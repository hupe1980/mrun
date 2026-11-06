//! The pipeline across the pool's Macs (026 R3): the host's [`PoolEngine`] and every other
//! Mac's [`serve_stage`] loop, over the ring.
//!
//! The host runs the embeddings, its layers and the head; each stage runs its layers. Every
//! state operation the prefix cache drives is applied on the host, then sent as an op frame
//! around the ring and applied by every stage in the same order, so all Macs hold the same
//! positions, segments and checkpoints (II). For prefill and forward the frame goes first and the
//! hidden state follows it; a forward's hidden state comes back to the host for the logits.
//! After each op a status frame goes round: every stage compares its length and the id it
//! assigned with the previous Mac's and passes on whether all agree; any disagreement or failure
//! fails the call with [`EngineError::Backend`] naming the stage. The host's state budget is
//! capped to what every stage holds ([`NuvecEngine::cap_state_tokens`]), so a stage's move
//! succeeds whenever the host's does. A pool cancels between prefill chunks, never inside one:
//! a chunk half sent would leave the ring out of step.

use mrun_engine::{
    CancelToken, Checkpoint, CheckpointId, Engine, EngineError, MemoryBudget, ModelInfo,
    NuvecEngine, Pipe, Sampling, Segment, SegmentId, Step, TokenMask,
};

/// How long one transfer on the ring may wait without progress. Long: a stage may compute a
/// large prefill chunk for many seconds without sending. A Mac gone without closing is found by
/// nuvec's keepalive (about 15 s) and by the heartbeat, which closes the ring.
pub const RING_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(24 * 3600);

/// Connects rank `rank` of the ring over `nodes` with the pool's `secret`: always authenticated,
/// and encrypted unless the pool is `unencrypted`, its host's choice (constitution IV). The host
/// passes the mode to every stage, so all ranks agree.
pub fn connect(
    nodes: &[Vec<String>],
    rank: usize,
    secret: &[u8; 32],
    unencrypted: bool,
) -> Result<Pipe, String> {
    Pipe::connect(nodes, rank, secret, !unencrypted, RING_TIMEOUT).map_err(|e| e.to_string())
}

const OP_WORDS: usize = 4;
const STATUS_WORDS: usize = 4;

/// What crosses the ring before an operation (no text, only numbers).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u32)]
pub enum Op {
    Stop = 0,
    SetLimit = 1,
    Prefill = 2,
    Forward = 3,
    Trim = 4,
    Checkpoint = 5,
    Restore = 6,
    DropCheckpoint = 7,
    DetachTail = 8,
    Attach = 9,
    DropSegment = 10,
    Reset = 11,
    ResetAll = 12,
}

impl Op {
    fn from_word(word: u32) -> Option<Self> {
        Some(match word {
            0 => Self::Stop,
            1 => Self::SetLimit,
            2 => Self::Prefill,
            3 => Self::Forward,
            4 => Self::Trim,
            5 => Self::Checkpoint,
            6 => Self::Restore,
            7 => Self::DropCheckpoint,
            8 => Self::DetachTail,
            9 => Self::Attach,
            10 => Self::DropSegment,
            11 => Self::Reset,
            12 => Self::ResetAll,
            _ => return None,
        })
    }
}

fn frame(op: Op, arg: u64) -> [u32; OP_WORDS] {
    [op as u32, arg as u32, (arg >> 32) as u32, 0]
}

fn arg(words: &[u32]) -> u64 {
    u64::from(words[1]) | (u64::from(words[2]) << 32)
}

fn status(ok: bool, len: u32, value: u64) -> [u32; STATUS_WORDS] {
    [u32::from(ok), len, value as u32, (value >> 32) as u32]
}

/// `MRUN_POOL_DEBUG`: every operation on the ring, on stderr.
fn debug(line: impl FnOnce() -> String) {
    if std::env::var_os("MRUN_POOL_DEBUG").is_some() {
        eprintln!("pool: {}", line());
    }
}

fn ring(error: impl std::fmt::Display) -> EngineError {
    EngineError::Backend(format!("pool ring: {error}"))
}

/// Why the pool closed its ring (a member lost), shared with the pool's health watch.
pub type RingReason = std::sync::Arc<std::sync::Mutex<Option<String>>>;

/// The host's engine: its part of the model plus the ring to the other parts.
pub struct PoolEngine {
    inner: NuvecEngine,
    pipe: Pipe,
    reason: RingReason,
    /// A failure that left the ring out of step: every later call fails with it.
    broken: Option<String>,
}

impl std::fmt::Debug for PoolEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "PoolEngine({:?}, {:?})", self.inner.part(), self.pipe)
    }
}

impl PoolEngine {
    /// `inner` is the host's part (with the head); `stage_tokens` the smallest state capacity
    /// (in positions) of the other Macs' parts.
    pub fn new(mut inner: NuvecEngine, pipe: Pipe, stage_tokens: u64, reason: RingReason) -> Self {
        inner.cap_state_tokens(stage_tokens);
        inner.set_pipe(Some(pipe.clone()));
        Self {
            inner,
            pipe,
            reason,
            broken: None,
        }
    }

    /// A ring failure, named after the lost Mac when the pool knows it (E060).
    fn ring(&self, error: impl std::fmt::Display) -> EngineError {
        match self
            .reason
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .as_deref()
        {
            Some(why) => EngineError::Backend(format!("E060 pool member lost: {why}")),
            None => ring(error),
        }
    }

    fn usable(&self) -> Result<(), EngineError> {
        match &self.broken {
            Some(why) => Err(EngineError::Backend(why.clone())),
            None => Ok(()),
        }
    }

    fn send(&self, op: Op, arg: u64) -> Result<(), EngineError> {
        debug(|| format!("host: {op:?} {arg}"));
        self.pipe
            .send_words(&frame(op, arg))
            .map_err(|e| self.ring(e))
    }

    /// Sends the host's status and checks the one that came round.
    fn finish(&mut self, ok: bool, value: u64) -> Result<(), EngineError> {
        let done = self
            .pipe
            .send_words(&status(ok, self.inner.len(), value))
            .and_then(|()| self.pipe.recv_words(STATUS_WORDS));
        let agreed = match done {
            Ok(words) => words[0] == 1,
            Err(e) => {
                let EngineError::Backend(why) = self.ring(e) else {
                    unreachable!()
                };
                self.broken = Some(why);
                false
            }
        };
        if agreed || !ok {
            return Ok(());
        }
        let why = self
            .broken
            .get_or_insert_with(|| "a Mac of the pool failed or fell out of step".into())
            .clone();
        Err(EngineError::Backend(why))
    }

    /// A backend failure during a transfer (the ring closed under it) named after the lost Mac.
    fn named(&self, error: EngineError) -> EngineError {
        match (
            &error,
            self.reason
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .as_deref(),
        ) {
            (EngineError::Backend(_), Some(why)) => {
                EngineError::Backend(format!("E060 pool member lost: {why}"))
            }
            _ => error,
        }
    }

    /// An operation applied here, then on every stage.
    fn mirrored<T>(
        &mut self,
        op: Op,
        arg: u64,
        here: impl FnOnce(&mut NuvecEngine) -> Result<T, EngineError>,
        value: impl Fn(&T) -> u64,
    ) -> Result<T, EngineError> {
        self.usable()?;
        let done = here(&mut self.inner)?;
        self.send(op, arg)?;
        self.finish(true, value(&done))?;
        Ok(done)
    }

    /// An infallible operation: a failure on the ring breaks the pool for later calls.
    fn mirrored_quietly(&mut self, op: Op, arg: u64, here: impl FnOnce(&mut NuvecEngine)) {
        here(&mut self.inner);
        if self.broken.is_none() && self.send(op, arg).is_ok() {
            let _ = self.finish(true, 0);
        }
    }
}

impl Drop for PoolEngine {
    fn drop(&mut self) {
        if self.broken.is_none() && self.send(Op::Stop, 0).is_ok() {
            let _ = self.finish(true, 0);
        }
        self.inner.set_pipe(None);
        self.pipe.close();
    }
}

impl Engine for PoolEngine {
    fn info(&self) -> &ModelInfo {
        self.inner.info()
    }

    fn budget(&self) -> &MemoryBudget {
        self.inner.budget()
    }

    fn len(&self) -> u32 {
        self.inner.len()
    }

    fn allocated(&self) -> u64 {
        self.inner.allocated()
    }

    fn set_limit(&mut self, total: u32) -> Result<(), EngineError> {
        self.mirrored(
            Op::SetLimit,
            u64::from(total),
            |e| e.set_limit(total),
            |_| 0,
        )
    }

    fn prefill_chunk(
        &mut self,
        tokens: &[u32],
        _cancel: &CancelToken,
    ) -> Result<bool, EngineError> {
        self.usable()?;
        if tokens.is_empty() {
            return Ok(true);
        }
        self.inner.check_append(tokens.len() as u32)?;
        self.send(Op::Prefill, tokens.len() as u64)?;
        let done = self.inner.prefill_chunk(tokens, &CancelToken::new());
        self.finish(done.is_ok(), 0)?;
        done.map_err(|e| self.named(e))
    }

    fn forward(&mut self, last: u32) -> Result<Step, EngineError> {
        self.usable()?;
        self.inner.check_append(1)?;
        self.send(Op::Forward, 0)?;
        let done = self.inner.forward(last);
        self.finish(done.is_ok(), 0)?;
        done.map_err(|e| self.named(e))
    }

    fn sample(
        &mut self,
        step: Step,
        sampling: &Sampling,
        mask: Option<&TokenMask>,
    ) -> Result<u32, EngineError> {
        // With a ring attached the host feeds no pipelined step: sampling changes no state.
        self.inner.sample(step, sampling, mask)
    }

    fn top_logprobs(&mut self, step: &Step, n: usize) -> Result<Vec<(u32, f32)>, EngineError> {
        self.inner.top_logprobs(step, n)
    }

    fn trim(&mut self, len: u32) -> Result<(), EngineError> {
        self.mirrored(Op::Trim, u64::from(len), |e| e.trim(len), |_| 0)
    }

    fn checkpoint(&mut self) -> Result<Checkpoint, EngineError> {
        self.mirrored(Op::Checkpoint, 0, |e| e.checkpoint(), |c| c.id.0)
    }

    fn restore(&mut self, id: CheckpointId) -> Result<(), EngineError> {
        self.mirrored(Op::Restore, id.0, |e| e.restore(id), |_| 0)
    }

    fn drop_checkpoint(&mut self, id: CheckpointId) {
        self.mirrored_quietly(Op::DropCheckpoint, id.0, |e| e.drop_checkpoint(id));
    }

    fn detach_tail(&mut self, at: u32) -> Result<Segment, EngineError> {
        self.mirrored(
            Op::DetachTail,
            u64::from(at),
            |e| e.detach_tail(at),
            |s| s.id.0,
        )
    }

    fn attach(&mut self, id: SegmentId) -> Result<(), EngineError> {
        self.mirrored(Op::Attach, id.0, |e| e.attach(id), |_| 0)
    }

    fn drop_segment(&mut self, id: SegmentId) {
        self.mirrored_quietly(Op::DropSegment, id.0, |e| e.drop_segment(id));
    }

    fn reset(&mut self) {
        self.mirrored_quietly(Op::Reset, 0, Engine::reset);
    }

    fn reset_all(&mut self) {
        self.mirrored_quietly(Op::ResetAll, 0, Engine::reset_all);
        // A reset after a failure starts over in step only if the ring still works.
        if self
            .broken
            .as_deref()
            .is_some_and(|b| !b.starts_with("pool ring"))
        {
            self.broken = None;
        }
    }
}

/// Applies `op` to a stage's part; the value is the id an op assigned.
fn apply(engine: &mut NuvecEngine, op: Op, arg: u64) -> Result<u64, EngineError> {
    let never = CancelToken::new();
    match op {
        Op::Stop => Ok(0),
        Op::SetLimit => engine.set_limit(arg as u32).map(|()| 0),
        // A stage's input is the previous part's hidden state; the ids are not needed.
        Op::Prefill => engine
            .prefill_chunk(&vec![0; arg as usize], &never)
            .map(|_| 0),
        Op::Forward => engine.forward(0).map(|_| 0),
        Op::Trim => engine.trim(arg as u32).map(|()| 0),
        Op::Checkpoint => engine.checkpoint().map(|c| c.id.0),
        Op::Restore => engine.restore(CheckpointId(arg)).map(|()| 0),
        Op::DropCheckpoint => {
            engine.drop_checkpoint(CheckpointId(arg));
            Ok(0)
        }
        Op::DetachTail => engine.detach_tail(arg as u32).map(|s| s.id.0),
        Op::Attach => engine.attach(SegmentId(arg)).map(|()| 0),
        Op::DropSegment => {
            engine.drop_segment(SegmentId(arg));
            Ok(0)
        }
        Op::Reset => {
            engine.reset();
            Ok(0)
        }
        Op::ResetAll => {
            engine.reset_all();
            Ok(0)
        }
    }
}

/// A stage's loop: applies the host's operations to `engine` (its part, the ring attached) in
/// order until the host stops or the ring fails.
pub fn serve_stage(engine: &mut NuvecEngine) -> Result<(), String> {
    let pipe = engine
        .pipe()
        .cloned()
        .ok_or("a stage needs the ring attached")?;
    loop {
        // The host closed the ring between operations (it serves again, or the pool is over).
        let Ok(words) = pipe.recv_words(OP_WORDS) else {
            return Ok(());
        };
        let op = Op::from_word(words[0]).ok_or("an unknown pool operation")?;
        debug(|| format!("stage {}: {op:?}", pipe.rank()));
        if !pipe.last() {
            pipe.send_words(&words).map_err(|e| e.to_string())?;
        }
        let applied = apply(engine, op, arg(&words));
        debug(|| format!("stage {}: {op:?} applied: {applied:?}", pipe.rank()));
        // A failed prefill or forward leaves hidden states unsent: the ring cannot stay in step,
        // so it closes, and the host's waiting transfer fails at once instead of hanging.
        if let (Err(e), Op::Prefill | Op::Forward) = (&applied, op) {
            pipe.close();
            return Err(format!("{op:?} failed: {e}"));
        }
        let before = pipe.recv_words(STATUS_WORDS).map_err(|e| e.to_string())?;
        let value = |w: &[u32]| u64::from(w[2]) | (u64::from(w[3]) << 32);
        let agreed = match &applied {
            Ok(v) => before[0] == 1 && before[1] == engine.len() && value(&before) == *v,
            Err(_) => false,
        };
        let mine = status(agreed, engine.len(), applied.as_ref().copied().unwrap_or(0));
        pipe.send_words(&mine).map_err(|e| e.to_string())?;
        if let Err(e) = applied {
            eprintln!("pool stage: {op:?} failed: {e}");
        }
        if op == Op::Stop {
            return Ok(());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ports() -> Vec<Vec<String>> {
        (0..2)
            .map(|_| {
                let port = std::net::TcpListener::bind("127.0.0.1:0")
                    .unwrap()
                    .local_addr()
                    .unwrap()
                    .port();
                vec![format!("127.0.0.1:{port}")]
            })
            .collect()
    }

    /// Two ranks in two threads, each in its mode; what rank 1 received, or each one's error.
    fn ring(modes: [bool; 2]) -> Result<Vec<u32>, String> {
        let nodes = ports();
        let timeout = std::time::Duration::from_secs(3);
        // Both ends stay open until rank 1 has its words.
        let done = std::sync::Arc::new(std::sync::Barrier::new(2));
        let ranks: Vec<_> = (0..2)
            .map(|rank| {
                let (nodes, done) = (nodes.clone(), done.clone());
                std::thread::spawn(move || {
                    mrun_engine::tensor::use_device(mrun_engine::tensor::Device::Cpu);
                    let got =
                        Pipe::connect(&nodes, rank, &[7; 32], modes[rank], timeout).map(|pipe| {
                            let got = if rank == 0 {
                                pipe.send_words(&[1, 2, 3, 4]).map(|()| Vec::new())
                            } else {
                                pipe.recv_words(4)
                            };
                            (pipe, got)
                        });
                    done.wait();
                    got.and_then(|(_pipe, got)| got).map_err(|e| e.to_string())
                })
            })
            .collect();
        let results: Vec<Result<Vec<u32>, String>> =
            ranks.into_iter().map(|r| r.join().unwrap()).collect();
        match (&results[0], &results[1]) {
            (Ok(_), Ok(words)) => Ok(words.clone()),
            _ => Err(format!("{results:?}")),
        }
    }

    #[test]
    fn a_ring_carries_the_same_numbers_encrypted_or_not_and_both_ends_must_agree() {
        assert_eq!(ring([false, false]), Ok(vec![1, 2, 3, 4]));
        assert_eq!(ring([true, true]), Ok(vec![1, 2, 3, 4]));
        let e = ring([true, false]).unwrap_err();
        assert!(e.contains("does not encrypt"), "{e}");
    }

    #[test]
    fn frames_round_trip() {
        let f = frame(Op::DetachTail, (7u64 << 32) | 9);
        assert_eq!(Op::from_word(f[0]), Some(Op::DetachTail));
        assert_eq!(arg(&f), (7u64 << 32) | 9);
        assert_eq!(Op::from_word(99), None);
        assert_eq!(status(true, 5, 1 << 33), [1, 5, 0, 2]);
    }
}
