//! `ScriptedEngine`: a deterministic [`Engine`] (contract v2) for tests on any OS, without a GPU.
//!
//! It emits a fixed token script and models what the runtime relies on. Its bookkeeping is
//! `NuvecEngine`'s own ([`Ledger`]): lazy growth, `NeedsMemory` with nothing changed, the limit, one
//! pending pipelined step, non-trimmable position-bound state (hybrid mode), checkpoints, moves
//! that shrink the live allocation to `align(at)` and credit a consumed segment, and the poisoned
//! state after a failed move. On top it models per-layer cancellation inside a chunk and the
//! token-mask width check. Position-bound state is modelled as the tokens it has consumed, so
//! restoring a checkpoint or segment that does not match the live sequence is detected on the
//! next use.

use mrun_engine::budget::{self, BudgetError, DeviceMemory, Overrides, RESERVED_CHECKPOINTS};
use mrun_engine::ledger::Ledger;
use mrun_engine::{
    CancelToken, Checkpoint, CheckpointId, Engine, EngineError, LayerKind, MemoryBudget, ModelInfo,
    Numerics, Sampling, Segment, SegmentId, Step, TokenMask, verify_serially,
};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

/// End-of-sequence token used by [`ScriptedEngine`].
pub const SCRIPT_EOS: u32 = u32::MAX - 1;
/// Attention bytes per token.
pub const KV_BYTES: u64 = 16;

/// Byte costs of one family's reference model (4-bit weights, bf16 state).
#[derive(Clone, Copy, Debug)]
pub struct FamilyProfile {
    pub family: &'static str,
    pub layers: u32,
    /// Kind of layer `i`.
    pub layer_kind: fn(u32) -> LayerKind,
    pub kv_bytes_per_token: u64,
    /// Position-bound bytes per checkpoint.
    pub recurrent_bytes: u64,
    pub max_position: u32,
    /// Approximate weight bytes, for budget fixtures.
    pub weights: u64,
}

const KIB: u64 = 1 << 10;
const MIB: u64 = 1 << 20;

/// Reference models of the 0.1 families: Qwen3.6-35B-A3B, Qwen3-Coder-30B-A3B and Qwen3-8B.
pub const PROFILES: [FamilyProfile; 3] = [
    FamilyProfile {
        family: "qwen3_5_moe",
        layers: 40,
        layer_kind: |i| {
            if (i + 1) % 4 == 0 {
                LayerKind::FullAttention
            } else {
                LayerKind::Linear
            }
        },
        kv_bytes_per_token: 20 * KIB,
        recurrent_bytes: 64_389_120,
        max_position: 262_144,
        weights: 20 * 1024 * MIB,
    },
    FamilyProfile {
        family: "qwen3_moe",
        layers: 48,
        layer_kind: |_| LayerKind::FullAttention,
        kv_bytes_per_token: 96 * KIB,
        recurrent_bytes: 0,
        max_position: 262_144,
        weights: 16 * 1024 * MIB,
    },
    FamilyProfile {
        family: "qwen3",
        layers: 36,
        layer_kind: |_| LayerKind::FullAttention,
        kv_bytes_per_token: 144 * KIB,
        recurrent_bytes: 0,
        max_position: 40_960,
        weights: 4_400 * MIB,
    },
];

impl FamilyProfile {
    pub fn get(family: &str) -> Option<&'static Self> {
        PROFILES.iter().find(|profile| profile.family == family)
    }

    /// Budget on a machine with `physical` bytes (all usable by the GPU) and `scratch`.
    pub fn budget(&self, physical: u64, scratch: u64) -> Result<MemoryBudget, BudgetError> {
        let info = ModelInfo {
            identity: String::new(),
            family: self.family.into(),
            layer_kinds: Vec::new(),
            kv_heads: 1,
            head_dim: 1,
            vocab: 1,
            max_position: self.max_position,
            kv_bytes_per_token: self.kv_bytes_per_token,
            recurrent_bytes: self.recurrent_bytes,
            eos_tokens: Vec::new(),
            numerics: Numerics::scripted(),
            ..ModelInfo::default()
        };
        let device = DeviceMemory {
            physical,
            recommended_working_set: physical,
            available: None,
            estimated: false,
        };
        let top = budget::plan(device, Overrides::default())?;
        budget::with_state_capacity(top, self.weights, &|_| scratch, &info)
    }
}

#[derive(Debug)]
struct Pending {
    token: u32,
    step: u64,
}

#[derive(Clone, Debug)]
struct StoredSegment {
    tokens: Vec<u32>,
    recurrent: Option<Vec<u32>>,
}

#[derive(Debug)]
pub struct ScriptedEngine {
    info: ModelInfo,
    script: Vec<u32>,
    next: usize,
    /// Live positions, including a pending pipelined token.
    tokens: Vec<u32>,
    /// Hybrid only: tokens the live position-bound state has consumed; `None` after a detach.
    recurrent: Option<Vec<u32>>,
    /// Checkpoints hold the tokens their state has consumed.
    ledger: Ledger<Pending, StoredSegment, Vec<u32>>,
    current: Option<u64>,
    /// Cancels once this many prefill layers have run in total (one-shot).
    pub cancel_after_layers: Option<u64>,
    pub panic_on_forward: bool,
    /// The next move (`detach_tail` or `attach`) fails half-way (one-shot).
    pub fail_next_move: bool,
    /// Sleep before each sampled token (keepalive, deadline and disconnect tests).
    pub step_delay: Option<std::time::Duration>,
    /// Sleep per layer of a prefill chunk; cancellation is checked between layers (prefill
    /// disconnect tests).
    pub layer_delay: Option<std::time::Duration>,
    /// Incremented by every `reset_all` call (shared so tests can observe the worker thread).
    pub reset_all_calls: Arc<AtomicUsize>,
    pub layers_run: u64,
    /// Every token fed by a prefill chunk, `forward` or a pipelined step, in order.
    pub processed: Vec<u32>,
    /// Committed prefill chunks as `(start, end)`.
    pub chunks: Vec<(u32, u32)>,
    /// Highest `allocated()` seen (memory-bound tests).
    pub peak_allocated: u64,
    /// Draft tokens a verify takes (006); 0, the default, keeps the runtime from drafting. Its
    /// verify is serial decoding with the trim window of a verify pass.
    pub max_draft: u32,
}

impl ScriptedEngine {
    /// Attention-only model with four layers; `script` is sampled in order, then [`SCRIPT_EOS`].
    pub fn new(script: Vec<u32>, capacity_tokens: u32) -> Self {
        Self::build(
            script,
            capacity_tokens,
            vec![LayerKind::FullAttention; 4],
            0,
        )
    }

    /// Hybrid model (three linear layers, one attention layer) with `recurrent_bytes` of state.
    pub fn new_hybrid(script: Vec<u32>, capacity_tokens: u32, recurrent_bytes: u64) -> Self {
        let mut kinds = vec![LayerKind::Linear; 3];
        kinds.push(LayerKind::FullAttention);
        Self::build(script, capacity_tokens, kinds, recurrent_bytes)
    }

    fn build(script: Vec<u32>, capacity: u32, kinds: Vec<LayerKind>, recurrent: u64) -> Self {
        let info = ModelInfo {
            identity: "scripted".into(),
            family: if recurrent > 0 {
                "qwen3_5_moe"
            } else {
                "qwen3"
            }
            .into(),
            layer_kinds: kinds,
            kv_heads: 1,
            head_dim: 1,
            vocab: u32::MAX,
            max_position: capacity,
            kv_bytes_per_token: KV_BYTES,
            recurrent_bytes: recurrent,
            eos_tokens: vec![SCRIPT_EOS],
            numerics: Numerics::scripted(),
            ..ModelInfo::default()
        };
        let budget = MemoryBudget {
            state_capacity_bytes: u64::from(capacity) * KV_BYTES
                + recurrent * (1 + RESERVED_CHECKPOINTS),
            kv_capacity_tokens: capacity,
            ..MemoryBudget::default()
        };
        Self::with_info(script, info, budget)
    }

    /// Engine with the byte costs of a family's reference model under `budget`.
    pub fn for_family(script: Vec<u32>, profile: &FamilyProfile, budget: MemoryBudget) -> Self {
        let info = ModelInfo {
            identity: "scripted".into(),
            family: profile.family.into(),
            layer_kinds: (0..profile.layers).map(profile.layer_kind).collect(),
            kv_heads: 1,
            head_dim: 1,
            vocab: u32::MAX,
            max_position: profile.max_position,
            kv_bytes_per_token: profile.kv_bytes_per_token,
            recurrent_bytes: profile.recurrent_bytes,
            eos_tokens: vec![SCRIPT_EOS],
            numerics: Numerics::scripted(),
            ..ModelInfo::default()
        };
        Self::with_info(script, info, budget)
    }

    fn with_info(script: Vec<u32>, info: ModelInfo, budget: MemoryBudget) -> Self {
        let hybrid = info.is_hybrid();
        Self {
            ledger: Ledger::new(&info, budget),
            info,
            script,
            next: 0,
            tokens: Vec::new(),
            recurrent: hybrid.then(Vec::new),
            current: None,
            cancel_after_layers: None,
            panic_on_forward: false,
            fail_next_move: false,
            step_delay: None,
            layer_delay: None,
            reset_all_calls: Arc::default(),
            layers_run: 0,
            processed: Vec::new(),
            chunks: Vec::new(),
            peak_allocated: 0,
            max_draft: 0,
        }
    }

    pub fn with_state_capacity_bytes(mut self, bytes: u64) -> Self {
        self.ledger.budget.state_capacity_bytes = bytes;
        self
    }

    /// The model's logits width (a real model's padded `vocab_size`); masks may be narrower.
    pub fn with_vocab(mut self, vocab: u32) -> Self {
        self.info.vocab = vocab;
        self
    }

    /// Live tokens, including a pending pipelined token.
    pub fn tokens(&self) -> &[u32] {
        &self.tokens
    }

    /// Allocated live positions.
    pub fn cap(&self) -> u32 {
        self.ledger.capacity()
    }

    pub fn has_pending(&self) -> bool {
        self.ledger.pending().is_some()
    }

    pub fn checkpoint_count(&self) -> usize {
        self.ledger.checkpoint_count()
    }

    fn hybrid(&self) -> bool {
        self.info.is_hybrid()
    }

    fn note_peak(&mut self) {
        self.peak_allocated = self.peak_allocated.max(self.ledger.allocated());
    }

    /// The live position-bound state must match the attention state.
    fn check_consumed(&self) -> Result<(), EngineError> {
        match &self.recurrent {
            Some(consumed) if self.hybrid() && *consumed != self.tokens => {
                Err(EngineError::Backend(
                    "position-bound state does not match the attention state".into(),
                ))
            }
            _ => Ok(()),
        }
    }

    fn feed(&mut self, tokens: &[u32]) {
        self.tokens.extend_from_slice(tokens);
        if let Some(consumed) = &mut self.recurrent {
            consumed.extend_from_slice(tokens);
        }
        self.processed.extend_from_slice(tokens);
    }

    fn roll_back(&mut self) {
        if self.ledger.roll_back().is_some() {
            self.tokens.pop();
            if let Some(consumed) = &mut self.recurrent {
                consumed.pop();
            }
        }
    }

    fn zero_recurrent(&mut self) {
        self.recurrent = self.hybrid().then(Vec::new);
    }

    /// Plans an append of `n` tokens, rolls back a pending step and grows; returns the capacity
    /// before, to give back on a cancel.
    fn grow(&mut self, n: u32) -> Result<u32, EngineError> {
        let growth = self.ledger.plan_append(n)?;
        self.roll_back();
        self.check_consumed()?;
        let before = self.ledger.capacity();
        if let Some(capacity) = growth {
            // A growth moves the state into a larger allocation, like the engine's resize.
            self.moved("growth")?;
            self.ledger.set_capacity(capacity);
            self.note_peak();
        }
        Ok(before)
    }

    /// A move that fails half-way when `fail_next_move` is set.
    fn moved(&mut self, what: &str) -> Result<(), EngineError> {
        if std::mem::take(&mut self.fail_next_move) {
            return Err(self.ledger.poison(format!("{what}: scripted failure")));
        }
        Ok(())
    }
}

impl Engine for ScriptedEngine {
    fn info(&self) -> &ModelInfo {
        &self.info
    }

    fn budget(&self) -> &MemoryBudget {
        &self.ledger.budget
    }

    fn len(&self) -> u32 {
        self.ledger.len()
    }

    fn allocated(&self) -> u64 {
        self.ledger.allocated()
    }

    fn set_limit(&mut self, total: u32) -> Result<(), EngineError> {
        self.ledger.set_limit(total)
    }

    fn prefill_chunk(&mut self, tokens: &[u32], cancel: &CancelToken) -> Result<bool, EngineError> {
        if tokens.is_empty() {
            return Ok(true);
        }
        let capacity = self.grow(tokens.len() as u32)?;
        self.current = None;
        let start = self.ledger.len();
        let layers = self.info.layers();
        for layer in 1..=layers {
            if let Some(delay) = self.layer_delay {
                std::thread::sleep(delay);
            }
            self.layers_run += 1;
            if self
                .cancel_after_layers
                .is_some_and(|n| self.layers_run >= n)
            {
                self.cancel_after_layers = None;
                cancel.cancel();
            }
            // The partial chunk was never written to the live state; its growth is given back.
            if layer < layers && cancel.is_cancelled() {
                self.ledger.set_capacity(capacity);
                return Ok(false);
            }
        }
        self.feed(tokens);
        self.ledger.advance(tokens.len() as u32);
        self.chunks.push((start, start + tokens.len() as u32));
        Ok(true)
    }

    fn forward(&mut self, last: u32) -> Result<Step, EngineError> {
        assert!(!self.panic_on_forward, "scripted panic");
        self.ledger.usable()?;
        if let Some(pending) = self.ledger.reuse_pending(|pending| pending.token == last) {
            self.current = Some(pending.step);
            return Ok(Step::new(pending.step));
        }
        self.grow(1)?;
        self.feed(&[last]);
        self.ledger.advance(1);
        let id = self.ledger.id();
        self.current = Some(id);
        Ok(Step::new(id))
    }

    fn sample(
        &mut self,
        step: Step,
        _sampling: &Sampling,
        mask: Option<&TokenMask>,
    ) -> Result<u32, EngineError> {
        self.ledger.usable()?;
        if self.current != Some(step.id()) {
            return Err(EngineError::InvalidStep);
        }
        if let Some(mask) = mask {
            mask.check_width(self.info.vocab)?;
        }
        self.current = None;
        if let Some(delay) = self.step_delay {
            std::thread::sleep(delay);
        }
        if mask.is_some_and(TokenMask::is_none) {
            return Err(EngineError::EmptyMask);
        }
        let token = self.script.get(self.next).copied().unwrap_or(SCRIPT_EOS);
        self.next += 1;
        if let Some(mask) = mask
            && token != SCRIPT_EOS
            && !mask.is_allowed(token)
        {
            return Err(EngineError::Backend(format!(
                "script token {token} is masked"
            )));
        }
        if self.ledger.has_room() {
            self.feed(&[token]);
            let step = self.ledger.id();
            self.ledger.push_pending(Pending { token, step });
        }
        Ok(token)
    }

    fn top_logprobs(&mut self, step: &Step, n: usize) -> Result<Vec<(u32, f32)>, EngineError> {
        if self.current != Some(step.id()) {
            return Err(EngineError::InvalidStep);
        }
        let top = self.script.get(self.next).copied().unwrap_or(SCRIPT_EOS);
        Ok(std::iter::once((top, -0.01))
            .chain(
                (0..)
                    .filter(|&token| token != top)
                    .map(|token| (token, -2.0 - token as f32)),
            )
            .take(n)
            .collect())
    }

    fn max_draft(&self) -> u32 {
        self.max_draft
    }

    fn verify(
        &mut self,
        last: u32,
        draft: &[u32],
        sampling: &Sampling,
    ) -> Result<Vec<u32>, EngineError> {
        if self.max_draft > 0 && draft.len() > self.max_draft as usize {
            return Err(EngineError::DraftTooLong {
                draft: draft.len() as u32,
                max: self.max_draft,
            });
        }
        // `last` goes at the pending step's position, or at the end.
        let start = self.ledger.len() - u32::from(self.ledger.pending().is_some());
        let committed = verify_serially(self, last, draft, sampling)?;
        if self.max_draft > 0 {
            self.ledger.open_window(start);
        }
        Ok(committed)
    }

    fn trim(&mut self, len: u32) -> Result<(), EngineError> {
        let windowed = self.ledger.window().is_some();
        let cut = self.ledger.trim(len)?;
        if cut.rolled_back.is_some()
            && let Some(consumed) = &mut self.recurrent
        {
            consumed.pop();
        }
        // Inside a verify's window the state after `len` positions is known.
        if windowed && let Some(consumed) = &mut self.recurrent {
            consumed.truncate(len as usize);
        }
        if cut.cut_step {
            self.current = None;
        }
        self.tokens.truncate(len as usize);
        if cut.zero_recurrent {
            self.zero_recurrent();
        }
        Ok(())
    }

    fn checkpoint(&mut self) -> Result<Checkpoint, EngineError> {
        let bytes = self.ledger.plan_checkpoint()?;
        self.roll_back();
        let consumed = self
            .recurrent
            .clone()
            .unwrap_or_else(|| self.tokens.clone());
        let checkpoint = self.ledger.commit_checkpoint(bytes, consumed);
        self.note_peak();
        Ok(checkpoint)
    }

    fn restore(&mut self, id: CheckpointId) -> Result<(), EngineError> {
        let position = self.ledger.plan_restore(id)?;
        let consumed = self.ledger.checkpoint_data(id).cloned();
        self.roll_back();
        self.current = None;
        self.tokens.truncate(position as usize);
        self.ledger.commit_restore(position);
        if self.hybrid() {
            self.recurrent = consumed;
        }
        self.note_peak();
        Ok(())
    }

    fn drop_checkpoint(&mut self, id: CheckpointId) {
        self.ledger.drop_checkpoint(id);
    }

    fn detach_tail(&mut self, at: u32) -> Result<Segment, EngineError> {
        let plan = self.ledger.plan_detach(at)?;
        self.roll_back();
        self.current = None;
        self.moved("detach")?;
        let tokens = self.tokens.split_off(at as usize);
        let recurrent = self.recurrent.take();
        let segment = self
            .ledger
            .commit_detach(plan, StoredSegment { tokens, recurrent });
        self.note_peak();
        Ok(segment)
    }

    fn attach(&mut self, id: SegmentId) -> Result<(), EngineError> {
        let plan = self.ledger.plan_attach(id)?;
        let seg = self.ledger.take_segment(id).expect("planned");
        self.roll_back();
        self.current = None;
        self.moved("attach")?;
        self.tokens.extend_from_slice(&seg.tokens);
        if self.hybrid() {
            self.recurrent = seg.recurrent;
        }
        self.ledger.commit_attach(plan);
        self.note_peak();
        Ok(())
    }

    fn drop_segment(&mut self, id: SegmentId) {
        self.ledger.drop_segment(id);
    }

    fn reset(&mut self) {
        self.ledger.reset();
        self.current = None;
        self.tokens.clear();
        self.zero_recurrent();
    }

    fn reset_all(&mut self) {
        self.reset();
        self.ledger.reset_all();
        self.reset_all_calls.fetch_add(1, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const RB: u64 = 1_000;

    fn engine(capacity: u32) -> ScriptedEngine {
        ScriptedEngine::new(vec![7, 8, 9], capacity)
    }

    fn hybrid(capacity: u32) -> ScriptedEngine {
        ScriptedEngine::new_hybrid(vec![7, 8, 9], capacity, RB)
    }

    fn seq(range: std::ops::Range<u32>) -> Vec<u32> {
        range.collect()
    }

    fn prefill(engine: &mut ScriptedEngine, tokens: &[u32]) {
        assert!(engine.prefill_chunk(tokens, &CancelToken::new()).unwrap());
    }

    #[test]
    fn allocation_grows_lazily_within_the_limit() {
        let mut engine = engine(100_000);
        engine.set_limit(10_000).unwrap();
        assert_eq!(engine.allocated(), 0);
        prefill(&mut engine, &seq(0..100));
        assert_eq!(engine.cap(), 4352, "input + 4096, aligned");
        prefill(&mut engine, &seq(100..4400));
        assert_eq!(engine.cap(), 8704, "doubled");
        assert!(matches!(
            engine.prefill_chunk(&seq(0..6000), &CancelToken::new()),
            Err(EngineError::ReservationExceeded { .. })
        ));
        assert_eq!(
            engine.set_limit(100_001),
            Err(EngineError::Capacity {
                requested: 100_001,
                available: 100_000
            })
        );
    }

    #[test]
    fn needs_memory_reports_the_shortfall_and_changes_nothing() {
        let mut engine = engine(100_000).with_state_capacity_bytes(1_000 * KV_BYTES);
        engine.set_limit(10_000).unwrap();
        let error = engine
            .prefill_chunk(&seq(0..100), &CancelToken::new())
            .unwrap_err();
        assert_eq!(
            error,
            EngineError::NeedsMemory {
                bytes: (4352 - 1_000) * KV_BYTES
            }
        );
        assert_eq!(engine.len(), 0);
        assert_eq!(engine.allocated(), 0);
    }

    #[test]
    fn a_cancelled_chunk_is_not_committed() {
        let mut engine = engine(10_000);
        engine.set_limit(10_000).unwrap();
        prefill(&mut engine, &seq(0..10));
        engine.cancel_after_layers = Some(engine.layers_run + 2);
        let cancel = CancelToken::new();
        assert!(!engine.prefill_chunk(&seq(10..20), &cancel).unwrap());
        assert_eq!(engine.tokens(), seq(0..10));
        assert!(cancel.is_cancelled());
    }

    #[test]
    fn pipelined_step_is_reused_or_rolled_back() {
        let mut engine = engine(100);
        engine.set_limit(100).unwrap();
        prefill(&mut engine, &[1, 2]);
        let step = engine.forward(3).unwrap();
        assert_eq!(engine.sample(step, &Sampling::GREEDY, None).unwrap(), 7);
        assert!(engine.has_pending());
        assert_eq!(engine.tokens(), [1, 2, 3, 7]);
        // The same token reuses the pending step.
        let step = engine.forward(7).unwrap();
        assert_eq!(engine.tokens(), [1, 2, 3, 7]);
        assert_eq!(engine.sample(step, &Sampling::GREEDY, None).unwrap(), 8);
        // A different token (forced) rolls the pending one back.
        engine.forward(42).unwrap();
        assert_eq!(engine.tokens(), [1, 2, 3, 7, 42]);
    }

    #[test]
    fn trim_to_the_length_confirms_a_pending_step() {
        let mut engine = engine(100);
        engine.set_limit(100).unwrap();
        prefill(&mut engine, &[1, 2]);
        let step = engine.forward(3).unwrap();
        engine.sample(step, &Sampling::GREEDY, None).unwrap();
        engine.trim(4).unwrap();
        assert!(!engine.has_pending());
        assert_eq!(engine.tokens(), [1, 2, 3, 7]);
        let step = engine.forward(7).unwrap();
        engine.sample(step, &Sampling::GREEDY, None).unwrap();
        engine.trim(4).unwrap();
        assert_eq!(
            engine.tokens(),
            [1, 2, 3, 7],
            "below the pending token: rolled back"
        );
    }

    #[test]
    fn hybrid_state_trims_only_at_its_position_or_zero() {
        let mut engine = hybrid(100);
        engine.set_limit(100).unwrap();
        prefill(&mut engine, &seq(0..10));
        assert_eq!(
            engine.trim(5),
            Err(EngineError::NotTrimmable { len: 5, active: 10 })
        );
        engine.trim(10).unwrap();
        let step = engine.forward(10).unwrap();
        engine.sample(step, &Sampling::GREEDY, None).unwrap();
        engine.trim(11).unwrap();
        engine.trim(0).unwrap();
        assert_eq!(engine.len(), 0);
        prefill(&mut engine, &seq(0..3));
    }

    #[test]
    fn no_pipelining_without_room_in_the_allocation() {
        let mut engine = engine(3);
        engine.set_limit(3).unwrap();
        prefill(&mut engine, &[1, 2]);
        let step = engine.forward(3).unwrap();
        engine.sample(step, &Sampling::GREEDY, None).unwrap();
        assert!(!engine.has_pending());
    }

    #[test]
    fn hybrid_checkpoints_restore_earlier_positions() {
        let mut engine = hybrid(100);
        engine.set_limit(100).unwrap();
        prefill(&mut engine, &seq(0..4));
        let at4 = engine.checkpoint().unwrap();
        prefill(&mut engine, &seq(4..8));
        let at8 = engine.checkpoint().unwrap();
        prefill(&mut engine, &seq(8..12));
        engine.restore(at4.id).unwrap();
        assert_eq!(engine.tokens(), seq(0..4));
        prefill(&mut engine, &seq(4..8));
        engine.restore(at8.id).unwrap();
        prefill(&mut engine, &[99]);
        engine.trim(0).unwrap();
        assert!(matches!(
            engine.restore(at8.id),
            Err(EngineError::InvalidRange { .. })
        ));
    }

    #[test]
    fn detach_and_attach_move_state_without_duplicating_it() {
        // A 90K conversation at a 105K capacity survives a 15K-token side request.
        let mut engine = engine(105_000).with_state_capacity_bytes(105_000 * KV_BYTES);
        engine.set_limit(100_000).unwrap();
        prefill(&mut engine, &seq(0..90_000));
        let tail = engine.detach_tail(15_000).unwrap();
        assert_eq!((tail.start, tail.end), (15_000, 90_000));
        assert_eq!(engine.cap(), 15_104, "shrunk to align(15_000)");
        engine.set_limit(30_000).unwrap();
        prefill(&mut engine, &seq(200_000..210_000));
        let side = engine.detach_tail(15_000).unwrap();
        engine.drop_segment(side.id);
        engine.attach(tail.id).unwrap();
        assert_eq!(engine.tokens(), seq(0..90_000));
        assert!(engine.peak_allocated <= 105_000 * KV_BYTES);
        assert!(matches!(
            engine.attach(tail.id),
            Err(EngineError::UnknownSegment(_))
        ));
    }

    #[test]
    fn hybrid_detach_moves_the_position_bound_state() {
        let mut engine = hybrid(1_000);
        engine.set_limit(1_000).unwrap();
        prefill(&mut engine, &seq(0..4));
        let at4 = engine.checkpoint().unwrap();
        prefill(&mut engine, &seq(4..8));
        let tail = engine.detach_tail(4).unwrap();
        assert_eq!(
            engine.prefill_chunk(&[1], &CancelToken::new()),
            Err(EngineError::NoRecurrentState)
        );
        engine.attach(tail.id).unwrap();
        prefill(&mut engine, &[8]);
        let tail = engine.detach_tail(4).unwrap();
        engine.restore(at4.id).unwrap();
        prefill(&mut engine, &seq(50..52));
        engine.drop_segment(tail.id);
    }

    #[test]
    fn small_machines_refuse_or_shrink_large_families() {
        let gib = 1 << 30;
        let qwen36 = FamilyProfile::get("qwen3_5_moe").unwrap();
        assert!(qwen36.budget(16 * gib, gib).is_err());
        let qwen3 = FamilyProfile::get("qwen3").unwrap();
        let small = qwen3.budget(16 * gib, gib).unwrap();
        assert_eq!(small.kv_capacity_tokens, 40_960);
    }

    #[test]
    fn masks_narrower_than_the_logits_are_padded_and_wider_ones_refused() {
        // Qwen: the tokenizer has 151,669 tokens, the logits 151,936.
        let mut engine = ScriptedEngine::new(vec![7, 8], 100).with_vocab(151_936);
        engine.set_limit(100).unwrap();
        prefill(&mut engine, &[1, 2]);
        let mut narrow = TokenMask::none_allowed(151_669);
        narrow.allow(7);
        let step = engine.forward(3).unwrap();
        assert_eq!(engine.sample(step, &Sampling::GREEDY, Some(&narrow)), Ok(7));
        let wide = TokenMask::none_allowed(151_937);
        let step = engine.forward(7).unwrap();
        assert_eq!(
            engine.sample(step, &Sampling::GREEDY, Some(&wide)),
            Err(EngineError::MaskTooWide {
                mask: 151_937,
                vocab: 151_936
            })
        );
    }

    #[test]
    fn a_cancelled_chunk_gives_its_growth_back() {
        let mut engine = engine(100_000);
        engine.set_limit(100_000).unwrap();
        prefill(&mut engine, &seq(0..100));
        let (cap, allocated) = (engine.cap(), engine.allocated());
        engine.cancel_after_layers = Some(engine.layers_run + 1);
        assert!(
            !engine
                .prefill_chunk(&seq(100..9_000), &CancelToken::new())
                .unwrap()
        );
        assert_eq!((engine.cap(), engine.allocated()), (cap, allocated));
    }

    #[test]
    fn trim_releases_capacity_beyond_a_growth_step() {
        let mut engine = engine(100_000);
        engine.set_limit(100_000).unwrap();
        prefill(&mut engine, &seq(0..20_000));
        engine.trim(1_000).unwrap();
        assert_eq!(engine.cap(), 1_024 + mrun_engine::GROWTH);
        assert_eq!(engine.allocated(), u64::from(engine.cap()) * KV_BYTES);
    }

    #[test]
    fn attach_rolls_back_a_pending_step_first() {
        let mut engine = engine(10_000);
        engine.set_limit(10_000).unwrap();
        prefill(&mut engine, &seq(0..300));
        let tail = engine.detach_tail(100).unwrap();
        let step = engine.forward(5).unwrap();
        engine.sample(step, &Sampling::GREEDY, None).unwrap();
        assert!(engine.has_pending());
        // The confirmed length (101) is not the segment start: refused, nothing changed.
        assert!(matches!(
            engine.attach(tail.id),
            Err(EngineError::InvalidRange { .. })
        ));
        assert!(engine.has_pending());
        // A pending step after the segment start: attach rolls it back instead of committing it.
        engine.trim(99).unwrap();
        let step = engine.forward(99).unwrap();
        engine.sample(step, &Sampling::GREEDY, None).unwrap();
        assert_eq!((engine.len(), engine.has_pending()), (101, true));
        engine.attach(tail.id).unwrap();
        assert!(!engine.has_pending());
        assert_eq!(engine.tokens(), seq(0..300));
        assert_eq!(engine.cap(), 512, "align(end)");
    }

    #[test]
    fn a_failed_move_poisons_the_engine_until_reset_all() {
        let mut engine = engine(1_000);
        engine.set_limit(1_000).unwrap();
        prefill(&mut engine, &seq(0..400));
        engine.fail_next_move = true;
        let error = engine.detach_tail(300).unwrap_err();
        assert!(matches!(error, EngineError::Poisoned(_)));
        assert!(matches!(
            engine.prefill_chunk(&[1], &CancelToken::new()),
            Err(EngineError::Poisoned(_))
        ));
        engine.reset();
        assert!(matches!(engine.forward(1), Err(EngineError::Poisoned(_))));
        engine.reset_all();
        prefill(&mut engine, &[1]);
    }

    #[test]
    fn reset_releases_the_live_state_only() {
        let mut engine = engine(1_000);
        engine.set_limit(1_000).unwrap();
        prefill(&mut engine, &seq(0..400));
        let segment = engine.detach_tail(300).unwrap();
        engine.reset();
        assert_eq!(engine.len(), 0);
        assert_eq!(engine.allocated(), segment.bytes);
        engine.reset_all();
        assert_eq!(engine.allocated(), 0);
        assert_eq!(engine.reset_all_calls.load(Ordering::Relaxed), 1);
    }
}
