//! The per-request generation loop, generic over [`Engine`] and a [`ReusePolicy`]: admission,
//! reuse, prefill of all but the last prompt token, pipelined decode with a decode hook, stop and
//! UTF-8 handling, and a finish that leaves the state holding committed tokens only.

use crate::chat::prompt::RenderedPrompt;
use crate::chat::stop::{Fed, StopMatcher};
use crate::chat::tokenizer::StreamDecoder;
use mrun_engine::{CancelToken, Checkpoint, Engine, EngineError, PrefillPlan, Sampling, TokenMask};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{Mutex, PoisonError};
use std::time::{Duration, Instant};

/// Receives events; returns `false` when the consumer is gone (the request is then cancelled).
pub trait EventSink {
    fn send(&mut self, event: Event) -> bool;
}

impl EventSink for Vec<Event> {
    fn send(&mut self, event: Event) -> bool {
        self.push(event);
        true
    }
}

/// Per-request decode logic that runs on the engine thread (tool-call parser, grammar).
pub trait DecodeHook: Send {
    /// Mask for the first sampled step (a grammar armed from the start of the answer).
    fn first_mask(&mut self) -> Option<TokenMask> {
        None
    }
    /// Sees every output token, sampled or forced; returns the mask for the next sampled step.
    fn on_token(&mut self, id: u32) -> Option<TokenMask>;
    /// Tokens to append next without sampling; asked whenever none are queued.
    fn forced_tokens(&mut self) -> Vec<u32> {
        Vec::new()
    }
    /// Treat EOS tokens as ordinary output, so only the output limit ends generation.
    fn ignore_eos(&self) -> bool {
        false
    }
}

impl std::fmt::Debug for dyn DecodeHook {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("DecodeHook")
    }
}

#[derive(Debug)]
pub struct GenerateRequest {
    pub prompt: RenderedPrompt,
    pub sampling: Sampling,
    pub max_tokens: u32,
    pub stop: Vec<String>,
    pub cancel: CancelToken,
    pub hook: Option<Box<dyn DecodeHook>>,
}

impl GenerateRequest {
    /// Greedy request without stop sequences or hook.
    pub fn new(prompt: RenderedPrompt, max_tokens: u32) -> Self {
        Self {
            prompt,
            sampling: Sampling::GREEDY,
            max_tokens,
            stop: Vec::new(),
            cancel: CancelToken::new(),
            hook: None,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum FinishReason {
    /// A stop sequence (`stop`) or the end-of-sequence token that ended generation.
    Stop {
        stop: Option<String>,
        eos_token: Option<u32>,
    },
    Length,
    Cancelled,
    /// Refused before any processing (capacity, invalid input).
    Rejected(String),
    /// The engine failed; the state was reset.
    Error(String),
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Stats {
    pub prompt_tokens: u32,
    pub cached_tokens: u32,
    pub generated_tokens: u32,
    /// Output limit actually applied: the requested one, clamped to the remaining capacity.
    pub max_tokens: u32,
    /// From the start of `generate` (the worker's dequeue) to the first output token.
    pub ttft: Option<Duration>,
    pub prefill_tps: f64,
    pub decode_tps: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Event {
    Started {
        prompt: u32,
        cached: u32,
        max_tokens: u32,
    },
    Progress {
        done: u32,
        total: u32,
    },
    /// One per output token, in order. `text` is what became visible now (possibly empty);
    /// `held` counts bytes of decoded text held back as a possible stop sequence.
    Token {
        id: u32,
        text: String,
        held: u32,
    },
    /// Held text released at a finish without a stop-sequence match.
    Flush {
        text: String,
    },
}

/// The single final report of a request.
#[derive(Clone, Debug, PartialEq)]
pub struct Finished {
    pub reason: FinishReason,
    /// Prompt tokens held in the state when the request ended.
    pub prefilled: u32,
    pub stats: Stats,
}

/// Owned form of a [`PrefillPlan`].
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ChunkPlan {
    pub boundaries: Vec<u32>,
    pub checkpoints: Vec<u32>,
}

impl ChunkPlan {
    pub fn as_plan(&self) -> PrefillPlan<'_> {
        PrefillPlan {
            boundaries: &self.boundaries,
            checkpoints: &self.checkpoints,
        }
    }
}

/// What `prepare` leaves for the loop.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct Prepared {
    /// Prompt tokens the state holds; prefill starts here.
    pub resume: u32,
    pub chunks: ChunkPlan,
}

/// Memory-pressure level reported by the system.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum Pressure {
    Warning = 1,
    Critical = 2,
}

/// Pending memory-pressure event shared between the pressure source and the worker. Raising a
/// level also asks the running request's prefill to yield after its current chunk.
#[derive(Debug, Default)]
pub struct PressureGauge {
    level: AtomicU8,
    running: Mutex<Option<CancelToken>>,
}

impl PressureGauge {
    /// Records `level` (the highest pending one wins) and requests a yield of the running request.
    pub fn raise(&self, level: Pressure) {
        self.level.fetch_max(level as u8, Ordering::AcqRel);
        if let Some(cancel) = &*self.running.lock().unwrap_or_else(PoisonError::into_inner) {
            cancel.request_yield();
        }
    }

    /// The pending level, once.
    pub fn take(&self) -> Option<Pressure> {
        match self.level.swap(0, Ordering::AcqRel) {
            1 => Some(Pressure::Warning),
            2 => Some(Pressure::Critical),
            _ => None,
        }
    }

    fn set_running(&self, cancel: Option<CancelToken>) {
        *self.running.lock().unwrap_or_else(PoisonError::into_inner) = cancel;
    }
}

/// Decides which state a request reuses and what is kept after it.
pub trait ReusePolicy {
    /// Bytes the request will allocate before its first decode step beyond what the state holds:
    /// attention growth to `P + min(output, 4096)` plus planned checkpoints. Pure.
    fn initial_need(&self, engine: &dyn Engine, prompt: &RenderedPrompt, total: u32) -> u64;
    /// Frees cached state, aiming for `bytes`; never the state `protect` would reuse. Returns the
    /// bytes freed (0 when nothing is left).
    fn evict(
        &mut self,
        engine: &mut dyn Engine,
        bytes: u64,
        protect: Option<&RenderedPrompt>,
    ) -> u64;
    /// Leaves the engine holding a prefix of the prompt of at most `P − 1` tokens, reserves
    /// `total` and returns that prefix's length with the prefill plan.
    fn prepare(
        &mut self,
        engine: &mut dyn Engine,
        prompt: &RenderedPrompt,
        total: u32,
    ) -> Result<Prepared, EngineError>;
    /// Acts on memory pressure; resets the live state only when `idle`.
    fn on_pressure(&mut self, engine: &mut dyn Engine, level: Pressure, idle: bool);
    /// Records a finished request. The engine holds `committed[..target]` (plus possibly
    /// uncommitted tokens beyond it); the policy cuts it to a prefix of `committed`, takes
    /// ownership of `captured` and returns the length the state holds.
    fn commit(
        &mut self,
        engine: &mut dyn Engine,
        prompt: &RenderedPrompt,
        committed: &[u32],
        target: u32,
        captured: Vec<Checkpoint>,
    ) -> u32;
    /// Forgets all state (after an engine error or a panic).
    fn clear(&mut self, engine: &mut dyn Engine);
}

/// Progress of one request, used to decide what is committed at the finish.
#[derive(Debug)]
struct Turn {
    /// When `generate` was called (the worker's dequeue).
    started: Instant,
    stats: Stats,
    /// Prompt tokens confirmed in the state.
    prefilled: u32,
    /// The last prompt token has been fed.
    decoding: bool,
    /// Output tokens fed and confirmed.
    forwarded: u32,
    /// Output tokens (an EOS that ended generation excluded).
    output: Vec<u32>,
    /// Byte offset in the decoded text where each output token's text ends, once complete.
    text_ends: Vec<usize>,
    decoded_bytes: usize,
    emitted_bytes: usize,
    captured: Vec<Checkpoint>,
}

impl Turn {
    /// Output tokens whose text was fully emitted.
    fn committed_output(&self) -> usize {
        self.text_ends
            .iter()
            .take_while(|&&end| end <= self.emitted_bytes)
            .count()
    }
}

/// Runs one request to completion; the result is its only `Finished`. `pressure` is read at
/// prefill yields and between decode tokens.
pub fn generate<E: Engine, P: ReusePolicy>(
    engine: &mut E,
    policy: &mut P,
    detok: &mut StreamDecoder<'_>,
    mut request: GenerateRequest,
    sink: &mut dyn EventSink,
    pressure: &PressureGauge,
) -> Finished {
    let mut turn = Turn {
        started: Instant::now(),
        stats: Stats {
            prompt_tokens: request.prompt.len(),
            ..Stats::default()
        },
        prefilled: 0,
        decoding: false,
        forwarded: 0,
        output: Vec::new(),
        text_ends: Vec::new(),
        decoded_bytes: 0,
        emitted_bytes: 0,
        captured: Vec::new(),
    };
    pressure.set_running(Some(request.cancel.clone()));
    let outcome = run(
        engine,
        policy,
        detok,
        &mut request,
        sink,
        pressure,
        &mut turn,
    );
    pressure.set_running(None);
    let reason = outcome.unwrap_or_else(FinishReason::Error);
    let prefilled = match reason {
        FinishReason::Rejected(_) => 0,
        FinishReason::Error(_) => {
            for checkpoint in turn.captured {
                engine.drop_checkpoint(checkpoint.id);
            }
            policy.clear(engine);
            0
        }
        _ => {
            let prompt = &request.prompt;
            let shown = turn.committed_output();
            let mut committed = prompt.tokens.clone();
            committed.extend_from_slice(&turn.output[..shown]);
            let target = if turn.decoding {
                prompt.len() + turn.forwarded.min(shown as u32)
            } else {
                turn.prefilled
            };
            let reached = policy.commit(engine, prompt, &committed, target, turn.captured);
            reached.min(prompt.len())
        }
    };
    Finished {
        reason,
        prefilled,
        stats: turn.stats,
    }
}

/// Retries `op` after the policy evicts, while `NeedsMemory` can be satisfied.
fn with_memory<E: Engine, P: ReusePolicy, T>(
    engine: &mut E,
    policy: &mut P,
    mut op: impl FnMut(&mut E) -> Result<T, EngineError>,
) -> Result<T, String> {
    loop {
        match op(engine) {
            Err(EngineError::NeedsMemory { bytes }) if policy.evict(engine, bytes, None) > 0 => {}
            result => return result.map_err(|error| error.to_string()),
        }
    }
}

fn relieve_pressure<E: Engine, P: ReusePolicy>(
    engine: &mut E,
    policy: &mut P,
    pressure: &PressureGauge,
) {
    if let Some(level) = pressure.take() {
        policy.on_pressure(engine, level, false);
    }
}

/// Returns the finish reason, or `Err` with a message when the engine failed.
fn run<E: Engine, P: ReusePolicy>(
    engine: &mut E,
    policy: &mut P,
    detok: &mut StreamDecoder<'_>,
    request: &mut GenerateRequest,
    sink: &mut dyn EventSink,
    pressure: &PressureGauge,
    turn: &mut Turn,
) -> Result<FinishReason, String> {
    let prompt = &request.prompt;
    let cancel = &request.cancel;
    if prompt.is_empty() {
        return Ok(FinishReason::Rejected("the prompt is empty".into()));
    }
    if request.max_tokens == 0 {
        return Ok(FinishReason::Rejected(
            "max tokens must be at least 1".into(),
        ));
    }
    if let Err(message) = request.sampling.validate() {
        return Ok(FinishReason::Rejected(message));
    }
    let mut matcher = match StopMatcher::new(request.stop.clone()) {
        Ok(matcher) => matcher,
        Err(error) => return Ok(FinishReason::Rejected(error.to_string())),
    };
    // Refuse only a prompt that leaves no room for one output token; clamp the output limit.
    let capacity = engine.budget().kv_capacity_tokens;
    let Some(room) = u32::try_from(prompt.tokens.len())
        .ok()
        .and_then(|len| capacity.checked_sub(len))
        .filter(|room| *room > 0)
    else {
        let requested = (prompt.tokens.len() as u64 + 1).min(u64::from(u32::MAX)) as u32;
        let error = EngineError::Capacity {
            requested,
            available: capacity,
        };
        return Ok(FinishReason::Rejected(error.to_string()));
    };
    let len = prompt.len();
    let limit = request.max_tokens.min(room);
    turn.stats.max_tokens = limit;
    let total = len + limit;

    // Free what the request will allocate before switching state, sparing what it reuses.
    let need = policy.initial_need(engine, prompt, total);
    let free = engine
        .budget()
        .state_capacity_bytes
        .saturating_sub(engine.allocated());
    if need > free {
        policy.evict(engine, need - free, Some(prompt));
    }
    let prepared = policy
        .prepare(engine, prompt, total)
        .map_err(|error| error.to_string())?;
    let keep = prepared.resume;
    turn.prefilled = keep;
    turn.stats.cached_tokens = keep;
    if !sink.send(Event::Started {
        prompt: len,
        cached: keep,
        max_tokens: limit,
    }) {
        return Ok(FinishReason::Cancelled);
    }

    let prefill_started = Instant::now();
    let pending = &prompt.tokens[keep as usize..prompt.tokens.len() - 1];
    let pending_len = pending.len() as u32;
    let mut done = 0;
    let mut consumer_gone = false;
    while done < pending_len {
        let filled = with_memory(engine, policy, |engine| {
            engine.prefill(
                &pending[done as usize..],
                &prepared.chunks.as_plan(),
                cancel,
                &mut |n| {
                    if !consumer_gone
                        && !sink.send(Event::Progress {
                            done: done + n,
                            total: pending_len,
                        })
                    {
                        consumer_gone = true;
                        cancel.cancel();
                    }
                },
            )
        })?;
        done += filled.committed;
        turn.prefilled = keep + done;
        turn.captured.extend(filled.checkpoints);
        if filled.yielded {
            cancel.clear_yield();
            relieve_pressure(engine, policy, pressure);
        } else if done < pending_len {
            return Ok(FinishReason::Cancelled);
        }
    }

    let eos = engine.info().eos_tokens.clone();
    let mut hook = request.hook.take();
    let ignore_eos = hook.as_ref().is_some_and(|hook| hook.ignore_eos());
    let mut mask: Option<TokenMask> = None;
    let mut forced = VecDeque::new();
    let mut last = prompt.tokens[prompt.tokens.len() - 1];
    let mut decode_started = None;
    if let Some(hook) = &mut hook {
        mask = hook.first_mask();
        forced.extend(hook.forced_tokens());
    }
    loop {
        if cancel.is_cancelled() {
            return Ok(FinishReason::Cancelled);
        }
        relieve_pressure(engine, policy, pressure);
        let step = with_memory(engine, policy, |engine| engine.forward(last))?;
        if turn.decoding {
            turn.forwarded += 1;
        } else {
            turn.decoding = true;
            turn.prefilled = len;
        }
        // A forced token needs no sampling: the step's logits are dropped.
        let next = match forced.pop_front() {
            Some(token) => token,
            None => {
                let pipeline = limit - turn.stats.generated_tokens > 1;
                engine
                    .sample(step, &request.sampling, mask.take().as_ref(), pipeline)
                    .map_err(|error| error.to_string())?
            }
        };
        if decode_started.is_none() {
            turn.stats.ttft = Some(turn.started.elapsed());
            let prefill = prefill_started.elapsed().as_secs_f64();
            if prefill > 0.0 {
                turn.stats.prefill_tps = f64::from(len - keep) / prefill;
            }
            decode_started = Some(Instant::now());
        }
        if !ignore_eos && eos.contains(&next) {
            flush(sink, &mut matcher, turn);
            let eos_token = Some(next);
            return Ok(finish_decode(
                turn,
                decode_started,
                FinishReason::Stop {
                    stop: None,
                    eos_token,
                },
            ));
        }
        turn.stats.generated_tokens += 1;
        turn.output.push(next);
        if let Some(hook) = &mut hook {
            mask = hook.on_token(next);
            if forced.is_empty() {
                forced.extend(hook.forced_tokens());
            }
        }
        let mut fed = Fed::default();
        if let Some(text) = detok.step(next).map_err(|error| error.to_string())? {
            turn.decoded_bytes += text.len();
            let ends = turn.output.len() - turn.text_ends.len();
            turn.text_ends
                .extend(std::iter::repeat_n(turn.decoded_bytes, ends));
            fed = matcher.feed(&text);
        }
        if !emit(sink, next, fed.emit, matcher.held(), turn) {
            return Ok(FinishReason::Cancelled);
        }
        if let Some(stop) = fed.stopped {
            let reason = FinishReason::Stop {
                stop: Some(stop),
                eos_token: None,
            };
            return Ok(finish_decode(turn, decode_started, reason));
        }
        if turn.stats.generated_tokens >= limit {
            flush(sink, &mut matcher, turn);
            return Ok(finish_decode(turn, decode_started, FinishReason::Length));
        }
        last = next;
    }
}

/// Reports one output token; `false` when the consumer is gone.
fn emit(sink: &mut dyn EventSink, id: u32, text: String, held: usize, turn: &mut Turn) -> bool {
    let bytes = text.len();
    let delivered = sink.send(Event::Token {
        id,
        text,
        held: held as u32,
    });
    if delivered {
        turn.emitted_bytes += bytes;
    }
    delivered
}

/// Releases held text at a finish without a stop match.
fn flush(sink: &mut dyn EventSink, matcher: &mut StopMatcher, turn: &mut Turn) {
    let text = matcher.finish();
    let bytes = text.len();
    if bytes > 0 && sink.send(Event::Flush { text }) {
        turn.emitted_bytes += bytes;
    }
}

fn finish_decode(
    turn: &mut Turn,
    decode_started: Option<Instant>,
    reason: FinishReason,
) -> FinishReason {
    let stats = &mut turn.stats;
    if let Some(start) = decode_started {
        let elapsed = start.elapsed().as_secs_f64();
        if elapsed > 0.0 && stats.generated_tokens > 1 {
            stats.decode_tps = f64::from(stats.generated_tokens - 1) / elapsed;
        }
    }
    reason
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chat::tokenizer::Tokenizer;
    use crate::chat::tokenizer::test_support::byte_tokenizer;
    use crate::runtime::scripted::{KV_BYTES, SCRIPT_EOS, ScriptedEngine};
    use crate::runtime::session::Session;
    use std::sync::{Arc, Mutex};

    const RB: u64 = 1_000;
    const EOS_STOP: FinishReason = FinishReason::Stop {
        stop: None,
        eos_token: Some(SCRIPT_EOS),
    };

    impl AsRef<Session> for Session {
        fn as_ref(&self) -> &Session {
            self
        }
    }

    struct Run {
        events: Vec<Event>,
        finished: Finished,
        engine: ScriptedEngine,
        session: Session,
    }

    fn ids(text: &str) -> Vec<u32> {
        byte_tokenizer().encode(text).unwrap()
    }

    fn request(prompt: Vec<u32>, max_tokens: u32) -> GenerateRequest {
        GenerateRequest::new(RenderedPrompt::from_tokens(prompt), max_tokens)
    }

    fn generate_with<P: ReusePolicy + AsRef<Session>>(
        engine: &mut ScriptedEngine,
        policy: &mut P,
        tokenizer: &Tokenizer,
        request: GenerateRequest,
        sink: &mut dyn EventSink,
        pressure: &PressureGauge,
    ) -> Finished {
        let finished = generate(
            engine,
            policy,
            &mut tokenizer.stream(),
            request,
            sink,
            pressure,
        );
        let session = policy.as_ref();
        assert_eq!(
            engine.tokens(),
            session.tokens(),
            "session mirrors the state"
        );
        assert!(!engine.has_pending());
        assert!(session.committed().starts_with(session.tokens()));
        finished
    }

    fn generate_into(
        engine: &mut ScriptedEngine,
        session: &mut Session,
        tokenizer: &Tokenizer,
        request: GenerateRequest,
        sink: &mut dyn EventSink,
    ) -> Finished {
        let pressure = PressureGauge::default();
        generate_with(engine, session, tokenizer, request, sink, &pressure)
    }

    fn run_with(engine: ScriptedEngine, session: Session, request: GenerateRequest) -> Run {
        let (mut engine, mut session) = (engine, session);
        let mut events = Vec::new();
        let finished = generate_into(
            &mut engine,
            &mut session,
            &byte_tokenizer(),
            request,
            &mut events,
        );
        Run {
            events,
            finished,
            engine,
            session,
        }
    }

    fn text(events: &[Event]) -> String {
        events
            .iter()
            .filter_map(|event| match event {
                Event::Token { text, .. } | Event::Flush { text } => Some(text.as_str()),
                _ => None,
            })
            .collect()
    }

    fn concat(parts: &[&[u32]]) -> Vec<u32> {
        parts.concat()
    }

    #[test]
    fn streams_text_until_eos_and_reports_stats() {
        let run = run_with(
            ScriptedEngine::new(ids("Hi €!"), 1_000),
            Session::default(),
            request(ids("hello"), 50),
        );
        assert_eq!(text(&run.events), "Hi €!");
        let Finished {
            reason,
            prefilled,
            stats,
        } = &run.finished;
        assert_eq!(*reason, EOS_STOP);
        assert_eq!(*prefilled, 5);
        assert_eq!((stats.prompt_tokens, stats.generated_tokens), (5, 7));
        assert_eq!(stats.max_tokens, 50);
        assert!(stats.ttft.is_some());
        assert_eq!(
            run.events[0],
            Event::Started {
                prompt: 5,
                cached: 0,
                max_tokens: 50
            }
        );
        // The pipelined EOS step was trimmed; prompt and reply stay.
        assert_eq!(run.session.tokens(), ids("helloHi €!"));
    }

    #[test]
    fn follow_up_turn_processes_only_new_tokens() {
        let first = run_with(
            ScriptedEngine::new(ids("ok"), 1_000),
            Session::default(),
            request(ids("abc"), 10),
        );
        let prompt = concat(&[first.session.tokens(), &ids(" xyz")]);
        let mut engine = first.engine;
        engine.processed.clear();
        let second = run_with(engine, first.session, request(prompt.clone(), 10));
        assert_eq!(
            second.events[0],
            Event::Started {
                prompt: prompt.len() as u32,
                cached: 5,
                max_tokens: 10
            }
        );
        assert!(second.engine.processed.starts_with(&prompt[5..]));
    }

    #[test]
    fn edited_history_reuses_only_the_common_prefix() {
        let first = run_with(
            ScriptedEngine::new(Vec::new(), 1_000),
            Session::default(),
            request(ids("abcdef"), 1),
        );
        let second = run_with(first.engine, first.session, request(ids("abXdef"), 1));
        assert_eq!(
            second.events[0],
            Event::Started {
                prompt: 6,
                cached: 2,
                max_tokens: 1
            }
        );
    }

    #[test]
    fn identical_prompt_reprocesses_only_the_last_token() {
        let prompt = ids("same prompt");
        let first = run_with(
            ScriptedEngine::new(Vec::new(), 1_000),
            Session::default(),
            request(prompt.clone(), 1),
        );
        let mut engine = first.engine;
        engine.processed.clear();
        let second = run_with(engine, first.session, request(prompt.clone(), 1));
        assert_eq!(second.engine.processed, [*prompt.last().unwrap()]);
    }

    #[test]
    fn stop_sequence_is_neither_shown_nor_committed() {
        let mut req = request(ids("q"), 50);
        req.stop = vec!["</done>".into()];
        let run = run_with(
            ScriptedEngine::new(ids("answer</done>more"), 1_000),
            Session::default(),
            req,
        );
        assert_eq!(text(&run.events), "answer");
        assert_eq!(
            run.finished.reason,
            FinishReason::Stop {
                stop: Some("</done>".into()),
                eos_token: None
            }
        );
        assert_eq!(run.session.committed(), ids("qanswer"));
        assert_eq!(run.session.tokens(), ids("qanswer"));
    }

    #[test]
    fn output_limit_ends_with_length_and_keeps_the_fed_tokens() {
        let run = run_with(
            ScriptedEngine::new(ids("abcdefgh"), 1_000),
            Session::default(),
            request(ids("q"), 3),
        );
        assert_eq!(text(&run.events), "abc");
        assert_eq!(run.finished.reason, FinishReason::Length);
        assert_eq!(run.finished.stats.generated_tokens, 3);
        assert_eq!(run.session.committed(), ids("qabc"));
        // The last sampled token was never fed.
        assert_eq!(run.session.tokens(), ids("qab"));
    }

    #[test]
    fn incomplete_character_is_neither_shown_nor_committed() {
        let run = run_with(
            ScriptedEngine::new(ids("a€"), 1_000),
            Session::default(),
            request(ids("q"), 2),
        );
        assert_eq!(text(&run.events), "a");
        assert_eq!(run.session.committed(), ids("qa"));
    }

    #[test]
    fn output_limit_is_clamped_to_capacity() {
        let run = run_with(
            ScriptedEngine::new(ids("xyz"), 12),
            Session::default(),
            request(ids("0123456789"), 100),
        );
        assert_eq!(run.finished.reason, FinishReason::Length);
        assert_eq!(run.finished.stats.max_tokens, 2);
        assert_eq!(run.finished.stats.generated_tokens, 2);
    }

    #[test]
    fn prompt_without_room_for_output_is_rejected() {
        let prompt = ids("0123456789");
        let at = run_with(
            ScriptedEngine::new(ids("xyz"), 11),
            Session::default(),
            request(prompt.clone(), 3),
        );
        assert_eq!(at.finished.reason, FinishReason::Length);
        let over = run_with(
            ScriptedEngine::new(ids("xyz"), 10),
            Session::default(),
            request(prompt, 3),
        );
        assert!(
            matches!(&over.finished.reason, FinishReason::Rejected(message) if message.contains("11 tokens"))
        );
        assert_eq!(over.finished.prefilled, 0);
        assert!(over.engine.processed.is_empty(), "nothing processed");
    }

    #[test]
    fn rejected_request_keeps_the_cached_conversation() {
        let first = run_with(
            ScriptedEngine::new(Vec::new(), 20),
            Session::default(),
            request(ids("abcdefgh"), 1),
        );
        let cached = first.session.tokens().to_vec();
        let rejected = run_with(
            first.engine,
            first.session,
            request(ids("0123456789abcdefghijk"), 1),
        );
        assert!(matches!(
            rejected.finished.reason,
            FinishReason::Rejected(_)
        ));
        assert_eq!(rejected.session.tokens(), cached);
    }

    #[test]
    fn invalid_requests_are_rejected_up_front() {
        let mut bad_stop = request(ids("q"), 5);
        bad_stop.stop = vec!["s".into(); 5];
        let mut bad_sampling = request(ids("q"), 5);
        bad_sampling.sampling.top_p = 0.0;
        for req in [
            request(Vec::new(), 5),
            request(ids("q"), 0),
            bad_stop,
            bad_sampling,
        ] {
            let run = run_with(
                ScriptedEngine::new(Vec::new(), 100),
                Session::default(),
                req,
            );
            assert!(matches!(run.finished.reason, FinishReason::Rejected(_)));
        }
    }

    #[test]
    fn cancel_mid_chunk_keeps_whole_chunks() {
        let mut engine = ScriptedEngine::new_hybrid(ids("never"), 10_000, RB);
        // Four layers per chunk: layer 6 is inside the second chunk.
        engine.cancel_after_layers = Some(6);
        let prompt = ids(&"x".repeat(1_000));
        let mut req = request(prompt.clone(), 10);
        req.prompt.generation_start = 400;
        let run = run_with(engine, Session::default(), req);
        assert_eq!(run.finished.reason, FinishReason::Cancelled);
        assert_eq!(run.finished.prefilled, 400);
        assert_eq!(run.session.tokens(), &prompt[..400]);
        assert_eq!(run.session.checkpoints().len(), 1);
        assert!(text(&run.events).is_empty());
    }

    #[test]
    fn cancel_during_decode_keeps_exactly_the_shown_tokens() {
        struct CancelAfterTokens(Vec<Event>, CancelToken);
        impl EventSink for CancelAfterTokens {
            fn send(&mut self, event: Event) -> bool {
                self.0.push(event);
                if self.0.len() == 3 {
                    self.1.cancel();
                }
                true
            }
        }
        let tokenizer = byte_tokenizer();
        let req = request(ids("q"), 50);
        let mut sink = CancelAfterTokens(Vec::new(), req.cancel.clone());
        let mut engine = ScriptedEngine::new(ids("abcdef"), 1_000);
        let mut session = Session::default();
        let finished = generate_into(&mut engine, &mut session, &tokenizer, req, &mut sink);
        assert_eq!(finished.reason, FinishReason::Cancelled);
        assert_eq!(text(&sink.0), "ab");
        assert_eq!(session.committed(), ids("qab"));
        // "b" was only fed as a pending pipelined step.
        assert_eq!(session.tokens(), ids("qa"));
    }

    #[test]
    fn consumer_gone_cancels() {
        struct Gone;
        impl EventSink for Gone {
            fn send(&mut self, _: Event) -> bool {
                false
            }
        }
        let tokenizer = byte_tokenizer();
        let finished = generate_into(
            &mut ScriptedEngine::new(ids("abc"), 100),
            &mut Session::default(),
            &tokenizer,
            request(ids("q"), 10),
            &mut Gone,
        );
        assert_eq!(finished.reason, FinishReason::Cancelled);
    }

    #[test]
    fn needs_memory_without_cache_to_evict_fails_and_resets() {
        let first = run_with(
            ScriptedEngine::new(ids("ok"), 10_000).with_state_capacity_bytes(300 * KV_BYTES),
            Session::default(),
            request(ids("abc"), 5),
        );
        assert_eq!(first.finished.reason, EOS_STOP);
        let failed = run_with(
            first.engine,
            first.session,
            request(ids("abcdefghijklmnop"), 1_000),
        );
        assert!(
            matches!(&failed.finished.reason, FinishReason::Error(message) if message.contains("bytes")),
            "{:?}",
            failed.finished.reason
        );
        assert_eq!(failed.finished.prefilled, 0);
        assert!(failed.session.tokens().is_empty());
        assert_eq!(failed.engine.allocated(), 0);
    }

    #[test]
    fn needs_memory_evicts_the_older_session_checkpoint() {
        // Room for the live state plus two checkpoints, not three.
        let mut engine = ScriptedEngine::new_hybrid(Vec::new(), 1_000, RB)
            .with_state_capacity_bytes(1_000 + 3 * RB);
        let mut session = Session::default();
        let tokenizer = byte_tokenizer();
        let mut prompt = Vec::new();
        for turn in 0..3 {
            let start = prompt.len() as u32 + 3;
            prompt.extend(ids("user"));
            prompt.extend(ids("|reply"));
            let mut req = request(prompt.clone(), 1);
            req.prompt.generation_start = start;
            let finished =
                generate_into(&mut engine, &mut session, &tokenizer, req, &mut Vec::new());
            assert_eq!(finished.reason, EOS_STOP, "turn {turn}");
            assert!(session.checkpoints().len() <= 2);
        }
        assert_eq!(engine.checkpoint_count(), 2);
        let positions: Vec<_> = session.checkpoints().iter().map(|c| c.position).collect();
        assert_eq!(positions, [13, 23]);
    }

    #[test]
    fn hybrid_stop_sequence_falls_back_to_the_generation_checkpoint() {
        let prompt = ids("<sys><user>hi<asst>");
        let start = 15;
        let mut req = request(prompt.clone(), 50);
        req.stop = vec!["</done>".into()];
        req.prompt.generation_start = start;
        let run = run_with(
            ScriptedEngine::new_hybrid(ids("answer</done>"), 1_000, RB),
            Session::default(),
            req,
        );
        assert_eq!(text(&run.events), "answer");
        assert_eq!(run.session.committed(), concat(&[&prompt, &ids("answer")]));
        assert_eq!(run.session.tokens(), &prompt[..start as usize]);
    }

    #[test]
    fn hybrid_next_turn_resumes_from_the_deepest_checkpoint() {
        let first_prompt = ids("<user>hi<asst>");
        let mut req = request(first_prompt.clone(), 50);
        req.prompt.generation_start = 8;
        let first = run_with(
            ScriptedEngine::new_hybrid(ids("hello there"), 1_000, RB),
            Session::default(),
            req,
        );
        assert_eq!(first.session.tokens(), ids("<user>hi<asst>hello there"));

        // The template re-renders the reply differently: divergence inside it.
        let prompt = ids("<user>hi<asst>hello world<user>more<asst>");
        let mut req = request(prompt.clone(), 1);
        req.prompt.generation_start = prompt.len() as u32 - 6;
        let mut engine = first.engine;
        engine.processed.clear();
        let second = run_with(engine, first.session, req);
        assert_eq!(
            second.events[0],
            Event::Started {
                prompt: prompt.len() as u32,
                cached: 8,
                max_tokens: 1
            }
        );
        assert!(second.engine.processed.starts_with(&prompt[8..]));
        assert_eq!(second.finished.reason, EOS_STOP);
        let positions: Vec<_> = second
            .session
            .checkpoints()
            .iter()
            .map(|c| c.position)
            .collect();
        assert_eq!(positions, [8, prompt.len() as u32 - 6]);
    }

    #[test]
    fn yield_resumes_prefill_with_identical_chunks() {
        let prompt = ids(&"y".repeat(500));
        let chunks = |yield_after: Option<u64>| {
            let mut engine = ScriptedEngine::new_hybrid(Vec::new(), 1_000, RB);
            engine.yield_after_layers = yield_after;
            let mut req = request(prompt.clone(), 1);
            req.prompt.generation_start = 200;
            let run = run_with(engine, Session::default(), req);
            assert_eq!(run.finished.reason, EOS_STOP);
            run.engine.chunks
        };
        let straight = chunks(None);
        assert_eq!(straight, [(0, 200), (200, 499)]);
        assert_eq!(chunks(Some(2)), straight);
    }

    /// Session that records what the loop asks of it.
    #[derive(Debug, Default)]
    struct Recording {
        session: Session,
        /// `(bytes, protect given)` per `evict`.
        evictions: Vec<(u64, bool)>,
        /// `(level, active_len)` per `on_pressure`.
        pressure: Vec<(Pressure, u32)>,
        prepare_delay: Duration,
    }

    impl AsRef<Session> for Recording {
        fn as_ref(&self) -> &Session {
            &self.session
        }
    }

    impl ReusePolicy for Recording {
        fn initial_need(&self, engine: &dyn Engine, prompt: &RenderedPrompt, total: u32) -> u64 {
            self.session.initial_need(engine, prompt, total)
        }
        fn evict(
            &mut self,
            engine: &mut dyn Engine,
            bytes: u64,
            protect: Option<&RenderedPrompt>,
        ) -> u64 {
            self.evictions.push((bytes, protect.is_some()));
            self.session.evict(engine, bytes, protect)
        }
        fn prepare(
            &mut self,
            engine: &mut dyn Engine,
            prompt: &RenderedPrompt,
            total: u32,
        ) -> Result<Prepared, EngineError> {
            std::thread::sleep(self.prepare_delay);
            self.session.prepare(engine, prompt, total)
        }
        fn on_pressure(&mut self, engine: &mut dyn Engine, level: Pressure, idle: bool) {
            assert!(!idle, "never idle inside a request");
            self.pressure.push((level, engine.active_len()));
            self.session.on_pressure(engine, level, idle);
        }
        fn commit(
            &mut self,
            engine: &mut dyn Engine,
            prompt: &RenderedPrompt,
            committed: &[u32],
            target: u32,
            captured: Vec<Checkpoint>,
        ) -> u32 {
            self.session
                .commit(engine, prompt, committed, target, captured)
        }
        fn clear(&mut self, engine: &mut dyn Engine) {
            self.session.clear(engine);
        }
    }

    /// Hook that records the tokens it sees and forces or masks on cue.
    #[derive(Debug, Default)]
    struct ScriptHook {
        seen: Arc<Mutex<Vec<u32>>>,
        /// Forced once after this token is seen.
        force_after: Option<(u32, Vec<u32>)>,
        /// Mask for the step after this token: allows only the listed tokens.
        mask_after: Option<(u32, Vec<u32>)>,
        ignore_eos: bool,
        queued: Vec<u32>,
    }

    impl DecodeHook for ScriptHook {
        fn on_token(&mut self, id: u32) -> Option<TokenMask> {
            self.seen.lock().unwrap().push(id);
            if let Some((_, tokens)) = self.force_after.take_if(|(after, _)| *after == id) {
                self.queued = tokens;
            }
            let (after, allowed) = self.mask_after.as_ref()?;
            (*after == id).then(|| {
                let mut mask = TokenMask::none_allowed(1024);
                allowed.iter().for_each(|&token| mask.allow(token));
                mask
            })
        }
        fn forced_tokens(&mut self) -> Vec<u32> {
            std::mem::take(&mut self.queued)
        }
        fn ignore_eos(&self) -> bool {
            self.ignore_eos
        }
    }

    fn with_hook(prompt: &str, max_tokens: u32, hook: ScriptHook) -> GenerateRequest {
        let mut req = request(ids(prompt), max_tokens);
        req.hook = Some(Box::new(hook));
        req
    }

    fn token_ids(events: &[Event]) -> Vec<u32> {
        events
            .iter()
            .filter_map(|event| match event {
                Event::Token { id, .. } => Some(*id),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn every_output_token_is_reported_with_its_held_bytes() {
        let mut req = request(ids("q"), 50);
        req.stop = vec!["##".into()];
        let run = run_with(
            ScriptedEngine::new(ids("a€#"), 1_000),
            Session::default(),
            req,
        );
        let tokens: Vec<_> = run
            .events
            .iter()
            .filter(|event| !matches!(event, Event::Started { .. }))
            .cloned()
            .collect();
        let euro = ids("€");
        let token = |id, text: &str, held| Event::Token {
            id,
            text: text.into(),
            held,
        };
        assert_eq!(
            tokens,
            [
                token(ids("a")[0], "a", 0),
                token(euro[0], "", 0),
                token(euro[1], "", 0),
                token(euro[2], "€", 0),
                token(ids("#")[0], "", 1),
                Event::Flush { text: "#".into() },
            ]
        );
        assert_eq!(run.finished.reason, EOS_STOP);
        assert_eq!(run.session.committed(), ids("qa€#"));
    }

    #[test]
    fn hook_masks_apply_to_the_next_sampled_step() {
        let (a, b) = (ids("a")[0], ids("b")[0]);
        let allowed = ScriptHook {
            mask_after: Some((a, vec![b])),
            ..ScriptHook::default()
        };
        let run = run_with(
            ScriptedEngine::new(ids("ab"), 1_000),
            Session::default(),
            with_hook("q", 10, allowed),
        );
        assert_eq!(run.finished.reason, EOS_STOP);
        // The scripted engine fails when the token it would sample is masked out.
        let blocked = ScriptHook {
            mask_after: Some((a, vec![a])),
            ..ScriptHook::default()
        };
        let run = run_with(
            ScriptedEngine::new(ids("ab"), 1_000),
            Session::default(),
            with_hook("q", 10, blocked),
        );
        assert!(
            matches!(&run.finished.reason, FinishReason::Error(message) if message.contains("masked")),
            "{:?}",
            run.finished.reason
        );
    }

    #[test]
    fn forced_tokens_are_appended_without_sampling() {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let hook = ScriptHook {
            seen: seen.clone(),
            force_after: Some((ids("a")[0], ids("XY"))),
            ..ScriptHook::default()
        };
        let run = run_with(
            ScriptedEngine::new(ids("ab"), 1_000),
            Session::default(),
            with_hook("q", 10, hook),
        );
        assert_eq!(text(&run.events), "aXYb");
        assert_eq!(*seen.lock().unwrap(), ids("aXYb"));
        assert_eq!(token_ids(&run.events), ids("aXYb"));
        assert_eq!(run.finished.stats.generated_tokens, 4);
        assert!(run.engine.processed.starts_with(&ids("qaXYb")));
        assert_eq!(run.session.tokens(), ids("qaXYb"));
    }

    #[test]
    fn ignore_eos_runs_to_the_output_limit() {
        let hook = ScriptHook {
            ignore_eos: true,
            ..ScriptHook::default()
        };
        let run = run_with(
            ScriptedEngine::new(ids("a"), 1_000),
            Session::default(),
            with_hook("q", 3, hook),
        );
        assert_eq!(run.finished.reason, FinishReason::Length);
        assert_eq!(
            token_ids(&run.events),
            [ids("a")[0], SCRIPT_EOS, SCRIPT_EOS]
        );
    }

    #[test]
    fn pressure_gauge_is_edge_triggered_and_yields_the_running_request() {
        let gauge = PressureGauge::default();
        assert_eq!(gauge.take(), None);
        gauge.raise(Pressure::Critical);
        gauge.raise(Pressure::Warning);
        assert_eq!(gauge.take(), Some(Pressure::Critical));
        assert_eq!(gauge.take(), None);
        let cancel = CancelToken::new();
        gauge.set_running(Some(cancel.clone()));
        gauge.raise(Pressure::Warning);
        assert!(cancel.yield_requested());
    }

    #[test]
    fn pressure_is_handled_at_prefill_yields_and_between_tokens() {
        let prompt = ids(&"p".repeat(5_000));
        let mut engine = ScriptedEngine::new(ids("abc"), 10_000);
        // Four layers per chunk: the yield lands in the first 2,048-token chunk.
        engine.yield_after_layers = Some(2);
        let mut policy = Recording::default();
        let pressure = PressureGauge::default();
        pressure.raise(Pressure::Warning);
        struct RaiseOnToken<'a>(Vec<Event>, &'a PressureGauge);
        impl EventSink for RaiseOnToken<'_> {
            fn send(&mut self, event: Event) -> bool {
                if matches!(event, Event::Token { .. }) {
                    self.1.raise(Pressure::Critical);
                }
                self.0.push(event);
                true
            }
        }
        let mut sink = RaiseOnToken(Vec::new(), &pressure);
        let finished = generate_with(
            &mut engine,
            &mut policy,
            &byte_tokenizer(),
            request(prompt.clone(), 2),
            &mut sink,
            &pressure,
        );
        assert_eq!(finished.reason, FinishReason::Length);
        let unchunked = ScriptedEngine::new(Vec::new(), 10_000);
        let mut straight = run_with(unchunked, Session::default(), request(prompt, 1)).engine;
        straight.chunks.truncate(engine.chunks.len());
        assert_eq!(engine.chunks, straight.chunks, "same split after the yield");
        // Warning at the yield (after the first chunk), Critical before the second token.
        assert_eq!(
            policy.pressure,
            [(Pressure::Warning, 2_048), (Pressure::Critical, 5_001)]
        );
    }

    #[test]
    fn initial_need_is_evicted_before_prepare_sparing_the_prompt() {
        // Room for the live state plus two checkpoints, not three.
        let mut engine = ScriptedEngine::new_hybrid(Vec::new(), 1_000, RB)
            .with_state_capacity_bytes(1_000 + 3 * RB);
        let mut policy = Recording::default();
        let tokenizer = byte_tokenizer();
        let mut prompt = Vec::new();
        for _ in 0..3 {
            let start = prompt.len() as u32 + 3;
            prompt.extend(ids("user|reply"));
            let mut req = request(prompt.clone(), 1);
            req.prompt.generation_start = start;
            let pressure = PressureGauge::default();
            let finished = generate_with(
                &mut engine,
                &mut policy,
                &tokenizer,
                req,
                &mut Vec::new(),
                &pressure,
            );
            assert_eq!(finished.reason, EOS_STOP);
        }
        // Turn 3 grows attention from 21 to 31 positions (160 bytes) and plans a checkpoint
        // (RB) with 664 bytes free: the shortfall is evicted up front, never on `NeedsMemory`.
        assert_eq!(policy.evictions, [(160 + RB - 664, true)]);
        assert_eq!(policy.session.checkpoints().len(), 2);
    }

    #[test]
    fn session_initial_need_counts_growth_and_planned_checkpoints() {
        let session = Session::default();
        let plain = ScriptedEngine::new(Vec::new(), 100_000);
        let prompt = RenderedPrompt::from_tokens(ids("0123456789"));
        assert_eq!(session.initial_need(&plain, &prompt, 100), 100 * KV_BYTES);
        assert_eq!(
            session.initial_need(&plain, &prompt, 50_000),
            (10 + 4096) * KV_BYTES
        );
        let hybrid = ScriptedEngine::new_hybrid(Vec::new(), 100_000, RB);
        let mut prompt = prompt;
        prompt.generation_start = 5;
        assert_eq!(
            session.initial_need(&hybrid, &prompt, 100),
            100 * KV_BYTES + 2 * RB,
            "live recurrent state plus the planned checkpoint"
        );
    }

    #[test]
    fn ttft_counts_from_the_call() {
        let mut policy = Recording {
            prepare_delay: Duration::from_millis(20),
            ..Recording::default()
        };
        let finished = generate_with(
            &mut ScriptedEngine::new(ids("a"), 100),
            &mut policy,
            &byte_tokenizer(),
            request(ids("q"), 5),
            &mut Vec::new(),
            &PressureGauge::default(),
        );
        assert!(finished.stats.ttft.unwrap() >= Duration::from_millis(20));
    }
}
