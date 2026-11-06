//! The per-request generation loop (D028: the runtime owns it): admission, cache preparation,
//! the prefill chunk loop with checkpoints, memory pressure and `NeedsMemory` retries, pipelined
//! decode with a decode hook and a loop guard, stop and UTF-8 handling, and a finish that leaves
//! the state holding committed tokens only.
//!
//! Failures are typed (D052): a request-level failure (empty grammar mask, detokenizer error,
//! unmet `NeedsMemory`, a reservation overrun) ends that request and keeps the cache; only a
//! backend failure resets the state.

use crate::cache::miss::{Excerpt, MissReport};
use crate::cache::{PrefixStore, Prepared};
use crate::chat::prompt::RenderedPrompt;
use crate::chat::stop::{Fed, StopMatcher};
use crate::chat::tokenizer::StreamDecoder;
use crate::runtime::pressure::PressureGauge;
use mrun_engine::{CancelToken, Checkpoint, Engine, EngineError, Sampling, TokenMask};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Decode tokens between memory-pressure checks.
const PRESSURE_EVERY: u32 = 64;
/// Loop guard (D033): the last `LOOP_WINDOW` output tokens repeating with a period up to
/// `LOOP_MAX_PERIOD` end generation.
const LOOP_WINDOW: usize = 512;
const LOOP_MAX_PERIOD: usize = 128;
/// Tokens between loop checks.
const LOOP_CHECK_EVERY: usize = 16;

/// The tool-call opener and closer tokens: the loop guard ignores everything from an opener to
/// its closer, so long repetitive arguments (file contents, tables) are never cut (D052).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CallFence {
    pub open: u32,
    pub close: u32,
}

/// The loop guard's view of the output: tokens outside tool calls, the last [`LOOP_WINDOW`].
#[derive(Debug, Default)]
struct LoopGuard {
    fence: Option<CallFence>,
    inside: bool,
    window: Vec<u32>,
    seen: usize,
}

impl LoopGuard {
    fn new(fence: Option<CallFence>) -> Self {
        Self {
            fence,
            ..Self::default()
        }
    }

    /// Notes one output token; `true` when the output outside calls is looping.
    fn push(&mut self, id: u32) -> bool {
        if let Some(fence) = self.fence {
            if self.inside {
                self.inside = id != fence.close;
                return false;
            }
            if id == fence.open {
                self.inside = true;
                return false;
            }
        }
        self.window.push(id);
        if self.window.len() >= 2 * LOOP_WINDOW {
            self.window.drain(..LOOP_WINDOW);
        }
        self.seen += 1;
        self.seen.is_multiple_of(LOOP_CHECK_EVERY) && looping(&self.window)
    }
}

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
    /// Sees every output token; returns the mask for the next sampled step.
    fn on_token(&mut self, id: u32) -> Option<TokenMask>;
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
    /// Tool-call delimiters of the model's syntax, for the loop guard.
    pub call_fence: Option<CallFence>,
    /// Who waits for it (026: a pool member's id; `None` for this Mac): waiting requests are
    /// served in turns between them.
    pub turn: Option<String>,
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
            call_fence: None,
            turn: None,
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
    /// The loop guard ended a repeating output (D033).
    Repetition,
    Cancelled,
    /// Refused before any processing.
    Rejected(Rejection),
    /// The request failed after admission.
    Error(Failure),
}

/// Why the runtime refused a request before any processing.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Rejection {
    /// The prompt leaves no room for one output token.
    TooLong { prompt: u32, capacity: u32 },
    /// Invalid input: an empty prompt, a zero output limit, sampling or stop sequences.
    Invalid(String),
}

impl std::fmt::Display for Rejection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TooLong { prompt, capacity } => write!(
                f,
                "{}",
                EngineError::Capacity {
                    requested: prompt.saturating_add(1),
                    available: *capacity,
                }
            ),
            Self::Invalid(message) => f.write_str(message),
        }
    }
}

/// A failure after admission.
#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
pub enum Failure {
    /// Memory for the request's state could not be found, even after evicting everything
    /// stored (unmet `NeedsMemory`, growth past the reservation). With one lane it cannot fit
    /// later either, so the message is the prompt-too-long text naming the tokens needed and
    /// the tokens the state reached ([`out_of_memory`]). The cache is kept.
    #[error("{0}")]
    OutOfMemory(String),
    /// The output grammar allows no token. The cache is kept.
    #[error("the output grammar allows no token")]
    EmptyMask,
    /// The output could not be decoded to text. The cache is kept.
    #[error("cannot decode the output: {0}")]
    Detokenize(String),
    /// The engine failed or its state is unknown; the whole state was reset.
    #[error("{0}")]
    Backend(String),
}

impl From<EngineError> for Failure {
    fn from(error: EngineError) -> Self {
        match error {
            EngineError::NeedsMemory { .. }
            | EngineError::ReservationExceeded { .. }
            | EngineError::Capacity { .. } => Self::OutOfMemory(error.to_string()),
            EngineError::EmptyMask => Self::EmptyMask,
            _ => Self::Backend(error.to_string()),
        }
    }
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
    pub miss: Option<MissReport>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Event {
    Started {
        prompt: u32,
        cached: u32,
        max_tokens: u32,
        miss: Option<MissReport>,
        excerpt: Option<Arc<Excerpt>>,
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

/// Runs one request to completion; the result is its only `Finished`.
pub fn generate<E: Engine>(
    engine: &mut E,
    store: &mut PrefixStore,
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
    let outcome = run(
        engine,
        store,
        detok,
        &mut request,
        sink,
        pressure,
        &mut turn,
    );
    let reason = outcome.unwrap_or_else(|failure| {
        FinishReason::Error(match failure {
            Failure::OutOfMemory(detail) => {
                let needed = request.prompt.len() + turn.output.len() as u32 + 1;
                Failure::OutOfMemory(out_of_memory(needed, engine.len(), &detail))
            }
            other => other,
        })
    });
    let prefilled = match reason {
        FinishReason::Rejected(_) => 0,
        FinishReason::Error(Failure::Backend(_)) => {
            for checkpoint in turn.captured {
                engine.drop_checkpoint(checkpoint.id);
            }
            store.clear(engine);
            // The state is unknown (a move failed half-way poisons the engine, FR-028):
            // `reset_all` clears the poison; the store already holds nothing.
            engine.reset_all();
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
            let reached = store.commit(engine, prompt, &committed, target, turn.captured);
            reached.min(prompt.len())
        }
    };
    Finished {
        reason,
        prefilled,
        stats: turn.stats,
    }
}

/// The message of an unmet `NeedsMemory`: Anthropic's and OpenAI's prompt-too-long text (so
/// clients compact), with `needed` positions against the `reached` ones memory held.
pub fn out_of_memory(needed: u32, reached: u32, detail: &str) -> String {
    format!(
        "prompt is too long: {needed} tokens > {reached} maximum (sequence-state memory ran out \
         at {reached} tokens with the cache shed: {detail})"
    )
}

/// Retries `op` after the store evicts, while `NeedsMemory` can be satisfied; then after
/// dropping this request's own checkpoints, shallowest first: they save later work and must never
/// cost this request its memory.
fn with_memory<E: Engine, T>(
    engine: &mut E,
    store: &mut PrefixStore,
    captured: &mut Vec<Checkpoint>,
    mut op: impl FnMut(&mut E) -> Result<T, EngineError>,
) -> Result<T, Failure> {
    loop {
        match op(engine) {
            Err(EngineError::NeedsMemory { bytes }) if store.evict(engine, bytes, None) > 0 => {}
            Err(EngineError::NeedsMemory { .. }) if !captured.is_empty() => {
                engine.drop_checkpoint(captured.remove(0).id);
            }
            result => return result.map_err(Failure::from),
        }
    }
}

fn relieve_pressure<E: Engine>(engine: &mut E, store: &mut PrefixStore, pressure: &PressureGauge) {
    if let Some(level) = pressure.take() {
        store.on_pressure(engine, level, false);
    }
}

/// Whether the last [`LOOP_WINDOW`] tokens repeat with a period of at most [`LOOP_MAX_PERIOD`].
fn looping(output: &[u32]) -> bool {
    if output.len() < LOOP_WINDOW {
        return false;
    }
    let window = &output[output.len() - LOOP_WINDOW..];
    (1..=LOOP_MAX_PERIOD).any(|p| window[p..].iter().zip(window).all(|(a, b)| a == b))
}

/// Returns the finish reason, or the failure that ended the request.
fn run<E: Engine>(
    engine: &mut E,
    store: &mut PrefixStore,
    detok: &mut StreamDecoder<'_>,
    request: &mut GenerateRequest,
    sink: &mut dyn EventSink,
    pressure: &PressureGauge,
    turn: &mut Turn,
) -> Result<FinishReason, Failure> {
    let prompt = &request.prompt;
    let cancel = &request.cancel;
    let invalid = |message: String| Ok(FinishReason::Rejected(Rejection::Invalid(message)));
    if prompt.is_empty() {
        return invalid("the prompt is empty".into());
    }
    if request.max_tokens == 0 {
        return invalid("max tokens must be at least 1".into());
    }
    if let Err(message) = request.sampling.validate() {
        return invalid(message);
    }
    let mut matcher = match StopMatcher::new(request.stop.clone()) {
        Ok(matcher) => matcher,
        Err(error) => return invalid(error.to_string()),
    };
    // Refuse only a prompt that leaves no room for one output token; clamp the output limit.
    let capacity = engine.budget().kv_capacity_tokens;
    let Some(room) = u32::try_from(prompt.tokens.len())
        .ok()
        .and_then(|len| capacity.checked_sub(len))
        .filter(|room| *room > 0)
    else {
        let prompt = prompt.tokens.len().min(u32::MAX as usize) as u32;
        return Ok(FinishReason::Rejected(Rejection::TooLong {
            prompt,
            capacity,
        }));
    };
    let len = prompt.len();
    let limit = request.max_tokens.min(room);
    turn.stats.max_tokens = limit;
    let total = len + limit;

    // Memory is found when it is needed: moves are memory-neutral, attaching evicts other
    // branches, and prefill growth evicts by rank once the old tail is a branch too.
    relieve_pressure(engine, store, pressure);
    let Prepared {
        resume,
        chunk_ends,
        checkpoints,
        miss,
        excerpt,
    } = store.prepare(engine, prompt, total)?;
    turn.prefilled = resume;
    turn.stats.cached_tokens = resume;
    turn.stats.miss = Some(miss);
    if !sink.send(Event::Started {
        prompt: len,
        cached: resume,
        max_tokens: limit,
        miss: Some(miss),
        excerpt,
    }) {
        return Ok(FinishReason::Cancelled);
    }

    // Prefill all but the last prompt token, one chunk at a time.
    let prefill_started = Instant::now();
    let pending_total = (len - 1).saturating_sub(resume);
    let mut at = resume;
    for end in chunk_ends {
        let chunk = &prompt.tokens[at as usize..end as usize];
        let committed = with_memory(engine, store, &mut turn.captured, |engine| {
            engine.prefill_chunk(chunk, cancel)
        })?;
        if !committed {
            return Ok(FinishReason::Cancelled);
        }
        at = end;
        turn.prefilled = at;
        if checkpoints.contains(&at) {
            let checkpoint = with_memory(engine, store, &mut turn.captured, |engine| {
                engine.checkpoint()
            })?;
            turn.captured.push(checkpoint);
        }
        if !sink.send(Event::Progress {
            done: at - resume,
            total: pending_total,
        }) {
            cancel.cancel();
            return Ok(FinishReason::Cancelled);
        }
        relieve_pressure(engine, store, pressure);
    }

    let eos = engine.info().eos_tokens.clone();
    let mut hook = request.hook.take();
    let ignore_eos = hook.as_ref().is_some_and(|hook| hook.ignore_eos());
    let mut mask = hook.as_mut().and_then(|hook| hook.first_mask());
    let mut last = prompt.tokens[prompt.tokens.len() - 1];
    let mut decode_started = None;
    let mut guard = LoopGuard::new(request.call_fence);
    loop {
        if cancel.is_cancelled() {
            return Ok(FinishReason::Cancelled);
        }
        if turn.stats.generated_tokens.is_multiple_of(PRESSURE_EVERY) {
            relieve_pressure(engine, store, pressure);
        }
        let step = with_memory(engine, store, &mut turn.captured, |engine| {
            engine.forward(last)
        })?;
        if turn.decoding {
            turn.forwarded += 1;
        } else {
            turn.decoding = true;
            turn.prefilled = len;
        }
        let next = engine.sample(step, &request.sampling, mask.take().as_ref())?;
        if decode_started.is_none() {
            turn.stats.ttft = Some(turn.started.elapsed());
            let prefill = prefill_started.elapsed().as_secs_f64();
            if prefill > 0.0 {
                turn.stats.prefill_tps = f64::from(len - resume) / prefill;
            }
            decode_started = Some(Instant::now());
        }
        if !ignore_eos && eos.contains(&next) {
            flush(sink, &mut matcher, turn);
            let reason = FinishReason::Stop {
                stop: None,
                eos_token: Some(next),
            };
            return Ok(finish_decode(turn, decode_started, reason));
        }
        turn.stats.generated_tokens += 1;
        turn.output.push(next);
        if let Some(hook) = &mut hook {
            mask = hook.on_token(next);
        }
        let mut fed = Fed::default();
        let text = detok
            .step(next)
            .map_err(|error| Failure::Detokenize(error.to_string()))?;
        if let Some(text) = text {
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
        // Off with `ignore_eos`: forced-length runs (bench) generate to the limit on purpose.
        if guard.push(next) && !ignore_eos {
            flush(sink, &mut matcher, turn);
            return Ok(finish_decode(
                turn,
                decode_started,
                FinishReason::Repetition,
            ));
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
    use crate::cache::CacheMode;
    use crate::chat::tokenizer::test_support::byte_tokenizer;
    use crate::runtime::pressure::Pressure;
    use crate::runtime::scripted::{KV_BYTES, SCRIPT_EOS, ScriptedEngine};

    const EOS_STOP: FinishReason = FinishReason::Stop {
        stop: None,
        eos_token: Some(SCRIPT_EOS),
    };

    struct Run {
        events: Vec<Event>,
        finished: Finished,
    }

    fn ids(text: &str) -> Vec<u32> {
        byte_tokenizer().encode(text).unwrap()
    }

    fn request(prompt: Vec<u32>, max_tokens: u32) -> GenerateRequest {
        GenerateRequest::new(RenderedPrompt::from_tokens(prompt), max_tokens)
    }

    fn store() -> PrefixStore {
        PrefixStore::new(CacheMode::Cached)
    }

    fn run_on(
        engine: &mut ScriptedEngine,
        store: &mut PrefixStore,
        request: GenerateRequest,
        pressure: &PressureGauge,
    ) -> Run {
        let tokenizer = byte_tokenizer();
        let mut events = Vec::new();
        let finished = generate(
            engine,
            store,
            &mut tokenizer.stream(),
            request,
            &mut events,
            pressure,
        );
        assert_eq!(
            engine.tokens(),
            store.live_tokens(),
            "the store mirrors the state"
        );
        assert!(!engine.has_pending());
        Run { events, finished }
    }

    fn run(engine: &mut ScriptedEngine, store: &mut PrefixStore, request: GenerateRequest) -> Run {
        run_on(engine, store, request, &PressureGauge::default())
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

    #[test]
    fn eos_finishes_and_keeps_prompt_and_output() {
        let mut engine = ScriptedEngine::new(ids("ok"), 1_000);
        let mut store = store();
        let r = run(&mut engine, &mut store, request(ids("hello"), 10));
        assert_eq!(r.finished.reason, EOS_STOP);
        assert_eq!(text(&r.events), "ok");
        assert_eq!(r.finished.stats.generated_tokens, 2);
        assert_eq!(engine.tokens(), ids("hellook"));
    }

    #[test]
    fn a_follow_up_reuses_the_conversation() {
        let mut engine = ScriptedEngine::new(ids("ab"), 1_000);
        let mut store = store();
        run(&mut engine, &mut store, request(ids("hello"), 10));
        let mut next = ids("helloab");
        next.extend(ids(" more"));
        let r = run(&mut engine, &mut store, request(next.clone(), 10));
        assert_eq!(r.finished.stats.cached_tokens, 7);
        assert_eq!(r.finished.stats.miss.unwrap().cause, None);
    }

    #[test]
    fn length_and_stop_sequences_end_generation() {
        let mut engine = ScriptedEngine::new(ids("abcdef"), 1_000);
        let mut store = store();
        let r = run(&mut engine, &mut store, request(ids("x"), 3));
        assert_eq!(r.finished.reason, FinishReason::Length);
        assert_eq!(text(&r.events), "abc");

        let mut engine = ScriptedEngine::new(ids("abcdef"), 1_000);
        let mut store = self::store();
        let mut req = request(ids("x"), 10);
        req.stop = vec!["cd".into()];
        let r = run(&mut engine, &mut store, req);
        assert_eq!(
            r.finished.reason,
            FinishReason::Stop {
                stop: Some("cd".into()),
                eos_token: None
            }
        );
        assert_eq!(text(&r.events), "ab");
        // Tokens whose text was held back by the stop match are not committed.
        assert_eq!(engine.tokens(), ids("xab"));
    }

    #[test]
    fn prompts_that_leave_no_room_are_rejected_and_limits_clamped() {
        let mut engine = ScriptedEngine::new(ids("abc"), 6);
        let mut store = store();
        let r = run(&mut engine, &mut store, request(ids("123456"), 10));
        assert!(matches!(r.finished.reason, FinishReason::Rejected(_)));
        let r = run(&mut engine, &mut store, request(ids("12345"), 10));
        assert_eq!(r.finished.stats.max_tokens, 1);
        assert_eq!(r.finished.reason, FinishReason::Length);
    }

    #[test]
    fn cancel_during_prefill_keeps_whole_chunks() {
        let mut engine = ScriptedEngine::new(vec![], 100_000);
        engine.cancel_after_layers = Some(4 * 2 + 2);
        let mut store = store();
        let prompt: Vec<u32> = (0..5_000).collect();
        let r = run(&mut engine, &mut store, request(prompt, 4));
        assert_eq!(r.finished.reason, FinishReason::Cancelled);
        // Two whole chunks of 2048 committed; the third was discarded.
        assert_eq!(r.finished.prefilled, 4096);
        assert_eq!(engine.len(), 4096);
    }

    #[test]
    fn needs_memory_evicts_branches_and_retries() {
        // A 600-token side conversation fills memory; the next request evicts it.
        let mut engine =
            ScriptedEngine::new(vec![], 10_000).with_state_capacity_bytes(6_000 * KV_BYTES);
        let mut store = store();
        let first: Vec<u32> = (0..600).collect();
        run(&mut engine, &mut store, request(first, 1));
        let other: Vec<u32> = (10_000..11_200).collect();
        let r = run(&mut engine, &mut store, request(other, 1));
        assert!(
            !matches!(r.finished.reason, FinishReason::Error(_)),
            "{:?}",
            r.finished
        );
        assert!(engine.allocated() <= 6_000 * KV_BYTES);
    }

    #[test]
    fn pressure_sheds_branches_between_chunks() {
        let mut engine = ScriptedEngine::new(vec![], 100_000);
        let mut store = store();
        let main: Vec<u32> = (0..1_000).collect();
        run(&mut engine, &mut store, request(main, 1));
        let side: Vec<u32> = (50_000..50_600).collect();
        run(&mut engine, &mut store, request(side, 1));
        assert_eq!(store.branches().len(), 1);
        let gauge = PressureGauge::default();
        gauge.raise(Pressure::Critical);
        let next: Vec<u32> = (60_000..60_010).collect();
        run_on(&mut engine, &mut store, request(next, 1), &gauge);
        // The main conversation's branch was shed; only the side request just detached is left.
        assert_eq!(store.branches(), [(0, 600)]);
        assert_eq!(store.stats().pressure_events, 1);
    }

    #[test]
    fn hybrid_models_checkpoint_at_planned_positions() {
        let mut engine = ScriptedEngine::new_hybrid(vec![], 100_000, 1_000);
        let mut store = store();
        let mut prompt = RenderedPrompt::from_tokens((0..3_000).collect());
        prompt.system_tools_end = 500;
        prompt.generation_start = 2_990;
        let r = run(&mut engine, &mut store, GenerateRequest::new(prompt, 1));
        assert_eq!(r.finished.reason, EOS_STOP);
        assert_eq!(
            engine.checkpoint_count(),
            3,
            "system end, generation start and prompt end − 1"
        );
    }

    #[test]
    fn the_loop_guard_ends_repeating_output() {
        let script: Vec<u32> = ids("ab").into_iter().cycle().take(2_000).collect();
        let mut engine = ScriptedEngine::new(script, 100_000);
        let mut store = store();
        let r = run(&mut engine, &mut store, request(ids("x"), 1_500));
        assert_eq!(r.finished.reason, FinishReason::Repetition);
        assert!(r.finished.stats.generated_tokens <= 528);
        assert!(!looping(&(0..600).collect::<Vec<u32>>()));
    }

    /// A hook that only asks for `ignore_eos`.
    struct IgnoreEos;

    impl DecodeHook for IgnoreEos {
        fn on_token(&mut self, _id: u32) -> Option<TokenMask> {
            None
        }
        fn ignore_eos(&self) -> bool {
            true
        }
    }

    #[test]
    fn xapi_loop_guard_off_with_ignore_eos() {
        // Forced-length runs (bench, FR-031) generate repeating output to the limit.
        let script: Vec<u32> = ids("ab").into_iter().cycle().take(2_000).collect();
        let mut engine = ScriptedEngine::new(script, 100_000);
        let mut store = store();
        let mut req = request(ids("x"), 1_500);
        req.hook = Some(Box::new(IgnoreEos));
        let r = run(&mut engine, &mut store, req);
        assert_eq!(r.finished.reason, FinishReason::Length);
        assert_eq!(r.finished.stats.generated_tokens, 1_500);
    }

    /// A hook whose mask allows nothing after `after` tokens.
    struct Starve {
        after: u32,
        seen: u32,
    }

    impl DecodeHook for Starve {
        fn on_token(&mut self, _id: u32) -> Option<TokenMask> {
            self.seen += 1;
            (self.seen >= self.after).then(|| TokenMask::none_allowed(1_000))
        }
    }

    #[test]
    fn request_failures_keep_the_cache() {
        let mut engine = ScriptedEngine::new(ids("abcdef"), 1_000);
        let mut store = store();
        run(&mut engine, &mut store, request(ids("hello"), 1));
        let mut req = request(ids("hello world"), 10);
        req.hook = Some(Box::new(Starve { after: 2, seen: 0 }));
        let r = run(&mut engine, &mut store, req);
        assert_eq!(r.finished.reason, FinishReason::Error(Failure::EmptyMask));
        // The prompt stays cached: the next request with it reuses it.
        assert_eq!(r.finished.prefilled, ids("hello world").len() as u32);
        let r = run(&mut engine, &mut store, request(ids("hello world!"), 1));
        assert!(
            r.finished.stats.cached_tokens >= 11,
            "{:?}",
            r.finished.stats
        );
    }

    #[test]
    fn a_failed_move_fails_the_request_and_resets_the_engine() {
        let mut engine = ScriptedEngine::new(ids("ok"), 1_000);
        let mut store = store();
        run(&mut engine, &mut store, request(ids("hello"), 10));
        engine.fail_next_move = true;
        let r = run(&mut engine, &mut store, request(ids(&"x".repeat(300)), 10));
        assert!(
            matches!(&r.finished.reason, FinishReason::Error(Failure::Backend(m)) if m.contains("reset_all")),
            "{:?}",
            r.finished.reason
        );
        assert_eq!(r.finished.prefilled, 0);
        // Not poisoned any more: the next request runs from an empty state.
        let r = run(&mut engine, &mut store, request(ids("hello"), 10));
        assert_eq!(r.finished.reason, EOS_STOP);
        assert_eq!(r.finished.stats.cached_tokens, 0);
    }

    #[test]
    fn engine_errors_are_typed() {
        assert_eq!(
            Failure::from(EngineError::NeedsMemory { bytes: 1 }),
            Failure::OutOfMemory("1 more bytes of sequence-state memory are needed".into())
        );
        assert!(matches!(
            Failure::from(EngineError::ReservationExceeded {
                needed: 2,
                reserved: 1
            }),
            Failure::OutOfMemory(_)
        ));
        assert_eq!(Failure::from(EngineError::EmptyMask), Failure::EmptyMask);
        assert!(matches!(
            Failure::from(EngineError::Backend("x".into())),
            Failure::Backend(_)
        ));
        let too_long = Rejection::TooLong {
            prompt: 6,
            capacity: 6,
        };
        assert!(too_long.to_string().contains('7'));
    }

    #[test]
    fn the_loop_guard_ignores_tool_call_arguments() {
        let (open, close) = (ids("<")[0], ids(">")[0]);
        let mut script = vec![open];
        script.extend(ids("ab").into_iter().cycle().take(1_500));
        script.push(close);
        script.extend(ids("done"));
        let mut engine = ScriptedEngine::new(script, 100_000);
        let mut store = store();
        let mut req = request(ids("x"), 2_000);
        req.call_fence = Some(CallFence { open, close });
        let r = run(&mut engine, &mut store, req);
        assert_eq!(r.finished.reason, EOS_STOP);
        assert!(text(&r.events).ends_with(">done"));
    }

    #[test]
    fn invalid_input_is_rejected_before_any_work() {
        let mut engine = ScriptedEngine::new(vec![], 1_000);
        let mut store = store();
        let r = run(&mut engine, &mut store, request(vec![], 4));
        assert!(matches!(r.finished.reason, FinishReason::Rejected(_)));
        let mut bad = request(ids("x"), 4);
        bad.sampling.top_p = 0.0;
        let r = run(&mut engine, &mut store, bad);
        assert!(matches!(r.finished.reason, FinishReason::Rejected(_)));
        assert!(r.events.is_empty());
    }
}
