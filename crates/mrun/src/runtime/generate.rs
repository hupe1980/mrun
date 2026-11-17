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
use crate::runtime::draft::{Drafter, Pace};
use crate::runtime::pressure::PressureGauge;
use mrun_engine::{CancelToken, Checkpoint, Engine, EngineError, Sampling, TokenMask};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Decode tokens between memory-pressure checks.
const PRESSURE_EVERY: u32 = 64;
/// Tokens the engine's drafter (030) drafts per round: verifying up to 4 rows reads each weight
/// once with row-exact products, so 3 drafts cost little more than a step (measured on Qwen3.5-9B
/// against 5 and 7). A full prompt-lookup draft still takes the engine's maximum.
const MODEL_DRAFT: usize = 3;
/// Loop guard (D033): the last `LOOP_WINDOW` output tokens repeating with a period up to
/// `LOOP_MAX_PERIOD` (two full copies at least: a looping paragraph, not only a phrase) end
/// generation.
const LOOP_WINDOW: usize = 512;
const LOOP_MAX_PERIOD: usize = 256;
/// Tokens between loop checks.
const LOOP_CHECK_EVERY: usize = 16;

/// The tool-call opener and closer tokens: the loop guard ignores everything from an opener to
/// its closer, so long repetitive arguments (file contents, tables) are never cut (D052). Inside
/// the reasoning (`think`: its open and close tokens; `thinking`: the reply starts in it) an
/// opener is only text: a model thinking about tools must not switch the guard off.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CallFence {
    pub open: u32,
    pub close: u32,
    pub think: Option<(u32, u32)>,
    pub thinking: bool,
}

/// The loop guard's view of the output: tokens outside tool calls, the last [`LOOP_WINDOW`].
#[derive(Debug, Default)]
struct LoopGuard {
    fence: Option<CallFence>,
    inside: bool,
    /// In the reasoning, where a call opener is text.
    thinking: bool,
    window: Vec<u32>,
    seen: usize,
}

impl LoopGuard {
    fn new(fence: Option<CallFence>) -> Self {
        Self {
            fence,
            thinking: fence.is_some_and(|f| f.thinking),
            ..Self::default()
        }
    }

    /// Notes one output token; `true` when the output outside calls is looping.
    fn push(&mut self, id: u32) -> bool {
        if let Some((open, close)) = self.fence.and_then(|f| f.think) {
            if id == open {
                self.thinking = true;
            } else if id == close {
                self.thinking = false;
            }
        }
        if let Some(fence) = self.fence.filter(|_| !self.thinking) {
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
    /// Delivers events held for a consumer that was not reading (lanes never wait on one).
    fn flush(&mut self) -> Delivery {
        Delivery::Done
    }
}

/// Whether a sink's held events reached the consumer.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Delivery {
    Done,
    /// Some still wait, since this instant.
    Waiting(Instant),
    /// The consumer is gone.
    Gone,
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
    /// Whether any step may get a mask. Speculative decoding (006) samples drafted positions
    /// without one, so it runs only under hooks that never mask.
    fn masks(&self) -> bool {
        true
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
    /// Speculative decoding may run (006; the engine handle also applies the server's setting).
    pub speculate: bool,
    /// A side request (`x-claude-code-request-class: auxiliary`): admitted before normal ones
    /// when a lane frees (011 FR-005).
    pub auxiliary: bool,
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
            speculate: true,
            auxiliary: false,
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
    /// An image could not be read (009). The cache is kept.
    #[error("{0}")]
    Image(String),
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
    /// Speculative decoding (006): draft tokens verified, kept, and verify passes.
    pub drafted: u32,
    pub accepted: u32,
    pub passes: u32,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Event {
    Started {
        prompt: u32,
        cached: u32,
        max_tokens: u32,
        miss: Option<MissReport>,
        excerpt: Option<Arc<Excerpt>>,
        /// The lane it runs on (011).
        lane: u32,
    },
    Progress {
        done: u32,
        total: u32,
    },
    /// A verify pass (006) ran: `drafted` tokens checked, `accepted` kept; its tokens follow.
    Verified {
        drafted: u32,
        accepted: u32,
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

/// Runs one request to completion on the store's selected lane; the result is its only
/// `Finished`. The same steps the scheduler runs ([`crate::runtime::lanes`]) with one lane.
pub fn generate<E: Engine>(
    engine: &mut E,
    store: &mut PrefixStore,
    detok: &mut StreamDecoder<'_>,
    request: GenerateRequest,
    sink: &mut dyn EventSink,
    pressure: &PressureGauge,
) -> Finished {
    let lane = store.lane();
    let job = Job::new(request, lane, Box::new(Borrowed(sink)), detok.clone());
    crate::runtime::lanes::run_one(engine, store, job, pressure)
}

/// A borrowed sink as a job's own.
struct Borrowed<'s>(&'s mut dyn EventSink);

impl EventSink for Borrowed<'_> {
    fn send(&mut self, event: Event) -> bool {
        self.0.send(event)
    }

    fn flush(&mut self) -> Delivery {
        self.0.flush()
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
pub(crate) fn with_memory<E: Engine, T>(
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

pub(crate) fn relieve_pressure<E: Engine>(
    engine: &mut E,
    store: &mut PrefixStore,
    pressure: &PressureGauge,
) {
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

/// What a step of a job left: go on, or it ended (with its finish reason, or a failure).
pub type Next = Result<Option<FinishReason>, Failure>;

/// One request on one lane (011), advanced a step at a time by the scheduler: [`Job::start`],
/// then each prefill chunk ([`Job::next_chunk`], [`Job::chunk_done`]), then decode rounds, then
/// [`Job::finish`]. Every call expects the job's lane selected in the store and the engine.
pub struct Job<'a> {
    pub request: GenerateRequest,
    pub lane: usize,
    sink: Box<dyn EventSink + 'a>,
    detok: StreamDecoder<'a>,
    turn: Turn,
    decoding: Option<Decode>,
    /// Prompt tokens committed to the state.
    at: u32,
    resume: u32,
    chunk_ends: std::collections::VecDeque<u32>,
    /// Positions to capture a checkpoint at, after the chunk ending there.
    checkpoints: Vec<u32>,
    prefill_started: Instant,
    /// The stop matcher and output limit, from the start until decoding begins.
    ready: Option<(StopMatcher, u32)>,
    /// Prompt tokens plus the clamped output limit, held while the job runs (FR-007).
    pub reservation: u32,
}

/// The decode phase's context.
struct Decode {
    matcher: StopMatcher,
    hook: Option<Box<dyn DecodeHook>>,
    mask: Option<TokenMask>,
    guard: LoopGuard,
    eos: Vec<u32>,
    ignore_eos: bool,
    limit: u32,
    max_draft: usize,
    drafter: Option<Drafter>,
    pace: Pace,
    /// The token fed next.
    last: u32,
    started: Option<Instant>,
}

impl std::fmt::Debug for Job<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Job")
            .field("lane", &self.lane)
            .field("at", &self.at)
            .field("decoding", &self.decoding.is_some())
            .finish_non_exhaustive()
    }
}

impl<'a> Job<'a> {
    pub fn new(
        request: GenerateRequest,
        lane: usize,
        sink: Box<dyn EventSink + 'a>,
        detok: StreamDecoder<'a>,
    ) -> Self {
        let turn = Turn {
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
        Self {
            request,
            lane,
            sink,
            detok,
            turn,
            decoding: None,
            at: 0,
            resume: 0,
            chunk_ends: Default::default(),
            checkpoints: Vec::new(),
            prefill_started: Instant::now(),
            ready: None,
            reservation: 0,
        }
    }

    /// The output limit a request gets alone: its own, clamped to the context left after its
    /// prompt; `None` when the prompt leaves no room (it is rejected at the start).
    pub fn limit(request: &GenerateRequest, capacity: u32) -> Option<u32> {
        let len = u32::try_from(request.prompt.tokens.len()).ok()?;
        let room = capacity.checked_sub(len).filter(|room| *room > 0)?;
        Some(request.max_tokens.min(room))
    }

    /// What it reserves when admitted: its prompt and its clamped output limit (FR-007).
    pub fn need(request: &GenerateRequest, capacity: u32) -> u32 {
        let len = request.prompt.len();
        Self::limit(request, capacity).map_or(len, |limit| len + limit)
    }

    pub fn cancelled(&self) -> bool {
        self.request.cancel.is_cancelled()
    }

    /// Whether the job reads its prompt (a chunk is next) or decodes.
    pub fn decoding(&self) -> bool {
        self.decoding.is_some()
    }

    /// Prompt tokens it still has to read.
    pub fn left(&self) -> u32 {
        self.chunk_ends.back().map_or(0, |&end| end - self.at)
    }

    /// The next prefill chunk's tokens.
    pub fn next_chunk(&self) -> Option<&[u32]> {
        let end = *self.chunk_ends.front()?;
        Some(&self.request.prompt.tokens[self.at as usize..end as usize])
    }

    /// Gives the selected lane this request's images (009); before every chunk and decoding,
    /// since another request may have used the lane meanwhile.
    pub fn media<E: Engine>(&self, engine: &mut E) -> Result<(), Failure> {
        let spans: Vec<_> = self
            .request
            .prompt
            .images
            .iter()
            .map(|p| p.span())
            .collect();
        engine.set_media(&spans).map_err(Failure::from)
    }

    /// Encodes the images the next chunk reaches that are not encoded yet; `Ok(false)` when
    /// cancelled.
    pub fn encode_for<E: Engine>(&self, engine: &mut E) -> Result<bool, Failure> {
        let Some(&end) = self.chunk_ends.front() else {
            return Ok(true);
        };
        let Some(vision) = engine.info().vision else {
            return Ok(true);
        };
        for placed in &self.request.prompt.images {
            let key = u64::from(placed.key);
            let first = placed.start;
            let past = first + placed.image.tokens();
            if first >= end || past <= self.at || !engine.needs_image(key) {
                continue;
            }
            let pixels = placed.image.pixels(&vision).map_err(Failure::Image)?;
            if !engine
                .encode_image(key, &pixels, &self.request.cancel)
                .map_err(Failure::from)?
            {
                return Ok(false);
            }
        }
        Ok(true)
    }

    /// Delivers held events; whether they all left, wait, or the consumer is gone.
    pub fn flush(&mut self) -> Delivery {
        self.sink.flush()
    }

    /// Validates, plans the reuse (003 `prepare`) and reports `Started`.
    pub fn start<E: Engine>(
        &mut self,
        engine: &mut E,
        store: &mut PrefixStore,
        pressure: &PressureGauge,
    ) -> Next {
        let prompt = &self.request.prompt;
        let invalid =
            |message: String| Ok(Some(FinishReason::Rejected(Rejection::Invalid(message))));
        if prompt.is_empty() {
            return invalid("the prompt is empty".into());
        }
        if !prompt.images.is_empty() && engine.info().vision.is_none() {
            return invalid("the served model reads no images".into());
        }
        if self.request.max_tokens == 0 {
            return invalid("max tokens must be at least 1".into());
        }
        if let Err(message) = self.request.sampling.validate() {
            return invalid(message);
        }
        let matcher = match StopMatcher::new(self.request.stop.clone()) {
            Ok(matcher) => matcher,
            Err(error) => return invalid(error.to_string()),
        };
        // Refuse only a prompt that leaves no room for one output token; clamp the output limit.
        let capacity = engine.budget().kv_capacity_tokens;
        let Some(limit) = Self::limit(&self.request, capacity) else {
            let prompt = prompt.tokens.len().min(u32::MAX as usize) as u32;
            return Ok(Some(FinishReason::Rejected(Rejection::TooLong {
                prompt,
                capacity,
            })));
        };
        let len = prompt.len();
        self.turn.stats.max_tokens = limit;
        self.reservation = len + limit;

        // Memory is found when it is needed: moves are memory-neutral, attaching evicts other
        // branches, and prefill growth evicts by rank once the old tail is a branch too.
        relieve_pressure(engine, store, pressure);
        let Prepared {
            resume,
            chunk_ends,
            checkpoints,
            miss,
            excerpt,
        } = store.prepare(engine, prompt, len + limit)?;
        self.turn.prefilled = resume;
        self.turn.stats.cached_tokens = resume;
        self.turn.stats.miss = Some(miss);
        if !self.sink.send(Event::Started {
            prompt: len,
            cached: resume,
            max_tokens: limit,
            miss: Some(miss),
            excerpt,
            lane: self.lane as u32,
        }) {
            return Ok(Some(FinishReason::Cancelled));
        }
        self.resume = resume;
        self.at = resume;
        self.chunk_ends = chunk_ends.into();
        self.checkpoints = checkpoints;
        self.prefill_started = Instant::now();
        self.ready = Some((matcher, limit));
        self.media(engine)?;
        if self.chunk_ends.is_empty() {
            self.begin_decode(engine);
        }
        Ok(None)
    }

    /// After the chunk [`Self::next_chunk`] gave was committed: a checkpoint where planned,
    /// progress, pressure; decoding begins after the last chunk.
    pub fn chunk_done<E: Engine>(
        &mut self,
        engine: &mut E,
        store: &mut PrefixStore,
        pressure: &PressureGauge,
    ) -> Next {
        let end = self.chunk_ends.pop_front().expect("a chunk ran");
        self.at = end;
        self.turn.prefilled = end;
        if self.checkpoints.contains(&end) {
            let checkpoint = with_memory(engine, store, &mut self.turn.captured, |engine| {
                engine.checkpoint()
            })?;
            self.turn.captured.push(checkpoint);
        }
        let total = (self.request.prompt.len() - 1).saturating_sub(self.resume);
        if !self.sink.send(Event::Progress {
            done: end - self.resume,
            total,
        }) {
            self.request.cancel.cancel();
            return Ok(Some(FinishReason::Cancelled));
        }
        relieve_pressure(engine, store, pressure);
        if self.chunk_ends.is_empty() {
            self.begin_decode(engine);
        }
        Ok(None)
    }

    /// The decode phase's context, once the prompt is read.
    fn begin_decode<E: Engine>(&mut self, engine: &E) {
        let (matcher, limit) = self.ready.take().expect("started");
        let mut hook = self.request.hook.take();
        let ignore_eos = hook.as_ref().is_some_and(|hook| hook.ignore_eos());
        let mask = hook.as_mut().and_then(|hook| hook.first_mask());
        // Speculative decoding (006): prompt lookup over the prompt and the reply, verified
        // exactly.
        let max_draft = if self.request.speculate && hook.as_ref().is_none_or(|hook| !hook.masks())
        {
            engine.max_draft() as usize
        } else {
            0
        };
        let tokens = &self.request.prompt.tokens;
        self.decoding = Some(Decode {
            matcher,
            hook,
            mask,
            guard: LoopGuard::new(self.request.call_fence),
            eos: engine.info().eos_tokens.clone(),
            ignore_eos,
            limit,
            max_draft,
            drafter: (max_draft > 0).then(|| Drafter::new(tokens)),
            pace: Pace::default(),
            last: tokens[tokens.len() - 1],
            started: None,
        });
    }

    /// The token the next decode step feeds.
    pub fn last(&self) -> u32 {
        self.decoding.as_ref().expect("decoding").last
    }

    /// The mask for the next sampled token (a grammar's), taken for one step.
    pub fn take_mask(&mut self) -> Option<TokenMask> {
        self.decoding.as_mut().expect("decoding").mask.take()
    }

    /// One decode round of this job alone: a step, or (`solo`, 006) a verify of a draft.
    pub fn decode_round<E: Engine>(
        &mut self,
        engine: &mut E,
        store: &mut PrefixStore,
        pressure: &PressureGauge,
        solo: bool,
    ) -> Next {
        if self.cancelled() {
            return Ok(Some(FinishReason::Cancelled));
        }
        if self
            .turn
            .stats
            .generated_tokens
            .is_multiple_of(PRESSURE_EVERY)
        {
            relieve_pressure(engine, store, pressure);
        }
        let round = Instant::now();
        let d = self.decoding.as_mut().expect("decoding");
        // Room for the draft: the output limit counts every committed token, the final sample
        // included.
        let room = (d.limit - self.turn.stats.generated_tokens).saturating_sub(1) as usize;
        let last = d.last;
        let draft = match &d.drafter {
            Some(drafter) if solo && d.mask.is_none() && d.pace.drafting() => {
                // A full prompt-lookup match costs nothing to make; otherwise the engine's own
                // drafter (030), when it has one. Either way the verify keeps only what plain
                // decoding samples.
                let n = d.max_draft.min(room);
                let lookup = drafter.draft(n);
                if lookup.len() >= n || engine.info().drafter.is_none() {
                    lookup
                } else {
                    match engine.draft(last, n.min(MODEL_DRAFT) as u32) {
                        Ok(model) if !model.is_empty() => model,
                        _ => lookup,
                    }
                }
            }
            _ => Vec::new(),
        };
        let committed = if draft.is_empty() {
            let step = with_memory(engine, store, &mut self.turn.captured, |engine| {
                engine.forward(last)
            })?;
            let mask = d.mask.take();
            vec![engine.sample(step, &self.request.sampling, mask.as_ref())?]
        } else {
            let sampling = self.request.sampling;
            let committed = with_memory(engine, store, &mut self.turn.captured, |engine| {
                engine.verify(last, &draft, &sampling)
            })?;
            self.turn.stats.drafted += draft.len() as u32;
            self.turn.stats.accepted += committed.len() as u32 - 1;
            self.turn.stats.passes += 1;
            if !self.sink.send(Event::Verified {
                drafted: draft.len() as u32,
                accepted: committed.len() as u32 - 1,
            }) {
                return Ok(Some(FinishReason::Cancelled));
            }
            committed
        };
        self.commit(&committed, !draft.is_empty(), round)
    }

    /// Takes the tokens a step (one) or a verify (the kept drafts and a sample) committed.
    pub fn commit(&mut self, committed: &[u32], verified: bool, round: Instant) -> Next {
        let len = self.request.prompt.len();
        // `last` is fed now; it is an output token after the first step.
        if self.turn.decoding {
            self.turn.forwarded += 1;
        } else {
            self.turn.decoding = true;
            self.turn.prefilled = len;
        }
        let d = self.decoding.as_mut().expect("decoding");
        if d.started.is_none() {
            self.turn.stats.ttft = Some(self.turn.started.elapsed());
            let prefill = self.prefill_started.elapsed().as_secs_f64();
            if prefill > 0.0 {
                self.turn.stats.prefill_tps = f64::from(len - self.resume) / prefill;
            }
            d.started = Some(Instant::now());
        }
        let kept = committed.len() - 1;
        for (i, &next) in committed.iter().enumerate() {
            if let Some(reason) = self.take(next)? {
                return Ok(Some(match reason {
                    FinishReason::Cancelled => reason,
                    reason => self.finish_decode(reason),
                }));
            }
            let d = self.decoding.as_mut().expect("decoding");
            if let Some(drafter) = &mut d.drafter {
                drafter.push(next);
            }
            // A kept draft token was fed in the verify; the final sample is fed next round.
            if i < kept {
                self.turn.forwarded += 1;
            }
        }
        let d = self.decoding.as_mut().expect("decoding");
        // The first step (the prompt's) is not a decode step; the rest time the pace.
        if self.turn.stats.generated_tokens > committed.len() as u32 {
            if verified {
                d.pace.round(committed.len(), round.elapsed());
            } else {
                d.pace.plain(round.elapsed());
            }
        }
        d.last = committed[kept];
        Ok(None)
    }

    /// The per-token path: EOS, the hook, detokenizing, stop sequences, the output limit and
    /// the loop guard. `Some` ends the request.
    fn take(&mut self, next: u32) -> Next {
        let d = self.decoding.as_mut().expect("decoding");
        if !d.ignore_eos && d.eos.contains(&next) {
            Self::flush_text(&mut *self.sink, &mut d.matcher, &mut self.turn);
            return Ok(Some(FinishReason::Stop {
                stop: None,
                eos_token: Some(next),
            }));
        }
        let turn = &mut self.turn;
        turn.stats.generated_tokens += 1;
        turn.output.push(next);
        if let Some(hook) = &mut d.hook {
            d.mask = hook.on_token(next);
        }
        let mut fed = Fed::default();
        let text = self
            .detok
            .step(next)
            .map_err(|error| Failure::Detokenize(error.to_string()))?;
        if let Some(text) = text {
            turn.decoded_bytes += text.len();
            let ends = turn.output.len() - turn.text_ends.len();
            turn.text_ends
                .extend(std::iter::repeat_n(turn.decoded_bytes, ends));
            fed = d.matcher.feed(&text);
        }
        let held = d.matcher.held() as u32;
        let bytes = fed.emit.len();
        if !self.sink.send(Event::Token {
            id: next,
            text: fed.emit,
            held,
        }) {
            return Ok(Some(FinishReason::Cancelled));
        }
        turn.emitted_bytes += bytes;
        if let Some(stop) = fed.stopped {
            return Ok(Some(FinishReason::Stop {
                stop: Some(stop),
                eos_token: None,
            }));
        }
        if turn.stats.generated_tokens >= d.limit {
            Self::flush_text(&mut *self.sink, &mut d.matcher, turn);
            return Ok(Some(FinishReason::Length));
        }
        // Off with `ignore_eos`: forced-length runs (bench) generate to the limit on purpose.
        if d.guard.push(next) && !d.ignore_eos {
            Self::flush_text(&mut *self.sink, &mut d.matcher, turn);
            return Ok(Some(FinishReason::Repetition));
        }
        Ok(None)
    }

    /// Releases held text at a finish without a stop match.
    fn flush_text(sink: &mut dyn EventSink, matcher: &mut StopMatcher, turn: &mut Turn) {
        let text = matcher.finish();
        let bytes = text.len();
        if bytes > 0 && sink.send(Event::Flush { text }) {
            turn.emitted_bytes += bytes;
        }
    }

    fn finish_decode(&mut self, reason: FinishReason) -> FinishReason {
        let started = self.decoding.as_ref().and_then(|d| d.started);
        let stats = &mut self.turn.stats;
        if let Some(start) = started {
            let elapsed = start.elapsed().as_secs_f64();
            if elapsed > 0.0 && stats.generated_tokens > 1 {
                stats.decode_tps = f64::from(stats.generated_tokens - 1) / elapsed;
            }
        }
        reason
    }

    /// This request's own checkpoints so far (dropped first when memory runs out).
    pub fn captured(&mut self) -> &mut Vec<Checkpoint> {
        &mut self.turn.captured
    }

    /// Ends the job: the state keeps what it committed (003 `commit`), or everything is dropped
    /// after a backend failure. Its lane must be selected. Returns the report and the sink,
    /// which may still hold events for the client.
    pub fn finish<E: Engine>(
        self,
        engine: &mut E,
        store: &mut PrefixStore,
        outcome: Result<FinishReason, Failure>,
    ) -> (Finished, Box<dyn EventSink + 'a>) {
        let Job {
            request,
            turn,
            sink,
            ..
        } = self;
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
                engine.select(store.lane());
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
        (
            Finished {
                reason,
                prefilled,
                stats: turn.stats,
            },
            sink,
        )
    }
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
        fn masks(&self) -> bool {
            false
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
        req.call_fence = Some(CallFence {
            open,
            close,
            think: None,
            thinking: false,
        });
        let r = run(&mut engine, &mut store, req);
        assert_eq!(r.finished.reason, EOS_STOP);
        assert!(text(&r.events).ends_with(">done"));
    }

    #[test]
    fn a_call_opener_while_thinking_does_not_blind_the_loop_guard() {
        // The reply starts in the reasoning, mentions the opener and then loops: stopped.
        let (open, close, think_end) = (ids("<")[0], ids(">")[0], ids("|")[0]);
        let mut script = vec![open];
        script.extend(ids("ab").into_iter().cycle().take(1_500));
        let mut engine = ScriptedEngine::new(script, 100_000);
        let mut req = request(ids("x"), 2_000);
        req.call_fence = Some(CallFence {
            open,
            close,
            think: Some((ids("~")[0], think_end)),
            thinking: true,
        });
        let r = run(&mut engine, &mut store(), req);
        assert_eq!(r.finished.reason, FinishReason::Repetition);
    }

    #[test]
    fn a_looping_paragraph_is_caught_too() {
        // A 200-token paragraph repeated: longer than a phrase, still a loop.
        let paragraph: Vec<u32> = (0..200u32)
            .map(|i| ids("abcdefghijklmnopqrstuvwxyz")[(i * 7 % 26) as usize] + (i / 26) % 2)
            .collect();
        let script: Vec<u32> = paragraph.iter().copied().cycle().take(3_000).collect();
        let mut engine = ScriptedEngine::new(script, 100_000);
        let r = run(&mut engine, &mut store(), request(ids("x"), 2_500));
        assert_eq!(r.finished.reason, FinishReason::Repetition);
        assert!(r.finished.stats.generated_tokens < 1_000);
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

    /// A sink that disconnects after `left` tokens.
    struct Leaves {
        events: Vec<Event>,
        left: usize,
    }

    impl EventSink for Leaves {
        fn send(&mut self, event: Event) -> bool {
            if matches!(event, Event::Token { .. }) {
                if self.left == 0 {
                    return false;
                }
                self.left -= 1;
            }
            self.events.push(event);
            true
        }
    }

    /// One request with speculation off and on (006), on an attention-only and a hybrid engine:
    /// the same events, finish, stats, live state and cache; returns the speculative run's
    /// drafted and accepted counts.
    fn same_with_speculation(
        script: &[u32],
        prompt: &[u32],
        max_tokens: u32,
        leave_after: Option<usize>,
        setup: impl Fn(&mut GenerateRequest),
    ) -> (u32, u32) {
        let mut counts = (0, 0);
        for hybrid in [false, true] {
            let mut runs = Vec::new();
            for max_draft in [0, 7] {
                let mut engine = if hybrid {
                    ScriptedEngine::new_hybrid(script.to_vec(), 10_000, 64)
                } else {
                    ScriptedEngine::new(script.to_vec(), 10_000)
                };
                engine.max_draft = max_draft;
                let mut store = store();
                let mut req = request(prompt.to_vec(), max_tokens);
                setup(&mut req);
                let tokenizer = byte_tokenizer();
                let mut sink = Leaves {
                    events: Vec::new(),
                    left: leave_after.unwrap_or(usize::MAX),
                };
                let finished = generate(
                    &mut engine,
                    &mut store,
                    &mut tokenizer.stream(),
                    req,
                    &mut sink,
                    &PressureGauge::default(),
                );
                assert_eq!(
                    engine.tokens(),
                    store.live_tokens(),
                    "the store mirrors the state"
                );
                assert!(!engine.has_pending());
                // A follow-up reuses what the request left.
                let live = engine.tokens().to_vec();
                let mut follow = live.clone();
                follow.extend(ids("?"));
                let again = run(&mut engine, &mut store, request(follow, 1));
                runs.push((
                    sink.events,
                    finished,
                    live,
                    again.finished.stats.cached_tokens,
                ));
            }
            let [
                (plain_events, plain, plain_tokens, plain_cached),
                (events, spec, tokens, cached),
            ] = <[_; 2]>::try_from(runs).unwrap();
            let tokens_only = |events: Vec<Event>| -> Vec<Event> {
                events
                    .into_iter()
                    .filter(|e| !matches!(e, Event::Verified { .. }))
                    .collect()
            };
            assert_eq!(
                tokens_only(events),
                plain_events,
                "events (hybrid: {hybrid})"
            );
            assert_eq!(spec.reason, plain.reason, "finish (hybrid: {hybrid})");
            if hybrid {
                // A verify's trim window cuts position-bound state where plain decoding cannot
                // (a stop inside fed tokens): the state kept is exact and never less.
                assert!(spec.prefilled >= plain.prefilled, "prefilled (hybrid)");
                assert!(tokens.starts_with(&plain_tokens), "live state (hybrid)");
                assert!(cached >= plain_cached, "reuse after (hybrid)");
            } else {
                assert_eq!(spec.prefilled, plain.prefilled, "prefilled");
                assert_eq!(tokens, plain_tokens, "live state");
                assert_eq!(cached, plain_cached, "reuse after");
            }
            let strip = |stats: &Stats| Stats {
                ttft: None,
                prefill_tps: 0.0,
                decode_tps: 0.0,
                drafted: 0,
                accepted: 0,
                passes: 0,
                ..stats.clone()
            };
            assert_eq!(
                strip(&spec.stats),
                strip(&plain.stats),
                "stats (hybrid: {hybrid})"
            );
            assert_eq!((plain.stats.drafted, plain.stats.passes), (0, 0));
            counts = (spec.stats.drafted, spec.stats.accepted);
        }
        counts
    }

    #[test]
    fn speculation_copies_from_the_prompt_and_changes_nothing() {
        let prompt = ids("the quick brown fox jumps over the lazy dog. the quick");
        let script = ids(" brown fox jumps over the lazy cat.");
        let (drafted, accepted) = same_with_speculation(&script, &prompt, 100, None, |_| {});
        assert!(
            drafted > 0 && accepted > 0,
            "drafted {drafted}, accepted {accepted}"
        );
        // A reply that leaves the copy inside a draft: "cdefghi" drafted, "cdefg" kept.
        let prompt = ids("abcdefghijklmnop abcdefghijklmnop ab");
        let (drafted, accepted) =
            same_with_speculation(&ids("cdefgX."), &prompt, 100, None, |_| {});
        assert!(accepted < drafted, "drafted {drafted}, accepted {accepted}");
    }

    #[test]
    fn speculation_ends_inside_a_batch_as_plain_decoding_does() {
        let prompt = ids("the quick brown fox jumps over the lazy dog. the quick");
        let script = ids(" brown fox jumps over the lazy dog.");
        // A stop sequence, the output limit and a disconnect inside accepted drafts.
        same_with_speculation(&script, &prompt, 100, None, |r| {
            r.stop = vec!["over".into()]
        });
        same_with_speculation(&script, &prompt, 9, None, |_| {});
        same_with_speculation(&script, &prompt, 100, Some(7), |_| {});
        // EOS as an accepted draft token: the prompt holds it after the copied text.
        let mut prompt = ids("ab");
        prompt.push(SCRIPT_EOS);
        prompt.extend(ids("xyz. ab"));
        let (drafted, _) = same_with_speculation(&ids("ab"), &prompt, 100, None, |_| {});
        assert!(drafted > 0);
        // The loop guard, fed from the reply's own repetition.
        let script: Vec<u32> = ids("ab").into_iter().cycle().take(2_000).collect();
        let (drafted, accepted) = same_with_speculation(&script, &ids("x"), 1_500, None, |_| {});
        assert!(accepted > drafted / 2, "a repeating reply drafts itself");
    }

    #[test]
    fn no_speculation_under_masks_or_when_off() {
        let prompt = ids("the quick brown fox. the quick");
        let script = ids(" brown fox.");
        for (speculate, hook) in [(false, false), (true, true)] {
            let mut engine = ScriptedEngine::new(script.clone(), 10_000);
            engine.max_draft = 7;
            let mut req = request(prompt.clone(), 100);
            req.speculate = speculate;
            if hook {
                req.hook = Some(Box::new(Starve {
                    after: 1_000,
                    seen: 0,
                }));
            }
            let r = run(&mut engine, &mut store(), req);
            assert_eq!(r.finished.stats.passes, 0);
        }
        // `ignore_eos` alone never masks: speculation runs.
        let mut engine = ScriptedEngine::new(script, 10_000);
        engine.max_draft = 7;
        let mut req = request(prompt, 12);
        req.hook = Some(Box::new(IgnoreEos));
        let r = run(&mut engine, &mut store(), req);
        assert!(r.finished.stats.passes > 0);
    }
}
