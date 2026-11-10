//! One API request end to end: render, splice, admit, generate, parse, and stream `GenEvent`s
//! to the encoder, with the queue deadline and cancellation (002 FR-009–FR-011). Keepalives are
//! the stream writer's: they follow the bytes sent (D033).
//!
//! The request runs on the engine worker; a tokio task relays its events and keeps the
//! request's tracker entry and metrics (012). When the response is dropped (client gone,
//! streaming or not) the relay cancels the request, whether it is still queued, prefilling or
//! generating. Only non-streaming requests have a queue deadline: streaming ones receive
//! keepalives while they wait (D052). A reply that ends with EOS is recorded in the splice
//! ledger (D030).

use crate::api::log;
use crate::api::metrics::Outcome;
use crate::api::model::{
    ApiError, ApiKind, ChatRequest, ErrorKind, Finish, GenEvent, Message, ReasoningCtl, ToolChoice,
    ToolDef, Usage, Warning,
};
use crate::api::sessions::{Fingerprint, Finished};
use crate::api::status::{Entry, Phase, Stage, Tracker};
use crate::cache::miss::MissReport;
use crate::chat::constrain;
use crate::chat::hook::{Arm, Constraint, RequestHook, SingleCall};
use crate::chat::render::{self, RenderContext};
use crate::chat::splice::{self, Ledger, Reply, ReplyParts, Spliced};
use crate::chat::syntax::registry::ChatSyntax;
use crate::chat::syntax::{NO_TOKEN, ParseEnd, ParseLog, ParseOut, ParserConfig, Specials};
use crate::cli::serve::{Loaded, ServeArgs};
use crate::runtime::generate::{
    CallFence, Event, Failure, FinishReason, GenerateRequest, Rejection,
};
use crate::runtime::handle::{Generation, SubmitError, Ticket};
use mrun_engine::TokenMask;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::{Duration, Instant};
use tokio::sync::mpsc;

/// Silence after which a keepalive is sent, in every phase.
pub const KEEPALIVE: Duration = Duration::from_secs(10);
/// The most output a request without a limit gets, below the remaining capacity, in all three
/// APIs (D052; `--max-output` replaces it). Anthropic requires `max_tokens`; mrun accepts its
/// absence (documented leniency).
pub const DEFAULT_MAX_OUTPUT: u32 = 32_768;

/// A started request.
#[derive(Debug)]
pub struct Turn {
    pub events: mpsc::Receiver<GenEvent>,
    /// Output limit asked for (or the API default), before clamping.
    pub requested_max: u32,
    /// The client sent the limit (only then is clamping it worth a warning).
    pub explicit_max: bool,
    pub warnings: Vec<Warning>,
    /// Prompt tokens taken from earlier replies (D030).
    pub spliced: u32,
}

/// Timing knobs, from `mrun serve` flags (tests shorten them).
#[derive(Clone, Copy, Debug)]
pub struct Timing {
    pub keepalive: Duration,
    pub queue_wait: Duration,
}

impl Timing {
    /// A deadline of 0 s would fail every non-streaming request: at least 1 s.
    pub fn from_args(args: &ServeArgs) -> Self {
        Self {
            keepalive: KEEPALIVE,
            queue_wait: Duration::from_secs(args.queue_wait_s.max(1)),
        }
    }
}

/// The model id responses report: the requested one (D033), else the served model's.
pub fn model_id(loaded: &Loaded, request: &ChatRequest) -> String {
    if request.model.is_empty() {
        loaded.model_id.clone()
    } else {
        request.model.clone()
    }
}

/// Renders and queues `request`; errors here happen before any response header.
pub async fn start(
    loaded: Arc<Loaded>,
    request: ChatRequest,
    args: &ServeArgs,
    timing: Timing,
    tracker: Arc<Tracker>,
) -> Result<Turn, ApiError> {
    let syntax = ChatSyntax::get(loaded.syntax);
    let sampling = request.sampling.resolve_valid(loaded.defaults)?;
    if request.max_output_tokens == Some(0) {
        let param = match request.api {
            ApiKind::Responses => "max_output_tokens",
            ApiKind::Chat | ApiKind::Messages => "max_tokens",
        };
        return Err(ApiError::invalid(
            format!("{param}: must be at least 1"),
            param,
        ));
    }
    let prepared = prepare(&loaded, &request).await?;
    loaded
        .handle
        .ledger()
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .note_turn(&request.client, &prepared.spliced);
    let prompt_tokens = prepared.rendered.prompt.len();
    let capacity = loaded.handle.budget().kv_capacity_tokens;
    if prompt_tokens + 1 > capacity {
        return Err(prompt_too_long(prompt_tokens, capacity));
    }
    let requested_max = request.max_output_tokens.unwrap_or_else(|| {
        (capacity - prompt_tokens).min(args.max_output.unwrap_or(DEFAULT_MAX_OUTPUT).max(1))
    });
    // A non-streaming request has no keepalive to hold its client: refuse it now when it
    // would wait past the deadline (D052).
    let queue = loaded.handle.queue();
    if !request.stream && queue.estimated_wait() > timing.queue_wait {
        let retry = queue.estimated_wait().as_secs();
        return Err(ApiError::new(
            ErrorKind::QueueTimeout,
            format!(
                "the estimated wait in the queue ({retry} s) exceeds its deadline ({} s)",
                timing.queue_wait.as_secs()
            ),
        )
        .with_retry_after(retry));
    }
    let mut warnings = Vec::new();
    if request
        .messages
        .iter()
        .any(|m| matches!(m, Message::Tool(r) if r.images_omitted > 0))
    {
        warnings.push(Warning::ImageOmitted);
    }

    let rendered = prepared.rendered;
    let parser = syntax.new_parser(&ParserConfig {
        tools: request.tools.clone(),
        reasoning_open: rendered.reasoning_open,
        buffer_args: request.api == ApiKind::Messages,
        specials: Some(Arc::new(Specials::new(loaded.tokenizer.added_tokens()))),
    });
    // Only a reply generated after the whole prompt (no prefill, no ignore-EOS) is spliceable.
    let record = (!request.prefill && !request.ignore_eos).then(|| Recording {
        prompt: rendered.prompt.tokens.clone(),
        generation_start: rendered.prompt.generation_start,
        ledger: loaded.handle.ledger().clone(),
        tools: request.tools.clone(),
    });
    let replay = Finished {
        member: request.member.clone(),
        client: request.client.clone(),
        fingerprint: Fingerprint::of(&rendered.prompt, !request.tools.is_empty()),
        excerpt: None,
    };
    let mut generate = GenerateRequest::new(rendered.prompt, requested_max);
    generate.sampling = sampling;
    generate.stop = request.stop.clone();
    generate.hook = hook(&loaded, &request, rendered.reasoning_open).boxed();
    generate.call_fence = call_fence(&loaded);
    generate.turn = request.member.clone();
    let generation = loaded.handle.generate(generate).map_err(|e| match e {
        SubmitError::QueueFull(full) => {
            ApiError::new(ErrorKind::QueueFull, "the request queue is full")
                .with_retry_after(full.retry_after)
        }
        SubmitError::Stopped => ApiError::new(ErrorKind::Internal, e.to_string()),
    })?;

    let (tx, events) = mpsc::channel(64);
    let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
    tracker.insert(Entry {
        ticket: Some(generation.ticket.clone()),
        member: request.member.clone(),
        ..Entry::queued(id, request.api, prompt_tokens)
    });
    let relay = Relay {
        tx,
        parser,
        api: request.api,
        stream: request.stream,
        requested_max,
        prompt_tokens,
        queue_wait: timing.queue_wait,
        single_call: request.tool_choice == ToolChoice::Auto && !request.parallel_tool_calls,
        calls_ended: 0,
        reasoning_tokens: 0,
        tracker,
        id,
        generating: false,
        miss: None,
        replay: Some(replay),
        spliced: prepared.spliced.tokens,
        record,
        generated: Vec::new(),
        reply: ReplyParts::default(),
    };
    tokio::spawn(relay.run(generation));
    Ok(Turn {
        events,
        requested_max,
        explicit_max: request.max_output_tokens.is_some(),
        warnings,
        spliced: prepared.spliced.tokens,
    })
}

/// A rendered request, with earlier replies spliced in.
#[derive(Debug)]
pub struct Prepared {
    pub rendered: render::Rendered,
    /// The earlier replies spliced in.
    pub spliced: Spliced,
}

/// Renders and splices off the async runtime (templates and tokenization of long prompts take
/// a while); logs the render's conditions.
pub async fn prepare(loaded: &Arc<Loaded>, request: &ChatRequest) -> Result<Prepared, ApiError> {
    let loaded = loaded.clone();
    let request = request.clone();
    tokio::task::spawn_blocking(move || {
        let mut rendered = render::render(
            &request,
            &RenderContext {
                template: &loaded.template,
                tokenizer: &loaded.tokenizer,
                syntax: loaded.syntax,
                supports_thinking: loaded.supports_thinking,
            },
        )?;
        for entry in &rendered.logs {
            eprintln!("mrun: {}", entry.line(request.api));
        }
        let ledger = loaded
            .handle
            .ledger()
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        let spliced = splice::splice(
            &ledger,
            &mut rendered.prompt,
            &rendered.replies,
            &request.messages,
            &request.tools,
        );
        Ok(Prepared { rendered, spliced })
    })
    .await
    .map_err(|_| ApiError::new(ErrorKind::Internal, "rendering failed"))?
}

/// The decode hook: the answer grammar (`required`/named calls, JSON formats), the `none` ban,
/// a single call when parallel calls are off, and `ignore_eos`. A grammar that fails to compile
/// leaves the request unconstrained (logged `unenforced` with the field).
fn hook(loaded: &Loaded, request: &ChatRequest, reasoning_open: bool) -> RequestHook {
    let tokenizer = &loaded.tokenizer;
    let mut constraint = None;
    let mut ban = None;
    if let Some(env) = loaded.grammar.clone() {
        let plan = constrain::plan(request, loaded.syntax, &env);
        for path in &plan.unenforced {
            log::unenforced(path);
        }
        let answer = plan
            .answer
            .as_deref()
            .and_then(|lark| match env.matcher(lark) {
                Ok(matcher) => Some(matcher),
                // The error can quote the schema: log the field only (FR-014).
                Err(_) => {
                    log::unenforced(plan.answer_field);
                    None
                }
            });
        let (open, end) = (
            tokenizer.token_id("<think>"),
            tokenizer.token_id("</think>"),
        );
        let thinking = request.reasoning.is_on() && loaded.supports_thinking;
        let arm = match (reasoning_open, thinking, open, end) {
            (true, _, _, Some(end)) => Arm::AfterReasoning { end },
            (false, true, Some(open), Some(end)) => Arm::OptionalThink { open, end },
            _ => Arm::Now,
        };
        constraint = answer.map(|answer| Constraint {
            env: env.clone(),
            answer,
            arm,
        });
        ban = plan
            .ban_opener
            .then(|| env.special(constrain::OPENER))
            .flatten()
            .map(|token| constrain::ban(env.vocab(), token));
    }
    let single_call = (request.tool_choice == ToolChoice::Auto && !request.parallel_tool_calls)
        .then(|| tokenizer.token_id(constrain::CLOSER))
        .flatten()
        .map(|closer| {
            let eos = &loaded.handle.info().eos_tokens;
            let vocab = eos
                .iter()
                .map(|id| id + 1)
                .fold(tokenizer.vocab_size(), u32::max);
            let mut eos_only = TokenMask::none_allowed(vocab);
            for &id in eos {
                eos_only.allow(id);
            }
            SingleCall { closer, eos_only }
        });
    if let ReasoningCtl::On {
        budget: Some(_), ..
    } = request.reasoning
    {
        log::unenforced("thinking.budget_tokens");
    }
    RequestHook::new(request.ignore_eos, constraint, ban, single_call)
}

/// The tool-call delimiters of the model's syntax, for the loop guard (D052).
fn call_fence(loaded: &Loaded) -> Option<CallFence> {
    let token = |text: &str| loaded.tokenizer.token_id(text);
    Some(CallFence {
        open: token(constrain::OPENER)?,
        close: token(constrain::CLOSER)?,
    })
}

/// Anthropic's and OpenAI's prompt-too-long error (context-length class).
pub fn prompt_too_long(prompt: u32, capacity: u32) -> ApiError {
    ApiError::new(
        ErrorKind::ContextLengthExceeded,
        format!(
            "prompt is too long: {prompt} tokens > {} maximum",
            capacity.saturating_sub(1)
        ),
    )
}

/// A request the runtime refused before any processing.
fn rejected(rejection: Rejection) -> ApiError {
    match rejection {
        Rejection::TooLong { prompt, capacity } => prompt_too_long(prompt, capacity),
        Rejection::Invalid(message) => ApiError::new(ErrorKind::InvalidRequest, message),
    }
}

/// A request that failed after admission. Unmet memory after the cache was shed cannot fit
/// later with one lane, so it is the context-length class (clients compact; the message names
/// the numbers); `overloaded` stays for transient causes (queue full, shutdown). Everything else
/// is a deterministic server error.
fn failed(failure: Failure) -> ApiError {
    match failure {
        Failure::OutOfMemory(message) => ApiError::new(ErrorKind::ContextLengthExceeded, message),
        other => ApiError::new(ErrorKind::Internal, other.to_string()),
    }
}

fn shutting_down() -> ApiError {
    ApiError::new(ErrorKind::Overloaded, "the server is shutting down").with_retry_after(5)
}

/// Waits until `deadline`, or forever without one.
async fn until(deadline: Option<tokio::time::Instant>) {
    match deadline {
        Some(deadline) => tokio::time::sleep_until(deadline).await,
        None => std::future::pending().await,
    }
}

/// What a finished reply needs to enter the ledger.
struct Recording {
    prompt: Vec<u32>,
    generation_start: u32,
    ledger: Arc<Mutex<Ledger>>,
    /// The request's tools: the canonical form drops schema-default arguments.
    tools: Vec<ToolDef>,
}

struct Relay {
    tx: mpsc::Sender<GenEvent>,
    parser: Box<dyn crate::chat::syntax::SyntaxParser>,
    api: ApiKind,
    /// Streaming requests have no queue deadline (D052).
    stream: bool,
    requested_max: u32,
    prompt_tokens: u32,
    queue_wait: Duration,
    /// `parallel_tool_calls: false` under `auto`: nothing after the first call.
    single_call: bool,
    calls_ended: u32,
    reasoning_tokens: u32,
    tracker: Arc<Tracker>,
    id: u64,
    generating: bool,
    miss: Option<MissReport>,
    /// What session replay keeps of the request (025).
    replay: Option<Finished>,
    spliced: u32,
    record: Option<Recording>,
    /// Generated ids, for the ledger.
    generated: Vec<u32>,
    /// The parsed reply as the client receives it, for its canonical form.
    reply: ReplyParts,
}

static NEXT_ID: AtomicU64 = AtomicU64::new(1);

impl Relay {
    async fn run(mut self, generation: Generation) {
        let outcome = self.relay(&generation).await;
        let stop = match outcome {
            Outcome::Completed => "end",
            Outcome::Cancelled => "stopped",
            Outcome::Failed | Outcome::Rejected => "error",
        };
        self.tracker
            .finish(self.id, self.miss.as_ref(), stop, self.replay.take());
        self.tracker.metrics.record_outcome(self.api, outcome);
    }

    /// Sends a failure as the terminal event.
    async fn fail(&mut self, error: ApiError) -> Outcome {
        if self.tx.send(GenEvent::Failed(error)).await.is_err() {
            return Outcome::Cancelled;
        }
        Outcome::Failed
    }

    async fn relay(&mut self, generation: &Generation) -> Outcome {
        let deadline = (!self.stream).then(|| tokio::time::Instant::now() + self.queue_wait);
        let ticket = &generation.ticket;
        let mut started = false;
        let mut stage = self.tracker.watch_stage();
        loop {
            // Shutting down: a request that has not started ends now (D052).
            if *stage.borrow_and_update() != Stage::Serving && !started {
                ticket.cancel();
                return self.fail(shutting_down()).await;
            }
            let next = tokio::select! {
                _ = self.tx.closed() => {
                    // The response is gone: stop the request in whatever phase it is.
                    ticket.cancel();
                    return Outcome::Cancelled;
                }
                _ = stage.changed() => continue,
                _ = until(deadline.filter(|_| !started)) => {
                    ticket.cancel();
                    let error = ApiError::new(
                        ErrorKind::QueueTimeout,
                        "the request waited too long in the queue",
                    );
                    return self.fail(error).await;
                }
                next = generation.events.recv_async() => next,
            };
            let Ok(event) = next else {
                break; // events closed: `Finished` follows
            };
            let mut out = ParseOut::default();
            match event {
                Event::Started {
                    prompt,
                    cached,
                    max_tokens,
                    miss,
                    excerpt,
                } => {
                    started = true;
                    self.miss = miss;
                    if let Some(replay) = &mut self.replay {
                        replay.excerpt = excerpt;
                    }
                    self.tracker.update(self.id, |entry| {
                        entry.phase = Phase::Prefill;
                        entry.prompt_tokens = prompt;
                        entry.cached = Some(cached);
                        entry.progress = cached;
                        entry.cause = miss.and_then(|m| m.cause);
                        entry.unreused = Some(miss.map_or(0, |m| m.unreused));
                    });
                    let started = GenEvent::Started {
                        input: prompt,
                        cached,
                        max_tokens,
                        miss,
                    };
                    if !self.send(vec![started], ticket).await {
                        return Outcome::Cancelled;
                    }
                    continue;
                }
                Event::Progress { done, .. } => {
                    self.tracker.update(self.id, |entry| {
                        entry.progress = entry.cached.unwrap_or(0) + done;
                    });
                    continue;
                }
                Event::Verified { drafted, accepted } => {
                    self.tracker.update(self.id, |entry| {
                        let [d, a, p] = entry.speculation.get_or_insert_default();
                        *d += drafted;
                        *a += accepted;
                        *p += 1;
                    });
                    continue;
                }
                Event::Token { id, text, held } => {
                    let first = !self.generating;
                    self.generating = true;
                    self.tracker.update(self.id, |entry| {
                        if first {
                            entry.phase = Phase::Generating;
                            entry.progress = entry.prompt_tokens;
                            entry.first_token = Some(Instant::now());
                        }
                        entry.generated += 1;
                    });
                    self.generated.push(id);
                    self.parser.feed(id, &text, held, &mut out);
                    if out
                        .events
                        .iter()
                        .any(|e| matches!(e, GenEvent::ReasoningDelta(_)))
                    {
                        self.reasoning_tokens += 1;
                    }
                }
                Event::Flush { text } => self.parser.feed(NO_TOKEN, &text, 0, &mut out),
            }
            if !self.forward(out, ticket).await {
                return Outcome::Cancelled;
            }
        }
        let Ok(finished) = generation.finished.recv_async().await else {
            let error = ApiError::new(ErrorKind::Internal, "the engine thread has stopped");
            return self.fail(error).await;
        };
        let stats = &finished.stats;
        let usage = Usage {
            input: if stats.prompt_tokens > 0 {
                stats.prompt_tokens
            } else {
                self.prompt_tokens
            },
            cached: stats.cached_tokens,
            output: stats.generated_tokens,
            reasoning: self.reasoning_tokens.min(stats.generated_tokens),
            speculation: (stats.passes > 0).then_some([
                stats.drafted,
                stats.accepted,
                stats.passes,
            ]),
        };
        let mut out = ParseOut::default();
        let mut eos = None;
        let terminal = match finished.reason {
            FinishReason::Cancelled if *stage.borrow() != Stage::Serving => {
                return self.fail(shutting_down()).await;
            }
            FinishReason::Cancelled => return Outcome::Cancelled,
            FinishReason::Rejected(rejection) => return self.fail(rejected(rejection)).await,
            FinishReason::Error(failure) => return self.fail(failed(failure)).await,
            FinishReason::Repetition => {
                self.parser.finish(ParseEnd::Length, &mut out);
                GenEvent::Finished {
                    reason: Finish::Repetition,
                    stop_seq: None,
                    usage,
                }
            }
            FinishReason::Length => {
                self.parser.finish(ParseEnd::Length, &mut out);
                let clamped = stats.max_tokens < self.requested_max;
                GenEvent::Finished {
                    reason: if clamped {
                        Finish::ContextFull
                    } else {
                        Finish::Length
                    },
                    stop_seq: None,
                    usage,
                }
            }
            FinishReason::Stop { stop, eos_token } => {
                self.parser.finish(ParseEnd::Stop, &mut out);
                let calls = self.calls_ended
                    + out
                        .events
                        .iter()
                        .filter(|e| matches!(e, GenEvent::ToolEnd { .. }))
                        .count() as u32;
                eos = eos_token.filter(|_| stop.is_none());
                let reason = if stop.is_some() {
                    Finish::StopSequence
                } else if calls > 0 {
                    Finish::ToolCalls
                } else {
                    Finish::Stop
                };
                GenEvent::Finished {
                    reason,
                    stop_seq: stop,
                    usage,
                }
            }
        };
        if !self.forward(out, ticket).await {
            return Outcome::Cancelled;
        }
        if let (Some(eos), Some(record)) = (eos, self.record.take()) {
            record
                .ledger
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .record(Reply {
                    prompt: record.prompt,
                    generation_start: record.generation_start,
                    generated: std::mem::take(&mut self.generated),
                    eos,
                    canonical: self.reply.canonical(&record.tools),
                });
        }
        if self.tx.send(terminal).await.is_err() {
            return Outcome::Cancelled;
        }
        self.tracker.metrics.record_completed(
            self.api,
            usage,
            self.miss.as_ref(),
            self.spliced,
            &finished.stats,
        );
        Outcome::Completed
    }

    /// Assigns call ids, logs parser entries, notes the reply for the ledger and sends the
    /// events on; with a single call allowed, nothing after its end.
    async fn forward(&mut self, out: ParseOut, ticket: &Ticket) -> bool {
        for entry in &out.logs {
            // Names come from the model's output: log the field only (FR-014).
            match entry {
                ParseLog::ToolUnknown(_) => log::unenforced("tools"),
            }
        }
        let mut events = Vec::with_capacity(out.events.len());
        for mut event in out.events {
            if self.single_call && self.calls_ended > 0 {
                // The hook allows only EOS after the first call; this covers engines that
                // ignore masks.
                break;
            }
            match &mut event {
                GenEvent::ToolStart { id, name, .. } => {
                    if id.is_empty() {
                        *id = crate::api::sse::new_id(match self.api {
                            ApiKind::Messages => "toolu_",
                            ApiKind::Chat | ApiKind::Responses => "call_",
                        });
                    }
                    self.reply.calls.push((name.clone(), String::new()));
                }
                GenEvent::ToolEnd {
                    raw_args, valid, ..
                } => {
                    self.calls_ended += 1;
                    if let Some(call) = self.reply.calls.last_mut() {
                        call.1 = raw_args.clone();
                    }
                    // Anthropic shows a call whose arguments do not parse as text (FR-004).
                    if self.api == ApiKind::Messages && !*valid {
                        let text = self.reply.calls.pop().map(|c| c.1).unwrap_or_default();
                        self.reply.text.push_str(&text);
                    }
                }
                GenEvent::TextDelta(text) => self.reply.text.push_str(text),
                GenEvent::ReasoningDelta(text) => self.reply.reasoning.push_str(text),
                _ => {}
            }
            events.push(event);
        }
        self.send(events, ticket).await
    }

    async fn send(&mut self, events: Vec<GenEvent>, ticket: &Ticket) -> bool {
        for event in events {
            if self.tx.send(event).await.is_err() {
                ticket.cancel();
                return false;
            }
        }
        true
    }
}
