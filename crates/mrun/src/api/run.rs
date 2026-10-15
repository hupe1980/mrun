//! One API request end to end: render, admit, generate, parse, and stream `GenEvent`s to the
//! encoder, with keepalives, the queue deadline and cancellation (002 FR-009–FR-011).
//!
//! The request runs on the engine worker; a tokio task relays its events. When the response is
//! dropped (client gone, streaming or not) the relay cancels the request, whether it is still
//! queued, prefilling or generating.

use crate::api::model::{
    ApiError, ApiKind, ChatRequest, ErrorKind, Finish, GenEvent, Usage, Warning,
};
use crate::api::status::{Active, Tracker};
use crate::chat::render::{self, RenderContext, RenderLog};
use crate::chat::syntax::registry::ChatSyntax;
use crate::chat::syntax::{ParseEnd, ParseLog, ParseOut, ParserConfig};
use crate::cli::serve::{Loaded, ServeArgs};
use crate::runtime::generate::{Event, FinishReason, GenerateRequest};
use crate::runtime::handle::SubmitError;
use mrun_engine::CancelToken;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};
use tokio::sync::mpsc;

/// Silence after which a keepalive is sent, in every phase.
pub const KEEPALIVE: Duration = Duration::from_secs(10);
/// Default output limit of Chat Completions and Messages requests without one.
pub const DEFAULT_MAX_OUTPUT: u32 = 16_384;

/// What the encoder receives.
#[derive(Clone, Debug, PartialEq)]
pub enum Tick {
    Event(GenEvent),
    Keepalive,
}

/// A started request.
#[derive(Debug)]
pub struct Turn {
    pub ticks: mpsc::Receiver<Tick>,
    pub model: String,
    pub prompt_tokens: u32,
    /// Output limit asked for (or the API default), before clamping.
    pub requested_max: u32,
    pub warnings: Vec<Warning>,
}

/// Timing knobs, from `mrun serve` flags (tests shorten them).
#[derive(Clone, Copy, Debug)]
pub struct Timing {
    pub keepalive: Duration,
    pub queue_wait: Duration,
}

impl Timing {
    pub fn from_args(args: &ServeArgs) -> Self {
        Self {
            keepalive: KEEPALIVE,
            queue_wait: Duration::from_secs(args.queue_wait_s),
        }
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
    let rendering = render_request(&loaded, &request).await?;
    for log in &rendering.logs {
        log_render(log);
    }
    let prompt_tokens = rendering.prompt.len();
    let capacity = loaded.handle.budget().kv_capacity_tokens;
    if prompt_tokens + 1 > capacity {
        return Err(prompt_too_long(prompt_tokens, capacity));
    }
    let requested_max = request.max_output_tokens.unwrap_or(match request.api {
        ApiKind::Responses => capacity - prompt_tokens,
        _ => args.max_output.unwrap_or(DEFAULT_MAX_OUTPUT),
    });
    let mut warnings = Vec::new();
    if request
        .messages
        .iter()
        .any(|m| matches!(m, crate::api::model::Message::Tool(r) if r.images_omitted > 0))
    {
        warnings.push(Warning::ImageOmitted);
    }

    let parser = syntax.new_parser(&ParserConfig {
        tools: request.tools.clone(),
        reasoning_open: rendering.reasoning_open,
        buffer_args: request.api == ApiKind::Messages,
    });
    let mut generate = GenerateRequest::new(rendering.prompt, requested_max.max(1));
    generate.sampling = request.sampling.resolve(loaded.defaults);
    generate.stop = request.stop.clone();
    let cancel = generate.cancel.clone();
    let generation = loaded.handle.generate(generate).map_err(|e| match e {
        SubmitError::QueueFull(full) => {
            ApiError::new(ErrorKind::QueueFull, "the request queue is full")
                .with_retry_after(full.retry_after)
        }
        SubmitError::Stopped => ApiError::new(ErrorKind::Internal, e.to_string()),
    })?;

    let (tx, ticks) = mpsc::channel(64);
    let relay = Relay {
        tx,
        cancel,
        parser,
        api: request.api,
        requested_max,
        prompt_tokens,
        timing,
        calls_seen: 0,
        reasoning_tokens: 0,
        tracker,
        id: NEXT_ID.fetch_add(1, Ordering::Relaxed),
        generating: false,
    };
    tokio::spawn(relay.run(generation));
    Ok(Turn {
        ticks,
        model: loaded.model_id.clone(),
        prompt_tokens,
        requested_max,
        warnings,
    })
}

/// Renders off the async runtime (templates and tokenization of long prompts take a while).
pub async fn render_request(
    loaded: &Arc<Loaded>,
    request: &ChatRequest,
) -> Result<render::Rendered, ApiError> {
    let loaded = loaded.clone();
    let request = request.clone();
    tokio::task::spawn_blocking(move || {
        render::render(
            &request,
            &RenderContext {
                template: &loaded.template,
                tokenizer: &loaded.tokenizer,
                syntax: loaded.syntax,
                supports_thinking: loaded.supports_thinking,
            },
        )
    })
    .await
    .map_err(|_| ApiError::new(ErrorKind::Internal, "rendering failed"))?
}

pub fn prompt_too_long(prompt: u32, capacity: u32) -> ApiError {
    ApiError::new(
        ErrorKind::ContextLengthExceeded,
        format!(
            "prompt is too long: {prompt} tokens > {} maximum",
            capacity.saturating_sub(1)
        ),
    )
}

fn log_render(log: &RenderLog) {
    let name = match log {
        RenderLog::HistoryArgsRaw(_) => "history-args-raw",
        RenderLog::HistoryCallsSplit(_) => "history-calls-split",
        RenderLog::ThinkingUnsupported => "thinking-unsupported",
        RenderLog::MarksMismatch => "marks-mismatch",
    };
    eprintln!("mrun: {name}");
}

struct Relay {
    tx: mpsc::Sender<Tick>,
    cancel: CancelToken,
    parser: Box<dyn crate::chat::syntax::SyntaxParser>,
    api: ApiKind,
    requested_max: u32,
    prompt_tokens: u32,
    timing: Timing,
    calls_seen: u32,
    reasoning_tokens: u32,
    tracker: Arc<Tracker>,
    id: u64,
    generating: bool,
}

static NEXT_ID: AtomicU64 = AtomicU64::new(1);

impl Relay {
    async fn run(self, generation: crate::runtime::handle::Generation) {
        let tracker = self.tracker.clone();
        let id = self.id;
        self.relay(generation).await;
        tracker.update(|active| {
            if active.as_ref().is_some_and(|a| a.id == id) {
                *active = None;
            }
        });
    }

    async fn relay(mut self, generation: crate::runtime::handle::Generation) {
        let arrived = Instant::now();
        let mut started = false;
        loop {
            let wait = if started {
                self.timing.keepalive
            } else {
                // While queued, wake for the deadline as well as for keepalives.
                self.timing
                    .keepalive
                    .min(self.timing.queue_wait.saturating_sub(arrived.elapsed()))
                    .max(Duration::from_millis(1))
            };
            let next = tokio::select! {
                _ = self.tx.closed() => {
                    // The response is gone: stop the request in whatever phase it is.
                    self.cancel.cancel();
                    return;
                }
                next = tokio::time::timeout(wait, generation.events.recv_async()) => next,
            };
            match next {
                Err(_) => {
                    if !started && arrived.elapsed() >= self.timing.queue_wait {
                        self.cancel.cancel();
                        let error = ApiError::new(
                            ErrorKind::QueueTimeout,
                            "the request waited too long in the queue",
                        );
                        let _ = self.tx.send(Tick::Event(GenEvent::Failed(error))).await;
                        return;
                    }
                    if self.tx.send(Tick::Keepalive).await.is_err() {
                        self.cancel.cancel();
                        return;
                    }
                }
                Ok(Err(_)) => break, // events closed: `Finished` follows
                Ok(Ok(event)) => {
                    let mut out = ParseOut::default();
                    match event {
                        Event::Started {
                            prompt,
                            cached,
                            max_tokens,
                        } => {
                            started = true;
                            let (id, api) = (self.id, self.api);
                            self.tracker.update(|active| {
                                *active = Some(Active {
                                    id,
                                    api,
                                    prompt_tokens: prompt,
                                    cached,
                                    phase: "prefill",
                                    progress: cached,
                                });
                            });
                            let started = GenEvent::Started {
                                input: prompt,
                                cached,
                                max_tokens,
                            };
                            if !self.send(vec![started]).await {
                                return;
                            }
                            continue;
                        }
                        Event::Progress { done, .. } => {
                            let id = self.id;
                            self.tracker.update(|active| {
                                if let Some(a) = active.as_mut().filter(|a| a.id == id) {
                                    a.progress = done;
                                }
                            });
                            continue;
                        }
                        Event::Token { id, text, .. } => {
                            if !self.generating {
                                self.generating = true;
                                let me = self.id;
                                self.tracker.update(|active| {
                                    if let Some(a) = active.as_mut().filter(|a| a.id == me) {
                                        a.phase = "generating";
                                    }
                                });
                            }
                            self.parser.feed(id, &text, &mut out);
                            if out
                                .events
                                .iter()
                                .any(|e| matches!(e, GenEvent::ReasoningDelta(_)))
                            {
                                self.reasoning_tokens += 1;
                            }
                        }
                        Event::Flush { text } => self.parser.feed(u32::MAX, &text, &mut out),
                    }
                    if !self.forward(out).await {
                        return;
                    }
                }
            }
        }
        let Ok(finished) = generation.finished.recv_async().await else {
            let error = ApiError::new(ErrorKind::Internal, "the engine thread has stopped");
            let _ = self.tx.send(Tick::Event(GenEvent::Failed(error))).await;
            return;
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
        };
        let mut out = ParseOut::default();
        let terminal = match finished.reason {
            FinishReason::Cancelled => return,
            FinishReason::Rejected(message) => GenEvent::Failed(
                if message.contains("too long") || message.contains("capacity") {
                    ApiError::new(ErrorKind::ContextLengthExceeded, message)
                } else {
                    ApiError::new(ErrorKind::InvalidRequest, message)
                },
            ),
            FinishReason::Error(message) => {
                GenEvent::Failed(ApiError::new(ErrorKind::Overloaded, message))
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
            FinishReason::Stop { stop, .. } => {
                self.parser.finish(ParseEnd::Stop, &mut out);
                let calls = self.calls_seen
                    + out
                        .events
                        .iter()
                        .filter(|e| matches!(e, GenEvent::ToolStart { .. }))
                        .count() as u32;
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
        if !self.forward(out).await {
            return;
        }
        let _ = self.tx.send(Tick::Event(terminal)).await;
    }

    /// Assigns call ids, logs parser entries and sends the events on.
    async fn forward(&mut self, out: ParseOut) -> bool {
        for log in &out.logs {
            match log {
                ParseLog::ToolUnknown(name) => eprintln!("mrun: tool-unknown ({name})"),
                ParseLog::BuiltinRecipient(name) => eprintln!("mrun: tool-unknown ({name})"),
            }
        }
        let mut events = out.events;
        for event in &mut events {
            if let GenEvent::ToolStart { id, .. } = event {
                self.calls_seen += 1;
                if id.is_empty() {
                    *id = crate::api::sse::new_id(match self.api {
                        ApiKind::Messages => "toolu_",
                        ApiKind::Chat | ApiKind::Responses => "call_",
                    });
                }
            }
        }
        self.send(events).await
    }

    async fn send(&mut self, events: Vec<GenEvent>) -> bool {
        for event in events {
            if self.tx.send(Tick::Event(event)).await.is_err() {
                self.cancel.cancel();
                return false;
            }
        }
        true
    }
}
