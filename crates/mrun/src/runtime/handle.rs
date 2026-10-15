//! `EngineHandle`: owns an engine on a dedicated thread; async callers stream events through a
//! bounded channel. A consumer that stops reading for the stall timeout cancels its request.

use crate::chat::tokenizer::Tokenizer;
use crate::runtime::generate::{self, Event, EventSink, GenerateRequest, Session};
use crossbeam_channel::{Receiver, Sender, bounded, unbounded};
use mrun_engine::{Engine, MemoryBudget, ModelInfo};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

/// Events buffered per request before a slow consumer starts the stall timer.
pub const EVENT_CAPACITY: usize = 64;
/// Default time a consumer may stop reading before its request is cancelled.
pub const DEFAULT_STALL_TIMEOUT: Duration = Duration::from_secs(10);

enum Command {
    Generate {
        request: GenerateRequest,
        events: Sender<Event>,
    },
    Reset,
}

#[derive(Debug, Clone)]
pub struct EngineHandle {
    commands: Sender<Command>,
    info: ModelInfo,
    budget: MemoryBudget,
    stall_timeout: Duration,
}

impl std::fmt::Debug for Command {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Generate { .. } => f.write_str("Generate"),
            Self::Reset => f.write_str("Reset"),
        }
    }
}

impl EngineHandle {
    /// Starts the worker thread; `factory` builds the engine on that thread.
    pub fn spawn<E, F>(
        factory: F,
        tokenizer: Arc<Tokenizer>,
        stall_timeout: Duration,
    ) -> Result<Self, String>
    where
        E: Engine + 'static,
        F: FnOnce() -> Result<E, String> + Send + 'static,
    {
        let (commands, inbox) = unbounded::<Command>();
        let (ready_tx, ready_rx) = bounded(1);
        thread::Builder::new()
            .name("mrun-engine".into())
            .spawn(move || {
                let mut engine = match factory() {
                    Ok(engine) => engine,
                    Err(error) => {
                        let _ = ready_tx.send(Err(error));
                        return;
                    }
                };
                let _ = ready_tx.send(Ok((engine.info().clone(), *engine.budget())));
                worker(&mut engine, &tokenizer, &inbox, stall_timeout);
            })
            .map_err(|error| format!("cannot start engine thread: {error}"))?;
        let (info, budget) = ready_rx
            .recv()
            .map_err(|_| "engine thread exited during startup".to_string())??;
        Ok(Self {
            commands,
            info,
            budget,
            stall_timeout,
        })
    }

    pub fn info(&self) -> &ModelInfo {
        &self.info
    }

    pub fn budget(&self) -> &MemoryBudget {
        &self.budget
    }

    pub fn stall_timeout(&self) -> Duration {
        self.stall_timeout
    }

    /// Queues a request; events arrive on the returned receiver, ending with `Finished`.
    pub fn generate(&self, request: GenerateRequest) -> Result<Receiver<Event>, String> {
        let (events, receiver) = bounded(EVENT_CAPACITY);
        self.commands
            .send(Command::Generate { request, events })
            .map_err(|_| "the engine thread has stopped".to_string())?;
        Ok(receiver)
    }

    /// Forgets the conversation state.
    pub fn reset(&self) -> Result<(), String> {
        self.commands
            .send(Command::Reset)
            .map_err(|_| "the engine thread has stopped".to_string())
    }
}

struct ChannelSink {
    events: Sender<Event>,
    stall_timeout: Duration,
}

impl EventSink for ChannelSink {
    fn send(&mut self, event: Event) -> bool {
        self.events.send_timeout(event, self.stall_timeout).is_ok()
    }
}

fn worker<E: Engine>(
    engine: &mut E,
    tokenizer: &Tokenizer,
    inbox: &Receiver<Command>,
    stall_timeout: Duration,
) {
    let mut session = Session::default();
    for command in inbox {
        match command {
            Command::Generate { request, events } => {
                let mut sink = ChannelSink {
                    events,
                    stall_timeout,
                };
                let mut detok = tokenizer.stream();
                generate::generate(engine, &mut session, &mut detok, request, &mut sink);
            }
            Command::Reset => session.clear(engine),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chat::tokenizer::test_support::byte_tokenizer;
    use crate::runtime::generate::FinishReason;
    use crate::runtime::scripted::ScriptedEngine;
    use mrun_engine::{CancelToken, Sampling};

    fn request(tokenizer: &Tokenizer, prompt: &str) -> GenerateRequest {
        GenerateRequest {
            prompt: tokenizer.encode(prompt).unwrap(),
            sampling: Sampling::GREEDY,
            max_tokens: 100,
            stop: Vec::new(),
            cancel: CancelToken::new(),
        }
    }

    #[test]
    fn worker_streams_events_and_survives_a_dropped_consumer() {
        let tokenizer = Arc::new(byte_tokenizer());
        let script = tokenizer.encode(&"z".repeat(200)).unwrap();
        let handle = EngineHandle::spawn(
            move || Ok(ScriptedEngine::new(script, 10_000)),
            tokenizer.clone(),
            Duration::from_millis(50),
        )
        .unwrap();
        assert_eq!(handle.info().identity, "scripted");

        // A consumer that never reads: the request is cancelled after the stall timeout.
        let abandoned = handle.generate(request(&tokenizer, "first")).unwrap();
        drop(abandoned);

        let events: Vec<_> = handle
            .generate(request(&tokenizer, "second"))
            .unwrap()
            .iter()
            .collect();
        assert!(matches!(
            events.last(),
            Some(Event::Finished {
                reason: FinishReason::Length,
                ..
            })
        ));
    }

    #[test]
    fn factory_errors_are_reported() {
        let tokenizer = Arc::new(byte_tokenizer());
        let error = EngineHandle::spawn(
            || Err::<ScriptedEngine, _>("no model".to_string()),
            tokenizer,
            DEFAULT_STALL_TIMEOUT,
        )
        .unwrap_err();
        assert_eq!(error, "no model");
    }
}
