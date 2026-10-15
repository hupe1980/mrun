//! `EngineHandle`: owns an engine on a dedicated thread; callers stream events through a bounded
//! channel (blocking `recv` or `recv_async`) and then read the request's `Finished` from a
//! one-slot channel. A consumer that stops reading for the stall timeout cancels its request,
//! and the engine never waits on that consumer again; `Finished` is still delivered. The engine
//! stays loaded until the last handle is dropped; idle time never unloads it.

use crate::chat::tokenizer::Tokenizer;
use crate::runtime::generate::{
    self, Event, EventSink, FinishReason, Finished, GenerateRequest, ReusePolicy, Stats,
};
use crate::runtime::session::Session;
use flume::{Receiver, Sender, bounded, unbounded};
use mrun_engine::{Engine, MemoryBudget, ModelInfo};
use std::panic::{AssertUnwindSafe, catch_unwind};
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
        finished: Sender<Finished>,
    },
    Reset,
}

impl std::fmt::Debug for Command {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Generate { .. } => f.write_str("Generate"),
            Self::Reset => f.write_str("Reset"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct EngineHandle {
    commands: Sender<Command>,
    info: ModelInfo,
    budget: MemoryBudget,
}

/// Receivers of one request: drain `events` (closed at the finish), then read `finished`.
#[derive(Debug)]
pub struct Generation {
    pub events: Receiver<Event>,
    pub finished: Receiver<Finished>,
}

impl Generation {
    /// Blocks until the request ends, collecting its events.
    pub fn wait(self) -> Result<(Vec<Event>, Finished), String> {
        let events = self.events.iter().collect();
        let finished = self
            .finished
            .recv()
            .map_err(|_| "the engine thread has stopped".to_string())?;
        Ok((events, finished))
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
        })
    }

    pub fn info(&self) -> &ModelInfo {
        &self.info
    }

    pub fn budget(&self) -> &MemoryBudget {
        &self.budget
    }

    /// Queues a request.
    pub fn generate(&self, request: GenerateRequest) -> Result<Generation, String> {
        let (events, events_rx) = bounded(EVENT_CAPACITY);
        let (finished, finished_rx) = bounded(1);
        self.commands
            .send(Command::Generate {
                request,
                events,
                finished,
            })
            .map_err(|_| "the engine thread has stopped".to_string())?;
        Ok(Generation {
            events: events_rx,
            finished: finished_rx,
        })
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
    /// Set once the consumer disconnected or stalled; nothing more is sent to it.
    gone: bool,
}

impl EventSink for ChannelSink {
    fn send(&mut self, event: Event) -> bool {
        if !self.gone {
            self.gone = self.events.send_timeout(event, self.stall_timeout).is_err();
        }
        !self.gone
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
            Command::Generate {
                request,
                events,
                finished,
            } => {
                let mut sink = ChannelSink {
                    events,
                    stall_timeout,
                    gone: false,
                };
                let outcome = catch_unwind(AssertUnwindSafe(|| {
                    let mut detok = tokenizer.stream();
                    generate::generate(engine, &mut session, &mut detok, request, &mut sink)
                }));
                // Close the event stream first so a consumer draining it moves on to `Finished`.
                drop(sink);
                let done = outcome.unwrap_or_else(|_| {
                    // A panic left the engine state unknown: drop everything it holds.
                    engine.reset_all();
                    session = Session::default();
                    Finished {
                        reason: FinishReason::Error("internal error; state reset".into()),
                        prefilled: 0,
                        stats: Stats::default(),
                    }
                });
                // The slot is empty and never waited on; this fails only if the consumer left.
                let _ = finished.try_send(done);
            }
            Command::Reset => session.clear(engine),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chat::tokenizer::test_support::byte_tokenizer;
    use crate::runtime::scripted::ScriptedEngine;
    use mrun_engine::{CancelToken, Sampling};
    use std::future::Future;
    use std::pin::pin;
    use std::task::{Context, Poll, Wake, Waker};

    fn request(tokenizer: &Tokenizer, prompt: &str) -> GenerateRequest {
        GenerateRequest {
            prompt: tokenizer.encode(prompt).unwrap(),
            sampling: Sampling::GREEDY,
            max_tokens: 100,
            stop: Vec::new(),
            cancel: CancelToken::new(),
            generation_start: None,
        }
    }

    fn spawn(
        engine: impl FnOnce() -> ScriptedEngine + Send + 'static,
    ) -> (EngineHandle, Arc<Tokenizer>) {
        let tokenizer = Arc::new(byte_tokenizer());
        let handle = EngineHandle::spawn(
            move || Ok(engine()),
            tokenizer.clone(),
            Duration::from_millis(50),
        )
        .unwrap();
        (handle, tokenizer)
    }

    fn long_script(tokenizer: &Tokenizer) -> Vec<u32> {
        tokenizer.encode(&"z".repeat(200)).unwrap()
    }

    #[test]
    fn worker_streams_events_and_survives_a_dropped_consumer() {
        let script = long_script(&byte_tokenizer());
        let (handle, tokenizer) = spawn(move || ScriptedEngine::new(script, 10_000));
        assert_eq!(handle.info().identity, "scripted");

        drop(handle.generate(request(&tokenizer, "first")).unwrap());

        let (events, finished) = handle
            .generate(request(&tokenizer, "second"))
            .unwrap()
            .wait()
            .unwrap();
        assert_eq!(finished.reason, FinishReason::Length);
        assert!(matches!(events[0], Event::Started { .. }));
    }

    #[test]
    fn finished_arrives_although_the_consumer_never_reads_events() {
        let script = long_script(&byte_tokenizer());
        let (handle, tokenizer) = spawn(move || ScriptedEngine::new(script, 10_000));
        let stalled = handle.generate(request(&tokenizer, "q")).unwrap();
        let finished = stalled.finished.recv().unwrap();
        assert_eq!(finished.reason, FinishReason::Cancelled);
        assert_eq!(stalled.events.len(), EVENT_CAPACITY);
        // The worker keeps serving.
        let (_, next) = handle
            .generate(request(&tokenizer, "again"))
            .unwrap()
            .wait()
            .unwrap();
        assert_eq!(next.reason, FinishReason::Length);
    }

    #[test]
    fn a_panic_finishes_with_an_error_and_the_worker_survives() {
        let resets = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let observed = resets.clone();
        let (handle, tokenizer) = spawn(move || {
            let mut engine = ScriptedEngine::new(Vec::new(), 100);
            engine.panic_on_forward = true;
            engine.reset_all_calls = observed;
            engine
        });
        for _ in 0..2 {
            let (_, finished) = handle
                .generate(request(&tokenizer, "q"))
                .unwrap()
                .wait()
                .unwrap();
            assert!(matches!(finished.reason, FinishReason::Error(_)));
            assert_eq!(finished.prefilled, 0);
        }
        assert_eq!(resets.load(std::sync::atomic::Ordering::Relaxed), 2);
    }

    #[test]
    fn async_consumer_receives_events_and_finished() {
        struct Unpark(thread::Thread);
        impl Wake for Unpark {
            fn wake(self: Arc<Self>) {
                self.0.unpark();
            }
        }
        fn block_on<T>(future: impl Future<Output = T>) -> T {
            let waker = Waker::from(Arc::new(Unpark(thread::current())));
            let mut context = Context::from_waker(&waker);
            let mut future = pin!(future);
            loop {
                match future.as_mut().poll(&mut context) {
                    Poll::Ready(value) => return value,
                    Poll::Pending => thread::park(),
                }
            }
        }

        let script = byte_tokenizer().encode("hey").unwrap();
        let (handle, tokenizer) = spawn(move || ScriptedEngine::new(script, 1_000));
        let generation = handle.generate(request(&tokenizer, "q")).unwrap();
        let (text, finished) = block_on(async {
            let mut text = String::new();
            while let Ok(event) = generation.events.recv_async().await {
                if let Event::Token { text: piece, .. } = event {
                    text.push_str(&piece);
                }
            }
            (text, generation.finished.recv_async().await.unwrap())
        });
        assert_eq!(text, "hey");
        assert_eq!(finished.reason, FinishReason::Stop { stop: None });
    }

    #[test]
    fn the_model_stays_resident_across_idle_periods() {
        let built = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let alive = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let (counter, owned) = (built.clone(), alive.clone());
        let script = long_script(&byte_tokenizer());
        let (handle, tokenizer) = spawn(move || {
            counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let mut engine = ScriptedEngine::new(script, 10_000);
            // The engine holds this Arc for as long as it exists.
            engine.reset_all_calls = owned;
            engine
        });
        let mut req = request(&tokenizer, "first");
        req.max_tokens = 2;
        handle.generate(req).unwrap().wait().unwrap();
        // Idle for several stall timeouts: the worker blocks on its inbox and has no timer.
        thread::sleep(Duration::from_millis(300));
        assert_eq!(Arc::strong_count(&alive), 2, "engine still loaded");
        handle.reset().unwrap();
        let (events, finished) = handle
            .generate(request(&tokenizer, "first"))
            .unwrap()
            .wait()
            .unwrap();
        assert_eq!(finished.reason, FinishReason::Length);
        assert!(
            matches!(events[0], Event::Started { cached: 0, .. }),
            "state was reset"
        );
        assert_eq!(
            built.load(std::sync::atomic::Ordering::Relaxed),
            1,
            "loaded once"
        );
        assert_eq!(
            alive.load(std::sync::atomic::Ordering::Relaxed),
            0,
            "no reset_all"
        );
        assert_eq!(Arc::strong_count(&alive), 2);
        drop(handle);
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
