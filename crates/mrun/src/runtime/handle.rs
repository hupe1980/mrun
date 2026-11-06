//! `EngineHandle`: owns an engine on a dedicated thread; callers stream events through a bounded
//! channel (blocking `recv` or `recv_async`) and then read the request's `Finished` from a
//! one-slot channel. A consumer that stops reading for the stall timeout cancels its request,
//! and the engine never waits on that consumer again; `Finished` is still delivered. The engine
//! stays loaded until the last handle is dropped; idle time never unloads it.

use crate::cache::{CacheMode, CacheStats, ClearReport, PrefixStore};
use crate::chat::splice::Ledger;
use crate::chat::tokenizer::Tokenizer;
use crate::runtime::generate::{
    self, Event, EventSink, Failure, FinishReason, Finished, GenerateRequest, Stats,
};
use crate::runtime::pressure::{self, PressureGauge};
use crate::runtime::queue::{QueueFull, QueueGauge};
use flume::{Receiver, Sender, bounded, unbounded};
use mrun_engine::{CancelToken, Engine, MemoryBudget, ModelInfo};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, PoisonError};
use std::thread;
use std::time::{Duration, Instant};

/// Events buffered per request before a slow consumer starts the stall timer.
pub const EVENT_CAPACITY: usize = 64;
/// Default time a consumer may stop reading before its request is cancelled.
pub const DEFAULT_STALL_TIMEOUT: Duration = Duration::from_secs(10);

enum Command {
    Generate {
        request: GenerateRequest,
        ticket: Ticket,
        events: Sender<Event>,
        finished: Sender<Finished>,
    },
    Reset,
    ClearCache {
        reply: Sender<Option<ClearReport>>,
    },
}

impl std::fmt::Debug for Command {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Generate { .. } => f.write_str("Generate"),
            Self::Reset => f.write_str("Reset"),
            Self::ClearCache { .. } => f.write_str("ClearCache"),
        }
    }
}

/// How often the idle worker looks for memory pressure.
const IDLE_PRESSURE_POLL: Duration = Duration::from_secs(1);

#[derive(Debug, Clone)]
pub struct EngineHandle {
    commands: Sender<Command>,
    info: ModelInfo,
    budget: MemoryBudget,
    pressure: Arc<PressureGauge>,
    queue: Arc<QueueGauge>,
    /// The policy's cache statistics after the last command (`None` for `Session`).
    cache: Arc<Mutex<Option<CacheStats>>>,
    /// Reply records for token splicing, shared with the prefix store (D051).
    ledger: Arc<Mutex<Ledger>>,
}

/// Why a request was not queued.
#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
pub enum SubmitError {
    #[error("the queue is full; retry after {} s", .0.retry_after)]
    QueueFull(QueueFull),
    #[error("the engine thread has stopped")]
    Stopped,
}

/// A queued request's hold on its queue slot and its cancel token. Cancelling frees the slot at
/// once (the worker skips the request when it reaches it); the worker frees it on dequeue.
#[derive(Clone, Debug)]
pub struct Ticket {
    inner: Arc<TicketInner>,
}

#[derive(Debug)]
struct TicketInner {
    cancel: CancelToken,
    queued: AtomicBool,
    queue: Arc<QueueGauge>,
}

impl Ticket {
    fn new(cancel: CancelToken, queue: Arc<QueueGauge>) -> Self {
        Self {
            inner: Arc::new(TicketInner {
                cancel,
                queued: AtomicBool::new(true),
                queue,
            }),
        }
    }

    /// Cancels the request in whatever phase it is.
    pub fn cancel(&self) {
        self.inner.cancel.cancel();
        self.leave_queue();
    }

    pub fn is_cancelled(&self) -> bool {
        self.inner.cancel.is_cancelled()
    }

    /// Still waiting for the worker.
    pub fn queued(&self) -> bool {
        self.inner.queued.load(Ordering::Acquire)
    }

    fn leave_queue(&self) {
        if self.inner.queued.swap(false, Ordering::AcqRel) {
            self.inner.queue.release();
        }
    }
}

/// Receivers of one request: drain `events` (closed at the finish), then read `finished`.
#[derive(Debug)]
pub struct Generation {
    pub events: Receiver<Event>,
    pub finished: Receiver<Finished>,
    pub ticket: Ticket,
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
    /// Starts the worker thread with the prefix cache keyed by the model identity (`mrun run`);
    /// `factory` builds the engine on that thread.
    pub fn spawn<E, F>(
        factory: F,
        tokenizer: Arc<Tokenizer>,
        stall_timeout: Duration,
    ) -> Result<Self, String>
    where
        E: Engine + 'static,
        F: FnOnce() -> Result<E, String> + Send + 'static,
    {
        Self::spawn_with(factory, tokenizer, stall_timeout, |_: &ModelInfo| {
            PrefixStore::new(CacheMode::Cached)
        })
    }

    /// Starts the worker thread with the prefix store `policy` builds from the loaded model.
    pub fn spawn_with<E, F, Q>(
        factory: F,
        tokenizer: Arc<Tokenizer>,
        stall_timeout: Duration,
        policy: Q,
    ) -> Result<Self, String>
    where
        E: Engine + 'static,
        F: FnOnce() -> Result<E, String> + Send + 'static,
        Q: FnOnce(&ModelInfo) -> PrefixStore + Send + 'static,
    {
        let (commands, inbox) = unbounded::<Command>();
        let (ready_tx, ready_rx) = bounded(1);
        let pressure = Arc::new(PressureGauge::default());
        pressure::watch(&pressure);
        let gauge = pressure.clone();
        let queue = Arc::new(QueueGauge::default());
        let worker_queue = queue.clone();
        let cache = Arc::new(Mutex::new(None));
        let worker_cache = cache.clone();
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
                let mut policy = policy(engine.info());
                *worker_cache.lock().unwrap_or_else(PoisonError::into_inner) = Some(policy.stats());
                let _ = ready_tx.send(Ok((
                    engine.info().clone(),
                    *engine.budget(),
                    policy.ledger(),
                )));
                worker(
                    &mut engine,
                    &mut policy,
                    &tokenizer,
                    &inbox,
                    stall_timeout,
                    &gauge,
                    &worker_queue,
                    &worker_cache,
                );
            })
            .map_err(|error| format!("cannot start engine thread: {error}"))?;
        let (info, budget, ledger) = ready_rx
            .recv()
            .map_err(|_| "engine thread exited during startup".to_string())??;
        Ok(Self {
            commands,
            info,
            budget,
            pressure,
            queue,
            cache,
            ledger,
        })
    }

    /// Finished replies for token splicing (D030).
    pub fn ledger(&self) -> &Arc<Mutex<Ledger>> {
        &self.ledger
    }

    /// Cache statistics after the last command.
    pub fn cache_stats(&self) -> Option<CacheStats> {
        *self.cache.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// `mrun cache clear`: queued like a request; returns the receiver of its report.
    pub fn clear_cache(&self) -> Result<Receiver<Option<ClearReport>>, SubmitError> {
        let (reply, receiver) = bounded(1);
        self.commands
            .send(Command::ClearCache { reply })
            .map_err(|_| SubmitError::Stopped)?;
        Ok(receiver)
    }

    /// Waiting requests and the running flag; the capacity is set with `--queue`.
    pub fn queue(&self) -> &Arc<QueueGauge> {
        &self.queue
    }

    pub fn info(&self) -> &ModelInfo {
        &self.info
    }

    pub fn budget(&self) -> &MemoryBudget {
        &self.budget
    }

    /// Where memory-pressure levels are raised (the system poller, tests).
    pub fn pressure(&self) -> &Arc<PressureGauge> {
        &self.pressure
    }

    /// Queues a request, or refuses at once when the queue is full. A request whose cancel
    /// token fires while it waits finishes `Cancelled` without reaching the engine.
    pub fn generate(&self, request: GenerateRequest) -> Result<Generation, SubmitError> {
        self.queue.admit().map_err(SubmitError::QueueFull)?;
        let ticket = Ticket::new(request.cancel.clone(), self.queue.clone());
        let (events, events_rx) = bounded(EVENT_CAPACITY);
        let (finished, finished_rx) = bounded(1);
        if self
            .commands
            .send(Command::Generate {
                request,
                ticket: ticket.clone(),
                events,
                finished,
            })
            .is_err()
        {
            ticket.leave_queue();
            return Err(SubmitError::Stopped);
        }
        Ok(Generation {
            events: events_rx,
            finished: finished_rx,
            ticket,
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

#[allow(clippy::too_many_arguments)]
fn worker<E: Engine>(
    engine: &mut E,
    policy: &mut PrefixStore,
    tokenizer: &Tokenizer,
    inbox: &Receiver<Command>,
    stall_timeout: Duration,
    pressure: &PressureGauge,
    queue: &QueueGauge,
    cache: &Mutex<Option<CacheStats>>,
) {
    let mut waiting: std::collections::VecDeque<Command> = std::collections::VecDeque::new();
    let mut last_turn: Option<Option<String>> = None;
    loop {
        // Everything that arrived waits here, so the next request can be chosen in turns.
        while let Ok(command) = inbox.try_recv() {
            waiting.push_back(command);
        }
        let next = if waiting.is_empty() {
            inbox.recv_timeout(IDLE_PRESSURE_POLL)
        } else {
            Ok(next_turn(&mut waiting, last_turn.as_ref()))
        };
        let command = match next {
            Ok(command) => command,
            Err(flume::RecvTimeoutError::Timeout) => {
                // Idle: memory pressure sheds cache here too (critical also drops the live state).
                if let Some(level) = pressure.take() {
                    policy.on_pressure(engine, level, true);
                    *cache.lock().unwrap_or_else(PoisonError::into_inner) = Some(policy.stats());
                }
                continue;
            }
            Err(flume::RecvTimeoutError::Disconnected) => break,
        };
        match command {
            Command::Generate {
                request,
                ticket,
                events,
                finished,
            } => {
                ticket.leave_queue();
                last_turn = Some(request.turn.clone());
                if request.cancel.is_cancelled() {
                    // Cancelled or expired while waiting: never reaches the engine.
                    drop(events);
                    let _ = finished.try_send(Finished {
                        reason: FinishReason::Cancelled,
                        prefilled: 0,
                        stats: Stats::default(),
                    });
                    continue;
                }
                queue.set_running(true);
                let began = Instant::now();
                let mut sink = ChannelSink {
                    events,
                    stall_timeout,
                    gone: false,
                };
                let outcome = catch_unwind(AssertUnwindSafe(|| {
                    let mut detok = tokenizer.stream();
                    generate::generate(engine, policy, &mut detok, request, &mut sink, pressure)
                }));
                // Close the event stream first so a consumer draining it moves on to `Finished`.
                drop(sink);
                let done = outcome.unwrap_or_else(|_| {
                    // A panic left the engine state unknown: drop everything it holds.
                    engine.reset_all();
                    policy.clear(engine);
                    Finished {
                        reason: FinishReason::Error(Failure::Backend(
                            "internal error; state reset".into(),
                        )),
                        prefilled: 0,
                        stats: Stats::default(),
                    }
                });
                queue.set_running(false);
                queue.record_service(began.elapsed());
                // The slot is empty and never waited on; this fails only if the consumer left.
                let _ = finished.try_send(done);
            }
            Command::Reset => policy.clear(engine),
            Command::ClearCache { reply } => {
                let _ = reply.try_send(Some(policy.clear_cache(engine)));
            }
        }
        *cache.lock().unwrap_or_else(PoisonError::into_inner) = Some(policy.stats());
    }
}

/// The next command: control commands first, then the oldest request of another member than
/// the one just served (026 FR-007), else the oldest.
fn next_turn(
    waiting: &mut std::collections::VecDeque<Command>,
    last: Option<&Option<String>>,
) -> Command {
    let turn = |c: &Command| match c {
        Command::Generate { request, .. } => Some(request.turn.clone()),
        _ => None,
    };
    let at = waiting
        .iter()
        .position(|c| turn(c).is_none())
        .or_else(|| {
            last.and_then(|last| waiting.iter().position(|c| turn(c).as_ref() != Some(last)))
        })
        .unwrap_or(0);
    waiting.remove(at).expect("a waiting command")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chat::prompt::RenderedPrompt;
    use crate::chat::tokenizer::test_support::byte_tokenizer;
    use crate::runtime::pressure::Pressure;
    use crate::runtime::scripted::ScriptedEngine;
    use std::future::Future;
    use std::pin::pin;
    use std::task::{Context, Poll, Wake, Waker};

    fn request(tokenizer: &Tokenizer, prompt: &str) -> GenerateRequest {
        let tokens = tokenizer.encode(prompt).unwrap();
        GenerateRequest::new(RenderedPrompt::from_tokens(tokens), 100)
    }

    #[test]
    fn members_take_turns() {
        let tokenizer = byte_tokenizer();
        let command = |turn: Option<&str>, prompt: &str| {
            let mut request = request(&tokenizer, prompt);
            request.turn = turn.map(str::to_string);
            let (events, _) = bounded(1);
            let (finished, _) = bounded(1);
            Command::Generate {
                request,
                ticket: Ticket::new(CancelToken::new(), Arc::default()),
                events,
                finished,
            }
        };
        let mut waiting: std::collections::VecDeque<Command> = [
            command(Some("anna"), "a1"),
            command(Some("anna"), "a2"),
            command(None, "b1"),
            command(Some("anna"), "a3"),
        ]
        .into();
        let mut order = Vec::new();
        let mut last = Some(Some("anna".to_string()));
        while !waiting.is_empty() {
            if let Command::Generate { request, .. } = next_turn(&mut waiting, last.as_ref()) {
                order.push(
                    tokenizer
                        .inner()
                        .decode(&request.prompt.tokens, false)
                        .unwrap(),
                );
                last = Some(request.turn);
            }
        }
        assert_eq!(order, ["b1", "a1", "a2", "a3"]);
        waiting.push_back(Command::Reset);
        assert!(matches!(next_turn(&mut waiting, None), Command::Reset));
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
    fn xapi_queue_full_immediate_error_and_cancel_while_queued() {
        let tokenizer = Arc::new(byte_tokenizer());
        let script = long_script(&tokenizer);
        let handle = EngineHandle::spawn(
            move || Ok(ScriptedEngine::new(script, 10_000)),
            tokenizer.clone(),
            Duration::from_secs(5),
        )
        .unwrap();
        handle.queue().set_capacity(1);
        // The first request blocks the worker: its consumer reads nothing.
        let busy = handle.generate(request(&tokenizer, "busy")).unwrap();
        while handle.queue().waiting() > 0 || !handle.queue().running() {
            thread::yield_now();
        }
        let waiting = request(&tokenizer, "waiting");
        let cancel = waiting.cancel.clone();
        let queued = handle.generate(waiting).unwrap();
        assert_eq!(handle.queue().waiting(), 1);
        let refused = handle.generate(request(&tokenizer, "third")).unwrap_err();
        assert!(
            matches!(refused, SubmitError::QueueFull(full) if (1..=60).contains(&full.retry_after))
        );

        // Cancelling frees the slot at once, while the worker is still busy.
        queued.ticket.cancel();
        assert!(cancel.is_cancelled());
        assert_eq!(handle.queue().waiting(), 0);
        let third = handle.generate(request(&tokenizer, "third")).unwrap();
        third.ticket.cancel();
        drop(busy);
        let (events, finished) = queued.wait().unwrap();
        assert!(events.is_empty());
        assert_eq!(
            (finished.reason, finished.prefilled),
            (FinishReason::Cancelled, 0)
        );
        assert_eq!(handle.queue().waiting(), 0);
        // Order is kept and the worker serves the next request.
        let (_, next) = handle
            .generate(request(&tokenizer, "next"))
            .unwrap()
            .wait()
            .unwrap();
        assert_eq!(next.reason, FinishReason::Length);
    }

    #[test]
    fn critical_pressure_while_idle_drops_the_live_state() {
        let script = long_script(&byte_tokenizer());
        let (handle, tokenizer) = spawn(move || ScriptedEngine::new(script, 10_000));
        handle
            .generate(request(&tokenizer, "first"))
            .unwrap()
            .wait()
            .unwrap();
        assert!(handle.cache_stats().unwrap().live_tokens > 0);
        handle.pressure().raise(Pressure::Critical);
        let deadline = std::time::Instant::now() + Duration::from_secs(5);
        // Waits for the live state to go, not for the first pressure event: the machine's own
        // pressure (a warning while the first request ran) may already have counted one.
        while handle.cache_stats().unwrap().live_tokens > 0 {
            assert!(
                std::time::Instant::now() < deadline,
                "the idle worker never saw it"
            );
            thread::sleep(Duration::from_millis(50));
        }
        assert!(handle.cache_stats().unwrap().pressure_events > 0);
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
        assert!(matches!(
            finished.reason,
            FinishReason::Stop { stop: None, .. }
        ));
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
