//! Parallel lanes (011): the scheduler that runs the jobs of up to four lanes on the engine
//! thread.
//!
//! Each iteration delivers held events, ends cancelled jobs, advances prefill and decodes:
//!
//! - **Prefill**: a layer per iteration of the chunk of the reading job with the fewest prompt
//!   tokens left (FR-019), so a short prompt that arrives takes over from a long one at the next
//!   layer (the long one's chunk waits, paused, on its lane), and decoding lanes step once
//!   prefill has had [`PREFILL_SHARE`] decode steps' worth of time (FR-003). One lane runs
//!   whole chunks (`prefill_chunk`), as it always did.
//! - **Decode**: every decoding job whose client takes events, in one batched step
//!   (`forward_lanes`, `sample_lanes`); a job alone takes its own `forward`/`sample`, and
//!   verifies drafts (006) only while it is the only busy job (FR-015).
//! - **Events** never block the thread: a job whose client stops reading waits (its events
//!   held) while the others go on, and is cancelled after the stall timeout. A finished job's
//!   held events are delivered before its `Finished`.
//!
//! With one busy job the engine sees exactly the calls one lane always made (FR-002).

use crate::cache::PrefixStore;
use crate::runtime::generate::{Delivery, EventSink, Failure, FinishReason, Finished, Job, Next};
use crate::runtime::generate::{relieve_pressure, with_memory};
use crate::runtime::pressure::PressureGauge;
use mrun_engine::{ChunkProgress, Engine, EngineError, Pick};
use std::time::{Duration, Instant};

/// Prefill gets this many decode steps' worth of time before decoding lanes step again.
pub const PREFILL_SHARE: u32 = 4;

struct Running<'a, T> {
    job: Job<'a>,
    tag: T,
    /// Its prefill chunk is in flight (begun, possibly paused between layers).
    in_flight: bool,
    /// Admission order (ties of FR-019).
    order: u64,
}

/// A finished job whose events are still on their way to its client.
struct Draining<'a, T> {
    sink: Box<dyn EventSink + 'a>,
    tag: T,
    finished: Finished,
    since: Instant,
}

/// The jobs of every lane; `T` is what the caller keeps with each (its reply channel).
pub struct Lanes<'a, T> {
    jobs: Vec<Option<Running<'a, T>>>,
    draining: Vec<Draining<'a, T>>,
    /// Prefill time since the last decode step, and the fastest recent decode step.
    prefill_since: Duration,
    decode_time: Duration,
    admitted: u64,
    stall_timeout: Duration,
}

impl<T> std::fmt::Debug for Lanes<'_, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Lanes")
            .field("lanes", &self.jobs.len())
            .field("busy", &self.busy())
            .finish_non_exhaustive()
    }
}

impl<'a, T> Lanes<'a, T> {
    pub fn new(lanes: usize, stall_timeout: Duration) -> Self {
        Self {
            jobs: (0..lanes.max(1)).map(|_| None).collect(),
            draining: Vec::new(),
            prefill_since: Duration::ZERO,
            decode_time: Duration::ZERO,
            admitted: 0,
            stall_timeout,
        }
    }

    pub fn lanes(&self) -> usize {
        self.jobs.len()
    }

    /// Lanes without a job.
    pub fn free(&self) -> Vec<usize> {
        (0..self.jobs.len())
            .filter(|&l| self.jobs[l].is_none())
            .collect()
    }

    /// Jobs running.
    pub fn busy(&self) -> usize {
        self.jobs.iter().flatten().count()
    }

    /// Whether anything is left to do: running jobs or events still on their way.
    pub fn active(&self) -> bool {
        self.busy() > 0 || !self.draining.is_empty()
    }

    /// Context the running jobs reserve (FR-007).
    pub fn reserved(&self) -> u32 {
        self.jobs.iter().flatten().map(|r| r.job.reservation).sum()
    }

    /// Starts `job` on its lane (free); a job that ends at once (rejected, cancelled) is
    /// returned finished.
    pub fn admit<E: Engine>(
        &mut self,
        engine: &mut E,
        store: &mut PrefixStore,
        pressure: &PressureGauge,
        mut job: Job<'a>,
        tag: T,
    ) {
        let lane = job.lane;
        debug_assert!(self.jobs[lane].is_none(), "a free lane");
        store.select(engine, lane);
        store.set_busy(lane, true);
        let started = job.start(engine, store, pressure);
        self.admitted += 1;
        self.jobs[lane] = Some(Running {
            job,
            tag,
            in_flight: false,
            order: self.admitted,
        });
        self.settle(engine, store, lane, started);
    }

    /// One iteration; returns the jobs whose `Finished` is ready, and whether anything moved.
    pub fn step<E: Engine>(
        &mut self,
        engine: &mut E,
        store: &mut PrefixStore,
        pressure: &PressureGauge,
    ) -> (Vec<(T, Finished)>, bool) {
        let mut moved = false;
        // Clients: held events, gone or stalled consumers, cancellations.
        let mut waiting = vec![false; self.jobs.len()];
        #[expect(
            clippy::needless_range_loop,
            reason = "the body ends jobs through `self`"
        )]
        for lane in 0..self.jobs.len() {
            let Some(running) = &mut self.jobs[lane] else {
                continue;
            };
            match running.job.flush() {
                Delivery::Done => {}
                Delivery::Waiting(since) if since.elapsed() < self.stall_timeout => {
                    waiting[lane] = true;
                }
                Delivery::Waiting(_) | Delivery::Gone => running.job.request.cancel.cancel(),
            }
            if running.job.cancelled() {
                self.end(engine, store, lane, Ok(FinishReason::Cancelled));
                moved = true;
            }
        }
        moved |= self.prefill(engine, store, pressure, &waiting);
        moved |= self.decode(engine, store, pressure, &waiting);
        let ready = self.deliver();
        moved |= !ready.is_empty();
        (ready, moved)
    }

    /// Lanes decoding with a client that takes events.
    fn decoders(&self, waiting: &[bool]) -> Vec<usize> {
        (0..self.jobs.len())
            .filter(|&l| !waiting[l])
            .filter(|&l| self.jobs[l].as_ref().is_some_and(|r| r.job.decoding()))
            .collect()
    }

    /// Whether a job still reads its prompt.
    fn reading(&self) -> bool {
        self.jobs.iter().flatten().any(|r| !r.job.decoding())
    }

    fn prefill<E: Engine>(
        &mut self,
        engine: &mut E,
        store: &mut PrefixStore,
        pressure: &PressureGauge,
        waiting: &[bool],
    ) -> bool {
        // The reading job with the fewest tokens left (FR-019); with several lanes it takes over
        // from a longer one between two layers.
        let Some(lane) = (0..self.jobs.len())
            .filter_map(|l| self.jobs[l].as_ref().map(|r| (l, r)))
            .filter(|(_, r)| !r.job.decoding())
            .min_by_key(|(_, r)| (r.job.left(), r.order))
            .map(|(l, _)| l)
        else {
            return false;
        };
        // Between decode steps only while prefill has its share.
        let decoding = !self.decoders(waiting).is_empty();
        if decoding && self.prefill_since >= self.decode_time * PREFILL_SHARE {
            return false;
        }
        store.select(engine, lane);
        let running = self.jobs[lane].as_mut().expect("busy");
        let cancel = running.job.request.cancel.clone();
        if engine.lanes() == 1 {
            // One lane: the whole chunk, as one lane always ran it.
            let tokens = running.job.next_chunk().expect("reading").to_vec();
            let ran = with_memory(engine, store, running.job.captured(), |engine| {
                engine.prefill_chunk(&tokens, &cancel)
            });
            let next = match ran {
                Ok(true) => self.chunk_done(engine, store, pressure, lane),
                Ok(false) => Ok(Some(FinishReason::Cancelled)),
                Err(failure) => Err(failure),
            };
            self.settle(engine, store, lane, next);
            return true;
        }
        if !running.in_flight {
            let tokens = running.job.next_chunk().expect("reading").to_vec();
            let begun = with_memory(engine, store, running.job.captured(), |engine| {
                engine.begin_chunk(&tokens)
            });
            if let Err(failure) = begun {
                self.end(engine, store, lane, Err(failure));
                return true;
            }
            self.jobs[lane].as_mut().expect("busy").in_flight = true;
        }
        let began = Instant::now();
        let progress = engine.advance_chunk(1, &cancel);
        self.prefill_since += began.elapsed();
        let next = match progress {
            Ok(ChunkProgress::More) => return true,
            Ok(ChunkProgress::Done) => {
                self.jobs[lane].as_mut().expect("busy").in_flight = false;
                self.chunk_done(engine, store, pressure, lane)
            }
            Ok(ChunkProgress::Cancelled) => {
                self.jobs[lane].as_mut().expect("busy").in_flight = false;
                Ok(Some(FinishReason::Cancelled))
            }
            Err(e) => {
                self.jobs[lane].as_mut().expect("busy").in_flight = false;
                Err(Failure::from(e))
            }
        };
        self.settle(engine, store, lane, next);
        true
    }

    fn chunk_done<E: Engine>(
        &mut self,
        engine: &mut E,
        store: &mut PrefixStore,
        pressure: &PressureGauge,
        lane: usize,
    ) -> Next {
        let running = self.jobs[lane].as_mut().expect("busy");
        running.job.chunk_done(engine, store, pressure)
    }

    fn decode<E: Engine>(
        &mut self,
        engine: &mut E,
        store: &mut PrefixStore,
        pressure: &PressureGauge,
        waiting: &[bool],
    ) -> bool {
        let mut decoders = self.decoders(waiting);
        if decoders.is_empty() {
            return false;
        }
        if self.reading() && self.prefill_since < self.decode_time * PREFILL_SHARE {
            return false;
        }
        let began = Instant::now();
        if let [lane] = decoders[..] {
            // One decoding job: its own step, a verify when it is the only job (FR-015).
            let solo = self.busy() == 1;
            store.select(engine, lane);
            let running = self.jobs[lane].as_mut().expect("busy");
            let next = running.job.decode_round(engine, store, pressure, solo);
            self.settle(engine, store, lane, next);
        } else {
            relieve_pressure(engine, store, pressure);
            // One forward for every decoding lane; a row that fails ends its job alone.
            let steps = loop {
                let rows: Vec<(usize, u32)> = decoders
                    .iter()
                    .map(|&l| (l, self.jobs[l].as_ref().expect("busy").job.last()))
                    .collect();
                if rows.is_empty() {
                    break Vec::new();
                }
                match engine.forward_lanes(&rows) {
                    Ok(steps) => break steps,
                    Err((row, EngineError::NeedsMemory { bytes })) => {
                        let lane = rows[row].0;
                        store.select(engine, lane);
                        if store.evict(engine, bytes, None) > 0 {
                            continue;
                        }
                        let running = self.jobs[lane].as_mut().expect("busy");
                        if let Some(c) = running.job.captured().first().map(|c| c.id) {
                            running.job.captured().remove(0);
                            engine.drop_checkpoint(c);
                            continue;
                        }
                        let e = EngineError::NeedsMemory { bytes };
                        self.end(engine, store, lane, Err(Failure::from(e)));
                        decoders.remove(row);
                    }
                    Err((row, e)) => {
                        self.end(engine, store, rows[row].0, Err(Failure::from(e)));
                        decoders.retain(|&l| self.jobs[l].is_some());
                    }
                }
            };
            if !steps.is_empty() {
                let samplings: Vec<_> = decoders
                    .iter()
                    .map(|&l| self.jobs[l].as_ref().expect("busy").job.request.sampling)
                    .collect();
                let masks: Vec<_> = decoders
                    .iter()
                    .map(|&l| self.jobs[l].as_mut().expect("busy").job.take_mask())
                    .collect();
                let picks = decoders
                    .iter()
                    .zip(steps)
                    .zip(samplings.iter().zip(&masks))
                    .map(|((&lane, step), (sampling, mask))| Pick {
                        lane,
                        step,
                        sampling,
                        mask: mask.as_ref(),
                    })
                    .collect();
                let sampled = engine.sample_lanes(picks);
                for (&lane, token) in decoders.iter().zip(sampled) {
                    let next = match token {
                        Ok(token) => {
                            let running = self.jobs[lane].as_mut().expect("busy");
                            running.job.commit(&[token], false, began)
                        }
                        Err(e) => Err(Failure::from(e)),
                    };
                    self.settle(engine, store, lane, next);
                }
            }
        }
        // A step right after a prefill layer also waits for that layer on the GPU; the fastest
        // recent step is the decode step itself (it may grow 2 % a step as contexts deepen).
        let took = began.elapsed();
        self.decode_time = if self.decode_time.is_zero() {
            took
        } else {
            took.min(self.decode_time.mul_f64(1.02))
        };
        self.prefill_since = Duration::ZERO;
        true
    }

    /// Ends the job on `lane` if `next` says so.
    fn settle<E: Engine>(
        &mut self,
        engine: &mut E,
        store: &mut PrefixStore,
        lane: usize,
        next: Next,
    ) {
        match next {
            Ok(None) => {}
            Ok(Some(reason)) => self.end(engine, store, lane, Ok(reason)),
            Err(failure) => self.end(engine, store, lane, Err(failure)),
        }
    }

    /// Finishes the job on `lane`. A backend failure resets every lane's state, so every other
    /// job fails with it.
    fn end<E: Engine>(
        &mut self,
        engine: &mut E,
        store: &mut PrefixStore,
        lane: usize,
        outcome: Result<FinishReason, Failure>,
    ) {
        let Some(running) = self.jobs[lane].take() else {
            return;
        };
        store.select(engine, lane);
        if running.in_flight {
            engine.abort_chunk();
        }
        let backend = matches!(outcome, Err(Failure::Backend(_)));
        let failure = outcome.as_ref().err().cloned();
        let (finished, sink) = running.job.finish(engine, store, outcome);
        store.set_busy(lane, false);
        self.draining.push(Draining {
            sink,
            tag: running.tag,
            finished,
            since: Instant::now(),
        });
        if backend {
            let failure = failure.expect("a failure");
            for other in 0..self.jobs.len() {
                if self.jobs[other].is_some() {
                    self.end(engine, store, other, Err(failure.clone()));
                }
            }
        }
    }

    /// Fails every job (a panic left the state unknown); the caller resets the engine.
    pub fn fail_all(&mut self, failure: &Failure) -> Vec<(T, Finished)> {
        let mut out = Vec::new();
        for slot in &mut self.jobs {
            if let Some(running) = slot.take() {
                out.push((
                    running.tag,
                    Finished {
                        reason: FinishReason::Error(failure.clone()),
                        prefilled: 0,
                        stats: Default::default(),
                    },
                ));
            }
        }
        out.extend(self.draining.drain(..).map(|d| (d.tag, d.finished)));
        out
    }

    /// The finished jobs whose events all reached their clients (or whose clients are gone or
    /// stalled past the timeout); their sinks close here, before `Finished` goes out.
    fn deliver(&mut self) -> Vec<(T, Finished)> {
        let mut ready = Vec::new();
        let mut i = 0;
        while i < self.draining.len() {
            let d = &mut self.draining[i];
            let done = match d.sink.flush() {
                Delivery::Done | Delivery::Gone => true,
                Delivery::Waiting(_) => d.since.elapsed() >= self.stall_timeout,
            };
            if done {
                let d = self.draining.swap_remove(i);
                drop(d.sink);
                ready.push((d.tag, d.finished));
            } else {
                i += 1;
            }
        }
        ready
    }
}

/// Runs `job` alone to its end on its lane (tests, the pool host, `generate`).
pub fn run_one<E: Engine>(
    engine: &mut E,
    store: &mut PrefixStore,
    job: Job<'_>,
    pressure: &PressureGauge,
) -> Finished {
    let lanes = store.lanes();
    let mut one: Lanes<'_, ()> = Lanes::new(lanes, Duration::MAX);
    one.admit(engine, store, pressure, job, ());
    loop {
        let (mut ready, _) = one.step(engine, store, pressure);
        if let Some(((), finished)) = ready.pop() {
            return finished;
        }
    }
}
