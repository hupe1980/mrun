//! Admission for the single engine worker (002 FR-010): a bounded FIFO of waiting requests.
//!
//! The worker runs one request at a time; up to `capacity` more wait. A full queue refuses at
//! once with a `retry-after` hint. Waiting deadlines and keepalives are enforced by the caller
//! (it cancels the request's token), so the worker drops a cancelled request when it reaches it
//! without touching the engine.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

/// Default number of waiting requests (`--queue`).
pub const DEFAULT_QUEUE: usize = 16;
/// Default time a request may wait before it starts (`--queue-wait-s`).
pub const DEFAULT_QUEUE_WAIT_S: u64 = 600;
/// Assumed duration of one request, for `retry-after`.
const RETRY_PER_REQUEST_S: u64 = 10;

#[derive(Debug)]
pub struct QueueGauge {
    capacity: AtomicUsize,
    waiting: AtomicUsize,
    running: AtomicBool,
}

impl Default for QueueGauge {
    fn default() -> Self {
        Self {
            capacity: AtomicUsize::new(usize::MAX),
            waiting: AtomicUsize::new(0),
            running: AtomicBool::new(false),
        }
    }
}

/// The queue was full; retry after this many seconds (1–60).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct QueueFull {
    pub retry_after: u64,
}

impl QueueGauge {
    pub fn set_capacity(&self, capacity: usize) {
        self.capacity.store(capacity, Ordering::Relaxed);
    }

    /// Requests waiting to start (not counting the running one).
    pub fn waiting(&self) -> usize {
        self.waiting.load(Ordering::Relaxed)
    }

    pub fn running(&self) -> bool {
        self.running.load(Ordering::Relaxed)
    }

    /// Takes a waiting slot, or refuses when all are taken.
    pub fn admit(&self) -> Result<(), QueueFull> {
        let capacity = self.capacity.load(Ordering::Relaxed);
        let waiting = self.waiting.fetch_add(1, Ordering::AcqRel);
        if waiting >= capacity {
            self.waiting.fetch_sub(1, Ordering::AcqRel);
            let ahead = waiting as u64 + u64::from(self.running());
            return Err(QueueFull {
                retry_after: (ahead * RETRY_PER_REQUEST_S).clamp(1, 60),
            });
        }
        Ok(())
    }

    /// The worker took a request off the queue.
    pub(crate) fn dequeued(&self) {
        self.waiting.fetch_sub(1, Ordering::AcqRel);
    }

    /// Returns a slot whose request never reached the worker.
    pub(crate) fn release(&self) {
        self.waiting.fetch_sub(1, Ordering::AcqRel);
    }

    pub(crate) fn set_running(&self, running: bool) {
        self.running.store(running, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_queue_refuses_with_a_bounded_retry_after() {
        let gauge = QueueGauge::default();
        gauge.set_capacity(2);
        gauge.set_running(true);
        gauge.admit().unwrap();
        gauge.admit().unwrap();
        assert_eq!(gauge.admit(), Err(QueueFull { retry_after: 30 }));
        assert_eq!(gauge.waiting(), 2);
        gauge.dequeued();
        gauge.admit().unwrap();
        gauge.set_capacity(100);
        for _ in 0..10 {
            gauge.admit().unwrap();
        }
        gauge.set_capacity(0);
        assert_eq!(gauge.admit().unwrap_err().retry_after, 60);
    }
}
