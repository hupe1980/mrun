//! Admission for the single engine worker (002 FR-010): a bounded FIFO of waiting requests.
//!
//! The worker runs one request at a time; up to `capacity` more wait. A full queue refuses at
//! once with a `retry-after` hint. Waiting deadlines and keepalives are enforced by the caller
//! (it cancels the request's token), so the worker drops a cancelled request when it reaches it
//! without touching the engine. The gauge keeps a moving average of request durations, so a
//! non-streaming request can be refused at admission when its estimated wait exceeds the
//! deadline (D052).

use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::time::Duration;

/// Default number of waiting requests (`--queue`).
pub const DEFAULT_QUEUE: usize = 16;
/// Default time a request may wait before it starts (`--queue-wait-s`).
pub const DEFAULT_QUEUE_WAIT_S: u64 = 240;
/// Assumed duration of one request, for `retry-after`.
const RETRY_PER_REQUEST_S: u64 = 10;

#[derive(Debug)]
pub struct QueueGauge {
    capacity: AtomicUsize,
    waiting: AtomicUsize,
    running: AtomicBool,
    /// Moving average of request durations on the worker, in milliseconds (0: none yet).
    service_ms: AtomicU64,
}

impl Default for QueueGauge {
    fn default() -> Self {
        Self {
            capacity: AtomicUsize::new(usize::MAX),
            waiting: AtomicUsize::new(0),
            running: AtomicBool::new(false),
            service_ms: AtomicU64::new(0),
        }
    }
}

/// The queue was full; retry after this many seconds (1–60).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct QueueFull {
    pub retry_after: u64,
}

impl QueueGauge {
    /// At least one request may always wait: with none, every request would be refused,
    /// since a request waits from its submission until the worker takes it.
    pub fn set_capacity(&self, capacity: usize) {
        self.capacity.store(capacity.max(1), Ordering::Relaxed);
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

    /// Returns a slot: the worker took its request, or it was cancelled while waiting.
    pub(crate) fn release(&self) {
        self.waiting.fetch_sub(1, Ordering::AcqRel);
    }

    /// Notes how long the worker spent on one request.
    pub(crate) fn record_service(&self, took: Duration) {
        let ms = took.as_millis().min(u128::from(u64::MAX)) as u64;
        let old = self.service_ms.load(Ordering::Relaxed);
        let new = if old == 0 {
            ms.max(1)
        } else {
            (old * 3 + ms) / 4
        };
        self.service_ms.store(new.max(1), Ordering::Relaxed);
    }

    /// Estimated wait of a request admitted now: the requests ahead times the average duration
    /// (zero before any request finished).
    pub fn estimated_wait(&self) -> Duration {
        let ahead = self.waiting() as u64 + u64::from(self.running());
        Duration::from_millis(ahead.saturating_mul(self.service_ms.load(Ordering::Relaxed)))
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
        gauge.release();
        gauge.admit().unwrap();
        gauge.set_capacity(100);
        for _ in 0..10 {
            gauge.admit().unwrap();
        }
        gauge.set_capacity(0);
        assert_eq!(gauge.admit().unwrap_err().retry_after, 60);
    }

    #[test]
    fn a_zero_capacity_still_admits_one_and_waits_are_estimated() {
        let gauge = QueueGauge::default();
        gauge.set_capacity(0);
        gauge.admit().unwrap();
        assert_eq!(gauge.estimated_wait(), Duration::ZERO);
        gauge.record_service(Duration::from_secs(8));
        gauge.set_running(true);
        assert_eq!(gauge.estimated_wait(), Duration::from_secs(16));
        gauge.record_service(Duration::from_secs(4));
        assert_eq!(gauge.estimated_wait(), Duration::from_secs(14));
    }
}
