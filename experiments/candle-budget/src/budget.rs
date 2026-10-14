//! Experimental reservation ledger (T117).
//!
//! Every byte is reserved before the allocation it covers is attempted. Charges belong to one
//! unique backing: aliased views share it, submitted GPU work retains it, and uncertain
//! completion quarantines it until verified teardown. This models enforcement; it does not
//! prove that Candle routes every allocation through it (see `gpu-inventory.md`).

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Denied {
    Insufficient { requested: u64, available: u64 },
    Overflow,
    Quarantined,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Totals {
    pub capacity: u64,
    pub reserved: u64,
    pub live_leases: usize,
    pub quarantined: u64,
    pub releases: u64,
}

#[derive(Debug, Eq, PartialEq)]
pub enum Guarded<E> {
    Denied(Denied),
    Failed(E),
}

/// Checked byte size of a dense tensor; Candle multiplies these without a check.
pub fn tensor_bytes(dims: &[usize], dtype_size: usize) -> Result<u64, Denied> {
    dims.iter()
        .try_fold(dtype_size, |bytes, &dim| bytes.checked_mul(dim))
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or(Denied::Overflow)
}

/// Checked mirror of Candle's pooled `buf_size` (`next_power_of_two`).
pub fn pool_bucket(bytes: u64) -> Result<u64, Denied> {
    bytes.checked_next_power_of_two().ok_or(Denied::Overflow)
}

#[derive(Clone)]
pub struct Budget {
    inner: Arc<Inner>,
}

struct Inner {
    capacity: u64,
    state: Mutex<State>,
}

#[derive(Default)]
struct State {
    reserved: u64,
    live_leases: usize,
    quarantined: u64,
    releases: u64,
    next_id: u64,
}

impl Budget {
    pub fn new(capacity: u64) -> Self {
        Self {
            inner: Arc::new(Inner {
                capacity,
                state: Mutex::new(State::default()),
            }),
        }
    }

    /// Reserves `bytes` for one unique backing. Denial never allocates.
    pub fn reserve(&self, bytes: u64) -> Result<Lease, Denied> {
        let id = {
            let mut state = self.inner.lock();
            if state.quarantined > 0 {
                return Err(Denied::Quarantined);
            }
            let total = state.reserved.checked_add(bytes).ok_or(Denied::Overflow)?;
            if total > self.inner.capacity {
                return Err(Denied::Insufficient {
                    requested: bytes,
                    available: self.inner.capacity - state.reserved,
                });
            }
            state.reserved = total;
            state.live_leases += 1;
            state.next_id += 1;
            state.next_id
        };
        Ok(Lease {
            charge: Arc::new(Charge {
                budget: Arc::clone(&self.inner),
                id,
                bytes,
                quarantined: AtomicBool::new(false),
            }),
        })
    }

    /// Reserves before entering `construct`; a constructor failure releases the charge once.
    pub fn guarded<T, E>(
        &self,
        bytes: u64,
        construct: impl FnOnce() -> Result<T, E>,
    ) -> Result<(T, Lease), Guarded<E>> {
        let lease = self.reserve(bytes).map_err(Guarded::Denied)?;
        match construct() {
            Ok(value) => Ok((value, lease)),
            Err(error) => Err(Guarded::Failed(error)),
        }
    }

    pub fn totals(&self) -> Totals {
        let state = self.inner.lock();
        Totals {
            capacity: self.inner.capacity,
            reserved: state.reserved,
            live_leases: state.live_leases,
            quarantined: state.quarantined,
            releases: state.releases,
        }
    }

    /// New work may be admitted only while nothing is quarantined.
    pub fn ready(&self) -> bool {
        self.inner.lock().quarantined == 0
    }

    /// Releases quarantined charges after the owning device/context was torn down and every
    /// lease referencing it is gone. Returns false, changing nothing, while any lease is live.
    pub fn release_after_teardown(&self) -> bool {
        let mut state = self.inner.lock();
        if state.live_leases != 0 {
            return false;
        }
        state.reserved -= state.quarantined;
        if state.quarantined > 0 {
            state.releases += 1;
        }
        state.quarantined = 0;
        true
    }
}

impl Inner {
    fn lock(&self) -> MutexGuard<'_, State> {
        // Ledger arithmetic completes before any panic point, so poisoned state is consistent.
        self.state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}

struct Charge {
    budget: Arc<Inner>,
    id: u64,
    bytes: u64,
    quarantined: AtomicBool,
}

impl Drop for Charge {
    fn drop(&mut self) {
        let mut state = self.budget.lock();
        state.live_leases -= 1;
        // A quarantined charge stays reserved until `release_after_teardown`.
        if !self.quarantined.load(Ordering::Acquire) {
            state.reserved -= self.bytes;
            state.releases += 1;
        }
    }
}

/// The owning handle of one unique backing charge.
pub struct Lease {
    charge: Arc<Charge>,
}

/// An aliased CPU or GPU view of a backing; it keeps the single charge alive.
#[derive(Clone)]
pub struct View {
    _charge: Arc<Charge>,
}

/// A retain held by submitted GPU work until its completion is observed.
#[must_use = "submitted work must be completed or quarantined"]
pub struct Submission {
    charge: Arc<Charge>,
}

impl Lease {
    pub fn id(&self) -> u64 {
        self.charge.id
    }

    pub fn bytes(&self) -> u64 {
        self.charge.bytes
    }

    pub fn view(&self) -> View {
        View {
            _charge: Arc::clone(&self.charge),
        }
    }

    pub fn submit(&self) -> Submission {
        Submission {
            charge: Arc::clone(&self.charge),
        }
    }
}

impl Submission {
    /// Observed successful completion: drop the retain.
    pub fn complete(self) {}

    /// Completion error or unknown outcome: the backing may still be referenced by the device.
    pub fn fail_uncertain(self) {
        if !self.charge.quarantined.swap(true, Ordering::AcqRel) {
            self.charge.budget.lock().quarantined += self.charge.bytes;
        }
    }
}

/// Model of Candle's pooled allocator: rounded buckets whose capacity stays charged after the
/// tensor is dropped, until an explicit trim after synchronization (G-01..G-04).
pub struct Pool {
    budget: Budget,
    free: BTreeMap<u64, Vec<Lease>>,
}

/// A pooled buffer handed to a tensor; return it with [`Pool::recycle`].
pub struct Pooled {
    pub lease: Lease,
    pub requested: u64,
}

impl Pool {
    pub fn new(budget: Budget) -> Self {
        Self {
            budget,
            free: BTreeMap::new(),
        }
    }

    /// Reuses any free buffer at least as large (as Candle does), else reserves a new bucket.
    pub fn acquire(&mut self, bytes: u64) -> Result<Pooled, Denied> {
        let reusable = self
            .free
            .range_mut(bytes..)
            .find_map(|(_, leases)| leases.pop());
        let lease = match reusable {
            Some(lease) => lease,
            None => self.budget.reserve(pool_bucket(bytes)?)?,
        };
        self.free.retain(|_, leases| !leases.is_empty());
        Ok(Pooled {
            lease,
            requested: bytes,
        })
    }

    /// Returns a buffer to the pool; its capacity stays charged.
    pub fn recycle(&mut self, buffer: Pooled) {
        self.free
            .entry(buffer.lease.bytes())
            .or_default()
            .push(buffer.lease);
    }

    /// Bytes held by idle pooled buffers.
    pub fn idle_bytes(&self) -> u64 {
        self.free
            .iter()
            .map(|(bucket, leases)| bucket * leases.len() as u64)
            .sum()
    }

    /// Releases idle capacity; callers must first observe completion of all submitted work.
    pub fn trim_after_synchronize(&mut self) {
        self.free.clear();
    }
}

/// Size-only ledger for the patched Candle Metal gate (patch 0001): raw constructors are
/// admitted by exact length and released only when the pool frees after a completed wait.
pub struct GateLedger {
    capacity: u64,
    reserved: std::sync::atomic::AtomicU64,
}

impl GateLedger {
    pub const fn new(capacity: u64) -> Self {
        Self {
            capacity,
            reserved: std::sync::atomic::AtomicU64::new(0),
        }
    }

    pub fn admit(&self, bytes: u64) -> Result<(), Denied> {
        self.reserved
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |reserved| {
                reserved
                    .checked_add(bytes)
                    .filter(|total| *total <= self.capacity)
            })
            .map(|_| ())
            .map_err(|reserved| match reserved.checked_add(bytes) {
                None => Denied::Overflow,
                Some(_) => Denied::Insufficient {
                    requested: bytes,
                    available: self.capacity - reserved,
                },
            })
    }

    /// Saturates instead of underflowing: a report for bytes never admitted (an allocation
    /// that predates the gate) is a coverage error surfaced by `reserved()` tests, not a wrap.
    pub fn release(&self, bytes: u64) {
        let _ = self
            .reserved
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |reserved| {
                Some(reserved.saturating_sub(bytes))
            });
    }

    pub fn reserved(&self) -> u64 {
        self.reserved.load(Ordering::Acquire)
    }
}
