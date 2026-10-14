use super::{Error, Totals};
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

const MAX_BACKINGS: usize = 1024;
const MIN_SCRATCH: u64 = 2 * 1024 * 1024 * 1024;

#[derive(Default)]
struct Allocation {
    ceiling: u64,
    committed: u64,
    pins: usize,
}

struct State {
    ceiling: u64,
    next_id: u64,
    totals: Totals,
    allocations: BTreeMap<u64, Allocation>,
}

pub struct Ledger(Arc<Mutex<State>>);

struct Backing {
    id: u64,
    state: Arc<Mutex<State>>,
}

pub struct Lease(Arc<Backing>);
pub struct Pin(Arc<Backing>);

impl Ledger {
    pub fn new(ceiling: u64) -> Self {
        Self(Arc::new(Mutex::new(State {
            ceiling,
            next_id: 0,
            totals: Totals {
                reserved: 0,
                committed: 0,
                backings: 0,
                pins: 0,
            },
            allocations: BTreeMap::new(),
        })))
    }

    pub fn reserve(&self, bytes: u64) -> Result<Lease, Error> {
        let mut state = self.0.lock().map_err(|_| Error::Unavailable)?;
        let total = state
            .totals
            .reserved
            .checked_add(bytes)
            .ok_or(Error::Budget)?;
        if bytes == 0 || total > state.ceiling || state.allocations.len() >= MAX_BACKINGS {
            return Err(Error::Budget);
        }
        let id = state.next_id.checked_add(1).ok_or(Error::Budget)?;
        state.allocations.insert(
            id,
            Allocation {
                ceiling: bytes,
                ..Allocation::default()
            },
        );
        state.next_id = id;
        state.totals.reserved = total;
        state.totals.backings += 1;
        Ok(Lease(Arc::new(Backing {
            id,
            state: self.0.clone(),
        })))
    }

    pub fn totals(&self) -> Totals {
        self.0
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .totals
    }
}

impl Lease {
    pub fn commit(&self, bytes: u64) -> Result<(), Error> {
        let mut state = self.0.state.lock().map_err(|_| Error::Unavailable)?;
        let allocation = state
            .allocations
            .get(&self.0.id)
            .ok_or(Error::Unavailable)?;
        if bytes > allocation.ceiling || (bytes < allocation.committed && allocation.pins != 0) {
            return Err(Error::Growth);
        }
        let total = state
            .totals
            .committed
            .checked_sub(allocation.committed)
            .and_then(|total| total.checked_add(bytes))
            .ok_or(Error::Growth)?;
        state
            .allocations
            .get_mut(&self.0.id)
            .ok_or(Error::Unavailable)?
            .committed = bytes;
        state.totals.committed = total;
        Ok(())
    }

    pub fn view(&self) -> Self {
        Self(self.0.clone())
    }

    pub fn pin(&self) -> Result<Pin, Error> {
        let mut state = self.0.state.lock().map_err(|_| Error::Unavailable)?;
        let total = state.totals.pins.checked_add(1).ok_or(Error::Growth)?;
        let allocation = state
            .allocations
            .get_mut(&self.0.id)
            .ok_or(Error::Unavailable)?;
        allocation.pins = allocation.pins.checked_add(1).ok_or(Error::Growth)?;
        state.totals.pins = total;
        Ok(Pin(self.0.clone()))
    }
}

impl Drop for Pin {
    fn drop(&mut self) {
        let mut state = self
            .0
            .state
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        if let Some(allocation) = state.allocations.get_mut(&self.0.id) {
            allocation.pins -= 1;
            state.totals.pins -= 1;
        }
    }
}

impl Drop for Backing {
    fn drop(&mut self) {
        let mut state = self.state.lock().unwrap_or_else(|error| error.into_inner());
        if let Some(allocation) = state.allocations.remove(&self.id) {
            state.totals.reserved -= allocation.ceiling;
            state.totals.committed -= allocation.committed;
            state.totals.backings -= 1;
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Demand {
    pub weights: u64,
    pub kv: u64,
    pub scratch: u64,
    pub host: u64,
    pub serving: u64,
}

impl Demand {
    pub fn conservative(
        weights: u64,
        kv: u64,
        host: u64,
        serving: u64,
        scratch_peak: u64,
    ) -> Result<Self, Error> {
        let shared = weights.checked_add(kv).ok_or(Error::Budget)?;
        let quarter = shared.checked_add(3).ok_or(Error::Budget)? / 4;
        let scratch = MIN_SCRATCH.max(quarter).max(scratch_peak);
        let demand = Self {
            weights,
            kv,
            scratch,
            host,
            serving,
        };
        demand.total()?;
        Ok(demand)
    }

    pub fn total(&self) -> Result<u64, Error> {
        [self.weights, self.kv, self.scratch, self.host, self.serving]
            .into_iter()
            .try_fold(0_u64, |total, bytes| {
                total.checked_add(bytes).ok_or(Error::Budget)
            })
    }

    pub fn reserve_all(&self, ledger: &Ledger) -> Result<Vec<Lease>, Error> {
        [self.weights, self.kv, self.scratch, self.host, self.serving]
            .into_iter()
            .filter(|bytes| *bytes != 0)
            .map(|bytes| ledger.reserve(bytes))
            .collect()
    }
}
