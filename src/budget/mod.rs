use std::time::Duration;

pub mod ledger;
pub mod observations;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Error {
    Unavailable,
    Missing,
    Stale,
    Budget,
    Device,
    Context { allowance: u32 },
    Empty,
    Growth,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Severity {
    Nominal,
    Elevated,
    Critical,
}

#[derive(Clone, Copy, Debug)]
pub struct Sample<T> {
    pub at: Duration,
    pub value: T,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Observations {
    pub physical: Option<Sample<u64>>,
    pub available: Option<Sample<u64>>,
    pub device_recommended: Option<Sample<u64>>,
    pub device_allocated: Option<Sample<u64>>,
    pub footprint: Option<Sample<u64>>,
    pub rss: Option<Sample<u64>>,
    pub swap: Option<Sample<u64>>,
    pub pageouts: Option<Sample<u64>>,
    pub pressure: Option<Sample<Severity>>,
}

#[derive(Debug)]
pub struct Admission;

impl Admission {
    pub fn select(
        _observations: Observations,
        _now: Duration,
        _budget: Option<u64>,
        _allow_increase: bool,
    ) -> Result<Self, Error> {
        Err(Error::Unavailable)
    }

    pub fn ceiling(&self) -> u64 {
        0
    }

    pub fn check(
        &self,
        _observations: Observations,
        _now: Duration,
        _owned: u64,
        _delta: u64,
        _device_delta: u64,
    ) -> Result<(), Error> {
        Err(Error::Unavailable)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Totals {
    pub reserved: u64,
    pub committed: u64,
    pub backings: usize,
    pub pins: usize,
}

pub fn validate_context(
    _prompt: u32,
    _output: u32,
    _context_limit: u32,
    _output_limit: u32,
) -> Result<(), Error> {
    Err(Error::Unavailable)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Action {
    StopAdmission,
    EvictUnpinned,
    Cancel,
    Synchronize,
    Unload,
}

pub fn pressure_actions(_severity: Option<Severity>, _safe_after_eviction: bool) -> Vec<Action> {
    vec![Action::StopAdmission]
}
