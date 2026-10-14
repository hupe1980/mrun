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

/// Fixed host reserve kept outside every automatic or fresh-headroom allowance (FR-015).
pub const HOST_RESERVE: u64 = 8 * 1024 * 1024 * 1024;
/// Maximum observation age; a sample from the future is equally unusable (research C-04).
pub const MAX_OBSERVATION_AGE: Duration = Duration::from_millis(250);

/// Owned-memory admission ceiling selected once from complete, fresh observations.
///
/// Observations are a deliberately conservative allowance, not a hard process-RSS limit.
#[derive(Debug)]
pub struct Admission {
    ceiling: u64,
}

impl Admission {
    pub fn select(
        observations: Observations,
        now: Duration,
        budget: Option<u64>,
        allow_increase: bool,
    ) -> Result<Self, Error> {
        let fresh = Fresh::new(observations, now)?;
        // floor(0.75 * physical) without overflow; fits u64 because it is below physical.
        let three_quarters = (u128::from(fresh.physical) * 3 / 4) as u64;
        let automatic = three_quarters.min(fresh.headroom());
        let ceiling = match budget {
            None => automatic,
            Some(requested) if requested <= automatic => requested,
            Some(requested) if allow_increase && requested <= fresh.physical => requested,
            Some(_) => return Err(Error::Budget),
        };
        if ceiling == 0 {
            return Err(Error::Budget);
        }
        Ok(Self { ceiling })
    }

    pub fn ceiling(&self) -> u64 {
        self.ceiling
    }

    /// Checks additional owned and device growth against the retained ceiling and fresh headroom.
    ///
    /// Already-owned bytes count only against the ceiling; they are not subtracted again from the
    /// freshly observed allowance, which already reflects their residency.
    pub fn check(
        &self,
        observations: Observations,
        now: Duration,
        owned: u64,
        delta: u64,
        device_delta: u64,
    ) -> Result<(), Error> {
        let fresh = Fresh::new(observations, now)?;
        if owned
            .checked_add(delta)
            .is_none_or(|total| total > self.ceiling)
            || delta > fresh.headroom()
        {
            return Err(Error::Budget);
        }
        if fresh
            .device_allocated
            .checked_add(device_delta)
            .is_none_or(|total| total > fresh.device_recommended)
        {
            return Err(Error::Device);
        }
        Ok(())
    }
}

struct Fresh {
    physical: u64,
    available: u64,
    device_recommended: u64,
    device_allocated: u64,
}

impl Fresh {
    fn new(observations: Observations, now: Duration) -> Result<Self, Error> {
        observations::require_complete(&observations)?;
        let stale = |at: Duration| at > now || now - at > MAX_OBSERVATION_AGE;
        let times = [
            observations.physical.map(|sample| sample.at),
            observations.available.map(|sample| sample.at),
            observations.device_recommended.map(|sample| sample.at),
            observations.device_allocated.map(|sample| sample.at),
            observations.footprint.map(|sample| sample.at),
            observations.rss.map(|sample| sample.at),
            observations.swap.map(|sample| sample.at),
            observations.pageouts.map(|sample| sample.at),
            observations.pressure.map(|sample| sample.at),
        ];
        if times.into_iter().flatten().any(stale) {
            return Err(Error::Stale);
        }
        let value =
            |sample: Option<Sample<u64>>| sample.map(|sample| sample.value).ok_or(Error::Missing);
        match observations.pressure.map(|sample| sample.value) {
            Some(Severity::Nominal) => {}
            Some(Severity::Elevated | Severity::Critical) => return Err(Error::Unavailable),
            None => return Err(Error::Missing),
        }
        Ok(Self {
            physical: value(observations.physical)?,
            available: value(observations.available)?,
            device_recommended: value(observations.device_recommended)?,
            device_allocated: value(observations.device_allocated)?,
        })
    }

    fn headroom(&self) -> u64 {
        self.available.saturating_sub(HOST_RESERVE)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Totals {
    pub reserved: u64,
    pub committed: u64,
    pub backings: usize,
    pub pins: usize,
}

/// Reserves the rendered prompt plus the full requested output before evaluation.
pub fn validate_context(
    prompt: u32,
    output: u32,
    context_limit: u32,
    output_limit: u32,
) -> Result<(), Error> {
    if prompt == 0 || output == 0 {
        return Err(Error::Empty);
    }
    let allowance = context_limit.saturating_sub(prompt).min(output_limit);
    if output > allowance {
        return Err(Error::Context { allowance });
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Action {
    StopAdmission,
    EvictUnpinned,
    Cancel,
    Synchronize,
    Unload,
}

/// Orders pressure handling; unknown severity is handled as critical.
pub fn pressure_actions(severity: Option<Severity>, safe_after_eviction: bool) -> Vec<Action> {
    match severity {
        Some(Severity::Nominal) => Vec::new(),
        Some(Severity::Elevated | Severity::Critical) | None if safe_after_eviction => {
            vec![Action::StopAdmission, Action::EvictUnpinned]
        }
        Some(Severity::Elevated | Severity::Critical) | None => vec![
            Action::StopAdmission,
            Action::EvictUnpinned,
            Action::Cancel,
            Action::Synchronize,
            Action::Unload,
        ],
    }
}
