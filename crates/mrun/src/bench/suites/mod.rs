//! The three suites (005 FR-001) and what they share: the HTTP client, cache resets and the
//! random source for nonces and cancellation points.

pub mod agent;
pub mod cancel;
pub mod synthetic;

use crate::api::model::ApiKind;
use crate::bench::canary::{self, Canary};
use crate::bench::result::{Cell, CellStatus, Failure, Reset};
use crate::bench::systems::System;
use std::collections::HashMap;

/// State shared by the suites of one run.
#[derive(Debug)]
pub struct Ctx {
    pub client: reqwest::Client,
    pub interactive: bool,
    pub resets: Vec<Reset>,
    pub rng: Rng,
    /// The run's truncation canary line.
    pub canary: String,
    calibrations: HashMap<(String, &'static str), Canary>,
}

impl Ctx {
    pub fn new(interactive: bool) -> Self {
        let mut rng = Rng::from_time();
        Self {
            client: crate::bench::driver::client(),
            interactive,
            resets: Vec::new(),
            canary: canary::line(&mut rng),
            rng,
            calibrations: HashMap::new(),
        }
    }

    /// The canary measured on `system` over `api`, calibrated once per run.
    pub async fn canary(&mut self, system: &System, api: ApiKind) -> Canary {
        let key = (system.name.clone(), api.as_str());
        if let Some(found) = self.calibrations.get(&key) {
            return found.clone();
        }
        let found = canary::calibrate(&self.client, &mut self.rng, system, api, &self.canary).await;
        self.calibrations.insert(key, found.clone());
        found
    }

    /// Resets `system`'s caches and records it.
    pub async fn reset(&mut self, system: &System, rep: u32) {
        let reset = system.reset(&self.client, rep, self.interactive).await;
        self.resets.push(reset);
    }
}

/// A cell that produced no samples, with the reason.
pub fn not_run(system: &str, cell: &str, api: &str, unit: &str, reason: &str) -> Cell {
    Cell {
        system: system.into(),
        cell: cell.into(),
        api: api.into(),
        status: CellStatus::NotRun,
        unit: unit.into(),
        thermal_start: "unknown".into(),
        thermal_end: "unknown".into(),
        failures: vec![Failure {
            rep: 0,
            error: reason.into(),
        }],
        ..Cell::default()
    }
}

/// xorshift64*: nonces, canaries and cancellation points only need to differ between runs.
#[derive(Debug)]
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed | 1)
    }

    fn from_time() -> Self {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(1, |d| d.as_nanos() as u64);
        Self::new(nanos ^ u64::from(std::process::id()))
    }

    /// Uniform in `0..n`.
    pub fn below(&mut self, n: u64) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d) % n.max(1)
    }
}
