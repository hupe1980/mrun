//! The three suites (005 FR-001) and what they share: the HTTP client, cache resets and the
//! random source for nonces and cancellation points.

pub mod agent;
pub mod cancel;
pub mod synthetic;

use crate::bench::result::{Cell, CellStatus, Failure, Reset};
use crate::bench::systems::System;

/// State shared by the suites of one run.
#[derive(Debug)]
pub struct Ctx {
    pub client: reqwest::Client,
    pub interactive: bool,
    pub resets: Vec<Reset>,
    pub rng: Rng,
}

impl Ctx {
    pub fn new(interactive: bool) -> Self {
        Self {
            client: crate::bench::driver::client(),
            interactive,
            resets: Vec::new(),
            rng: Rng::from_time(),
        }
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
        failures: vec![Failure {
            rep: 0,
            error: reason.into(),
        }],
        ..Cell::default()
    }
}

/// xorshift64*: nonces and cancellation points only need to differ between runs.
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
