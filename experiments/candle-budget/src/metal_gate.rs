//! Wires [`GateLedger`] into patched Candle (patch 0001). Requires the `metal-gate` feature and
//! a Candle source with `patches/0001-metal-allocation-gate.patch` applied.

use crate::budget::GateLedger;

/// The experiment's 64 MiB GPU backing ceiling (README).
pub const EXPERIMENT_GPU_CEILING: u64 = 64 * 1024 * 1024;

pub static LEDGER: GateLedger = GateLedger::new(EXPERIMENT_GPU_CEILING);

fn gate(length: usize) -> Result<(), String> {
    LEDGER
        .admit(length as u64)
        .map_err(|denied| format!("mrun reservation denied: {denied:?}"))
}

fn release(length: usize) {
    LEDGER.release(length as u64);
}

/// Installs the process-wide gate before any Metal device is created.
pub fn install() -> candle_core::Result<()> {
    candle_metal_kernels::metal::install_allocation_gate(gate, release)
        .map_err(candle_core::Error::wrap)
}
