//! T119 lifetime probes. Portable controls model Candle's pool and GPU submission lifetimes and
//! run on every host; real Metal probes need macOS plus `--features metal-gate` against Candle
//! with `patches/0001-metal-allocation-gate.patch`. Portable passes are not Metal evidence.

use mrun_candle_budget::budget::{Budget, Denied, GateLedger, Pool};

const MIB: u64 = 1024 * 1024;
const CEILING: u64 = 64 * MIB;

#[test]
fn pooled_reuse_keeps_capacity_charged_until_synchronized_trim() {
    let budget = Budget::new(CEILING);
    let mut pool = Pool::new(budget.clone());
    let first = pool.acquire(3000).unwrap();
    assert_eq!(
        first.lease.bytes(),
        4096,
        "power-of-two bucket like Candle buf_size"
    );
    pool.recycle(first);
    assert_eq!(
        budget.totals().reserved,
        4096,
        "dropped tensor keeps pooled capacity charged"
    );
    let larger_reuse = pool.acquire(1000).unwrap();
    assert_eq!(
        larger_reuse.lease.bytes(),
        4096,
        "smaller request reuses the larger buffer"
    );
    assert_eq!(budget.totals().reserved, 4096, "reuse adds no new charge");
    pool.recycle(larger_reuse);
    assert_eq!(pool.idle_bytes(), 4096);
    pool.trim_after_synchronize();
    assert_eq!(budget.totals().reserved, 0);
    assert_eq!(budget.totals().live_leases, 0);
}

#[test]
fn denied_growth_leaves_existing_backing_and_baseline_intact() {
    let budget = Budget::new(CEILING);
    let mut pool = Pool::new(budget.clone());
    let resident = pool.acquire(32 * MIB).unwrap();
    assert!(matches!(
        pool.acquire(32 * MIB + 1),
        Err(Denied::Insufficient { requested, available }) if requested == 64 * MIB && available == 32 * MIB
    ));
    assert_eq!(budget.totals().reserved, 32 * MIB);
    drop(resident);
    assert_eq!(budget.totals().reserved, 0);
}

#[test]
fn submitted_work_retains_backing_until_observed_completion() {
    let budget = Budget::new(CEILING);
    let lease = budget.reserve(MIB).unwrap();
    let snapshot = budget.reserve(MIB).unwrap();
    let in_flight = lease.submit();
    drop(lease);
    assert_eq!(
        budget.totals().reserved,
        2 * MIB,
        "queued GPU work keeps the charge"
    );
    in_flight.complete();
    assert_eq!(
        budget.totals().reserved,
        MIB,
        "independent snapshot backing remains charged"
    );
    drop(snapshot);
    assert_eq!(budget.totals().reserved, 0);
    assert!(budget.ready());
}

#[test]
fn uncertain_completion_quarantines_and_blocks_readiness_until_teardown() {
    let budget = Budget::new(CEILING);
    let lease = budget.reserve(MIB).unwrap();
    let view = lease.view();
    let in_flight = lease.submit();
    in_flight.fail_uncertain();
    drop((lease, view));
    let totals = budget.totals();
    assert_eq!(
        totals.reserved, MIB,
        "quarantined backing is never released early"
    );
    assert_eq!(totals.quarantined, MIB);
    assert!(!budget.ready());
    assert_eq!(budget.reserve(1).err(), Some(Denied::Quarantined));
    assert!(budget.release_after_teardown());
    assert_eq!(budget.totals().reserved, 0);
    assert!(budget.ready());
}

#[test]
fn teardown_is_refused_while_any_lease_is_live() {
    let budget = Budget::new(CEILING);
    let lease = budget.reserve(MIB).unwrap();
    lease.submit().fail_uncertain();
    assert!(!budget.release_after_teardown());
    drop(lease);
    assert!(budget.release_after_teardown());
    assert_eq!(
        budget.totals(),
        mrun_candle_budget::budget::Totals {
            capacity: CEILING,
            releases: 1,
            ..Default::default()
        }
    );
}

#[test]
fn gate_ledger_admits_exact_lengths_denies_growth_and_rejects_overflow() {
    let ledger = GateLedger::new(CEILING);
    ledger.admit(48 * MIB).unwrap();
    assert!(matches!(
        ledger.admit(17 * MIB),
        Err(Denied::Insufficient { available, .. }) if available == 16 * MIB
    ));
    assert_eq!(
        ledger.reserved(),
        48 * MIB,
        "denial leaves the ledger unchanged"
    );
    ledger.admit(16 * MIB).unwrap();
    let full = GateLedger::new(u64::MAX);
    full.admit(u64::MAX).unwrap();
    assert_eq!(full.admit(1), Err(Denied::Overflow));
    ledger.release(64 * MIB);
    assert_eq!(ledger.reserved(), 0);
}

/// Real Metal probes (macOS + patched Candle). Not executed on the Linux host of 2026-10-14.
#[cfg(all(target_os = "macos", feature = "metal-gate"))]
mod metal {
    use super::*;
    use candle_core::{DType, Device, Tensor};
    use mrun_candle_budget::metal_gate;

    #[test]
    fn gated_upload_copy_readback_denied_growth_and_synchronized_release() {
        metal_gate::install().unwrap();
        let baseline = metal_gate::LEDGER.reserved();
        let device = Device::new_metal(0).unwrap();
        let host = vec![1f32; 1 << 20];
        let uploaded = Tensor::from_vec(host.clone(), (1024, 1024), &device).unwrap();
        let copied = (&uploaded + 1.0).unwrap();
        let readback: Vec<f32> = copied.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(readback[0], 2.0);
        assert!(metal_gate::LEDGER.reserved() > baseline);
        assert!(metal_gate::LEDGER.reserved() <= CEILING);
        let denied = Tensor::zeros((CEILING as usize / 4 + 1,), DType::F32, &device);
        assert!(
            denied.is_err(),
            "growth past the 64 MiB ceiling must be denied before allocation"
        );
        drop((uploaded, copied));
        device.synchronize().unwrap();
        // `drop_unused_buffers` runs inside the completed wait; pool capacity returns.
        assert_eq!(metal_gate::LEDGER.reserved(), baseline);
    }
}
