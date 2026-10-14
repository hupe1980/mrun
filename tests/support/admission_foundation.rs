use crate::budget::ledger::{Lease, Ledger};
use crate::budget::{self, Action, Admission, Error, Observations, Sample, Severity};
use std::time::Duration;

const GIB: u64 = 1024 * 1024 * 1024;

fn observations() -> Observations {
    let sample = |value| {
        Some(Sample {
            at: Duration::ZERO,
            value,
        })
    };
    Observations {
        physical: sample(48 * GIB),
        available: sample(40 * GIB),
        device_recommended: sample(36 * GIB),
        device_allocated: sample(0),
        footprint: sample(0),
        rss: sample(0),
        swap: sample(0),
        pageouts: sample(0),
        pressure: Some(Sample {
            at: Duration::ZERO,
            value: Severity::Nominal,
        }),
    }
}

#[test]
fn admission_accepts_a_small_lawful_fit_and_freezes_selected_ceiling() {
    let admission = Admission::select(observations(), Duration::ZERO, None, false).unwrap();
    assert_eq!(admission.ceiling(), 32 * GIB);
    admission
        .check(observations(), Duration::ZERO, 5 * GIB, GIB, GIB)
        .unwrap();
}

#[test]
fn additional_availability_is_not_reduced_by_existing_owned_bytes_twice() {
    let admission = Admission::select(observations(), Duration::ZERO, None, false).unwrap();
    let mut fresh = observations();
    fresh.available.as_mut().unwrap().value = 10 * GIB;
    fresh.device_allocated.as_mut().unwrap().value = 20 * GIB;
    admission
        .check(fresh, Duration::ZERO, 20 * GIB, GIB, GIB)
        .unwrap();
    assert_eq!(admission.ceiling(), 32 * GIB);
    assert_eq!(
        admission.check(fresh, Duration::ZERO, 20 * GIB, 3 * GIB, GIB),
        Err(Error::Budget)
    );
}

#[test]
fn every_missing_observation_blocks_admission() {
    for missing in 0..9 {
        let mut sample = observations();
        match missing {
            0 => sample.physical = None,
            1 => sample.available = None,
            2 => sample.device_recommended = None,
            3 => sample.device_allocated = None,
            4 => sample.footprint = None,
            5 => sample.rss = None,
            6 => sample.swap = None,
            7 => sample.pageouts = None,
            _ => sample.pressure = None,
        }
        assert_eq!(
            Admission::select(sample, Duration::ZERO, None, false).unwrap_err(),
            Error::Missing
        );
    }
}

#[test]
fn device_limit_and_pressure_are_independent_of_owned_budget() {
    let admission = Admission::select(observations(), Duration::ZERO, None, false).unwrap();
    let mut sample = observations();
    sample.device_allocated.as_mut().unwrap().value = 36 * GIB;
    assert_eq!(
        admission.check(sample, Duration::ZERO, 0, GIB, GIB),
        Err(Error::Device)
    );
    sample.device_allocated.as_mut().unwrap().value = 0;
    sample.pressure.as_mut().unwrap().value = Severity::Critical;
    assert_eq!(
        admission.check(sample, Duration::ZERO, 0, GIB, GIB),
        Err(Error::Unavailable)
    );
}

#[test]
fn lower_budgets_apply_and_increases_need_explicit_opt_in() {
    let lower = Admission::select(observations(), Duration::ZERO, Some(8 * GIB), false).unwrap();
    assert_eq!(lower.ceiling(), 8 * GIB);
    assert_eq!(
        lower.check(observations(), Duration::ZERO, 8 * GIB, 1, 0),
        Err(Error::Budget)
    );
    assert_eq!(
        Admission::select(observations(), Duration::ZERO, Some(33 * GIB), false).unwrap_err(),
        Error::Budget
    );
    assert_eq!(
        Admission::select(observations(), Duration::ZERO, Some(33 * GIB), true)
            .unwrap()
            .ceiling(),
        33 * GIB
    );
}

#[test]
fn stale_and_future_observations_do_not_authorize_allocations() {
    assert_eq!(
        Admission::select(observations(), Duration::from_secs(2), None, false).unwrap_err(),
        Error::Stale
    );
    let mut sample = observations();
    sample.available.as_mut().unwrap().at = Duration::from_secs(1);
    assert_eq!(
        Admission::select(sample, Duration::ZERO, None, false).unwrap_err(),
        Error::Stale
    );
}

#[test]
fn reservation_precedes_commit_and_growth_cannot_exceed_lease() {
    let ledger = Ledger::new(1024);
    assert!(ledger.reserve(1025).is_err());
    assert_eq!(ledger.totals().reserved, 0);
    let lease = ledger.reserve(1024).unwrap();
    assert_eq!(ledger.totals().reserved, 1024);
    assert_eq!(ledger.totals().committed, 0);
    lease.commit(512).unwrap();
    assert_eq!(ledger.totals().committed, 512);
    assert_eq!(lease.commit(1025), Err(Error::Growth));
    assert_eq!(ledger.totals().committed, 512);
    drop(lease);
    assert_eq!(ledger.totals().reserved, 0);
    assert_eq!(ledger.totals().committed, 0);
}

#[test]
fn unified_views_and_gpu_pins_charge_one_backing_until_last_safe_release() {
    let ledger = Ledger::new(1024);
    let host = ledger.reserve(1024).unwrap();
    host.commit(1024).unwrap();
    let device = host.view();
    let pin = device.pin().unwrap();
    assert_eq!(ledger.totals().backings, 1);
    assert_eq!(ledger.totals().reserved, 1024);
    assert_eq!(ledger.totals().pins, 1);
    drop(host);
    drop(device);
    assert_eq!(ledger.totals().reserved, 1024);
    drop(pin);
    assert_eq!(ledger.totals().backings, 0);
    assert_eq!(ledger.totals().pins, 0);
    assert_eq!(ledger.totals().reserved, 0);
}

#[test]
fn ledger_transfer_repeated_commit_and_pinned_shrink_preserve_unique_charges() {
    fn transfer(lease: Lease) -> Lease {
        lease
    }
    let ledger = Ledger::new(1024);
    let lease = transfer(ledger.reserve(1024).unwrap());
    lease.commit(512).unwrap();
    lease.commit(512).unwrap();
    assert_eq!(ledger.totals().committed, 512);
    let pin = lease.pin().unwrap();
    assert_eq!(lease.commit(0), Err(Error::Growth));
    drop(pin);
    lease.commit(0).unwrap();
    drop(lease);
    assert_eq!(ledger.totals().reserved, 0);
}

#[test]
fn ledger_conservative_scratch_and_failed_bundle_return_reservations() {
    use budget::ledger::Demand;
    let demand = Demand::conservative(4 * GIB, 4 * GIB, 1024, 1024, 0).unwrap();
    assert_eq!(demand.scratch, 2 * GIB);
    assert_eq!(
        Demand::conservative(12 * GIB, GIB, 0, 0, 0)
            .unwrap()
            .scratch,
        13 * GIB / 4
    );
    assert_eq!(
        Demand::conservative(GIB, GIB, 0, 0, 3 * GIB)
            .unwrap()
            .scratch,
        3 * GIB
    );
    assert!(Demand::conservative(u64::MAX, 1, 0, 0, 0).is_err());
    let ledger = Ledger::new(8 * GIB);
    assert!(demand.reserve_all(&ledger).is_err());
    assert_eq!(ledger.totals().reserved, 0);
    let ledger = Ledger::new(demand.total().unwrap());
    let leases = demand.reserve_all(&ledger).unwrap();
    assert_eq!(ledger.totals().reserved, demand.total().unwrap());
    drop(leases);
    assert_eq!(ledger.totals().reserved, 0);
}

#[test]
fn prompt_and_output_growth_have_exact_numeric_boundaries() {
    budget::validate_context(14_336, 2048, 16_384, 2048).unwrap();
    budget::validate_context(255, 1, 256, 64).unwrap();
    assert_eq!(
        budget::validate_context(14_337, 2048, 16_384, 2048),
        Err(Error::Context { allowance: 2047 })
    );
    assert_eq!(
        budget::validate_context(256, 1, 256, 64),
        Err(Error::Context { allowance: 0 })
    );
    assert_eq!(
        budget::validate_context(1, 65, 256, 64),
        Err(Error::Context { allowance: 64 })
    );
    assert_eq!(budget::validate_context(0, 1, 256, 64), Err(Error::Empty));
    assert_eq!(budget::validate_context(1, 0, 256, 64), Err(Error::Empty));
    assert!(budget::validate_context(u32::MAX, 1, 16_384, 2048).is_err());
}

#[test]
fn pressure_orders_eviction_before_cancel_synchronize_and_unload() {
    assert!(budget::pressure_actions(Some(Severity::Nominal), true).is_empty());
    assert_eq!(
        budget::pressure_actions(Some(Severity::Elevated), true),
        vec![Action::StopAdmission, Action::EvictUnpinned]
    );
    assert_eq!(
        budget::pressure_actions(Some(Severity::Critical), false),
        vec![
            Action::StopAdmission,
            Action::EvictUnpinned,
            Action::Cancel,
            Action::Synchronize,
            Action::Unload
        ]
    );
    assert_eq!(
        budget::pressure_actions(None, false),
        budget::pressure_actions(Some(Severity::Critical), false)
    );
}
