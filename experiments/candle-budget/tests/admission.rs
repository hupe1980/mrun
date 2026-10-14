//! T116 allocation-contract probes. Real Candle CPU constructors run under a counting global
//! allocator; the ledger must deny before constructor entry and roll back exactly once.

mod support;

use candle_core::{DType, Device, Tensor};
use mrun_candle_budget::budget::{self, Budget, Denied, Guarded};
use std::cell::Cell;

#[global_allocator]
static ALLOCATOR: support::Counting = support::Counting;

/// Tensor/storage/layout metadata observed per Candle CPU constructor at the pinned revision.
const METADATA: u64 = 1024;
const MIB: u64 = 1024 * 1024;

type Case<'a> = (&'a str, Box<dyn Fn() -> Tensor + 'a>, &'a [usize], DType);

fn demand(dims: &[usize], dtype: DType) -> u64 {
    budget::tensor_bytes(dims, dtype.size_in_bytes()).unwrap() + METADATA
}

#[test]
fn lawful_small_reservation_admits_constructor_and_covers_its_allocations() {
    let _serial = support::serial();
    let ledger = Budget::new(MIB);
    let bytes = demand(&[32, 32], DType::F32);
    let (result, observed) = support::measure(|| {
        ledger.guarded(bytes, || Tensor::zeros((32, 32), DType::F32, &Device::Cpu))
    });
    let (tensor, lease) = result.expect("lawful reservation admits the constructor");
    assert!(observed.calls > 0, "constructor entry must be observed");
    assert!(
        observed.thread_bytes <= lease.bytes(),
        "{} observed > {} reserved",
        observed.thread_bytes,
        lease.bytes()
    );
    assert_eq!(ledger.totals().reserved, bytes);
    drop((tensor, lease));
    assert_eq!(ledger.totals().reserved, 0);
}

#[test]
fn insufficient_reservation_is_denied_before_constructor_entry() {
    let _serial = support::serial();
    let ledger = Budget::new(4096);
    let entered = Cell::new(false);
    let (result, observed) = support::measure(|| {
        ledger.guarded(8192, || {
            entered.set(true);
            Tensor::zeros((32, 64), DType::F32, &Device::Cpu)
        })
    });
    assert!(matches!(
        result,
        Err(Guarded::Denied(Denied::Insufficient {
            requested: 8192,
            available: 4096
        }))
    ));
    assert!(
        !entered.get(),
        "denied work must never enter the constructor"
    );
    assert_eq!(observed.thread_calls, 0, "denial itself must not allocate");
    assert_eq!(ledger.totals().reserved, 0);
}

#[test]
fn arithmetic_and_rounding_overflow_are_rejected() {
    let _serial = support::serial();
    assert_eq!(
        budget::tensor_bytes(&[usize::MAX, 2], 4),
        Err(Denied::Overflow)
    );
    assert_eq!(
        budget::tensor_bytes(&[1 << 40, 1 << 40], 8),
        Err(Denied::Overflow)
    );
    assert_eq!(budget::tensor_bytes(&[4, 8], 2), Ok(64));
    // Mirrors Candle's pooled `buf_size`, which is `next_power_of_two` without a check.
    assert_eq!(budget::pool_bucket(0), Ok(1));
    assert_eq!(budget::pool_bucket(5000), Ok(8192));
    assert_eq!(budget::pool_bucket(1 << 63), Ok(1 << 63));
    assert_eq!(budget::pool_bucket((1 << 63) + 1), Err(Denied::Overflow));
    let ledger = Budget::new(u64::MAX);
    let held = ledger.reserve(u64::MAX - 1).unwrap();
    assert_eq!(ledger.reserve(2).err(), Some(Denied::Overflow));
    drop(held);
    assert_eq!(ledger.totals().reserved, 0);
}

#[test]
fn aliased_views_charge_once_and_independent_staging_charges_separately() {
    let _serial = support::serial();
    let ledger = Budget::new(MIB);
    let backing = ledger.reserve(4096).unwrap();
    let gpu_view = backing.view();
    let cpu_view = gpu_view.clone();
    assert_eq!(ledger.totals().reserved, 4096);
    assert_eq!(ledger.totals().live_leases, 1);
    let staging = ledger.reserve(4096).unwrap();
    assert_eq!(ledger.totals().reserved, 8192);
    assert_eq!(ledger.totals().live_leases, 2);
    drop((backing, staging));
    assert_eq!(
        ledger.totals().reserved,
        4096,
        "views keep the backing charged"
    );
    drop(gpu_view);
    assert_eq!(ledger.totals().reserved, 4096);
    drop(cpu_view);
    assert_eq!(ledger.totals().reserved, 0);
    assert_eq!(ledger.totals().releases, 2);
}

#[test]
fn failed_constructor_rolls_back_exactly_once() {
    let _serial = support::serial();
    let ledger = Budget::new(MIB);
    let result: Result<((), _), _> = ledger.guarded(4096, || Err("constructor failed"));
    assert_eq!(result.err(), Some(Guarded::Failed("constructor failed")));
    assert_eq!(ledger.totals().reserved, 0);
    assert_eq!(ledger.totals().releases, 1);
    assert_eq!(ledger.totals().live_leases, 0);
}

#[test]
fn simple_cpu_constructors_fit_their_output_demand() {
    let _serial = support::serial();
    let device = Device::Cpu;
    let a = Tensor::ones((256, 256), DType::F32, &device).unwrap();
    let b = Tensor::ones((256, 256), DType::F32, &device).unwrap();
    let cases: [Case; 5] = [
        (
            "zeros",
            Box::new(|| Tensor::zeros((256, 256), DType::F32, &device).unwrap()),
            &[256, 256],
            DType::F32,
        ),
        (
            "from_vec",
            Box::new(|| Tensor::from_vec(vec![0f32; 65_536], (256, 256), &device).unwrap()),
            &[256, 256],
            DType::F32,
        ),
        (
            "to_dtype",
            Box::new(|| a.to_dtype(DType::F16).unwrap()),
            &[256, 256],
            DType::F16,
        ),
        (
            "contiguous",
            Box::new(|| a.t().unwrap().contiguous().unwrap()),
            &[256, 256],
            DType::F32,
        ),
        (
            "cat",
            Box::new(|| Tensor::cat(&[&a, &b], 0).unwrap()),
            &[512, 256],
            DType::F32,
        ),
    ];
    for (name, construct, dims, dtype) in cases {
        let (tensor, observed) = support::measure(&construct);
        let reserved = demand(dims, dtype);
        assert!(
            observed.peak <= reserved,
            "{name}: peak {} > {reserved}",
            observed.peak
        );
        drop(tensor);
    }
}

/// Counterexample on the unpatched candidate: output-only demand does not bound CPU matmul,
/// whose gemm packing scratch is allocated inside the op with no Candle hook.
#[test]
fn unpatched_cpu_matmul_exceeds_output_only_demand() {
    let _serial = support::serial();
    let device = Device::Cpu;
    let a = Tensor::ones((256, 256), DType::F32, &device).unwrap();
    let b = Tensor::ones((256, 256), DType::F32, &device).unwrap();
    let output_only = demand(&[256, 256], DType::F32);
    let (product, observed) = support::measure(|| a.matmul(&b).unwrap());
    assert!(
        observed.peak > 4 * output_only,
        "expected op-internal scratch beyond output demand: peak {} vs {output_only}",
        observed.peak
    );
    drop(product);
}
