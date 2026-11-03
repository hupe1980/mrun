//! `chunk_len` calibration on Metal (FR-007, D050; T068, T077): the warm-up's time bound is
//! stored per (device, family, quantization, numerics class, identity) and reused, so two starts
//! produce the same chunk plan; a stored entry is used as is (never recomputed); the CPU is not
//! calibrated; the bound never changes during the process.

#![cfg(feature = "engine")]

mod common;

use mrun_engine::budget::{DeviceMemory, GIB};
use mrun_engine::{CancelToken, Engine, LoadOptions, MlxEngine, tensor};

fn load(path: &std::path::Path, cpu: bool) -> MlxEngine {
    MlxEngine::load(
        &common::fixture("tiny_qwen3_5"),
        LoadOptions {
            identity: "tiny_qwen3_5".into(),
            cpu,
            calibration: Some(path.to_path_buf()),
            fallback_device: Some(DeviceMemory {
                physical: 16 * GIB,
                recommended_working_set: 16 * GIB,
                available: None,
                estimated: false,
            }),
            ..LoadOptions::default()
        },
    )
    .unwrap()
}

#[test]
fn two_starts_produce_the_same_chunk_plan() {
    if !tensor::metal_available() {
        return;
    }
    std::thread::spawn(|| {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("calibration.toml");
        let first = load(&path, false);
        let plan = first.info().time_len.clone();
        assert!(!plan.is_unbounded(), "Metal engines are calibrated");
        // A tiny model is fast: the memory bound stands at every depth.
        for depth in [0, 1000, 4000] {
            assert_eq!(first.info().chunk_len(depth), mrun_engine::chunk_len(depth));
        }
        drop(first);
        let text = std::fs::read_to_string(&path).unwrap();
        assert!(text.contains("tiny_qwen3_5"), "{text}");
        assert_eq!(
            load(&path, false).info().time_len,
            plan,
            "the second start reuses it"
        );

        // A stored entry is taken as is: an edited (slow) one shortens the chunks.
        let edited = text.replace(&format!("[0, {}]", plan.points()[0].1), "[0, 256]");
        assert_ne!(edited, text);
        std::fs::write(&path, &edited).unwrap();
        let mut engine = load(&path, false);
        assert_eq!(engine.info().chunk_len(0), 256);
        // Fixed for the process: work does not change it.
        let before = engine.info().time_len.clone();
        engine.set_limit(64).unwrap();
        assert!(engine.prefill_chunk(&[1; 40], &CancelToken::new()).unwrap());
        assert_eq!(engine.info().time_len, before);
    })
    .join()
    .unwrap();
}

#[test]
fn the_cpu_is_not_calibrated() {
    std::thread::spawn(|| {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("calibration.toml");
        let engine = load(&path, true);
        assert!(engine.info().time_len.is_unbounded());
        assert_eq!(engine.info().numerics.gpu_arch, "cpu");
        assert!(!engine.info().numerics.nax);
        assert!(!path.exists());
    })
    .join()
    .unwrap();
}

#[test]
fn numerics_never_claim_nax_without_the_kernels() {
    let numerics = tensor::numerics();
    if !tensor::NAX_COMPILED {
        // A build for macOS 15.0 (or an older SDK) has no NAX kernels.
        assert!(!numerics.nax, "{numerics}");
    }
    // The class carries NAX exactly when the backend runs its kernels (016 FR-016).
    let status = tensor::nax_status();
    assert_eq!(
        numerics.nax,
        status.starts_with("on"),
        "{numerics}: {status}"
    );
    if tensor::metal_available() {
        assert!(numerics.gpu_arch.starts_with("applegpu"), "{numerics}");
    } else {
        assert_eq!(numerics.gpu_arch, "cpu");
    }
}
