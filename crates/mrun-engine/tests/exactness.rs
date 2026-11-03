//! State primitives are exact (SC-008, US5): after detach/attach, and after checkpoint restore
//! (hybrids) or a cut (attention-only) followed by a re-prefill with the same chunk split, the
//! next-token log-probabilities are bit-identical to the uninterrupted run; and a move never
//! holds more than one layer beyond what it moves (R13), with no copy at all of the live prefix
//! when the capacity stays (FR-030).
//!
//! The tiny fixtures run always (CPU and Metal); the real models (`MRUN_TEST_MODELS`) are
//! `#[ignore]`.

#![cfg(feature = "engine")]

mod common;

use common::{Device, assert_bitwise, load, logprobs, on_each_device, prefill};
use mrun_engine::tensor::{self, memory};
use mrun_engine::{Engine, LayerKind, MlxEngine};

const TINY: [&str; 3] = ["tiny_qwen3_5", "tiny_qwen3_5_moe", "tiny_qwen3_moe"];

/// A deterministic prompt of `n` tokens below `vocab`.
/// The backend's memory counters are process-wide, so a test that bounds memory must not overlap
/// with another test of this binary that allocates: they run one at a time.
static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn serial() -> std::sync::MutexGuard<'static, ()> {
    SERIAL
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

fn prompt(n: u32, vocab: u32) -> Vec<u32> {
    (0..n).map(|i| (i * 37 + 11) % vocab.min(30_000)).collect()
}

/// Next-token log-probabilities after feeding `last`.
fn next(engine: &mut MlxEngine, last: u32) -> Vec<f32> {
    let step = engine.forward(last).unwrap();
    logprobs(engine, &step)
}

/// The SC-008 checks on `engine` over `tokens` (prefilled with chunk ends `ends`), cut at `at`
/// (one of the ends).
fn assert_exact(engine: &mut MlxEngine, tokens: &[u32], ends: &[u32], at: u32, what: &str) {
    let (last, body) = tokens.split_last().unwrap();
    let limit = tokens.len() as u32 + 4;
    let hybrid = engine.info().is_hybrid();

    engine.reset();
    engine.set_limit(limit).unwrap();
    prefill(engine, body, ends);
    let want = next(engine, *last);

    // Detach the tail and attach it again.
    engine.reset();
    engine.set_limit(limit).unwrap();
    prefill(engine, body, ends);
    let segment = engine.detach_tail(at).unwrap();
    assert_eq!(engine.len(), at);
    engine.attach(segment.id).unwrap();
    assert_bitwise(
        &next(engine, *last),
        &want,
        &format!("{what}: detach/attach"),
    );

    // Drop the tail and recompute it with the same split.
    engine.reset();
    engine.set_limit(limit).unwrap();
    prefill(engine, &body[..at as usize], ends);
    let checkpoint = engine.checkpoint().unwrap();
    prefill(engine, &body[at as usize..], ends);
    if hybrid {
        let segment = engine.detach_tail(at).unwrap();
        engine.drop_segment(segment.id);
        engine.restore(checkpoint.id).unwrap();
    } else {
        engine.trim(at).unwrap();
    }
    assert_eq!(engine.len(), at);
    prefill(engine, &body[at as usize..], ends);
    assert_bitwise(
        &next(engine, *last),
        &want,
        &format!(
            "{what}: re-prefill after {}",
            if hybrid { "restore" } else { "cut" }
        ),
    );
    engine.drop_checkpoint(checkpoint.id);
    engine.reset();
}

#[test]
fn tiny_moves_and_recomputation_are_bit_identical() {
    let _serial = serial();
    on_each_device(|device: Device| {
        for name in TINY {
            let mut engine = load(name, device);
            let tokens = prompt(40, engine.info().vocab);
            assert_exact(
                &mut engine,
                &tokens,
                &[5, 11, 24],
                11,
                &format!("{name} {device:?}"),
            );
        }
    });
}

/// Bytes of one attention layer's K and V at `capacity` positions.
fn layer_bytes(engine: &MlxEngine, capacity: u32) -> u64 {
    let info = engine.info();
    let attention = info
        .layer_kinds
        .iter()
        .filter(|k| **k == LayerKind::FullAttention)
        .count() as u64;
    info.kv_bytes_per_token / attention * u64::from(capacity)
}

/// Peak active memory above the start during `op`.
fn peak_during<T>(op: impl FnOnce() -> T) -> (T, u64) {
    memory::clear_cache();
    let start = memory::active();
    memory::reset_peak();
    let out = op();
    (out, memory::peak().saturating_sub(start))
}

/// Peak memory of moves (R13, FR-030) on `engine` with `n` prompt tokens: a detach that shrinks
/// the live state holds at most the segment plus one layer, one that keeps the capacity holds
/// only the segment (the live prefix is not copied), and an attach at most one layer.
fn assert_moves_hold_one_layer(engine: &mut MlxEngine, n: u32, slack: u64, what: &str) {
    let tokens = prompt(n, engine.info().vocab);
    // Capacity above align(at): the detach shrinks the live layers.
    let limit = (n + 4096).min(engine.info().max_position);
    engine.reset();
    engine.set_limit(limit).unwrap();
    common::prefill_plan(engine, &tokens, None);
    let at = n / 4;
    let layer = layer_bytes(engine, limit);
    let (segment, peak) = peak_during(|| engine.detach_tail(at).unwrap());
    assert!(
        peak <= segment.bytes + layer + slack,
        "{what}: shrinking detach peaked {peak} above the start, segment {} + layer {layer}",
        segment.bytes
    );
    let (result, peak) = peak_during(|| engine.attach(segment.id));
    result.unwrap();
    let layer = layer_bytes(engine, limit);
    assert!(
        peak <= layer + slack,
        "{what}: attach peaked {peak} above the start, one layer is {layer}"
    );

    // Capacity at the limit, at most align(at): the detach keeps the live buffers (FR-030).
    engine.reset();
    engine.set_limit(n).unwrap();
    common::prefill_plan(engine, &tokens, None);
    let at = n - 64;
    let (segment, peak) = peak_during(|| engine.detach_tail(at).unwrap());
    assert!(
        peak <= segment.bytes + slack,
        "{what}: a detach at unchanged capacity peaked {peak}, the segment is {} (one layer {})",
        segment.bytes,
        layer_bytes(engine, n)
    );
    engine.drop_segment(segment.id);
    engine.reset();
}

#[test]
fn tiny_moves_hold_at_most_one_extra_layer() {
    if !tensor::metal_available() {
        return;
    }
    let _serial = serial();
    std::thread::spawn(|| {
        // 3,000 positions: one attention layer is 768 KB, well above the allocator's rounding.
        let mut engine = load("tiny_qwen3_5", Device::Gpu);
        assert_moves_hold_one_layer(&mut engine, 3000, 96 << 10, "tiny_qwen3_5");
    })
    .join()
    .unwrap();
}

#[test]
#[ignore = "needs real weights: MRUN_TEST_MODELS"]
fn real_models_are_exact_and_move_one_layer_at_a_time() {
    let _serial = serial();
    for (name, dir) in common::real_models() {
        let mut engine = common::load_real(&dir);
        let vocab = engine.info().vocab;
        // Three chunks of the engine's plan, cut at the second chunk boundary.
        let first = engine.info().chunk_len(0);
        let second = engine.info().chunk_len(first);
        let at = first + second;
        let tokens = prompt(at + 700, vocab);
        assert_exact(&mut engine, &tokens, &[first, at], at, &name);
        assert_moves_hold_one_layer(&mut engine, 6000, 4 << 20, &name);
        eprintln!("{name}: exact; moves within one layer");
    }
}
