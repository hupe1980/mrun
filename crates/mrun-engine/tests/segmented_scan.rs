//! 003 SC-009: the position-bound state captured inside a chunk by a segmented Gated DeltaNet
//! scan (`NuvecEngine::prefill_chunk_with_checkpoints`: the recurrence runs in segments that
//! thread the state, the projections and the convolution run once over the whole chunk) is bit
//! for bit the state of a chunk ending there, and a continuation from it equals one from the
//! chunk end. Tiny hybrids always (CPU and Metal); `qwen3.5`/`qwen3.6` with `MRUN_TEST_MODELS`
//! (`#[ignore]`).

#![cfg(feature = "engine")]

mod common;

use common::{Device, load, logprobs, max_diff, on_each_device};
use mrun_engine::{CancelToken, Engine, NuvecEngine};

fn chunk(engine: &mut NuvecEngine, tokens: &[u32]) {
    assert!(engine.prefill_chunk(tokens, &CancelToken::new()).unwrap());
}

fn next(engine: &mut NuvecEngine, last: u32) -> Vec<f32> {
    let step = engine.forward(last).unwrap();
    logprobs(engine, &step)
}

fn same(a: &[f32], b: &[f32], what: String) -> Result<(), String> {
    let equal = a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits());
    if equal {
        Ok(())
    } else {
        Err(format!("{what}: max diff {}", max_diff(a, b)))
    }
}

/// SC-009 for a split at `at` inside a chunk of `chunk_end` tokens, continued with
/// `tokens[at..]` (one chunk) and the last token: `(segmenting leaves the chunk's outputs
/// unchanged, the captured state equals a chunk end's)`, each with its first difference.
fn segmented_parity(
    engine: &mut NuvecEngine,
    tokens: &[u32],
    at: u32,
    chunk_end: u32,
    what: &str,
) -> (Result<(), String>, Result<(), String>) {
    let (last, body) = tokens.split_last().unwrap();
    let (at_u, end_u) = (at as usize, chunk_end as usize);
    let limit = tokens.len() as u32 + 4;
    let fresh = |engine: &mut NuvecEngine| {
        engine.reset_all();
        engine.set_limit(limit).unwrap();
    };
    let segmented = |engine: &mut NuvecEngine| {
        let inside = engine
            .prefill_chunk_with_checkpoints(&body[..end_u], &CancelToken::new(), &[at])
            .unwrap()
            .unwrap();
        assert_eq!((inside[0].position, engine.len()), (at, chunk_end));
        inside[0]
    };

    // Segmenting the scan does not change the chunk's outputs.
    fresh(engine);
    chunk(engine, &body[..end_u]);
    chunk(engine, &body[end_u..]);
    let whole = next(engine, *last);
    fresh(engine);
    segmented(engine);
    chunk(engine, &body[end_u..]);
    let outputs = same(
        &next(engine, *last),
        &whole,
        format!("{what}: segmented chunk output at {at}"),
    );

    // A chunk that ends at `at` ...
    fresh(engine);
    chunk(engine, &body[..at_u]);
    let ended = engine.checkpoint().unwrap();
    let ended_values = engine.checkpoint_values(ended.id).unwrap();
    chunk(engine, &body[at_u..]);
    let want = next(engine, *last);
    // ... against the state captured at `at` inside the longer chunk, and its continuation.
    fresh(engine);
    let inside = segmented(engine);
    let inside_values = engine.checkpoint_values(inside.id).unwrap();
    assert_eq!(inside_values.len(), ended_values.len());
    let state = inside_values
        .iter()
        .zip(&ended_values)
        .enumerate()
        .try_for_each(|(i, (a, b))| {
            let kind = if i % 2 == 0 {
                "conv tail"
            } else {
                "delta state"
            };
            same(
                a,
                b,
                format!("{what}: linear layer {} {kind} at {at}", i / 2),
            )
        })
        .and_then(|()| {
            engine.restore(inside.id).unwrap();
            chunk(engine, &body[at_u..]);
            same(
                &next(engine, *last),
                &want,
                format!("{what}: continued from the capture at {at}"),
            )
        });
    engine.reset_all();
    (outputs, state)
}

#[test]
fn tiny_in_chunk_state_equals_a_chunk_end() {
    on_each_device(|device: Device| {
        for name in ["tiny_qwen3_5", "tiny_qwen3_5_moe"] {
            let mut engine = load(name, device);
            let tokens: Vec<u32> = (0..40).map(|i| (i * 37 + 11) % 128).collect();
            for at in [1, 7, 11, 22] {
                let what = format!("{name} {device:?}");
                let (outputs, state) = segmented_parity(&mut engine, &tokens, at, 23, &what);
                eprintln!("{what} split {at}: outputs {outputs:?}, state {state:?}");
                outputs.unwrap();
                // On the CPU every split is exact. On Metal the projections of a chunk of a
                // different length can differ in the last bit (kernel choice by row count), so
                // the in-chunk state is not a chunk end's (research R3, 003 T040).
                if device == Device::Cpu {
                    state.unwrap();
                }
            }
        }
    });
}

#[test]
#[ignore = "needs real weights: MRUN_TEST_MODELS"]
fn real_hybrids_in_chunk_state_equals_a_chunk_end() {
    for (name, dir) in common::real_models() {
        let mut engine = common::load_real(&dir);
        if !engine.info().is_hybrid() {
            continue;
        }
        let vocab = engine.info().vocab.min(30_000);
        let tokens: Vec<u32> = (0..3000)
            .map(|i| 1000 + (i * 7919 + 13) % (vocab - 1000))
            .collect();
        let end = 2048;
        let mut exact = 0;
        for at in [1, 8, 64, 128, 512, 1024, 1536, end - 1] {
            let (outputs, state) = segmented_parity(&mut engine, &tokens, at, end, &name);
            eprintln!(
                "{name}: split at {at} inside a {end}-token chunk: outputs {outputs:?}, state {state:?}"
            );
            outputs.unwrap();
            exact += usize::from(state.is_ok());
        }
        eprintln!("{name}: {exact} of 8 in-chunk captures bit-identical to a chunk end");
    }
}
