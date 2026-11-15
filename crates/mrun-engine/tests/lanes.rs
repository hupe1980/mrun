//! Lanes are exact (011 FR-014, SC-004): on every tiny fixture, on the CPU and on Metal, greedy
//! and seeded, a batched decode step of 2–4 lanes at different depths gives every lane the
//! log-probabilities and tokens it gets decoding alone, bit for bit, pipelined steps included;
//! and a prefill chunk run a layer at a time between another lane's decode steps leaves the
//! state a whole chunk leaves.

#![cfg(feature = "engine")]

mod common;

use common::{Device, Reference, assert_bitwise, load, logprobs, on_each_device, prefill};
use mrun_engine::{CancelToken, ChunkProgress, Engine, NuvecEngine, Pick, Sampling};

const FAMILIES: [&str; 4] = [
    "tiny_qwen3_moe",
    "tiny_qwen3_5",
    "tiny_qwen3_5_moe",
    "tiny_qwen3_5_moe_hf",
];
const STEPS: usize = 10;
const LIMIT: u32 = 96;

fn seeded() -> Sampling {
    Sampling {
        temperature: 0.9,
        top_p: 0.95,
        repetition_penalty: 1.3,
        seed: Some(11),
        ..Sampling::GREEDY
    }
}

/// Lane `b`'s prompt: the fixture's body cut shorter per lane (different depths), and its last
/// token.
fn prompt(name: &str, b: usize) -> (Vec<u32>, u32) {
    let r = Reference::read(name);
    let body = r.body();
    let cut = body.len().saturating_sub(3 * b).max(1);
    (
        body[..cut].to_vec(),
        body.get(cut).copied().unwrap_or(r.last()),
    )
}

/// Decoding alone: each step's log-probabilities and the sampled tokens.
fn solo(name: &str, device: Device, b: usize, sampling: &Sampling) -> (Vec<Vec<f32>>, Vec<u32>) {
    let mut engine = load(name, device);
    engine.set_limit(LIMIT).unwrap();
    let (body, mut token) = prompt(name, b);
    prefill(&mut engine, &body, &[]);
    let (mut rows, mut tokens) = (Vec::new(), Vec::new());
    for _ in 0..STEPS {
        let step = engine.forward(token).unwrap();
        rows.push(logprobs(&mut engine, &step));
        token = engine.sample(step, sampling, None).unwrap();
        tokens.push(token);
    }
    (rows, tokens)
}

fn lanes_engine(name: &str, device: Device, lanes: usize) -> NuvecEngine {
    let mut engine = load(name, device);
    engine.force_lanes(lanes);
    assert_eq!(engine.lanes(), lanes);
    engine
}

fn batched_rows_are_solo_rows(device: Device) {
    for name in FAMILIES {
        for sampling in [Sampling::GREEDY, seeded()] {
            let solos: Vec<_> = (0..4).map(|b| solo(name, device, b, &sampling)).collect();
            for width in 2..=4 {
                let what = |b: usize, i: usize| {
                    format!("{name} {device:?} {sampling:?}: {width} lanes, lane {b}, step {i}")
                };
                let mut engine = lanes_engine(name, device, width);
                let mut last = Vec::new();
                for b in 0..width {
                    engine.select(b);
                    engine.set_limit(LIMIT).unwrap();
                    let (body, token) = prompt(name, b);
                    prefill(&mut engine, &body, &[]);
                    last.push(token);
                }
                engine.select(0);
                for i in 0..STEPS {
                    let rows: Vec<(usize, u32)> = (0..width).map(|b| (b, last[b])).collect();
                    let steps = engine.forward_lanes(&rows).unwrap();
                    for (b, step) in steps.iter().enumerate() {
                        engine.select(b);
                        assert_bitwise(&logprobs(&mut engine, step), &solos[b].0[i], &what(b, i));
                    }
                    let picks = steps
                        .into_iter()
                        .enumerate()
                        .map(|(lane, step)| Pick {
                            lane,
                            step,
                            sampling: &sampling,
                            mask: None,
                        })
                        .collect();
                    for (b, token) in engine.sample_lanes(picks).into_iter().enumerate() {
                        let token = token.unwrap();
                        assert_eq!(token, solos[b].1[i], "{}", what(b, i));
                        last[b] = token;
                    }
                }
            }
        }
    }
}

fn an_interleaved_chunk_is_a_whole_chunk(device: Device) {
    for name in FAMILIES {
        let sampling = Sampling::GREEDY;
        let (rows0, tokens0) = solo(name, device, 0, &sampling);
        let (rows1, _) = solo(name, device, 1, &sampling);
        let (rows2, _) = solo(name, device, 2, &sampling);
        let mut engine = lanes_engine(name, device, 3);
        let (body0, mut token0) = prompt(name, 0);
        engine.set_limit(LIMIT).unwrap();
        prefill(&mut engine, &body0, &[]);
        // Lane 0 decodes between layers; lane 1 starts its chunk, lane 2 takes over after one
        // layer and finishes first, then lane 1 resumes where it paused.
        let mut i = 0;
        let mut decode = |engine: &mut NuvecEngine| {
            if i >= STEPS {
                return;
            }
            let step = engine.forward_lanes(&[(0, token0)]).unwrap().remove(0);
            engine.select(0);
            assert_bitwise(
                &logprobs(engine, &step),
                &rows0[i],
                &format!("{name} {device:?}: lane 0, step {i}"),
            );
            token0 = engine.sample(step, &sampling, None).unwrap();
            assert_eq!(token0, tokens0[i]);
            i += 1;
        };
        let (body1, token1) = prompt(name, 1);
        let (body2, token2) = prompt(name, 2);
        let mut run = |engine: &mut NuvecEngine, lane: usize, layers: usize| loop {
            engine.select(lane);
            let progress = engine.advance_chunk(1, &CancelToken::new()).unwrap();
            decode(engine);
            if progress == ChunkProgress::Done {
                return true;
            }
            if layers == 1 {
                return false;
            }
        };
        engine.select(1);
        engine.set_limit(LIMIT).unwrap();
        engine.begin_chunk(&body1).unwrap();
        assert!(!run(&mut engine, 1, 1), "lane 1 paused after a layer");
        engine.select(2);
        engine.set_limit(LIMIT).unwrap();
        engine.begin_chunk(&body2).unwrap();
        assert!(run(&mut engine, 2, usize::MAX));
        assert!(run(&mut engine, 1, usize::MAX));
        for (lane, token, rows) in [(1, token1, &rows1), (2, token2, &rows2)] {
            engine.select(lane);
            let step = engine.forward(token).unwrap();
            assert_bitwise(
                &logprobs(&mut engine, &step),
                &rows[0],
                &format!("{name} {device:?}: lane {lane} after interleaved chunks"),
            );
        }
        // A cancelled interleaved chunk leaves the lane as it was.
        engine.select(1);
        let before = engine.len();
        engine.begin_chunk(&[1, 2, 3]).unwrap();
        let cancel = CancelToken::new();
        engine.advance_chunk(1, &cancel).unwrap();
        cancel.cancel();
        assert_eq!(
            engine.advance_chunk(usize::MAX, &cancel).unwrap(),
            ChunkProgress::Cancelled
        );
        assert_eq!(engine.len(), before);
    }
}

#[test]
fn a_batched_step_gives_every_lane_its_solo_rows() {
    on_each_device(batched_rows_are_solo_rows);
}

#[test]
fn a_chunk_between_other_lanes_steps_is_a_whole_chunk() {
    on_each_device(an_interleaved_chunk_is_a_whole_chunk);
}

#[test]
fn a_lane_count_that_does_not_fit_is_refused_with_its_numbers() {
    use mrun_engine::budget::{DeviceMemory, GIB};
    use mrun_engine::{LoadOptions, NuvecLoadError};
    let load = |lanes| {
        NuvecEngine::load(
            &common::fixture("tiny_qwen3_5_moe"),
            LoadOptions {
                identity: "tiny".into(),
                cpu: true,
                fallback_device: Some(DeviceMemory {
                    physical: 16 * GIB,
                    recommended_working_set: 16 * GIB,
                    available: None,
                    estimated: false,
                }),
                lanes,
                ..LoadOptions::default()
            },
        )
    };
    // The fixture holds 4,096 positions: one lane, automatic or asked for.
    assert_eq!(load(None).unwrap().lanes(), 1);
    assert_eq!(load(Some(1)).unwrap().lanes(), 1);
    match load(Some(2)) {
        Err(
            error @ NuvecLoadError::Lanes {
                requested: 2,
                capacity,
                fits: 1,
            },
        ) => {
            let message = error.to_string();
            assert!(message.contains("2 lanes need 65536 tokens"), "{message}");
            assert!(message.contains(&capacity.to_string()), "{message}");
            assert!(message.ends_with("; 1 lane fits"), "{message}");
        }
        other => panic!("{:?}", other.map(|e| e.lanes())),
    }
}
