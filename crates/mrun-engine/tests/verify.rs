//! Speculative verification is serial decoding (006 FR-001, FR-002, SC-001): on every tiny
//! fixture, on Metal (the verify pass) and on the CPU (serial, no pass), greedy and seeded (top-p, repetition penalty), a verify of
//! drafts that are right, wrong at each position, reusing a pending step or cut short by the
//! reservation keeps exactly the tokens serial decoding samples, and every later step has serial
//! decoding's log-probabilities bit for bit; trims to each position of a verify leave serial
//! decoding's state.

#![cfg(feature = "engine")]

mod common;

use common::{Device, Reference, assert_bitwise, load, logprobs, on_each_device, prefill};
use mrun_engine::{Engine, NuvecEngine, Sampling};

const FAMILIES: [&str; 4] = [
    "tiny_qwen3_moe",
    "tiny_qwen3_5",
    "tiny_qwen3_5_moe",
    "tiny_qwen3_5_moe_hf",
];
/// Decode steps compared after the prompt.
const STEPS: usize = 12;
const LIMIT: u32 = 64;

fn seeded() -> Sampling {
    Sampling {
        temperature: 0.9,
        top_p: 0.95,
        repetition_penalty: 1.3,
        seed: Some(7),
        ..Sampling::GREEDY
    }
}

/// Serial decoding from the prompt: `STEPS` tokens and the log-probabilities before each
/// (`STEPS + 1` rows).
struct Serial {
    tokens: Vec<u32>,
    rows: Vec<Vec<f32>>,
}

fn prompted(name: &str, device: Device, limit: u32) -> (NuvecEngine, u32) {
    let reference = Reference::read(name);
    let mut engine = load(name, device);
    engine.set_limit(limit).unwrap();
    prefill(&mut engine, reference.body(), &[]);
    (engine, reference.last())
}

fn serial(name: &str, device: Device, sampling: &Sampling) -> Serial {
    let (mut engine, mut token) = prompted(name, device, LIMIT);
    let (mut tokens, mut rows) = (Vec::new(), Vec::new());
    for i in 0..=STEPS {
        let step = engine.forward(token).unwrap();
        rows.push(logprobs(&mut engine, &step));
        if i == STEPS {
            break;
        }
        token = engine.sample(step, sampling, None).unwrap();
        tokens.push(token);
    }
    Serial { tokens, rows }
}

/// Decodes on from `done` committed tokens (the last of them fed next) and checks every step
/// against serial decoding.
fn continues(
    engine: &mut NuvecEngine,
    serial: &Serial,
    done: usize,
    sampling: &Sampling,
    what: &str,
) {
    let mut token = serial.tokens[done - 1];
    for i in done..=STEPS {
        let step = engine.forward(token).unwrap();
        assert_bitwise(
            &logprobs(engine, &step),
            &serial.rows[i],
            &format!("{what}, step {i}"),
        );
        if i == STEPS {
            break;
        }
        token = engine.sample(step, sampling, None).unwrap();
        assert_eq!(token, serial.tokens[i], "{what}, token {i}");
    }
}

fn verifies_as_serial_decoding(device: Device) {
    for name in FAMILIES {
        for sampling in [Sampling::GREEDY, seeded()] {
            let serial = serial(name, device, &sampling);
            let what = |case: &str| format!("{name} {device:?} {case} ({sampling:?})");

            let (engine, _) = prompted(name, device, LIMIT);
            let vocab = engine.info().vocab;
            // The CPU has no row-exact products: its verify is serial decoding (`max_draft` 0),
            // still exact; the same drafts check it.
            let max = match device {
                Device::Gpu => {
                    assert_eq!(engine.max_draft(), 7, "{name}: 8 row-exact rows");
                    7
                }
                Device::Cpu => {
                    assert_eq!(engine.max_draft(), 0, "{name}: no verify pass on the CPU");
                    7
                }
            };

            // Right drafts: all kept, plus the sample after them.
            for n in [1, max] {
                let (mut engine, last) = prompted(name, device, LIMIT);
                let got = engine.verify(last, &serial.tokens[..n], &sampling).unwrap();
                assert_eq!(
                    got,
                    serial.tokens[..=n],
                    "{}",
                    what(&format!("right draft of {n}"))
                );
                continues(
                    &mut engine,
                    &serial,
                    n + 1,
                    &sampling,
                    &what("after a right draft"),
                );
            }

            // Wrong at position j: the first j kept, then serial decoding's own token.
            for j in 0..max {
                let mut draft = serial.tokens[..max].to_vec();
                draft[j] = (draft[j] + 1) % vocab;
                let (mut engine, last) = prompted(name, device, LIMIT);
                let got = engine.verify(last, &draft, &sampling).unwrap();
                assert_eq!(
                    got,
                    serial.tokens[..=j],
                    "{}",
                    what(&format!("wrong at {j}"))
                );
                continues(
                    &mut engine,
                    &serial,
                    j + 1,
                    &sampling,
                    &what(&format!("after wrong at {j}")),
                );
            }

            // After a plain step: the pending step of the sampled token is row 0.
            let (mut engine, last) = prompted(name, device, LIMIT);
            let step = engine.forward(last).unwrap();
            let first = engine.sample(step, &sampling, None).unwrap();
            assert_eq!(first, serial.tokens[0]);
            let got = engine
                .verify(first, &serial.tokens[1..=max], &sampling)
                .unwrap();
            assert_eq!(
                got,
                serial.tokens[1..=max + 1],
                "{}",
                what("after a pending step")
            );
            continues(
                &mut engine,
                &serial,
                max + 2,
                &sampling,
                &what("after a pending step"),
            );

            // A reservation with room for four more positions (`last` and three drafts): the
            // pass shortens the draft (serial decoding would fail like `forward`; the runtime
            // never drafts past the output limit).
            if device == Device::Gpu {
                let room = Reference::read(name).body().len() as u32 + 1 + 3;
                let (mut engine, last) = prompted(name, device, room);
                let got = engine
                    .verify(last, &serial.tokens[..max], &sampling)
                    .unwrap();
                assert!(got.len() <= 4, "{}", what("short reservation"));
                assert_eq!(
                    got,
                    serial.tokens[..got.len()],
                    "{}",
                    what("short reservation")
                );
            }
        }

        // Trims to each position of a verify pass leave serial decoding's state (greedy: a trim
        // does not rewind the draw count, so sampled decoding would draw on).
        if device == Device::Cpu {
            continue;
        }
        let serial = serial(name, device, &Sampling::GREEDY);
        let max = 7;
        for j in 0..=max + 1 {
            let (mut engine, last) = prompted(name, device, LIMIT);
            let start = engine.len();
            engine
                .verify(last, &serial.tokens[..max], &Sampling::GREEDY)
                .unwrap();
            engine.trim(start + j as u32).unwrap();
            let token = if j == 0 { last } else { serial.tokens[j - 1] };
            let step = engine.forward(token).unwrap();
            assert_bitwise(
                &logprobs(&mut engine, &step),
                &serial.rows[j],
                &format!("{name} {device:?}: trimmed to {j} of a verify"),
            );
        }

        // The next state change ends the verify: its states are dropped (they count against the
        // budget only until then) and a hybrid refuses to trim back into it.
        let (mut engine, last) = prompted(name, device, LIMIT);
        let start = engine.len();
        engine
            .verify(last, &serial.tokens[..max], &Sampling::GREEDY)
            .unwrap();
        let end = engine.len();
        engine.trim(end - 1).unwrap();
        prefill(&mut engine, &[serial.tokens[0]], &[]);
        if engine.info().is_hybrid() {
            assert!(engine.trim(start + 1).is_err(), "{name}: the window closed");
        }
    }
}

#[test]
fn a_verify_is_serial_decoding() {
    on_each_device(verifies_as_serial_decoding);
}
