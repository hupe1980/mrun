//! The DFlash drafter never changes output (030 SC-001): on the tiny fixtures with a
//! random-weight drafter, on the CPU and Metal, greedy and seeded, rounds of drafts (the
//! drafter's own, and right or wrong ones to reach every kept length) decode exactly the tokens
//! serial decoding samples, through segments, lanes and a cancelled chunk; and the drafter's
//! context follows the sequence.

#![cfg(all(feature = "engine", feature = "test-support"))]

mod common;

use common::{Device, fixture, on_each_device, prefill};
use mrun_engine::budget::{DeviceMemory, GIB};
use mrun_engine::{CancelToken, Engine, LoadOptions, NuvecEngine, Sampling, testing};
use std::path::Path;

const FAMILIES: [&str; 3] = ["tiny_qwen3_moe", "tiny_qwen3_5", "tiny_qwen3_5_moe"];
const PROMPT: usize = 24;
const STEPS: usize = 24;
const LIMIT: u32 = 96;

fn seeded() -> Sampling {
    Sampling {
        temperature: 0.9,
        top_p: 0.95,
        repetition_penalty: 1.3,
        seed: Some(7),
        ..Sampling::GREEDY
    }
}

fn prompt() -> Vec<u32> {
    (0..PROMPT as u32).map(|i| (i * 7 + 3) % 120).collect()
}

/// A tiny drafter for fixture `name` (its width, vocabulary and first and last layer).
fn drafter(name: &str, dir: &Path) {
    let config: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(fixture(name).join("config.json")).unwrap())
            .unwrap();
    let text = if config["text_config"].is_object() {
        &config["text_config"]
    } else {
        &config
    };
    let layers = text["num_hidden_layers"].as_u64().unwrap() as usize;
    testing::write_tiny_dflash(
        dir,
        text["hidden_size"].as_i64().unwrap() as i32,
        text["vocab_size"].as_u64().unwrap() as u32,
        &[0, layers - 1],
    )
    .unwrap();
}

fn load(name: &str, device: Device, drafter: Option<&Path>) -> NuvecEngine {
    let mut engine = NuvecEngine::load(
        &fixture(name),
        LoadOptions {
            identity: name.into(),
            cpu: device == Device::Cpu,
            fallback_device: Some(DeviceMemory {
                physical: 16 * GIB,
                recommended_working_set: 16 * GIB,
                available: None,
                estimated: false,
            }),
            drafter: drafter.map(Path::to_path_buf),
            ..LoadOptions::default()
        },
    )
    .unwrap_or_else(|e| panic!("{name}: {e}"));
    engine.set_limit(LIMIT).unwrap();
    engine
}

/// Serial decoding: `STEPS` tokens after the prompt.
fn serial(name: &str, device: Device, sampling: &Sampling) -> Vec<u32> {
    let mut engine = load(name, device, None);
    let prompt = prompt();
    prefill(&mut engine, &prompt[..PROMPT - 1], &[]);
    let mut token = prompt[PROMPT - 1];
    let mut tokens = Vec::new();
    while tokens.len() < STEPS {
        let step = engine.forward(token).unwrap();
        token = engine.sample(step, sampling, None).unwrap();
        tokens.push(token);
    }
    tokens
}

/// Speculative rounds from `last` until `want.len()` tokens: the drafter's draft on even rounds,
/// on odd ones the right continuation with position `round % 7` wrong (every kept length).
fn rounds(engine: &mut NuvecEngine, mut last: u32, want: &[u32], sampling: &Sampling, what: &str) {
    let max = engine.max_draft() as usize;
    let mut got: Vec<u32> = Vec::new();
    let mut round = 0;
    while got.len() < want.len() {
        let left = want.len() - got.len();
        let n = max.min(left.saturating_sub(1));
        let draft = if round % 2 == 0 {
            let draft = engine.draft(last, n as u32).unwrap();
            assert!(draft.len() <= n, "{what}: draft too long");
            draft
        } else {
            let mut draft = want[got.len()..got.len() + n].to_vec();
            if let Some(d) = draft.get_mut(round % 7) {
                *d = (*d + 1) % 100;
            }
            draft
        };
        let kept = engine.verify(last, &draft, sampling).unwrap();
        got.extend(&kept);
        last = *kept.last().unwrap();
        // The drafter has every computed position up to the pending step's.
        let (_, end) = engine.draft_span().expect("a drafter context");
        assert!(
            end + 1 >= engine.len(),
            "{what}: context ends at {end}, len {}",
            engine.len()
        );
        round += 1;
    }
    got.truncate(want.len());
    assert_eq!(got, want, "{what}");
}

fn decodes_as_serial(device: Device) {
    let dir = tempfile::tempdir().unwrap();
    for name in FAMILIES {
        drafter(name, dir.path());
        for sampling in [Sampling::GREEDY, seeded()] {
            let want = serial(name, device, &sampling);
            let what = format!("{name} {device:?} {sampling:?}");
            let mut engine = load(name, device, Some(dir.path()));
            assert!(engine.info().drafter.is_some());
            let prompt = prompt();
            prefill(&mut engine, &prompt[..PROMPT - 1], &[]);
            // The tiny drafter's window is 8: it keeps the last 7 positions.
            assert_eq!(engine.draft_span(), Some((16, PROMPT as u32 - 1)), "{what}");
            let draft = engine.draft(prompt[PROMPT - 1], 7).unwrap();
            assert_eq!(draft.len(), 7, "{what}: a full draft from the prompt");
            rounds(&mut engine, prompt[PROMPT - 1], &want, &sampling, &what);
        }
    }
}

#[test]
fn drafts_never_change_output() {
    on_each_device(decodes_as_serial);
}

fn follows_segments_and_lanes(device: Device) {
    let dir = tempfile::tempdir().unwrap();
    let name = "tiny_qwen3_5";
    drafter(name, dir.path());
    let want = serial(name, device, &Sampling::GREEDY);
    let prompt = prompt();

    // A segment carries the context of its positions, charged with it.
    let mut plain = load(name, device, None);
    let mut engine = load(name, device, Some(dir.path()));
    for e in [&mut plain, &mut engine] {
        prefill(e, &prompt[..PROMPT - 1], &[]);
    }
    let bare = plain.detach_tail(10).unwrap();
    let segment = engine.detach_tail(10).unwrap();
    let per_position = engine.info().drafter.as_ref().unwrap().context_bytes
        / u64::from(engine.info().drafter.as_ref().unwrap().window - 1 + 256);
    // The context holds positions 16..23 (its window): the segment takes all of them.
    assert_eq!(segment.bytes - bare.bytes, 7 * per_position);
    assert_eq!(engine.draft_span(), Some((0, 0)));
    engine.attach(segment.id).unwrap();
    assert_eq!(engine.draft_span(), Some((16, PROMPT as u32 - 1)));
    rounds(
        &mut engine,
        prompt[PROMPT - 1],
        &want,
        &Sampling::GREEDY,
        &format!("{name} {device:?} after attach"),
    );

    // A cancelled chunk feeds nothing.
    let mut engine = load(name, device, Some(dir.path()));
    prefill(&mut engine, &prompt[..10], &[]);
    let cancel = CancelToken::new();
    cancel.cancel();
    assert!(
        !engine
            .prefill_chunk(&prompt[10..PROMPT - 1], &cancel)
            .unwrap()
    );
    assert_eq!(engine.draft_span(), Some((3, 10)));

    // Lanes: another lane's steps leave this lane's context alone, and batched steps feed each
    // lane's own.
    let mut engine = load(name, device, Some(dir.path()));
    engine.force_lanes(2);
    engine.set_limit(LIMIT).unwrap();
    prefill(&mut engine, &prompt[..PROMPT - 1], &[]);
    engine.select(1);
    engine.set_limit(LIMIT).unwrap();
    prefill(&mut engine, &prompt[..5], &[]);
    let mut steps = engine
        .forward_lanes(&[(0, prompt[PROMPT - 1]), (1, 9)])
        .unwrap();
    engine.select(1);
    assert_eq!(engine.draft_span(), Some((0, 6)));
    engine.select(0);
    assert_eq!(engine.draft_span(), Some((16, PROMPT as u32)));
    let token = engine
        .sample(steps.swap_remove(0), &Sampling::GREEDY, None)
        .unwrap();
    assert_eq!(token, want[0]);
    rounds(
        &mut engine,
        token,
        &want[1..],
        &Sampling::GREEDY,
        &format!("{name} {device:?} lanes"),
    );
}

#[test]
fn the_context_follows_segments_lanes_and_cancels() {
    on_each_device(follows_segments_and_lanes);
}

#[test]
fn a_mismatched_drafter_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    testing::write_tiny_dflash(dir.path(), 32, 128, &[0, 1]).unwrap();
    let error = NuvecEngine::load(
        &fixture("tiny_qwen3_5"),
        LoadOptions {
            identity: "tiny".into(),
            cpu: true,
            fallback_device: Some(DeviceMemory {
                physical: 16 * GIB,
                recommended_working_set: 16 * GIB,
                available: None,
                estimated: false,
            }),
            drafter: Some(dir.path().to_path_buf()),
            ..LoadOptions::default()
        },
    )
    .expect_err("refused");
    assert!(error.to_string().contains("hidden_size"), "{error}");
}
