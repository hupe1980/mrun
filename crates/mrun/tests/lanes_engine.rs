//! Lanes on the real engine through the worker (011 SC-004): requests running at once on two
//! lanes of a tiny model (one decoding while the other reads its prompt a layer at a time, then
//! both decoding in one batched, pipelined step) produce exactly the tokens each produces alone.

#![cfg(feature = "engine")]

use mrun::chat::prompt::RenderedPrompt;
use mrun::chat::tokenizer::test_support::byte_tokenizer;
use mrun::runtime::generate::{DecodeHook, Event, GenerateRequest};
use mrun::runtime::handle::{DEFAULT_STALL_TIMEOUT, EngineHandle};
use mrun_engine::{Sampling, TokenMask};
use std::sync::Arc;

/// Generates to the output limit (EOS is an ordinary token), so lengths are fixed.
struct ToTheLimit;

impl DecodeHook for ToTheLimit {
    fn on_token(&mut self, _id: u32) -> Option<TokenMask> {
        None
    }
    fn ignore_eos(&self) -> bool {
        true
    }
    fn masks(&self) -> bool {
        false
    }
}

fn handle(lanes: usize) -> EngineHandle {
    EngineHandle::spawn(
        move || {
            let dir = mrun_engine::testing::fixture("tiny_qwen3_5_moe");
            let mut engine =
                mrun_engine::testing::load_cpu(&dir, None).map_err(|e| e.to_string())?;
            engine.force_lanes(lanes);
            Ok(engine)
        },
        Arc::new(byte_tokenizer()),
        DEFAULT_STALL_TIMEOUT,
    )
    .unwrap()
}

fn request(seed: u32, len: u32, max_tokens: u32, sampling: Sampling) -> GenerateRequest {
    let tokens = (0..len).map(|i| 2 + (i * 7 + seed) % 120).collect();
    let mut request = GenerateRequest::new(RenderedPrompt::from_tokens(tokens), max_tokens);
    request.sampling = sampling;
    request.hook = Some(Box::new(ToTheLimit));
    request
}

fn tokens(events: &[Event]) -> Vec<u32> {
    events
        .iter()
        .filter_map(|e| match e {
            Event::Token { id, .. } => Some(*id),
            _ => None,
        })
        .collect()
}

#[test]
fn two_lanes_give_each_request_its_solo_tokens() {
    let seeded = Sampling {
        temperature: 0.8,
        top_p: 0.9,
        seed: Some(5),
        ..Sampling::GREEDY
    };
    // A decodes long; B reads a prompt of two chunks meanwhile, then both decode.
    let a = || request(1, 40, 60, seeded);
    let b = || request(9, 3_000, 30, Sampling::GREEDY);
    let solo = |r: GenerateRequest| tokens(&handle(1).generate(r).unwrap().wait().unwrap().0);
    let (solo_a, solo_b) = (solo(a()), solo(b()));
    assert_eq!((solo_a.len(), solo_b.len()), (60, 30));

    let both = handle(2);
    assert_eq!(both.lanes(), 2);
    let ga = both.generate(a()).unwrap();
    // B arrives while A decodes.
    let Event::Started { lane: lane_a, .. } = ga.events.recv().unwrap() else {
        panic!("A starts");
    };
    let gb = both.generate(b()).unwrap();
    let ta = std::thread::spawn(move || ga.wait().unwrap());
    let (events_b, finished_b) = gb.wait().unwrap();
    let (events_a, _) = ta.join().unwrap();
    let lane = |events: &[Event]| {
        events.iter().find_map(|e| match e {
            Event::Started { lane, .. } => Some(*lane),
            _ => None,
        })
    };
    assert_ne!(
        lane(&events_b),
        Some(lane_a),
        "B ran beside A, on the other lane"
    );
    assert_eq!(tokens(&events_a), solo_a);
    assert_eq!(tokens(&events_b), solo_b);
    assert_eq!(finished_b.stats.generated_tokens, 30);
}

/// Experiment: memory after each round of a two-lane soak on the GPU (011 SC-005).
#[test]
#[ignore = "experiment"]
fn memory_follows_the_ledger_over_many_rounds() {
    use mrun_engine::tensor::memory;
    let handle = EngineHandle::spawn(
        move || {
            let dir = mrun_engine::testing::fixture("tiny_qwen3_5_moe");
            let mut engine = mrun_engine::NuvecEngine::load(
                &dir,
                mrun_engine::LoadOptions {
                    identity: "tiny".into(),
                    fallback_device: Some(mrun_engine::budget::DeviceMemory {
                        physical: 16 << 30,
                        recommended_working_set: 16 << 30,
                        available: None,
                        estimated: false,
                    }),
                    ..Default::default()
                },
            )
            .map_err(|e| e.to_string())?;
            engine.force_lanes(std::env::var("LANES").map_or(2, |v| v.parse().unwrap()));
            Ok(engine)
        },
        Arc::new(byte_tokenizer()),
        DEFAULT_STALL_TIMEOUT,
    )
    .unwrap();
    let mut main: Vec<u32> = (0..800).map(|i| 2 + i % 100).collect();
    for round in 0..60u32 {
        let r = |seed: u32, len: u32, n: u32| request(seed, len, n, Sampling::GREEDY);
        let m = handle.generate({
            let mut q = r(0, 1, 20);
            q.prompt = RenderedPrompt::from_tokens(main.clone());
            q
        });
        let (events, _) = m.unwrap().wait().unwrap();
        main.extend(tokens(&events));
        main.extend((0..30).map(|i| 2 + (i * 3 + round) % 100));
        if main.len() > 3_000 {
            main.truncate(800);
        }
        let subs: Vec<_> = (0..3)
            .map(|i| handle.generate(r(500 + round * 3 + i, 600, 15)).unwrap())
            .collect();
        for s in subs {
            s.wait().unwrap();
        }
        let stats = handle.cache_stats().unwrap();
        eprintln!(
            "round {round}: active {:.1} MiB, live {} tokens, branches {} ({:.1} MiB), checkpoints {} ({:.1} MiB)",
            memory::active_allocated() as f64 / 1048576.0,
            stats.live_tokens,
            stats.branches,
            stats.buffer_bytes as f64 / 1048576.0,
            stats.checkpoints,
            stats.checkpoint_bytes as f64 / 1048576.0
        );
    }
}
