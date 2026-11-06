//! A pool is exact with a real model (026 T016, SC-003, SC-004): a small pulled model split
//! between this process (the host: embeddings, the first half of the layers, head) and a child
//! process (a stage: the second half), joined by the ring over loopback on the GPU, gives the same
//! greedy tokens and bit-identical log-probabilities as the whole model in one process, for a
//! prompt and for a follow-up that continues from the reused state.
//!
//! Ignored by default (a few minutes on the GPU; the model must be pulled):
//! `cargo test -p mrun --test pool_exact -- --ignored` (`MRUN_POOL_EXACT_MODEL`, default qwen3.5).

#![cfg(feature = "engine")]

use mrun::pool::engine::{PoolEngine, connect, serve_stage};
use mrun_engine::{CancelToken, Engine, LoadOptions, NuvecEngine, Part, Sampling};
use std::process::Command;

const STAGE: &str = "MRUN_TEST_POOL_EXACT_STAGE";
const TEST: &str = "a_split_real_model_replies_as_the_whole_model";

fn model_name() -> String {
    std::env::var("MRUN_POOL_EXACT_MODEL").unwrap_or_else(|_| "qwen3.5".into())
}

fn load(part: Option<Part>) -> NuvecEngine {
    let model = mrun::cli::model::open(&model_name()).expect("pull the model first");
    NuvecEngine::load(
        &model.dir,
        LoadOptions {
            identity: model.identity.value.clone(),
            fallback_device: mrun::hub::device_memory(),
            calibration: mrun_engine::calibration::default_path(),
            part,
            ..LoadOptions::default()
        },
    )
    .unwrap()
}

fn layers() -> usize {
    mrun::pool::serve::shape(&mrun::cli::model::open(&model_name()).unwrap())
        .layer_kinds
        .len()
}

fn nodes(ports: &[u16]) -> Vec<Vec<String>> {
    ports
        .iter()
        .map(|p| vec![format!("127.0.0.1:{p}")])
        .collect()
}

/// Greedy decoding of `n` tokens after `last`: each step's top log-probabilities (bits) and token.
fn decode(engine: &mut dyn Engine, mut last: u32, n: usize) -> (Vec<String>, u32) {
    let mut seen = Vec::new();
    for _ in 0..n {
        let step = engine.forward(last).unwrap();
        let top: Vec<(u32, u32)> = engine
            .top_logprobs(&step, 5)
            .unwrap()
            .into_iter()
            .map(|(t, p)| (t, p.to_bits()))
            .collect();
        let token = engine.sample(step, &Sampling::GREEDY, None).unwrap();
        seen.push(format!("{top:?} {token}"));
        last = token;
    }
    (seen, last)
}

/// A prompt, a reply, then a follow-up appended to the same state (no reset: reused).
fn script(engine: &mut dyn Engine) -> Vec<String> {
    let model = mrun::cli::model::open(&model_name()).unwrap();
    let tokens = |text: &str| model.tokenizer.encode(text).unwrap();
    let never = CancelToken::new();
    engine.set_limit(4096).unwrap();
    let prompt = tokens("Write a Rust function that adds two numbers, and explain it briefly.");
    let (head, last) = prompt.split_at(prompt.len() - 1);
    assert!(engine.prefill_chunk(head, &never).unwrap());
    let (mut seen, reply_last) = decode(engine, last[0], 24);
    // The follow-up continues from the state kept (lengths are not compared: the whole model
    // may hold a pipelined step a pool does not).
    let follow = tokens(" Now make it generic over any type that can be added.");
    let mut chunk = vec![reply_last];
    chunk.extend_from_slice(&follow[..follow.len() - 1]);
    assert!(engine.prefill_chunk(&chunk, &never).unwrap());
    let (more, _) = decode(engine, follow[follow.len() - 1], 16);
    seen.extend(more);
    seen
}

#[test]
#[ignore = "a few minutes on the GPU with a pulled model (the Mac may be shared)"]
fn a_split_real_model_replies_as_the_whole_model() {
    let half = layers() / 2;
    if let Ok(ports) = std::env::var(STAGE) {
        let ports: Vec<u16> = ports.split(',').map(|p| p.parse().unwrap()).collect();
        let mut stage = load(Some(Part {
            first: half,
            end: layers(),
            head: false,
        }));
        stage.set_pipe(Some(connect(&nodes(&ports), 1, &[5; 32], false).unwrap()));
        serve_stage(&mut stage).unwrap();
        return;
    }
    let whole = {
        let mut engine = load(None);
        script(&mut engine)
    };
    let ports: Vec<u16> = (0..2)
        .map(|_| {
            std::net::TcpListener::bind("127.0.0.1:0")
                .unwrap()
                .local_addr()
                .unwrap()
                .port()
        })
        .collect();
    let mut child = Command::new(std::env::current_exe().unwrap())
        .args([
            TEST,
            "--exact",
            "--nocapture",
            "--ignored",
            "--test-threads=1",
        ])
        .env(
            STAGE,
            ports
                .iter()
                .map(u16::to_string)
                .collect::<Vec<_>>()
                .join(","),
        )
        .spawn()
        .unwrap();
    let host = load(Some(Part {
        first: 0,
        end: half,
        head: true,
    }));
    let pipe = connect(&nodes(&ports), 0, &[5; 32], false).unwrap();
    let mut pool = PoolEngine::new(host, pipe, u64::MAX, Default::default());
    let split = script(&mut pool);
    drop(pool);
    assert!(child.wait().unwrap().success(), "the stage failed");
    assert_eq!(split.len(), whole.len());
    for (i, (s, w)) in split.iter().zip(&whole).enumerate() {
        assert_eq!(s, w, "step {i} differs");
    }
}
