//! The pipeline across a pool (026 T013): a tiny model split between this process (the host:
//! embeddings, layer 1, head) and a child process (a stage: layer 2), joined by the ring over
//! loopback, does what the whole model does in one process: the same next-token log-probabilities
//! bit for bit and the same greedy tokens, through every state operation the prefix cache uses
//! (moves, trims, checkpoints, resets), with both Macs' state in step.

#![cfg(feature = "engine")]

use mrun::pool::engine::{PoolEngine, connect, serve_stage};
use mrun_engine::budget::{DeviceMemory, GIB};
use mrun_engine::{CancelToken, Engine, LoadOptions, NuvecEngine, Part, Sampling, testing};
use std::path::Path;
use std::process::Command;

const STAGE: &str = "MRUN_TEST_POOL_STAGE";
const TEST: &str = "a_split_model_does_what_the_whole_model_does";

fn load(dir: &Path, part: Option<Part>) -> NuvecEngine {
    mrun_engine::tensor::use_device(mrun_engine::tensor::Device::Cpu);
    NuvecEngine::load(
        dir,
        LoadOptions {
            identity: "tiny".into(),
            cpu: true,
            fallback_device: Some(DeviceMemory {
                physical: 16 * GIB,
                recommended_working_set: 16 * GIB,
                available: None,
                estimated: false,
            }),
            part,
            ..LoadOptions::default()
        },
    )
    .unwrap()
}

fn nodes(ports: &[u16]) -> Vec<Vec<String>> {
    ports
        .iter()
        .map(|p| vec![format!("127.0.0.1:{p}")])
        .collect()
}

/// The next token's top log-probabilities and the greedy choice.
fn step(engine: &mut dyn Engine, last: u32) -> (Vec<(u32, u32)>, u32) {
    let step = engine.forward(last).unwrap();
    let top = engine
        .top_logprobs(&step, 5)
        .unwrap()
        .into_iter()
        .map(|(t, p)| (t, p.to_bits()))
        .collect();
    let token = engine.sample(step, &Sampling::GREEDY, None).unwrap();
    (top, token)
}

/// The same operations on any engine; returns everything they produced.
fn script(engine: &mut dyn Engine) -> Vec<String> {
    let never = CancelToken::new();
    let mut seen = Vec::new();
    engine.set_limit(64).unwrap();
    assert!(engine.prefill_chunk(&[1, 5, 9, 2], &never).unwrap());
    let mut last = 7;
    for _ in 0..6 {
        let (top, token) = step(engine, last);
        seen.push(format!("{top:?} {token}"));
        last = token;
    }
    // A branch: move the tail out, run another continuation, come back. (Lengths are not
    // compared: the whole model holds a pipelined step a pool does not feed.)
    let segment = engine.detach_tail(4).unwrap();
    // Ids are internal (the whole model counts pipelined steps too); the stage's agree with the
    // host's, or the status frame would have failed the call.
    seen.push(format!("segment {}..{}", segment.start, segment.end));
    assert!(engine.prefill_chunk(&[3, 3, 8], &never).unwrap());
    let (top, _) = step(engine, 6);
    seen.push(format!("{top:?}"));
    engine.trim(4).unwrap();
    engine.attach(segment.id).unwrap();
    let checkpoint = engine.checkpoint().unwrap();
    seen.push(format!("checkpoint at {}", checkpoint.position));
    let (top, _) = step(engine, last);
    seen.push(format!("{top:?}"));
    engine.drop_checkpoint(checkpoint.id);
    engine.reset();
    assert_eq!(engine.len(), 0);
    assert!(engine.prefill_chunk(&[2, 2], &never).unwrap());
    let (top, _) = step(engine, 4);
    seen.push(format!("{top:?}"));
    seen
}

#[test]
fn a_split_model_does_what_the_whole_model_does() {
    if let Ok(spec) = std::env::var(STAGE) {
        // The child: the stage on layer 2.
        let (dir, ports) = spec.split_once('|').unwrap();
        let ports: Vec<u16> = ports.split(',').map(|p| p.parse().unwrap()).collect();
        let mut stage = load(
            Path::new(dir),
            Some(Part {
                first: 1,
                end: 2,
                head: false,
            }),
        );
        stage.set_pipe(Some(connect(&nodes(&ports), 1, &[9; 32], false).unwrap()));
        serve_stage(&mut stage).unwrap();
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    testing::write_tiny_qwen3(dir.path(), 64).unwrap();
    let ports: Vec<u16> = (0..2)
        .map(|_| {
            std::net::TcpListener::bind("127.0.0.1:0")
                .unwrap()
                .local_addr()
                .unwrap()
                .port()
        })
        .collect();
    let spec = format!(
        "{}|{}",
        dir.path().display(),
        ports
            .iter()
            .map(u16::to_string)
            .collect::<Vec<_>>()
            .join(",")
    );
    let mut child = Command::new(std::env::current_exe().unwrap())
        .args([TEST, "--exact", "--nocapture", "--test-threads=1"])
        .env(STAGE, spec)
        .spawn()
        .unwrap();
    let whole = script(&mut load(dir.path(), None));
    let host = load(
        dir.path(),
        Some(Part {
            first: 0,
            end: 1,
            head: true,
        }),
    );
    let pipe = connect(&nodes(&ports), 0, &[9; 32], false).unwrap();
    let mut pool = PoolEngine::new(host, pipe, u64::MAX, Default::default());
    let split = script(&mut pool);
    drop(pool); // stops the stage
    assert!(child.wait().unwrap().success(), "the stage failed");
    assert_eq!(split, whole);
}
