//! What the allocator holds follows the ledger (constitution III; found by 011's soak): after
//! prefill chunks, a checkpoint, a moved branch and decode steps on real models, the bytes nuvec
//! holds stay within the ledger's count. Before, a Gated DeltaNet state kept its chunk's output
//! alive and the conv tail its chunk's whole input: 2.7× the count on `qwen3.6`.
//!
//! `MRUN_TEST_MODELS=qwen3.6 cargo test --release -p mrun-engine --features engine,test-support
//! --test memory_ledger -- --ignored`

#![cfg(feature = "engine")]

mod common;

use mrun_engine::tensor::memory;
use mrun_engine::{CancelToken, Engine, Sampling};

/// Allocator slack the ledger does not count: small arrays (logits, a step's token) and
/// alignment.
const SLACK: u64 = 8 << 20;

#[test]
#[ignore = "real models (MRUN_TEST_MODELS)"]
fn the_allocator_holds_what_the_ledger_counts() {
    for (name, dir) in common::real_models() {
        let mut e = common::load_real(&dir);
        // Empty lanes are counted with their recurrent state before they hold any.
        let empty = (e.lanes() as u64 - 1) * e.info().recurrent_bytes;
        let check = |e: &mrun_engine::NuvecEngine, what: &str| {
            memory::clear_cache();
            let (held, counted) = (memory::active_allocated(), e.allocated());
            assert!(
                held <= counted + SLACK,
                "{name} {what}: {} MiB held, {} MiB counted",
                held >> 20,
                counted >> 20
            );
            assert!(
                counted.saturating_sub(empty) <= held + SLACK + (64 << 20),
                "{name} {what}: the count is far above what is held"
            );
        };
        e.set_limit(30_000).unwrap();
        let tokens: Vec<u32> = (0..6000).map(|i| 100 + (i * 37) % 5000).collect();
        for chunk in tokens.chunks(2048) {
            e.prefill_chunk(chunk, &CancelToken::new()).unwrap();
            check(&e, "after a chunk");
        }
        let checkpoint = e.checkpoint().unwrap();
        check(&e, "after a checkpoint");
        let segment = e.detach_tail(2000).unwrap();
        check(&e, "after a detach");
        e.attach(segment.id).unwrap();
        let mut token = 7;
        for _ in 0..4 {
            let step = e.forward(token).unwrap();
            token = e.sample(step, &Sampling::GREEDY, None).unwrap();
        }
        e.trim(e.len() - 1).unwrap();
        check(&e, "after decode steps");
        e.drop_checkpoint(checkpoint.id);
    }
}
