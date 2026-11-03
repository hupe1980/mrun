//! Query-blocked attention (D059, FR-017): when a chunk's unfused score buffer would exceed
//! `SCORE_BLOCK_BYTES`, attention runs in query blocks over the same keys. On the tiny hybrids
//! (head dim 32, so MLX's unfused path) blocks of a few rows are forced and compared with the
//! unblocked run: bit-identical for bf16 checkpoints (blocks keep at least 9 rows, so MLX stays
//! on the unfused path the whole chunk takes); the f32 fixture differs below 1e-5.

#![cfg(feature = "engine")]

mod common;

use common::{Device, load, max_diff, next, on_each_device, prefill};

use mrun_engine::Engine;
use mrun_engine::models::set_score_block_bytes;

#[test]
fn blocked_attention_matches_the_whole_chunk() {
    on_each_device(|device: Device| {
        for name in ["tiny_qwen3_5", "tiny_qwen3_5_moe", "tiny_qwen3_5_moe_hf"] {
            let tokens: Vec<u32> = (0..40).map(|i| (i * 37 + 11) % 128).collect();
            let (last, body) = tokens.split_last().unwrap();
            let run = |limit: u64| {
                set_score_block_bytes(limit);
                let mut engine = load(name, device);
                engine.set_limit(64).unwrap();
                prefill(&mut engine, body, &[5, 23]);
                let out = next(&mut engine, *last);
                set_score_block_bytes(mrun_engine::models::SCORE_BLOCK_BYTES_DEFAULT);
                out
            };
            let whole = run(u64::MAX);
            // 4 heads × 39 keys × dtype: a limit of one 3-row block.
            for rows in [1u64, 3, 7, 16] {
                let blocked = run(4 * 40 * 4 * rows);
                let diff = max_diff(&blocked, &whole);
                let same = blocked
                    .iter()
                    .zip(&whole)
                    .all(|(a, b)| a.to_bits() == b.to_bits());
                eprintln!(
                    "{name} {device:?} blocks of ~{rows} rows: bitwise {same}, max diff {diff}"
                );
                // bf16 checkpoints (the published models' dtype) are bit-identical; the f32
                // Hugging Face fixture differs in the last bits (f32 matmuls of other row counts).
                if name.ends_with("_hf") {
                    assert!(diff < 1e-5, "{name} {device:?}: {diff}");
                } else {
                    assert!(
                        same,
                        "{name} {device:?}: blocks of ~{rows} rows differ by {diff}"
                    );
                }
            }
        }
    });
}
