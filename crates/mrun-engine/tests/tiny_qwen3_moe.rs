//! A tiny random-weight `qwen3_moe` checkpoint (4-bit, 8-bit router, one dense MLP layer;
//! `tests/fixtures/gen_tiny.py`): bitwise parity with mlx-lm 0.32.0 on the CPU and on Metal
//! (one-chunk prefill with sorted expert dispatch, chunked prefill with unsorted dispatch, token
//! by token stepping, pipelined greedy decode), and the attention-state primitives.

#![cfg(feature = "engine")]

mod common;

use common::{Device, Reference, assert_bitwise, load, next, on_each_device, prefill};
use mrun_engine::{CancelToken, Engine};

const NAME: &str = "tiny_qwen3_moe";

#[test]
fn matches_mlx_lm_bitwise() {
    on_each_device(|device| common::assert_parity(NAME, device));
}

#[test]
fn cancel_mid_chunk_and_moves_are_exact() {
    on_each_device(|device: Device| {
        let reference = Reference::read(NAME);
        let want = &reference.run(device).chunked;
        let (body, last) = (reference.body(), reference.last());
        let mut engine = load(NAME, device);
        engine.set_limit(64).unwrap();
        prefill(&mut engine, &body[..11], &reference.chunk_ends);
        let cancel = CancelToken::new();
        cancel.cancel();
        assert!(!engine.prefill_chunk(&body[11..], &cancel).unwrap());
        assert_eq!(engine.len(), 11, "a cancelled chunk is not committed");
        prefill(&mut engine, &body[11..], &[]);
        assert_eq!(engine.len(), 23);

        // Detach above the first chunk, attach back: the same next-token distribution.
        let segment = engine.detach_tail(5).unwrap();
        assert_eq!((segment.start, segment.end, engine.len()), (5, 23, 5));
        engine.attach(segment.id).unwrap();
        assert_bitwise(&next(&mut engine, last), want, "after detach + attach");
        // Attention state is cut anywhere: back to 11 and re-prefilled the same way.
        engine.trim(11).unwrap();
        prefill(&mut engine, &body[11..], &[]);
        assert_bitwise(&next(&mut engine, last), want, "after trim");
    });
}
