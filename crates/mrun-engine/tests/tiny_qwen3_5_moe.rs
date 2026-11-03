//! Tiny random-weight `qwen3_5_moe` checkpoints (`tests/fixtures/gen_tiny.py`): two Gated
//! DeltaNet and two gated attention layers, sparse MoE with a shared expert.
//!
//! - `tiny_qwen3_5_moe`: 4-bit MLX layout whose 8-bit routers are named only in
//!   `quantization_config` (as in the published Qwen3.6 conversion), `Dk = 128` (the packed
//!   kernel on Metal);
//! - `tiny_qwen3_5_moe_hf`: dense f32 Hugging Face layout (fused experts, `[C, 1, K]` conv
//!   kernels, norms stored as offsets from 1, vision and MTP tensors to drop).
//!
//! Both are bitwise identical to mlx-lm 0.32.0 on the CPU and on Metal. The position-bound
//! state follows the contract (and `ScriptedEngine`): checkpoints, trim rules, pipelined
//! rollback, cancelled chunks, detach/attach and `NoRecurrentState`.

#![cfg(feature = "engine")]

mod common;

use common::{Device, Reference, assert_bitwise, greedy, load, next, on_each_device, prefill};
use mrun_engine::{CancelToken, Engine, EngineError, Sampling};

const NAME: &str = "tiny_qwen3_5_moe";

#[test]
fn matches_mlx_lm_bitwise() {
    on_each_device(|device| common::assert_parity(NAME, device));
}

#[test]
fn hugging_face_layout_matches_mlx_lm_bitwise() {
    on_each_device(|device| common::assert_parity("tiny_qwen3_5_moe_hf", device));
}

#[test]
fn state_sizes_follow_the_layers() {
    let engine = load(NAME, Device::Cpu);
    let info = engine.info();
    assert!(info.is_hybrid());
    // Two linear layers: conv tail 3 × (2·2·128 + 4·16) bf16 + delta state 4·16·128 f32.
    assert_eq!(info.recurrent_bytes, 2 * (3 * 576 * 2 + 4 * 16 * 128 * 4));
    // Two attention layers: K and V of 2 heads × 32 in bf16.
    assert_eq!(info.kv_bytes_per_token, 2 * 2 * 2 * 32 * 2);
}

#[test]
fn checkpoint_restore_reproduces_the_continuation() {
    on_each_device(|device: Device| {
        let reference = Reference::read(NAME);
        let run = reference.run(device);
        let (body, last) = (reference.body(), reference.last());
        let mut engine = load(NAME, device);
        engine.set_limit(64).unwrap();
        prefill(&mut engine, body, &[]);
        let base = engine.allocated();
        let checkpoint = engine.checkpoint().unwrap();
        assert_eq!(checkpoint.position, 23);
        assert_eq!(checkpoint.bytes, engine.info().recurrent_bytes);
        assert_eq!(engine.allocated(), base + checkpoint.bytes);

        let (rows, tokens) = greedy(&mut engine, last, 4);
        assert_eq!(tokens, run.greedy[..4]);
        assert_bitwise(&rows[0], &run.single[0], "first continuation");
        engine.restore(checkpoint.id).unwrap();
        assert_eq!(engine.len(), 23);
        let (again, tokens_again) = greedy(&mut engine, last, 4);
        assert_eq!(tokens_again, tokens);
        for (i, (a, b)) in again.iter().zip(&rows).enumerate() {
            assert_bitwise(a, b, &format!("after restore, step {i}"));
        }
        // The checkpoint stays until dropped.
        engine.restore(checkpoint.id).unwrap();
        engine.drop_checkpoint(checkpoint.id);
        assert_eq!(engine.allocated(), base);
        assert!(matches!(
            engine.restore(checkpoint.id),
            Err(EngineError::UnknownCheckpoint(_))
        ));
    });
}

#[test]
fn trim_allows_only_the_live_length_the_pending_cut_or_zero() {
    on_each_device(|device: Device| {
        let reference = Reference::read(NAME);
        let run = reference.run(device);
        let (body, last) = (reference.body(), reference.last());
        let mut engine = load(NAME, device);
        engine.set_limit(64).unwrap();
        prefill(&mut engine, body, &[]);
        assert_eq!(
            engine.trim(22),
            Err(EngineError::NotTrimmable {
                len: 22,
                active: 23
            })
        );
        engine.trim(23).unwrap();

        // A pipelined step is rolled back exactly: recurrent state included.
        let step = engine.forward(last).unwrap();
        let sampled = engine.sample(step, &Sampling::GREEDY, None).unwrap();
        assert_eq!(sampled, run.greedy[0]);
        assert_eq!(engine.len(), 25, "the sampled token is already fed");
        engine.trim(24).unwrap();
        assert_eq!(engine.len(), 24);
        assert_bitwise(
            &next(&mut engine, sampled),
            &run.single[1],
            "after rollback",
        );

        // Confirming a pending step keeps it.
        let step = engine.forward(run.greedy[1]).unwrap();
        engine.sample(step, &Sampling::GREEDY, None).unwrap();
        assert_eq!(engine.len(), 27);
        engine.trim(27).unwrap();
        assert_eq!(engine.len(), 27);
        assert!(matches!(
            engine.trim(25),
            Err(EngineError::NotTrimmable { .. })
        ));

        // Zero resets the recurrent state: the prompt runs as on a fresh engine.
        engine.trim(0).unwrap();
        prefill(&mut engine, body, &[]);
        assert_bitwise(&next(&mut engine, last), &run.single[0], "after trim(0)");
    });
}

#[test]
fn a_cancelled_chunk_leaves_the_recurrent_state_at_its_start() {
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
        assert_eq!(engine.len(), 11);
        prefill(&mut engine, &body[11..], &[]);
        assert_bitwise(&next(&mut engine, last), want, "after a cancelled chunk");
    });
}

#[test]
fn detach_moves_the_recurrent_state_and_attach_restores_it() {
    on_each_device(|device: Device| {
        let reference = Reference::read(NAME);
        let run = reference.run(device);
        let (body, last) = (reference.body(), reference.last());
        let mut engine = load(NAME, device);
        engine.set_limit(64).unwrap();
        prefill(&mut engine, &body[..11], &reference.chunk_ends);
        let at_11 = engine.checkpoint().unwrap();
        prefill(&mut engine, &body[11..], &[]);
        let before = engine.allocated();

        let segment = engine.detach_tail(11).unwrap();
        assert_eq!((segment.start, segment.end, engine.len()), (11, 23, 11));
        // The segment holds 12 positions and the recurrent state, which left the live state; the
        // live allocation (64 positions, the limit) is already at most align(11).
        let (kv, recurrent) = (
            engine.info().kv_bytes_per_token,
            engine.info().recurrent_bytes,
        );
        assert_eq!(segment.bytes, 12 * kv + recurrent);
        assert_eq!(engine.allocated(), before + 12 * kv);
        assert_eq!(engine.forward(last), Err(EngineError::NoRecurrentState));
        assert_eq!(
            engine.prefill_chunk(&body[11..], &CancelToken::new()),
            Err(EngineError::NoRecurrentState)
        );
        assert_eq!(engine.checkpoint(), Err(EngineError::NoRecurrentState));

        // Attaching brings back the state at 23.
        engine.attach(segment.id).unwrap();
        assert_eq!(engine.len(), 23);
        assert_bitwise(&next(&mut engine, last), &run.chunked, "after attach");

        // Detach the tail (now 24 positions) and drop it; restoring the checkpoint at 11 and
        // re-prefilling with the same chunk ends gives the same result (SC-008).
        let segment = engine.detach_tail(11).unwrap();
        assert_eq!(segment.end, 24);
        engine.drop_segment(segment.id);
        engine.restore(at_11.id).unwrap();
        prefill(&mut engine, &body[11..], &[]);
        assert_bitwise(&next(&mut engine, last), &run.chunked, "after restore");
    });
}
