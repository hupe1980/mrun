//! Failure atomicity (FR-028, SC-012) with the engine's failure-injection hook
//! (`MlxEngine::fail_next`): an MLX error inside `prefill_chunk` (after any layer), `forward`
//! or `sample`'s readback leaves the state, `len()` and `allocated()` as before the call, and
//! the next calls compute exactly what an undisturbed engine computes; an error half-way through
//! a move (`detach_tail`, `attach`) poisons the engine until `reset_all`.

#![cfg(feature = "engine")]

mod common;

use common::{Device, Reference, assert_bitwise, load, next, on_each_device, prefill};
use mrun_engine::{CancelToken, Engine, EngineError, FailPoint, Sampling};

const FIXTURES: [&str; 3] = ["tiny_qwen3_5", "tiny_qwen3_5_moe", "tiny_qwen3_moe"];

fn injected(result: Result<impl std::fmt::Debug, EngineError>, what: &str) {
    match result {
        Err(EngineError::Backend(message)) => assert!(message.contains("injected"), "{what}"),
        other => panic!("{what}: expected the injected failure, got {other:?}"),
    }
}

#[test]
fn failed_prefill_forward_and_sample_change_nothing() {
    on_each_device(|device: Device| {
        for name in FIXTURES {
            let reference = Reference::read(name);
            let run = reference.run(device);
            let (body, last) = (reference.body(), reference.last());
            let what = format!("{name} {device:?}");
            let mut engine = load(name, device);
            engine.set_limit(64).unwrap();
            prefill(&mut engine, &body[..11], &reference.chunk_ends);
            let layers = engine.info().layers() as usize;

            // Prefill: a failure after every layer in turn (the first chunk also grows).
            for layer in 0..layers {
                let (len, allocated) = (engine.len(), engine.allocated());
                engine.fail_next(FailPoint::PrefillLayer(layer));
                injected(
                    engine.prefill_chunk(&body[11..], &CancelToken::new()),
                    &format!("{what}: prefill layer {layer}"),
                );
                assert_eq!(
                    (engine.len(), engine.allocated()),
                    (len, allocated),
                    "{what}"
                );
            }
            prefill(&mut engine, &body[11..], &[]);
            assert_eq!(engine.len(), 23);

            // Forward.
            let allocated = engine.allocated();
            engine.fail_next(FailPoint::Forward);
            injected(engine.forward(last), &format!("{what}: forward"));
            assert_eq!(
                (engine.len(), engine.allocated()),
                (23, allocated),
                "{what}"
            );
            assert_bitwise(
                &next(&mut engine, last),
                &run.chunked,
                &format!("{what}: after forward"),
            );

            // Sample: the readback fails after the next step was built and fed.
            engine.trim(0).unwrap();
            prefill(&mut engine, body, &[]);
            let step = engine.forward(last).unwrap();
            let (len, allocated) = (engine.len(), engine.allocated());
            engine.fail_next(FailPoint::SampleReadback);
            injected(
                engine.sample(step, &Sampling::GREEDY, None),
                &format!("{what}: sample"),
            );
            assert_eq!(
                (engine.len(), engine.allocated()),
                (len, allocated),
                "{what}"
            );
            assert_bitwise(
                &next(&mut engine, run.greedy[0]),
                &run.single[1],
                &format!("{what}: after sample"),
            );
            // Nothing stays armed.
            let step = engine.forward(run.greedy[1]).unwrap();
            assert_eq!(
                engine.sample(step, &Sampling::GREEDY, None).unwrap(),
                run.greedy[2]
            );
        }
    });
}

#[test]
fn a_failed_move_poisons_until_reset_all() {
    on_each_device(|device: Device| {
        for name in FIXTURES {
            let reference = Reference::read(name);
            let run = reference.run(device);
            let (body, last) = (reference.body(), reference.last());
            let what = format!("{name} {device:?}");
            for point in [FailPoint::Detach, FailPoint::Attach] {
                let mut engine = load(name, device);
                engine.set_limit(64).unwrap();
                prefill(&mut engine, body, &reference.chunk_ends);
                if point == FailPoint::Detach {
                    engine.fail_next(point);
                    let result = engine.detach_tail(11);
                    assert!(
                        matches!(result, Err(EngineError::Poisoned(_))),
                        "{what}: {result:?}"
                    );
                } else {
                    let segment = engine.detach_tail(11).unwrap();
                    engine.fail_next(point);
                    let result = engine.attach(segment.id);
                    assert!(
                        matches!(result, Err(EngineError::Poisoned(_))),
                        "{what}: {result:?}"
                    );
                }
                assert!(matches!(
                    engine.forward(last),
                    Err(EngineError::Poisoned(_))
                ));
                assert!(matches!(
                    engine.prefill_chunk(&body[..3], &CancelToken::new()),
                    Err(EngineError::Poisoned(_))
                ));
                assert!(matches!(engine.checkpoint(), Err(EngineError::Poisoned(_))));
                engine.reset_all();
                // As a fresh engine: nothing but the (zero) live position-bound state.
                let fresh = engine.info().recurrent_bytes;
                assert_eq!((engine.len(), engine.allocated()), (0, fresh), "{what}");
                engine.set_limit(64).unwrap();
                prefill(&mut engine, body, &reference.chunk_ends);
                assert_bitwise(
                    &next(&mut engine, last),
                    &run.chunked,
                    &format!("{what}: {point:?}, after reset_all"),
                );
            }
        }
    });
}
