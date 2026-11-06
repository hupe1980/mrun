//! Parts of a model (026 T003): every split of the tiny checkpoints into a head part and a tail
//! part, chained, gives the whole model's logits bit for bit, and each part reads only its own
//! tensors.

#![cfg(all(feature = "engine", feature = "test-support"))]

use mrun_engine::engine::LayerKind;
use mrun_engine::load::{Weights, unused_tensor, weights_bytes, weights_bytes_of};
use mrun_engine::models::{self, FamilyConfig, Model, Part};
use mrun_engine::state::{LayerKv, LayerState, Recurrent};
use mrun_engine::tensor::{self, Array, Device};
use mrun_engine::testing;
use std::path::Path;

fn load(dir: &Path, part: Part) -> Box<dyn Model> {
    let config: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("config.json")).unwrap()).unwrap();
    let family = FamilyConfig::from_config(&config).unwrap();
    let text = config
        .get("text_config")
        .filter(|t| t.is_object())
        .unwrap_or(&config);
    let weights = Weights::load(dir, |name| !unused_tensor(name) && part.keeps(name)).unwrap();
    models::build(family.family.name, &config, text, weights, part).unwrap()
}

fn layers(dir: &Path) -> (usize, u32, u32) {
    let config: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("config.json")).unwrap()).unwrap();
    let info = FamilyConfig::from_config(&config)
        .unwrap()
        .model_info(String::new(), Vec::new());
    (info.layer_kinds.len(), info.kv_heads, info.head_dim)
}

/// Runs `ids` through layers `range` of `model` on fresh state.
fn run(model: &dyn Model, x: Array, range: std::ops::Range<usize>, dir: &Path) -> Array {
    let (_, kv_heads, head_dim) = layers(dir);
    let mut x = x;
    for i in range {
        let mut state = match model.layer_kinds()[i] {
            LayerKind::FullAttention => LayerState::Kv(
                LayerKv::zeroed(kv_heads, head_dim, 64, model.state_dtype()).unwrap(),
            ),
            LayerKind::Linear => LayerState::Recurrent(Recurrent::default()),
        };
        x = model.layer(i, &x, &mut state, 0).unwrap();
    }
    x
}

fn check(dir: &Path) {
    let (n, _, _) = layers(dir);
    let ids = Array::from_u32(&[1, 5, 9, 2, 7, 3], &[1, 6]);
    let whole = load(dir, Part::whole(n));
    let expected = whole
        .logits(&run(whole.as_ref(), whole.embed(&ids).unwrap(), 0..n, dir))
        .unwrap()
        .to_vec_f32()
        .unwrap();
    for split in 1..n {
        let head = load(
            dir,
            Part {
                first: 0,
                end: split,
                head: true,
            },
        );
        let tail = load(
            dir,
            Part {
                first: split,
                end: n,
                head: false,
            },
        );
        let x = run(head.as_ref(), head.embed(&ids).unwrap(), 0..split, dir);
        let x = run(tail.as_ref(), x, split..n, dir);
        let got = head.logits(&x).unwrap().to_vec_f32().unwrap();
        let same = expected
            .iter()
            .zip(&got)
            .all(|(a, b)| a.to_bits() == b.to_bits());
        assert!(
            same && got.len() == expected.len(),
            "{}: split at {split}",
            dir.display()
        );
        assert!(tail.embed(&ids).is_err(), "a tail part has no embeddings");
        assert_eq!(tail.hidden(), whole.hidden());
        // A part reads only its own tensors (nuvec's footprint of an unaligned file is the byte
        // range it reads, so the head part may count the file up to its last tensor).
        let of = |part: Part| {
            weights_bytes_of(dir, move |name| !unused_tensor(name) && part.keeps(name))
        };
        let (h, t) = (
            of(Part {
                first: 0,
                end: split,
                head: true,
            }),
            of(Part {
                first: split,
                end: n,
                head: false,
            }),
        );
        let all = weights_bytes(dir);
        assert!(
            h <= all && t < all && t > 0,
            "{}: split at {split}: {h} + {t} of {all}",
            dir.display()
        );
    }
}

#[test]
fn every_split_chains_to_the_whole_model() {
    tensor::use_device(Device::Cpu);
    let dir = tempfile::tempdir().unwrap();
    testing::write_tiny_qwen3(dir.path(), 64).unwrap();
    check(dir.path());
    for name in ["tiny_qwen3_moe", "tiny_qwen3_5", "tiny_qwen3_5_moe"] {
        check(&testing::fixture(name));
    }
}

#[test]
fn a_part_outside_the_layers_is_refused() {
    tensor::use_device(Device::Cpu);
    let dir = tempfile::tempdir().unwrap();
    testing::write_tiny_qwen3(dir.path(), 64).unwrap();
    let mut options = mrun_engine::LoadOptions {
        identity: "tiny".into(),
        cpu: true,
        fallback_device: Some(mrun_engine::budget::DeviceMemory {
            physical: 16 * mrun_engine::budget::GIB,
            recommended_working_set: 16 * mrun_engine::budget::GIB,
            available: None,
            estimated: false,
        }),
        ..Default::default()
    };
    options.part = Some(Part {
        first: 1,
        end: 9,
        head: false,
    });
    let error = mrun_engine::NuvecEngine::load(dir.path(), options.clone()).unwrap_err();
    assert!(error.to_string().contains("not a range"), "{error}");
    // A tail part loads, holds the state of its own layer only, and runs on zeros.
    options.part = Some(Part {
        first: 1,
        end: 2,
        head: false,
    });
    let mut tail = mrun_engine::NuvecEngine::load(dir.path(), options).unwrap();
    use mrun_engine::Engine;
    let whole = testing::load_cpu(dir.path(), None).unwrap();
    assert_eq!(
        tail.info().kv_bytes_per_token * 2,
        whole.info().kv_bytes_per_token
    );
    tail.set_limit(8).unwrap();
    assert!(
        tail.prefill_chunk(&[0; 4], &mrun_engine::CancelToken::new())
            .unwrap()
    );
    assert_eq!(tail.len(), 4);
}
