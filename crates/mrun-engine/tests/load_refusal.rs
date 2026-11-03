//! Load refusals (SC-010, FR-002): an unknown `model_type`, an unknown wrapped
//! `text_config.model_type`, an unsupported quantization mode and unsupported per-path bits are
//! refused from `config.json` alone (their fixtures hold no `*.safetensors`, so no tensor data
//! can have been read); an unknown tensor name is refused after only the safetensors headers were
//! read (MLX's peak memory does not grow by the tensor's size during the failed load).

#![cfg(feature = "engine")]

mod common;

use mrun_engine::budget::{DeviceMemory, GIB};
use mrun_engine::models::LoadError;
use mrun_engine::tensor::{self, Array, memory};
use mrun_engine::{LoadOptions, MlxEngine, MlxLoadError};

fn load(dir: &std::path::Path) -> Result<MlxEngine, MlxLoadError> {
    MlxEngine::load(
        dir,
        LoadOptions {
            identity: "refusal".into(),
            cpu: true,
            fallback_device: Some(DeviceMemory {
                physical: 16 * GIB,
                recommended_working_set: 16 * GIB,
                available: None,
                estimated: false,
            }),
            ..LoadOptions::default()
        },
    )
}

fn config_only(name: &str) -> std::path::PathBuf {
    let dir = common::fixture(name);
    let tensors = std::fs::read_dir(&dir)
        .unwrap()
        .filter_map(Result::ok)
        .filter(|e| e.path().extension().is_some_and(|x| x == "safetensors"))
        .count();
    assert_eq!(tensors, 0, "{name} must hold only config.json");
    dir
}

#[test]
fn unknown_architectures_are_refused_naming_every_family() {
    std::thread::spawn(|| {
        for (name, arch) in [
            ("refuse_model_type", "llama4+llama4_text"),
            ("refuse_text_model_type", "qwen3_5+mamba2"),
        ] {
            let error = load(&config_only(name)).unwrap_err();
            assert!(
                matches!(
                    error,
                    MlxLoadError::Model(LoadError::UnsupportedArchitecture { .. })
                ),
                "{name}: {error}"
            );
            let message = error.to_string();
            assert!(message.contains(&format!("'{arch}'")), "{message}");
            assert!(
                message.contains("qwen3_5_moe, qwen3_5, qwen3_moe, qwen3"),
                "{message}"
            );
        }
    })
    .join()
    .unwrap();
}

#[test]
fn unsupported_quantization_is_refused_from_the_config() {
    std::thread::spawn(|| {
        for (name, says) in [
            ("refuse_quant_mode", "mode mxfp4"),
            ("refuse_quant_bits", "down_proj"),
        ] {
            let error = load(&config_only(name)).unwrap_err();
            match &error {
                MlxLoadError::Model(LoadError::InvalidConfig { key, reason }) => {
                    assert_eq!(key, "quantization", "{name}: {error}");
                    assert!(reason.contains(says), "{name}: {reason}");
                }
                other => panic!("{name}: {other}"),
            }
        }
    })
    .join()
    .unwrap();
}

#[test]
fn an_unknown_tensor_is_refused_without_reading_tensor_data() {
    std::thread::spawn(|| {
        tensor::use_device(tensor::Device::Cpu);
        let dir = tempfile::tempdir().unwrap();
        let source = common::fixture("tiny_qwen3_5");
        std::fs::copy(source.join("config.json"), dir.path().join("config.json")).unwrap();
        // The fixture's tensors plus an unknown 8 MiB one.
        let mut tensors = tensor::load_safetensors(&source.join("model.safetensors")).unwrap();
        let bogus = Array::from_f32(&vec![0.5; 2 << 20], &[2048, 1024]);
        tensors.push((
            "language_model.model.layers.0.mlp.bogus.weight".into(),
            bogus,
        ));
        let refs: Vec<(&str, &Array)> = tensors.iter().map(|(n, a)| (n.as_str(), a)).collect();
        tensor::save_safetensors(&dir.path().join("model.safetensors"), &refs).unwrap();
        drop(refs);
        drop(tensors);

        memory::clear_cache();
        // Allocated memory: mapping the file is not reading it.
        let start = memory::active_allocated();
        memory::reset_peak();
        let error = load(dir.path()).unwrap_err();
        let grown = memory::peak_allocated().saturating_sub(start);
        match &error {
            MlxLoadError::Model(LoadError::InvalidConfig { key, reason }) => {
                assert_eq!(key, "weights", "{error}");
                assert!(reason.contains("mlp.bogus.weight"), "{reason}");
            }
            other => panic!("{other}"),
        }
        assert!(
            grown < 1 << 20,
            "the failed load grew MLX memory by {grown} bytes: tensor data was read"
        );
    })
    .join()
    .unwrap();
}
