//! Test support (feature `test-support`): tiny random-weight checkpoints that `mrun`'s suites
//! (prefix cache, splicing, exactness oracle) and this crate's tests load on a real
//! [`MlxEngine`](crate::MlxEngine).
//!
//! - [`write_tiny_qwen3`]: a dense `qwen3` (hidden 32, 2 layers, f32) with any vocabulary — 64
//!   rows for fast tests, [`QWEN3_VOCAB`] rows so the real Qwen3 tokenizer's ids are valid;
//! - [`fixture`]: the committed tiny checkpoints in `crates/mrun-engine/tests/fixtures/`
//!   (`tiny_qwen3_5_moe`: hybrid Gated DeltaNet + attention, vocabulary 128);
//! - [`load_cpu`]: loads one on the CPU backend with a fixed 16 GiB device.

use crate::budget::{DeviceMemory, GIB, Overrides};
use crate::tensor::{self, Array};
use crate::{LoadOptions, MlxEngine, MlxLoadError};
use std::path::{Path, PathBuf};

/// Rows of the real Qwen3 vocabulary (`vocab_size` of Qwen3-8B).
pub const QWEN3_VOCAB: u32 = 151_936;
/// `<|im_end|>` and `<|endoftext|>` in the Qwen3 vocabulary: the tiny model's EOS tokens when
/// it has the real vocabulary.
pub const QWEN3_EOS: [u32; 2] = [151_645, 151_643];

/// Shape of a tiny `qwen3`.
#[derive(Clone, Copy, Debug)]
pub struct TinyQwen3 {
    pub vocab: u32,
    /// `max_position_embeddings`: bounds the token capacity.
    pub max_position: u32,
    pub seed: u64,
}

impl TinyQwen3 {
    pub fn new(vocab: u32) -> Self {
        Self {
            vocab,
            max_position: 8192,
            seed: 7,
        }
    }
}

/// Deterministic pseudo-random values in `[-scale, scale)`.
fn noise(seed: &mut u64, n: usize, scale: f32) -> Vec<f32> {
    (0..n)
        .map(|_| {
            *seed = seed
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((*seed >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0) * scale
        })
        .collect()
}

/// Writes a tiny random-weight `qwen3` with `vocab` rows to `dir` (`config.json`,
/// `model.safetensors`). EOS: token 1, or [`QWEN3_EOS`] with the real vocabulary.
pub fn write_tiny_qwen3(dir: &Path, vocab: u32) -> Result<(), tensor::Error> {
    write_tiny_qwen3_with(dir, TinyQwen3::new(vocab))
}

pub fn write_tiny_qwen3_with(dir: &Path, shape: TinyQwen3) -> Result<(), tensor::Error> {
    const HIDDEN: i32 = 32;
    const HEADS: i32 = 4;
    const KV_HEADS: i32 = 2;
    const HEAD_DIM: i32 = 8;
    const FFN: i32 = 48;
    const LAYERS: usize = 2;
    let vocab = shape.vocab as i32;
    let eos = if shape.vocab == QWEN3_VOCAB {
        serde_json::json!(QWEN3_EOS)
    } else {
        serde_json::json!(1)
    };
    let config = serde_json::json!({
        "model_type": "qwen3",
        "hidden_size": HIDDEN,
        "intermediate_size": FFN,
        "num_hidden_layers": LAYERS,
        "num_attention_heads": HEADS,
        "num_key_value_heads": KV_HEADS,
        "head_dim": HEAD_DIM,
        "vocab_size": vocab,
        "max_position_embeddings": shape.max_position,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "tie_word_embeddings": false,
        "torch_dtype": "float32",
        "eos_token_id": eos
    });
    std::fs::write(dir.join("config.json"), config.to_string())
        .map_err(|e| tensor::Error(e.to_string()))?;

    let mut seed = shape.seed;
    let mut tensors: Vec<(String, Array)> = Vec::new();
    let mut dense = |name: String, rows: i32, cols: i32, seed: &mut u64| {
        let data = noise(seed, (rows * cols) as usize, 0.3);
        tensors.push((name, Array::from_f32(&data, &[rows, cols])));
    };
    dense("model.embed_tokens.weight".into(), vocab, HIDDEN, &mut seed);
    dense("lm_head.weight".into(), vocab, HIDDEN, &mut seed);
    for i in 0..LAYERS {
        let p = format!("model.layers.{i}");
        dense(
            format!("{p}.self_attn.q_proj.weight"),
            HEADS * HEAD_DIM,
            HIDDEN,
            &mut seed,
        );
        dense(
            format!("{p}.self_attn.k_proj.weight"),
            KV_HEADS * HEAD_DIM,
            HIDDEN,
            &mut seed,
        );
        dense(
            format!("{p}.self_attn.v_proj.weight"),
            KV_HEADS * HEAD_DIM,
            HIDDEN,
            &mut seed,
        );
        dense(
            format!("{p}.self_attn.o_proj.weight"),
            HIDDEN,
            HEADS * HEAD_DIM,
            &mut seed,
        );
        dense(format!("{p}.mlp.gate_proj.weight"), FFN, HIDDEN, &mut seed);
        dense(format!("{p}.mlp.up_proj.weight"), FFN, HIDDEN, &mut seed);
        dense(format!("{p}.mlp.down_proj.weight"), HIDDEN, FFN, &mut seed);
    }
    let mut norm = |name: String, n: i32, seed: &mut u64| {
        let data: Vec<f32> = noise(seed, n as usize, 0.1)
            .iter()
            .map(|x| 1.0 + x)
            .collect();
        tensors.push((name, Array::from_f32(&data, &[n])));
    };
    for i in 0..LAYERS {
        let p = format!("model.layers.{i}");
        norm(format!("{p}.input_layernorm.weight"), HIDDEN, &mut seed);
        norm(
            format!("{p}.post_attention_layernorm.weight"),
            HIDDEN,
            &mut seed,
        );
        norm(format!("{p}.self_attn.q_norm.weight"), HEAD_DIM, &mut seed);
        norm(format!("{p}.self_attn.k_norm.weight"), HEAD_DIM, &mut seed);
    }
    norm("model.norm.weight".into(), HIDDEN, &mut seed);
    let refs: Vec<(&str, &Array)> = tensors.iter().map(|(n, a)| (n.as_str(), a)).collect();
    tensor::save_safetensors(&dir.join("model.safetensors"), &refs)
}

/// A committed tiny checkpoint of this crate (`tests/fixtures/<name>`).
pub fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

/// Loads `dir` on the CPU backend with a 16 GiB device, optionally lowering the budget.
pub fn load_cpu(dir: &Path, budget: Option<u64>) -> Result<MlxEngine, MlxLoadError> {
    load_on(dir, budget, true)
}

fn load_on(dir: &Path, budget: Option<u64>, cpu: bool) -> Result<MlxEngine, MlxLoadError> {
    MlxEngine::load(
        dir,
        LoadOptions {
            identity: dir.display().to_string(),
            cpu,
            fallback_device: Some(DeviceMemory {
                physical: 16 * GIB,
                recommended_working_set: 16 * GIB,
                available: None,
                estimated: false,
            }),
            overrides: Overrides { budget },
            ..LoadOptions::default()
        },
    )
}
