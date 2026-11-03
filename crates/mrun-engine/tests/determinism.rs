//! Run-to-run determinism (spec 016 FR-005, N2): the same request gives bit-identical logits
//! twice in one process and in a second process, on each device. Covered: every tiny family
//! (dense `qwen3`, `qwen3_moe`, the hybrids `qwen3_5` and `qwen3_5_moe`), each prefilled in chunks
//! and then decoded greedily through the engine's pipelined path (`sample` submits the token and
//! the next forward with `async_eval` before reading the token back), plus a dense and an
//! affine-quantized product long enough in K to take the split-K kernels (nuvec 0.4.1:
//! `gemm_splitk` for few output tiles over K ≥ max(M, N); `qmm_splitk` above the vector kernels' row limit; MLX has the
//! same split-K paths).
//!
//! The second process is this test binary itself, running [`child_writes_outputs`] with
//! `MRUN_DETERMINISM_OUT` set; without that variable the child test does nothing.

#![cfg(feature = "engine")]

mod common;

use std::collections::BTreeMap;
use std::path::Path;

use common::{Device, Reference, devices, greedy, load, prefill};
use mrun_engine::budget::{DeviceMemory, GIB};
use mrun_engine::tensor::{self, Array, Dtype};
use mrun_engine::{Engine, LoadOptions, MlxEngine};

const OUT: &str = "MRUN_DETERMINISM_OUT";
const STEPS: usize = 8;

/// Every output, as bits, by `"<device>/<case>"`.
type Outputs = BTreeMap<String, Vec<u32>>;

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

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|v| v.to_bits()).collect()
}

/// A tiny random-weight dense `qwen3` (the same in every process: fixed seed, f32 weights).
fn write_tiny_qwen3(dir: &Path) {
    const VOCAB: i32 = 64;
    const HIDDEN: i32 = 32;
    const HEADS: i32 = 4;
    const KV_HEADS: i32 = 2;
    const HEAD_DIM: i32 = 8;
    const FFN: i32 = 48;
    let config = serde_json::json!({
        "model_type": "qwen3", "hidden_size": HIDDEN, "intermediate_size": FFN,
        "num_hidden_layers": 2, "num_attention_heads": HEADS, "num_key_value_heads": KV_HEADS,
        "head_dim": HEAD_DIM, "vocab_size": VOCAB, "max_position_embeddings": 512,
        "rms_norm_eps": 1e-6, "rope_theta": 10000.0, "tie_word_embeddings": false,
        "torch_dtype": "float32", "eos_token_id": 1
    });
    std::fs::write(dir.join("config.json"), config.to_string()).unwrap();
    let mut seed = 7u64;
    let mut tensors: Vec<(String, Array)> = Vec::new();
    let mut matrix = |name: String, shape: [i32; 2]| {
        let data = noise(&mut seed, (shape[0] * shape[1]) as usize, 0.3);
        tensors.push((name, Array::from_f32(&data, &shape)));
    };
    matrix("model.embed_tokens.weight".into(), [VOCAB, HIDDEN]);
    matrix("lm_head.weight".into(), [VOCAB, HIDDEN]);
    for i in 0..2 {
        let p = format!("model.layers.{i}");
        matrix(
            format!("{p}.self_attn.q_proj.weight"),
            [HEADS * HEAD_DIM, HIDDEN],
        );
        matrix(
            format!("{p}.self_attn.k_proj.weight"),
            [KV_HEADS * HEAD_DIM, HIDDEN],
        );
        matrix(
            format!("{p}.self_attn.v_proj.weight"),
            [KV_HEADS * HEAD_DIM, HIDDEN],
        );
        matrix(
            format!("{p}.self_attn.o_proj.weight"),
            [HIDDEN, HEADS * HEAD_DIM],
        );
        matrix(format!("{p}.mlp.gate_proj.weight"), [FFN, HIDDEN]);
        matrix(format!("{p}.mlp.up_proj.weight"), [FFN, HIDDEN]);
        matrix(format!("{p}.mlp.down_proj.weight"), [HIDDEN, FFN]);
    }
    let mut norm = |name: String, n: i32| {
        let data: Vec<f32> = noise(&mut seed, n as usize, 0.1)
            .iter()
            .map(|x| 1.0 + x)
            .collect();
        tensors.push((name, Array::from_f32(&data, &[n])));
    };
    for i in 0..2 {
        let p = format!("model.layers.{i}");
        norm(format!("{p}.input_layernorm.weight"), HIDDEN);
        norm(format!("{p}.post_attention_layernorm.weight"), HIDDEN);
        norm(format!("{p}.self_attn.q_norm.weight"), HEAD_DIM);
        norm(format!("{p}.self_attn.k_norm.weight"), HEAD_DIM);
    }
    norm("model.norm.weight".into(), HIDDEN);
    let refs: Vec<(&str, &Array)> = tensors.iter().map(|(n, a)| (n.as_str(), a)).collect();
    tensor::save_safetensors(&dir.join("model.safetensors"), &refs).unwrap();
}

fn load_dir(dir: &Path, device: Device) -> MlxEngine {
    MlxEngine::load(
        dir,
        LoadOptions {
            identity: "tiny_qwen3".into(),
            cpu: device == Device::Cpu,
            fallback_device: Some(DeviceMemory {
                physical: 16 * GIB,
                recommended_working_set: 16 * GIB,
                available: None,
                estimated: false,
            }),
            ..LoadOptions::default()
        },
    )
    .unwrap()
}

/// Chunked prefill, then [`STEPS`] pipelined greedy steps: every row's bits and the tokens.
fn decode(out: &mut Outputs, key: &str, mut engine: MlxEngine, prompt: &[u32], ends: &[u32]) {
    engine.set_limit(64).unwrap();
    let (last, body) = prompt.split_last().unwrap();
    prefill(&mut engine, body, ends);
    let (rows, tokens) = greedy(&mut engine, *last, STEPS);
    for (i, row) in rows.iter().enumerate() {
        out.insert(format!("{key}/step{i}"), bits(row));
    }
    out.insert(format!("{key}/tokens"), tokens);
}

/// Products long enough in K for the split-K kernels.
fn products(out: &mut Outputs, key: &str) {
    let mut seed = 11u64;
    // Dense: 2×4 output tiles over K = 4096 (batch 1, K ≥ max(M, N)).
    let x = Array::from_f32(&noise(&mut seed, 32 * 4096, 1.0), &[32, 4096]);
    let w = Array::from_f32(&noise(&mut seed, 4096 * 64, 0.05), &[4096, 64]);
    for dtype in [Dtype::Float32, Dtype::Bfloat16] {
        let (xd, wd) = (
            tensor::astype(&x, dtype).unwrap(),
            tensor::astype(&w, dtype).unwrap(),
        );
        let y = tensor::astype(&tensor::matmul(&xd, &wd).unwrap(), Dtype::Float32).unwrap();
        out.insert(
            format!("{key}/matmul_splitk_{dtype:?}"),
            bits(&y.to_vec_f32().unwrap()),
        );
    }
    // Affine 4-bit, group 64: 40 rows over K = 4096, above every GPU generation's vector-kernel
    // row limit (nuvec's `qmv_batch_limit`, at most 25 for this shape), so `qmm_splitk` runs.
    let (n, k) = (256i32, 4096i32);
    let packed: Vec<u32> = (0..(n * k / 8) as u32)
        .map(|i| i.wrapping_mul(2_654_435_761).rotate_left(i % 32))
        .collect();
    let wq = Array::from_u32(&packed, &[n, k / 8]);
    let groups = (n * k / 64) as usize;
    let scales = Array::from_f32(&noise(&mut seed, groups, 0.02), &[n, k / 64]);
    let biases = Array::from_f32(&noise(&mut seed, groups, 0.1), &[n, k / 64]);
    let xq = Array::from_f32(&noise(&mut seed, 40 * k as usize, 1.0), &[40, k]);
    let (scales, biases, xq) = (
        tensor::astype(&scales, Dtype::Bfloat16).unwrap(),
        tensor::astype(&biases, Dtype::Bfloat16).unwrap(),
        tensor::astype(&xq, Dtype::Bfloat16).unwrap(),
    );
    let wq = tensor::Quantized::new(wq, scales, Some(biases), 64, 4).unwrap();
    let y = tensor::quantized_matmul(&xq, &wq).unwrap();
    let y = tensor::astype(&y, Dtype::Float32).unwrap();
    out.insert(
        format!("{key}/qmm_splitk_bf16"),
        bits(&y.to_vec_f32().unwrap()),
    );
}

/// All cases on `device`, on a thread of its own (streams are per thread).
fn run_on(device: Device) -> Outputs {
    std::thread::spawn(move || {
        let key = format!("{device:?}");
        let mut out = Outputs::new();
        products(&mut out, &key);
        let dir = tempfile::tempdir().unwrap();
        write_tiny_qwen3(dir.path());
        let prompt: Vec<u32> = (0..20u32).map(|i| (i * 7 + 3) % 64).collect();
        let engine = load_dir(dir.path(), device);
        decode(
            &mut out,
            &format!("{key}/tiny_qwen3"),
            engine,
            &prompt,
            &[5, 11],
        );
        for name in ["tiny_qwen3_moe", "tiny_qwen3_5", "tiny_qwen3_5_moe"] {
            let reference = Reference::read(name);
            decode(
                &mut out,
                &format!("{key}/{name}"),
                load(name, device),
                &reference.prompt,
                &reference.chunk_ends,
            );
        }
        out
    })
    .join()
    .unwrap_or_else(|_| panic!("failed on {device:?}"))
}

fn run_all() -> Outputs {
    devices().into_iter().flat_map(run_on).collect()
}

/// The names of the first outputs of `a` that differ from `b`'s, with the first differing bit.
fn differences(a: &Outputs, b: &Outputs) -> Vec<String> {
    assert_eq!(
        a.keys().collect::<Vec<_>>(),
        b.keys().collect::<Vec<_>>(),
        "the same cases"
    );
    a.iter()
        .filter(|(key, bits)| b[*key] != **bits)
        .map(|(key, bits)| {
            let at = bits.iter().zip(&b[key]).position(|(x, y)| x != y);
            format!("{key} (first difference at element {at:?})")
        })
        .collect()
}

/// The second process: writes [`run_all`]'s outputs to `MRUN_DETERMINISM_OUT` (nothing without it).
#[test]
fn child_writes_outputs() {
    let Ok(path) = std::env::var(OUT) else {
        return;
    };
    std::fs::write(path, serde_json::to_vec(&run_all()).unwrap()).unwrap();
}

#[test]
fn the_same_request_gives_the_same_bits_in_one_process_and_in_two() {
    if std::env::var_os(OUT).is_some() {
        return;
    }
    let first = run_all();
    let second = run_all();
    let differ = differences(&first, &second);
    assert!(differ.is_empty(), "in one process: {differ:#?}");

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("child.json");
    let status = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["--exact", "child_writes_outputs", "--test-threads", "1"])
        .env(OUT, &path)
        .status()
        .unwrap();
    assert!(status.success(), "child process: {status}");
    let child: Outputs = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    let differ = differences(&first, &child);
    assert!(differ.is_empty(), "across processes: {differ:#?}");
    eprintln!(
        "{} outputs bit-identical in one process and across processes",
        first.len()
    );
}
