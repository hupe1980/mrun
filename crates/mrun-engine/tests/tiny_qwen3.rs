//! A tiny random-weight `qwen3` checkpoint on the MLX CPU backend: chunked prefill matches
//! token-by-token stepping (001 T013), cancel keeps exactly the whole chunks (T019a), and moves
//! and trims are exact (T033 attention part, SC-008).

#![cfg(feature = "engine")]

use mrun_engine::budget::{DeviceMemory, GIB};
use mrun_engine::tensor::{self, Array};
use mrun_engine::{CancelToken, Engine, EngineError, LoadOptions, MlxEngine, Sampling, TokenMask};
use std::path::Path;

const VOCAB: i32 = 64;
const HIDDEN: i32 = 32;
const HEADS: i32 = 4;
const KV_HEADS: i32 = 2;
const HEAD_DIM: i32 = 8;
const FFN: i32 = 48;
const LAYERS: usize = 2;

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

fn write_model(dir: &Path) {
    let config = serde_json::json!({
        "model_type": "qwen3",
        "hidden_size": HIDDEN,
        "intermediate_size": FFN,
        "num_hidden_layers": LAYERS,
        "num_attention_heads": HEADS,
        "num_key_value_heads": KV_HEADS,
        "head_dim": HEAD_DIM,
        "vocab_size": VOCAB,
        "max_position_embeddings": 512,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "tie_word_embeddings": false,
        "torch_dtype": "float32",
        "eos_token_id": 1
    });
    std::fs::write(dir.join("config.json"), config.to_string()).unwrap();

    let mut seed = 7u64;
    let mut tensors: Vec<(String, Array)> = Vec::new();
    let mut dense = |name: String, rows: i32, cols: i32, seed: &mut u64| {
        let data = noise(seed, (rows * cols) as usize, 0.3);
        tensors.push((name, Array::from_f32(&data, &[rows, cols])));
    };
    dense("model.embed_tokens.weight".into(), VOCAB, HIDDEN, &mut seed);
    dense("lm_head.weight".into(), VOCAB, HIDDEN, &mut seed);
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
    tensor::save_safetensors(&dir.join("model.safetensors"), &refs).unwrap();
}

fn engine() -> (tempfile::TempDir, MlxEngine) {
    let dir = tempfile::tempdir().unwrap();
    write_model(dir.path());
    let engine = MlxEngine::load(
        dir.path(),
        LoadOptions {
            identity: "tiny".into(),
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
    .unwrap();
    (dir, engine)
}

fn tokens() -> Vec<u32> {
    (0..20u32).map(|i| (i * 7 + 3) % VOCAB as u32).collect()
}

/// Log-probabilities of every token at the next position after feeding `last`.
fn next_logprobs(engine: &mut MlxEngine, last: u32) -> Vec<f32> {
    let step = engine.forward(last).unwrap();
    let mut top = engine.top_logprobs(&step, VOCAB as usize).unwrap();
    top.sort_by_key(|&(id, _)| id);
    top.into_iter().map(|(_, lp)| lp).collect()
}

/// Prefills `tokens` in chunks ending at `boundaries` (absolute positions), as the runtime does.
fn prefill(engine: &mut MlxEngine, tokens: &[u32], boundaries: &[u32]) -> u32 {
    let start = engine.len();
    let mut at = 0usize;
    for end in boundaries
        .iter()
        .map(|&b| (b - start) as usize)
        .filter(|&b| b > 0 && b < tokens.len())
        .chain([tokens.len()])
    {
        assert!(
            engine
                .prefill_chunk(&tokens[at..end], &CancelToken::new())
                .unwrap()
        );
        at = end;
    }
    tokens.len() as u32
}

fn max_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

#[test]
fn chunked_prefill_matches_stepping() {
    let t = tokens();
    let (_dir, mut chunked) = engine();
    chunked.set_limit(64).unwrap();
    assert_eq!(prefill(&mut chunked, &t[..19], &[5, 11]), 19);
    let a = next_logprobs(&mut chunked, t[19]);

    let (_dir, mut stepped) = engine();
    stepped.set_limit(64).unwrap();
    for &token in &t[..19] {
        stepped.forward(token).unwrap();
    }
    let b = next_logprobs(&mut stepped, t[19]);
    assert_eq!(chunked.len(), 20);
    assert_eq!(stepped.len(), 20);
    assert!(max_diff(&a, &b) < 1e-3, "max diff {}", max_diff(&a, &b));
}

#[test]
fn cancel_keeps_whole_chunks_and_continues_exactly() {
    let t = tokens();
    let (_dir, mut reference) = engine();
    reference.set_limit(64).unwrap();
    prefill(&mut reference, &t[..19], &[6]);
    let expected = next_logprobs(&mut reference, t[19]);

    let (_dir, mut engine) = engine();
    engine.set_limit(64).unwrap();
    assert_eq!(prefill(&mut engine, &t[..6], &[]), 6);
    let cancel = CancelToken::new();
    cancel.cancel();
    assert!(!engine.prefill_chunk(&t[6..19], &cancel).unwrap());
    assert_eq!(engine.len(), 6, "a cancelled chunk is not committed");
    // Continue with the rest at the same chunk ends: bit-identical to the uninterrupted run.
    assert_eq!(prefill(&mut engine, &t[6..19], &[]), 13);
    assert_eq!(next_logprobs(&mut engine, t[19]), expected);
}

#[test]
fn detach_attach_and_trim_are_exact() {
    let t: Vec<u32> = (0..400u32).map(|i| (i * 7 + 3) % VOCAB as u32).collect();
    let (_dir, mut engine) = engine();
    engine.set_limit(500).unwrap();
    prefill(&mut engine, &t[..399], &[]);
    let expected = next_logprobs(&mut engine, t[399]);
    let full = engine.allocated();

    engine.trim(399).unwrap();
    let segment = engine.detach_tail(10).unwrap();
    assert_eq!((segment.start, segment.end), (10, 399));
    assert_eq!(engine.len(), 10);
    assert!(
        engine.allocated() - segment.bytes < full,
        "the live state shrank to align(10) = 256 positions"
    );
    engine.attach(segment.id).unwrap();
    assert_eq!(engine.len(), 399);
    assert!(matches!(
        engine.attach(segment.id),
        Err(EngineError::UnknownSegment(_))
    ));
    assert_eq!(
        next_logprobs(&mut engine, t[399]),
        expected,
        "after detach + attach"
    );

    // Trimming back and re-feeding the same token reproduces the logits too.
    engine.trim(399).unwrap();
    assert_eq!(next_logprobs(&mut engine, t[399]), expected, "after trim");
}

#[test]
fn sampling_pipelines_the_next_step() {
    let t = tokens();
    let (_dir, mut engine) = engine();
    engine.set_limit(64).unwrap();
    prefill(&mut engine, &t[..9], &[]);
    let step = engine.forward(t[9]).unwrap();
    let next = engine.sample(step, &Sampling::GREEDY, None).unwrap();
    assert_eq!(engine.len(), 11, "the sampled token is already fed");
    // Forwarding that token reuses the pending step; trimming below it rolls it back.
    let step = engine.forward(next).unwrap();
    assert_eq!(engine.len(), 11);
    engine.sample(step, &Sampling::GREEDY, None).unwrap();
    engine.trim(11).unwrap();
    assert_eq!(engine.len(), 11);
}

#[test]
fn masks_narrower_than_the_logits_are_padded() {
    // A tokenizer vocabulary below the model's (padded) `vocab_size`, as on Qwen.
    let t = tokens();
    let (_dir, mut engine) = engine();
    engine.set_limit(64).unwrap();
    prefill(&mut engine, &t[..9], &[]);
    let vocab = engine.info().vocab;
    let mut mask = TokenMask::none_allowed(vocab - 3);
    mask.allow(5);
    let step = engine.forward(t[9]).unwrap();
    assert_eq!(engine.sample(step, &Sampling::GREEDY, Some(&mask)), Ok(5));
    let wide = TokenMask::none_allowed(vocab + 1);
    let step = engine.forward(5).unwrap();
    assert_eq!(
        engine.sample(step, &Sampling::GREEDY, Some(&wide)),
        Err(EngineError::MaskTooWide {
            mask: vocab + 1,
            vocab
        })
    );
}

#[test]
fn a_cancelled_chunk_gives_its_growth_back() {
    let t = tokens();
    let (_dir, mut engine) = engine();
    engine.set_limit(500).unwrap();
    let cancel = CancelToken::new();
    cancel.cancel();
    assert!(!engine.prefill_chunk(&t[..19], &cancel).unwrap());
    assert_eq!((engine.len(), engine.allocated()), (0, 0));
    prefill(&mut engine, &t[..19], &[]);
    let expected = {
        let (_dir, mut fresh) = self::engine();
        fresh.set_limit(500).unwrap();
        prefill(&mut fresh, &t[..19], &[]);
        next_logprobs(&mut fresh, t[19])
    };
    assert_eq!(next_logprobs(&mut engine, t[19]), expected);
}

#[test]
fn admission_and_growth_limits() {
    let (_dir, mut engine) = engine();
    let capacity = engine.budget().kv_capacity_tokens;
    assert_eq!(capacity, 512, "max position bounds the capacity");
    assert!(matches!(
        engine.set_limit(capacity + 1),
        Err(EngineError::Capacity { requested, available }) if requested == capacity + 1 && available == capacity
    ));
    engine.set_limit(8).unwrap();
    assert_eq!(
        engine.allocated(),
        0,
        "nothing is allocated before the first growth"
    );
    let t = tokens();
    assert!(matches!(
        engine.prefill_chunk(&t[..9], &CancelToken::new()),
        Err(EngineError::ReservationExceeded { needed: 9, .. })
    ));
    prefill(&mut engine, &t[..7], &[]);
    assert!(engine.allocated() > 0);
    assert_eq!(engine.len(), 7);
}

#[test]
fn greedy_generation_is_deterministic() {
    let t = tokens();
    let run = || {
        let (_dir, mut engine) = engine();
        engine.set_limit(64).unwrap();
        prefill(&mut engine, &t[..9], &[]);
        let mut last = t[9];
        let mut out = Vec::new();
        for _ in 0..8 {
            let step = engine.forward(last).unwrap();
            last = engine.sample(step, &Sampling::GREEDY, None).unwrap();
            out.push(last);
        }
        out
    };
    assert_eq!(run(), run());
}
