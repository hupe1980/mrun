//! Shared helpers for the tiny-fixture tests: loading a checkpoint from `tests/fixtures/` on the
//! CPU or on Metal, the mlx-lm reference outputs written by `fixtures/gen_tiny.py`, and the
//! prefill/step drivers the runtime would use.

#![allow(dead_code)]

use std::path::{Path, PathBuf};

use mrun_engine::budget::{DeviceMemory, GIB};
use mrun_engine::tensor;
use mrun_engine::{CancelToken, Engine, LoadOptions, NuvecEngine, Sampling};
use serde::Deserialize;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Device {
    Cpu,
    Gpu,
}

/// The CPU, plus Metal when present; `MRUN_TEST_DEVICE=cpu|gpu` keeps one (to tell a backend's
/// CPU path from its GPU path).
pub fn devices() -> Vec<Device> {
    let mut devices = vec![Device::Cpu];
    if tensor::metal_available() {
        devices.push(Device::Gpu);
    }
    match std::env::var("MRUN_TEST_DEVICE").as_deref() {
        Ok("cpu") => devices.retain(|d| *d == Device::Cpu),
        Ok("gpu") => devices.retain(|d| *d == Device::Gpu),
        _ => {}
    }
    devices
}

/// Runs `test` once per device, each on its own thread (nuvec streams are per thread).
pub fn on_each_device(test: fn(Device)) {
    for device in devices() {
        std::thread::spawn(move || test(device))
            .join()
            .unwrap_or_else(|_| panic!("failed on {device:?}"));
    }
}

pub fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

pub fn load(name: &str, device: Device) -> NuvecEngine {
    NuvecEngine::load(
        &fixture(name),
        LoadOptions {
            identity: name.into(),
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
    .unwrap_or_else(|e| panic!("{name}: {e}"))
}

/// Outputs of one implementation on one device: mlx-lm's (recorded) or this engine's own fresh
/// run (see [`Reference::run`]).
#[derive(Clone, Debug, Deserialize)]
pub struct Run {
    /// Log-probabilities after the last prompt token and after each greedy token.
    pub single: Vec<Vec<f32>>,
    pub greedy: Vec<u32>,
    pub chunked: Vec<f32>,
    pub stepped: Vec<f32>,
}

#[derive(Debug, Deserialize)]
pub struct Reference {
    pub prompt: Vec<u32>,
    pub chunk_ends: Vec<u32>,
    cpu: Run,
    gpu: Option<Run>,
    /// The fixture's name (set by [`Reference::read`]).
    #[serde(skip)]
    name: String,
    /// This engine's own runs, per device (computed on first use).
    #[serde(skip)]
    own: [std::sync::OnceLock<Run>; 2],
}

/// Largest log-probability difference allowed between this engine and mlx-lm's recorded rows
/// (spec 016 R10: nuvec rounds bf16 element-wise ops once in f32, MLX per step; measured on the
/// tiny fixtures: ≤ 0.34 on the GPU, ≤ 0.63 on the CPU).
pub const NUVEC_MLX_TOLERANCE: f32 = 1.0;

impl Reference {
    pub fn read(name: &str) -> Self {
        let text = std::fs::read_to_string(fixture(name).join("reference.json")).unwrap();
        let mut reference: Self = serde_json::from_str(&text).unwrap();
        reference.name = name.to_string();
        reference
    }

    /// mlx-lm's recorded outputs on `device` (`fixtures/gen_tiny.py`).
    pub fn recorded(&self, device: Device) -> &Run {
        match device {
            Device::Cpu => &self.cpu,
            Device::Gpu => self.gpu.as_ref().expect("reference has no GPU run"),
        }
    }

    /// The outputs the exactness checks compare against bitwise: this engine's own fresh runs on
    /// `device` (single-chunk prefill and greedy steps, the chunked prefill, token by token),
    /// because bitwise equality with MLX's rounding is not available (spec 016 R10); the
    /// comparison with mlx-lm is [`assert_parity`]'s tolerance check, never a precondition of
    /// these.
    pub fn run(&self, device: Device) -> &Run {
        let slot = &self.own[usize::from(device == Device::Gpu)];
        slot.get_or_init(|| self.own_run(device))
    }

    /// This engine's runs of the fixture, built as `gen_tiny.py` builds mlx-lm's.
    fn own_run(&self, device: Device) -> Run {
        let (body, last) = (self.body(), self.last());
        let steps = self.recorded(device).greedy.len();
        let fresh = || {
            let mut engine = load(&self.name, device);
            engine.set_limit(64).unwrap();
            engine
        };
        let mut engine = fresh();
        prefill(&mut engine, body, &[]);
        let (single, greedy) = greedy(&mut engine, last, steps);
        let mut engine = fresh();
        prefill(&mut engine, body, &self.chunk_ends);
        let chunked = next(&mut engine, last);
        let mut engine = fresh();
        for &token in body {
            engine.forward(token).unwrap();
        }
        let stepped = next(&mut engine, last);
        Run {
            single,
            greedy,
            chunked,
            stepped,
        }
    }

    /// The prompt without its last token, which `forward` feeds.
    pub fn body(&self) -> &[u32] {
        &self.prompt[..self.prompt.len() - 1]
    }

    pub fn last(&self) -> u32 {
        self.prompt[self.prompt.len() - 1]
    }
}

/// Prefills `tokens` at `len()` in chunks ending before the absolute positions `ends`.
pub fn prefill(engine: &mut NuvecEngine, tokens: &[u32], ends: &[u32]) {
    let start = engine.len();
    let mut at = 0usize;
    for end in ends
        .iter()
        .filter(|&&e| e > start)
        .map(|&e| (e - start) as usize)
        .filter(|&e| e < tokens.len())
        .chain([tokens.len()])
    {
        assert!(
            engine
                .prefill_chunk(&tokens[at..end], &CancelToken::new())
                .unwrap()
        );
        at = end;
    }
}

/// Log-probabilities of the whole vocabulary at the current step, by token id.
pub fn logprobs(engine: &mut NuvecEngine, step: &mrun_engine::Step) -> Vec<f32> {
    let vocab = engine.info().vocab as usize;
    let mut top = engine.top_logprobs(step, vocab).unwrap();
    top.sort_by_key(|&(id, _)| id);
    top.into_iter().map(|(_, lp)| lp).collect()
}

/// Feeds `token` and returns the next position's log-probabilities.
pub fn next(engine: &mut NuvecEngine, token: u32) -> Vec<f32> {
    let step = engine.forward(token).unwrap();
    logprobs(engine, &step)
}

/// Feeds `token`, then samples `n` greedy tokens (pipelined); returns each step's
/// log-probabilities and the tokens.
pub fn greedy(engine: &mut NuvecEngine, token: u32, n: usize) -> (Vec<Vec<f32>>, Vec<u32>) {
    let mut rows = Vec::new();
    let mut tokens = Vec::new();
    let mut step = engine.forward(token).unwrap();
    for i in 0..=n {
        rows.push(logprobs(engine, &step));
        if i == n {
            break;
        }
        let next = engine.sample(step, &Sampling::GREEDY, None).unwrap();
        tokens.push(next);
        step = engine.forward(next).unwrap();
    }
    (rows, tokens)
}

pub fn max_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

/// Bitwise equality of two log-probability rows, with the largest difference on failure.
pub fn assert_bitwise(got: &[f32], want: &[f32], what: &str) {
    let same = got.len() == want.len()
        && got
            .iter()
            .zip(want)
            .all(|(a, b)| a.to_bits() == b.to_bits());
    assert!(same, "{what}: max diff {}", max_diff(got, want));
}

/// mlx-lm parity on `device`: the same greedy tokens and each row within
/// [`NUVEC_MLX_TOLERANCE`] of mlx-lm's recorded rows.
pub fn assert_parity(name: &str, device: Device) {
    assert_parity_within_tolerance(name, &Reference::read(name), device);
}

/// Real models named by `MRUN_TEST_MODELS` (comma-separated aliases `qwen3`, `qwen3.5`,
/// `qwen3.6`, Hugging Face repo ids or directories), resolved in the Hugging Face cache; each
/// with its reference name (the repo's last path part). Panics when the variable is unset or a
/// named model is missing, so an `--ignored` run never passes without running anything (T072).
pub fn real_models() -> Vec<(String, PathBuf)> {
    let list = std::env::var("MRUN_TEST_MODELS")
        .expect("set MRUN_TEST_MODELS (e.g. qwen3,qwen3.5) to run the real-model tests");
    list.split(',')
        .map(str::trim)
        .filter(|m| !m.is_empty())
        .map(|model| {
            let repo = match model {
                "qwen3" => "mlx-community/Qwen3-8B-4bit",
                "qwen3.5" => "mlx-community/Qwen3.5-9B-4bit",
                "qwen3.6" => "mlx-community/Qwen3.6-35B-A3B-4bit",
                other => other,
            };
            let dir = Path::new(repo);
            if dir.join("config.json").is_file() {
                // A Hugging Face cache snapshot names its repo (`models--org--name/...`).
                let name = dir
                    .components()
                    .filter_map(|c| c.as_os_str().to_str())
                    .find_map(|c| c.strip_prefix("models--"))
                    .and_then(|r| r.rsplit("--").next())
                    .map_or_else(
                        || dir.file_name().unwrap().to_string_lossy().into_owned(),
                        str::to_string,
                    );
                return (name, dir.to_path_buf());
            }
            let dir = hf_snapshot(repo).unwrap_or_else(|| panic!("{repo} is not downloaded"));
            (repo.rsplit('/').next().unwrap().to_string(), dir)
        })
        .collect()
}

/// The snapshot of `repo` in the Hugging Face cache (`refs/main`, else the only snapshot).
pub fn hf_snapshot(repo: &str) -> Option<PathBuf> {
    let hub = std::env::var_os("HF_HUB_CACHE")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HF_HOME").map(|h| PathBuf::from(h).join("hub")))
        .or_else(|| {
            std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache/huggingface/hub"))
        })?;
    let root = hub.join(format!("models--{}", repo.replace('/', "--")));
    let snapshots = root.join("snapshots");
    if let Ok(main) = std::fs::read_to_string(root.join("refs/main")) {
        let dir = snapshots.join(main.trim());
        if dir.join("config.json").is_file() {
            return Some(dir);
        }
    }
    std::fs::read_dir(&snapshots)
        .ok()?
        .filter_map(Result::ok)
        .map(|e| e.path())
        .find(|p| p.join("config.json").is_file())
}

/// Loads a real model on Metal with mrun's own calibration file (the chunk plan mrun uses).
pub fn load_real(dir: &Path) -> NuvecEngine {
    let identity = mrun_engine::identity::identify(
        dir,
        &mrun_engine::calibration::default_path()
            .expect("home directory")
            .with_file_name("identity.toml"),
    )
    .expect("identity")
    .value;
    NuvecEngine::load(
        dir,
        LoadOptions {
            identity,
            calibration: mrun_engine::calibration::default_path(),
            ..LoadOptions::default()
        },
    )
    .unwrap_or_else(|e| panic!("{}: {e}", dir.display()))
}

/// Prefills `tokens` at `len()` in the engine's own chunk plan (`ModelInfo::chunk_len`), or in
/// fixed chunks of `split` tokens.
pub fn prefill_plan(engine: &mut NuvecEngine, tokens: &[u32], split: Option<u32>) {
    let mut at = 0usize;
    while at < tokens.len() {
        let len = split.unwrap_or_else(|| engine.info().chunk_len(engine.len())) as usize;
        let end = (at + len).min(tokens.len());
        assert!(
            engine
                .prefill_chunk(&tokens[at..end], &CancelToken::new())
                .unwrap()
        );
        at = end;
    }
}

/// nuvec against mlx-lm's recorded rows: the same greedy tokens, every row within
/// [`NUVEC_MLX_TOLERANCE`]; prints the largest difference per path for the research record.
fn assert_parity_within_tolerance(name: &str, reference: &Reference, device: Device) {
    let (own, recorded) = (reference.run(device), reference.recorded(device));
    // Greedy tokens agree until a near-tie (spec 016 US2 AS3): where they first differ, mlx-lm's
    // own row must rank the two tokens within the tolerance; later rows follow different tokens
    // and are not compared.
    let diverged = own
        .greedy
        .iter()
        .zip(&recorded.greedy)
        .position(|(a, b)| a != b);
    if let Some(step) = diverged {
        let row = &recorded.single[step];
        let (theirs, ours) = (recorded.greedy[step], own.greedy[step]);
        let gap = row[theirs as usize] - row[ours as usize];
        eprintln!(
            "{name} {device:?}: greedy token {step} differs at a gap of {gap} in mlx-lm's row"
        );
        assert!(
            gap <= NUVEC_MLX_TOLERANCE,
            "{name} {device:?}: greedy token {step} differs ({ours} vs {theirs}) at a gap of \
             {gap}, not a near-tie"
        );
    }
    let compared = diverged.map_or(own.single.len(), |step| step + 1);
    let single = own.single[..compared]
        .iter()
        .zip(&recorded.single)
        .map(|(a, b)| max_diff(a, b))
        .fold(0.0, f32::max);
    let chunked = max_diff(&own.chunked, &recorded.chunked);
    let stepped = max_diff(&own.stepped, &recorded.stepped);
    eprintln!(
        "{name} {device:?} vs mlx-lm: max diff single {single}, chunked {chunked}, stepped {stepped}"
    );
    for (path, diff) in [
        ("single", single),
        ("chunked", chunked),
        ("stepped", stepped),
    ] {
        assert!(
            diff <= NUVEC_MLX_TOLERANCE,
            "{name} {device:?}: {path} differs from mlx-lm by {diff} (> {NUVEC_MLX_TOLERANCE})"
        );
    }
}
