//! Fixtures shared by the prefix-cache suites (003 T007–T009), included with
//! `#[path = "fixtures/support.rs"] mod support;`:
//!
//! - the real Qwen3-8B chat template (`qwen3-8b/chat_template.jinja`, pinned revision in its
//!   header) and the real BPE `tokenizer.json` from the Hugging Face cache ([`qwen3_tokenizer`];
//!   tests skip with a message when it is absent);
//! - [`ScriptHook`]: a `DecodeHook` that masks everything but the next scripted token, so a real
//!   engine produces given replies through the real decode path;
//! - tiny `NuvecEngine` checkpoints (feature `engine`) from `mrun-engine`'s `test-support`.

#![allow(dead_code)]

use mrun::chat::tokenizer::Tokenizer;
use mrun::runtime::generate::DecodeHook;
use mrun_engine::TokenMask;
use std::path::PathBuf;

/// `mlx-community/Qwen3-8B-4bit` at the revision the template was copied from.
pub const QWEN3_REPO: &str = "models--mlx-community--Qwen3-8B-4bit";
pub const QWEN3_REVISION: &str = "545dc4251c05440727734bcd94334791f6ab0192";

/// The real Qwen3-8B chat template.
pub const QWEN3_TEMPLATE: &str = include_str!("qwen3-8b/chat_template.jinja");

/// `tokenizer.json` of Qwen3-8B-4bit: `$MRUN_TEST_MODELS/Qwen3-8B-4bit/tokenizer.json`, else the
/// Hugging Face cache (`$HF_HUB_CACHE`, `$HF_HOME/hub`, `~/.cache/huggingface/hub`) at the
/// pinned revision.
pub fn qwen3_tokenizer_path() -> Option<PathBuf> {
    let mut candidates = Vec::new();
    if let Some(dir) = std::env::var_os("MRUN_TEST_MODELS") {
        candidates.push(PathBuf::from(dir).join("Qwen3-8B-4bit/tokenizer.json"));
    }
    let hub = std::env::var_os("HF_HUB_CACHE")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HF_HOME").map(|h| PathBuf::from(h).join("hub")))
        .or_else(|| {
            std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache/huggingface/hub"))
        });
    if let Some(hub) = hub {
        candidates.push(
            hub.join(QWEN3_REPO)
                .join("snapshots")
                .join(QWEN3_REVISION)
                .join("tokenizer.json"),
        );
    }
    candidates.into_iter().find(|p| p.is_file())
}

/// The snapshot of a downloaded Hugging Face repository (`org/name`): its `main` ref, else any
/// complete one.
pub fn hf_snapshot(repo: &str) -> Option<PathBuf> {
    let hub = std::env::var_os("HF_HUB_CACHE")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HF_HOME").map(|h| PathBuf::from(h).join("hub")))
        .or_else(|| {
            std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache/huggingface/hub"))
        })?;
    let dir = hub.join(format!("models--{}", repo.replace('/', "--")));
    let snapshots = dir.join("snapshots");
    let main = std::fs::read_to_string(dir.join("refs/main"))
        .ok()
        .map(|revision| snapshots.join(revision.trim()));
    // Without a `main` ref (a download by revision), any complete snapshot.
    let any = std::fs::read_dir(&snapshots)
        .ok()?
        .filter_map(Result::ok)
        .map(|e| e.path());
    main.into_iter()
        .chain(any)
        .find(|s| s.join("config.json").is_file() && s.join("tokenizer.json").is_file())
}

/// The real Qwen3 BPE tokenizer, or `None` (with a skip message) when it is not downloaded.
pub fn qwen3_tokenizer() -> Option<Tokenizer> {
    let Some(path) = qwen3_tokenizer_path() else {
        eprintln!(
            "skipped: Qwen3-8B-4bit tokenizer.json not found (set MRUN_TEST_MODELS or run \
             `hf download mlx-community/Qwen3-8B-4bit tokenizer.json --revision {QWEN3_REVISION}`)"
        );
        return None;
    };
    Some(Tokenizer::from_file(&path).expect("load tokenizer.json"))
}

/// `<|im_end|>` in the Qwen3 vocabulary.
pub const IM_END: u32 = 151_645;

/// Masks every token but the next one of `script` (003 T009): the engine computes real logits
/// and samples the scripted token through the ordinary decode path. After the script, only
/// `end` is allowed.
#[derive(Debug)]
pub struct ScriptHook {
    script: Vec<u32>,
    next: usize,
    end: u32,
    vocab: u32,
}

impl ScriptHook {
    pub fn new(script: Vec<u32>, end: u32, vocab: u32) -> Self {
        Self {
            script,
            next: 0,
            end,
            vocab,
        }
    }

    fn mask(&self) -> TokenMask {
        let mut mask = TokenMask::none_allowed(self.vocab);
        mask.allow(self.script.get(self.next).copied().unwrap_or(self.end));
        mask
    }
}

impl DecodeHook for ScriptHook {
    fn first_mask(&mut self) -> Option<TokenMask> {
        Some(self.mask())
    }

    fn on_token(&mut self, id: u32) -> Option<TokenMask> {
        assert_eq!(self.script.get(self.next).copied(), Some(id), "scripted");
        self.next += 1;
        Some(self.mask())
    }
}

#[cfg(feature = "engine")]
pub mod tiny {
    use mrun_engine::NuvecEngine;
    use mrun_engine::testing::{self, TinyQwen3};

    /// A tiny `qwen3` on the CPU with its model directory (kept alive with the engine).
    pub struct Tiny {
        pub engine: NuvecEngine,
        pub dir: Option<tempfile::TempDir>,
    }

    /// Tiny `qwen3` with `vocab` rows; `state_bytes` caps the state capacity.
    pub fn qwen3(vocab: u32, state_bytes: Option<u64>) -> Tiny {
        qwen3_with(TinyQwen3::new(vocab), state_bytes)
    }

    pub fn qwen3_with(shape: TinyQwen3, state_bytes: Option<u64>) -> Tiny {
        let dir = tempfile::tempdir().unwrap();
        testing::write_tiny_qwen3_with(dir.path(), shape).unwrap();
        let engine = load(dir.path(), state_bytes);
        Tiny {
            engine,
            dir: Some(dir),
        }
    }

    /// The committed tiny hybrid (`qwen3_5_moe`: two Gated DeltaNet and two attention layers,
    /// vocabulary 128, 4096 positions).
    pub fn hybrid(state_bytes: Option<u64>) -> Tiny {
        Tiny {
            engine: load(&testing::fixture("tiny_qwen3_5_moe"), state_bytes),
            dir: None,
        }
    }

    /// Loads `dir` on the CPU; with
    /// `state_bytes`, the budget is lowered so the state capacity is about that many bytes
    /// (weights and scratch measured by a first load).
    fn load(dir: &std::path::Path, state_bytes: Option<u64>) -> NuvecEngine {
        use mrun_engine::Engine;
        let engine = testing::load_cpu(dir, None).unwrap();
        let Some(state) = state_bytes else {
            return engine;
        };
        let b = *engine.budget();
        drop(engine);
        // Scratch depends on the capacity; settle on the scratch the smaller budget needs.
        let mut scratch = b.scratch;
        for _ in 0..4 {
            match testing::load_cpu(dir, Some(b.weights + scratch + state)) {
                Ok(engine) => return engine,
                Err(mrun_engine::NuvecLoadError::Budget(
                    mrun_engine::budget::BudgetError::DoesNotFit { scratch: s, .. },
                )) => scratch = s,
                Err(e) => panic!("{e}"),
            }
        }
        panic!("no budget fits {state} bytes of state")
    }
}

/// An engine wrapper that records, for every sampled step, the log-probabilities of the whole
/// vocabulary and the sampled id (the cache-on side of the SC-005 oracle).
pub struct Recorder<E> {
    pub inner: E,
    pub rows: Vec<Vec<f32>>,
    pub sampled: Vec<u32>,
}

impl<E: mrun_engine::Engine> Recorder<E> {
    pub fn new(inner: E) -> Self {
        Self {
            inner,
            rows: Vec::new(),
            sampled: Vec::new(),
        }
    }

    /// The rows and ids recorded since the last call.
    pub fn take(&mut self) -> (Vec<Vec<f32>>, Vec<u32>) {
        (
            std::mem::take(&mut self.rows),
            std::mem::take(&mut self.sampled),
        )
    }
}

/// Log-probabilities of the whole vocabulary at `step`, by token id.
pub fn logprobs(engine: &mut dyn mrun_engine::Engine, step: &mrun_engine::Step) -> Vec<f32> {
    let vocab = engine.info().vocab as usize;
    let mut top = engine.top_logprobs(step, vocab).unwrap();
    top.sort_by_key(|&(id, _)| id);
    top.into_iter().map(|(_, lp)| lp).collect()
}

impl<E: mrun_engine::Engine> mrun_engine::Engine for Recorder<E> {
    fn info(&self) -> &mrun_engine::ModelInfo {
        self.inner.info()
    }
    fn budget(&self) -> &mrun_engine::MemoryBudget {
        self.inner.budget()
    }
    fn len(&self) -> u32 {
        self.inner.len()
    }
    fn allocated(&self) -> u64 {
        self.inner.allocated()
    }
    fn set_limit(&mut self, total: u32) -> Result<(), mrun_engine::EngineError> {
        self.inner.set_limit(total)
    }
    fn prefill_chunk(
        &mut self,
        tokens: &[u32],
        cancel: &mrun_engine::CancelToken,
    ) -> Result<bool, mrun_engine::EngineError> {
        self.inner.prefill_chunk(tokens, cancel)
    }
    fn forward(&mut self, last: u32) -> Result<mrun_engine::Step, mrun_engine::EngineError> {
        self.inner.forward(last)
    }
    fn sample(
        &mut self,
        step: mrun_engine::Step,
        sampling: &mrun_engine::Sampling,
        mask: Option<&TokenMask>,
    ) -> Result<u32, mrun_engine::EngineError> {
        let row = logprobs(&mut self.inner, &step);
        let id = self.inner.sample(step, sampling, mask)?;
        self.rows.push(row);
        self.sampled.push(id);
        Ok(id)
    }
    fn top_logprobs(
        &mut self,
        step: &mrun_engine::Step,
        n: usize,
    ) -> Result<Vec<(u32, f32)>, mrun_engine::EngineError> {
        self.inner.top_logprobs(step, n)
    }
    fn trim(&mut self, len: u32) -> Result<(), mrun_engine::EngineError> {
        self.inner.trim(len)
    }
    fn checkpoint(&mut self) -> Result<mrun_engine::Checkpoint, mrun_engine::EngineError> {
        self.inner.checkpoint()
    }
    fn restore(&mut self, id: mrun_engine::CheckpointId) -> Result<(), mrun_engine::EngineError> {
        self.inner.restore(id)
    }
    fn drop_checkpoint(&mut self, id: mrun_engine::CheckpointId) {
        self.inner.drop_checkpoint(id);
    }
    fn detach_tail(&mut self, at: u32) -> Result<mrun_engine::Segment, mrun_engine::EngineError> {
        self.inner.detach_tail(at)
    }
    fn attach(&mut self, id: mrun_engine::SegmentId) -> Result<(), mrun_engine::EngineError> {
        self.inner.attach(id)
    }
    fn drop_segment(&mut self, id: mrun_engine::SegmentId) {
        self.inner.drop_segment(id);
    }
    fn reset(&mut self) {
        self.inner.reset();
    }
    fn reset_all(&mut self) {
        self.inner.reset_all();
    }
}

/// One request served from the cache, as the oracle needs it.
pub struct Served {
    pub prompt: mrun::chat::prompt::RenderedPrompt,
    /// Where prefill started (the cached tokens).
    pub resume: u32,
    /// The live schedule after the request (its prefix at `resume` is the reused part).
    pub schedule: Vec<mrun::cache::schedule::Span>,
    /// The live tokens after the request.
    pub live: Vec<u32>,
    /// Cache-on log-probabilities and sampled ids, one per sampled step.
    pub rows: Vec<Vec<f32>>,
    pub sampled: Vec<u32>,
    /// The masks' scripted ids, when the request had a [`ScriptHook`].
    pub script: Option<(Vec<u32>, u32)>,
}

/// The exactness oracle (003 SC-005, D049, R10): on `fresh` (an empty engine of the same model),
/// replays the recorded forward schedule of the reused positions (chunks as recorded, earlier
/// replies teacher-forced one token at a time), then the request's own plan
/// (`chunks::plan(info, prompt)` above `resume`), then decodes teacher-forced; every step's
/// log-probabilities and sampled id must equal the cache-on run's bit for bit. Returns `false`
/// when the case cannot be checked (the request kept less than its reused part).
pub fn oracle(fresh: &mut dyn mrun_engine::Engine, served: &Served, what: &str) -> bool {
    use mrun::cache::schedule::{self, Span};
    use mrun::cache::{PrefixStore, chunks};
    use mrun_engine::{CancelToken, Sampling};
    let prompt = &served.prompt;
    let tokens = &prompt.tokens;
    let p = prompt.len();
    if schedule::len(&served.schedule) < served.resume || served.live.len() < served.resume as usize
    {
        return false;
    }
    let reused = schedule::prefix(&served.schedule, &served.live, served.resume);
    assert_eq!(
        &served.live[..served.resume as usize],
        &tokens[..served.resume as usize],
        "{what}: no stale reuse"
    );
    let marks = PrefixStore::checkpoint_marks(fresh, prompt);
    let own = chunks::above(&chunks::plan(fresh.info(), prompt, &marks), served.resume);
    // The store recorded the request's own chunks as planned (FR-005, FR-015).
    if schedule::len(&served.schedule) >= p - 1 {
        let recorded: Vec<u32> = served
            .schedule
            .iter()
            .filter_map(|s| match s {
                Span::Chunk { start, end, .. } if *start >= served.resume && *end < p => Some(*end),
                _ => None,
            })
            .collect();
        assert_eq!(recorded, own, "{what}: recorded chunks follow the plan");
    }
    fresh.reset_all();
    // A cut chunk is replayed whole, then trimmed back.
    let widest = reused
        .iter()
        .map(|s| match s {
            Span::Chunk { end, .. } | Span::Steps { end, .. } => *end,
        })
        .max()
        .unwrap_or(0);
    fresh
        .set_limit((p + served.sampled.len() as u32 + 1).max(widest))
        .unwrap();
    let cancel = CancelToken::new();
    for span in &reused {
        match span {
            Span::Chunk { start, end, tail } => {
                let held = span.held_end();
                let mut feed = tokens[*start as usize..held as usize].to_vec();
                feed.extend_from_slice(tail);
                assert_eq!(feed.len() as u32, end - start);
                assert!(fresh.prefill_chunk(&feed, &cancel).unwrap());
                if !tail.is_empty() {
                    fresh.trim(held).unwrap();
                }
            }
            Span::Steps { start, end } => {
                for pos in *start..*end {
                    fresh.forward(tokens[pos as usize]).unwrap();
                }
            }
        }
    }
    assert_eq!(
        fresh.len(),
        served.resume,
        "{what}: replayed the reused part"
    );
    let mut at = served.resume;
    for end in own {
        assert!(
            fresh
                .prefill_chunk(&tokens[at as usize..end as usize], &cancel)
                .unwrap()
        );
        at = end;
    }
    let mut last = tokens[p as usize - 1];
    let vocab = fresh.info().vocab;
    for (i, (want, &id)) in served.rows.iter().zip(&served.sampled).enumerate() {
        let step = fresh.forward(last).unwrap();
        let got = logprobs(fresh, &step);
        let same = got.len() == want.len()
            && got
                .iter()
                .zip(want)
                .all(|(a, b)| a.to_bits() == b.to_bits());
        if !same {
            let diff = got
                .iter()
                .zip(want)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            panic!("{what}: step {i} differs from the recomputation (max diff {diff})");
        }
        let mask = served.script.as_ref().map(|(script, end)| {
            let mut mask = TokenMask::none_allowed(vocab);
            mask.allow(script.get(i).copied().unwrap_or(*end));
            mask
        });
        let sampled = fresh
            .sample(step, &Sampling::GREEDY, mask.as_ref())
            .unwrap();
        assert_eq!(sampled, id, "{what}: step {i} sampled id");
        last = sampled;
    }
    true
}
