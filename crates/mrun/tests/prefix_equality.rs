//! Exactness oracle (003 SC-005, D049, R10; T030): a request served from the cache gets
//! bit-identical output — next-token log-probabilities over the whole vocabulary and sampled ids
//! — to recomputing its prompt from empty state with the same chunk plan. For every request the
//! oracle replays, on a fresh engine of the same model, the forward schedule the store recorded
//! for the reused positions (FR-015; earlier replies teacher-forced one token at a time) and the
//! request's own plan (`chunks::plan(info, prompt)` above the resume position, FR-005), then decodes
//! teacher-forced. One request at a time; this does not compare with `--no-cache` (D049).
//!
//! Cases: appends, a subagent and a title request between main turns (holder chains),
//! interleaved conversations, an edit that cuts the trunk inside a chunk, rewinds, identical
//! retries, cancels and their retries, spliced replies (the next prompt carries the model's own
//! ids), and on the hybrid checkpoint restores — on tiny `qwen3` and the tiny `qwen3_5_moe`
//! hybrid (on the CPU).

#![cfg(feature = "engine")]

#[path = "fixtures/support.rs"]
mod support;

use mrun::cache::{CacheMode, PrefixStore};
use mrun::chat::prompt::{MessageMark, RenderedPrompt, Role};
use mrun::chat::tokenizer::test_support::byte_tokenizer;
use mrun::runtime::generate::{Event, EventSink, FinishReason, GenerateRequest, generate};
use mrun::runtime::pressure::PressureGauge;
use mrun_engine::{Engine, NuvecEngine};
use support::{Recorder, Served, oracle, tiny};

/// A conversation under construction: tokens and message marks.
#[derive(Clone)]
struct Conv {
    tokens: Vec<u32>,
    marks: Vec<MessageMark>,
    system_tools_end: u32,
    seed: u64,
    /// Ordinary token ids are drawn from `2..top` (1 is the tiny models' EOS).
    top: u32,
}

impl Conv {
    fn new(seed: u64, system: usize, top: u32) -> Self {
        let mut conv = Self {
            tokens: Vec::new(),
            marks: Vec::new(),
            system_tools_end: 0,
            seed,
            top,
        };
        conv.push(Role::System, system);
        conv.system_tools_end = conv.tokens.len() as u32;
        conv
    }

    fn draw(&mut self) -> u32 {
        self.seed = self
            .seed
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        2 + ((self.seed >> 33) % u64::from(self.top - 2)) as u32
    }

    fn push(&mut self, role: Role, n: usize) {
        self.marks.push(MessageMark {
            start: self.tokens.len() as u32,
            role,
            message: Some(self.marks.len() as u32),
        });
        for _ in 0..n {
            let t = self.draw();
            self.tokens.push(t);
        }
    }

    fn user(mut self, n: usize) -> Self {
        self.push(Role::User, n);
        self
    }

    fn tool(mut self, n: usize) -> Self {
        self.push(Role::Tool, n);
        self
    }

    /// Appends the reply the cache holds after `prompt` (the model's own ids: a splice).
    fn reply(mut self, live: &[u32]) -> Self {
        let start = self.tokens.len();
        assert_eq!(
            &live[..start],
            &self.tokens[..],
            "the reply follows this prompt"
        );
        self.marks.push(MessageMark {
            start: start as u32,
            role: Role::Assistant,
            message: Some(self.marks.len() as u32),
        });
        self.tokens.extend_from_slice(&live[start..]);
        self
    }

    /// Replaces token `at` (an edit).
    fn edit(mut self, at: usize) -> Self {
        let t = self.tokens[at];
        self.tokens[at] = if t + 1 < self.top { t + 1 } else { 2 };
        self
    }

    /// The first `n` tokens (a rewind).
    fn cut(mut self, n: usize) -> Self {
        self.tokens.truncate(n);
        self.marks.retain(|m| (m.start as usize) < n);
        self
    }

    fn prompt(&self) -> RenderedPrompt {
        RenderedPrompt {
            tokens: self.tokens.clone(),
            messages: self.marks.clone(),
            system_tools_end: self.system_tools_end,
            generation_start: self.tokens.len() as u32,
        }
    }
}

/// Cancels the request at its `n`-th progress event (after that many chunks).
struct CancelAfter(usize);

impl EventSink for CancelAfter {
    fn send(&mut self, event: Event) -> bool {
        if matches!(event, Event::Progress { .. }) {
            self.0 = self.0.saturating_sub(1);
            return self.0 > 0;
        }
        true
    }
}

struct Bench {
    engine: Recorder<NuvecEngine>,
    /// An empty engine of the same model; `None`: the oracle runs on the served engine after
    /// the scenario (real models, one copy of the weights).
    fresh: Option<NuvecEngine>,
    /// Cases waiting for the oracle when there is no second engine.
    pending: Vec<(Served, String)>,
    store: PrefixStore,
    checked: usize,
    /// Resume position of the last request.
    resume: u32,
    /// The last checked request, for the negative control.
    last: Option<Served>,
    tokenizer: mrun::chat::tokenizer::Tokenizer,
}

impl Bench {
    fn new(engine: NuvecEngine, fresh: NuvecEngine) -> Self {
        Self::with_tokenizer(engine, Some(fresh), byte_tokenizer())
    }

    fn with_tokenizer(
        engine: NuvecEngine,
        fresh: Option<NuvecEngine>,
        tokenizer: mrun::chat::tokenizer::Tokenizer,
    ) -> Self {
        Self {
            pending: Vec::new(),
            engine: Recorder::new(engine),
            fresh,
            store: PrefixStore::new(CacheMode::Cached),
            checked: 0,
            resume: 0,
            last: None,
            tokenizer,
        }
    }

    /// Serves `conv` (cancelled after `cancel_after` chunks when set) and checks it against the
    /// oracle. Returns the live tokens.
    fn serve(&mut self, conv: &Conv, cancel_after: Option<usize>, what: &str) -> Vec<u32> {
        let prompt = conv.prompt();
        let mut events = Vec::new();
        let mut cancel = cancel_after.map(CancelAfter);
        let sink: &mut dyn EventSink = match &mut cancel {
            Some(c) => c,
            None => &mut events,
        };
        let finished = generate(
            &mut self.engine,
            &mut self.store,
            &mut self.tokenizer.stream(),
            GenerateRequest::new(prompt.clone(), 6),
            sink,
            &PressureGauge::default(),
        );
        assert!(
            !matches!(finished.reason, FinishReason::Error(_)),
            "{what}: {:?}",
            finished.reason
        );
        let (rows, sampled) = self.engine.take();
        assert_eq!(self.engine.len(), self.store.live_tokens().len() as u32);
        self.store.assert_invariants(&self.engine);
        let live = self.store.live_tokens().to_vec();
        self.resume = finished.stats.cached_tokens;
        if cancel_after.is_some() {
            assert_eq!(finished.reason, FinishReason::Cancelled, "{what}");
            return live;
        }
        let served = Served {
            prompt,
            resume: finished.stats.cached_tokens,
            schedule: self.store.schedule().to_vec(),
            live: live.clone(),
            rows,
            sampled,
            script: None,
        };
        assert!(!served.rows.is_empty(), "{what}: decoded");
        match &mut self.fresh {
            Some(fresh) => {
                if oracle(fresh, &served, what) {
                    self.checked += 1;
                }
                self.last = Some(served);
            }
            None => self.pending.push((served, what.to_string())),
        }
        live
    }
}

#[test]
fn cache_hits_are_bit_identical_to_recomputation_on_tiny_qwen3() {
    let model = tiny::qwen3(64, None);
    let fresh = mrun_engine::testing::load_cpu(model.dir.as_ref().unwrap().path(), None).unwrap();
    let mut b = Bench::new(model.engine, fresh);
    attention_cases(&mut b, 64);
    assert_eq!(b.checked, 21, "every served request was checked");
    eprintln!("tiny qwen3: {} requests bit-identical", b.checked);
}

impl Bench {
    /// Runs the deferred oracle cases on the served engine, emptied.
    fn finish(&mut self) {
        if self.pending.is_empty() {
            return;
        }
        self.store.clear(&mut self.engine);
        for (served, what) in std::mem::take(&mut self.pending) {
            if oracle(&mut self.engine.inner, &served, &what) {
                self.checked += 1;
            }
        }
    }
}

/// A downloaded model on Metal, with its own tokenizer; `None` (skipped) when absent, unless
/// `MRUN_REQUIRE_MODELS=1`, where a missing model fails the test instead of passing silently.
fn real_model(repo: &str) -> Option<Bench> {
    let Some(dir) = support::hf_snapshot(repo) else {
        assert!(
            std::env::var("MRUN_REQUIRE_MODELS").as_deref() != Ok("1"),
            "{repo} is not downloaded and MRUN_REQUIRE_MODELS=1"
        );
        return None;
    };
    let tokenizer =
        mrun::chat::tokenizer::Tokenizer::from_file(&dir.join("tokenizer.json")).ok()?;
    let engine = NuvecEngine::load(
        &dir,
        mrun_engine::LoadOptions {
            identity: repo.into(),
            ..mrun_engine::LoadOptions::default()
        },
    )
    .unwrap();
    Some(Bench::with_tokenizer(engine, None, tokenizer))
}

/// Qwen3.5-9B-4bit (hybrid) on Metal (macOS, downloaded; `--ignored`).
#[test]
#[ignore = "needs Qwen3.5-9B-4bit in the Hugging Face cache and Metal"]
fn cache_hits_are_bit_identical_to_recomputation_on_qwen3_5() {
    let Some(mut b) = real_model("mlx-community/Qwen3.5-9B-4bit") else {
        eprintln!("skipped: Qwen3.5-9B-4bit not downloaded");
        return;
    };
    hybrid_cases(&mut b, 150_000);
    b.finish();
    assert_eq!(b.checked, 10, "every served request was checked");
}

/// Qwen3.6-35B-A3B-4bit (hybrid MoE) on Metal (macOS, downloaded; `--ignored`).
#[test]
#[ignore = "needs Qwen3.6-35B-A3B-4bit in the Hugging Face cache and Metal"]
fn cache_hits_are_bit_identical_to_recomputation_on_qwen3_6() {
    let Some(mut b) = real_model("mlx-community/Qwen3.6-35B-A3B-4bit") else {
        eprintln!("skipped: Qwen3.6-35B-A3B-4bit not downloaded");
        return;
    };
    hybrid_cases(&mut b, 150_000);
    b.finish();
    assert_eq!(b.checked, 10, "every served request was checked");
}

/// Qwen3-8B-4bit on Metal (macOS, downloaded; `--ignored`).
#[test]
#[ignore = "needs Qwen3-8B-4bit in the Hugging Face cache and Metal"]
fn cache_hits_are_bit_identical_to_recomputation_on_qwen3_8b() {
    let Some(tokenizer) = support::qwen3_tokenizer() else {
        return;
    };
    let dir = support::qwen3_tokenizer_path()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf();
    let load = || {
        NuvecEngine::load(
            &dir,
            mrun_engine::LoadOptions {
                identity: "qwen3-8b".into(),
                ..mrun_engine::LoadOptions::default()
            },
        )
        .unwrap()
    };
    let mut b = Bench::with_tokenizer(load(), None, tokenizer);
    // Ordinary (non-special) ids of the Qwen3 vocabulary.
    attention_cases(&mut b, 151_000);
    b.finish();
    assert_eq!(b.checked, 21, "every served request was checked");
}

/// Appends, switches, interleaving, a retry, an edit inside a chunk, a rewind, a cancel and its
/// retry, on an attention-only model whose ordinary token ids are `2..top`.
fn attention_cases(b: &mut Bench, top: u32) {
    // Main: five turns, each prompt the previous one with the model's reply spliced in.
    let mut main = Conv::new(11, 300, top).user(500);
    let mut live = b.serve(&main, None, "main 1");
    for turn in 2..=5 {
        main = main.reply(&live).tool(350 + 40 * turn);
        live = b.serve(&main, None, &format!("main {turn}"));
    }
    // A subagent with its own system prompt, then a title request sharing main's system prompt,
    // then main again through the holder chain.
    let sub = Conv::new(12, 280, top).user(400);
    let sub_live = b.serve(&sub, None, "subagent 1");
    let sub = sub.reply(&sub_live).tool(300);
    b.serve(&sub, None, "subagent 2");
    let title = main.clone().cut(300).user(60);
    b.serve(&title, None, "title");
    main = main.reply(&live).user(200);
    live = b.serve(&main, None, "main back");
    // Interleaved conversations.
    let mut other = Conv::new(13, 320, top).user(300);
    for round in 0..2 {
        let other_live = b.serve(&other, None, &format!("other {round}"));
        other = other.reply(&other_live).user(280);
        main = main.reply(&live).tool(260);
        live = b.serve(&main, None, &format!("main interleaved {round}"));
    }
    // An identical retry resumes at P − 1.
    b.serve(&main, None, "retry");
    assert_eq!(b.resume, main.prompt().len() - 1);
    // An edit inside the last chunk cuts the trunk inside it; the edited conversation goes on.
    let at = main.tokens.len() - 120;
    let edited = main.clone().edit(at);
    let edited_live = b.serve(&edited, None, "edit");
    let edited = edited.reply(&edited_live).user(300);
    b.serve(&edited, None, "after the edit");
    // A rewind to an earlier point of main, then main's own continuation.
    let rewind = main.clone().cut(1500).user(100);
    b.serve(&rewind, None, "rewind");
    main = main.reply(&live).user(150);
    live = b.serve(&main, None, "main after rewind");
    // A long new prompt cancelled after its first chunk, then retried: it resumes at the whole
    // chunk the cancel kept.
    let long = Conv::new(14, 300, top).user(2600);
    b.serve(&long, Some(1), "cancel");
    b.serve(&long, None, "after cancel");
    assert_eq!(b.resume, 300, "resumed at the whole chunk the cancel kept");
    b.serve(
        &long.clone().reply(b.store.live_tokens()).user(90),
        None,
        "long 2",
    );
    // Main once more, after everything.
    main = main.reply(&live).tool(400);
    b.serve(&main, None, "main last");
}

#[test]
fn cache_hits_are_bit_identical_to_recomputation_on_the_tiny_hybrid() {
    let model = tiny::hybrid(None);
    let fresh = tiny::hybrid(None).engine;
    let mut b = Bench::new(model.engine, fresh);
    // Token ids below 94 decode to ASCII with the byte tokenizer.
    hybrid_cases(&mut b, 94);
    assert_eq!(b.checked, 10, "every served request was checked");
    eprintln!("tiny hybrid: {} requests bit-identical", b.checked);
}

/// Appends, an identical retry and an edit (checkpoint restores), a subagent and a title
/// request, a cancel and its retry, on a hybrid model whose ordinary ids are `2..top`.
fn hybrid_cases(b: &mut Bench, top: u32) {
    let mut main = Conv::new(21, 280, top).user(400);
    let mut live = b.serve(&main, None, "hybrid main 1");
    for turn in 2..=3 {
        main = main.reply(&live).tool(300);
        live = b.serve(&main, None, &format!("hybrid main {turn}"));
    }
    // An identical retry restores the checkpoint at P − 1.
    b.serve(&main, None, "hybrid retry");
    assert_eq!(b.resume, main.prompt().len() - 1);
    // An edit of the last tool result restores the checkpoint at its start.
    let last_tool = main.marks.last().unwrap().start as usize;
    b.serve(&main.clone().edit(last_tool + 50), None, "hybrid edit");
    assert_eq!(
        b.resume, last_tool as u32,
        "restored the last tool result's checkpoint"
    );
    // A subagent, a title request, and back to main.
    let sub = Conv::new(22, 260, top).user(300);
    b.serve(&sub, None, "hybrid subagent");
    b.serve(&main.clone().cut(280).user(40), None, "hybrid title");
    main = main.reply(&live).user(200);
    live = b.serve(&main, None, "hybrid main back");
    // Cancelled after one chunk, then retried.
    let long = Conv::new(23, 300, top).user(2400);
    b.serve(&long, Some(1), "hybrid cancel");
    b.serve(&long, None, "hybrid after cancel");
    main = main.reply(&live).user(100);
    b.serve(&main, None, "hybrid main last");
}

/// The oracle is not vacuous: recomputing a cached request with a different schedule for its
/// reused positions (the earlier reply prefilled as part of a chunk instead of decoded) changes
/// the low bits, and the oracle reports it.
#[test]
fn the_oracle_detects_a_different_schedule() {
    use mrun::cache::schedule::Span;
    let model = tiny::qwen3(64, None);
    let fresh = mrun_engine::testing::load_cpu(model.dir.as_ref().unwrap().path(), None).unwrap();
    let mut b = Bench::new(model.engine, fresh);
    let main = Conv::new(31, 300, 64).user(500);
    let live = b.serve(&main, None, "turn 1");
    let main = main.reply(&live).user(300);
    b.serve(&main, None, "turn 2");
    let mut served = b.last.take().unwrap();
    assert!(
        served
            .schedule
            .iter()
            .any(|s| matches!(s, Span::Steps { .. })),
        "the reused part holds decoded tokens"
    );
    // Everything below the resume position as one chunk.
    let resume = served.resume;
    served.schedule.retain(|s| s.start() >= resume);
    served.schedule.insert(
        0,
        Span::Chunk {
            start: 0,
            end: resume,
            tail: std::sync::Arc::from(Vec::new()),
        },
    );
    let mut fresh = b.fresh.take().unwrap();
    let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        oracle(&mut fresh, &served, "re-chunked")
    }));
    assert!(caught.is_err(), "a different schedule gives different bits");
}
