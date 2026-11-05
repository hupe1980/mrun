//! The prefix cache on `NuvecEngine` with tiny random-weight models on the CPU device (003 T022,
//! T024–T026, T028):
//!
//! - SC-002: with the state capacity set so main fills ≥ 75 % of it, main → subagent → title →
//!   main and main → subagent → subagent → main return to everything main committed, three times
//!   in a row, main protected throughout;
//! - SC-006: nuvec's active memory during every move stays within the state before it plus one layer
//!   of the moved range, and the accounted state within the capacity;
//! - SC-004: ≥ 100 requests of the mutation suite;
//! - SC-007 on the tiny hybrid: an edit of the last user message after ten tool turns resumes at
//!   its start, an identical retry at `P − 1`, and a budget below the protected set keeps the
//!   anchor and the newest user message;
//! - SC-008: a pressure event during a request is acted on at the next chunk end and leaves the
//!   request's output bit-identical to a run without it.

#![cfg(feature = "engine")]

#[path = "fixtures/mutations.rs"]
mod mutations;
#[path = "fixtures/support.rs"]
mod support;

use mrun::cache::miss::MissReport;
use mrun::cache::{CacheMode, PrefixStore};
use mrun::chat::prompt::{MessageMark, RenderedPrompt, Role};
use mrun::chat::tokenizer::Tokenizer;
use mrun::chat::tokenizer::test_support::byte_tokenizer;
use mrun::runtime::generate::{Event, EventSink, FinishReason, GenerateRequest, generate};
use mrun::runtime::pressure::{Pressure, PressureGauge};
use mrun_engine::tensor::memory;
use mrun_engine::{
    CancelToken, Checkpoint, CheckpointId, Engine, EngineError, LayerKind, MemoryBudget, ModelInfo,
    NuvecEngine, Sampling, Segment, SegmentId, Step, TokenMask, align,
};
use support::{Recorder, ScriptHook, tiny};

/// nuvec's memory counters are process-wide: the tests of this file run one at a time.
static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn serial() -> std::sync::MutexGuard<'static, ()> {
    SERIAL
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// Slack for the allocator's rounding of buffer sizes and the small index and scalar arrays a
/// move evaluates beside the layer copies.
const SCALARS: u64 = 64 * 1024;

/// Watches every move (SC-006): nuvec's peak active memory during `detach_tail` and `attach`
/// against the active memory before it plus one layer of the moved range, and the accounted
/// state against the capacity after every call.
struct Moves<E> {
    inner: E,
    moves: usize,
    worst: i64,
    /// Positions of each detached segment.
    segments: std::collections::HashMap<SegmentId, u32>,
    /// The request's growth ceiling (`set_limit`).
    limit: u32,
}

impl<E: Engine> Moves<E> {
    fn new(inner: E) -> Self {
        Self {
            inner,
            moves: 0,
            worst: i64::MIN,
            segments: std::collections::HashMap::new(),
            limit: 0,
        }
    }

    /// Bytes of one attention layer for `positions` (capacity is allocated in `ALIGN` steps).
    fn layer_bytes(&self, positions: u32) -> u64 {
        let info = self.inner.info();
        let attention = info
            .layer_kinds
            .iter()
            .filter(|k| **k == LayerKind::FullAttention)
            .count()
            .max(1) as u64;
        info.kv_bytes_per_token / attention * u64::from(align(positions))
    }

    /// Runs a move of `moved` positions whose rebuilt live layer holds `rebuilt` positions.
    fn watch<T>(&mut self, moved: u32, rebuilt: u32, op: impl FnOnce(&mut E) -> T) -> T {
        memory::clear_cache();
        let before = memory::active();
        memory::reset_peak();
        let out = op(&mut self.inner);
        let peak = memory::peak();
        // One layer at a time: the moved range of one layer plus that layer rebuilt at its new
        // capacity, never a second copy of the state (plus allocator rounding and scalars).
        let one_layer = self.layer_bytes(moved) + self.layer_bytes(rebuilt);
        let bound = before + one_layer + SCALARS;
        self.worst = self.worst.max(peak as i64 - bound as i64);
        assert!(
            peak <= bound,
            "SC-006: a move of {moved} positions peaked at {peak} > {before} + one layer ({one_layer})"
        );
        // Strictly below a copy of the moved range in every layer.
        let layers = self
            .inner
            .info()
            .layer_kinds
            .iter()
            .filter(|k| **k == LayerKind::FullAttention)
            .count() as u64;
        if layers > 1 && moved >= 1024 {
            assert!(
                peak - before < layers * self.layer_bytes(moved),
                "SC-006: no whole copy of the moved state"
            );
        }
        self.moves += 1;
        self.accounted();
        out
    }

    fn accounted(&self) {
        assert!(
            self.inner.allocated() <= self.inner.budget().state_capacity_bytes,
            "SC-006: accounted state within the capacity"
        );
    }
}

impl<E: Engine> Engine for Moves<E> {
    fn info(&self) -> &ModelInfo {
        self.inner.info()
    }
    fn budget(&self) -> &MemoryBudget {
        self.inner.budget()
    }
    fn len(&self) -> u32 {
        self.inner.len()
    }
    fn allocated(&self) -> u64 {
        self.inner.allocated()
    }
    fn set_limit(&mut self, total: u32) -> Result<(), EngineError> {
        self.limit = total;
        self.inner.set_limit(total)
    }
    fn prefill_chunk(&mut self, tokens: &[u32], cancel: &CancelToken) -> Result<bool, EngineError> {
        let out = self.inner.prefill_chunk(tokens, cancel);
        self.accounted();
        out
    }
    fn forward(&mut self, last: u32) -> Result<Step, EngineError> {
        let out = self.inner.forward(last);
        self.accounted();
        out
    }
    fn sample(
        &mut self,
        step: Step,
        sampling: &Sampling,
        mask: Option<&TokenMask>,
    ) -> Result<u32, EngineError> {
        self.inner.sample(step, sampling, mask)
    }
    fn top_logprobs(&mut self, step: &Step, n: usize) -> Result<Vec<(u32, f32)>, EngineError> {
        self.inner.top_logprobs(step, n)
    }
    fn trim(&mut self, len: u32) -> Result<(), EngineError> {
        self.inner.trim(len)
    }
    fn checkpoint(&mut self) -> Result<Checkpoint, EngineError> {
        self.inner.checkpoint()
    }
    fn restore(&mut self, id: CheckpointId) -> Result<(), EngineError> {
        self.inner.restore(id)
    }
    fn drop_checkpoint(&mut self, id: CheckpointId) {
        self.inner.drop_checkpoint(id);
    }
    fn detach_tail(&mut self, at: u32) -> Result<Segment, EngineError> {
        let moved = self.inner.len().saturating_sub(at);
        let segment = self.watch(moved, at, |e| e.detach_tail(at))?;
        self.segments
            .insert(segment.id, segment.end - segment.start);
        Ok(segment)
    }
    fn attach(&mut self, id: SegmentId) -> Result<(), EngineError> {
        // Each layer is rebuilt at its new capacity (the live prefix, the segment and growth up
        // to the limit) before the old layer and the segment's layer are freed: the transient is
        // one layer of that capacity.
        let moved = self.segments.get(&id).copied().unwrap_or(0);
        let capacity = (self.inner.len() + moved).max(self.limit);
        self.watch(0, capacity, |e| e.attach(id))
    }
    fn drop_segment(&mut self, id: SegmentId) {
        self.inner.drop_segment(id);
    }
    fn reset(&mut self) {
        self.inner.reset();
    }
    fn reset_all(&mut self) {
        self.inner.reset_all();
    }
}

/// A conversation of `(role, length)` messages with tokens drawn from `2..top`.
struct Conv {
    prompt: RenderedPrompt,
    seed: u64,
    top: u32,
}

impl Conv {
    fn new(seed: u64, system: u32, top: u32) -> Self {
        let mut c = Self {
            prompt: RenderedPrompt::default(),
            seed,
            top,
        };
        c.push(Role::System, system);
        c.prompt.system_tools_end = c.prompt.len();
        c
    }

    fn push(&mut self, role: Role, n: u32) -> &mut Self {
        let index = self.prompt.messages.len() as u32;
        self.prompt.messages.push(MessageMark {
            start: self.prompt.len(),
            role,
            message: Some(index),
        });
        for _ in 0..n {
            self.seed = self
                .seed
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let t = 2 + ((self.seed >> 33) % u64::from(self.top - 2)) as u32;
            self.prompt.tokens.push(t);
        }
        self.prompt.generation_start = self.prompt.len();
        self
    }

    /// Continues with what the store committed (prompt and reply).
    fn reply(&mut self, live: &[u32]) -> &mut Self {
        let start = self.prompt.len() as usize;
        assert_eq!(&live[..start], &self.prompt.tokens[..]);
        let index = self.prompt.messages.len() as u32;
        self.prompt.messages.push(MessageMark {
            start: start as u32,
            role: Role::Assistant,
            message: Some(index),
        });
        self.prompt.tokens.extend_from_slice(&live[start..]);
        self.prompt.generation_start = self.prompt.len();
        self
    }
}

struct Server<E: Engine> {
    engine: E,
    store: PrefixStore,
    tokenizer: Tokenizer,
}

impl<E: Engine> Server<E> {
    fn new(engine: E) -> Self {
        Self {
            engine,
            store: PrefixStore::new(CacheMode::Cached),
            tokenizer: byte_tokenizer(),
        }
    }

    fn serve_with(
        &mut self,
        prompt: &RenderedPrompt,
        max_tokens: u32,
        sink: &mut dyn EventSink,
        gauge: &PressureGauge,
    ) -> MissReport {
        // ASCII replies (the byte tokenizer shows them whole), then the tiny models' EOS.
        let mut request = GenerateRequest::new(prompt.clone(), max_tokens);
        let script: Vec<u32> = (0..max_tokens.saturating_sub(1))
            .map(|i| 40 + i % 20)
            .collect();
        let vocab = self.engine.info().vocab;
        request.hook = Some(Box::new(ScriptHook::new(script, 1, vocab)));
        let finished = generate(
            &mut self.engine,
            &mut self.store,
            &mut self.tokenizer.stream(),
            request,
            sink,
            gauge,
        );
        assert!(
            !matches!(finished.reason, FinishReason::Error(_)),
            "{:?}",
            finished.reason
        );
        self.store.assert_invariants(&self.engine);
        finished.stats.miss.unwrap()
    }

    fn serve(&mut self, prompt: &RenderedPrompt) -> MissReport {
        let mut events = Vec::new();
        self.serve_with(prompt, 4, &mut events, &PressureGauge::default())
    }

    fn live(&self) -> Vec<u32> {
        self.store.live_tokens().to_vec()
    }
}

/// SC-002 and SC-006 on tiny `qwen3`: main fills ≥ 75 % of the state capacity; subagents and a
/// title request between its turns never cost it a token, three rounds in a row.
#[test]
fn main_survives_subagents_and_titles_at_capacity() {
    let _serial = serial();
    // Capacity of 5,120 positions (256 bytes each): main reaches ~ 4,000.
    let state = 5_120 * 256;
    let tiny::Tiny { engine, dir: _dir } = tiny::qwen3(64, Some(state));
    let capacity = engine.budget().kv_capacity_tokens;
    assert!(capacity <= 8_192, "capacity {capacity}");
    let mut s = Server::new(Moves::new(engine));
    let mut main = Conv::new(1, 300, 64);
    main.push(Role::User, capacity * 2 / 5);
    s.serve(&main.prompt);
    let live = s.live();
    main.reply(&live).push(Role::Tool, capacity * 7 / 20);
    s.serve(&main.prompt);
    let mut committed = s.live();
    assert!(
        committed.len() as u32 * 4 >= capacity * 3,
        "main fills ≥ 75 %: {} of {capacity}",
        committed.len()
    );
    for round in 0..3 {
        // main → subagent → title → main
        let mut sub = Conv::new(10 + round, 250, 64);
        sub.push(Role::User, capacity / 10);
        s.serve(&sub.prompt);
        let mut title = Conv::new(1, 300, 64);
        title.push(Role::User, 30);
        assert_eq!(title.prompt.tokens[..300], main.prompt.tokens[..300]);
        s.serve(&title.prompt);
        main.reply(&committed).push(Role::Tool, 20);
        let back = s.serve(&main.prompt);
        assert!(
            back.cached as usize >= committed.len(),
            "round {round}: main back after a title: {back:?}"
        );
        assert_eq!((back.cause, back.gap), (None, 0));
        committed = s.live();
        // main → subagent → subagent → main
        for k in 0..2 {
            let mut sub = Conv::new(20 + 2 * round + k, 260, 64);
            sub.push(Role::User, capacity / 8);
            s.serve(&sub.prompt);
        }
        main.reply(&committed).push(Role::Tool, 20);
        let back = s.serve(&main.prompt);
        assert!(
            back.cached as usize >= committed.len(),
            "round {round}: main back after two subagents: {back:?}"
        );
        committed = s.live();
        // Protected throughout: two subagents never evict it.
        assert!(s.store.turns().0 >= 2, "main ran consecutive turns");
    }
    assert!(s.engine.moves >= 12, "moves watched: {}", s.engine.moves);
    eprintln!(
        "SC-006: {} moves, worst margin {} bytes",
        s.engine.moves, s.engine.worst
    );
}

/// The mutation suite (SC-004) on tiny `qwen3` with a 4,096-row vocabulary: unique ids for
/// message starts and edits, ASCII replies forced by a one-token mask.
#[test]
fn mutation_suite_on_tiny_qwen3() {
    let _serial = serial();
    struct EngineRig {
        engine: NuvecEngine,
        store: PrefixStore,
        tokenizer: Tokenizer,
    }
    struct CancelFirst;
    impl EventSink for CancelFirst {
        fn send(&mut self, event: Event) -> bool {
            !matches!(event, Event::Progress { .. })
        }
    }
    impl mutations::Rig for EngineRig {
        fn serve(&mut self, prompt: &RenderedPrompt, reply: &[u32], cancel: bool) -> MissReport {
            let mut events = Vec::new();
            let sink: &mut dyn EventSink = if cancel {
                &mut CancelFirst
            } else {
                &mut events
            };
            let mut request = GenerateRequest::new(prompt.clone(), 40);
            let vocab = self.engine.info().vocab;
            request.hook = Some(Box::new(ScriptHook::new(reply.to_vec(), 1, vocab)));
            let finished = generate(
                &mut self.engine,
                &mut self.store,
                &mut self.tokenizer.stream(),
                request,
                sink,
                &PressureGauge::default(),
            );
            assert!(
                !matches!(finished.reason, FinishReason::Error(_)),
                "{finished:?}"
            );
            finished.stats.miss.unwrap()
        }
        fn store(&self) -> &PrefixStore {
            &self.store
        }
        fn check(&self) {
            self.store.assert_invariants(&self.engine);
        }
        fn clear(&mut self) {
            self.store.clear_cache(&mut self.engine);
        }
        fn pressure(&mut self) {
            self.store
                .on_pressure(&mut self.engine, Pressure::Warning, false);
        }
        fn hybrid(&self) -> bool {
            false
        }
    }
    let shape = mrun_engine::testing::TinyQwen3 {
        max_position: 8192,
        ..mrun_engine::testing::TinyQwen3::new(4096)
    };
    // About 6,000 positions of state.
    let tiny::Tiny { engine, dir: _dir } = tiny::qwen3_with(shape, Some(6_000 * 256));
    let mut rig = EngineRig {
        engine,
        store: PrefixStore::new(CacheMode::Cached),
        tokenizer: byte_tokenizer(),
    };
    let alphabet = mutations::Alphabet {
        unique: (100, 4096),
        filler: (2, 90),
    };
    let sizes = mutations::Sizes {
        system: 200,
        message: 300,
        reply: 12,
    };
    let summary = mutations::run(&mut rig, 2026, 110, alphabet, sizes);
    assert!(summary.requests >= 100 && summary.exact > 40, "{summary:?}");
    assert!(rig.store.stats().evicted_total > 0, "eviction exercised");
    eprintln!("MLX mutation suite: {summary:?}");
}

/// SC-007 on the tiny hybrid: checkpoint marks serve edits and retries.
#[test]
fn hybrid_checkpoints_serve_edits_and_retries() {
    let _serial = serial();
    let tiny::Tiny { engine, .. } = tiny::hybrid(None);
    let mut s = Server::new(engine);
    let mut main = Conv::new(5, 300, 94);
    main.push(Role::User, 200);
    s.serve(&main.prompt);
    for _ in 0..10 {
        let live = s.live();
        main.reply(&live).push(Role::Tool, 60);
        let turn = s.serve(&main.prompt);
        assert_eq!(turn.cause, None, "{turn:?}");
    }
    // A new user message, one more turn, then an edit of that last user message.
    let live = s.live();
    main.reply(&live).push(Role::User, 120);
    let user_start = main.prompt.messages.last().unwrap().start;
    s.serve(&main.prompt);
    let retry = s.serve(&main.prompt);
    assert_eq!(
        retry.cached,
        main.prompt.len() - 1,
        "a retry resumes at P − 1"
    );
    let mut edited = main.prompt.clone();
    let at = user_start as usize + 60;
    edited.tokens[at] = if edited.tokens[at] == 2 { 3 } else { 2 };
    let edit = s.serve(&edited);
    assert_eq!(
        edit.cached, user_start,
        "an edit resumes at the user message: {edit:?}"
    );
    assert_eq!(edit.divergence, Some(at as u32));
}

/// SC-007's budget part on the tiny hybrid: a checkpoint budget below the protected set keeps
/// the anchor and the newest user message.
#[test]
fn a_small_checkpoint_budget_keeps_the_anchor_and_the_newest_user() {
    let _serial = serial();
    // Recurrent state: the budget (25 % of the capacity) holds two checkpoints.
    let probe = tiny::hybrid(None).engine;
    let (recurrent, kv) = (
        probe.info().recurrent_bytes,
        probe.info().kv_bytes_per_token,
    );
    drop(probe);
    let state = 4 * 2 * recurrent + 4 * 2 * recurrent / 3 + 8_192 * kv;
    let tiny::Tiny { engine, .. } = tiny::hybrid(Some(state));
    let budget = engine.budget().state_capacity_bytes / 4;
    let mut s = Server::new(engine);
    let mut main = Conv::new(6, 300, 94);
    main.push(Role::User, 100);
    s.serve(&main.prompt);
    for k in 0..4 {
        let live = s.live();
        main.reply(&live)
            .push(if k % 2 == 0 { Role::Tool } else { Role::User }, 80);
        s.serve(&main.prompt);
    }
    let checkpoints = s.store.checkpoints().to_vec();
    let bytes: u64 = checkpoints.iter().map(|c| c.bytes).sum();
    assert!(bytes <= budget, "within the budget");
    let positions: Vec<u32> = checkpoints.iter().map(|c| c.position).collect();
    let newest_user = main
        .prompt
        .messages
        .iter()
        .rev()
        .find(|m| m.role == Role::User)
        .unwrap()
        .start;
    assert!(
        positions.contains(&300),
        "the anchor survives: {positions:?}"
    );
    assert!(
        positions.contains(&newest_user),
        "the newest user message survives: {positions:?}"
    );
}

/// Raises a pressure level at the first progress event (after the first chunk).
struct RaiseAt<'a> {
    gauge: &'a PressureGauge,
    level: Pressure,
    raised: bool,
}

impl EventSink for RaiseAt<'_> {
    fn send(&mut self, event: Event) -> bool {
        if matches!(event, Event::Progress { .. }) && !self.raised {
            self.gauge.raise(self.level);
            self.raised = true;
        }
        true
    }
}

/// SC-008 on tiny `qwen3`: a pressure event during a long prefill sheds branches at the next
/// chunk end; the request's output is bit-identical to a run without the event.
#[test]
fn pressure_mid_request_leaves_the_output_bit_identical() {
    let _serial = serial();
    let run = |level: Option<Pressure>| {
        let tiny::Tiny { engine, dir: _dir } = tiny::qwen3(64, None);
        let mut s = Server::new(Recorder::new(engine));
        // Two stored branches, then a long new conversation.
        for seed in [1, 2] {
            let mut c = Conv::new(seed, 300, 64);
            c.push(Role::User, 400);
            s.serve(&c.prompt);
        }
        s.engine.take();
        let mut long = Conv::new(3, 300, 64);
        long.push(Role::User, 4000);
        let gauge = PressureGauge::default();
        let mut sink = RaiseAt {
            gauge: &gauge,
            level: level.unwrap_or(Pressure::Warning),
            raised: level.is_none(),
        };
        s.serve_with(&long.prompt, 8, &mut sink, &gauge);
        let (rows, ids) = s.engine.take();
        (rows, ids, s.store.stats(), s.live())
    };
    let (rows, ids, quiet, live) = run(None);
    for level in [Pressure::Warning, Pressure::Critical] {
        let (p_rows, p_ids, stats, p_live) = run(Some(level));
        assert_eq!(stats.pressure_events, 1, "{level:?}: acted on");
        assert!(stats.branches < quiet.branches, "{level:?}: branches shed");
        assert_eq!(p_ids, ids, "{level:?}: same ids");
        assert_eq!(p_live, live, "{level:?}: same committed tokens");
        for (a, b) in p_rows.iter().zip(&rows) {
            assert!(
                a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits()),
                "{level:?}: bit-identical log-probabilities"
            );
        }
    }
}
