//! `PrefixStore` v2 on `ScriptedEngine` (003 US1–US4, D031): hits, appends, subagent switches that
//! keep the parent, the tiny-tail rule, edits, `--no-cache`, eviction that protects multi-turn
//! conversations, and hybrid checkpoints.

use mrun::cache::miss::{Cause, MissReport};
use mrun::cache::{CacheMode, MIN_BRANCH, PrefixStore};
use mrun::chat::prompt::{MessageMark, RenderedPrompt, Role};
use mrun::runtime::generate::{Event, Finished, GenerateRequest, generate};
use mrun::runtime::pressure::PressureGauge;
use mrun::runtime::scripted::{SCRIPT_EOS, ScriptedEngine};
use mrun_engine::Engine;

#[path = "fixtures/mutations.rs"]
mod mutations;

/// A conversation: `(role, length, seed)` per message; tokens are `seed * 1000 + i` so
/// messages with equal seeds are equal. Ends with a two-token generation header.
fn conv(messages: &[(Role, u32, u32)]) -> RenderedPrompt {
    let mut tokens = Vec::new();
    let mut marks = Vec::new();
    for (i, &(role, len, seed)) in messages.iter().enumerate() {
        marks.push(MessageMark {
            start: tokens.len() as u32,
            role,
            message: Some(i as u32),
        });
        tokens.extend((0..len).map(|t| seed * 1000 + t));
    }
    let system_tools_end = if messages.first().is_some_and(|m| m.0 == Role::System) {
        messages[0].1
    } else {
        0
    };
    let generation_start = tokens.len() as u32;
    tokens.extend([7, 8]);
    RenderedPrompt {
        tokens,
        messages: marks,
        system_tools_end,
        generation_start,
    }
}

/// The next turn: the previous prompt, its reply, a closing token and a new message.
fn next_turn(prev: &RenderedPrompt, reply: &[u32], user: (u32, u32)) -> RenderedPrompt {
    let mut next = prev.clone();
    next.messages.push(MessageMark {
        start: prev.generation_start,
        role: Role::Assistant,
        message: Some(prev.messages.len() as u32),
    });
    next.tokens.extend_from_slice(reply);
    next.tokens.push(9);
    next.messages.push(MessageMark {
        start: next.tokens.len() as u32,
        role: Role::User,
        message: Some(prev.messages.len() as u32 + 1),
    });
    next.tokens.extend((0..user.0).map(|t| user.1 * 1000 + t));
    next.generation_start = next.tokens.len() as u32;
    next.tokens.extend([7, 8]);
    next
}

struct Rig {
    engine: ScriptedEngine,
    store: PrefixStore,
}

impl Rig {
    fn new(engine: ScriptedEngine, mode: CacheMode) -> Self {
        Self {
            engine,
            store: PrefixStore::new(mode),
        }
    }

    /// Runs one request; returns the miss report and the number of tokens processed.
    fn run(&mut self, prompt: &RenderedPrompt, max_tokens: u32) -> (MissReport, usize, Finished) {
        let before = self.engine.processed.len();
        let mut events: Vec<Event> = Vec::new();
        let mut detok_tokenizer = mrun::chat::tokenizer::test_support::byte_tokenizer();
        let tokenizer = &mut detok_tokenizer;
        let finished = generate(
            &mut self.engine,
            &mut self.store,
            &mut tokenizer.stream(),
            GenerateRequest::new(prompt.clone(), max_tokens),
            &mut events,
            &PressureGauge::default(),
        );
        assert_eq!(
            self.engine.tokens(),
            self.store.live_tokens(),
            "the store mirrors the live state"
        );
        assert!(self.engine.allocated() <= self.engine.budget().state_capacity_bytes);
        let miss = finished
            .stats
            .miss
            .unwrap_or_else(|| panic!("no miss report: {:?}", finished.reason));
        (miss, self.engine.processed.len() - before, finished)
    }
}

fn reply(n: u32) -> Vec<u32> {
    (0..n).map(|i| 50 + i % 40).collect()
}

fn script(replies: &[Vec<u32>]) -> Vec<u32> {
    let mut s = Vec::new();
    for r in replies {
        s.extend(r);
        s.push(SCRIPT_EOS);
    }
    s
}

#[test]
fn full_hit_and_append_reuse_everything_before_the_new_tokens() {
    let r1 = reply(20);
    let mut rig = Rig::new(
        ScriptedEngine::new(script(&[r1.clone(), reply(5), reply(5)]), 10_000),
        CacheMode::Cached,
    );
    let p1 = conv(&[(Role::System, 300, 1), (Role::User, 50, 2)]);
    let (miss, _, _) = rig.run(&p1, 100);
    assert_eq!(miss.cause, Some(Cause::New));

    let p2 = next_turn(&p1, &r1, (40, 3));
    let (miss, processed, _) = rig.run(&p2, 100);
    assert_eq!(miss.cause, None, "{miss:?}");
    assert_eq!(miss.cached, p1.len() + r1.len() as u32);
    assert!(
        processed < 60,
        "only the new turn is processed: {processed}"
    );

    // The same prompt again: everything but its last token.
    let (miss, _, _) = rig.run(&p2, 100);
    assert_eq!(miss.cached, p2.len() - 1);
}

#[test]
fn subagent_and_title_requests_keep_the_main_conversation() {
    let r_main = reply(30);
    let mut rig = Rig::new(
        ScriptedEngine::new(
            script(&[r_main.clone(), reply(10), reply(4), reply(5)]),
            20_000,
        ),
        CacheMode::Cached,
    );
    let main = conv(&[(Role::System, 400, 1), (Role::User, 3000, 2)]);
    rig.run(&main, 100);
    // A subagent: same system prompt and tools, a different task.
    let sub = conv(&[(Role::System, 400, 1), (Role::User, 200, 5)]);
    let (miss, _, _) = rig.run(&sub, 100);
    assert_eq!(miss.cause, Some(Cause::New));
    assert_eq!(miss.cached, 400);
    assert_eq!(
        rig.store.branches().len(),
        1,
        "the main tail moved out, not dropped"
    );
    // A title request with nothing in common.
    let title = conv(&[(Role::User, 30, 9)]);
    rig.run(&title, 100);
    // Back to main: its whole prompt and reply come back.
    let main2 = next_turn(&main, &r_main, (20, 6));
    let (miss, processed, _) = rig.run(&main2, 100);
    assert_eq!(miss.cause, None, "{miss:?}");
    assert_eq!(miss.cached, main.len() + r_main.len() as u32);
    assert!(processed < 30, "{processed}");
}

#[test]
fn a_short_tail_is_discarded_and_reported_as_evicted() {
    let mut rig = Rig::new(
        ScriptedEngine::new(script(&[reply(5), reply(5), reply(5)]), 10_000),
        CacheMode::Cached,
    );
    let main = conv(&[(Role::System, 400, 1), (Role::User, MIN_BRANCH - 100, 2)]);
    rig.run(&main, 100);
    let sub = conv(&[(Role::System, 400, 1), (Role::User, 50, 5)]);
    rig.run(&sub, 100);
    assert!(rig.store.branches().is_empty());
    let (miss, _, _) = rig.run(&main, 100);
    assert_eq!(miss.cause, Some(Cause::Evicted));
    assert_eq!(miss.cached, 400);
}

#[test]
fn edits_are_reported_with_the_message_that_diverged() {
    let r1 = reply(10);
    let mut rig = Rig::new(
        ScriptedEngine::new(script(&[r1.clone(), reply(5), reply(5)]), 10_000),
        CacheMode::Cached,
    );
    let p1 = conv(&[(Role::System, 300, 1), (Role::User, 50, 2)]);
    rig.run(&p1, 100);
    let mut p2 = next_turn(&p1, &r1, (40, 3));
    // The re-rendered reply differs from what the model wrote.
    let at = p1.generation_start as usize + 3;
    p2.tokens[at] = 999_999;
    let (miss, _, _) = rig.run(&p2, 100);
    assert_eq!(miss.cause, Some(Cause::Diverged));
    assert_eq!(miss.divergence, Some(at as u32));
    let message = miss.message.unwrap();
    assert_eq!((message.index, message.role), (2, Role::Assistant));
    // The tool list changed, inside the system region: a new conversation as far as the cache
    // can tell, with the message that changed named.
    let mut p3 = p1.clone();
    p3.tokens[10] = 999_998;
    let (miss, _, _) = rig.run(&p3, 100);
    assert_eq!(miss.cause, Some(Cause::New));
    assert_eq!(miss.message.unwrap().role, Role::System);
}

#[test]
fn no_cache_mode_processes_everything() {
    let mut rig = Rig::new(
        ScriptedEngine::new(script(&[reply(5), reply(5)]), 10_000),
        CacheMode::Off,
    );
    let p = conv(&[(Role::System, 100, 1), (Role::User, 50, 2)]);
    rig.run(&p, 100);
    let (miss, processed, _) = rig.run(&p, 100);
    assert_eq!((miss.cause, miss.cached), (Some(Cause::Disabled), 0));
    assert!(processed >= p.len() as usize);
}

#[test]
fn branches_are_evicted_lru_when_memory_runs_out() {
    // Capacity for ~2 conversations: the third request evicts the oldest branch, never the
    // state it reuses.
    let mut rig = Rig::new(
        ScriptedEngine::new(script(&[reply(5), reply(5), reply(5), reply(5)]), 3_000),
        CacheMode::Cached,
    );
    let a = conv(&[(Role::System, 100, 1), (Role::User, 1000, 2)]);
    let b = conv(&[(Role::System, 100, 1), (Role::User, 1000, 3)]);
    let c = conv(&[(Role::System, 100, 1), (Role::User, 1000, 4)]);
    rig.run(&a, 50);
    rig.run(&b, 50);
    assert_eq!(rig.store.branches().len(), 1);
    rig.run(&c, 50);
    // `a` (oldest) was evicted to make room; `b` is kept.
    let (miss, _, _) = rig.run(&b, 50);
    assert_eq!(miss.cause, None, "{miss:?}");
    assert_eq!(miss.cached, b.len() - 1);
}

#[test]
fn hybrid_resumes_from_checkpoints_with_a_gap() {
    const RB: u64 = 1_000;
    let r1 = reply(10);
    let mut rig = Rig::new(
        ScriptedEngine::new_hybrid(
            script(&[r1.clone(), reply(5), reply(5), reply(5)]),
            10_000,
            RB,
        ),
        CacheMode::Cached,
    );
    let p1 = conv(&[(Role::System, 300, 1), (Role::User, 400, 2)]);
    rig.run(&p1, 100);
    // An appended turn resumes exactly at the end of the previous reply.
    let p2 = next_turn(&p1, &r1, (40, 3));
    let (miss, _, _) = rig.run(&p2, 100);
    assert_eq!((miss.cause, miss.gap), (None, 0), "{miss:?}");
    // Editing the last user message: resume from the checkpoint at its start, re-processing the
    // tokens between it and the divergence.
    let mut p3 = p2.clone();
    let last_user = p3.messages.last().unwrap().start;
    p3.tokens[last_user as usize + 20] = 999_999;
    let (miss, _, _) = rig.run(&p3, 100);
    assert_eq!(miss.cause, Some(Cause::Diverged));
    assert_eq!(miss.cached, last_user);
    assert_eq!(miss.gap, 20);
    // A subagent and back: the main tail's end state travels with its segment.
    let sub = conv(&[(Role::System, 300, 1), (Role::User, 400, 7)]);
    let (miss, _, _) = rig.run(&sub, 100);
    assert_eq!(miss.cached, 300, "the system checkpoint");
    // Position-bound state exists only where it was computed: the same prompt (a retry or a
    // regeneration) resumes from the checkpoint at `P − 1` (D053).
    let (miss, _, _) = rig.run(&p3, 100);
    assert_eq!(miss.cause, None, "{miss:?}");
    assert_eq!((miss.cached, miss.gap), (p3.len() - 1, 0));
}

/// Small deterministic generator (xorshift).
/// `ScriptedEngine` under the shared mutation suite (SC-004, T024).
struct ScriptedRig {
    engine: ScriptedEngine,
    store: PrefixStore,
}

/// Cancels at the first progress event (after one chunk).
struct CancelFirst;

impl mrun::runtime::generate::EventSink for CancelFirst {
    fn send(&mut self, event: Event) -> bool {
        !matches!(event, Event::Progress { .. })
    }
}

impl mutations::Rig for ScriptedRig {
    fn serve(&mut self, prompt: &RenderedPrompt, _reply: &[u32], cancel: bool) -> MissReport {
        let mut events: Vec<Event> = Vec::new();
        let tokenizer = mrun::chat::tokenizer::test_support::byte_tokenizer();
        let sink: &mut dyn mrun::runtime::generate::EventSink = if cancel {
            &mut CancelFirst
        } else {
            &mut events
        };
        let finished = generate(
            &mut self.engine,
            &mut self.store,
            &mut tokenizer.stream(),
            GenerateRequest::new(prompt.clone(), 30),
            sink,
            &PressureGauge::default(),
        );
        finished.stats.miss.expect("a miss report")
    }
    fn store(&self) -> &PrefixStore {
        &self.store
    }
    fn check(&self) {
        assert_eq!(self.engine.tokens(), self.store.live_tokens());
        self.store.assert_invariants(&self.engine);
    }
    fn clear(&mut self) {
        self.store.clear_cache(&mut self.engine);
    }
    fn pressure(&mut self) {
        self.store.on_pressure(
            &mut self.engine,
            mrun::runtime::pressure::Pressure::Warning,
            false,
        );
    }
    fn hybrid(&self) -> bool {
        self.engine.info().recurrent_bytes > 0
    }
}

/// Random agent traffic under tight memory (`fixtures/mutations.rs`): expected cause,
/// divergence, message, gap and `unreused` per injected change, no stale reuse, invariants.
fn mutation_suite(hybrid: bool, seed: u64) -> mutations::Summary {
    let replies: Vec<Vec<u32>> = (0..400).map(|i| reply(1 + (i % 19) as u32)).collect();
    let engine = if hybrid {
        ScriptedEngine::new_hybrid(script(&replies), 6_000, 1_000)
    } else {
        ScriptedEngine::new(script(&replies), 6_000)
    };
    let mut rig = ScriptedRig {
        engine,
        store: PrefixStore::new(CacheMode::Cached),
    };
    let alphabet = mutations::Alphabet {
        unique: (1_000_000, 2_000_000),
        filler: (100, 900),
    };
    let sizes = mutations::Sizes {
        system: 200,
        message: 300,
        reply: 20,
    };
    let summary = mutations::run(&mut rig, seed, 150, alphabet, sizes);
    let stats = rig.store.stats();
    assert!(stats.evicted_total > 0, "eviction was exercised: {stats:?}");
    assert!(stats.cached_tokens_total > 0, "and reuse: {stats:?}");
    summary
}

#[test]
fn mutation_suite_attention() {
    let mut causes = std::collections::BTreeMap::new();
    for seed in [1, 7, 42, 2026, 5, 13, 77, 1234] {
        let s = mutation_suite(false, seed);
        assert!(s.exact >= 50 && s.evicted > 0, "{s:?}");
        for (k, v) in s.by_cause {
            *causes.entry(k).or_insert(0) += v;
        }
    }
    for cause in ["hit", "new", "diverged", "evicted"] {
        assert!(
            causes.get(cause).is_some_and(|&n| n > 0),
            "{cause}: {causes:?}"
        );
    }
}

#[test]
fn mutation_suite_hybrid() {
    let mut causes = std::collections::BTreeMap::new();
    for seed in [3, 11, 99, 1015, 8, 21, 64, 4096] {
        let s = mutation_suite(true, seed);
        assert!(s.exact >= 50 && s.evicted > 0, "{s:?}");
        for (k, v) in s.by_cause {
            *causes.entry(k).or_insert(0) += v;
        }
    }
    for cause in ["hit", "new", "diverged", "evicted"] {
        assert!(
            causes.get(cause).is_some_and(|&n| n > 0),
            "{cause}: {causes:?}"
        );
    }
}

#[test]
fn a_cancelled_prefill_keeps_its_whole_chunks_for_the_retry() {
    let mut engine = ScriptedEngine::new(script(&[reply(5)]), 20_000);
    // Four layers per chunk: cancel inside the third chunk.
    engine.cancel_after_layers = Some(4 * 2 + 2);
    let mut rig = Rig::new(engine, CacheMode::Cached);
    let p = conv(&[(Role::System, 400, 1), (Role::User, 6000, 2)]);
    let (_, _, finished) = rig.run(&p, 50);
    assert_eq!(
        finished.reason,
        mrun::runtime::generate::FinishReason::Cancelled
    );
    let kept = rig.store.live_tokens().len() as u32;
    assert!(kept > 0 && kept < p.len(), "whole chunks kept: {kept}");
    // The client retries: the kept chunks are reused, not recomputed.
    let (miss, _, _) = rig.run(&p, 50);
    assert_eq!(miss.cached, kept);
    assert_eq!(miss.cause, None);
}

#[test]
fn an_80k_main_conversation_survives_subagents_and_titles_at_105k_capacity() {
    // Qwen3-Coder on a 48 GB Mac holds about 105K tokens: a subagent or a title request must not
    // cost the main conversation (SC-002, structural part).
    let r_main = reply(40);
    let mut rig = Rig::new(
        ScriptedEngine::new(
            script(&[r_main.clone(), reply(20), reply(30), reply(5), reply(5)]),
            105_000,
        ),
        CacheMode::Cached,
    );
    let main = conv(&[(Role::System, 2_000, 1), (Role::User, 80_000, 2)]);
    rig.run(&main, 100);
    let sub = conv(&[(Role::System, 2_000, 1), (Role::User, 3_000, 3)]);
    rig.run(&sub, 100);
    let main2 = next_turn(&main, &r_main, (50, 4));
    let (miss, processed, _) = rig.run(&main2, 100);
    assert_eq!(
        (miss.cause, miss.cached),
        (None, main.len() + r_main.len() as u32)
    );
    assert!(processed < 100, "{processed}");
    // And a title request in between keeps it too.
    let title = conv(&[(Role::User, 500, 5)]);
    rig.run(&title, 100);
    let (miss, _, _) = rig.run(&main2, 100);
    assert_eq!(miss.cached, main2.len() - 1, "{miss:?}");
}

#[test]
fn hybrid_tool_loop_then_a_new_user_turn_resumes_near_the_end() {
    const RB: u64 = 1_000;
    let replies: Vec<Vec<u32>> = (0..16).map(|i| reply(8 + i)).collect();
    let mut rig = Rig::new(
        ScriptedEngine::new_hybrid(script(&replies), 20_000, RB),
        CacheMode::Cached,
    );
    let mut p = conv(&[(Role::System, 300, 1), (Role::User, 100, 2)]);
    // Ten tool turns: each appends the model's call and a tool result.
    for (turn, reply) in replies.iter().take(11).enumerate() {
        let (miss, _, _) = rig.run(&p, 50);
        if turn > 0 {
            assert_eq!((miss.cause, miss.gap), (None, 0), "turn {turn}: {miss:?}");
        }
        let mut next = next_turn(&p, reply, (60, 10 + turn as u32));
        next.messages.last_mut().unwrap().role = Role::Tool;
        p = next;
    }
    // A tool result retried with different content resumes at its start (checkpoint).
    let (miss, _, _) = rig.run(&p, 50);
    assert_eq!((miss.cause, miss.gap), (None, 0), "{miss:?}");
    let mut retried = p.clone();
    let tool_start = retried.messages.last().unwrap().start;
    retried.tokens[tool_start as usize + 5] = 777_777;
    let (miss, _, _) = rig.run(&retried, 50);
    assert_eq!(miss.cause, Some(Cause::Diverged));
    assert_eq!((miss.cached, miss.gap), (tool_start, 5));
}

#[test]
fn attention_state_restores_exactly_at_any_position() {
    // Attention state is exact at any length: a match anywhere resumes with no gap and no
    // checkpoints.
    use mrun::runtime::scripted::FamilyProfile;
    let profile = FamilyProfile::get("qwen3").unwrap();
    let budget = profile.budget(48 << 30, 1 << 30).unwrap();
    let replies: Vec<Vec<u32>> = (0..30).map(|_| reply(5)).collect();
    let mut rig = Rig::new(
        ScriptedEngine::for_family(script(&replies), profile, budget),
        CacheMode::Cached,
    );
    let base = conv(&[(Role::System, 300, 1), (Role::User, 700, 2)]);
    rig.run(&base, 20);
    for at in [64u32, 255, 256, 257, 300, 420, 900] {
        let mut edited = base.clone();
        edited.tokens[at as usize] = 800_000 + at;
        let (miss, _, _) = rig.run(&edited, 20);
        assert_eq!(
            (miss.cached, miss.gap),
            (at, 0),
            "divergence at {at}: {miss:?}"
        );
        assert_eq!(
            rig.engine.checkpoint_count(),
            0,
            "no checkpoints without position-bound state"
        );
        // And back to the original: the moved-out tail returns (or was short and recomputed).
        let (miss, _, _) = rig.run(&base, 20);
        assert_eq!(miss.gap, 0);
    }
}

#[test]
fn lookup_stays_fast_at_100k_tokens() {
    // T004c: the lookup that `initial_need` and `prepare` run, over a 100K-token trunk and a
    // few branches, stays well under the per-request budget.
    let mut rig = Rig::new(
        ScriptedEngine::new(script(&[reply(2), reply(2), reply(2), reply(2)]), 200_000),
        CacheMode::Cached,
    );
    let main = conv(&[(Role::System, 1_000, 1), (Role::User, 99_000, 2)]);
    let other = conv(&[(Role::System, 1_000, 1), (Role::User, 30_000, 3)]);
    rig.run(&main, 5);
    rig.run(&other, 5);
    rig.run(&main, 5);
    let started = std::time::Instant::now();
    let rounds = 20;
    for _ in 0..rounds {
        std::hint::black_box(rig.store.evict(&mut rig.engine, 0, Some(&main)));
    }
    let per_lookup = started.elapsed() / rounds;
    let bound = if cfg!(debug_assertions) {
        std::time::Duration::from_millis(20)
    } else {
        std::time::Duration::from_millis(1)
    };
    assert!(per_lookup < bound, "{per_lookup:?}");
}

#[test]
fn a_multi_turn_parent_outlives_two_subagents_under_memory_pressure() {
    // D031: plain LRU would evict the parent (used before both subagents); a conversation that
    // ran two or more turns is protected while one-shot branches go first.
    let r1 = reply(20);
    let r2 = reply(20);
    let replies = [
        r1.clone(),
        r2.clone(),
        reply(5),
        reply(5),
        reply(5),
        reply(5),
    ];
    let mut rig = Rig::new(
        ScriptedEngine::new(script(&replies), 9_000),
        CacheMode::Cached,
    );
    let main = conv(&[(Role::System, 200, 1), (Role::User, 3_000, 2)]);
    rig.run(&main, 100);
    let main2 = next_turn(&main, &r1, (100, 3));
    rig.run(&main2, 100);
    // Two subagents with the shared system prompt, each large enough to force eviction.
    let sub1 = conv(&[(Role::System, 200, 1), (Role::User, 3_000, 5)]);
    rig.run(&sub1, 100);
    let sub2 = conv(&[(Role::System, 200, 1), (Role::User, 3_000, 6)]);
    rig.run(&sub2, 100);
    // The parent comes back whole; the first subagent was evicted instead.
    let main3 = next_turn(&main2, &r2, (50, 4));
    let (miss, _, _) = rig.run(&main3, 100);
    assert_eq!(miss.cause, None, "{miss:?}");
    assert_eq!(miss.cached, main2.len() + r2.len() as u32);
    let (miss, _, _) = rig.run(&sub1, 100);
    assert_eq!(miss.cause, Some(Cause::Evicted), "{miss:?}");
}

#[test]
fn a_title_request_does_not_orphan_branches_it_cuts_below() {
    // A title request shares nothing: the trunk is cut to 0, so the main conversation's branch
    // (detached at the end of the shared system prompt) is reached through the branch that now
    // holds that prompt (holder chain), not dropped.
    let r_main = reply(30);
    let mut rig = Rig::new(
        ScriptedEngine::new(
            script(&[r_main.clone(), reply(10), reply(4), reply(5)]),
            20_000,
        ),
        CacheMode::Cached,
    );
    let main = conv(&[(Role::System, 400, 1), (Role::User, 3000, 2)]);
    rig.run(&main, 100);
    let sub = conv(&[(Role::System, 400, 1), (Role::User, 200, 5)]);
    rig.run(&sub, 100);
    let title = conv(&[(Role::User, 30, 9)]);
    rig.run(&title, 100);
    assert_eq!(rig.store.branches().len(), 2, "{:?}", rig.store.branches());
    let main2 = next_turn(&main, &r_main, (20, 6));
    let (miss, processed, _) = rig.run(&main2, 100);
    assert_eq!(miss.cause, None, "{miss:?}");
    assert_eq!(miss.cached, main.len() + r_main.len() as u32);
    assert!(processed < 30, "{processed}");
}

#[test]
fn a_parent_stays_protected_across_repeated_round_trips() {
    // H1 / D053: the turn count moves with the state through detach and attach, so the parent
    // keeps its protection after any number of subagent round trips (SC-002, repeated). Memory
    // holds about two conversations: every second subagent must evict something, and it must
    // never be the parent.
    let main_replies: Vec<Vec<u32>> = (0..5).map(|_| reply(20)).collect();
    let mut script_replies = vec![main_replies[0].clone(), main_replies[1].clone()];
    for round in 0..3 {
        script_replies.extend([reply(5), reply(5), main_replies[round + 2].clone()]);
    }
    let mut rig = Rig::new(
        ScriptedEngine::new(script(&script_replies), 9_000),
        CacheMode::Cached,
    );
    // Two consecutive turns make main a protected parent.
    let first = conv(&[(Role::System, 200, 1), (Role::User, 3_000, 2)]);
    rig.run(&first, 100);
    let mut main = next_turn(&first, &main_replies[0], (50, 3));
    rig.run(&main, 100);
    assert_eq!(rig.store.turns().0, 2);
    let mut fresh = 10;
    for round in 0..3 {
        for _ in 0..2 {
            fresh += 1;
            let sub = conv(&[(Role::System, 200, 1), (Role::User, 3_000, fresh)]);
            let (miss, _, _) = rig.run(&sub, 100);
            assert_eq!(miss.cached, 200, "round {round}: the shared system prompt");
        }
        let reply = &main_replies[round + 1];
        let next = next_turn(&main, reply, (50, 100 + round as u32));
        let (miss, _, _) = rig.run(&next, 100);
        assert_eq!(miss.cause, None, "round {round}: {miss:?}");
        assert_eq!(
            miss.cached,
            main.len() + reply.len() as u32,
            "round {round}: main came back whole"
        );
        assert_eq!(rig.store.turns().0, round as u32 + 3, "round {round}");
        main = next;
    }
}

#[test]
fn eviction_keeps_the_whole_holder_chain_of_the_protected_path() {
    // L2: the main conversation is reachable only through the branch holding the shared
    // system prompt (after a title request cut the trunk to 0); protecting main must protect
    // that holder too.
    let r_main = reply(30);
    let mut rig = Rig::new(
        ScriptedEngine::new(
            script(&[r_main.clone(), reply(10), reply(4), reply(5)]),
            20_000,
        ),
        CacheMode::Cached,
    );
    let main = conv(&[(Role::System, 400, 1), (Role::User, 3000, 2)]);
    rig.run(&main, 100);
    let sub = conv(&[(Role::System, 400, 1), (Role::User, 200, 5)]);
    rig.run(&sub, 100);
    let title = conv(&[(Role::User, 30, 9)]);
    rig.run(&title, 100);
    assert_eq!(rig.store.branches().len(), 2);
    let main2 = next_turn(&main, &r_main, (20, 6));
    rig.store.evict(&mut rig.engine, u64::MAX, Some(&main2));
    assert_eq!(rig.store.branches().len(), 2, "holder and main kept");
    let (miss, _, _) = rig.run(&main2, 100);
    assert_eq!(miss.cause, None, "{miss:?}");
    assert_eq!(miss.cached, main.len() + r_main.len() as u32);
}

#[test]
fn a_hybrid_resume_far_behind_the_match_reports_gap() {
    // D053: a prompt contained in the trunk with no checkpoint near its end resumes far
    // behind the match: cause `gap`, and the gap is unreused work.
    let mut rig = Rig::new(
        ScriptedEngine::new_hybrid(script(&[reply(5), reply(5)]), 10_000, 1_000),
        CacheMode::Cached,
    );
    let long = RenderedPrompt::from_tokens((0..3_000).map(|t| 5_000 + t).collect());
    rig.run(&long, 10);
    let short = RenderedPrompt::from_tokens(long.tokens[..1_000].to_vec());
    let (miss, _, _) = rig.run(&short, 10);
    assert_eq!(miss.cause, Some(Cause::Gap), "{miss:?}");
    assert_eq!((miss.cached, miss.gap, miss.unreused), (0, 999, 999));
}

#[test]
fn a_dropped_tail_counts_as_unreused() {
    let mut rig = Rig::new(
        ScriptedEngine::new(script(&[reply(5), reply(5), reply(5)]), 10_000),
        CacheMode::Cached,
    );
    let main = conv(&[(Role::System, 400, 1), (Role::User, MIN_BRANCH - 100, 2)]);
    rig.run(&main, 100);
    let (miss, _, _) = rig.run(&main, 100);
    assert_eq!(
        (miss.cause, miss.unreused),
        (None, 0),
        "a hit leaves nothing"
    );
    let sub = conv(&[(Role::System, 400, 1), (Role::User, 50, 5)]);
    rig.run(&sub, 100);
    let (miss, _, _) = rig.run(&main, 100);
    assert_eq!(miss.cause, Some(Cause::Evicted));
    // The whole prompt was on the dropped path: everything but the system prompt (and the
    // last token, always processed) is unreused.
    assert_eq!((miss.cached, miss.unreused), (400, main.len() - 1 - 400));
    assert!(!miss.ring_wrapped);
}

/// A hybrid prompt with an anchor, three user messages and two replies; checkpoints are
/// planned at 300 (anchor and first user), 750, 1200 and `P − 1`.
fn three_user_turns() -> RenderedPrompt {
    conv(&[
        (Role::System, 300, 1),
        (Role::User, 400, 2),
        (Role::Assistant, 50, 3),
        (Role::User, 400, 4),
        (Role::Assistant, 50, 5),
        (Role::User, 400, 6),
    ])
}

#[test]
fn checkpoint_budget_thins_by_gap_and_keeps_the_protected_marks() {
    // D053: the budget (25 % of the state) holds three of five checkpoints: the unprotected ones
    // go (smallest gap first); the anchor, the newest user message and the generation header
    // stay.
    let rb = 14_000;
    let mut rig = Rig::new(
        ScriptedEngine::new_hybrid(script(&[reply(5)]), 10_000, rb),
        CacheMode::Cached,
    );
    assert_eq!(rig.engine.budget().state_capacity_bytes / 4 / rb, 3);
    let p = three_user_turns();
    rig.run(&p, 10);
    let at: Vec<u32> = rig.store.checkpoints().iter().map(|c| c.position).collect();
    assert_eq!(at, [300, 1200, 1600]);
}

#[test]
fn the_protected_set_shrinks_to_fit_the_budget() {
    let rb = 40_000;
    let mut rig = Rig::new(
        ScriptedEngine::new_hybrid(script(&[reply(5)]), 10_000, rb),
        CacheMode::Cached,
    );
    assert_eq!(rig.engine.budget().state_capacity_bytes / 4 / rb, 1);
    rig.run(&three_user_turns(), 10);
    let at: Vec<u32> = rig.store.checkpoints().iter().map(|c| c.position).collect();
    assert_eq!(
        at,
        [1200],
        "the newest user message outlives the closer anchor"
    );
}

/// FR-015 (T016): a retried identical request resumes at `P − 1`, commits exactly its own
/// prompt and reply (with their marks' checkpoints and forward schedule), and drops the state
/// the first reply left beyond the prompt.
#[test]
fn an_identical_retry_resumes_at_p_minus_1_and_commits_its_own_reply() {
    let first = reply(300);
    let second = reply(7);
    let mut rig = Rig::new(
        ScriptedEngine::new_hybrid(script(&[first.clone(), second.clone()]), 20_000, 1_000),
        CacheMode::Cached,
    );
    let p = three_user_turns();
    let (miss, _, _) = rig.run(&p, 400);
    assert_eq!(miss.cause, Some(Cause::New));
    let mut committed = p.tokens.clone();
    committed.extend(&first);
    assert_eq!(rig.store.live_tokens(), &committed[..]);
    rig.store.assert_invariants(&rig.engine);
    // The retry: everything but the last prompt token is reused.
    let (miss, _, _) = rig.run(&p, 400);
    assert_eq!(
        (miss.cached, miss.cause, miss.processed),
        (p.len() - 1, None, 1)
    );
    let mut committed = p.tokens.clone();
    committed.extend(&second);
    assert_eq!(rig.store.live_tokens(), &committed[..], "its own reply");
    // The trunk was cut at `P − 1` (the last prompt token is fed again); the first reply's
    // tail is kept as a branch (≥ 256 tokens): a request continuing it still finds it.
    assert_eq!(rig.store.branches(), [(p.len() - 1, p.len() + 300)]);
    rig.store.assert_invariants(&rig.engine);
    let schedule = rig.store.schedule();
    assert!(matches!(
        schedule.last(),
        Some(mrun::cache::schedule::Span::Steps { end, .. }) if *end == committed.len() as u32
    ));
}

/// T035: Anthropic tool results are tool marks (002 FR-037), so a long tool loop never fills
/// the protected set; with the budget full of protected checkpoints, captures at unprotected
/// marks (last tool result, `P − 1`) are skipped before they are taken.
#[test]
fn tool_loops_keep_the_protected_set_and_skip_unaffordable_captures() {
    let rb = 14_000;
    let mut rig = Rig::new(
        ScriptedEngine::new_hybrid(script(&vec![reply(5); 30]), 10_000, rb),
        CacheMode::Cached,
    );
    assert_eq!(rig.engine.budget().state_capacity_bytes / 4 / rb, 3);
    let mut p = three_user_turns();
    rig.run(&p, 10);
    for turn in 0..12u32 {
        let live = rig.store.live_tokens().to_vec();
        let mut next = p.clone();
        next.messages.push(MessageMark {
            start: p.generation_start,
            role: Role::Assistant,
            message: Some(p.messages.len() as u32),
        });
        next.tokens = live;
        next.messages.push(MessageMark {
            start: next.tokens.len() as u32,
            role: Role::Tool,
            message: Some(p.messages.len() as u32 + 1),
        });
        next.tokens.extend((0..40).map(|t| (100 + turn) * 1000 + t));
        next.generation_start = next.tokens.len() as u32;
        next.tokens.extend([7, 8]);
        if turn == 11 {
            // The budget holds the anchor, the newest user message and the newest generation
            // header, all protected.
            let at: Vec<u32> = rig.store.checkpoints().iter().map(|c| c.position).collect();
            assert_eq!(at, [300, 1200, p.generation_start]);
            let prepared = rig
                .store
                .prepare(&mut rig.engine, &next, next.len() + 10)
                .unwrap();
            assert_eq!(
                prepared.checkpoints,
                [next.generation_start],
                "nothing droppable: only the protected header is captured"
            );
            rig.store.clear(&mut rig.engine);
            break;
        }
        let (miss, _, _) = rig.run(&next, 10);
        assert_eq!(miss.cause, None, "turn {turn}: {miss:?}");
        rig.store.assert_invariants(&rig.engine);
        p = next;
    }
}

/// SC-007 on the position-bound profile: after ten tool turns an edit of the last user message
/// resumes at its start; an identical retry at `P − 1`.
#[test]
fn hybrid_edits_and_retries_resume_at_their_marks() {
    let mut rig = Rig::new(
        ScriptedEngine::new_hybrid(script(&vec![reply(6); 30]), 50_000, 1_000),
        CacheMode::Cached,
    );
    let mut p = conv(&[(Role::System, 300, 1), (Role::User, 200, 2)]);
    rig.run(&p, 10);
    for turn in 0..11u32 {
        let live = rig.store.live_tokens().to_vec();
        let role = if turn == 10 { Role::User } else { Role::Tool };
        let index = p.messages.len() as u32;
        p.messages.push(MessageMark {
            start: p.generation_start,
            role: Role::Assistant,
            message: Some(index),
        });
        p.tokens = live;
        p.messages.push(MessageMark {
            start: p.tokens.len() as u32,
            role,
            message: Some(index + 1),
        });
        p.tokens.extend((0..80).map(|t| (200 + turn) * 1000 + t));
        p.generation_start = p.tokens.len() as u32;
        p.tokens.extend([7, 8]);
        let (miss, _, _) = rig.run(&p, 10);
        assert_eq!(miss.cause, None);
    }
    let user = p.messages.last().unwrap().start;
    let (retry, _, _) = rig.run(&p, 10);
    assert_eq!(retry.cached, p.len() - 1);
    let mut edited = p.clone();
    edited.tokens[user as usize + 40] = 999_999;
    let (edit, _, _) = rig.run(&edited, 10);
    assert_eq!(edit.cached, user, "{edit:?}");
    assert_eq!(edit.cause, Some(Cause::Diverged));
}

/// 021: a plain follow-up re-renders the earlier reply (templates drop empty think blocks and
/// earlier reasoning; D051 splices only the current turn), so it diverges right after the
/// generation header. A hybrid resumes from the header's checkpoint, not the user message's
/// start: the 7,981 unreused tokens of 020 R9.
#[test]
fn a_hybrid_follow_up_resumes_at_the_earlier_generation_header() {
    let mut rig = Rig::new(
        ScriptedEngine::new_hybrid(script(&[reply(40), reply(40)]), 50_000, 1_000),
        CacheMode::Cached,
    );
    // Thinking on: the header opens a think block (`<|im_start|>assistant\n<think>\n`), so
    // the prompt end − 1 lies past the divergence.
    let mut first = conv(&[(Role::User, 8_000, 2)]);
    first.tokens.extend([6, 4]);
    rig.run(&first, 50);
    // The template renders the earlier reply without it: `<|im_start|>assistant\n` and text.
    let g = first.generation_start as usize;
    let mut follow_up = next_turn(&first, &(5_000..5_030).collect::<Vec<_>>(), (20, 3));
    follow_up.tokens.drain(g + 2..g + 4);
    for m in follow_up.messages.iter_mut().skip(2) {
        m.start -= 2;
    }
    follow_up.generation_start -= 2;
    let (miss, processed, _) = rig.run(&follow_up, 10);
    assert_eq!(miss.cause, Some(Cause::Diverged), "{miss:?}");
    assert_eq!(miss.divergence, Some(first.generation_start + 2));
    assert_eq!((miss.cached, miss.gap), (first.generation_start, 2));
    assert!(processed < 100, "{processed} tokens processed: {miss:?}");
}

/// SC-008 on `ScriptedEngine`: a pressure event raised during prefill is acted on at the next
/// chunk end (branches shed) and the request's output and committed state are those of a run
/// without it.
#[test]
fn pressure_during_prefill_leaves_the_request_unchanged() {
    struct RaiseAt<'a>(
        &'a PressureGauge,
        mrun::runtime::pressure::Pressure,
        Vec<u32>,
    );
    impl mrun::runtime::generate::EventSink for RaiseAt<'_> {
        fn send(&mut self, event: Event) -> bool {
            match event {
                Event::Progress { .. } => self.0.raise(self.1),
                Event::Token { id, .. } => self.2.push(id),
                _ => {}
            }
            true
        }
    }
    let run = |level: Option<mrun::runtime::pressure::Pressure>| {
        let mut rig = Rig::new(
            ScriptedEngine::new(script(&[reply(5), reply(5), reply(20)]), 50_000),
            CacheMode::Cached,
        );
        rig.run(&conv(&[(Role::System, 300, 1), (Role::User, 400, 2)]), 10);
        rig.run(&conv(&[(Role::System, 300, 3), (Role::User, 400, 4)]), 10);
        let long = conv(&[(Role::System, 300, 5), (Role::User, 5_000, 6)]);
        let gauge = PressureGauge::default();
        let mut sink = RaiseAt(
            &gauge,
            level.unwrap_or(mrun::runtime::pressure::Pressure::Warning),
            Vec::new(),
        );
        let tokenizer = mrun::chat::tokenizer::test_support::byte_tokenizer();
        let processed = rig.engine.processed.len();
        let quiet = PressureGauge::default();
        generate(
            &mut rig.engine,
            &mut rig.store,
            &mut tokenizer.stream(),
            GenerateRequest::new(long, 30),
            &mut sink,
            if level.is_some() { &gauge } else { &quiet },
        );
        let processed = rig.engine.processed[processed..].to_vec();
        (
            sink.2,
            processed,
            rig.store.live_tokens().to_vec(),
            rig.store.stats(),
        )
    };
    let (ids, processed, live, quiet) = run(None);
    for level in [
        mrun::runtime::pressure::Pressure::Warning,
        mrun::runtime::pressure::Pressure::Critical,
    ] {
        let (p_ids, p_processed, p_live, stats) = run(Some(level));
        assert!(
            stats.pressure_events >= 1 && stats.branches < quiet.branches,
            "{level:?}"
        );
        assert_eq!(
            (p_ids, p_processed, p_live),
            (ids.clone(), processed.clone(), live.clone())
        );
    }
}

/// `conv` with its first `shared` tokens taken from `from` (two system prompts that start alike).
fn sharing(mut prompt: RenderedPrompt, from: &RenderedPrompt, shared: usize) -> RenderedPrompt {
    prompt.tokens[..shared].copy_from_slice(&from.tokens[..shared]);
    prompt
}

#[test]
fn claude_code_side_requests_keep_the_hybrid_main_conversation() {
    // Live run (Claude Code → Qwen3.6, hybrid): the main conversation (16K) was dropped when a
    // ~30K side request with a different system prompt arrived that shares only its first 3
    // tokens. No checkpoint lies at or below 3, so the trunk resumes from 0 — and a tail
    // detached at 3 has lost its holder: it must be detached where the trunk resumes.
    let r_main = reply(30);
    let mut rig = Rig::new(
        ScriptedEngine::new_hybrid(
            script(&[r_main.clone(), reply(10), reply(10), reply(5)]),
            262_144,
            1_000,
        ),
        CacheMode::Cached,
    );
    let main = conv(&[(Role::System, 15_000, 1), (Role::User, 900, 2)]);
    rig.run(&main, 32_768);
    let side = sharing(
        conv(&[(Role::System, 29_000, 3), (Role::User, 1_000, 4)]),
        &main,
        3,
    );
    let (miss, _, _) = rig.run(&side, 2_112);
    assert_eq!(miss.cause, Some(Cause::New), "{miss:?}");
    rig.store.assert_invariants(&rig.engine);
    assert_eq!(rig.store.branches().len(), 1, "the main tail is kept");
    let side2 = sharing(
        conv(&[(Role::System, 29_000, 3), (Role::User, 1_100, 5)]),
        &main,
        3,
    );
    let (miss, _, _) = rig.run(&side2, 10_240);
    assert_eq!(miss.cached, 29_000, "{miss:?}");
    rig.store.assert_invariants(&rig.engine);
    let main2 = next_turn(&main, &r_main, (200, 6));
    let (miss, processed, _) = rig.run(&main2, 32_768);
    assert_eq!(miss.cause, None, "{miss:?}");
    assert_eq!(miss.cached, main.len() + r_main.len() as u32);
    assert!(processed < 300, "{processed}");
    rig.store.assert_invariants(&rig.engine);
    assert_eq!(
        rig.store.stats().evicted_total,
        0,
        "{:?}",
        rig.store.stats()
    );
}
