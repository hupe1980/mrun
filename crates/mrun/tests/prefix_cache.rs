//! `PrefixStore` on `ScriptedEngine` (003 US1–US4): hits, appends, subagent switches that keep
//! the parent, the tiny-tail rule, edits, `--no-cache`, eviction and hybrid checkpoints.

use mrun::cache::miss::{Cause, MissReport};
use mrun::cache::{CacheMode, MIN_BRANCH, PrefixStore};
use mrun::chat::prompt::{MessageMark, RenderedPrompt, Role};
use mrun::runtime::generate::{
    Event, Finished, GenerateRequest, PressureGauge, ReusePolicy, generate,
};
use mrun::runtime::scripted::{SCRIPT_EOS, ScriptedEngine};
use mrun_engine::Engine;

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
            store: PrefixStore::new("key".into(), mode),
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
fn edits_are_changed_and_report_the_message() {
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
    assert_eq!(miss.cause, Some(Cause::Changed));
    assert_eq!(miss.divergence, Some(at as u32));
    let message = miss.message.unwrap();
    assert_eq!((message.index, message.role), (2, Role::Assistant));
    // The tool list changed: inside the system region.
    let mut p3 = p1.clone();
    p3.tokens[10] = 999_998;
    let (miss, _, _) = rig.run(&p3, 100);
    assert_eq!(miss.cause, Some(Cause::Changed));
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
    assert_eq!(miss.cause, Some(Cause::Changed));
    assert_eq!(miss.cached, last_user);
    assert_eq!(miss.gap, 20);
    // A subagent and back: the main tail's end state travels with its segment.
    let sub = conv(&[(Role::System, 300, 1), (Role::User, 400, 7)]);
    let (miss, _, _) = rig.run(&sub, 100);
    assert_eq!(miss.cached, 300, "the system checkpoint");
    // Position-bound state exists only where it was computed: the same prompt resumes from the
    // generation-start checkpoint, one token before `P − 1`.
    let (miss, _, _) = rig.run(&p3, 100);
    assert_eq!(miss.cause, None, "{miss:?}");
    assert_eq!((miss.cached, miss.gap), (p3.generation_start, 1));
}

/// Small deterministic generator (xorshift).
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

/// Random agent traffic under tight memory: appended turns, subagents, title requests, edits of
/// earlier messages, retries and repeats. After every request the engine holds exactly the
/// tokens the store believes (no stale reuse), memory stays within the budget, and the report
/// is consistent.
fn mutation_suite(hybrid: bool, seed: u64) {
    let replies: Vec<Vec<u32>> = (0..200).map(|i| reply(5 + (i % 7) as u32)).collect();
    let engine = if hybrid {
        ScriptedEngine::new_hybrid(script(&replies), 6_000, 1_000)
    } else {
        ScriptedEngine::new(script(&replies), 6_000)
    };
    let mut rig = Rig::new(engine, CacheMode::Cached);
    let mut rng = Rng(seed);
    let mut conversations: Vec<RenderedPrompt> = Vec::new();
    let mut fresh = 100;
    for step in 0..150 {
        let choice = rng.below(10);
        let prompt = if conversations.is_empty() || choice < 2 {
            // A new conversation, sometimes with the shared system prompt.
            fresh += 1;
            let system = if rng.below(2) == 0 { 1 } else { fresh + 500 };
            let p = conv(&[
                (Role::System, 200 + rng.below(200) as u32, system),
                (Role::User, 20 + rng.below(600) as u32, fresh),
            ]);
            conversations.push(p.clone());
            p
        } else {
            let i = rng.below(conversations.len() as u64) as usize;
            let base = conversations[i].clone();
            match choice {
                2..=5 => {
                    fresh += 1;
                    let r = reply(3 + rng.below(20) as u32);
                    next_turn(&base, &r, (10 + rng.below(100) as u32, fresh))
                }
                6 | 7 => {
                    // Edit a token somewhere before the header.
                    let mut p = base.clone();
                    let at = rng.below(u64::from(p.generation_start)) as usize;
                    p.tokens[at] = 900_000 + step;
                    p
                }
                _ => base,
            }
        };
        let (miss, _, finished) = rig.run(&prompt, 30);
        assert!(miss.cached < prompt.len(), "step {step}: {miss:?}");
        assert_eq!(miss.processed, prompt.len() - miss.cached);
        assert_eq!(finished.stats.cached_tokens, miss.cached);
        // The live state starts with exactly the prompt the request sent.
        let live = rig.store.live_tokens();
        let shared = live.len().min(prompt.tokens.len());
        assert_eq!(&live[..shared], &prompt.tokens[..shared], "step {step}");
        conversations.push(prompt);
        if conversations.len() > 12 {
            conversations.remove(0);
        }
    }
    let stats = rig.store.stats().unwrap();
    assert!(
        stats.evicted_total > 0,
        "the run exercised eviction: {stats:?}"
    );
    assert!(stats.cached_tokens_total > 0, "and reuse: {stats:?}");
}

#[test]
fn mutation_suite_attention() {
    for seed in [1, 7, 42, 2026] {
        mutation_suite(false, seed);
    }
}

#[test]
fn mutation_suite_hybrid() {
    for seed in [3, 11, 99, 1015] {
        mutation_suite(true, seed);
    }
}
