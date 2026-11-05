//! The mutation suite (003 SC-004, R-mutations; T024), shared by the `ScriptedEngine` suite
//! (`prefix_cache.rs`) and the `NuvecEngine` one (`prefix_engine.rs`); included with
//! `#[path = "fixtures/mutations.rs"] mod mutations;`.
//!
//! Random agent traffic — appended turns, identical retries, edits of the system prompt, of the
//! first and of later conversation messages and of the last message, rewinds, new conversations,
//! side requests sharing the system prompt (titles), interleaving, cancels and their retries,
//! memory pressure and `cache clear` — under tight memory. Every message starts with a token
//! used nowhere else and every edit writes one, so the longest prefix a prompt shares with any
//! path ever stored (`ideal`) is exact. After every request:
//!
//! - no stale reuse: the live state starts with the request's tokens up to `cached`;
//! - the data-model invariants hold (`assert_invariants`);
//! - when the store matched the ideal prefix: the divergence (if the closest path continues
//!   there), the cause, the message index and role, `gap` and `unreused` are the expected ones;
//! - when it matched less: the cause is `evicted` and `unreused` covers the loss, or (when the
//!   ring's resolution misses the loss) the report is the one for the closest stored path.

#![allow(dead_code)]

use mrun::cache::PrefixStore;
use mrun::cache::miss::{Cause, GAP_CAUSE, MessageRef, MissReport};
use mrun::chat::prompt::{MessageMark, RenderedPrompt, Role};

/// A server under test.
pub trait Rig {
    /// Serves `prompt`; the model replies with `reply` then its EOS. With `cancel`, the request
    /// is cancelled after its first prefill chunk.
    fn serve(&mut self, prompt: &RenderedPrompt, reply: &[u32], cancel: bool) -> MissReport;
    fn store(&self) -> &PrefixStore;
    /// Checks the store's invariants against the engine.
    fn check(&self);
    /// `mrun cache clear`.
    fn clear(&mut self);
    /// A memory-pressure warning between requests.
    fn pressure(&mut self);
    fn hybrid(&self) -> bool;
}

/// Token ranges: `unique` ids are never reused, `filler` ids fill messages and replies.
#[derive(Clone, Copy, Debug)]
pub struct Alphabet {
    pub unique: (u32, u32),
    pub filler: (u32, u32),
}

/// Lengths of messages, scaled to the rig's capacity.
#[derive(Clone, Copy, Debug)]
pub struct Sizes {
    pub system: u32,
    pub message: u32,
    pub reply: u32,
}

struct Gen {
    state: u64,
    next_unique: u32,
    alphabet: Alphabet,
}

impl Gen {
    fn next(&mut self) -> u64 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        self.state
    }

    fn below(&mut self, n: u32) -> u32 {
        (self.next() % u64::from(n.max(1))) as u32
    }

    fn unique(&mut self) -> u32 {
        let id = self.next_unique;
        assert!(id < self.alphabet.unique.1, "unique ids exhausted");
        self.next_unique += 1;
        id
    }

    fn filler(&mut self) -> u32 {
        let (lo, hi) = self.alphabet.filler;
        lo + self.below(hi - lo)
    }

    /// A message of `n` tokens starting with a unique one.
    fn message(&mut self, n: u32) -> Vec<u32> {
        let mut tokens = vec![self.unique()];
        tokens.extend((1..n.max(1)).map(|_| self.filler()));
        tokens
    }
}

/// Appends a message to `prompt` (before its generation header, which is re-added).
fn push(prompt: &mut RenderedPrompt, role: Role, tokens: Vec<u32>, header: [u32; 2]) {
    prompt.tokens.truncate(prompt.generation_start as usize);
    let index = prompt.messages.len() as u32;
    prompt.messages.push(MessageMark {
        start: prompt.tokens.len() as u32,
        role,
        message: Some(index),
    });
    prompt.tokens.extend(tokens);
    prompt.generation_start = prompt.tokens.len() as u32;
    prompt.tokens.extend(header);
}

fn lcp(a: &[u32], b: &[u32]) -> usize {
    a.iter().zip(b).take_while(|(x, y)| x == y).count()
}

/// A conversation: its last prompt and the path the store committed for it.
#[derive(Clone)]
struct Conversation {
    prompt: RenderedPrompt,
    committed: Vec<u32>,
}

/// What the suite saw.
#[derive(Debug, Default)]
pub struct Summary {
    pub requests: usize,
    pub exact: usize,
    pub evicted: usize,
    /// Requests that lost state the ring could not name (classified against what is stored).
    pub unseen: usize,
    pub by_cause: std::collections::BTreeMap<&'static str, usize>,
}

/// The expected report when the store matched the ideal prefix `m` (`continues`: the closest
/// path has a token at `m`).
fn expected_cause(prompt: &RenderedPrompt, m: u32, continues: bool, gap: u32) -> Option<Cause> {
    let p = prompt.len();
    if continues && m < p {
        let marks = &prompt.messages;
        let whole = marks.iter().enumerate().any(|(i, mark)| {
            let end = marks
                .get(i + 1)
                .map_or(prompt.generation_start, |n| n.start);
            mark.role != Role::System && mark.start >= prompt.system_tools_end && end <= m
        });
        return Some(if whole { Cause::Diverged } else { Cause::New });
    }
    if gap > GAP_CAUSE {
        Some(Cause::Gap)
    } else if m <= prompt.system_tools_end && p > prompt.system_tools_end.max(1) {
        Some(Cause::New)
    } else {
        None
    }
}

fn expected_message(prompt: &RenderedPrompt, at: u32) -> MessageRef {
    let mark = prompt
        .messages
        .iter()
        .rev()
        .find(|m| m.start <= at)
        .expect("a mark");
    MessageRef {
        index: mark.message.unwrap(),
        role: if at < prompt.system_tools_end {
            Role::System
        } else {
            mark.role
        },
    }
}

/// Runs `requests` random requests on `rig`; panics on the first wrong report.
pub fn run(
    rig: &mut dyn Rig,
    seed: u64,
    requests: usize,
    alphabet: Alphabet,
    sizes: Sizes,
) -> Summary {
    let mut g = Gen {
        state: seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1,
        next_unique: alphabet.unique.0,
        alphabet,
    };
    let header = [g.filler(), g.filler()];
    let shared_system = g.message(sizes.system);
    // Every path the store has held since the last clear.
    let mut history: Vec<Vec<u32>> = Vec::new();
    let mut conversations: Vec<Conversation> = Vec::new();
    let mut summary = Summary::default();
    let new_conversation = |g: &mut Gen, shared: bool, user: u32| {
        let system = if shared {
            shared_system.clone()
        } else {
            let n = sizes.system / 2 + g.below(sizes.system);
            g.message(n)
        };
        let mut prompt = RenderedPrompt {
            tokens: Vec::new(),
            messages: Vec::new(),
            system_tools_end: 0,
            generation_start: 0,
        };
        push(&mut prompt, Role::System, system, header);
        prompt.system_tools_end = prompt.generation_start;
        let user = g.message(user);
        push(&mut prompt, Role::User, user, header);
        prompt
    };
    for step in 0..requests {
        let what = g.below(100);
        let mut cancel = false;
        let mut target: Option<usize> = None;
        if what < 3 && step > 0 {
            rig.clear();
            history.clear();
        } else if what < 6 && step > 0 {
            rig.pressure();
        }
        let prompt = if conversations.is_empty() || what < 14 {
            let shared = g.below(2) == 0;
            let user = sizes.message / 2 + g.below(sizes.message * 2);
            new_conversation(&mut g, shared, user)
        } else if what < 20 {
            // A side request (title) sharing the system prompt.
            let n = 8 + g.below(24);
            new_conversation(&mut g, true, n)
        } else {
            let i = g.below(conversations.len() as u32) as usize;
            target = Some(i);
            let c = conversations[i].clone();
            let mut p = c.prompt.clone();
            let gen_start = p.generation_start as usize;
            let mut return_retry = false;
            match what {
                20..=54 => {
                    // The next turn: the committed path (prompt and reply, spliced) and a new
                    // user message or tool result.
                    if c.committed.len() > gen_start {
                        p.messages.push(MessageMark {
                            start: p.generation_start,
                            role: Role::Assistant,
                            message: Some(p.messages.len() as u32),
                        });
                        p.tokens = c.committed.clone();
                        p.generation_start = p.tokens.len() as u32;
                    }
                    let role = if g.below(2) == 0 {
                        Role::User
                    } else {
                        Role::Tool
                    };
                    let n = 8 + g.below(sizes.message);
                    let message = g.message(n);
                    push(&mut p, role, message, header);
                    p
                }
                55..=61 => p, // an identical retry
                62..=65 => {
                    // An edit of the system prompt.
                    let at = g.below(p.system_tools_end.max(1)) as usize;
                    p.tokens[at] = g.unique();
                    p
                }
                66..=77 => {
                    // An edit anywhere in the conversation (earlier, first or last message).
                    let from = p.system_tools_end as usize;
                    let at = from + g.below((gen_start - from) as u32) as usize;
                    p.tokens[at] = g.unique();
                    p
                }
                78..=85 => {
                    // A rewind to an earlier message, then a new user message.
                    let starts: Vec<u32> = p
                        .messages
                        .iter()
                        .filter(|m| m.start > p.system_tools_end)
                        .map(|m| m.start)
                        .collect();
                    if starts.is_empty() {
                        return_retry = true;
                    }
                    let to = if return_retry {
                        p.generation_start
                    } else {
                        starts[g.below(starts.len() as u32) as usize]
                    };
                    p.tokens.truncate(to as usize);
                    p.messages.retain(|m| m.start < to);
                    p.generation_start = to;
                    let n = 8 + g.below(sizes.message);
                    let message = g.message(n);
                    push(&mut p, Role::User, message, header);
                    p
                }
                _ => {
                    // Cancelled after its first chunk; the conversation retries it later.
                    cancel = true;
                    p
                }
            }
        };
        let reply: Vec<u32> = (0..1 + g.below(sizes.reply)).map(|_| g.filler()).collect();
        // The ideal match over every path held since the last clear.
        let p = prompt.len();
        let (mut ideal, mut ends, mut continues) = (0usize, false, false);
        for path in &history {
            let l = lcp(path, &prompt.tokens);
            if l > ideal {
                (ideal, ends, continues) = (l, false, false);
            }
            if l == ideal {
                ends |= path.len() == l;
                continues |= path.len() > l;
            }
        }
        let ideal = ideal as u32;
        let miss = rig.serve(&prompt, &reply, cancel);
        rig.check();
        let live = rig.store().live_tokens();
        let reused = (miss.cached as usize).min(live.len());
        assert_eq!(
            &live[..reused],
            &prompt.tokens[..reused],
            "step {step}: no stale reuse"
        );
        assert!(miss.cached < p && miss.processed == p - miss.cached);
        let matched = miss.cached + miss.gap;
        let cap = ideal.min(p - 1);
        let ctx = format!("step {step} (kind {what}): ideal {ideal} of {p}, {miss:?}");
        if std::env::var_os("MUTATIONS_DEBUG").is_some() {
            eprintln!("{ctx} branches {:?}", rig.store().branches());
        }
        assert!(matched <= cap, "{ctx}: matched more than any stored path");
        assert!(!miss.ring_wrapped, "{ctx}");
        if matched == cap {
            summary.exact += 1;
            let at = (continues && ideal < p).then_some(ideal);
            let alt = (ends || ideal >= p).then_some(None);
            assert!(
                miss.divergence == at || alt == Some(miss.divergence),
                "{ctx}: divergence"
            );
            let cause = expected_cause(&prompt, ideal, miss.divergence.is_some(), miss.gap);
            assert_eq!(miss.cause, cause, "{ctx}: cause");
            if let Some(at) = miss.divergence {
                assert_eq!(miss.message, Some(expected_message(&prompt, at)), "{ctx}");
            } else {
                assert_eq!(miss.message, None, "{ctx}");
            }
            if !rig.hybrid() {
                assert_eq!(miss.gap, 0, "{ctx}: attention resumes at the match");
            }
            assert_eq!(miss.unreused, miss.gap, "{ctx}: unreused");
        } else if miss.cause == Some(Cause::Evicted) {
            summary.evicted += 1;
            assert!(
                miss.unreused >= miss.gap && miss.unreused <= cap - miss.cached,
                "{ctx}: unreused"
            );
        } else {
            // Lost state the ring cannot see: it keeps prefix hashes one token past the drop
            // point, at multiples of 256 and at the full length (FR-018), so a prompt sharing a
            // dropped path only between those points is classified against what is stored.
            summary.unseen += 1;
            assert_eq!(miss.unreused, miss.gap, "{ctx}: unreused");
            let cause = expected_cause(&prompt, matched, miss.divergence.is_some(), miss.gap);
            assert_eq!(miss.cause, cause, "{ctx}: cause of an unseen loss");
        }
        *summary
            .by_cause
            .entry(miss.cause.map_or("hit", Cause::as_str))
            .or_default() += 1;
        summary.requests += 1;
        let committed = rig.store().live_tokens().to_vec();
        history.push(committed.clone());
        let conversation = Conversation {
            prompt: prompt.clone(),
            committed,
        };
        match (cancel, target) {
            // A cancelled request is retried as it was.
            (true, Some(_)) => {}
            (false, Some(i)) => conversations[i] = conversation,
            _ => conversations.push(conversation),
        }
        if conversations.len() > 10 {
            conversations.remove(0);
        }
    }
    summary
}
