//! Prefix cache v3 (003, D031, D053): the only reuse policy, used by `mrun serve` and `mrun run`.
//!
//! The engine's live state is the *trunk*. Other conversations are *branches*: each holds its full
//! token sequence, the trunk position `at` it was detached at, and an engine segment with
//! positions `[at, end)` (plus, for hybrid models, the position-bound state at `end` and its
//! checkpoints above `at`). Branches are flat: a branch is reachable while the trunk or another
//! branch (its holder chain) holds its prefix `[0, at)`; branches that lose it are dropped.
//!
//! A request moves state, never copies it: the trunk's tail is detached where the request leaves
//! it (kept as a branch when at least [`MIN_BRANCH`] tokens, else dropped), the matched branch's
//! chain is attached, and the trunk is cut back to the match. Eviction drops one-shot branches
//! before *protected* ones — conversations that ran two or more turns and were used within the
//! last [`PROTECT_REQUESTS`] requests (the parent a subagent returns to); a conversation's turn
//! count moves with its state — and least recently used first within each group. Reply records
//! for splicing (D051) live on the path holding their tokens. Nothing is written to disk.

pub mod chunks;
pub mod miss;
pub mod schedule;

use crate::chat::prompt::{RenderedPrompt, Role};
use crate::chat::splice::{HeldPath, Ledger, Record};
use crate::runtime::pressure::Pressure;
use miss::{Found, MissReport};
use mrun_engine::{Checkpoint, CheckpointId, Engine, EngineError, Segment, SegmentId};
use schedule::Schedule;
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex, PoisonError};

/// Shortest tail kept as a branch when the trunk is cut; shorter tails are dropped.
pub const MIN_BRANCH: u32 = 256;
/// Requests a multi-turn branch stays protected after its last use.
pub const PROTECT_REQUESTS: u64 = 16;
/// Dropped paths remembered for the `evicted` cause and `unreused`.
pub const DROPPED_RING: usize = 1024;
/// Spacing of the prefix hashes a ring entry keeps beyond its drop point.
const RING_STEP: usize = 256;
/// Share of the state capacity stored checkpoints may use (hybrid models).
const CHECKPOINT_SHARE: u64 = 4;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CacheMode {
    Cached,
    /// `--no-cache`: every request starts from an empty state.
    Off,
}

/// `mrun ps` and `/mrun/status` figures.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct CacheStats {
    pub live_tokens: u32,
    pub branches: u32,
    pub branch_tokens: u64,
    pub buffer_bytes: u64,
    pub checkpoints: u32,
    pub checkpoint_bytes: u64,
    pub prompt_tokens_total: u64,
    pub cached_tokens_total: u64,
    /// Sum of the requests' `unreused` tokens (D053, D056).
    pub unreused_tokens_total: u64,
    pub evicted_total: u64,
    /// The dropped-path ring has overwritten entries since the last clear.
    pub ring_wrapped: bool,
    /// Reply records on stored paths (D051).
    pub records: u32,
    pub pressure_events: u64,
    pub last: Option<MissReport>,
}

/// Result of `mrun cache clear` (runs between requests).
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ClearReport {
    pub branches_dropped: u32,
    pub bytes_freed: u64,
}

/// What `prepare` leaves for the generation loop.
#[derive(Clone, Debug, PartialEq)]
pub struct Prepared {
    /// Prompt tokens the live state holds; prefill starts here.
    pub resume: u32,
    /// Ends of the prefill chunks ([`chunks::plan`]).
    pub chunk_ends: Vec<u32>,
    /// Positions where a checkpoint is captured after the chunk ending there (hybrids).
    pub checkpoints: Vec<u32>,
    pub miss: MissReport,
}

/// The mark a checkpoint was captured at (D053).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Mark {
    /// End of system + tools.
    Anchor,
    /// Start of a user message with non-tool-result content.
    User,
    /// Start of the last tool result.
    Tool,
    /// Start of the generation header: the next turn re-renders the reply from here (D051 splices
    /// only the current turn), so a follow-up resumes here instead of at its user message.
    Reply,
    /// The prompt end − 1.
    End,
}

#[derive(Debug)]
struct Branch {
    at: u32,
    /// The whole conversation: `tokens[..at]` is shared with the trunk.
    tokens: Arc<Vec<u32>>,
    segment: Segment,
    /// By position.
    checkpoints: Vec<Checkpoint>,
    /// Replies whose end lies above `at`.
    records: Vec<Record>,
    /// Forward schedule of the whole path (FR-015); `[..at]` is the prefix it was computed on.
    schedule: Schedule,
    last_used: u64,
    turns: u32,
}

impl Branch {
    fn bytes(&self) -> u64 {
        self.segment.bytes + self.checkpoints.iter().map(|c| c.bytes).sum::<u64>()
    }
}

/// Lookup result.
#[derive(Debug, Default)]
struct Plan {
    matched: u32,
    /// Segment ids of the branches to attach, in order (empty: the trunk is the best match).
    chain: Vec<u64>,
    /// The matched path has a token at `matched` that differs from the prompt.
    diverges: bool,
}

/// A dropped path: `(length, hash)` of its prefixes one token past the drop point, at every
/// multiple of [`RING_STEP`] above it, and at its full length.
#[derive(Debug)]
struct Dropped {
    points: Vec<(u32, u64)>,
}

#[derive(Debug)]
pub struct PrefixStore {
    mode: CacheMode,
    /// Tokens the engine's live state holds.
    trunk: Vec<u32>,
    /// Consecutive turns the trunk's conversation has run.
    turns: u32,
    /// The current request extends the path it resumed (decided in `prepare`).
    continues: bool,
    /// Trunk checkpoints, by position.
    checkpoints: Vec<Checkpoint>,
    /// Forward schedule of the live state (FR-015).
    schedule: Schedule,
    /// `prepare`'s resume position and chunk ends, for `commit` to record.
    planned: Option<(u32, Vec<u32>)>,
    /// Reply records on the trunk.
    records: Vec<Record>,
    branches: Vec<Branch>,
    /// Marks of stored checkpoints (trunk and branches).
    marks: HashMap<CheckpointId, Mark>,
    dropped: VecDeque<Dropped>,
    ring_wrapped: bool,
    ledger: Arc<Mutex<Ledger>>,
    requests: u64,
    stats: CacheStats,
}

fn common(a: &[u32], b: &[u32]) -> u32 {
    a.iter().zip(b).take_while(|(x, y)| x == y).count() as u32
}

const HASH_SEED: u64 = 0xcbf2_9ce4_8422_2325;

fn hash_step(h: u64, token: u32) -> u64 {
    (h ^ u64::from(token)).wrapping_mul(0x0000_0100_0000_01b3)
}

/// Rolling hashes of every prefix: `h[i]` covers `tokens[..i]`.
fn prefix_hashes(tokens: &[u32]) -> Vec<u64> {
    let mut hashes = Vec::with_capacity(tokens.len() + 1);
    let mut h = HASH_SEED;
    hashes.push(h);
    for &t in tokens {
        h = hash_step(h, t);
        hashes.push(h);
    }
    hashes
}

/// Adds `record` unless an equal one is there.
fn add_record(records: &mut Vec<Record>, record: Record) {
    if !records.contains(&record) {
        records.push(record);
    }
}

impl PrefixStore {
    pub fn new(mode: CacheMode) -> Self {
        Self {
            mode,
            trunk: Vec::new(),
            turns: 0,
            continues: false,
            checkpoints: Vec::new(),
            schedule: Schedule::new(),
            planned: None,
            records: Vec::new(),
            branches: Vec::new(),
            marks: HashMap::new(),
            dropped: VecDeque::new(),
            ring_wrapped: false,
            ledger: Arc::new(Mutex::new(Ledger::default())),
            requests: 0,
            stats: CacheStats::default(),
        }
    }

    /// The reply records the api layer splices from (D051); shared with the renderer.
    pub fn ledger(&self) -> Arc<Mutex<Ledger>> {
        self.ledger.clone()
    }

    /// Tokens the live state holds (tests check it against the engine).
    pub fn live_tokens(&self) -> &[u32] {
        &self.trunk
    }

    /// Checkpoints along the live state, by position.
    pub fn checkpoints(&self) -> &[Checkpoint] {
        &self.checkpoints
    }

    /// Forward schedule of the live state (FR-015), for the exactness oracle (SC-005).
    #[cfg(any(test, feature = "test-support"))]
    pub fn schedule(&self) -> &[schedule::Span] {
        &self.schedule
    }

    /// Stored branches as `(at, len)`, sorted, for tests.
    pub fn branches(&self) -> Vec<(u32, u32)> {
        let mut out: Vec<(u32, u32)> = self
            .branches
            .iter()
            .map(|b| (b.at, b.tokens.len() as u32))
            .collect();
        out.sort_unstable();
        out
    }

    /// Asserts the data-model invariants 1–3, 5, 8 and 9 (and 4's bound between operations);
    /// the suites call it after every request.
    #[cfg(any(test, feature = "test-support"))]
    pub fn assert_invariants(&self, engine: &dyn Engine) {
        let len = self.trunk.len() as u32;
        assert_eq!(engine.len(), len, "1: the store mirrors the live state");
        let holders = self.holders();
        let mut segments = Vec::new();
        for (b, chain) in self.branches.iter().zip(&holders) {
            let end = b.tokens.len() as u32;
            assert!(chain.is_some(), "2: the branch at {} is reachable", b.at);
            assert_eq!(
                (b.segment.start, b.segment.end),
                (b.at, end),
                "2: the segment covers the branch's range"
            );
            segments.push(b.segment.id);
            for c in &b.checkpoints {
                assert!(c.position > b.at && c.position <= end, "5: on its range");
            }
            for r in &b.records {
                assert!(
                    r.start < r.end && r.end > b.at && r.end <= end,
                    "9: on its range"
                );
            }
            assert!(
                schedule::is_contiguous(&b.schedule, end),
                "8: the branch's schedule covers it: {:?}",
                b.schedule
            );
        }
        segments.sort_unstable();
        segments.dedup();
        assert_eq!(
            segments.len(),
            self.branches.len(),
            "3: one segment per branch"
        );
        assert!(
            engine.allocated() <= engine.budget().state_capacity_bytes,
            "4: within the state capacity"
        );
        for c in &self.checkpoints {
            assert!(c.position <= len, "5: trunk checkpoints lie on the trunk");
        }
        let checkpoint_bytes: u64 = self
            .checkpoints
            .iter()
            .chain(self.branches.iter().flat_map(|b| &b.checkpoints))
            .map(|c| c.bytes)
            .sum();
        assert!(
            checkpoint_bytes <= engine.budget().state_capacity_bytes / CHECKPOINT_SHARE,
            "5: checkpoints within their budget"
        );
        for r in &self.records {
            assert!(
                r.start < r.end && r.end <= len,
                "9: trunk records lie on the trunk"
            );
        }
        assert!(
            schedule::is_contiguous(&self.schedule, len),
            "8: the trunk's schedule covers it: {:?}",
            self.schedule
        );
    }

    /// Consecutive turns of the trunk's conversation and of each branch, for tests.
    pub fn turns(&self) -> (u32, Vec<u32>) {
        (self.turns, self.branches.iter().map(|b| b.turns).collect())
    }

    pub fn stats(&self) -> CacheStats {
        self.stats
    }

    fn hybrid(engine: &dyn Engine) -> bool {
        engine.info().recurrent_bytes > 0
    }

    /// For each branch, the branches to attach, in order, before it can be attached: none when
    /// the trunk holds its prefix, else the shortest chain of a branch that holds it. `None` for
    /// an orphan. One pass in order of detach position (a holder detaches before what it holds).
    fn holders(&self) -> Vec<Option<Vec<u64>>> {
        let mut order: Vec<usize> = (0..self.branches.len()).collect();
        order.sort_by_key(|&i| self.branches[i].at);
        let mut chains: Vec<Option<Vec<u64>>> = vec![None; self.branches.len()];
        for &b in &order {
            let branch = &self.branches[b];
            let at = branch.at as usize;
            let own = schedule::prefix(&branch.schedule, &branch.tokens, branch.at);
            // A holder has the prefix's tokens and computed them by the same forward passes.
            let holds = |tokens: &[u32], sched: &[schedule::Span]| {
                tokens.len() >= at
                    && tokens[..at] == branch.tokens[..at]
                    && schedule::prefix(sched, tokens, branch.at) == own
            };
            chains[b] = if holds(&self.trunk, &self.schedule) {
                Some(Vec::new())
            } else {
                self.branches
                    .iter()
                    .enumerate()
                    .filter(|(_, c)| c.at < branch.at && holds(&c.tokens, &c.schedule))
                    .filter_map(|(i, c)| {
                        let mut chain = chains[i].clone()?;
                        chain.push(c.segment.id.0);
                        Some(chain)
                    })
                    .min_by_key(Vec::len)
            };
        }
        chains
    }

    /// Longest exact match over the trunk and the branches (through their holders); an equal
    /// match keeps the shorter move (the trunk over any branch, fewer attaches over more).
    fn lookup(&self, prompt: &[u32]) -> Plan {
        let t = common(&self.trunk, prompt);
        let mut best = Plan {
            matched: t,
            chain: Vec::new(),
            diverges: (t as usize) < self.trunk.len() && (t as usize) < prompt.len(),
        };
        let holders = self.holders();
        for (i, b) in self.branches.iter().enumerate() {
            let m = common(&b.tokens, prompt);
            if m <= b.at || m < best.matched {
                continue;
            }
            let Some(mut chain) = holders[i].clone() else {
                continue;
            };
            chain.push(b.segment.id.0);
            if m > best.matched || chain.len() < best.chain.len() {
                best = Plan {
                    matched: m,
                    chain,
                    diverges: (m as usize) < b.tokens.len() && (m as usize) < prompt.len(),
                };
            }
        }
        best
    }

    /// Against the dropped-path ring: whether the prompt continues a dropped path beyond
    /// `matched` (`evicted`), and the longest prefix it shares with one (to [`RING_STEP`]).
    /// A request reuses at most `P − 1` tokens, so a loss past that costs nothing.
    fn scan_dropped(&self, prompt: &[u32], matched: u32) -> (bool, u32) {
        if self.dropped.is_empty() {
            return (false, 0);
        }
        let reusable = (prompt.len() as u32).saturating_sub(1);
        let hashes = prefix_hashes(prompt);
        let (mut evicted, mut shared) = (false, 0);
        for entry in &self.dropped {
            // Prefix matches are monotone: every point up to the longest shared one matches.
            let n = entry.points.partition_point(|&(len, hash)| {
                (len as usize) < hashes.len() && hashes[len as usize] == hash
            });
            if n == 0 {
                continue;
            }
            // The entry continues past the match at any of its points: the prompt continues
            // state that was dropped.
            shared = shared.max(entry.points[n - 1].0);
            evicted |= entry.points[n - 1].0.min(reusable) > matched;
        }
        (evicted, shared)
    }

    /// Remembers that the path `tokens` lost its state from `from` on: a later prompt that
    /// matches it one token beyond `from` continues dropped state.
    fn remember_dropped(&mut self, tokens: &[u32], from: u32) {
        let first = from as usize + 1;
        if first > tokens.len() {
            return;
        }
        let mut points = Vec::new();
        let mut h = HASH_SEED;
        for (i, &t) in tokens.iter().enumerate() {
            h = hash_step(h, t);
            let len = i + 1;
            if len == first || (len > first && (len % RING_STEP == 0 || len == tokens.len())) {
                points.push((len as u32, h));
            }
        }
        if self.dropped.len() == DROPPED_RING {
            self.dropped.pop_front();
            self.ring_wrapped = true;
        }
        self.dropped.push_back(Dropped { points });
        self.stats.evicted_total += 1;
    }

    /// Eviction rank: one-shot or stale branches first, then least recently used.
    fn rank(&self, b: &Branch) -> (bool, u64) {
        let protected = b.turns >= 2 && self.requests - b.last_used < PROTECT_REQUESTS;
        (protected, b.last_used)
    }

    fn drop_branch(&mut self, engine: &mut dyn Engine, index: usize) -> u64 {
        let b = self.branches.swap_remove(index);
        engine.drop_segment(b.segment.id);
        for c in &b.checkpoints {
            engine.drop_checkpoint(c.id);
        }
        self.remember_dropped(&b.tokens, b.at);
        b.bytes()
    }

    /// Drops branches (lowest rank first, never one in `keep`) until `bytes` are freed.
    fn evict_branches(&mut self, engine: &mut dyn Engine, bytes: u64, keep: &[u64]) -> u64 {
        let mut freed = 0;
        while freed < bytes {
            let Some(index) = self
                .branches
                .iter()
                .enumerate()
                .filter(|(_, b)| !keep.contains(&b.segment.id.0))
                .min_by_key(|(_, b)| self.rank(b))
                .map(|(i, _)| i)
            else {
                break;
            };
            freed += self.drop_branch(engine, index);
        }
        freed
    }

    /// Drops trunk checkpoints above `at`.
    fn retain_checkpoints(&mut self, engine: &mut dyn Engine, at: u32) {
        let (keep, gone): (Vec<_>, Vec<_>) =
            self.checkpoints.drain(..).partition(|c| c.position <= at);
        for c in gone {
            engine.drop_checkpoint(c.id);
        }
        self.checkpoints = keep;
    }

    /// Deepest trunk checkpoint at or below `at`.
    fn checkpoint_at_or_below(&self, at: u32) -> Option<Checkpoint> {
        self.checkpoints
            .iter()
            .filter(|c| c.position <= at)
            .max_by_key(|c| c.position)
            .copied()
    }

    /// Truncates the trunk to `len` with the records it no longer holds.
    fn truncate_trunk(&mut self, len: u32) {
        schedule::truncate(&mut self.schedule, &self.trunk, len);
        self.trunk.truncate(len as usize);
        self.records.retain(|r| r.end <= len);
    }

    /// Leaves position-bound state at the trunk's end: the deepest checkpoint at or below it
    /// (evicting branches when restoring needs memory), or an empty state.
    fn resume_state(&mut self, engine: &mut dyn Engine) -> Result<(), EngineError> {
        let at = self.trunk.len() as u32;
        if let Some(c) = self.checkpoint_at_or_below(at) {
            let mut restored = engine.restore(c.id);
            if let Err(EngineError::NeedsMemory { bytes }) = restored
                && self.evict_branches(engine, bytes, &[]) > 0
            {
                restored = engine.restore(c.id);
            }
            match restored {
                Ok(()) => {
                    self.truncate_trunk(c.position);
                    return Ok(());
                }
                Err(EngineError::NeedsMemory { .. }) => {}
                Err(error) => return Err(error),
            }
        }
        engine.trim(0)?;
        self.truncate_trunk(0);
        Ok(())
    }

    /// Cuts the trunk to `at`. A tail of at least [`MIN_BRANCH`] tokens becomes a branch (with
    /// the trunk's turn count and the records above `at`); otherwise it is dropped. With
    /// `attach_follows` the tail is always detached (an attach supplies position-bound state
    /// next); otherwise models with such state resume from a checkpoint. Branches in `keep_ids`
    /// are never evicted to find memory. Returns whether the live position-bound state went with
    /// the tail.
    fn cut(
        &mut self,
        engine: &mut dyn Engine,
        at: u32,
        attach_follows: bool,
        keep_ids: &[u64],
    ) -> Result<bool, EngineError> {
        let len = self.trunk.len() as u32;
        if at >= len {
            return Ok(false);
        }
        let hybrid = Self::hybrid(engine);
        // Without an attach, a hybrid trunk resumes from the deepest checkpoint at or below `at`
        // (or from 0): the tail is detached there, or the trunk would no longer hold its prefix
        // and the branch would be pruned as an orphan right away (the live Claude Code run: a
        // side request sharing 3 tokens dropped the main conversation).
        // The prompt shares the dropped path up to `shared`: the miss classification counts a
        // continuation of it from there (the re-prefilled `[at, shared)` is not lost).
        let shared = at;
        let at = if hybrid && !attach_follows {
            self.checkpoint_at_or_below(at).map_or(0, |c| c.position)
        } else {
            at
        };
        let keep = self.mode == CacheMode::Cached && len - at >= MIN_BRANCH;
        if keep || (attach_follows && hybrid) {
            let mut detached = engine.detach_tail(at);
            if let Err(EngineError::NeedsMemory { bytes }) = detached
                && self.evict_branches(engine, bytes, keep_ids) > 0
            {
                detached = engine.detach_tail(at);
            }
            match detached {
                Ok(segment) => {
                    let (stay, moved): (Vec<_>, Vec<_>) =
                        self.checkpoints.drain(..).partition(|c| c.position <= at);
                    self.checkpoints = stay;
                    let (stay, records): (Vec<_>, Vec<_>) =
                        self.records.drain(..).partition(|r| r.end <= at);
                    self.records = stay;
                    let tokens = std::mem::take(&mut self.trunk);
                    self.trunk = tokens[..at as usize].to_vec();
                    let whole = self.schedule.clone();
                    schedule::truncate(&mut self.schedule, &tokens, at);
                    if keep {
                        self.branches.push(Branch {
                            at,
                            tokens: Arc::new(tokens),
                            segment,
                            checkpoints: moved,
                            records,
                            schedule: whole,
                            last_used: self.requests,
                            turns: self.turns,
                        });
                    } else {
                        engine.drop_segment(segment.id);
                        for c in &moved {
                            engine.drop_checkpoint(c.id);
                        }
                        self.remember_dropped(&tokens, at);
                    }
                    self.turns = 0;
                    return Ok(hybrid);
                }
                Err(EngineError::NeedsMemory { .. }) if !attach_follows => {}
                Err(error) => return Err(error),
            }
        }
        let dropped = self.trunk.clone();
        self.remember_dropped(&dropped, shared);
        self.retain_checkpoints(engine, at);
        self.turns = 0;
        self.truncate_trunk(at);
        if hybrid {
            self.resume_state(engine)?;
        } else {
            engine.trim(at)?;
        }
        Ok(false)
    }

    /// Attaches the branch with segment `id` at the trunk's end; the trunk takes its turn count
    /// and records. Branches in `keep` are never evicted to find memory. `false` when memory
    /// could not be found.
    fn attach(
        &mut self,
        engine: &mut dyn Engine,
        id: u64,
        keep: &[u64],
    ) -> Result<bool, EngineError> {
        let id = SegmentId(id);
        let mut attached = engine.attach(id);
        if let Err(EngineError::NeedsMemory { bytes }) = attached
            && self.evict_branches(engine, bytes, keep) > 0
        {
            attached = engine.attach(id);
        }
        match attached {
            Ok(()) => {
                let index = self
                    .branches
                    .iter()
                    .position(|b| b.segment.id == id)
                    .expect("kept");
                let b = self.branches.swap_remove(index);
                self.trunk = Arc::unwrap_or_clone(b.tokens);
                // The holder chain guarantees the trunk's prefix schedule is the branch's.
                self.schedule = b.schedule;
                self.checkpoints.extend(b.checkpoints);
                self.checkpoints.sort_by_key(|c| c.position);
                for record in b.records {
                    add_record(&mut self.records, record);
                }
                self.turns = b.turns;
                Ok(true)
            }
            Err(EngineError::NeedsMemory { .. }) => Ok(false),
            Err(error) => Err(error),
        }
    }

    /// Drops branches that no path leads to any more (orphans) or that the trunk retraces
    /// entirely (their records then belong to the trunk).
    fn prune(&mut self, engine: &mut dyn Engine) {
        loop {
            let holders = self.holders();
            let doomed = (0..self.branches.len()).find_map(|i| {
                let b = &self.branches[i];
                let retraced = b.tokens.len() <= self.trunk.len()
                    && self.trunk[..b.tokens.len()] == b.tokens[..];
                if retraced {
                    Some((i, false))
                } else {
                    holders[i].is_none().then_some((i, true))
                }
            });
            let Some((i, orphan)) = doomed else {
                break;
            };
            let b = self.branches.swap_remove(i);
            engine.drop_segment(b.segment.id);
            for c in &b.checkpoints {
                engine.drop_checkpoint(c.id);
            }
            // The trunk holds a retraced branch; an orphan's state is gone.
            if orphan {
                self.remember_dropped(&b.tokens, b.at);
            } else {
                for record in b.records {
                    add_record(&mut self.records, record);
                }
            }
        }
    }

    /// The mark of a checkpoint position in `prompt`.
    fn mark_of(prompt: &RenderedPrompt, position: u32) -> Mark {
        if position == prompt.system_tools_end {
            return Mark::Anchor;
        }
        if position == prompt.generation_start {
            return Mark::Reply;
        }
        match prompt.messages.iter().find(|m| m.start == position) {
            Some(m) if m.role == Role::User => Mark::User,
            Some(m) if m.role == Role::Tool => Mark::Tool,
            _ => Mark::End,
        }
    }

    /// Checkpoints to capture during this prefill (hybrids only), at API-neutral marks: the end
    /// of system and tools, each user message with non-tool-result content (tool results are
    /// `Role::Tool` marks whatever the API), the last tool result's start (retried tool calls
    /// resume there), the generation header's start (follow-ups, whose re-rendered reply
    /// diverges right after it) and the prompt end − 1 (retries and regenerations). Capturing inside a
    /// chunk needs a segmented scan in the engine (SC-009); until then these marks end chunks
    /// (FR-005). A function of the prompt only; a request captures the marks above its resume.
    pub fn checkpoint_marks(engine: &dyn Engine, prompt: &RenderedPrompt) -> Vec<u32> {
        if !Self::hybrid(engine) {
            return Vec::new();
        }
        let last_tool = prompt
            .messages
            .iter()
            .rev()
            .find(|m| m.role == Role::Tool)
            .map(|m| m.start);
        let mut at: Vec<u32> = prompt
            .messages
            .iter()
            .filter(|m| m.role == Role::User)
            .map(|m| m.start)
            .chain(last_tool)
            .chain([
                prompt.system_tools_end,
                prompt.generation_start,
                prompt.len().saturating_sub(1),
            ])
            .filter(|&p| p > 0 && p < prompt.len())
            .collect();
        at.sort_unstable();
        at.dedup();
        at
    }

    /// The protected checkpoints of one path: its system + tools anchor, its newest user message
    /// and its newest generation header.
    fn protected_on(&self, checkpoints: &[Checkpoint]) -> [Option<CheckpointId>; 3] {
        let of = |mark: Mark| {
            checkpoints
                .iter()
                .filter(|c| self.marks.get(&c.id) == Some(&mark))
                .max_by_key(|c| c.position)
                .map(|c| c.id)
        };
        [of(Mark::Anchor), of(Mark::User), of(Mark::Reply)]
    }

    /// The checkpoint the budget drops next: unprotected ones by the smallest gap to their
    /// predecessor on their path; then protected ones, least recently used path first (the
    /// trunk last), so the protected set shrinks to fit. `(branch, index)`; `None` = the trunk.
    fn thinnest(&self, protected_too: bool) -> Option<(Option<usize>, usize)> {
        let paths = std::iter::once((None, &self.checkpoints, u64::MAX)).chain(
            self.branches
                .iter()
                .enumerate()
                .map(|(i, b)| (Some(i), &b.checkpoints, b.last_used)),
        );
        // `((protected, path recency, gap), branch, index)`; the smallest key goes first.
        type Candidate = ((bool, u64, u32), Option<usize>, usize);
        let mut best: Option<Candidate> = None;
        for (owner, checkpoints, recency) in paths {
            let protected = self.protected_on(checkpoints);
            let mut previous = 0;
            for (i, c) in checkpoints.iter().enumerate() {
                let gap = c.position.saturating_sub(previous);
                previous = c.position;
                let key = if protected.contains(&Some(c.id)) {
                    if !protected_too {
                        continue;
                    }
                    (true, recency, gap)
                } else {
                    (false, 0, gap)
                };
                if best.as_ref().is_none_or(|(k, _, _)| key < *k) {
                    best = Some((key, owner, i));
                }
            }
        }
        best.map(|(_, owner, i)| (owner, i))
    }

    /// Keeps the checkpoints (trunk and branches) within [`CHECKPOINT_SHARE`] of the state
    /// capacity, dropping the thinnest first ([`Self::thinnest`]).
    fn budget_checkpoints(&mut self, engine: &mut dyn Engine) {
        let limit = engine.budget().state_capacity_bytes / CHECKPOINT_SHARE;
        loop {
            let total: u64 = self
                .checkpoints
                .iter()
                .chain(self.branches.iter().flat_map(|b| &b.checkpoints))
                .map(|c| c.bytes)
                .sum();
            if total <= limit {
                break;
            }
            let Some((owner, i)) = self.thinnest(true) else {
                break;
            };
            let c = match owner {
                None => self.checkpoints.remove(i),
                Some(b) => self.branches[b].checkpoints.remove(i),
            };
            engine.drop_checkpoint(c.id);
        }
    }

    /// Bytes of the stored checkpoints (trunk and branches).
    fn checkpoint_bytes(&self) -> u64 {
        self.checkpoints
            .iter()
            .chain(self.branches.iter().flat_map(|b| &b.checkpoints))
            .map(|c| c.bytes)
            .sum()
    }

    /// FR-004 before a capture: the planned captures (positions with their marks) that fit the
    /// checkpoint budget. Each capture that would exceed it first drops the unprotected
    /// checkpoint with the smallest gap; with none left, a capture of an unprotected mark (last
    /// tool result, prompt end − 1) is skipped. Captures of protected marks (system + tools,
    /// user messages, the generation header) are kept: `commit` then shrinks the protected set
    /// to fit.
    fn admit_captures(
        &mut self,
        engine: &mut dyn Engine,
        prompt: &RenderedPrompt,
        planned: Vec<u32>,
    ) -> Vec<u32> {
        let each = engine.info().recurrent_bytes;
        let limit = engine.budget().state_capacity_bytes / CHECKPOINT_SHARE;
        let mut total = self.checkpoint_bytes();
        let mut admitted = Vec::new();
        for position in planned {
            while total + each > limit {
                let Some((owner, i)) = self.thinnest(false) else {
                    break;
                };
                let c = match owner {
                    None => self.checkpoints.remove(i),
                    Some(b) => self.branches[b].checkpoints.remove(i),
                };
                engine.drop_checkpoint(c.id);
                total -= c.bytes;
            }
            let protected = matches!(
                Self::mark_of(prompt, position),
                Mark::Anchor | Mark::User | Mark::Reply
            );
            if total + each <= limit || protected {
                total += each;
                admitted.push(position);
            }
        }
        admitted
    }

    /// Places replies recorded since the last call on the path holding their tokens; a reply
    /// no stored path holds is dropped (D051).
    fn adopt_records(&mut self) {
        let pending = self
            .ledger
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .take_pending();
        for (tokens, record) in pending {
            let end = record.end as usize;
            let holds = |path: &[u32]| path.len() >= end && path[..end] == tokens[..];
            if holds(&self.trunk) {
                add_record(&mut self.records, record);
            } else if let Some(b) = self
                .branches
                .iter_mut()
                .find(|b| (b.at as usize) < end && holds(&b.tokens))
            {
                add_record(&mut b.records, record);
            }
        }
    }

    /// Places pending replies, then publishes the paths that carry records to the ledger the
    /// renderer splices from.
    fn publish_records(&mut self) {
        self.adopt_records();
        let mut paths = Vec::new();
        if !self.records.is_empty() {
            paths.push(HeldPath {
                tokens: Arc::new(self.trunk.clone()),
                records: self.records.clone(),
            });
        }
        for b in self.branches.iter().filter(|b| !b.records.is_empty()) {
            paths.push(HeldPath {
                tokens: b.tokens.clone(),
                records: b.records.clone(),
            });
        }
        self.ledger
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .publish(paths);
    }

    fn refresh_stats(&mut self) {
        let len = self.trunk.len() as u32;
        self.records.retain(|r| r.end <= len);
        let live: Vec<CheckpointId> = self
            .checkpoints
            .iter()
            .chain(self.branches.iter().flat_map(|b| &b.checkpoints))
            .map(|c| c.id)
            .collect();
        self.marks.retain(|id, _| live.contains(id));
        let s = &mut self.stats;
        s.live_tokens = len;
        s.branches = self.branches.len() as u32;
        s.branch_tokens = self
            .branches
            .iter()
            .map(|b| u64::from(b.segment.end - b.segment.start))
            .sum();
        s.buffer_bytes = self.branches.iter().map(|b| b.segment.bytes).sum();
        let all = self
            .checkpoints
            .iter()
            .chain(self.branches.iter().flat_map(|b| &b.checkpoints));
        let (count, bytes) = all.fold((0, 0), |(n, b), c| (n + 1, b + c.bytes));
        s.checkpoints = count;
        s.checkpoint_bytes = bytes;
        s.ring_wrapped = self.ring_wrapped;
        s.records = (self.records.len()
            + self.branches.iter().map(|b| b.records.len()).sum::<usize>())
            as u32;
    }

    /// Engine failure during a move: drop everything (the state is unknown). `reset_all` also
    /// clears an engine poisoned by the failed move.
    fn fail(&mut self, engine: &mut dyn Engine, error: EngineError) -> EngineError {
        self.clear(engine);
        engine.reset_all();
        error
    }

    /// Frees stored state, aiming for `bytes`, never the path `protect` would reuse (its whole
    /// holder chain); then checkpoints other than the deepest one under the trunk. Returns the
    /// bytes freed.
    pub fn evict(
        &mut self,
        engine: &mut dyn Engine,
        bytes: u64,
        protect: Option<&RenderedPrompt>,
    ) -> u64 {
        let plan = protect.map(|p| self.lookup(&p.tokens));
        let keep = plan.as_ref().map_or(&[][..], |p| p.chain.as_slice());
        let mut freed = self.evict_branches(engine, bytes, keep);
        // Before a request, keep the checkpoint it resumes from; during one (no `protect`), the
        // live state has moved past every stored trunk checkpoint.
        let deepest = plan
            .as_ref()
            .and_then(|p| self.checkpoint_at_or_below(p.matched).map(|c| c.id));
        let mut i = 0;
        while freed < bytes && i < self.checkpoints.len() {
            if Some(self.checkpoints[i].id) == deepest {
                i += 1;
                continue;
            }
            let c = self.checkpoints.remove(i);
            engine.drop_checkpoint(c.id);
            freed += c.bytes;
        }
        self.refresh_stats();
        freed
    }

    /// Switches the live state to the closest stored path, limits growth to `total` and plans
    /// the prefill of the rest.
    pub fn prepare(
        &mut self,
        engine: &mut dyn Engine,
        prompt: &RenderedPrompt,
        total: u32,
    ) -> Result<Prepared, EngineError> {
        self.requests += 1;
        engine.set_limit(total)?;
        self.adopt_records();
        let len = prompt.len();
        if self.mode == CacheMode::Off {
            self.clear(engine);
            let miss = miss::classify(
                prompt,
                Found {
                    disabled: true,
                    ..Found::default()
                },
            );
            return Ok(self.prepared(engine, prompt, 0, miss));
        }
        let plan = self.lookup(&prompt.tokens);
        let matched = plan.matched.min(len.saturating_sub(1));
        let (evicted, dropped_shared) = self.scan_dropped(&prompt.tokens, plan.matched);

        let mut stateless = false;
        for (k, id) in plan.chain.iter().enumerate() {
            // The rest of the chain must survive this step's memory search.
            let keep = &plan.chain[k..];
            let Some(at) = self
                .branches
                .iter()
                .find(|b| b.segment.id.0 == *id)
                .map(|b| b.at)
            else {
                break;
            };
            stateless = match self.cut(engine, at, true, keep) {
                Ok(stateless) => stateless,
                // No memory to move the tail out without evicting the chain: stop the switch.
                Err(EngineError::NeedsMemory { .. }) => break,
                Err(e) => return Err(self.fail(engine, e)),
            };
            if self.trunk.len() as u32 != at
                || !self
                    .attach(engine, *id, keep)
                    .map_err(|e| self.fail(engine, e))?
            {
                break;
            }
            stateless = false;
        }
        // The request continues the conversation it resumed when it extends that whole path;
        // the turn count, which moved with the state through the switch, grows at commit.
        self.continues = common(&self.trunk, &prompt.tokens) as usize >= self.trunk.len();
        // Cut to what the trunk really shares with the prompt: a switch that stopped early (no
        // memory for an attach) leaves a different path than the plan's.
        let shared = common(&self.trunk, &prompt.tokens).min(matched);
        if (self.trunk.len() as u32) > shared {
            stateless |= self
                .cut(engine, shared, false, &[])
                .map_err(|e| self.fail(engine, e))?;
        }
        if stateless {
            self.resume_state(engine)
                .map_err(|e| self.fail(engine, e))?;
        }
        let resume = self.trunk.len() as u32;
        self.retain_checkpoints(engine, resume);
        self.prune(engine);
        let found = Found {
            matched,
            resume,
            diverges: plan.diverges,
            evicted,
            disabled: false,
            shared: plan.matched.max(dropped_shared),
            ring_wrapped: self.ring_wrapped,
        };
        let miss = miss::classify(prompt, found);
        Ok(self.prepared(engine, prompt, resume, miss))
    }

    fn prepared(
        &mut self,
        engine: &mut dyn Engine,
        prompt: &RenderedPrompt,
        resume: u32,
        miss: MissReport,
    ) -> Prepared {
        self.stats.prompt_tokens_total += u64::from(prompt.len());
        self.stats.cached_tokens_total += u64::from(miss.cached);
        self.stats.unreused_tokens_total += u64::from(miss.unreused);
        self.stats.last = Some(miss);
        self.refresh_stats();
        let marks = Self::checkpoint_marks(engine, prompt);
        let chunk_ends = chunks::above(&chunks::plan(engine.info(), prompt, &marks), resume);
        self.planned = Some((resume, chunk_ends.clone()));
        let mut wanted: Vec<u32> = marks.iter().copied().filter(|&m| m > resume).collect();
        // The live state already sits at a mark (a tool result right after the reply): capture
        // it now, before prefill moves past it.
        let at_mark = self.mode == CacheMode::Cached
            && marks.contains(&resume)
            && engine.len() == resume
            && !self.checkpoints.iter().any(|c| c.position == resume);
        if at_mark {
            wanted.insert(0, resume);
        }
        let mut checkpoints = self.admit_captures(engine, prompt, wanted);
        if checkpoints.first() == Some(&resume) {
            checkpoints.remove(0);
            if let Ok(c) = engine.checkpoint() {
                self.marks.insert(c.id, Self::mark_of(prompt, resume));
                self.checkpoints.push(c);
                self.checkpoints.sort_by_key(|c| c.position);
            }
        }
        self.refresh_stats();
        Prepared {
            resume,
            chunk_ends,
            checkpoints,
            miss,
        }
    }

    /// Sheds stored state under memory pressure; the live state only when `idle`.
    pub fn on_pressure(&mut self, engine: &mut dyn Engine, level: Pressure, idle: bool) {
        self.stats.pressure_events += 1;
        let branch_bytes: u64 = self.branches.iter().map(Branch::bytes).sum();
        let target = match level {
            Pressure::Warning => branch_bytes.div_ceil(2),
            Pressure::Critical => u64::MAX,
        };
        self.evict_branches(engine, target, &[]);
        if level == Pressure::Critical {
            let deepest = self
                .checkpoint_at_or_below(self.trunk.len() as u32)
                .map(|c| c.id);
            let (keep, gone): (Vec<_>, Vec<_>) = self
                .checkpoints
                .drain(..)
                .partition(|c| Some(c.id) == deepest);
            for c in gone {
                engine.drop_checkpoint(c.id);
            }
            self.checkpoints = keep;
            if idle {
                self.reset_live(engine);
            }
        }
        self.refresh_stats();
        self.publish_records();
    }

    /// Records a finished request: the engine holds `committed[..target]` (possibly more); the
    /// trunk is cut to a prefix of `committed` and owns `captured`. Returns its length.
    pub fn commit(
        &mut self,
        engine: &mut dyn Engine,
        prompt: &RenderedPrompt,
        committed: &[u32],
        target: u32,
        captured: Vec<Checkpoint>,
    ) -> u32 {
        for c in &captured {
            self.marks.insert(c.id, Self::mark_of(prompt, c.position));
        }
        self.checkpoints.extend(captured);
        self.checkpoints.sort_by_key(|c| c.position);
        let reached = if engine.trim(target).is_ok() {
            target
        } else if let Some(c) = self.checkpoint_at_or_below(target)
            && engine.restore(c.id).is_ok()
        {
            c.position
        } else {
            engine.reset();
            0
        };
        // The forward passes this request ran (FR-015): the planned chunks it committed, then one
        // decode step per position from the last prompt token on.
        let held = self.trunk.len() as u32;
        let (_, ends) = self.planned.take().unwrap_or_default();
        let decode_from = prompt.len().saturating_sub(1);
        schedule::push_chunks(
            &mut self.schedule,
            ends.into_iter()
                .filter(|&e| e > held && e <= target.min(decode_from)),
        );
        if target > decode_from {
            schedule::push_steps(&mut self.schedule, target);
        }
        let ran = schedule::len(&self.schedule) as usize;
        debug_assert!(
            ran >= reached as usize,
            "every held position has a forward pass"
        );
        schedule::truncate(&mut self.schedule, &committed[..ran], reached);
        // The trunk held a prefix of `committed`; records above `reached` lose their tokens.
        self.records.retain(|r| r.end <= reached);
        self.trunk = committed[..reached as usize].to_vec();
        self.turns = if self.continues { self.turns + 1 } else { 1 };
        self.retain_checkpoints(engine, reached);
        self.budget_checkpoints(engine);
        self.prune(engine);
        if self.mode == CacheMode::Off {
            self.reset_live(engine);
        }
        self.adopt_records();
        self.refresh_stats();
        self.publish_records();
        self.trunk.len() as u32
    }

    fn reset_live(&mut self, engine: &mut dyn Engine) {
        engine.reset();
        self.truncate_trunk(0);
        self.turns = 0;
        for c in self.checkpoints.drain(..) {
            engine.drop_checkpoint(c.id);
        }
    }

    /// Forgets all state (after an engine error or a panic, or `mrun run`'s `/clear`).
    pub fn clear(&mut self, engine: &mut dyn Engine) {
        self.reset_live(engine);
        for b in self.branches.drain(..) {
            engine.drop_segment(b.segment.id);
            for c in &b.checkpoints {
                engine.drop_checkpoint(c.id);
            }
        }
        self.dropped.clear();
        self.ring_wrapped = false;
        self.marks.clear();
        self.refresh_stats();
        self.publish_records();
    }

    /// `mrun cache clear`: drops every branch, checkpoint and the live state.
    pub fn clear_cache(&mut self, engine: &mut dyn Engine) -> ClearReport {
        let branches_dropped = self.branches.len() as u32;
        let before = engine.allocated();
        self.clear(engine);
        ClearReport {
            branches_dropped,
            bytes_freed: before.saturating_sub(engine.allocated()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn branch(last_used: u64, turns: u32) -> Branch {
        Branch {
            at: 0,
            tokens: Arc::new(vec![1]),
            segment: Segment {
                id: SegmentId(1),
                start: 0,
                end: 1,
                bytes: 1,
            },
            checkpoints: Vec::new(),
            records: Vec::new(),
            schedule: Schedule::new(),
            last_used,
            turns,
        }
    }

    #[test]
    fn protection_needs_two_turns_and_expires_after_sixteen_requests() {
        let mut store = PrefixStore::new(CacheMode::Cached);
        store.requests = 10;
        assert_eq!(store.rank(&branch(10, 1)), (false, 10), "one turn");
        assert_eq!(store.rank(&branch(10, 2)), (true, 10));
        store.requests = 10 + PROTECT_REQUESTS - 1;
        assert!(store.rank(&branch(10, 2)).0, "still within the window");
        store.requests = 10 + PROTECT_REQUESTS;
        assert!(!store.rank(&branch(10, 2)).0, "expired");
    }

    #[test]
    fn the_dropped_ring_wraps_and_measures_shared_prefixes() {
        let mut store = PrefixStore::new(CacheMode::Cached);
        let path: Vec<u32> = (0..1000).collect();
        store.remember_dropped(&path, 100);
        let points: Vec<u32> = store.dropped[0].points.iter().map(|p| p.0).collect();
        assert_eq!(points, [101, 256, 512, 768, 1000]);
        // Shares 600 tokens: evicted past a match of 50, shared to the 512 point.
        let mut prompt: Vec<u32> = (0..600).collect();
        prompt.push(7);
        assert_eq!(store.scan_dropped(&prompt, 50), (true, 512));
        // Past the drop point but short of what the dropped path still shares: evicted.
        assert_eq!(store.scan_dropped(&prompt, 200), (true, 512));
        // Past every point the prompt shares: not evicted, still shared.
        assert_eq!(store.scan_dropped(&prompt, 512), (false, 512));
        assert_eq!(store.scan_dropped(&[5, 6], 0), (false, 0));
        assert!(!store.ring_wrapped);
        for n in 0..DROPPED_RING as u32 {
            store.remember_dropped(&[n + 5000, 1], 0);
        }
        assert!(store.ring_wrapped, "the first entry was overwritten");
        assert_eq!(store.dropped.len(), DROPPED_RING);
        assert_eq!(store.scan_dropped(&prompt, 50), (false, 0));
    }
}
