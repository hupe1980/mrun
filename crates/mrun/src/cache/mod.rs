//! Prefix cache (003): the reuse policy of `mrun serve`.
//!
//! The engine's live state is the *trunk*: one root path of tokens. Everything else is a stored
//! node: an engine segment holding positions `[start, end)` (and, for hybrid models, the
//! recurrent state at `end`), attached at position `start` of the trunk or of another stored
//! node. A request moves state, never copies it: the trunk's tail is split off where the request
//! leaves it (kept if at least 256 tokens, else discarded), the matched stored nodes are appended
//! whole, and a node matched only partly is split back where the match ends. Every split is a
//! real engine `split_off`, so no buffer is ever shared.
//!
//! Position-bound (linear-attention) state exists only where it was computed: checkpoints taken
//! during prefill let a request resume at the deepest one at or below the match; the tokens in
//! between (the gap) are processed again. Nothing is written to disk (FR-013).

pub mod chunks;
pub mod miss;

use crate::chat::prompt::{RenderedPrompt, Role};
use crate::runtime::generate::{Prepared, Pressure, ReusePolicy};
use miss::{Found, MissReport};
use mrun_engine::{Checkpoint, Engine, EngineError, Segment};
use std::collections::HashMap;

/// Shortest tail kept as a branch when the trunk is cut; shorter tails are discarded.
pub const MIN_BRANCH: u32 = 256;
/// Most checkpoints kept along one path (the shallowest but the first is dropped beyond).
pub const MAX_CHECKPOINTS: usize = 8;
/// Output positions the first growth of a request covers.
const FIRST_GROWTH: u32 = 4096;

/// Identity of the state a cache holds: model weights, tokenizer, template and state format.
pub type CacheKey = String;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CacheMode {
    Cached,
    /// `--no-cache`: every request starts from an empty state.
    Off,
}

/// Why stored state was removed.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EvictReason {
    Lru,
    Clear,
    Tiny,
    Pressure,
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
    pub evicted_total: u64,
    pub pressure_events: u64,
    pub last: Option<MissReport>,
}

/// Result of `mrun cache clear`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ClearReport {
    pub branches_dropped: u32,
    pub bytes_freed: u64,
    /// The live state was reset now (idle) or will be after the running request.
    pub live_reset_now: bool,
}

type NodeId = u64;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
enum Owner {
    Trunk,
    Node(NodeId),
}

#[derive(Debug)]
struct Node {
    parent: Owner,
    /// First position; the node holds `tokens` at `[start, start + len)`.
    start: u32,
    tokens: Vec<u32>,
    segment: Segment,
    checkpoints: Vec<Checkpoint>,
    last_used: u64,
}

impl Node {
    fn end(&self) -> u32 {
        self.start + self.tokens.len() as u32
    }
}

/// A child removed from `owner` at `pos` whose first token was `token`.
#[derive(Clone, Copy, Debug)]
struct Evicted {
    owner: Owner,
    pos: u32,
    token: u32,
}

/// Outcome of cutting the trunk.
#[derive(Clone, Copy, Debug)]
struct Cut {
    /// The live recurrent state went with a split-off segment (hybrid models).
    no_state: bool,
}

/// Lookup result: the nodes to move in, in order, and the match.
#[derive(Debug, Default)]
struct Plan {
    steps: Vec<NodeId>,
    matched: u32,
    /// Tokens of the matched path stored at `matched` (it does not end there).
    diverges: bool,
    evicted: bool,
}

#[derive(Debug)]
pub struct PrefixStore {
    key: CacheKey,
    mode: CacheMode,
    /// Tokens the engine's live state holds.
    trunk: Vec<u32>,
    trunk_checkpoints: Vec<Checkpoint>,
    nodes: HashMap<NodeId, Node>,
    next_id: NodeId,
    evicted: Vec<Evicted>,
    tick: u64,
    /// The previous key had state; reported once as cause `model`.
    rekeyed: bool,
    /// `cache clear` while busy: reset the live state after the request.
    reset_after: bool,
    stats: CacheStats,
}

impl PrefixStore {
    pub fn new(key: CacheKey, mode: CacheMode) -> Self {
        Self {
            key,
            mode,
            trunk: Vec::new(),
            trunk_checkpoints: Vec::new(),
            nodes: HashMap::new(),
            next_id: 1,
            evicted: Vec::new(),
            tick: 0,
            rekeyed: false,
            reset_after: false,
            stats: CacheStats::default(),
        }
    }

    pub fn key(&self) -> &str {
        &self.key
    }

    /// Tokens the live state holds (tests check it against the engine).
    pub fn live_tokens(&self) -> &[u32] {
        &self.trunk
    }

    /// Stored branches as `(start, len)`, for tests.
    pub fn branches(&self) -> Vec<(u32, u32)> {
        let mut out: Vec<(u32, u32)> = self
            .nodes
            .values()
            .map(|n| (n.start, n.tokens.len() as u32))
            .collect();
        out.sort_unstable();
        out
    }

    /// A different model identity or template: the old tree is discarded (FR-014).
    pub fn rekey(&mut self, engine: &mut dyn Engine, key: CacheKey) {
        if key == self.key {
            return;
        }
        let had_state = !self.trunk.is_empty() || !self.nodes.is_empty();
        self.clear(engine);
        self.key = key;
        self.rekeyed = had_state;
    }

    fn touch(&mut self) -> u64 {
        self.tick += 1;
        self.tick
    }

    fn children(&self, owner: Owner) -> impl Iterator<Item = (&NodeId, &Node)> {
        self.nodes.iter().filter(move |(_, n)| n.parent == owner)
    }

    fn common(a: &[u32], b: &[u32]) -> u32 {
        a.iter().zip(b).take_while(|(x, y)| x == y).count() as u32
    }

    /// Longest exact match over the trunk and the stored nodes. Ties keep the shorter move.
    fn lookup(&self, prompt: &[u32]) -> Plan {
        let t = Self::common(&self.trunk, prompt);
        let mut best = Plan {
            matched: t,
            diverges: t < self.trunk.len() as u32 && (t as usize) < prompt.len(),
            ..Plan::default()
        };
        // Depth-first over stored nodes attached at positions the prompt reaches.
        let mut stack: Vec<(Owner, u32, Vec<NodeId>)> = vec![(Owner::Trunk, t, Vec::new())];
        while let Some((owner, reach, path)) = stack.pop() {
            for (&id, node) in self.children(owner) {
                let at = node.start as usize;
                if node.start > reach || at >= prompt.len() || node.tokens[0] != prompt[at] {
                    continue;
                }
                let k = Self::common(&node.tokens, &prompt[at..]);
                let matched = node.start + k;
                let mut steps = path.clone();
                steps.push(id);
                if matched > best.matched {
                    best = Plan {
                        steps: steps.clone(),
                        matched,
                        diverges: matched < node.end() && (matched as usize) < prompt.len(),
                        evicted: false,
                    };
                }
                stack.push((Owner::Node(id), matched, steps));
            }
        }
        // A request continuing where a child was removed.
        let owner = best
            .steps
            .last()
            .map_or(Owner::Trunk, |&id| Owner::Node(id));
        if let Some(&token) = prompt.get(best.matched as usize) {
            best.evicted = self
                .evicted
                .iter()
                .any(|e| e.owner == owner && e.pos == best.matched && e.token == token);
        }
        best
    }

    /// Checkpoints to capture during this prefill (position-bound state only): the end of the
    /// system and tools, the start of the last user or tool message, and the generation start.
    fn planned_checkpoints(engine: &dyn Engine, prompt: &RenderedPrompt, resume: u32) -> Vec<u32> {
        if engine.info().recurrent_bytes == 0 {
            return Vec::new();
        }
        let last_turn = prompt
            .messages
            .iter()
            .rev()
            .find(|m| matches!(m.role, Role::User | Role::Tool))
            .map(|m| m.start);
        let mut at: Vec<u32> = [
            Some(prompt.system_tools_end),
            last_turn,
            Some(prompt.generation_start),
        ]
        .into_iter()
        .flatten()
        .filter(|&p| p > resume && p < prompt.len())
        .collect();
        at.sort_unstable();
        at.dedup();
        at
    }

    /// Cuts the trunk to `at`. A tail of at least [`MIN_BRANCH`] tokens becomes a stored node
    /// (with the checkpoints and children beyond `at`), making room among other branches if the
    /// split needs memory; otherwise the tail is discarded. Discarding position-bound state
    /// resumes from the deepest checkpoint at or below `at`, so the trunk may end earlier.
    /// `shrink`: no `append` follows, so the live allocation may shrink.
    fn cut_trunk(
        &mut self,
        engine: &mut dyn Engine,
        at: u32,
        shrink: bool,
        protect: &[NodeId],
    ) -> Result<Cut, EngineError> {
        let len = self.trunk.len() as u32;
        if at >= len {
            return Ok(Cut { no_state: false });
        }
        let hybrid = engine.info().recurrent_bytes > 0;
        if self.mode == CacheMode::Cached && len - at >= MIN_BRANCH {
            let mut split = engine.split_off(at, shrink);
            if let Err(EngineError::NeedsMemory { bytes }) = split {
                self.evict_branches(engine, bytes, protect, EvictReason::Lru);
                split = engine.split_off(at, shrink);
            }
            match split {
                Ok(segment) => {
                    self.store_tail(at, segment);
                    return Ok(Cut { no_state: hybrid });
                }
                Err(EngineError::NeedsMemory { .. }) => {}
                Err(error) => return Err(error),
            }
        }
        // Discard the tail.
        let moved: Vec<Checkpoint> = {
            let (stay, moved) = self
                .trunk_checkpoints
                .drain(..)
                .partition(|c| c.position <= at);
            self.trunk_checkpoints = stay;
            moved
        };
        self.discard(engine, at, moved);
        let kept = if !hybrid {
            engine.trim(at)?;
            at
        } else if let Some(checkpoint) = self.checkpoint_at_or_below(at) {
            engine.restore_checkpoint(checkpoint.id)?;
            checkpoint.position
        } else {
            engine.reset();
            0
        };
        self.trunk.truncate(kept as usize);
        Ok(Cut { no_state: false })
    }

    /// The trunk's tail from `at`, just split off as `segment`, becomes a stored node.
    fn store_tail(&mut self, at: u32, segment: Segment) {
        let id = self.next_id;
        self.next_id += 1;
        let last_used = self.touch();
        for node in self.nodes.values_mut() {
            if node.parent == Owner::Trunk && node.start > at {
                node.parent = Owner::Node(id);
            }
        }
        let (stay, moved): (Vec<Checkpoint>, Vec<Checkpoint>) = self
            .trunk_checkpoints
            .drain(..)
            .partition(|c| c.position <= at);
        self.trunk_checkpoints = stay;
        for e in &mut self.evicted {
            if e.owner == Owner::Trunk && e.pos > at {
                e.owner = Owner::Node(id);
            }
        }
        self.nodes.insert(
            id,
            Node {
                parent: Owner::Trunk,
                start: at,
                tokens: self.trunk[at as usize..].to_vec(),
                segment,
                checkpoints: moved,
                last_used,
            },
        );
        self.trunk.truncate(at as usize);
    }

    /// A discarded trunk tail: its checkpoints and the nodes hanging off it go too.
    fn discard(&mut self, engine: &mut dyn Engine, at: u32, checkpoints: Vec<Checkpoint>) {
        for checkpoint in checkpoints {
            engine.drop_checkpoint(checkpoint.id);
        }
        if let Some(&token) = self.trunk.get(at as usize) {
            self.evicted.push(Evicted {
                owner: Owner::Trunk,
                pos: at,
                token,
            });
        }
        let orphans: Vec<NodeId> = self
            .children(Owner::Trunk)
            .filter(|(_, n)| n.start > at)
            .map(|(&id, _)| id)
            .collect();
        for id in orphans {
            self.remove_subtree(engine, id, EvictReason::Tiny);
        }
    }

    /// Removes a node and everything stored below it.
    fn remove_subtree(&mut self, engine: &mut dyn Engine, id: NodeId, reason: EvictReason) -> u64 {
        let children: Vec<NodeId> = self.children(Owner::Node(id)).map(|(&c, _)| c).collect();
        let mut freed = 0;
        for child in children {
            freed += self.remove_subtree(engine, child, reason);
        }
        freed + self.remove_node(engine, id, reason)
    }

    fn remove_node(&mut self, engine: &mut dyn Engine, id: NodeId, _reason: EvictReason) -> u64 {
        let Some(node) = self.nodes.remove(&id) else {
            return 0;
        };
        engine.drop_segment(node.segment.id);
        let mut freed = node.segment.bytes;
        for checkpoint in node.checkpoints {
            freed += checkpoint.bytes;
            engine.drop_checkpoint(checkpoint.id);
        }
        self.evicted.retain(|e| e.owner != Owner::Node(id));
        self.evicted.push(Evicted {
            owner: node.parent,
            pos: node.start,
            token: node.tokens[0],
        });
        self.stats.evicted_total += 1;
        freed
    }

    /// Moves stored node `id` onto the end of the trunk (which must end at its start).
    fn move_in(&mut self, engine: &mut dyn Engine, id: NodeId) -> Result<(), EngineError> {
        let segment = self.nodes[&id].segment;
        engine.append(segment.id, true)?;
        let node = self.nodes.remove(&id).expect("node exists");
        debug_assert_eq!(node.start as usize, self.trunk.len());
        self.trunk.extend_from_slice(&node.tokens);
        self.trunk_checkpoints.extend(node.checkpoints);
        for child in self.nodes.values_mut() {
            if child.parent == Owner::Node(id) {
                child.parent = Owner::Trunk;
            }
        }
        for e in &mut self.evicted {
            if e.owner == Owner::Node(id) {
                e.owner = Owner::Trunk;
            }
        }
        Ok(())
    }

    /// Deepest trunk checkpoint at or below `at`.
    fn checkpoint_at_or_below(&self, at: u32) -> Option<Checkpoint> {
        self.trunk_checkpoints
            .iter()
            .filter(|c| c.position <= at)
            .max_by_key(|c| c.position)
            .copied()
    }

    /// Keeps at most [`MAX_CHECKPOINTS`] on the trunk: drops the shallowest but the first.
    fn cap_checkpoints(&mut self, engine: &mut dyn Engine) {
        self.trunk_checkpoints.sort_by_key(|c| c.position);
        while self.trunk_checkpoints.len() > MAX_CHECKPOINTS {
            let dropped = self.trunk_checkpoints.remove(1);
            engine.drop_checkpoint(dropped.id);
        }
    }

    /// Nodes on the path `protect` would reuse.
    fn protected(&self, protect: Option<&RenderedPrompt>) -> Vec<NodeId> {
        protect.map_or(Vec::new(), |p| self.lookup(&p.tokens).steps)
    }

    /// Least recently used stored leaf not in `protect`.
    fn lru_leaf(&self, protect: &[NodeId]) -> Option<NodeId> {
        self.nodes
            .iter()
            .filter(|&(id, _)| !protect.contains(id))
            .filter(|&(&id, _)| self.children(Owner::Node(id)).next().is_none())
            .min_by_key(|(_, n)| n.last_used)
            .map(|(&id, _)| id)
    }

    fn branch_bytes(&self) -> u64 {
        self.nodes
            .values()
            .map(|n| n.segment.bytes + n.checkpoints.iter().map(|c| c.bytes).sum::<u64>())
            .sum()
    }

    /// Drops stored leaves (LRU first) until `bytes` are freed or none is left.
    fn evict_branches(
        &mut self,
        engine: &mut dyn Engine,
        bytes: u64,
        protect: &[NodeId],
        reason: EvictReason,
    ) -> u64 {
        let mut freed = 0;
        while freed < bytes {
            let Some(id) = self.lru_leaf(protect) else {
                break;
            };
            freed += self.remove_node(engine, id, reason);
        }
        freed
    }

    fn refresh_stats(&mut self) {
        let s = &mut self.stats;
        s.live_tokens = self.trunk.len() as u32;
        s.branches = self.nodes.len() as u32;
        s.branch_tokens = self.nodes.values().map(|n| n.tokens.len() as u64).sum();
        s.buffer_bytes = self.nodes.values().map(|n| n.segment.bytes).sum();
        let checkpoints = self
            .trunk_checkpoints
            .iter()
            .chain(self.nodes.values().flat_map(|n| &n.checkpoints));
        let (count, bytes) = checkpoints.fold((0, 0), |(n, b), c| (n + 1, b + c.bytes));
        s.checkpoints = count;
        s.checkpoint_bytes = bytes;
    }

    /// Engine failure during a move: drop everything (the state is unknown).
    fn fail(&mut self, engine: &mut dyn Engine, error: EngineError) -> EngineError {
        self.clear(engine);
        error
    }
}

impl ReusePolicy for PrefixStore {
    fn initial_need(&self, engine: &dyn Engine, prompt: &RenderedPrompt, total: u32) -> u64 {
        let info = engine.info();
        let plan = self.lookup(&prompt.tokens);
        let matched = plan.matched.min(prompt.len().saturating_sub(1));
        let positions = total.min(prompt.len().saturating_add(FIRST_GROWTH));
        let checkpoints = Self::planned_checkpoints(engine, prompt, matched).len() as u64;
        u64::from(positions.saturating_sub(matched)) * info.kv_bytes_per_token
            + checkpoints * info.recurrent_bytes
    }

    fn evict(
        &mut self,
        engine: &mut dyn Engine,
        bytes: u64,
        protect: Option<&RenderedPrompt>,
    ) -> u64 {
        let protected = self.protected(protect);
        let mut freed = self.evict_branches(engine, bytes, &protected, EvictReason::Lru);
        // Then trunk checkpoints, shallowest first, keeping the one the request resumes from.
        let keep = protect.and_then(|p| {
            let matched = self.lookup(&p.tokens).matched;
            self.checkpoint_at_or_below(matched).map(|c| c.id)
        });
        self.trunk_checkpoints.sort_by_key(|c| c.position);
        let mut i = 0;
        while freed < bytes && i < self.trunk_checkpoints.len() {
            if Some(self.trunk_checkpoints[i].id) == keep {
                i += 1;
                continue;
            }
            let dropped = self.trunk_checkpoints.remove(i);
            engine.drop_checkpoint(dropped.id);
            freed += dropped.bytes;
        }
        self.refresh_stats();
        freed
    }

    fn prepare(
        &mut self,
        engine: &mut dyn Engine,
        prompt: &RenderedPrompt,
        total: u32,
    ) -> Result<Prepared, EngineError> {
        let len = prompt.len();
        let rekeyed = std::mem::take(&mut self.rekeyed);
        if self.mode == CacheMode::Off {
            self.trunk.clear();
            for checkpoint in self.trunk_checkpoints.drain(..) {
                engine.drop_checkpoint(checkpoint.id);
            }
            engine.reserve(0, total)?;
            let found = Found {
                disabled: true,
                ..Found::default()
            };
            let miss = miss::classify(prompt, found);
            return Ok(self.prepared(engine, prompt, 0, miss));
        }
        let stored = !self.trunk.is_empty() || !self.nodes.is_empty();
        let plan = self.lookup(&prompt.tokens);
        let matched = plan.matched.min(len.saturating_sub(1));

        // Switch: cut where the path leaves the trunk, move each stored node in.
        let mut no_state = false;
        let protect = plan.steps.clone();
        for &id in &plan.steps {
            let start = self.nodes[&id].start;
            no_state = self
                .cut_trunk(engine, start, false, &protect)
                .map_err(|e| self.fail(engine, e))?
                .no_state;
            if self.trunk.len() as u32 != start {
                // A discarded tail took the trunk below the branch point: start from here.
                break;
            }
            match self.move_in(engine, id) {
                Ok(()) => no_state = false,
                Err(EngineError::NeedsMemory { bytes }) => {
                    // Make room among other branches and retry once; else stop here.
                    self.evict_branches(engine, bytes, &protect, EvictReason::Lru);
                    if self.move_in(engine, id).is_err() {
                        break;
                    }
                    no_state = false;
                }
                Err(error) => return Err(self.fail(engine, error)),
            }
            if let Some(node_id) = self.nodes.get_mut(&id) {
                node_id.last_used = self.tick;
            }
        }
        // Split back where the match ends.
        if (self.trunk.len() as u32) > matched {
            no_state |= self
                .cut_trunk(engine, matched, true, &protect)
                .map_err(|e| self.fail(engine, e))?
                .no_state;
        }
        let mut resume = (self.trunk.len() as u32).min(matched);
        if no_state {
            // Position-bound state exists only where it was computed: resume from a checkpoint.
            match self.checkpoint_at_or_below(resume) {
                Some(checkpoint) => {
                    engine
                        .restore_checkpoint(checkpoint.id)
                        .map_err(|e| self.fail(engine, e))?;
                    resume = checkpoint.position;
                }
                None => resume = 0,
            }
            self.trunk.truncate(resume as usize);
        }
        for checkpoint in std::mem::take(&mut self.trunk_checkpoints) {
            if checkpoint.position <= resume {
                self.trunk_checkpoints.push(checkpoint);
            } else {
                engine.drop_checkpoint(checkpoint.id);
            }
        }
        engine.reserve(resume, total)?;
        self.trunk.truncate(resume as usize);

        let found = Found {
            matched,
            resume,
            diverges: plan.diverges,
            stored,
            evicted: plan.evicted,
            rekeyed,
            disabled: false,
        };
        let miss = miss::classify(prompt, found);
        Ok(self.prepared(engine, prompt, resume, miss))
    }

    fn on_pressure(&mut self, engine: &mut dyn Engine, level: Pressure, idle: bool) {
        self.stats.pressure_events += 1;
        let target = match level {
            Pressure::Warning => self.branch_bytes() / 2,
            Pressure::Critical => u64::MAX,
        };
        self.evict_branches(engine, target, &[], EvictReason::Pressure);
        if level == Pressure::Critical {
            for checkpoint in self.trunk_checkpoints.drain(..) {
                engine.drop_checkpoint(checkpoint.id);
            }
            if idle {
                engine.reset();
                self.trunk.clear();
            }
        }
        self.refresh_stats();
    }

    fn commit(
        &mut self,
        engine: &mut dyn Engine,
        _prompt: &RenderedPrompt,
        committed: &[u32],
        target: u32,
        captured: Vec<Checkpoint>,
    ) -> u32 {
        self.trunk_checkpoints.extend(captured);
        let reached = if engine.trim(target).is_ok() {
            target
        } else if let Some(checkpoint) = self.checkpoint_at_or_below(target)
            && engine.restore_checkpoint(checkpoint.id).is_ok()
        {
            checkpoint.position
        } else {
            engine.reset();
            0
        };
        self.trunk = committed[..reached as usize].to_vec();
        for checkpoint in std::mem::take(&mut self.trunk_checkpoints) {
            if checkpoint.position <= reached {
                self.trunk_checkpoints.push(checkpoint);
            } else {
                engine.drop_checkpoint(checkpoint.id);
            }
        }
        self.cap_checkpoints(engine);
        // A stored node the trunk now retraces entirely is redundant: its children move up.
        let redundant: Vec<NodeId> = self
            .children(Owner::Trunk)
            .filter(|(_, n)| {
                let (s, e) = (n.start as usize, n.end() as usize);
                e <= self.trunk.len() && self.trunk[s..e] == n.tokens[..]
            })
            .map(|(&id, _)| id)
            .collect();
        for id in redundant {
            for child in self.nodes.values_mut() {
                if child.parent == Owner::Node(id) {
                    child.parent = Owner::Trunk;
                }
            }
            if let Some(node) = self.nodes.remove(&id) {
                engine.drop_segment(node.segment.id);
                for checkpoint in node.checkpoints {
                    engine.drop_checkpoint(checkpoint.id);
                }
            }
        }
        // A removal record stays until the trunk retraces exactly that state again.
        let trunk = &self.trunk;
        self.evicted
            .retain(|e| !(e.owner == Owner::Trunk && trunk.get(e.pos as usize) == Some(&e.token)));
        if self.mode == CacheMode::Off || std::mem::take(&mut self.reset_after) {
            engine.reset();
            self.trunk.clear();
            for checkpoint in self.trunk_checkpoints.drain(..) {
                engine.drop_checkpoint(checkpoint.id);
            }
        }
        self.touch();
        self.refresh_stats();
        self.trunk.len() as u32
    }

    fn clear(&mut self, engine: &mut dyn Engine) {
        engine.reset();
        for checkpoint in self.trunk_checkpoints.drain(..) {
            engine.drop_checkpoint(checkpoint.id);
        }
        for (_, node) in self.nodes.drain() {
            engine.drop_segment(node.segment.id);
            for checkpoint in node.checkpoints {
                engine.drop_checkpoint(checkpoint.id);
            }
        }
        self.trunk.clear();
        self.evicted.clear();
        self.refresh_stats();
    }

    fn stats(&self) -> Option<CacheStats> {
        Some(self.stats)
    }

    fn clear_cache(&mut self, engine: &mut dyn Engine, idle: bool) -> Option<ClearReport> {
        let branches_dropped = self.nodes.len() as u32;
        let mut bytes_freed = 0;
        let ids: Vec<NodeId> = self.nodes.keys().copied().collect();
        for id in ids {
            bytes_freed += self.remove_node(engine, id, EvictReason::Clear);
        }
        for checkpoint in self.trunk_checkpoints.drain(..) {
            bytes_freed += checkpoint.bytes;
            engine.drop_checkpoint(checkpoint.id);
        }
        self.evicted.clear();
        if idle {
            engine.reset();
            self.trunk.clear();
        } else {
            self.reset_after = true;
        }
        self.refresh_stats();
        Some(ClearReport {
            branches_dropped,
            bytes_freed,
            live_reset_now: idle,
        })
    }
}

impl PrefixStore {
    fn prepared(
        &mut self,
        engine: &dyn Engine,
        prompt: &RenderedPrompt,
        resume: u32,
        miss: MissReport,
    ) -> Prepared {
        self.stats.prompt_tokens_total += u64::from(prompt.len());
        self.stats.cached_tokens_total += u64::from(miss.cached);
        self.stats.last = Some(miss);
        self.refresh_stats();
        let checkpoints = Self::planned_checkpoints(engine, prompt, resume);
        Prepared {
            resume,
            chunks: chunks::plan(prompt, resume, checkpoints),
            miss: Some(miss),
        }
    }
}
