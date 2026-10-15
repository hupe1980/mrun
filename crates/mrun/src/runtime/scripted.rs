//! `ScriptedEngine`: a deterministic [`Engine`] for tests on any OS (no MLX).
//!
//! It emits a fixed token script and models what the runtime relies on: byte accounting with
//! lazy growth, `NeedsMemory`, the reservation ceiling, chunk splits with per-layer cancel and
//! yield, one pending pipelined step, non-trimmable recurrent state (hybrid mode), checkpoints
//! and segments. Recurrent state is modelled as the tokens it has consumed, so restoring a
//! checkpoint or segment that does not match the live sequence is detected on the next use.

use mrun_engine::budget::{self, BudgetError, DeviceMemory, Overrides};
use mrun_engine::{
    CancelToken, Checkpoint, CheckpointId, Engine, EngineError, LayerKind, MAX_PLAN_CHECKPOINTS,
    MemoryBudget, ModelInfo, PrefillPlan, Prefilled, Sampling, Segment, SegmentId, Step, TokenMask,
};
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

/// End-of-sequence token used by [`ScriptedEngine`].
pub const SCRIPT_EOS: u32 = u32::MAX - 1;
/// Attention bytes per token.
pub const KV_BYTES: u64 = 16;
/// Allocation granularity of the live state, in positions.
const ALIGN: u32 = 256;

/// Byte costs of one family's reference model (4-bit or MXFP4 weights, bf16 state).
#[derive(Clone, Copy, Debug)]
pub struct FamilyProfile {
    pub family: &'static str,
    pub layers: u32,
    /// Kind of layer `i`.
    pub layer_kind: fn(u32) -> LayerKind,
    pub kv_bytes_per_token: u64,
    /// Position-bound bytes per checkpoint.
    pub recurrent_bytes: u64,
    pub max_position: u32,
    /// Approximate weight bytes, for budget fixtures.
    pub weights: u64,
}

const KIB: u64 = 1 << 10;
const MIB: u64 = 1 << 20;

/// Reference models: Qwen3.6-35B-A3B, Qwen3-Coder-30B-A3B, gpt-oss-20b and Qwen3-8B.
pub const PROFILES: [FamilyProfile; 4] = [
    FamilyProfile {
        family: "qwen3_5_moe",
        layers: 40,
        layer_kind: |i| {
            if (i + 1) % 4 == 0 {
                LayerKind::FullAttention
            } else {
                LayerKind::Linear
            }
        },
        kv_bytes_per_token: 20 * KIB,
        recurrent_bytes: 64_389_120,
        max_position: 262_144,
        weights: 20 * 1024 * MIB,
    },
    FamilyProfile {
        family: "qwen3_moe",
        layers: 48,
        layer_kind: |_| LayerKind::FullAttention,
        kv_bytes_per_token: 96 * KIB,
        recurrent_bytes: 0,
        max_position: 262_144,
        weights: 16 * 1024 * MIB,
    },
    FamilyProfile {
        family: "gpt_oss",
        layers: 24,
        layer_kind: |i| {
            if i % 2 == 0 {
                LayerKind::SlidingAttention { window: 128 }
            } else {
                LayerKind::FullAttention
            }
        },
        kv_bytes_per_token: 48 * KIB,
        recurrent_bytes: 0,
        max_position: 131_072,
        weights: 11_571 * MIB,
    },
    FamilyProfile {
        family: "qwen3",
        layers: 36,
        layer_kind: |_| LayerKind::FullAttention,
        kv_bytes_per_token: 144 * KIB,
        recurrent_bytes: 0,
        max_position: 40_960,
        weights: 4_400 * MIB,
    },
];

impl FamilyProfile {
    pub fn get(family: &str) -> Option<&'static Self> {
        PROFILES.iter().find(|profile| profile.family == family)
    }

    /// Budget on a machine with `physical` bytes (all usable by the GPU) and `scratch` probed.
    pub fn budget(&self, physical: u64, scratch: u64) -> Result<MemoryBudget, BudgetError> {
        let info = ModelInfo {
            identity: String::new(),
            family: self.family.into(),
            layer_kinds: Vec::new(),
            kv_heads: 1,
            head_dim: 1,
            vocab: 1,
            max_position: self.max_position,
            kv_bytes_per_token: self.kv_bytes_per_token,
            recurrent_bytes: self.recurrent_bytes,
            eos_tokens: Vec::new(),
        };
        let device = DeviceMemory {
            physical,
            recommended_working_set: physical,
        };
        let top = budget::plan(device, Overrides::default())?;
        let depth = budget::probe_depth(&top, self.weights, &info);
        budget::with_state_capacity(top, self.weights, scratch, depth, &info)
    }
}

#[derive(Debug)]
struct Pending {
    token: u32,
    step: u64,
}

#[derive(Debug)]
struct StoredCheckpoint {
    position: u32,
    /// Tokens the captured state has consumed.
    consumed: Vec<u32>,
    bytes: u64,
}

#[derive(Clone, Debug)]
struct StoredSegment {
    start: u32,
    tokens: Vec<u32>,
    recurrent: Option<Vec<u32>>,
    bytes: u64,
}

#[derive(Debug)]
pub struct ScriptedEngine {
    info: ModelInfo,
    budget: MemoryBudget,
    script: Vec<u32>,
    next: usize,
    /// Live positions, including a pending pipelined token.
    tokens: Vec<u32>,
    cap: u32,
    /// Hybrid only: tokens the live recurrent state has consumed; `None` after `split_off`.
    recurrent: Option<Vec<u32>>,
    reserved: u32,
    grown: bool,
    pending: Option<Pending>,
    current: Option<u64>,
    checkpoints: HashMap<CheckpointId, StoredCheckpoint>,
    segments: HashMap<SegmentId, StoredSegment>,
    next_id: u64,
    /// Positions added per growth after the first one of a request.
    pub growth_step: u32,
    /// Cancels once this many prefill layers have run in total (one-shot).
    pub cancel_after_layers: Option<u64>,
    /// Requests a yield once this many prefill layers have run in total (one-shot).
    pub yield_after_layers: Option<u64>,
    pub panic_on_forward: bool,
    /// Incremented by every `reset_all` call (shared so tests can observe the worker thread).
    pub reset_all_calls: Arc<AtomicUsize>,
    pub layers_run: u64,
    /// Every token fed by `prefill`, `forward` or a pipelined step, in order.
    pub processed: Vec<u32>,
    /// Committed prefill chunks as `(start, end)`.
    pub chunks: Vec<(u32, u32)>,
}

impl ScriptedEngine {
    /// Attention-only model with four layers; `script` is sampled in order, then [`SCRIPT_EOS`].
    pub fn new(script: Vec<u32>, capacity_tokens: u32) -> Self {
        Self::build(
            script,
            capacity_tokens,
            vec![LayerKind::FullAttention; 4],
            0,
        )
    }

    /// Hybrid model (three linear layers, one attention layer) with `recurrent_bytes` of state.
    pub fn new_hybrid(script: Vec<u32>, capacity_tokens: u32, recurrent_bytes: u64) -> Self {
        let mut kinds = vec![LayerKind::Linear; 3];
        kinds.push(LayerKind::FullAttention);
        Self::build(script, capacity_tokens, kinds, recurrent_bytes)
    }

    fn build(script: Vec<u32>, capacity: u32, kinds: Vec<LayerKind>, recurrent: u64) -> Self {
        let info = ModelInfo {
            identity: "scripted".into(),
            family: if recurrent > 0 {
                "qwen3_5_moe"
            } else {
                "qwen3"
            }
            .into(),
            layer_kinds: kinds,
            kv_heads: 1,
            head_dim: 1,
            vocab: u32::MAX,
            max_position: capacity,
            kv_bytes_per_token: KV_BYTES,
            recurrent_bytes: recurrent,
            eos_tokens: vec![SCRIPT_EOS],
        };
        let budget = MemoryBudget {
            probed_depth: capacity,
            state_capacity_bytes: u64::from(capacity) * KV_BYTES
                + recurrent * (1 + MAX_PLAN_CHECKPOINTS as u64),
            kv_capacity_tokens: capacity,
            ..MemoryBudget::default()
        };
        Self::with_info(script, info, budget)
    }

    /// Engine with the byte costs of a family's reference model under `budget`.
    pub fn for_family(script: Vec<u32>, profile: &FamilyProfile, budget: MemoryBudget) -> Self {
        let info = ModelInfo {
            identity: "scripted".into(),
            family: profile.family.into(),
            layer_kinds: (0..profile.layers).map(profile.layer_kind).collect(),
            kv_heads: 1,
            head_dim: 1,
            vocab: u32::MAX,
            max_position: profile.max_position,
            kv_bytes_per_token: profile.kv_bytes_per_token,
            recurrent_bytes: profile.recurrent_bytes,
            eos_tokens: vec![SCRIPT_EOS],
        };
        Self::with_info(script, info, budget)
    }

    fn with_info(script: Vec<u32>, info: ModelInfo, budget: MemoryBudget) -> Self {
        let hybrid = info.is_hybrid();
        Self {
            info,
            budget,
            script,
            next: 0,
            tokens: Vec::new(),
            cap: 0,
            recurrent: hybrid.then(Vec::new),
            reserved: 0,
            grown: false,
            pending: None,
            current: None,
            checkpoints: HashMap::new(),
            segments: HashMap::new(),
            next_id: 0,
            growth_step: 4096,
            cancel_after_layers: None,
            yield_after_layers: None,
            panic_on_forward: false,
            reset_all_calls: Arc::default(),
            layers_run: 0,
            processed: Vec::new(),
            chunks: Vec::new(),
        }
    }

    pub fn with_state_capacity_bytes(mut self, bytes: u64) -> Self {
        self.budget.state_capacity_bytes = bytes;
        self
    }

    /// Live tokens, including a pending pipelined token.
    pub fn tokens(&self) -> &[u32] {
        &self.tokens
    }

    /// Allocated live positions.
    pub fn cap(&self) -> u32 {
        self.cap
    }

    pub fn has_pending(&self) -> bool {
        self.pending.is_some()
    }

    pub fn checkpoint_count(&self) -> usize {
        self.checkpoints.len()
    }

    /// Token range `(start, end)` a stored segment holds.
    pub fn segment_range(&self, id: SegmentId) -> Option<(u32, u32)> {
        self.segments
            .get(&id)
            .map(|seg| (seg.start, seg.start + seg.tokens.len() as u32))
    }

    fn hybrid(&self) -> bool {
        self.info.is_hybrid()
    }

    fn id(&mut self) -> u64 {
        self.next_id += 1;
        self.next_id
    }

    fn confirmed_len(&self) -> u32 {
        (self.tokens.len() - usize::from(self.pending.is_some())) as u32
    }

    fn live_bytes(&self, cap: u32, len: u32, has_recurrent: bool) -> u64 {
        let recurrent = if self.hybrid() && has_recurrent && len > 0 {
            self.info.recurrent_bytes
        } else {
            0
        };
        u64::from(cap) * self.info.kv_bytes_per_token + recurrent
    }

    fn current_live_bytes(&self) -> u64 {
        self.live_bytes(self.cap, self.tokens.len() as u32, self.recurrent.is_some())
    }

    /// Fails with the shortfall if `extra` more bytes do not fit.
    fn ensure_free(&self, extra: u64) -> Result<(), EngineError> {
        let free = self
            .budget
            .state_capacity_bytes
            .saturating_sub(self.allocated());
        match extra.checked_sub(free) {
            Some(short) if short > 0 => Err(EngineError::NeedsMemory { bytes: short }),
            _ => Ok(()),
        }
    }

    /// Allocation needed to hold `needed` positions under the growth policy.
    fn grown_cap(&self, needed: u32) -> Result<u32, EngineError> {
        if needed <= self.cap {
            return Ok(self.cap);
        }
        let limit = self.reserved.min(self.budget.kv_capacity_tokens);
        if needed > limit {
            return Err(EngineError::ReservationExceeded {
                needed,
                reserved: self.reserved,
            });
        }
        let target = if self.grown {
            needed.max(self.cap.saturating_add(self.growth_step))
        } else {
            needed.saturating_add(self.growth_step)
        };
        Ok(round_up(target).min(limit))
    }

    /// Grows to hold `len` positions, checking memory (plus `extra` bytes) first.
    fn grow(&mut self, len: u32, extra: u64) -> Result<(), EngineError> {
        let cap = self.grown_cap(len)?;
        let growth = self
            .live_bytes(cap, len, true)
            .saturating_sub(self.current_live_bytes());
        self.ensure_free(growth + extra)?;
        if cap != self.cap {
            self.cap = cap;
            self.grown = true;
        }
        Ok(())
    }

    fn check_recurrent(&self) -> Result<(), EngineError> {
        if !self.hybrid() {
            return Ok(());
        }
        match &self.recurrent {
            None => Err(EngineError::NoRecurrentState),
            Some(consumed) if *consumed != self.tokens => Err(EngineError::Backend(
                "recurrent state does not match the attention state".into(),
            )),
            Some(_) => Ok(()),
        }
    }

    fn feed(&mut self, tokens: &[u32]) {
        self.tokens.extend_from_slice(tokens);
        if let Some(consumed) = &mut self.recurrent {
            consumed.extend_from_slice(tokens);
        }
        self.processed.extend_from_slice(tokens);
    }

    fn rollback(&mut self) {
        if self.pending.take().is_some() {
            self.tokens.pop();
            if let Some(consumed) = &mut self.recurrent {
                consumed.pop();
            }
        }
    }

    fn zero_recurrent(&mut self) {
        self.recurrent = self.hybrid().then(Vec::new);
    }

    fn store_checkpoint(&mut self) -> Checkpoint {
        let id = CheckpointId(self.id());
        let stored = StoredCheckpoint {
            position: self.tokens.len() as u32,
            consumed: self
                .recurrent
                .clone()
                .unwrap_or_else(|| self.tokens.clone()),
            bytes: self.info.recurrent_bytes,
        };
        let checkpoint = Checkpoint {
            id,
            position: stored.position,
            bytes: stored.bytes,
        };
        self.checkpoints.insert(id, stored);
        checkpoint
    }

    /// Counts one prefill layer and fires the configured one-shot cancel or yield.
    fn tick(&mut self, cancel: &CancelToken) {
        self.layers_run += 1;
        if self
            .cancel_after_layers
            .is_some_and(|n| self.layers_run >= n)
        {
            self.cancel_after_layers = None;
            cancel.cancel();
        }
        if self
            .yield_after_layers
            .is_some_and(|n| self.layers_run >= n)
        {
            self.yield_after_layers = None;
            cancel.request_yield();
        }
    }
}

fn round_up(positions: u32) -> u32 {
    positions.div_ceil(ALIGN).saturating_mul(ALIGN)
}

impl Engine for ScriptedEngine {
    fn info(&self) -> &ModelInfo {
        &self.info
    }

    fn budget(&self) -> &MemoryBudget {
        &self.budget
    }

    fn active_len(&self) -> u32 {
        self.tokens.len() as u32
    }

    fn allocated(&self) -> u64 {
        self.current_live_bytes()
            + self.segments.values().map(|seg| seg.bytes).sum::<u64>()
            + self.checkpoints.values().map(|ck| ck.bytes).sum::<u64>()
    }

    fn reserve(&mut self, keep: u32, total: u32) -> Result<(), EngineError> {
        let available = self.budget.kv_capacity_tokens;
        if total > available {
            return Err(EngineError::Capacity {
                requested: total,
                available,
            });
        }
        self.trim(keep)?;
        self.reserved = total;
        self.grown = false;
        Ok(())
    }

    fn prefill(
        &mut self,
        tokens: &[u32],
        plan: &PrefillPlan,
        cancel: &CancelToken,
        progress: &mut dyn FnMut(u32),
    ) -> Result<Prefilled, EngineError> {
        if tokens.is_empty() {
            return Ok(Prefilled::default());
        }
        self.rollback();
        self.check_recurrent()?;
        let start = self.tokens.len() as u32;
        let end = start + tokens.len() as u32;
        let planned = plan.checkpoints_in(start, end).count();
        if planned > MAX_PLAN_CHECKPOINTS {
            return Err(EngineError::InvalidRange {
                start,
                end,
                len: start,
            });
        }
        self.grow(end, planned as u64 * self.info.recurrent_bytes)?;

        let layers = self.info.layers();
        let mut result = Prefilled::default();
        let mut pos = start;
        while pos < end {
            let chunk_end = plan.chunk_end(pos, end);
            let mut yield_now = false;
            for layer in 1..=layers {
                self.tick(cancel);
                // The partial chunk was never written to the live state: discarding is free.
                if layer < layers && cancel.is_cancelled() {
                    return Ok(result);
                }
                yield_now |= cancel.yield_requested();
            }
            self.feed(&tokens[(pos - start) as usize..(chunk_end - start) as usize]);
            self.chunks.push((pos, chunk_end));
            pos = chunk_end;
            result.committed = pos - start;
            if plan.checkpoints.contains(&pos) {
                let checkpoint = self.store_checkpoint();
                result.checkpoints.push(checkpoint);
            }
            progress(result.committed);
            if cancel.is_cancelled() {
                break;
            }
            if yield_now && pos < end {
                result.yielded = true;
                break;
            }
        }
        Ok(result)
    }

    fn forward(&mut self, last: u32) -> Result<Step, EngineError> {
        assert!(!self.panic_on_forward, "scripted panic");
        self.check_recurrent()?;
        if let Some(pending) = self.pending.take_if(|pending| pending.token == last) {
            self.current = Some(pending.step);
            return Ok(Step::new(pending.step));
        }
        self.rollback();
        self.grow(self.active_len() + 1, 0)?;
        self.feed(&[last]);
        let id = self.id();
        self.current = Some(id);
        Ok(Step::new(id))
    }

    fn sample(
        &mut self,
        step: Step,
        _sampling: &Sampling,
        mask: Option<&TokenMask>,
        pipeline: bool,
    ) -> Result<u32, EngineError> {
        if self.current != Some(step.id()) {
            return Err(EngineError::InvalidStep);
        }
        self.current = None;
        let token = self.script.get(self.next).copied().unwrap_or(SCRIPT_EOS);
        self.next += 1;
        if let Some(mask) = mask
            && token != SCRIPT_EOS
            && !mask.is_allowed(token)
        {
            return Err(EngineError::Backend(format!(
                "script token {token} is masked"
            )));
        }
        if pipeline && (self.tokens.len() as u32) < self.cap {
            self.feed(&[token]);
            let step = self.id();
            self.pending = Some(Pending { token, step });
        }
        Ok(token)
    }

    fn top_logprobs(&mut self, step: &Step, n: usize) -> Result<Vec<(u32, f32)>, EngineError> {
        if self.current != Some(step.id()) {
            return Err(EngineError::InvalidStep);
        }
        let top = self.script.get(self.next).copied().unwrap_or(SCRIPT_EOS);
        Ok(std::iter::once((top, -0.01))
            .chain(
                (0..)
                    .filter(|&token| token != top)
                    .map(|token| (token, -2.0 - token as f32)),
            )
            .take(n)
            .collect())
    }

    fn trim(&mut self, len: u32) -> Result<(), EngineError> {
        let confirmed = self.confirmed_len();
        if len > confirmed {
            return Err(EngineError::InvalidRange {
                start: len,
                end: len,
                len: confirmed,
            });
        }
        if self.hybrid() && self.recurrent.is_some() && len != 0 && len != confirmed {
            return Err(EngineError::NotTrimmable {
                len,
                active: self.active_len(),
            });
        }
        self.rollback();
        if len < confirmed {
            self.current = None;
        }
        self.tokens.truncate(len as usize);
        if len == 0 {
            self.zero_recurrent();
        }
        Ok(())
    }

    fn checkpoint(&mut self) -> Result<Checkpoint, EngineError> {
        if self.hybrid() && self.recurrent.is_none() {
            return Err(EngineError::NoRecurrentState);
        }
        self.ensure_free(self.info.recurrent_bytes)?;
        self.rollback();
        Ok(self.store_checkpoint())
    }

    fn restore_checkpoint(&mut self, id: CheckpointId) -> Result<(), EngineError> {
        let stored = self
            .checkpoints
            .get(&id)
            .ok_or(EngineError::UnknownCheckpoint(id))?;
        let (position, confirmed) = (stored.position, self.confirmed_len());
        if position > confirmed {
            return Err(EngineError::InvalidRange {
                start: position,
                end: position,
                len: confirmed,
            });
        }
        let extra = self
            .live_bytes(self.cap, position, true)
            .saturating_sub(self.current_live_bytes());
        self.ensure_free(extra)?;
        let consumed = stored.consumed.clone();
        self.rollback();
        self.current = None;
        self.tokens.truncate(position as usize);
        if self.hybrid() {
            self.recurrent = Some(consumed);
        }
        Ok(())
    }

    fn drop_checkpoint(&mut self, id: CheckpointId) {
        self.checkpoints.remove(&id);
    }

    fn split_off(&mut self, at: u32, shrink: bool) -> Result<Segment, EngineError> {
        let confirmed = self.confirmed_len();
        if at > confirmed {
            return Err(EngineError::InvalidRange {
                start: at,
                end: confirmed,
                len: confirmed,
            });
        }
        if self.hybrid() && self.recurrent.is_none() {
            return Err(EngineError::NoRecurrentState);
        }
        let cap = if shrink {
            round_up(at).min(self.cap)
        } else {
            self.cap
        };
        let bytes = u64::from(confirmed - at) * self.info.kv_bytes_per_token
            + if self.hybrid() {
                self.info.recurrent_bytes
            } else {
                0
            };
        let after = self.live_bytes(cap, at, false) + bytes;
        self.ensure_free(after.saturating_sub(self.current_live_bytes()))?;
        self.rollback();
        self.current = None;
        let tail = self.tokens.split_off(at as usize);
        let recurrent = self.recurrent.take();
        self.cap = cap;
        let id = SegmentId(self.id());
        self.segments.insert(
            id,
            StoredSegment {
                start: at,
                tokens: tail,
                recurrent,
                bytes,
            },
        );
        Ok(Segment {
            id,
            start: at,
            end: confirmed,
            bytes,
        })
    }

    fn append(&mut self, id: SegmentId, consume: bool) -> Result<(), EngineError> {
        let seg = self
            .segments
            .get(&id)
            .ok_or(EngineError::UnknownSegment(id))?;
        let end = seg.start + seg.tokens.len() as u32;
        let confirmed = self.confirmed_len();
        if confirmed != seg.start {
            return Err(EngineError::InvalidRange {
                start: seg.start,
                end,
                len: confirmed,
            });
        }
        let cap = if end > self.cap {
            round_up(end).min(self.budget.kv_capacity_tokens).max(end)
        } else {
            self.cap
        };
        let freed = if consume { seg.bytes } else { 0 };
        let after = self.live_bytes(cap, end, seg.recurrent.is_some());
        self.ensure_free(after.saturating_sub(self.current_live_bytes() + freed))?;
        let seg = if consume {
            self.segments.remove(&id).expect("checked above")
        } else {
            seg.clone()
        };
        self.rollback();
        self.current = None;
        self.cap = cap;
        self.tokens.extend_from_slice(&seg.tokens);
        if self.hybrid() {
            self.recurrent = seg.recurrent;
        }
        Ok(())
    }

    fn drop_segment(&mut self, id: SegmentId) {
        self.segments.remove(&id);
    }

    fn reset(&mut self) {
        self.pending = None;
        self.current = None;
        self.tokens.clear();
        self.cap = 0;
        self.grown = false;
        self.zero_recurrent();
    }

    fn reset_all(&mut self) {
        self.reset();
        self.checkpoints.clear();
        self.segments.clear();
        self.reset_all_calls.fetch_add(1, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mrun_engine::chunk_len;

    const RB: u64 = 1_000;

    fn engine(capacity: u32) -> ScriptedEngine {
        ScriptedEngine::new(vec![7, 8, 9], capacity)
    }

    fn hybrid(capacity: u32) -> ScriptedEngine {
        ScriptedEngine::new_hybrid(vec![7, 8, 9], capacity, RB)
    }

    fn seq(range: std::ops::Range<u32>) -> Vec<u32> {
        range.collect()
    }

    fn prefill(engine: &mut ScriptedEngine, tokens: &[u32]) -> Prefilled {
        prefill_plan(engine, tokens, &PrefillPlan::default(), &CancelToken::new())
    }

    fn prefill_plan(
        engine: &mut ScriptedEngine,
        tokens: &[u32],
        plan: &PrefillPlan,
        cancel: &CancelToken,
    ) -> Prefilled {
        engine.prefill(tokens, plan, cancel, &mut |_| {}).unwrap()
    }

    #[test]
    fn chunks_follow_depth_and_plan() {
        let mut engine = engine(100_000);
        engine.reserve(0, 60_000).unwrap();
        let plan = PrefillPlan {
            boundaries: &[100],
            checkpoints: &[300],
        };
        let done = prefill_plan(&mut engine, &seq(0..50_000), &plan, &CancelToken::new());
        assert_eq!(done.committed, 50_000);
        assert_eq!(done.checkpoints.len(), 1);
        assert_eq!(done.checkpoints[0].position, 300);
        assert_eq!(engine.chunks[..3], [(0, 100), (100, 300), (300, 2348)]);
        for &(start, end) in &engine.chunks[2..] {
            assert_eq!(end - start, chunk_len(start).min(50_000 - start), "{start}");
        }
        assert_eq!(engine.tokens(), seq(0..50_000));
    }

    #[test]
    fn allocation_grows_lazily_within_the_reservation() {
        let mut engine = engine(20_000);
        engine.reserve(0, 10_000).unwrap();
        assert_eq!(engine.allocated(), 0, "reserve allocates nothing");
        prefill(&mut engine, &seq(0..100));
        assert_eq!(engine.cap(), 4352, "input + 4,096, rounded to 256");
        assert_eq!(engine.allocated(), 4352 * KV_BYTES);
        prefill(&mut engine, &seq(0..4300));
        assert_eq!(engine.cap(), 8448, "later growth adds 4,096");
        prefill(&mut engine, &seq(0..5600));
        assert_eq!(engine.cap(), 10_000, "clamped to the reservation");
        assert_eq!(
            engine.prefill(
                &[1],
                &PrefillPlan::default(),
                &CancelToken::new(),
                &mut |_| {}
            ),
            Err(EngineError::ReservationExceeded {
                needed: 10_001,
                reserved: 10_000
            })
        );
        assert_eq!(
            engine.reserve(0, 20_001),
            Err(EngineError::Capacity {
                requested: 20_001,
                available: 20_000
            })
        );
    }

    #[test]
    fn needs_memory_reports_the_shortfall_and_changes_nothing() {
        let mut engine = engine(1_000);
        engine.reserve(0, 1_000).unwrap();
        prefill(&mut engine, &seq(0..10));
        assert_eq!(engine.allocated(), engine.budget().state_capacity_bytes);
        let segment = engine.split_off(5, true).unwrap();
        engine.reserve(5, 1_000).unwrap();
        let before = (engine.tokens().to_vec(), engine.allocated());
        assert_eq!(
            engine.prefill(
                &seq(0..900),
                &PrefillPlan::default(),
                &CancelToken::new(),
                &mut |_| {}
            ),
            Err(EngineError::NeedsMemory {
                bytes: 5 * KV_BYTES
            })
        );
        assert_eq!((engine.tokens().to_vec(), engine.allocated()), before);
        engine.drop_segment(segment.id);
        assert_eq!(prefill(&mut engine, &seq(0..900)).committed, 900);
        assert_eq!(engine.allocated(), engine.budget().state_capacity_bytes);
    }

    #[test]
    fn cancel_mid_chunk_keeps_only_whole_chunks() {
        let mut engine = engine(10_000);
        engine.reserve(0, 10_000).unwrap();
        // Four layers per chunk: layer 6 is inside the second chunk.
        engine.cancel_after_layers = Some(6);
        let plan = PrefillPlan {
            boundaries: &[10, 20],
            checkpoints: &[],
        };
        let done = prefill_plan(&mut engine, &seq(0..30), &plan, &CancelToken::new());
        assert_eq!(done.committed, 10);
        assert!(!done.yielded);
        assert_eq!(engine.tokens(), seq(0..10));
        assert_eq!(engine.layers_run, 6);
    }

    #[test]
    fn yield_finishes_the_chunk_and_resumes_with_identical_splits() {
        let plan = PrefillPlan {
            boundaries: &[10, 20],
            checkpoints: &[],
        };
        let mut straight = engine(10_000);
        straight.reserve(0, 10_000).unwrap();
        prefill_plan(&mut straight, &seq(0..3000), &plan, &CancelToken::new());

        let mut yielding = engine(10_000);
        yielding.reserve(0, 10_000).unwrap();
        yielding.yield_after_layers = Some(6);
        let cancel = CancelToken::new();
        let first = prefill_plan(&mut yielding, &seq(0..3000), &plan, &cancel);
        assert_eq!((first.committed, first.yielded), (20, true));
        cancel.clear_yield();
        let rest = prefill_plan(&mut yielding, &seq(20..3000), &plan, &cancel);
        assert_eq!((rest.committed, rest.yielded), (2980, false));
        assert_eq!(yielding.chunks, straight.chunks);
        assert_eq!(yielding.tokens(), straight.tokens());
    }

    #[test]
    fn pipelined_step_is_reused_or_rolled_back() {
        let mut engine = engine(100);
        engine.reserve(0, 100).unwrap();
        prefill(&mut engine, &[1, 2]);
        let step = engine.forward(3).unwrap();
        assert_eq!(engine.top_logprobs(&step, 2).unwrap()[0].0, 7);
        assert_eq!(engine.sample(step, &Sampling::GREEDY, None, true), Ok(7));
        assert!(engine.has_pending());
        assert_eq!(engine.active_len(), 4);
        // Same token: no recomputation.
        let step = engine.forward(7).unwrap();
        assert_eq!(engine.processed, [1, 2, 3, 7]);
        assert_eq!(engine.sample(step, &Sampling::GREEDY, None, true), Ok(8));
        // Another token: the pending step is rolled back first.
        let step = engine.forward(42).unwrap();
        assert_eq!(engine.tokens(), [1, 2, 3, 7, 42]);
        assert_eq!(
            engine.sample(Step::new(999), &Sampling::GREEDY, None, true),
            Err(EngineError::InvalidStep)
        );
        assert_eq!(engine.sample(step, &Sampling::GREEDY, None, true), Ok(9));
        engine.trim(5).unwrap();
        assert!(!engine.has_pending());
        assert_eq!(engine.tokens(), [1, 2, 3, 7, 42]);
    }

    #[test]
    fn no_pipelining_without_room_in_the_allocation() {
        let mut engine = engine(100);
        engine.reserve(0, 3).unwrap();
        prefill(&mut engine, &[1, 2]);
        assert_eq!(engine.cap(), 3);
        let step = engine.forward(3).unwrap();
        engine.sample(step, &Sampling::GREEDY, None, true).unwrap();
        assert!(!engine.has_pending());
    }

    #[test]
    fn hybrid_state_trims_only_at_its_position() {
        let mut engine = hybrid(100);
        engine.reserve(0, 100).unwrap();
        prefill(&mut engine, &seq(0..10));
        assert_eq!(engine.allocated(), engine.cap() as u64 * KV_BYTES + RB);
        assert_eq!(
            engine.trim(5),
            Err(EngineError::NotTrimmable { len: 5, active: 10 })
        );
        let step = engine.forward(10).unwrap();
        engine.sample(step, &Sampling::GREEDY, None, true).unwrap();
        assert_eq!(engine.active_len(), 12);
        engine.trim(11).unwrap();
        assert_eq!(engine.tokens(), seq(0..11));
        engine.trim(0).unwrap();
        prefill(&mut engine, &[5]);
        assert_eq!(engine.tokens(), [5]);
    }

    #[test]
    fn hybrid_checkpoints_restore_earlier_positions() {
        let mut engine = hybrid(100);
        engine.reserve(0, 100).unwrap();
        let plan = PrefillPlan {
            boundaries: &[],
            checkpoints: &[4],
        };
        let done = prefill_plan(&mut engine, &seq(0..8), &plan, &CancelToken::new());
        let at4 = done.checkpoints[0];
        assert_eq!((at4.position, at4.bytes), (4, RB));
        let at8 = engine.checkpoint().unwrap();
        assert_eq!(engine.allocated(), engine.cap() as u64 * KV_BYTES + 3 * RB);
        engine.restore_checkpoint(at4.id).unwrap();
        assert_eq!(engine.tokens(), seq(0..4));
        prefill(&mut engine, &[40, 50]);
        // A checkpoint of another sequence leaves inconsistent state, caught on next use.
        engine.trim(0).unwrap();
        prefill(&mut engine, &seq(100..110));
        engine.restore_checkpoint(at8.id).unwrap();
        assert!(matches!(engine.forward(1), Err(EngineError::Backend(_))));
        engine.drop_checkpoint(at8.id);
        assert_eq!(
            engine.restore_checkpoint(at8.id),
            Err(EngineError::UnknownCheckpoint(at8.id))
        );
    }

    #[test]
    fn too_many_planned_checkpoints_are_refused() {
        let mut engine = hybrid(100);
        engine.reserve(0, 100).unwrap();
        let plan = PrefillPlan {
            boundaries: &[],
            checkpoints: &[1, 2, 3, 4],
        };
        let error = engine
            .prefill(&seq(0..8), &plan, &CancelToken::new(), &mut |_| {})
            .unwrap_err();
        assert_eq!(
            error,
            EngineError::InvalidRange {
                start: 0,
                end: 8,
                len: 0
            }
        );
        assert_eq!(engine.active_len(), 0);
    }

    #[test]
    fn split_off_and_append_move_state_without_duplicating_it() {
        let mut engine = engine(2_000);
        engine.reserve(0, 1_000).unwrap();
        prefill(&mut engine, &seq(0..600));
        let cap = engine.cap();
        let total = engine.allocated();
        let tail = engine.split_off(300, false).unwrap();
        assert_eq!(
            (tail.start, tail.end, tail.bytes),
            (300, 600, 300 * KV_BYTES)
        );
        assert_eq!(engine.segment_range(tail.id), Some((300, 600)));
        assert_eq!(engine.cap(), cap, "kept for an append");
        engine.append(tail.id, true).unwrap();
        assert_eq!(engine.tokens(), seq(0..600));
        assert_eq!(engine.allocated(), total);
        assert_eq!(engine.segment_range(tail.id), None);

        let tail = engine.split_off(300, true).unwrap();
        assert_eq!(engine.cap(), 512, "shrunk to round_up(300, 256)");
        prefill(&mut engine, &seq(1000..1010));
        let other = engine.split_off(300, true).unwrap();
        engine.append(tail.id, false).unwrap();
        assert_eq!(engine.tokens(), seq(0..600));
        assert_eq!(engine.segment_range(tail.id), Some((300, 600)), "copied");
        assert_eq!(
            engine.append(other.id, true),
            Err(EngineError::InvalidRange {
                start: 300,
                end: 310,
                len: 600
            })
        );
    }

    #[test]
    fn hybrid_split_off_moves_the_recurrent_state() {
        let mut engine = hybrid(1_000);
        engine.reserve(0, 1_000).unwrap();
        let plan = PrefillPlan {
            boundaries: &[],
            checkpoints: &[4],
        };
        let at4 = prefill_plan(&mut engine, &seq(0..8), &plan, &CancelToken::new()).checkpoints[0];
        let before = engine.allocated();
        let tail = engine.split_off(4, false).unwrap();
        assert_eq!(tail.bytes, 4 * KV_BYTES + RB);
        assert_eq!(engine.allocated(), before + 4 * KV_BYTES, "recurrent moved");
        assert_eq!(engine.forward(1), Err(EngineError::NoRecurrentState));
        assert_eq!(
            engine.prefill(
                &[1],
                &PrefillPlan::default(),
                &CancelToken::new(),
                &mut |_| {}
            ),
            Err(EngineError::NoRecurrentState)
        );
        engine.append(tail.id, true).unwrap();
        assert_eq!(engine.tokens(), seq(0..8));
        assert!(engine.forward(8).is_ok());

        let tail = engine.split_off(4, false).unwrap();
        engine.restore_checkpoint(at4.id).unwrap();
        prefill(&mut engine, &[9, 9]);
        assert_eq!(engine.tokens(), [0, 1, 2, 3, 9, 9]);
        engine.drop_segment(tail.id);
    }

    #[test]
    fn family_profiles_match_the_registry_byte_costs_and_capacities_on_48_gb() {
        use mrun_engine::budget::GIB;
        // (family, token capacity range) from the worked 48 GB example.
        for (family, capacity) in [
            ("qwen3_5_moe", 262_144..=262_144),
            ("qwen3_moe", 105_000..=120_000),
            ("gpt_oss", 131_072..=131_072),
            ("qwen3", 40_960..=40_960),
        ] {
            let profile = FamilyProfile::get(family).unwrap();
            let budget = profile.budget(48 * GIB, 2 * GIB).unwrap();
            assert_eq!(budget.budget, 28 * GIB);
            assert!(
                capacity.contains(&budget.kv_capacity_tokens),
                "{family}: {}",
                budget.kv_capacity_tokens
            );
            let mut engine = ScriptedEngine::for_family(vec![7], profile, budget);
            assert_eq!(engine.info().family, family);
            assert_eq!(engine.info().layers(), profile.layers);
            assert_eq!(engine.info().is_hybrid(), profile.recurrent_bytes > 0);
            let total = budget.kv_capacity_tokens;
            engine.reserve(0, total).unwrap();
            prefill(&mut engine, &seq(0..total));
            assert!(
                engine.allocated() <= budget.state_capacity_bytes,
                "{family}"
            );
        }
        assert_eq!(
            PROFILES
                .iter()
                .map(|profile| (0..profile.layers)
                    .filter(|&i| (profile.layer_kind)(i) == LayerKind::Linear)
                    .count())
                .collect::<Vec<_>>(),
            [30, 0, 0, 0]
        );
    }

    #[test]
    fn small_machines_refuse_or_shrink_large_families() {
        use mrun_engine::budget::GIB;
        let coder = FamilyProfile::get("qwen3_moe").unwrap();
        assert!(matches!(
            coder.budget(16 * GIB, 2 * GIB),
            Err(BudgetError::DoesNotFit { .. })
        ));
        let qwen3 = FamilyProfile::get("qwen3").unwrap();
        let small = qwen3.budget(16 * GIB, GIB).unwrap();
        assert!(small.kv_capacity_tokens < qwen3.max_position);
        assert!(small.kv_capacity_tokens > 4096);
    }

    #[test]
    fn reset_releases_the_live_state_only() {
        let mut engine = hybrid(1_000);
        engine.reserve(0, 1_000).unwrap();
        prefill(&mut engine, &seq(0..8));
        engine.checkpoint().unwrap();
        engine.reset();
        assert_eq!(engine.active_len(), 0);
        assert_eq!(engine.allocated(), RB);
        assert_eq!(engine.checkpoint_count(), 1);
        prefill(&mut engine, &[1]);
    }

    #[test]
    fn reset_all_also_drops_segments_and_checkpoints() {
        let mut engine = hybrid(1_000);
        engine.reserve(0, 1_000).unwrap();
        prefill(&mut engine, &seq(0..8));
        let checkpoint = engine.checkpoint().unwrap();
        let tail = engine.split_off(4, true).unwrap();
        engine.reset_all();
        assert_eq!(engine.active_len(), 0);
        assert_eq!(engine.allocated(), 0);
        assert_eq!(engine.checkpoint_count(), 0);
        assert_eq!(engine.segment_range(tail.id), None);
        assert_eq!(
            engine.restore_checkpoint(checkpoint.id),
            Err(EngineError::UnknownCheckpoint(checkpoint.id))
        );
        prefill(&mut engine, &[1]);
    }
}
