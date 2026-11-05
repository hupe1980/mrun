//! Bookkeeping every [`Engine`](crate::Engine) shares (contract v2, D028): the live length with
//! its pending pipelined step, the attention capacity and the growth limit, the stored
//! checkpoints and segments, the byte accounting and the poisoned state after a failed move.
//!
//! `NuvecEngine` and the scripted test double both keep their state here, so the double cannot
//! credit memory the real engine would not. The pattern of every operation is the same: a pure
//! `plan_*` call validates and checks memory without changing anything (so an error leaves the
//! state as it was), the engine rolls back a pending step and does the physical work, then a
//! `commit_*` call records the result. Physical failures after the plan poison the ledger.
//!
//! The payloads are the engine's: `P` a pending step, `S` a segment's contents, `C` a
//! checkpoint's position-bound state.

use std::collections::HashMap;

use crate::engine::{
    Checkpoint, CheckpointId, EngineError, GROWTH, MemoryBudget, ModelInfo, Segment, SegmentId,
    align, grown_capacity,
};

/// Capacity to keep after cutting the live state to `len`: `align(len) + GROWTH` when more than
/// that is allocated, so a cut releases memory but a following prefill does not regrow at once.
pub fn shrunk_capacity(capacity: u32, len: u32) -> Option<u32> {
    let target = align(len).saturating_add(GROWTH);
    (capacity > target).then_some(target)
}

/// Capacity for attaching a segment that ends at `end`: `align(end)`, clamped to the request
/// limit when the segment fits it (a stored branch may be longer than the request), never below
/// `end` or the current capacity. `ReservationExceeded` beyond the token capacity.
pub fn attach_capacity(
    capacity: u32,
    end: u32,
    limit: u32,
    kv_capacity: u32,
) -> Result<u32, EngineError> {
    if end > kv_capacity {
        return Err(EngineError::ReservationExceeded {
            needed: end,
            reserved: kv_capacity,
        });
    }
    let target = if end <= limit {
        align(end).min(limit)
    } else {
        align(end)
    };
    Ok(capacity.max(target.min(kv_capacity).max(end)))
}

/// A planned `detach_tail`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DetachPlan {
    pub start: u32,
    /// The length before a pending step.
    pub end: u32,
    /// Live capacity afterwards: `align(start)`, at most the current one.
    pub capacity: u32,
    pub bytes: u64,
}

/// A planned `attach`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AttachPlan {
    pub segment: Segment,
    pub capacity: u32,
}

/// What a `trim` asks of the engine after the ledger recorded it.
#[derive(Debug)]
pub struct Trim<P> {
    /// The pending step that was rolled back (the engine restores what it fed).
    pub rolled_back: Option<P>,
    /// The current step's logits describe a position that is gone.
    pub cut_step: bool,
    /// Position-bound state goes back to zeros (`len == 0`).
    pub zero_recurrent: bool,
    /// New, smaller capacity to shrink the live allocation to.
    pub shrink: Option<u32>,
}

#[derive(Debug)]
struct Stored<T, D> {
    meta: T,
    data: D,
}

#[derive(Debug)]
pub struct Ledger<P, S, C> {
    pub budget: MemoryBudget,
    kv_bytes_per_token: u64,
    /// Position-bound bytes of the live state; 0 for attention-only models.
    recurrent_bytes: u64,
    /// Positions in the live state, a pending token included.
    len: u32,
    capacity: u32,
    /// Growth ceiling from `set_limit`.
    limit: u32,
    /// Hybrids: the live position-bound state exists (false after `detach_tail`).
    recurrent_live: bool,
    pending: Option<P>,
    segments: HashMap<SegmentId, Stored<Segment, S>>,
    checkpoints: HashMap<CheckpointId, Stored<Checkpoint, C>>,
    next_id: u64,
    poisoned: Option<String>,
}

impl<P, S, C> Ledger<P, S, C> {
    pub fn new(info: &ModelInfo, budget: MemoryBudget) -> Self {
        Self {
            budget,
            kv_bytes_per_token: info.kv_bytes_per_token,
            recurrent_bytes: if info.is_hybrid() {
                info.recurrent_bytes
            } else {
                0
            },
            len: 0,
            capacity: 0,
            limit: 0,
            recurrent_live: true,
            pending: None,
            segments: HashMap::new(),
            checkpoints: HashMap::new(),
            next_id: 0,
            poisoned: None,
        }
    }

    pub fn hybrid(&self) -> bool {
        self.recurrent_bytes > 0
    }

    /// Positions held, a pending token included.
    pub fn len(&self) -> u32 {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Positions before a pending step.
    pub fn confirmed_len(&self) -> u32 {
        self.len - u32::from(self.pending.is_some())
    }

    pub fn capacity(&self) -> u32 {
        self.capacity
    }

    pub fn recurrent_live(&self) -> bool {
        self.recurrent_live
    }

    pub fn pending(&self) -> Option<&P> {
        self.pending.as_ref()
    }

    /// Bytes of the live state at `capacity`, with or without its position-bound state.
    pub fn live_bytes(&self, capacity: u32, recurrent: bool) -> u64 {
        u64::from(capacity) * self.kv_bytes_per_token
            + if recurrent { self.recurrent_bytes } else { 0 }
    }

    fn current_live_bytes(&self) -> u64 {
        self.live_bytes(self.capacity, self.recurrent_live)
    }

    /// Live allocation + segments + checkpoints.
    pub fn allocated(&self) -> u64 {
        self.current_live_bytes()
            + self.segments.values().map(|s| s.meta.bytes).sum::<u64>()
            + self.checkpoints.values().map(|c| c.meta.bytes).sum::<u64>()
    }

    /// `NeedsMemory` with the shortfall when `bytes` more do not fit the state capacity.
    pub fn ensure_free(&self, bytes: u64) -> Result<(), EngineError> {
        let free = self
            .budget
            .state_capacity_bytes
            .saturating_sub(self.allocated());
        if bytes > free {
            return Err(EngineError::NeedsMemory {
                bytes: bytes - free,
            });
        }
        Ok(())
    }

    pub fn set_limit(&mut self, total: u32) -> Result<(), EngineError> {
        self.usable()?;
        let available = self.budget.kv_capacity_tokens;
        if total > available {
            return Err(EngineError::Capacity {
                requested: total,
                available,
            });
        }
        self.limit = total;
        Ok(())
    }

    /// A fresh id for a step, checkpoint or segment.
    pub fn id(&mut self) -> u64 {
        self.next_id += 1;
        self.next_id
    }

    /// `Poisoned` after a failed move, until `reset_all`.
    pub fn usable(&self) -> Result<(), EngineError> {
        match &self.poisoned {
            Some(why) => Err(EngineError::Poisoned(why.clone())),
            None => Ok(()),
        }
    }

    /// Usable and, for hybrids, holding live position-bound state.
    pub fn check_recurrent(&self) -> Result<(), EngineError> {
        self.usable()?;
        if self.hybrid() && !self.recurrent_live {
            return Err(EngineError::NoRecurrentState);
        }
        Ok(())
    }

    /// Records a failed move; the returned error is what the engine reports.
    pub fn poison(&mut self, why: impl Into<String>) -> EngineError {
        let why = why.into();
        self.poisoned = Some(why.clone());
        EngineError::Poisoned(why)
    }

    /// Plans appending `n` positions after rolling back a pending step: the capacity to grow
    /// to, if any, checked against the limit and the free memory. Changes nothing.
    pub fn plan_append(&self, n: u32) -> Result<Option<u32>, EngineError> {
        self.plan_append_with_checkpoints(n, 0)
    }

    /// [`Self::plan_append`] that also stores `checkpoints` checkpoints of the position-bound
    /// state taken inside the appended positions (their bytes are checked with the growth).
    pub fn plan_append_with_checkpoints(
        &self,
        n: u32,
        checkpoints: u32,
    ) -> Result<Option<u32>, EngineError> {
        self.check_recurrent()?;
        let needed = self.confirmed_len().saturating_add(n);
        let limit = self.limit.min(self.budget.kv_capacity_tokens);
        let next = grown_capacity(self.capacity, needed, limit)?;
        let growth = next.map_or(0, |next| {
            u64::from(next - self.capacity) * self.kv_bytes_per_token
        });
        let stored = u64::from(checkpoints) * self.recurrent_bytes;
        if growth + stored > 0 {
            self.ensure_free(growth + stored)?;
        }
        Ok(next)
    }

    /// Records a new attention capacity (growth, or its undoing after a cancelled chunk).
    pub fn set_capacity(&mut self, capacity: u32) {
        self.capacity = capacity;
    }

    /// Positions appended to the live state.
    pub fn advance(&mut self, n: u32) {
        self.len += n;
    }

    /// Whether a pipelined step fits the allocation.
    pub fn has_room(&self) -> bool {
        self.len < self.capacity
    }

    /// Records a fed pipelined step.
    pub fn push_pending(&mut self, pending: P) {
        debug_assert!(self.pending.is_none());
        self.len += 1;
        self.pending = Some(pending);
    }

    /// Takes the pending step when `reuse` accepts it: its position is kept (confirmed).
    pub fn reuse_pending(&mut self, reuse: impl FnOnce(&P) -> bool) -> Option<P> {
        self.pending.take_if(|p| reuse(p))
    }

    /// Drops a pending step, un-counting its position; the engine restores what it fed.
    pub fn roll_back(&mut self) -> Option<P> {
        let pending = self.pending.take()?;
        self.len -= 1;
        Some(pending)
    }

    /// Validates and records a cut to `len` (see [`crate::Engine::trim`]).
    pub fn trim(&mut self, len: u32) -> Result<Trim<P>, EngineError> {
        self.usable()?;
        let (active, confirmed) = (self.len, self.confirmed_len());
        if len > active {
            return Err(EngineError::InvalidRange {
                start: len,
                end: len,
                len: active,
            });
        }
        if len == active && len > 0 {
            // Keeping every position confirms a pending step.
            self.pending = None;
            return Ok(Trim {
                rolled_back: None,
                cut_step: false,
                zero_recurrent: false,
                shrink: None,
            });
        }
        if self.hybrid() && self.recurrent_live && len != 0 && len != confirmed {
            return Err(EngineError::NotTrimmable { len, active });
        }
        let rolled_back = self.roll_back();
        self.len = len;
        let zero_recurrent = len == 0 && self.hybrid();
        if zero_recurrent {
            // Cutting to nothing zeroes the position-bound state (it describes position 0).
            self.recurrent_live = true;
        }
        let shrink = shrunk_capacity(self.capacity, len);
        if let Some(capacity) = shrink {
            self.capacity = capacity;
        }
        Ok(Trim {
            rolled_back,
            cut_step: len < confirmed,
            zero_recurrent,
            shrink,
        })
    }

    /// Bytes of a checkpoint at the current position, checked against the free memory.
    pub fn plan_checkpoint(&self) -> Result<u64, EngineError> {
        self.check_recurrent()?;
        let bytes = self.recurrent_bytes;
        self.ensure_free(bytes)?;
        Ok(bytes)
    }

    /// Stores a checkpoint of the state at `len()` (after the engine rolled back a pending step).
    pub fn commit_checkpoint(&mut self, bytes: u64, data: C) -> Checkpoint {
        debug_assert!(self.pending.is_none());
        let checkpoint = Checkpoint {
            id: CheckpointId(self.id()),
            position: self.len,
            bytes,
        };
        self.checkpoints.insert(
            checkpoint.id,
            Stored {
                meta: checkpoint,
                data,
            },
        );
        checkpoint
    }

    /// Stores a checkpoint of the state at `position` (at most `len()`), taken during the
    /// append that reached `len()` (in-chunk checkpoints, 003 SC-009).
    pub fn commit_checkpoint_at(&mut self, position: u32, bytes: u64, data: C) -> Checkpoint {
        debug_assert!(position <= self.len);
        let checkpoint = Checkpoint {
            id: CheckpointId(self.id()),
            position,
            bytes,
        };
        self.checkpoints.insert(
            checkpoint.id,
            Stored {
                meta: checkpoint,
                data,
            },
        );
        checkpoint
    }

    pub fn checkpoint_data(&self, id: CheckpointId) -> Option<&C> {
        self.checkpoints.get(&id).map(|c| &c.data)
    }

    /// Validates a restore of checkpoint `id` and checks the memory it needs. Changes nothing.
    pub fn plan_restore(&self, id: CheckpointId) -> Result<u32, EngineError> {
        self.usable()?;
        let stored = self
            .checkpoints
            .get(&id)
            .ok_or(EngineError::UnknownCheckpoint(id))?;
        let (position, confirmed) = (stored.meta.position, self.confirmed_len());
        if position > confirmed {
            return Err(EngineError::InvalidRange {
                start: position,
                end: position,
                len: confirmed,
            });
        }
        let extra = self
            .live_bytes(self.capacity, true)
            .saturating_sub(self.current_live_bytes());
        self.ensure_free(extra)?;
        Ok(position)
    }

    /// Records a restore to `position` (after the engine rolled back a pending step).
    pub fn commit_restore(&mut self, position: u32) {
        debug_assert!(self.pending.is_none());
        self.len = position;
        self.recurrent_live = true;
    }

    pub fn drop_checkpoint(&mut self, id: CheckpointId) -> bool {
        self.checkpoints.remove(&id).is_some()
    }

    /// Plans moving `[at, confirmed_len())` into a segment. Layer by layer the live allocation
    /// shrinks as the tail moves out: only the net change has to fit. Changes nothing.
    pub fn plan_detach(&self, at: u32) -> Result<DetachPlan, EngineError> {
        self.usable()?;
        let end = self.confirmed_len();
        if at > end {
            return Err(EngineError::InvalidRange {
                start: at,
                end,
                len: end,
            });
        }
        self.check_recurrent()?;
        let capacity = align(at).min(self.capacity);
        let bytes = u64::from(end - at) * self.kv_bytes_per_token + self.recurrent_bytes;
        let after = self.live_bytes(capacity, false) + bytes;
        self.ensure_free(after.saturating_sub(self.current_live_bytes()))?;
        Ok(DetachPlan {
            start: at,
            end,
            capacity,
            bytes,
        })
    }

    /// Records a finished detach (after the engine rolled back a pending step and moved the
    /// tail).
    pub fn commit_detach(&mut self, plan: DetachPlan, data: S) -> Segment {
        debug_assert!(self.pending.is_none());
        let segment = Segment {
            id: SegmentId(self.id()),
            start: plan.start,
            end: plan.end,
            bytes: plan.bytes,
        };
        self.len = plan.start;
        self.capacity = plan.capacity;
        if self.hybrid() {
            self.recurrent_live = false;
        }
        self.segments.insert(
            segment.id,
            Stored {
                meta: segment,
                data,
            },
        );
        segment
    }

    /// Plans attaching segment `id` at the length before a pending step. The segment is freed
    /// layer by layer as it is copied: its bytes count toward the room. Changes nothing.
    pub fn plan_attach(&self, id: SegmentId) -> Result<AttachPlan, EngineError> {
        self.usable()?;
        let segment = self
            .segments
            .get(&id)
            .ok_or(EngineError::UnknownSegment(id))?
            .meta;
        let len = self.confirmed_len();
        if len != segment.start {
            return Err(EngineError::InvalidRange {
                start: segment.start,
                end: segment.end,
                len,
            });
        }
        let capacity = attach_capacity(
            self.capacity,
            segment.end,
            self.limit,
            self.budget.kv_capacity_tokens,
        )?;
        let after = self.live_bytes(capacity, true);
        self.ensure_free(after.saturating_sub(self.current_live_bytes() + segment.bytes))?;
        Ok(AttachPlan { segment, capacity })
    }

    /// Takes a planned segment's contents out for the move.
    pub fn take_segment(&mut self, id: SegmentId) -> Option<S> {
        self.segments.remove(&id).map(|s| s.data)
    }

    /// Records a finished attach (after the engine rolled back a pending step and moved the
    /// segment in).
    pub fn commit_attach(&mut self, plan: AttachPlan) {
        debug_assert!(self.pending.is_none());
        self.len = plan.segment.end;
        self.capacity = plan.capacity;
        self.recurrent_live = true;
    }

    /// The range a stored segment holds.
    pub fn segment(&self, id: SegmentId) -> Option<Segment> {
        self.segments.get(&id).map(|s| s.meta)
    }

    pub fn drop_segment(&mut self, id: SegmentId) -> bool {
        self.segments.remove(&id).is_some()
    }

    pub fn checkpoint_count(&self) -> usize {
        self.checkpoints.len()
    }

    /// Empties the live state; segments and checkpoints stay, and so does a poisoned state.
    pub fn reset(&mut self) {
        self.len = 0;
        self.capacity = 0;
        self.pending = None;
        self.recurrent_live = true;
    }

    /// `reset`, dropping every segment and checkpoint and clearing a poisoned state.
    pub fn reset_all(&mut self) {
        self.reset();
        self.segments.clear();
        self.checkpoints.clear();
        self.poisoned = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::LayerKind;

    const KV: u64 = 16;
    const RB: u64 = 1_000;

    type Test = Ledger<u32, (), ()>;

    fn ledger(hybrid: bool, tokens: u32) -> Test {
        let info = ModelInfo {
            identity: "test".into(),
            family: "test".into(),
            layer_kinds: if hybrid {
                vec![LayerKind::Linear, LayerKind::FullAttention]
            } else {
                vec![LayerKind::FullAttention]
            },
            kv_heads: 1,
            head_dim: 1,
            vocab: 10,
            max_position: tokens,
            kv_bytes_per_token: KV,
            recurrent_bytes: if hybrid { RB } else { 0 },
            eos_tokens: Vec::new(),
            ..ModelInfo::default()
        };
        let budget = MemoryBudget {
            state_capacity_bytes: u64::from(tokens) * KV + 3 * RB,
            kv_capacity_tokens: tokens,
            ..MemoryBudget::default()
        };
        let mut ledger = Ledger::new(&info, budget);
        ledger.set_limit(tokens).unwrap();
        ledger
    }

    fn append(ledger: &mut Test, n: u32) {
        if let Some(capacity) = ledger.plan_append(n).unwrap() {
            ledger.set_capacity(capacity);
        }
        ledger.advance(n);
    }

    #[test]
    fn plans_change_nothing_on_error() {
        let mut l = ledger(false, 100_000);
        l.budget.state_capacity_bytes = 1_000 * KV;
        append(&mut l, 0);
        assert_eq!(
            l.plan_append(100),
            Err(EngineError::NeedsMemory {
                bytes: (4352 - 1_000) * KV
            })
        );
        assert_eq!((l.len(), l.capacity(), l.allocated()), (0, 0, 0));
        // A pending step stays when the plan fails.
        l.budget.state_capacity_bytes = 100_000 * KV;
        append(&mut l, 10);
        l.push_pending(7);
        l.set_limit(20).unwrap();
        assert!(l.plan_append(5_000).is_err());
        assert_eq!((l.len(), l.pending()), (11, Some(&7)));
        assert!(l.plan_attach(SegmentId(99)).is_err());
        assert!(l.plan_detach(50).is_err());
        assert_eq!(l.len(), 11);
    }

    #[test]
    fn appending_after_a_pending_step_counts_from_the_confirmed_length() {
        let mut l = ledger(false, 4352);
        append(&mut l, 4351);
        l.push_pending(1);
        assert!(!l.has_room());
        // 4351 confirmed + 1 fits the capacity once the pending step is rolled back.
        assert_eq!(l.plan_append(1), Ok(None));
        assert_eq!(l.roll_back(), Some(1));
        assert_eq!(l.len(), 4351);
    }

    #[test]
    fn trim_releases_capacity_beyond_one_growth_step() {
        let mut l = ledger(false, 100_000);
        append(&mut l, 20_000);
        assert_eq!(l.capacity(), 24_320);
        let cut = l.trim(19_990).unwrap();
        assert_eq!(cut.shrink, None, "within a growth step");
        let cut = l.trim(1_000).unwrap();
        assert_eq!(cut.shrink, Some(1_024 + GROWTH));
        assert_eq!(l.capacity(), 5_120);
        assert_eq!(l.allocated(), 5_120 * KV);
        assert_eq!(shrunk_capacity(4352, 0), Some(4096));
        assert_eq!(shrunk_capacity(4096, 0), None);
    }

    #[test]
    fn trim_rules_for_position_bound_state() {
        let mut l = ledger(true, 1_000);
        append(&mut l, 10);
        assert_eq!(
            l.trim(5).unwrap_err(),
            EngineError::NotTrimmable { len: 5, active: 10 }
        );
        l.push_pending(3);
        let cut = l.trim(10).unwrap();
        assert_eq!(cut.rolled_back, Some(3));
        assert!(!cut.cut_step);
        l.push_pending(4);
        assert!(l.trim(11).unwrap().rolled_back.is_none(), "confirmed");
        assert_eq!(l.len(), 11);
        let cut = l.trim(0).unwrap();
        assert!(cut.zero_recurrent && cut.cut_step);
    }

    #[test]
    fn attach_rolls_back_the_pending_step_and_allocates_exactly() {
        let mut l = ledger(false, 100_000);
        append(&mut l, 9_000);
        let plan = l.plan_detach(1_000).unwrap();
        assert_eq!((plan.capacity, plan.bytes), (1_024, 8_000 * KV));
        let segment = l.commit_detach(plan, ());
        assert_eq!((l.len(), l.capacity()), (1_000, 1_024));
        // A pending token after the detach point: attach compares the confirmed length.
        l.push_pending(5);
        l.set_limit(2_000).unwrap();
        let plan = l.plan_attach(segment.id).unwrap();
        assert_eq!(plan.capacity, 9_216, "align(end), not geometric growth");
        assert_eq!(l.len(), 1_001, "planning changes nothing");
        assert_eq!(l.roll_back(), Some(5));
        l.take_segment(segment.id).unwrap();
        l.commit_attach(plan);
        assert_eq!((l.len(), l.capacity()), (9_000, 9_216));
        assert_eq!(attach_capacity(0, 9_000, 9_100, 100_000), Ok(9_100));
        assert_eq!(attach_capacity(0, 9_000, 100, 100_000), Ok(9_216));
        assert_eq!(attach_capacity(20_000, 9_000, 100, 100_000), Ok(20_000));
        assert!(attach_capacity(0, 9_000, 100, 8_000).is_err());
    }

    #[test]
    fn hybrid_detach_moves_the_position_bound_bytes() {
        let mut l = ledger(true, 1_000);
        append(&mut l, 300);
        let before = l.allocated();
        let segment = l.commit_detach(l.plan_detach(100).unwrap(), ());
        assert_eq!(segment.bytes, 200 * KV + RB);
        assert!(l.allocated() <= before);
        assert_eq!(l.plan_append(1), Err(EngineError::NoRecurrentState));
        assert_eq!(l.plan_checkpoint(), Err(EngineError::NoRecurrentState));
    }

    #[test]
    fn a_poisoned_ledger_refuses_until_reset_all() {
        let mut l = ledger(false, 1_000);
        append(&mut l, 10);
        let error = l.poison("layer 3");
        assert_eq!(error, EngineError::Poisoned("layer 3".into()));
        assert_eq!(l.plan_append(1), Err(error.clone()));
        assert_eq!(l.trim(0).unwrap_err(), error);
        l.reset();
        assert_eq!(l.plan_detach(0).unwrap_err(), error);
        l.reset_all();
        assert_eq!(l.plan_append(1).map(|c| c.is_some()), Ok(true));
    }

    #[test]
    fn checkpoints_and_segments_count_toward_allocated() {
        let mut l = ledger(true, 1_000);
        append(&mut l, 10);
        let bytes = l.plan_checkpoint().unwrap();
        let checkpoint = l.commit_checkpoint(bytes, ());
        assert_eq!(checkpoint.position, 10);
        assert_eq!(l.allocated(), l.capacity() as u64 * KV + 2 * RB);
        assert_eq!(l.plan_restore(checkpoint.id), Ok(10));
        assert!(l.drop_checkpoint(checkpoint.id));
        assert!(matches!(
            l.plan_restore(checkpoint.id),
            Err(EngineError::UnknownCheckpoint(_))
        ));
    }
}
