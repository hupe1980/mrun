//! The engine contract: primitives the runtime drives. No MLX types appear here.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

/// Kind of a decoder layer. Attention layers (full or sliding-window) hold positional K/V that
/// can be cut at any position; linear layers hold position-bound recurrent state.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LayerKind {
    FullAttention,
    /// Attends to the last `window` positions of the full K/V.
    SlidingAttention {
        window: u32,
    },
    Linear,
}

/// What the runtime needs to know about the loaded model.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ModelInfo {
    /// Content identity (see [`crate::identity`]).
    pub identity: String,
    /// Registry family name (the checkpoint's `model_type`, e.g. `qwen3_5_moe`).
    pub family: String,
    pub layer_kinds: Vec<LayerKind>,
    pub kv_heads: u32,
    pub head_dim: u32,
    pub vocab: u32,
    pub max_position: u32,
    /// Bytes of K and V for one token across all attention layers.
    pub kv_bytes_per_token: u64,
    /// Bytes of recurrent state across all linear layers; 0 for attention-only models.
    pub recurrent_bytes: u64,
    pub eos_tokens: Vec<u32>,
}

impl ModelInfo {
    pub fn layers(&self) -> u32 {
        self.layer_kinds.len() as u32
    }

    /// Whether the model has recurrent state, which cannot be cut to an earlier position.
    pub fn is_hybrid(&self) -> bool {
        self.layer_kinds.contains(&LayerKind::Linear)
    }
}

/// Memory budget computed at startup: what mrun may use and how it is split.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct MemoryBudget {
    pub physical: u64,
    pub recommended_working_set: u64,
    pub reserve: u64,
    /// `min(fraction × physical, recommended) − reserve`.
    pub budget: u64,
    pub weights: u64,
    pub scratch: u64,
    /// Depth at which scratch was probed; token capacity never exceeds it.
    pub probed_depth: u32,
    /// `budget − weights − scratch`: the bound on [`Engine::allocated`].
    pub state_capacity_bytes: u64,
    pub kv_capacity_tokens: u32,
}

/// Sampling settings for one request. Temperature 0 means greedy.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sampling {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: u32,
    pub min_p: f32,
    pub seed: Option<u64>,
}

impl Sampling {
    pub const GREEDY: Self = Self {
        temperature: 0.0,
        top_p: 1.0,
        top_k: 0,
        min_p: 0.0,
        seed: None,
    };

    pub fn is_greedy(&self) -> bool {
        self.temperature == 0.0
    }

    /// Rejects values that would produce NaN logits or an empty candidate set.
    pub fn validate(&self) -> Result<(), String> {
        if !self.temperature.is_finite() || self.temperature < 0.0 {
            return Err(format!(
                "temperature must be 0 or more, got {}",
                self.temperature
            ));
        }
        if !(self.top_p > 0.0 && self.top_p <= 1.0) {
            return Err(format!("top_p must be in (0, 1], got {}", self.top_p));
        }
        if !(0.0..=1.0).contains(&self.min_p) {
            return Err(format!("min_p must be in [0, 1], got {}", self.min_p));
        }
        Ok(())
    }
}

impl Default for Sampling {
    fn default() -> Self {
        Self::GREEDY
    }
}

/// Allowed next tokens (constrained decoding). Bit `i` set means token `i` is allowed.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TokenMask {
    bits: Vec<u64>,
    vocab: u32,
}

impl TokenMask {
    pub fn none_allowed(vocab: u32) -> Self {
        Self {
            bits: vec![0; vocab.div_ceil(64) as usize],
            vocab,
        }
    }

    pub fn allow(&mut self, token: u32) {
        if token < self.vocab {
            self.bits[(token / 64) as usize] |= 1 << (token % 64);
        }
    }

    pub fn is_allowed(&self, token: u32) -> bool {
        token < self.vocab && self.bits[(token / 64) as usize] & (1 << (token % 64)) != 0
    }
}

#[derive(Debug, Default)]
struct Flags {
    cancel: AtomicBool,
    yield_: AtomicBool,
}

/// Shared cancellation and yield flags, checked by the engine after every layer and token.
#[derive(Clone, Debug, Default)]
pub struct CancelToken(Arc<Flags>);

impl CancelToken {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn cancel(&self) {
        self.0.cancel.store(true, Ordering::Release);
    }

    pub fn is_cancelled(&self) -> bool {
        self.0.cancel.load(Ordering::Acquire)
    }

    /// Asks `prefill` to return after the current whole chunk (memory pressure).
    pub fn request_yield(&self) {
        self.0.yield_.store(true, Ordering::Release);
    }

    pub fn yield_requested(&self) -> bool {
        self.0.yield_.load(Ordering::Acquire)
    }

    /// Withdraws a yield request once the caller has acted on it.
    pub fn clear_yield(&self) {
        self.0.yield_.store(false, Ordering::Release);
    }
}

/// Prefill chunk length at `depth`: `clamp(2048 × 16384 / max(depth, 1), 128, 2048)`.
pub fn chunk_len(depth: u32) -> u32 {
    (2048 * 16384 / u64::from(depth.max(1))).clamp(128, 2048) as u32
}

/// Checkpoints one `prefill` call may capture.
pub const MAX_PLAN_CHECKPOINTS: usize = 3;

/// Absolute, ascending positions where prefill chunks must end; recurrent state is also
/// checkpointed at each `checkpoints` position. More than [`MAX_PLAN_CHECKPOINTS`] in one call
/// is refused with `InvalidRange`.
#[derive(Clone, Copy, Debug, Default)]
pub struct PrefillPlan<'a> {
    pub boundaries: &'a [u32],
    pub checkpoints: &'a [u32],
}

impl PrefillPlan<'_> {
    /// End of the chunk starting at `start` in a call ending at `end`. Depends only on `start`
    /// and the plan, so equal inputs always produce equal splits.
    pub fn chunk_end(&self, start: u32, end: u32) -> u32 {
        let planned = self
            .boundaries
            .iter()
            .chain(self.checkpoints)
            .copied()
            .filter(|&at| at > start);
        planned
            .chain([end, start.saturating_add(chunk_len(start))])
            .min()
            .unwrap_or(end)
    }

    /// Planned checkpoint positions in `(start, end]`.
    pub fn checkpoints_in(&self, start: u32, end: u32) -> impl Iterator<Item = u32> + '_ {
        self.checkpoints
            .iter()
            .copied()
            .filter(move |&at| at > start && at <= end)
    }
}

/// Result of one `prefill` call.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct Prefilled {
    /// Tokens processed in whole chunks; the rest was not processed.
    pub committed: u32,
    /// Checkpoints captured at planned positions.
    pub checkpoints: Vec<Checkpoint>,
    /// Returned early after a whole chunk because a yield was requested.
    pub yielded: bool,
}

/// Logits of one forward pass, consumed by `sample`.
#[derive(Debug, Eq, PartialEq)]
pub struct Step(u64);

impl Step {
    pub fn new(id: u64) -> Self {
        Self(id)
    }

    pub fn id(&self) -> u64 {
        self.0
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct CheckpointId(pub u64);

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct SegmentId(pub u64);

/// Stored deep copy of all recurrent state at `position`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Checkpoint {
    pub id: CheckpointId,
    pub position: u32,
    pub bytes: u64,
}

/// State for positions `[start, end)` moved out of the live sequence (plus, for hybrids, the
/// recurrent state at `end`).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Segment {
    pub id: SegmentId,
    pub start: u32,
    pub end: u32,
    pub bytes: u64,
}

#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
pub enum EngineError {
    #[error("request needs {requested} tokens of sequence state but only {available} fit")]
    Capacity { requested: u32, available: u32 },
    #[error("{bytes} more bytes of sequence-state memory are needed")]
    NeedsMemory { bytes: u64 },
    #[error("recurrent state at {active} cannot be cut to {len}")]
    NotTrimmable { len: u32, active: u32 },
    #[error("the live sequence has no recurrent state")]
    NoRecurrentState,
    #[error("growth to {needed} tokens exceeds the reservation of {reserved}")]
    ReservationExceeded { needed: u32, reserved: u32 },
    #[error("invalid range {start}..{end} for a sequence of {len} tokens")]
    InvalidRange { start: u32, end: u32, len: u32 },
    #[error("unknown segment {0:?}")]
    UnknownSegment(SegmentId),
    #[error("unknown checkpoint {0:?}")]
    UnknownCheckpoint(CheckpointId),
    #[error("step is not the current one")]
    InvalidStep,
    #[error("engine failure: {0}")]
    Backend(String),
}

/// Primitive operations on the single live sequence plus stored checkpoints and segments.
///
/// Every allocating call checks `budget().state_capacity_bytes − allocated()` first and returns
/// `NeedsMemory` with nothing changed. At most one pipelined step is pending; `trim`, `reserve`,
/// `checkpoint` and `split_off` roll it back first. Engines are created on the worker thread and
/// never move, so no `Send` bound is required.
pub trait Engine {
    fn info(&self) -> &ModelInfo;
    fn budget(&self) -> &MemoryBudget;

    /// Positions held by the live state, including a pending pipelined token.
    fn active_len(&self) -> u32;
    /// Bytes held: live allocation + segments + checkpoints.
    fn allocated(&self) -> u64;

    /// Trims the live state to `keep` (rules of `trim`) and records `total` as the ceiling for
    /// later growth. Allocates nothing. `Capacity` if `total` exceeds the token capacity;
    /// `reserve(0, …)` always succeeds otherwise and zeroes recurrent state.
    fn reserve(&mut self, keep: u32, total: u32) -> Result<(), EngineError>;

    /// Appends `tokens` at `active_len()` in chunks split by [`PrefillPlan::chunk_end`],
    /// checkpointing at planned positions. Memory for the whole call is checked first. A cancel
    /// discards the partial chunk; a yield request returns after the current whole chunk.
    fn prefill(
        &mut self,
        tokens: &[u32],
        plan: &PrefillPlan,
        cancel: &CancelToken,
        progress: &mut dyn FnMut(u32),
    ) -> Result<Prefilled, EngineError>;

    /// Feeds `last` and computes the logits for the next position. A pending step that fed
    /// exactly `last` is returned without recomputation; any other pending step is rolled back.
    fn forward(&mut self, last: u32) -> Result<Step, EngineError>;
    /// Samples from `step`. With `pipeline` and room in the current allocation, the sampled
    /// token is fed at once as a pending step.
    fn sample(
        &mut self,
        step: Step,
        sampling: &Sampling,
        mask: Option<&TokenMask>,
        pipeline: bool,
    ) -> Result<u32, EngineError>;
    /// Top-`n` log-probabilities of `step`.
    fn top_logprobs(&mut self, step: &Step, n: usize) -> Result<Vec<(u32, f32)>, EngineError>;

    /// Cuts the live state to `len`. Recurrent state allows only `len == active_len()`,
    /// `active_len() − 1` with a pending step, or 0 (zeroes it); otherwise `NotTrimmable`.
    fn trim(&mut self, len: u32) -> Result<(), EngineError>;

    /// Deep copy of the recurrent state at `active_len()` (zero bytes for attention-only models).
    fn checkpoint(&mut self) -> Result<Checkpoint, EngineError>;
    /// Copies checkpoint `id` into the live recurrent state and cuts attention to its position
    /// (which must be `<= active_len()`). The checkpoint stays stored.
    fn restore_checkpoint(&mut self, id: CheckpointId) -> Result<(), EngineError>;
    fn drop_checkpoint(&mut self, id: CheckpointId);

    /// Moves positions `[at, active_len())` out into a segment. With `shrink` the live
    /// allocation is reduced to `round_up(at, 256)`; without it (an `append` follows) it is
    /// kept. Hybrids move the live recurrent state into the segment; `prefill` and `forward`
    /// then fail with `NoRecurrentState` until `restore_checkpoint` or `append`.
    fn split_off(&mut self, at: u32, shrink: bool) -> Result<Segment, EngineError>;
    /// Copies segment `id` back in at its start (`active_len()` must equal it). With `consume`
    /// the segment is freed as it is copied; otherwise it stays stored.
    fn append(&mut self, id: SegmentId, consume: bool) -> Result<(), EngineError>;
    fn drop_segment(&mut self, id: SegmentId);

    /// Empties the live state and releases its allocation; segments and checkpoints stay.
    fn reset(&mut self);
    /// `reset` plus dropping every stored segment and checkpoint (recovery after a panic).
    fn reset_all(&mut self);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunk_len_follows_depth() {
        for (depth, len) in [
            (0, 2048),
            (1, 2048),
            (16_384, 2048),
            (32_768, 1024),
            (65_536, 512),
            (131_072, 256),
            (262_144, 128),
            (262_000, 128),
            (u32::MAX, 128),
        ] {
            assert_eq!(chunk_len(depth), len, "depth {depth}");
        }
    }

    #[test]
    fn chunks_end_at_plan_positions() {
        let plan = PrefillPlan {
            boundaries: &[100, 5000],
            checkpoints: &[300],
        };
        assert_eq!(plan.chunk_end(0, 4000), 100);
        assert_eq!(plan.chunk_end(100, 4000), 300);
        assert_eq!(plan.chunk_end(300, 4000), 2348);
        assert_eq!(plan.chunk_end(2348, 4000), 4000);
        assert_eq!(plan.checkpoints_in(0, 299).count(), 0);
        assert_eq!(plan.checkpoints_in(0, 300).collect::<Vec<_>>(), [300]);
    }

    #[test]
    fn only_linear_layers_make_a_model_hybrid() {
        let mut info = ModelInfo {
            identity: "test".into(),
            family: "gpt_oss".into(),
            layer_kinds: vec![
                LayerKind::SlidingAttention { window: 128 },
                LayerKind::FullAttention,
            ],
            kv_heads: 8,
            head_dim: 64,
            vocab: 1,
            max_position: 1,
            kv_bytes_per_token: 0,
            recurrent_bytes: 0,
            eos_tokens: Vec::new(),
        };
        assert_eq!(info.layers(), 2);
        assert!(!info.is_hybrid());
        info.layer_kinds.push(LayerKind::Linear);
        assert!(info.is_hybrid());
    }

    #[test]
    fn token_mask_allows_only_set_tokens() {
        let mut mask = TokenMask::none_allowed(130);
        mask.allow(0);
        mask.allow(129);
        mask.allow(500);
        assert!(mask.is_allowed(0));
        assert!(mask.is_allowed(129));
        assert!(!mask.is_allowed(1));
        assert!(!mask.is_allowed(500));
    }

    #[test]
    fn cancel_and_yield_are_shared() {
        let token = CancelToken::new();
        let clone = token.clone();
        assert!(!token.is_cancelled() && !token.yield_requested());
        clone.request_yield();
        assert!(token.yield_requested());
        token.clear_yield();
        assert!(!clone.yield_requested());
        clone.cancel();
        assert!(token.is_cancelled());
    }

    #[test]
    fn invalid_sampling_is_rejected() {
        for bad in [
            Sampling {
                temperature: f32::NAN,
                ..Sampling::GREEDY
            },
            Sampling {
                temperature: -1.0,
                ..Sampling::GREEDY
            },
            Sampling {
                top_p: 0.0,
                ..Sampling::GREEDY
            },
            Sampling {
                min_p: 1.5,
                ..Sampling::GREEDY
            },
        ] {
            assert!(bad.validate().is_err(), "{bad:?}");
        }
        assert!(Sampling::GREEDY.validate().is_ok());
    }

    #[test]
    fn greedy_is_temperature_zero() {
        assert!(Sampling::GREEDY.is_greedy());
        assert!(
            !Sampling {
                temperature: 0.7,
                ..Sampling::GREEDY
            }
            .is_greedy()
        );
    }
}
