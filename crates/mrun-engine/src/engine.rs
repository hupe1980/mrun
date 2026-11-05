//! The engine contract: primitives the runtime drives. No tensor-backend types appear here.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

/// Kind of a decoder layer. Attention layers hold positional K/V that can be cut at any
/// position; linear layers hold position-bound recurrent state.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LayerKind {
    FullAttention,
    Linear,
}

/// The tensor backend's name (nuvec, the pure-Rust MLX port), as `/mrun/status`
/// (`engine.backend`, `numerics.backend`) and bench records report it.
pub const BACKEND: &str = "nuvec";

/// Numerics class of the GPU and kernels (FR-027, D044): bits can differ between classes, so
/// exactness comparisons, parity references, persisted state (007) and benchmark results carry it
/// beside the model identity.
#[derive(Clone, Debug, Default, Eq, Hash, PartialEq)]
pub struct Numerics {
    /// nuvec runs its NAX (M5 Neural Accelerator) kernels, which change the bits.
    pub nax: bool,
    /// GPU architecture from the device info (e.g. `applegpu_g16s`), `cpu` without Metal.
    pub gpu_arch: String,
}

/// How the GPU kernels are provided (016 FR-009).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Kernels {
    /// A Metal library compiled at build time and embedded in the binary.
    Precompiled,
    /// Shader sources embedded and compiled when the process starts.
    Runtime,
    /// No Metal support in this build.
    Unavailable,
}

impl Kernels {
    /// `precompiled`, `runtime` or `unavailable` (status, doctor, `--version`).
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Precompiled => "precompiled",
            Self::Runtime => "runtime",
            Self::Unavailable => "unavailable",
        }
    }
}

/// The tensor backend of a build (016 FR-008): what `--version`, `doctor`, `/mrun/status` and
/// bench records report besides [`BACKEND`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BackendInfo {
    /// nuvec's crate version.
    pub version: &'static str,
    pub kernels: Kernels,
}

impl Numerics {
    /// The class name: `nuvec`, or `nuvec-nax` with the NAX kernels.
    pub fn class(&self) -> &'static str {
        if self.nax { "nuvec-nax" } else { "nuvec" }
    }

    /// The fixed class `ScriptedEngine` reports.
    pub fn scripted() -> Self {
        Self {
            nax: false,
            gpu_arch: "scripted".into(),
        }
    }

    /// Compact key, e.g. `applegpu_g16s/nuvec` (calibration and reference files).
    pub fn key(&self) -> String {
        format!("{}/{}", self.gpu_arch, self.class())
    }
}

impl std::fmt::Display for Numerics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}, {}", self.gpu_arch, self.class())
    }
}

/// The calibrated time bound of [`ModelInfo::chunk_len`] (FR-007, D050): at each measured depth,
/// a chunk length whose layer took at most [`LAYER_TIME_BOUND_MS`]: the memory-bound
/// [`chunk_len`] itself where that was fast enough, else the longest multiple of
/// [`CHUNK_ALIGN`] that was. Fixed for the process (and persisted per device, model and
/// numerics class), so the chunk plan stays a pure function of the prompt. Empty means
/// unbounded (no calibration, e.g. the CPU).
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct TimeLen {
    /// `(depth, len)`, depths ascending, lengths non-increasing and at least 128.
    points: Vec<(u32, u32)>,
}

/// Chunk lengths are multiples of this (or the memory bound itself).
pub const CHUNK_ALIGN: u32 = 128;
/// One layer of a prefill chunk may take at most this long (FR-007, D059): a cancel waits for
/// about one layer in flight (measured p95 93–150 ms with layers up to 235 ms), so SC-004's
/// 300 ms gate is the real check.
pub const LAYER_TIME_BOUND_MS: u32 = 250;

impl TimeLen {
    /// No time bound.
    pub fn unbounded() -> Self {
        Self::default()
    }

    /// From measured `(depth, len)` points: sorted by depth, at least [`CHUNK_ALIGN`], and made
    /// non-increasing with depth (a length that is fast enough at some depth is fast enough at
    /// every shallower one).
    pub fn from_points(mut points: Vec<(u32, u32)>) -> Self {
        points.sort_unstable();
        points.dedup_by_key(|p| p.0);
        let mut bound = u32::MAX;
        for (_, len) in &mut points {
            *len = (*len).max(CHUNK_ALIGN).min(bound);
            bound = *len;
        }
        Self { points }
    }

    pub fn points(&self) -> &[(u32, u32)] {
        &self.points
    }

    pub fn is_unbounded(&self) -> bool {
        self.points.is_empty()
    }

    /// The bound for a chunk starting at `depth`. At a measured depth its length; between two,
    /// `1 / len` is interpolated linearly (exact when a layer's time is affine in `len` and
    /// `len × depth`: weights plus attention), rounded down to a multiple of [`CHUNK_ALIGN`],
    /// and `u32::MAX` where that reaches the memory bound (which is itself `1 / len`-linear);
    /// beyond the last point its length. `u32::MAX` when unbounded.
    pub fn at(&self, depth: u32) -> u32 {
        let Some(&(last_depth, last_len)) = self.points.last() else {
            return u32::MAX;
        };
        if depth >= last_depth {
            return last_len;
        }
        let next = self.points.partition_point(|(d, _)| *d <= depth);
        if next == 0 {
            return self.points[0].1;
        }
        let ((d0, l0), (d1, l1)) = (self.points[next - 1], self.points[next]);
        if depth == d0 {
            return l0;
        }
        let t = f64::from(depth - d0) / f64::from(d1 - d0);
        let len = 1.0 / ((1.0 - t) / f64::from(l0) + t / f64::from(l1));
        if len + 0.5 >= f64::from(chunk_len(depth)) {
            return u32::MAX;
        }
        let align = f64::from(CHUNK_ALIGN);
        // A hair of tolerance so a depth just past a point keeps that point's length.
        let blocks = (len / align + 0.01).floor();
        ((blocks * align) as u32).clamp(l1, l0)
    }
}

/// What the runtime needs to know about the loaded model.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
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
    /// Numerics class the engine runs with (FR-027).
    pub numerics: Numerics,
    /// Calibrated time bound on chunk lengths (FR-007); unbounded until calibrated.
    pub time_len: TimeLen,
}

impl ModelInfo {
    /// Prefill chunk length at `depth` for this model on this machine (FR-007):
    /// `max(128, min(chunk_len(depth), time_len(depth)))`.
    pub fn chunk_len(&self, depth: u32) -> u32 {
        chunk_len(depth)
            .min(self.time_len.at(depth))
            .max(CHUNK_ALIGN)
    }

    pub fn layers(&self) -> u32 {
        self.layer_kinds.len() as u32
    }

    /// Whether the model has recurrent state, which cannot be cut to an earlier position.
    pub fn is_hybrid(&self) -> bool {
        self.layer_kinds.contains(&LayerKind::Linear)
    }
}

/// The term of the budget formula that binds (D062).
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum Binding {
    /// Metal's recommended working set.
    #[default]
    WorkingSet,
    /// `physical − 2 GiB`.
    Physical,
    /// Available memory minus the safety margin: other processes hold the rest.
    Available,
    /// `--budget-gib` lowered it.
    Flag,
}

impl Binding {
    pub fn label(self) -> &'static str {
        match self {
            Self::WorkingSet => "working set",
            Self::Physical => "physical",
            Self::Available => "available memory",
            Self::Flag => "--budget-gib",
        }
    }
}

impl std::fmt::Display for Binding {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.label())
    }
}

/// Memory budget computed at startup: what mrun may use and how it is split.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct MemoryBudget {
    pub physical: u64,
    pub recommended_working_set: u64,
    /// What macOS could hand out without swapping at startup (free + inactive + purgeable +
    /// speculative pages, D062); `None` where it is not read (not macOS).
    pub available: Option<u64>,
    /// `min(recommended working set, physical − 2 GiB, available − margin)`, lowered by the
    /// override (D062).
    pub budget: u64,
    /// Which term of the formula `budget` is.
    pub binding: Binding,
    pub weights: u64,
    pub scratch: u64,
    /// The depth scratch's position-dependent terms are taken at: the capacity-limited depth
    /// (FR-017), the fixed point of capacity and scratch.
    pub scratch_depth: u32,
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
    /// mlx-lm's repetition penalty over the last [`crate::sample::REPETITION_CONTEXT`] token ids
    /// (prompt and generated); 1 is off.
    pub repetition_penalty: f32,
}

impl Sampling {
    pub const GREEDY: Self = Self {
        temperature: 0.0,
        top_p: 1.0,
        top_k: 0,
        min_p: 0.0,
        seed: None,
        repetition_penalty: 1.0,
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
        if !(self.repetition_penalty.is_finite() && self.repetition_penalty > 0.0) {
            return Err(format!(
                "repetition_penalty must be more than 0, got {}",
                self.repetition_penalty
            ));
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

    /// Mask from a bitset in 32-bit words (bit `i % 32` of word `i / 32` allows token `i`), the
    /// layout grammar engines produce. Missing words allow nothing; bits at or above `vocab` are
    /// ignored.
    pub fn from_words(words: &[u32], vocab: u32) -> Self {
        let mut mask = Self::none_allowed(vocab);
        for (i, &word) in words.iter().enumerate().take(vocab.div_ceil(32) as usize) {
            mask.bits[i / 2] |= u64::from(word) << (32 * (i % 2));
        }
        if !vocab.is_multiple_of(64)
            && let Some(last) = mask.bits.last_mut()
        {
            *last &= (1 << (vocab % 64)) - 1;
        }
        mask
    }

    pub fn allow(&mut self, token: u32) {
        if token < self.vocab {
            self.bits[(token / 64) as usize] |= 1 << (token % 64);
        }
    }

    pub fn is_allowed(&self, token: u32) -> bool {
        token < self.vocab && self.bits[(token / 64) as usize] & (1 << (token % 64)) != 0
    }

    /// Whether no token is allowed.
    pub fn is_none(&self) -> bool {
        self.bits.iter().all(|&w| w == 0)
    }

    /// Tokens the mask covers (a tokenizer's vocabulary, which may be narrower than the
    /// model's logits).
    pub fn vocab(&self) -> u32 {
        self.vocab
    }

    /// One flag per logit of a `width`-wide output, expanded a word at a time; positions at or
    /// beyond the mask's vocabulary (padding rows of the embedding) are not allowed. `None` when
    /// the mask is wider than `width`.
    pub fn to_flags(&self, width: u32) -> Option<Vec<bool>> {
        if self.vocab > width {
            return None;
        }
        let mut flags = Vec::with_capacity(width as usize);
        for &word in &self.bits {
            let take = (self.vocab as usize - flags.len()).min(64);
            flags.extend((0..take).map(|bit| word >> bit & 1 == 1));
        }
        flags.resize(width as usize, false);
        Some(flags)
    }

    /// `MaskTooWide` when the mask covers more tokens than the model has logits.
    pub fn check_width(&self, width: u32) -> Result<(), EngineError> {
        if self.vocab > width {
            return Err(EngineError::MaskTooWide {
                mask: self.vocab,
                vocab: width,
            });
        }
        Ok(())
    }
}

/// Shared cancellation flag, checked by the engine after every layer and between tokens.
#[derive(Clone, Debug, Default)]
pub struct CancelToken(Arc<AtomicBool>);

impl CancelToken {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn cancel(&self) {
        self.0.store(true, Ordering::Release);
    }

    pub fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::Acquire)
    }
}

/// Prefill chunk length at `depth`: `clamp(2048 × 16384 / max(depth, 1), 128, 2048)`. Bounds the
/// attention work of one layer (nuvec, as MLX, falls back to unfused attention for some head
/// sizes, whose scores grow with chunk × depth) and so the cancellation latency.
pub fn chunk_len(depth: u32) -> u32 {
    (2048 * 16384 / u64::from(depth.max(1))).clamp(128, 2048) as u32
}

/// Growth step of the attention state, in tokens.
pub const GROWTH: u32 = 4096;
/// Attention state capacity is a multiple of this many positions.
pub const ALIGN: u32 = 256;

/// `n` rounded up to [`ALIGN`].
pub fn align(n: u32) -> u32 {
    n.div_ceil(ALIGN).saturating_mul(ALIGN)
}

/// Capacity after growing to hold `needed` positions: `align(max(needed + GROWTH, 2 ×
/// capacity))`, clamped to `limit`. Geometric, so a prefill that grows one chunk at a time copies
/// O(n) positions in total. `Ok(None)` when `capacity` already suffices; `ReservationExceeded`
/// beyond `limit`. Shared by every engine so they allocate alike.
pub fn grown_capacity(capacity: u32, needed: u32, limit: u32) -> Result<Option<u32>, EngineError> {
    if needed <= capacity {
        return Ok(None);
    }
    if needed > limit {
        return Err(EngineError::ReservationExceeded {
            needed,
            reserved: limit,
        });
    }
    let target = needed
        .saturating_add(GROWTH)
        .max(capacity.saturating_mul(2));
    Ok(Some(align(target).min(limit).max(needed)))
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

/// Stored copy of all position-bound state at `position` (zero bytes for attention-only models,
/// whose positional state is exact at any length).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Checkpoint {
    pub id: CheckpointId,
    pub position: u32,
    pub bytes: u64,
}

/// State for positions `[start, end)` detached from the live sequence (plus, for models with
/// position-bound state, that state at `end`).
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
    #[error("position-bound state at {active} cannot be cut to {len}")]
    NotTrimmable { len: u32, active: u32 },
    #[error("the live sequence has no position-bound state")]
    NoRecurrentState,
    #[error("growth to {needed} tokens exceeds the limit of {reserved}")]
    ReservationExceeded { needed: u32, reserved: u32 },
    #[error("invalid range {start}..{end} for a sequence of {len} tokens")]
    InvalidRange { start: u32, end: u32, len: u32 },
    #[error("the token mask allows no token")]
    EmptyMask,
    #[error("a token mask over {mask} tokens is wider than the model's {vocab} logits")]
    MaskTooWide { mask: u32, vocab: u32 },
    #[error("unknown segment {0:?}")]
    UnknownSegment(SegmentId),
    #[error("unknown checkpoint {0:?}")]
    UnknownCheckpoint(CheckpointId),
    #[error("step is not the current one")]
    InvalidStep,
    #[error("engine failure: {0}")]
    Backend(String),
    /// A move failed half-way: the live state and the segment in flight are unknown. Every
    /// call but `drop_*`, `reset` and `reset_all` fails with this until `reset_all`.
    #[error("engine state lost in a failed move ({0}); reset_all required")]
    Poisoned(String),
}

/// Primitive operations on the single live sequence plus stored checkpoints and segments
/// (contract v2, D028). The runtime owns the chunk loop, the chunk plan, checkpoint placement,
/// memory-pressure handling and `NeedsMemory` retries.
///
/// Every allocating call checks `budget().state_capacity_bytes − allocated()` first and returns
/// `NeedsMemory` (like every other error before the state is touched) with nothing changed. At
/// most one pipelined step is pending: `sample` may feed the sampled token at once; `forward`
/// with that token reuses it, `trim` to `len()` confirms it, and `prefill_chunk`, any other
/// `forward` or `trim`, `checkpoint`, `restore`, `detach_tail` and `attach` roll it back
/// (`set_limit` leaves it alone). The bookkeeping is shared by every engine
/// ([`crate::ledger`]). Engines are created on the worker thread and never move, so no `Send`
/// bound is required.
pub trait Engine {
    fn info(&self) -> &ModelInfo;
    fn budget(&self) -> &MemoryBudget;

    /// Positions held by the live state, including a pending pipelined token.
    fn len(&self) -> u32;
    /// The live state holds no positions.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// Bytes held: live allocation + segments + checkpoints.
    fn allocated(&self) -> u64;

    /// Sets the ceiling for growth until the next call (the request's prompt + output limit).
    /// `Capacity` if it exceeds the token capacity. Allocates nothing and keeps a pending step.
    fn set_limit(&mut self, total: u32) -> Result<(), EngineError>;

    /// Appends one chunk at `len()` (after rolling back a pending step), at most two layers in
    /// flight, checking `cancel` after every layer. `Ok(true)`: committed; `Ok(false)`:
    /// cancelled, and like an error the state (allocation included) is as before the call.
    fn prefill_chunk(&mut self, tokens: &[u32], cancel: &CancelToken) -> Result<bool, EngineError>;

    /// Feeds `last` and computes the logits for the next position.
    fn forward(&mut self, last: u32) -> Result<Step, EngineError>;
    /// Samples from `step`; `EmptyMask` when `mask` allows nothing, `MaskTooWide` when it covers
    /// more tokens than `info().vocab` (a narrower mask leaves the rest disallowed). Feeds the
    /// sampled token as a pending step when the allocation has room.
    fn sample(
        &mut self,
        step: Step,
        sampling: &Sampling,
        mask: Option<&TokenMask>,
    ) -> Result<u32, EngineError>;
    /// Top-`n` log-probabilities of `step`, most likely first.
    fn top_logprobs(&mut self, step: &Step, n: usize) -> Result<Vec<(u32, f32)>, EngineError>;

    /// Cuts the live state to `len`. Position-bound state allows only `len == len()` (after a
    /// pending step: `len() − 1`) or 0 (zeroes it); otherwise `NotTrimmable`. Capacity more than
    /// [`GROWTH`] beyond the new length is released.
    fn trim(&mut self, len: u32) -> Result<(), EngineError>;

    /// Copy of the position-bound state at `len()` (zero bytes for attention-only models).
    fn checkpoint(&mut self) -> Result<Checkpoint, EngineError>;
    /// Copies checkpoint `id` into the live position-bound state and cuts positional state to its
    /// position (which must be `<= len()`). The checkpoint stays stored.
    fn restore(&mut self, id: CheckpointId) -> Result<(), EngineError>;
    fn drop_checkpoint(&mut self, id: CheckpointId);

    /// Moves positions `[at, len())` into a segment, one layer at a time (copy the layer's tail,
    /// shrink the live layer to `align(at)`, release, next layer), so the transient is one layer.
    /// Models with position-bound state move it into the segment; prefill and forward then fail
    /// with `NoRecurrentState` until `restore` or `attach`.
    fn detach_tail(&mut self, at: u32) -> Result<Segment, EngineError>;
    /// Moves segment `id` back in at its start (the length before a pending step must equal it),
    /// one layer at a time, freeing each segment layer as it is copied, into a capacity of
    /// `align(end)` (within the limit when the segment fits it). The segment is consumed.
    fn attach(&mut self, id: SegmentId) -> Result<(), EngineError>;
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
    fn time_len_bounds_chunks_by_depth() {
        let info = ModelInfo::default();
        // Uncalibrated: the memory bound alone.
        assert_eq!(info.chunk_len(0), 2048);
        assert_eq!(info.chunk_len(65_536), 512);
        let info = ModelInfo {
            time_len: TimeLen::from_points(vec![(16_384, 896), (0, 1536), (65_536, 2048)]),
            ..ModelInfo::default()
        };
        // Sorted and non-increasing with depth.
        assert_eq!(
            info.time_len.points(),
            [(0, 1536), (16_384, 896), (65_536, 896)]
        );
        assert_eq!(info.chunk_len(0), 1536);
        assert_eq!(info.chunk_len(1), 1536);
        // 1 / len interpolated: 1 / (0.5 / 1536 + 0.5 / 896) = 1131.8 → 1024.
        assert_eq!(info.chunk_len(8192), 1024);
        assert_eq!(info.chunk_len(16_383), 896);
        assert_eq!(info.chunk_len(16_384), 896);
        assert_eq!(info.chunk_len(32_768), 896);
        assert_eq!(info.chunk_len(65_536), 512, "the memory bound is lower");
        assert_eq!(info.chunk_len(200_000), 167, "2^25 / depth");
        // Never below 128.
        let slow = ModelInfo {
            time_len: TimeLen::from_points(vec![(0, 5)]),
            ..ModelInfo::default()
        };
        assert_eq!(slow.chunk_len(0), 128);
        assert_eq!(slow.chunk_len(1_000_000), 128);
    }

    #[test]
    fn numerics_class_prints_and_keys() {
        let n = Numerics {
            nax: false,
            gpu_arch: "applegpu_g16s".into(),
        };
        assert_eq!(n.to_string(), "applegpu_g16s, nuvec");
        assert_eq!(n.key(), "applegpu_g16s/nuvec");
        let nax = Numerics {
            nax: true,
            gpu_arch: "applegpu_g17s".into(),
        };
        assert_eq!(nax.key(), "applegpu_g17s/nuvec-nax");
        assert_eq!(Numerics::scripted().gpu_arch, "scripted");
    }

    #[test]
    fn growth_is_geometric() {
        // align(max(needed + 4096, 2 × capacity)), clamped to the limit.
        assert_eq!(grown_capacity(0, 100, 100_000), Ok(Some(4352)));
        assert_eq!(grown_capacity(4352, 4353, 100_000), Ok(Some(8704)));
        assert_eq!(grown_capacity(8704, 20_000, 100_000), Ok(Some(24_320)));
        assert_eq!(grown_capacity(4352, 4000, 100_000), Ok(None));
        assert_eq!(grown_capacity(0, 100, 300), Ok(Some(300)));
        assert_eq!(
            grown_capacity(0, 400, 300),
            Err(EngineError::ReservationExceeded {
                needed: 400,
                reserved: 300
            })
        );
    }

    #[test]
    fn only_linear_layers_make_a_model_hybrid() {
        let mut info = ModelInfo {
            identity: "test".into(),
            family: "qwen3".into(),
            layer_kinds: vec![LayerKind::FullAttention; 2],
            kv_heads: 8,
            head_dim: 64,
            vocab: 1,
            max_position: 1,
            kv_bytes_per_token: 0,
            recurrent_bytes: 0,
            eos_tokens: Vec::new(),
            ..ModelInfo::default()
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
    fn token_mask_from_words_reads_a_32_bit_bitset() {
        // Tokens 0, 33, 64 and 70 set; 70 lies beyond the vocabulary and is dropped.
        let mask = TokenMask::from_words(&[1, 1 << 1, 1 | (1 << 6)], 70);
        let allowed: Vec<u32> = (0..96).filter(|&t| mask.is_allowed(t)).collect();
        assert_eq!(allowed, [0, 33, 64]);
        let mut expected = TokenMask::none_allowed(70);
        for token in [0, 33, 64] {
            expected.allow(token);
        }
        assert_eq!(mask, expected);
        // Missing words allow nothing.
        assert_eq!(TokenMask::from_words(&[], 70), TokenMask::none_allowed(70));
    }

    #[test]
    fn cancel_is_shared() {
        let token = CancelToken::new();
        let clone = token.clone();
        assert!(!token.is_cancelled());
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
            Sampling {
                repetition_penalty: 0.0,
                ..Sampling::GREEDY
            },
            Sampling {
                repetition_penalty: f32::INFINITY,
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
