//! The engine contract: primitives the runtime drives. No MLX types appear here.

use std::fmt;
use std::ops::Range;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

/// Model families the engine can run.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Architecture {
    Qwen3,
    Qwen3Moe,
}

/// What the runtime needs to know about the loaded model.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ModelInfo {
    /// Content identity (see [`crate::identity`]).
    pub identity: String,
    pub architecture: Architecture,
    pub layers: u32,
    pub kv_heads: u32,
    pub head_dim: u32,
    pub vocab: u32,
    pub max_position: u32,
    /// Bytes of K and V for one token across all layers.
    pub kv_bytes_per_token: u64,
    pub eos_tokens: Vec<u32>,
}

/// Memory budget computed at startup: what mrun may use and how it is split.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct MemoryBudget {
    pub physical: u64,
    pub recommended_working_set: u64,
    pub reserve: u64,
    pub budget: u64,
    pub weights: u64,
    pub scratch: u64,
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

    pub fn vocab(&self) -> u32 {
        self.vocab
    }
}

/// Shared cancellation flag, checked by the engine between layers and tokens.
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

/// Handle to an owned deep copy of a token range of the active cache.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct KvSegmentId(pub u64);

/// A snapshot together with its size.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Segment {
    pub id: KvSegmentId,
    pub bytes: u64,
}

#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
pub enum EngineError {
    #[error("request needs {requested} tokens of attention state but only {available} fit")]
    Capacity { requested: u32, available: u32 },
    #[error("unknown KV segment {0:?}")]
    UnknownSegment(KvSegmentId),
    #[error("invalid range {start}..{end} for an active cache of {len} tokens")]
    InvalidRange { start: u32, end: u32, len: u32 },
    #[error("engine failure: {0}")]
    Backend(String),
}

/// Primitive operations on the single active sequence plus owned snapshots.
///
/// Engines are created on the worker thread and never move, so no `Send` bound is required.
pub trait Engine {
    fn info(&self) -> &ModelInfo;
    fn budget(&self) -> &MemoryBudget;
    /// Tokens currently held in the active cache.
    fn active_len(&self) -> u32;
    /// Ensure the active cache can grow to `total_tokens`. Evicting cached segments is the
    /// caller's job; this fails only when even an empty cache cannot hold `total_tokens`.
    fn reserve(&mut self, total_tokens: u32) -> Result<(), EngineError>;
    /// Append `tokens` to the active cache. Returns how many were processed, fewer than
    /// `tokens.len()` only when cancelled.
    fn prefill(
        &mut self,
        tokens: &[u32],
        cancel: &CancelToken,
        progress: &mut dyn FnMut(u32),
    ) -> Result<u32, EngineError>;
    /// Feed `last` and return the next sampled token.
    fn decode(
        &mut self,
        last: u32,
        sampling: &Sampling,
        mask: Option<&TokenMask>,
    ) -> Result<u32, EngineError>;
    /// Shrink the active cache to `len` tokens (no-op if already shorter).
    fn trim(&mut self, len: u32);
    /// Owned deep copy of active tokens `range`.
    fn snapshot(&mut self, range: Range<u32>) -> Result<Segment, EngineError>;
    /// Replace the active cache with `segments` (contiguous from position 0), then trim to `keep`.
    fn restore(&mut self, segments: &[KvSegmentId], keep: u32) -> Result<(), EngineError>;
    fn drop_segment(&mut self, id: KvSegmentId);
    /// Empty the active cache.
    fn reset(&mut self);
}

impl fmt::Display for Architecture {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Qwen3 => "qwen3",
            Self::Qwen3Moe => "qwen3_moe",
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn cancel_token_is_shared() {
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
