//! Engine side of mrun: the primitive [`Engine`] contract and model identity.
//!
//! The MLX implementation (feature `mlx`) lands with tasks T004–T015 of feature 001.

pub mod engine;
pub mod identity;

pub use engine::{
    Architecture, CancelToken, Engine, EngineError, KvSegmentId, MemoryBudget, ModelInfo, Sampling,
    Segment, TokenMask,
};
