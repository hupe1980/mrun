//! Engine side of mrun: the primitive [`Engine`] contract and model identity.
//!
//! The MLX implementation will live behind the `mlx` cargo feature.

pub mod engine;
pub mod identity;

pub use engine::{
    Architecture, CancelToken, Engine, EngineError, KvSegmentId, MemoryBudget, ModelInfo, Sampling,
    Segment, TokenMask,
};
