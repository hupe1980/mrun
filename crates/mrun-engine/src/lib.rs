//! Engine side of mrun: the primitive [`Engine`] contract and model identity.
//!
//! The MLX implementation will live behind the `mlx` cargo feature.

pub mod budget;
pub mod engine;
pub mod identity;
pub mod models;

pub use engine::{
    CancelToken, Checkpoint, CheckpointId, Engine, EngineError, LayerKind, MAX_PLAN_CHECKPOINTS,
    MemoryBudget, ModelInfo, PrefillPlan, Prefilled, Sampling, Segment, SegmentId, Step, TokenMask,
    chunk_len,
};
