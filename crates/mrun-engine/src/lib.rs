//! Engine side of mrun: the primitive [`Engine`] contract, model identity, the memory budget and
//! the family registry.
//!
//! The tensor backend (`tensor`: `mlx` or `nuvec` feature, spec 016) is behind `engine`.

pub mod budget;
pub mod calibration;
pub mod engine;
pub mod identity;
pub mod ledger;
#[cfg(feature = "engine")]
pub mod load;
#[cfg(feature = "engine")]
pub mod mlx_engine;
pub mod models;
#[cfg(feature = "engine")]
pub mod sample;
#[cfg(feature = "engine")]
pub mod state;
pub mod sysmem;
#[cfg(feature = "engine")]
pub mod tensor;
#[cfg(all(feature = "engine", feature = "test-support"))]
pub mod testing;

pub use engine::{
    ALIGN, Backend, Binding, CHUNK_ALIGN, CancelToken, Checkpoint, CheckpointId, Engine,
    EngineError, GROWTH, LAYER_TIME_BOUND_MS, LayerKind, MemoryBudget, ModelInfo, Numerics,
    Sampling, Segment, SegmentId, Step, TimeLen, TokenMask, align, chunk_len, grown_capacity,
};
#[cfg(feature = "engine")]
pub use mlx_engine::{FailPoint, LoadOptions, MlxEngine, MlxLoadError};
pub use models::SyntaxId;
