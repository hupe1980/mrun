//! Engine side of mrun: the primitive [`Engine`] contract, model identity, the memory budget and
//! the family registry.
//!
//! The nuvec engine and its tensor layer (`tensor`) are behind the `engine` feature.

pub mod budget;
pub mod calibration;
pub mod engine;
pub mod identity;
pub mod ledger;
#[cfg(feature = "engine")]
pub mod load;
pub mod models;
#[cfg(feature = "engine")]
pub mod nuvec_engine;
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
    ALIGN, BACKEND, Binding, CHUNK_ALIGN, CancelToken, Checkpoint, CheckpointId, Engine,
    EngineError, GROWTH, LAYER_TIME_BOUND_MS, LayerKind, MemoryBudget, ModelInfo, Numerics,
    Sampling, Segment, SegmentId, Step, TimeLen, TokenMask, align, chunk_len, grown_capacity,
};
#[cfg(feature = "engine")]
pub use models::Part;
pub use models::SyntaxId;
#[cfg(feature = "engine")]
pub use nuvec_engine::{FailPoint, LoadOptions, NuvecEngine, NuvecLoadError};
#[cfg(feature = "engine")]
pub use tensor::Pipe;
