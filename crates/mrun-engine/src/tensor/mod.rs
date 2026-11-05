//! The tensor backend: arrays, ops, `fast::*`, evaluation, memory and device queries, and
//! safetensors I/O, as one surface over nuvec (D065, 022).
//!
//! The engine and the model families call only `crate::tensor`, never nuvec directly.

mod nuvec;
pub use self::nuvec::*;
