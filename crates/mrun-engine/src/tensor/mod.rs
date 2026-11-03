//! The tensor backend: arrays, ops, `fast::*`, evaluation, memory and device queries, and
//! safetensors I/O, as one surface (the tensor boundary of spec 016).
//!
//! The engine and the model families call only `crate::tensor`; the backend behind it is chosen at
//! build time. Today that is MLX through `mrun-mlx-sys`; spec 016 (D065) adds nuvec and then
//! removes MLX.

#[cfg(all(feature = "mlx", feature = "nuvec"))]
compile_error!("enable one tensor backend: `mlx` or `nuvec`, not both");
#[cfg(not(any(feature = "mlx", feature = "nuvec")))]
compile_error!("the `engine` feature needs a backend: enable `mlx` or `nuvec`");

#[cfg(feature = "mlx")]
mod mlx;
#[cfg(feature = "mlx")]
pub use mlx::*;
#[cfg(feature = "nuvec")]
mod nuvec;
#[cfg(feature = "nuvec")]
pub use self::nuvec::*;
