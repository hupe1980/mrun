//! Embeds MLX's GPU kernels (D034): with the `mlx` feature, `mrun-mlx-sys` exports the path of
//! its zstd-compressed `mlx.metallib` as `DEP_MLXC_METALLIB_ZST`. This script hands that path to
//! `src/mlx/metallib.rs` (which `include_bytes!` it) together with a short content hash for the
//! extracted file's name, and sets `cfg(mrun_metallib)`; `cfg(mlx_nax)` when MLX was built with its
//! NAX kernels (`DEP_MLXC_NAX`). CPU-only and Linux builds export no
//! metallib, so nothing is embedded; without the `mlx` feature this script does nothing.

use std::{env, fs, path::PathBuf};

use sha2::{Digest, Sha256};

fn main() {
    println!("cargo::rerun-if-changed=build.rs");
    println!("cargo::rustc-check-cfg=cfg(mrun_metallib)");
    println!("cargo::rustc-check-cfg=cfg(mlx_nax)");
    if env::var_os("CARGO_FEATURE_MLX").is_none() {
        return;
    }
    // MLX compiled its NAX kernels (SDK and deployment target >= 26.2, D044): part of the
    // numerics class (FR-027).
    println!("cargo::rerun-if-env-changed=DEP_MLXC_NAX");
    if env::var("DEP_MLXC_NAX").is_ok_and(|v| v == "1") {
        println!("cargo::rustc-cfg=mlx_nax");
    }
    println!("cargo::rerun-if-env-changed=DEP_MLXC_METALLIB_ZST");
    let Some(path) = env::var_os("DEP_MLXC_METALLIB_ZST").map(PathBuf::from) else {
        return;
    };
    println!("cargo::rerun-if-changed={}", path.display());
    let bytes = fs::read(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    let hash: String = Sha256::digest(&bytes)
        .iter()
        .take(6)
        .map(|b| format!("{b:02x}"))
        .collect();
    println!("cargo::rustc-env=BUILD_METALLIB_ZST={}", path.display());
    println!("cargo::rustc-env=BUILD_METALLIB_HASH={hash}");
    println!("cargo::rustc-cfg=mrun_metallib");
}
