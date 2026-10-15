//! `mrun --version`: version, commit, build date, vendored MLX versions and platform, embedded at
//! build time; printing it opens no socket (008 FR-004, FR-011).

/// `X.Y.Z (<commit7>, <date>) mlx <v> mlx-c <v> <os>-<arch>`.
pub const LONG: &str = env!("BUILD_MRUN_VERSION_LONG");
pub const MLX: &str = env!("BUILD_MLX_VERSION");
pub const MLX_C: &str = env!("BUILD_MLX_C_VERSION");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn long_version_names_everything() {
        assert!(LONG.starts_with(env!("CARGO_PKG_VERSION")));
        assert!(LONG.contains(&format!("mlx {MLX} mlx-c {MLX_C}")));
        let date = LONG.split(", ").nth(1).unwrap().split(')').next().unwrap();
        assert_eq!(date.len(), 10, "{LONG}");
    }

    #[test]
    fn mlx_pins_match_mlx_sys() {
        let build = include_str!("../../../mlx-sys/build.rs");
        assert!(build.contains(&format!("const MLX_VERSION: &str = \"{MLX}\";")));
        assert!(build.contains(&format!("mlx-c v{MLX_C}")) || build.contains(&format!("v{MLX_C}")));
    }
}
