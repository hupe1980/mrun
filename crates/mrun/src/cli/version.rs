//! `mrun --version`: version, commit, build date, the tensor backend's version and platform,
//! embedded at build time; printing it opens no socket (008 FR-004, FR-011; 016 FR-008).

use mrun_engine::engine::BackendInfo;

/// `X.Y.Z (<commit7>, <date>) <backend> <os>-<arch> numerics <nax compiled|no nax|nuvec|cpu>`,
/// with `<backend>` `mlx <v> mlx-c <v>` or `nuvec <v>`: the build half of the numerics class
/// (FR-027, D044; 008 T030; 016 FR-008): nuvec, or whether MLX's NAX kernels are compiled in
/// (`cpu` without an engine). The GPU half (architecture, NAX available at runtime) needs the
/// Metal device: `mrun --version` prints [`long`], and `mrun run --info` and `/mrun/status` print
/// the class too.
pub const LONG: &str = match NUMERICS {
    Build::Nax => concat!(env!("BUILD_MRUN_VERSION_LONG"), " numerics nax compiled"),
    Build::NoNax => concat!(env!("BUILD_MRUN_VERSION_LONG"), " numerics no nax"),
    Build::Nuvec => concat!(env!("BUILD_MRUN_VERSION_LONG"), " numerics nuvec"),
    Build::Cpu => concat!(env!("BUILD_MRUN_VERSION_LONG"), " numerics cpu"),
};

pub const MLX: &str = env!("BUILD_MLX_VERSION");
pub const MLX_C: &str = env!("BUILD_MLX_C_VERSION");

/// `mrun --version`: [`LONG`] plus the GPU half of the numerics class (008 T030) and how the
/// kernels are provided (016 FR-009): `…, available|not available, gpu <family>, kernels
/// <precompiled|runtime>`. It opens the Metal device (no socket); without an engine it is
/// [`LONG`].
pub fn long() -> String {
    #[cfg(feature = "engine")]
    {
        let numerics = mrun_engine::tensor::numerics();
        format!(
            "{LONG}, {}, gpu {}, kernels {}",
            if numerics.nax {
                "available"
            } else {
                "not available"
            },
            numerics.gpu_arch,
            mrun_engine::tensor::backend_info().kernels.as_str()
        )
    }
    #[cfg(not(feature = "engine"))]
    {
        LONG.to_string()
    }
}

/// The build's tensor backend (016 FR-008), `None` without an engine.
pub fn backend() -> Option<BackendInfo> {
    #[cfg(feature = "engine")]
    {
        Some(mrun_engine::tensor::backend_info())
    }
    #[cfg(not(feature = "engine"))]
    {
        None
    }
}

/// The build half of the numerics class.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Build {
    Nax,
    NoNax,
    Nuvec,
    Cpu,
}

#[cfg(feature = "engine")]
pub const NUMERICS: Build = if matches!(
    mrun_engine::engine::Backend::BUILD,
    mrun_engine::engine::Backend::Nuvec
) {
    Build::Nuvec
} else if mrun_engine::tensor::NAX_COMPILED {
    Build::Nax
} else {
    Build::NoNax
};
#[cfg(not(feature = "engine"))]
pub const NUMERICS: Build = Build::Cpu;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn long_version_names_everything() {
        assert!(LONG.starts_with(env!("CARGO_PKG_VERSION")));
        match NUMERICS {
            Build::Nuvec => {
                let info = backend().unwrap();
                assert!(LONG.contains(&format!("nuvec {} ", info.version)), "{LONG}");
            }
            _ => assert!(LONG.contains(&format!("mlx {MLX} mlx-c {MLX_C}")), "{LONG}"),
        }
        let date = LONG.split(", ").nth(1).unwrap().split(')').next().unwrap();
        assert_eq!(date.len(), 10, "{LONG}");
        let numerics = match NUMERICS {
            Build::Nax => " numerics nax compiled",
            Build::NoNax => " numerics no nax",
            Build::Nuvec => " numerics nuvec",
            Build::Cpu => " numerics cpu",
        };
        assert!(LONG.ends_with(numerics), "{LONG}");
    }

    #[test]
    fn the_long_version_adds_the_gpu_half() {
        let long = long();
        assert!(long.starts_with(LONG));
        if cfg!(feature = "engine") {
            assert!(long.contains("available, gpu "), "{long}");
            let kernels = backend().unwrap().kernels.as_str();
            assert!(long.ends_with(&format!(", kernels {kernels}")), "{long}");
        } else {
            assert_eq!(backend(), None);
        }
    }

    /// Only in the repository: a packaged `mrun` has no sibling `mrun-mlx-sys` sources (the
    /// build script sets `mrun_repo` and the path when they exist).
    #[cfg(mrun_repo)]
    #[test]
    fn mlx_pins_match_mlx_sys() {
        let build = include_str!(env!("BUILD_MLX_SYS_BUILD_RS"));
        assert!(build.contains(&format!("name: \"mlx\",\n    version: \"{MLX}\",")));
        assert!(build.contains(&format!("name: \"mlx-c\",\n    version: \"{MLX_C}\",")));
    }
}
