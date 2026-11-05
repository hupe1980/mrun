//! `mrun --version`: version, commit, build date, the engine's version and platform, embedded at
//! build time; printing it opens no socket (008 FR-004, FR-011; 022 contract `cli-output`).

use mrun_engine::engine::BackendInfo;

/// `X.Y.Z (<commit7>, <date>) nuvec <v> <os>-<arch>` (`X.Y.Z (<commit7>, <date>) <os>-<arch>`
/// without the engine). The GPU part needs the Metal device: `mrun --version` prints [`long`].
pub const LONG: &str = env!("BUILD_MRUN_VERSION_LONG");

/// `mrun --version`: [`LONG`] plus the GPU family, how the kernels are provided (016 FR-009) and
/// whether the NAX kernels run, and why not: `…, gpu <family>, kernels <precompiled|runtime>, nax
/// <on|off (<reason>)>`. It opens the Metal device (no socket); without an engine it is [`LONG`].
pub fn long() -> String {
    #[cfg(feature = "engine")]
    {
        let numerics = mrun_engine::tensor::numerics();
        format!(
            "{LONG}, gpu {}, kernels {}, nax {}",
            numerics.gpu_arch,
            mrun_engine::tensor::backend_info().kernels.as_str(),
            mrun_engine::tensor::nax_status()
        )
    }
    #[cfg(not(feature = "engine"))]
    {
        LONG.to_string()
    }
}

/// The build's engine (016 FR-008), `None` without one.
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn long_version_names_everything() {
        assert!(LONG.starts_with(env!("CARGO_PKG_VERSION")));
        match backend() {
            Some(info) => assert!(
                LONG.contains(&format!(" nuvec {} ", info.version)),
                "{LONG}"
            ),
            None => assert!(!LONG.contains("nuvec"), "{LONG}"),
        }
        let date = LONG.split(", ").nth(1).unwrap().split(')').next().unwrap();
        assert_eq!(date.len(), 10, "{LONG}");
        for word in ["mlx", "numerics", "available"] {
            assert!(!LONG.contains(word), "{LONG}");
        }
    }

    #[test]
    fn the_long_version_adds_the_gpu_part() {
        let long = long();
        assert!(long.starts_with(LONG));
        match backend() {
            Some(info) => {
                assert!(long.contains(", gpu "), "{long}");
                assert!(
                    long.contains(&format!(", kernels {}, nax ", info.kernels.as_str())),
                    "{long}"
                );
                let nax = long.rsplit(", nax ").next().unwrap();
                assert!(
                    nax == "on" || nax.starts_with("on (") || nax.starts_with("off ("),
                    "{long}"
                );
            }
            None => assert_eq!(long, LONG),
        }
    }
}
