//! `mrun --version` works without a network (008 FR-004, FR-011): it is embedded at build time,
//! so the binary prints it with networking unavailable (`unshare -rn` where permitted).

use std::process::Command;

#[test]
fn version_prints_without_a_network() {
    let binary = env!("CARGO_BIN_EXE_mrun");
    let isolated = Command::new("unshare")
        .args(["-rn", binary, "--version"])
        .output()
        .ok()
        .filter(|o| o.status.success());
    let output =
        isolated.unwrap_or_else(|| Command::new(binary).arg("--version").output().unwrap());
    assert!(output.status.success());
    let text = String::from_utf8_lossy(&output.stdout);
    assert!(
        text.starts_with(&format!("mrun {}", env!("CARGO_PKG_VERSION"))),
        "{text}"
    );
    // The engine, its kernels and the NAX state (022 contract `cli-output`); no MLX wording.
    if cfg!(feature = "engine") {
        assert!(text.contains(" nuvec "), "{text}");
        assert!(
            text.contains(", kernels precompiled, nax ")
                || text.contains(", kernels runtime, nax "),
            "{text}"
        );
    }
    for word in ["mlx", "numerics", "available"] {
        assert!(!text.contains(word), "{text}");
    }
}
