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
    // The tensor backend and its numerics class (016 FR-008).
    if cfg!(feature = "nuvec") {
        assert!(
            text.contains(" nuvec ") && text.contains(" numerics nuvec"),
            "{text}"
        );
        assert!(
            text.trim_end().ends_with("kernels precompiled") || text.contains("kernels runtime"),
            "{text}"
        );
    } else {
        assert!(text.contains(" mlx ") && text.contains(" mlx-c "), "{text}");
    }
}
