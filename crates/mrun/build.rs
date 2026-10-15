//! Embeds the build identity shown by `mrun --version` (008 FR-004):
//! `X.Y.Z (<commit7>, <date>) mlx <v> mlx-c <v> <os>-<arch>`.

use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

/// Vendored MLX and mlx-c versions (pinned in `crates/mlx-sys/build.rs`; a test keeps them equal).
const MLX_VERSION: &str = "0.32.4";
const MLX_C_VERSION: &str = "0.7.0";

fn main() {
    println!("cargo:rerun-if-env-changed=MRUN_BUILD_COMMIT");
    println!("cargo:rerun-if-env-changed=SOURCE_DATE_EPOCH");
    println!("cargo:rerun-if-changed=../../.git/HEAD");
    let commit = std::env::var("MRUN_BUILD_COMMIT")
        .ok()
        .or_else(|| {
            let out = Command::new("git")
                .args(["rev-parse", "--short=7", "HEAD"])
                .output()
                .ok()?;
            out.status
                .success()
                .then(|| String::from_utf8_lossy(&out.stdout).trim().to_string())
        })
        .unwrap_or_else(|| "unknown".into());
    // Reproducible builds pin the date with SOURCE_DATE_EPOCH.
    let secs = std::env::var("SOURCE_DATE_EPOCH")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(|| {
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map_or(0, |d| d.as_secs())
        });
    let os = match std::env::var("CARGO_CFG_TARGET_OS").as_deref() {
        Ok("macos") => "macos".to_string(),
        Ok(other) => other.to_string(),
        Err(_) => "unknown".into(),
    };
    let arch = match std::env::var("CARGO_CFG_TARGET_ARCH").as_deref() {
        Ok("aarch64") => "arm64".to_string(),
        Ok(other) => other.to_string(),
        Err(_) => "unknown".into(),
    };
    let version = std::env::var("CARGO_PKG_VERSION").unwrap_or_default();
    println!(
        "cargo:rustc-env=BUILD_MRUN_VERSION_LONG={version} ({commit}, {}) mlx {MLX_VERSION} mlx-c {MLX_C_VERSION} {os}-{arch}",
        date(secs)
    );
    println!("cargo:rustc-env=BUILD_MLX_VERSION={MLX_VERSION}");
    println!("cargo:rustc-env=BUILD_MLX_C_VERSION={MLX_C_VERSION}");
}

/// UTC `YYYY-MM-DD` of a Unix timestamp (civil-from-days).
fn date(secs: u64) -> String {
    let z = (secs / 86_400) as i64 + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + i64::from(month <= 2);
    format!("{year:04}-{month:02}-{day:02}")
}
