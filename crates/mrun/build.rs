//! Embeds the build identity shown by `mrun --version` (008 FR-004):
//! `X.Y.Z (<commit7>, <date>) mlx <v> mlx-c <v> <os>-<arch>`.

use std::path::PathBuf;
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

/// Vendored MLX and mlx-c versions (pinned in `crates/mrun-mlx-sys/build.rs`; a test keeps them equal).
const MLX_VERSION: &str = "0.32.4";
const MLX_C_VERSION: &str = "0.7.0";

fn main() {
    println!("cargo:rerun-if-env-changed=MRUN_BUILD_COMMIT");
    println!("cargo:rerun-if-env-changed=SOURCE_DATE_EPOCH");
    println!("cargo::rustc-check-cfg=cfg(mrun_repo)");
    // Inside the repository (not a packaged crate, which has neither `.git` nor the sibling
    // crates): watch HEAD for the commit, and let tests read the sibling `mrun-mlx-sys` sources.
    let manifest_dir = PathBuf::from(std::env::var_os("CARGO_MANIFEST_DIR").unwrap_or_default());
    let repo = manifest_dir.join("../..");
    let in_repo = repo.join(".git").exists();
    if in_repo {
        for path in git(&["rev-parse", "--git-path", "HEAD"]).into_iter().chain(
            git(&["symbolic-ref", "-q", "HEAD"])
                .and_then(|r| git(&["rev-parse", "--git-path", &r])),
        ) {
            let path = manifest_dir.join(path);
            if path.exists() {
                println!("cargo:rerun-if-changed={}", path.display());
            }
        }
    }
    let sys_build = repo.join("crates/mrun-mlx-sys/build.rs");
    if sys_build.is_file() {
        println!("cargo::rustc-cfg=mrun_repo");
        println!(
            "cargo::rustc-env=BUILD_MLX_SYS_BUILD_RS={}",
            sys_build.display()
        );
    }
    let commit = std::env::var("MRUN_BUILD_COMMIT")
        .ok()
        .filter(|c| !c.is_empty())
        .or_else(|| {
            if in_repo {
                git(&["rev-parse", "--short=7", "HEAD"])
            } else {
                None
            }
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
    // The tensor backend in the version line (spec 016): `nuvec <pinned version>` on the nuvec
    // build, the vendored MLX and mlx-c versions otherwise.
    let backend = if std::env::var_os("CARGO_FEATURE_NUVEC").is_some() {
        format!("nuvec {}", nuvec_version())
    } else {
        format!("mlx {MLX_VERSION} mlx-c {MLX_C_VERSION}")
    };
    println!(
        "cargo:rustc-env=BUILD_MRUN_VERSION_LONG={version} ({commit}, {}) {backend} {os}-{arch}",
        date(secs)
    );
    println!("cargo:rustc-env=BUILD_MLX_VERSION={MLX_VERSION}");
    println!("cargo:rustc-env=BUILD_MLX_C_VERSION={MLX_C_VERSION}");
    embed_ui();
}

/// Embeds the built window (`ui/dist`, spec 018) as `UI_BUNDLE: &[(name, bytes)]` in
/// `OUT_DIR/ui_bundle.rs`. A missing bundle embeds nothing (the host then serves a page saying
/// so) except in release and CI builds, which fail: a release never ships without its window.
fn embed_ui() {
    let dist = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("ui/dist");
    println!("cargo:rerun-if-changed={}", dist.display());
    let mut files: Vec<String> = std::fs::read_dir(&dist)
        .map(|d| {
            d.flatten()
                .filter(|e| e.path().is_file())
                .map(|e| e.file_name().to_string_lossy().into_owned())
                .filter(|n| !n.ends_with(".map"))
                .collect()
        })
        .unwrap_or_default();
    files.sort();
    if files.is_empty() {
        println!(
            "cargo:warning=crates/mrun/ui/dist is absent: `mrun ui` shows no window (cd crates/mrun/ui && npm ci && npm run build)"
        );
        let release = std::env::var("PROFILE").is_ok_and(|p| p == "release");
        if release || std::env::var_os("CI").is_some() {
            panic!(
                "crates/mrun/ui/dist is absent: build the window first (cd crates/mrun/ui && npm ci && npm run build)"
            );
        }
    }
    let mut src = String::from(
        "/// The built window: (file name, bytes).\npub static UI_BUNDLE: &[(&str, &[u8])] = &[\n",
    );
    for f in &files {
        src.push_str(&format!(
            "    ({f:?}, include_bytes!({:?})),\n",
            dist.join(f).display().to_string()
        ));
    }
    src.push_str("];\n");
    let out = PathBuf::from(std::env::var_os("OUT_DIR").unwrap_or_default());
    std::fs::write(out.join("ui_bundle.rs"), src).expect("write ui_bundle.rs");
}

/// The nuvec version `mrun-engine` pins (`nuvec = { version = "=X.Y.Z", … }`), or `?` when the
/// sibling manifest is not there (a packaged crate).
fn nuvec_version() -> String {
    let manifest =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../mrun-engine/Cargo.toml");
    println!("cargo:rerun-if-changed={}", manifest.display());
    std::fs::read_to_string(manifest)
        .ok()
        .and_then(|text| {
            let line = text.lines().find(|l| l.starts_with("nuvec = {"))?;
            let start = line.find("version = \"")? + "version = \"".len();
            let rest = &line[start..];
            Some(rest[..rest.find('"')?].trim_start_matches('=').to_string())
        })
        .unwrap_or_else(|| "?".into())
}

/// Runs `git <args>` in the crate directory; trimmed stdout on success.
fn git(args: &[&str]) -> Option<String> {
    let out = Command::new("git").args(args).output().ok()?;
    out.status
        .success()
        .then(|| String::from_utf8_lossy(&out.stdout).trim().to_string())
        .filter(|s| !s.is_empty())
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
