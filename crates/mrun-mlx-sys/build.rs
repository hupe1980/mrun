//! Builds mlx-c v0.7.0 (`a341b49`) and MLX v0.32.4 (`36969c0`) from sha256-pinned source tars
//! with CMake and links both statically.
//!
//! - macOS: Metal backend + CPU (Accelerate); links Metal, Foundation, QuartzCore, Accelerate.
//!   `mlx.metallib` is exported as `DEP_MLXC_METALLIB` and copied next to the target binaries
//!   (development builds); a zstd-compressed copy (`lib/mlx.metallib.zst`, ~15 MB instead of
//!   ~215 MB) is exported as `DEP_MLXC_METALLIB_ZST` so dependants can embed it (D034).
//! - Linux: CPU backend only, BLAS/LAPACK (OpenBLAS + LAPACKE headers).
//!
//! The build is shared: it is installed once per configuration (MLX and mlx-c commits, patches,
//! target, build type, Metal, SDK version, resolved deployment target) in the user cache
//! directory, so debug, release,
//! test and check builds, rust-analyzer, `cargo install`, other worktrees and builds after
//! `cargo clean` reuse one MLX build instead of compiling it again. An OS file lock serializes
//! concurrent builds of the same configuration, and another one guards the shared downloads.
//!
//! Environment:
//! - `MLX_SYS_PREBUILT_DIR`: install prefix with `lib/libmlxc.a` and `lib/libmlx.a` (and on
//!   macOS `lib/mlx.metallib`) from an earlier build; skips fetching and CMake.
//! - `MLX_SYS_CACHE_DIR`: where builds and downloaded sources are kept. Default:
//!   `~/Library/Caches/mrun-mlx-sys` on macOS, `$XDG_CACHE_HOME/mrun-mlx-sys` or
//!   `~/.cache/mrun-mlx-sys` on Linux (`<target-dir>/mrun-mlx-sys` only when `HOME` is unset).
//! - `MLX_SYS_SOURCE_DIR`: directory holding the source tars (`mlx-<commit>.tar[.gz]`,
//!   `mlx-c-<commit>.tar[.gz]`) for offline builds; they are checked against the same hashes.
//! - `MLX_SYS_METAL`: `1` requires the Metal backend, `0` builds CPU only; unset on macOS means
//!   Metal, failing with instructions when the Metal compiler cannot compile a kernel.
//! - `MLX_SYS_BUILD_TYPE`: CMake build type (default `Release`, also for debug cargo profiles).
//! - `MLX_SYS_LINUX_LIBS`: comma-separated system libraries for Linux (default `openblas`, which
//!   carries LAPACK too; MLX's CMake finds the same library).
//! - `MACOSX_DEPLOYMENT_TARGET`: the macOS version MLX is built for; unset means this Mac's own
//!   version, so a source build on macOS 26.2 or later compiles the M5 Neural Accelerator
//!   (`*_nax`) kernels (D044). It is passed to CMake and is part of the cache key, as is the SDK
//!   version (`xcrun --show-sdk-version`).
//! - `MLX_SYS_REQUIRE_NAX`: `1` fails the build when the deployment target is 26.2 or later but
//!   `mlx.metallib` holds no `*_nax` kernel (release CI); otherwise that is a warning.
//! - `DOCS_RS`: set by docs.rs; nothing is built (the bindings are checked in).

use std::{
    env,
    fs::{self, File, TryLockError},
    io::{self, Read},
    path::{Path, PathBuf},
    process::{Command, Stdio},
    time::SystemTime,
};

use sha2::{Digest, Sha256};

/// A pinned upstream source: the GitHub archive of `commit`. `sha256` is the hash of the
/// uncompressed tar, i.e. the output of `git archive --format=tar --prefix=<name>-<commit>/
/// <commit>`; GitHub's `archive/<commit>.tar.gz` is that same tar, gzip-compressed. Hashing the
/// uncompressed tar keeps the pin independent of the gzip implementation.
struct Source {
    name: &'static str,
    version: &'static str,
    commit: &'static str,
    sha256: &'static str,
    repo: &'static str,
}

const MLX: Source = Source {
    name: "mlx",
    version: "0.32.4",
    commit: "36969c094354086e15d2a327a9c50feb20b9cf11",
    sha256: "7b05da60c9f0080dbccb11a0c70d23df85d29219d38f6d20732b43dc8616f592",
    repo: "ml-explore/mlx",
};

const MLX_C: Source = Source {
    name: "mlx-c",
    version: "0.7.0",
    commit: "a341b4925024b88b2c593468f16e12f5e17315da",
    sha256: "0a186bd06f557720ea010f5cfd010b7c533b12b23be1863d44758f28145b2fab",
    repo: "ml-explore/mlx-c",
};

/// Bumped whenever the CMake configuration below changes, so old cached builds are not reused.
const RECIPE: u32 = 1;

const ENV: [&str; 9] = [
    "MLX_SYS_PREBUILT_DIR",
    "MLX_SYS_CACHE_DIR",
    "MLX_SYS_SOURCE_DIR",
    "MLX_SYS_METAL",
    "MLX_SYS_BUILD_TYPE",
    "MLX_SYS_LINUX_LIBS",
    "MACOSX_DEPLOYMENT_TARGET",
    "MLX_SYS_REQUIRE_NAX",
    "DOCS_RS",
];

/// The first macOS version whose SDK and deployment target give MLX its NAX kernels (D044).
const NAX_FROM: (u32, u32) = (26, 2);

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=patches");
    for var in ENV {
        println!("cargo:rerun-if-env-changed={var}");
    }

    let out_dir = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR"));
    let target_os = env::var("CARGO_CFG_TARGET_OS").expect("CARGO_CFG_TARGET_OS");
    if env::var_os("DOCS_RS").is_some() {
        write_build_info(&out_dir, false, false);
        return;
    }
    let metal = target_os == "macos" && metal_enabled(&out_dir);
    let nax = metal && nax_target(&target_os);
    write_build_info(&out_dir, metal, nax);
    // `DEP_MLXC_NAX` for dependants (`mrun-engine` sets `cfg(mlx_nax)` from it).
    println!("cargo:nax={}", u8::from(nax));

    let (prefix, lock) = match env::var_os("MLX_SYS_PREBUILT_DIR") {
        Some(dir) => {
            let prefix = PathBuf::from(dir);
            let lock = prefix.join("lib").join(".mlx.metallib.zst.lock");
            (prefix, lock)
        }
        None => cached_build(&out_dir, &target_os, metal),
    };
    let lib_dir = prefix.join("lib");
    for lib in ["libmlxc.a", "libmlx.a"] {
        assert!(
            lib_dir.join(lib).is_file(),
            "{} not found (MLX_SYS_PREBUILT_DIR must be an install prefix with lib/{lib})",
            lib_dir.join(lib).display()
        );
    }
    // A removed or replaced cache entry (or prebuilt prefix) reruns this script.
    println!(
        "cargo:rerun-if-changed={}",
        lib_dir.join("libmlx.a").display()
    );

    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    println!("cargo:rustc-link-lib=static=mlxc");
    println!("cargo:rustc-link-lib=static=mlx");
    println!("cargo:root={}", prefix.display());
    println!("cargo:include={}", prefix.join("include").display());
    println!("cargo:metal={}", u8::from(metal));

    match target_os.as_str() {
        "macos" => {
            for framework in ["Metal", "Foundation", "QuartzCore", "Accelerate"] {
                println!("cargo:rustc-link-lib=framework={framework}");
            }
            println!("cargo:rustc-link-lib=dylib=c++");
            let metallib = lib_dir.join("mlx.metallib");
            if metallib.is_file() {
                println!("cargo:rerun-if-changed={}", metallib.display());
                println!("cargo:metallib={}", metallib.display());
                check_nax(&metallib);
                let zst = compressed_metallib(&metallib, &lock, &out_dir);
                println!("cargo:metallib_zst={}", zst.display());
                copy_metallib_next_to_binaries(&out_dir, &metallib);
            } else if metal {
                println!(
                    "cargo:warning=mlx.metallib not found in {}; Metal kernels will fail to load",
                    lib_dir.display()
                );
            }
        }
        "linux" => {
            let libs = env::var("MLX_SYS_LINUX_LIBS").unwrap_or_else(|_| "openblas".into());
            for lib in libs.split(',').map(str::trim).filter(|l| !l.is_empty()) {
                println!("cargo:rustc-link-lib=dylib={lib}");
            }
            println!("cargo:rustc-link-lib=dylib=stdc++");
        }
        other => panic!("mrun-mlx-sys supports macOS and Linux, not {other}"),
    }
}

/// `src/lib.rs` includes this: the pinned versions and whether Metal was built.
fn write_build_info(out_dir: &Path, metal: bool, nax: bool) {
    let info = format!(
        "/// MLX version this crate links.\npub const MLX_VERSION: &str = \"{}\";\n\
         /// mlx-c version this crate links.\npub const MLX_C_VERSION: &str = \"{}\";\n\
         /// Whether MLX was built with the Metal backend.\npub const METAL: bool = {metal};\n\
         /// Whether MLX was built with its NAX (M5 Neural Accelerator) kernels: Metal, SDK and\n\
         /// deployment target 26.2 or later (D044).\npub const NAX: bool = {nax};\n",
        MLX.version, MLX_C.version
    );
    fs::write(out_dir.join("build_info.rs"), info).expect("write build_info.rs");
}

/// Whether to build the Metal backend on macOS: `MLX_SYS_METAL`, else required. The check
/// compiles a one-line kernel, because `xcrun -f metal` succeeds on Xcode installs whose Metal
/// toolchain component is missing (Xcode 26 downloads it separately).
fn metal_enabled(out_dir: &Path) -> bool {
    match env::var("MLX_SYS_METAL").as_deref() {
        Ok("0" | "off" | "OFF" | "false") => return false,
        Ok("1" | "on" | "ON" | "true") | Err(_) => {}
        Ok(other) => panic!("MLX_SYS_METAL must be 0 or 1, got {other:?}"),
    }
    let dir = out_dir.join("metal-check");
    fs::create_dir_all(&dir).expect("create metal-check dir");
    let source = dir.join("check.metal");
    fs::write(
        &source,
        "kernel void check(device float* x [[buffer(0)]]) { x[0] = 1.0f; }\n",
    )
    .expect("write check.metal");
    let output = Command::new("xcrun")
        .args(["-sdk", "macosx", "metal", "-c"])
        .arg(&source)
        .arg("-o")
        .arg(dir.join("check.air"))
        .stdin(Stdio::null())
        .output();
    let detail = match output {
        Ok(out) if out.status.success() => return true,
        Ok(out) => String::from_utf8_lossy(&out.stderr).trim().to_string(),
        Err(e) => format!("cannot run xcrun: {e}"),
    };
    panic!(
        "\n\nThe Metal compiler cannot compile a kernel, and MLX needs it to build its GPU kernels:\n\
         \n    {}\n\
         \nInstall Xcode (App Store or developer.apple.com), then run:\n\
         \n    sudo xcode-select -s /Applications/Xcode.app\n\
         \n    xcodebuild -downloadComponent MetalToolchain\n\
         \nTo build the CPU backend only (slow; tests and development), set MLX_SYS_METAL=0.\n",
        detail.replace('\n', "\n    ")
    );
}

/// Returns the install prefix of the build for this configuration, building it first when no
/// other build has, and the path of the configuration's lock file.
fn cached_build(out_dir: &Path, target_os: &str, metal: bool) -> (PathBuf, PathBuf) {
    let root = cache_root(out_dir);
    fs::create_dir_all(&root).expect("create MLX cache dir");
    let patches = patches();
    let build_type = env::var("MLX_SYS_BUILD_TYPE").unwrap_or_else(|_| "Release".into());

    let mut key = Sha256::new();
    for part in [
        RECIPE.to_string(),
        MLX.commit.into(),
        MLX_C.commit.into(),
        env::var("TARGET").expect("TARGET"),
        build_type.clone(),
        metal.to_string(),
        sdk_version(target_os).unwrap_or_default(),
        deployment_target(target_os).unwrap_or_default(),
        env::var("MLX_SYS_LINUX_LIBS").unwrap_or_default(),
    ] {
        key.update(part.as_bytes());
        key.update([0]);
    }
    for patch in &patches {
        key.update(fs::read(patch).expect("read patch"));
    }
    let key = &hex(&key.finalize())[..16];
    let name = format!("mlx-{}-{key}", MLX.version);
    let prefix = root.join(&name);
    let done = prefix.join(".complete");
    let lock_path = root.join(format!("{name}.lock"));
    if done.is_file() {
        return (prefix, lock_path);
    }

    let _lock = Lock::acquire(&lock_path, "another build of MLX")
        .unwrap_or_else(|e| panic!("cannot lock {}: {e}", lock_path.display()));
    if done.is_file() {
        return (prefix, lock_path); // built by the process we waited for
    }
    let work = root.join(format!("{name}.partial"));
    let _ = fs::remove_dir_all(&work);
    let _ = fs::remove_dir_all(&prefix);
    let src = work.join("src");
    let mlx_src = extract(&root, &src, &MLX);
    let mlx_c_src = extract(&root, &src, &MLX_C);
    for patch in &patches {
        let file = fs::File::open(patch).expect("open patch");
        run(Command::new("patch")
            .args(["-p1", "--forward", "--silent", "-d"])
            .arg(&mlx_c_src)
            .stdin(Stdio::from(file)));
    }

    let mut config = cmake::Config::new(&mlx_c_src);
    config
        .out_dir(&work)
        .profile(&build_type)
        .define("FETCHCONTENT_SOURCE_DIR_MLX", &mlx_src)
        .define("FETCHCONTENT_QUIET", "ON")
        .define("BUILD_SHARED_LIBS", "OFF")
        .define("CMAKE_INSTALL_LIBDIR", "lib")
        .define("CMAKE_POSITION_INDEPENDENT_CODE", "ON")
        .define("MLX_C_BUILD_EXAMPLES", "OFF")
        .define("MLX_BUILD_TESTS", "OFF")
        .define("MLX_BUILD_EXAMPLES", "OFF")
        .define("MLX_BUILD_BENCHMARKS", "OFF")
        .define("MLX_BUILD_PYTHON_BINDINGS", "OFF")
        .define("MLX_BUILD_PYTHON_STUBS", "OFF")
        .define("MLX_BUILD_GGUF", "OFF")
        .define("MLX_BUILD_SAFETENSORS", "ON")
        .define("MLX_BUILD_CPU", "ON")
        .define("MLX_BUILD_CUDA", "OFF")
        .define("MLX_USE_CCACHE", "OFF")
        .define("MLX_BUILD_METAL", if metal { "ON" } else { "OFF" });
    // The resolved deployment target is part of the cache key; CMake builds for exactly it.
    if let Some(target) = deployment_target(target_os) {
        config.define("CMAKE_OSX_DEPLOYMENT_TARGET", target);
    }
    config.build();

    // Keep only the install tree (lib/, include/, share/): the build tree is several GB.
    let _ = fs::remove_dir_all(work.join("build"));
    let _ = fs::remove_dir_all(&src);
    fs::write(work.join(".complete"), "").expect("mark build complete");
    fs::rename(&work, &prefix).expect("publish MLX build");
    (prefix, lock_path)
}

/// Output of a command that succeeded, trimmed; `None` otherwise.
fn output(cmd: &mut Command) -> Option<String> {
    let out = cmd
        .stdin(Stdio::null())
        .stderr(Stdio::null())
        .output()
        .ok()?;
    let text = String::from_utf8(out.stdout).ok()?.trim().to_string();
    (out.status.success() && !text.is_empty()).then_some(text)
}

/// The macOS SDK version the Metal kernels are compiled with (macOS targets only).
fn sdk_version(target_os: &str) -> Option<String> {
    if target_os != "macos" {
        return None;
    }
    output(Command::new("xcrun").args(["--sdk", "macosx", "--show-sdk-version"]))
}

/// `MACOSX_DEPLOYMENT_TARGET`, else this Mac's version (`sw_vers -productVersion`), for macOS
/// targets; the build script runs on the Mac it builds for.
fn deployment_target(target_os: &str) -> Option<String> {
    if target_os != "macos" {
        return None;
    }
    env::var("MACOSX_DEPLOYMENT_TARGET")
        .ok()
        .filter(|t| !t.is_empty())
        .or_else(|| output(Command::new("sw_vers").arg("-productVersion")))
}

/// Whether both the SDK and the deployment target are at least [`NAX_FROM`], so MLX compiles its
/// NAX kernels.
fn nax_target(target_os: &str) -> bool {
    let at_least = |v: Option<String>| {
        v.as_deref()
            .and_then(major_minor)
            .is_some_and(|v| v >= NAX_FROM)
    };
    at_least(sdk_version(target_os)) && at_least(deployment_target(target_os))
}

/// `"26.2.1"` → `(26, 2)`.
fn major_minor(version: &str) -> Option<(u32, u32)> {
    let mut parts = version.split('.').map(str::parse::<u32>);
    let major = parts.next()?.ok()?;
    let minor = parts.next().unwrap_or(Ok(0)).ok()?;
    Some((major, minor))
}

/// With a deployment target of 26.2 or later the metallib must hold NAX kernels (their names
/// end in `_nax`); a warning, or with `MLX_SYS_REQUIRE_NAX=1` a failed build, when it does not.
fn check_nax(metallib: &Path) {
    let Some(target) = deployment_target("macos") else {
        return;
    };
    if major_minor(&target).is_none_or(|v| v < NAX_FROM) {
        return;
    }
    let bytes = fs::read(metallib).expect("read mlx.metallib");
    if bytes.windows(4).any(|w| w == b"_nax") {
        return;
    }
    let message = format!(
        "{} holds no *_nax kernels although the deployment target is {target} (NAX needs the \
         macOS {}.{} SDK or later)",
        metallib.display(),
        NAX_FROM.0,
        NAX_FROM.1
    );
    if env::var("MLX_SYS_REQUIRE_NAX").is_ok_and(|v| v == "1") {
        panic!("{message}");
    }
    println!("cargo:warning={message}");
}

/// `MLX_SYS_CACHE_DIR`, else the user cache directory (D034): `~/Library/Caches/mrun-mlx-sys`
/// on macOS, `$XDG_CACHE_HOME/mrun-mlx-sys` or `~/.cache/mrun-mlx-sys` elsewhere. Only without
/// `HOME` does it fall back to `<target-dir>/mrun-mlx-sys` (next to the profile directories).
fn cache_root(out_dir: &Path) -> PathBuf {
    if let Some(dir) = env::var_os("MLX_SYS_CACHE_DIR").filter(|d| !d.is_empty()) {
        return PathBuf::from(dir);
    }
    let home = env::var_os("HOME")
        .filter(|h| !h.is_empty())
        .map(PathBuf::from);
    // The build script runs on the host, so `cfg!` names the host OS (where the cache lives).
    if cfg!(target_os = "macos") {
        if let Some(home) = home {
            return home.join("Library/Caches/mrun-mlx-sys");
        }
    } else {
        let xdg = env::var_os("XDG_CACHE_HOME")
            .map(PathBuf::from)
            .filter(|d| d.is_absolute());
        if let Some(dir) = xdg.or_else(|| home.map(|h| h.join(".cache"))) {
            return dir.join("mrun-mlx-sys");
        }
    }
    // OUT_DIR = <target-dir>[/<triple>]/<profile>/build/<pkg>-<hash>/out
    out_dir
        .ancestors()
        .nth(4)
        .map_or_else(|| out_dir.join("mlx"), |dir| dir.join("mrun-mlx-sys"))
}

/// `patches/mlx-c-*.patch`, sorted.
fn patches() -> Vec<PathBuf> {
    let dir = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"))
        .join("patches");
    let mut patches: Vec<PathBuf> = fs::read_dir(dir)
        .map(|dir| {
            dir.filter_map(|e| e.ok().map(|e| e.path()))
                .filter(|p| {
                    p.file_name()
                        .and_then(|n| n.to_str())
                        .is_some_and(|n| n.starts_with("mlx-c-") && n.ends_with(".patch"))
                })
                .collect()
        })
        .unwrap_or_default();
    patches.sort();
    patches
}

/// Extracts the verified source tar of `source` into `dest`; returns its top directory.
fn extract(root: &Path, dest: &Path, source: &Source) -> PathBuf {
    let tar = fetch(root, source);
    fs::create_dir_all(dest).expect("create source dir");
    run(Command::new("tar").arg("-xf").arg(&tar).arg("-C").arg(dest));
    let dir = dest.join(format!("{}-{}", source.name, source.commit));
    assert!(
        dir.join("CMakeLists.txt").is_file(),
        "unexpected layout of the {} source tar",
        source.name
    );
    dir
}

/// Returns the path of the verified, uncompressed source tar of `source` in
/// `<root>/downloads/`, taking it from `MLX_SYS_SOURCE_DIR`, GitHub's archive or a shallow git
/// fetch, in that order. Builds of other configurations share the directory: a lock serializes
/// fetches, and temporary files carry the process id.
fn fetch(root: &Path, source: &Source) -> PathBuf {
    let downloads = root.join("downloads");
    fs::create_dir_all(&downloads).expect("create downloads dir");
    let tar = downloads.join(format!("{}-{}.tar", source.name, source.commit));
    let lock_path = downloads.join(".lock");
    let _lock = Lock::acquire(&lock_path, "another download of the MLX sources")
        .unwrap_or_else(|e| panic!("cannot lock {}: {e}", lock_path.display()));
    if tar.is_file() && sha256_file(&tar) == source.sha256 {
        return tar;
    }
    let pid = std::process::id();
    let temp = |suffix: &str| {
        downloads.join(format!(
            "{}-{}.{suffix}.{pid}.partial",
            source.name, source.commit
        ))
    };
    let partial = temp("tar");

    let mut tried = Vec::new();
    if let Some(dir) = env::var_os("MLX_SYS_SOURCE_DIR") {
        let dir = PathBuf::from(dir);
        let base = format!("{}-{}.tar", source.name, source.commit);
        for local in [dir.join(&base), dir.join(format!("{base}.gz"))] {
            if !local.is_file() {
                continue;
            }
            if local.extension().is_some_and(|e| e == "gz") {
                gunzip(&local, &partial);
            } else {
                fs::copy(&local, &partial).expect("copy source tar");
            }
            if accept(&partial, &tar, source) {
                return tar;
            }
            tried.push(format!("{} (hash mismatch)", local.display()));
        }
        tried.push(format!("{} (no {base}[.gz])", dir.display()));
    }

    let gz = temp("tar.gz");
    let url = format!(
        "https://github.com/{}/archive/{}.tar.gz",
        source.repo, source.commit
    );
    let downloaded = Command::new("curl")
        .args(["-fsSL", "--retry", "3", "--connect-timeout", "20"])
        .args(["--max-time", "600", "-o"])
        .arg(&gz)
        .arg(&url)
        .status()
        .is_ok_and(|s| s.success());
    if downloaded {
        gunzip(&gz, &partial);
        let _ = fs::remove_file(&gz);
        if accept(&partial, &tar, source) {
            return tar;
        }
        tried.push(format!("{url} (hash mismatch)"));
    } else {
        let _ = fs::remove_file(&gz);
        tried.push(format!("{url} (download failed)"));
    }

    let repo = temp("git");
    let _ = fs::remove_dir_all(&repo);
    // A stalled transfer (below 1 KB/s for 60 s) aborts instead of hanging the build.
    let git = || {
        let mut cmd = Command::new("git");
        cmd.args([
            "-c",
            "http.lowSpeedLimit=1000",
            "-c",
            "http.lowSpeedTime=60",
        ]);
        cmd
    };
    let fetched = git()
        .arg("init")
        .arg("-q")
        .arg(&repo)
        .status()
        .is_ok_and(|s| s.success())
        && git()
            .arg("-C")
            .arg(&repo)
            .args(["fetch", "-q", "--depth", "1"])
            .arg(format!("https://github.com/{}.git", source.repo))
            .arg(source.commit)
            .status()
            .is_ok_and(|s| s.success())
        && git()
            .arg("-C")
            .arg(&repo)
            .args(["archive", "--format=tar"])
            .arg(format!("--prefix={}-{}/", source.name, source.commit))
            .arg("-o")
            .arg(&partial)
            .arg(source.commit)
            .status()
            .is_ok_and(|s| s.success());
    let _ = fs::remove_dir_all(&repo);
    if fetched && accept(&partial, &tar, source) {
        return tar;
    }
    let _ = fs::remove_file(&partial);
    tried.push("git fetch (failed or hash mismatch)".into());
    panic!(
        "could not obtain the {} {} source (commit {}, sha256 {}):\n  {}\n\
         For offline builds put {}-{}.tar[.gz] into MLX_SYS_SOURCE_DIR.",
        source.name,
        source.version,
        source.commit,
        source.sha256,
        tried.join("\n  "),
        source.name,
        source.commit
    );
}

/// Moves `partial` to `tar` when its hash matches the pin; deletes it otherwise.
fn accept(partial: &Path, tar: &Path, source: &Source) -> bool {
    if sha256_file(partial) == source.sha256 {
        fs::rename(partial, tar).expect("store source tar");
        true
    } else {
        let _ = fs::remove_file(partial);
        false
    }
}

/// An exclusive OS file lock (`flock`), released when the process exits or the guard drops,
/// so a crashed build never leaves a stale lock behind. The lock file itself stays.
struct Lock(#[allow(dead_code)] File);

impl Lock {
    fn acquire(path: &Path, holder: &str) -> io::Result<Self> {
        let file = fs::OpenOptions::new()
            .create(true)
            .truncate(false)
            .write(true)
            .open(path)?;
        match file.try_lock() {
            Ok(()) => {}
            Err(TryLockError::WouldBlock) => {
                eprintln!("waiting for {holder} (lock {})", path.display());
                file.lock()?;
            }
            Err(TryLockError::Error(e)) => return Err(e),
        }
        Ok(Self(file))
    }
}

/// Returns `lib/mlx.metallib.zst` next to `metallib`, compressing it first (zstd level 19, once
/// per configuration, under the configuration's lock) when it is missing or older than the
/// metallib. When the prefix is not writable (a read-only `MLX_SYS_PREBUILT_DIR`), the compressed
/// copy goes to `OUT_DIR` instead.
fn compressed_metallib(metallib: &Path, lock: &Path, out_dir: &Path) -> PathBuf {
    let zst = metallib.with_file_name("mlx.metallib.zst");
    if is_newer(&zst, metallib) {
        return zst;
    }
    let (zst, _lock) = match Lock::acquire(lock, "another compression of mlx.metallib") {
        Ok(guard) => (zst, Some(guard)),
        Err(_) => (out_dir.join("mlx.metallib.zst"), None),
    };
    if is_newer(&zst, metallib) {
        return zst; // compressed by the process we waited for
    }
    let partial = zst.with_extension(format!("zst.{}.partial", std::process::id()));
    if let Err(e) = compress(metallib, &partial) {
        let _ = fs::remove_file(&partial);
        panic!("compressing {} failed: {e}", metallib.display());
    }
    fs::rename(&partial, &zst).expect("publish mlx.metallib.zst");
    zst
}

fn is_newer(file: &Path, than: &Path) -> bool {
    let modified = |p: &Path| fs::metadata(p).and_then(|m| m.modified()).ok();
    match (modified(file), modified(than)) {
        (Some(a), Some(b)) => a >= b,
        (Some(_), None) => true,
        _ => false,
    }
}

fn compress(src: &Path, dst: &Path) -> io::Result<()> {
    let started = SystemTime::now();
    let mut input = io::BufReader::new(File::open(src)?);
    let mut encoder = zstd::Encoder::new(File::create(dst)?, 19)?;
    let workers = std::thread::available_parallelism().map_or(1, |n| n.get());
    encoder.multithread(u32::try_from(workers).unwrap_or(1))?;
    io::copy(&mut input, &mut encoder)?;
    encoder.finish()?.sync_all()?;
    eprintln!(
        "compressed {} in {:?}",
        src.display(),
        started.elapsed().unwrap_or_default()
    );
    Ok(())
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn gunzip(gz: &Path, tar: &Path) {
    let out = fs::File::create(tar).expect("create tar");
    let status = Command::new("gzip")
        .arg("-dc")
        .arg(gz)
        .stdout(Stdio::from(out))
        .status()
        .expect("run gzip");
    assert!(status.success(), "gzip -dc {} failed", gz.display());
}

fn sha256_file(path: &Path) -> String {
    let mut file = fs::File::open(path).expect("open for hashing");
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = file.read(&mut buf).expect("read for hashing");
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    hex(&hasher.finalize())
}

/// MLX looks for `mlx.metallib` next to the executable: copy it into `<target>/<profile>/` and
/// its `deps/` and `examples/` (test and example binaries).
fn copy_metallib_next_to_binaries(out_dir: &Path, metallib: &Path) {
    // OUT_DIR = <target-dir>[/<triple>]/<profile>/build/<pkg>-<hash>/out
    let Some(profile_dir) = out_dir.ancestors().nth(3) else {
        return;
    };
    for dir in [
        profile_dir.to_path_buf(),
        profile_dir.join("deps"),
        profile_dir.join("examples"),
    ] {
        if fs::create_dir_all(&dir).is_ok() {
            let _ = fs::copy(metallib, dir.join("mlx.metallib"));
        }
    }
}

fn run(cmd: &mut Command) {
    let status = cmd
        .status()
        .unwrap_or_else(|e| panic!("failed to run {cmd:?}: {e}"));
    assert!(status.success(), "{cmd:?} failed with {status}");
}
