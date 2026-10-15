//! Builds mlx-c (git submodule `vendor/mlx-c`, v0.7.0 `a341b49`) and MLX v0.32.4 (`36969c0`,
//! sha256-pinned source tarball) with CMake and links both statically.
//!
//! - macOS: Metal backend + CPU (Accelerate); links Metal, Foundation, QuartzCore, Accelerate.
//!   `mlx.metallib` is exported as `DEP_MLXC_METALLIB` and copied next to the target binaries.
//! - Linux: CPU backend only, BLAS/LAPACK (OpenBLAS + LAPACKE headers).
//!
//! Environment:
//! - `MLX_SYS_PREBUILT_DIR`: directory containing `lib/libmlxc.a` and `lib/libmlx.a` (the
//!   install prefix of an earlier build); skips CMake entirely.
//! - `MLX_SYS_MLX_TARBALL`: local MLX source (`.tar` or `.tar.gz`) instead of downloading it;
//!   it is checked against the same pinned hash.
//! - `MLX_SYS_BUILD_TYPE`: CMake build type (default `Release`, also for debug cargo profiles).
//! - `MLX_SYS_LINUX_LIBS`: comma-separated system libraries for Linux (default `openblas`, which
//!   carries LAPACK too; MLX's CMake finds the same library).

use std::{
    env, fs,
    io::Read,
    path::{Path, PathBuf},
    process::{Command, Stdio},
};

use sha2::{Digest, Sha256};

const MLX_VERSION: &str = "0.32.4";
const MLX_COMMIT: &str = "36969c094354086e15d2a327a9c50feb20b9cf11";
/// sha256 of the uncompressed source tar, i.e. the output of
/// `git archive --format=tar --prefix=mlx-<commit>/ <commit>`; GitHub's
/// `archive/<commit>.tar.gz` is that same tar, gzip-compressed. Hashing the uncompressed tar
/// keeps the pin independent of the gzip implementation.
const MLX_TAR_SHA256: &str = "7b05da60c9f0080dbccb11a0c70d23df85d29219d38f6d20732b43dc8616f592";
const MLX_GIT_URL: &str = "https://github.com/ml-explore/mlx.git";

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=vendor/mlx-c/CMakeLists.txt");
    println!("cargo:rerun-if-changed=vendor/mlx-c/mlx/c");
    println!("cargo:rerun-if-changed=patches");
    for var in [
        "MLX_SYS_PREBUILT_DIR",
        "MLX_SYS_MLX_TARBALL",
        "MLX_SYS_BUILD_TYPE",
        "MLX_SYS_LINUX_LIBS",
    ] {
        println!("cargo:rerun-if-env-changed={var}");
    }

    let target_os = env::var("CARGO_CFG_TARGET_OS").expect("CARGO_CFG_TARGET_OS");
    let out_dir = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR"));

    let prefix = match env::var_os("MLX_SYS_PREBUILT_DIR") {
        Some(dir) => PathBuf::from(dir),
        None => build(&out_dir, &target_os),
    };
    let lib_dir = prefix.join("lib");
    for lib in ["libmlxc.a", "libmlx.a"] {
        assert!(
            lib_dir.join(lib).is_file(),
            "{} not found (MLX_SYS_PREBUILT_DIR must be an install prefix with lib/{lib})",
            lib_dir.join(lib).display()
        );
    }

    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    println!("cargo:rustc-link-lib=static=mlxc");
    println!("cargo:rustc-link-lib=static=mlx");
    println!("cargo:root={}", prefix.display());
    println!("cargo:include={}", prefix.join("include").display());

    match target_os.as_str() {
        "macos" => {
            for framework in ["Metal", "Foundation", "QuartzCore", "Accelerate"] {
                println!("cargo:rustc-link-lib=framework={framework}");
            }
            println!("cargo:rustc-link-lib=dylib=c++");
            let metallib = lib_dir.join("mlx.metallib");
            if metallib.is_file() {
                println!("cargo:metallib={}", metallib.display());
                copy_metallib_next_to_binaries(&out_dir, &metallib);
            } else {
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
        other => panic!("mlx-sys supports macOS and Linux, not {other}"),
    }
}

/// Configures, builds and installs mlx-c (and MLX through it) into `out_dir`; returns the prefix.
fn build(out_dir: &Path, target_os: &str) -> PathBuf {
    let mlx_c = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"))
        .join("vendor/mlx-c");
    assert!(
        mlx_c.join("CMakeLists.txt").is_file(),
        "{} is empty: run `git submodule update --init crates/mlx-sys/vendor/mlx-c`",
        mlx_c.display()
    );
    let mlx_c = patched_mlx_c(&mlx_c, out_dir);
    let mlx_src = mlx_source(out_dir);

    let build_type = env::var("MLX_SYS_BUILD_TYPE").unwrap_or_else(|_| "Release".into());
    let mut config = cmake::Config::new(&mlx_c);
    config
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
        .define("MLX_USE_CCACHE", "OFF");
    match target_os {
        "macos" => {
            config.define("MLX_BUILD_METAL", "ON");
        }
        _ => {
            config.define("MLX_BUILD_METAL", "OFF");
        }
    }
    config.build()
}

/// Copies the mlx-c submodule to `OUT_DIR/mlx-c` and applies `patches/mlx-c-*.patch` (the
/// submodule itself stays pristine). Skipped when the inputs are unchanged, so CMake does not
/// recompile mlx-c on every build-script run.
fn patched_mlx_c(vendor: &Path, out_dir: &Path) -> PathBuf {
    let manifest_dir = vendor.parent().and_then(Path::parent).expect("crate dir");
    let mut patches: Vec<PathBuf> = fs::read_dir(manifest_dir.join("patches"))
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

    let mut files = Vec::new();
    collect_files(vendor, vendor, &mut files);
    files.sort();
    let mut hasher = Sha256::new();
    for rel in &files {
        hasher.update(rel.to_string_lossy().as_bytes());
        hasher.update(fs::read(vendor.join(rel)).expect("read mlx-c source"));
    }
    for patch in &patches {
        hasher.update(fs::read(patch).expect("read patch"));
    }
    let digest = hex(&hasher.finalize());

    let dst = out_dir.join("mlx-c");
    let stamp = out_dir.join("mlx-c.stamp");
    if dst.join("CMakeLists.txt").is_file()
        && fs::read_to_string(&stamp).is_ok_and(|s| s.trim() == digest)
    {
        return dst;
    }
    let _ = fs::remove_dir_all(&dst);
    for rel in &files {
        let to = dst.join(rel);
        fs::create_dir_all(to.parent().expect("parent")).expect("create mlx-c dir");
        fs::copy(vendor.join(rel), &to).expect("copy mlx-c source");
    }
    for patch in &patches {
        let file = fs::File::open(patch).expect("open patch");
        run(Command::new("patch")
            .args(["-p1", "--forward", "--silent", "-d"])
            .arg(&dst)
            .stdin(Stdio::from(file)));
    }
    fs::write(&stamp, &digest).expect("write stamp");
    dst
}

/// Lists the files under `dir` relative to `root`, skipping VCS metadata and the parts of
/// mlx-c the build does not use.
fn collect_files(root: &Path, dir: &Path, out: &mut Vec<PathBuf>) {
    for entry in fs::read_dir(dir).expect("read mlx-c dir").flatten() {
        let path = entry.path();
        let name = entry.file_name();
        if dir == root
            && [".git", "docs", "examples", "python"]
                .iter()
                .any(|n| name == *n)
        {
            continue;
        }
        if path.is_dir() {
            collect_files(root, &path, out);
        } else {
            out.push(path.strip_prefix(root).expect("prefix").to_path_buf());
        }
    }
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// Returns the extracted MLX source directory, fetching and verifying the pinned tarball first.
fn mlx_source(out_dir: &Path) -> PathBuf {
    let src_root = out_dir.join("mlx-src");
    let src = src_root.join(format!("mlx-{MLX_COMMIT}"));
    let stamp = src_root.join(".verified");
    if src.join("CMakeLists.txt").is_file()
        && fs::read_to_string(&stamp).is_ok_and(|s| s.trim() == MLX_TAR_SHA256)
    {
        return src;
    }

    let tar = out_dir.join(format!("mlx-{MLX_VERSION}-{MLX_COMMIT}.tar"));
    if !(tar.is_file() && sha256_file(&tar) == MLX_TAR_SHA256) {
        fetch_tar(out_dir, &tar);
    }
    let digest = sha256_file(&tar);
    if digest != MLX_TAR_SHA256 {
        let _ = fs::remove_file(&tar);
        panic!("MLX source tarball hash mismatch: expected {MLX_TAR_SHA256}, got {digest}");
    }

    let _ = fs::remove_dir_all(&src_root);
    fs::create_dir_all(&src_root).expect("create mlx-src");
    run(Command::new("tar")
        .arg("-xf")
        .arg(&tar)
        .arg("-C")
        .arg(&src_root));
    assert!(
        src.join("CMakeLists.txt").is_file(),
        "unexpected tarball layout"
    );
    fs::write(&stamp, MLX_TAR_SHA256).expect("write stamp");
    src
}

/// Writes the uncompressed MLX source tar to `tar`: from `MLX_SYS_MLX_TARBALL`, else GitHub's
/// archive, else `git archive` of a shallow fetch of the pinned commit.
fn fetch_tar(out_dir: &Path, tar: &Path) {
    if let Some(local) = env::var_os("MLX_SYS_MLX_TARBALL") {
        let local = PathBuf::from(local);
        if local.extension().is_some_and(|e| e == "gz" || e == "tgz") {
            gunzip(&local, tar);
        } else {
            fs::copy(&local, tar).expect("copy MLX_SYS_MLX_TARBALL");
        }
        return;
    }

    let gz = out_dir.join("mlx.tar.gz");
    let url = format!("https://github.com/ml-explore/mlx/archive/{MLX_COMMIT}.tar.gz");
    let downloaded = Command::new("curl")
        .args(["-fsSL", "--retry", "3", "-o"])
        .arg(&gz)
        .arg(&url)
        .status()
        .is_ok_and(|s| s.success());
    if downloaded {
        gunzip(&gz, tar);
        let _ = fs::remove_file(&gz);
        if sha256_file(tar) == MLX_TAR_SHA256 {
            return;
        }
        println!("cargo:warning={url} did not match the pinned hash; falling back to git");
    } else {
        println!("cargo:warning=could not download {url}; falling back to git");
    }

    let repo = out_dir.join("mlx-git");
    let _ = fs::remove_dir_all(&repo);
    fs::create_dir_all(&repo).expect("create mlx-git");
    run(Command::new("git").arg("init").arg("-q").arg(&repo));
    run(Command::new("git").arg("-C").arg(&repo).args([
        "fetch",
        "-q",
        "--depth",
        "1",
        MLX_GIT_URL,
        MLX_COMMIT,
    ]));
    run(Command::new("git").arg("-C").arg(&repo).args([
        "archive",
        "--format=tar",
        &format!("--prefix=mlx-{MLX_COMMIT}/"),
        "-o",
        tar.to_str().expect("utf-8 OUT_DIR"),
        MLX_COMMIT,
    ]));
    let _ = fs::remove_dir_all(&repo);
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

/// MLX looks for `mlx.metallib` next to the executable: copy it into `target/<profile>/` and
/// its `deps/` and `examples/` (test and example binaries).
fn copy_metallib_next_to_binaries(out_dir: &Path, metallib: &Path) {
    // OUT_DIR = target/<profile>/build/mlx-sys-<hash>/out
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
