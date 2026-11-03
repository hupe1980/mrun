//! The embedded `mlx.metallib` (D034): MLX's GPU kernels, zstd-compressed into the binary so
//! `cargo install` and release tarballs need no file next to the executable.
//!
//! [`ensure`] runs once, before the first MLX call. A `mlx.metallib` next to the executable
//! wins (development builds, where `mrun-mlx-sys` copies it into the target directory); the
//! executable path is canonicalized first, so a symlinked binary (Homebrew's `bin/mrun` into the
//! Cellar) finds the file beside its real location, and MLX is pointed at that file. Otherwise the embedded copy is extracted once to
//! `~/Library/Caches/mrun/mlx-<MLX version>-<hash>.metallib` (temporary file plus rename, so
//! concurrent first runs are safe) and MLX is pointed at it. Builds without a metallib (CPU-only,
//! Linux) embed nothing and this is a no-op. A failure (cache not writable, disk full) is kept and
//! returned by every call, naming the path; engine load fails with it (008 FR-007).

#![allow(unsafe_code)]

#[cfg(mrun_metallib)]
mod embedded {
    use std::ffi::CString;
    use std::io;
    use std::path::{Path, PathBuf};

    use mlx_sys as sys;

    /// The zstd-compressed metallib exported by `mrun-mlx-sys` (`DEP_MLXC_METALLIB_ZST`).
    static COMPRESSED: &[u8] = include_bytes!(env!("BUILD_METALLIB_ZST"));
    /// First 12 hex digits of the SHA-256 of [`COMPRESSED`], computed by `build.rs`.
    const HASH: &str = env!("BUILD_METALLIB_HASH");

    pub(super) fn ensure() -> Result<(), String> {
        let beside = std::env::current_exe()
            .ok()
            .and_then(|exe| super::beside_executable(&exe));
        if let Some(path) = beside {
            return set_path(&path)
                .map_err(|e| format!("cannot use the GPU kernels at {}: {e}", path.display()));
        }
        let dir = cache_dir()
            .map_err(|e| format!("cannot extract the GPU kernels (mlx.metallib): {e}"))?;
        let path = dir.join(format!("mlx-{}-{HASH}.metallib", sys::MLX_VERSION));
        super::extract(COMPRESSED, &dir, &path)
            .and_then(|()| set_path(&path))
            .map_err(|e| {
                format!(
                    "cannot extract the GPU kernels to {}: {e} (needs about {} MB free)",
                    path.display(),
                    EXTRACTED_MB
                )
            })
    }

    /// Size of the extracted metallib, rounded up, for the error message.
    const EXTRACTED_MB: u64 = 215;

    /// `~/Library/Caches/mrun` (macOS) or `$XDG_CACHE_HOME/mrun` / `~/.cache/mrun`.
    fn cache_dir() -> io::Result<PathBuf> {
        let home = std::env::var_os("HOME")
            .filter(|h| !h.is_empty())
            .map(PathBuf::from);
        let base = if cfg!(target_os = "macos") {
            home.map(|h| h.join("Library/Caches"))
        } else {
            std::env::var_os("XDG_CACHE_HOME")
                .map(PathBuf::from)
                .filter(|d| d.is_absolute())
                .or_else(|| home.map(|h| h.join(".cache")))
        };
        base.map(|b| b.join("mrun"))
            .ok_or_else(|| io::Error::other("HOME is not set"))
    }

    fn set_path(path: &Path) -> io::Result<()> {
        let c_path = CString::new(path.as_os_str().as_encoded_bytes())
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, e))?;
        // SAFETY: a NUL-terminated path valid for the call; mlx-c copies it.
        let code = unsafe { sys::mlx_metal_set_metallib_path(c_path.as_ptr()) };
        super::super::check(code).map_err(|e| io::Error::other(e.0))
    }
}

/// `mlx.metallib` beside the executable's real location (symlinks resolved), when present.
#[cfg_attr(not(mrun_metallib), allow(dead_code))]
fn beside_executable(exe: &std::path::Path) -> Option<std::path::PathBuf> {
    let real = exe.canonicalize().unwrap_or_else(|_| exe.to_path_buf());
    let path = real.parent()?.join("mlx.metallib");
    path.is_file().then_some(path)
}

/// Decompresses `compressed` (zstd) to `path` in `dir` unless it is already there: a temporary
/// file in the same directory, synced, then renamed, so concurrent first runs never see a
/// partial file and the loser of a race keeps the winner's identical copy.
#[cfg_attr(not(mrun_metallib), allow(dead_code))]
fn extract(
    compressed: &[u8],
    dir: &std::path::Path,
    path: &std::path::Path,
) -> std::io::Result<()> {
    use std::io::Write;
    if path.is_file() {
        return Ok(());
    }
    std::fs::create_dir_all(dir)?;
    let temp = tempfile::NamedTempFile::new_in(dir)?;
    {
        let mut out = std::io::BufWriter::with_capacity(1 << 20, temp.as_file());
        zstd::stream::copy_decode(compressed, &mut out)?;
        out.flush()?;
    }
    temp.as_file().sync_all()?;
    match temp.persist(path) {
        Ok(_) => Ok(()),
        // Another process may have won the race; its file is just as good.
        Err(_) if path.is_file() => Ok(()),
        Err(e) => Err(e.error),
    }
}

/// Makes MLX find its GPU kernels; see the module docs. Runs once (from `mlx::init`, before any
/// other MLX call); every call returns the first call's outcome.
pub fn ensure() -> Result<(), String> {
    static OUTCOME: std::sync::OnceLock<Result<(), String>> = std::sync::OnceLock::new();
    OUTCOME
        .get_or_init(|| {
            // Errors of the calls below reach the handler, not mlx-c's default (which exits).
            super::install_handler();
            #[cfg(mrun_metallib)]
            return embedded::ensure();
            #[cfg(not(mrun_metallib))]
            Ok(())
        })
        .clone()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(unix)]
    fn the_metallib_beside_a_symlinked_executable_is_found_at_its_real_location() {
        let root = tempfile::tempdir().unwrap();
        let cellar = root.path().join("Cellar/mrun/0.1.0/bin");
        let bin = root.path().join("bin");
        std::fs::create_dir_all(&cellar).unwrap();
        std::fs::create_dir_all(&bin).unwrap();
        let exe = cellar.join("mrun");
        std::fs::write(&exe, b"").unwrap();
        let link = bin.join("mrun");
        std::os::unix::fs::symlink(&exe, &link).unwrap();
        // Nothing beside either: the embedded copy is used.
        assert_eq!(beside_executable(&link), None);
        // Beside the real executable, not beside the link.
        std::fs::write(cellar.join("mlx.metallib"), b"kernels").unwrap();
        let found = beside_executable(&link).unwrap();
        assert_eq!(found, cellar.canonicalize().unwrap().join("mlx.metallib"));
        // A file beside the link only does not count.
        std::fs::remove_file(cellar.join("mlx.metallib")).unwrap();
        std::fs::write(bin.join("mlx.metallib"), b"other").unwrap();
        assert_eq!(beside_executable(&link), None);
    }

    #[test]
    fn extraction_is_idempotent_and_safe_under_concurrency() {
        let content = vec![7u8; 1 << 20];
        let compressed = zstd::stream::encode_all(&content[..], 3).unwrap();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("mlx-x.metallib");
        std::thread::scope(|scope| {
            for _ in 0..8 {
                scope.spawn(|| extract(&compressed, dir.path(), &path).unwrap());
            }
        });
        assert_eq!(std::fs::read(&path).unwrap(), content);
        // No temporary file is left behind.
        assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 1);
        // A second call keeps the file as it is (no rewrite).
        let modified = std::fs::metadata(&path).unwrap().modified().unwrap();
        extract(b"not zstd", dir.path(), &path).unwrap();
        assert_eq!(
            std::fs::metadata(&path).unwrap().modified().unwrap(),
            modified
        );
    }
}
