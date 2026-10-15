//! Verification against a manifest: fast checks (size, modification time) on every use and a full
//! re-hash for `mrun verify` (FR-004). A same-size edit that keeps the modification time passes
//! the fast check; only the re-hash catches it.

use super::Manifest;
use crate::errors::{CliError, Code};
use sha1::Sha1;
use sha2::{Digest, Sha256};
use std::fs;
use std::io::{self, Read};
use std::path::Path;
use std::time::UNIX_EPOCH;

/// `ready` or `corrupt` with the problems found.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Status {
    Ready,
    Corrupt(Vec<String>),
}

impl Status {
    pub fn label(&self) -> &'static str {
        match self {
            Self::Ready => "ready",
            Self::Corrupt(_) => "corrupt",
        }
    }
}

/// Size and modification time of every file in `dir` (following snapshot symlinks) against the
/// manifest.
pub fn fast_check(manifest: &Manifest, dir: &Path) -> Status {
    let problems: Vec<String> = manifest
        .files
        .iter()
        .filter_map(|file| match fs::metadata(dir.join(&file.path)) {
            Err(_) => Some(format!("{} is missing", file.path)),
            Ok(meta) if meta.len() != file.size => Some(format!(
                "{} has {} bytes, expected {}",
                file.path,
                meta.len(),
                file.size
            )),
            Ok(meta) if mtime_ns(&meta) != file.mtime_ns => {
                Some(format!("{} was modified", file.path))
            }
            Ok(_) => None,
        })
        .collect();
    if problems.is_empty() {
        Status::Ready
    } else {
        Status::Corrupt(problems)
    }
}

/// Re-hashes every file; returns the mismatches.
pub fn full_check(manifest: &Manifest, dir: &Path) -> Vec<String> {
    manifest
        .files
        .iter()
        .filter_map(|file| match sha256_file(&dir.join(&file.path)) {
            Err(error) => Some(format!("{}: {error}", file.path)),
            Ok(sha) if sha != file.sha256 => Some(format!("{} does not match its hash", file.path)),
            Ok(_) => None,
        })
        .collect()
}

/// Records the current modification times after a full check passed, so a file that was only
/// touched is no longer reported by fast checks.
pub fn refresh_mtimes(manifest: &mut Manifest, dir: &Path) {
    for file in &mut manifest.files {
        if let Ok(meta) = fs::metadata(dir.join(&file.path)) {
            file.mtime_ns = mtime_ns(&meta);
        }
    }
}

/// E032 naming the problems, with the repair command.
pub fn corrupt_error(manifest: &Manifest, problems: &[String]) -> CliError {
    CliError::new(
        Code::ModelCorrupt,
        format!(
            "model '{}' is corrupt: {}",
            manifest.name(),
            problems.join("; ")
        ),
    )
    .hint(format!("run 'mrun pull {}' to repair it", manifest.name()))
}

pub fn mtime_ns(meta: &fs::Metadata) -> u64 {
    meta.modified()
        .ok()
        .and_then(|time| time.duration_since(UNIX_EPOCH).ok())
        .map_or(0, |duration| duration.as_nanos() as u64)
}

pub fn sha256_file(path: &Path) -> io::Result<String> {
    let mut hasher = Sha256::new();
    stream(path, |chunk| hasher.update(chunk))?;
    Ok(hex(&hasher.finalize()))
}

pub fn sha256_bytes(bytes: &[u8]) -> String {
    hex(&Sha256::digest(bytes))
}

/// Git blob id (`sha1("blob <len>\0" + content)`): the hub's hash of files outside LFS.
pub fn git_sha1_file(path: &Path) -> io::Result<String> {
    let mut hasher = Sha1::new();
    hasher.update(format!("blob {}\0", fs::metadata(path)?.len()));
    stream(path, |chunk| hasher.update(chunk))?;
    Ok(hex(&hasher.finalize()))
}

fn stream(path: &Path, mut each: impl FnMut(&[u8])) -> io::Result<()> {
    let mut file = fs::File::open(path)?;
    let mut buffer = vec![0; 1 << 20];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            return Ok(());
        }
        each(&buffer[..read]);
    }
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hub::test_support;

    #[test]
    fn a_flipped_byte_with_preserved_mtime_is_caught_by_verify_only() {
        let root = tempfile::tempdir().unwrap();
        let ctx = test_support::ctx(root.path(), "http://unused");
        let manifest = test_support::cached_model(
            &ctx.paths,
            Some("m"),
            "org/m",
            &"a".repeat(40),
            &[("config.json", b"{}"), ("model.safetensors", b"weights")],
        );
        let dir = manifest.snapshot_dir(&ctx.paths);
        assert_eq!(fast_check(&manifest, &dir), Status::Ready);
        assert!(full_check(&manifest, &dir).is_empty());

        let weights = dir.join("model.safetensors");
        let modified = fs::metadata(&weights).unwrap().modified().unwrap();
        fs::write(&weights, b"weightz").unwrap();
        fs::File::options()
            .write(true)
            .open(&weights)
            .unwrap()
            .set_modified(modified)
            .unwrap();

        assert_eq!(fast_check(&manifest, &dir), Status::Ready);
        assert_eq!(
            full_check(&manifest, &dir),
            ["model.safetensors does not match its hash"]
        );
    }

    #[test]
    fn size_and_mtime_changes_are_caught_by_fast_checks() {
        let root = tempfile::tempdir().unwrap();
        let ctx = test_support::ctx(root.path(), "http://unused");
        let mut manifest = test_support::cached_model(
            &ctx.paths,
            Some("m"),
            "org/m",
            &"a".repeat(40),
            &[("config.json", b"{}"), ("model.safetensors", b"weights")],
        );
        let dir = manifest.snapshot_dir(&ctx.paths);
        let weights = dir.join("model.safetensors");

        let file = fs::File::options().write(true).open(&weights).unwrap();
        file.set_modified(UNIX_EPOCH + std::time::Duration::from_secs(1))
            .unwrap();
        let Status::Corrupt(problems) = fast_check(&manifest, &dir) else {
            panic!("touched file passed")
        };
        assert_eq!(problems, ["model.safetensors was modified"]);
        // Same content: verify passes and refreshes the record.
        assert!(full_check(&manifest, &dir).is_empty());
        refresh_mtimes(&mut manifest, &dir);
        assert_eq!(fast_check(&manifest, &dir), Status::Ready);

        fs::write(&weights, b"short").unwrap();
        assert_eq!(
            fast_check(&manifest, &dir),
            Status::Corrupt(vec!["model.safetensors has 5 bytes, expected 7".into()])
        );
        fs::remove_file(&weights).unwrap();
        assert_eq!(fast_check(&manifest, &dir).label(), "corrupt");
    }

    #[test]
    fn git_blob_ids_match_git() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("f");
        fs::write(&path, b"hello\n").unwrap();
        // `printf 'hello\n' | git hash-object --stdin`
        assert_eq!(
            git_sha1_file(&path).unwrap(),
            "ce013625030ba8dba906f756967f9e9ca394464a"
        );
    }
}
