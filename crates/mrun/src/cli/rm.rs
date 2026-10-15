//! `mrun rm <model> [--yes]` (FR-003, research R10). The HF cache is shared and records no owner:
//! rm deletes the pinned revision's snapshot and the blobs no other revision of the repository
//! links to, removes refs only when they point to that revision, and warns that other tools
//! using that revision lose it.

use crate::errors::{CliError, Code};
use crate::hub::{self, Ctx, Manifest, Paths, human_bytes, short};
use std::collections::BTreeSet;
use std::fs;
use std::io;
use std::path::Path;

#[derive(Debug, Clone, clap::Args)]
pub struct RmArgs {
    /// Alias or repo id (org/name[@revision]) of a pulled model.
    pub model: String,
    /// Remove without asking.
    #[arg(long, short)]
    pub yes: bool,
}

/// What `remove` deleted.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct Removed {
    pub blobs: usize,
    pub bytes: u64,
    /// Blobs kept because another revision links to them.
    pub kept_shared: usize,
    /// Refs removed (relative to `refs/`).
    pub refs: Vec<String>,
}

/// Deletes the manifest's revision from the HF cache and the manifest itself.
pub fn remove(paths: &Paths, manifest: &Manifest) -> Result<Removed, CliError> {
    let repo_dir = paths.repo_dir(&manifest.repo);
    let snapshots = repo_dir.join("snapshots");
    let ours = snapshots.join(&manifest.revision);
    let failed = |error: io::Error| {
        CliError::new(
            Code::Internal,
            format!("cannot remove from {}: {error}", repo_dir.display()),
        )
        .hint("check permissions of the Hugging Face cache (HF_HUB_CACHE)")
    };

    let mut linked = BTreeSet::new();
    collect_blobs(&ours, &mut linked).map_err(failed)?;
    let mut shared = BTreeSet::new();
    if let Ok(entries) = fs::read_dir(&snapshots) {
        for entry in entries.flatten() {
            if entry.file_name() != manifest.revision.as_str() {
                collect_blobs(&entry.path(), &mut shared).map_err(failed)?;
            }
        }
    }

    let mut removed = Removed::default();
    if ours.exists() {
        fs::remove_dir_all(&ours).map_err(failed)?;
    }
    for blob in &linked {
        if shared.contains(blob) {
            removed.kept_shared += 1;
            continue;
        }
        let path = repo_dir.join("blobs").join(blob);
        if let Ok(meta) = fs::metadata(&path) {
            fs::remove_file(&path).map_err(failed)?;
            removed.blobs += 1;
            removed.bytes += meta.len();
        }
    }
    remove_refs(
        &repo_dir.join("refs"),
        "",
        &manifest.revision,
        &mut removed.refs,
    )
    .map_err(failed)?;
    // No revision left: drop the repository (as `hf cache delete` does), partial files included.
    if fs::read_dir(&snapshots).is_ok_and(|mut entries| entries.next().is_none()) {
        fs::remove_dir_all(&repo_dir).map_err(failed)?;
    }
    match fs::remove_file(manifest.path(paths)) {
        Err(error) if error.kind() != io::ErrorKind::NotFound => return Err(failed(error)),
        _ => {}
    }
    Ok(removed)
}

/// Blob names the snapshot's symlinks point to.
fn collect_blobs(dir: &Path, blobs: &mut BTreeSet<String>) -> io::Result<()> {
    let entries = match fs::read_dir(dir) {
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(()),
        other => other?,
    };
    for entry in entries {
        let entry = entry?;
        let kind = entry.file_type()?;
        if kind.is_dir() {
            collect_blobs(&entry.path(), blobs)?;
        } else if kind.is_symlink()
            && let Some(name) = fs::read_link(entry.path())?.file_name()
        {
            blobs.insert(name.to_string_lossy().into_owned());
        }
    }
    Ok(())
}

/// Removes refs (e.g. `main`, `pr/1`) whose content is `revision`.
fn remove_refs(
    dir: &Path,
    prefix: &str,
    revision: &str,
    removed: &mut Vec<String>,
) -> io::Result<()> {
    let Ok(entries) = fs::read_dir(dir) else {
        return Ok(());
    };
    for entry in entries {
        let entry = entry?;
        let name = format!("{prefix}{}", entry.file_name().to_string_lossy());
        if entry.file_type()?.is_dir() {
            remove_refs(&entry.path(), &format!("{name}/"), revision, removed)?;
        } else if fs::read_to_string(entry.path())?.trim() == revision {
            fs::remove_file(entry.path())?;
            removed.push(name);
        }
    }
    Ok(())
}

pub fn run(args: RmArgs) -> u8 {
    let ctx = Ctx::from_env();
    super::exit((|| {
        let manifest = hub::find_manifest(&ctx.paths, &args.model)?;
        eprintln!(
            "warning: the Hugging Face cache is shared; other tools using {} at revision {} lose it",
            manifest.repo,
            short(&manifest.revision)
        );
        let question = format!(
            "remove '{}' ({}@{}, {})?",
            manifest.name(),
            manifest.repo,
            short(&manifest.revision),
            human_bytes(manifest.size())
        );
        if !super::confirm(&question, args.yes)? {
            eprintln!("nothing removed");
            return Ok(());
        }
        let removed = remove(&ctx.paths, &manifest)?;
        let kept = if removed.kept_shared > 0 {
            format!("; kept {} files another revision uses", removed.kept_shared)
        } else {
            String::new()
        };
        eprintln!(
            "removed '{}': {} freed{kept}",
            manifest.name(),
            human_bytes(removed.bytes)
        );
        Ok(())
    })())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hub::test_support;

    #[test]
    fn removes_one_revision_and_keeps_blobs_and_refs_of_others() {
        let root = tempfile::tempdir().unwrap();
        let ctx = test_support::ctx(root.path(), "http://unused");
        let (old, new) = ("a".repeat(40), "b".repeat(40));
        // Another tool's revision shares config.json with ours.
        let other = test_support::cached_model(
            &ctx.paths,
            Some("other"),
            "org/m",
            &new,
            &[
                ("config.json", b"{}"),
                ("model.safetensors", b"new weights"),
            ],
        );
        fs::remove_file(other.path(&ctx.paths)).unwrap();
        let ours = test_support::cached_model(
            &ctx.paths,
            Some("ours"),
            "org/m",
            &old,
            &[
                ("config.json", b"{}"),
                ("model.safetensors", b"old weights"),
            ],
        );
        let repo = ctx.paths.repo_dir("org/m");
        fs::create_dir_all(repo.join("refs/pr")).unwrap();
        fs::write(repo.join("refs/main"), &new).unwrap();
        fs::write(repo.join("refs/pr/1"), &old).unwrap();

        let removed = remove(&ctx.paths, &ours).unwrap();
        assert_eq!(removed.blobs, 1);
        assert_eq!(removed.bytes, b"old weights".len() as u64);
        assert_eq!(removed.kept_shared, 1);
        assert_eq!(removed.refs, ["pr/1"]);
        assert!(!repo.join("snapshots").join(&old).exists());
        assert!(!ours.path(&ctx.paths).exists());
        // The other revision still loads.
        let other_dir = repo.join("snapshots").join(&new);
        assert_eq!(fs::read(other_dir.join("config.json")).unwrap(), b"{}");
        assert_eq!(
            fs::read(other_dir.join("model.safetensors")).unwrap(),
            b"new weights"
        );
        assert_eq!(fs::read_to_string(repo.join("refs/main")).unwrap(), new);
    }

    #[test]
    fn removing_the_last_revision_removes_the_repository() {
        let root = tempfile::tempdir().unwrap();
        let ctx = test_support::ctx(root.path(), "http://unused");
        let rev = "a".repeat(40);
        let manifest = test_support::cached_model(
            &ctx.paths,
            Some("m"),
            "org/m",
            &rev,
            &[("config.json", b"{}")],
        );
        let repo = ctx.paths.repo_dir("org/m");
        fs::create_dir_all(repo.join("refs")).unwrap();
        fs::write(repo.join("refs/main"), &rev).unwrap();
        fs::write(repo.join("blobs/partial.incomplete"), b"x").unwrap();
        let removed = remove(&ctx.paths, &manifest).unwrap();
        assert_eq!(removed.refs, ["main"]);
        assert!(!repo.exists());
        assert!(hub::manifests(&ctx.paths).is_empty());
    }
}
