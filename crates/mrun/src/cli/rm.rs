//! `mrun rm <model> [--yes]` (FR-003): deletes the model's directory in mrun's store and its
//! manifest (022: one manifest owns one directory). The Hugging Face cache is never touched.
//! Removing the model the running server serves needs a confirmation of its own.

use crate::cli::server;
use crate::errors::{CliError, Code};
use crate::hub::{self, Ctx, Manifest, Paths, human_bytes, short};
use std::fs;
use std::io;

#[derive(Debug, Clone, clap::Args)]
pub struct RmArgs {
    /// Alias or repo id (org/name[@revision]) of a pulled model.
    pub model: String,
    /// Remove without asking.
    #[arg(long, short)]
    pub yes: bool,
}

/// Deletes the manifest's model directory and the manifest; returns the bytes freed.
pub fn remove(paths: &Paths, manifest: &Manifest) -> Result<u64, CliError> {
    // The revision and repo are joined into a path below; a manifest that could point outside
    // the store is refused (manifests are validated on load too).
    manifest.validate().map_err(|problem| {
        CliError::new(
            Code::ModelCorrupt,
            format!("refusing to remove {}: {problem}", manifest.name()),
        )
        .hint("delete the manifest in MRUN_HOME/manifests by hand")
    })?;
    let dir = manifest.dir(paths);
    let failed = |error: io::Error| {
        CliError::new(
            Code::Internal,
            format!("cannot remove {}: {error}", dir.display()),
        )
        .hint("check permissions of mrun's model store (MRUN_HOME, ~/.mrun)")
    };
    let freed = fs::read_dir(&dir)
        .map(|entries| {
            entries
                .flatten()
                .filter_map(|entry| entry.metadata().ok())
                .map(|meta| meta.len())
                .sum()
        })
        .unwrap_or(0);
    match fs::remove_dir_all(&dir) {
        Err(error) if error.kind() != io::ErrorKind::NotFound => return Err(failed(error)),
        _ => {}
    }
    // The repository's directory goes with its last revision.
    if let Some(repo) = dir.parent()
        && fs::read_dir(repo).is_ok_and(|mut entries| entries.next().is_none())
    {
        let _ = fs::remove_dir(repo);
    }
    match fs::remove_file(manifest.path(paths)) {
        Err(error) if error.kind() != io::ErrorKind::NotFound => return Err(failed(error)),
        _ => {}
    }
    Ok(freed)
}

/// Whether the server on `port` has `manifest`'s model loaded (same content identity).
fn served(port: u16, manifest: &Manifest) -> bool {
    server::request(port, "GET", "/mrun/status").is_ok_and(|(_, status)| {
        status["model"]["identity"].as_str() == Some(manifest.identity.as_str())
    })
}

pub fn run(args: RmArgs) -> u8 {
    let ctx = Ctx::from_env();
    super::exit((|| {
        let manifest = hub::find_manifest(&ctx.paths, &args.model)?;
        let port = server::default_port();
        if served(port, &manifest) {
            eprintln!(
                "warning: the server on port {port} serves '{}'; it keeps running but cannot load it again",
                manifest.name()
            );
            if !super::confirm("remove it anyway?", args.yes)? {
                eprintln!("nothing removed; 'mrun stop' stops the server first");
                return Ok(());
            }
        }
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
        let freed = remove(&ctx.paths, &manifest)?;
        eprintln!(
            "removed '{}': {} freed",
            manifest.name(),
            human_bytes(freed)
        );
        Ok(())
    })())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hub::test_support;

    #[test]
    fn removes_the_model_and_keeps_other_revisions_and_the_hf_cache() {
        let root = tempfile::tempdir().unwrap();
        let ctx = test_support::ctx(root.path(), "http://unused");
        let (old, new) = ("a".repeat(40), "b".repeat(40));
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
        // Another tool's copy in the Hugging Face cache.
        let cached = ctx.paths.repo_dir("org/m").join("snapshots").join(&old);
        fs::create_dir_all(&cached).unwrap();
        fs::write(cached.join("config.json"), b"{}").unwrap();

        let freed = remove(&ctx.paths, &ours).unwrap();
        assert_eq!(freed, (b"{}".len() + b"old weights".len()) as u64);
        assert!(!ours.dir(&ctx.paths).exists());
        assert!(!ours.path(&ctx.paths).exists());
        assert_eq!(
            fs::read(other.dir(&ctx.paths).join("model.safetensors")).unwrap(),
            b"new weights"
        );
        assert!(other.path(&ctx.paths).exists());
        assert_eq!(fs::read(cached.join("config.json")).unwrap(), b"{}");

        // The last revision takes the repository's directory with it.
        remove(&ctx.paths, &other).unwrap();
        assert!(!ctx.paths.models().join("org--m").exists());
        assert!(hub::manifests(&ctx.paths).is_empty());
    }

    #[test]
    fn a_manifest_pointing_outside_the_store_is_refused() {
        let root = tempfile::tempdir().unwrap();
        let ctx = test_support::ctx(root.path(), "http://unused");
        let mut manifest = test_support::cached_model(
            &ctx.paths,
            Some("m"),
            "org/m",
            &"a".repeat(40),
            &[("config.json", b"{}")],
        );
        let victim = root.path().join("victim");
        fs::create_dir_all(&victim).unwrap();
        manifest.revision = "../../../../victim".into();
        let error = remove(&ctx.paths, &manifest).unwrap_err();
        assert_eq!(error.code, Code::ModelCorrupt, "{error}");
        assert!(victim.exists());
        // The same manifest on disk is skipped when manifests are loaded.
        fs::write(
            manifest.path(&ctx.paths),
            toml::to_string(&manifest).unwrap(),
        )
        .unwrap();
        assert!(hub::find_manifest(&ctx.paths, "m").is_err());
    }
}
