//! `mrun verify <model>`: re-hashes every file against the manifest (FR-004). A passing run also
//! refreshes the recorded modification times, so a file that was only touched reads `ready`
//! again.

use crate::errors::CliError;
use crate::hub::{self, Ctx, Paths, verify};

#[derive(Debug, Clone, clap::Args)]
pub struct VerifyArgs {
    /// Alias or repo id (org/name[@revision]) of a pulled model.
    pub model: String,
}

/// Verifies the model; returns the number of files checked.
pub fn verify_model(paths: &Paths, name: &str) -> Result<usize, CliError> {
    let mut manifest = hub::find_manifest(paths, name)?;
    let dir = manifest.snapshot_dir(paths);
    crate::cli::info(&format!(
        "hashing {} files ({})…",
        manifest.files.len(),
        hub::human_bytes(manifest.size())
    ));
    let problems = verify::full_check_with(&manifest, &dir, true);
    if !problems.is_empty() {
        return Err(verify::corrupt_error(&manifest, &problems));
    }
    let before = manifest.files.clone();
    verify::refresh_mtimes(&mut manifest, &dir);
    if manifest.files != before {
        manifest.save(paths)?;
    }
    Ok(manifest.files.len())
}

pub fn run(args: VerifyArgs) -> u8 {
    let ctx = Ctx::from_env();
    super::exit(verify_model(&ctx.paths, &args.model).map(|files| {
        eprintln!("'{}' verified: {files} files match", args.model);
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::errors::Code;
    use crate::hub::test_support;
    use std::fs;

    #[test]
    fn verify_reports_corruption_with_the_repair_command() {
        let root = tempfile::tempdir().unwrap();
        let ctx = test_support::ctx(root.path(), "http://unused");
        let manifest = test_support::cached_model(
            &ctx.paths,
            Some("m"),
            "org/m",
            &"a".repeat(40),
            &[("config.json", b"{}"), ("model.safetensors", b"weights")],
        );
        assert_eq!(verify_model(&ctx.paths, "m").unwrap(), 2);
        let weights = manifest.snapshot_dir(&ctx.paths).join("model.safetensors");
        fs::write(&weights, b"weightz").unwrap();
        let error = verify_model(&ctx.paths, "m").unwrap_err();
        assert_eq!(error.code, Code::ModelCorrupt);
        assert_eq!(error.exit() as u8, 3);
        assert_eq!(
            error.hint.as_deref(),
            Some("run 'mrun pull m' to repair it")
        );
    }
}
