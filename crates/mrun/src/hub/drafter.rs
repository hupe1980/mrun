//! DFlash drafters pulled with their alias (030): pinned files under
//! `models/<org--name>/<commit>/`, verified against their blob ids, safetensors realigned for
//! mapping. No manifest: a drafter is part of its alias, not a model of its own, so `mrun ls`
//! never lists it.

use std::fs;
use std::path::{Path, PathBuf};

use super::aliases::{Alias, Drafter};
use super::download::{
    Fetch, Remote, RemoteFile, blob_matches, fetch_with_retries, free_space, incomplete, place,
    write_error,
};
use super::human_bytes;
use super::progress::Report;
use super::{Ctx, Paths};
use crate::errors::{CliError, Code};

/// Written once every file is verified.
const COMPLETE: &str = ".complete";

/// Where `drafter`'s files live in the store.
pub fn dir(paths: &Paths, drafter: &Drafter) -> PathBuf {
    paths
        .models()
        .join(drafter.repo.replace('/', "--"))
        .join(drafter.revision)
}

/// The drafter directory of `alias` when its drafter is pulled.
pub fn pulled(paths: &Paths, alias: &Alias) -> Option<PathBuf> {
    let drafter = alias.drafter.as_ref()?;
    let dir = dir(paths, drafter);
    dir.join(COMPLETE).is_file().then_some(dir)
}

fn files(drafter: &Drafter) -> Vec<RemoteFile> {
    drafter
        .files
        .iter()
        .map(|&(path, size, blob)| RemoteFile {
            path: path.to_owned(),
            size,
            blob: blob.to_owned(),
            lfs: blob.len() == 64,
        })
        .collect()
}

/// Pulls `alias`'s drafter, if it has one and it is not here yet; returns its directory.
pub async fn pull(ctx: &Ctx, alias: &Alias, report: &Report) -> Result<Option<PathBuf>, CliError> {
    let Some(drafter) = alias.drafter.as_ref() else {
        return Ok(None);
    };
    if let Some(dir) = pulled(&ctx.paths, alias) {
        return Ok(Some(dir));
    }
    report.note(&format!(
        "Pulling its DFlash drafter {} ({}, license {}): faster replies, the same output.",
        drafter.repo,
        human_bytes(drafter.size()),
        drafter.license
    ));
    let dir = dir(&ctx.paths, drafter);
    fs::create_dir_all(&dir).map_err(|error| write_error(error, &dir))?;
    let files = files(drafter);
    let missing: u64 = files
        .iter()
        .filter(|f| !dir.join(&f.path).is_file())
        .map(|f| f.size)
        .sum();
    if let Some(free) = ctx.free_space.or_else(|| free_space(&dir))
        && free < missing + missing / 20
    {
        return Err(CliError::new(
            Code::DiskFull,
            format!(
                "not enough disk space for the drafter: {} needed, {} available",
                human_bytes(missing + missing / 20),
                human_bytes(free)
            ),
        )
        .hint("free disk space and run the same 'mrun pull' again"));
    }
    let remote = Remote::new(&ctx.hub, drafter.repo)?;
    let at = Fetch {
        commit: drafter.revision,
        target: alias.name,
    };
    for file in &files {
        let target = dir.join(&file.path);
        if target.is_file() {
            continue;
        }
        fetch_with_retries(&remote, at, file, &dir, report).await?;
        place(&incomplete(&dir, file), &target, file, true)?;
    }
    let complete = dir.join(COMPLETE);
    fs::write(&complete, drafter.revision).map_err(|error| write_error(error, &complete))?;
    report.note(&format!("drafter ready: {}", drafter.repo));
    Ok(Some(dir))
}

/// Whether the drafter in `dir` still matches its pins (`mrun doctor`): every file of the
/// expected size, small files by hash.
pub fn intact(dir: &Path, drafter: &Drafter) -> bool {
    files(drafter).iter().all(|f| {
        let path = dir.join(&f.path);
        if f.lfs {
            // Realigned: its original bytes plus at most the pad.
            fs::metadata(&path).is_ok_and(|m| m.len() >= f.size)
        } else {
            blob_matches(&path, f)
        }
    })
}
