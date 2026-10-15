//! `mrun pull`: hub metadata, the architecture gate (`config.json` before any weight, FR-014),
//! the size/license/fit preview, a disk-space check, resumable downloads into the HF cache
//! layout, verification and the manifest.
//!
//! Plain HTTP against the hub: `GET {endpoint}/api/models/{repo}/revision/{rev}?blobs=true` for
//! the commit, file list, sizes and hashes, and `GET {endpoint}/{repo}/resolve/{commit}/{file}`
//! for content. A file downloads into `blobs/<hash>.incomplete` (resumed with `Range`), is
//! verified (sha256 for LFS files, the git blob id otherwise) and renamed to `blobs/<hash>`;
//! `snapshots/<commit>/<file>` links to it, as `huggingface_hub` does.

use super::verify::{self, git_sha1_file, sha256_file};
use super::{
    Ctx, Hub, Manifest, ManifestFile, Quantization, TARGET_48GB, aliases, fit, human_bytes,
    manifests, not_found, short, split_revision,
};
use crate::errors::{CliError, Code};
use mrun_engine::identity;
use mrun_engine::models::{FamilyConfig, LoadError, SyntaxId};
use reqwest::{Response, StatusCode, header};
use serde::Deserialize;
use serde_json::Value;
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};

/// Quantization modes the engine loads.
const QUANT_MODES: [&str; 2] = ["affine", "mxfp4"];

/// Asks the user; `Err` when nobody can answer (no terminal and no `--yes`).
pub type Confirm<'a> = &'a mut dyn FnMut(&str) -> Result<bool, CliError>;

#[derive(Debug, Deserialize)]
struct RepoInfo {
    sha: String,
    #[serde(default)]
    siblings: Vec<Sibling>,
    #[serde(default, rename = "cardData")]
    card_data: Value,
    #[serde(default)]
    tags: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct Sibling {
    rfilename: String,
    size: Option<u64>,
    #[serde(rename = "blobId")]
    blob_id: Option<String>,
    lfs: Option<Lfs>,
}

#[derive(Debug, Deserialize)]
struct Lfs {
    sha256: String,
    size: u64,
}

/// A file to fetch: its blob name is its hash (LFS sha256 or git blob id).
#[derive(Debug)]
struct RemoteFile {
    path: String,
    size: u64,
    blob: String,
    lfs: bool,
}

/// Files a model needs, top level only (the identity function reads the same set).
fn wanted(path: &str) -> bool {
    !path.contains('/')
        && (path.ends_with(".safetensors")
            || path.ends_with(".json")
            || path.ends_with(".jinja")
            || path.ends_with(".tiktoken")
            || path == "tokenizer.model")
}

struct Remote<'a> {
    client: reqwest::Client,
    hub: &'a Hub,
    repo: &'a str,
}

impl Remote<'_> {
    /// GET with the token; `from > 0` asks for a range. 200 and 206 are returned.
    async fn get(&self, url: &str, from: u64) -> Result<Response, CliError> {
        let mut request = self.client.get(url);
        if let Some(token) = &self.hub.token {
            request = request.bearer_auth(token);
        }
        if from > 0 {
            request = request.header(header::RANGE, format!("bytes={from}-"));
        }
        let response = request.send().await.map_err(|error| {
            CliError::new(
                Code::Network,
                format!("cannot reach {}: {error}", self.hub.endpoint),
            )
            .hint("check the connection and run the same 'mrun pull' again; HF_ENDPOINT selects a mirror")
        })?;
        let status = response.status();
        if status.is_success() {
            return Ok(response);
        }
        let terms = format!("{}/{}", self.hub.endpoint, self.repo);
        Err(match status {
            StatusCode::UNAUTHORIZED | StatusCode::FORBIDDEN if self.hub.token.is_none() => {
                CliError::new(
                    Code::Gated,
                    format!("{} needs a Hugging Face token ({status})", self.repo),
                )
                .hint(format!(
                    "set HF_TOKEN or run 'hf auth login', and accept the terms at {terms}"
                ))
            }
            StatusCode::UNAUTHORIZED | StatusCode::FORBIDDEN => CliError::new(
                Code::Gated,
                format!(
                    "the Hugging Face token has no access to {} ({status})",
                    self.repo
                ),
            )
            .hint(format!("accept the terms at {terms}, or check HF_TOKEN")),
            StatusCode::NOT_FOUND => CliError::new(
                Code::ModelNotFound,
                format!("{url} was not found on {}", self.hub.endpoint),
            )
            .hint("check the repo id and revision (org/name@revision)"),
            _ => CliError::new(Code::Network, format!("{url} returned {status}"))
                .hint("try again later; HF_ENDPOINT selects a mirror"),
        })
    }

    fn resolve_url(&self, commit: &str, path: &str) -> String {
        format!(
            "{}/{}/resolve/{commit}/{path}",
            self.hub.endpoint, self.repo
        )
    }
}

/// `mrun pull <alias|org/name[@rev]>`. Returns the written manifest.
pub async fn pull(ctx: &Ctx, target: &str, confirm: Confirm<'_>) -> Result<Manifest, CliError> {
    if ctx.hub.offline {
        return Err(CliError::new(
            Code::Network,
            "HF_HUB_OFFLINE is set, and pull needs the network",
        )
        .hint("unset HF_HUB_OFFLINE to download; 'mrun ls' lists the models already here"));
    }
    let alias = aliases::get(target);
    let (repo, revision) = match alias {
        Some(alias) => (alias.repo, alias.revision),
        None => match split_revision(target) {
            (repo, rev) if is_repo_id(repo) => (repo, rev.unwrap_or("main")),
            _ => {
                return Err(not_found(target, &manifests(&ctx.paths)));
            }
        },
    };
    let remote = Remote {
        client: reqwest::Client::builder()
            .user_agent(concat!("mrun/", env!("CARGO_PKG_VERSION")))
            .build()
            .map_err(|error| CliError::new(Code::Internal, error.to_string()))?,
        hub: &ctx.hub,
        repo,
    };

    // Metadata, then config.json: nothing else is fetched before the gate.
    let api = format!(
        "{}/api/models/{repo}/revision/{revision}?blobs=true",
        ctx.hub.endpoint
    );
    let info: RepoInfo = remote.get(&api, 0).await?.json().await.map_err(|error| {
        CliError::new(Code::Network, format!("invalid reply from {api}: {error}"))
    })?;
    let commit = info.sha.as_str();
    let files = remote_files(&info, repo)?;
    let config = remote
        .get(&remote.resolve_url(commit, "config.json"), 0)
        .await?
        .bytes()
        .await
        .map_err(|error| CliError::new(Code::Network, format!("config.json of {repo}: {error}")))?;
    let config: Value = serde_json::from_slice(&config).map_err(|error| {
        gate(format!(
            "config.json of {repo} is not valid JSON: {error} (nothing downloaded)"
        ))
    })?;
    let family = FamilyConfig::from_config(&config).map_err(|error| match error {
        LoadError::UnsupportedArchitecture { .. } => gate(error.report(repo).replacen(
            '\n',
            " (nothing downloaded)\n",
            1,
        )),
        other => gate(format!(
            "cannot use config.json of {repo}: {other} (nothing downloaded)"
        )),
    })?;
    let quantization = quantization(&config);
    if let Some(mode) = quantization.as_ref().and_then(|q| q.mode.as_deref())
        && !QUANT_MODES.contains(&mode)
    {
        return Err(gate(format!(
            "unsupported quantization mode '{mode}' in {repo} (nothing downloaded)\n  supported modes: {}",
            QUANT_MODES.join(", ")
        )));
    }
    if let Some(alias) = alias
        && alias.family != family.family.name
    {
        return Err(gate(format!(
            "alias '{}' expects family {} but {repo} is {} (nothing downloaded)",
            alias.name, alias.family, family.family.name
        )));
    }

    // Preview: size, license, fit.
    let size: u64 = files.iter().map(|file| file.size).sum();
    let weights: u64 = files
        .iter()
        .filter(|file| file.path.ends_with(".safetensors"))
        .map(|file| file.size)
        .sum();
    let model_info = family.model_info(String::new(), Vec::new());
    let license = license(&info);
    let syntax = syntax_name(family.family.syntax);
    eprintln!(
        "{repo}@{} ({}, {syntax})\n  size {}, license {license}",
        short(commit),
        family.family.name,
        human_bytes(size)
    );
    if let Some(device) = ctx.device {
        let fit = fit(device, weights, &model_info);
        if fit.fits {
            eprintln!(
                "  fits this machine: up to {} tokens of context",
                fit.max_context
            );
        } else {
            let question = format!(
                "{repo} does not fit this machine's memory budget of {}; download anyway?",
                human_bytes(fit.budget)
            );
            let refused = |hint: &str| {
                CliError::new(
                    Code::DoesNotFit,
                    format!(
                        "{repo} does not fit the memory budget of {} (nothing downloaded)",
                        human_bytes(fit.budget)
                    ),
                )
                .hint(hint.to_string())
            };
            match confirm(&question) {
                Ok(true) => {}
                Ok(false) => return Err(refused("pick a smaller model ('mrun pull qwen3-coder')")),
                Err(_) => return Err(refused("pass --yes to download it anyway")),
            }
        }
    }

    // Disk space for what is missing (+5%), counting resumable partial files.
    let repo_dir = ctx.paths.repo_dir(repo);
    let blobs = repo_dir.join("blobs");
    fs::create_dir_all(&blobs).map_err(|error| write_error(error, &blobs))?;
    let mut missing = 0;
    let mut partial = 0;
    for file in &files {
        if !is_file_of_size(&blobs.join(&file.blob), file.size) {
            let have = fs::metadata(incomplete(&blobs, file))
                .map(|meta| meta.len())
                .unwrap_or(0)
                .min(file.size);
            partial += have;
            missing += file.size - have;
        }
    }
    if missing > 0 {
        let needed = missing + missing / 20;
        if let Some(free) = ctx.free_space.or_else(|| free_space(&blobs))
            && free < needed
        {
            let kept = if partial > 0 {
                format!("; partial downloads kept: {}", human_bytes(partial))
            } else {
                String::new()
            };
            return Err(CliError::new(
                Code::DiskFull,
                format!(
                    "not enough disk space in {}: {} needed, {} available{kept}",
                    ctx.paths.hf_cache.display(),
                    human_bytes(needed),
                    human_bytes(free)
                ),
            )
            .hint("free disk space, or set HF_HUB_CACHE to a larger disk, and run the same 'mrun pull' again"));
        }
    }

    // Download (or reuse) and verify every file, then link the snapshot.
    let snapshot = repo_dir.join("snapshots").join(commit);
    for file in &files {
        let blob = blobs.join(&file.blob);
        if is_file_of_size(&blob, file.size) && !blob_matches(&blob, file) {
            eprintln!(
                "  {} does not match its hash; downloading it again",
                file.path
            );
            fs::remove_file(&blob).map_err(|error| write_error(error, &blob))?;
        }
        if !blob.exists() {
            fetch(&remote, commit, file, &blobs, target).await?;
        }
        if let Some(alias) = alias
            && let Some((_, pinned)) = alias.sha256.iter().find(|(path, _)| *path == file.path)
            && *pinned != file.blob
        {
            return Err(CliError::new(
                Code::ModelCorrupt,
                format!(
                    "{} of '{}' differs from the pinned hash",
                    file.path, alias.name
                ),
            )
            .hint("the hub changed the pinned revision; update mrun or pull the repo id"));
        }
        let link = snapshot.join(&file.path);
        link_blob(&link, &file.blob, &file.path).map_err(|error| write_error(error, &link))?;
    }
    if revision != commit {
        let reference = repo_dir.join("refs").join(revision);
        let written = reference
            .parent()
            .map_or(Ok(()), fs::create_dir_all)
            .and_then(|()| fs::write(&reference, commit));
        written.map_err(|error| write_error(error, &reference))?;
    }

    // Manifest.
    let mut entries = Vec::with_capacity(files.len());
    for file in &files {
        let blob = blobs.join(&file.blob);
        let meta = fs::metadata(&blob).map_err(|error| write_error(error, &blob))?;
        let sha256 = if file.lfs {
            file.blob.clone()
        } else {
            sha256_file(&blob).map_err(|error| write_error(error, &blob))?
        };
        entries.push(ManifestFile {
            path: file.path.clone(),
            size: file.size,
            sha256,
            mtime_ns: verify::mtime_ns(&meta),
        });
    }
    let identity = identity::identify(&snapshot, &ctx.paths.identity_cache())
        .map_err(|error| CliError::new(Code::Internal, error.to_string()))?;
    let alias_name = alias.map(|alias| alias.name.to_string()).or_else(|| {
        // A repo pulled by id after its alias keeps the alias.
        manifests(&ctx.paths)
            .into_iter()
            .find(|m| m.repo == repo && m.alias.is_some())
            .and_then(|m| m.alias)
    });
    let manifest = Manifest {
        alias: alias_name,
        repo: repo.to_string(),
        revision: commit.to_string(),
        license,
        family: family.family.name.to_string(),
        syntax: syntax.to_string(),
        architecture: config["model_type"]
            .as_str()
            .unwrap_or_default()
            .to_string(),
        text_architecture: config["text_config"]["model_type"]
            .as_str()
            .map(str::to_string),
        state_bytes_per_token: family.kv_bytes_per_token,
        checkpoint_bytes: family.recurrent_bytes,
        max_position: family.max_position,
        capacity_48gb: fit(TARGET_48GB, weights, &model_info).max_context,
        identity: identity.value,
        quantization,
        files: entries,
    };
    manifest.save(&ctx.paths)?;
    eprintln!(
        "verified {} files; '{}' is ready in {}",
        manifest.files.len(),
        manifest.name(),
        snapshot.display()
    );
    Ok(manifest)
}

fn gate(message: String) -> CliError {
    CliError::new(Code::UnsupportedArchitecture, message)
}

fn is_repo_id(repo: &str) -> bool {
    let mut parts = repo.split('/');
    matches!((parts.next(), parts.next(), parts.next()), (Some(org), Some(name), None) if !org.is_empty() && !name.is_empty())
}

fn remote_files(info: &RepoInfo, repo: &str) -> Result<Vec<RemoteFile>, CliError> {
    let mut files: Vec<RemoteFile> = info
        .siblings
        .iter()
        .filter(|sibling| wanted(&sibling.rfilename))
        .filter_map(|sibling| {
            let (blob, size, lfs) = match &sibling.lfs {
                Some(lfs) => (lfs.sha256.clone(), lfs.size, true),
                None => (sibling.blob_id.clone()?, sibling.size?, false),
            };
            Some(RemoteFile {
                path: sibling.rfilename.clone(),
                size,
                blob: blob.to_ascii_lowercase(),
                lfs,
            })
        })
        .collect();
    // config.json first; the gate already read it.
    files.sort_by_key(|file| file.path != "config.json");
    if files.first().is_none_or(|file| file.path != "config.json") {
        return Err(gate(format!(
            "{repo} has no config.json (nothing downloaded)"
        )));
    }
    if !files.iter().any(|file| file.path.ends_with(".safetensors")) {
        return Err(gate(format!(
            "{repo} has no *.safetensors weights (nothing downloaded)"
        )));
    }
    Ok(files)
}

fn quantization(config: &Value) -> Option<Quantization> {
    let q = ["quantization", "quantization_config"]
        .iter()
        .map(|key| &config[key])
        .find(|value| value.is_object())?;
    let number = |key: &str| q[key].as_u64().and_then(|n| u32::try_from(n).ok());
    Some(Quantization {
        bits: number("bits"),
        group_size: number("group_size"),
        mode: q["mode"].as_str().map(str::to_string),
    })
}

fn license(info: &RepoInfo) -> String {
    info.card_data["license"]
        .as_str()
        .map(str::to_string)
        .or_else(|| {
            info.tags
                .iter()
                .find_map(|tag| tag.strip_prefix("license:").map(str::to_string))
        })
        .unwrap_or_else(|| "unknown".into())
}

pub fn syntax_name(syntax: SyntaxId) -> &'static str {
    match syntax {
        SyntaxId::Hermes => "hermes",
        SyntaxId::QwenXml => "qwen_xml",
        SyntaxId::Harmony => "harmony",
    }
}

fn incomplete(blobs: &Path, file: &RemoteFile) -> PathBuf {
    blobs.join(format!("{}.incomplete", file.blob))
}

fn is_file_of_size(path: &Path, size: u64) -> bool {
    fs::metadata(path).is_ok_and(|meta| meta.is_file() && meta.len() == size)
}

fn blob_matches(path: &Path, file: &RemoteFile) -> bool {
    let hash = if file.lfs {
        sha256_file(path)
    } else {
        git_sha1_file(path)
    };
    hash.is_ok_and(|hash| hash == file.blob)
}

/// Downloads one file into `blobs/<hash>.incomplete`, resuming what is there, verifies it and
/// renames it into place. Partial files are kept on any failure except a hash mismatch.
async fn fetch(
    remote: &Remote<'_>,
    commit: &str,
    file: &RemoteFile,
    blobs: &Path,
    target: &str,
) -> Result<(), CliError> {
    let partial = incomplete(blobs, file);
    let mut offset = fs::metadata(&partial).map(|meta| meta.len()).unwrap_or(0);
    if offset > file.size {
        offset = 0;
    }
    if offset < file.size {
        let mut response = remote
            .get(&remote.resolve_url(commit, &file.path), offset)
            .await?;
        let resume = offset > 0 && response.status() == StatusCode::PARTIAL_CONTENT;
        if !resume {
            offset = 0;
        }
        eprintln!(
            "  {} {} ({})",
            if resume { "resuming" } else { "downloading" },
            file.path,
            human_bytes(file.size)
        );
        let mut options = fs::OpenOptions::new();
        if resume {
            options.append(true);
        } else {
            options.create(true).write(true).truncate(true);
        }
        let mut out = options
            .open(&partial)
            .map_err(|error| write_error(error, &partial))?;
        let interrupted = |offset: u64, reason: String| {
            CliError::new(
                Code::Network,
                format!(
                    "download of {} interrupted after {} of {}: {reason}",
                    file.path,
                    human_bytes(offset),
                    human_bytes(file.size)
                ),
            )
            .hint(format!("run 'mrun pull {target}' again to resume"))
        };
        loop {
            match response.chunk().await {
                Ok(Some(chunk)) => {
                    out.write_all(&chunk)
                        .map_err(|error| write_error(error, &partial))?;
                    offset += chunk.len() as u64;
                }
                Ok(None) => break,
                Err(error) => return Err(interrupted(offset, error.to_string())),
            }
        }
        out.flush().map_err(|error| write_error(error, &partial))?;
        if offset != file.size {
            return Err(interrupted(offset, "the connection closed early".into()));
        }
    }
    if !blob_matches(&partial, file) {
        let _ = fs::remove_file(&partial);
        return Err(CliError::new(
            Code::Network,
            format!(
                "{} does not match its hash; the download was removed",
                file.path
            ),
        )
        .hint(format!("run 'mrun pull {target}' again")));
    }
    let blob = blobs.join(&file.blob);
    fs::rename(&partial, &blob).map_err(|error| write_error(error, &blob))
}

/// A write failure: a full disk is E052 (partial files are kept for resume).
fn write_error(error: io::Error, path: &Path) -> CliError {
    if error.kind() == io::ErrorKind::StorageFull {
        let kept = fs::metadata(path).map(|meta| meta.len()).unwrap_or(0);
        return CliError::new(
            Code::DiskFull,
            format!(
                "the disk is full while writing {} ({} kept for resume)",
                path.display(),
                human_bytes(kept)
            ),
        )
        .hint("free disk space and run the same 'mrun pull' again");
    }
    CliError::new(
        Code::Internal,
        format!("cannot write {}: {error}", path.display()),
    )
    .hint("check permissions of the Hugging Face cache (HF_HUB_CACHE)")
}

/// `snapshots/<commit>/<path>` → `../../blobs/<blob>` (relative, as `huggingface_hub` links).
pub fn link_blob(link: &Path, blob: &str, path: &str) -> io::Result<()> {
    if let Some(parent) = link.parent() {
        fs::create_dir_all(parent)?;
    }
    if fs::symlink_metadata(link).is_ok() {
        fs::remove_file(link)?;
    }
    let target = format!(
        "{}../../blobs/{blob}",
        "../".repeat(path.matches('/').count())
    );
    #[cfg(unix)]
    {
        std::os::unix::fs::symlink(target, link)
    }
    #[cfg(not(unix))]
    {
        fs::copy(link.parent().unwrap_or(link).join(target), link).map(drop)
    }
}

/// Free bytes on the disk holding `dir` (`df -Pk`), if it can be read.
fn free_space(dir: &Path) -> Option<u64> {
    let out = std::process::Command::new("df")
        .arg("-Pk")
        .arg(dir)
        .output()
        .ok()?;
    let text = String::from_utf8_lossy(&out.stdout);
    let kib: u64 = text
        .lines()
        .nth(1)?
        .split_whitespace()
        .nth(3)?
        .parse()
        .ok()?;
    kib.checked_mul(1024)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wanted_files_are_top_level_model_files() {
        for path in [
            "config.json",
            "model-00001-of-00002.safetensors",
            "model.safetensors.index.json",
            "chat_template.jinja",
            "tokenizer.model",
        ] {
            assert!(wanted(path), "{path}");
        }
        for path in ["README.md", "original/model.safetensors", ".gitattributes"] {
            assert!(!wanted(path), "{path}");
        }
    }

    #[test]
    fn repo_ids_have_one_slash() {
        assert!(is_repo_id("org/name"));
        assert!(!is_repo_id("name"));
        assert!(!is_repo_id("org/"));
        assert!(!is_repo_id("a/b/c"));
    }

    #[test]
    fn free_space_reads_df() {
        assert!(free_space(Path::new(".")).is_some_and(|free| free > 0));
    }
}
