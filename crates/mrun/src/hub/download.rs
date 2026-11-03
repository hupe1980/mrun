//! `mrun pull`: hub metadata, pinned hashes and the architecture gate (`config.json` before any
//! weight, FR-014), the size/license/fit preview, a disk-space check, resumable downloads into
//! the HF cache layout, verification and the manifest.
//!
//! Plain HTTP against the hub: `GET {endpoint}/api/models/{repo}/revision/{rev}?blobs=true` for
//! the commit, file list, sizes and hashes, and `GET {endpoint}/{repo}/resolve/{commit}/{file}`
//! for content. A file downloads into `blobs/<hash>.incomplete` (resumed with `Range`, retried
//! with backoff), is verified (sha256 for LFS files, the git blob id otherwise) and renamed to
//! `blobs/<hash>`; `snapshots/<commit>/<file>` links to it, and `.locks/` guards each blob, as
//! `huggingface_hub` does.

use super::aliases::{self, Alias};
use super::progress::Report;
use super::verify::{self, git_sha1_file, sha256_file};
use super::{
    Ctx, Fit, Hub, Manifest, ManifestFile, Quantization, TARGET_48GB, fit, human_bytes, is_blob_id,
    is_commit, is_file_name, is_repo_id, is_revision, manifests, short, split_revision,
};
use crate::errors::{CliError, Code};
use mrun_engine::identity;
use mrun_engine::models::{FamilyConfig, LoadError, SyntaxId};
use reqwest::{Response, StatusCode, header};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::time::Duration;

/// Quantization modes the engine loads.
const QUANT_MODES: [&str; 2] = ["affine", "mxfp4"];
const CONNECT_TIMEOUT: Duration = Duration::from_secs(20);
/// Longest wait for the next chunk of a download.
const READ_TIMEOUT: Duration = Duration::from_secs(60);
/// Resumed attempts after a failed download, waiting 1, 2, 4, 8 and 16 s.
const RETRIES: u32 = 5;

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

impl<'a> Remote<'a> {
    fn new(hub: &'a Hub, repo: &'a str) -> Result<Self, CliError> {
        let client = reqwest::Client::builder()
            .user_agent(concat!("mrun/", env!("CARGO_PKG_VERSION")))
            .connect_timeout(CONNECT_TIMEOUT)
            .read_timeout(READ_TIMEOUT)
            .build()
            .map_err(|error| CliError::new(Code::Internal, error.to_string()))?;
        Ok(Self { client, hub, repo })
    }

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
        if response.status().is_success() {
            return Ok(response);
        }
        let error_code = response
            .headers()
            .get("x-error-code")
            .and_then(|value| value.to_str().ok())
            .map(str::to_string);
        Err(self.status_error(response.status(), error_code.as_deref(), url))
    }

    /// The hub's `X-Error-Code` names the cause. A 401 without one is a missing or private repo
    /// (the hub does not tell them apart), as `huggingface_hub` reads it; a gated repo says
    /// `GatedRepo`.
    fn status_error(&self, status: StatusCode, error_code: Option<&str>, url: &str) -> CliError {
        let repo = self.repo;
        let terms = format!("{}/{repo}", self.hub.endpoint);
        let needs_token = || match &self.hub.token {
            None => CliError::new(
                Code::Gated,
                format!("{repo} needs a Hugging Face token ({status})"),
            )
            .hint(format!(
                "set HF_TOKEN or run 'hf auth login', and accept the terms at {terms}"
            )),
            Some(_) => CliError::new(
                Code::Gated,
                format!("the Hugging Face token has no access to {repo} ({status})"),
            )
            .hint(format!("accept the terms at {terms}, or check HF_TOKEN")),
        };
        match (error_code, status) {
            (Some("RepositoryNotFound"), _) | (None, StatusCode::UNAUTHORIZED) => CliError::new(
                Code::ModelNotFound,
                format!(
                    "{repo} does not exist on {} (or it is private)",
                    self.hub.endpoint
                ),
            )
            .hint("check the repo id (org/name); a private repo needs HF_TOKEN or 'hf auth login'"),
            (Some("RevisionNotFound"), _) => {
                CliError::new(Code::ModelNotFound, format!("{repo} has no such revision"))
                    .hint("check the revision (org/name@revision)")
            }
            (Some("EntryNotFound"), _) => {
                CliError::new(Code::ModelNotFound, format!("{url} does not exist"))
                    .hint("the revision lacks a file the model needs; pick another revision")
            }
            (Some("GatedRepo"), _) | (_, StatusCode::UNAUTHORIZED | StatusCode::FORBIDDEN) => {
                needs_token()
            }
            (_, StatusCode::NOT_FOUND) => CliError::new(
                Code::ModelNotFound,
                format!("{url} was not found on {}", self.hub.endpoint),
            )
            .hint("check the repo id and revision (org/name@revision)"),
            _ => CliError::new(Code::Network, format!("{url} returned {status}"))
                .hint("try again later; HF_ENDPOINT selects a mirror"),
        }
    }

    fn resolve_url(&self, commit: &str, path: &str) -> String {
        format!(
            "{}/{}/resolve/{commit}/{path}",
            self.hub.endpoint, self.repo
        )
    }
}

/// What `pull` downloads, read from the hub; the architecture gate and the pins have passed.
#[derive(Debug)]
pub struct Plan {
    /// What the user typed, for hints.
    pub target: String,
    /// The alias named, or the alias pinned to this commit.
    pub alias: Option<&'static Alias>,
    pub repo: String,
    /// The revision asked for (a branch, tag or commit).
    pub revision: String,
    pub commit: String,
    pub license: String,
    files: Vec<RemoteFile>,
    family: FamilyConfig,
    config: Value,
    quantization: Option<Quantization>,
}

impl Plan {
    pub fn size(&self) -> u64 {
        self.files.iter().map(|file| file.size).sum()
    }

    pub fn weights(&self) -> u64 {
        self.files
            .iter()
            .filter(|file| file.path.ends_with(".safetensors"))
            .map(|file| file.size)
            .sum()
    }

    /// Weight bytes the engine loads: an alias's pinned number (vision tower excluded), else
    /// the file sizes.
    pub fn loaded_weights(&self) -> u64 {
        self.alias
            .map_or_else(|| self.weights(), |alias| alias.loaded_weights())
    }

    pub fn model_info(&self) -> mrun_engine::ModelInfo {
        self.family.model_info(String::new(), Vec::new())
    }

    pub fn fit(&self, device: mrun_engine::budget::DeviceMemory) -> Fit {
        fit(device, self.loaded_weights(), &self.model_info())
    }

    /// `org/name@abc1234 (family, syntax)` and the size and license.
    pub fn summary(&self) -> String {
        format!(
            "{}@{} ({}, {})\n  size {}, license {}",
            self.repo,
            short(&self.commit),
            self.family.family.name,
            syntax_name(self.family.family.syntax),
            human_bytes(self.size()),
            self.license
        )
    }
}

/// `mrun pull <alias|org/name[@rev]>`: the plan, the preview, a confirmation when the model does
/// not fit, the download. Returns the written manifest.
pub async fn pull(ctx: &Ctx, target: &str, confirm: Confirm<'_>) -> Result<Manifest, CliError> {
    pull_with(ctx, target, confirm, &Report::Human).await
}

/// [`pull`] reporting to `report`: human lines and bars, or the 018 JSON lines (`plan`,
/// `progress`, `done`; the caller prints `confirm` and `error`).
pub async fn pull_with(
    ctx: &Ctx,
    target: &str,
    confirm: Confirm<'_>,
    report: &Report,
) -> Result<Manifest, CliError> {
    let plan = plan(ctx, target).await?;
    let fit = announce(ctx, &plan, report);
    let done = |manifest: &Manifest, already: bool| {
        report.emit(serde_json::json!({
            "event": "done",
            "model": manifest.name(),
            "path": manifest.snapshot_dir(&ctx.paths),
            "already_pulled": already,
        }));
    };
    if let Some(manifest) = already_pulled(ctx, &plan) {
        report.note(&format!(
            "already pulled: {} is intact — run 'mrun launch'",
            manifest.name()
        ));
        done(&manifest, true);
        return Ok(manifest);
    }
    if let Some((fit, _)) = fit
        && !fit.fits
    {
        let smaller = aliases::default_alias(ctx.device).name;
        let refused = |hint: String| {
            CliError::new(
                Code::DoesNotFit,
                format!(
                    "{} does not fit the memory budget of {} (nothing downloaded)",
                    plan.repo,
                    human_bytes(fit.budget)
                ),
            )
            .hint(hint)
        };
        let question = format!(
            "{} does not fit this machine's memory budget of {}; download anyway?",
            plan.repo,
            human_bytes(fit.budget)
        );
        match confirm(&question) {
            Ok(true) => {}
            Ok(false) => {
                return Err(refused(format!(
                    "pick a model that fits ('mrun pull {smaller}')"
                )));
            }
            Err(_) => return Err(refused("pass --yes to download it anyway".into())),
        }
    }
    let manifest = download_with(ctx, &plan, report).await?;
    report.note(&format!(
        "ready: {} ({} files verified) — run 'mrun launch'",
        manifest.name(),
        manifest.files.len()
    ));
    done(&manifest, false);
    Ok(manifest)
}

/// `mrun pull --dry-run` (020 FR-005): the plan line or `plan` event, then nothing else: no
/// confirmation, no download, nothing written, whether or not the model fits.
pub async fn dry_run(ctx: &Ctx, target: &str, report: &Report) -> Result<Plan, CliError> {
    let plan = plan(ctx, target).await?;
    announce(ctx, &plan, report);
    Ok(plan)
}

/// Reports the plan (the line, or the `plan` event) and returns its fit.
fn announce(ctx: &Ctx, plan: &Plan, report: &Report) -> Option<(Fit, bool)> {
    let fit = ctx
        .device
        .map(|device| (plan.fit(device), device.estimated));
    report.note(&plan_line(plan, fit));
    report.emit(serde_json::json!({
        "event": "plan",
        "model": plan.alias.map_or(plan.repo.as_str(), |alias| alias.name),
        "repo": plan.repo,
        "revision": plan.commit,
        "files": plan.files.len(),
        "bytes": plan.size(),
        "fit": fit.map(|(fit, _)| FitClass::of(&fit)),
        "verdict": verdict(fit),
        "family": plan.family.family.name,
    }));
    fit
}

/// The CLI's fit classes (D048): `tight` fits with less than
/// [`budget::DEFAULT_ALIAS_MIN_TOKENS`](mrun_engine::budget::DEFAULT_ALIAS_MIN_TOKENS) tokens of
/// context, too little for the coding agents.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum FitClass {
    Fits,
    Tight,
    WontFit,
}

impl FitClass {
    pub fn of(fit: &Fit) -> Self {
        if !fit.fits {
            Self::WontFit
        } else if fit.max_context < mrun_engine::budget::DEFAULT_ALIAS_MIN_TOKENS {
            Self::Tight
        } else {
            Self::Fits
        }
    }

    /// The word of `ls`'s FITS column.
    pub fn word(self) -> &'static str {
        match self {
            Self::Fits => "yes",
            Self::Tight => "tight",
            Self::WontFit => "no",
        }
    }
}

/// `n` with its thousands grouped by commas (`221,819`), as the page shows counts (019 R4).
pub fn group(n: u64) -> String {
    let digits = n.to_string();
    let mut out = String::with_capacity(digits.len() + digits.len() / 3);
    for (i, digit) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(digit);
    }
    out
}

/// The fit part of the plan line: `fits with 262,144 tokens`, `does not fit the budget of 36.0
/// GB`, ` (estimated)` appended; `fit unknown` without memory figures. `fit` carries whether
/// the device's figures are estimated.
pub fn fit_phrase(fit: Option<(Fit, bool)>) -> String {
    let Some((fit, estimated)) = fit else {
        return "fit unknown".into();
    };
    let estimated = if estimated { " (estimated)" } else { "" };
    if fit.fits {
        format!(
            "fits with {} tokens{estimated}",
            group(u64::from(fit.max_context))
        )
    } else {
        format!(
            "does not fit the budget of {}{estimated}",
            human_bytes(fit.budget)
        )
    }
}

/// The fit as one plain sentence (`ls --json`'s `verdict`, `pull --json`'s plan): the plan
/// line's [`fit_phrase`], saying when the context is tight.
pub fn verdict(fit: Option<(Fit, bool)>) -> String {
    let phrase = fit_phrase(fit);
    let mut chars = phrase.chars();
    let mut sentence: String = chars
        .next()
        .map(|first| first.to_uppercase().chain(chars).collect())
        .unwrap_or_default();
    if let Some((fit, _)) = fit
        && FitClass::of(&fit) == FitClass::Tight
    {
        sentence.push_str(&format!(
            ", less than the {} coding agents need",
            group(u64::from(mrun_engine::budget::DEFAULT_ALIAS_MIN_TOKENS))
        ));
    }
    sentence.push('.');
    sentence
}

/// The plan line (FR-001): `qwen3.5 → org/name@abc1234, 6.0 GB, license apache-2.0, fits with
/// 262,144 tokens`.
pub fn plan_line(plan: &Plan, fit: Option<(Fit, bool)>) -> String {
    let name = match plan.alias {
        Some(alias) => format!("{} → ", alias.name),
        None => String::new(),
    };
    format!(
        "{name}{}@{}, {}, license {}, {}",
        plan.repo,
        short(&plan.commit),
        human_bytes(plan.size()),
        plan.license,
        fit_phrase(fit)
    )
}

/// The manifest of this exact pull (same name, repo and commit) when every file is present
/// and passes the fast check: nothing to download.
fn already_pulled(ctx: &Ctx, plan: &Plan) -> Option<Manifest> {
    let name = plan.alias.map(|alias| alias.name);
    let mut manifest = manifests(&ctx.paths)
        .into_iter()
        .find(|m| m.repo == plan.repo && m.revision == plan.commit && m.alias_name() == name)?;
    let complete = plan
        .files
        .iter()
        .all(|file| manifest.files.iter().any(|known| known.path == file.path));
    (complete && super::check(&ctx.paths, &mut manifest) == verify::Status::Ready)
        .then_some(manifest)
}

/// Metadata and `config.json` of `target`; nothing else is fetched. Refuses unsupported
/// architectures and quantizations, and an alias whose pins the hub no longer matches.
pub async fn plan(ctx: &Ctx, target: &str) -> Result<Plan, CliError> {
    if ctx.hub.offline {
        return Err(CliError::new(
            Code::Network,
            "HF_HUB_OFFLINE is set, and pull needs the network",
        )
        .hint("unset HF_HUB_OFFLINE to download; 'mrun ls' lists the models already here"));
    }
    let named = aliases::get(target);
    let (repo, revision) = match named {
        Some(alias) => (alias.repo, alias.revision),
        None => match split_revision(target) {
            (repo, Some(rev)) if is_repo_id(repo) && !is_revision(rev) => {
                return Err(CliError::new(
                    Code::InvalidArguments,
                    format!("'{rev}' is not a valid revision"),
                )
                .hint("name a branch, tag or commit without '/' or '..' (org/name@main)"));
            }
            (repo, rev) if is_repo_id(repo) => (repo, rev.unwrap_or("main")),
            _ => return Err(not_a_model(target, &manifests(&ctx.paths))),
        },
    };
    let remote = Remote::new(&ctx.hub, repo)?;
    let api = format!(
        "{}/api/models/{repo}/revision/{revision}?blobs=true",
        ctx.hub.endpoint
    );
    let info: RepoInfo = remote.get(&api, 0).await?.json().await.map_err(|error| {
        CliError::new(Code::Network, format!("invalid reply from {api}: {error}"))
    })?;
    if !is_commit(&info.sha) {
        return Err(invalid_reply(
            &api,
            "a commit id that is not 40 lowercase hex",
        ));
    }
    let files = remote_files(&info, repo)?;
    let alias = named.or_else(|| aliases::pinned(repo, &info.sha));
    if let Some(alias) = alias {
        check_pins(alias, &info.sha, &files)?;
    }

    let config = remote
        .get(&remote.resolve_url(&info.sha, "config.json"), 0)
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
    Ok(Plan {
        target: target.to_string(),
        alias,
        repo: repo.to_string(),
        revision: revision.to_string(),
        commit: info.sha.clone(),
        license: license(&info),
        files,
        family,
        config,
        quantization,
    })
}

/// The hub's commit and blob ids against the alias' pins, before anything is downloaded.
fn check_pins(alias: &Alias, commit: &str, files: &[RemoteFile]) -> Result<(), CliError> {
    let changed = |what: String| {
        CliError::new(
            Code::ModelCorrupt,
            format!(
                "{what} of '{}' differs from the pinned one (nothing downloaded)",
                alias.name
            ),
        )
        .hint(format!(
            "the hub changed {}; update mrun, or pull the repo id",
            alias.repo
        ))
    };
    if commit != alias.revision {
        return Err(changed(format!("the commit {}", short(commit))));
    }
    if let Some(file) = files
        .iter()
        .find(|file| alias.blob(&file.path) != Some(file.blob.as_str()))
    {
        return Err(changed(file.path.clone()));
    }
    if files.len() != alias.blobs.len() {
        return Err(changed("the file list".into()));
    }
    Ok(())
}

/// Downloads (or reuses) and verifies every file of `plan`, links the snapshot and writes the
/// manifest.
pub async fn download(ctx: &Ctx, plan: &Plan) -> Result<Manifest, CliError> {
    download_with(ctx, plan, &Report::Human).await
}

/// [`download`] reporting progress to `report`.
pub async fn download_with(ctx: &Ctx, plan: &Plan, report: &Report) -> Result<Manifest, CliError> {
    let repo = plan.repo.as_str();
    let remote = Remote::new(&ctx.hub, repo)?;
    let repo_dir = ctx.paths.repo_dir(repo);
    let blobs = repo_dir.join("blobs");
    fs::create_dir_all(&blobs).map_err(|error| write_error(error, &blobs))?;
    check_disk(ctx, plan, &blobs)?;

    let recorded = manifests(&ctx.paths);
    let snapshot = repo_dir.join("snapshots").join(&plan.commit);
    for file in &plan.files {
        let lock = ctx.paths.blob_lock(repo, &file.blob);
        let _lock = lock_blob(&lock, &file.path).await?;
        let blob = blobs.join(&file.blob);
        if is_file_of_size(&blob, file.size)
            && needs_hash(file, &blob, &recorded)
            && !blob_matches(&blob, file)
        {
            eprintln!(
                "  {} does not match its hash; downloading it again",
                file.path
            );
            fs::remove_file(&blob).map_err(|error| write_error(error, &blob))?;
        }
        if !blob.exists() {
            fetch_with_retries(&remote, plan, file, &blobs, report).await?;
        }
        let link = snapshot.join(&file.path);
        link_blob(&link, &file.blob, &file.path).map_err(|error| write_error(error, &link))?;
    }
    // Branches and tags get a ref; a commit (or its prefix) is not a name.
    if !plan.commit.starts_with(&plan.revision) {
        let reference = repo_dir.join("refs").join(&plan.revision);
        let written = reference
            .parent()
            .map_or(Ok(()), fs::create_dir_all)
            .and_then(|()| fs::write(&reference, &plan.commit));
        written.map_err(|error| write_error(error, &reference))?;
    }
    let manifest = manifest(ctx, plan, &blobs, &snapshot)?;
    manifest.save(&ctx.paths)?;
    // One manifest per revision: an earlier one under another name (repo id vs alias) goes.
    for other in recorded {
        if other.repo == manifest.repo
            && other.revision == manifest.revision
            && other.path(&ctx.paths) != manifest.path(&ctx.paths)
        {
            let _ = fs::remove_file(other.path(&ctx.paths));
        }
    }
    Ok(manifest)
}

/// Disk space for what is missing (+5%), counting resumable partial files.
fn check_disk(ctx: &Ctx, plan: &Plan, blobs: &Path) -> Result<(), CliError> {
    let mut missing = 0;
    let mut partial = 0;
    for file in &plan.files {
        if !is_file_of_size(&blobs.join(&file.blob), file.size) {
            let have = fs::metadata(incomplete(blobs, file))
                .map(|meta| meta.len())
                .unwrap_or(0)
                .min(file.size);
            partial += have;
            missing += file.size - have;
        }
    }
    if missing == 0 {
        return Ok(());
    }
    let needed = missing + missing / 20;
    match ctx.free_space.or_else(|| free_space(blobs)) {
        Some(free) if free < needed => {
            let kept = if partial > 0 {
                format!("; partial downloads kept: {}", human_bytes(partial))
            } else {
                String::new()
            };
            Err(CliError::new(
                Code::DiskFull,
                format!(
                    "not enough disk space in {}: {} needed, {} available{kept}",
                    ctx.paths.hf_cache.display(),
                    human_bytes(needed),
                    human_bytes(free)
                ),
            )
            .hint("free disk space, or set HF_HUB_CACHE to a larger disk, and run the same 'mrun pull' again"))
        }
        _ => Ok(()),
    }
}

/// Whether a blob of the right size must be hashed before it is trusted: small git files always;
/// LFS blobs only when a manifest recorded them with another modification time (someone wrote
/// to them). Blobs other tools downloaded are trusted by size, as `huggingface_hub` does.
fn needs_hash(file: &RemoteFile, blob: &Path, recorded: &[Manifest]) -> bool {
    if !file.lfs {
        return true;
    }
    let mtime = fs::metadata(blob).map_or(0, |meta| verify::mtime_ns(&meta));
    recorded
        .iter()
        .flat_map(|manifest| &manifest.files)
        .any(|known| known.sha256 == file.blob && known.mtime_ns != mtime)
}

fn manifest(ctx: &Ctx, plan: &Plan, blobs: &Path, snapshot: &Path) -> Result<Manifest, CliError> {
    let mut entries = Vec::with_capacity(plan.files.len());
    for file in &plan.files {
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
    let identity = identity::identify(snapshot, &ctx.paths.identity_cache())
        .map_err(|error| CliError::new(Code::Internal, error.to_string()))?;
    let family = &plan.family;
    let model_info = family.model_info(String::new(), Vec::new());
    Ok(Manifest {
        alias: plan.alias.map(|alias| alias.name.to_string()),
        repo: plan.repo.clone(),
        revision: plan.commit.clone(),
        license: plan.license.clone(),
        family: family.family.name.to_string(),
        syntax: syntax_name(family.family.syntax).to_string(),
        architecture: plan.config["model_type"]
            .as_str()
            .unwrap_or_default()
            .to_string(),
        text_architecture: plan.config["text_config"]["model_type"]
            .as_str()
            .map(str::to_string),
        state_bytes_per_token: family.kv_bytes_per_token,
        checkpoint_bytes: family.recurrent_bytes,
        max_position: family.max_position,
        capacity_48gb: fit(TARGET_48GB, plan.loaded_weights(), &model_info).max_context,
        identity: identity.value,
        quantization: plan.quantization.clone(),
        files: entries,
    })
}

fn gate(message: String) -> CliError {
    CliError::new(Code::UnsupportedArchitecture, message)
}

/// Hub metadata that would be unsafe to join into a cache path; nothing is downloaded.
fn invalid_reply(api: &str, what: &str) -> CliError {
    CliError::new(
        Code::Network,
        format!("{api} returned {what} (nothing downloaded)"),
    )
    .hint("the hub or the mirror in HF_ENDPOINT sent malformed metadata; unset HF_ENDPOINT or try again later")
}

/// E031 for a name that is neither an alias nor a repo id, with a did-you-mean.
fn not_a_model(target: &str, local: &[Manifest]) -> CliError {
    let mut error = super::not_found(target, local);
    error.message = format!("'{target}' is not an alias or an org/name repo id");
    error
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
                blob,
                lfs,
            })
        })
        .collect();
    // Blob ids name files in `blobs/` and `.locks/`: only hashes are accepted.
    if let Some(file) = files
        .iter()
        .find(|file| !is_blob_id(&file.blob) || !is_file_name(&file.path))
    {
        return Err(invalid_reply(
            &format!("the file list of {repo}"),
            &format!("an invalid blob id or name for '{}'", file.path),
        ));
    }
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

/// Takes `huggingface_hub`'s lock for a blob, waiting while another download holds it.
async fn lock_blob(lock: &Path, path: &str) -> Result<fs::File, CliError> {
    if let Some(dir) = lock.parent() {
        fs::create_dir_all(dir).map_err(|error| write_error(error, dir))?;
    }
    let file = fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .write(true)
        .open(lock)
        .map_err(|error| write_error(error, lock))?;
    let mut announced = false;
    loop {
        match file.try_lock() {
            Ok(()) => return Ok(file),
            Err(fs::TryLockError::WouldBlock) => {
                if !announced {
                    eprintln!("  waiting for another download of {path}");
                    announced = true;
                }
                tokio::time::sleep(Duration::from_millis(500)).await;
            }
            Err(fs::TryLockError::Error(error)) => return Err(write_error(error, lock)),
        }
    }
}

/// [`fetch`] with up to [`RETRIES`] resumed attempts after network failures.
async fn fetch_with_retries(
    remote: &Remote<'_>,
    plan: &Plan,
    file: &RemoteFile,
    blobs: &Path,
    report: &Report,
) -> Result<(), CliError> {
    let mut attempt = 0;
    loop {
        match fetch(remote, plan, file, blobs, report).await {
            Err(error) if error.code == Code::Network && attempt < RETRIES => {
                let wait = Duration::from_secs(1 << attempt);
                attempt += 1;
                eprintln!(
                    "  {} (retry {attempt}/{RETRIES} in {} s)",
                    error.message,
                    wait.as_secs()
                );
                tokio::time::sleep(wait).await;
            }
            other => return other,
        }
    }
}

/// Downloads one file into `blobs/<hash>.incomplete`, resuming what is there, verifies it and
/// renames it into place. Partial files are kept on any failure except a hash mismatch.
async fn fetch(
    remote: &Remote<'_>,
    plan: &Plan,
    file: &RemoteFile,
    blobs: &Path,
    report: &Report,
) -> Result<(), CliError> {
    let partial = incomplete(blobs, file);
    let mut offset = fs::metadata(&partial).map(|meta| meta.len()).unwrap_or(0);
    if offset > file.size {
        offset = 0;
    }
    let again = || format!("run 'mrun pull {}' again", plan.target);
    if offset < file.size {
        let mut response = remote
            .get(&remote.resolve_url(&plan.commit, &file.path), offset)
            .await?;
        let resume = offset > 0 && response.status() == StatusCode::PARTIAL_CONTENT;
        if !resume {
            offset = 0;
        }
        let verb = if resume { "resuming" } else { "downloading" };
        let progress = report.start(verb, &file.path, file.size, offset);
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
            .hint(format!("{} to resume", again()))
        };
        loop {
            match response.chunk().await {
                Ok(Some(chunk)) => {
                    out.write_all(&chunk)
                        .map_err(|error| write_error(error, &partial))?;
                    offset += chunk.len() as u64;
                    progress.set(offset);
                }
                Ok(None) => break,
                Err(error) => return Err(interrupted(offset, error.to_string())),
            }
        }
        progress.finish();
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
        .hint(again()));
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
pub fn free_space(dir: &Path) -> Option<u64> {
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
    fn counts_group_thousands() {
        let cases = [
            (0, "0"),
            (999, "999"),
            (1000, "1,000"),
            (40960, "40,960"),
            (221_819, "221,819"),
            (1_048_576, "1,048,576"),
        ];
        for (n, text) in cases {
            assert_eq!(group(n), text);
        }
    }

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
        assert!(is_repo_id("mlx-community/Qwen3.5-9B-4bit"));
        assert!(!is_repo_id("name"));
        assert!(!is_repo_id("org/"));
        assert!(!is_repo_id("a/b/c"));
        assert!(!is_repo_id("../name"));
        assert!(!is_repo_id("org/.."));
        assert!(!is_repo_id("org/a--b"));
        assert!(!is_repo_id("org/a b"));
    }

    #[test]
    fn hub_metadata_that_could_escape_the_cache_is_refused() {
        assert!(is_commit(&"a".repeat(40)));
        for bad in [
            "../../../etc",
            &"A".repeat(40),
            &"a".repeat(39),
            &"g".repeat(40),
        ] {
            assert!(!is_commit(bad), "{bad}");
        }
        assert!(is_blob_id(&"0".repeat(40)) && is_blob_id(&"f".repeat(64)));
        for bad in [
            "../x",
            &"a".repeat(41),
            &"F".repeat(64),
            "",
            &format!("{}/", "a".repeat(39)),
        ] {
            assert!(!is_blob_id(bad), "{bad}");
        }
        for good in ["main", "v1.0", "abc1234", "release_2"] {
            assert!(is_revision(good), "{good}");
        }
        for bad in [
            "../main", "a/b", "/main", "..", ".hidden", "-x", "a\\b", "", "a b",
        ] {
            assert!(!is_revision(bad), "{bad}");
        }

        let sibling = |name: &str, blob: &str| Sibling {
            rfilename: name.into(),
            size: Some(1),
            blob_id: Some(blob.into()),
            lfs: None,
        };
        let info = |blob: &str| RepoInfo {
            sha: "a".repeat(40),
            siblings: vec![
                sibling("config.json", &"1".repeat(40)),
                sibling("model.safetensors", blob),
            ],
            card_data: Value::Null,
            tags: Vec::new(),
        };
        assert!(remote_files(&info(&"2".repeat(40)), "org/m").is_ok());
        let error = remote_files(&info("../../../../tmp/x"), "org/m").unwrap_err();
        assert!(error.message.contains("invalid blob id"), "{error}");
        let error = remote_files(&info(&"A".repeat(40)), "org/m").unwrap_err();
        assert!(error.message.contains("model.safetensors"), "{error}");
    }

    #[tokio::test]
    async fn pull_refuses_revisions_that_could_escape_the_cache() {
        let root = tempfile::tempdir().unwrap();
        let ctx = super::super::test_support::ctx(root.path(), "http://127.0.0.1:9");
        for target in ["org/m@../../x", "org/m@a/b", "org/m@/x"] {
            let error = plan(&ctx, target).await.unwrap_err();
            assert_eq!(error.code, Code::InvalidArguments, "{target}: {error}");
        }
        let error = plan(&ctx, "nonsense").await.unwrap_err();
        assert_eq!(error.code, Code::ModelNotFound);
        assert!(
            error
                .message
                .contains("not an alias or an org/name repo id"),
            "{error}"
        );
    }

    #[test]
    fn free_space_reads_df() {
        assert!(free_space(Path::new(".")).is_some_and(|free| free > 0));
    }

    fn remote(path: &str, blob: &str) -> RemoteFile {
        RemoteFile {
            path: path.into(),
            size: 1,
            blob: blob.into(),
            lfs: false,
        }
    }

    #[test]
    fn pins_are_compared_before_anything_is_fetched() {
        let alias = aliases::get("qwen3.5").unwrap();
        let files: Vec<RemoteFile> = alias
            .blobs
            .iter()
            .map(|(path, blob)| remote(path, blob))
            .collect();
        assert!(check_pins(alias, alias.revision, &files).is_ok());

        let error = check_pins(alias, &"f".repeat(40), &files).unwrap_err();
        assert_eq!(error.code, Code::ModelCorrupt);
        assert!(error.message.contains("the commit fffffff"), "{error}");

        let mut changed = files;
        changed[1].blob = "0".repeat(40);
        let error = check_pins(alias, alias.revision, &changed).unwrap_err();
        assert!(error.message.starts_with(&changed[1].path), "{error}");

        changed.truncate(1);
        changed[0] = remote(alias.blobs[0].0, alias.blobs[0].1);
        let error = check_pins(alias, alias.revision, &changed).unwrap_err();
        assert!(error.message.contains("the file list"), "{error}");
    }

    #[test]
    fn hub_error_codes_name_the_cause() {
        let hub = Hub {
            endpoint: "https://hub.example".into(),
            token: None,
            offline: false,
        };
        let remote = Remote::new(&hub, "org/m").unwrap();
        let cases = [
            (Some("RepositoryNotFound"), 401, Code::ModelNotFound),
            (Some("GatedRepo"), 403, Code::Gated),
            (Some("RevisionNotFound"), 404, Code::ModelNotFound),
            (Some("EntryNotFound"), 404, Code::ModelNotFound),
            (None, 401, Code::ModelNotFound),
            (None, 403, Code::Gated),
            (None, 404, Code::ModelNotFound),
            (None, 503, Code::Network),
        ];
        for (error_code, status, expected) in cases {
            let status = StatusCode::from_u16(status).unwrap();
            let error = remote.status_error(status, error_code, "https://hub.example/x");
            assert_eq!(error.code, expected, "{error_code:?} {status}");
        }
        let missing = remote.status_error(
            StatusCode::UNAUTHORIZED,
            Some("RepositoryNotFound"),
            "https://hub.example/x",
        );
        assert!(missing.message.contains("does not exist"), "{missing}");
    }

    #[tokio::test]
    async fn blob_locks_are_exclusive() {
        let root = tempfile::tempdir().unwrap();
        let lock = root.path().join(".locks/models--org--m/abc.lock");
        let held = lock_blob(&lock, "f").await.unwrap();
        let other = fs::File::open(&lock).unwrap();
        assert!(matches!(
            other.try_lock(),
            Err(fs::TryLockError::WouldBlock)
        ));
        drop(held);
        // A child forked by another test (between fork and exec) can briefly share the
        // descriptor, so the release may take a moment to show.
        let deadline = std::time::Instant::now() + Duration::from_secs(2);
        while other.try_lock().is_err() {
            assert!(
                std::time::Instant::now() < deadline,
                "the lock was not released"
            );
            std::thread::sleep(Duration::from_millis(10));
        }
    }
}
