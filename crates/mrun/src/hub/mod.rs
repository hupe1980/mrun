//! mrun's model store and state in `MRUN_HOME`: one manifest per pulled model and its files in
//! `models/<org>--<name>/<commit>/`, safetensors laid out for mapping (022, [`realign`]). The
//! shared Hugging Face hub cache (`models--org--name/{blobs,snapshots,refs}`) is only read: pulls
//! import from it, and an unpulled repo there still loads by name.
//!
//! Resolution order (004 contract): existing path, then a manifest (alias or repo id), then a
//! repo id already in the HF cache.

pub mod aliases;
pub mod download;
pub mod progress;
pub mod realign;
pub mod verify;

use crate::errors::{CliError, Code};
use mrun_engine::ModelInfo;
use mrun_engine::budget::{self, DeviceMemory, Overrides};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};

/// Where models and mrun's state live.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Paths {
    /// `HF_HUB_CACHE`, else `HF_HOME/hub`, else `~/.cache/huggingface/hub`.
    pub hf_cache: PathBuf,
    /// `HF_HOME`, else `$XDG_CACHE_HOME/huggingface`, else `~/.cache/huggingface` (holds the
    /// token of `hf auth login`).
    pub hf_home: PathBuf,
    /// `MRUN_HOME`, else `~/.mrun`.
    pub mrun_home: PathBuf,
}

impl Paths {
    /// From an environment lookup; empty values count as unset. Without `HOME` the account's
    /// home directory is used.
    pub fn from_vars(var: &dyn Fn(&str) -> Option<String>) -> Self {
        let var = |key: &str| {
            var(key)
                .filter(|value| !value.is_empty())
                .map(PathBuf::from)
        };
        let home = var("HOME")
            .or_else(std::env::home_dir)
            .unwrap_or_else(|| PathBuf::from("."));
        let hf_home = var("HF_HOME").unwrap_or_else(|| {
            var("XDG_CACHE_HOME")
                .unwrap_or_else(|| home.join(".cache"))
                .join("huggingface")
        });
        Self {
            hf_cache: var("HF_HUB_CACHE").unwrap_or_else(|| hf_home.join("hub")),
            mrun_home: var("MRUN_HOME").unwrap_or_else(|| home.join(".mrun")),
            hf_home,
        }
    }

    pub fn from_env() -> Self {
        Self::from_vars(&|key| std::env::var(key).ok())
    }

    /// `models--org--name` in the HF cache.
    pub fn repo_dir(&self, repo: &str) -> PathBuf {
        self.hf_cache.join(repo_folder(repo))
    }

    /// mrun's model store: `models/<org>--<name>/<commit>/`.
    pub fn models(&self) -> PathBuf {
        self.mrun_home.join("models")
    }

    pub fn manifests(&self) -> PathBuf {
        self.mrun_home.join("manifests")
    }

    /// Cache of file hashes used by 001's identity function.
    pub fn identity_cache(&self) -> PathBuf {
        self.mrun_home.join("identity.toml")
    }
}

fn repo_folder(repo: &str) -> String {
    format!("models--{}", repo.replace('/', "--"))
}

/// A commit id: 40 lowercase hex digits. Commits come from the hub (or a mirror) and from
/// manifests and are joined into store paths, so nothing else is accepted.
pub fn is_commit(text: &str) -> bool {
    is_lower_hex(text, 40)
}

/// A blob id: a git blob id (40 lowercase hex) or an LFS sha256 (64 lowercase hex).
pub fn is_blob_id(text: &str) -> bool {
    is_lower_hex(text, 40) || is_lower_hex(text, 64)
}

fn is_lower_hex(text: &str, len: usize) -> bool {
    text.len() == len && text.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
}

/// A revision the user may name (`@main`, `@v1.0`, `@abc1234`): it becomes a URL segment and
/// a path component in the HF cache, so `/`, `\`, `..`, control characters and a leading `.` or `-` are
/// refused.
pub fn is_revision(text: &str) -> bool {
    !text.is_empty()
        && text.len() <= 128
        && !text.contains("..")
        && !text.starts_with(['.', '-'])
        && text
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b'-'))
}

/// `org/name` with exactly one slash and hub-safe characters in each part (`[A-Za-z0-9._-]`,
/// no `.` or `..` part, no `--` that would collide in `org--name` folder names).
pub fn is_repo_id(repo: &str) -> bool {
    let part = |text: &str| {
        !text.is_empty()
            && text != "."
            && text != ".."
            && !text.contains("--")
            && text
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b'-'))
    };
    let mut parts = repo.split('/');
    matches!((parts.next(), parts.next(), parts.next()), (Some(org), Some(name), None) if part(org) && part(name))
}

/// A file name in a model directory: one path component, never `.` or `..`.
pub fn is_file_name(text: &str) -> bool {
    !text.is_empty() && text != "." && text != ".." && !text.contains(['/', '\\', '\0'])
}

/// The hub to download from.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Hub {
    /// `HF_ENDPOINT` (a mirror), else `https://huggingface.co`; no trailing slash.
    pub endpoint: String,
    /// `HF_TOKEN`, else the token file saved by `hf auth login` (`$HF_HOME/token`).
    pub token: Option<String>,
    /// `HF_HUB_OFFLINE=1`: `pull` fails fast.
    pub offline: bool,
}

impl Hub {
    pub const DEFAULT_ENDPOINT: &str = "https://huggingface.co";

    /// One line on stderr, once per process, when downloads come from somewhere other than the
    /// Hugging Face hub (`HF_ENDPOINT`), so a mirror set in a profile is never silent.
    pub fn announce_endpoint(&self) {
        static ONCE: std::sync::Once = std::sync::Once::new();
        if self.endpoint != Self::DEFAULT_ENDPOINT {
            ONCE.call_once(|| {
                eprintln!(
                    "note: HF_ENDPOINT is set; models are downloaded from {}",
                    self.endpoint
                );
            });
        }
    }

    pub fn from_vars(var: &dyn Fn(&str) -> Option<String>, paths: &Paths) -> Self {
        let var = |key: &str| var(key).filter(|value| !value.trim().is_empty());
        let token = var("HF_TOKEN").or_else(|| {
            fs::read_to_string(paths.hf_home.join("token"))
                .ok()
                .filter(|token| !token.trim().is_empty())
        });
        Self {
            endpoint: var("HF_ENDPOINT")
                .unwrap_or_else(|| Self::DEFAULT_ENDPOINT.into())
                .trim_end_matches('/')
                .to_string(),
            token: token.map(|token| token.trim().to_string()),
            offline: var("HF_HUB_OFFLINE").is_some_and(|value| {
                matches!(value.to_lowercase().as_str(), "1" | "true" | "yes" | "on")
            }),
        }
    }
}

/// Everything a model command needs from the environment; tests build it directly.
#[derive(Clone, Debug)]
pub struct Ctx {
    pub paths: Paths,
    pub hub: Hub,
    /// This machine's memory, for fit and maximum context; `None` when unknown.
    pub device: Option<DeviceMemory>,
    /// Free bytes on the store's disk; `None` asks the system (`df`).
    pub free_space: Option<u64>,
}

impl Ctx {
    pub fn from_env() -> Self {
        let var = |key: &str| std::env::var(key).ok();
        let paths = Paths::from_vars(&var);
        let hub = Hub::from_vars(&var, &paths);
        hub.announce_endpoint();
        let budget =
            var("MRUN_BUDGET_GIB").and_then(|text| crate::cli::model::parse_budget_gib(&text).ok());
        Self {
            hub,
            paths,
            device: device_memory(),
            free_space: None,
        }
        .with_budget(budget)
    }

    /// The machine as `--budget-gib` makes it look (D060): the working set lowered to `budget`
    /// (never raised), so the default pick and fit follow the lowered budget as the load will.
    /// This is how small Macs are simulated on the qualification Mac.
    pub fn with_budget(mut self, budget: Option<u64>) -> Self {
        if let (Some(device), Some(budget)) = (self.device.as_mut(), budget) {
            device.recommended_working_set = device.recommended_working_set.min(budget);
        }
        self
    }
}

/// This machine's memory. On a Mac with Metal it is the device's own figure, Metal's device info
/// (`recommendedMaxWorkingSetSize`), the same number the engine budgets with, so `pull`, `ls`,
/// `show` and `launch` fit before a load as the load will. Only without a Metal device (Linux, a
/// build without the engine) does it fall back to the system's numbers, marked `estimated`.
pub fn device_memory() -> Option<DeviceMemory> {
    #[cfg(feature = "engine")]
    if let Some((physical, recommended_working_set)) = mrun_engine::tensor::device_memory()
        && recommended_working_set > 0
    {
        return Some(DeviceMemory {
            physical,
            recommended_working_set,
            available: mrun_engine::sysmem::available(),
            estimated: false,
        });
    }
    system_memory()
}

/// `51.5 GB physical, 40.2 GB working set (Metal)`: the memory figures and where they come
/// from, for `show` and `doctor` (FR-004, FR-018).
pub fn describe_memory(device: DeviceMemory) -> String {
    let available = device
        .available
        .map(|bytes| format!(", {} available now", human_bytes(bytes)))
        .unwrap_or_default();
    format!(
        "{} physical, {} working set ({}){available}",
        human_bytes(device.physical),
        human_bytes(device.recommended_working_set),
        memory_source(device)
    )
}

/// Where [`device_memory`]'s working set comes from.
pub fn memory_source(device: DeviceMemory) -> &'static str {
    if device.estimated {
        "estimated: wired limit or 2/3 of physical"
    } else {
        "Metal recommended working set"
    }
}

#[cfg(target_os = "linux")]
fn system_memory() -> Option<DeviceMemory> {
    DeviceMemory::read_linux().ok()
}

#[cfg(target_os = "macos")]
fn system_memory() -> Option<DeviceMemory> {
    let sysctl = |name: &str| -> Option<u64> {
        let out = std::process::Command::new("sysctl")
            .args(["-n", name])
            .output()
            .ok()?;
        String::from_utf8_lossy(&out.stdout).trim().parse().ok()
    };
    let physical = sysctl("hw.memsize")?;
    Some(DeviceMemory {
        physical,
        recommended_working_set: estimated_working_set(
            physical,
            sysctl("iogpu.wired_limit_mb").unwrap_or(0),
        ),
        available: mrun_engine::sysmem::available(),
        estimated: true,
    })
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
fn system_memory() -> Option<DeviceMemory> {
    None
}

/// Share of physical memory (⅔) the fallback assumes Metal recommends. One conservative share for
/// every size (D048: no RAM-size boundaries); Metal's own figure is at least this on the
/// machines measured (67 % on 16 GB, 78 % on 48 GB).
pub const ESTIMATED_WORKING_SET_SHARE: f64 = 2.0 / 3.0;

/// The fallback estimate of Metal's recommended working set, used only when no Metal device
/// answers: `iogpu.wired_limit_mb` when set, else [`ESTIMATED_WORKING_SET_SHARE`] of physical
/// memory. Never above physical memory.
pub fn estimated_working_set(physical: u64, wired_limit_mb: u64) -> u64 {
    if wired_limit_mb > 0 {
        return (wired_limit_mb << 20).min(physical);
    }
    (physical as f64 * ESTIMATED_WORKING_SET_SHARE) as u64
}

/// `quantization` of `config.json`.
#[derive(Clone, Debug, Default, Deserialize, Serialize, Eq, PartialEq)]
pub struct Quantization {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bits: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub group_size: Option<u32>,
    /// `affine` (the default) or `mxfp4`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mode: Option<String>,
}

/// One file of a manifest: content hash plus the fast-check record.
#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
pub struct ManifestFile {
    /// Path relative to the model directory.
    pub path: String,
    pub size: u64,
    pub sha256: String,
    /// Modification time of the file in the store, nanoseconds since the epoch.
    pub mtime_ns: u64,
    /// Spaces added after a safetensors header ([`realign`]); on disk the file is `size + pad`.
    #[serde(default, skip_serializing_if = "is_zero")]
    pub pad: u32,
}

fn is_zero(n: &u32) -> bool {
    *n == 0
}

/// `MRUN_HOME/manifests/<alias or org--name>.toml` (data model, ModelManifest).
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct Manifest {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub alias: Option<String>,
    pub repo: String,
    /// Pinned 40-hex commit.
    pub revision: String,
    pub license: String,
    pub family: String,
    pub syntax: String,
    /// `config.json` `model_type`.
    pub architecture: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text_architecture: Option<String>,
    /// 001 `kv_bytes_per_token`.
    pub state_bytes_per_token: u64,
    /// 001 `recurrent_bytes`; 0 without position-bound state.
    pub checkpoint_bytes: u64,
    pub max_position: u32,
    /// Token capacity on the 48 GB target.
    pub capacity_48gb: u32,
    /// 001 content identity.
    pub identity: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub quantization: Option<Quantization>,
    pub files: Vec<ManifestFile>,
    /// A part (026): only the files of layers `[first, end)` (0-based), for a pool stage.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub part: Option<[usize; 2]>,
}

impl Manifest {
    /// Refuses a manifest whose repo, revision or file names would leave the store when joined
    /// into a path (a tampered manifest, or one written from a hostile mirror's metadata).
    pub fn validate(&self) -> Result<(), String> {
        if !is_repo_id(&self.repo) {
            return Err(format!("invalid repo id '{}'", self.repo));
        }
        if !is_commit(&self.revision) {
            return Err(format!(
                "revision '{}' is not a 40-hex commit",
                self.revision
            ));
        }
        if let Some(alias) = &self.alias
            && !is_file_name(alias)
        {
            return Err(format!("invalid alias '{alias}'"));
        }
        if let Some(file) = self
            .files
            .iter()
            .find(|file| !is_file_name(&file.path) || !is_lower_hex(&file.sha256, 64))
        {
            return Err(format!("invalid file entry '{}'", file.path));
        }
        if let Some(file) = self.files.iter().find(|file| {
            u64::from(file.pad) > realign::MAX_PAD
                || (file.pad > 0 && !file.path.ends_with(".safetensors"))
        }) {
            return Err(format!("invalid pad {} for '{}'", file.pad, file.path));
        }
        Ok(())
    }

    /// The alias: recorded, or the one pinned to this revision (a model pulled by repo id).
    pub fn alias_name(&self) -> Option<&str> {
        self.alias
            .as_deref()
            .or_else(|| aliases::pinned(&self.repo, &self.revision).map(|alias| alias.name))
    }

    /// What users type: the alias, else the repo id.
    pub fn name(&self) -> &str {
        self.alias_name().unwrap_or(&self.repo)
    }

    fn file_stem(&self) -> String {
        self.alias
            .clone()
            .unwrap_or_else(|| self.repo.replace('/', "--"))
    }

    pub fn path(&self, paths: &Paths) -> PathBuf {
        paths.manifests().join(format!("{}.toml", self.file_stem()))
    }

    /// The model's directory in the store.
    pub fn dir(&self, paths: &Paths) -> PathBuf {
        paths
            .models()
            .join(self.repo.replace('/', "--"))
            .join(&self.revision)
    }

    /// Bytes on disk.
    pub fn size(&self) -> u64 {
        self.files
            .iter()
            .map(|file| file.size + u64::from(file.pad))
            .sum()
    }

    pub fn weights(&self) -> u64 {
        self.files
            .iter()
            .filter(|file| file.path.ends_with(".safetensors"))
            .map(|file| file.size)
            .sum()
    }

    /// Weight bytes the engine loads: an alias's pinned number (vision tower excluded), else the
    /// `*.safetensors` sizes. For a model pulled by repo id that carries a vision tower the
    /// engine skips, this overestimates by the tower's size (≈ 0.9 GB for Qwen3.5/3.6).
    pub fn loaded_weights(&self) -> u64 {
        match aliases::pinned(&self.repo, &self.revision) {
            Some(alias) => alias.loaded_weights(),
            None => self.weights(),
        }
    }

    /// Fit on `device` with the default budget.
    pub fn fit(&self, device: DeviceMemory) -> Fit {
        fit(device, self.loaded_weights(), &self.model_info())
    }

    /// The fields 001's budget functions read.
    pub fn model_info(&self) -> ModelInfo {
        model_info(
            &self.family,
            self.state_bytes_per_token,
            self.checkpoint_bytes,
            self.max_position,
        )
    }

    pub fn save(&self, paths: &Paths) -> Result<(), CliError> {
        let path = self.path(paths);
        let text = toml::to_string(self).map_err(|error| internal(&path, error))?;
        fs::create_dir_all(paths.manifests()).map_err(|error| internal(&path, error))?;
        let mut temp =
            tempfile::NamedTempFile::new_in(paths.manifests()).map_err(|e| internal(&path, e))?;
        std::io::Write::write_all(&mut temp, text.as_bytes()).map_err(|e| internal(&path, e))?;
        temp.persist(&path).map_err(|e| internal(&path, e.error))?;
        Ok(())
    }
}

/// A [`ModelInfo`] with only the fields 001's budget functions read.
pub fn model_info(
    family: &str,
    kv_bytes_per_token: u64,
    recurrent_bytes: u64,
    max_position: u32,
) -> ModelInfo {
    ModelInfo {
        identity: String::new(),
        family: family.to_string(),
        layer_kinds: Vec::new(),
        kv_heads: 0,
        head_dim: 0,
        vocab: 0,
        max_position,
        kv_bytes_per_token,
        recurrent_bytes,
        eos_tokens: Vec::new(),
        ..ModelInfo::default()
    }
}

fn internal(path: &Path, error: impl std::fmt::Display) -> CliError {
    CliError::new(
        Code::Internal,
        format!("cannot write {}: {error}", path.display()),
    )
    .hint("check that the directory is writable (MRUN_HOME sets mrun's state directory)")
}

/// Every manifest, sorted by name; unreadable ones are skipped with a warning.
pub fn manifests(paths: &Paths) -> Vec<Manifest> {
    let Ok(entries) = fs::read_dir(paths.manifests()) else {
        return Vec::new();
    };
    let mut all: Vec<Manifest> = entries
        .filter_map(|entry| {
            let path = entry.ok()?.path();
            if path.extension()? != "toml" {
                return None;
            }
            let parsed = fs::read_to_string(&path)
                .map_err(|error| error.to_string())
                .and_then(|text| {
                    toml::from_str::<Manifest>(&text).map_err(|error| error.to_string())
                })
                .and_then(|manifest| manifest.validate().map(|()| manifest));
            match parsed {
                Ok(manifest) => Some(manifest),
                Err(error) => {
                    eprintln!("warning: skipping {}: {error}", path.display());
                    None
                }
            }
        })
        .collect();
    all.sort_by(|a, b| a.name().cmp(b.name()));
    all
}

/// The manifest named by `name`: an alias, a repo id or `repo@revision` (a prefix of the commit).
pub fn find_manifest(paths: &Paths, name: &str) -> Result<Manifest, CliError> {
    let all = manifests(paths);
    let (repo, revision) = split_revision(name);
    // A manifest of an alias' pinned revision is that alias, even when pulled by repo id.
    let is_alias = |m: &Manifest| m.alias_name() == Some(name);
    let mut matches: Vec<&Manifest> = all
        .iter()
        .filter(|m| {
            is_alias(m)
                || (m.repo == repo && revision.is_none_or(|rev| m.revision.starts_with(rev)))
        })
        .collect();
    if matches.iter().any(|m| is_alias(m)) {
        matches.retain(|m| is_alias(m));
    }
    match matches.as_slice() {
        [one] => Ok((*one).clone()),
        [] => Err(not_found(name, &all)),
        many => Err(CliError::new(
            Code::ModelAmbiguous,
            format!(
                "'{name}' matches several models: {}",
                many.iter()
                    .map(|m| format!("{}@{}", m.repo, short(&m.revision)))
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
        )
        .hint("name one with org/name@revision")),
    }
}

/// `org/name@rev` → (`org/name`, `rev`).
pub fn split_revision(name: &str) -> (&str, Option<&str>) {
    match name.split_once('@') {
        Some((repo, rev)) if !rev.is_empty() => (repo, Some(rev)),
        _ => (name, None),
    }
}

/// E031 with a did-you-mean among local models and curated aliases.
pub fn not_found(name: &str, local: &[Manifest]) -> CliError {
    let known = local
        .iter()
        .map(|m| m.name().to_string())
        .chain(aliases::ALIASES.iter().map(|a| a.name.to_string()));
    let suggestion = known
        .map(|candidate| (edit_distance(name, &candidate), candidate))
        .filter(|(distance, _)| *distance <= 2)
        .min();
    let error = CliError::new(Code::ModelNotFound, format!("model '{name}' not found"));
    match (suggestion, aliases::get(name)) {
        (_, Some(alias)) => error.hint(format!("run 'mrun pull {}' first", alias.name)),
        (Some((_, candidate)), None) => error.hint(format!(
            "did you mean '{candidate}'? Run 'mrun ls' to see local models."
        )),
        (None, None) => error.hint("run 'mrun ls' to see local models, or 'mrun pull <org/name>'"),
    }
}

fn edit_distance(a: &str, b: &str) -> usize {
    let b: Vec<char> = b.chars().collect();
    let mut row: Vec<usize> = (0..=b.len()).collect();
    for (i, ca) in a.chars().enumerate() {
        let mut previous = row[0];
        row[0] = i + 1;
        for (j, cb) in b.iter().enumerate() {
            let current = row[j + 1];
            row[j + 1] = (previous + usize::from(ca != *cb))
                .min(row[j] + 1)
                .min(current + 1);
            previous = current;
        }
    }
    row[b.len()]
}

/// First 7 characters of a commit.
pub fn short(revision: &str) -> &str {
    &revision[..revision.len().min(7)]
}

/// A model ready to load.
#[derive(Clone, Debug)]
pub struct LocalModel {
    pub dir: PathBuf,
    /// Present for pulled models; their fast checks passed.
    pub manifest: Option<Manifest>,
}

/// Resolves `name` for loading: an existing path, then a manifest (fast-checked; a mismatch is
/// refused with the repair command), then a repo id already in the HF cache. A path that has the
/// name of a pulled model or an alias is ambiguous.
pub fn resolve(paths: &Paths, name: &str) -> Result<LocalModel, CliError> {
    let manifest = match find_manifest(paths, name) {
        Ok(manifest) => Some(manifest),
        Err(error) if error.code == Code::ModelNotFound => None,
        Err(error) => return Err(error),
    };
    let path = Path::new(name);
    if path.exists() {
        if manifest.is_some() || aliases::get(name).is_some() {
            return Err(shadowed(name));
        }
        return Ok(LocalModel {
            dir: path.to_path_buf(),
            manifest: None,
        });
    }
    if let Some(mut manifest) = manifest {
        if let verify::Status::Corrupt(problems) = check(paths, &mut manifest) {
            return Err(verify::corrupt_error(&manifest, &problems));
        }
        return Ok(LocalModel {
            dir: manifest.dir(paths),
            manifest: Some(manifest),
        });
    }
    let (repo, revision) = split_revision(name);
    if !is_repo_id(repo) || revision.is_some_and(|rev| !is_revision(rev)) {
        return Err(not_found(name, &manifests(paths)));
    }
    let repo_dir = paths.repo_dir(repo);
    let revision = match revision {
        Some(rev) => rev.to_string(),
        None => fs::read_to_string(repo_dir.join("refs/main")).unwrap_or_default(),
    };
    let revision = revision.trim();
    let dir = repo_dir.join("snapshots").join(revision);
    if is_revision(revision) && dir.join("config.json").exists() {
        return Ok(LocalModel {
            dir,
            manifest: None,
        });
    }
    Err(not_found(name, &manifests(paths)))
}

fn shadowed(name: &str) -> CliError {
    CliError::new(
        Code::ModelAmbiguous,
        format!("'{name}' is both a directory here and a model name"),
    )
    .hint(format!(
        "write ./{name} for the directory, or run from another directory for the model"
    ))
}

/// Fast checks of a pulled model. A file that was only touched is re-hashed; when its content
/// still matches, the new modification time is recorded so the next check is fast again.
pub fn check(paths: &Paths, manifest: &mut Manifest) -> verify::Status {
    let dir = manifest.dir(paths);
    let before = manifest.files.clone();
    let status = verify::fast_check(manifest, &dir);
    if manifest.files != before
        && let Err(error) = manifest.save(paths)
    {
        eprintln!("warning: {error}");
    }
    status
}

/// The model to serve when none is named: the per-machine default alias when it is pulled, else
/// the largest pulled model that fits, else the default alias (which then needs a pull). E040
/// when memory is known, no alias leaves [`aliases::DEFAULT_MIN_CONTEXT`] tokens of context and
/// no pulled model fits. Memory other processes hold does not change the pick (D062): the
/// load refuses with E041 instead, naming what to quit or the smaller model.
pub fn default_model(ctx: &Ctx) -> Result<String, CliError> {
    let Some(device) = ctx.device.map(DeviceMemory::ignoring_available) else {
        return Ok(aliases::ALIASES[0].name.to_string());
    };
    let alias = aliases::machine_alias(device);
    let pulled = manifests(&ctx.paths);
    if let Some(alias) = alias
        && pulled.iter().any(|m| m.alias_name() == Some(alias.name))
    {
        return Ok(alias.name.to_string());
    }
    let fitting = pulled
        .iter()
        .filter(|m| m.fit(device).fits)
        .max_by_key(|m| m.weights())
        .map(|m| m.name().to_string());
    if let Some(name) = fitting.or_else(|| alias.map(|a| a.name.to_string())) {
        return Ok(name);
    }
    let smallest = &aliases::ALIASES[0];
    let fit = smallest.fit(device);
    Err(CliError::new(
        Code::DoesNotFit,
        format!(
            "no model leaves {} tokens of context on this Mac: {} leaves {} in a budget of {}",
            aliases::DEFAULT_MIN_CONTEXT,
            smallest.name,
            fit.max_context,
            human_bytes(fit.budget)
        ),
    )
    .hint("pass --model with a smaller model; a Mac with more memory leaves more context"))
}

/// Memory fit on a device (001 budget functions, the family's scratch estimate).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Fit {
    pub budget: u64,
    pub weights: u64,
    pub scratch: u64,
    pub fits: bool,
    /// Tokens of context that fit; 0 when the model does not fit.
    pub max_context: u32,
}

/// Fit with the default budget (the recommended working set, D048).
pub fn fit(device: DeviceMemory, weights: u64, info: &ModelInfo) -> Fit {
    fit_with(device, None, weights, info)
}

/// The scratch the hub plans a model with before it is loaded: the family's alias estimate,
/// fixed plus per position, solved against the capacity by the engine (D048).
fn scratch_model(family: &str) -> budget::LinearScratch {
    budget::LinearScratch {
        fixed: aliases::scratch_estimate(family),
        per_position: aliases::scratch_per_position(family),
    }
}

/// Fit with an optional budget override in bytes (`--budget-gib`; it can only lower the
/// budget), through the engine's capacity function (D048).
pub fn fit_with(device: DeviceMemory, budget: Option<u64>, weights: u64, info: &ModelInfo) -> Fit {
    let scratch = scratch_model(&info.family);
    let mut fit = Fit {
        budget: 0,
        weights,
        scratch: scratch.fixed,
        fits: false,
        max_context: 0,
    };
    let overrides = Overrides { budget };
    if let Ok(plan) = budget::plan(device, overrides) {
        fit.budget = plan.budget;
        if let Ok(full) = budget::capacity_on(device, overrides, weights, scratch, info) {
            fit.fits = true;
            fit.scratch = full.scratch;
            fit.max_context = full.kv_capacity_tokens;
        }
    }
    fit
}

/// The budget a model needs for `tokens` of context: weights, the family's scratch estimate,
/// the position-bound state with its reserved checkpoints and the attention state (the inverse
/// of [`fit`]).
pub fn budget_needed(tokens: u32, weights: u64, info: &ModelInfo) -> u64 {
    weights
        + scratch_model(&info.family).at(tokens)
        + budget::position_bound_reserve(info.recurrent_bytes)
        + u64::from(tokens) * info.kv_bytes_per_token
}

/// D062: refuses (E041) before any weight is read when the model's weights, scratch and state
/// for [`budget::DEFAULT_ALIAS_MIN_TOKENS`] (at most its maximum position) exceed the available
/// memory less the margin, naming the largest other processes. Unknown available memory (not
/// macOS) passes: the budget then follows D048.
pub fn check_available(
    device: DeviceMemory,
    name: &str,
    weights: u64,
    info: &ModelInfo,
) -> Result<(), CliError> {
    let Some(room) = device.available_room() else {
        return Ok(());
    };
    let need = budget_needed(floor_tokens(info), weights, info);
    if need <= room {
        return Ok(());
    }
    let others = mrun_engine::sysmem::processes();
    Err(memory_short(
        name,
        need,
        device,
        &others[..others.len().min(3)],
    ))
}

fn floor_tokens(info: &ModelInfo) -> u32 {
    budget::DEFAULT_ALIAS_MIN_TOKENS.min(info.max_position)
}

/// The E041 refusal: what `name` needs, what is available, the largest other processes and
/// the largest alias that would fit now.
pub fn memory_short(
    name: &str,
    need: u64,
    device: DeviceMemory,
    others: &[mrun_engine::sysmem::Process],
) -> CliError {
    let room = device.available_room().unwrap_or(0);
    let gib = |bytes: u64| format!("{:.1} GiB", bytes as f64 / budget::GIB as f64);
    let largest = if others.is_empty() {
        String::new()
    } else {
        format!(
            "; largest other processes: {}",
            mrun_engine::sysmem::describe(others)
        )
    };
    let smaller = aliases::ALIASES
        .iter()
        .rev()
        .filter(|alias| alias.name != name)
        .find(|alias| {
            let info = model_info(
                alias.family,
                alias.kv_bytes_per_token,
                alias.recurrent_bytes,
                alias.max_position,
            );
            budget_needed(floor_tokens(&info), alias.loaded_weights(), &info) <= room
        });
    let hint = match smaller {
        Some(alias) => format!("Quit some, or use {}.", alias.name),
        None => "Quit some and rerun; no smaller model fits either.".to_string(),
    };
    CliError::new(
        Code::MemoryShort,
        format!(
            "{name} needs {} but only {} are available ({} free, less a {} margin for other apps){largest}",
            gib(need),
            gib(room),
            gib(device.available.unwrap_or(0)),
            gib(budget::margin(device.physical))
        ),
    )
    .hint(hint)
}

/// The 48 GB target of the data model's `capacity_48gb`.
pub const TARGET_48GB: DeviceMemory = DeviceMemory {
    physical: 48 * budget::GIB,
    recommended_working_set: 48 * budget::GIB / 100 * 78,
    available: None,
    estimated: false,
};

/// `4.6 GB`, `512 MB` (decimal units, as disk tools show them).
pub fn human_bytes(bytes: u64) -> String {
    const GB: f64 = 1e9;
    const MB: f64 = 1e6;
    let value = bytes as f64;
    if value >= GB {
        format!("{:.1} GB", value / GB)
    } else if value >= MB {
        format!("{:.0} MB", value / MB)
    } else {
        format!("{bytes} B")
    }
}

/// Fakes for tests: a context in a directory and cached models (also for integration tests,
/// with feature `test-support`).
#[cfg(any(test, feature = "test-support"))]
pub mod test_support {
    use super::*;

    /// Paths and hub inside `root`.
    pub fn ctx(root: &Path, endpoint: &str) -> Ctx {
        let paths = Paths {
            hf_cache: root.join("hf/hub"),
            hf_home: root.join("hf"),
            mrun_home: root.join("mrun"),
        };
        Ctx {
            paths,
            hub: Hub {
                endpoint: endpoint.into(),
                token: None,
                offline: false,
            },
            device: Some(TARGET_48GB),
            free_space: Some(u64::MAX),
        }
    }

    /// A pulled model: `repo` at `revision` with its files (path, content) in the store.
    pub fn cached_model(
        paths: &Paths,
        alias: Option<&str>,
        repo: &str,
        revision: &str,
        files: &[(&str, &[u8])],
    ) -> Manifest {
        let dir = paths.models().join(repo.replace('/', "--")).join(revision);
        fs::create_dir_all(&dir).unwrap();
        let mut entries = Vec::new();
        for (path, content) in files {
            let file = dir.join(path);
            fs::write(&file, content).unwrap();
            entries.push(ManifestFile {
                path: path.to_string(),
                size: content.len() as u64,
                sha256: verify::sha256_bytes(content),
                mtime_ns: verify::mtime_ns(&fs::metadata(&file).unwrap()),
                pad: 0,
            });
        }
        let manifest = Manifest {
            alias: alias.map(str::to_string),
            repo: repo.into(),
            revision: revision.into(),
            license: "apache-2.0".into(),
            family: "qwen3".into(),
            syntax: "hermes".into(),
            architecture: "qwen3".into(),
            text_architecture: None,
            state_bytes_per_token: 147_456,
            checkpoint_bytes: 0,
            max_position: 40_960,
            capacity_48gb: 0,
            identity: "0".repeat(64),
            quantization: None,
            files: entries,
            part: None,
        };
        manifest.save(paths).unwrap();
        manifest
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn vars(pairs: &[(&str, &str)]) -> impl Fn(&str) -> Option<String> {
        let map: HashMap<String, String> = pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        move |key| map.get(key).cloned()
    }

    #[test]
    fn cache_locations_follow_hf_precedence() {
        let paths = Paths::from_vars(&vars(&[("HOME", "/h")]));
        assert_eq!(paths.hf_cache, Path::new("/h/.cache/huggingface/hub"));
        assert_eq!(paths.mrun_home, Path::new("/h/.mrun"));
        let paths = Paths::from_vars(&vars(&[("HOME", "/h"), ("XDG_CACHE_HOME", "/x")]));
        assert_eq!(paths.hf_cache, Path::new("/x/huggingface/hub"));
        assert_eq!(paths.models(), Path::new("/h/.mrun/models"));
        // Without HOME the account's home directory is used, never a relative path.
        assert!(Paths::from_vars(&vars(&[])).mrun_home.is_absolute());
        let paths = Paths::from_vars(&vars(&[("HOME", "/h"), ("HF_HOME", "/hf")]));
        assert_eq!(paths.hf_cache, Path::new("/hf/hub"));
        let paths = Paths::from_vars(&vars(&[
            ("HF_HOME", "/hf"),
            ("HF_HUB_CACHE", "/cache"),
            ("MRUN_HOME", "/m"),
        ]));
        assert_eq!(paths.hf_cache, Path::new("/cache"));
        assert_eq!(paths.mrun_home, Path::new("/m"));
        assert_eq!(
            paths.repo_dir("org/name"),
            Path::new("/cache/models--org--name")
        );
    }

    #[test]
    fn hub_reads_endpoint_token_file_and_offline() {
        let root = tempfile::tempdir().unwrap();
        let home = root.path().to_str().unwrap();
        let paths = Paths::from_vars(&vars(&[("HF_HOME", home)]));
        let hub = Hub::from_vars(&vars(&[]), &paths);
        assert_eq!(hub.endpoint, "https://huggingface.co");
        assert_eq!(hub.token, None);
        assert!(!hub.offline);

        fs::write(root.path().join("token"), "hf_file\n").unwrap();
        let hub = Hub::from_vars(
            &vars(&[
                ("HF_ENDPOINT", "https://mirror.example/"),
                ("HF_HUB_OFFLINE", "1"),
            ]),
            &paths,
        );
        assert_eq!(hub.endpoint, "https://mirror.example");
        assert_eq!(hub.token.as_deref(), Some("hf_file"));
        assert!(hub.offline);
        let hub = Hub::from_vars(&vars(&[("HF_TOKEN", "hf_env")]), &paths);
        assert_eq!(hub.token.as_deref(), Some("hf_env"));
    }

    #[test]
    fn resolve_prefers_paths_then_manifests_then_cache() {
        let root = tempfile::tempdir().unwrap();
        let ctx = test_support::ctx(root.path(), "http://unused");
        let rev = "a".repeat(40);
        let manifest = test_support::cached_model(
            &ctx.paths,
            Some("small"),
            "org/small",
            &rev,
            &[("config.json", b"{}"), ("model.safetensors", b"w")],
        );
        let local = resolve(&ctx.paths, "small").unwrap();
        assert_eq!(local.dir, manifest.dir(&ctx.paths));
        assert!(local.manifest.is_some());
        assert_eq!(
            find_manifest(&ctx.paths, "org/small@aaaa").unwrap().name(),
            "small"
        );

        let dir = root.path().join("small");
        fs::create_dir_all(&dir).unwrap();
        let cwd_relative = dir.to_str().unwrap();
        assert!(
            resolve(&ctx.paths, cwd_relative)
                .unwrap()
                .manifest
                .is_none()
        );

        // A repo in the cache without a manifest resolves through refs/main.
        let other = "b".repeat(40);
        let cached = ctx.paths.repo_dir("org/cached");
        fs::create_dir_all(cached.join("snapshots").join(&other)).unwrap();
        fs::write(
            cached.join("snapshots").join(&other).join("config.json"),
            "{}",
        )
        .unwrap();
        fs::create_dir_all(cached.join("refs")).unwrap();
        fs::write(cached.join("refs/main"), &other).unwrap();
        assert!(
            resolve(&ctx.paths, "org/cached")
                .unwrap()
                .manifest
                .is_none()
        );

        let error = resolve(&ctx.paths, "smal").unwrap_err();
        assert_eq!(error.code, Code::ModelNotFound);
        assert!(error.hint.unwrap().contains("did you mean 'small'"));
        let error = resolve(&ctx.paths, "qwen3.6").unwrap_err();
        assert_eq!(error.hint.as_deref(), Some("run 'mrun pull qwen3.6' first"));
    }

    #[test]
    fn a_directory_named_like_a_model_is_ambiguous() {
        let root = tempfile::tempdir().unwrap();
        let ctx = test_support::ctx(root.path(), "http://unused");
        // `src` is a directory in the crate root, where cargo runs tests.
        let name = "src";
        assert!(resolve(&ctx.paths, name).unwrap().manifest.is_none());
        // A pulled model of the same name.
        let files: &[(&str, &[u8])] = &[("config.json", b"{}")];
        test_support::cached_model(&ctx.paths, Some(name), "org/m", &"a".repeat(40), files);
        let error = resolve(&ctx.paths, name).unwrap_err();
        assert_eq!(error.code, Code::ModelAmbiguous);
        assert!(error.hint.unwrap().starts_with(&format!("write ./{name}")));
    }

    #[test]
    fn the_working_set_estimate_follows_the_wired_limit_or_one_share() {
        let gib = budget::GIB;
        assert_eq!(estimated_working_set(48 * gib, 40_000), 40_000 << 20);
        // One share at every size: no RAM-size boundary (D048).
        for size in [8, 16, 36, 37, 48, 128] {
            assert_eq!(
                estimated_working_set(size * gib, 0),
                (size as f64 * ESTIMATED_WORKING_SET_SHARE * gib as f64) as u64,
                "{size} GiB"
            );
        }
        assert_eq!(
            estimated_working_set(8 * gib, 1 << 20),
            8 * gib,
            "capped at physical"
        );
    }

    #[test]
    fn the_default_model_prefers_the_machines_alias() {
        let root = tempfile::tempdir().unwrap();
        let mut ctx = test_support::ctx(root.path(), "http://unused");
        assert_eq!(default_model(&ctx).unwrap(), "qwen3.6");
        let files: &[(&str, &[u8])] = &[("config.json", b"{}"), ("model.safetensors", b"w")];
        test_support::cached_model(&ctx.paths, None, "org/small", &"a".repeat(40), files);
        assert_eq!(
            default_model(&ctx).unwrap(),
            "org/small",
            "a pulled model that fits"
        );
        test_support::cached_model(
            &ctx.paths,
            Some("qwen3.6"),
            "org/big",
            &"b".repeat(40),
            files,
        );
        assert_eq!(default_model(&ctx).unwrap(), "qwen3.6");
        ctx.device = None;
        assert_eq!(default_model(&ctx).unwrap(), "qwen3.5");
        // 8 GB: nothing pulled fits and no alias leaves the minimum context.
        let empty = tempfile::tempdir().unwrap();
        let mut small = test_support::ctx(empty.path(), "http://unused");
        small.device = Some(DeviceMemory {
            physical: 8 * budget::GIB,
            recommended_working_set: 5 * budget::GIB,
            available: None,
            estimated: false,
        });
        let error = default_model(&small).unwrap_err();
        assert_eq!(error.code, Code::DoesNotFit, "{error}");
        // `--budget-gib` at a 16 GB Mac's working set picks what that Mac picks (D060).
        let empty = tempfile::tempdir().unwrap();
        let mut target = test_support::ctx(empty.path(), "http://unused");
        target.device = Some(DeviceMemory {
            physical: 51_539_607_552,
            recommended_working_set: 40_200_896_512,
            available: None,
            estimated: false,
        });
        assert_eq!(default_model(&target).unwrap(), "qwen3.6");
        let simulated = target.with_budget(Some(11_453_250_000));
        assert_eq!(default_model(&simulated).unwrap(), "qwen3.5");
        let raised = simulated.with_budget(Some(100 * budget::GIB));
        assert_eq!(
            raised.device.unwrap().recommended_working_set,
            11_453_250_000,
            "never raised"
        );
    }

    /// D062: other processes holding memory refuse the load with E041 and the numbers; they
    /// do not change the default pick.
    #[test]
    fn short_available_memory_is_refused_with_the_largest_processes() {
        use mrun_engine::sysmem::Process;
        let device = |available: Option<u64>| DeviceMemory {
            physical: 48 * budget::GIB,
            recommended_working_set: 40_200_896_512,
            available,
            estimated: false,
        };
        let alias = aliases::ALIASES
            .iter()
            .find(|a| a.name == "qwen3.6")
            .unwrap();
        let info = model_info(
            alias.family,
            alias.kv_bytes_per_token,
            alias.recurrent_bytes,
            alias.max_position,
        );
        let weights = alias.loaded_weights();
        let need = budget_needed(65_536, weights, &info);
        assert!(check_available(device(None), "qwen3.6", weights, &info).is_ok());
        assert!(check_available(device(Some(44 * budget::GIB)), "qwen3.6", weights, &info).is_ok());
        assert!(
            check_available(
                device(Some(need + 6 * budget::GIB)),
                "qwen3.6",
                weights,
                &info
            )
            .is_ok()
        );
        let others = [
            Process {
                pid: 2,
                name: "com.apple.Virtualization.VirtualMachine".into(),
                resident: 34_681_651_200,
            },
            Process {
                pid: 3,
                name: "rust-analyzer".into(),
                resident: 4_617_089_843,
            },
        ];
        let short = device(Some(20 * budget::GIB + budget::GIB / 5));
        let error = memory_short("qwen3.6", need, short, &others);
        assert_eq!(error.code, Code::MemoryShort);
        assert_eq!(error.code.id(), "E041");
        assert_eq!(
            error.to_string(),
            format!(
                "error[E041]: qwen3.6 needs {:.1} GiB but only 14.2 GiB are available (20.2 GiB \
                 free, less a 6.0 GiB margin for other apps); largest other processes: \
                 com.apple.Virtualization.VirtualMachine 32.3 GiB, rust-analyzer 4.3 GiB\n  \
                 hint: Quit some, or use qwen3.5.",
                need as f64 / budget::GIB as f64
            )
        );
        let none = memory_short("qwen3.6", need, device(Some(7 * budget::GIB)), &[]);
        assert!(none.to_string().contains("no smaller model fits"), "{none}");
        assert!(check_available(short, "qwen3.6", weights, &info).is_err());
        // The default pick ignores what other processes hold.
        let root = tempfile::tempdir().unwrap();
        let mut ctx = test_support::ctx(root.path(), "http://unused");
        ctx.device = Some(short);
        assert_eq!(default_model(&ctx).unwrap(), "qwen3.6");
        // `ls` fit counts it.
        assert!(alias.fit(device(None)).max_context >= 65_536);
        assert!(alias.fit(short).max_context < 65_536);
    }

    #[test]
    fn several_revisions_of_a_repo_are_ambiguous() {
        let root = tempfile::tempdir().unwrap();
        let ctx = test_support::ctx(root.path(), "http://unused");
        let files: &[(&str, &[u8])] = &[("config.json", b"{}")];
        test_support::cached_model(&ctx.paths, Some("one"), "org/m", &"a".repeat(40), files);
        test_support::cached_model(&ctx.paths, Some("two"), "org/m", &"b".repeat(40), files);
        let error = find_manifest(&ctx.paths, "org/m").unwrap_err();
        assert_eq!(error.code, Code::ModelAmbiguous);
        assert_eq!(
            find_manifest(&ctx.paths, "org/m@bbb").unwrap().name(),
            "two"
        );
    }

    #[test]
    fn a_budget_override_lowers_the_fit_and_budget_needed_inverts_it() {
        let info = model_info("qwen3", 147_456, 0, 40_960);
        let full = fit(TARGET_48GB, 5_000_000_000, &info);
        let lowered = fit_with(TARGET_48GB, Some(8 * budget::GIB), 5_000_000_000, &info);
        assert_eq!(lowered.budget, 8 * budget::GIB);
        assert!(lowered.max_context < full.max_context);
        // Never above physical − 2 GiB, whatever the working set says.
        let wired = DeviceMemory {
            physical: 16 * budget::GIB,
            recommended_working_set: 16 * budget::GIB,
            available: None,
            estimated: false,
        };
        assert_eq!(fit(wired, 1, &info).budget, 14 * budget::GIB);
        // An override above the automatic budget is refused, not raised.
        assert!(!fit_with(wired, Some(15 * budget::GIB), 1, &info).fits);
        let need = budget_needed(lowered.max_context, 5_000_000_000, &info);
        assert!(need <= 8 * budget::GIB && need + info.kv_bytes_per_token > 8 * budget::GIB);
    }

    #[test]
    fn fit_uses_the_budget_functions() {
        let manifest = Manifest {
            state_bytes_per_token: 147_456,
            max_position: 40_960,
            ..test_manifest()
        };
        let fit = fit(TARGET_48GB, 5_000_000_000, &manifest.model_info());
        assert!(fit.fits);
        assert_eq!(fit.max_context, 40_960);
        let fit = super::fit(TARGET_48GB, 60_000_000_000, &manifest.model_info());
        assert!(!fit.fits);
        assert_eq!(fit.max_context, 0);
    }

    fn test_manifest() -> Manifest {
        Manifest {
            alias: None,
            repo: "org/m".into(),
            revision: "a".repeat(40),
            license: "mit".into(),
            family: "qwen3".into(),
            syntax: "hermes".into(),
            architecture: "qwen3".into(),
            text_architecture: None,
            state_bytes_per_token: 1,
            checkpoint_bytes: 0,
            max_position: 1,
            capacity_48gb: 0,
            identity: String::new(),
            quantization: Some(Quantization {
                bits: Some(4),
                group_size: Some(64),
                mode: None,
            }),
            files: vec![ManifestFile {
                path: "config.json".into(),
                size: 2,
                sha256: "0".repeat(64),
                mtime_ns: 1,
                pad: 0,
            }],
            part: None,
        }
    }

    #[test]
    fn pads_are_validated_and_counted_on_disk() {
        let mut manifest = test_manifest();
        manifest.files.push(ManifestFile {
            path: "model.safetensors".into(),
            size: 100,
            sha256: "1".repeat(64),
            mtime_ns: 1,
            pad: 7,
        });
        assert_eq!(manifest.validate(), Ok(()));
        assert_eq!(manifest.size(), 2 + 100 + 7);
        assert_eq!(manifest.weights(), 100, "weights are the Hub's bytes");
        manifest.files[1].pad = 16;
        assert!(manifest.validate().unwrap_err().contains("invalid pad 16"));
        manifest.files[1].pad = 0;
        manifest.files[0].pad = 1;
        assert!(manifest.validate().unwrap_err().contains("'config.json'"));
    }

    #[test]
    fn manifest_round_trips_through_toml() {
        let root = tempfile::tempdir().unwrap();
        let ctx = test_support::ctx(root.path(), "http://unused");
        let manifest = test_manifest();
        manifest.save(&ctx.paths).unwrap();
        assert!(ctx.paths.manifests().join("org--m.toml").exists());
        assert_eq!(manifests(&ctx.paths), [manifest]);
    }

    #[test]
    fn human_sizes() {
        assert_eq!(human_bytes(4_600_000_000), "4.6 GB");
        assert_eq!(human_bytes(512_000_000), "512 MB");
        assert_eq!(human_bytes(12), "12 B");
    }
}
