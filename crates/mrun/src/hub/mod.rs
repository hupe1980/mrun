//! Models in the shared Hugging Face hub cache (`models--org--name/{blobs,snapshots,refs}`) and
//! mrun's own state in `MRUN_HOME`: one manifest per pulled model.
//!
//! Resolution order (004 contract): existing path, then a manifest (alias or repo id), then a
//! repo id already in the HF cache.

pub mod aliases;
pub mod download;
pub mod verify;

use crate::errors::{CliError, Code};
use mrun_engine::ModelInfo;
use mrun_engine::budget::{self, DeviceMemory, Overrides, SCRATCH_ESTIMATE};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};

/// Where models and mrun's state live.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Paths {
    /// `HF_HUB_CACHE`, else `HF_HOME/hub`, else `~/.cache/huggingface/hub`.
    pub hf_cache: PathBuf,
    /// `HF_HOME`, else `~/.cache/huggingface` (holds the token of `hf auth login`).
    pub hf_home: PathBuf,
    /// `MRUN_HOME`, else `~/.mrun`.
    pub mrun_home: PathBuf,
}

impl Paths {
    /// From an environment lookup; empty values count as unset.
    pub fn from_vars(var: &dyn Fn(&str) -> Option<String>) -> Self {
        let var = |key: &str| {
            var(key)
                .filter(|value| !value.is_empty())
                .map(PathBuf::from)
        };
        let home = var("HOME").unwrap_or_default();
        let hf_home = var("HF_HOME").unwrap_or_else(|| home.join(".cache/huggingface"));
        Self {
            hf_cache: var("HF_HUB_CACHE").unwrap_or_else(|| hf_home.join("hub")),
            mrun_home: var("MRUN_HOME").unwrap_or_else(|| home.join(".mrun")),
            hf_home,
        }
    }

    /// `models--org--name` in the HF cache.
    pub fn repo_dir(&self, repo: &str) -> PathBuf {
        self.hf_cache
            .join(format!("models--{}", repo.replace('/', "--")))
    }

    pub fn manifests(&self) -> PathBuf {
        self.mrun_home.join("manifests")
    }

    /// Cache of file hashes used by 001's identity function.
    pub fn identity_cache(&self) -> PathBuf {
        self.mrun_home.join("identity.toml")
    }
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
    pub fn from_vars(var: &dyn Fn(&str) -> Option<String>, paths: &Paths) -> Self {
        let var = |key: &str| var(key).filter(|value| !value.trim().is_empty());
        let token = var("HF_TOKEN").or_else(|| {
            fs::read_to_string(paths.hf_home.join("token"))
                .ok()
                .filter(|token| !token.trim().is_empty())
        });
        Self {
            endpoint: var("HF_ENDPOINT")
                .unwrap_or_else(|| "https://huggingface.co".into())
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
    /// Free bytes on the cache's disk; `None` asks the system (`df`).
    pub free_space: Option<u64>,
}

impl Ctx {
    pub fn from_env() -> Self {
        let var = |key: &str| std::env::var(key).ok();
        let paths = Paths::from_vars(&var);
        Self {
            hub: Hub::from_vars(&var, &paths),
            paths,
            device: device_memory(),
            free_space: None,
        }
    }
}

/// This machine's memory: `/proc/meminfo` on Linux, `hw.memsize` on macOS (the working set is
/// taken as all of it; the budget's fraction caps it first).
pub fn device_memory() -> Option<DeviceMemory> {
    #[cfg(target_os = "linux")]
    {
        DeviceMemory::read_linux().ok()
    }
    #[cfg(target_os = "macos")]
    {
        let out = std::process::Command::new("sysctl")
            .args(["-n", "hw.memsize"])
            .output()
            .ok()?;
        let physical = String::from_utf8_lossy(&out.stdout).trim().parse().ok()?;
        Some(DeviceMemory {
            physical,
            recommended_working_set: physical,
        })
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    {
        None
    }
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
    /// Path relative to the snapshot directory.
    pub path: String,
    pub size: u64,
    pub sha256: String,
    /// Modification time of the file content (the blob), nanoseconds since the epoch.
    pub mtime_ns: u64,
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
}

impl Manifest {
    /// What users type: the alias, else the repo id.
    pub fn name(&self) -> &str {
        self.alias.as_deref().unwrap_or(&self.repo)
    }

    fn file_stem(&self) -> String {
        self.alias
            .clone()
            .unwrap_or_else(|| self.repo.replace('/', "--"))
    }

    pub fn path(&self, paths: &Paths) -> PathBuf {
        paths.manifests().join(format!("{}.toml", self.file_stem()))
    }

    pub fn snapshot_dir(&self, paths: &Paths) -> PathBuf {
        paths
            .repo_dir(&self.repo)
            .join("snapshots")
            .join(&self.revision)
    }

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

    /// The fields 001's budget functions read.
    pub fn model_info(&self) -> ModelInfo {
        ModelInfo {
            identity: self.identity.clone(),
            family: self.family.clone(),
            layer_kinds: Vec::new(),
            kv_heads: 0,
            head_dim: 0,
            vocab: 0,
            max_position: self.max_position,
            kv_bytes_per_token: self.state_bytes_per_token,
            recurrent_bytes: self.checkpoint_bytes,
            eos_tokens: Vec::new(),
        }
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
                .and_then(|text| toml::from_str(&text).map_err(|error| error.to_string()));
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
    let mut matches: Vec<&Manifest> = all
        .iter()
        .filter(|m| {
            m.alias.as_deref() == Some(name)
                || (m.repo == repo && revision.is_none_or(|rev| m.revision.starts_with(rev)))
        })
        .collect();
    if matches.iter().any(|m| m.alias.as_deref() == Some(name)) {
        matches.retain(|m| m.alias.as_deref() == Some(name));
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

/// Resolves `name` for loading: an existing path wins, then a manifest (fast-checked; a mismatch
/// is refused with the repair command), then a repo id already in the HF cache.
pub fn resolve(paths: &Paths, name: &str) -> Result<LocalModel, CliError> {
    let path = Path::new(name);
    if path.exists() {
        return Ok(LocalModel {
            dir: path.to_path_buf(),
            manifest: None,
        });
    }
    match find_manifest(paths, name) {
        Ok(manifest) => {
            let dir = manifest.snapshot_dir(paths);
            if let verify::Status::Corrupt(problems) = verify::fast_check(&manifest, &dir) {
                return Err(verify::corrupt_error(&manifest, &problems));
            }
            return Ok(LocalModel {
                dir,
                manifest: Some(manifest),
            });
        }
        Err(error) if error.code != Code::ModelNotFound => return Err(error),
        Err(_) => {}
    }
    let (repo, revision) = split_revision(name);
    let repo_dir = paths.repo_dir(repo);
    let revision = match revision {
        Some(rev) => rev.to_string(),
        None => fs::read_to_string(repo_dir.join("refs/main")).unwrap_or_default(),
    };
    let dir = repo_dir.join("snapshots").join(revision.trim());
    if repo.contains('/') && !revision.trim().is_empty() && dir.join("config.json").exists() {
        return Ok(LocalModel {
            dir,
            manifest: None,
        });
    }
    Err(not_found(name, &manifests(paths)))
}

/// Memory fit on a device (001 budget functions, default overrides, estimated scratch).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Fit {
    pub budget: u64,
    pub weights: u64,
    pub scratch: u64,
    pub fits: bool,
    /// Tokens of context that fit; 0 when the model does not fit.
    pub max_context: u32,
}

pub fn fit(device: DeviceMemory, weights: u64, info: &ModelInfo) -> Fit {
    let mut fit = Fit {
        budget: 0,
        weights,
        scratch: SCRATCH_ESTIMATE,
        fits: false,
        max_context: 0,
    };
    if let Ok(plan) = budget::plan(device, Overrides::default()) {
        fit.budget = plan.budget;
        fit.fits = budget::check_fits(&plan, weights, SCRATCH_ESTIMATE, info).is_ok();
        if fit.fits {
            fit.max_context = budget::probe_depth(&plan, weights, info);
        }
    }
    fit
}

/// The 48 GB target of the data model's `capacity_48gb`.
pub const TARGET_48GB: DeviceMemory = DeviceMemory {
    physical: 48 * budget::GIB,
    recommended_working_set: 48 * budget::GIB,
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

#[cfg(test)]
pub(crate) mod test_support {
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

    /// A manifest for `repo` at `revision` whose files are cached as blobs (path, content).
    pub fn cached_model(
        paths: &Paths,
        alias: Option<&str>,
        repo: &str,
        revision: &str,
        files: &[(&str, &[u8])],
    ) -> Manifest {
        let repo_dir = paths.repo_dir(repo);
        let snapshot = repo_dir.join("snapshots").join(revision);
        fs::create_dir_all(repo_dir.join("blobs")).unwrap();
        fs::create_dir_all(&snapshot).unwrap();
        let mut entries = Vec::new();
        for (path, content) in files {
            let sha = verify::sha256_bytes(content);
            let blob = repo_dir.join("blobs").join(&sha);
            fs::write(&blob, content).unwrap();
            let link = snapshot.join(path);
            let _ = fs::remove_file(&link);
            download::link_blob(&link, &sha, path).unwrap();
            entries.push(ManifestFile {
                path: path.to_string(),
                size: content.len() as u64,
                sha256: sha,
                mtime_ns: verify::mtime_ns(&fs::metadata(&blob).unwrap()),
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
        assert_eq!(local.dir, manifest.snapshot_dir(&ctx.paths));
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
        let error = resolve(&ctx.paths, "qwen3-coder").unwrap_err();
        assert_eq!(
            error.hint.as_deref(),
            Some("run 'mrun pull qwen3-coder' first")
        );
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
            }],
        }
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
