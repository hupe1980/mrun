//! Model content identity: a hash over the files that determine a model's output.
//!
//! `identity = sha256` over the sorted `(relative path, file sha256)` pairs of every file that
//! determines output. In the Hugging Face cache, weight files are symlinks to LFS blobs named by
//! their sha256, so they need no hashing. Other files are hashed once and cached by
//! (canonical path, size, mtime, inode).

use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs;
use std::io::{self, Read};
use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;

/// Files besides `*.safetensors` that determine output.
const OUTPUT_FILES: [&str; 6] = [
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "generation_config.json",
    "special_tokens_map.json",
];

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Identity {
    /// 64-hex content identity.
    pub value: String,
    /// The identity previously recorded for this model directory, if it differs.
    pub changed_from: Option<String>,
    /// `(relative path, sha256)` for every identifying file.
    pub files: Vec<(String, String)>,
}

#[derive(Debug, thiserror::Error)]
pub enum IdentityError {
    #[error("{0} is not a model directory: no *.safetensors and config.json found")]
    NotAModel(PathBuf),
    #[error("cannot read {path}: {source}")]
    Io { path: PathBuf, source: io::Error },
    #[error("{path} is not valid JSON: {message}")]
    Json { path: PathBuf, message: String },
    #[error("identity cache {path} is corrupt: {message}")]
    Cache { path: PathBuf, message: String },
}

#[derive(Clone, Debug, Default, Deserialize, Serialize, Eq, PartialEq)]
struct FileRecord {
    size: u64,
    mtime_ns: u128,
    inode: u64,
    sha256: String,
}

#[derive(Debug, Default, Deserialize, Serialize)]
struct Cache {
    #[serde(default)]
    files: BTreeMap<String, FileRecord>,
    #[serde(default)]
    models: BTreeMap<String, String>,
}

/// Computes the identity of the model in `dir`, using and updating the cache file `cache_path`.
pub fn identify(dir: &Path, cache_path: &Path) -> Result<Identity, IdentityError> {
    let io_err = |path: &Path| {
        let path = path.to_path_buf();
        move |source| IdentityError::Io { path, source }
    };
    let dir = fs::canonicalize(dir).map_err(io_err(dir))?;
    let mut cache = load_cache(cache_path)?;
    let mut names = Vec::new();
    for entry in fs::read_dir(&dir).map_err(io_err(&dir))? {
        let name = entry
            .map_err(io_err(&dir))?
            .file_name()
            .to_string_lossy()
            .into_owned();
        if name.ends_with(".safetensors") || OUTPUT_FILES.contains(&name.as_str()) {
            names.push(name);
        }
    }
    names.sort();
    if !names.iter().any(|name| name.ends_with(".safetensors"))
        || !names.iter().any(|name| name == "config.json")
    {
        return Err(IdentityError::NotAModel(dir));
    }

    let mut files = Vec::with_capacity(names.len());
    for name in names {
        let sha = file_sha256(&dir.join(&name), &mut cache)?;
        files.push((name, sha));
    }
    let mut hasher = Sha256::new();
    for (name, sha) in &files {
        hasher.update(name.as_bytes());
        hasher.update([0]);
        hasher.update(sha.as_bytes());
        hasher.update([b'\n']);
    }
    let value = hex(&hasher.finalize());
    let key = dir.to_string_lossy().into_owned();
    let changed_from = cache
        .models
        .insert(key, value.clone())
        .filter(|previous| *previous != value);
    store_cache(cache_path, &cache)?;
    Ok(Identity {
        value,
        changed_from,
        files,
    })
}

/// End-of-sequence token IDs of the model in `dir`, sorted: `eos_token_id` from `config.json`
/// (root or `text_config`) and `generation_config.json` (int or list), the tokenizer's EOS token
/// from `tokenizer_config.json` and `special_tokens_map.json`, and the family's `extra` end
/// tokens. Token texts missing from the vocabulary are skipped.
pub fn eos_union(dir: &Path, extra: &[&str]) -> Result<Vec<u32>, IdentityError> {
    let config = read_json(&dir.join("config.json"))?;
    let generation = read_json(&dir.join("generation_config.json"))?;
    let tokenizer_config = read_json(&dir.join("tokenizer_config.json"))?;
    let special = read_json(&dir.join("special_tokens_map.json"))?;
    let tokenizer = read_json(&dir.join("tokenizer.json"))?;

    let mut ids = Vec::new();
    for value in [
        &config["eos_token_id"],
        &config["text_config"]["eos_token_id"],
        &generation["eos_token_id"],
    ] {
        match value {
            Value::Array(items) => ids.extend(items.iter().filter_map(token_id)),
            other => ids.extend(token_id(other)),
        }
    }
    let texts = [&tokenizer_config["eos_token"], &special["eos_token"]]
        .into_iter()
        .filter_map(|token| token.as_str().or_else(|| token["content"].as_str()))
        .chain(extra.iter().copied());
    for text in texts {
        ids.extend(vocab_id(text, &tokenizer, &tokenizer_config));
    }
    ids.sort_unstable();
    ids.dedup();
    Ok(ids)
}

fn token_id(value: &Value) -> Option<u32> {
    value.as_u64().and_then(|id| u32::try_from(id).ok())
}

/// ID of `text` in `tokenizer.json` (added tokens, then the model vocabulary) or in
/// `tokenizer_config.json` `added_tokens_decoder`.
fn vocab_id(text: &str, tokenizer: &Value, tokenizer_config: &Value) -> Option<u32> {
    let added = tokenizer["added_tokens"]
        .as_array()
        .into_iter()
        .flatten()
        .find(|token| token["content"] == text)
        .and_then(|token| token_id(&token["id"]));
    added
        .or_else(|| token_id(&tokenizer["model"]["vocab"][text]))
        .or_else(|| {
            tokenizer_config["added_tokens_decoder"]
                .as_object()?
                .iter()
                .find(|(_, token)| token["content"] == text)
                .and_then(|(id, _)| id.parse().ok())
        })
}

/// Parsed JSON file, or `Null` when it does not exist.
fn read_json(path: &Path) -> Result<Value, IdentityError> {
    match fs::read_to_string(path) {
        Ok(text) => serde_json::from_str(&text).map_err(|error| IdentityError::Json {
            path: path.to_path_buf(),
            message: error.to_string(),
        }),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(Value::Null),
        Err(source) => Err(IdentityError::Io {
            path: path.to_path_buf(),
            source,
        }),
    }
}

fn file_sha256(path: &Path, cache: &mut Cache) -> Result<String, IdentityError> {
    let io_err = |source| IdentityError::Io {
        path: path.to_path_buf(),
        source,
    };
    let target = fs::canonicalize(path).map_err(io_err)?;
    if let Some(sha) = hf_lfs_blob_sha(&target) {
        return Ok(sha);
    }
    let meta = fs::metadata(&target).map_err(io_err)?;
    let record = FileRecord {
        size: meta.len(),
        mtime_ns: meta
            .modified()
            .ok()
            .and_then(|time| time.duration_since(UNIX_EPOCH).ok())
            .map_or(0, |duration| duration.as_nanos()),
        inode: inode(&meta),
        sha256: String::new(),
    };
    let key = target.to_string_lossy().into_owned();
    if let Some(cached) = cache.files.get(&key)
        && cached.size == record.size
        && cached.mtime_ns == record.mtime_ns
        && cached.inode == record.inode
    {
        return Ok(cached.sha256.clone());
    }
    let mut file = fs::File::open(&target).map_err(io_err)?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0; 1 << 20];
    loop {
        let read = file.read(&mut buffer).map_err(io_err)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    let sha256 = hex(&hasher.finalize());
    cache.files.insert(
        key,
        FileRecord {
            sha256: sha256.clone(),
            ..record
        },
    );
    Ok(sha256)
}

/// In the HF hub cache, LFS files live at `.../blobs/<sha256>`.
fn hf_lfs_blob_sha(target: &Path) -> Option<String> {
    let name = target.file_name()?.to_str()?;
    let parent = target.parent()?.file_name()?.to_str()?;
    (parent == "blobs" && name.len() == 64 && name.bytes().all(|b| b.is_ascii_hexdigit()))
        .then(|| name.to_ascii_lowercase())
}

#[cfg(unix)]
fn inode(meta: &fs::Metadata) -> u64 {
    std::os::unix::fs::MetadataExt::ino(meta)
}

#[cfg(not(unix))]
fn inode(_meta: &fs::Metadata) -> u64 {
    0
}

fn load_cache(path: &Path) -> Result<Cache, IdentityError> {
    match fs::read_to_string(path) {
        Ok(text) => toml::from_str(&text).map_err(|error| IdentityError::Cache {
            path: path.to_path_buf(),
            message: error.to_string(),
        }),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(Cache::default()),
        Err(source) => Err(IdentityError::Io {
            path: path.to_path_buf(),
            source,
        }),
    }
}

fn store_cache(path: &Path, cache: &Cache) -> Result<(), IdentityError> {
    let io_err = |source| IdentityError::Io {
        path: path.to_path_buf(),
        source,
    };
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(io_err)?;
    }
    let text = toml::to_string(cache).map_err(|error| IdentityError::Cache {
        path: path.to_path_buf(),
        message: error.to_string(),
    })?;
    let temp = path.with_extension("toml.tmp");
    fs::write(&temp, text).map_err(io_err)?;
    fs::rename(&temp, path).map_err(io_err)
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model_dir(root: &Path) -> PathBuf {
        let dir = root.join("model");
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("config.json"), r#"{"model_type":"qwen3"}"#).unwrap();
        fs::write(dir.join("tokenizer.json"), "{}").unwrap();
        fs::write(dir.join("model.safetensors"), b"weights").unwrap();
        fs::write(dir.join("README.md"), "ignored").unwrap();
        dir
    }

    #[test]
    fn identity_is_stable_and_ignores_unrelated_files() {
        let root = tempfile::tempdir().unwrap();
        let dir = model_dir(root.path());
        let cache = root.path().join("identity.toml");
        let first = identify(&dir, &cache).unwrap();
        fs::write(dir.join("README.md"), "changed").unwrap();
        let second = identify(&dir, &cache).unwrap();
        assert_eq!(first.value, second.value);
        assert_eq!(second.changed_from, None);
        assert_eq!(first.value.len(), 64);
        let names: Vec<_> = first.files.iter().map(|(name, _)| name.as_str()).collect();
        assert_eq!(
            names,
            ["config.json", "model.safetensors", "tokenizer.json"]
        );
    }

    #[test]
    fn changed_content_is_reported() {
        let root = tempfile::tempdir().unwrap();
        let dir = model_dir(root.path());
        let cache = root.path().join("identity.toml");
        let first = identify(&dir, &cache).unwrap();
        fs::write(dir.join("model.safetensors"), b"other weights").unwrap();
        let second = identify(&dir, &cache).unwrap();
        assert_ne!(first.value, second.value);
        assert_eq!(second.changed_from.as_deref(), Some(first.value.as_str()));
    }

    #[cfg(unix)]
    #[test]
    fn hf_lfs_blob_names_are_used_without_hashing() {
        let root = tempfile::tempdir().unwrap();
        let blobs = root.path().join("blobs");
        fs::create_dir_all(&blobs).unwrap();
        let sha = "ab".repeat(32);
        fs::write(blobs.join(&sha), b"not actually hashed").unwrap();
        let snapshot = root.path().join("snapshots/rev");
        fs::create_dir_all(&snapshot).unwrap();
        std::os::unix::fs::symlink(blobs.join(&sha), snapshot.join("model.safetensors")).unwrap();
        fs::write(snapshot.join("config.json"), "{}").unwrap();
        let identity = identify(&snapshot, &root.path().join("identity.toml")).unwrap();
        assert!(
            identity
                .files
                .contains(&("model.safetensors".to_string(), sha))
        );
    }

    #[test]
    fn eos_union_joins_every_source_and_follows_generation_config() {
        let root = tempfile::tempdir().unwrap();
        let dir = model_dir(root.path());
        let write = |name: &str, value: serde_json::Value| {
            fs::write(dir.join(name), value.to_string()).unwrap();
        };
        write(
            "config.json",
            serde_json::json!({"model_type": "gpt_oss", "eos_token_id": 200002}),
        );
        write(
            "generation_config.json",
            serde_json::json!({"eos_token_id": [200002, 199999]}),
        );
        write(
            "tokenizer.json",
            serde_json::json!({
                "added_tokens": [
                    {"id": 200012, "content": "<|call|>"},
                    {"id": 200002, "content": "<|return|>"}
                ],
                "model": {"vocab": {"</s>": 2}}
            }),
        );
        write(
            "tokenizer_config.json",
            serde_json::json!({
                "eos_token": {"content": "<|return|>"},
                "added_tokens_decoder": {"200007": {"content": "<|end|>"}}
            }),
        );
        write(
            "special_tokens_map.json",
            serde_json::json!({"eos_token": "</s>"}),
        );
        let extra = ["<|call|>", "<|return|>", "<|end|>", "<|missing|>"];
        assert_eq!(
            eos_union(&dir, &extra).unwrap(),
            [2, 199999, 200002, 200007, 200012]
        );

        write(
            "generation_config.json",
            serde_json::json!({"eos_token_id": 7}),
        );
        assert_eq!(eos_union(&dir, &[]).unwrap(), [2, 7, 200002]);
    }

    #[test]
    fn eos_union_reads_wrapped_configs_and_tolerates_missing_files() {
        let root = tempfile::tempdir().unwrap();
        let dir = root.path();
        fs::write(
            dir.join("config.json"),
            r#"{"model_type":"qwen3_5_moe","text_config":{"eos_token_id":248044}}"#,
        )
        .unwrap();
        assert_eq!(eos_union(dir, &["<|im_end|>"]).unwrap(), [248044]);
        fs::write(dir.join("generation_config.json"), "{").unwrap();
        assert!(matches!(
            eos_union(dir, &[]),
            Err(IdentityError::Json { .. })
        ));
    }

    #[test]
    fn non_model_directories_are_rejected() {
        let root = tempfile::tempdir().unwrap();
        let error = identify(root.path(), &root.path().join("c.toml")).unwrap_err();
        assert!(matches!(error, IdentityError::NotAModel(_)));
    }
}
