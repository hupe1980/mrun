//! `mrun pull` against a mock hub on 127.0.0.1: metadata, the architecture gate, resume and
//! retry, corruption, disk space, gated and missing repos, the model store (realigned weights,
//! locks, import from the Hugging Face cache, which is never written), `HF_ENDPOINT` and the
//! manifest. No real network.

use axum::body::{Body, Bytes};
use axum::extract::{Request, State};
use axum::http::{StatusCode, header};
use axum::response::{IntoResponse, Response};
use futures_util::StreamExt;
use mrun::cli::ls;
use mrun::cli::verify::verify_model;
use mrun::errors::{CliError, Code};
use mrun::hub::download::pull;
use mrun::hub::verify::{git_sha1_file, sha256_bytes};
use mrun::hub::{Ctx, Hub, Manifest, Paths};
use mrun_engine::budget::{DeviceMemory, GIB};
use serde_json::json;
use std::collections::HashSet;
use std::fs;
use std::path::Path;
use std::sync::{Arc, Mutex};

const REPO: &str = "org/tiny";
const COMMIT: &str = "0123456789abcdef0123456789abcdef01234567";

/// A safetensors file of `n` bytes of `fill`, its header unpadded as MLX writes it.
fn safetensors(fill: u8, n: usize) -> Vec<u8> {
    let header = format!(r#"{{"w":{{"dtype":"U8","shape":[{n}],"data_offsets":[0,{n}]}}}}"#);
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend_from_slice(header.as_bytes());
    bytes.extend(std::iter::repeat_n(fill, n));
    bytes
}

/// Where the data of the safetensors file at `path` starts.
fn data_start(path: &Path) -> u64 {
    let bytes = fs::read(path).unwrap();
    8 + u64::from_le_bytes(bytes[..8].try_into().unwrap())
}

fn config(model_type: &str) -> Vec<u8> {
    json!({
        "model_type": model_type, "num_hidden_layers": 2, "num_attention_heads": 4,
        "num_key_value_heads": 2, "head_dim": 8, "hidden_size": 32, "vocab_size": 64,
        "max_position_embeddings": 4096, "torch_dtype": "bfloat16",
        "quantization": {"bits": 4, "group_size": 64}
    })
    .to_string()
    .into_bytes()
}

#[derive(Default)]
struct Mock {
    /// (path, content, stored in LFS)
    files: Vec<(String, Vec<u8>, bool)>,
    token: Option<String>,
    /// Files whose next download stops halfway.
    cut_once: Mutex<HashSet<String>>,
    /// Files whose next download has a flipped byte.
    corrupt_once: Mutex<HashSet<String>>,
    /// `path [range]` of every request.
    log: Mutex<Vec<String>>,
    /// A hostile mirror: this commit id instead of [`COMMIT`] in the metadata.
    sha: Option<String>,
    /// A hostile mirror: this LFS sha256 for every LFS file.
    lfs_sha: Option<String>,
}

impl Mock {
    fn new(model_type: &str) -> Self {
        Self {
            files: vec![
                ("config.json".into(), config(model_type), false),
                ("tokenizer.json".into(), b"{}".to_vec(), false),
                ("README.md".into(), b"# tiny".to_vec(), false),
                ("model.safetensors".into(), safetensors(7, 64 * 1024), true),
            ],
            ..Self::default()
        }
    }

    fn log(&self) -> Vec<String> {
        self.log.lock().unwrap().clone()
    }

    fn weight_requests(&self) -> usize {
        self.log()
            .iter()
            .filter(|line| line.contains(".safetensors"))
            .count()
    }
}

fn git_blob_id(content: &[u8]) -> String {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("f");
    fs::write(&path, content).unwrap();
    git_sha1_file(&path).unwrap()
}

async fn handle(State(mock): State<Arc<Mock>>, request: Request) -> Response {
    let path = request.uri().path().to_string();
    let range = request
        .headers()
        .get(header::RANGE)
        .and_then(|value| value.to_str().ok())
        .map(str::to_string);
    mock.log
        .lock()
        .unwrap()
        .push(format!("{path} {}", range.clone().unwrap_or_default()));
    if let Some(token) = &mock.token {
        let auth = request
            .headers()
            .get(header::AUTHORIZATION)
            .and_then(|value| value.to_str().ok());
        if auth != Some(&format!("Bearer {token}")) {
            return hub_error(StatusCode::UNAUTHORIZED, "GatedRepo");
        }
    }
    if let Some(rev) = path.strip_prefix(&format!("/api/models/{REPO}/revision/")) {
        if rev != "main" && rev != COMMIT {
            return hub_error(StatusCode::NOT_FOUND, "RevisionNotFound");
        }
        let siblings: Vec<_> = mock
            .files
            .iter()
            .map(|(name, content, lfs)| {
                let mut sibling = json!({
                    "rfilename": name, "size": content.len(), "blobId": git_blob_id(content)
                });
                if *lfs {
                    let sha = mock
                        .lfs_sha
                        .clone()
                        .unwrap_or_else(|| sha256_bytes(content));
                    sibling["lfs"] = json!({"sha256": sha, "size": content.len()});
                }
                sibling
            })
            .collect();
        return axum::Json(json!({
            "id": REPO, "sha": mock.sha.as_deref().unwrap_or(COMMIT), "siblings": siblings,
            "cardData": {"license": "apache-2.0"}
        }))
        .into_response();
    }
    if path.starts_with("/api/models/") {
        // The hub answers 401 for a repo that does not exist (or is private).
        return hub_error(StatusCode::UNAUTHORIZED, "RepositoryNotFound");
    }
    let Some(file) = path.strip_prefix(&format!("/{REPO}/resolve/{COMMIT}/")) else {
        return StatusCode::NOT_FOUND.into_response();
    };
    let Some((_, content, _)) = mock.files.iter().find(|(name, ..)| name == file) else {
        return StatusCode::NOT_FOUND.into_response();
    };
    let mut content = content.clone();
    if mock.corrupt_once.lock().unwrap().remove(file) {
        content[0] ^= 1;
    }
    let from: usize = range
        .as_deref()
        .and_then(|range| range.strip_prefix("bytes="))
        .and_then(|range| range.strip_suffix('-'))
        .and_then(|from| from.parse().ok())
        .unwrap_or(0);
    let total = content.len();
    let body = content[from..].to_vec();
    let length = body.len();
    let status = if from > 0 {
        StatusCode::PARTIAL_CONTENT
    } else {
        StatusCode::OK
    };
    let body = if mock.cut_once.lock().unwrap().remove(file) {
        let half = Bytes::from(body[..length / 2].to_vec());
        // The error comes after the headers and the first half are out.
        let cut = futures_util::stream::once(async {
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
            Err(std::io::Error::other("connection cut"))
        });
        Body::from_stream(futures_util::stream::iter([Ok(half)]).chain(cut))
    } else {
        Body::from(body)
    };
    let mut response = (status, body).into_response();
    response
        .headers_mut()
        .insert(header::CONTENT_LENGTH, length.into());
    if from > 0 {
        response.headers_mut().insert(
            header::CONTENT_RANGE,
            format!("bytes {from}-{}/{total}", total - 1)
                .parse()
                .unwrap(),
        );
    }
    response
}

fn hub_error(status: StatusCode, code: &'static str) -> Response {
    let mut response = status.into_response();
    response
        .headers_mut()
        .insert("x-error-code", code.parse().unwrap());
    response
}

/// Serves `mock`; returns its URL.
async fn serve(mock: Arc<Mock>) -> String {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let app = axum::Router::new().fallback(handle).with_state(mock);
    tokio::spawn(async move { axum::serve(listener, app).await });
    url
}

/// A context in `root` whose hub is `HF_ENDPOINT=url` (a mirror).
fn ctx(root: &Path, url: &str) -> Ctx {
    let root = root.to_str().unwrap().to_string();
    let vars = move |key: &str| match key {
        "HOME" => Some(root.clone()),
        "HF_ENDPOINT" => Some(url.to_string()),
        _ => None,
    };
    let paths = Paths::from_vars(&vars);
    Ctx {
        hub: Hub::from_vars(&vars, &paths),
        paths,
        device: Some(DeviceMemory {
            physical: 48 * GIB,
            recommended_working_set: 48 * GIB,
            available: None,
            estimated: false,
        }),
        free_space: Some(u64::MAX),
    }
}

fn never(_: &str) -> Result<bool, CliError> {
    panic!("no confirmation expected")
}

async fn pull_tiny(ctx: &Ctx) -> Result<Manifest, CliError> {
    pull(ctx, REPO, &mut never).await
}

#[tokio::test]
async fn pull_downloads_verifies_links_and_records_a_manifest() {
    let mock = Arc::new(Mock::new("qwen3"));
    let url = serve(mock.clone()).await;
    let root = tempfile::tempdir().unwrap();
    let ctx = ctx(root.path(), &url);
    assert_eq!(ctx.hub.endpoint, url, "HF_ENDPOINT selects the mirror");

    let manifest = pull_tiny(&ctx).await.unwrap();
    assert_eq!(manifest.alias, None);
    assert_eq!(manifest.revision, COMMIT);
    assert_eq!(manifest.family, "qwen3");
    assert_eq!(manifest.syntax, "hermes");
    assert_eq!(manifest.license, "apache-2.0");
    assert_eq!(manifest.quantization.as_ref().unwrap().bits, Some(4));
    assert_eq!(manifest.identity.len(), 64);
    let paths: Vec<_> = manifest.files.iter().map(|f| f.path.as_str()).collect();
    assert_eq!(
        paths,
        ["config.json", "tokenizer.json", "model.safetensors"]
    );
    // config.json is fetched first (the gate), README.md never.
    let log = mock.log();
    assert!(log[0].starts_with("/api/models/org/tiny/revision/main"));
    assert!(log[1].contains("/config.json"));
    assert!(!log.iter().any(|line| line.contains("README")));

    // mrun's store: the weights realigned for mapping, their pad recorded; the Hugging Face
    // cache is not written.
    let dir = ctx.paths.models().join("org--tiny").join(COMMIT);
    assert_eq!(manifest.dir(&ctx.paths), dir);
    let weights = dir.join("model.safetensors");
    assert_eq!(data_start(&weights) % 16, 0);
    let pad = manifest.files[2].pad;
    assert!(pad > 0, "MLX's unpadded header needs a pad");
    assert_eq!(
        fs::metadata(&weights).unwrap().len(),
        safetensors(7, 64 * 1024).len() as u64 + u64::from(pad)
    );
    assert_eq!(fs::read(dir.join("config.json")).unwrap(), config("qwen3"));
    assert!(!ctx.paths.hf_cache.exists());
    assert!(ctx.paths.manifests().join("org--tiny.toml").exists());

    let rows = pulled_rows(&ctx);
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].status, "ready");
    assert_eq!(rows[0].fits, Some(true));
    assert_eq!(rows[0].max_context, Some(4096));

    // Already in the store: verified and recorded without downloading again.
    let requests = mock.log().len();
    pull_tiny(&ctx).await.unwrap();
    let again = &mock.log()[requests..];
    assert_eq!(again.len(), 2, "metadata and config.json only: {again:?}");

    // Each file is locked while it is written; no lock is left behind.
    assert!(!dir.join("model.safetensors.lock").exists());
}

#[tokio::test]
async fn a_pull_by_commit_lands_in_the_store() {
    let mock = Arc::new(Mock::new("qwen3"));
    let url = serve(mock.clone()).await;
    let root = tempfile::tempdir().unwrap();
    let ctx = ctx(root.path(), &url);
    let manifest = pull(&ctx, &format!("{REPO}@{COMMIT}"), &mut never)
        .await
        .unwrap();
    assert!(manifest.dir(&ctx.paths).join("config.json").exists());
    assert!(!ctx.paths.hf_cache.exists());
}

/// The mock's files in a Hugging Face cache layout: blobs named by hash, snapshot links to them.
fn cache_like_huggingface_hub(paths: &Paths, mock: &Mock) {
    let repo = paths.repo_dir(REPO);
    let snapshot = repo.join("snapshots").join(COMMIT);
    fs::create_dir_all(repo.join("blobs")).unwrap();
    fs::create_dir_all(&snapshot).unwrap();
    for (name, content, lfs) in &mock.files {
        let blob = if *lfs {
            sha256_bytes(content)
        } else {
            git_blob_id(content)
        };
        fs::write(repo.join("blobs").join(&blob), content).unwrap();
        std::os::unix::fs::symlink(Path::new("../../blobs").join(&blob), snapshot.join(name))
            .unwrap();
    }
}

/// Every file under `dir` with its bytes and modification time.
fn snapshot_of(dir: &Path) -> Vec<(std::path::PathBuf, Vec<u8>, std::time::SystemTime)> {
    let mut files: Vec<_> = files_under(dir)
        .into_iter()
        .map(|path| {
            let meta = fs::metadata(&path).unwrap();
            (
                path.clone(),
                fs::read(&path).unwrap(),
                meta.modified().unwrap(),
            )
        })
        .collect();
    files.sort();
    files
}

/// 022 FR-013: a model the Hugging Face cache holds is imported without downloading its files,
/// realigned into the store, and the cache is left byte for byte as it was.
#[tokio::test]
async fn a_model_in_the_hugging_face_cache_is_imported_without_a_download() {
    let mock = Arc::new(Mock::new("qwen3"));
    let url = serve(mock.clone()).await;
    let root = tempfile::tempdir().unwrap();
    let ctx = ctx(root.path(), &url);
    cache_like_huggingface_hub(&ctx.paths, &mock);
    let before = snapshot_of(&ctx.paths.hf_cache);

    let manifest = pull_tiny(&ctx).await.unwrap();
    let fetched: Vec<_> = mock
        .log()
        .into_iter()
        .filter(|line| line.contains("/resolve/"))
        .collect();
    assert_eq!(fetched.len(), 1, "only the gate's config.json: {fetched:?}");
    assert_eq!(
        snapshot_of(&ctx.paths.hf_cache),
        before,
        "the cache is untouched"
    );
    let weights = manifest.dir(&ctx.paths).join("model.safetensors");
    assert_eq!(data_start(&weights) % 16, 0);
    assert_eq!(verify_model(&ctx.paths, REPO).unwrap(), 3);

    // Newer huggingface_hub links the repository's blob on into a shared store whose file has
    // another name: still imported.
    let root = tempfile::tempdir().unwrap();
    let ctx = self::ctx(root.path(), &url);
    cache_like_huggingface_hub(&ctx.paths, &mock);
    let sha = sha256_bytes(&mock.files[3].1);
    let blob = ctx.paths.repo_dir(REPO).join("blobs").join(&sha);
    let shared = ctx.paths.hf_cache.join("blobs/xx");
    fs::create_dir_all(&shared).unwrap();
    fs::rename(&blob, shared.join("other-name")).unwrap();
    std::os::unix::fs::symlink(shared.join("other-name"), &blob).unwrap();
    let before = mock.weight_requests();
    pull_tiny(&ctx).await.unwrap();
    assert_eq!(
        mock.weight_requests(),
        before,
        "imported through the shared store"
    );
    assert_eq!(verify_model(&ctx.paths, REPO).unwrap(), 3);

    // A cached blob that does not match (here: the wrong size) is downloaded instead.
    let root = tempfile::tempdir().unwrap();
    let ctx = self::ctx(root.path(), &url);
    cache_like_huggingface_hub(&ctx.paths, &mock);
    let blob = ctx
        .paths
        .repo_dir(REPO)
        .join("blobs")
        .join(sha256_bytes(&mock.files[3].1));
    fs::write(&blob, b"short").unwrap();
    let before = mock.weight_requests();
    pull_tiny(&ctx).await.unwrap();
    assert_eq!(mock.weight_requests(), before + 1);
    assert_eq!(verify_model(&ctx.paths, REPO).unwrap(), 3);
}

/// Rows of pulled models (`ls` also lists the curated aliases).
fn pulled_rows(ctx: &Ctx) -> Vec<ls::Row> {
    ls::rows(ctx)
        .into_iter()
        .filter(|row| row.status != "not pulled")
        .collect()
}

#[tokio::test]
async fn unsupported_architecture_is_refused_before_any_weight() {
    let mock = Arc::new(Mock::new("llama4"));
    let url = serve(mock.clone()).await;
    let root = tempfile::tempdir().unwrap();
    let ctx = ctx(root.path(), &url);

    let error = pull_tiny(&ctx).await.unwrap_err();
    assert_eq!(error.code, Code::UnsupportedArchitecture);
    assert_eq!(error.exit() as u8, 3);
    let text = error.to_string();
    assert!(
        text.starts_with(
            "error[E034]: unsupported architecture 'llama4' in org/tiny (nothing downloaded)\n  supported families: "
        ),
        "{text}"
    );
    assert_eq!(mock.weight_requests(), 0);
    assert!(!ctx.paths.models().exists());
    assert!(pulled_rows(&ctx).is_empty(), "no manifest written");
}

#[tokio::test]
async fn unsupported_quantization_mode_is_refused() {
    let mut mock = Mock::new("qwen3");
    let mut config: serde_json::Value = serde_json::from_slice(&mock.files[0].1).unwrap();
    config["quantization"]["mode"] = json!("nvfp4");
    mock.files[0].1 = config.to_string().into_bytes();
    let mock = Arc::new(mock);
    let url = serve(mock.clone()).await;
    let root = tempfile::tempdir().unwrap();
    let error = pull_tiny(&ctx(root.path(), &url)).await.unwrap_err();
    assert_eq!(error.code, Code::UnsupportedArchitecture);
    assert!(error.message.contains("'nvfp4'"), "{error}");
    assert_eq!(mock.weight_requests(), 0);
}

#[tokio::test]
async fn an_interrupted_download_resumes_without_refetching_completed_bytes() {
    let mock = Arc::new(Mock::new("qwen3"));
    mock.cut_once
        .lock()
        .unwrap()
        .insert("model.safetensors".into());
    let url = serve(mock.clone()).await;
    let root = tempfile::tempdir().unwrap();
    let ctx = ctx(root.path(), &url);

    // The cut is retried within the same pull, from where it stopped.
    pull_tiny(&ctx).await.unwrap();
    let dir = ctx.paths.models().join("org--tiny").join(COMMIT);
    assert!(!dir.join("model.safetensors.incomplete").exists());
    let requests = mock
        .log()
        .into_iter()
        .filter(|line| line.contains("model.safetensors"))
        .collect::<Vec<_>>();
    assert_eq!(requests.len(), 2);
    assert!(
        requests[1].ends_with(&format!("bytes={}-", mock.files[3].1.len() / 2)),
        "{requests:?}"
    );
    assert_eq!(verify_model(&ctx.paths, REPO).unwrap(), 3);
}

#[tokio::test]
async fn a_download_with_the_wrong_hash_is_removed_and_fetched_again() {
    let mock = Arc::new(Mock::new("qwen3"));
    mock.corrupt_once
        .lock()
        .unwrap()
        .insert("model.safetensors".into());
    let url = serve(mock.clone()).await;
    let root = tempfile::tempdir().unwrap();
    let ctx = ctx(root.path(), &url);

    pull_tiny(&ctx).await.unwrap();
    assert_eq!(
        mock.weight_requests(),
        2,
        "fetched again after the mismatch"
    );
    let dir = ctx.paths.models().join("org--tiny").join(COMMIT);
    assert!(!fs::read_dir(&dir).unwrap().any(|entry| {
        entry
            .unwrap()
            .file_name()
            .to_string_lossy()
            .ends_with(".incomplete")
    }));
    assert_eq!(verify_model(&ctx.paths, REPO).unwrap(), 3);
}

#[tokio::test]
async fn corruption_is_repaired_by_pulling_again() {
    let mock = Arc::new(Mock::new("qwen3"));
    let url = serve(mock.clone()).await;
    let root = tempfile::tempdir().unwrap();
    let ctx = ctx(root.path(), &url);
    let manifest = pull_tiny(&ctx).await.unwrap();

    let weights = manifest.dir(&ctx.paths).join("model.safetensors");
    let mut content = fs::read(&weights).unwrap();
    content[100] ^= 1;
    fs::write(&weights, &content).unwrap();
    assert_eq!(pulled_rows(&ctx)[0].status, "corrupt");
    let error = mrun::hub::resolve(&ctx.paths, REPO).unwrap_err();
    assert_eq!(error.code, Code::ModelCorrupt);
    assert_eq!(
        error.hint.as_deref(),
        Some("run 'mrun pull org/tiny' to repair it")
    );

    let before = mock.weight_requests();
    pull_tiny(&ctx).await.unwrap();
    assert_eq!(
        mock.weight_requests(),
        before + 1,
        "the bad blob is fetched again"
    );
    assert_eq!(pulled_rows(&ctx)[0].status, "ready");
    assert!(mrun::hub::resolve(&ctx.paths, REPO).is_ok());
}

/// T012: a present file whose size and modification time match the manifest is trusted without
/// hashing; one whose time changed is hashed again, alone, and its time is refreshed when it
/// matches.
#[tokio::test]
async fn present_blobs_are_hashed_only_when_their_time_changed() {
    let mut mock = Mock::new("qwen3");
    mock.files.push((
        "model-2.safetensors".into(),
        safetensors(9, 32 * 1024),
        true,
    ));
    let mock = Arc::new(mock);
    let url = serve(mock.clone()).await;
    let root = tempfile::tempdir().unwrap();
    let ctx = ctx(root.path(), &url);
    let manifest = pull_tiny(&ctx).await.unwrap();
    let snapshot = manifest.dir(&ctx.paths);
    let recorded = |manifest: &Manifest, path: &str| {
        manifest
            .files
            .iter()
            .find(|f| f.path == path)
            .unwrap()
            .mtime_ns
    };
    let set_mtime = |path: &Path, time: std::time::SystemTime| {
        fs::OpenOptions::new()
            .write(true)
            .open(path)
            .unwrap()
            .set_modified(time)
            .unwrap();
    };

    // Same size, same time, different bytes: trusted without a hash (nothing fetched).
    let first = snapshot.join("model.safetensors");
    let time = fs::metadata(&first).unwrap().modified().unwrap();
    let mut bytes = fs::read(&first).unwrap();
    let last = bytes.len() - 1;
    bytes[last] ^= 1;
    fs::write(&first, &bytes).unwrap();
    set_mtime(&first, time);
    let before = mock.weight_requests();
    pull_tiny(&ctx).await.unwrap();
    assert_eq!(
        mock.weight_requests(),
        before,
        "matching size and time: no hash"
    );

    // The second file's time changes, its bytes do not: hashed alone, it matches, nothing is
    // fetched and the manifest records the new time. The first is still not hashed.
    let second = snapshot.join("model-2.safetensors");
    let later = std::time::SystemTime::now() + std::time::Duration::from_secs(60);
    set_mtime(&second, later);
    let refreshed = pull_tiny(&ctx).await.unwrap();
    assert_eq!(mock.weight_requests(), before);
    assert_ne!(
        recorded(&refreshed, "model-2.safetensors"),
        recorded(&manifest, "model-2.safetensors"),
        "the time is refreshed after a matching hash"
    );

    // Now the first file's time changes: hashed, mismatched, fetched again (only it).
    set_mtime(&first, later);
    pull_tiny(&ctx).await.unwrap();
    let fetched: Vec<String> = mock.log()[..]
        .iter()
        .filter(|line| line.contains(".safetensors"))
        .skip(before)
        .cloned()
        .collect();
    assert_eq!(fetched.len(), 1, "{fetched:?}");
    assert!(fetched[0].contains("/model.safetensors"), "{fetched:?}");
}

/// T050: the plan line before, the result line after, and `already pulled` (exit 0) for an
/// intact model, as `mrun pull` prints them.
#[tokio::test]
async fn pull_prints_the_plan_the_result_and_already_pulled() {
    let mock = Arc::new(Mock::new("qwen3"));
    let url = serve(mock.clone()).await;
    let root = tempfile::tempdir().unwrap();
    let run = |root: std::path::PathBuf, url: String| {
        tokio::task::spawn_blocking(move || {
            std::process::Command::new(env!("CARGO_BIN_EXE_mrun"))
                .args(["pull", REPO])
                .env("HOME", &root)
                .env("MRUN_HOME", root.join(".mrun"))
                .env("HF_ENDPOINT", &url)
                .env_remove("HF_HOME")
                .env_remove("HF_HUB_CACHE")
                .env_remove("HF_TOKEN")
                .env_remove("XDG_CACHE_HOME")
                .output()
                .unwrap()
        })
    };
    let out = run(root.path().to_path_buf(), url.clone()).await.unwrap();
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "{stderr}");
    let lines: Vec<&str> = stderr
        .lines()
        .filter(|line| !line.starts_with("  ") && !line.starts_with("note: "))
        .collect();
    assert_eq!(lines.len(), 2, "{stderr}");
    assert!(
        lines[0].starts_with("org/tiny@0123456, ")
            && lines[0].contains(", license apache-2.0, fits with 4,096 tokens"),
        "{stderr}"
    );
    assert_eq!(
        lines[1],
        "ready: org/tiny (3 files verified) — run 'mrun launch'"
    );

    let requests = mock.log().len();
    let out = run(root.path().to_path_buf(), url).await.unwrap();
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "{stderr}");
    assert!(
        stderr.contains("already pulled: org/tiny is intact — run 'mrun launch'"),
        "{stderr}"
    );
    assert_eq!(
        mock.log().len() - requests,
        2,
        "metadata and config.json only"
    );
}

#[tokio::test]
async fn too_little_disk_space_stops_before_downloading() {
    let mock = Arc::new(Mock::new("qwen3"));
    let url = serve(mock.clone()).await;
    let root = tempfile::tempdir().unwrap();
    let ctx = Ctx {
        free_space: Some(1000),
        ..ctx(root.path(), &url)
    };
    let error = pull_tiny(&ctx).await.unwrap_err();
    assert_eq!(error.code, Code::DiskFull);
    assert_eq!(error.exit() as u8, 6);
    assert!(
        error.message.contains("needed, 1000 B available"),
        "{error}"
    );
    assert_eq!(mock.weight_requests(), 0);
}

#[tokio::test]
async fn gated_repos_need_a_token() {
    let mock = Arc::new(Mock {
        token: Some("hf_secret".into()),
        ..Mock::new("qwen3")
    });
    let url = serve(mock.clone()).await;
    let root = tempfile::tempdir().unwrap();
    let ctx = ctx(root.path(), &url);
    let error = pull_tiny(&ctx).await.unwrap_err();
    assert_eq!(error.code, Code::Gated);
    assert_eq!(error.exit() as u8, 6);
    assert!(error.hint.unwrap().contains("HF_TOKEN"));

    // The token file saved by `hf auth login` is used.
    fs::create_dir_all(&ctx.paths.hf_home).unwrap();
    fs::write(ctx.paths.hf_home.join("token"), "hf_secret\n").unwrap();
    let ctx = Ctx {
        hub: Hub::from_vars(
            &|key: &str| (key == "HF_ENDPOINT").then(|| url.clone()),
            &ctx.paths,
        ),
        ..ctx
    };
    pull_tiny(&ctx).await.unwrap();
}

#[tokio::test]
async fn a_model_that_does_not_fit_needs_confirmation() {
    let mock = Arc::new(Mock::new("qwen3"));
    let url = serve(mock.clone()).await;
    let root = tempfile::tempdir().unwrap();
    let ctx = Ctx {
        // 2.5 GiB: physical − 2 GiB leaves 0.5 GiB, below the 1 GiB scratch estimate.
        device: Some(DeviceMemory {
            physical: 5 * GIB / 2,
            recommended_working_set: 5 * GIB / 2,
            available: None,
            estimated: false,
        }),
        ..ctx(root.path(), &url)
    };
    let mut no_terminal = |_: &str| -> Result<bool, CliError> {
        Err(CliError::new(Code::NeedsConfirmation, "no terminal"))
    };
    let error = pull(&ctx, REPO, &mut no_terminal).await.unwrap_err();
    assert_eq!(error.code, Code::DoesNotFit);
    assert_eq!(error.exit() as u8, 4);
    assert_eq!(
        error.hint.as_deref(),
        Some("pass --yes to download it anyway")
    );
    assert_eq!(mock.weight_requests(), 0);

    let mut yes = |_: &str| -> Result<bool, CliError> { Ok(true) };
    pull(&ctx, REPO, &mut yes).await.unwrap();
}

#[tokio::test]
async fn offline_mode_and_unknown_names_fail_fast() {
    let mock = Arc::new(Mock::new("qwen3"));
    let url = serve(mock.clone()).await;
    let root = tempfile::tempdir().unwrap();
    let mut ctx = ctx(root.path(), &url);
    ctx.hub.offline = true;
    let error = pull_tiny(&ctx).await.unwrap_err();
    assert_eq!(error.code, Code::Network);
    assert!(error.message.contains("HF_HUB_OFFLINE"));

    ctx.hub.offline = false;
    let error = pull(&ctx, "qwen3-codr", &mut never).await.unwrap_err();
    assert_eq!(error.code, Code::ModelNotFound);
    let error = pull(&ctx, "org/missing", &mut never).await.unwrap_err();
    assert_eq!(
        error.code,
        Code::ModelNotFound,
        "X-Error-Code beats the 401"
    );
    assert!(error.message.contains("does not exist"), "{error}");
    let error = pull(&ctx, &format!("{REPO}@nope"), &mut never)
        .await
        .unwrap_err();
    assert_eq!(error.code, Code::ModelNotFound);
    assert!(error.message.contains("no such revision"), "{error}");
    assert!(mock.log().iter().all(|line| !line.contains("resolve")));
}

#[tokio::test]
async fn hostile_metadata_never_reaches_a_path() {
    let root = tempfile::tempdir().unwrap();
    for (sha, lfs_sha, what) in [
        (Some("../../../../escaped"), None, "commit id"),
        (Some(&"A".repeat(40) as &str), None, "commit id"),
        (None, Some("../../../../escaped"), "invalid blob id"),
    ] {
        let mock = Arc::new(Mock {
            sha: sha.map(str::to_string),
            lfs_sha: lfs_sha.map(str::to_string),
            ..Mock::new("qwen3")
        });
        let url = serve(mock.clone()).await;
        let ctx = ctx(root.path(), &url);
        let error = pull_tiny(&ctx).await.unwrap_err();
        assert_eq!(error.code, Code::Network, "{error}");
        assert!(error.message.contains(what), "{error}");
        assert!(
            mock.log().iter().all(|line| !line.contains("/resolve/")),
            "nothing is fetched"
        );
    }
    assert!(!root.path().join("escaped").exists());
    let error = pull(
        &ctx(root.path(), "http://127.0.0.1:9"),
        "org/tiny@../x",
        &mut never,
    )
    .await
    .unwrap_err();
    assert_eq!(error.code, Code::InvalidArguments);
}

/// Every file under `dir`.
fn files_under(dir: &Path) -> Vec<std::path::PathBuf> {
    let mut found = Vec::new();
    let mut todo = vec![dir.to_path_buf()];
    while let Some(dir) = todo.pop() {
        for entry in fs::read_dir(&dir).into_iter().flatten().flatten() {
            let path = entry.path();
            if path.is_dir() {
                todo.push(path);
            } else {
                found.push(path);
            }
        }
    }
    found
}

/// 020 FR-005, SC-004: `pull --dry-run` prints the plan (with the family) and writes nothing,
/// with and without `--json`, and a model that does not fit still exits 0 without asking.
#[tokio::test]
async fn a_dry_run_prints_the_plan_and_writes_nothing() {
    let mock = Arc::new(Mock::new("qwen3"));
    let url = serve(mock.clone()).await;
    let root = tempfile::tempdir().unwrap();
    let out = run_with(root.path(), &url, &["pull", REPO, "--dry-run", "--json"]).await;
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8(out.stdout).unwrap();
    let lines: Vec<serde_json::Value> = stdout
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    assert_eq!(lines.len(), 1, "{stdout}");
    let plan = &lines[0];
    assert_eq!(plan["event"], "plan");
    assert_eq!(plan["model"], REPO);
    assert_eq!(plan["repo"], REPO);
    assert_eq!(plan["revision"], COMMIT);
    assert_eq!(plan["files"], 3);
    assert_eq!(plan["family"], "qwen3");
    assert!(plan["bytes"].as_u64().unwrap() > 64 * 1024, "{plan}");
    assert!(
        plan["verdict"].is_string() && plan.get("fit").is_some(),
        "{plan}"
    );

    let out = run_with(root.path(), &url, &["pull", REPO, "--dry-run"]).await;
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "{stderr}");
    assert!(
        stderr.lines().any(|l| l.starts_with("org/tiny@0123456, ")),
        "{stderr}"
    );
    assert!(!stderr.contains("ready:"), "{stderr}");

    assert_eq!(mock.weight_requests(), 0);
    assert_eq!(files_under(root.path()), Vec::<std::path::PathBuf>::new());

    // Too big for this machine: still the plan, no question, no error.
    let small = Ctx {
        device: Some(DeviceMemory {
            physical: 5 * GIB / 2,
            recommended_working_set: 5 * GIB / 2,
            available: None,
            estimated: false,
        }),
        ..ctx(root.path(), &url)
    };
    let buffer = Arc::new(Mutex::new(Vec::new()));
    let plan = mrun::cli::pull::plan_json(
        &small,
        REPO,
        &mrun::hub::progress::JsonLines::to(Shared(buffer.clone())),
    )
    .await
    .unwrap();
    assert_eq!(plan.repo, REPO);
    let text = String::from_utf8(buffer.lock().unwrap().clone()).unwrap();
    let event: serde_json::Value = serde_json::from_str(text.trim()).unwrap();
    assert_eq!(event["fit"], "wont_fit");
    assert_eq!(mock.weight_requests(), 0);
    assert!(files_under(root.path()).is_empty());

    // Errors are pull's: an unsupported architecture is one `error` event.
    let mock = Arc::new(Mock::new("llama"));
    let url = serve(mock.clone()).await;
    let out = run_with(root.path(), &url, &["pull", REPO, "--dry-run", "--json"]).await;
    assert!(!out.status.success());
    let stdout = String::from_utf8(out.stdout).unwrap();
    let event: serde_json::Value = serde_json::from_str(stdout.trim()).unwrap();
    assert_eq!(event["event"], "error");
    assert!(event["hint"].is_string(), "{event}");
    assert_eq!(mock.weight_requests(), 0);
}

async fn run_with(root: &Path, url: &str, args: &[&str]) -> std::process::Output {
    let (root, url) = (root.to_path_buf(), url.to_string());
    let args: Vec<String> = args.iter().map(|a| (*a).to_string()).collect();
    tokio::task::spawn_blocking(move || {
        std::process::Command::new(env!("CARGO_BIN_EXE_mrun"))
            .args(&args)
            .env("HOME", &root)
            .env("MRUN_HOME", root.join(".mrun"))
            .env("HF_ENDPOINT", &url)
            .env_remove("HF_HOME")
            .env_remove("HF_HUB_CACHE")
            .env_remove("HF_TOKEN")
            .env_remove("XDG_CACHE_HOME")
            .stdin(std::process::Stdio::null())
            .output()
            .unwrap()
    })
    .await
    .unwrap()
}

/// A writer into a shared buffer.
struct Shared(Arc<Mutex<Vec<u8>>>);

impl std::io::Write for Shared {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0.lock().unwrap().extend_from_slice(buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

/// A sharded model: layer 0 and the embeddings in one shard, layer 1 and the head in the other.
fn sharded() -> Mock {
    let index = json!({"weight_map": {
        "model.embed_tokens.weight": "model-00001-of-00002.safetensors",
        "model.layers.0.self_attn.q_proj.weight": "model-00001-of-00002.safetensors",
        "model.layers.1.self_attn.q_proj.weight": "model-00002-of-00002.safetensors",
        "lm_head.weight": "model-00002-of-00002.safetensors",
    }});
    let mut mock = Mock::new("qwen3");
    mock.files
        .retain(|(name, ..)| !name.ends_with(".safetensors"));
    mock.files.extend([
        (
            "model.safetensors.index.json".into(),
            index.to_string().into_bytes(),
            false,
        ),
        (
            "model-00001-of-00002.safetensors".into(),
            safetensors(1, 4096),
            true,
        ),
        (
            "model-00002-of-00002.safetensors".into(),
            safetensors(2, 4096),
            true,
        ),
    ]);
    mock
}

/// 026 T029: a pool member downloads only the shards of its layers, kept as a part in the store
/// (listed by `mrun ls`); a whole pull completes it with the files it has.
#[tokio::test]
async fn a_part_downloads_only_its_layers_and_a_pull_completes_it() {
    let mock = Arc::new(sharded());
    let url = serve(mock.clone()).await;
    let root = tempfile::tempdir().unwrap();
    let ctx = ctx(root.path(), &url);
    let report = mrun::hub::progress::Report::default();
    let part = mrun::hub::download::pull_part(&ctx, REPO, 1, 2, &report)
        .await
        .unwrap();
    assert_eq!(part.part, Some([1, 2]));
    let files: Vec<&str> = part.files.iter().map(|f| f.path.as_str()).collect();
    assert!(
        files.contains(&"model-00002-of-00002.safetensors"),
        "{files:?}"
    );
    assert!(
        !files.contains(&"model-00001-of-00002.safetensors"),
        "{files:?}"
    );
    assert!(
        !mock.log().iter().any(|l| l.contains("model-00001")),
        "the other shard is not downloaded"
    );
    let rows = pulled_rows(&ctx);
    assert_eq!(rows[0].part, Some([1, 2]));
    assert!(
        ls::render(&rows, "qwen3.6", None).contains("layers 2–2 · for pools"),
        "{}",
        ls::render(&rows, "qwen3.6", None)
    );

    // The whole model: only the missing shard comes down, and the manifest is whole.
    let before = mock.weight_requests();
    let whole = pull_tiny(&ctx).await.unwrap();
    assert_eq!(whole.part, None);
    assert_eq!(whole.files.len(), part.files.len() + 1);
    assert_eq!(mock.weight_requests(), before + 1, "{:?}", mock.log());
    assert_eq!(
        pulled_rows(&ctx).len(),
        1,
        "one model, not a part beside it"
    );
}
