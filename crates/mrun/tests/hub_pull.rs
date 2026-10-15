//! `mrun pull` against a mock hub on 127.0.0.1: metadata, the architecture gate, resume,
//! corruption, disk space, gated repos, `HF_ENDPOINT` and the manifest. No real network.

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
}

impl Mock {
    fn new(model_type: &str) -> Self {
        Self {
            files: vec![
                ("config.json".into(), config(model_type), false),
                ("tokenizer.json".into(), b"{}".to_vec(), false),
                ("README.md".into(), b"# tiny".to_vec(), false),
                ("model.safetensors".into(), vec![7; 64 * 1024], true),
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
            return StatusCode::UNAUTHORIZED.into_response();
        }
    }
    if let Some(rev) = path.strip_prefix(&format!("/api/models/{REPO}/revision/")) {
        if rev != "main" && rev != COMMIT {
            return StatusCode::NOT_FOUND.into_response();
        }
        let siblings: Vec<_> = mock
            .files
            .iter()
            .map(|(name, content, lfs)| {
                let mut sibling = json!({
                    "rfilename": name, "size": content.len(), "blobId": git_blob_id(content)
                });
                if *lfs {
                    sibling["lfs"] =
                        json!({"sha256": sha256_bytes(content), "size": content.len()});
                }
                sibling
            })
            .collect();
        return axum::Json(json!({
            "id": REPO, "sha": COMMIT, "siblings": siblings,
            "cardData": {"license": "apache-2.0"}
        }))
        .into_response();
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

    // HF cache layout: snapshot symlinks into blobs named by hash, refs/main = commit.
    let repo = ctx.paths.repo_dir(REPO);
    let weights = repo
        .join("snapshots")
        .join(COMMIT)
        .join("model.safetensors");
    let target = fs::read_link(&weights).unwrap();
    assert_eq!(
        target,
        Path::new("../../blobs").join(sha256_bytes(&vec![7; 64 * 1024]))
    );
    assert_eq!(fs::read_to_string(repo.join("refs/main")).unwrap(), COMMIT);
    assert!(ctx.paths.manifests().join("org--tiny.toml").exists());

    let rows = ls::rows(&ctx);
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].status, "ready");
    assert_eq!(rows[0].fits, Some(true));
    assert_eq!(rows[0].max_context, Some(4096));

    // Already in the cache: verified and recorded without downloading again.
    let requests = mock.log().len();
    pull_tiny(&ctx).await.unwrap();
    let again = &mock.log()[requests..];
    assert_eq!(again.len(), 2, "metadata and config.json only: {again:?}");
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
            "error[E034]: unsupported architecture 'llama4' in org/tiny (nothing downloaded)\n  supported families: qwen3_5_moe (qwen3.6), qwen3_moe (qwen3-coder), gpt_oss (gpt-oss), qwen3"
        ),
        "{text}"
    );
    assert_eq!(mock.weight_requests(), 0);
    assert!(!ctx.paths.repo_dir(REPO).join("blobs").exists());
    assert!(ls::rows(&ctx).is_empty(), "no manifest written");
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

    let error = pull_tiny(&ctx).await.unwrap_err();
    assert_eq!(error.code, Code::Network);
    assert_eq!(error.exit() as u8, 6);
    assert!(error.message.contains("interrupted"), "{error}");
    let blobs = ctx.paths.repo_dir(REPO).join("blobs");
    let sha = sha256_bytes(&vec![7; 64 * 1024]);
    let partial = blobs.join(format!("{sha}.incomplete"));
    let kept = fs::metadata(&partial).unwrap().len();
    assert_eq!(kept, 32 * 1024, "the received half is kept");

    pull_tiny(&ctx).await.unwrap();
    assert!(!partial.exists());
    let resumed = mock
        .log()
        .into_iter()
        .filter(|line| line.contains("model.safetensors"))
        .collect::<Vec<_>>();
    assert_eq!(resumed.len(), 2);
    assert!(
        resumed[1].ends_with(&format!("bytes={kept}-")),
        "{resumed:?}"
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

    let error = pull_tiny(&ctx).await.unwrap_err();
    assert!(error.message.contains("does not match its hash"), "{error}");
    let blobs = ctx.paths.repo_dir(REPO).join("blobs");
    assert!(!fs::read_dir(&blobs).unwrap().any(|entry| {
        entry
            .unwrap()
            .file_name()
            .to_string_lossy()
            .ends_with(".incomplete")
    }));
    pull_tiny(&ctx).await.unwrap();
}

#[tokio::test]
async fn corruption_is_repaired_by_pulling_again() {
    let mock = Arc::new(Mock::new("qwen3"));
    let url = serve(mock.clone()).await;
    let root = tempfile::tempdir().unwrap();
    let ctx = ctx(root.path(), &url);
    let manifest = pull_tiny(&ctx).await.unwrap();

    let weights = manifest.snapshot_dir(&ctx.paths).join("model.safetensors");
    let mut content = fs::read(&weights).unwrap();
    content[100] ^= 1;
    fs::write(&weights, &content).unwrap();
    assert_eq!(ls::rows(&ctx)[0].status, "corrupt");
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
    assert_eq!(ls::rows(&ctx)[0].status, "ready");
    assert!(mrun::hub::resolve(&ctx.paths, REPO).is_ok());
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
        device: Some(DeviceMemory {
            physical: 4 * GIB,
            recommended_working_set: 4 * GIB,
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
    assert_eq!(error.code, Code::ModelNotFound);
    assert!(mock.log().iter().all(|line| !line.contains("resolve")));
}
