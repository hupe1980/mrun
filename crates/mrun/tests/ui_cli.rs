//! 018 CLI additions the UI host reads: `pull --json` against a mock hub on 127.0.0.1,
//! `ls --all --json` and `connect <client> --json` in a temporary home. No real network.

use axum::extract::{Request, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use mrun::cli::pull::pull_json;
use mrun::errors::Code;
use mrun::hub::progress::JsonLines;
use mrun::hub::verify::{git_sha1_file, sha256_bytes};
use mrun::hub::{Ctx, Hub, Paths};
use mrun_engine::budget::{DeviceMemory, GIB};
use serde_json::{Value, json};
use std::io::Write;
use std::path::Path;
use std::process::Command;
use std::sync::{Arc, Mutex};

const REPO: &str = "org/tiny";
const COMMIT: &str = "0123456789abcdef0123456789abcdef01234567";

/// (path, content, stored in LFS)
type Files = Vec<(&'static str, Vec<u8>, bool)>;

fn files() -> Files {
    let config = json!({
        "model_type": "qwen3", "num_hidden_layers": 2, "num_attention_heads": 4,
        "num_key_value_heads": 2, "head_dim": 8, "hidden_size": 32, "vocab_size": 64,
        "max_position_embeddings": 4096, "torch_dtype": "bfloat16",
        "quantization": {"bits": 4, "group_size": 64}
    });
    vec![
        ("config.json", config.to_string().into_bytes(), false),
        ("tokenizer.json", b"{}".to_vec(), false),
        ("model.safetensors", vec![7; 64 * 1024], true),
    ]
}

fn git_blob_id(content: &[u8]) -> String {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("f");
    std::fs::write(&path, content).unwrap();
    git_sha1_file(&path).unwrap()
}

/// The hub API's metadata and `resolve` routes for [`REPO`] at [`COMMIT`].
async fn handle(State(files): State<Arc<Files>>, request: Request) -> Response {
    let path = request.uri().path().to_string();
    if path.starts_with(&format!("/api/models/{REPO}/revision/")) {
        let siblings: Vec<Value> = files
            .iter()
            .map(|(name, content, lfs)| {
                let mut sibling =
                    json!({"rfilename": name, "size": content.len(), "blobId": git_blob_id(content)});
                if *lfs {
                    sibling["lfs"] = json!({"sha256": sha256_bytes(content), "size": content.len()});
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
    match files.iter().find(|(name, ..)| *name == file) {
        Some((_, content, _)) => content.clone().into_response(),
        None => StatusCode::NOT_FOUND.into_response(),
    }
}

async fn serve() -> String {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let app = axum::Router::new()
        .fallback(handle)
        .with_state(Arc::new(files()));
    tokio::spawn(async move { axum::serve(listener, app).await });
    url
}

/// `mrun` with a temporary home, `HF_ENDPOINT=url` and no ambient HF settings.
fn mrun(root: &Path, url: &str) -> Command {
    let mut command = Command::new(env!("CARGO_BIN_EXE_mrun"));
    command
        .env("HOME", root)
        .env("MRUN_HOME", root.join(".mrun"))
        .env("HF_ENDPOINT", url)
        .env("CODEX_HOME", root.join(".codex"))
        .env("XDG_CONFIG_HOME", root.join(".config"))
        .env("MRUN_PORT", free_port().to_string())
        .env_remove("HF_HOME")
        .env_remove("HF_HUB_CACHE")
        .env_remove("HF_TOKEN")
        .env_remove("XDG_CACHE_HOME")
        .env_remove("MRUN_BUDGET_GIB");
    command
}

/// A port nothing listens on.
fn free_port() -> u16 {
    std::net::TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port()
}

fn json_lines(stdout: &[u8]) -> Vec<Value> {
    String::from_utf8_lossy(stdout)
        .lines()
        .map(|line| serde_json::from_str(line).unwrap_or_else(|_| panic!("not JSON: {line}")))
        .collect()
}

#[tokio::test]
async fn pull_json_prints_plan_progress_and_done_lines() {
    let url = serve().await;
    let root = tempfile::tempdir().unwrap();
    let run = |root: std::path::PathBuf, url: String| {
        tokio::task::spawn_blocking(move || {
            mrun(&root, &url)
                .args(["pull", REPO, "--json"])
                .output()
                .unwrap()
        })
    };
    let out = run(root.path().to_path_buf(), url.clone()).await.unwrap();
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "{stderr}");
    assert!(!stderr.contains("ready:"), "no human lines: {stderr}");
    let lines = json_lines(&out.stdout);
    let plan = &lines[0];
    assert_eq!(plan["event"], "plan");
    assert_eq!(plan["model"], REPO);
    assert_eq!(plan["files"], 3);
    let total: u64 = files()
        .iter()
        .map(|(_, content, _)| content.len() as u64)
        .sum();
    assert_eq!(plan["bytes"], total);
    assert!(
        plan["verdict"]
            .as_str()
            .unwrap()
            .starts_with("Fits with 4,096 tokens"),
        "{plan}"
    );
    let progress: Vec<&Value> = lines.iter().filter(|l| l["event"] == "progress").collect();
    for (name, content, _) in files() {
        let of_file: Vec<_> = progress.iter().filter(|l| l["file"] == name).collect();
        assert!(!of_file.is_empty(), "{name}: {lines:?}");
        let last = of_file.last().unwrap();
        assert_eq!(last["bytes"], content.len(), "{name}");
        assert_eq!(last["total"], content.len());
        assert!(last["rate"].is_u64());
    }
    let done = lines.last().unwrap();
    assert_eq!(done["event"], "done");
    assert_eq!(done["model"], REPO);
    assert_eq!(done["already_pulled"], false);
    assert!(
        Path::new(done["path"].as_str().unwrap())
            .join("config.json")
            .is_file()
    );

    // Again: plan and done only.
    let out = run(root.path().to_path_buf(), url).await.unwrap();
    let lines = json_lines(&out.stdout);
    let events: Vec<_> = lines.iter().map(|l| l["event"].as_str().unwrap()).collect();
    assert_eq!(events, ["plan", "done"]);
    assert_eq!(lines[1]["already_pulled"], true);
}

/// A buffer the JSON lines go to.
#[derive(Clone, Default)]
struct Buffer(Arc<Mutex<Vec<u8>>>);

impl Write for Buffer {
    fn write(&mut self, data: &[u8]) -> std::io::Result<usize> {
        self.0.lock().unwrap().extend_from_slice(data);
        Ok(data.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

#[tokio::test]
async fn pull_json_without_yes_prints_confirm_and_needs_yes() {
    let url = serve().await;
    let root = tempfile::tempdir().unwrap();
    let home = root.path().to_str().unwrap().to_string();
    let vars = move |key: &str| match key {
        "HOME" => Some(home.clone()),
        "HF_ENDPOINT" => Some(url.clone()),
        _ => None,
    };
    let paths = Paths::from_vars(&vars);
    let ctx = Ctx {
        hub: Hub::from_vars(&vars, &paths),
        paths,
        // 2.5 GiB: too small for the scratch estimate, so the model does not fit.
        device: Some(DeviceMemory {
            physical: 5 * GIB / 2,
            recommended_working_set: 5 * GIB / 2,
            available: None,
            estimated: false,
        }),
        free_space: Some(u64::MAX),
    };
    let buffer = Buffer::default();
    let error = pull_json(&ctx, REPO, false, &JsonLines::to(buffer.clone()))
        .await
        .unwrap_err();
    assert_eq!(error.code, Code::NeedsConfirmation);
    assert_eq!(error.exit() as u8, 2);
    let lines = json_lines(&buffer.0.lock().unwrap());
    let events: Vec<_> = lines.iter().map(|l| l["event"].as_str().unwrap()).collect();
    assert_eq!(events, ["plan", "confirm", "error"]);
    assert_eq!(lines[0]["fit"], "wont_fit");
    assert!(
        lines[0]["verdict"]
            .as_str()
            .unwrap()
            .starts_with("Does not fit the budget of "),
        "{}",
        lines[0]
    );
    assert!(
        lines[1]["question"]
            .as_str()
            .unwrap()
            .ends_with("download anyway?")
    );
    assert_eq!(lines[2]["code"], "E002");
    assert_eq!(lines[2]["hint"], "pass --yes to download it anyway");
    assert!(!ctx.paths.repo_dir(REPO).join("snapshots").exists());

    let buffer = Buffer::default();
    pull_json(&ctx, REPO, true, &JsonLines::to(buffer.clone()))
        .await
        .unwrap();
    let lines = json_lines(&buffer.0.lock().unwrap());
    assert!(lines.iter().all(|l| l["event"] != "confirm"));
    assert_eq!(lines.last().unwrap()["event"], "done");
}

#[test]
fn ls_all_json_lists_aliases_without_the_network() {
    let root = tempfile::tempdir().unwrap();
    // Nothing answers here: `ls` must not need it.
    let url = format!("http://127.0.0.1:{}", free_port());
    let out = mrun(root.path(), &url)
        .args(["ls", "--all", "--json", "--budget-gib", "36"])
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let rows: Vec<Value> = serde_json::from_slice(&out.stdout).unwrap();
    let names: Vec<_> = rows.iter().map(|r| r["name"].as_str().unwrap()).collect();
    assert_eq!(names, ["qwen3.5", "qwen3.6"]);
    for row in &rows {
        for key in [
            "name",
            "alias",
            "repo",
            "revision",
            "size",
            "pulled",
            "fit",
            "verdict",
            "max_context",
            "recommended",
            "preview",
            "serving",
            "status",
            "fits",
        ] {
            assert!(row.get(key).is_some(), "{key} missing: {row}");
        }
        assert_eq!(row["pulled"], false);
        assert_eq!(row["serving"], false);
        assert!(row["verdict"].as_str().unwrap().ends_with('.'));
    }
    assert_eq!(rows[0]["preview"], true);
    assert_eq!(
        rows.iter().filter(|r| r["recommended"] == true).count(),
        1,
        "{rows:?}"
    );
}

#[test]
fn connect_json_shows_state_then_what_write_and_restore_did() {
    let root = tempfile::tempdir().unwrap();
    let url = format!("http://127.0.0.1:{}", free_port());
    let port = free_port();
    let empty_path = root.path().join("bin");
    std::fs::create_dir_all(&empty_path).unwrap();
    let connect = |extra: &[&str]| {
        let out = mrun(root.path(), &url)
            .env("PATH", &empty_path)
            // One port: the configuration names the server's URL.
            .env("MRUN_PORT", port.to_string())
            .args(["connect", "codex", "--json"])
            .args(extra)
            .output()
            .unwrap();
        assert!(
            out.status.success(),
            "{}",
            String::from_utf8_lossy(&out.stderr)
        );
        serde_json::from_slice::<Value>(&out.stdout).unwrap()
    };
    let profile = root.path().join(".codex/mrun.config.toml");
    let state = connect(&[]);
    assert_eq!(state["client"], "codex");
    assert_eq!(state["installed"], false);
    assert_eq!(state["version"], Value::Null);
    assert_eq!(state["connected"], false);
    assert_eq!(state["refusal"], Value::Null);
    assert_eq!(
        state["files"],
        json!([{"path": profile, "change": "create"}])
    );
    assert!(!profile.exists());

    let written = connect(&["--write", "--yes"]);
    assert_eq!(written["connected"], true);
    assert_eq!(
        written["files"],
        json!([{"path": profile, "change": "create"}])
    );
    assert!(profile.is_file());
    assert_eq!(connect(&[])["files"], json!([]));

    let restored = connect(&["--restore", "--yes"]);
    assert_eq!(
        restored["files"],
        json!([{"path": profile, "change": "restore"}])
    );
    assert_eq!(restored["connected"], false);
    assert!(!profile.exists());
}

/// 020 FR-008: `connect --json`'s `blocks` are exactly what `mrun connect <client>` prints, for
/// the print-only clients too.
#[test]
fn connect_json_blocks_are_the_printed_blocks() {
    let root = tempfile::tempdir().unwrap();
    let url = format!("http://127.0.0.1:{}", free_port());
    let port = free_port().to_string();
    for client in ["zed", "crush"] {
        let run = |extra: &[&str]| {
            let out = mrun(root.path(), &url)
                .args(["connect", client, "--port", &port])
                .args(extra)
                .output()
                .unwrap();
            assert!(
                out.status.success(),
                "{}",
                String::from_utf8_lossy(&out.stderr)
            );
            String::from_utf8(out.stdout).unwrap()
        };
        let printed = run(&[]);
        let state: Value = serde_json::from_str(&run(&["--json"])).unwrap();
        let blocks = state["blocks"].as_array().unwrap();
        assert!(!blocks.is_empty(), "{client}: {state}");
        let rebuilt: String = blocks
            .iter()
            .map(|b| {
                format!(
                    "# {} ({})\n{}\n\n",
                    b["path"].as_str().unwrap(),
                    b["note"].as_str().unwrap(),
                    b["text"].as_str().unwrap()
                )
            })
            .collect();
        assert_eq!(rebuilt, printed, "{client}");
    }
}
