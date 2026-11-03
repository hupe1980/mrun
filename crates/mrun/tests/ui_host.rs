//! The UI host (018 contracts/ui-host.md): security refusals, the page without its key, the
//! proxy to the server, and actions that run exactly the command Copy as Command shows.

use axum::Router;
use axum::routing::{get, post};
use mrun::ui::host::{Host, router};
use mrun::ui::store::Store;
use std::time::Duration;

const TOKEN: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

async fn serve(app: Router) -> u16 {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    port
}

/// A host on a free port, showing the server on `server_port`, with its home in `dir`.
async fn host(dir: &std::path::Path, server_port: u16) -> (u16, Host) {
    host_with(dir, server_port, None).await
}

async fn host_with(
    dir: &std::path::Path,
    server_port: u16,
    cli: Option<std::path::PathBuf>,
) -> (u16, Host) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let mut host = Host::new(TOKEN.into(), port, server_port, Store::open(dir));
    if let Some(cli) = cli {
        host.cli = std::sync::Arc::new(cli);
    }
    host.downloads = std::sync::Arc::new(dir.join("Downloads"));
    host.reveal = false;
    let app = router(host.clone());
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    (port, host)
}

fn client() -> reqwest::Client {
    reqwest::Client::builder().no_proxy().build().unwrap()
}

fn get_with(port: u16, path: &str, token: Option<&str>) -> reqwest::RequestBuilder {
    let mut request = client().get(format!("http://127.0.0.1:{port}{path}"));
    if let Some(token) = token {
        request = request.bearer_auth(token);
    }
    request
}

#[tokio::test]
async fn controls_need_the_key_and_the_page_never_carries_it() {
    let dir = tempfile::tempdir().unwrap();
    let (port, _) = host(dir.path(), 1).await;
    for path in [
        "/app/state",
        "/app/chats",
        "/mrun/status",
        "/health",
        "/v1/models",
    ] {
        let none = get_with(port, path, None).send().await.unwrap();
        assert_eq!(none.status(), 401, "{path} without the key");
        let wrong = get_with(port, path, Some(&"f".repeat(64)))
            .send()
            .await
            .unwrap();
        assert_eq!(wrong.status(), 401, "{path} with a wrong key");
    }
    let page = get_with(port, "/", None).send().await.unwrap();
    assert_eq!(page.status(), 200);
    let csp = page.headers()["content-security-policy"]
        .to_str()
        .unwrap()
        .to_string();
    assert!(
        csp.contains("default-src 'self'") && csp.contains("frame-ancestors 'none'"),
        "{csp}"
    );
    let html = page.text().await.unwrap();
    assert!(!html.contains(TOKEN));
    let state = get_with(port, "/app/state", Some(TOKEN))
        .send()
        .await
        .unwrap();
    assert_eq!(state.status(), 200);
    assert_eq!(state.headers()["cache-control"], "no-store");
    let body: serde_json::Value = state.json().await.unwrap();
    assert_eq!(body["server"]["running"], false);
    assert_eq!(body["prefs"]["keep_chats"], serde_json::Value::Null);
}

#[tokio::test]
async fn other_hosts_and_origins_are_refused_even_with_the_key() {
    let dir = tempfile::tempdir().unwrap();
    let (port, _) = host(dir.path(), 1).await;
    // DNS rebinding: a name that resolves to 127.0.0.1 but is not this host's address.
    let rebound = get_with(port, "/app/state", Some(TOKEN))
        .header("Host", format!("evil.example:{port}"))
        .send()
        .await
        .unwrap();
    assert_eq!(rebound.status(), 403);
    let foreign = get_with(port, "/app/state", Some(TOKEN))
        .header("Origin", "http://evil.example")
        .send()
        .await
        .unwrap();
    assert_eq!(foreign.status(), 403);
    let other_port = get_with(port, "/app/state", Some(TOKEN))
        .header("Origin", "http://127.0.0.1:1")
        .send()
        .await
        .unwrap();
    assert_eq!(other_port.status(), 403);
    let own = get_with(port, "/app/state", Some(TOKEN))
        .header("Origin", format!("http://127.0.0.1:{port}"))
        .send()
        .await
        .unwrap();
    assert_eq!(own.status(), 200);
}

#[tokio::test]
async fn the_proxy_streams_the_server_and_reports_it_down() {
    let server = Router::new()
        .route("/mrun/status", get(|| async { axum::Json(serde_json::json!({"schema_version": 1})) }))
        .route(
            "/v1/messages",
            post(|| async {
                (
                    [("content-type", "text/event-stream"), ("x-mrun-cache-cached", "7")],
                    ": mrun {\"cached\":7,\"processed\":3}\n\nevent: message_stop\ndata: {\"type\":\"message_stop\"}\n\n",
                )
            }),
        )
        .route(
            "/v1/messages/count_tokens",
            post(|| async { axum::Json(serde_json::json!({"input_tokens": 12})) }),
        );
    let server_port = serve(server).await;
    let dir = tempfile::tempdir().unwrap();
    let (port, _) = host(dir.path(), server_port).await;
    let status: serde_json::Value = get_with(port, "/mrun/status", Some(TOKEN))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(status["schema_version"], 1);
    let stream = client()
        .post(format!("http://127.0.0.1:{port}/v1/messages"))
        .bearer_auth(TOKEN)
        .header("content-type", "application/json")
        .body("{}")
        .send()
        .await
        .unwrap();
    assert_eq!(stream.headers()["content-type"], "text/event-stream");
    assert_eq!(stream.headers()["x-mrun-cache-cached"], "7");
    let text = stream.text().await.unwrap();
    assert!(text.starts_with(": mrun {\"cached\":7"), "{text}");
    // 020 R2: the chat's token counts.
    let counted: serde_json::Value = client()
        .post(format!("http://127.0.0.1:{port}/v1/messages/count_tokens"))
        .bearer_auth(TOKEN)
        .header("content-type", "application/json")
        .body("{}")
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(counted["input_tokens"], 12);
    let unkeyed = client()
        .post(format!("http://127.0.0.1:{port}/v1/messages/count_tokens"))
        .body("{}")
        .send()
        .await
        .unwrap();
    assert_eq!(unkeyed.status(), 401);
    // Nothing listens on the server port: 502, never a connection elsewhere.
    let (down, _) = host(dir.path(), 1).await;
    let response = get_with(down, "/mrun/status", Some(TOKEN))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 502);
}

#[tokio::test]
async fn chats_are_refused_until_keeping_is_chosen() {
    let dir = tempfile::tempdir().unwrap();
    let (port, _) = host(dir.path(), 1).await;
    let chat = serde_json::json!({"id": "0123456789abcdef", "title": "t", "updated": "2026-11-01"});
    let put = |port| {
        client()
            .put(format!(
                "http://127.0.0.1:{port}/app/chats/0123456789abcdef"
            ))
            .bearer_auth(TOKEN)
            .json(&chat)
    };
    assert_eq!(put(port).send().await.unwrap().status(), 409);
    let prefs = client()
        .put(format!("http://127.0.0.1:{port}/app/prefs"))
        .bearer_auth(TOKEN)
        .json(&serde_json::json!({"keep_chats": true, "last_model": "--evil"}))
        .send()
        .await
        .unwrap();
    assert_eq!(prefs.status(), 200);
    let prefs: serde_json::Value = prefs.json().await.unwrap();
    assert_eq!(
        prefs["last_model"],
        serde_json::Value::Null,
        "the page cannot set last_model"
    );
    assert_eq!(put(port).send().await.unwrap().status(), 204);
    let list: serde_json::Value = get_with(port, "/app/chats", Some(TOKEN))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(list[0]["id"], "0123456789abcdef");
}

/// A fake `mrun` that records its arguments, prints a pull's JSON lines and a listing for `ls`.
#[cfg(unix)]
fn fake_cli(dir: &std::path::Path) -> std::path::PathBuf {
    use std::os::unix::fs::PermissionsExt;
    let script = dir.join("fake-mrun");
    let log = dir.join("argv.log");
    std::fs::write(
        &script,
        format!(
            "#!/bin/sh\necho \"$@\" >> {log}\ncase \"$1 $2 $3\" in\n  'pull org/bad --dry-run') echo '{{\"event\":\"error\",\"code\":\"E040\",\"message\":\"unsupported architecture\",\"hint\":\"pick another\"}}'; exit 5;;\n  *--dry-run) echo '{{\"event\":\"plan\",\"model\":\"org/ok\",\"family\":\"qwen3\",\"bytes\":10}}'; exit 0;;\n  'connect zed '*|'connect crush '*) echo \"{{\\\"client\\\":\\\"$2\\\",\\\"blocks\\\":[]}}\"; exit 0;;\nesac\ncase \"$1\" in\n  pull) echo '{{\"event\":\"progress\",\"file\":\"a\",\"bytes\":5,\"total\":10,\"rate\":5}}'; echo '{{\"event\":\"done\",\"model\":\"m\"}}';;\n  rm) echo 'error: no such model' >&2; exit 3;;\n  ls) echo '[{{\"name\":\"m\",\"serving\":false}}]';;\nesac\nexit 0\n",
            log = log.display()
        ),
    )
    .unwrap();
    std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o755)).unwrap();
    script
}

#[cfg(unix)]
#[tokio::test]
async fn actions_run_exactly_the_command_they_show() {
    let dir = tempfile::tempdir().unwrap();
    let (port, _) = host_with(dir.path(), 1, Some(fake_cli(dir.path()))).await;
    for (kind, target) in [
        ("pull", "qwen3.6"),
        ("connect", "claude-code"),
        ("remove", "qwen3.5"),
    ] {
        let shown: serde_json::Value = get_with(
            port,
            &format!("/app/command?kind={kind}&target={target}"),
            Some(TOKEN),
        )
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
        let started: serde_json::Value = client()
            .post(format!("http://127.0.0.1:{port}/app/actions"))
            .bearer_auth(TOKEN)
            .json(&serde_json::json!({"kind": kind, "target": target}))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        let id = started["job"]["id"].as_str().unwrap().to_string();
        let mut job = serde_json::Value::Null;
        for _ in 0..50 {
            job = get_with(port, &format!("/app/jobs/{id}"), Some(TOKEN))
                .send()
                .await
                .unwrap()
                .json()
                .await
                .unwrap();
            if job["state"] != "running" {
                break;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        let ran = std::fs::read_to_string(dir.path().join("argv.log")).unwrap();
        let last = ran.lines().last().unwrap().to_string();
        let argv: Vec<String> = shown["argv"][0]
            .as_array()
            .unwrap()
            .iter()
            .map(|a| a.as_str().unwrap().to_string())
            .collect();
        assert_eq!(
            last,
            argv.join(" "),
            "{kind}: ran what Copy as Command shows"
        );
        assert_eq!(shown["text"], format!("mrun {}", argv.join(" ")));
        match kind {
            "pull" => {
                assert_eq!(job["state"], "done");
                assert_eq!(job["progress"]["bytes"], 5);
            }
            "remove" => {
                assert_eq!(job["state"], "failed");
                assert_eq!(job["error"], "error: no such model");
            }
            _ => assert_eq!(job["state"], "done"),
        }
    }
    let bad = client()
        .post(format!("http://127.0.0.1:{port}/app/actions"))
        .bearer_auth(TOKEN)
        .json(&serde_json::json!({"kind": "pull", "target": "--help"}))
        .send()
        .await
        .unwrap();
    assert_eq!(bad.status(), 400);
}

/// How often the fake CLI ran `ls`.
#[cfg(unix)]
fn ls_runs(dir: &std::path::Path) -> usize {
    std::fs::read_to_string(dir.join("argv.log"))
        .unwrap_or_default()
        .lines()
        .filter(|l| l.starts_with("ls "))
        .count()
}

#[cfg(unix)]
async fn wait_job(port: u16, id: &str) -> serde_json::Value {
    let mut job = serde_json::Value::Null;
    for _ in 0..50 {
        job = get_with(port, &format!("/app/jobs/{id}"), Some(TOKEN))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        if job["state"] != "running" {
            break;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    job
}

/// 019 FR-008, FR-016: Models runs `ls` for the window's server port, answers from its cache
/// while refreshing, `fresh=1` waits for a run, and a finished pull drops the cache.
#[cfg(unix)]
#[tokio::test]
async fn listings_answer_from_the_cache_and_refresh() {
    let dir = tempfile::tempdir().unwrap();
    let (port, _) = host_with(dir.path(), 9123, Some(fake_cli(dir.path()))).await;
    let shown: serde_json::Value = get_with(port, "/app/command?kind=models", Some(TOKEN))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(shown["text"], "mrun ls --json --all --port 9123");
    let models = || get_with(port, "/app/models", Some(TOKEN)).send();

    let first = models().await.unwrap();
    assert_eq!(first.status(), 200);
    assert!(
        first.headers().get("x-mrun-stale").is_none(),
        "nothing cached yet"
    );
    let body: serde_json::Value = first.json().await.unwrap();
    assert_eq!(body[0]["name"], "m");
    assert_eq!(ls_runs(dir.path()), 1);
    let ran = std::fs::read_to_string(dir.path().join("argv.log")).unwrap();
    assert_eq!(ran.lines().last(), Some("ls --json --all --port 9123"));

    let second = models().await.unwrap();
    assert_eq!(second.headers()["x-mrun-stale"], "1");
    let body: serde_json::Value = second.json().await.unwrap();
    assert_eq!(body[0]["name"], "m");
    for _ in 0..50 {
        if ls_runs(dir.path()) == 2 {
            break;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    assert_eq!(ls_runs(dir.path()), 2, "one background refresh");

    let fresh = get_with(port, "/app/models?fresh=1", Some(TOKEN))
        .send()
        .await
        .unwrap();
    assert!(fresh.headers().get("x-mrun-stale").is_none());
    assert_eq!(ls_runs(dir.path()), 3, "fresh=1 waits for a run");

    let started: serde_json::Value = client()
        .post(format!("http://127.0.0.1:{port}/app/actions"))
        .bearer_auth(TOKEN)
        .json(&serde_json::json!({"kind": "pull", "target": "qwen3.6"}))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let job = wait_job(port, started["job"]["id"].as_str().unwrap()).await;
    assert_eq!(job["state"], "done");
    let before = ls_runs(dir.path());
    let after = models().await.unwrap();
    assert!(
        after.headers().get("x-mrun-stale").is_none(),
        "a finished pull drops the cached listing"
    );
    assert_eq!(ls_runs(dir.path()), before + 1);

    // Agents: cached the same way.
    let agents = || get_with(port, "/app/agents", Some(TOKEN)).send();
    assert!(
        agents()
            .await
            .unwrap()
            .headers()
            .get("x-mrun-stale")
            .is_none()
    );
    assert_eq!(agents().await.unwrap().headers()["x-mrun-stale"], "1");
}

/// 020 FR-006: the plan is `pull <repo> --dry-run --json`'s event, a refusal is its error
/// event, and Copy as Command shows the dry run.
#[cfg(unix)]
#[tokio::test]
async fn the_plan_is_the_clis_dry_run() {
    let dir = tempfile::tempdir().unwrap();
    let (port, _) = host_with(dir.path(), 1, Some(fake_cli(dir.path()))).await;
    let ran = || {
        std::fs::read_to_string(dir.path().join("argv.log"))
            .unwrap_or_default()
            .lines()
            .last()
            .map(str::to_string)
    };
    let plan = get_with(port, "/app/plan?model=org/ok@main", Some(TOKEN))
        .send()
        .await
        .unwrap();
    assert_eq!(plan.status(), 200);
    let plan: serde_json::Value = plan.json().await.unwrap();
    assert_eq!(plan["event"], "plan");
    assert_eq!(plan["family"], "qwen3");
    assert_eq!(ran().as_deref(), Some("pull org/ok@main --dry-run --json"));

    let refused = get_with(port, "/app/plan?model=org/bad", Some(TOKEN))
        .send()
        .await
        .unwrap();
    assert_eq!(refused.status(), 422);
    let refused: serde_json::Value = refused.json().await.unwrap();
    assert_eq!(refused["error"], "unsupported architecture");
    assert_eq!(refused["hint"], "pick another");

    let runs = std::fs::read_to_string(dir.path().join("argv.log"))
        .unwrap()
        .lines()
        .count();
    for bad in ["--help", "", "a%20b", "x;rm"] {
        let response = get_with(port, &format!("/app/plan?model={bad}"), Some(TOKEN))
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), 400, "{bad:?}");
    }
    assert_eq!(
        std::fs::read_to_string(dir.path().join("argv.log"))
            .unwrap()
            .lines()
            .count(),
        runs,
        "a bad target never runs"
    );

    let shown: serde_json::Value =
        get_with(port, "/app/command?kind=plan&target=org/ok", Some(TOKEN))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
    assert_eq!(shown["text"], "mrun pull org/ok --dry-run");
    assert_eq!(
        shown["argv"],
        serde_json::json!([["pull", "org/ok", "--dry-run"]])
    );
    let bad = get_with(port, "/app/command?kind=plan&target=--help", Some(TOKEN))
        .send()
        .await
        .unwrap();
    assert_eq!(bad.status(), 400);
}

/// 020 FR-008: Other apps are `connect <app> --port <port> --json` of Zed and Crush.
#[cfg(unix)]
#[tokio::test]
async fn other_apps_are_the_print_only_targets() {
    let dir = tempfile::tempdir().unwrap();
    let (port, _) = host_with(dir.path(), 9123, Some(fake_cli(dir.path()))).await;
    let apps: serde_json::Value = get_with(port, "/app/apps", Some(TOKEN))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(
        apps,
        serde_json::json!([
            {"client": "zed", "blocks": []},
            {"client": "crush", "blocks": []},
        ])
    );
    let ran = std::fs::read_to_string(dir.path().join("argv.log")).unwrap();
    assert_eq!(
        ran.lines().collect::<Vec<_>>(),
        [
            "connect zed --port 9123 --json",
            "connect crush --port 9123 --json"
        ]
    );
}

/// 020 FR-009, SC-006: an export lands in the downloads directory whatever its name, is
/// numbered when the name is taken, and is refused over 10 MB.
#[tokio::test]
async fn exports_stay_in_downloads_and_are_numbered() {
    let dir = tempfile::tempdir().unwrap();
    let (port, host) = host(dir.path(), 1).await;
    let downloads = host.downloads.to_path_buf();
    let export = |name: &str, text: String| {
        client()
            .post(format!("http://127.0.0.1:{port}/app/export"))
            .bearer_auth(TOKEN)
            .json(&serde_json::json!({"name": name, "text": text}))
            .send()
    };
    let saved = |response: reqwest::Response| async move {
        assert_eq!(response.status(), 200);
        let body: serde_json::Value = response.json().await.unwrap();
        std::path::PathBuf::from(body["path"].as_str().unwrap())
    };
    for (name, file) in [
        ("../../etc/passwd", "etcpasswd.md"),
        ("/abs", "abs.md"),
        ("", "Chat.md"),
        ("..", "Chat 2.md"),
        (".hidden: a/b?", "hidden ab.md"),
    ] {
        let path = saved(export(name, "# t\n".into()).await.unwrap()).await;
        assert_eq!(path, downloads.join(file), "{name:?}");
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "# t\n");
    }
    let long = saved(export(&"x".repeat(200), String::new()).await.unwrap()).await;
    assert_eq!(long, downloads.join(format!("{}.md", "x".repeat(80))));

    let first = saved(export("Plan", "one".into()).await.unwrap()).await;
    let second = saved(export("Plan", "two".into()).await.unwrap()).await;
    let third = saved(export("Plan", "three".into()).await.unwrap()).await;
    assert_eq!(first, downloads.join("Plan.md"));
    assert_eq!(second, downloads.join("Plan 2.md"));
    assert_eq!(third, downloads.join("Plan 3.md"));
    assert_eq!(std::fs::read_to_string(&first).unwrap(), "one");

    let big = export("Big", "a".repeat(10 * 1024 * 1024 + 1))
        .await
        .unwrap();
    assert_eq!(big.status(), 413);
    assert!(!downloads.join("Big.md").exists());
    let entries = std::fs::read_dir(&downloads).unwrap().count();
    assert_eq!(entries, 9, "nothing outside the downloads directory");
    assert!(!dir.path().join("etc").exists());
}
