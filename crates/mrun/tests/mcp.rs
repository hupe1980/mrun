//! 028: the MCP client against mrun's own tools (stdio, caged) and a fake streamable-HTTP server
//! (JSON and SSE answers, session id).

use axum::http::{HeaderMap, StatusCode};
use axum::response::IntoResponse;
use mrun::mcp::{Registry, cage};
use serde_json::{Value, json};
use std::path::Path;

fn write_config(home: &Path, servers: Value) {
    std::fs::write(
        home.join("mcp.json"),
        json!({"mcpServers": servers}).to_string(),
    )
    .unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn the_workspace_tools_run_caged_in_their_folder() {
    // A sandbox that does not answer fails the test instead of hanging CI.
    tokio::time::timeout(std::time::Duration::from_secs(90), workspace_tools())
        .await
        .expect("the caged workspace server did not answer in 90 s");
}

async fn workspace_tools() {
    if !cage::available() {
        eprintln!("skipped: this system cannot cage children");
        return;
    }
    let home = tempfile::tempdir().unwrap();
    let root = tempfile::tempdir().unwrap();
    let root = root.path().canonicalize().unwrap();
    let outside = tempfile::tempdir_in(std::env::home_dir().unwrap()).unwrap();
    let outside = outside.path().canonicalize().unwrap();
    write_config(
        home.path(),
        json!({"workspace": {
            "command": env!("CARGO_BIN_EXE_mrun"),
            "args": ["mcp", "serve", "workspace", "--root", root],
            // A config file the cage could not read must not stop mrun's own tools.
            "env": {"PROJECT_TOKEN": "s3cret-value", "MRUN_CONFIG": home.path().join("broken.toml")},
            "sandbox": {"workspace": root, "write": true, "hosts": []},
        }}),
    );
    std::fs::write(home.path().join("broken.toml"), "not = [toml").unwrap();
    let registry = Registry::new(home.path().to_path_buf());
    let list = registry.list(true).await;
    let server = &list["servers"][0];
    if server["error"]
        .as_str()
        .is_some_and(|e| e.contains("capability gaps"))
    {
        eprintln!("skipped: {}", server["error"]);
        return;
    }
    assert!(server["error"].is_null(), "{list}");
    assert_eq!(server["tools"].as_array().unwrap().len(), 6);
    assert_eq!(server["tools"][0]["qualified"], "workspace__read");
    assert!(
        !list.to_string().contains("s3cret"),
        "no env value in the listing"
    );

    let wrote = registry
        .call(
            "workspace",
            "write",
            json!({"path": "a.txt", "content": "hi"}),
        )
        .await;
    assert!(!wrote.is_error, "{}", wrote.text);
    assert!(
        wrote.denials.iter().all(|d| d.fix.is_none()),
        "no false Allow: {:?}",
        wrote.denials
    );
    assert_eq!(std::fs::read_to_string(root.join("a.txt")).unwrap(), "hi");

    let refused = registry
        .call("workspace", "read", json!({"path": "../etc"}))
        .await;
    assert!(refused.is_error && refused.text.contains("outside the workspace"));

    // The env value reaches the child through its environment (bash sees it).
    let env = registry
        .call(
            "workspace",
            "bash",
            json!({"command": "printf %s \"$PROJECT_TOKEN\""}),
        )
        .await;
    assert!(env.text.starts_with("s3cret-value"), "{}", env.text);

    // bash itself is caged: writing outside the folder fails.
    let escape = format!("echo x > '{}/b'", outside.display());
    let caged = registry
        .call("workspace", "bash", json!({"command": escape}))
        .await;
    assert!(caged.is_error, "{}", caged.text);
    assert!(!outside.join("b").exists());
}

async fn fake(headers: HeaderMap, body: axum::Json<Value>) -> axum::response::Response {
    let msg = body.0;
    let Some(id) = msg.get("id").cloned() else {
        return StatusCode::ACCEPTED.into_response();
    };
    if headers.get("authorization").and_then(|v| v.to_str().ok()) != Some("Bearer t0ken") {
        return StatusCode::UNAUTHORIZED.into_response();
    }
    let method = msg["method"].as_str().unwrap_or("");
    if method != "initialize" && headers.get("mcp-session-id").is_none() {
        return StatusCode::BAD_REQUEST.into_response();
    }
    let result = match method {
        "initialize" => {
            json!({"protocolVersion": "2025-06-18", "capabilities": {"tools": {}}, "serverInfo": {"name": "fake"}})
        }
        "tools/list" if msg["params"]["cursor"].is_null() => {
            json!({"tools": [{"name": "echo", "description": "Echo", "inputSchema": {"type": "object"}}], "nextCursor": "2"})
        }
        "tools/list" => json!({"tools": [{"name": "fail"}]}),
        "tools/call" if msg["params"]["name"] == "echo" => {
            json!({"content": [{"type": "text", "text": msg["params"]["arguments"]["text"]}]})
        }
        _ => json!({"content": [{"type": "text", "text": "nope"}], "isError": true}),
    };
    let reply = json!({"jsonrpc": "2.0", "id": id, "result": result});
    let mut resp = if method == "tools/call" {
        // Answers a call as an event stream, after a notification.
        let body = format!(
            "event: message\ndata: {}\n\nevent: message\ndata: {reply}\n\n",
            json!({"jsonrpc": "2.0", "method": "notifications/progress"})
        );
        ([("content-type", "text/event-stream")], body).into_response()
    } else {
        axum::Json(reply).into_response()
    };
    resp.headers_mut()
        .insert("mcp-session-id", "abc".parse().unwrap());
    resp
}

#[tokio::test]
async fn a_remote_server_over_streamable_http() {
    let app = axum::Router::new().route("/mcp", axum::routing::post(fake));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await });
    let home = tempfile::tempdir().unwrap();
    write_config(
        home.path(),
        json!({"remote": {"url": format!("http://{addr}/mcp"), "headers": {"Authorization": "Bearer t0ken"}}}),
    );
    let registry = Registry::new(home.path().to_path_buf());
    let list = registry.list(true).await;
    let server = &list["servers"][0];
    assert_eq!(server["host"], "127.0.0.1");
    let names: Vec<&str> = server["tools"]
        .as_array()
        .unwrap()
        .iter()
        .map(|t| t["name"].as_str().unwrap())
        .collect();
    assert_eq!(names, ["echo", "fail"], "both pages: {list}");
    assert!(
        !list.to_string().contains("t0ken"),
        "no header in the listing"
    );
    let echo = registry
        .call("remote", "echo", json!({"text": "hello"}))
        .await;
    assert_eq!((echo.text.as_str(), echo.is_error), ("hello", false));
    let fail = registry.call("remote", "fail", json!({})).await;
    assert!(fail.is_error);

    write_config(
        home.path(),
        json!({"remote": {"url": format!("http://{addr}/mcp"), "headers": {"Authorization": "Bearer wrong"}}}),
    );
    let fresh = Registry::new(home.path().to_path_buf());
    let list = fresh.list(true).await;
    assert_eq!(list["servers"][0]["error"], "127.0.0.1: HTTP 401");
}

#[tokio::test]
async fn a_server_that_cannot_start_says_why() {
    let home = tempfile::tempdir().unwrap();
    write_config(
        home.path(),
        json!({"broken": {"command": "/bin/sh", "args": ["-c", "echo missing config >&2; exit 1"], "sandbox": false}}),
    );
    let registry = Registry::new(home.path().to_path_buf());
    let list = registry.list(true).await;
    let error = list["servers"][0]["error"].as_str().unwrap();
    assert!(error.starts_with("the server stopped"), "{error}");
    let call = registry.call("nothere", "x", json!({})).await;
    assert!(call.is_error && call.text.contains("no server nothere"));
}
