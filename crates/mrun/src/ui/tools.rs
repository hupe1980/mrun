//! The window's tool routes (028): list the servers of mcp.json, call a tool the user allowed,
//! allow what a sandbox stopped, import, add mrun's own tools, and choose a folder for them.

use super::host::{Host, error};
use crate::mcp::{cage::Fix, config};
use axum::extract::{Query, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde_json::{Value, json};

#[derive(Debug, serde::Deserialize)]
pub struct ListQuery {
    /// Start the servers (first open of the picker, Refresh).
    #[serde(default)]
    start: bool,
    /// Stop them first, so the file is read again.
    #[serde(default)]
    refresh: bool,
}

/// `GET /app/mcp`: every server without secrets; with `start`, its tools or its error.
pub async fn list(State(host): State<Host>, Query(q): Query<ListQuery>) -> Response {
    if q.refresh {
        host.mcp.stop(None).await;
    }
    axum::Json(host.mcp.list(q.start || q.refresh).await).into_response()
}

#[derive(Debug, serde::Deserialize)]
pub struct CallBody {
    server: String,
    tool: String,
    #[serde(default)]
    arguments: Value,
}

/// `POST /app/mcp/call`: the page sends a call only after the user allowed it.
pub async fn call(State(host): State<Host>, axum::Json(b): axum::Json<CallBody>) -> Response {
    axum::Json(host.mcp.call(&b.server, &b.tool, b.arguments).await).into_response()
}

#[derive(Debug, serde::Deserialize)]
pub struct AllowBody {
    server: String,
    fix: Value,
}

/// `POST /app/mcp/allow`: adds a denial's fix to the server's block and restarts it with it.
pub async fn allow(State(host): State<Host>, axum::Json(b): axum::Json<AllowBody>) -> Response {
    let fix = match (
        b.fix["host"].as_str(),
        serde_json::from_value::<config::Allowed>(b.fix.clone()),
    ) {
        (Some(h), _) => Fix::Host { host: h.into() },
        (None, Ok(a)) => Fix::Path(a),
        _ => return error(StatusCode::BAD_REQUEST, "nothing to allow"),
    };
    let result = match &fix {
        Fix::Host { host: h } => config::allow(host.mcp.home(), &b.server, None, Some(h)),
        Fix::Path(a) => config::allow(host.mcp.home(), &b.server, Some(a.clone()), None),
    };
    if let Err(e) = result {
        return error(StatusCode::BAD_REQUEST, &e);
    }
    host.mcp.stop(Some(&b.server)).await;
    axum::Json(json!({"command": fix.command(&b.server)})).into_response()
}

#[derive(Debug, serde::Deserialize)]
pub struct ImportBody {
    client: String,
}

/// `POST /app/mcp/import`: copies Claude Code's or Claude Desktop's servers.
pub async fn import(State(host): State<Host>, axum::Json(b): axum::Json<ImportBody>) -> Response {
    let Some(file) = config::client_config(&b.client) else {
        return error(StatusCode::BAD_REQUEST, "unknown client");
    };
    let entries = match config::servers_in(&file) {
        Ok(e) => e,
        Err(e) => return error(StatusCode::NOT_FOUND, &e),
    };
    match config::add(host.mcp.home(), entries) {
        Ok(added) => axum::Json(json!({"added": added})).into_response(),
        Err(e) => error(StatusCode::BAD_REQUEST, &e),
    }
}

#[derive(Debug, serde::Deserialize)]
pub struct AddBody {
    kit: String,
    dir: Option<std::path::PathBuf>,
}

/// `POST /app/mcp/add`: mrun's own `workspace` (in a folder) or `fetch`.
pub async fn add(State(host): State<Host>, axum::Json(b): axum::Json<AddBody>) -> Response {
    let dir = match b.dir.as_ref().map(|d| d.canonicalize()) {
        Some(Ok(d)) => Some(d),
        Some(Err(e)) => return error(StatusCode::BAD_REQUEST, &e.to_string()),
        None => None,
    };
    let Some(entry) = config::builtin(&b.kit, dir.as_deref()) else {
        return error(StatusCode::BAD_REQUEST, "workspace needs a folder");
    };
    match config::add(host.mcp.home(), vec![entry]) {
        Ok(added) if added.is_empty() => error(
            StatusCode::CONFLICT,
            &format!("mcp.json already has {}", b.kit),
        ),
        Ok(added) => axum::Json(json!({"added": added})).into_response(),
        Err(e) => error(StatusCode::BAD_REQUEST, &e),
    }
}

/// `POST /app/mcp/choose`: the system's folder chooser (no folder: `null`).
pub async fn choose() -> Response {
    let out = tokio::process::Command::new("osascript")
        .args([
            "-e",
            "POSIX path of (choose folder with prompt \"The folder mrun's workspace tools may read, change and run in\")",
        ])
        .output()
        .await;
    let dir = out
        .ok()
        .filter(|o| o.status.success())
        .map(|o| {
            String::from_utf8_lossy(&o.stdout)
                .trim()
                .trim_end_matches('/')
                .to_string()
        })
        .filter(|d| !d.is_empty());
    axum::Json(json!({"dir": dir})).into_response()
}
