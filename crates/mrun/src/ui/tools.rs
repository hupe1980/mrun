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
    /// `search`: `exa`, `brave` or `searxng`.
    provider: Option<String>,
    /// `browser`: `any` (default), `local` (also this Mac's servers) or `only` (sites asked for).
    reach: Option<String>,
}

/// A system dialog on this Mac; `None` when cancelled. Keys are typed here, never in the page
/// (029 FR-007).
async fn ask(prompt: &str, hidden: bool) -> Option<String> {
    let script = format!(
        "text returned of (display dialog \"{prompt}\" default answer \"\"{} buttons {{\"Cancel\", \"Add\"}} default button \"Add\" with title \"mrun\")",
        if hidden { " with hidden answer" } else { "" }
    );
    let out = tokio::process::Command::new("osascript")
        .args(["-e", &script])
        .output()
        .await
        .ok()?;
    let text = String::from_utf8_lossy(&out.stdout).trim().to_string();
    (out.status.success() && !text.is_empty()).then_some(text)
}

/// `search`'s entry: Brave's key and SearXNG's address come from a dialog.
async fn search_entry(provider: &str) -> Result<(String, serde_json::Value), String> {
    let (key, url) = match provider {
        "brave" => (
            ask("Brave Search API key (search.brave.com/api)", true).await,
            None,
        ),
        "searxng" => (
            None,
            ask(
                "Address of your SearXNG instance, for example http://127.0.0.1:8888",
                false,
            )
            .await,
        ),
        _ => (None, None),
    };
    if provider == "brave" && key.is_none() || provider == "searxng" && url.is_none() {
        return Err("cancelled".into());
    }
    config::search(provider, url.as_deref(), |name| {
        (name == "BRAVE_API_KEY").then(|| key.clone()).flatten()
    })
    .map(|(entry, _)| entry)
}

/// `POST /app/mcp/add`: mrun's own `workspace` (in a folder), `fetch`, `browser` or `search`.
pub async fn add(State(host): State<Host>, axum::Json(b): axum::Json<AddBody>) -> Response {
    if b.kit == "browser" && cfg!(feature = "browser") {
        let mut options = config::BrowserOptions::default();
        match b.reach.as_deref().unwrap_or("any") {
            "local" => options.allow_local = true,
            "only" => {
                let Some(sites) = ask(
                    "Sites the browser may use, separated by spaces (github.com docs.rs http://localhost:3000)",
                    false,
                )
                .await
                else {
                    return error(StatusCode::BAD_REQUEST, "cancelled");
                };
                options.origins = sites
                    .split([' ', ','])
                    .filter(|s| !s.is_empty())
                    .map(str::to_string)
                    .collect();
                options.allow_local = options
                    .origins
                    .iter()
                    .any(|o| o.contains("localhost") || o.contains("127.0.0.1"));
            }
            _ => {}
        }
        let entry = match config::browser(&options, |_| None) {
            Ok(entry) => entry,
            Err(e) => return error(StatusCode::BAD_REQUEST, &e),
        };
        return match config::add(host.mcp.home(), vec![entry]) {
            Ok(added) if added.is_empty() => {
                error(StatusCode::CONFLICT, "mcp.json already has browser")
            }
            Ok(added) => axum::Json(json!({"added": added})).into_response(),
            Err(e) => error(StatusCode::BAD_REQUEST, &e),
        };
    }
    if b.kit == "search" {
        let entry = match search_entry(b.provider.as_deref().unwrap_or("exa")).await {
            Ok(e) => e,
            Err(e) => return error(StatusCode::BAD_REQUEST, &e),
        };
        return match config::add(host.mcp.home(), vec![entry]) {
            Ok(added) if added.is_empty() => {
                error(StatusCode::CONFLICT, "mcp.json already has search")
            }
            Ok(added) => axum::Json(json!({"added": added})).into_response(),
            Err(e) => error(StatusCode::BAD_REQUEST, &e),
        };
    }
    let dir = match b.dir.as_ref().map(|d| d.canonicalize()) {
        Some(Ok(d)) => Some(d),
        Some(Err(e)) => return error(StatusCode::BAD_REQUEST, &e.to_string()),
        None => None,
    };
    let Some(entry) = config::builtin(&b.kit, dir.as_deref()) else {
        let why = if b.kit == "workspace" {
            "workspace needs a folder".to_string()
        } else {
            format!("this mrun was built without the {} tools", b.kit)
        };
        return error(StatusCode::BAD_REQUEST, &why);
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
