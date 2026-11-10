//! The UI host's HTTP surface (contracts/ui-host.md): the embedded page, `/app/*` actions and
//! state, and a proxy to the server for status and chat, all on one loopback origin.
//!
//! Security (research R5): Host and Origin must be this host's own loopback address (DNS
//! rebinding and other pages are refused), and everything but the page's files needs the
//! per-start token, which reaches the page only in the URL fragment.

use super::actions::{self, Jobs, Kind, State as JobState};
use super::store::Store;
use crate::api::is_loopback_host;
use axum::Router;
use axum::body::Body;
use axum::extract::{Path, Query, Request, State};
use axum::http::{HeaderMap, HeaderValue, Method, StatusCode, Uri, header};
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use serde_json::{Value, json};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

include!(concat!(env!("OUT_DIR"), "/ui_bundle.rs"));

/// Who started the server the window shows (Edge Cases: switching a terminal's server asks).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StartedBy {
    Ui,
    Other,
}

/// State shared by the host's routes.
#[derive(Clone, Debug)]
pub struct Host {
    pub token: Arc<String>,
    pub ui_port: u16,
    pub server_port: u16,
    pub jobs: Arc<Jobs>,
    pub store: Arc<Store>,
    pub started: Arc<Mutex<Option<StartedBy>>>,
    /// Bumped by `/app/focus`; the window shell brings itself forward on a change.
    pub focus: tokio::sync::watch::Sender<u64>,
    /// Set by `/app/quit` (the page confirmed quitting during a download).
    pub quit: tokio::sync::watch::Sender<bool>,
    /// The `mrun` actions run (this executable; tests put a fake here).
    pub cli: Arc<std::path::PathBuf>,
    /// Where Save as Markdown writes (020 FR-009): `~/Downloads`; tests use a temp dir.
    pub downloads: Arc<std::path::PathBuf>,
    /// Whether a saved export is shown in Finder (`open -R`); tests turn it off.
    pub reveal: bool,
    listings: Arc<Listings>,
    client: reqwest::Client,
}

/// The last good Models and Agents listings, in memory only (019 research R9).
#[derive(Debug, Default)]
struct Listings {
    entries: Mutex<HashMap<&'static str, Listing>>,
}

#[derive(Debug, Default)]
struct Listing {
    value: Option<Value>,
    /// Bumped by [`Listings::invalidate`]: a run that started before it never stores its result.
    generation: u64,
    refreshing: bool,
}

impl Listings {
    fn lock(&self) -> std::sync::MutexGuard<'_, HashMap<&'static str, Listing>> {
        self.entries
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    fn get(&self, key: &'static str) -> Option<Value> {
        self.lock().get(key).and_then(|l| l.value.clone())
    }

    fn generation(&self, key: &'static str) -> u64 {
        self.lock().get(key).map_or(0, |l| l.generation)
    }

    /// The generation to refresh at, `None` when a background refresh already runs.
    fn begin_refresh(&self, key: &'static str) -> Option<u64> {
        let mut entries = self.lock();
        let listing = entries.entry(key).or_default();
        (!listing.refreshing).then(|| {
            listing.refreshing = true;
            listing.generation
        })
    }

    fn store(&self, key: &'static str, generation: u64, value: Value) {
        let mut entries = self.lock();
        let listing = entries.entry(key).or_default();
        if listing.generation == generation {
            listing.value = Some(value);
        }
    }

    fn end_refresh(&self, key: &'static str) {
        if let Some(listing) = self.lock().get_mut(key) {
            listing.refreshing = false;
        }
    }

    /// A finished job changed what the listing shows.
    fn invalidate(&self, key: &'static str) {
        let mut entries = self.lock();
        let listing = entries.entry(key).or_default();
        listing.value = None;
        listing.generation += 1;
    }
}

impl Host {
    pub fn new(token: String, ui_port: u16, server_port: u16, store: Store) -> Self {
        Self {
            token: Arc::new(token),
            ui_port,
            server_port,
            jobs: Arc::new(Jobs::default()),
            store: Arc::new(store),
            started: Arc::new(Mutex::new(None)),
            focus: tokio::sync::watch::channel(0).0,
            quit: tokio::sync::watch::channel(false).0,
            cli: Arc::new(actions::cli_program()),
            downloads: Arc::new(
                std::env::var_os("HOME")
                    .map_or_else(|| ".".into(), std::path::PathBuf::from)
                    .join("Downloads"),
            ),
            reveal: true,
            listings: Arc::new(Listings::default()),
            client: reqwest::Client::builder()
                .no_proxy()
                .build()
                .unwrap_or_default(),
        }
    }
}

/// The routes the page proxies to the server (contract: proxied routes).
const PROXIED: [&str; 7] = [
    "/health",
    "/mrun/status",
    "/mrun/sessions",
    "/mrun/pool",
    "/mrun/pool/plan",
    "/metrics",
    "/v1/models",
];

/// The client class the window's own requests carry (025: its chats are labelled "Chat").
const USER_AGENT: &str = concat!("mrun-ui/", env!("CARGO_PKG_VERSION"));

pub fn router(host: Host) -> Router {
    Router::new()
        .route("/", get(index))
        .route("/app/state", get(app_state))
        .route("/app/models", get(models))
        .route("/app/agents", get(agents))
        .route("/app/apps", get(apps))
        .route("/app/cloud", get(cloud))
        .route("/app/cloud/model", get(cloud_model))
        .route("/app/plan", get(plan))
        .route("/app/export", post(export))
        .route("/app/config", get(config))
        .route("/app/actions", post(start_action))
        .route("/app/jobs/{id}", get(job).delete(cancel_job))
        .route("/app/command", get(command))
        .route("/app/chats", get(chats).delete(delete_chats))
        .route(
            "/app/chats/{id}",
            get(chat).put(save_chat).delete(delete_chat),
        )
        .route("/app/prefs", axum::routing::put(prefs))
        .route("/app/open", post(open))
        .route("/app/focus", post(focus))
        .route("/app/quit", post(quit))
        .route("/health", get(proxy))
        .route("/mrun/status", get(proxy))
        .route("/mrun/sessions", get(proxy))
        .route("/mrun/pool", get(proxy))
        .route("/mrun/pool/plan", get(proxy))
        .route("/mrun/requests/{id}/excerpt", get(proxy))
        .route("/metrics", get(proxy))
        .route("/v1/models", get(proxy))
        .route("/v1/models/{id}", get(proxy))
        .route("/v1/messages", post(proxy))
        .route("/v1/messages/count_tokens", post(proxy))
        .route("/v1/chat/completions", post(proxy))
        .fallback(asset)
        .layer(middleware::from_fn_with_state(host.clone(), guard))
        .with_state(host)
}

/// Host and Origin for every request, the token for everything but the page's files, and the
/// response headers of the contract.
async fn guard(State(host): State<Host>, request: Request, next: Next) -> Response {
    let port = host.ui_port;
    let host_ok = request
        .headers()
        .get(header::HOST)
        .and_then(|v| v.to_str().ok())
        .is_some_and(|h| is_loopback_host(h, port));
    if !host_ok {
        return error(StatusCode::FORBIDDEN, "not this window's address");
    }
    if !crate::api::origin_allowed(request.headers(), port) {
        return error(
            StatusCode::FORBIDDEN,
            "cross-origin requests are not accepted",
        );
    }
    let path = request.uri().path();
    let protected = path.starts_with("/app/")
        || path.starts_with("/v1/")
        || path.starts_with("/mrun/")
        || PROXIED.contains(&path);
    if protected && !authorized(request.headers(), &host.token) {
        return error(StatusCode::UNAUTHORIZED, "unauthorized");
    }
    let api = path.starts_with("/app/");
    let mut response = next.run(request).await;
    let headers = response.headers_mut();
    headers.insert(
        header::CONTENT_SECURITY_POLICY,
        HeaderValue::from_static(
            "default-src 'self'; connect-src 'self'; img-src 'self' data:; style-src 'self' 'unsafe-inline'; frame-ancestors 'none'",
        ),
    );
    headers.insert(
        header::X_CONTENT_TYPE_OPTIONS,
        HeaderValue::from_static("nosniff"),
    );
    headers.insert(
        header::REFERRER_POLICY,
        HeaderValue::from_static("no-referrer"),
    );
    if api {
        headers.insert(header::CACHE_CONTROL, HeaderValue::from_static("no-store"));
    }
    response
}

/// `Authorization: Bearer <token>`, compared in constant time.
fn authorized(headers: &HeaderMap, token: &str) -> bool {
    let Some(given) = headers
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "))
    else {
        return false;
    };
    given.len() == token.len()
        && given
            .bytes()
            .zip(token.bytes())
            .fold(0u8, |acc, (a, b)| acc | (a ^ b))
            == 0
}

fn error(status: StatusCode, message: &str) -> Response {
    (status, axum::Json(json!({"error": message}))).into_response()
}

fn content_type(name: &str) -> &'static str {
    match name.rsplit('.').next() {
        Some("html") => "text/html; charset=utf-8",
        Some("js") => "text/javascript; charset=utf-8",
        Some("css") => "text/css; charset=utf-8",
        Some("svg") => "image/svg+xml",
        Some("png") => "image/png",
        Some("json") => "application/json",
        _ => "application/octet-stream",
    }
}

fn file(name: &str) -> Option<Response> {
    let (_, bytes) = UI_BUNDLE.iter().find(|(n, _)| *n == name)?;
    Some(([(header::CONTENT_TYPE, content_type(name))], *bytes).into_response())
}

async fn index() -> Response {
    file("index.html").unwrap_or_else(|| {
        (
            [(header::CONTENT_TYPE, "text/html; charset=utf-8")],
            "<!doctype html><title>mrun</title><p>This build of mrun has no window: build it with <code>cd crates/mrun/ui &amp;&amp; npm ci &amp;&amp; npm run build</code>.</p>",
        )
            .into_response()
    })
}

/// A file of the built page by its flat name (no directories to walk).
async fn asset(uri: Uri) -> Response {
    let name = uri.path().trim_start_matches('/');
    if name.contains('/') || name.starts_with('.') {
        return error(StatusCode::NOT_FOUND, "not found");
    }
    file(name).unwrap_or_else(|| error(StatusCode::NOT_FOUND, "not found"))
}

/// Forwards a request to the server unchanged and streams its answer back; the page closing
/// the stream closes the upstream connection, which is how Stop reaches the server (D052).
async fn proxy(
    State(host): State<Host>,
    method: Method,
    uri: Uri,
    headers: HeaderMap,
    body: Body,
) -> Response {
    let path = uri.path_and_query().map_or(uri.path(), |p| p.as_str());
    let url = format!("http://127.0.0.1:{}{path}", host.server_port);
    let Ok(body) = axum::body::to_bytes(body, crate::api::MAX_BODY_BYTES).await else {
        return error(StatusCode::PAYLOAD_TOO_LARGE, "request too large");
    };
    let mut upstream = host
        .client
        .request(method, &url)
        .header(header::USER_AGENT, USER_AGENT)
        .body(body);
    for name in [header::CONTENT_TYPE, header::ACCEPT] {
        if let Some(value) = headers.get(&name) {
            upstream = upstream.header(name, value);
        }
    }
    if let Some(value) = headers.get("anthropic-version") {
        upstream = upstream.header("anthropic-version", value);
    }
    let response = match upstream.send().await {
        Ok(response) => response,
        Err(_) => return error(StatusCode::BAD_GATEWAY, "server_down"),
    };
    let status = response.status();
    let mut out = Response::builder().status(status.as_u16());
    for name in [header::CONTENT_TYPE, header::CACHE_CONTROL] {
        if let Some(value) = response.headers().get(&name) {
            out = out.header(name, value);
        }
    }
    for (name, value) in response.headers() {
        if name.as_str().starts_with("x-mrun-") {
            out = out.header(name, value);
        }
    }
    out.body(Body::from_stream(response.bytes_stream()))
        .unwrap_or_else(|_| error(StatusCode::BAD_GATEWAY, "server_down"))
}

fn server_running(port: u16) -> bool {
    std::net::TcpStream::connect_timeout(
        &([127, 0, 0, 1], port).into(),
        std::time::Duration::from_millis(250),
    )
    .is_ok()
}

async fn app_state(State(host): State<Host>) -> Response {
    let port = host.server_port;
    let running = tokio::task::spawn_blocking(move || server_running(port))
        .await
        .unwrap_or(false);
    let started_by = {
        let mut started = host
            .started
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if !running {
            *started = None;
        } else if started.is_none() {
            // A server an earlier window started is still this window's.
            let ours = host.store.prefs().server_pid.is_some()
                && host.store.prefs().server_pid == server_pid(port);
            *started = Some(if ours {
                StartedBy::Ui
            } else {
                StartedBy::Other
            });
        }
        match *started {
            Some(StartedBy::Ui) => json!("ui"),
            Some(StartedBy::Other) => json!("other"),
            None => Value::Null,
        }
    };
    let prefs = host.store.prefs();
    axum::Json(json!({
        "ui_version": env!("CARGO_PKG_VERSION"),
        "server": {"port": port, "running": running, "started_by": started_by},
        "prefs": {
            "keep_chats": prefs.keep_chats,
            "menu_bar": prefs.menu_bar,
            "open_at_login": prefs.open_at_login,
            "server_port": port,
        },
        "jobs": host.jobs.list(),
    }))
    .into_response()
}

/// Runs a read-only CLI command and returns its JSON output.
async fn fetch_json(host: &Host, argv: &[&str]) -> Result<Value, String> {
    let text = actions::read(&host.cli, argv).await?;
    serde_json::from_str::<Value>(&text).map_err(|_| "mrun printed no JSON".to_string())
}

/// Runs a read-only CLI command and answers with its JSON output.
async fn read_json(host: &Host, argv: &[&str]) -> Response {
    match fetch_json(host, argv).await {
        Ok(value) => axum::Json(value).into_response(),
        Err(message) => error(StatusCode::BAD_GATEWAY, &message),
    }
}

/// A listing the host caches (contract: `GET /app/models`, `GET /app/agents`).
#[derive(Clone, Copy, Debug)]
enum Read {
    Models,
    Agents,
    Cloud,
}

impl Read {
    /// The key a job's [`Kind::listing`] names.
    fn key(self) -> &'static str {
        match self {
            Self::Models => "models",
            Self::Agents => "agents",
            Self::Cloud => "cloud",
        }
    }

    async fn fetch(self, host: &Host) -> Result<Value, String> {
        match self {
            Self::Models => {
                let argv = actions::models_argv(host.server_port);
                let argv: Vec<&str> = argv.iter().map(String::as_str).collect();
                fetch_json(host, &argv).await
            }
            Self::Agents => Ok(agent_list(host).await),
            Self::Cloud => Ok(cloud_listing(host).await),
        }
    }
}

/// Stale-while-revalidate (019 FR-016): a cached listing answers at once, marked
/// `x-mrun-stale: 1`, while one background run refreshes it; `?fresh=1`, and a first read,
/// wait for a run. Only successful runs are kept.
async fn listing(host: Host, read: Read, q: &HashMap<String, String>) -> Response {
    let key = read.key();
    let fresh = q.get("fresh").is_some_and(|v| v == "1");
    if !fresh && let Some(value) = host.listings.get(key) {
        if let Some(generation) = host.listings.begin_refresh(key) {
            let host = host.clone();
            tokio::spawn(async move {
                if let Ok(value) = read.fetch(&host).await {
                    host.listings.store(key, generation, value);
                }
                host.listings.end_refresh(key);
            });
        }
        let mut response = axum::Json(value).into_response();
        response
            .headers_mut()
            .insert("x-mrun-stale", HeaderValue::from_static("1"));
        return response;
    }
    let generation = host.listings.generation(key);
    match read.fetch(&host).await {
        Ok(value) => {
            host.listings.store(key, generation, value.clone());
            axum::Json(value).into_response()
        }
        Err(message) => error(StatusCode::BAD_GATEWAY, &message),
    }
}

async fn models(State(host): State<Host>, Query(q): Query<HashMap<String, String>>) -> Response {
    listing(host, Read::Models, &q).await
}

async fn config(State(host): State<Host>) -> Response {
    read_json(&host, &["config", "show", "--json"]).await
}

async fn agents(State(host): State<Host>, Query(q): Query<HashMap<String, String>>) -> Response {
    listing(host, Read::Agents, &q).await
}

async fn cloud(State(host): State<Host>, Query(q): Query<HashMap<String, String>>) -> Response {
    listing(host, Read::Cloud, &q).await
}

/// A model to add to the cloud section: `cloud models <repo> --json` (its weights from the Hub,
/// the GPUs it fits).
async fn cloud_model(
    State(host): State<Host>,
    Query(q): Query<HashMap<String, String>>,
) -> Response {
    let repo = q.get("repo").map_or("", String::as_str);
    let argv = match actions::cloud_model_argv(repo) {
        Ok(argv) => argv,
        Err(message) => return error(StatusCode::BAD_REQUEST, &message),
    };
    let argv: Vec<&str> = argv.iter().map(String::as_str).collect();
    read_json(&host, &argv).await
}

/// The cloud section (024 data model: CloudListing): the catalog, RunPod's GPUs (none without
/// a key), the pods and the key's source, and the models the user added.
async fn cloud_listing(host: &Host) -> Value {
    let port = host.server_port.to_string();
    let status_argv = ["cloud", "status", "--json", "--port", &port];
    let (offers, gpus, status) = tokio::join!(
        fetch_json(host, &["cloud", "models", "--json"]),
        fetch_json(host, &["cloud", "gpus", "--json"]),
        fetch_json(host, &status_argv),
    );
    json!({
        "offers": offers.unwrap_or_else(|_| json!([])),
        "gpus": gpus.unwrap_or_else(|_| json!([])),
        "status": status.unwrap_or_else(|e| json!({"pods": [], "key": null, "error": e})),
        "added": host.store.prefs().cloud_models,
    })
}

/// Other apps (020 FR-008): `connect --json` of the print-only targets, with their blocks.
async fn apps(State(host): State<Host>) -> Response {
    axum::Json(connect_states(&host, &actions::APPS).await).into_response()
}

async fn agent_list(host: &Host) -> Value {
    connect_states(host, &actions::CLIENTS).await
}

async fn connect_states(host: &Host, clients: &[&str]) -> Value {
    let port = host.server_port.to_string();
    let mut list = Vec::new();
    for &client in clients {
        let entry = match actions::read(&host.cli, &["connect", client, "--port", &port, "--json"])
            .await
        {
            Ok(text) => serde_json::from_str(&text).unwrap_or_else(|_| json!({"client": client})),
            Err(message) => json!({"client": client, "installed": false, "connected": false,
                "version": null, "files": [], "refusal": message}),
        };
        list.push(entry);
    }
    Value::Array(list)
}

// No `Debug`: `secret` is a RunPod key.
#[derive(serde::Deserialize)]
struct ActionBody {
    kind: String,
    #[serde(default)]
    target: String,
    #[serde(default)]
    confirm: bool,
    /// `cloud-login`'s key, written to the command's stdin (024 FR-005).
    #[serde(default)]
    secret: Option<String>,
}

async fn start_action(
    State(host): State<Host>,
    axum::Json(body): axum::Json<ActionBody>,
) -> Response {
    let Some(kind) = Kind::parse(&body.kind) else {
        return error(StatusCode::BAD_REQUEST, "unknown action");
    };
    let prefs = host.store.prefs();
    let steps = match actions::steps(
        kind,
        &body.target,
        host.server_port,
        prefs.last_model.as_deref(),
    ) {
        Ok(steps) => steps,
        Err(message) => return error(StatusCode::BAD_REQUEST, &message),
    };
    let other = *host
        .started
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        == Some(StartedBy::Other);
    if kind.touches_server() && kind != Kind::Start && other && !body.confirm {
        return error(
            StatusCode::CONFLICT,
            "this server was started outside the window; confirm to change it",
        );
    }
    let stdin = match (kind, body.secret) {
        (Kind::CloudLogin, Some(secret)) if !secret.trim().is_empty() => Some(secret),
        (Kind::CloudLogin, _) => return error(StatusCode::BAD_REQUEST, "no key"),
        _ => None,
    };
    let started = host.started.clone();
    let store = host.store.clone();
    let target = body.target.clone();
    let server_port = host.server_port;
    let listings = host.listings.clone();
    let on_done = move |state: JobState| {
        // Even a failed or cancelled job may have changed something (a partial pull).
        listings.invalidate(kind.listing());
        if state != JobState::Done {
            return;
        }
        let mut started = started
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        match kind {
            Kind::Use | Kind::Start | Kind::CloudUp => {
                *started = Some(StartedBy::Ui);
                if kind == Kind::Use {
                    let _ = store.update(&json!({"last_model": target}));
                }
                if kind != Kind::CloudUp {
                    let _ = store.update(&json!({"server_pid": server_pid(server_port)}));
                }
            }
            Kind::Stop | Kind::CloudDown => *started = None,
            _ => {}
        }
    };
    match host.jobs.start(
        host.cli.to_path_buf(),
        kind,
        &body.target,
        steps,
        stdin,
        on_done,
    ) {
        Some(job) => axum::Json(json!({"job": job})).into_response(),
        None => error(StatusCode::CONFLICT, "this action is already running"),
    }
}

async fn job(State(host): State<Host>, Path(id): Path<String>) -> Response {
    match host.jobs.get(&id) {
        Some(job) => axum::Json(job).into_response(),
        None => error(StatusCode::NOT_FOUND, "no such job"),
    }
}

async fn cancel_job(State(host): State<Host>, Path(id): Path<String>) -> Response {
    match host.jobs.cancel(&id) {
        Some(job) => {
            // A cancelled cloud start may have rented a pod: end its billing.
            if let Some(argv) = actions::undo(job.kind, host.server_port) {
                let host = host.clone();
                tokio::spawn(async move {
                    let argv: Vec<&str> = argv.iter().map(String::as_str).collect();
                    let _ = actions::read(&host.cli, &argv).await;
                    host.listings.invalidate("cloud");
                });
            }
            axum::Json(job).into_response()
        }
        None => error(StatusCode::NOT_FOUND, "no such job"),
    }
}

/// Add a model's plan (020 FR-006): `pull <repo> --dry-run --json`'s `plan` event, or its
/// `error` event with 422.
async fn plan(State(host): State<Host>, Query(q): Query<HashMap<String, String>>) -> Response {
    let target = q.get("model").map_or("", String::as_str);
    let mut argv = match actions::plan_argv(target) {
        Ok(argv) => argv,
        Err(message) => return error(StatusCode::BAD_REQUEST, &message),
    };
    argv.push("--json".into());
    let argv: Vec<&str> = argv.iter().map(String::as_str).collect();
    let (ok, stdout, stderr) = match actions::output(&host.cli, &argv).await {
        Ok(out) => out,
        Err(message) => return error(StatusCode::BAD_GATEWAY, &message),
    };
    let events: Vec<Value> = stdout
        .lines()
        .filter_map(|line| serde_json::from_str(line).ok())
        .collect();
    let event = |name: &str| events.iter().find(|e| e["event"] == name).cloned();
    match (ok, event("plan"), event("error")) {
        (true, Some(plan), _) => axum::Json(plan).into_response(),
        (_, _, Some(refusal)) => (
            StatusCode::UNPROCESSABLE_ENTITY,
            axum::Json(json!({
                "error": refusal["message"],
                "message": refusal["message"],
                "hint": refusal["hint"],
            })),
        )
            .into_response(),
        _ if stderr.is_empty() => error(StatusCode::BAD_GATEWAY, "mrun printed no plan"),
        _ => error(StatusCode::BAD_GATEWAY, &stderr),
    }
}

async fn command(State(host): State<Host>, Query(q): Query<HashMap<String, String>>) -> Response {
    let target = q.get("target").cloned().unwrap_or_default();
    let read = match q.get("kind").map(String::as_str) {
        Some("models") => Some(Ok(actions::models_argv(host.server_port))),
        Some("plan") => Some(actions::plan_argv(&target)),
        _ => None,
    };
    if let Some(argv) = read {
        return match argv {
            Ok(argv) => {
                let steps = vec![argv];
                axum::Json(json!({"argv": steps, "text": actions::command_line(&steps)}))
                    .into_response()
            }
            Err(message) => error(StatusCode::BAD_REQUEST, &message),
        };
    }
    let kind = q.get("kind").and_then(|k| Kind::parse(k));
    let Some(kind) = kind else {
        return error(StatusCode::BAD_REQUEST, "unknown action");
    };
    let prefs = host.store.prefs();
    match actions::steps(kind, &target, host.server_port, prefs.last_model.as_deref()) {
        Ok(steps) => axum::Json(json!({"argv": steps, "text": actions::command_line(&steps)}))
            .into_response(),
        Err(message) => error(StatusCode::BAD_REQUEST, &message),
    }
}

async fn chats(State(host): State<Host>) -> Response {
    axum::Json(host.store.chats()).into_response()
}

async fn chat(State(host): State<Host>, Path(id): Path<String>) -> Response {
    match host.store.chat(&id) {
        Some(chat) => axum::Json(chat).into_response(),
        None => error(StatusCode::NOT_FOUND, "no such chat"),
    }
}

async fn save_chat(
    State(host): State<Host>,
    Path(id): Path<String>,
    axum::Json(chat): axum::Json<Value>,
) -> Response {
    match host.store.save_chat(&id, &chat) {
        Ok(true) => StatusCode::NO_CONTENT.into_response(),
        Ok(false) => error(StatusCode::CONFLICT, "keeping chats is off"),
        Err(e) => error(StatusCode::BAD_REQUEST, &e.to_string()),
    }
}

async fn delete_chat(State(host): State<Host>, Path(id): Path<String>) -> Response {
    match host.store.delete_chat(&id) {
        Ok(()) => StatusCode::NO_CONTENT.into_response(),
        Err(e) => error(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string()),
    }
}

async fn delete_chats(State(host): State<Host>) -> Response {
    match host.store.delete_chats() {
        Ok(deleted) => axum::Json(json!({"deleted": deleted})).into_response(),
        Err(e) => error(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string()),
    }
}

async fn prefs(State(host): State<Host>, axum::Json(patch): axum::Json<Value>) -> Response {
    // `last_model` and `window` are the window's own bookkeeping, not settings the page writes.
    let mut patch = patch;
    if let Value::Object(fields) = &mut patch {
        fields.remove("last_model");
        fields.remove("window");
        fields.remove("server_pid");
    }
    match host.store.update(&patch) {
        Ok(prefs) => axum::Json(prefs).into_response(),
        Err(e) => error(StatusCode::BAD_REQUEST, &e.to_string()),
    }
}

#[derive(Debug, serde::Deserialize)]
struct OpenBody {
    what: String,
    url: Option<String>,
}

/// Opens the config file in the default editor, the log or the chats in Finder, or an
/// `https://` link in the default browser.
async fn open(axum::Json(body): axum::Json<OpenBody>) -> Response {
    let home = crate::cli::mrun_home();
    let args: Vec<String> = match body.what.as_str() {
        "config" => {
            let location = crate::config::location();
            match location.path() {
                Some(file) if file.is_file() => vec!["-t".into(), file.display().to_string()],
                _ => vec![home.display().to_string()],
            }
        }
        // Console follows the log live (`mrun logs -f` in a terminal).
        "log" => vec![
            "-a".into(),
            "Console".into(),
            home.join("server.log").display().to_string(),
        ],
        "chats" => vec![home.join("chats").display().to_string()],
        "url" => match body.url {
            Some(url) if url.starts_with("https://") && !url.contains(char::is_whitespace) => {
                vec![url]
            }
            _ => return error(StatusCode::BAD_REQUEST, "only https links open"),
        },
        _ => return error(StatusCode::BAD_REQUEST, "nothing to open"),
    };
    match std::process::Command::new("open").args(&args).status() {
        Ok(_) => StatusCode::NO_CONTENT.into_response(),
        Err(e) => error(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string()),
    }
}

/// The largest export Save as Markdown writes (020 research R7).
const EXPORT_MAX_BYTES: usize = 10 * 1024 * 1024;

#[derive(Debug, serde::Deserialize)]
struct ExportBody {
    name: String,
    text: String,
}

/// Save as Markdown (020 FR-009): writes `<downloads>/<name>.md`, or `<name> 2.md` and so on
/// when taken, and shows it in Finder. The page names the file, never its directory.
async fn export(State(host): State<Host>, body: Body) -> Response {
    // JSON escapes can double the text's size; the text itself is checked below.
    let Ok(bytes) = axum::body::to_bytes(body, 2 * EXPORT_MAX_BYTES + 64 * 1024).await else {
        return error(StatusCode::PAYLOAD_TOO_LARGE, "the chat is over 10 MB");
    };
    let body: ExportBody = match serde_json::from_slice(&bytes) {
        Ok(body) => body,
        Err(e) => return error(StatusCode::BAD_REQUEST, &e.to_string()),
    };
    if body.text.len() > EXPORT_MAX_BYTES {
        return error(StatusCode::PAYLOAD_TOO_LARGE, "the chat is over 10 MB");
    }
    let path = match write_export(&host.downloads, &export_name(&body.name), &body.text) {
        Ok(path) => path,
        Err(e) => return error(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string()),
    };
    if host.reveal {
        let _ = std::process::Command::new("open")
            .arg("-R")
            .arg(&path)
            .status();
    }
    axum::Json(json!({"path": path})).into_response()
}

/// `[A-Za-z0-9 ._-]`, at most 80 characters, no leading dot; "Chat" when nothing is left.
fn export_name(name: &str) -> String {
    let kept: String = name
        .chars()
        .filter(|c| c.is_ascii_alphanumeric() || " ._-".contains(*c))
        .take(80)
        .collect();
    let name = kept.trim().trim_start_matches('.').trim();
    if name.is_empty() {
        "Chat".into()
    } else {
        name.to_string()
    }
}

/// Creates `<dir>/<name>.md`, never replacing a file: a taken name gets " 2", " 3" and so on.
fn write_export(
    dir: &std::path::Path,
    name: &str,
    text: &str,
) -> std::io::Result<std::path::PathBuf> {
    use std::io::Write;
    std::fs::create_dir_all(dir)?;
    for n in 1..10_000 {
        let file = if n == 1 {
            format!("{name}.md")
        } else {
            format!("{name} {n}.md")
        };
        let path = dir.join(file);
        match std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
        {
            Ok(mut out) => {
                out.write_all(text.as_bytes())?;
                return Ok(path);
            }
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {}
            Err(e) => return Err(e),
        }
    }
    Err(std::io::Error::other("too many exports with this name"))
}

async fn focus(State(host): State<Host>) -> Response {
    host.focus.send_modify(|n| *n += 1);
    StatusCode::NO_CONTENT.into_response()
}

/// The pid of the background server on `port` (`mrun serve --detach` writes it), if any.
fn server_pid(port: u16) -> Option<u32> {
    std::fs::read_to_string(crate::cli::stop::pid_file(port))
        .ok()?
        .trim()
        .parse()
        .ok()
}

impl Host {
    /// Quitting stops the server this window started (`mrun stop`): its model leaves memory with
    /// the app. A server started elsewhere keeps serving its clients.
    pub fn stop_own_server(&self) {
        let ours = *self
            .started
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            == Some(StartedBy::Ui);
        if ours {
            let _ = std::process::Command::new(self.cli.as_ref())
                .args(["stop", "--port", &self.server_port.to_string()])
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::null())
                .status();
        }
    }
}

async fn quit(State(host): State<Host>) -> Response {
    let _ = host.quit.send(true);
    StatusCode::NO_CONTENT.into_response()
}
