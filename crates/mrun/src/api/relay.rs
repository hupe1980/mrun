//! `mrun serve --relay <agent-url>`: the local side of remote GPU mode (013 FR-012–FR-014).
//!
//! The server keeps every local surface on 127.0.0.1 behind the loopback guard (`/health`,
//! `/v1/models`, `/mrun/status`, so `launch`, `connect` and `ps` work unchanged) and relays the
//! three APIs and `count_tokens` to the agent in the pod with the pod token. It applies the
//! client fixes that do not depend on mrun's engine: the model id echo, the attribution-nonce
//! normalization (D021), unknown-field tolerance for Messages (D055) and keepalives. Engine-only
//! fields (budget, cache, miss causes, splicing) are absent, not faked; `backend` says what
//! served.

use crate::api::model::{ApiError, ApiKind, ErrorKind};
use crate::api::run::{DEFAULT_MAX_OUTPUT, KEEPALIVE};
use crate::api::{MAX_BODY_BYTES, is_loopback_host, log, messages, status};
use crate::remote::forward::{self, Options, Upstream};
use crate::remote::sse::map_event_json;
use crate::remote::{self, api_of};
use axum::Router;
use axum::body::Bytes;
use axum::extract::{DefaultBodyLimit, Path, Request, State};
use axum::http::{HeaderName, HeaderValue, StatusCode, Uri, header};
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, head, post};
use serde_json::{Value, json};
use std::net::{Ipv4Addr, SocketAddr};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::{Duration, Instant};

/// The variable holding the pod token for the relay (never a flag: flags show in `ps`).
pub const TOKEN_VARIABLE: &str = "MRUN_RELAY_TOKEN";

/// The privacy note shown wherever remote mode is visible (Principle IV).
pub const PRIVACY_NOTE: &str = "remote mode: prompts and code are sent to RunPod";

/// Top-level Messages fields forwarded to vLLM; others are dropped and logged (D055).
const MESSAGES_FIELDS: [&str; 14] = [
    "model",
    "messages",
    "max_tokens",
    "system",
    "stop_sequences",
    "stream",
    "temperature",
    "top_p",
    "top_k",
    "tools",
    "tool_choice",
    "metadata",
    "thinking",
    // mrun's own Messages extension (the page's sampling control); vLLM takes it too.
    "repetition_penalty",
];

/// The pod the relay talks to (for status; not secret).
#[derive(Clone, Debug, Default)]
pub struct PodMeta {
    pub id: Option<String>,
    pub gpu: Option<String>,
    pub cost_per_hr: Option<f64>,
    /// When the pod was created (billing starts then, before the agent runs).
    pub since_unix: Option<u64>,
}

/// How the relay runs.
#[derive(Clone, Debug)]
pub struct RelayConfig {
    pub port: u16,
    pub agent_url: String,
    pub token: String,
    pub pod: PodMeta,
    pub keepalive: Duration,
    /// How often the agent's status is polled.
    pub poll: Duration,
    pub max_output: Option<u32>,
    /// Where the pod's log is saved when the agent fails (`None`: not saved).
    pub failure_log: Option<std::path::PathBuf>,
}

#[derive(Debug, Default)]
struct Remote {
    status: Option<Value>,
    reachable: bool,
    error: Option<String>,
    /// Served model id and context, once the agent is ready.
    served: Option<(String, u32)>,
    /// When `status` was last read from the agent.
    seen: Option<Instant>,
    /// The pod's log was saved for the current failure (once per failure).
    log_saved: bool,
}

/// Why a pod that answered before no longer does, from its last status (`uptime_s`, `idle_s`,
/// `max_runtime_s`, `idle_stop_s`) aged by the time since it was read: the agent stops its own
/// pod after the maximum runtime or the idle period (013 FR-010), and RunPod's proxy then answers
/// 404 or 502 without a body.
fn gone_reason(status: &Value, since_seen: Duration) -> String {
    let aged = |key: &str| status[key].as_u64().map(|s| s + since_seen.as_secs());
    let limit = |key: &str| status[key].as_u64().filter(|&s| s > 0);
    let model = status["model"].as_str().unwrap_or("<model>");
    let why = match (
        aged("uptime_s"),
        limit("max_runtime_s"),
        aged("idle_s"),
        limit("idle_stop_s"),
    ) {
        (Some(up), Some(max), _, _) if up >= max => format!(
            "the cloud pod stopped itself after its maximum runtime ({})",
            crate::remote::human_duration(max)
        ),
        (_, _, Some(idle), Some(stop)) if idle >= stop => format!(
            "the cloud pod stopped itself after {} without requests",
            crate::remote::human_duration(stop)
        ),
        _ => "the cloud pod does not answer (stopped, or RunPod's proxy is down)".to_string(),
    };
    format!("{why}; 'mrun cloud up --model {model}' starts a new one, 'mrun cloud status' shows it")
}

#[derive(Clone, Debug)]
struct RelayState {
    config: Arc<RelayConfig>,
    addr: SocketAddr,
    upstream: Upstream,
    remote: Arc<Mutex<Remote>>,
    started: Instant,
}

impl RelayState {
    fn remote(&self) -> std::sync::MutexGuard<'_, Remote> {
        self.remote.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn served(&self) -> Option<(String, u32)> {
        self.remote().served.clone()
    }

    fn agent(&self, key: &str) -> Value {
        self.remote()
            .status
            .as_ref()
            .map_or(Value::Null, |s| s[key].clone())
    }

    /// Why a pod that was serving no longer answers the status poll, if it does not.
    fn gone(&self) -> Option<String> {
        let remote = self.remote();
        match (&remote.status, remote.seen) {
            (Some(status), Some(seen)) if !remote.reachable && remote.served.is_some() => {
                Some(gone_reason(status, seen.elapsed()))
            }
            _ => None,
        }
    }

    fn backend(&self) -> Option<String> {
        self.agent("backend").as_str().map(str::to_string)
    }

    /// The agent's failure (a failed start, or a model that stopped after it served); older
    /// agents send only the phase.
    fn failure(&self) -> String {
        match self.agent("error").as_str() {
            Some(error) => error.to_string(),
            None => {
                let tail = self.agent("log_tail");
                let last = tail
                    .as_array()
                    .and_then(|t| t.last())
                    .and_then(Value::as_str)
                    .unwrap_or("see 'mrun cloud status'");
                format!("the remote model failed to start: {last}")
            }
        }
    }
}

/// Saves the pod's full log once the agent has failed, before the pod stops itself and RunPod
/// wipes its disk: the root cause of a crash hours in is only there.
async fn save_failure_log(state: RelayState) {
    eprintln!("mrun: {}", state.failure());
    let Some(path) = state.config.failure_log.clone() else {
        return;
    };
    if crate::cli::cloud::save_log(&state.config.agent_url, &state.config.token, &path)
        .await
        .is_none()
    {
        // Try again at the next poll.
        state.remote().log_saved = false;
    }
}

/// A started relay.
#[derive(Debug)]
pub struct RunningRelay {
    pub addr: SocketAddr,
    shutdown: Option<tokio::sync::oneshot::Sender<()>>,
    task: tokio::task::JoinHandle<std::io::Result<()>>,
}

impl RunningRelay {
    pub async fn shutdown(mut self) {
        if let Some(tx) = self.shutdown.take() {
            let _ = tx.send(());
        }
        let _ = self.task.await;
    }

    pub async fn wait(self) -> std::io::Result<()> {
        self.task.await.unwrap_or(Ok(()))
    }
}

/// Binds `127.0.0.1:port` (0 picks one), starts polling the agent and serving.
pub async fn start(config: RelayConfig) -> std::io::Result<RunningRelay> {
    let listener = tokio::net::TcpListener::bind((Ipv4Addr::LOCALHOST, config.port)).await?;
    let addr = listener.local_addr()?;
    // The agent sends a keepalive every 10 s; 60 s of silence is a dead connection.
    let upstream = Upstream::new(
        &config.agent_url,
        config.token.clone(),
        Some(Duration::from_secs(60)),
    )
    .map_err(std::io::Error::other)?;
    let state = RelayState {
        config: Arc::new(config),
        addr,
        upstream,
        remote: Arc::default(),
        started: Instant::now(),
    };
    tokio::spawn(poll(state.clone()));
    let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel::<()>();
    let app = router(state);
    let task = tokio::spawn(async move {
        axum::serve(listener, app)
            .with_graceful_shutdown(async move {
                tokio::select! {
                    _ = shutdown_rx => {}
                    () = crate::cli::serve::terminate() => {}
                }
            })
            .await
    });
    Ok(RunningRelay {
        addr,
        shutdown: Some(shutdown_tx),
        task,
    })
}

/// Polls the agent's status; learns the served model and context once it is ready.
async fn poll(state: RelayState) {
    loop {
        match state
            .upstream
            .get_json("/mrun/status", Duration::from_secs(10))
            .await
        {
            Ok(status) => {
                let ready = status["phase"] == "ready";
                let failed = status["phase"] == "failed";
                let save_log = {
                    let mut remote = state.remote();
                    remote.status = Some(status.clone());
                    remote.seen = Some(Instant::now());
                    remote.reachable = true;
                    remote.error = None;
                    if failed {
                        // A failed agent serves nothing: forget the model so requests get the
                        // failure, not a forward to a dead vLLM.
                        remote.served = None;
                    }
                    let first = failed && !remote.log_saved;
                    remote.log_saved = failed;
                    first
                };
                if save_log {
                    tokio::spawn(save_failure_log(state.clone()));
                }
                if ready
                    && state.served().is_none()
                    && let Ok(models) = state
                        .upstream
                        .get_json("/v1/models", Duration::from_secs(10))
                        .await
                {
                    let model = &models["data"][0];
                    if let Some(id) = model["id"].as_str() {
                        let context = model["max_model_len"]
                            .as_u64()
                            .or_else(|| status["max_model_len"].as_u64())
                            .unwrap_or(0) as u32;
                        state.remote().served = Some((id.to_string(), context));
                    }
                }
            }
            Err(error) => {
                let mut remote = state.remote();
                remote.reachable = false;
                remote.error = Some(match error.status {
                    Some(401) => "the agent refused the pod token".to_string(),
                    _ => error.message,
                });
            }
        }
        tokio::time::sleep(state.config.poll).await;
    }
}

fn router(state: RelayState) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/api/hello", head(hello).get(hello))
        .route("/v1/models", get(models))
        .route("/v1/models/{id}", get(model))
        .route("/mrun/status", get(relay_status))
        .route("/mrun/sessions", get(sessions))
        .route("/v1/messages", post(api))
        .route("/v1/messages/count_tokens", post(api))
        .route("/v1/chat/completions", post(api))
        .route("/v1/responses", post(api))
        .layer(DefaultBodyLimit::max(MAX_BODY_BYTES))
        .layer(middleware::from_fn_with_state(state.clone(), guard))
        .with_state(state)
}

/// The loopback Host/Origin guard of the local server (002 FR-012).
async fn guard(State(state): State<RelayState>, request: Request, next: Next) -> Response {
    let port = state.addr.port();
    let host = request
        .headers()
        .get(header::HOST)
        .and_then(|v| v.to_str().ok())
        .map(str::to_owned);
    let path = request.uri().path().to_string();
    if !host.as_deref().is_some_and(|h| is_loopback_host(h, port)) {
        return crate::api::forbidden(
            &path,
            "Host header is not a loopback address of this server",
        );
    }
    if !crate::api::origin_allowed(request.headers(), port) {
        return crate::api::forbidden(&path, "cross-origin requests are not accepted");
    }
    next.run(request).await
}

async fn hello() -> StatusCode {
    StatusCode::OK
}

/// Session replay covers models on this Mac (025): the relay has no miss reports to replay.
async fn sessions() -> Response {
    axum::Json(
        json!({"schema_version": 1, "kept": 0, "dropped": 0, "remote": true,
        "sessions": [], "requests": []}),
    )
    .into_response()
}

async fn health(State(state): State<RelayState>) -> Response {
    // A stopped pod is not "ok": `launch` and clients would start against a dead model.
    if let Some(message) = state.gone() {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            axum::Json(json!({"status": "error", "loaded": false, "error": message})),
        )
            .into_response();
    }
    if let Some((id, _)) = state.served() {
        return axum::Json(json!({"status": "ok", "model": id, "loaded": true,
            "backend": state.backend()}))
        .into_response();
    }
    let phase = state.agent("phase");
    if phase == "failed" {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            axum::Json(json!({"status": "error", "loaded": false, "error": state.failure()})),
        )
            .into_response();
    }
    (
        StatusCode::SERVICE_UNAVAILABLE,
        axum::Json(json!({"status": "loading", "model": state.agent("model"), "loaded": false})),
    )
        .into_response()
}

fn listed(state: &RelayState) -> Option<(String, u32, u32)> {
    let (id, context) = state.served()?;
    let max_output = state
        .config
        .max_output
        .unwrap_or(DEFAULT_MAX_OUTPUT)
        .min(context.saturating_sub(1));
    Some((id, context, max_output))
}

async fn models(State(state): State<RelayState>) -> Response {
    let data: Vec<Value> = listed(&state)
        .map(|(id, context, max_output)| status::model_json(&id, Some(context), Some(max_output)))
        .into_iter()
        .collect();
    let first = data.first().map(|m| m["id"].clone()).unwrap_or(Value::Null);
    axum::Json(json!({"object": "list", "data": data, "has_more": false,
        "first_id": first, "last_id": first}))
    .into_response()
}

async fn model(State(state): State<RelayState>, Path(id): Path<String>) -> Response {
    match listed(&state) {
        Some((_, context, max_output)) => {
            axum::Json(status::model_json(&id, Some(context), Some(max_output))).into_response()
        }
        None => (
            StatusCode::NOT_FOUND,
            axum::Json(json!({"type": "error", "error": {"type": "not_found_error",
                "message": "the remote model is not ready"}})),
        )
            .into_response(),
    }
}

/// `/mrun/status` schema v1 with engine-only fields null, plus `backend` and `pod`.
async fn relay_status(State(state): State<RelayState>) -> Response {
    let (agent, reachable, error, seen) = {
        let remote = state.remote();
        (
            remote.status.clone().unwrap_or(Value::Null),
            remote.reachable,
            remote.error.clone(),
            remote.seen,
        )
    };
    let served = state.served();
    let pod = &state.config.pod;
    let active = agent["active"].as_u64().unwrap_or(0);
    let cost = pod.cost_per_hr.or_else(|| agent["cost_per_hr"].as_f64());
    // A pod that answered before and no longer does has most likely stopped (it stops itself,
    // 013 FR-010): uptime and spend are counted up to when it was last seen, never further.
    let unseen_s = seen
        .filter(|_| !reachable)
        .map(|seen| seen.elapsed().as_secs());
    let (uptime, spend) = spend(
        pod.since_unix,
        cost,
        crate::api::sse::unix_now(),
        unseen_s,
        &agent,
    );
    let body = json!({
        "schema_version": 1,
        "version": env!("CARGO_PKG_VERSION"),
        "numerics": null,
        "pid": std::process::id(),
        "model": served.as_ref().map(|(id, _)| json!({"id": id, "identity": null,
            "family": "remote", "syntax": null})),
        "budget": null,
        "active": null,
        "queued": 0,
        "cache": null,
        "recording": false,
        "power": {"assertion": false},
        "last": null,
        "requests": [],
        "lanes": {"configured": 1, "busy": active},
        "backend": agent["backend"],
        "context": served.as_ref().map(|(_, context)| *context),
        "pod": {
            "id": pod.id.clone().or_else(|| agent["pod_id"].as_str().map(str::to_string)),
            "gpu": pod.gpu.clone().or_else(|| agent["gpu"].as_str().map(str::to_string)),
            "cost_per_hr": cost,
            "url": state.config.agent_url,
            "reachable": reachable,
            "error": error,
            "phase": agent["phase"],
            "reason": agent["reason"],
            "uptime_s": uptime,
            "spend_usd": spend,
            "unreachable_s": unseen_s,
            "idle_s": agent["idle_s"],
            "idle_stop_s": agent["idle_stop_s"],
            "max_runtime_s": agent["max_runtime_s"],
            "requests_total": agent["requests_total"],
            "log_tail": agent["log_tail"],
            "relay_uptime_s": state.started.elapsed().as_secs(),
        },
        "remote_note": PRIVACY_NOTE,
    });
    axum::Json(body).into_response()
}

/// Uptime and spend of the pod: from its creation (billing starts then) to `now`, or to when it
/// was last seen if it stopped answering (`unseen_s` ago); the agent's own figures when the
/// creation time or price is unknown.
fn spend(
    since_unix: Option<u64>,
    cost: Option<f64>,
    now: u64,
    unseen_s: Option<u64>,
    agent: &Value,
) -> (Value, Value) {
    match (since_unix, cost) {
        (Some(since), Some(cost)) => {
            let until = now.saturating_sub(unseen_s.unwrap_or(0));
            let uptime = until.saturating_sub(since);
            (
                Value::from(uptime),
                Value::from((cost * uptime as f64 / 3600.0 * 1000.0).round() / 1000.0),
            )
        }
        _ => (agent["uptime_s"].clone(), agent["spend_usd"].clone()),
    }
}

/// Sets the `model` a response reports (the requested id, 002 FR-018) where present.
pub fn echo_model(value: &mut Value, model: &str) {
    for pointer in ["/model", "/message/model", "/response/model"] {
        if let Some(slot) = value.pointer_mut(pointer)
            && slot.is_string()
        {
            *slot = model.into();
        }
    }
}

/// D021 on the relay: zeroes Claude Code's `cch=` nonce in a leading attribution system text
/// (vLLM's prefix cache needs a stable prefix too), and drops Messages fields vLLM does not
/// take (D055, logged).
fn prepare_messages(body: &mut Value) {
    match &mut body["system"] {
        Value::String(text) => *text = messages::normalize_billing_nonce(text),
        Value::Array(blocks) => {
            if let Some(Value::String(text)) = blocks.first_mut().and_then(|b| b.get_mut("text")) {
                *text = messages::normalize_billing_nonce(text);
            }
        }
        _ => {}
    }
    if let Some(map) = body.as_object_mut() {
        let dropped: Vec<String> = map
            .keys()
            .filter(|k| !MESSAGES_FIELDS.contains(&k.as_str()))
            .cloned()
            .collect();
        for key in dropped {
            map.remove(&key);
            log::ignored_field(&key);
        }
    }
}

async fn api(State(state): State<RelayState>, uri: Uri, body: Bytes) -> Response {
    let path = uri.path();
    let api = api_of(path);
    // A pod that was serving and no longer answers the status poll: say why instead of passing on
    // the proxy's bare 404/502.
    if let Some(message) = state.gone() {
        return remote::error_response(api, &ApiError::new(ErrorKind::Internal, message));
    }
    let Some((served, _)) = state.served() else {
        let error = match state.agent("phase").as_str() {
            Some("failed") => ApiError::new(ErrorKind::Internal, state.failure()),
            _ => ApiError::new(ErrorKind::NotReady, "the remote model is loading")
                .with_retry_after(5),
        };
        return remote::error_response(api, &error);
    };
    let mut json = match serde_json::from_slice::<Value>(&body) {
        Ok(value @ Value::Object(_)) => value,
        _ => {
            return remote::error_response(
                api,
                &ApiError::new(ErrorKind::InvalidRequest, "the body is not a JSON object"),
            );
        }
    };
    let requested = json["model"]
        .as_str()
        .filter(|m| !m.is_empty())
        .unwrap_or(&served)
        .to_string();
    json["model"] = served.into();
    if api == Some(ApiKind::Messages) {
        prepare_messages(&mut json);
        // vLLM requires `max_tokens`; mrun gives a missing one the server default (leniency).
        if path == "/v1/messages" && json.get("max_tokens").is_none() {
            let max_output = listed(&state).map_or(DEFAULT_MAX_OUTPUT, |(_, _, m)| m);
            json["max_tokens"] = max_output.into();
        }
    }
    let (stream, include_usage) = forward::stream_flags(&json);
    let echo = requested.clone();
    let echo_whole = requested.clone();
    let mut headers = Vec::new();
    if let Some(backend) = state.backend()
        && let Ok(value) = HeaderValue::from_str(&backend)
    {
        headers.push((HeaderName::from_static("mrun-backend"), value));
    }
    let options = Options {
        api,
        stream,
        include_usage,
        model: requested,
        keepalive: state.config.keepalive,
        event_transform: Some(Box::new(move |event: &str| {
            if event.contains("\"model\"") {
                map_event_json(event, |v| echo_model(v, &echo))
            } else {
                event.to_string()
            }
        })),
        json_transform: Some(Box::new(move |v: &mut Value| echo_model(v, &echo_whole))),
        headers,
        ..Options::default()
    };
    forward::forward(
        &state.upstream,
        reqwest::Method::POST,
        path,
        Some(Bytes::from(json.to_string())),
        options,
    )
    .await
}

/// `mrun serve --relay`: runs until SIGTERM or Ctrl-C. Returns the exit code.
pub fn run(port: u16, agent_url: &str, pod: PodMeta, max_output: Option<u32>) -> u8 {
    let token = std::env::var(TOKEN_VARIABLE).unwrap_or_default();
    if token.is_empty() {
        eprintln!(
            "error: {TOKEN_VARIABLE} is not set\n  hint: 'mrun cloud up' starts the relay with the pod token"
        );
        return 2;
    }
    let runtime = match tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
    {
        Ok(runtime) => runtime,
        Err(error) => {
            eprintln!("error: cannot start the async runtime: {error}");
            return 1;
        }
    };
    let config = RelayConfig {
        port,
        agent_url: agent_url.to_string(),
        token,
        failure_log: pod.id.as_deref().map(crate::cli::cloud::saved_log_path),
        pod,
        // The agent sends keepalives every 10 s; the relay's own come 2 s later, so they only
        // fill in when the agent's are lost (no doubled keepalives on a healthy link).
        keepalive: KEEPALIVE + Duration::from_secs(2),
        poll: Duration::from_secs(2),
        max_output,
    };
    runtime.block_on(async move {
        let relay = match start(config).await {
            Ok(relay) => relay,
            Err(error) => {
                eprintln!(
                    "error: cannot listen on 127.0.0.1:{port}: {error}\n  hint: is another server running? Try --port or 'mrun ps'."
                );
                return 1;
            }
        };
        crate::cli::info(&format!(
            "mrun relaying http://{} to {agent_url} ({PRIVACY_NOTE})",
            relay.addr
        ));
        match relay.wait().await {
            Ok(()) => 0,
            Err(error) => {
                eprintln!("error: relay failed: {error}");
                1
            }
        }
    })
}

#[cfg(test)]
mod tests {
    #[test]
    fn a_gone_pod_says_which_limit_stopped_it() {
        let status = json!({"model": "Qwen/Qwen3-Coder-Next-FP8", "uptime_s": 10_790,
            "max_runtime_s": 10_800, "idle_s": 5, "idle_stop_s": 1_800});
        let reason = super::gone_reason(&status, Duration::from_secs(30));
        assert!(reason.contains("maximum runtime (3h"), "{reason}");
        assert!(
            reason.contains("mrun cloud up --model Qwen/Qwen3-Coder-Next-FP8"),
            "{reason}"
        );
        let idle = json!({"uptime_s": 100, "max_runtime_s": 10_800, "idle_s": 1_790,
            "idle_stop_s": 1_800});
        assert!(super::gone_reason(&idle, Duration::from_secs(20)).contains("without requests"));
        let fresh = json!({"uptime_s": 100, "max_runtime_s": 10_800, "idle_s": 1,
            "idle_stop_s": 1_800});
        assert!(super::gone_reason(&fresh, Duration::from_secs(2)).contains("does not answer"));
    }

    use super::*;

    #[test]
    fn spend_stops_when_the_pod_stops_answering() {
        let none = serde_json::Value::Null;
        // Created at 1,000; now 1,000 + 9.5 h; unseen for the last 1.5 h.
        let now = 1_000 + 34_200;
        let (up, spent) = spend(Some(1_000), Some(1.69), now, Some(5_400), &none);
        assert_eq!(up, 28_800, "counted to the last answer, not to now");
        assert_eq!(spent.as_f64(), Some(13.52));
        let (up, _) = spend(Some(1_000), Some(1.69), now, None, &none);
        assert_eq!(up, 34_200, "a pod that answers is counted to now");
        let agent = serde_json::json!({"uptime_s": 7, "spend_usd": 0.1});
        assert_eq!(spend(None, Some(1.69), now, None, &agent).0, 7);
    }

    #[test]
    fn messages_bodies_get_the_nonce_zeroed_and_unknown_fields_dropped() {
        let mut body = json!({
            "model": "claude-x",
            "system": [{"type": "text", "text": "x-anthropic-billing-header: cc_version=2.1; cch=9f3c1;\nYou are Claude Code."}],
            "messages": [],
            "max_tokens": 10,
            "repetition_penalty": 1.1,
            "context_management": {"edits": []},
            "output_config": {"effort": "high"},
        });
        prepare_messages(&mut body);
        assert!(
            body["system"][0]["text"]
                .as_str()
                .unwrap()
                .contains("cch=00000;")
        );
        assert!(body.get("context_management").is_none());
        assert!(body.get("output_config").is_none());
        assert_eq!(body["max_tokens"], 10);
        assert_eq!(body["repetition_penalty"], 1.1, "forwarded (019 FR-019)");
        let mut plain = json!({"system": "x-anthropic-billing-header: cch=12;\nhi"});
        prepare_messages(&mut plain);
        assert_eq!(plain["system"], "x-anthropic-billing-header: cch=00;\nhi");
    }

    #[test]
    fn the_requested_model_is_echoed_where_responses_name_one() {
        let mut chat = json!({"model": "served", "choices": []});
        echo_model(&mut chat, "asked");
        assert_eq!(chat["model"], "asked");
        let mut start = json!({"type": "message_start", "message": {"model": "served"}});
        echo_model(&mut start, "asked");
        assert_eq!(start["message"]["model"], "asked");
        let mut created = json!({"response": {"model": "served"}});
        echo_model(&mut created, "asked");
        assert_eq!(created["response"]["model"], "asked");
        let mut none = json!({"type": "ping"});
        echo_model(&mut none, "asked");
        assert_eq!(none, json!({"type": "ping"}));
    }
}
