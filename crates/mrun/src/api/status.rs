//! `GET /v1/models` and `GET /mrun/status` (schema v1, 002 contract).

use crate::api::handler;
use crate::api::model::ApiKind;
use crate::api::run::DEFAULT_MAX_OUTPUT;
use crate::cache::miss::{MissReport, role_str};
use crate::chat::syntax::registry::ChatSyntax;
use crate::cli::serve::ServerState;
use axum::extract::State;
use axum::response::{IntoResponse, Response};
use serde_json::{Value, json};
use std::sync::Mutex;

/// The request on the engine right now, for `mrun ps`.
#[derive(Clone, Debug, PartialEq)]
pub struct Active {
    pub id: u64,
    pub api: ApiKind,
    pub prompt_tokens: u32,
    pub cached: u32,
    /// `queued` until it starts, then `prefill` or `generating`.
    pub phase: &'static str,
    /// Prefill tokens done, of the prompt.
    pub progress: u32,
}

#[derive(Debug, Default)]
pub struct Tracker {
    active: Mutex<Option<Active>>,
}

impl Tracker {
    pub fn update(&self, f: impl FnOnce(&mut Option<Active>)) {
        f(&mut self.active.lock().unwrap_or_else(|e| e.into_inner()));
    }

    pub fn active(&self) -> Option<Active> {
        self.active
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }
}

/// `GET /v1/models`: the loaded model with this machine's capacity (OpenAI list shape; the
/// vLLM-style `max_model_len` is what OpenCode's local discovery reads).
pub async fn models(State(state): State<ServerState>) -> Response {
    let loaded = match handler::ready(&state) {
        Ok(loaded) => loaded,
        Err(error) => return crate::api::chat_completions::SHAPE.error(&error),
    };
    let capacity = loaded.handle.budget().kv_capacity_tokens;
    let max_output = state.args.max_output.unwrap_or(DEFAULT_MAX_OUTPUT);
    axum::Json(json!({
        "object": "list",
        "data": [{
            "id": loaded.model_id,
            "object": "model",
            "type": "model",
            "created": 0,
            "owned_by": "mrun",
            "display_name": loaded.model_id,
            "context_window": capacity,
            "max_model_len": capacity,
            "max_output_tokens": max_output.min(capacity.saturating_sub(1)),
        }],
    }))
    .into_response()
}

/// A miss report as `/mrun/status` and the `: mrun` comment show it.
pub fn miss_json(miss: &MissReport) -> Value {
    json!({
        "cached": miss.cached,
        "processed": miss.processed,
        "cause": miss.cause.map(|c| c.as_str()),
        "divergence": miss.divergence,
        "message": miss.message.map(|m| json!({"index": m.index, "role": role_str(m.role)})),
        "gap": miss.gap,
    })
}

/// `POST /mrun/cache/clear`: drops branches and checkpoints, and the live state when idle. While
/// a request runs the clear is queued behind it (202, live state reset after the request).
pub async fn clear_cache(State(state): State<ServerState>) -> Response {
    let loaded = match handler::ready(&state) {
        Ok(loaded) => loaded,
        Err(error) => return crate::api::chat_completions::SHAPE.error(&error),
    };
    let busy = loaded.handle.queue().running() || loaded.handle.queue().waiting() > 0;
    let receiver = match loaded.handle.clear_cache() {
        Ok(receiver) => receiver,
        Err(error) => {
            let error = crate::api::model::ApiError::new(
                crate::api::model::ErrorKind::Internal,
                error.to_string(),
            );
            return crate::api::chat_completions::SHAPE.error(&error);
        }
    };
    if busy {
        return (
            axum::http::StatusCode::ACCEPTED,
            axum::Json(json!({"branches_dropped": null, "bytes_freed": null,
                "live_reset": "after-request"})),
        )
            .into_response();
    }
    match receiver.recv_async().await {
        Ok(Some(report)) => axum::Json(json!({
            "branches_dropped": report.branches_dropped,
            "bytes_freed": report.bytes_freed,
            "live_reset": if report.live_reset_now { "now" } else { "after-request" },
        }))
        .into_response(),
        _ => axum::Json(json!({"branches_dropped": 0, "bytes_freed": 0, "live_reset": "now"}))
            .into_response(),
    }
}

/// `GET /mrun/status`
pub async fn status(State(state): State<ServerState>) -> Response {
    let mut body = json!({
        "schema_version": 1,
        "version": env!("CARGO_PKG_VERSION"),
        "model": Value::Null,
        "budget": Value::Null,
        "active": Value::Null,
        "queued": 0,
        "cache": Value::Null,
        "recording": state.args.record.is_some(),
        "power": {"assertion": false},
        "last": Value::Null,
    });
    if let Ok(loaded) = handler::ready(&state) {
        let info = loaded.handle.info();
        let budget = loaded.handle.budget();
        body["model"] = json!({
            "id": loaded.model_id,
            "identity": info.identity,
            "family": info.family,
            "syntax": ChatSyntax::get(loaded.syntax).name,
            "architecture": info.family,
            "quantization": Value::Null,
        });
        body["budget"] = json!({
            "budget": budget.budget,
            "weights": budget.weights,
            "scratch": budget.scratch,
            "kv_capacity_tokens": budget.kv_capacity_tokens,
            "kv_bytes_per_token": info.kv_bytes_per_token,
        });
        body["queued"] = loaded.handle.queue().waiting().into();
        if let Some(stats) = loaded.handle.cache_stats() {
            body["cache"] = json!({
                "live_tokens": stats.live_tokens,
                "branches": stats.branches,
                "branch_tokens": stats.branch_tokens,
                "buffer_bytes": stats.buffer_bytes,
                "checkpoints": stats.checkpoints,
                "checkpoint_bytes": stats.checkpoint_bytes,
                "prompt_tokens_total": stats.prompt_tokens_total,
                "cached_tokens_total": stats.cached_tokens_total,
                "evicted_total": stats.evicted_total,
                "pressure_events": stats.pressure_events,
                "enabled": !state.args.no_cache,
            });
            body["last"] = stats.last.as_ref().map_or(Value::Null, miss_json);
        }
        if let Some(active) = state.tracker.active() {
            body["active"] = json!({
                "id": active.id,
                "api": active.api.as_str(),
                "prompt_tokens": active.prompt_tokens,
                "cached": active.cached,
                "phase": active.phase,
                "progress": active.progress,
            });
        }
    }
    axum::Json(body).into_response()
}
