//! `GET /v1/models` and `GET /mrun/status` (schema v1, 002 contract).

use crate::api::handler;
use crate::api::model::ApiKind;
use crate::api::run::DEFAULT_MAX_OUTPUT;
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
