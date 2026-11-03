//! A fake vLLM for tests and CPU pods (`mrun agent --stub-upstream`, `MRUN_AGENT_STUB=true`).
//!
//! It requires its API key like vLLM, rejects a `model` other than the served one (vLLM answers
//! 404), and answers the three APIs, `count_tokens`, `/v1/models`, `/version` and `/health`.
//! The last user text steers it: `slow:<s>` delays the response by `s` seconds, `long:<n>`
//! streams `n` tokens 100 ms apart, `error` answers 400; anything else gets a short reply.

use crate::api::sse;
use axum::Router;
use axum::body::{Body, Bytes};
use axum::extract::State;
use axum::http::{HeaderMap, StatusCode, Uri, header};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use serde_json::{Value, json};
use std::net::SocketAddr;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// What the stub saw (tests read it).
#[derive(Debug, Default)]
pub struct StubStats {
    /// API requests that passed the key check.
    pub requests: AtomicU64,
    /// Streams dropped before their end (the client went away).
    pub aborted: AtomicU64,
    /// The last API request body.
    pub last_body: Mutex<Option<Value>>,
}

#[derive(Clone, Debug)]
struct StubState {
    key: String,
    model: String,
    max_model_len: u32,
    stats: Arc<StubStats>,
}

/// A running stub.
#[derive(Debug)]
pub struct Stub {
    pub addr: SocketAddr,
    pub key: String,
    pub stats: Arc<StubStats>,
}

/// Starts the stub on 127.0.0.1 (a free port) serving `model`.
pub async fn start(model: &str, max_model_len: u32, key: String) -> std::io::Result<Stub> {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
    let addr = listener.local_addr()?;
    let stats = Arc::new(StubStats::default());
    let state = StubState {
        key: key.clone(),
        model: model.to_string(),
        max_model_len,
        stats: stats.clone(),
    };
    let app = Router::new()
        .route("/health", get(|| async { StatusCode::OK }))
        .route(
            "/version",
            get(|| async { axum::Json(json!({"version": "stub"})) }),
        )
        .route("/v1/models", get(models))
        .route("/v1/chat/completions", post(api))
        .route("/v1/messages", post(api))
        .route("/v1/messages/count_tokens", post(api))
        .route("/v1/responses", post(api))
        .with_state(state);
    tokio::spawn(async move {
        let _ = axum::serve(listener, app).await;
    });
    Ok(Stub { addr, key, stats })
}

fn authorized(state: &StubState, headers: &HeaderMap) -> bool {
    crate::remote::bearer(headers).is_some_and(|t| crate::remote::tokens_equal(t, &state.key))
}

fn unauthorized() -> Response {
    (
        StatusCode::UNAUTHORIZED,
        axum::Json(json!({"error": "Unauthorized"})),
    )
        .into_response()
}

async fn models(State(state): State<StubState>, headers: HeaderMap) -> Response {
    if !authorized(&state, &headers) {
        return unauthorized();
    }
    axum::Json(json!({"object": "list", "data": [{
        "id": state.model, "object": "model", "owned_by": "vllm",
        "max_model_len": state.max_model_len,
    }]}))
    .into_response()
}

/// The last user text of a Chat, Messages or Responses body.
fn last_user_text(body: &Value) -> String {
    let text_of = |content: &Value| -> String {
        match content {
            Value::String(s) => s.clone(),
            Value::Array(parts) => parts
                .iter()
                .filter_map(|p| p["text"].as_str())
                .collect::<Vec<_>>()
                .join(""),
            _ => String::new(),
        }
    };
    if let Some(messages) = body["messages"].as_array()
        && let Some(last) = messages.iter().rev().find(|m| m["role"] == "user")
    {
        return text_of(&last["content"]);
    }
    match &body["input"] {
        Value::String(s) => s.clone(),
        Value::Array(items) => items
            .iter()
            .rev()
            .find(|i| i["role"] == "user")
            .map(|i| text_of(&i["content"]))
            .unwrap_or_default(),
        _ => String::new(),
    }
}

/// Counts a stream dropped before it finished.
struct DropCounter {
    stats: Arc<StubStats>,
    finished: bool,
}

impl Drop for DropCounter {
    fn drop(&mut self) {
        if !self.finished {
            self.stats.aborted.fetch_add(1, Ordering::SeqCst);
        }
    }
}

async fn api(
    State(state): State<StubState>,
    uri: Uri,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    if !authorized(&state, &headers) {
        return unauthorized();
    }
    let Ok(body) = serde_json::from_slice::<Value>(&body) else {
        return (
            StatusCode::BAD_REQUEST,
            axum::Json(json!({"error": {"message": "not JSON"}})),
        )
            .into_response();
    };
    state.stats.requests.fetch_add(1, Ordering::SeqCst);
    *state
        .stats
        .last_body
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(body.clone());
    let model = body["model"].as_str().unwrap_or_default().to_string();
    if model != state.model {
        return (
            StatusCode::NOT_FOUND,
            axum::Json(
                json!({"error": {"message": format!("The model `{model}` does not exist."),
                "type": "NotFoundError", "code": 404}}),
            ),
        )
            .into_response();
    }
    let path = uri.path();
    if path == "/v1/messages/count_tokens" {
        return axum::Json(json!({"input_tokens": body.to_string().len() / 4})).into_response();
    }
    let text = last_user_text(&body);
    if text == "error" {
        return (
            StatusCode::BAD_REQUEST,
            axum::Json(json!({"error": {"message": "stub error", "type": "BadRequestError"}})),
        )
            .into_response();
    }
    if let Some(seconds) = text
        .strip_prefix("slow:")
        .and_then(|s| s.trim().parse::<f64>().ok())
    {
        tokio::time::sleep(Duration::from_secs_f64(seconds)).await;
    }
    let (count, gap) = match text
        .strip_prefix("long:")
        .and_then(|s| s.trim().parse().ok())
    {
        Some(n) => (n, Duration::from_millis(100)),
        None => (3, Duration::ZERO),
    };
    let pieces: Vec<String> = (0..count)
        .map(|i| {
            if count == 3 {
                ["hello", " from", " the stub"][i].to_string()
            } else {
                format!("t{i} ")
            }
        })
        .collect();
    let api = crate::remote::api_of(path);
    let stream = body["stream"].as_bool().unwrap_or(false);
    if !stream {
        return axum::Json(whole(api, &model, &pieces.concat())).into_response();
    }
    let frames = frames(
        api,
        &model,
        &pieces,
        body["stream_options"]["include_usage"] == true,
    );
    let counter = DropCounter {
        stats: state.stats.clone(),
        finished: false,
    };
    let body = futures_util::stream::unfold(
        (frames.into_iter().enumerate(), counter),
        move |(mut it, mut counter)| async move {
            match it.next() {
                Some((i, frame)) => {
                    if i > 0 && !gap.is_zero() {
                        tokio::time::sleep(gap).await;
                    }
                    Some((Ok::<_, std::io::Error>(Bytes::from(frame)), (it, counter)))
                }
                None => {
                    counter.finished = true;
                    drop(counter);
                    None
                }
            }
        },
    );
    (
        [(header::CONTENT_TYPE, "text/event-stream")],
        Body::from_stream(body),
    )
        .into_response()
}

fn whole(api: Option<crate::api::model::ApiKind>, model: &str, text: &str) -> Value {
    use crate::api::model::ApiKind;
    match api {
        Some(ApiKind::Messages) => json!({
            "id": "msg_stub", "type": "message", "role": "assistant", "model": model,
            "content": [{"type": "text", "text": text}], "stop_reason": "end_turn",
            "stop_sequence": null, "usage": {"input_tokens": 5, "output_tokens": 3}}),
        Some(ApiKind::Responses) => json!({
            "id": "resp_stub", "object": "response", "status": "completed", "model": model,
            "output": [{"type": "message", "id": "msg_stub", "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": text, "annotations": []}]}],
            "usage": {"input_tokens": 5, "output_tokens": 3, "total_tokens": 8}}),
        _ => json!({
            "id": "chatcmpl-stub", "object": "chat.completion", "created": 1, "model": model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}}),
    }
}

fn frames(
    api: Option<crate::api::model::ApiKind>,
    model: &str,
    pieces: &[String],
    include_usage: bool,
) -> Vec<String> {
    use crate::api::model::ApiKind;
    let mut out = Vec::new();
    match api {
        Some(ApiKind::Messages) => {
            out.push(sse::event(
                "message_start",
                &json!({"type": "message_start", "message": {
                "id": "msg_stub", "type": "message", "role": "assistant", "model": model,
                "content": [], "stop_reason": null, "stop_sequence": null,
                "usage": {"input_tokens": 5, "output_tokens": 0}}}),
            ));
            out.push(sse::event(
                "content_block_start",
                &json!({"type": "content_block_start",
                "index": 0, "content_block": {"type": "text", "text": ""}}),
            ));
            for piece in pieces {
                out.push(sse::event(
                    "content_block_delta",
                    &json!({"type": "content_block_delta",
                    "index": 0, "delta": {"type": "text_delta", "text": piece}}),
                ));
            }
            out.push(sse::event(
                "content_block_stop",
                &json!({"type": "content_block_stop", "index": 0}),
            ));
            out.push(sse::event(
                "message_delta",
                &json!({"type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": null},
                "usage": {"output_tokens": pieces.len()}}),
            ));
            out.push(sse::event("message_stop", &json!({"type": "message_stop"})));
        }
        Some(ApiKind::Responses) => {
            let response = |status: &str, output: Value| {
                json!({"id": "resp_stub",
                "object": "response", "status": status, "model": model, "output": output})
            };
            out.push(sse::event(
                "response.created",
                &json!({"type": "response.created",
                "response": response("in_progress", json!([]))}),
            ));
            for piece in pieces {
                out.push(sse::event(
                    "response.output_text.delta",
                    &json!({
                    "type": "response.output_text.delta", "item_id": "msg_stub",
                    "output_index": 0, "content_index": 0, "delta": piece}),
                ));
            }
            let mut done = response(
                "completed",
                json!([{"type": "message", "id": "msg_stub",
                "role": "assistant", "status": "completed",
                "content": [{"type": "output_text", "text": pieces.concat(), "annotations": []}]}]),
            );
            done["usage"] = json!({"input_tokens": 5, "output_tokens": pieces.len(),
                "total_tokens": 5 + pieces.len()});
            out.push(sse::event(
                "response.completed",
                &json!({"type": "response.completed",
                "response": done}),
            ));
        }
        _ => {
            let chunk = |delta: Value, finish: Value| {
                sse::data(&json!({"id": "chatcmpl-stub",
                "object": "chat.completion.chunk", "created": 1, "model": model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}))
            };
            out.push(chunk(
                json!({"role": "assistant", "content": ""}),
                Value::Null,
            ));
            for piece in pieces {
                out.push(chunk(json!({"content": piece}), Value::Null));
            }
            out.push(chunk(json!({}), "stop".into()));
            if include_usage {
                out.push(sse::data(&json!({"id": "chatcmpl-stub",
                    "object": "chat.completion.chunk", "created": 1, "model": model,
                    "choices": [], "usage": {"prompt_tokens": 5, "completion_tokens": 3,
                    "total_tokens": 8}})));
            }
            out.push("data: [DONE]\n\n".into());
        }
    }
    out
}
