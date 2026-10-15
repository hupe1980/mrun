//! The request flow shared by the three APIs: readiness, JSON parsing, starting the turn, and
//! turning its ticks into a streaming or non-streaming response with mrun metadata.

use crate::api::collect::Collected;
use crate::api::model::{ApiError, ApiKind, ErrorKind, GenEvent, Warning};
use crate::api::run::{self, Tick, Turn};
use crate::api::sse;
use crate::cache::miss::{MissReport, role_str};
use crate::cli::serve::{Loaded, Readiness, ServerState};
use axum::body::{Body, Bytes};
use axum::http::{HeaderMap, HeaderValue, StatusCode, header};
use axum::response::{IntoResponse, Response};
use serde_json::{Value, json};
use std::sync::Arc;

/// Streaming encoder of one API.
pub trait StreamEncode: Send + 'static {
    fn encode(&mut self, event: &GenEvent) -> Vec<String>;
    fn keepalive(&mut self) -> String;
    /// Frames sent with the headers, before anything is queued (Responses `response.created`).
    fn open(&mut self) -> Vec<String> {
        Vec::new()
    }
}

/// How one API shapes errors and complete bodies.
#[derive(Debug)]
pub struct ApiShape {
    pub api: ApiKind,
    pub error_body: fn(&ApiError) -> Value,
}

impl ApiShape {
    pub fn error(&self, error: &ApiError) -> Response {
        let mut response =
            (error.status(self.api), axum::Json((self.error_body)(error))).into_response();
        if let Some(seconds) = error.retry_after {
            response
                .headers_mut()
                .insert(header::RETRY_AFTER, HeaderValue::from(seconds));
        }
        response
    }
}

pub fn ready(state: &ServerState) -> Result<Arc<Loaded>, ApiError> {
    match &*state.readiness.borrow() {
        Readiness::Ready(loaded) => Ok(loaded.clone()),
        Readiness::Loading => {
            Err(ApiError::new(ErrorKind::NotReady, "the model is loading").with_retry_after(5))
        }
        Readiness::Failed(error) => Err(ApiError::new(ErrorKind::NotReady, error.clone())),
    }
}

/// Records the request when `--record` is on, and applies mrun's own request headers.
pub fn observe(
    state: &ServerState,
    api: ApiKind,
    headers: &HeaderMap,
    body: &Bytes,
) -> RequestFlags {
    if let Some(recorder) = &state.recorder {
        recorder.record(api, headers, body);
    }
    RequestFlags {
        ignore_eos: headers
            .get("x-mrun-ignore-eos")
            .is_some_and(|v| v.as_bytes().eq_ignore_ascii_case(b"on")),
    }
}

/// Request options carried in mrun headers.
#[derive(Clone, Copy, Debug, Default)]
pub struct RequestFlags {
    pub ignore_eos: bool,
}

pub fn parse(body: &Bytes) -> Result<Value, ApiError> {
    serde_json::from_slice(body)
        .map_err(|e| ApiError::invalid(format!("invalid JSON body: {e}"), "body"))
}

pub async fn start(
    state: &ServerState,
    request: crate::api::model::ChatRequest,
) -> Result<Turn, ApiError> {
    let loaded = ready(state)?;
    run::start(
        loaded,
        request,
        &state.args,
        state.timing,
        state.tracker.clone(),
    )
    .await
}

/// Collects a non-streaming turn; returns the folded result and in-band warnings.
pub async fn collect(mut turn: Turn) -> Result<(Collected, Vec<Warning>), ApiError> {
    let mut collected = Collected::default();
    let mut warnings = turn.warnings.clone();
    while let Some(tick) = turn.ticks.recv().await {
        let Tick::Event(event) = tick else { continue };
        note_warnings(
            &event,
            turn.explicit_max.then_some(turn.requested_max),
            &mut warnings,
        );
        let terminal = event.is_terminal();
        collected.push(event);
        if terminal {
            break;
        }
    }
    if let Some(error) = collected.error.take() {
        return Err(error);
    }
    if collected.finish.is_none() {
        return Err(ApiError::new(
            ErrorKind::Internal,
            "the request ended without a result",
        ));
    }
    Ok((collected, warnings))
}

/// `requested_max`: the client's own limit; a clamped server default is not reported.
fn note_warnings(event: &GenEvent, requested_max: Option<u32>, warnings: &mut Vec<Warning>) {
    let warning = match event {
        GenEvent::Started { max_tokens, .. }
            if requested_max.is_some_and(|requested| *max_tokens < requested) =>
        {
            Warning::MaxTokensClamped
        }
        GenEvent::ToolEnd { valid: false, .. } => Warning::ToolArgsInvalid,
        _ => return,
    };
    if !warnings.contains(&warning) {
        warnings.push(warning);
    }
}

/// A JSON response with mrun metadata headers.
pub fn json_response(body: Value, collected: &Collected, warnings: &[Warning]) -> Response {
    let mut response = axum::Json(body).into_response();
    let headers = response.headers_mut();
    headers.insert("x-mrun-cache-cached", HeaderValue::from(collected.cached));
    headers.insert(
        "x-mrun-cache-processed",
        HeaderValue::from(collected.input - collected.cached.min(collected.input)),
    );
    headers.insert("x-mrun-max-tokens", HeaderValue::from(collected.max_tokens));
    if let Some(miss) = &collected.miss {
        if let Some(cause) = miss.cause {
            headers.insert(
                "x-mrun-cache-cause",
                HeaderValue::from_static(cause.as_str()),
            );
        }
        if let Some(at) = miss.divergence {
            headers.insert("x-mrun-cache-divergence", HeaderValue::from(at));
        }
        if let Some(message) = miss.message
            && let Ok(value) =
                HeaderValue::from_str(&format!("{}:{}", message.index, role_str(message.role)))
        {
            headers.insert("x-mrun-cache-message", value);
        }
        if miss.gap > 0 {
            headers.insert("x-mrun-cache-gap", HeaderValue::from(miss.gap));
        }
    }
    if !warnings.is_empty() {
        let list = warnings
            .iter()
            .map(|w| w.as_str())
            .collect::<Vec<_>>()
            .join(",");
        if let Ok(value) = HeaderValue::from_str(&list) {
            headers.insert("x-mrun-warning", value);
        }
    }
    response
}

/// The `: mrun` comment opening a stream: reuse, the miss report and the applied limit.
fn start_comment(input: u32, cached: u32, max_tokens: u32, miss: Option<&MissReport>) -> String {
    let mut meta = json!({"cached": cached, "processed": input - cached.min(input),
        "max_tokens": max_tokens});
    if let Some(miss) = miss {
        meta["cause"] = miss.cause.map(|c| c.as_str()).into();
        meta["divergence"] = miss.divergence.into();
        meta["message"] = miss
            .message
            .map(|m| json!({"index": m.index, "role": role_str(m.role)}))
            .unwrap_or(Value::Null);
        meta["gap"] = miss.gap.into();
    }
    sse::comment(&format!("mrun {meta}"))
}

/// An SSE response: headers now, then frames as ticks arrive. `: mrun` comments carry the
/// metadata: reuse and the applied limit before the first event, warnings before the last.
pub fn stream_response(turn: Turn, mut encoder: impl StreamEncode) -> Response {
    let opening = encoder.open();
    let warnings = turn.warnings.clone();
    let state = (turn, encoder, Some(opening), false, warnings);
    let stream = futures_util::stream::unfold(
        state,
        |(mut turn, mut encoder, opening, done, mut warnings)| async move {
            if let Some(frames) = opening {
                return Some((
                    Ok::<_, std::convert::Infallible>(Bytes::from(frames.concat())),
                    (turn, encoder, None, done, warnings),
                ));
            }
            if done {
                return None;
            }
            let tick = turn.ticks.recv().await?;
            let (frames, done) = match tick {
                Tick::Keepalive => (encoder.keepalive(), false),
                Tick::Event(event) => {
                    let mut frames = String::new();
                    note_warnings(
                        &event,
                        turn.explicit_max.then_some(turn.requested_max),
                        &mut warnings,
                    );
                    if let GenEvent::Started {
                        input,
                        cached,
                        max_tokens,
                        miss,
                    } = &event
                    {
                        frames.push_str(&start_comment(
                            *input,
                            *cached,
                            *max_tokens,
                            miss.as_ref(),
                        ));
                    }
                    let done = event.is_terminal();
                    if done && !warnings.is_empty() {
                        let list: Vec<&str> = warnings.iter().map(|w| w.as_str()).collect();
                        let meta = json!({"warnings": list});
                        frames.push_str(&sse::comment(&format!("mrun {meta}")));
                    }
                    frames.push_str(&encoder.encode(&event).concat());
                    (frames, done)
                }
            };
            Some((
                Ok(Bytes::from(frames)),
                (turn, encoder, None, done, warnings),
            ))
        },
    );
    sse_response(Body::from_stream(stream))
}

/// An SSE response that reports `error` in-stream: the encoder's opening frames, then its
/// failure event (Responses prompt-too-long, which Codex handles only in this form).
pub fn stream_error_response(mut encoder: impl StreamEncode, error: ApiError) -> Response {
    let mut frames = encoder.open();
    frames.extend(encoder.encode(&GenEvent::Failed(error)));
    sse_response(Body::from(frames.concat()))
}

fn sse_response(body: Body) -> Response {
    let mut response = Response::new(body);
    let headers = response.headers_mut();
    headers.insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("text/event-stream"),
    );
    headers.insert(header::CACHE_CONTROL, HeaderValue::from_static("no-cache"));
    *response.status_mut() = StatusCode::OK;
    response
}
