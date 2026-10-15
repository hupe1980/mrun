//! The request flow shared by the three APIs: readiness, JSON parsing, starting the turn, and
//! turning its ticks into a streaming or non-streaming response with mrun metadata.

use crate::api::collect::Collected;
use crate::api::model::{ApiError, ApiKind, ErrorKind, GenEvent, Warning};
use crate::api::run::{self, Tick, Turn};
use crate::api::sse;
use crate::cli::serve::{Loaded, Readiness, ServerState};
use axum::body::{Body, Bytes};
use axum::http::{HeaderValue, StatusCode, header};
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

pub fn parse(body: &Bytes) -> Result<Value, ApiError> {
    serde_json::from_slice(body)
        .map_err(|e| ApiError::invalid(format!("invalid JSON body: {e}"), "body"))
}

pub async fn start(
    state: &ServerState,
    request: crate::api::model::ChatRequest,
) -> Result<Turn, ApiError> {
    let loaded = ready(state)?;
    run::start(loaded, request, &state.args, state.timing).await
}

/// Collects a non-streaming turn; returns the folded result and in-band warnings.
pub async fn collect(mut turn: Turn) -> Result<(Collected, Vec<Warning>), ApiError> {
    let mut collected = Collected::default();
    let mut warnings = turn.warnings.clone();
    while let Some(tick) = turn.ticks.recv().await {
        let Tick::Event(event) = tick else { continue };
        note_warnings(&event, turn.requested_max, &mut warnings);
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

fn note_warnings(event: &GenEvent, requested_max: u32, warnings: &mut Vec<Warning>) {
    let warning = match event {
        GenEvent::Started { max_tokens, .. } if *max_tokens < requested_max => {
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
    headers.insert("x-mrun-max-tokens", HeaderValue::from(collected.max_tokens));
    if !warnings.is_empty() {
        let list = warnings
            .iter()
            .map(|w| w.as_str())
            .collect::<Vec<_>>()
            .join(",");
        if let Ok(value) = HeaderValue::from_str(&list) {
            headers.insert("x-mrun-warnings", value);
        }
    }
    response
}

/// An SSE response: headers now, then frames as ticks arrive. The first `: mrun` comment
/// carries the applied output limit and warnings known at the start.
pub fn stream_response(turn: Turn, mut encoder: impl StreamEncode) -> Response {
    let opening = encoder.open();
    let state = (turn, encoder, Some(opening), false);
    let stream =
        futures_util::stream::unfold(state, |(mut turn, mut encoder, opening, done)| async move {
            if let Some(frames) = opening {
                return Some((
                    Ok::<_, std::convert::Infallible>(Bytes::from(frames.concat())),
                    (turn, encoder, None, done),
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
                    if let GenEvent::Started { max_tokens, .. } = &event {
                        let mut warnings = turn.warnings.clone();
                        note_warnings(&event, turn.requested_max, &mut warnings);
                        let meta = json!({"max_tokens": max_tokens,
                        "warnings": warnings.iter().map(|w| w.as_str()).collect::<Vec<_>>()});
                        frames.push_str(&sse::comment(&format!("mrun {meta}")));
                    }
                    let done = event.is_terminal();
                    frames.push_str(&encoder.encode(&event).concat());
                    (frames, done)
                }
            };
            Some((Ok(Bytes::from(frames)), (turn, encoder, None, done)))
        });
    let mut response = Response::new(Body::from_stream(stream));
    let headers = response.headers_mut();
    headers.insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("text/event-stream"),
    );
    headers.insert(header::CACHE_CONTROL, HeaderValue::from_static("no-cache"));
    *response.status_mut() = StatusCode::OK;
    response
}
