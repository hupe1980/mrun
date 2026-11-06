//! The request flow shared by the three APIs: readiness, body limits, JSON parsing, `--record`,
//! starting the turn, and turning its events into a streaming or non-streaming response with
//! mrun metadata. Each API supplies its decoder, encoder and error shape through [`Api`].
//!
//! A non-streaming body is the stream encoder's snapshot after the same events, so the two
//! forms cannot diverge.

use crate::api::decode::DecodeOptions;
use crate::api::metrics::Outcome;
use crate::api::model::{ApiError, ApiKind, ChatRequest, ErrorKind, Finish, GenEvent, Warning};
use crate::api::run::{self, Turn};
use crate::api::sse;
use crate::api::status::Stage;
use crate::cache::miss::{MissReport, role_str};
use crate::cli::serve::{Loaded, Readiness, ServerState};
use axum::body::{Body, Bytes};
use axum::extract::State;
use axum::extract::rejection::BytesRejection;
use axum::http::{HeaderMap, HeaderValue, StatusCode, header};
use axum::response::{IntoResponse, Response};
use serde_json::{Value, json};
use std::sync::Arc;
use std::time::Duration;
use tokio::time::Instant;

/// The encoder of one API: SSE frames per event, and the non-streaming body they add up to.
pub trait StreamEncode: Send + 'static {
    fn encode(&mut self, event: &GenEvent) -> Vec<String>;
    fn keepalive(&mut self) -> String;
    /// Frames sent with the headers, before anything is queued (Responses `response.created`).
    fn open(&mut self) -> Vec<String> {
        Vec::new()
    }
    /// The non-streaming body: what the events encoded so far amount to.
    fn snapshot(&self) -> Value;
}

/// One wire API: its decoder, encoder and error shape.
pub trait Api: Send + Sync + 'static {
    const KIND: ApiKind;
    type Encoder: StreamEncode;
    fn decode(body: &Value, options: DecodeOptions) -> Result<ChatRequest, ApiError>;
    /// The encoder for `request`; `model` is the id responses report.
    fn encoder(model: &str, request: &ChatRequest) -> Self::Encoder;
    fn error_body(error: &ApiError) -> Value;
    /// A start error of a streaming request that the client handles only in-stream (Responses
    /// prompt-too-long, which Codex compacts on).
    fn in_stream(_error: &ApiError) -> bool {
        false
    }
}

/// An error response in `A`'s shape, with `retry-after` when set.
pub fn error<A: Api>(error: &ApiError) -> Response {
    let mut response = (error.status(A::KIND), axum::Json(A::error_body(error))).into_response();
    if let Some(seconds) = error.retry_after {
        response
            .headers_mut()
            .insert(header::RETRY_AFTER, HeaderValue::from(seconds));
    }
    response
}

pub fn ready(state: &ServerState) -> Result<Arc<Loaded>, ApiError> {
    match &*state.readiness.borrow() {
        Readiness::Ready(loaded) => Ok(loaded.clone()),
        Readiness::Loading => {
            Err(ApiError::new(ErrorKind::NotReady, "the model is loading").with_retry_after(5))
        }
        // A failed load is permanent: a server error, never overload that invites retries.
        Readiness::Failed(error) => Err(ApiError::new(ErrorKind::Internal, error.clone())),
    }
}

/// The body, or the API error for one that could not be read (413 over the limit).
pub fn body(body: Result<Bytes, BytesRejection>) -> Result<Bytes, ApiError> {
    body.map_err(|rejection| {
        if rejection.status() == StatusCode::PAYLOAD_TOO_LARGE {
            ApiError::new(
                ErrorKind::PayloadTooLarge,
                format!(
                    "the request body is larger than {} bytes",
                    crate::api::MAX_BODY_BYTES
                ),
            )
        } else {
            ApiError::invalid(rejection.body_text(), "body")
        }
    })
}

pub fn parse(body: &Bytes) -> Result<Value, ApiError> {
    serde_json::from_slice(body)
        .map_err(|e| ApiError::invalid(format!("invalid JSON body: {e}"), "body"))
}

/// `x-mrun-ignore-eos: on` (FR-031).
fn ignore_eos(headers: &HeaderMap) -> bool {
    headers
        .get("x-mrun-ignore-eos")
        .is_some_and(|v| v.as_bytes().eq_ignore_ascii_case(b"on"))
}

/// The loaded model, unless the server is loading or shutting down.
fn accepting(state: &ServerState) -> Result<Arc<Loaded>, ApiError> {
    if state.tracker.stage() != Stage::Serving {
        return Err(
            ApiError::new(ErrorKind::Overloaded, "the server is shutting down").with_retry_after(5),
        );
    }
    ready(state)
}

/// `POST` of one API: decode, start, then stream or collect.
pub async fn serve<A: Api>(
    State(state): State<ServerState>,
    headers: HeaderMap,
    bytes: Result<Bytes, BytesRejection>,
) -> Response {
    let rejected = |e: &ApiError| {
        state
            .tracker
            .metrics
            .record_outcome(A::KIND, Outcome::Rejected);
        error::<A>(e)
    };
    let parsed = body(bytes).and_then(|bytes| {
        let value = parse(&bytes);
        if let Some(recorder) = &state.recorder {
            recorder.record(A::KIND, &headers, &bytes, value.is_ok());
        }
        value
    });
    let mut request = match parsed.and_then(|value| A::decode(&value, state.decode_options())) {
        Ok(request) => request,
        Err(e) => return rejected(&e),
    };
    request.ignore_eos = ignore_eos(&headers);
    request.client = crate::chat::splice::client_class(
        headers
            .get(header::USER_AGENT)
            .and_then(|v| v.to_str().ok()),
    );
    request.member = headers
        .get(crate::pool::MEMBER_HEADER)
        .and_then(|v| v.to_str().ok())
        .filter(|id| id.len() == 16 && id.chars().all(|c| c.is_ascii_hexdigit()))
        .map(str::to_string);
    let loaded = match accepting(&state) {
        Ok(loaded) => loaded,
        Err(e) => return rejected(&e),
    };
    let stream = request.stream;
    let encoder = A::encoder(&run::model_id(&loaded, &request), &request);
    let turn = match run::start(
        loaded,
        request,
        &state.args,
        state.timing,
        state.tracker.clone(),
    )
    .await
    {
        Ok(turn) => turn,
        Err(e) if stream && A::in_stream(&e) => {
            state
                .tracker
                .metrics
                .record_outcome(A::KIND, Outcome::Rejected);
            return stream_error_response(encoder, e);
        }
        Err(e) => return rejected(&e),
    };
    if stream {
        return stream_response(turn, encoder, state.timing.keepalive);
    }
    match collect(turn, encoder).await {
        Ok((body, meta)) => {
            let mut response = axum::Json(body).into_response();
            meta.headers(response.headers_mut());
            response
        }
        Err(e) => error::<A>(&e),
    }
}

/// Runs a non-streaming turn through `encoder`; returns its snapshot and the metadata.
pub async fn collect<E: StreamEncode>(
    mut turn: Turn,
    mut encoder: E,
) -> Result<(Value, Meta), ApiError> {
    let mut meta = Meta::of(&turn);
    while let Some(event) = turn.events.recv().await {
        meta.observe(&event, &turn);
        if let GenEvent::Failed(error) = event {
            return Err(error);
        }
        encoder.encode(&event);
        if event.is_terminal() {
            return Ok((encoder.snapshot(), meta));
        }
    }
    Err(ApiError::new(
        ErrorKind::Internal,
        "the request ended without a result",
    ))
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
        GenEvent::Finished {
            reason: Finish::Repetition,
            ..
        } => Warning::RepetitionStopped,
        _ => return,
    };
    if !warnings.contains(&warning) {
        warnings.push(warning);
    }
}

/// mrun's response metadata: `x-mrun-*` headers on JSON bodies, `: mrun {…}` comments in
/// streams (reuse and the applied limit first, warnings before the terminal event).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Meta {
    pub input: u32,
    pub cached: u32,
    pub max_tokens: u32,
    pub miss: Option<MissReport>,
    pub spliced: u32,
    pub warnings: Vec<Warning>,
}

impl Meta {
    /// What is known when the turn starts: splicing and render warnings.
    pub fn of(turn: &Turn) -> Self {
        Self {
            spliced: turn.spliced,
            warnings: turn.warnings.clone(),
            ..Self::default()
        }
    }

    /// Notes reuse and the applied limit from `Started`, and in-band warnings.
    pub fn observe(&mut self, event: &GenEvent, turn: &Turn) {
        note_warnings(
            event,
            turn.explicit_max.then_some(turn.requested_max),
            &mut self.warnings,
        );
        if let GenEvent::Started {
            input,
            cached,
            max_tokens,
            miss,
        } = event
        {
            self.input = *input;
            self.cached = *cached;
            self.max_tokens = *max_tokens;
            self.miss = *miss;
        }
    }

    fn processed(&self) -> u32 {
        self.input - self.cached.min(self.input)
    }

    fn warning_list(&self) -> String {
        self.warnings
            .iter()
            .map(|w| w.as_str())
            .collect::<Vec<_>>()
            .join(",")
    }

    pub fn headers(&self, headers: &mut HeaderMap) {
        headers.insert("x-mrun-cache-cached", HeaderValue::from(self.cached));
        headers.insert(
            "x-mrun-cache-processed",
            HeaderValue::from(self.processed()),
        );
        headers.insert("x-mrun-max-tokens", HeaderValue::from(self.max_tokens));
        if self.spliced > 0 {
            headers.insert("x-mrun-spliced", HeaderValue::from(self.spliced));
        }
        if let Some(miss) = &self.miss {
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
            if miss.unreused > 0 {
                headers.insert("x-mrun-cache-unreused", HeaderValue::from(miss.unreused));
            }
            if miss.ring_wrapped {
                headers.insert(
                    "x-mrun-cache-ring-wrapped",
                    HeaderValue::from_static("true"),
                );
            }
        }
        if !self.warnings.is_empty()
            && let Ok(value) = HeaderValue::from_str(&self.warning_list())
        {
            headers.insert("x-mrun-warning", value);
        }
    }

    /// The `: mrun` comment opening a stream: reuse, the miss report and the applied limit.
    pub fn start_comment(&self) -> String {
        let mut meta = json!({"cached": self.cached, "processed": self.processed(),
            "max_tokens": self.max_tokens});
        if self.spliced > 0 {
            meta["spliced"] = self.spliced.into();
        }
        if let Some(miss) = &self.miss
            && let Value::Object(fields) = crate::api::status::miss_json(miss)
        {
            // Reuse comes from `Started` above; the rest of the miss report as `/mrun/status`
            // shows it.
            for (key, value) in fields {
                if key != "cached" && key != "processed" {
                    meta[key] = value;
                }
            }
        }
        sse::comment(&format!("mrun {meta}"))
    }

    /// The `: mrun` comment before the terminal event, when there are warnings.
    pub fn end_comment(&self) -> Option<String> {
        let list: Vec<&str> = self.warnings.iter().map(|w| w.as_str()).collect();
        (!list.is_empty()).then(|| sse::comment(&format!("mrun {}", json!({"warnings": list}))))
    }
}

/// The state of a streamed response.
struct Streaming<E> {
    turn: Turn,
    encoder: E,
    opening: Option<Vec<String>>,
    meta: Meta,
    keepalive: Duration,
    /// When the last bytes went out: a keepalive follows `keepalive` after them, whatever the
    /// engine does (D033).
    last_sent: Instant,
    done: bool,
}

impl<E: StreamEncode> Streaming<E> {
    /// The next bytes to send; `None` at the end.
    async fn next(&mut self) -> Option<Bytes> {
        if let Some(frames) = self.opening.take()
            && !frames.is_empty()
        {
            return Some(self.sent(frames.concat()));
        }
        loop {
            if self.done {
                return None;
            }
            let deadline = self.last_sent + self.keepalive;
            let event = match tokio::time::timeout_at(deadline, self.turn.events.recv()).await {
                Err(_) => {
                    let frame = self.encoder.keepalive();
                    return Some(self.sent(frame));
                }
                Ok(None) => return None,
                Ok(Some(event)) => event,
            };
            let frames = self.frames(&event);
            if !frames.is_empty() {
                return Some(self.sent(frames));
            }
        }
    }

    fn frames(&mut self, event: &GenEvent) -> String {
        let mut frames = String::new();
        self.meta.observe(event, &self.turn);
        if matches!(event, GenEvent::Started { .. }) {
            frames.push_str(&self.meta.start_comment());
        }
        self.done = event.is_terminal();
        if self.done
            && let Some(comment) = self.meta.end_comment()
        {
            frames.push_str(&comment);
        }
        frames.push_str(&self.encoder.encode(event).concat());
        frames
    }

    fn sent(&mut self, frames: String) -> Bytes {
        self.last_sent = Instant::now();
        Bytes::from(frames)
    }
}

/// An SSE response: headers now, then frames as events arrive, and a keepalive whenever
/// `keepalive` passes without bytes.
pub fn stream_response(
    turn: Turn,
    mut encoder: impl StreamEncode,
    keepalive: Duration,
) -> Response {
    let opening = encoder.open();
    let meta = Meta::of(&turn);
    let state = Streaming {
        turn,
        encoder,
        opening: Some(opening),
        meta,
        keepalive,
        last_sent: Instant::now(),
        done: false,
    };
    let stream = futures_util::stream::unfold(state, |mut state| async move {
        let bytes = state.next().await?;
        Some((Ok::<_, std::convert::Infallible>(bytes), state))
    });
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::model::{Finish, Usage};
    use tokio::sync::mpsc;

    struct Echo;

    impl StreamEncode for Echo {
        fn encode(&mut self, event: &GenEvent) -> Vec<String> {
            match event {
                GenEvent::TextDelta(text) => vec![format!("data: {text}\n\n")],
                GenEvent::Finished { .. } => vec!["data: end\n\n".into()],
                _ => Vec::new(),
            }
        }

        fn keepalive(&mut self) -> String {
            ": keepalive\n\n".into()
        }

        fn snapshot(&self) -> Value {
            Value::Null
        }
    }

    fn turn() -> (mpsc::Sender<GenEvent>, Turn) {
        let (tx, events) = mpsc::channel(64);
        let turn = Turn {
            events,
            requested_max: 10,
            explicit_max: false,
            warnings: Vec::new(),
            spliced: 0,
        };
        (tx, turn)
    }

    fn streaming(turn: Turn, keepalive: Duration) -> Streaming<Echo> {
        Streaming {
            turn,
            encoder: Echo,
            opening: None,
            meta: Meta::default(),
            keepalive,
            last_sent: Instant::now(),
            done: false,
        }
    }

    // Paused tokio time (the stream's clock): the keepalive window is exact, not wall-clock, so the
    // test cannot flake on a loaded machine.
    #[tokio::test(start_paused = true)]
    async fn xapi_keepalive_follows_the_last_byte_not_engine_events() {
        let (tx, turn) = turn();
        let keepalive = Duration::from_millis(400);
        let mut stream = streaming(turn, keepalive);
        // Events that produce no bytes (buffered tool-call deltas) arrive more often than the
        // keepalive interval: the keepalive still comes one interval after the last byte.
        let feeder = tokio::spawn(async move {
            for _ in 0..8 {
                tokio::time::sleep(Duration::from_millis(100)).await;
                tx.send(GenEvent::ToolArgsDelta {
                    idx: 0,
                    json: "x".into(),
                })
                .await
                .unwrap();
            }
            tx
        });
        let start = Instant::now();
        let bytes = stream.next().await.unwrap();
        assert_eq!(&bytes[..], b": keepalive\n\n");
        let elapsed = start.elapsed();
        assert!(
            elapsed >= keepalive && elapsed < keepalive * 2,
            "{elapsed:?}"
        );
        let tx = feeder.await.unwrap();
        tx.send(GenEvent::TextDelta("hi".into())).await.unwrap();
        assert_eq!(&stream.next().await.unwrap()[..], b"data: hi\n\n");
        tx.send(GenEvent::Finished {
            reason: Finish::Stop,
            stop_seq: None,
            usage: Usage::default(),
        })
        .await
        .unwrap();
        assert_eq!(&stream.next().await.unwrap()[..], b"data: end\n\n");
        assert!(stream.next().await.is_none());
    }

    #[test]
    fn meta_headers_and_comments_agree() {
        let meta = Meta {
            input: 10,
            cached: 4,
            max_tokens: 99,
            miss: None,
            spliced: 7,
            warnings: vec![Warning::ToolArgsInvalid],
        };
        let mut headers = HeaderMap::new();
        meta.headers(&mut headers);
        assert_eq!(headers["x-mrun-cache-processed"], "6");
        assert_eq!(headers["x-mrun-spliced"], "7");
        assert_eq!(headers["x-mrun-warning"], "tool-args-invalid");
        let comment = meta.start_comment();
        assert!(comment.contains("\"processed\":6") && comment.contains("\"spliced\":7"));
        assert!(meta.end_comment().unwrap().contains("tool-args-invalid"));
    }

    #[test]
    fn meta_carries_the_whole_miss_report() {
        use crate::cache::miss::{Cause, MessageRef};
        use crate::chat::prompt::Role;
        let meta = Meta {
            input: 100,
            cached: 40,
            max_tokens: 9,
            miss: Some(MissReport {
                prompt: 100,
                cached: 40,
                processed: 60,
                divergence: Some(50),
                message: Some(MessageRef {
                    index: 2,
                    role: Role::Assistant,
                }),
                gap: 300,
                unreused: 20,
                ring_wrapped: true,
                cause: Some(Cause::Diverged),
            }),
            ..Meta::default()
        };
        let mut headers = HeaderMap::new();
        meta.headers(&mut headers);
        for (name, value) in [
            ("x-mrun-cache-cause", "diverged"),
            ("x-mrun-cache-divergence", "50"),
            ("x-mrun-cache-message", "2:assistant"),
            ("x-mrun-cache-gap", "300"),
            ("x-mrun-cache-unreused", "20"),
            ("x-mrun-cache-ring-wrapped", "true"),
        ] {
            assert_eq!(headers[name], value, "{name}");
        }
        let comment = meta.start_comment();
        let json: Value =
            serde_json::from_str(comment.trim().strip_prefix(": mrun ").unwrap()).unwrap();
        assert_eq!(
            json,
            json!({"cached": 40, "processed": 60, "max_tokens": 9, "cause": "diverged",
                "divergence": 50, "message": {"index": 2, "role": "assistant"}, "gap": 300,
                "unreused": 20, "ring_wrapped": true})
        );
    }
}
