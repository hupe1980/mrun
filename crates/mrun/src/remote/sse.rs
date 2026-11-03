//! Relaying an upstream SSE stream: whole events only (split at a blank line), a keepalive in
//! the API's own format once `interval` passes without a byte sent (never inside an event), an
//! optional per-event transform, and one API error event when the upstream fails.
//!
//! The relay owns the upstream stream: when the client goes away the server drops the relay,
//! which drops the upstream response and closes its connection (the upstream aborts the request,
//! 013 FR-009).

use crate::api::model::ApiKind;
use crate::api::sse;
use axum::body::Bytes;
use futures_util::{Stream, StreamExt};
use serde_json::{Value, json};
use std::collections::VecDeque;
use std::pin::Pin;
use std::time::Duration;
use tokio::time::Instant;

/// Why the upstream stream ended early.
#[derive(Clone, Debug, PartialEq)]
pub struct UpstreamError {
    /// The upstream's HTTP status; `None` when it could not be reached or the stream broke.
    pub status: Option<u16>,
    pub message: String,
}

impl UpstreamError {
    /// From a non-2xx status and its body (`error.message`, `message` or `detail`, else text).
    pub fn from_body(status: u16, body: &str) -> Self {
        let message = body_message(body);
        Self {
            status: Some(status),
            // A client shows the message alone: never an empty one.
            message: if message.is_empty() {
                format!("the remote model answered HTTP {status} without a message")
            } else {
                message
            },
        }
    }
}

/// The message of an error body in any of the three APIs' shapes or FastAPI's.
pub fn body_message(body: &str) -> String {
    if let Ok(value) = serde_json::from_str::<Value>(body) {
        for candidate in [
            &value["error"]["message"],
            &value["message"],
            &value["detail"],
            &value["error"],
        ] {
            if let Some(text) = candidate.as_str().filter(|t| !t.trim().is_empty()) {
                return text.to_string();
            }
        }
        return String::new();
    }
    let text: String = body.chars().take(300).collect();
    text.trim().to_string()
}

/// Upstream bytes.
pub type ByteStream = Pin<Box<dyn Stream<Item = Result<Bytes, UpstreamError>> + Send>>;

/// A transform applied to each whole event's text (without the trailing blank line).
pub type EventTransform = Box<dyn FnMut(&str) -> String + Send>;

/// The keepalive of one API, learning ids from the events it sees.
#[derive(Debug)]
pub struct Keepalive {
    api: Option<ApiKind>,
    model: String,
    /// Chat: the upstream chunk id and `created`.
    chat: Option<(String, u64)>,
    /// Chat: `stream_options.include_usage` (keepalive chunks then carry `usage: null`).
    include_usage: bool,
    /// Responses: the upstream response object.
    response: Option<Value>,
}

impl Keepalive {
    pub fn new(api: Option<ApiKind>, model: &str, include_usage: bool) -> Self {
        Self {
            api,
            model: model.to_string(),
            chat: None,
            include_usage,
            response: None,
        }
    }

    /// Learns the chunk id (Chat) or response object (Responses) from an upstream event.
    pub fn observe(&mut self, event: &str) {
        let Some(data) = data_json(event) else { return };
        match self.api {
            Some(ApiKind::Chat) if self.chat.is_none() => {
                if let Some(id) = data["id"].as_str() {
                    self.chat = Some((id.to_string(), data["created"].as_u64().unwrap_or(0)));
                }
            }
            Some(ApiKind::Responses) if data["response"]["id"].is_string() => {
                self.response = Some(data["response"].clone());
            }
            _ => {}
        }
    }

    /// The keepalive frame (002 formats: Messages `ping`, Chat an empty-`choices` chunk,
    /// Responses `response.in_progress`; an SSE comment where no format applies yet).
    pub fn frame(&self) -> String {
        match self.api {
            Some(ApiKind::Messages) => sse::event("ping", &json!({"type": "ping"})),
            Some(ApiKind::Chat) => {
                let (id, created) = self
                    .chat
                    .clone()
                    .unwrap_or_else(|| ("chatcmpl-keepalive".into(), sse::unix_now()));
                let mut chunk = json!({
                    "id": id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": self.model,
                    "choices": [],
                });
                if self.include_usage {
                    chunk["usage"] = Value::Null;
                }
                sse::data(&chunk)
            }
            Some(ApiKind::Responses) => match &self.response {
                Some(response) => {
                    let mut response = response.clone();
                    response["status"] = "in_progress".into();
                    response["output"] = json!([]);
                    sse::event(
                        "response.in_progress",
                        &json!({"type": "response.in_progress", "response": response}),
                    )
                }
                None => sse::comment("keepalive"),
            },
            None => sse::comment("keepalive"),
        }
    }

    /// One error event in the API's stream format.
    pub fn error_frame(&self, error: &UpstreamError) -> String {
        let (anthropic, openai, code) = match error.status {
            Some(400 | 404 | 413 | 422) => (
                "invalid_request_error",
                "invalid_request_error",
                "invalid_request",
            ),
            Some(401 | 403) => (
                "authentication_error",
                "invalid_request_error",
                "invalid_api_key",
            ),
            Some(429 | 503 | 529) | None => ("overloaded_error", "server_error", "server_busy"),
            Some(_) => ("api_error", "server_error", "server_error"),
        };
        let message = &error.message;
        match self.api {
            Some(ApiKind::Messages) => sse::event(
                "error",
                &json!({"type": "error", "error": {"type": anthropic, "message": message}}),
            ),
            Some(ApiKind::Responses) => {
                let mut response = self.response.clone().unwrap_or_else(|| {
                    json!({"id": sse::new_id("resp_"), "object": "response",
                           "created_at": sse::unix_now(), "model": self.model, "output": []})
                });
                response["status"] = "failed".into();
                response["error"] = json!({"code": code, "message": message});
                sse::event(
                    "response.failed",
                    &json!({"type": "response.failed", "response": response}),
                )
            }
            _ => sse::data(&json!({"error": {"message": message, "type": openai,
                "param": null, "code": code}})),
        }
    }
}

/// The JSON of an event's `data:` lines.
pub fn data_json(event: &str) -> Option<Value> {
    let data: Vec<&str> = event
        .lines()
        .filter_map(|line| line.strip_prefix("data:"))
        .map(|rest| rest.strip_prefix(' ').unwrap_or(rest))
        .collect();
    if data.is_empty() {
        return None;
    }
    serde_json::from_str(&data.join("\n")).ok()
}

/// Rewrites the JSON of an event's `data:` with `f`; other lines (and non-JSON data such as
/// `[DONE]`) stay as they are.
pub fn map_event_json(event: &str, f: impl FnOnce(&mut Value)) -> String {
    let Some(mut value) = data_json(event) else {
        return event.to_string();
    };
    f(&mut value);
    let mut out = String::with_capacity(event.len());
    let mut wrote = false;
    for line in event.lines() {
        if line.starts_with("data:") {
            if !wrote {
                out.push_str("data: ");
                out.push_str(&value.to_string());
                out.push('\n');
                wrote = true;
            }
        } else {
            out.push_str(line);
            out.push('\n');
        }
    }
    out.pop();
    out
}

/// Position and length of the first event terminator (`\n\n` or `\r\n\r\n`).
fn boundary(buf: &[u8]) -> Option<(usize, usize)> {
    let lf = buf.windows(2).position(|w| w == b"\n\n").map(|p| (p, 2));
    let crlf = buf
        .windows(4)
        .position(|w| w == b"\r\n\r\n")
        .map(|p| (p, 4));
    match (lf, crlf) {
        (Some(a), Some(b)) => Some(if a.0 <= b.0 { a } else { b }),
        (a, b) => a.or(b),
    }
}

struct State {
    upstream: ByteStream,
    buf: Vec<u8>,
    out: VecDeque<Bytes>,
    keepalive: Keepalive,
    interval: Duration,
    last_sent: Instant,
    transform: Option<EventTransform>,
    done: bool,
    /// Held until the relay ends or is dropped (activity accounting).
    _guard: Option<Box<dyn Send>>,
}

impl State {
    fn push_events(&mut self) {
        while let Some((at, len)) = boundary(&self.buf) {
            let raw: Vec<u8> = self.buf.drain(..at + len).collect();
            let text = String::from_utf8_lossy(&raw[..at]).into_owned();
            self.keepalive.observe(&text);
            let text = match &mut self.transform {
                Some(transform) => transform(&text),
                None => text,
            };
            self.out.push_back(Bytes::from(format!("{text}\n\n")));
        }
    }
}

/// Relays `upstream` to the client (see the module doc). `guard` is dropped when the relay ends.
pub fn relay(
    upstream: ByteStream,
    keepalive: Keepalive,
    interval: Duration,
    transform: Option<EventTransform>,
    guard: Option<Box<dyn Send>>,
) -> impl Stream<Item = Result<Bytes, std::io::Error>> + Send {
    let state = State {
        upstream,
        buf: Vec::new(),
        out: VecDeque::new(),
        keepalive,
        interval,
        last_sent: Instant::now(),
        transform,
        done: false,
        _guard: guard,
    };
    futures_util::stream::unfold(state, |mut st| async move {
        loop {
            if let Some(bytes) = st.out.pop_front() {
                st.last_sent = Instant::now();
                return Some((Ok(bytes), st));
            }
            if st.done {
                return None;
            }
            let deadline = st.last_sent + st.interval;
            tokio::select! {
                item = st.upstream.next() => match item {
                    Some(Ok(bytes)) => {
                        st.buf.extend_from_slice(&bytes);
                        st.push_events();
                    }
                    Some(Err(error)) => {
                        if !st.buf.is_empty() {
                            // A cut-off event is dropped; the error event follows.
                            st.buf.clear();
                        }
                        let frame = st.keepalive.error_frame(&error);
                        st.out.push_back(Bytes::from(frame));
                        st.done = true;
                    }
                    None => {
                        if !st.buf.is_empty() {
                            let rest = std::mem::take(&mut st.buf);
                            let mut text = String::from_utf8_lossy(&rest).into_owned();
                            if let Some(transform) = &mut st.transform {
                                text = transform(&text);
                            }
                            st.out.push_back(Bytes::from(format!("{text}\n\n")));
                        }
                        st.done = true;
                    }
                },
                () = tokio::time::sleep_until(deadline) => {
                    if st.buf.is_empty() {
                        st.out.push_back(Bytes::from(st.keepalive.frame()));
                    } else {
                        // Inside an event: wait for its end rather than corrupt it.
                        st.last_sent = Instant::now();
                    }
                }
            }
        }
    })
}

#[cfg(test)]
mod tests {
    #[test]
    fn an_upstream_error_never_has_an_empty_message() {
        for body in [
            r#"{"error":{"message":"","type":"BadRequestError"}}"#,
            r#"{"message":"  "}"#,
            "",
        ] {
            assert_eq!(
                super::UpstreamError::from_body(404, body).message,
                "the remote model answered HTTP 404 without a message",
                "{body:?}"
            );
        }
        let vllm = r#"{"object":"error","message":"too long","type":"BadRequestError","code":400}"#;
        assert_eq!(
            super::UpstreamError::from_body(400, vllm).message,
            "too long"
        );
        assert_eq!(
            super::UpstreamError::from_body(502, "<html>Bad gateway</html>").message,
            "<html>Bad gateway</html>"
        );
    }

    use super::*;
    use std::time::Instant as StdInstant;

    fn chunks(items: Vec<(u64, &'static str)>) -> ByteStream {
        Box::pin(futures_util::stream::unfold(
            items.into_iter(),
            |mut it| async move {
                let (delay, text) = it.next()?;
                tokio::time::sleep(Duration::from_millis(delay)).await;
                Some((Ok(Bytes::from(text)), it))
            },
        ))
    }

    async fn collect(
        stream: impl Stream<Item = Result<Bytes, std::io::Error>>,
    ) -> Vec<(u128, String)> {
        let start = StdInstant::now();
        let mut out = Vec::new();
        futures_util::pin_mut!(stream);
        while let Some(item) = stream.next().await {
            out.push((
                start.elapsed().as_millis(),
                String::from_utf8(item.unwrap().to_vec()).unwrap(),
            ));
        }
        out
    }

    #[tokio::test]
    async fn split_events_are_never_interleaved_with_keepalives() {
        // One event arrives in two halves 300 ms apart; the interval is 100 ms.
        let upstream = chunks(vec![
            (0, "event: message_start\ndata: {\"type\":"),
            (300, "\"message_start\"}\n\nevent: x\ndata: {}\n\n"),
        ]);
        let ka = Keepalive::new(Some(ApiKind::Messages), "m", false);
        let frames = collect(relay(upstream, ka, Duration::from_millis(100), None, None)).await;
        let texts: Vec<&str> = frames.iter().map(|(_, t)| t.as_str()).collect();
        assert_eq!(
            texts,
            [
                "event: message_start\ndata: {\"type\":\"message_start\"}\n\n",
                "event: x\ndata: {}\n\n"
            ]
        );
    }

    #[tokio::test]
    async fn keepalives_follow_the_last_byte_in_each_apis_format() {
        for (api, expect) in [
            (Some(ApiKind::Messages), "event: ping\n"),
            (Some(ApiKind::Chat), "\"choices\":[]"),
            (Some(ApiKind::Responses), ": keepalive"),
        ] {
            let upstream = chunks(vec![(
                250,
                "data: {\"id\":\"chatcmpl-1\",\"created\":7}\n\n",
            )]);
            let ka = Keepalive::new(api, "served", false);
            let frames = collect(relay(upstream, ka, Duration::from_millis(100), None, None)).await;
            assert!(frames.len() >= 3, "{frames:?}");
            assert!(frames[0].1.contains(expect), "{api:?} {frames:?}");
            assert!(frames[0].0 >= 90 && frames[0].0 < 200, "{frames:?}");
            assert!(frames.last().unwrap().1.contains("chatcmpl-1"));
        }
    }

    #[test]
    fn chat_and_responses_keepalives_carry_upstream_ids() {
        let mut ka = Keepalive::new(Some(ApiKind::Chat), "m", true);
        ka.observe("data: {\"id\":\"chatcmpl-9\",\"created\":5,\"choices\":[]}");
        let frame = data_json(&ka.frame()).unwrap();
        assert_eq!(frame["id"], "chatcmpl-9");
        assert_eq!(frame["created"], 5);
        assert!(frame["usage"].is_null() && frame.get("usage").is_some());
        let mut ka = Keepalive::new(Some(ApiKind::Responses), "m", false);
        ka.observe(
            "event: response.created\ndata: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_1\",\"status\":\"in_progress\",\"output\":[]}}",
        );
        let frame = ka.frame();
        assert!(frame.starts_with("event: response.in_progress\n"));
        assert_eq!(data_json(&frame).unwrap()["response"]["id"], "resp_1");
    }

    #[tokio::test]
    async fn an_upstream_error_becomes_one_api_error_event() {
        for (api, check) in [
            (Some(ApiKind::Messages), "event: error"),
            (Some(ApiKind::Chat), "\"error\""),
            (Some(ApiKind::Responses), "event: response.failed"),
        ] {
            let upstream: ByteStream = Box::pin(futures_util::stream::iter([Err(
                UpstreamError::from_body(400, r#"{"error":{"message":"bad tool"}}"#),
            )]));
            let ka = Keepalive::new(api, "m", false);
            let frames = collect(relay(upstream, ka, Duration::from_secs(10), None, None)).await;
            assert_eq!(frames.len(), 1);
            assert!(frames[0].1.contains(check), "{frames:?}");
            assert!(frames[0].1.contains("bad tool"));
            assert!(data_json(&frames[0].1).is_some());
        }
    }

    #[tokio::test]
    async fn transforms_apply_per_event() {
        let upstream = chunks(vec![(
            0,
            "data: {\"model\":\"served\"}\r\n\r\ndata: [DONE]\n\n",
        )]);
        let ka = Keepalive::new(Some(ApiKind::Chat), "m", false);
        let transform: EventTransform =
            Box::new(|event| map_event_json(event, |v| v["model"] = "asked".into()));
        let frames = collect(relay(
            upstream,
            ka,
            Duration::from_secs(10),
            Some(transform),
            None,
        ))
        .await;
        let texts: Vec<&str> = frames.iter().map(|(_, t)| t.as_str()).collect();
        assert_eq!(
            texts,
            ["data: {\"model\":\"asked\"}\n\n", "data: [DONE]\n\n"]
        );
        assert_eq!(body_message(r#"{"detail":"nope"}"#), "nope");
        assert_eq!(body_message("plain"), "plain");
    }
}
