//! Forwarding one request upstream (agent → vLLM, relay → agent).
//!
//! A streaming request is answered at once (status 200, `text/event-stream`) and relayed through
//! [`sse::relay`], which sends keepalives while the upstream is silent and turns a failed
//! upstream into one API error event. A non-streaming request waits for the upstream and returns
//! its status and body. Only the body and a bearer go upstream: no client headers, no query
//! string (D055: `?beta=true` and `anthropic-beta` stay local).

use super::sse::{self, ByteStream, EventTransform, Keepalive, UpstreamError};
use crate::api::model::{ApiError, ApiKind, ErrorKind};
use axum::body::{Body, Bytes};
use axum::http::{HeaderName, HeaderValue, StatusCode, header};
use axum::response::{IntoResponse, Response};
use futures_util::StreamExt;
use serde_json::Value;
use std::time::Duration;

/// Where requests go and the bearer they carry.
#[derive(Clone, Debug)]
pub struct Upstream {
    client: reqwest::Client,
    base: String,
    bearer: String,
}

impl Upstream {
    /// `read_timeout` bounds the silence between two reads (the relay uses it: the agent sends
    /// keepalives, so a long silence means a dead connection); `None` waits as long as needed.
    pub fn new(
        base: &str,
        bearer: String,
        read_timeout: Option<Duration>,
    ) -> Result<Self, reqwest::Error> {
        let mut builder = reqwest::Client::builder()
            .user_agent(mrun_runpod::USER_AGENT)
            .connect_timeout(Duration::from_secs(10));
        if let Some(timeout) = read_timeout {
            builder = builder.read_timeout(timeout);
        }
        Ok(Self {
            client: builder.build()?,
            base: base.trim_end_matches('/').to_string(),
            bearer,
        })
    }

    pub fn base(&self) -> &str {
        &self.base
    }

    /// The upstream URL of `path` (for log lines).
    fn request_url(&self, path: &str) -> String {
        format!("{}{path}", self.base)
    }

    fn request(&self, method: reqwest::Method, path: &str) -> reqwest::RequestBuilder {
        self.client
            .request(method, format!("{}{path}", self.base))
            .bearer_auth(&self.bearer)
    }

    /// `GET path` as JSON (status, `/v1/models`, `/version`), with a short timeout.
    pub async fn get_json(&self, path: &str, timeout: Duration) -> Result<Value, UpstreamError> {
        let response = self
            .request(reqwest::Method::GET, path)
            .timeout(timeout)
            .send()
            .await
            .map_err(|e| UpstreamError {
                status: None,
                message: e.to_string(),
            })?;
        let status = response.status().as_u16();
        let text = response.text().await.unwrap_or_default();
        if !(200..300).contains(&status) {
            return Err(UpstreamError::from_body(status, &text));
        }
        serde_json::from_str(&text).map_err(|e| UpstreamError {
            status: Some(status),
            message: e.to_string(),
        })
    }
}

/// How one request is forwarded.
#[derive(Default)]
pub struct Options {
    pub api: Option<ApiKind>,
    pub stream: bool,
    /// Chat `stream_options.include_usage`.
    pub include_usage: bool,
    /// Model id keepalive chunks report.
    pub model: String,
    pub keepalive: Duration,
    pub event_transform: Option<EventTransform>,
    pub json_transform: Option<JsonTransform>,
    /// Kept until the response is complete (activity accounting).
    pub guard: Option<Box<dyn Send>>,
    /// Added to the response (`mrun-backend` on the relay).
    pub headers: Vec<(HeaderName, HeaderValue)>,
    /// Given a 400 body, a corrected request body to send once more (the output-limit clamp).
    pub retry: Option<Retry>,
}

/// Applied to a non-streaming JSON body.
pub type JsonTransform = Box<dyn FnOnce(&mut Value) + Send>;

/// See [`Options::retry`].
pub type Retry = Box<dyn Fn(&str) -> Option<Bytes> + Send + Sync>;

impl std::fmt::Debug for Options {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Options")
            .field("api", &self.api)
            .field("stream", &self.stream)
            .field("model", &self.model)
            .finish_non_exhaustive()
    }
}

/// `stream` and `stream_options.include_usage` of a JSON request body.
pub fn stream_flags(body: &Value) -> (bool, bool) {
    (
        body["stream"].as_bool().unwrap_or(false),
        body["stream_options"]["include_usage"]
            .as_bool()
            .unwrap_or(false),
    )
}

/// Forwards `method path` with `body` and builds the client's response.
pub async fn forward(
    upstream: &Upstream,
    method: reqwest::Method,
    path: &str,
    body: Option<Bytes>,
    mut options: Options,
) -> Response {
    let headers = std::mem::take(&mut options.headers);
    let send = send(
        upstream.clone(),
        method,
        path.to_string(),
        body,
        options.retry.take(),
    );
    let mut response = if options.stream {
        streaming(send, options)
    } else {
        whole(send.await, options).await
    };
    for (name, value) in headers {
        response.headers_mut().insert(name, value);
    }
    response
}

/// What the upstream answered.
enum Sent {
    Ok(reqwest::Response),
    /// A non-2xx status with its body and content type.
    Status(u16, Bytes, Option<HeaderValue>),
    Unreachable(String),
}

/// Sends the request, and once more with the corrected body when `retry` fixes a 400.
async fn send(
    upstream: Upstream,
    method: reqwest::Method,
    path: String,
    body: Option<Bytes>,
    retry: Option<Retry>,
) -> Sent {
    let build = |body: Option<Bytes>| {
        let mut request = upstream.request(method.clone(), &path);
        if let Some(body) = body {
            request = request
                .header(reqwest::header::CONTENT_TYPE, "application/json")
                .body(body);
        }
        request
    };
    let mut attempt = 0;
    let mut body = body;
    loop {
        attempt += 1;
        let response = match build(body.clone()).send().await {
            Ok(response) => response,
            Err(error) => return Sent::Unreachable(error.to_string()),
        };
        let status = response.status().as_u16();
        if (200..300).contains(&status) {
            return Sent::Ok(response);
        }
        let content_type = response
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|v| HeaderValue::from_bytes(v.as_bytes()).ok());
        let bytes = match response.bytes().await {
            Ok(bytes) => bytes,
            Err(error) => return Sent::Unreachable(error.to_string()),
        };
        let text = String::from_utf8_lossy(&bytes);
        if status == 400
            && attempt == 1
            && let Some(retry) = &retry
            && let Some(corrected) = retry(&text)
        {
            eprintln!("mrun: {path} answered 400, sending once more with a clamped output limit");
            body = Some(corrected);
            continue;
        }
        // The one trace of a failed request (no prompt content: the status and the error text).
        let message = super::sse::body_message(&text);
        let detail = if message.is_empty() {
            format!(
                "no message; body {:?}",
                text.chars().take(300).collect::<String>()
            )
        } else {
            message
        };
        eprintln!(
            "mrun: {} answered {status}: {detail}",
            upstream.request_url(&path)
        );
        return Sent::Status(status, bytes, content_type);
    }
}

fn streaming(send: impl Future<Output = Sent> + Send + 'static, options: Options) -> Response {
    let upstream = upstream_stream(send);
    let keepalive = Keepalive::new(options.api, &options.model, options.include_usage);
    let body = sse::relay(
        upstream,
        keepalive,
        options.keepalive,
        options.event_transform,
        options.guard,
    );
    let mut response = Response::new(Body::from_stream(body));
    let headers = response.headers_mut();
    headers.insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("text/event-stream"),
    );
    headers.insert(header::CACHE_CONTROL, HeaderValue::from_static("no-cache"));
    headers.insert(
        HeaderName::from_static("x-accel-buffering"),
        HeaderValue::from_static("no"),
    );
    response
}

async fn whole(sent: Sent, options: Options) -> Response {
    let json = || HeaderValue::from_static("application/json");
    let unreachable = |message: String| {
        super::error_response(
            options.api,
            &ApiError::new(
                ErrorKind::Overloaded,
                format!("the remote model is unreachable: {message}"),
            )
            .with_retry_after(5),
        )
    };
    let (status, content_type, bytes) = match sent {
        Sent::Unreachable(message) => return unreachable(message),
        Sent::Status(status, bytes, content_type) => {
            (status, content_type.unwrap_or_else(json), bytes)
        }
        Sent::Ok(response) => {
            let status = response.status().as_u16();
            let content_type = response
                .headers()
                .get(reqwest::header::CONTENT_TYPE)
                .and_then(|v| HeaderValue::from_bytes(v.as_bytes()).ok())
                .unwrap_or_else(json);
            let bytes = match response.bytes().await {
                Ok(bytes) => bytes,
                Err(error) => return unreachable(error.to_string()),
            };
            let bytes = match options.json_transform {
                Some(transform) => match serde_json::from_slice::<Value>(&bytes) {
                    Ok(mut value) => {
                        transform(&mut value);
                        Bytes::from(value.to_string())
                    }
                    Err(_) => bytes,
                },
                None => bytes,
            };
            (status, content_type, bytes)
        }
    };
    drop(options.guard);
    let status = StatusCode::from_u16(status).unwrap_or(StatusCode::BAD_GATEWAY);
    (status, [(header::CONTENT_TYPE, content_type)], bytes).into_response()
}

/// The upstream's body as a stream; a transport failure or a non-2xx status becomes one error.
fn upstream_stream(send: impl Future<Output = Sent> + Send + 'static) -> ByteStream {
    Box::pin(
        futures_util::stream::once(send).flat_map(|sent| -> ByteStream {
            match sent {
                Sent::Unreachable(message) => {
                    Box::pin(futures_util::stream::iter([Err(UpstreamError {
                        status: None,
                        message: format!("the remote model is unreachable: {message}"),
                    })]))
                }
                Sent::Status(status, bytes, _) => Box::pin(futures_util::stream::iter([Err(
                    UpstreamError::from_body(status, &String::from_utf8_lossy(&bytes)),
                )])),
                Sent::Ok(response) => Box::pin(futures_util::stream::unfold(
                    Some(response),
                    |state| async move {
                        let mut response = state?;
                        match response.chunk().await {
                            Ok(Some(bytes)) => Some((Ok(bytes), Some(response))),
                            Ok(None) => None,
                            Err(error) => Some((
                                Err(UpstreamError {
                                    status: None,
                                    message: format!("the remote stream broke: {error}"),
                                }),
                                None,
                            )),
                        }
                    },
                )),
            }
        }),
    )
}

/// Input tokens named in vLLM's "context too long" message ("... your request has 40000
/// input tokens ...", "... (40000 in the messages, 32000 in the completion)").
pub fn input_tokens_in(message: &str) -> Option<u32> {
    for marker in [" input tokens", " in the messages", " tokens in the prompt"] {
        if let Some(at) = message.find(marker) {
            let digits: String = message[..at]
                .chars()
                .rev()
                .take_while(char::is_ascii_digit)
                .collect::<Vec<_>>()
                .into_iter()
                .rev()
                .collect();
            if let Ok(n) = digits.parse() {
                return Some(n);
            }
        }
    }
    None
}

/// The output-limit field a request of `api` uses (Chat prefers `max_completion_tokens`).
fn limit_field(api: Option<ApiKind>, body: &Value) -> Option<&'static str> {
    match api? {
        ApiKind::Messages => Some("max_tokens"),
        ApiKind::Responses => Some("max_output_tokens"),
        ApiKind::Chat if body.get("max_completion_tokens").is_some() => {
            Some("max_completion_tokens")
        }
        ApiKind::Chat => Some("max_tokens"),
    }
}

/// mrun clamps an output limit to the remaining context (002); vLLM refuses it instead. The
/// retry lowers the limit to `max_model_len - input - 1` when vLLM's 400 names the input size.
pub fn clamp_retry(api: Option<ApiKind>, body: Value, max_model_len: u32) -> Option<Retry> {
    let field = limit_field(api, &body)?;
    let asked = body.get(field)?.as_u64()?;
    Some(Box::new(move |error_body: &str| {
        let message = sse::body_message(error_body);
        let input = input_tokens_in(&message)?;
        let room = u64::from(max_model_len).checked_sub(u64::from(input) + 1)?;
        if room == 0 || room >= asked {
            return None;
        }
        let mut body = body.clone();
        body[field] = room.into();
        Some(Bytes::from(body.to_string()))
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn context_overflow_retries_with_the_remaining_room() {
        assert_eq!(
            input_tokens_in(
                "'max_tokens' or 'max_completion_tokens' is too large: 32000. This model's maximum context length is 65536 tokens and your request has 40000 input tokens (32000 > 65536 - 40000)."
            ),
            Some(40000)
        );
        assert_eq!(
            input_tokens_in(
                "This model's maximum context length is 65536 tokens. However, you requested 72000 tokens (40000 in the messages, 32000 in the completion)."
            ),
            Some(40000)
        );
        assert_eq!(input_tokens_in("other"), None);
        let body = json!({"model": "m", "max_tokens": 32000});
        let retry = clamp_retry(Some(ApiKind::Messages), body, 65536).unwrap();
        let error = json!({"error": {"message": "your request has 40000 input tokens"}});
        let corrected: Value = serde_json::from_slice(&retry(&error.to_string()).unwrap()).unwrap();
        assert_eq!(corrected["max_tokens"], 25535);
        // The prompt alone overflows: no retry, the error goes to the client.
        let error = json!({"error": {"message": "your request has 70000 input tokens"}});
        assert!(retry(&error.to_string()).is_none());
        let chat = json!({"max_completion_tokens": 100});
        assert!(clamp_retry(Some(ApiKind::Chat), chat, 1000).is_some());
        assert!(clamp_retry(Some(ApiKind::Responses), json!({}), 1000).is_none());
    }
}
