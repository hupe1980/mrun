//! The members' way to the pool's model (026 FR-005): a member's agents keep talking to their
//! own Mac on loopback; its server forwards their API calls to the host over the pinned pool
//! connection, and the host's pool port hands them to its own loopback server, naming the member
//! (from its certificate) in [`super::MEMBER_HEADER`]. Responses stream back as they come.

use axum::body::Body;
use axum::http::{HeaderMap, Method, StatusCode, header};
use axum::response::{IntoResponse, Response};

/// The API paths a member forwards.
pub fn relayed(path: &str) -> bool {
    path.starts_with("/v1/")
}

/// Forwards one request to `url` with `client` and streams the answer back.
pub async fn forward(
    client: &reqwest::Client,
    method: Method,
    url: String,
    headers: &HeaderMap,
    extra: Option<(&str, &str)>,
    body: Body,
) -> Response {
    let Ok(body) = axum::body::to_bytes(body, crate::api::MAX_BODY_BYTES).await else {
        return (StatusCode::PAYLOAD_TOO_LARGE, "request too large").into_response();
    };
    let mut upstream = client.request(method, &url).body(body);
    for name in [header::CONTENT_TYPE, header::ACCEPT, header::USER_AGENT] {
        if let Some(value) = headers.get(&name) {
            upstream = upstream.header(name, value);
        }
    }
    for name in ["anthropic-version", "anthropic-beta", "x-mrun-ignore-eos"] {
        if let Some(value) = headers.get(name) {
            upstream = upstream.header(name, value);
        }
    }
    if let Some((name, value)) = extra {
        upstream = upstream.header(name, value);
    }
    let response = match upstream.send().await {
        Ok(response) => response,
        Err(e) => {
            return (
                StatusCode::BAD_GATEWAY,
                axum::Json(serde_json::json!({"error": {"code": "E060",
                    "message": format!("the pool's host stopped answering: {e}")}})),
            )
                .into_response();
        }
    };
    let mut out = Response::builder().status(response.status().as_u16());
    for (name, value) in response.headers() {
        let keep = name == header::CONTENT_TYPE
            || name == header::CACHE_CONTROL
            || name == header::RETRY_AFTER
            || name.as_str().starts_with("x-mrun-");
        if keep {
            out = out.header(name, value);
        }
    }
    out.body(Body::from_stream(response.bytes_stream()))
        .unwrap_or_else(|_| StatusCode::BAD_GATEWAY.into_response())
}
