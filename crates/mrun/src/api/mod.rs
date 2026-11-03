//! HTTP API: router, the loopback Host/Origin guard, `/health`, `/metrics` and client probes.

pub mod chat_completions;
pub mod decode;
pub mod handler;
pub mod log;
pub mod messages;
pub mod metrics;
pub mod model;
pub mod record;
pub mod relay;
pub mod responses;
pub mod run;
pub mod sse;
pub mod status;

use crate::cli::serve::{Readiness, ServerState};
use axum::Router;
use axum::extract::DefaultBodyLimit;
use axum::extract::{Request, State};
use axum::http::{HeaderMap, StatusCode, header};
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, head, post};
use serde_json::json;

/// Largest request body (base64 screenshots in agent history).
pub const MAX_BODY_BYTES: usize = 64 * 1024 * 1024;

/// Builds the router. Query strings (`?beta=true`) and `anthropic-beta` headers are accepted on
/// every route (002 FR-017): nothing rejects them.
pub fn router(state: ServerState) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/api/hello", head(hello).get(hello))
        .route(
            "/v1/chat/completions",
            post(handler::serve::<chat_completions::Chat>),
        )
        .route("/v1/messages", post(handler::serve::<messages::Messages>))
        .route("/v1/models", get(status::models))
        .route("/v1/models/{id}", get(status::model))
        .route("/metrics", get(metrics::metrics))
        .route("/mrun/status", get(status::status))
        .route("/mrun/cache/clear", post(status::clear_cache))
        .route("/v1/messages/count_tokens", post(messages::count_tokens))
        .route(
            "/v1/responses",
            post(handler::serve::<responses::Responses>),
        )
        .layer(DefaultBodyLimit::max(MAX_BODY_BYTES))
        .layer(middleware::from_fn_with_state(
            state.clone(),
            loopback_guard,
        ))
        .with_state(state)
}

async fn health(State(state): State<ServerState>) -> Response {
    let readiness = state.readiness.borrow().clone();
    match readiness {
        Readiness::Ready(loaded) => (
            StatusCode::OK,
            axum::Json(json!({"status": "ok", "model": loaded.model_id, "loaded": true})),
        )
            .into_response(),
        Readiness::Loading => (
            StatusCode::SERVICE_UNAVAILABLE,
            axum::Json(json!({"status": "loading", "model": state.args.model, "loaded": false})),
        )
            .into_response(),
        Readiness::Failed(error) => (
            StatusCode::SERVICE_UNAVAILABLE,
            axum::Json(json!({"status": "error", "error": error, "loaded": false})),
        )
            .into_response(),
    }
}

async fn hello() -> StatusCode {
    StatusCode::OK
}

/// Accepts only loopback `Host` values with the server port and no foreign browser `Origin`
/// (002 FR-012). Rejections are 403 before any work; no CORS headers are ever sent.
async fn loopback_guard(
    State(state): State<ServerState>,
    request: Request,
    next: Next,
) -> Response {
    let port = state.addr.port();
    let host = request
        .headers()
        .get(header::HOST)
        .and_then(|v| v.to_str().ok())
        .map(str::to_owned)
        .or_else(|| request.uri().authority().map(|a| a.as_str().to_owned()));
    let path = request.uri().path();
    if !host.as_deref().is_some_and(|h| is_loopback_host(h, port)) {
        return forbidden(path, "Host header is not a loopback address of this server");
    }
    if !origin_allowed(request.headers(), port) {
        return forbidden(path, "cross-origin requests are not accepted");
    }
    next.run(request).await
}

/// 403 in the error shape of the API the path belongs to (contract: status codes); other
/// routes answer in the OpenAI shape.
pub(crate) fn forbidden(path: &str, message: &str) -> Response {
    let error = model::ApiError::new(model::ErrorKind::Forbidden, message);
    if path.starts_with("/v1/messages") {
        handler::error::<messages::Messages>(&error)
    } else if path.starts_with("/v1/responses") {
        handler::error::<responses::Responses>(&error)
    } else {
        handler::error::<chat_completions::Chat>(&error)
    }
}

const LOOPBACK_NAMES: [&str; 3] = ["127.0.0.1", "localhost", "[::1]"];

/// `127.0.0.1:PORT`, `localhost:PORT` or `[::1]:PORT`.
pub fn is_loopback_host(host: &str, port: u16) -> bool {
    let Some((name, host_port)) = host.rsplit_once(':') else {
        return false;
    };
    host_port.parse::<u16>().ok() == Some(port)
        && LOOPBACK_NAMES.iter().any(|n| n.eq_ignore_ascii_case(name))
}

/// No `Origin`, or the server's own (`http://<loopback>:PORT`).
pub(crate) fn origin_allowed(headers: &HeaderMap, port: u16) -> bool {
    match headers.get(header::ORIGIN) {
        None => true,
        Some(value) => value
            .to_str()
            .ok()
            .and_then(|origin| origin.strip_prefix("http://"))
            .is_some_and(|rest| is_loopback_host(rest, port)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loopback_hosts_need_the_server_port() {
        for host in [
            "127.0.0.1:8484",
            "localhost:8484",
            "LOCALHOST:8484",
            "[::1]:8484",
        ] {
            assert!(is_loopback_host(host, 8484), "{host}");
        }
        for host in [
            "127.0.0.1",
            "127.0.0.1:8485",
            "evil.example:8484",
            "127.0.0.1.evil.example:8484",
            "[::1]",
            "",
        ] {
            assert!(!is_loopback_host(host, 8484), "{host}");
        }
    }

    #[tokio::test]
    async fn xapi_origin_403_api_shape() {
        let body = |response: Response| async move {
            let status = response.status().as_u16();
            let bytes = axum::body::to_bytes(response.into_body(), 1 << 16)
                .await
                .unwrap();
            (
                status,
                serde_json::from_slice::<serde_json::Value>(&bytes).unwrap(),
            )
        };
        let (status, chat) = body(forbidden("/v1/chat/completions", "no")).await;
        assert_eq!(status, 403);
        assert_eq!(chat["error"]["type"], "invalid_request_error");
        assert_eq!(chat["error"]["code"], "forbidden");
        let (status, msg) = body(forbidden("/v1/messages/count_tokens", "no")).await;
        assert_eq!(status, 403);
        assert_eq!(msg["type"], "error");
        assert_eq!(msg["error"]["type"], "permission_error");
        let (status, resp) = body(forbidden("/v1/responses", "no")).await;
        assert_eq!(status, 403);
        assert_eq!(resp["error"]["type"], "invalid_request_error");
        assert_eq!(resp["error"]["code"], "forbidden");
        let (_, other) = body(forbidden("/mrun/status", "no")).await;
        assert_eq!(other["error"]["code"], "forbidden");
    }

    #[test]
    fn only_the_servers_own_origin_passes() {
        let mut headers = HeaderMap::new();
        assert!(origin_allowed(&headers, 8484));
        headers.insert(header::ORIGIN, "http://localhost:8484".parse().unwrap());
        assert!(origin_allowed(&headers, 8484));
        for origin in [
            "https://localhost:8484",
            "http://evil.example",
            "http://localhost:3000",
            "null",
        ] {
            headers.insert(header::ORIGIN, origin.parse().unwrap());
            assert!(!origin_allowed(&headers, 8484), "{origin}");
        }
    }
}
