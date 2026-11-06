//! Remote GPU mode (013, D064): what the in-pod agent (`mrun agent`) and the local relay
//! (`mrun serve --relay`) share: forwarding a request upstream with keepalives and abort on
//! disconnect, the per-API keepalive and error events, bearer checks and random tokens.

pub mod forward;
pub mod sse;

use crate::api::model::{ApiError, ApiKind};
use crate::api::{chat_completions, handler, messages, responses};
use axum::http::{HeaderMap, header};
use axum::response::Response;

/// Port the agent listens on inside the pod (the pod's one exposed HTTP port).
pub const AGENT_PORT: u16 = 8000;

/// The API a path belongs to (`count_tokens` is Messages).
pub fn api_of(path: &str) -> Option<ApiKind> {
    if path.starts_with("/v1/messages") {
        Some(ApiKind::Messages)
    } else if path.starts_with("/v1/chat/completions") {
        Some(ApiKind::Chat)
    } else if path.starts_with("/v1/responses") {
        Some(ApiKind::Responses)
    } else {
        None
    }
}

/// An error response in the shape of `api` (OpenAI's for other paths).
pub fn error_response(api: Option<ApiKind>, error: &ApiError) -> Response {
    match api {
        Some(ApiKind::Messages) => handler::error::<messages::Messages>(error),
        Some(ApiKind::Responses) => handler::error::<responses::Responses>(error),
        _ => handler::error::<chat_completions::Chat>(error),
    }
}

/// The token of an `Authorization: Bearer <token>` header (`x-api-key` is not accepted: the
/// agent is reached by mrun only).
pub fn bearer(headers: &HeaderMap) -> Option<&str> {
    let value = headers.get(header::AUTHORIZATION)?.to_str().ok()?;
    let (scheme, token) = value.split_once(' ')?;
    scheme.eq_ignore_ascii_case("bearer").then(|| token.trim())
}

/// Compares secrets in time independent of where they differ.
pub fn tokens_equal(a: &str, b: &str) -> bool {
    let (a, b) = (a.as_bytes(), b.as_bytes());
    if a.len() != b.len() {
        return false;
    }
    a.iter().zip(b).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

/// `bytes` random bytes from the OS, hex-encoded (pod tokens, vLLM keys).
pub fn random_hex(bytes: usize) -> std::io::Result<String> {
    use std::io::Read;
    let mut buf = vec![0u8; bytes];
    std::fs::File::open("/dev/urandom")?.read_exact(&mut buf)?;
    Ok(buf.iter().map(|b| format!("{b:02x}")).collect())
}

/// The bytes of lowercase or uppercase hex `text`; `None` for anything else.
pub fn from_hex(text: &str) -> Option<Vec<u8>> {
    if !text.len().is_multiple_of(2) {
        return None;
    }
    (0..text.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(text.get(i..i + 2)?, 16).ok())
        .collect()
}

/// `30m`, `2h`, `90s`, `1h30m` or plain seconds.
pub fn parse_duration(text: &str) -> Result<std::time::Duration, String> {
    let text = text.trim();
    if let Ok(seconds) = text.parse::<u64>() {
        return Ok(std::time::Duration::from_secs(seconds));
    }
    let mut total = 0u64;
    let mut number = String::new();
    for c in text.chars() {
        if c.is_ascii_digit() {
            number.push(c);
            continue;
        }
        let unit = match c {
            's' => 1,
            'm' => 60,
            'h' => 3600,
            'd' => 86_400,
            _ => return Err(format!("'{text}' is not a duration (e.g. 30m, 8h, 90s)")),
        };
        let n: u64 = number
            .parse()
            .map_err(|_| format!("'{text}' is not a duration (e.g. 30m, 8h, 90s)"))?;
        total += n * unit;
        number.clear();
    }
    if !number.is_empty() || total == 0 {
        return Err(format!("'{text}' is not a duration (e.g. 30m, 8h, 90s)"));
    }
    Ok(std::time::Duration::from_secs(total))
}

/// `1h05m`, `12m`, `40s`.
pub fn human_duration(seconds: u64) -> String {
    match seconds {
        s if s >= 3600 => format!("{}h{:02}m", s / 3600, (s % 3600) / 60),
        s if s >= 60 => format!("{}m", s / 60),
        s => format!("{s}s"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn bearer_tokens_compare_exactly() {
        let mut headers = HeaderMap::new();
        assert_eq!(bearer(&headers), None);
        headers.insert(header::AUTHORIZATION, "Bearer abc".parse().unwrap());
        assert_eq!(bearer(&headers), Some("abc"));
        headers.insert(header::AUTHORIZATION, "Basic abc".parse().unwrap());
        assert_eq!(bearer(&headers), None);
        assert!(tokens_equal("abc", "abc"));
        assert!(!tokens_equal("abc", "abd"));
        assert!(!tokens_equal("abc", "abcd"));
        let token = random_hex(32).unwrap();
        assert_eq!(token.len(), 64);
        assert_ne!(token, random_hex(32).unwrap());
    }

    #[test]
    fn durations_parse() {
        assert_eq!(parse_duration("30m"), Ok(Duration::from_secs(1800)));
        assert_eq!(parse_duration("1h30m"), Ok(Duration::from_secs(5400)));
        assert_eq!(parse_duration("90"), Ok(Duration::from_secs(90)));
        assert!(parse_duration("abc").is_err());
        assert!(parse_duration("10x").is_err());
        assert!(parse_duration("0m").is_err());
        assert_eq!(human_duration(3900), "1h05m");
        assert_eq!(human_duration(125), "2m");
        assert_eq!(human_duration(5), "5s");
        assert_eq!(api_of("/v1/messages/count_tokens"), Some(ApiKind::Messages));
        assert_eq!(api_of("/v1/models"), None);
    }
}
