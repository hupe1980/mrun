//! Typed errors. No variant ever carries the API key: messages come from response bodies, which
//! RunPod does not echo keys into, and transport errors are rendered without URLs' query strings
//! (mrun never puts the key in a URL).

/// What went wrong talking to RunPod.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// 401 or 403: the key is wrong, revoked, or not allowed this call (a pod-scoped key gets
    /// 403 from REST v1 `stop`).
    #[error("RunPod refused the API key (HTTP {status})")]
    Unauthorized { status: u16 },
    /// 404: no such pod or volume.
    #[error("not found")]
    NotFound,
    /// Another non-2xx answer, with the message from the body.
    #[error("RunPod answered HTTP {status}: {message}")]
    Api { status: u16, message: String },
    /// Cloudflare in front of RunPod blocked the request (error 1010: the user agent).
    #[error("Cloudflare blocked the request (error 1010, user agent)")]
    Blocked,
    /// The request did not complete (DNS, TLS, connection, timeout).
    #[error("cannot reach RunPod: {0}")]
    Transport(String),
    /// The answer was not the expected JSON.
    #[error("unexpected answer from RunPod: {0}")]
    Decode(String),
}

impl Error {
    /// Maps a non-2xx status and its body.
    pub(crate) fn from_status(status: u16, body: &str) -> Self {
        if body.contains("error code: 1010") || body.contains("Error 1010") {
            return Self::Blocked;
        }
        match status {
            401 | 403 => Self::Unauthorized { status },
            404 => Self::NotFound,
            _ => Self::Api {
                status,
                message: message_of(body),
            },
        }
    }

    /// RunPod has no machine for the request right now: the GPU type is sold out (often despite
    /// a stock label), or no host of it meets the pod's minimums. A later try may succeed.
    pub fn is_sold_out(&self) -> bool {
        let Self::Api { message, .. } = self else {
            return false;
        };
        let message = message.to_ascii_lowercase();
        message.contains("no instances currently available")
            || message.contains("does not have the resources")
    }

    pub(crate) fn transport(error: &reqwest::Error) -> Self {
        // `without_url`: the URL never carries the key, but keep error text short and stable.
        let mut text = error.to_string();
        if let Some(url) = error.url() {
            text = text.replace(url.as_str(), url.path());
        }
        Self::Transport(text)
    }
}

/// The `error`/`message` field of a JSON body, else the first 240 characters of the text.
fn message_of(body: &str) -> String {
    if let Ok(value) = serde_json::from_str::<serde_json::Value>(body) {
        for key in ["error", "message"] {
            match &value[key] {
                serde_json::Value::String(text) => return text.clone(),
                serde_json::Value::Object(map) => {
                    if let Some(serde_json::Value::String(text)) = map.get("message") {
                        return text.clone();
                    }
                }
                _ => {}
            }
        }
        if let Some(text) = value["errors"][0]["message"].as_str() {
            return text.to_string();
        }
    }
    body.chars()
        .take(240)
        .collect::<String>()
        .trim()
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn statuses_map_to_variants() {
        assert!(matches!(
            Error::from_status(403, ""),
            Error::Unauthorized { status: 403 }
        ));
        assert!(matches!(Error::from_status(404, "{}"), Error::NotFound));
        assert!(matches!(
            Error::from_status(403, "<html>error code: 1010</html>"),
            Error::Blocked
        ));
        let Error::Api { status, message } =
            Error::from_status(400, r#"{"error":"no GPU of that type is available"}"#)
        else {
            panic!()
        };
        assert_eq!(
            (status, message.as_str()),
            (400, "no GPU of that type is available")
        );
        let Error::Api { message, .. } =
            Error::from_status(500, r#"{"errors":[{"message":"boom"}]}"#)
        else {
            panic!()
        };
        assert_eq!(message, "boom");
    }

    #[test]
    fn sold_out_is_told_apart_from_other_errors() {
        let sold_out = Error::from_status(
            500,
            r#"{"error":"create pod: There are no instances currently available"}"#,
        );
        assert!(sold_out.is_sold_out());
        assert!(!Error::from_status(500, r#"{"error":"boom"}"#).is_sold_out());
        assert!(!Error::NotFound.is_sold_out());
    }
}
