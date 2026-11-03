//! Server-sent event framing and ids shared by the three encoders.

use serde_json::Value;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

/// `data: <json>\n\n`
pub fn data(value: &Value) -> String {
    format!("data: {value}\n\n")
}

/// `event: <name>\ndata: <json>\n\n`
pub fn event(name: &str, value: &Value) -> String {
    format!("event: {name}\ndata: {value}\n\n")
}

/// `: <text>\n\n` (ignored by SSE clients; keeps the connection alive).
pub fn comment(text: &str) -> String {
    format!(": {text}\n\n")
}

/// A process-unique id with an API-style prefix (`chatcmpl-`, `call_`, `msg_`, `toolu_`,
/// `resp_`, `fc_`, `rs_`).
pub fn new_id(prefix: &str) -> String {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let start = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos() as u64);
    let n = NEXT.fetch_add(1, Ordering::Relaxed);
    format!(
        "{prefix}{:012x}{:06x}",
        start & 0xffff_ffff_ffff,
        n & 0xff_ffff
    )
}

/// Seconds since the Unix epoch.
pub fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ids_are_unique_and_prefixed() {
        let a = new_id("call_");
        let b = new_id("call_");
        assert!(a.starts_with("call_") && a != b);
        assert_eq!(
            event("ping", &serde_json::json!({"type": "ping"})),
            "event: ping\ndata: {\"type\":\"ping\"}\n\n"
        );
    }
}
