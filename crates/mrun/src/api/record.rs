//! `mrun serve --record <dir>` (002 FR-021): one JSON line per request with the body, the
//! arrival time and allow-listed session headers, for `mrun bench` trace replay (005). Nothing
//! about the response is written. Files are created with mode 0600.

use crate::api::model::ApiKind;
use crate::chat::template::Date;
use axum::http::HeaderMap;
use serde_json::{Map, Value, json};
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

/// Session headers kept in records; values are cut at 256 bytes.
pub const SESSION_HEADERS: [&str; 4] = [
    "x-claude-code-session-id",
    "thread-id",
    "x-session-id",
    "x-claude-code-prev-tool-durations",
];
const MAX_HEADER_BYTES: usize = 256;

#[derive(Debug)]
pub struct Recorder {
    path: PathBuf,
    file: Mutex<File>,
}

impl Recorder {
    /// Creates `<dir>/mrun-<unix seconds>.jsonl`.
    pub fn create(dir: &Path) -> std::io::Result<Self> {
        std::fs::create_dir_all(dir)?;
        let secs = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| d.as_secs());
        let path = dir.join(format!("mrun-{secs}.jsonl"));
        let mut options = OpenOptions::new();
        options.create(true).append(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }
        let file = options.open(&path)?;
        Ok(Self {
            path,
            file: Mutex::new(file),
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn record(&self, api: ApiKind, headers: &HeaderMap, body: &[u8]) {
        let body: Value = serde_json::from_slice(body).unwrap_or(Value::Null);
        let mut session = Map::new();
        for name in SESSION_HEADERS {
            if let Some(value) = headers.get(name).and_then(|v| v.to_str().ok()) {
                let mut end = value.len().min(MAX_HEADER_BYTES);
                while !value.is_char_boundary(end) {
                    end -= 1;
                }
                session.insert(name.into(), value[..end].into());
            }
        }
        let line = json!({"api": api.as_str(), "arrived": rfc3339_now(),
            "session": session, "body": body});
        let mut file = self.file.lock().unwrap_or_else(|e| e.into_inner());
        if let Err(error) = writeln!(file, "{line}") {
            eprintln!("mrun: cannot write the record: {error}");
        }
    }
}

/// UTC time with milliseconds, e.g. `2026-10-15T08:30:00.123Z`.
pub fn rfc3339_now() -> String {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    rfc3339(now.as_secs(), now.subsec_millis())
}

fn rfc3339(secs: u64, millis: u32) -> String {
    let (y, m, d) = Date::from_unix_secs(secs).ymd();
    let t = secs % 86_400;
    format!(
        "{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}.{millis:03}Z",
        t / 3600,
        t / 60 % 60,
        t % 60
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn xapi_record_fields_and_session_headers() {
        let dir = tempfile::tempdir().unwrap();
        let recorder = Recorder::create(dir.path()).unwrap();
        let mut headers = HeaderMap::new();
        headers.insert("x-claude-code-session-id", "s1".parse().unwrap());
        headers.insert("x-claude-code-prev-tool-durations", "1200".parse().unwrap());
        headers.insert("authorization", "Bearer secret".parse().unwrap());
        recorder.record(ApiKind::Messages, &headers, br#"{"messages": []}"#);
        let text = std::fs::read_to_string(recorder.path()).unwrap();
        let line: Value = serde_json::from_str(text.trim()).unwrap();
        assert_eq!(line["api"], "messages");
        assert_eq!(line["session"]["x-claude-code-session-id"], "s1");
        assert_eq!(line["session"]["x-claude-code-prev-tool-durations"], "1200");
        assert!(line["session"].get("authorization").is_none());
        assert_eq!(line["body"], json!({"messages": []}));
        assert!(line["arrived"].as_str().unwrap().ends_with('Z'));
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = std::fs::metadata(recorder.path())
                .unwrap()
                .permissions()
                .mode();
            assert_eq!(mode & 0o777, 0o600);
        }
    }

    #[test]
    fn rfc3339_formats_utc_millis() {
        assert_eq!(rfc3339(1_760_517_000, 7), "2025-10-15T08:30:00.007Z");
    }
}
