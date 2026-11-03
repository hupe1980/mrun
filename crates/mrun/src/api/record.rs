//! `mrun serve --record <dir>` (002 FR-021): one JSON line per request with the body, the
//! arrival time and allow-listed session headers, for `mrun bench` trace replay (005). Nothing
//! about the response is written. Files are created with mode 0600.
//!
//! The body is written as the client sent it (newlines, which valid JSON has only between
//! tokens, become spaces), from a dedicated writer thread, so recording neither parses the body
//! again nor blocks the async runtime on disk. The writer's queue is bounded: when the disk
//! falls that far behind, records are dropped (and logged) instead of growing memory.

use crate::api::model::ApiKind;
use crate::chat::template::Date;
use axum::http::HeaderMap;
use serde_json::{Map, json};
use std::fs::OpenOptions;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::mpsc::{SyncSender, TrySendError, sync_channel};
use std::thread::JoinHandle;
use std::time::{SystemTime, UNIX_EPOCH};

/// Session headers kept in records; values are cut at 256 bytes. `x-claude-code-request-class`
/// tells main-loop turns from side requests (002 FR-021, 005 FR-020).
pub const SESSION_HEADERS: [&str; 5] = [
    "x-claude-code-session-id",
    "thread-id",
    "x-session-id",
    "x-claude-code-prev-tool-durations",
    "x-claude-code-request-class",
];
const MAX_HEADER_BYTES: usize = 256;
/// Records queued for the writer before new ones are dropped.
const QUEUE_RECORDS: usize = 256;

#[derive(Debug)]
pub struct Recorder {
    path: PathBuf,
    lines: Mutex<Option<SyncSender<Vec<u8>>>>,
    writer: Mutex<Option<JoinHandle<()>>>,
}

impl Recorder {
    /// Creates `<dir>/mrun-<unix seconds>.jsonl` and its writer thread.
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
        let (lines, inbox) = sync_channel::<Vec<u8>>(QUEUE_RECORDS);
        let writer = std::thread::Builder::new()
            .name("mrun-record".into())
            .spawn(move || {
                let mut file = BufWriter::new(file);
                while let Ok(line) = inbox.recv() {
                    // Write what else is queued, then flush once.
                    let mut result = file.write_all(&line);
                    for line in inbox.try_iter() {
                        result = result.and_then(|()| file.write_all(&line));
                    }
                    if let Err(error) = result.and_then(|()| file.flush()) {
                        eprintln!("mrun: cannot write the record: {error}");
                    }
                }
            })?;
        Ok(Self {
            path,
            lines: Mutex::new(Some(lines)),
            writer: Mutex::new(Some(writer)),
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Queues one record; `json` says whether `body` is valid JSON (else `null` is recorded).
    pub fn record(&self, api: ApiKind, headers: &HeaderMap, body: &[u8], json: bool) {
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
        let head = json!({"api": api.as_str(), "arrived": rfc3339_now(), "session": session});
        let head = head.to_string();
        let mut line = Vec::with_capacity(head.len() + body.len() + 10);
        line.extend_from_slice(&head.as_bytes()[..head.len() - 1]);
        line.extend_from_slice(b",\"body\":");
        if json {
            line.extend(
                body.iter()
                    .map(|&b| if b == b'\n' || b == b'\r' { b' ' } else { b }),
            );
        } else {
            line.extend_from_slice(b"null");
        }
        line.extend_from_slice(b"}\n");
        if let Some(lines) = &*self.lines.lock().unwrap_or_else(|e| e.into_inner())
            && let Err(TrySendError::Full(_)) = lines.try_send(line)
        {
            eprintln!("mrun: record-dropped (the writer is behind)");
        }
    }

    /// Writes every queued record and stops the writer.
    pub fn close(&self) {
        self.lines.lock().unwrap_or_else(|e| e.into_inner()).take();
        if let Some(writer) = self.writer.lock().unwrap_or_else(|e| e.into_inner()).take() {
            let _ = writer.join();
        }
    }
}

impl Drop for Recorder {
    fn drop(&mut self) {
        self.close();
    }
}

/// UTC time with milliseconds, e.g. `2026-10-15T08:30:00.123Z`.
pub fn rfc3339_now() -> String {
    rfc3339_of(SystemTime::now())
}

pub fn rfc3339_of(time: SystemTime) -> String {
    let at = time.duration_since(UNIX_EPOCH).unwrap_or_default();
    rfc3339(at.as_secs(), at.subsec_millis())
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
    use serde_json::Value;

    #[test]
    fn xapi_record_fields_and_session_headers() {
        let dir = tempfile::tempdir().unwrap();
        let recorder = Recorder::create(dir.path()).unwrap();
        let mut headers = HeaderMap::new();
        headers.insert("x-claude-code-session-id", "s1".parse().unwrap());
        headers.insert("x-claude-code-prev-tool-durations", "1200".parse().unwrap());
        headers.insert("x-claude-code-request-class", "main".parse().unwrap());
        headers.insert("authorization", "Bearer secret".parse().unwrap());
        recorder.record(
            ApiKind::Messages,
            &headers,
            b"{\"messages\": [],\n  \"model\": \"m\"}",
            true,
        );
        recorder.record(ApiKind::Chat, &headers, b"{not json", false);
        recorder.close();
        let text = std::fs::read_to_string(recorder.path()).unwrap();
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines.len(), 2, "{text}");
        let line: Value = serde_json::from_str(lines[0]).unwrap();
        assert!(
            lines[0].contains("{\"messages\": [],   \"model\": \"m\"}"),
            "raw bytes"
        );
        let second: Value = serde_json::from_str(lines[1]).unwrap();
        assert_eq!(second["body"], Value::Null);
        assert_eq!(line["api"], "messages");
        assert_eq!(line["session"]["x-claude-code-session-id"], "s1");
        assert_eq!(line["session"]["x-claude-code-prev-tool-durations"], "1200");
        assert_eq!(line["session"]["x-claude-code-request-class"], "main");
        assert!(line["session"].get("authorization").is_none());
        assert_eq!(line["body"], json!({"messages": [], "model": "m"}));
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
