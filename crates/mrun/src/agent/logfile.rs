//! The agent's full log in the pod (013 FR-015): every line vLLM prints on stdout and stderr and
//! the agent's own lines (prefixed `[agent]`), each with a UTC timestamp, in one file capped at
//! [`MAX_BYTES`] with one rotation to `<file>.1`. `GET /mrun/logs` reads it by offset.
//!
//! Offsets are logical: bytes written since the agent started, across rotations, so a client
//! that follows the log keeps its place when the file rotates. Bytes older than the rotated file
//! are gone; a read before them starts at the oldest byte kept.
//!
//! vLLM does not log prompts or outputs unless started with `--enable-log-requests` /
//! `--enable-log-outputs`, which the agent never passes; the file holds startup, load and
//! error output, never request contents.

use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::{Mutex, PoisonError};

/// Where the agent writes its log in the pod (`/tmp/mrun-agent` is the agent binary itself).
pub const DEFAULT_PATH: &str = "/tmp/mrun/vllm.log";
/// Size at which the file rotates to `<file>.1` (one rotated file is kept).
pub const MAX_BYTES: u64 = 50 * 1024 * 1024;
/// Most bytes one read returns.
pub const READ_LIMIT: usize = 4 * 1024 * 1024;

#[derive(Debug)]
struct Inner {
    file: Option<File>,
    /// Logical offset of the current file's first byte.
    base: u64,
    /// Bytes in the current file.
    len: u64,
    /// Logical offset of the rotated file's first byte, if there is one.
    rotated: Option<u64>,
}

/// The log file and its rotation state.
#[derive(Debug)]
pub struct LogFile {
    path: PathBuf,
    max_bytes: u64,
    inner: Mutex<Inner>,
}

/// One read: text from `start` up to `next`, and the log's current end and oldest kept byte.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Chunk {
    pub bytes: Vec<u8>,
    /// Where the bytes start (the requested offset, or the oldest byte kept if that is later).
    pub start: u64,
    /// The offset after the bytes: where the next read continues.
    pub next: u64,
    /// The log's end when read.
    pub end: u64,
    /// The oldest byte still kept.
    pub first: u64,
}

impl LogFile {
    /// Creates (truncates) the log at `path`; its directory is created. On error the log is
    /// disabled (lines are dropped, reads are empty) and the error is returned with it.
    pub fn create(path: &Path, max_bytes: u64) -> (Self, Option<std::io::Error>) {
        let opened = (|| {
            if let Some(dir) = path.parent() {
                std::fs::create_dir_all(dir)?;
            }
            let _ = std::fs::remove_file(rotated_path(path));
            OpenOptions::new()
                .create(true)
                .write(true)
                .truncate(true)
                .open(path)
        })();
        let (file, error) = match opened {
            Ok(file) => (Some(file), None),
            Err(error) => (None, Some(error)),
        };
        let log = Self {
            path: path.to_path_buf(),
            max_bytes: max_bytes.max(1024),
            inner: Mutex::new(Inner {
                file,
                base: 0,
                len: 0,
                rotated: None,
            }),
        };
        (log, error)
    }

    /// A log that keeps nothing (tests that do not read it).
    pub fn disabled() -> Self {
        Self {
            path: PathBuf::new(),
            max_bytes: MAX_BYTES,
            inner: Mutex::new(Inner {
                file: None,
                base: 0,
                len: 0,
                rotated: None,
            }),
        }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Appends `<timestamp> <line>\n` (whole lines under the lock, so readers never see half a
    /// line), rotating first when the line would pass the cap.
    pub fn append(&self, line: &str) {
        let text = format!(
            "{} {}\n",
            crate::api::record::rfc3339_now(),
            line.trim_end_matches(['\r', '\n'])
        );
        let mut inner = self.inner.lock().unwrap_or_else(PoisonError::into_inner);
        if inner.file.is_none() {
            return;
        }
        if inner.len > 0 && inner.len + text.len() as u64 > self.max_bytes {
            self.rotate(&mut inner);
        }
        if let Some(file) = inner.file.as_mut()
            && file.write_all(text.as_bytes()).is_ok()
        {
            inner.len += text.len() as u64;
        }
    }

    fn rotate(&self, inner: &mut Inner) {
        inner.file = None;
        let _ = std::fs::rename(&self.path, rotated_path(&self.path));
        inner.rotated = Some(inner.base);
        inner.base += inner.len;
        inner.len = 0;
        inner.file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&self.path)
            .ok();
    }

    /// The log's end offset.
    pub fn end(&self) -> u64 {
        let inner = self.inner.lock().unwrap_or_else(PoisonError::into_inner);
        inner.base + inner.len
    }

    /// Up to `limit` bytes from `offset` (clamped to the bytes kept). Blocking file I/O.
    pub fn read(&self, offset: u64, limit: usize) -> Chunk {
        let inner = self.inner.lock().unwrap_or_else(PoisonError::into_inner);
        let end = inner.base + inner.len;
        let first = inner.rotated.unwrap_or(inner.base);
        let start = offset.clamp(first, end);
        let mut bytes = Vec::new();
        let mut at = start;
        // The rotated part, then the current file.
        if at < inner.base
            && let Some(rotated) = inner.rotated
        {
            let want = ((inner.base - at) as usize).min(limit);
            read_at(&rotated_path(&self.path), at - rotated, want, &mut bytes);
            at += bytes.len() as u64;
        }
        if at >= inner.base && bytes.len() < limit && at < end {
            let want = ((end - at) as usize).min(limit - bytes.len());
            let before = bytes.len();
            read_at(&self.path, at - inner.base, want, &mut bytes);
            at += (bytes.len() - before) as u64;
        }
        Chunk {
            bytes,
            start,
            next: at,
            end,
            first,
        }
    }
}

fn rotated_path(path: &Path) -> PathBuf {
    let mut name = path.as_os_str().to_os_string();
    name.push(".1");
    PathBuf::from(name)
}

fn read_at(path: &Path, offset: u64, want: usize, out: &mut Vec<u8>) {
    let Ok(mut file) = File::open(path) else {
        return;
    };
    if file.seek(SeekFrom::Start(offset)).is_err() {
        return;
    }
    let _ = file.take(want as u64).read_to_end(out);
}

/// `bytes` up to and including its last newline (whole lines), unless it has none.
pub fn whole_lines(bytes: &[u8]) -> &[u8] {
    match bytes.iter().rposition(|&b| b == b'\n') {
        Some(i) => &bytes[..=i],
        None => bytes,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lines_are_timestamped_and_read_by_offset() {
        let dir = tempfile::tempdir().unwrap();
        let (log, error) = LogFile::create(&dir.path().join("sub/vllm.log"), MAX_BYTES);
        assert!(error.is_none());
        log.append("INFO hello");
        log.append("[agent] ready\n");
        let all = log.read(0, READ_LIMIT);
        let text = String::from_utf8(all.bytes.clone()).unwrap();
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines.len(), 2);
        assert!(lines[0].ends_with("Z INFO hello"), "{text}");
        assert!(lines[0].starts_with("20") && lines[0].as_bytes()[10] == b'T');
        assert!(lines[1].ends_with(" [agent] ready"));
        assert_eq!(
            (all.start, all.next, all.end, all.first),
            (0, all.end, log.end(), 0)
        );
        // From an offset: the rest; past the end: nothing, at the end.
        let first_len = lines[0].len() as u64 + 1;
        let rest = log.read(first_len, READ_LIMIT);
        assert_eq!(rest.bytes, format!("{}\n", lines[1]).into_bytes());
        let past = log.read(10_000, READ_LIMIT);
        assert!(past.bytes.is_empty());
        assert_eq!((past.start, past.next), (log.end(), log.end()));
        // A limit splits the read; `next` continues it.
        let head = log.read(0, 5);
        assert_eq!((head.bytes.len(), head.next), (5, 5));
        assert_eq!(log.read(head.next, READ_LIMIT).bytes, all.bytes[5..]);
    }

    #[test]
    fn rotation_keeps_one_file_and_logical_offsets() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("vllm.log");
        let (log, _) = LogFile::create(&path, 1024);
        let mut seen = 0;
        for i in 0..100 {
            log.append(&format!("line {i:03} {}", "x".repeat(40)));
            seen += 1;
        }
        assert_eq!(seen, 100);
        assert!(path.exists() && rotated_path(&path).exists());
        assert!(std::fs::metadata(&path).unwrap().len() <= 1024);
        let end = log.end();
        // Every line is 75 bytes (24 timestamp, space, 49, newline): 7500 bytes written.
        assert_eq!(end, 7500);
        let chunk = log.read(0, READ_LIMIT);
        assert!(chunk.first > 0, "the oldest lines are gone");
        assert_eq!(chunk.start, chunk.first);
        assert_eq!(chunk.next, end);
        let text = String::from_utf8(chunk.bytes).unwrap();
        assert!(text.ends_with(&format!("line 099 {}\n", "x".repeat(40))));
        assert!(
            text.lines().all(|l| l.contains(" line ")),
            "whole lines across both files"
        );
        // A read inside the rotated file continues into the current one.
        let mid = log.read(chunk.first + 75, READ_LIMIT);
        assert_eq!(mid.next, end);
        assert_eq!(mid.bytes.len() as u64, end - chunk.first - 75);
        assert_eq!(whole_lines(b"a\nb\nc"), b"a\nb\n");
        assert_eq!(whole_lines(b"abc"), b"abc");
    }

    #[test]
    fn an_unwritable_path_disables_the_log() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("file");
        std::fs::write(&file, b"x").unwrap();
        let (log, error) = LogFile::create(&file.join("vllm.log"), MAX_BYTES);
        assert!(error.is_some());
        log.append("dropped");
        assert_eq!(log.end(), 0);
        assert!(log.read(0, READ_LIMIT).bytes.is_empty());
    }
}
