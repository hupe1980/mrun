//! Per-file progress for `pull` and `verify` (FR-001, FR-006): on a terminal a bar with bytes,
//! rate and ETA redrawn every 200 ms; otherwise one line per file. `pull --json` reports through
//! [`Report::Json`] instead: one JSON object per line on stdout (018 data model), no bars.

use super::human_bytes;
use indicatif::{ProgressBar, ProgressDrawTarget, ProgressStyle};
use serde_json::{Value, json};
use std::io::{self, IsTerminal, Write};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Redraws per second on a terminal (200 ms).
const REDRAW_HZ: u8 = 5;
/// Shortest time between two `progress` lines of one file (at most 5 per second).
const JSON_INTERVAL: Duration = Duration::from_millis(200);

/// Where `pull` reports: human lines on stderr with bars, or JSON lines.
#[derive(Clone, Debug, Default)]
pub enum Report {
    #[default]
    Human,
    Json(JsonLines),
}

impl Report {
    /// A human line on stderr; JSON mode leaves it to the events.
    pub fn note(&self, line: &str) {
        if let Self::Human = self {
            eprintln!("{line}");
        }
    }

    /// An event line; human mode has none.
    pub fn emit(&self, event: Value) {
        if let Self::Json(out) = self {
            out.emit(&event);
        }
    }

    /// Starts the progress of one file, as [`Progress::start`] or as `progress` events.
    pub fn start(&self, verb: &str, path: &str, size: u64, offset: u64) -> Progress {
        match self {
            Self::Human => Progress::start(verb, path, size, offset),
            Self::Json(out) => Progress::json(out.clone(), path, size, offset),
        }
    }
}

/// A shared line writer: stdout, or a buffer in tests.
#[derive(Clone)]
pub struct JsonLines(Arc<Mutex<Box<dyn Write + Send>>>);

impl std::fmt::Debug for JsonLines {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("JsonLines")
    }
}

impl JsonLines {
    pub fn stdout() -> Self {
        Self::to(io::stdout())
    }

    pub fn to(writer: impl Write + Send + 'static) -> Self {
        Self(Arc::new(Mutex::new(Box::new(writer))))
    }

    /// Writes `event` as one line and flushes, so a reader sees it at once. Write errors (a
    /// closed pipe) are ignored: the pull itself goes on.
    pub fn emit(&self, event: &Value) {
        let mut out = self.0.lock().unwrap_or_else(|poison| poison.into_inner());
        let _ = writeln!(out, "{event}");
        let _ = out.flush();
    }
}

/// `progress` events of one file, throttled to [`JSON_INTERVAL`].
#[derive(Debug)]
struct JsonProgress {
    out: JsonLines,
    file: String,
    total: u64,
    state: Mutex<Throttle>,
}

#[derive(Debug)]
struct Throttle {
    bytes: u64,
    /// Time and bytes of the last line, for the rate.
    sent_at: Instant,
    sent_bytes: u64,
    rate: u64,
}

/// When a [`JsonProgress`] line is printed.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Line {
    /// Always (the file starts).
    Start,
    /// When [`JSON_INTERVAL`] passed since the last line.
    Tick,
    /// When bytes arrived since the last line (the file ends).
    Last,
}

impl JsonProgress {
    /// Records the bytes done (`bytes` maps the old count to the new) and prints a line when
    /// `when` says so.
    fn update(&self, bytes: impl FnOnce(u64) -> u64, when: Line) {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        state.bytes = bytes(state.bytes);
        let elapsed = state.sent_at.elapsed();
        let print = match when {
            Line::Start => true,
            Line::Tick => elapsed >= JSON_INTERVAL,
            Line::Last => state.bytes != state.sent_bytes,
        };
        if !print {
            return;
        }
        let secs = elapsed.as_secs_f64();
        if when != Line::Start && secs > 0.0 {
            state.rate = (state.bytes.saturating_sub(state.sent_bytes) as f64 / secs) as u64;
        }
        state.sent_at = Instant::now();
        state.sent_bytes = state.bytes;
        self.out.emit(&json!({
            "event": "progress",
            "file": self.file,
            "bytes": state.bytes,
            "total": self.total,
            "rate": state.rate,
        }));
    }
}

/// Progress of one file; a no-op bar when stderr is not a terminal.
#[derive(Debug)]
pub struct Progress(Option<ProgressBar>, Option<JsonProgress>);

impl Progress {
    /// Starts the progress of `path` (`size` bytes, `offset` already done). `verb` names the
    /// work: `downloading`, `resuming`, `hashing`.
    pub fn start(verb: &str, path: &str, size: u64, offset: u64) -> Self {
        Self::start_on(io::stderr().is_terminal(), verb, path, size, offset)
    }

    fn start_on(terminal: bool, verb: &str, path: &str, size: u64, offset: u64) -> Self {
        if !terminal {
            eprintln!("  {}", line(verb, path, size));
            return Self(None, None);
        }
        let style = ProgressStyle::with_template(
            "  {msg} {wide_bar} {bytes}/{total_bytes} {bytes_per_sec} ETA {eta}",
        )
        .expect("valid template");
        let bar = ProgressBar::with_draw_target(
            Some(size),
            ProgressDrawTarget::stderr_with_hz(REDRAW_HZ),
        )
        .with_style(style)
        .with_message(format!("{verb} {path}"))
        .with_position(offset);
        bar.reset_eta();
        Self(Some(bar), None)
    }

    /// `progress` lines on `out`: one at the start, at most one per [`JSON_INTERVAL`] while
    /// bytes arrive, and the last at [`Progress::finish`].
    fn json(out: JsonLines, path: &str, size: u64, offset: u64) -> Self {
        let progress = JsonProgress {
            out,
            file: path.to_string(),
            total: size,
            state: Mutex::new(Throttle {
                bytes: offset,
                sent_at: Instant::now(),
                sent_bytes: offset,
                rate: 0,
            }),
        };
        progress.update(|bytes| bytes, Line::Start);
        Self(None, Some(progress))
    }

    /// Sets the bytes done.
    pub fn set(&self, offset: u64) {
        if let Some(bar) = &self.0 {
            bar.set_position(offset);
        }
        if let Some(json) = &self.1 {
            json.update(|_| offset, Line::Tick);
        }
    }

    /// Adds `bytes` to the bytes done.
    pub fn add(&self, bytes: u64) {
        if let Some(bar) = &self.0 {
            bar.inc(bytes);
        }
        if let Some(json) = &self.1 {
            json.update(|done| done + bytes, Line::Tick);
        }
    }

    pub fn finish(&self) {
        if let Some(bar) = &self.0 {
            bar.finish();
        }
        if let Some(json) = &self.1 {
            json.update(|bytes| bytes, Line::Last);
        }
    }
}

/// The line printed per file without a terminal: `downloading model.safetensors (4.6 GB)`.
fn line(verb: &str, path: &str, size: u64) -> String {
    format!("{verb} {path} ({})", human_bytes(size))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn without_a_terminal_each_file_is_one_line() {
        assert_eq!(
            line("hashing", "model.safetensors", 4_600_000_000),
            "hashing model.safetensors (4.6 GB)"
        );
        let quiet = Progress::start_on(false, "hashing", "a", 10, 0);
        assert!(quiet.0.is_none());
        quiet.add(5);
        quiet.finish();
        let bar = Progress::start_on(true, "downloading", "a", 10, 4);
        bar.add(3);
        assert_eq!(bar.0.as_ref().unwrap().position(), 7);
        bar.set(10);
        bar.finish();
        assert!(bar.0.unwrap().is_finished());
    }

    /// A buffer the test reads back.
    #[derive(Clone, Default)]
    struct Buffer(Arc<Mutex<Vec<u8>>>);

    impl Write for Buffer {
        fn write(&mut self, data: &[u8]) -> io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(data);
            Ok(data.len())
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    impl Buffer {
        fn lines(&self) -> Vec<Value> {
            String::from_utf8(self.0.lock().unwrap().clone())
                .unwrap()
                .lines()
                .map(|line| serde_json::from_str(line).unwrap())
                .collect()
        }
    }

    #[test]
    fn json_progress_is_throttled_and_ends_at_the_total() {
        let buffer = Buffer::default();
        let report = Report::Json(JsonLines::to(buffer.clone()));
        report.note("not printed in JSON mode");
        let progress = report.start("downloading", "model.safetensors", 1000, 100);
        // Many updates inside one interval: no line until it passes.
        for done in (200..=900).step_by(100) {
            progress.set(done);
        }
        assert_eq!(buffer.lines().len(), 1, "only the start line");
        std::thread::sleep(JSON_INTERVAL);
        progress.add(50);
        progress.add(50);
        progress.finish();
        progress.finish();
        let lines = buffer.lines();
        assert_eq!(
            lines[0],
            json!({"event": "progress", "file": "model.safetensors", "bytes": 100, "total": 1000, "rate": 0})
        );
        assert_eq!(lines[1]["bytes"], 950);
        assert!(lines[1]["rate"].as_u64().unwrap() > 0, "{lines:?}");
        assert_eq!(lines[2]["bytes"], 1000);
        assert_eq!(lines.len(), 3, "a second finish prints nothing: {lines:?}");
        assert!(progress.0.is_none(), "no bar in JSON mode");
    }
}
