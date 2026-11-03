//! `mrun logs [-f] [-n N]`: the log of servers started with `mrun serve --detach`
//! (`MRUN_HOME/server.log`, rotated to `server.log.1` at 10 MB when a server starts).

use crate::errors::{CliError, Code};
use std::fs;
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::time::Duration;

/// Size at which a starting server rotates the log.
const ROTATE_AT: u64 = 10 * 1000 * 1000;

#[derive(Debug, clap::Args)]
pub struct LogsArgs {
    /// Keep printing new lines as the server writes them.
    #[arg(short, long)]
    pub follow: bool,
    /// Number of lines to print from the end.
    #[arg(short = 'n', long, default_value_t = 50)]
    pub lines: usize,
}

/// Where a detached server writes its log.
pub fn log_file() -> PathBuf {
    crate::cli::mrun_home().join("server.log")
}

/// Opens the log for a starting server, moving a log of 10 MB or more to `server.log.1` first
/// (one old log is kept).
pub fn open_for_server() -> io::Result<fs::File> {
    let path = log_file();
    if let Some(dir) = path.parent() {
        fs::create_dir_all(dir)?;
    }
    rotate(&path, ROTATE_AT)?;
    fs::OpenOptions::new().create(true).append(true).open(path)
}

fn rotate(path: &Path, limit: u64) -> io::Result<()> {
    if fs::metadata(path).is_ok_and(|meta| meta.len() >= limit) {
        fs::rename(path, path.with_extension("log.1"))?;
    }
    Ok(())
}

/// The last `n` lines of the log (empty when there is none).
pub fn tail(n: usize) -> String {
    let text = fs::read_to_string(log_file()).unwrap_or_default();
    last_lines(&text, n).to_string()
}

/// The last `n` lines the log gained after byte `start` (the log's length when a server was
/// started), so a failed start does not repeat an earlier run's error (008 local release dry run).
pub fn tail_since(start: u64, n: usize) -> String {
    let bytes = fs::read(log_file()).unwrap_or_default();
    since(&bytes, start, n)
}

fn since(bytes: &[u8], start: u64, n: usize) -> String {
    let start = usize::try_from(start)
        .unwrap_or(usize::MAX)
        .min(bytes.len());
    let text = String::from_utf8_lossy(&bytes[start..]);
    last_lines(&text, n).to_string()
}

fn last_lines(text: &str, n: usize) -> &str {
    let trimmed = text.trim_end_matches('\n');
    let start = trimmed
        .rmatch_indices('\n')
        .nth(n.saturating_sub(1))
        .map_or(0, |(at, _)| at + 1);
    if n == 0 { "" } else { &trimmed[start..] }
}

pub fn run(args: LogsArgs) -> u8 {
    crate::cli::exit(logs(&args))
}

fn logs(args: &LogsArgs) -> Result<(), CliError> {
    let path = log_file();
    let mut file = fs::File::open(&path).map_err(|_| {
        CliError::new(
            Code::ServerUnreachable,
            format!("no server log at {}", path.display()),
        )
        .hint("servers started with 'mrun serve --detach' (or by 'mrun launch') write it")
    })?;
    let mut text = String::new();
    file.read_to_string(&mut text)
        .map_err(|error| CliError::new(Code::Internal, error.to_string()))?;
    let tail = last_lines(&text, args.lines);
    if !tail.is_empty() {
        println!("{tail}");
    }
    if args.follow {
        follow(&path, text.len() as u64)
            .map_err(|error| CliError::new(Code::Internal, error.to_string()))?;
    }
    Ok(())
}

/// Prints what is appended to `path` from `offset` on; starts over when the log was rotated.
fn follow(path: &Path, mut offset: u64) -> io::Result<()> {
    let mut out = io::stdout();
    loop {
        std::thread::sleep(Duration::from_millis(500));
        let Ok(mut file) = fs::File::open(path) else {
            continue;
        };
        let len = file.metadata()?.len();
        if len < offset {
            offset = 0;
        }
        if len > offset {
            file.seek(SeekFrom::Start(offset))?;
            let mut chunk = Vec::new();
            file.take(len - offset).read_to_end(&mut chunk)?;
            out.write_all(&chunk)?;
            out.flush()?;
            offset = len;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn last_lines_counts_from_the_end() {
        assert_eq!(last_lines("a\nb\nc\n", 2), "b\nc");
        assert_eq!(last_lines("a\nb\nc", 5), "a\nb\nc");
        assert_eq!(last_lines("a\nb\n", 0), "");
        assert_eq!(last_lines("", 3), "");
    }

    #[test]
    fn a_tail_since_a_start_leaves_out_earlier_runs() {
        let log = b"mrun listening\nerror[E031]: model 'a' not found\nmrun listening\nloaded b\n";
        let start = u64::try_from(log.len() - "mrun listening\nloaded b\n".len()).unwrap();
        assert_eq!(since(log, start, 5), "mrun listening\nloaded b");
        assert_eq!(since(log, 0, 1), "loaded b");
        assert_eq!(
            since(log, 10_000, 5),
            "",
            "a start past the end shows nothing"
        );
    }

    #[test]
    fn large_logs_rotate_and_keep_one() {
        let dir = tempfile::tempdir().unwrap();
        let log = dir.path().join("server.log");
        fs::write(&log, "old\n").unwrap();
        rotate(&log, 100).unwrap();
        assert!(log.exists(), "small logs stay");
        rotate(&log, 4).unwrap();
        assert!(!log.exists());
        assert_eq!(
            fs::read_to_string(dir.path().join("server.log.1")).unwrap(),
            "old\n"
        );
    }
}
