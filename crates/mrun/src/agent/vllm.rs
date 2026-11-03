//! Supervising vLLM inside the pod (013 FR-007): its command line (loopback only, a random
//! `--api-key`, prefix caching, the model's tool and reasoning parsers), its output (echoed to
//! the agent's log and written in full to the log file, the last lines kept for `failed`), and
//! its exit. vLLM's request logging (`--enable-log-requests`, `--enable-log-outputs`) stays off,
//! so prompts and outputs never reach a log.

use super::logfile::LogFile;
use std::collections::VecDeque;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitStatus, Stdio};
use std::sync::{Arc, Mutex, PoisonError};

/// Lines of vLLM output kept for a failure report.
pub const LOG_LINES: usize = 60;
/// Exception lines kept apart from the tail (the last ones): a crash's root cause is printed inside
/// the engine's traceback and scrolls out of the tail under the server's traceback that follows.
pub const ERROR_LINES: usize = 8;

/// Variables vLLM never sees: the pod-scoped RunPod key and mrun's own secrets.
pub const HIDDEN_ENV: [&str; 4] = [
    "RUNPOD_API_KEY",
    "MRUN_POD_TOKEN",
    "MRUN_BOOTSTRAP",
    "MRUN_AGENT_URL",
];

/// How vLLM is started.
#[derive(Clone, Debug, PartialEq)]
pub struct VllmSpec {
    /// The executable (`vllm`).
    pub program: String,
    pub model: String,
    pub port: u16,
    pub api_key: String,
    pub max_model_len: u32,
    /// `--max-num-seqs` (hybrid models need it at most their Mamba cache blocks).
    pub max_num_seqs: u32,
    pub tool_parser: String,
    pub reasoning_parser: Option<String>,
    /// The model's own flags, after mrun's (a repeated flag overrides mrun's).
    pub extra_args: Vec<String>,
}

impl VllmSpec {
    /// `serve <model> --host 127.0.0.1 ...`.
    pub fn args(&self) -> Vec<String> {
        let mut args: Vec<String> = [
            "serve",
            &self.model,
            "--host",
            "127.0.0.1",
            "--port",
            &self.port.to_string(),
            "--api-key",
            &self.api_key,
            "--served-model-name",
            &self.model,
            "--enable-prefix-caching",
            "--enable-auto-tool-choice",
            "--tool-call-parser",
            &self.tool_parser,
            "--max-model-len",
            &self.max_model_len.to_string(),
            "--max-num-seqs",
            &self.max_num_seqs.to_string(),
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        // The model's own value replaces mrun's (vLLM warns on a repeated flag).
        if !self
            .extra_args
            .iter()
            .any(|a| a == "--gpu-memory-utilization")
        {
            args.extend(["--gpu-memory-utilization".to_string(), "0.90".to_string()]);
        }
        if let Some(parser) = &self.reasoning_parser {
            args.extend(["--reasoning-parser".to_string(), parser.clone()]);
        }
        args.extend(self.extra_args.iter().cloned());
        args
    }

    /// The command with its arguments and its environment (secrets removed).
    pub fn command(&self) -> Command {
        let mut command = Command::new(&self.program);
        command.args(self.args());
        for name in HIDDEN_ENV {
            command.env_remove(name);
        }
        command
    }
}

/// The last lines of a child's output, plus the last exception lines seen (the root cause of a
/// failed start usually comes first and is followed by a long traceback).
#[derive(Clone, Debug, Default)]
pub struct LogRing(Arc<Mutex<(VecDeque<String>, Vec<String>)>>);

/// A line's content without vLLM's log prefix (`(EngineCore pid=N) ERROR 10-19 09:02:22
/// [core.py:1510] `), so traceback frames inside log records look like plain traceback frames.
fn content(line: &str) -> &str {
    let mut l = line.trim_start();
    // The log file's timestamp (`2026-10-19T09:02:22.123Z `).
    if let Some((stamp, rest)) = l.split_once(' ')
        && stamp.len() == 24
        && stamp.ends_with('Z')
        && stamp.as_bytes().get(10) == Some(&b'T')
    {
        l = rest.trim_start();
    }
    l = l.strip_prefix("[agent] ").unwrap_or(l);
    if l.starts_with('(')
        && let Some(i) = l.find(") ")
    {
        l = &l[i + 2..];
    }
    for level in ["ERROR ", "WARNING ", "INFO ", "DEBUG ", "CRITICAL "] {
        if let Some(rest) = l.strip_prefix(level) {
            l = rest.find("] ").map_or(rest, |i| &rest[i + 2..]);
            break;
        }
    }
    l.trim_start()
}

/// Whether a line is an exception line (`ValueError: …`, `torch.OutOfMemoryError: CUDA out of
/// memory`), the line that names a failure's root cause; not traceback frames, carets or the
/// generic "Engine core initialization failed" summary.
pub fn is_error_line(line: &str) -> bool {
    let l = content(line);
    if l.contains("Engine core initialization failed") || l.contains("EngineCore failed to start") {
        return false;
    }
    let name = l.split([':', ' ']).next().unwrap_or("");
    (name.ends_with("Error") || name.ends_with("Exception") || name.ends_with("Interrupt"))
        && name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '.' || c == '_')
        || l.contains("CUDA out of memory")
}

impl LogRing {
    pub fn push(&self, line: String) {
        let mut guard = self.0.lock().unwrap_or_else(PoisonError::into_inner);
        let (lines, errors) = &mut *guard;
        if is_error_line(&line) {
            if errors.len() == ERROR_LINES {
                errors.remove(0);
            }
            errors.push(line.clone());
        }
        if lines.len() == LOG_LINES {
            lines.pop_front();
        }
        lines.push_back(line);
    }

    /// The last exception lines that have scrolled out of the tail, then the tail.
    pub fn tail(&self) -> Vec<String> {
        let guard = self.0.lock().unwrap_or_else(PoisonError::into_inner);
        let (lines, errors) = &*guard;
        let mut out: Vec<String> = errors
            .iter()
            .filter(|e| !lines.contains(e))
            .cloned()
            .collect();
        if !out.is_empty() {
            out.push("…".into());
        }
        out.extend(lines.iter().cloned());
        out
    }
}

/// A started child: its log and, once it exits, its status.
#[derive(Debug)]
pub struct Supervised {
    pub log: LogRing,
    pub exited: Arc<Mutex<Option<ExitStatus>>>,
    pid: u32,
}

impl Supervised {
    pub fn exit_status(&self) -> Option<ExitStatus> {
        *self.exited.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Kills the child (SIGKILL through `kill`; best effort).
    pub fn kill(&self) {
        let _ = Command::new("kill")
            .args(["-9", &self.pid.to_string()])
            .status();
    }
}

/// Spawns `command` with piped output; threads echo each line to stderr (prefixed `vllm: `),
/// append it to `file` and keep the last [`LOG_LINES`].
pub fn spawn(mut command: Command, file: Option<Arc<LogFile>>) -> std::io::Result<Supervised> {
    command
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let mut child: Child = command.spawn()?;
    let log = LogRing::default();
    let pid = child.id();
    let pipes: Vec<Box<dyn std::io::Read + Send>> = [
        child
            .stdout
            .take()
            .map(|p| Box::new(p) as Box<dyn std::io::Read + Send>),
        child
            .stderr
            .take()
            .map(|p| Box::new(p) as Box<dyn std::io::Read + Send>),
    ]
    .into_iter()
    .flatten()
    .collect();
    for pipe in pipes {
        let log = log.clone();
        let file = file.clone();
        std::thread::spawn(move || {
            for line in BufReader::new(pipe).lines().map_while(Result::ok) {
                eprintln!("vllm: {line}");
                if let Some(file) = &file {
                    file.append(&line);
                }
                log.push(line);
            }
        });
    }
    let exited = Arc::new(Mutex::new(None));
    let slot = exited.clone();
    std::thread::spawn(move || {
        let status = child.wait().ok();
        *slot.lock().unwrap_or_else(PoisonError::into_inner) = status;
    });
    Ok(Supervised { log, exited, pid })
}

/// Bytes of regular files under `dir` (symlinks not followed: the Hugging Face cache links
/// snapshots to blobs, so each blob counts once).
pub fn dir_size(dir: &Path) -> u64 {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return 0;
    };
    entries
        .flatten()
        .map(|entry| match entry.file_type() {
            Ok(t) if t.is_dir() => dir_size(&entry.path()),
            Ok(t) if t.is_file() => entry.metadata().map_or(0, |m| m.len()),
            _ => 0,
        })
        .sum()
}

/// Where Hugging Face downloads go: `HF_HOME`, else `~/.cache/huggingface`.
pub fn hf_home() -> PathBuf {
    std::env::var_os("HF_HOME").map_or_else(
        || {
            std::env::var_os("HOME")
                .map_or_else(|| PathBuf::from("/root"), PathBuf::from)
                .join(".cache/huggingface")
        },
        PathBuf::from,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec() -> VllmSpec {
        VllmSpec {
            program: "vllm".into(),
            model: "Qwen/Qwen3-4B-Instruct-2507".into(),
            port: 8001,
            api_key: "k".repeat(64),
            max_model_len: 65_536,
            max_num_seqs: 16,
            tool_parser: "hermes".into(),
            reasoning_parser: None,
            extra_args: vec![],
        }
    }

    #[test]
    fn vllm_binds_loopback_with_its_own_key_and_parsers() {
        let args = spec().args();
        let pair = |flag: &str| {
            args.iter()
                .position(|a| a == flag)
                .map(|i| args[i + 1].clone())
        };
        assert_eq!(args[..2], ["serve", "Qwen/Qwen3-4B-Instruct-2507"]);
        assert_eq!(pair("--host").as_deref(), Some("127.0.0.1"));
        assert_eq!(pair("--api-key").map(|k| k.len()), Some(64));
        assert_eq!(pair("--tool-call-parser").as_deref(), Some("hermes"));
        assert_eq!(pair("--max-model-len").as_deref(), Some("65536"));
        assert_eq!(pair("--max-num-seqs").as_deref(), Some("16"));
        assert!(args.contains(&"--enable-prefix-caching".to_string()));
        assert!(!args.contains(&"--reasoning-parser".to_string()));
        // Request logging stays off: prompts and outputs never reach a log.
        assert!(
            !args
                .iter()
                .any(|a| a.contains("log-requests") || a.contains("log-outputs"))
        );
        let with = VllmSpec {
            reasoning_parser: Some("qwen3".into()),
            ..spec()
        };
        assert!(
            with.args()
                .ends_with(&["--reasoning-parser".into(), "qwen3".into()])
        );
        let extra = VllmSpec {
            extra_args: vec!["--gpu-memory-utilization".into(), "0.92".into()],
            ..spec()
        };
        assert!(
            extra
                .args()
                .ends_with(&["--gpu-memory-utilization".into(), "0.92".into()]),
            "the model's flags come last"
        );
        assert_eq!(
            extra
                .args()
                .iter()
                .filter(|a| *a == "--gpu-memory-utilization")
                .count(),
            1,
            "mrun's 0.90 is left out when the model sets its own"
        );
        assert!(spec().args().contains(&"0.90".to_string()));
        let command = spec().command();
        let removed: Vec<_> = command
            .get_envs()
            .filter(|(_, v)| v.is_none())
            .map(|(k, _)| k.to_string_lossy().into_owned())
            .collect();
        for name in HIDDEN_ENV {
            assert!(removed.contains(&name.to_string()), "{name}");
        }
    }

    #[test]
    fn a_childs_output_tail_and_exit_are_kept() {
        let mut command = Command::new("sh");
        command.args([
            "-c",
            "i=0; while [ $i -lt 30 ]; do echo line$i; i=$((i+1)); done; echo oops >&2; exit 3",
        ]);
        let dir = tempfile::tempdir().unwrap();
        let (file, _) = LogFile::create(
            &dir.path().join("vllm.log"),
            super::super::logfile::MAX_BYTES,
        );
        let file = Arc::new(file);
        let child = spawn(command, Some(file.clone())).unwrap();
        let start = std::time::Instant::now();
        while child.exit_status().is_none() && start.elapsed().as_secs() < 30 {
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
        // The reader threads may still be draining the pipes when the exit is seen; a loaded test
        // machine (the full parallel suite) can take seconds, so the waits allow 30 s.
        while !(child.log.tail().contains(&"oops".to_string())
            && child.log.tail().contains(&"line29".to_string()))
            && start.elapsed().as_secs() < 30
        {
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
        assert_eq!(child.exit_status().and_then(|s| s.code()), Some(3));
        let tail = child.log.tail();
        assert_eq!(
            tail.len(),
            31,
            "30 stdout lines and the stderr line fit the tail"
        );
        assert!(tail.contains(&"oops".to_string()));
        assert!(tail.contains(&"line0".to_string()));
        // The file has every line, stdout and stderr, timestamped.
        let text = String::from_utf8(file.read(0, usize::MAX).bytes).unwrap();
        assert_eq!(text.lines().count(), 31, "{text}");
        assert!(text.lines().any(|l| l.ends_with("Z oops")));
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a"), [0u8; 100]).unwrap();
        std::fs::create_dir(dir.path().join("sub")).unwrap();
        std::fs::write(dir.path().join("sub/b"), [0u8; 50]).unwrap();
        assert_eq!(dir_size(dir.path()), 150);
    }

    #[test]
    fn the_root_cause_survives_a_long_traceback() {
        let log = LogRing::default();
        log.push("INFO loading".into());
        log.push("ValueError: To serve at least one request with max_model_len 65536, 9.4 GiB KV cache is needed".into());
        for i in 0..200 {
            log.push(format!("  File \"/x.py\", line {i}, in f"));
        }
        log.push("RuntimeError: Engine core initialization failed. See root cause above.".into());
        let tail = log.tail();
        assert!(tail[0].starts_with("ValueError: To serve"), "{tail:?}");
        assert_eq!(tail[1], "…");
        assert_eq!(tail.len(), 2 + LOG_LINES);
        assert!(!is_error_line("  File \"/a.py\", line 3"));
        assert!(!is_error_line(
            "RuntimeError: Engine core initialization failed."
        ));
        assert!(is_error_line("torch.OutOfMemoryError: CUDA out of memory."));
        // vLLM wraps engine tracebacks in log records; the exception line is the root cause.
        let vllm = "(EngineCore pid=1567) ERROR 10-19 09:02:22 [core.py:1510] ";
        assert!(!is_error_line(&format!(
            "{vllm}EngineCore failed to start."
        )));
        assert!(!is_error_line(&format!(
            "{vllm}Traceback (most recent call last):"
        )));
        assert!(!is_error_line(&format!(
            "{vllm}  File \"/x/core.py\", line 1482, in run"
        )));
        assert!(is_error_line(&format!(
            "{vllm}NotImplementedError: No compiled kernel for sm120"
        )));
        // Lines of the log file carry a timestamp first.
        assert!(is_error_line(&format!(
            "2026-10-19T09:02:22.123Z {vllm}ValueError: no KV cache"
        )));
        assert!(!is_error_line("2026-10-19T09:02:22.123Z INFO loading"));
    }
}
