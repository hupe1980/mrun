//! Command implementations shared by the binary and the integration tests.

pub mod cache;
pub mod clients;
pub mod connect;
pub mod doctor;
pub mod launch;
pub mod ls;
pub mod ps;
pub mod pull;
pub mod rm;
pub mod serve;
pub mod server;
pub mod show;
pub mod stop;
pub mod verify;
pub mod version;

use crate::errors::{CliError, Code};
use std::io::{BufRead, IsTerminal, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};

static QUIET: AtomicBool = AtomicBool::new(false);

/// `-q/--quiet`: informational messages are suppressed; warnings and errors are not.
pub fn set_quiet(quiet: bool) {
    QUIET.store(quiet, Ordering::Relaxed);
}

/// Prints an informational message to stderr unless `--quiet`.
pub fn info(message: &str) {
    if !QUIET.load(Ordering::Relaxed) {
        eprintln!("{message}");
    }
}

/// mrun's own state directory: `MRUN_HOME`, else `~/.mrun`.
pub fn mrun_home() -> PathBuf {
    if let Some(home) = std::env::var_os("MRUN_HOME").filter(|h| !h.is_empty()) {
        return PathBuf::from(home);
    }
    let home = std::env::var_os("HOME").map_or_else(|| PathBuf::from("."), PathBuf::from);
    home.join(".mrun")
}

/// Asks `question` with y/N on a terminal; `--yes` answers yes. Without a terminal and without
/// `--yes` nobody can answer: E002.
pub fn confirm(question: &str, yes: bool) -> Result<bool, CliError> {
    if yes {
        return Ok(true);
    }
    if !std::io::stdin().is_terminal() {
        return Err(CliError::new(
            Code::NeedsConfirmation,
            format!("{question} (needs confirmation)"),
        )
        .hint("pass --yes to confirm when stdin is not a terminal"));
    }
    eprint!("{question} [y/N] ");
    let _ = std::io::stderr().flush();
    let mut answer = String::new();
    std::io::stdin()
        .lock()
        .read_line(&mut answer)
        .map_err(|error| CliError::new(Code::Internal, error.to_string()))?;
    Ok(matches!(answer.trim(), "y" | "Y" | "yes" | "Yes"))
}

/// Exit code of a command result, printing the error.
pub fn exit(result: Result<(), CliError>) -> u8 {
    match result {
        Ok(()) => 0,
        Err(error) => error.report(),
    }
}

/// Finds `name` on `PATH`.
pub fn which(name: &str) -> Option<PathBuf> {
    let path = std::env::var_os("PATH")?;
    std::env::split_paths(&path)
        .map(|dir| dir.join(name))
        .find(|candidate| candidate.is_file())
}
