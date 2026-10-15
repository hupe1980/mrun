//! Command implementations shared by the binary and the integration tests.

pub mod clients;
pub mod connect;
pub mod doctor;
pub mod launch;
pub mod ps;
pub mod serve;
pub mod server;
pub mod stop;

use std::path::PathBuf;

/// mrun's own state directory: `MRUN_HOME`, else `~/.mrun`.
pub fn mrun_home() -> PathBuf {
    if let Some(home) = std::env::var_os("MRUN_HOME").filter(|h| !h.is_empty()) {
        return PathBuf::from(home);
    }
    let home = std::env::var_os("HOME").map_or_else(|| PathBuf::from("."), PathBuf::from);
    home.join(".mrun")
}

/// Asks a yes/no question on the terminal; `false` without a terminal (use `--yes`).
pub fn confirm(question: &str) -> bool {
    use std::io::{BufRead, IsTerminal, Write};
    if !std::io::stdin().is_terminal() {
        return false;
    }
    eprint!("{question} [y/N] ");
    let _ = std::io::stderr().flush();
    let mut answer = String::new();
    std::io::stdin().lock().read_line(&mut answer).is_ok()
        && matches!(answer.trim(), "y" | "Y" | "yes" | "Yes")
}

/// Finds `name` on `PATH`.
pub fn which(name: &str) -> Option<PathBuf> {
    let path = std::env::var_os("PATH")?;
    std::env::split_paths(&path)
        .map(|dir| dir.join(name))
        .find(|candidate| candidate.is_file())
}
