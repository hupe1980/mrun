//! `mrun stop`: stops the mrun server on a port, detached (pid file per port in `MRUN_HOME`) or
//! in the foreground (pid from `/mrun/status`). The pid is checked to be an mrun process before
//! it gets SIGTERM; the server drains open requests and exits.

use crate::cli::server;
use crate::errors::{CliError, Code};
use std::path::PathBuf;
use std::process::Command;
use std::time::{Duration, Instant};

/// How long a server may take to finish open requests and exit.
const STOP_TIMEOUT: Duration = Duration::from_secs(60);

/// Where a detached server on `port` records its process id.
pub fn pid_file(port: u16) -> PathBuf {
    crate::cli::mrun_home().join(format!("server-{port}.pid"))
}

#[derive(Debug, clap::Args)]
pub struct StopArgs {
    /// Port of the server.
    #[arg(long, env = "MRUN_PORT", default_value_t = server::default_port())]
    pub port: u16,
}

pub fn run(args: StopArgs) -> u8 {
    crate::cli::exit(stop(args.port).map(|pid| match pid {
        Some(pid) => crate::cli::info(&format!("mrun stopped (pid {pid})")),
        // Nothing to stop is the state the user asked for.
        None => crate::cli::info(&format!(
            "no mrun server is running on port {}; nothing to stop",
            args.port
        )),
    }))
}

/// Stops the mrun server on `port`; returns its pid, or `None` when nothing runs there.
pub fn stop(port: u16) -> Result<Option<u32>, CliError> {
    let file = pid_file(port);
    let pid = std::fs::read_to_string(&file)
        .ok()
        .and_then(|text| text.trim().parse::<u32>().ok());
    let up = server::health(port) != server::Health::Down;
    let Some(pid) = pid else {
        if !up {
            return Ok(None);
        }
        // A foreground server (`mrun serve` without `--detach`) has no pid file: it reports its
        // pid in `/mrun/status`. It is stopped the same way once the pid is an mrun process.
        let Some(pid) = reported_pid(port) else {
            return Err(CliError::new(
                Code::ServerUnreachable,
                format!("the server on port {port} does not report its process id"),
            )
            .hint("'mrun ps' shows what runs on the port"));
        };
        if !is_mrun_process(pid) {
            return Err(CliError::new(
                Code::ServerUnreachable,
                format!("process {pid} on port {port} is not an mrun server"),
            )
            .hint("'mrun ps' shows what runs on the port"));
        }
        return Ok(terminate(pid, &file)?.then_some(pid));
    };
    // Nothing answers and the pid is not an mrun process: a stale pid file of a server that is
    // gone (a busy server that does not answer still runs as `mrun` and is stopped below).
    if !up && !is_mrun_process(pid) {
        let _ = std::fs::remove_file(&file);
        crate::cli::info(&format!(
            "removed the stale pid file {} (process {pid} is not an mrun server)",
            file.display()
        ));
        return Ok(None);
    }
    if !is_our_server(pid, port) {
        let _ = std::fs::remove_file(&file);
        return Err(CliError::new(
            Code::ServerUnreachable,
            format!("process {pid} is not the mrun server on port {port} (stale pid file removed)"),
        )
        .hint("'mrun ps' shows what runs on the port"));
    }
    Ok(terminate(pid, &file)?.then_some(pid))
}

/// Sends SIGTERM (the server finishes open requests and exits) and waits for the process to go;
/// removes the pid file. `false` when the process exited before the signal.
fn terminate(pid: u32, file: &std::path::Path) -> Result<bool, CliError> {
    let sent = Command::new("kill")
        .args(["-TERM", &pid.to_string()])
        .status()
        .is_ok_and(|status| status.success());
    if !sent {
        // The server exited between the check and the signal.
        let _ = std::fs::remove_file(file);
        return Ok(false);
    }
    let deadline = Instant::now() + STOP_TIMEOUT;
    while process_name(pid).is_some() {
        if Instant::now() > deadline {
            return Err(CliError::new(
                Code::ServerUnreachable,
                format!(
                    "the server (pid {pid}) is still running {} s after SIGTERM",
                    STOP_TIMEOUT.as_secs()
                ),
            )
            .hint(format!(
                "it may be finishing a long request; 'kill -9 {pid}' ends it at once"
            )));
        }
        std::thread::sleep(Duration::from_millis(200));
    }
    let _ = std::fs::remove_file(file);
    Ok(true)
}

/// The pid the server on `port` reports in `/mrun/status`.
fn reported_pid(port: u16) -> Option<u32> {
    server::request(port, "GET", "/mrun/status")
        .ok()
        .and_then(|(_, status)| status["pid"].as_u64())
        .and_then(|pid| u32::try_from(pid).ok())
}

/// The process runs an mrun executable (guards against a reused pid): the executable's file
/// name, not any path component, names mrun.
pub(crate) fn is_mrun_process(pid: u32) -> bool {
    process_name(pid).is_some_and(|name| is_mrun_executable(&name))
}

fn is_mrun_executable(name: &str) -> bool {
    std::path::Path::new(name)
        .file_name()
        .and_then(|file| file.to_str())
        .is_some_and(|file| file.starts_with("mrun"))
}

/// The pid belongs to the server on `port`: the server reports it in `/mrun/status`, or (when it
/// reports none) the process is an mrun executable.
fn is_our_server(pid: u32, port: u16) -> bool {
    match reported_pid(port) {
        Some(reported) => reported == pid,
        None => is_mrun_process(pid),
    }
}

/// `ps -p <pid> -o comm=`: the executable of a running process.
fn process_name(pid: u32) -> Option<String> {
    let output = Command::new("ps")
        .args(["-p", &pid.to_string(), "-o", "comm="])
        .output()
        .ok()?;
    let name = String::from_utf8_lossy(&output.stdout).trim().to_string();
    (output.status.success() && !name.is_empty()).then_some(name)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn foreign_processes_are_not_ours() {
        // pid 1 (launchd or init) is never an mrun server.
        assert!(!is_our_server(1, 1));
        assert!(process_name(1).is_some());
        assert_eq!(process_name(u32::MAX >> 1), None);
        assert!(!is_mrun_process(1));
    }

    #[test]
    fn only_the_executable_name_counts() {
        assert!(is_mrun_executable("/opt/homebrew/bin/mrun"));
        assert!(is_mrun_executable("mrun"));
        assert!(!is_mrun_executable("/Users/me/mrun/target/debug/cli-1234"));
        assert!(!is_mrun_executable("/usr/bin/python3"));
    }
}
