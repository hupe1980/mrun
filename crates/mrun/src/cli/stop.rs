//! `mrun stop`: stops a server started with `mrun serve --detach` (pid file in `MRUN_HOME`).

use crate::cli::server;
use crate::errors::{CliError, Code};
use std::path::PathBuf;
use std::time::{Duration, Instant};

/// Where a detached server records its process id.
pub fn pid_file() -> PathBuf {
    crate::cli::mrun_home().join("server.pid")
}

/// Where a detached server writes its log.
pub fn log_file() -> PathBuf {
    crate::cli::mrun_home().join("server.log")
}

#[derive(Debug, clap::Args)]
pub struct StopArgs {
    /// Port of the server.
    #[arg(long, env = "MRUN_PORT", default_value_t = server::default_port())]
    pub port: u16,
}

pub fn run(args: StopArgs) -> u8 {
    match stop(args.port) {
        Ok(()) => 0,
        Err(error) => error.report(),
    }
}

fn stop(port: u16) -> Result<(), CliError> {
    let pid = std::fs::read_to_string(pid_file())
        .ok()
        .and_then(|text| text.trim().parse::<u32>().ok());
    let Some(pid) = pid else {
        if server::request(port, "GET", "/health").is_ok() {
            return Err(CliError::new(
                Code::ServerUnreachable,
                format!("the server on port {port} was not started with --detach"),
            )
            .hint("stop it in the terminal where it runs (Ctrl-C)"));
        }
        return Err(CliError::new(
            Code::ServerUnreachable,
            "no detached mrun server is running",
        ));
    };
    // SIGTERM: the server finishes open requests and exits.
    let sent = std::process::Command::new("kill")
        .args(["-TERM", &pid.to_string()])
        .status()
        .is_ok_and(|s| s.success());
    if !sent {
        let _ = std::fs::remove_file(pid_file());
        return Err(CliError::new(
            Code::ServerUnreachable,
            format!("no process {pid} (the server had already stopped)"),
        ));
    }
    let deadline = Instant::now() + Duration::from_secs(60);
    while Instant::now() < deadline && server::request(port, "GET", "/health").is_ok() {
        std::thread::sleep(Duration::from_millis(200));
    }
    let _ = std::fs::remove_file(pid_file());
    crate::cli::info(&format!("mrun stopped (pid {pid})"));
    Ok(())
}
