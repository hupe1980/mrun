//! `mrun cache clear`: drops the running server's prefix cache (003 FR-012).

use crate::cli::server;

#[derive(Debug, clap::Subcommand)]
pub enum CacheCommand {
    /// Drop all cached state of the running server (the live conversation when idle).
    Clear {
        /// Port of the server.
        #[arg(long, env = "MRUN_PORT", default_value_t = server::default_port())]
        port: u16,
    },
}

pub fn run(command: CacheCommand) -> u8 {
    let CacheCommand::Clear { port } = command;
    match server::request(port, "POST", "/mrun/cache/clear") {
        Ok((_, body)) => {
            if body["live_reset"] == "now" {
                crate::cli::info(&format!(
                    "cleared {} branches ({} MiB); the live conversation was reset",
                    body["branches_dropped"],
                    body["bytes_freed"].as_u64().unwrap_or(0) >> 20
                ));
            } else {
                crate::cli::info("cleared; the live conversation resets after the running request");
            }
            0
        }
        Err(error) => error.report(),
    }
}
