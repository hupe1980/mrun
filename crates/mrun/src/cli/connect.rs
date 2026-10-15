//! `mrun connect <client>`: prints the persistent configuration for a client and where it goes,
//! with the running server's values. Writes nothing (004 FR-006).

use crate::cli::clients::{self, Client};
use crate::cli::server::{self, ServerInfo};

#[derive(Debug, clap::Args)]
pub struct ConnectArgs {
    pub client: Client,
    /// Port of the running server.
    #[arg(long, env = "MRUN_PORT", default_value_t = server::default_port())]
    pub port: u16,
}

pub fn run(args: ConnectArgs) -> u8 {
    let server = match ServerInfo::fetch(args.port) {
        Ok(server) => server,
        Err(error) => return error.report(),
    };
    for (target, text) in clients::persistent(args.client, &server, &crate::cli::mrun_home()) {
        println!("# {target}");
        println!("{text}");
        println!();
    }
    0
}
