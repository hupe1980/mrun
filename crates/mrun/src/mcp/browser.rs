//! mrun's `browser` tools: pageon's MCP server in the binary (`mrun mcp serve browser`). Pages
//! run in pageon's own sandbox, a content process with no files, network or programs; pageon
//! starts it as `mrun content` (this binary), which [`content`] serves.

/// Serves pageon's browser tools on stdin and stdout until stdin closes, with the sites, local
/// access and logins of `options` (secret values from `PAGEON_SECRET_<NAME>`).
pub fn serve(options: &super::config::BrowserOptions) -> Result<(), String> {
    let mut config = pageon::Config::default();
    for origin in &options.origins {
        config = config.origin(origin);
    }
    for secret in &options.secrets {
        config = config.secret_spec(secret);
    }
    let mut mcp = pageon::mcp::McpOptions::default();
    mcp.load.allow_private_network = options.allow_local;
    let stdin = std::io::stdin();
    pageon::mcp::serve_with(&config, mcp, stdin.lock(), std::io::stdout()).map(|_| ())
}

/// The caged content process pageon starts (`mrun content`).
pub fn content() -> std::process::ExitCode {
    match pageon::content::serve() {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("content process: {e}");
            std::process::ExitCode::from(70)
        }
    }
}
