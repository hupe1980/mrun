//! mrun's `browser` tools: pageon's MCP server in the binary (`mrun mcp serve browser`). Pages
//! run in pageon's own sandbox, a content process with no files, network or programs; pageon
//! starts it as `mrun content` (this binary), which [`content`] serves.

/// Serves pageon's browser tools on stdin and stdout until stdin closes.
pub fn serve() -> Result<(), String> {
    let stdin = std::io::stdin();
    pageon::mcp::serve(
        pageon::Live::default(),
        pageon::mcp::McpOptions::default(),
        stdin.lock(),
        std::io::stdout(),
    )
    .map(|_| ())
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
