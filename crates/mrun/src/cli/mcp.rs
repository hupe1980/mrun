//! `mrun mcp` (028): the MCP servers mrun's own chats use (`~/.mrun/mcp.json`), and mrun's own
//! tools as MCP servers any client can start.

use crate::errors::{CliError, Code};
use crate::mcp::{Registry, config, serve};
use serde_json::Value;
use std::path::PathBuf;

#[derive(Debug, clap::Subcommand)]
pub enum McpCommand {
    /// The servers of mcp.json with their tools, sandbox and errors (starts them).
    Ls(LsArgs),
    /// Copy another client's servers into mcp.json (names it already has are kept).
    Import(ImportArgs),
    /// Add mrun's own tools: `workspace` (read, write, edit, glob, grep, bash in a folder) or
    /// `fetch` (web pages).
    Add(AddArgs),
    /// Let a caged server read, write or run a path, or reach a host.
    Allow(AllowArgs),
    /// Serve mrun's own tools over stdio (what mcp.json starts; any MCP client can).
    Serve(ServeArgs),
}

#[derive(Debug, clap::Args)]
pub struct LsArgs {
    /// Print JSON.
    #[arg(long)]
    pub json: bool,
}

#[derive(Debug, clap::Args)]
pub struct ImportArgs {
    /// The client whose servers to copy.
    #[arg(value_parser = ["claude-code", "claude-desktop"])]
    pub client: String,
}

#[derive(Debug, clap::Args)]
pub struct AddArgs {
    /// `workspace` or `fetch`.
    #[arg(value_parser = ["workspace", "fetch"])]
    pub kit: String,
    /// The workspace's folder (default: the current one).
    pub dir: Option<PathBuf>,
}

#[derive(Debug, clap::Args)]
#[group(id = "what", required = true, multiple = false)]
pub struct AllowArgs {
    /// The server.
    pub server: String,
    /// Let it read this path.
    #[arg(long, group = "what")]
    pub read: Option<PathBuf>,
    /// Let it write this path.
    #[arg(long, group = "what")]
    pub write: Option<PathBuf>,
    /// Let it run this program.
    #[arg(long, group = "what")]
    pub exec: Option<PathBuf>,
    /// Let it reach this host through the sandbox's proxy.
    #[arg(long, group = "what")]
    pub host: Option<String>,
}

#[derive(Debug, clap::Args)]
pub struct ServeArgs {
    /// `workspace` or `fetch`.
    #[arg(value_parser = ["workspace", "fetch"])]
    pub kit: String,
    /// The workspace's folder; every path stays inside it.
    #[arg(long, required_if_eq("kit", "workspace"))]
    pub root: Option<PathBuf>,
}

pub fn run(command: McpCommand) -> u8 {
    crate::cli::exit(match command {
        McpCommand::Ls(args) => ls(&args),
        McpCommand::Import(args) => import(&args),
        McpCommand::Add(args) => add(&args),
        McpCommand::Allow(args) => allow(&args),
        McpCommand::Serve(args) => serve_kit(&args),
    })
}

fn home() -> PathBuf {
    crate::cli::mrun_home()
}

fn invalid(message: impl Into<String>) -> CliError {
    CliError::new(Code::InvalidArguments, message)
}

fn runtime() -> Result<tokio::runtime::Runtime, CliError> {
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .map_err(|e| CliError::new(Code::Internal, e.to_string()))
}

/// `~` for the home folder in what is shown.
pub fn tilde(path: &str) -> String {
    match std::env::home_dir().and_then(|h| h.to_str().map(str::to_string)) {
        Some(h) if path.starts_with(&h) => format!("~{}", &path[h.len()..]),
        _ => path.to_string(),
    }
}

fn ls(args: &LsArgs) -> Result<(), CliError> {
    let registry = Registry::new(home());
    let list = runtime()?.block_on(registry.list(true));
    if args.json {
        println!(
            "{}",
            serde_json::to_string_pretty(&list).unwrap_or_default()
        );
        return Ok(());
    }
    print!("{}", render(&list));
    Ok(())
}

/// The listing as the terminal shows it.
pub fn render(list: &Value) -> String {
    let path = tilde(list["path"].as_str().unwrap_or(""));
    let servers = list["servers"].as_array().cloned().unwrap_or_default();
    let mut out = String::new();
    if servers.is_empty() {
        out.push_str(&format!(
            "No tool servers in {path}.\n\n  mrun mcp add workspace      read, edit and run in this folder\n  mrun mcp add fetch          read web pages\n  mrun mcp import claude-code  the servers Claude Code has\n"
        ));
        return out;
    }
    out.push_str(&format!("{path}\n"));
    for s in &servers {
        let name = s["name"].as_str().unwrap_or("");
        let what = match s["kind"].as_str() {
            Some("http") => format!("{} (remote)", s["host"].as_str().unwrap_or("")),
            _ => tilde(s["command"].as_str().unwrap_or("")),
        };
        let cage = match &s["sandbox"] {
            Value::Bool(false) if s["kind"] == "http" => String::new(),
            Value::Bool(false) => "  · not sandboxed".into(),
            b => {
                let dir = tilde(b["workspace"].as_str().unwrap_or("~"));
                let mode = if b["write"] == true {
                    "writes"
                } else {
                    "reads"
                };
                let net = match b["hosts"].as_array().map(Vec::len) {
                    Some(0) | None => "no network".to_string(),
                    Some(n) => format!("{n} hosts"),
                };
                format!("  · sandboxed: {mode} {dir}, {net}")
            }
        };
        out.push_str(&format!("\n{name}  {what}{cage}\n"));
        if let Some(e) = s["error"].as_str() {
            out.push_str(&format!("  ✗ {e}\n"));
        }
        let tools: Vec<&str> = s["tools"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|t| t["name"].as_str())
            .collect();
        if !tools.is_empty() {
            out.push_str(&format!("  {}\n", tools.join("  ")));
        }
    }
    for e in list["errors"].as_array().into_iter().flatten() {
        out.push_str(&format!("\n✗ {}\n", e.as_str().unwrap_or("")));
    }
    out
}

fn import(args: &ImportArgs) -> Result<(), CliError> {
    let file = config::client_config(&args.client).ok_or_else(|| invalid("unknown client"))?;
    let entries = config::servers_in(&file).map_err(invalid)?;
    let added = config::add(&home(), entries).map_err(invalid)?;
    if added.is_empty() {
        println!("Nothing new in {}.", tilde(&file.to_string_lossy()));
    } else {
        println!(
            "Added {} (each sandboxed unless its entry says \"sandbox\": false).",
            added.join(", ")
        );
    }
    Ok(())
}

fn add(args: &AddArgs) -> Result<(), CliError> {
    let dir = match &args.dir {
        Some(d) => d.clone(),
        None => std::env::current_dir().map_err(|e| invalid(e.to_string()))?,
    };
    let dir = dir
        .canonicalize()
        .map_err(|e| invalid(format!("{}: {e}", dir.display())))?;
    let entry = config::builtin(&args.kit, Some(&dir)).ok_or_else(|| invalid("unknown tools"))?;
    let name = entry.0.clone();
    let added = config::add(&home(), vec![entry]).map_err(invalid)?;
    if added.is_empty() {
        return Err(invalid(format!(
            "mcp.json already has a server named {name}"
        )));
    }
    match args.kit.as_str() {
        "workspace" => println!(
            "Added workspace: read, write, edit, glob, grep and bash in {}. Turn it on in a chat; each call asks first.",
            tilde(&dir.to_string_lossy())
        ),
        _ => println!("Added fetch: web pages as text. Each call names its host and asks first."),
    }
    Ok(())
}

fn allow(args: &AllowArgs) -> Result<(), CliError> {
    let path = |access: &str, p: &PathBuf| {
        let p = std::path::absolute(p).unwrap_or_else(|_| p.clone());
        config::Allowed {
            access: access.into(),
            path: p,
        }
    };
    let rule = args
        .read
        .as_ref()
        .map(|p| path("read", p))
        .or_else(|| args.write.as_ref().map(|p| path("write", p)))
        .or_else(|| args.exec.as_ref().map(|p| path("exec", p)));
    config::allow(&home(), &args.server, rule.clone(), args.host.as_deref()).map_err(invalid)?;
    match (rule, &args.host) {
        (Some(r), _) => println!("{}: allowed {} {}", args.server, r.access, r.path.display()),
        (None, Some(h)) => println!("{}: allowed {h}", args.server),
        _ => {}
    }
    Ok(())
}

fn serve_kit(args: &ServeArgs) -> Result<(), CliError> {
    let kit = match (args.kit.as_str(), &args.root) {
        ("workspace", Some(root)) => serve::Kit::Workspace(
            root.canonicalize()
                .map_err(|e| invalid(format!("{}: {e}", root.display())))?,
        ),
        ("fetch", _) => serve::Kit::Fetch,
        _ => return Err(invalid("workspace needs --root")),
    };
    runtime()?
        .block_on(serve::run(kit))
        .map_err(|e| CliError::new(Code::Internal, e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn the_listing_reads_well() {
        let empty = render(&json!({"path": "/x/mcp.json", "servers": [], "errors": []}));
        assert!(empty.contains("mrun mcp add workspace"));
        let list = json!({"path": "/x/mcp.json", "errors": [], "servers": [
            {"name": "workspace", "kind": "stdio", "command": "mrun mcp serve workspace --root /w",
             "sandbox": {"workspace": "/w", "write": true, "hosts": []},
             "tools": [{"name": "read"}, {"name": "bash"}]},
            {"name": "gh", "kind": "http", "host": "api.example.com", "sandbox": false, "error": "HTTP 401"},
        ]});
        let text = render(&list);
        assert!(
            text.contains(
                "workspace  mrun mcp serve workspace --root /w  · sandboxed: writes /w, no network"
            ),
            "{text}"
        );
        assert!(text.contains("  read  bash"));
        assert!(
            text.contains("gh  api.example.com (remote)\n  ✗ HTTP 401"),
            "{text}"
        );
    }
}
