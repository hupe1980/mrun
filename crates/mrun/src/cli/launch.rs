//! `mrun launch <client>`: starts a coding agent configured for the local server, for that
//! process only (004 FR-005). Starts a server first when none runs (after confirmation).

use crate::cli::clients::{self, CODEX_RESERVED, Client, SMALL_CONTEXT};
use crate::cli::server::{self, ServerInfo};
use crate::errors::{CliError, Code, Exit};
use std::process::Command;

/// Model `launch` offers when no server runs and none is named.
pub const DEFAULT_MODEL: &str = "qwen3.6";

#[derive(Debug, clap::Args)]
pub struct LaunchArgs {
    pub client: Client,
    /// Model to serve if a server has to be started.
    #[arg(long)]
    pub model: Option<String>,
    /// Print the environment and command instead of starting anything.
    #[arg(long)]
    pub dry_run: bool,
    /// Answer yes to confirmations (starting a server).
    #[arg(long, short = 'y')]
    pub yes: bool,
    /// Port of the server.
    #[arg(long, env = "MRUN_PORT", default_value_t = server::default_port())]
    pub port: u16,
    /// Arguments passed to the client.
    #[arg(last = true)]
    pub client_args: Vec<String>,
}

pub fn run(args: LaunchArgs) -> u8 {
    match launch(args) {
        Ok(code) => code,
        Err(error) => error.report(),
    }
}

fn launch(args: LaunchArgs) -> Result<u8, CliError> {
    if args.client == Client::Codex
        && let Some(conflict) = codex_conflict(&args.client_args)
    {
        eprintln!(
            "error: '-c {conflict}' conflicts with the provider mrun configures\n  hint: drop it, or configure Codex yourself with 'mrun connect codex'"
        );
        return Ok(Exit::Usage as u8);
    }
    let server = match ServerInfo::fetch(args.port) {
        Ok(server) => server,
        Err(error) if error.code == Code::ServerUnreachable && !args.dry_run => {
            start_server(&args)?;
            ServerInfo::fetch(args.port)?
        }
        Err(error) => return Err(error),
    };
    if let Some(model) = &args.model
        && !names_match(model, &server.model_id)
    {
        eprintln!(
            "error: the running server serves '{}', not '{model}'\n  hint: 'mrun stop', then 'mrun launch {} --model {model}'",
            server.model_id,
            args.client_name()
        );
        return Ok(Exit::Usage as u8);
    }
    if server.context < SMALL_CONTEXT {
        eprintln!(
            "warning: only {} tokens of context fit this Mac's memory budget; agents' system prompts and tool definitions alone take 15–25K. A Mac with more memory or a smaller model helps.",
            server.context
        );
    }
    let setup = clients::setup(args.client, &server, &crate::cli::mrun_home());
    if args.dry_run {
        for (key, value) in &setup.env {
            println!("{key}={}", shell_quote(value));
        }
        for (path, _) in &setup.files {
            println!("# writes {}", path.display());
        }
        let mut command = vec![args.client.binary().to_string()];
        command.extend(
            setup
                .args
                .iter()
                .chain(&args.client_args)
                .map(|a| shell_quote(a)),
        );
        println!("{}", command.join(" "));
        return Ok(0);
    }
    for (path, text) in &setup.files {
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir).map_err(|e| internal(&e))?;
        }
        std::fs::write(path, text).map_err(|e| internal(&e))?;
    }
    let Some(binary) = crate::cli::which(args.client.binary()) else {
        return Err(CliError::new(
            Code::ClientMissing,
            format!("'{}' is not installed or not on PATH", args.client.binary()),
        )
        .hint(format!("install it: {}", args.client.install_hint())));
    };
    if args.client == Client::ClaudeCode {
        check_claude_version(&binary);
    }
    let mut command = Command::new(&binary);
    command
        .args(&setup.args)
        .args(&args.client_args)
        .envs(setup.env.iter().map(|(k, v)| (k, v)));
    exec(command)
}

impl LaunchArgs {
    fn client_name(&self) -> &'static str {
        match self.client {
            Client::ClaudeCode => "claude-code",
            Client::Codex => "codex",
            Client::OpenCode => "opencode",
        }
    }
}

fn internal(error: &dyn std::fmt::Display) -> CliError {
    CliError::new(Code::Internal, error.to_string())
}

/// A served id matches a requested name when equal, or when the requested name (an alias such
/// as `qwen3-coder`) is the start of the served repository's name up to a `-` boundary
/// (`mlx-community/Qwen3-Coder-30B-…`), compared case-insensitively.
fn names_match(requested: &str, served: &str) -> bool {
    let (r, s) = (requested.to_lowercase(), served.to_lowercase());
    let name = s.rsplit('/').next().unwrap_or(&s);
    r == s
        || name == r
        || name
            .strip_prefix(r.as_str())
            .is_some_and(|rest| rest.starts_with('-'))
}

/// A user `-c key=value` that would replace mrun's provider selection.
fn codex_conflict(args: &[String]) -> Option<String> {
    let mut iter = args.iter();
    while let Some(arg) = iter.next() {
        let value = match arg.as_str() {
            "-c" | "--config" => iter.next().cloned(),
            _ => arg
                .strip_prefix("--config=")
                .or_else(|| arg.strip_prefix("-c"))
                .filter(|v| !v.is_empty())
                .map(str::to_string),
        };
        let Some(value) = value else { continue };
        let key = value.split('=').next().unwrap_or_default().trim();
        if CODEX_RESERVED
            .iter()
            .any(|r| key == *r || key.starts_with(&format!("{r}.")))
        {
            return Some(value);
        }
    }
    None
}

/// Starts `mrun serve --detach` after confirmation and waits for it.
fn start_server(args: &LaunchArgs) -> Result<(), CliError> {
    let model = args.model.as_deref().unwrap_or(DEFAULT_MODEL);
    let question = format!("No mrun server is running. Start one with {model}?");
    // Without a terminal and without --yes: report the missing server, not the prompt.
    if !crate::cli::confirm(&question, args.yes).unwrap_or(false) {
        return Err(CliError::new(
            Code::ServerUnreachable,
            format!("no mrun server is running on 127.0.0.1:{}", args.port),
        )
        .hint(format!(
            "start one with 'mrun serve --detach --model {model}', or pass --yes"
        )));
    }
    let exe = std::env::current_exe().map_err(|e| internal(&e))?;
    let status = Command::new(exe)
        .args(["serve", "--detach", "--model", model, "--port"])
        .arg(args.port.to_string())
        .status()
        .map_err(|e| internal(&e))?;
    if status.success() {
        Ok(())
    } else {
        Err(
            CliError::new(Code::ServerUnreachable, "the server did not start").hint(format!(
                "see {}",
                crate::cli::mrun_home().join("server.log").display()
            )),
        )
    }
}

fn check_claude_version(binary: &std::path::Path) {
    let Ok(output) = Command::new(binary).arg("--version").output() else {
        return;
    };
    let text = String::from_utf8_lossy(&output.stdout);
    if let Some(version) = clients::parse_version(&text)
        && version < clients::CLAUDE_CODE_MIN
    {
        let (a, b, c) = clients::CLAUDE_CODE_MIN;
        eprintln!(
            "warning: Claude Code {}.{}.{} is older than {a}.{b}.{c}, the oldest mrun supports\n  hint: claude update",
            version.0, version.1, version.2
        );
    }
}

#[cfg(unix)]
fn exec(mut command: Command) -> Result<u8, CliError> {
    use std::os::unix::process::CommandExt;
    let error = command.exec();
    Err(internal(&error))
}

#[cfg(not(unix))]
fn exec(mut command: Command) -> Result<u8, CliError> {
    let status = command.status().map_err(|e| internal(&e))?;
    Ok(status.code().unwrap_or(1) as u8)
}

/// Quotes a value for display as a shell word.
fn shell_quote(value: &str) -> String {
    if !value.is_empty()
        && value
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || "-_./:=@,+".contains(c))
    {
        value.to_string()
    } else {
        format!("'{}'", value.replace('\'', "'\\''"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn codex_provider_overrides_are_refused() {
        let args = |a: &[&str]| a.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        assert!(codex_conflict(&args(&["-c", "model_provider=\"openai\""])).is_some());
        assert!(codex_conflict(&args(&["-c", "model_providers.x.base_url=\"y\""])).is_some());
        assert!(codex_conflict(&args(&["-c", "model_reasoning_effort=\"high\""])).is_none());
        assert!(codex_conflict(&args(&["--full-auto"])).is_none());
    }

    #[test]
    fn served_names_match_aliases_and_repos() {
        assert!(names_match(
            "qwen3-coder",
            "mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit"
        ));
        assert!(names_match("org/x", "org/x"));
        assert!(!names_match(
            "gpt-oss",
            "mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit"
        ));
        assert!(names_match("qwen3.6", "mlx-community/Qwen3.6-35B-A3B-4bit"));
        assert!(!names_match("qwen3", "mlx-community/Qwen3.6-35B-A3B-4bit"));
    }

    #[test]
    fn values_are_quoted_for_the_shell() {
        assert_eq!(
            shell_quote("http://127.0.0.1:8484"),
            "http://127.0.0.1:8484"
        );
        assert_eq!(shell_quote(""), "''");
        assert_eq!(shell_quote("{\"a\": 1}"), "'{\"a\": 1}'");
        assert_eq!(shell_quote("it's"), "'it'\\''s'");
    }
}
