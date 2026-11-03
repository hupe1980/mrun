//! `mrun launch [client]`: starts a coding agent configured for the local server, for that
//! process only (004 FR-005, D037). Without a server it offers, in one confirmation, to pull the
//! model (this machine's default unless `--model` names one) and start the server; a loading
//! server is waited for, and so is a `mrun cloud up` still starting its pod; a server with another
//! model is offered a restart.

use crate::cli::clients::{self, CODEX_RESERVED, Client};
use crate::cli::serve::{self, StartError};
use crate::cli::server::{self, Health, LoadEnd, ServerInfo};
use crate::errors::{CliError, Code};
use crate::hub::{self, Ctx, download, human_bytes};
use std::ffi::OsString;
use std::process::Command;

#[derive(Debug, clap::Args)]
pub struct LaunchArgs {
    /// Client to start [default: the only one installed].
    pub client: Option<Client>,
    /// Model to serve [default: the running server's, else this machine's default].
    #[arg(long, env = "MRUN_MODEL")]
    pub model: Option<String>,
    /// Print the environment and command instead of starting anything.
    #[arg(long)]
    pub dry_run: bool,
    /// Answer yes to confirmations (pulling the model, starting or restarting the server).
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
    launch(args).unwrap_or_else(StartError::report)
}

fn launch(args: LaunchArgs) -> Result<u8, StartError> {
    let client = pick_client(args.client, &Client::installed())?;
    if client == Client::Codex
        && let Some(conflict) = codex_conflict(&args.client_args)
    {
        return Err(CliError::new(
            Code::InvalidArguments,
            format!("'-c {conflict}' conflicts with the provider mrun configures"),
        )
        .hint("drop it, or configure Codex yourself with 'mrun connect codex'")
        .into());
    }
    let server = ensure_server(&args, client)?;
    check_min_context(client, &server.model_id, server.context, None)?;
    if client == Client::ClaudeCode {
        warn_settings_overrides();
    }
    if let Some(note) = hub::aliases::preview_note(&server.model_id) {
        crate::cli::info(&note);
    }
    let setup = clients::setup(client, &server);
    if args.dry_run {
        print_dry_run(client, &setup, &args.client_args);
        return Ok(0);
    }
    for (path, text) in &setup.files {
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir).map_err(|e| internal(&e))?;
        }
        std::fs::write(path, text).map_err(|e| internal(&e))?;
    }
    let Some(binary) = crate::cli::which(client.binary()) else {
        return Err(client_missing(client).into());
    };
    if client == Client::ClaudeCode {
        check_claude_version(&binary);
    }
    crate::cli::info(&format!(
        "{} → {} at {} ({} tokens of context)",
        client.name(),
        server.model_id,
        server.base_url(),
        server.context
    ));
    let mut command = Command::new(&binary);
    command
        .args(&setup.args)
        .args(client_command_args(client, &args.client_args))
        .envs(setup.env.iter().map(|(k, v)| (k, v)));
    Ok(exec(command)?)
}

/// The client's arguments as `launch` passes them. OpenCode 2.x's UI and `run` talk to a shared
/// background service started without mrun's `OPENCODE_CONFIG_CONTENT` (it keeps the user's own
/// default model and working directory), so `launch` asks for a private server with
/// `--standalone`: first for the UI, right after `run` for `run`; unchanged when the user chose a
/// server (`--standalone`, `--server`) or another subcommand.
fn client_command_args(client: Client, client_args: &[String]) -> Vec<String> {
    let mut out = client_args.to_vec();
    if client != Client::OpenCode
        || out
            .iter()
            .any(|a| a == "--standalone" || a == "--server" || a.starts_with("--server="))
    {
        return out;
    }
    match out.first().map(String::as_str) {
        None => out.push("--standalone".into()),
        Some(first) if first.starts_with('-') => out.insert(0, "--standalone".into()),
        Some("run") => out.insert(1, "--standalone".into()),
        Some(_) => {}
    }
    out
}

/// The named client, else the only installed one.
fn pick_client(named: Option<Client>, installed: &[Client]) -> Result<Client, CliError> {
    if let Some(client) = named {
        return Ok(client);
    }
    match installed {
        [one] => Ok(*one),
        [] => Err(CliError::new(
            Code::ClientMissing,
            "no supported coding agent is installed (claude, codex or opencode)",
        )
        .hint(format!(
            "install one, e.g. {}",
            Client::ClaudeCode.install_hint()
        ))),
        many => Err(CliError::new(
            Code::InvalidArguments,
            format!(
                "several coding agents are installed: {}",
                many.iter().map(|c| c.name()).collect::<Vec<_>>().join(", ")
            ),
        )
        .hint(format!("name one: mrun launch {}", many[0].name()))),
    }
}

fn client_missing(client: Client) -> CliError {
    CliError::new(
        Code::ClientMissing,
        format!("'{}' is not installed or not on PATH", client.binary()),
    )
    .hint(format!("install it: {}", client.install_hint()))
}

/// A server serving the wanted model: the running one, a loading one once loaded, or a new one
/// (after pulling the model) with the user's consent.
fn ensure_server(args: &LaunchArgs, client: Client) -> Result<ServerInfo, StartError> {
    match server::health(args.port) {
        Health::Ready(_) => {
            let server = ServerInfo::fetch(args.port)?;
            match &args.model {
                Some(model) if !names_match(model, &server.model_id) => {
                    restart(args, client, model, &server.model_id)
                }
                _ => Ok(server),
            }
        }
        Health::Loading(model) => {
            let label = serve::loading_label(model.as_deref().unwrap_or("the model"));
            wait_for(args.port, &label)?;
            ServerInfo::fetch(args.port).map_err(Into::into)
        }
        Health::Failed(error) => Err(CliError::from(error).into()),
        Health::Down => {
            if let Some(starting) = crate::cli::cloud::starting(args.port) {
                if args.dry_run {
                    return Err(no_server(args.port)
                        .hint(format!(
                            "'mrun cloud up' (pid {}) is starting {} for this port; without --dry-run, launch waits for it",
                            starting.pid, starting.model
                        ))
                        .into());
                }
                wait_for_cloud(args.port, &starting)?;
                return ensure_server(args, client);
            }
            let model = match &args.model {
                Some(model) => model.clone(),
                // The server it starts inherits `MRUN_BUDGET_GIB`; pick as it will (D060).
                None => hub::default_model(
                    &Ctx::from_env().with_budget(
                        std::env::var("MRUN_BUDGET_GIB")
                            .ok()
                            .and_then(|v| crate::cli::model::parse_budget_gib(&v).ok()),
                    ),
                )?,
            };
            if args.dry_run {
                return Err(no_server(args.port)
                    .hint(format!(
                        "without --dry-run, launch offers to start 'mrun serve --detach --model {model}'"
                    ))
                    .into());
            }
            start(
                args,
                client,
                &model,
                &format!("No mrun server is running on port {}.", args.port),
            )?;
            ServerInfo::fetch(args.port).map_err(Into::into)
        }
    }
}

fn no_server(port: u16) -> CliError {
    CliError::new(
        Code::ServerUnreachable,
        format!("no mrun server is running on 127.0.0.1:{port}"),
    )
}

/// The running server serves `served`; offers to stop it and start `model`.
fn restart(
    args: &LaunchArgs,
    client: Client,
    model: &str,
    served: &str,
) -> Result<ServerInfo, StartError> {
    let other = || {
        CliError::new(
            Code::InvalidArguments,
            format!("the running server serves '{served}', not '{model}'"),
        )
        .hint(format!(
            "'mrun stop', then 'mrun launch {} --model {model}'; or drop --model",
            client.name()
        ))
    };
    if args.dry_run {
        return Err(other().into());
    }
    let question = format!(
        "The server on port {} serves '{served}'. Stop it and start '{model}'?",
        args.port
    );
    if !crate::cli::confirm(&question, args.yes)? {
        return Err(other().into());
    }
    crate::cli::stop::stop(args.port)?;
    start(args, client, model, "")?;
    ServerInfo::fetch(args.port).map_err(Into::into)
}

/// Starts a detached server with `model`, pulling it first when it is not here. One
/// confirmation covers both; `context` opens the question.
fn start(args: &LaunchArgs, client: Client, model: &str, context: &str) -> Result<(), StartError> {
    let ctx = Ctx::from_env();
    let (plan, local) = match hub::resolve(&ctx.paths, model) {
        Ok(local) => (None, local.manifest),
        Err(error) if error.code == Code::ModelNotFound => {
            (Some(block_on(download::plan(&ctx, model))?), None)
        }
        Err(error) => return Err(error.into()),
    };
    // Refuse before downloading or starting anything when the model cannot give the client
    // its minimum context here.
    if let Some(device) = ctx.device {
        let estimate = match (&plan, &local) {
            (Some(plan), _) => Some((plan.fit(device), plan.loaded_weights(), plan.model_info())),
            (None, Some(manifest)) => Some((
                manifest.fit(device),
                manifest.loaded_weights(),
                manifest.model_info(),
            )),
            (None, None) => None,
        };
        // D062: other processes may hold the memory the model needs; say so before anything
        // is downloaded or started.
        if let Some((_, weights, info)) = &estimate {
            hub::check_available(device, model, *weights, info)?;
        }
        if let Some((fit, weights, info)) = estimate
            && fit.fits
        {
            check_min_context(
                client,
                model,
                fit.max_context,
                Some((weights, &info, fit.budget)),
            )?;
        }
    }
    let question = match &plan {
        None => format!("{context} Start one with {model}?"),
        Some(plan) => format!(
            "{context} {model} is not downloaded yet:\n  {}\n  {}\nDownload it and start the server?",
            plan.summary(),
            fit_line(&ctx, plan)
        ),
    };
    let question = question.trim_start();
    if !crate::cli::confirm(question, args.yes).unwrap_or(false) {
        return Err(no_server(args.port)
            .hint(format!(
                "start one with 'mrun serve --detach --model {model}', or pass --yes"
            ))
            .into());
    }
    if let Some(plan) = &plan {
        block_on(download::download(&ctx, plan))?;
    }
    let child_args: Vec<OsString> = ["serve", "--model", model, "--port", &args.port.to_string()]
        .map(OsString::from)
        .into();
    serve::start_detached(args.port, &child_args, model, &[])
}

/// D048/D055: `launch` refuses a model whose context on this Mac is below the client's minimum,
/// naming the budget it would need. `model` is the served id or the name to start; `known`
/// carries weights, model info and the budget when the caller has them, else the manifest of
/// `model` is looked up.
fn check_min_context(
    client: Client,
    model: &str,
    context: u32,
    known: Option<(u64, &mrun_engine::ModelInfo, u64)>,
) -> Result<(), CliError> {
    let minimum = client.min_context();
    if context >= minimum {
        return Ok(());
    }
    let ctx = Ctx::from_env();
    let looked_up = || {
        let manifest = hub::find_manifest(&ctx.paths, model).ok()?;
        let budget = ctx.device.map_or(0, |device| manifest.fit(device).budget);
        Some((manifest.loaded_weights(), manifest.model_info(), budget))
    };
    let needed = match known {
        Some((weights, info, budget)) => Some((hub::budget_needed(minimum, weights, info), budget)),
        None => looked_up()
            .map(|(weights, info, budget)| (hub::budget_needed(minimum, weights, &info), budget)),
    };
    let hint = match needed {
        Some((needed, budget)) if budget > 0 => format!(
            "{model} needs a memory budget of about {} for {minimum} tokens; this Mac's is {}. Use a smaller model (--model), or a Mac with more memory",
            human_bytes(needed),
            human_bytes(budget)
        ),
        Some((needed, _)) => format!(
            "{model} needs a memory budget of about {} for {minimum} tokens; use a smaller model (--model)",
            human_bytes(needed)
        ),
        None => "use a smaller model (--model), or a Mac with more memory".into(),
    };
    Err(CliError::new(
        Code::DoesNotFit,
        format!(
            "{model} leaves {context} tokens of context on this Mac; {} needs at least {minimum}",
            client.name()
        ),
    )
    .hint(hint))
}

fn fit_line(ctx: &Ctx, plan: &download::Plan) -> String {
    match ctx.device.map(|device| plan.fit(device)) {
        Some(fit) if fit.fits => {
            format!("fits this Mac: up to {} tokens of context", fit.max_context)
        }
        Some(fit) => format!(
            "does NOT fit this Mac's memory budget of {}",
            human_bytes(fit.budget)
        ),
        None => "fit unknown (this machine's memory could not be read)".into(),
    }
}

/// Waits for a server another command is starting.
fn wait_for(port: u16, label: &str) -> Result<(), CliError> {
    let tail = || {
        eprintln!("{}", crate::cli::logs::tail(5));
    };
    match server::wait_until_loaded(port, label, None) {
        LoadEnd::Ready => Ok(()),
        LoadEnd::Failed(error) => {
            tail();
            Err(CliError::from(error))
        }
        LoadEnd::Exited(_) | LoadEnd::TimedOut => {
            tail();
            Err(no_server(port).hint("'mrun logs' shows what happened"))
        }
    }
}

/// Waits until the relay of a starting `mrun cloud up` serves on `port`; fails when that `cloud up`
/// ends without one (its own output says why).
fn wait_for_cloud(port: u16, starting: &crate::cli::cloud::Starting) -> Result<(), CliError> {
    use std::io::{IsTerminal, Write};
    let label = format!(
        "waiting for the cloud model {} ('mrun cloud up', pid {})",
        starting.model, starting.pid
    );
    let start = std::time::Instant::now();
    let live = std::io::stderr().is_terminal() && !crate::cli::is_quiet();
    if !live {
        crate::cli::info(&format!("{label}…"));
    }
    let ended = loop {
        if server::health(port) != Health::Down {
            break false;
        }
        if crate::cli::cloud::starting(port).is_none() {
            // The relay starts before the marker goes: look once more.
            break server::health(port) == Health::Down;
        }
        if live {
            eprint!("\r{label}… {} s", start.elapsed().as_secs());
            let _ = std::io::stderr().flush();
        }
        std::thread::sleep(std::time::Duration::from_secs(1));
    };
    let seconds = start.elapsed().as_secs();
    if live {
        eprintln!("\r{label}… {seconds} s");
    }
    if ended {
        return Err(no_server(port).hint(
            "'mrun cloud up' ended without starting the relay; its output says why ('mrun cloud status' shows the pod)",
        ));
    }
    Ok(())
}

fn block_on<T>(future: impl Future<Output = Result<T, CliError>>) -> Result<T, CliError> {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|e| internal(&e))?
        .block_on(future)
}

fn internal(error: &dyn std::fmt::Display) -> CliError {
    CliError::new(Code::Internal, error.to_string())
}

fn print_dry_run(client: Client, setup: &clients::Setup, client_args: &[String]) {
    for (key, value) in &setup.env {
        println!("{key}={}", shell_quote(value));
    }
    for (path, _) in &setup.files {
        println!("# writes {}", path.display());
    }
    let mut command = vec![client.binary().to_string()];
    let client_args = client_command_args(client, client_args);
    command.extend(
        setup
            .args
            .iter()
            .chain(&client_args)
            .map(|a| shell_quote(a)),
    );
    println!("{}", command.join(" "));
}

/// `~/.claude/settings.json` `env` beats the process environment `launch` sets.
fn warn_settings_overrides() {
    let overrides = clients::claude_settings_overrides();
    if !overrides.is_empty() {
        eprintln!(
            "warning: {} in \"env\", which overrides what launch sets\n  hint: remove them there (use 'mrun connect claude-code' for a separate settings file)",
            clients::describe_overrides(&overrides)
        );
    }
}

/// A served id matches a requested name when equal (ignoring case), when one is an alias pinned
/// to the other, or (for names that are not aliases) when the requested name is the start of
/// the served repository's name up to a `-` boundary (`Qwen3-Coder` → `org/Qwen3-Coder-30B-…`).
fn names_match(requested: &str, served: &str) -> bool {
    let same = |a: &str, b: &str| a.eq_ignore_ascii_case(b);
    if same(requested, served) {
        return true;
    }
    let pinned = |alias: &str, repo: &str| {
        hub::aliases::get(alias).is_some_and(|alias| same(alias.repo, repo))
    };
    if hub::aliases::get(requested).is_some() || hub::aliases::get(served).is_some() {
        return pinned(requested, served) || pinned(served, requested);
    }
    let (r, s) = (requested.to_lowercase(), served.to_lowercase());
    let name = s.rsplit('/').next().unwrap_or(&s);
    name == r
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
        assert!(names_match("qwen3.5", "mlx-community/Qwen3.5-9B-4bit"));
        assert!(names_match("mlx-community/Qwen3.5-9B-4bit", "qwen3.5"));
        assert!(names_match("org/x", "org/x"));
        assert!(names_match("qwen3.6", "mlx-community/Qwen3.6-35B-A3B-4bit"));
        assert!(!names_match(
            "qwen3.5",
            "mlx-community/Qwen3.6-35B-A3B-4bit"
        ));
        assert!(!names_match("qwen3.6", "mlx-community/Qwen3.5-9B-4bit"));
        assert!(!names_match(
            "qwen3.5",
            "org/Qwen3.5-Coder-30B-A3B-Instruct-4bit"
        ));
        assert!(names_match(
            "Qwen3-Coder",
            "org/Qwen3-Coder-30B-A3B-Instruct-4bit"
        ));
    }

    #[test]
    fn the_only_installed_client_is_picked() {
        use Client::*;
        assert_eq!(pick_client(Some(Codex), &[]).unwrap(), Codex);
        assert_eq!(pick_client(None, &[OpenCode]).unwrap(), OpenCode);
        let none = pick_client(None, &[]).unwrap_err();
        assert_eq!(none.code, Code::ClientMissing);
        let many = pick_client(None, &[ClaudeCode, Codex]).unwrap_err();
        assert_eq!(many.code, Code::InvalidArguments);
        assert!(many.message.contains("claude-code, codex"), "{many}");
    }

    #[test]
    fn claude_code_is_refused_below_its_minimum_context() {
        let info = hub::model_info("qwen3", 147_456, 0, 40_960);
        let error = check_min_context(
            Client::ClaudeCode,
            "org/m",
            40_960,
            Some((4_600_000_000, &info, 30 << 30)),
        )
        .unwrap_err();
        assert_eq!(error.code, Code::DoesNotFit);
        assert!(error.message.contains("needs at least 65536"), "{error}");
        assert!(
            error
                .hint
                .unwrap()
                .contains("needs a memory budget of about")
        );
        assert!(check_min_context(Client::ClaudeCode, "org/m", 65_536, None).is_ok());
        assert!(check_min_context(Client::Codex, "org/m", 32_768, None).is_ok());
        assert!(check_min_context(Client::Codex, "org/m", 32_767, None).is_err());
        assert!(check_min_context(Client::OpenCode, "org/m", 49_152, None).is_err());
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

    #[test]
    fn opencode_gets_a_private_server_so_the_launch_config_applies() {
        let a = |v: &[&str]| v.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        let oc = |v: &[&str]| client_command_args(Client::OpenCode, &a(v));
        assert_eq!(oc(&[]), a(&["--standalone"]));
        assert_eq!(oc(&["--continue"]), a(&["--standalone", "--continue"]));
        assert_eq!(
            oc(&["run", "--auto", "x"]),
            a(&["run", "--standalone", "--auto", "x"])
        );
        assert_eq!(oc(&["--standalone"]), a(&["--standalone"]));
        assert_eq!(oc(&["--server", "http://h"]), a(&["--server", "http://h"]));
        assert_eq!(oc(&["auth", "list"]), a(&["auth", "list"]));
        let cc = client_command_args(Client::ClaudeCode, &a(&["-p", "x"]));
        assert_eq!(cc, a(&["-p", "x"]));
    }
}
