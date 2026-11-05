//! `mrun serve`: flags, the engine factory and the HTTP server lifecycle.
//!
//! The listener comes up first; the engine loads on its own thread while `/health` answers 503.
//! The server binds loopback only (002 FR-012): there is no `--host` flag in 0.1.

use crate::api;
use crate::api::decode::DecodeOptions;
use crate::api::record::Recorder;
use crate::api::run::Timing;
use crate::api::status::Tracker;
use crate::cache::{CacheMode, PrefixStore};
use crate::chat::constrain::GrammarEnv;
use crate::chat::template::ChatTemplate;
use crate::chat::tokenizer::Tokenizer;
use crate::errors::{CliError, Code};
use crate::runtime::handle::EngineHandle;
use crate::runtime::power::{Action, PowerState};
use crate::runtime::queue::{DEFAULT_QUEUE, DEFAULT_QUEUE_WAIT_S};
use mrun_engine::{ModelInfo, Sampling, SyntaxId};
use std::net::{Ipv4Addr, SocketAddr};
use std::path::PathBuf;
use std::sync::Arc;
use tokio::net::TcpListener;
use tokio::sync::watch;

/// Default port of `mrun serve`.
pub const DEFAULT_PORT: u16 = 8484;

/// Flags of `mrun serve`. Every flag except `--record` also reads
/// `MRUN_<FLAG>` (004 contract, configuration sources).
#[derive(Debug, Clone, clap::Args)]
pub struct ServeArgs {
    /// Model to serve: alias, Hugging Face repo id (org/name) or local directory [default: this
    /// machine's default alias, or the largest pulled model that fits].
    #[arg(long, env = "MRUN_MODEL")]
    pub model: Option<String>,
    /// Port on 127.0.0.1.
    #[arg(long, env = "MRUN_PORT", default_value_t = DEFAULT_PORT)]
    pub port: u16,
    /// Output limit for requests that send none, on all three APIs; always clamped to the
    /// remaining capacity [default: min(remaining, 32,768)].
    #[arg(long, env = "MRUN_MAX_OUTPUT",
        value_parser = clap::value_parser!(u32).range(1..))]
    pub max_output: Option<u32>,
    /// Requests that may wait while one generates.
    #[arg(long, env = "MRUN_QUEUE", default_value_t = DEFAULT_QUEUE,
        value_parser = clap::builder::RangedU64ValueParser::<usize>::new().range(1..))]
    pub queue: usize,
    /// Seconds a non-streaming request may wait in the queue (below Claude Code's 300 s
    /// watchdog); one whose estimated wait is longer is refused at admission. Streaming
    /// requests have no queue deadline: keepalives hold their clients (D052).
    #[arg(long, env = "MRUN_QUEUE_WAIT_S", default_value_t = DEFAULT_QUEUE_WAIT_S,
        value_parser = clap::value_parser!(u64).range(1..))]
    pub queue_wait_s: u64,
    /// Lower the memory budget to this many GiB [default: the GPU's recommended working set,
    /// at most physical memory − 2 GiB]; it cannot raise it.
    #[arg(long, env = "MRUN_BUDGET_GIB", value_parser = crate::cli::model::parse_budget_gib)]
    pub budget_gib: Option<u64>,
    /// Disable prefix reuse across requests.
    #[arg(long, env = "MRUN_NO_CACHE")]
    pub no_cache: bool,
    /// Refuse images inside tool results instead of omitting them.
    #[arg(long, env = "MRUN_REJECT_TOOL_IMAGES")]
    pub reject_tool_images: bool,
    /// Record raw requests as JSON lines into this directory (for `mrun bench --trace`).
    #[arg(long, value_name = "DIR")]
    pub record: Option<PathBuf>,
    /// Run in the background (stop it with `mrun stop`).
    #[arg(long, env = "MRUN_DETACH")]
    pub detach: bool,
    /// Relay to a remote mrun agent instead of loading a model (remote GPU mode, 013; `mrun
    /// cloud up` starts it). The pod token comes from `MRUN_RELAY_TOKEN`.
    #[arg(long, value_name = "AGENT_URL")]
    pub relay: Option<String>,
    /// The relayed pod's id (status only).
    #[arg(long, hide = true, requires = "relay")]
    pub relay_pod: Option<String>,
    /// The relayed pod's GPU (status only).
    #[arg(long, hide = true, requires = "relay")]
    pub relay_gpu: Option<String>,
    /// The relayed pod's price in $/h (status only).
    #[arg(long, hide = true, requires = "relay")]
    pub relay_cost: Option<f64>,
    /// When the relayed pod was created (Unix seconds; status only: uptime and spend).
    #[arg(long, hide = true, requires = "relay")]
    pub relay_since: Option<u64>,
}

/// A loaded model, shared by every request.
#[derive(Debug)]
pub struct Loaded {
    /// Model id reported in responses.
    pub model_id: String,
    pub handle: EngineHandle,
    pub tokenizer: Arc<Tokenizer>,
    pub template: Arc<ChatTemplate>,
    /// The family's chat syntax (tool calls, reasoning, message framing).
    pub syntax: SyntaxId,
    /// The family has a thinking mode that requests can switch on.
    pub supports_thinking: bool,
    /// Sampling defaults from the model's generation config.
    pub defaults: Sampling,
    /// Constrained decoding over the model's vocabulary; `None` disables it.
    pub grammar: Option<Arc<GrammarEnv>>,
}

/// Engine readiness, observed by `/health` and the API routes.
#[derive(Debug, Clone)]
pub enum Readiness {
    Loading,
    Ready(Arc<Loaded>),
    /// Loading failed; the error carries the exit code.
    Failed(CliError),
}

/// Builds the engine on a blocking thread after the listener is up.
pub type EngineFactory = Box<dyn FnOnce(&ServeArgs) -> Result<Loaded, CliError> + Send>;

/// State shared by the router.
#[derive(Debug, Clone)]
pub struct ServerState {
    pub addr: SocketAddr,
    pub args: Arc<ServeArgs>,
    pub readiness: watch::Receiver<Readiness>,
    pub timing: Timing,
    pub tracker: Arc<Tracker>,
    /// `--record`: request bodies written for trace replay.
    pub recorder: Option<Arc<Recorder>>,
    /// The idle-sleep assertion's state, for `/mrun/status`.
    pub power: Arc<std::sync::Mutex<PowerState>>,
}

impl ServerState {
    pub fn decode_options(&self) -> DecodeOptions {
        DecodeOptions {
            reject_tool_images: self.args.reject_tool_images,
        }
    }
}

/// A started server: its address, readiness and a way to stop it.
#[derive(Debug)]
pub struct RunningServer {
    pub addr: SocketAddr,
    pub readiness: watch::Receiver<Readiness>,
    shutdown: Option<tokio::sync::oneshot::Sender<()>>,
    task: tokio::task::JoinHandle<std::io::Result<()>>,
}

impl RunningServer {
    /// Waits until the engine is ready or failed to load.
    pub async fn loaded(&mut self) -> Result<Arc<Loaded>, CliError> {
        let state = self
            .readiness
            .wait_for(|r| !matches!(r, Readiness::Loading))
            .await
            .map_err(|_| CliError::new(Code::Internal, "the server stopped"))?
            .clone();
        match state {
            Readiness::Ready(loaded) => Ok(loaded),
            Readiness::Failed(error) => Err(error),
            Readiness::Loading => unreachable!(),
        }
    }

    /// Stops accepting connections and waits for open requests to finish.
    pub async fn shutdown(mut self) -> std::io::Result<()> {
        if let Some(tx) = self.shutdown.take() {
            let _ = tx.send(());
        }
        self.task.await.unwrap_or(Ok(()))
    }

    /// Runs until the server stops on its own (it never does unless the listener fails).
    pub async fn wait(self) -> std::io::Result<()> {
        self.task.await.unwrap_or(Ok(()))
    }
}

/// Binds `127.0.0.1:port` (port 0 picks a free one), starts serving and loads the engine.
pub async fn start(args: ServeArgs, factory: EngineFactory) -> std::io::Result<RunningServer> {
    let timing = Timing::from_args(&args);
    start_with(args, factory, timing).await
}

/// [`start`] with explicit keepalive and queue timing (tests shorten them).
pub async fn start_with(
    args: ServeArgs,
    factory: EngineFactory,
    timing: Timing,
) -> std::io::Result<RunningServer> {
    let recorder = match &args.record {
        Some(dir) => Some(Arc::new(Recorder::create(dir)?)),
        None => None,
    };
    let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, args.port)).await?;
    let addr = listener.local_addr()?;
    let args = Arc::new(args);
    let (ready_tx, readiness) = watch::channel(Readiness::Loading);

    let load_args = args.clone();
    tokio::task::spawn_blocking(move || {
        let state = match factory(&load_args) {
            Ok(loaded) => {
                loaded.handle.queue().set_capacity(load_args.queue);
                Readiness::Ready(Arc::new(loaded))
            }
            Err(error) => Readiness::Failed(error),
        };
        let _ = ready_tx.send(state);
    });

    let state = ServerState {
        addr,
        args,
        readiness: readiness.clone(),
        timing,
        tracker: Arc::new(Tracker::default()),
        recorder,
        power: Arc::new(std::sync::Mutex::new(PowerState::default())),
    };
    tokio::spawn(keep_awake(
        readiness.clone(),
        state.power.clone(),
        crate::runtime::power::platform_assertion(),
    ));
    let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel::<()>();
    let tracker = state.tracker.clone();
    let app = api::router(state);
    let task = tokio::spawn(async move {
        axum::serve(listener, app)
            .with_graceful_shutdown(async move {
                tokio::select! {
                    _ = shutdown_rx => {}
                    _ = terminate() => {}
                }
                // D052: refuse new work, cancel queued requests, give running ones 5 s.
                tracker.drain(api::status::SHUTDOWN_GRACE).await;
            })
            .await
    });
    Ok(RunningServer {
        addr,
        readiness,
        shutdown: Some(shutdown_tx),
        task,
    })
}

/// Holds the idle-sleep assertion while requests are queued or running (008 FR-009).
async fn keep_awake(
    readiness: watch::Receiver<Readiness>,
    power: Arc<std::sync::Mutex<PowerState>>,
    mut assertion: Box<dyn crate::runtime::power::Assertion>,
) {
    let mut tick = tokio::time::interval(std::time::Duration::from_secs(1));
    loop {
        tick.tick().await;
        let busy = match &*readiness.borrow() {
            Readiness::Ready(loaded) => {
                let queue = loaded.handle.queue();
                queue.running() || queue.waiting() > 0
            }
            _ => false,
        };
        let action = power
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .observe(
                busy,
                std::time::Instant::now(),
                std::time::SystemTime::now(),
            );
        match action {
            Some(Action::Acquire) => {
                assertion.acquire();
            }
            Some(Action::Release) => assertion.release(),
            None => {}
        }
    }
}

/// Resolves on SIGTERM or Ctrl-C.
pub async fn terminate() {
    let ctrl_c = async {
        let _ = tokio::signal::ctrl_c().await;
    };
    #[cfg(unix)]
    let term = async {
        match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
            Ok(mut signal) => {
                signal.recv().await;
            }
            Err(_) => std::future::pending().await,
        }
    };
    #[cfg(not(unix))]
    let term = std::future::pending::<()>();
    tokio::select! {
        _ = ctrl_c => {}
        _ = term => {}
    }
}

/// The reuse policy of `serve`: the prefix cache, keyed by the model identity (`--no-cache`
/// turns reuse off).
pub fn cache_policy(args: &ServeArgs) -> impl FnOnce(&ModelInfo) -> PrefixStore + Send + 'static {
    let mode = if args.no_cache {
        CacheMode::Off
    } else {
        CacheMode::Cached
    };
    move |_: &ModelInfo| PrefixStore::new(mode)
}

/// The model `serve` loads: `--model`, else this machine's default.
fn model_name(args: &ServeArgs) -> Result<String, CliError> {
    match &args.model {
        Some(model) => Ok(model.clone()),
        None => {
            crate::hub::default_model(&crate::hub::Ctx::from_env().with_budget(args.budget_gib))
        }
    }
}

/// The engine factory of this build: the nuvec engine with feature `engine`, otherwise none.
#[cfg(feature = "engine")]
pub fn default_factory() -> EngineFactory {
    Box::new(|args: &ServeArgs| {
        let name = model_name(args)?;
        let model = crate::cli::model::open(&name)?;
        crate::cli::model::preflight(&model)?;
        let overrides = crate::cli::model::overrides(args.budget_gib);
        let handle = EngineHandle::spawn_with(
            crate::cli::model::engine_factory(&model, overrides),
            model.tokenizer.clone(),
            crate::runtime::handle::DEFAULT_STALL_TIMEOUT,
            cache_policy(args),
        )?;
        let grammar = GrammarEnv::new(&model.tokenizer, &handle.info().eos_tokens)
            .ok()
            .map(Arc::new);
        Ok(Loaded {
            model_id: name,
            syntax: model.syntax(),
            supports_thinking: model.supports_thinking(),
            defaults: model.defaults,
            grammar,
            handle,
            tokenizer: model.tokenizer,
            template: model.template,
        })
    })
}

/// The engine factory of this build: the nuvec engine with feature `engine`, otherwise none.
#[cfg(not(feature = "engine"))]
pub fn default_factory() -> EngineFactory {
    Box::new(|_args: &ServeArgs| {
        Err(
            CliError::new(Code::Internal, "this build of mrun has no inference engine")
                .hint("build with the default features (the `engine` feature)"),
        )
    })
}

/// `mrun serve --detach`: the same command line without `--detach`, in the background.
fn detach(args: &ServeArgs) -> Result<(), StartError> {
    let given: Vec<std::ffi::OsString> = std::env::args_os()
        .skip(1)
        .filter(|a| a != "--detach")
        .collect();
    // The child uses this process's resolution of the config file, not a fresh read (017 R5): the
    // file's values go first as flags (the user's own flags come later and win), and the child
    // reads no file.
    let settings = crate::config::settings(&crate::cli::app::cli_command());
    let child_args = with_file_args(&given, &crate::config::file_args("serve", &settings));
    let model = match &args.relay {
        Some(_) => "the remote model".to_string(),
        None => model_name(args)?,
    };
    start_detached(args.port, &child_args, &model, &[("MRUN_CONFIG", "")])
}

/// `given` (a `serve` command line) with `file_args` (`--name=value`) inserted right after `serve`,
/// except those whose flag the user typed: a flag beats the file, and clap refuses a flag twice.
fn with_file_args(given: &[std::ffi::OsString], file_args: &[String]) -> Vec<std::ffi::OsString> {
    let typed = |flag: &str| {
        given.iter().any(|a| {
            let a = a.to_string_lossy();
            a == flag || a.starts_with(&format!("{flag}="))
        })
    };
    let at = given.iter().position(|a| a == "serve").map_or(0, |i| i + 1);
    let mut out = given[..at].to_vec();
    out.extend(
        file_args
            .iter()
            .filter(|f| !typed(f.split_once('=').map_or(f.as_str(), |(flag, _)| flag)))
            .map(std::ffi::OsString::from),
    );
    out.extend_from_slice(&given[at..]);
    out
}

/// Runs `mrun <child_args>` (a `serve` command) as a background process with its log in
/// `MRUN_HOME/server.log` and returns once it serves a loaded model, with the server's own exit
/// code when loading fails (004 FR-009), or E010 after [`server::LOAD_TIMEOUT`].
pub fn start_detached(
    port: u16,
    child_args: &[std::ffi::OsString],
    model: &str,
    env: &[(&str, &str)],
) -> Result<(), StartError> {
    use crate::cli::{logs, server, stop::pid_file};
    if server::health(port) != server::Health::Down {
        return Err(CliError::new(
            Code::InvalidArguments,
            format!("a server already runs on 127.0.0.1:{port}"),
        )
        .hint("'mrun ps' shows it; 'mrun stop' stops a detached one; --port picks another")
        .into());
    }
    let log = logs::open_for_server().map_err(|error| {
        CliError::new(
            Code::Internal,
            format!("cannot open {}: {error}", logs::log_file().display()),
        )
        .hint("check that MRUN_HOME is writable")
    })?;
    // Only this start's lines are shown when it fails; the log keeps earlier runs.
    let log_start = log.metadata().map_or(0, |meta| meta.len());
    let exe = std::env::current_exe().map_err(|e| CliError::new(Code::Internal, e.to_string()))?;
    let mut command = std::process::Command::new(exe);
    command
        .args(child_args)
        .envs(env.iter().copied())
        .env_remove("MRUN_DETACH")
        .stdin(std::process::Stdio::null())
        .stdout(
            log.try_clone()
                .map_or_else(|_| std::process::Stdio::null(), Into::into),
        )
        .stderr(log);
    #[cfg(unix)]
    {
        // Its own process group: Ctrl-C in this terminal does not reach the server.
        use std::os::unix::process::CommandExt;
        command.process_group(0);
    }
    let mut child = command.spawn().map_err(|error| {
        CliError::new(Code::Internal, format!("cannot start the server: {error}"))
    })?;
    let pid = pid_file(port);
    let _ = std::fs::write(&pid, child.id().to_string());
    let failed = |code: u8| {
        let _ = std::fs::remove_file(&pid);
        eprintln!("{}", logs::tail_since(log_start, 5));
        StartError::Exit(code)
    };
    match server::wait_until_loaded(port, &loading_label(model), Some(&mut child)) {
        server::LoadEnd::Ready => {
            eprintln!(
                "mrun serving {model} at http://127.0.0.1:{port} (pid {}, log {})\n  next: mrun launch claude-code   (or codex, opencode); 'mrun stop' stops it",
                child.id(),
                logs::log_file().display()
            );
            Ok(())
        }
        server::LoadEnd::Exited(code) => Err(failed(code)),
        server::LoadEnd::Failed(error) => {
            eprintln!("{error}");
            Err(failed(1))
        }
        server::LoadEnd::TimedOut => {
            eprintln!("{}", logs::tail_since(log_start, 5));
            Err(CliError::new(
                Code::ServerUnreachable,
                format!(
                    "the server did not finish loading within {} minutes",
                    server::LOAD_TIMEOUT.as_secs() / 60
                ),
            )
            .hint("'mrun logs -f' shows what it does; 'mrun stop' stops it")
            .into())
        }
    }
}

/// Why a background server did not come up: an error to report, or the server's own exit code
/// (it reported in its log, whose tail was printed).
#[derive(Debug)]
pub enum StartError {
    Error(CliError),
    Exit(u8),
}

impl StartError {
    /// Reports the error (if not reported yet) and returns the exit code.
    pub fn report(self) -> u8 {
        match self {
            Self::Error(error) => error.report(),
            Self::Exit(code) => code,
        }
    }
}

impl From<CliError> for StartError {
    fn from(error: CliError) -> Self {
        Self::Error(error)
    }
}

/// `loading <model> (<size>)` for a pulled model, `loading <model>` otherwise.
pub fn loading_label(model: &str) -> String {
    let ctx = crate::hub::Ctx::from_env();
    match crate::hub::find_manifest(&ctx.paths, model) {
        Ok(manifest) => format!(
            "loading {model} ({})",
            crate::hub::human_bytes(manifest.size())
        ),
        Err(_) => format!("loading {model}"),
    }
}

/// `mrun serve`: runs until SIGTERM or Ctrl-C. Returns the process exit code.
pub fn run(args: ServeArgs, factory: EngineFactory) -> u8 {
    if args.detach {
        return detach(&args).map_or_else(StartError::report, |()| 0);
    }
    if let Some(agent_url) = &args.relay {
        let pod = crate::api::relay::PodMeta {
            id: args.relay_pod.clone(),
            gpu: args.relay_gpu.clone(),
            cost_per_hr: args.relay_cost,
            since_unix: args.relay_since,
        };
        return crate::api::relay::run(args.port, agent_url, pod, args.max_output);
    }
    let runtime = match tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
    {
        Ok(runtime) => runtime,
        Err(error) => {
            eprintln!("error: cannot start the async runtime: {error}");
            return 1;
        }
    };
    runtime.block_on(async move {
        let port = args.port;
        let record = args.record.clone();
        let mut server = match start(args, factory).await {
            Ok(server) => server,
            Err(error) => {
                eprintln!(
                    "error: cannot listen on 127.0.0.1:{port}: {error}\n  hint: is another server running? Try --port or 'mrun ps'."
                );
                return 1;
            }
        };
        let url = format!("http://{}", server.addr);
        if let Some(dir) = &record {
            eprintln!(
                "warning: recording every request body to {} (--record); prompts are written to disk",
                dir.display()
            );
        }
        crate::cli::info(&format!("mrun listening on {url} (loading model)"));
        match server.loaded().await {
            Ok(loaded) => {
                let syntax = crate::chat::syntax::registry::ChatSyntax::get(loaded.syntax).name;
                eprintln!(
                    "mrun serving {} ({}, {syntax}) at {url}",
                    loaded.model_id,
                    loaded.handle.info().family
                );
                eprintln!("  next: mrun launch claude-code   (or codex, opencode)");
            }
            Err(error) => {
                let code = error.report();
                let _ = server.shutdown().await;
                return code;
            }
        }
        match server.wait().await {
            Ok(()) => 0,
            Err(error) => {
                eprintln!("error: server failed: {error}");
                1
            }
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_detached_child_gets_the_files_values_except_typed_flags() {
        let os = |v: &[&str]| v.iter().map(std::ffi::OsString::from).collect::<Vec<_>>();
        let given = os(&["-q", "serve", "--port", "9100"]);
        let child = with_file_args(&given, &["--port=9000".into(), "--queue=4".into()]);
        // The user's --port beats the file's (and clap refuses a flag twice).
        assert_eq!(child, os(&["-q", "serve", "--queue=4", "--port", "9100"]));
        let given = os(&["serve", "--no-cache=false"]);
        let child = with_file_args(&given, &["--no-cache=true".into()]);
        assert_eq!(child, os(&["serve", "--no-cache=false"]));
    }
    use clap::Parser;

    #[derive(Parser)]
    struct Wrap {
        #[command(flatten)]
        serve: ServeArgs,
    }

    #[test]
    fn queue_flags_default_and_refuse_zero() {
        let args = Wrap::try_parse_from(["mrun"]).unwrap().serve;
        assert_eq!(
            (args.queue, args.queue_wait_s),
            (DEFAULT_QUEUE, DEFAULT_QUEUE_WAIT_S)
        );
        let args = Wrap::try_parse_from(["mrun", "--queue", "1", "--queue-wait-s", "1"]);
        assert_eq!(args.unwrap().serve.queue, 1);
        for flag in ["--queue", "--queue-wait-s", "--max-output"] {
            assert!(Wrap::try_parse_from(["mrun", flag, "0"]).is_err(), "{flag}");
        }
    }
}
