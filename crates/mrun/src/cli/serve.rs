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
use crate::runtime::handle::EngineHandle;
use crate::runtime::power::{Action, PowerState};
use mrun_engine::{ModelInfo, Sampling, SyntaxId};
use std::net::{Ipv4Addr, SocketAddr};
use std::path::PathBuf;
use std::sync::Arc;
use tokio::net::TcpListener;
use tokio::sync::watch;

/// Default port of `mrun serve`.
pub const DEFAULT_PORT: u16 = 8484;

/// Flags of `mrun serve`. Every flag except `--record` and `--allow-over` also reads
/// `MRUN_<FLAG>` (004 contract, configuration sources).
#[derive(Debug, Clone, clap::Args)]
pub struct ServeArgs {
    /// Model to serve: alias, Hugging Face repo id (org/name) or local directory.
    #[arg(long, env = "MRUN_MODEL")]
    pub model: Option<String>,
    /// Port on 127.0.0.1.
    #[arg(long, env = "MRUN_PORT", default_value_t = DEFAULT_PORT)]
    pub port: u16,
    /// Default output limit for requests that send none [default: 16384 for Chat
    /// Completions; Responses use the remaining capacity].
    #[arg(long, env = "MRUN_MAX_OUTPUT")]
    pub max_output: Option<u32>,
    /// Requests that may wait while one generates.
    #[arg(long, env = "MRUN_QUEUE", default_value_t = 16)]
    pub queue: usize,
    /// Seconds a request may wait in the queue.
    #[arg(long, env = "MRUN_QUEUE_WAIT_S", default_value_t = 600)]
    pub queue_wait_s: u64,
    /// Fraction of physical memory mrun may plan with.
    #[arg(long, env = "MRUN_MEMORY_FRACTION", default_value_t = 0.75)]
    pub memory_fraction: f64,
    /// Memory kept free for other applications and mrun's host heap, in GiB
    /// [default: physical memory / 6, clamped to 2..=8].
    #[arg(long, env = "MRUN_RESERVE_GIB")]
    pub reserve_gib: Option<u64>,
    /// Allow a budget above the automatic one.
    #[arg(long)]
    pub allow_over: bool,
    /// Disable prefix reuse across requests.
    #[arg(long, env = "MRUN_NO_CACHE")]
    pub no_cache: bool,
    /// Refuse images inside tool results instead of omitting them.
    #[arg(long, env = "MRUN_REJECT_TOOL_IMAGES")]
    pub reject_tool_images: bool,
    /// Record requests and responses as JSON lines into this directory.
    #[arg(long, value_name = "DIR")]
    pub record: Option<PathBuf>,
    /// Run in the background (stop it with `mrun stop`).
    #[arg(long, env = "MRUN_DETACH")]
    pub detach: bool,
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
    Failed(String),
}

/// Builds the engine on a blocking thread after the listener is up.
pub type EngineFactory = Box<dyn FnOnce(&ServeArgs) -> Result<Loaded, String> + Send>;

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
    pub async fn loaded(&mut self) -> Result<Arc<Loaded>, String> {
        let state = self
            .readiness
            .wait_for(|r| !matches!(r, Readiness::Loading))
            .await
            .map_err(|_| "the server stopped".to_string())?
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
    let app = api::router(state);
    let task = tokio::spawn(async move {
        axum::serve(listener, app)
            .with_graceful_shutdown(async move {
                tokio::select! {
                    _ = shutdown_rx => {}
                    _ = terminate() => {}
                }
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
async fn terminate() {
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
    move |info: &ModelInfo| PrefixStore::new(info.identity.clone(), mode)
}

/// The engine factory of this build: the MLX engine with feature `mlx`, otherwise none.
pub fn default_factory() -> EngineFactory {
    Box::new(|_args: &ServeArgs| {
        Err("this build of mrun has no inference engine yet (build with --features mlx)".into())
    })
}

/// `mrun serve --detach`: runs the server as a background process with its log in
/// `MRUN_HOME/server.log`, returns once it serves a loaded model (exit 0), or with the server's
/// own exit code when loading fails (004 FR-009).
fn detach(args: &ServeArgs) -> u8 {
    use crate::cli::stop::{log_file, pid_file};
    use std::time::Duration;
    let home = crate::cli::mrun_home();
    let opened = std::fs::create_dir_all(&home).and_then(|()| {
        std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(log_file())
    });
    let log = match opened {
        Ok(log) => log,
        Err(error) => {
            eprintln!("error: cannot open {}: {error}", log_file().display());
            return 1;
        }
    };
    if crate::cli::server::request(args.port, "GET", "/health").is_ok() {
        eprintln!(
            "error: a server already runs on 127.0.0.1:{}\n  hint: 'mrun ps' shows it; 'mrun stop' stops a detached one",
            args.port
        );
        return 1;
    }
    let child_args: Vec<std::ffi::OsString> = std::env::args_os()
        .skip(1)
        .filter(|a| a != "--detach")
        .collect();
    let mut command = match std::env::current_exe() {
        Ok(exe) => std::process::Command::new(exe),
        Err(error) => {
            eprintln!("error: {error}");
            return 1;
        }
    };
    command
        .args(&child_args)
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
    let mut child = match command.spawn() {
        Ok(child) => child,
        Err(error) => {
            eprintln!("error: cannot start the server: {error}");
            return 1;
        }
    };
    let _ = std::fs::write(pid_file(), child.id().to_string());
    loop {
        if let Ok(Some(status)) = child.try_wait() {
            let _ = std::fs::remove_file(pid_file());
            let tail = std::fs::read_to_string(log_file()).unwrap_or_default();
            let lines: Vec<&str> = tail.lines().collect();
            for line in &lines[lines.len().saturating_sub(5)..] {
                eprintln!("{line}");
            }
            return status.code().unwrap_or(1).clamp(1, 255) as u8;
        }
        if let Ok((200, _)) = crate::cli::server::request(args.port, "GET", "/health") {
            eprintln!(
                "mrun serving at http://127.0.0.1:{} (pid {}, log {})\n  next: mrun launch claude-code   (or codex, opencode); 'mrun stop' stops it",
                args.port,
                child.id(),
                log_file().display()
            );
            return 0;
        }
        std::thread::sleep(Duration::from_millis(200));
    }
}

/// `mrun serve`: runs until SIGTERM or Ctrl-C. Returns the process exit code.
pub fn run(args: ServeArgs, factory: EngineFactory) -> u8 {
    if args.detach {
        return detach(&args);
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
                eprintln!("error: {error}");
                let _ = server.shutdown().await;
                return 1;
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
