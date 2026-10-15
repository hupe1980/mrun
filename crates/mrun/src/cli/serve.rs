//! `mrun serve`: flags, the engine factory and the HTTP server lifecycle.
//!
//! The listener comes up first; the engine loads on its own thread while `/health` answers 503.
//! The server binds loopback only (002 FR-012): there is no `--host` flag in 0.1.

use crate::api;
use crate::api::decode::DecodeOptions;
use crate::api::record::Recorder;
use crate::api::run::Timing;
use crate::api::status::Tracker;
use crate::chat::constrain::GrammarEnv;
use crate::chat::template::ChatTemplate;
use crate::chat::tokenizer::Tokenizer;
use crate::runtime::handle::EngineHandle;
use mrun_engine::{Sampling, SyntaxId};
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
    };
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

/// The engine factory of this build: the MLX engine with feature `mlx`, otherwise none.
pub fn default_factory() -> EngineFactory {
    Box::new(|_args: &ServeArgs| {
        Err("this build of mrun has no inference engine yet (build with --features mlx)".into())
    })
}

/// `mrun serve`: runs until SIGTERM or Ctrl-C. Returns the process exit code.
pub fn run(args: ServeArgs, factory: EngineFactory) -> u8 {
    if args.detach {
        eprintln!(
            "error: --detach is not available yet\n  hint: run 'mrun serve' in its own terminal"
        );
        return 1;
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
        eprintln!("mrun listening on {url} (loading model)");
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
