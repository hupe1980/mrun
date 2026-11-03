//! `mrun agent`: the process in a RunPod pod (013, D064). It supervises vLLM on the pod's
//! loopback, requires the per-pod bearer token on every route but `/health` (FR-008), forwards
//! the three APIs, `count_tokens` and `/v1/models` with keepalives and abort on disconnect
//! (FR-009), reports its phase on `/mrun/status`, and stops its own pod when idle or past the
//! maximum runtime (FR-010). vLLM's full output and the agent's own lines go to a log file that
//! `GET /mrun/logs` serves by offset or follows (FR-015).
//!
//! It listens on all interfaces because RunPod's proxy reaches it from outside the container;
//! the token check is the boundary. vLLM itself listens on 127.0.0.1 with a key only the agent
//! knows.

pub mod lifecycle;
pub mod logfile;
pub mod stub;
pub mod vllm;

use crate::api::MAX_BODY_BYTES;
use crate::api::model::{ApiError, ErrorKind};
use crate::remote::forward::{self, Options, Upstream};
use crate::remote::{self, api_of};
use axum::Router;
use axum::body::Bytes;
use axum::extract::{DefaultBodyLimit, Request, State};
use axum::http::Uri;
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use lifecycle::{Reason, Stopper};
use serde_json::{Value, json};
use std::net::{Ipv4Addr, SocketAddr};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::{Duration, Instant};

/// Port vLLM listens on (127.0.0.1 only).
pub const VLLM_PORT: u16 = 8001;
/// How long vLLM may take to answer `/v1/models` after start (download plus load).
pub const LOAD_TIMEOUT: Duration = Duration::from_secs(45 * 60);
/// Sequences vLLM runs at once by default: one developer's agent with its parallel subagents
/// (D045 lanes are 2–4), well under a hybrid model's Mamba cache blocks.
pub const DEFAULT_MAX_NUM_SEQS: u32 = 16;
/// Restarts of a vLLM that exits after it served (a crash hours in should not end the session).
pub const MAX_RESTARTS: u32 = 1;
/// Silence after which a keepalive is sent.
pub const KEEPALIVE: Duration = Duration::from_secs(10);

/// Flags of `mrun agent`; in a pod they come from the environment `mrun cloud up` sets. The
/// pod token is read from `MRUN_POD_TOKEN` only (never a flag: flags show in `ps`).
#[derive(Debug, Clone, clap::Args)]
pub struct AgentArgs {
    /// Port on all interfaces (the pod's exposed HTTP port).
    #[arg(long, env = "MRUN_AGENT_PORT", default_value_t = remote::AGENT_PORT)]
    pub port: u16,
    /// Hugging Face repo vLLM serves.
    #[arg(long, env = "MRUN_MODEL")]
    pub model: String,
    /// Context length in tokens.
    #[arg(long, env = "MRUN_MAX_MODEL_LEN", default_value_t = 65_536)]
    pub max_model_len: u32,
    /// Most sequences vLLM runs at once (`--max-num-seqs`). vLLM's default (1024) exceeds the
    /// Mamba cache blocks of hybrid models on one GPU and fails CUDA graph capture (research R14).
    #[arg(long, env = "MRUN_MAX_NUM_SEQS", default_value_t = DEFAULT_MAX_NUM_SEQS)]
    pub max_num_seqs: u32,
    /// vLLM tool-call parser.
    #[arg(long, env = "MRUN_TOOL_PARSER", default_value = "hermes")]
    pub tool_parser: String,
    /// vLLM reasoning parser (none when empty).
    #[arg(long, env = "MRUN_REASONING_PARSER")]
    pub reasoning_parser: Option<String>,
    /// More vLLM flags, a JSON array of strings (the model's own, from the cloud catalog).
    #[arg(long, env = "MRUN_VLLM_ARGS")]
    pub vllm_args: Option<String>,
    /// Stop the pod after this long without a request.
    #[arg(long, env = "MRUN_IDLE_STOP", default_value = "30m",
        value_parser = remote::parse_duration)]
    pub idle_stop: Duration,
    /// Stop the pod after this long regardless.
    #[arg(long, env = "MRUN_MAX_RUNTIME", default_value = "8h",
        value_parser = remote::parse_duration)]
    pub max_runtime: Duration,
    /// Stop the pod this long after the model failed for good, once its log can be fetched
    /// (0: only the idle stop applies, e.g. to keep a failed pod for inspection).
    #[arg(long, env = "MRUN_FAILED_STOP", default_value = "5m",
        value_parser = remote::parse_duration)]
    pub failed_stop: Duration,
    /// The pod's price, for the spend estimate in status.
    #[arg(long, env = "MRUN_COST_PER_HR", default_value_t = 0.0)]
    pub cost_per_hr: f64,
    /// The pod's GPU, for status.
    #[arg(long, env = "MRUN_GPU")]
    pub gpu: Option<String>,
    /// The log file (vLLM's full output and the agent's lines; rotates at 50 MB to `<file>.1`).
    #[arg(long, env = "MRUN_AGENT_LOG", default_value = logfile::DEFAULT_PATH)]
    pub log_file: PathBuf,
    /// Serve a built-in fake vLLM instead of starting vLLM (tests, CPU pods).
    #[arg(long, env = "MRUN_AGENT_STUB", hide = true)]
    pub stub_upstream: bool,
}

/// What the agent forwards to.
#[derive(Clone, Debug, PartialEq)]
pub enum Backend {
    Vllm { program: String, port: u16 },
    Stub,
}

/// The agent's configuration.
#[derive(Clone, Debug)]
pub struct AgentConfig {
    pub bind: SocketAddr,
    pub token: String,
    pub model: String,
    pub max_model_len: u32,
    pub max_num_seqs: u32,
    pub tool_parser: String,
    pub reasoning_parser: Option<String>,
    pub vllm_args: Vec<String>,
    pub idle_stop: Duration,
    pub max_runtime: Duration,
    /// After a failure for good, the pod stops this long later (zero: never early).
    pub failed_stop: Duration,
    pub grace: Duration,
    pub cost_per_hr: f64,
    pub gpu: Option<String>,
    pub keepalive: Duration,
    pub backend: Backend,
    pub stopper: Stopper,
    /// How often the cost timers are checked.
    pub tick: Duration,
    pub load_timeout: Duration,
    /// The log file (`None`: no file, `/mrun/logs` is empty).
    pub log_path: Option<PathBuf>,
    /// Size at which the log file rotates.
    pub log_max_bytes: u64,
}

impl AgentConfig {
    /// From the flags and the environment (`MRUN_POD_TOKEN`, `RUNPOD_POD_ID`, `RUNPOD_API_KEY`).
    pub fn from_args(args: AgentArgs) -> Result<Self, String> {
        let token = std::env::var("MRUN_POD_TOKEN").unwrap_or_default();
        if token.len() < 32 {
            return Err("MRUN_POD_TOKEN is missing or shorter than 32 characters".into());
        }
        let vllm_args = parse_vllm_args(args.vllm_args.as_deref())?;
        Ok(Self {
            bind: SocketAddr::from((Ipv4Addr::UNSPECIFIED, args.port)),
            token,
            model: args.model,
            max_model_len: args.max_model_len,
            max_num_seqs: args.max_num_seqs.max(1),
            tool_parser: args.tool_parser,
            reasoning_parser: args.reasoning_parser.filter(|p| !p.trim().is_empty()),
            vllm_args,
            idle_stop: args.idle_stop,
            max_runtime: args.max_runtime,
            failed_stop: args.failed_stop,
            grace: lifecycle::GRACE,
            cost_per_hr: args.cost_per_hr,
            gpu: args.gpu.filter(|g| !g.is_empty()),
            keepalive: KEEPALIVE,
            backend: if args.stub_upstream {
                Backend::Stub
            } else {
                Backend::Vllm {
                    program: "vllm".into(),
                    port: VLLM_PORT,
                }
            },
            stopper: Stopper::from_env(),
            tick: Duration::from_secs(5),
            load_timeout: LOAD_TIMEOUT,
            log_path: Some(args.log_file),
            log_max_bytes: logfile::MAX_BYTES,
        })
    }
}

/// `MRUN_VLLM_ARGS`: a JSON array of strings (flags keep their spaces and quotes intact).
fn parse_vllm_args(json: Option<&str>) -> Result<Vec<String>, String> {
    match json.map(str::trim) {
        None | Some("") => Ok(vec![]),
        Some(json) => serde_json::from_str(json)
            .map_err(|e| format!("MRUN_VLLM_ARGS is not a JSON array of strings: {e}")),
    }
}

impl AgentConfig {
    /// How vLLM is started for this configuration.
    pub fn vllm_spec(&self, program: String, port: u16, api_key: String) -> vllm::VllmSpec {
        vllm::VllmSpec {
            program,
            model: self.model.clone(),
            port,
            api_key,
            max_model_len: self.max_model_len,
            max_num_seqs: self.max_num_seqs,
            tool_parser: self.tool_parser.clone(),
            reasoning_parser: self.reasoning_parser.clone(),
            extra_args: self.vllm_args.clone(),
        }
    }
}

/// Where the agent is (data model: agent session).
#[derive(Clone, Debug, PartialEq)]
pub enum Phase {
    Starting,
    Loading,
    Ready,
    /// vLLM failed; the last lines of its output.
    Failed(Vec<String>),
    Stopping(Reason),
}

impl Phase {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Starting => "starting",
            Self::Loading => "loading",
            Self::Ready => "ready",
            Self::Failed(_) => "failed",
            Self::Stopping(_) => "stopping",
        }
    }
}

/// Running requests and the time of the last start or end.
#[derive(Debug)]
pub struct Activity {
    pub active: AtomicUsize,
    last: Mutex<Instant>,
}

/// Counts one forwarded request while it runs.
struct ActiveGuard(Arc<Shared>);

impl ActiveGuard {
    fn new(shared: Arc<Shared>) -> Self {
        shared.activity.active.fetch_add(1, Ordering::SeqCst);
        shared.touch();
        Self(shared)
    }
}

impl Drop for ActiveGuard {
    fn drop(&mut self) {
        self.0.touch();
        self.0.activity.active.fetch_sub(1, Ordering::SeqCst);
    }
}

/// State shared by the routes, the supervisor and the timers.
#[derive(Debug)]
pub struct Shared {
    pub config: AgentConfig,
    phase: Mutex<Phase>,
    upstream: Mutex<Option<Upstream>>,
    pub activity: Activity,
    pub started: Instant,
    pub requests_total: AtomicU64,
    pub weights_bytes: AtomicU64,
    max_model_len: AtomicU32,
    backend_version: Mutex<Option<String>>,
    /// The stub's counters (stub backend only).
    pub stub: Mutex<Option<Arc<stub::StubStats>>>,
    /// The full log (`/mrun/logs`).
    pub log: Arc<logfile::LogFile>,
    /// The model served at least once (a later failure is a stop, not a failed start).
    was_ready: AtomicBool,
    /// When the agent failed for good.
    failed_at: Mutex<Option<Instant>>,
}

impl Shared {
    fn new(config: AgentConfig) -> Self {
        let now = Instant::now();
        let log = match &config.log_path {
            Some(path) => {
                let (log, error) = logfile::LogFile::create(path, config.log_max_bytes);
                if let Some(error) = error {
                    eprintln!(
                        "mrun agent: warning: no log file at {}: {error}",
                        path.display()
                    );
                }
                log
            }
            None => logfile::LogFile::disabled(),
        };
        Self {
            log: Arc::new(log),
            max_model_len: AtomicU32::new(config.max_model_len),
            config,
            phase: Mutex::new(Phase::Starting),
            upstream: Mutex::new(None),
            activity: Activity {
                active: AtomicUsize::new(0),
                last: Mutex::new(now),
            },
            started: now,
            requests_total: AtomicU64::new(0),
            weights_bytes: AtomicU64::new(0),
            backend_version: Mutex::new(None),
            stub: Mutex::new(None),
            was_ready: AtomicBool::new(false),
            failed_at: Mutex::new(None),
        }
    }

    /// An agent line: to stderr (`mrun agent: …`) and to the log file (`[agent] …`).
    pub fn note(&self, message: &str) {
        eprintln!("mrun agent: {message}");
        self.log.append(&format!("[agent] {message}"));
    }

    pub fn phase(&self) -> Phase {
        self.phase
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }

    pub fn set_phase(&self, phase: Phase) {
        let mut current = self.phase.lock().unwrap_or_else(PoisonError::into_inner);
        // Stopping is final.
        if !matches!(*current, Phase::Stopping(_)) {
            if current.as_str() != phase.as_str() {
                self.log
                    .append(&format!("[agent] phase {}", phase.as_str()));
                // The idle period starts when the model can serve (or has failed), not while the
                // weights download and load: that took 6–8 of a 30-minute idle stop (2026-10-20).
                if matches!(phase, Phase::Ready | Phase::Failed(_)) {
                    self.touch();
                }
                match phase {
                    Phase::Ready => self.was_ready.store(true, Ordering::SeqCst),
                    Phase::Failed(_) => {
                        *self
                            .failed_at
                            .lock()
                            .unwrap_or_else(PoisonError::into_inner) = Some(Instant::now());
                    }
                    _ => {}
                }
            }
            *current = phase;
        }
    }

    fn touch(&self) {
        *self
            .activity
            .last
            .lock()
            .unwrap_or_else(PoisonError::into_inner) = Instant::now();
    }

    /// Idle time since the model became ready (or failed) or the last request ended; `None`
    /// while a request runs or the model is still starting or loading (the maximum runtime
    /// bounds those).
    /// How long ago the agent failed for good, if it did.
    pub fn failed_for(&self, now: Instant) -> Option<Duration> {
        self.failed_at
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .map(|at| now.saturating_duration_since(at))
    }

    /// What a client is told while the agent has failed: a failed start, or a model that served
    /// and then stopped (vLLM exited hours in; not a start problem).
    pub fn failure(&self) -> Option<String> {
        let Phase::Failed(tail) = self.phase() else {
            return None;
        };
        let last = tail.last().cloned().unwrap_or_default();
        Some(if self.was_ready.load(Ordering::SeqCst) {
            format!("the remote model stopped: {last}")
        } else {
            format!("the remote model failed to start: {last}")
        })
    }

    pub fn idle_for(&self, now: Instant) -> Option<Duration> {
        if self.activity.active.load(Ordering::SeqCst) > 0
            || matches!(self.phase(), Phase::Starting | Phase::Loading)
        {
            return None;
        }
        let last = *self
            .activity
            .last
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        Some(now.saturating_duration_since(last))
    }

    fn upstream(&self) -> Option<Upstream> {
        self.upstream
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }

    fn backend_label(&self) -> Option<String> {
        let version = self
            .backend_version
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()?;
        Some(match self.config.backend {
            Backend::Stub => format!("stub {version}"),
            Backend::Vllm { .. } => format!("vllm {version}"),
        })
    }

    /// The `/mrun/status` body (contracts/agent-http.md).
    pub fn status_json(&self) -> Value {
        let now = Instant::now();
        let phase = self.phase();
        let uptime = now.duration_since(self.started).as_secs();
        let spend = self.config.cost_per_hr * uptime as f64 / 3600.0;
        json!({
            "schema_version": 1,
            "role": "agent",
            "version": env!("CARGO_PKG_VERSION"),
            "backend": self.backend_label(),
            "phase": phase.as_str(),
            "reason": match &phase { Phase::Stopping(r) => Value::from(r.as_str()), _ => Value::Null },
            "model": self.config.model,
            "max_model_len": self.max_model_len.load(Ordering::Relaxed),
            "weights_bytes": self.weights_bytes.load(Ordering::Relaxed),
            "log_tail": match &phase { Phase::Failed(tail) => json!(tail), _ => Value::Null },
            "error": self.failure(),
            "uptime_s": uptime,
            "idle_s": self.idle_for(now).map(|d| d.as_secs()),
            "idle_stop_s": self.config.idle_stop.as_secs(),
            "max_runtime_s": self.config.max_runtime.as_secs(),
            "active": self.activity.active.load(Ordering::SeqCst),
            "requests_total": self.requests_total.load(Ordering::Relaxed),
            "pod_id": self.config.stopper.pod_id(),
            "gpu": self.config.gpu,
            "cost_per_hr": self.config.cost_per_hr,
            "spend_usd": (spend * 1000.0).round() / 1000.0,
        })
    }
}

/// A started agent.
#[derive(Debug)]
pub struct RunningAgent {
    pub addr: SocketAddr,
    pub shared: Arc<Shared>,
    shutdown: Option<tokio::sync::oneshot::Sender<()>>,
    task: tokio::task::JoinHandle<std::io::Result<()>>,
}

impl RunningAgent {
    pub async fn shutdown(mut self) {
        if let Some(tx) = self.shutdown.take() {
            let _ = tx.send(());
        }
        let _ = self.task.await;
    }

    pub async fn wait(self) -> std::io::Result<()> {
        self.task.await.unwrap_or(Ok(()))
    }
}

/// Binds, starts the upstream (vLLM or the stub), the timers and the server.
pub async fn start(config: AgentConfig) -> std::io::Result<RunningAgent> {
    let listener = tokio::net::TcpListener::bind(config.bind).await?;
    let addr = listener.local_addr()?;
    let shared = Arc::new(Shared::new(config));
    shared.log.append(&format!(
        "[agent] mrun agent {} on {addr} (model {}, context {}, idle stop {}, max runtime {}, pod {})",
        env!("CARGO_PKG_VERSION"),
        shared.config.model,
        shared.config.max_model_len,
        remote::human_duration(shared.config.idle_stop.as_secs()),
        remote::human_duration(shared.config.max_runtime.as_secs()),
        shared.config.stopper.pod_id().unwrap_or("none")
    ));
    tokio::spawn(start_upstream(shared.clone()));
    let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel::<()>();
    let (exit_tx, exit_rx) = tokio::sync::oneshot::channel::<()>();
    let timers = shared.clone();
    tokio::spawn(async move {
        let tick = timers.config.tick;
        lifecycle::watch(timers.clone(), tick).await;
        if matches!(timers.config.stopper, Stopper::Exit) {
            let _ = exit_tx.send(());
        }
    });
    let app = router(shared.clone());
    let task = tokio::spawn(async move {
        axum::serve(listener, app)
            .with_graceful_shutdown(async move {
                tokio::select! {
                    _ = shutdown_rx => {}
                    // Only a sent exit stops the server (a dropped sender is no exit).
                    Ok(()) = exit_rx => {}
                    () = crate::cli::serve::terminate() => {}
                }
            })
            .await
    });
    Ok(RunningAgent {
        addr,
        shared,
        shutdown: Some(shutdown_tx),
        task,
    })
}

async fn start_upstream(shared: Arc<Shared>) {
    let key = match remote::random_hex(32) {
        Ok(key) => key,
        Err(error) => {
            shared.note(&format!("no randomness: {error}"));
            shared.set_phase(Phase::Failed(vec![format!("no randomness: {error}")]));
            return;
        }
    };
    match shared.config.backend.clone() {
        Backend::Stub => {
            match stub::start(
                &shared.config.model,
                shared.config.max_model_len,
                key.clone(),
            )
            .await
            {
                Ok(stub) => {
                    *shared.stub.lock().unwrap_or_else(PoisonError::into_inner) =
                        Some(stub.stats.clone());
                    connect(&shared, &format!("http://{}", stub.addr), key);
                    shared.set_phase(Phase::Loading);
                    probe_ready(&shared).await;
                }
                Err(error) => {
                    shared.note(&format!("the stub failed: {error}"));
                    shared.set_phase(Phase::Failed(vec![error.to_string()]));
                }
            }
        }
        Backend::Vllm { program, port } => {
            let spec = shared.config.vllm_spec(program, port, key.clone());
            shared.note(&format!(
                "starting vllm serve {} (127.0.0.1:{port}, context {}, max {} sequences, tools {}, reasoning {})",
                spec.model,
                spec.max_model_len,
                spec.max_num_seqs,
                spec.tool_parser,
                spec.reasoning_parser.as_deref().unwrap_or("off")
            ));
            if !spec.extra_args.is_empty() {
                shared.note(&format!(
                    "vllm flags of the model: {}",
                    spec.extra_args.join(" ")
                ));
            }
            connect(&shared, &format!("http://127.0.0.1:{port}"), key);
            let log = shared.log.clone();
            run_vllm(shared, move || {
                vllm::spawn(spec.command(), Some(log.clone()))
            })
            .await;
        }
    }
}

fn connect(shared: &Shared, base: &str, key: String) {
    if let Ok(upstream) = Upstream::new(base, key, None) {
        *shared
            .upstream
            .lock()
            .unwrap_or_else(PoisonError::into_inner) = Some(upstream);
    }
}

/// `true` once the upstream lists its model; records its context and version.
async fn probe_ready(shared: &Shared) -> bool {
    let Some(upstream) = shared.upstream() else {
        return false;
    };
    let Ok(models) = upstream
        .get_json("/v1/models", Duration::from_secs(5))
        .await
    else {
        return false;
    };
    if let Some(len) = models["data"][0]["max_model_len"].as_u64() {
        shared.max_model_len.store(len as u32, Ordering::Relaxed);
    }
    let version = upstream
        .get_json("/version", Duration::from_secs(5))
        .await
        .ok()
        .and_then(|v| v["version"].as_str().map(str::to_string))
        .unwrap_or_else(|| "unknown".into());
    *shared
        .backend_version
        .lock()
        .unwrap_or_else(PoisonError::into_inner) = Some(version);
    shared.set_phase(Phase::Ready);
    shared.note(&format!(
        "ready ({}, {})",
        shared.config.model,
        shared.backend_label().unwrap_or_default()
    ));
    true
}

/// Starts vLLM with `start` and supervises it. One that exits after it served is started again
/// ([`MAX_RESTARTS`] times; the weights are on disk, so a restart takes minutes); an exit before
/// it serves, or after the restarts, fails the agent.
async fn run_vllm(
    shared: Arc<Shared>,
    mut start: impl FnMut() -> std::io::Result<vllm::Supervised>,
) {
    let mut restarts = 0;
    loop {
        let child = match start() {
            Ok(child) => child,
            Err(error) => {
                shared.note(&format!("cannot start vllm: {error}"));
                shared.set_phase(Phase::Failed(vec![format!("cannot start vllm: {error}")]));
                return;
            }
        };
        // Loading again after a restart: requests wait (503), and it is not idle time.
        shared.set_phase(Phase::Loading);
        let Ended::AfterReady {
            status,
            served,
            mut tail,
        } = supervise(shared.clone(), child).await
        else {
            return;
        };
        let line = format!(
            "vllm exited after {} ({status})",
            remote::human_duration(served.as_secs())
        );
        if restarts < MAX_RESTARTS {
            restarts += 1;
            shared.note(&format!(
                "{line}; restarting it ({restarts} of {MAX_RESTARTS})"
            ));
            continue;
        }
        shared.note(&format!("{line}; not restarting it again"));
        tail.push(line);
        shared.set_phase(Phase::Failed(tail));
        return;
    }
}

/// How a supervised vLLM ended.
enum Ended {
    /// It failed before it served (or did not serve in time); the agent has failed.
    Failed,
    /// It exited after serving this long; the caller decides about a restart.
    AfterReady {
        status: std::process::ExitStatus,
        served: Duration,
        tail: Vec<String>,
    },
}

/// Polls vLLM every 2 s until it serves, fails or times out; then watches for its exit.
async fn supervise(shared: Arc<Shared>, child: vllm::Supervised) -> Ended {
    let start = Instant::now();
    let home = vllm::hf_home();
    let mut ready_at: Option<Instant> = None;
    loop {
        if let Some(status) = child.exit_status() {
            let mut tail = child.log.tail();
            if let Some(at) = ready_at {
                return Ended::AfterReady {
                    status,
                    served: at.elapsed(),
                    tail,
                };
            }
            tail.push(format!("vllm exited: {status}"));
            shared.note(&format!("vllm exited ({status})"));
            shared.set_phase(Phase::Failed(tail));
            return Ended::Failed;
        }
        if ready_at.is_none() {
            let dir = home.clone();
            if let Ok(bytes) = tokio::task::spawn_blocking(move || vllm::dir_size(&dir)).await {
                shared.weights_bytes.store(bytes, Ordering::Relaxed);
            }
            if probe_ready(&shared).await {
                ready_at = Some(Instant::now());
            } else if start.elapsed() >= shared.config.load_timeout {
                let mut tail = child.log.tail();
                tail.push(format!(
                    "vllm did not serve within {} minutes",
                    shared.config.load_timeout.as_secs() / 60
                ));
                shared.note(&format!(
                    "vllm did not serve within {} minutes; killing it",
                    shared.config.load_timeout.as_secs() / 60
                ));
                child.kill();
                shared.set_phase(Phase::Failed(tail));
                return Ended::Failed;
            }
        }
        tokio::time::sleep(Duration::from_secs(2)).await;
    }
}

fn router(shared: Arc<Shared>) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/mrun/status", get(status))
        .route("/mrun/logs", get(logs))
        .route("/v1/models", get(models))
        .route("/v1/messages", post(api))
        .route("/v1/messages/count_tokens", post(api))
        .route("/v1/chat/completions", post(api))
        .route("/v1/responses", post(api))
        .layer(DefaultBodyLimit::max(MAX_BODY_BYTES))
        .layer(middleware::from_fn_with_state(shared.clone(), auth))
        .with_state(shared)
}

/// 401 without the pod token on every route but `/health` (013 FR-008); nothing is forwarded
/// and the idle timer is not reset.
async fn auth(State(shared): State<Arc<Shared>>, request: Request, next: Next) -> Response {
    let path = request.uri().path().to_string();
    if path == "/health" {
        return next.run(request).await;
    }
    let ok = remote::bearer(request.headers())
        .is_some_and(|token| remote::tokens_equal(token, &shared.config.token));
    if !ok {
        return remote::error_response(
            api_of(&path),
            &ApiError::new(ErrorKind::Unauthorized, "a valid pod token is required"),
        );
    }
    next.run(request).await
}

async fn health(State(shared): State<Arc<Shared>>) -> Response {
    let phase = shared.phase();
    let status = match phase {
        Phase::Ready => "ok",
        Phase::Starting | Phase::Loading => "loading",
        Phase::Failed(_) => "failed",
        Phase::Stopping(_) => "stopping",
    };
    axum::Json(json!({"status": status, "phase": phase.as_str()})).into_response()
}

async fn status(State(shared): State<Arc<Shared>>) -> Response {
    axum::Json(shared.status_json()).into_response()
}

/// `GET /mrun/logs?offset=<bytes>[&follow=1]` (FR-015): the log from `offset` (default 0),
/// at most [`logfile::READ_LIMIT`] bytes; `x-mrun-log-start` is where the bytes start (a
/// rotated-away offset starts at the oldest byte kept), `x-mrun-log-offset` where the next read
/// continues, `x-mrun-log-end` the log's end. With `follow=1` the response streams whole lines
/// from `offset` as they are written, an empty line after each keepalive interval of silence, until
/// the client disconnects. Reading the log does not reset the idle timer.
async fn logs(State(shared): State<Arc<Shared>>, uri: Uri) -> Response {
    let mut offset = 0u64;
    let mut follow = false;
    for pair in uri.query().unwrap_or("").split('&') {
        match pair.split_once('=').unwrap_or((pair, "")) {
            ("offset", value) => match value.parse() {
                Ok(value) => offset = value,
                Err(_) => {
                    return remote::error_response(
                        None,
                        &ApiError::new(ErrorKind::InvalidRequest, "offset must be a byte count"),
                    );
                }
            },
            ("follow", value) => follow = matches!(value, "1" | "true" | ""),
            _ => {}
        }
    }
    let log = shared.log.clone();
    let chunk = {
        let log = log.clone();
        let limit = if follow { 0 } else { logfile::READ_LIMIT };
        tokio::task::spawn_blocking(move || log.read(offset, limit))
            .await
            .unwrap_or_else(|_| logfile::Chunk {
                bytes: Vec::new(),
                start: offset,
                next: offset,
                end: offset,
                first: 0,
            })
    };
    let headers = [
        ("content-type", "text/plain; charset=utf-8".to_string()),
        ("cache-control", "no-store".to_string()),
        ("x-mrun-log-start", chunk.start.to_string()),
        ("x-mrun-log-offset", chunk.next.to_string()),
        ("x-mrun-log-end", chunk.end.to_string()),
    ];
    let mut response = if follow {
        let keepalive = shared.config.keepalive;
        let state = (log, chunk.start, Instant::now());
        let stream =
            futures_util::stream::unfold(state, move |(log, mut at, mut sent)| async move {
                loop {
                    let reader = log.clone();
                    let chunk = tokio::task::spawn_blocking(move || reader.read(at, 64 * 1024))
                        .await
                        .ok()?;
                    let lines = logfile::whole_lines(&chunk.bytes);
                    if !lines.is_empty() {
                        at = chunk.start + lines.len() as u64;
                        let bytes = Bytes::copy_from_slice(lines);
                        return Some((Ok::<_, std::io::Error>(bytes), (log, at, Instant::now())));
                    }
                    at = chunk.start;
                    if sent.elapsed() >= keepalive {
                        sent = Instant::now();
                        return Some((Ok(Bytes::from_static(b"\n")), (log, at, sent)));
                    }
                    tokio::time::sleep(Duration::from_millis(200).min(keepalive)).await;
                }
            });
        axum::body::Body::from_stream(stream).into_response()
    } else {
        chunk.bytes.into_response()
    };
    for (name, value) in headers {
        if let Ok(value) = axum::http::HeaderValue::from_str(&value) {
            response.headers_mut().insert(name, value);
        }
    }
    response
}

/// The upstream if the agent serves; the API error otherwise.
fn serving(shared: &Shared) -> Result<Upstream, ApiError> {
    match shared.phase() {
        Phase::Ready => shared
            .upstream()
            .ok_or_else(|| ApiError::new(ErrorKind::Internal, "no upstream")),
        Phase::Starting | Phase::Loading => Err(ApiError::new(
            ErrorKind::NotReady,
            "the remote model is loading",
        )
        .with_retry_after(5)),
        Phase::Failed(_) => Err(ApiError::new(
            ErrorKind::Internal,
            shared.failure().unwrap_or_default(),
        )),
        Phase::Stopping(reason) => Err(ApiError::new(
            ErrorKind::Overloaded,
            format!("the pod is stopping ({})", reason.as_str()),
        )),
    }
}

async fn models(State(shared): State<Arc<Shared>>) -> Response {
    match serving(&shared) {
        Ok(upstream) => {
            forward::forward(
                &upstream,
                reqwest::Method::GET,
                "/v1/models",
                None,
                Options::default(),
            )
            .await
        }
        Err(error) => remote::error_response(None, &error),
    }
}

async fn api(State(shared): State<Arc<Shared>>, uri: Uri, body: Bytes) -> Response {
    let path = uri.path();
    let api = api_of(path);
    let upstream = match serving(&shared) {
        Ok(upstream) => upstream,
        Err(error) => return remote::error_response(api, &error),
    };
    let parsed: Value = serde_json::from_slice(&body).unwrap_or(Value::Null);
    let (stream, include_usage) = forward::stream_flags(&parsed);
    shared.requests_total.fetch_add(1, Ordering::Relaxed);
    let guard = ActiveGuard::new(shared.clone());
    let options = Options {
        api,
        stream,
        include_usage,
        model: parsed["model"]
            .as_str()
            .unwrap_or(&shared.config.model)
            .to_string(),
        keepalive: shared.config.keepalive,
        guard: Some(Box::new(guard)),
        retry: forward::clamp_retry(
            api,
            parsed.clone(),
            shared.max_model_len.load(Ordering::Relaxed),
        ),
        ..Options::default()
    };
    forward::forward(&upstream, reqwest::Method::POST, path, Some(body), options).await
}

/// `mrun agent`. Returns the exit code.
pub fn run(args: AgentArgs) -> u8 {
    let config = match AgentConfig::from_args(args) {
        Ok(config) => config,
        Err(message) => {
            eprintln!("error: {message}");
            return 2;
        }
    };
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
        eprintln!(
            "mrun agent {} on {} (model {}, idle stop {}, max runtime {}, pod {}, log {})",
            env!("CARGO_PKG_VERSION"),
            config.bind,
            config.model,
            remote::human_duration(config.idle_stop.as_secs()),
            remote::human_duration(config.max_runtime.as_secs()),
            config.stopper.pod_id().unwrap_or("none"),
            config
                .log_path
                .as_deref()
                .map_or("none".into(), |p| p.display().to_string())
        );
        let agent = match start(config).await {
            Ok(agent) => agent,
            Err(error) => {
                eprintln!("error: cannot listen: {error}");
                return 1;
            }
        };
        match agent.wait().await {
            Ok(()) => 0,
            Err(error) => {
                eprintln!("error: agent failed: {error}");
                1
            }
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[derive(Parser)]
    struct Cli {
        #[command(flatten)]
        agent: AgentArgs,
    }

    #[test]
    fn max_num_seqs_reaches_the_vllm_command_line() {
        let parse = |extra: &[&str]| {
            let mut argv = vec!["agent", "--model", "Qwen/Qwen3-Coder-Next-FP8"];
            argv.extend_from_slice(extra);
            Cli::try_parse_from(argv).unwrap().agent
        };
        let config = |args: AgentArgs| AgentConfig {
            bind: SocketAddr::from((Ipv4Addr::LOCALHOST, 0)),
            token: "t".repeat(32),
            model: args.model,
            max_model_len: args.max_model_len,
            max_num_seqs: args.max_num_seqs,
            tool_parser: args.tool_parser,
            reasoning_parser: args.reasoning_parser,
            vllm_args: vec![],
            idle_stop: args.idle_stop,
            max_runtime: args.max_runtime,
            failed_stop: args.failed_stop,
            grace: lifecycle::GRACE,
            cost_per_hr: 0.0,
            gpu: None,
            keepalive: KEEPALIVE,
            backend: Backend::Stub,
            stopper: Stopper::Exit,
            tick: Duration::from_secs(5),
            load_timeout: LOAD_TIMEOUT,
            log_path: None,
            log_max_bytes: logfile::MAX_BYTES,
        };
        let value = |args: Vec<String>| {
            let i = args.iter().position(|a| a == "--max-num-seqs").unwrap();
            args[i + 1].clone()
        };
        let default = config(parse(&[])).vllm_spec("vllm".into(), VLLM_PORT, "k".into());
        // Only when MRUN_MAX_NUM_SEQS is unset in the test environment.
        if std::env::var_os("MRUN_MAX_NUM_SEQS").is_none() {
            assert_eq!(value(default.args()), DEFAULT_MAX_NUM_SEQS.to_string());
        }
        let set =
            config(parse(&["--max-num-seqs", "8"])).vllm_spec("vllm".into(), 8001, "k".into());
        assert_eq!(value(set.args()), "8");
    }

    #[test]
    fn model_flags_come_as_a_json_array() {
        assert_eq!(parse_vllm_args(None), Ok(vec![]));
        assert_eq!(parse_vllm_args(Some(" ")), Ok(vec![]));
        assert_eq!(
            parse_vllm_args(Some(r#"["--speculative-config", "{\"method\": \"mtp\"}"]"#)),
            Ok(vec![
                "--speculative-config".to_string(),
                r#"{"method": "mtp"}"#.to_string()
            ])
        );
        assert!(parse_vllm_args(Some("--flag value")).is_err());
    }

    fn test_shared(failed_stop: Duration) -> Arc<Shared> {
        let args = Cli::try_parse_from(["agent", "--model", "m"])
            .unwrap()
            .agent;
        Arc::new(Shared::new(AgentConfig {
            bind: SocketAddr::from((Ipv4Addr::LOCALHOST, 0)),
            token: "t".repeat(32),
            model: args.model,
            max_model_len: args.max_model_len,
            max_num_seqs: args.max_num_seqs,
            tool_parser: args.tool_parser,
            reasoning_parser: None,
            vllm_args: vec![],
            idle_stop: args.idle_stop,
            max_runtime: args.max_runtime,
            failed_stop,
            grace: lifecycle::GRACE,
            cost_per_hr: 0.0,
            gpu: None,
            keepalive: KEEPALIVE,
            backend: Backend::Stub,
            stopper: Stopper::Exit,
            tick: Duration::from_secs(5),
            load_timeout: LOAD_TIMEOUT,
            log_path: None,
            log_max_bytes: logfile::MAX_BYTES,
        }))
    }

    /// A stand-in for vLLM that runs for `seconds`, then exits 0.
    fn exits_after(seconds: &str) -> std::io::Result<vllm::Supervised> {
        let mut command = std::process::Command::new("sh");
        command.args(["-c", &format!("sleep {seconds}")]);
        vllm::spawn(command, None)
    }

    #[tokio::test]
    async fn a_vllm_that_exits_after_serving_is_restarted_once_then_the_model_stopped() {
        let shared = test_shared(Duration::from_secs(300));
        // The stub answers /v1/models, so each started process counts as serving.
        let stub = stub::start("m", 65_536, "k".into()).await.unwrap();
        connect(&shared, &format!("http://{}", stub.addr), "k".into());
        let starts = Arc::new(AtomicU32::new(0));
        let counted = starts.clone();
        run_vllm(shared.clone(), move || {
            counted.fetch_add(1, Ordering::SeqCst);
            exits_after("1")
        })
        .await;
        assert_eq!(starts.load(Ordering::SeqCst), 1 + MAX_RESTARTS);
        assert_eq!(shared.phase().as_str(), "failed");
        let failure = shared.failure().unwrap();
        assert!(
            failure.starts_with("the remote model stopped: vllm exited after "),
            "{failure}"
        );
        assert!(!failure.contains("failed to start"), "{failure}");
        assert!(shared.failed_for(Instant::now()).is_some());
        assert_eq!(shared.status_json()["error"], failure.as_str());
    }

    #[tokio::test]
    async fn a_vllm_that_exits_before_serving_is_a_failed_start_without_a_restart() {
        let shared = test_shared(Duration::from_secs(300));
        let starts = Arc::new(AtomicU32::new(0));
        let counted = starts.clone();
        run_vllm(shared.clone(), move || {
            counted.fetch_add(1, Ordering::SeqCst);
            exits_after("0")
        })
        .await;
        assert_eq!(starts.load(Ordering::SeqCst), 1);
        let failure = shared.failure().unwrap();
        assert!(
            failure.starts_with("the remote model failed to start: vllm exited"),
            "{failure}"
        );
    }

    #[test]
    fn a_failed_agent_stops_its_pod_after_the_failed_stop_unless_it_is_zero() {
        let now = Instant::now();
        let shared = test_shared(Duration::from_secs(60));
        shared.set_phase(Phase::Ready);
        assert_eq!(
            lifecycle::due(&shared, now + Duration::from_secs(120)),
            None
        );
        shared.set_phase(Phase::Failed(vec!["x".into()]));
        assert_eq!(
            lifecycle::due(&shared, Instant::now()),
            None,
            "grace for the log"
        );
        assert_eq!(
            lifecycle::due(&shared, Instant::now() + Duration::from_secs(61)),
            Some(Reason::Failed)
        );
        // --keep-on-error: only the idle stop (30 minutes by default).
        let kept = test_shared(Duration::ZERO);
        kept.set_phase(Phase::Failed(vec!["x".into()]));
        assert_eq!(
            lifecycle::due(&kept, Instant::now() + Duration::from_secs(61)),
            None
        );
    }

    #[test]
    fn the_idle_period_starts_when_the_model_is_ready() {
        let args = Cli::try_parse_from(["agent", "--model", "m", "--idle-stop", "30m"])
            .unwrap()
            .agent;
        let shared = Shared::new(AgentConfig {
            bind: SocketAddr::from((Ipv4Addr::LOCALHOST, 0)),
            token: "t".repeat(32),
            model: args.model,
            max_model_len: args.max_model_len,
            max_num_seqs: args.max_num_seqs,
            tool_parser: args.tool_parser,
            reasoning_parser: args.reasoning_parser,
            vllm_args: vec![],
            idle_stop: args.idle_stop,
            max_runtime: Duration::from_secs(8 * 3600),
            failed_stop: args.failed_stop,
            grace: lifecycle::GRACE,
            cost_per_hr: 0.0,
            gpu: None,
            keepalive: KEEPALIVE,
            backend: Backend::Stub,
            stopper: Stopper::Exit,
            tick: Duration::from_secs(5),
            load_timeout: LOAD_TIMEOUT,
            log_path: None,
            log_max_bytes: logfile::MAX_BYTES,
        });
        let later = Instant::now() + Duration::from_secs(40 * 60);
        // 40 minutes of loading is not idle time.
        shared.set_phase(Phase::Loading);
        assert_eq!(shared.idle_for(later), None);
        assert_eq!(lifecycle::due(&shared, later), None);
        // Ready now: the 30 minutes start here.
        shared.set_phase(Phase::Ready);
        let soon = Instant::now() + Duration::from_secs(29 * 60);
        assert_eq!(lifecycle::due(&shared, soon), None);
        let after = Instant::now() + Duration::from_secs(31 * 60);
        assert_eq!(
            lifecycle::due(&shared, after),
            Some(lifecycle::Reason::Idle)
        );
    }
}
