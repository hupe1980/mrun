//! `mrun cloud`: remote GPU mode on RunPod (013, D064; contract cloud-cli.md).
//!
//! `up` creates one pod (vLLM image pinned by digest, the agent pinned by sha256, one exposed
//! HTTP port, a random pod token in its env), waits through the phases, and starts the local
//! relay (`mrun serve --relay`) with the token in the relay's environment only. Any failure after
//! the pod exists terminates it (unless `--keep-on-error`), Ctrl-C too. The account key is read
//! from `RUNPOD_API_KEY` or `./.env`, used for the API calls and nothing else (FR-001).
//!
//! The agent's full log (FR-015): `up --verbose` follows it while the pod starts; a failed or
//! interrupted start saves it to `~/.mrun/logs/cloud-<pod>.log` before the pod is terminated;
//! `cloud logs` reads or follows the running pod's log.
//!
//! While `up` waits, a marker (`cloud-starting-<port>.json` in `MRUN_HOME`, its pid and model)
//! tells `mrun launch` to wait for the relay instead of offering a local model.

pub mod catalog;

use crate::cli::server::{self, Health};
use crate::cli::{confirm, info};
use crate::errors::{CliError, Code};
use crate::remote::{self, AGENT_PORT, human_duration};
use catalog::Offer;
use mrun_runpod::{ApiKey, Client, CloudType, Compute, GpuType, Pod, PodCreate};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::io::IsTerminal;
use std::path::PathBuf;
use std::time::{Duration, Instant};

/// The vLLM image, pinned by digest (research R5: v0.32.0, CUDA 13.0, 2026-10-18). Not the
/// `-cu129` variant: its torchcodec is built for CUDA 13 and vLLM fails at import (R13).
pub const VLLM_IMAGE: &str = "vllm/vllm-openai:v0.32.0@sha256:befccf0e7e4089aa23926bb6f887a85c0d46c2a99e6de5c5ffdf5cb7957c5b34";
/// The image of CPU test pods (the agent serves its stub upstream).
pub const CPU_IMAGE: &str = "python:3.12-slim";
/// CUDA versions the image runs on (hosts whose driver supports CUDA 13.0).
const CUDA: [&str; 1] = ["13.0"];
/// The bootstrap the pod runs first (fetch or receive the agent, check it, exec it).
const BOOTSTRAP: &str = include_str!("bootstrap.py");
/// Prefix of every pod and volume mrun creates.
pub const PREFIX: &str = "mrun-";
/// How long `up` waits for the agent to serve.
const READY_TIMEOUT: Duration = Duration::from_secs(40 * 60);
const POLL: Duration = Duration::from_secs(5);
/// Pause between rounds of `--retry` (every fitting GPU sold out).
const RETRY_POLL: Duration = Duration::from_secs(60);
/// A network volume's price, $/GB/month (RunPod, 2026-10).
const VOLUME_PRICE_GB_MONTH: f64 = 0.07;

#[derive(Debug, clap::Subcommand)]
pub enum CloudCommand {
    /// Create a pod for a model, wait until it serves, and serve it on 127.0.0.1.
    Up(Box<UpArgs>),
    /// Terminate mrun's pods and stop the local relay.
    Down(DownArgs),
    /// The pod, its cost and the agent's phase.
    Status(StatusArgs),
    /// GPU types with VRAM and prices.
    Gpus(ListArgs),
    /// Curated models with the smallest GPU each fits on.
    Models(ListArgs),
    /// The running pod's full log (vLLM's output and the agent's lines).
    Logs(LogsArgs),
}

#[derive(Debug, Clone, clap::Args)]
pub struct LogsArgs {
    /// Keep printing new lines until Ctrl-C.
    #[arg(long, short = 'f')]
    pub follow: bool,
    /// Only the last N lines.
    #[arg(long, short = 'n', value_name = "N")]
    pub tail: Option<usize>,
}

#[derive(Debug, Clone, clap::Args)]
pub struct UpArgs {
    /// Hugging Face repo to serve (see 'mrun cloud models').
    #[arg(long)]
    pub model: String,
    /// GPU type, id or display name ('mrun cloud gpus') [default: the cheapest that fits].
    #[arg(long)]
    pub gpu: Option<String>,
    /// Rent in Community Cloud (cheaper, hosts run by third parties) instead of Secure Cloud.
    #[arg(long)]
    pub community: bool,
    /// The pod stops itself after this long without a request.
    #[arg(long, default_value = "30m", value_parser = remote::parse_duration)]
    pub idle_stop: Duration,
    /// The pod stops itself after this long regardless.
    #[arg(long, default_value = "8h", value_parser = remote::parse_duration)]
    pub max_runtime: Duration,
    /// Context in tokens [default: the catalog's, else 65,536].
    #[arg(long)]
    pub max_model_len: Option<u32>,
    /// Most requests vLLM runs at once (hybrid models fail to start above their Mamba cache
    /// blocks).
    #[arg(long, value_name = "N", default_value_t = crate::agent::DEFAULT_MAX_NUM_SEQS,
        value_parser = clap::value_parser!(u32).range(1..))]
    pub max_num_seqs: u32,
    /// vLLM tool-call parser [default: the catalog's, else hermes].
    #[arg(long)]
    pub tool_parser: Option<String>,
    /// vLLM reasoning parser [default: the catalog's].
    #[arg(long)]
    pub reasoning_parser: Option<String>,
    /// Port of the local server on 127.0.0.1.
    #[arg(long, env = "MRUN_PORT", default_value_t = crate::cli::serve::DEFAULT_PORT)]
    pub port: u16,
    /// Cache the weights on a network volume in this data center (e.g. EU-RO-1); the pod then
    /// runs there.
    #[arg(long, value_name = "DC")]
    pub volume_dc: Option<String>,
    /// While RunPod has no machine (sold out), try again for this long: every GPU that fits in
    /// the chosen cloud up to --max-price, round after round [default: no retry].
    #[arg(long, value_name = "DURATION", value_parser = remote::parse_duration)]
    pub retry: Option<Duration>,
    /// Most $/h a --retry may rent at [default: the price shown before the first try].
    #[arg(long, value_name = "USD", requires = "retry")]
    pub max_price: Option<f64>,
    /// With --retry: try both clouds, cheapest first (Community Cloud's hosts are run by third
    /// parties).
    #[arg(long, requires = "retry")]
    pub any_cloud: bool,
    /// Leave a pod whose model failed to start running, for inspection.
    #[arg(long)]
    pub keep_on_error: bool,
    /// Print the pod's full log (vLLM's output) live while it starts.
    #[arg(long, short = 'v')]
    pub verbose: bool,
    /// Where the pod fetches the agent (a Linux build of mrun) [default: this version's
    /// release asset].
    #[arg(long, requires = "agent_sha256", conflicts_with = "agent_file")]
    pub agent_url: Option<String>,
    /// The sha256 of --agent-url's file.
    #[arg(long)]
    pub agent_sha256: Option<String>,
    /// Upload this local Linux build of mrun as the agent (through the pod's proxy, with the
    /// pod token).
    #[arg(long, value_name = "PATH")]
    pub agent_file: Option<PathBuf>,
    /// Do not ask.
    #[arg(long, short = 'y')]
    pub yes: bool,
    /// A CPU pod ($0.06/h) for testing: with `--model stub` its agent serves a fake model;
    /// with another model vLLM fails to start (the failure path).
    #[arg(long, hide = true)]
    pub cpu: bool,
    /// Pod name [default: mrun-<model>].
    #[arg(long, hide = true)]
    pub name: Option<String>,
}

#[derive(Debug, Clone, clap::Args)]
pub struct DownArgs {
    /// Also delete mrun's network volumes (the cached weights).
    #[arg(long)]
    pub volume: bool,
    /// Do not ask before deleting volumes.
    #[arg(long, short = 'y')]
    pub yes: bool,
    /// Port of the local relay.
    #[arg(long, env = "MRUN_PORT", default_value_t = crate::cli::serve::DEFAULT_PORT)]
    pub port: u16,
}

#[derive(Debug, Clone, clap::Args)]
pub struct StatusArgs {
    #[arg(long)]
    pub json: bool,
    /// Port of the local relay.
    #[arg(long, env = "MRUN_PORT", default_value_t = crate::cli::serve::DEFAULT_PORT)]
    pub port: u16,
}

#[derive(Debug, Clone, clap::Args)]
pub struct ListArgs {
    #[arg(long)]
    pub json: bool,
}

pub fn run(command: CloudCommand) -> u8 {
    let runtime = match tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
    {
        Ok(runtime) => runtime,
        Err(error) => return CliError::new(Code::Internal, error.to_string()).report(),
    };
    let result = runtime.block_on(async move {
        match command {
            CloudCommand::Up(args) => up(*args).await,
            CloudCommand::Down(args) => down(args).await,
            CloudCommand::Status(args) => status(args).await,
            CloudCommand::Gpus(args) => gpus(args).await,
            CloudCommand::Models(args) => models(args).await,
            CloudCommand::Logs(args) => logs(args).await,
        }
    });
    crate::cli::exit(result)
}

/// The account key (environment, else `./.env`).
fn key() -> Result<ApiKey, CliError> {
    let dir = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    ApiKey::from_env_or_dotenv(&dir).ok_or_else(|| {
        CliError::new(Code::InvalidArguments, "RUNPOD_API_KEY is not set")
            .hint("create a key at https://console.runpod.io/user/settings and export RUNPOD_API_KEY (or put it in ./.env)")
    })
}

fn client() -> Result<Client, CliError> {
    Client::new(key()?).map_err(api_error)
}

fn api_error(error: mrun_runpod::Error) -> CliError {
    let hint = match &error {
        mrun_runpod::Error::Unauthorized { .. } => "check RUNPOD_API_KEY (read and write access)",
        mrun_runpod::Error::Blocked => "a proxy or firewall changed the request's user agent",
        _ => "RunPod's status page: https://uptime.runpod.io",
    };
    CliError::new(Code::Network, error.to_string()).hint(hint)
}

/// Pods mrun created (name prefix) that are not terminated.
async fn mrun_pods(client: &Client) -> Result<Vec<Pod>, CliError> {
    Ok(client
        .list_pods()
        .await
        .map_err(api_error)?
        .into_iter()
        .filter(|p| p.name.starts_with(PREFIX) && p.desired_status != "TERMINATED")
        .collect())
}

/// What a pod computes on, for status: RunPod's GPU type, else the GPU `cloud up` chose (the
/// API omits the GPU of a stopped pod, and of one still starting), "CPU" only for CPU pods, else
/// "GPU unknown".
fn pod_compute(pod: &Pod) -> String {
    if let Some(gpu) = &pod.gpu {
        return gpu.id.clone();
    }
    if pod.cpu_flavor_id.is_some() {
        return "CPU".into();
    }
    match pod.env.get("MRUN_GPU").filter(|g| !g.is_empty()) {
        Some(gpu) if gpu.starts_with("CPU") => "CPU".into(),
        Some(gpu) => gpu.clone(),
        None => "GPU unknown".into(),
    }
}

/// `mrun-<repo name>` in lowercase with `[a-z0-9-]` only.
pub fn slug(repo: &str) -> String {
    let name = repo.rsplit('/').next().unwrap_or(repo).to_ascii_lowercase();
    let cleaned: String = name
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '-' })
        .collect();
    format!("{PREFIX}{}", cleaned.trim_matches('-'))
}

/// An HTTP client for the pod's proxy URL.
fn http() -> reqwest::Client {
    reqwest::Client::builder()
        .user_agent(mrun_runpod::USER_AGENT)
        .connect_timeout(Duration::from_secs(10))
        .timeout(Duration::from_secs(20))
        .build()
        .unwrap_or_default()
}

/// `GET <base><path>` with the pod token: status and JSON body (`Null` when not JSON).
async fn agent_get(
    http: &reqwest::Client,
    base: &str,
    path: &str,
    token: &str,
) -> Option<(u16, Value)> {
    let response = http
        .get(format!("{base}{path}"))
        .bearer_auth(token)
        .send()
        .await
        .ok()?;
    let status = response.status().as_u16();
    let body = response.json::<Value>().await.unwrap_or(Value::Null);
    Some((status, body))
}

/// How the pod gets the agent.
#[derive(Debug)]
enum AgentSource {
    Url { url: String, sha256: String },
    File { bytes: Vec<u8>, sha256: String },
}

impl AgentSource {
    fn sha256(&self) -> &str {
        match self {
            Self::Url { sha256, .. } | Self::File { sha256, .. } => sha256,
        }
    }
}

/// The release asset of this version (008 adds it to releases).
fn release_url() -> String {
    let v = env!("CARGO_PKG_VERSION");
    format!(
        "https://github.com/hupe1980/mrun/releases/download/v{v}/mrun-{v}-x86_64-unknown-linux-gnu"
    )
}

async fn agent_source(args: &UpArgs) -> Result<AgentSource, CliError> {
    if let Some(path) = &args.agent_file {
        let bytes = std::fs::read(path).map_err(|e| {
            CliError::new(
                Code::InvalidArguments,
                format!("cannot read {}: {e}", path.display()),
            )
        })?;
        if bytes.get(..4) != Some(b"\x7fELF") {
            return Err(CliError::new(
                Code::InvalidArguments,
                format!("{} is not a Linux executable", path.display()),
            )
            .hint("cargo zigbuild --release --target x86_64-unknown-linux-gnu.2.28 --no-default-features -p mrun"));
        }
        let sha256 = hex(&Sha256::digest(&bytes));
        return Ok(AgentSource::File { bytes, sha256 });
    }
    if let (Some(url), Some(sha256)) = (&args.agent_url, &args.agent_sha256) {
        return Ok(AgentSource::Url {
            url: url.clone(),
            sha256: sha256.trim().to_ascii_lowercase(),
        });
    }
    let url = release_url();
    let missing = || {
        CliError::new(
            Code::Network,
            format!(
                "no agent release asset for mrun {} ({url})",
                env!("CARGO_PKG_VERSION")
            ),
        )
        .hint("pass --agent-file with a Linux build (see the cloud docs) or --agent-url with --agent-sha256")
    };
    let text = http()
        .get(format!("{url}.sha256"))
        .send()
        .await
        .ok()
        .filter(|r| r.status().is_success())
        .ok_or_else(missing)?
        .text()
        .await
        .map_err(|_| missing())?;
    let sha256 = text
        .split_whitespace()
        .next()
        .filter(|s| s.len() == 64)
        .ok_or_else(missing)?
        .to_ascii_lowercase();
    Ok(AgentSource::Url { url, sha256 })
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// The model's serving settings: the catalog's, flags override.
fn offer_for(args: &UpArgs) -> (Option<Offer>, u32, String, Option<String>) {
    let offer = if args.cpu && args.model == "stub" {
        Some(catalog::STUB)
    } else if args.cpu {
        None
    } else {
        catalog::find(&args.model)
    };
    let len = args
        .max_model_len
        .or(offer.map(|o| o.max_model_len))
        .unwrap_or(65_536);
    let tool = args
        .tool_parser
        .clone()
        .or(offer.map(|o| o.tool_parser.to_string()))
        .unwrap_or_else(|| "hermes".into());
    let reasoning = args
        .reasoning_parser
        .clone()
        .or(offer.and_then(|o| o.reasoning_parser.map(str::to_string)));
    (offer, len, tool, reasoning)
}

/// The GPU to rent and its price.
fn choose_gpu(
    args: &UpArgs,
    offer: Option<&Offer>,
    gpus: &[GpuType],
    cloud: CloudType,
) -> Result<(GpuType, f64), CliError> {
    let cloud_name = match cloud {
        CloudType::Secure => "Secure Cloud",
        CloudType::Community => "Community Cloud",
    };
    let smallest = |offer: &Offer| {
        catalog::fitting(offer, gpus, cloud)
            .first()
            .map(|(g, p)| format!("{} ({} GB, ${p:.2}/h)", g.display_name, g.memory_gb))
    };
    match &args.gpu {
        Some(name) => {
            let gpu = gpus.iter().find(|g| g.matches(name)).ok_or_else(|| {
                CliError::new(Code::InvalidArguments, format!("unknown GPU type '{name}'"))
                    .hint("'mrun cloud gpus' lists them")
            })?;
            if !gpu.is_nvidia() {
                return Err(CliError::new(
                    Code::InvalidArguments,
                    format!(
                        "{} is not an NVIDIA GPU; the vLLM image needs CUDA",
                        gpu.display_name
                    ),
                ));
            }
            if let Some(offer) = offer
                && !offer.fits(gpu)
            {
                return Err(CliError::new(
                    Code::DoesNotFit,
                    format!(
                        "{} needs about {:.0} GB of GPU memory (weights + 64K tokens of KV); {} has {} GB",
                        offer.repo,
                        offer.min_vram_gib(),
                        gpu.display_name,
                        gpu.memory_gb
                    ),
                )
                .hint(match smallest(offer) {
                    Some(s) => format!("the smallest that fits in {cloud_name}: {s}"),
                    None => "'mrun cloud models' shows what fits".into(),
                }));
            }
            let price = gpu.price(cloud).ok_or_else(|| {
                CliError::new(
                    Code::InvalidArguments,
                    format!("{} is not offered in {cloud_name}", gpu.display_name),
                )
                .hint(if cloud == CloudType::Secure {
                    "try --community"
                } else {
                    "drop --community"
                })
            })?;
            Ok((gpu.clone(), price))
        }
        None => {
            let Some(offer) = offer else {
                return Err(CliError::new(
                    Code::InvalidArguments,
                    format!(
                        "{} is not in the catalog, so mrun cannot pick a GPU",
                        args.model
                    ),
                )
                .hint("name one with --gpu ('mrun cloud gpus')"));
            };
            catalog::fitting(offer, gpus, cloud)
                .first()
                .map(|(g, p)| ((*g).clone(), *p))
                .ok_or_else(|| {
                    CliError::new(
                        Code::DoesNotFit,
                        format!("no GPU in {cloud_name} fits {}", offer.repo),
                    )
                })
        }
    }
}

/// What `--retry` tries, cheapest first: the named GPU only, else every GPU in `clouds` that fits
/// `offer`, at `max_price` or less. Sold-out labels are ignored: they are often wrong, and a
/// refused create costs nothing.
fn retry_candidates(
    args: &UpArgs,
    offer: Option<&Offer>,
    gpus: &[GpuType],
    clouds: &[CloudType],
    max_price: f64,
) -> Vec<(GpuType, CloudType, f64)> {
    let mut out: Vec<(GpuType, CloudType, f64)> = clouds
        .iter()
        .flat_map(|&cloud| match (&args.gpu, offer) {
            (None, Some(offer)) => catalog::fitting(offer, gpus, cloud)
                .into_iter()
                .map(|(g, p)| (g.clone(), cloud, p))
                .collect::<Vec<_>>(),
            _ => choose_gpu(args, offer, gpus, cloud)
                .into_iter()
                .map(|(g, p)| (g, cloud, p))
                .collect(),
        })
        .filter(|(_, _, p)| *p <= max_price + 1e-9)
        .collect();
    out.sort_by(|a, b| a.2.total_cmp(&b.2));
    out
}

fn cloud_name(cloud: CloudType) -> &'static str {
    match cloud {
        CloudType::Secure => "Secure",
        CloudType::Community => "Community",
    }
}

/// The local relay on `port` (`Some(agent url)`), another server (`Some("")`), or none.
fn local_server(port: u16) -> Option<String> {
    if server::health(port) == Health::Down {
        return None;
    }
    let url = server::request(port, "GET", "/mrun/status")
        .ok()
        .and_then(|(_, s)| s["pod"]["url"].as_str().map(str::to_string));
    Some(url.unwrap_or_default())
}

/// A `cloud up` that is starting a pod for a port: its model and pid.
#[derive(Clone, Debug, PartialEq)]
pub struct Starting {
    pub model: String,
    pub pid: u32,
}

fn starting_file(port: u16) -> PathBuf {
    crate::cli::mrun_home().join(format!("cloud-starting-{port}.json"))
}

/// The `cloud up` starting a pod for `port`, if one runs (a marker left by a killed one is
/// ignored: its pid is no mrun process).
pub fn starting(port: u16) -> Option<Starting> {
    read_starting(&starting_file(port)).filter(|s| crate::cli::stop::is_mrun_process(s.pid))
}

fn read_starting(path: &std::path::Path) -> Option<Starting> {
    let value: Value = serde_json::from_slice(&std::fs::read(path).ok()?).ok()?;
    Some(Starting {
        model: value["model"].as_str()?.to_string(),
        pid: u32::try_from(value["pid"].as_u64()?).ok()?,
    })
}

/// The marker of a starting `cloud up`, removed when `up` returns (ready, failed or interrupted).
struct StartingMark(PathBuf);

impl StartingMark {
    fn at(path: PathBuf, model: &str) -> Self {
        let body = json!({"model": model, "pid": std::process::id()}).to_string();
        if let Err(e) = write_private(&path, body.as_bytes()) {
            eprintln!("warning: could not write {}: {e}", path.display());
        }
        Self(path)
    }
}

impl Drop for StartingMark {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

/// What `cloud up` shows before it asks (013 FR-003).
struct Consent<'a> {
    model: &'a str,
    gpu: String,
    idle_stop: Duration,
    max_runtime: Duration,
    max_model_len: u32,
    max_num_seqs: u32,
    host_ram: Option<String>,
    image: Option<String>,
    volume: Option<String>,
    /// The retry window, the candidates, and whether they span both clouds.
    retry: Option<(Duration, Vec<String>, bool)>,
}

/// The consent screen's lines; a line whose value came from the config file ends with
/// " (config)" (017 FR-010). `from_file` says whether an argument of `cloud up` was set there.
fn consent_lines(c: &Consent, from_file: &dyn Fn(&str) -> bool) -> Vec<String> {
    let mark = |args: &[&str]| {
        if args.iter().any(|a| from_file(a)) {
            " (config)"
        } else {
            ""
        }
    };
    let mut lines = vec![format!("mrun cloud up: {}", c.model)];
    lines.push(format!(
        "  GPU       {}{}",
        c.gpu,
        mark(&["gpu", "community", "any_cloud"])
    ));
    lines.push(format!(
        "  limits    stops itself after {} idle, after {} at most{}",
        human_duration(c.idle_stop.as_secs()),
        human_duration(c.max_runtime.as_secs()),
        mark(&["idle_stop", "max_runtime"])
    ));
    lines.push(format!(
        "  context   {} tokens, up to {} requests at once (vLLM, prefix caching on){}",
        c.max_model_len,
        c.max_num_seqs,
        mark(&["max_model_len", "max_num_seqs"])
    ));
    if let Some(host_ram) = &c.host_ram {
        lines.push(format!("  host RAM  {host_ram}"));
    }
    if let Some(image) = &c.image {
        lines.push(format!("  image     {image}"));
    }
    if let Some(volume) = &c.volume {
        lines.push(format!("  volume    {volume}{}", mark(&["volume_dc"])));
    }
    if let Some((window, names, any_cloud)) = &c.retry {
        lines.push(format!(
            "  retry     while sold out, for up to {}: {}{}",
            human_duration(window.as_secs()),
            names.join(", "),
            mark(&["retry", "max_price", "any_cloud"])
        ));
        if *any_cloud {
            lines.push("            Community Cloud hosts are run by third parties".into());
        }
    }
    lines.push(
        "  privacy   prompts and code are sent to RunPod; nothing is sent until a client does"
            .into(),
    );
    lines
}

async fn up(args: UpArgs) -> Result<(), CliError> {
    let client = client()?;
    let (offer, max_model_len, tool_parser, reasoning_parser) = offer_for(&args);
    let model = args.model.clone();

    // One pod per account (v1): attach to the same model, refuse another.
    if let Some(pod) = mrun_pods(&client).await?.into_iter().next() {
        let pod_model = pod.env.get("MRUN_MODEL").cloned().unwrap_or_default();
        if pod_model == model && pod.is_running() {
            let _mark = StartingMark::at(starting_file(args.port), &model);
            info(&format!(
                "pod {} ({}) already serves {model}; attaching",
                pod.id, pod.name
            ));
            let token = pod.env.get("MRUN_POD_TOKEN").cloned().unwrap_or_default();
            let price = pod.cost_per_hr;
            let gpu = pod_compute(&pod);
            let base = pod.proxy_url(AGENT_PORT);
            if let Err(error) = wait_ready(&base, &token, None, Instant::now()).await {
                save_log(&base, &token, &saved_log_path(&pod.id)).await;
                return Err(error);
            }
            return start_relay(&args, &pod, &token, (&gpu, price), &model, None);
        }
        return Err(CliError::new(
            Code::InvalidArguments,
            format!(
                "pod {} ({}, {}, {}) already exists; mrun keeps one pod per account",
                pod.id, pod.name, pod_model, pod.desired_status
            ),
        )
        .hint("'mrun cloud down' terminates it"));
    }
    if local_server(args.port).is_some() {
        return Err(CliError::new(
            Code::InvalidArguments,
            format!("a server already runs on 127.0.0.1:{}", args.port),
        )
        .hint("'mrun stop' stops it, or pass --port"));
    }
    let _mark = StartingMark::at(starting_file(args.port), &model);

    let mut cloud = if args.community {
        CloudType::Community
    } else {
        CloudType::Secure
    };
    let clouds: &[CloudType] = if args.any_cloud {
        &[CloudType::Secure, CloudType::Community]
    } else if args.community {
        &[CloudType::Community]
    } else {
        &[CloudType::Secure]
    };
    let mut candidates = Vec::new();
    let (compute, mut gpu_label, mut price) = if args.cpu {
        (
            Compute::Cpu {
                flavor: "cpu3c".into(),
                vcpus: 2,
            },
            "CPU (cpu3c, 2 vCPU)".to_string(),
            0.06,
        )
    } else {
        let gpus = client.gpu_types().await.map_err(api_error)?;
        if offer.is_none() {
            info(&format!(
                "{model} is not in the catalog: no fit check; context {max_model_len}, tool parser {tool_parser}"
            ));
        }
        let (mut gpu, mut price) = choose_gpu(&args, offer.as_ref(), &gpus, cloud)?;
        if let Some(max) = args.max_price
            && max < price
        {
            return Err(CliError::new(
                Code::InvalidArguments,
                format!(
                    "{} costs ${price:.2}/h, above --max-price ${max:.2}",
                    gpu.display_name
                ),
            ));
        }
        if args.retry.is_some() {
            let max = args.max_price.unwrap_or(price);
            candidates = retry_candidates(&args, offer.as_ref(), &gpus, clouds, max);
            // The cheapest candidate first (in either cloud with --any-cloud).
            if let Some((g, c, p)) = candidates.first() {
                (gpu, cloud, price) = (g.clone(), *c, *p);
            }
        }
        (
            Compute::Gpu {
                type_id: gpu.id.clone(),
                cloud,
            },
            gpu.id.clone(),
            price,
        )
    };
    let source = agent_source(&args).await?;

    // Volume (opt-in, research R10).
    let mut volume = None;
    if let Some(dc) = &args.volume_dc {
        let name = slug(&model);
        let existing = client
            .list_volumes()
            .await
            .map_err(api_error)?
            .into_iter()
            .find(|v| v.name == name && &v.data_center_id == dc);
        let size = offer.map_or(60, |o| o.disk_gb());
        volume = Some((name, dc.clone(), existing, size));
    }

    // Consent (FR-003); values from the config file are marked (017 FR-010).
    let consent = Consent {
        model: &model,
        gpu: format!(
            "{gpu_label}, {} Cloud, ${price:.2}/h (${:.2} per 8 h)",
            cloud_name(cloud),
            price * 8.0
        ),
        idle_stop: args.idle_stop,
        max_runtime: args.max_runtime,
        max_model_len,
        max_num_seqs: args.max_num_seqs,
        host_ram: offer.filter(|o| o.host_bytes > 0 && !args.cpu).map(|o| {
            format!(
                "{:.0} GB of the weights stay in host RAM; the pod gets at least {} GB",
                o.host_bytes as f64 / 1e9,
                o.min_ram_gb
            )
        }),
        image: offer
            .and_then(|o| o.image)
            .filter(|_| !args.cpu)
            .map(|image| image.split('@').next().unwrap_or(image).to_string()),
        volume: volume.as_ref().map(|(name, dc, existing, size)| {
            format!(
                "{name} in {dc}, {size} GB (${:.2}/month, kept by 'cloud down'){}",
                *size as f64 * VOLUME_PRICE_GB_MONTH,
                if existing.is_some() { ", reused" } else { "" }
            )
        }),
        retry: args.retry.filter(|_| !candidates.is_empty()).map(|retry| {
            let names: Vec<String> = candidates
                .iter()
                .map(|(g, c, p)| {
                    if args.any_cloud {
                        format!("{} ({}, ${p:.2}/h)", g.display_name, cloud_name(*c))
                    } else {
                        format!("{} (${p:.2}/h)", g.display_name)
                    }
                })
                .collect();
            (retry, names, args.any_cloud)
        }),
    };
    let from_file = |arg: &str| crate::config::from_file("cloud up", arg).is_some();
    for line in consent_lines(&consent, &from_file) {
        eprintln!("{line}");
    }
    if !confirm("Create the pod?", args.yes)? {
        return Err(CliError::new(Code::InvalidArguments, "cancelled"));
    }

    let mut network_volume_id = None;
    // A volume this run created: deleted again if the start fails, so a sold-out GPU or a model
    // that does not start never leaves an empty volume billing by the month.
    let mut created_volume: Option<String> = None;
    let mut data_center_ids = Vec::new();
    if let Some((name, dc, existing, size)) = &volume {
        let id = match existing {
            Some(v) => v.id.clone(),
            None => {
                let id = client
                    .create_volume(name, *size, dc)
                    .await
                    .map_err(api_error)?
                    .id;
                created_volume = Some(id.clone());
                id
            }
        };
        network_volume_id = Some(id);
        data_center_ids.push(dc.clone());
    }

    let token = remote::random_hex(32).map_err(|e| CliError::new(Code::Internal, e.to_string()))?;
    let env = pod_env(&PodEnv {
        token: &token,
        model: &model,
        max_model_len,
        max_num_seqs: args.max_num_seqs,
        tool_parser: &tool_parser,
        reasoning_parser: reasoning_parser.as_deref(),
        idle_stop: args.idle_stop,
        max_runtime: args.max_runtime,
        price,
        gpu: &gpu_label,
        source: &source,
        // A CPU pod serves the stub; with another model it starts vLLM, which fails there
        // (the failure path, tested with `--cpu --model <repo>`).
        stub: args.cpu && model == "stub",
        volume: network_volume_id.is_some(),
        vllm_env: offer.map_or(&[], |o| o.vllm_env),
        vllm_args: offer.map_or(&[], |o| o.vllm_args),
        keep_on_error: args.keep_on_error,
    });
    let mut create = PodCreate {
        name: args.name.clone().unwrap_or_else(|| slug(&model)),
        image_name: if args.cpu {
            CPU_IMAGE
        } else {
            offer.and_then(|o| o.image).unwrap_or(VLLM_IMAGE)
        }
        .to_string(),
        compute,
        container_disk_gb: if args.cpu {
            10
        } else if network_volume_id.is_some() {
            40
        } else {
            40 + offer.map_or(60, |o| o.disk_gb())
        },
        volume_gb: 0,
        network_volume_id,
        data_center_ids,
        ports: vec![format!("{AGENT_PORT}/http")],
        env,
        docker_entrypoint: vec![
            "python3".into(),
            "-c".into(),
            "import os; exec(os.environ['MRUN_BOOTSTRAP'])".into(),
        ],
        docker_start_cmd: vec![],
        allowed_cuda_versions: if args.cpu {
            vec![]
        } else {
            CUDA.iter().map(|s| s.to_string()).collect()
        },
        min_ram_per_gpu_gb: offer
            .filter(|o| !args.cpu && o.min_ram_gb > 0)
            .map(|o| o.min_ram_gb),
    };
    debug_assert!(!create.body().to_string().contains("RUNPOD_API_KEY"));
    let deadline = args.retry.map(|d| Instant::now() + d);
    let max_price = args
        .max_price
        .or(candidates.iter().map(|c| c.2).reduce(f64::max))
        .unwrap_or(price);
    // The rest of this round of `--retry`, after the GPU just tried.
    let mut queue: std::collections::VecDeque<(GpuType, CloudType, f64)> =
        candidates.iter().skip(1).cloned().collect();
    let pod = loop {
        let error = match client.create_pod(&create).await {
            Ok(pod) => break pod,
            Err(error) => error,
        };
        if error.is_sold_out() && deadline.is_some_and(|d| Instant::now() < d) {
            info(&format!(
                "{gpu_label} ({} Cloud): sold out",
                cloud_name(cloud)
            ));
            if queue.is_empty() {
                let left = deadline.map_or(Duration::ZERO, |d| d - Instant::now());
                info(&format!(
                    "every GPU that fits up to ${max_price:.2}/h is sold out; next round in {} ({} left)",
                    human_duration(RETRY_POLL.as_secs()),
                    human_duration(left.as_secs())
                ));
                tokio::select! {
                    () = tokio::time::sleep(RETRY_POLL) => {}
                    _ = tokio::signal::ctrl_c() => {
                        drop_created_volume(&client, created_volume.as_deref()).await;
                        return Err(CliError::new(Code::InvalidArguments, "cancelled"));
                    }
                }
                // Fresh prices; on a transient error the last list again.
                if let Ok(gpus) = client.gpu_types().await {
                    candidates = retry_candidates(&args, offer.as_ref(), &gpus, clouds, max_price);
                }
                queue = candidates.iter().cloned().collect();
            }
            if let Some((gpu, c, p)) = queue.pop_front() {
                create.compute = Compute::Gpu {
                    type_id: gpu.id.clone(),
                    cloud: c,
                };
                cloud = c;
                create.env.insert("MRUN_GPU".into(), gpu.id.clone());
                create.env.insert("MRUN_COST_PER_HR".into(), format!("{p}"));
                gpu_label = gpu.id;
                price = p;
                continue;
            }
        }
        {
            let mut error = api_error(error);
            if let (Some(offer), Compute::Gpu { .. }) = (offer.as_ref(), &create.compute)
                && let Ok(gpus) = client.gpu_types().await
            {
                let others: Vec<String> = catalog::alternatives(offer, &gpus, &gpu_label, cloud)
                    .into_iter()
                    .take(3)
                    .map(|(g, c, p)| {
                        let flag = if c == CloudType::Community {
                            " --community"
                        } else {
                            ""
                        };
                        format!("--gpu \"{}\"{flag} (${p:.2}/h)", g.display_name)
                    })
                    .collect();
                if !others.is_empty() {
                    error = error.hint(format!("in stock and large enough: {}", others.join(", ")));
                }
            }
            drop_created_volume(&client, created_volume.as_deref()).await;
            return Err(error);
        }
    };
    let start = Instant::now();
    let created_unix = crate::api::sse::unix_now();
    info(&format!(
        "pod {} created ({}); waiting for it (Ctrl-C terminates it)",
        pod.id, create.name
    ));
    let upload = match source {
        AgentSource::File { bytes, .. } => Some(bytes),
        AgentSource::Url { .. } => None,
    };
    let price = if pod.cost_per_hr > 0.0 {
        pod.cost_per_hr
    } else {
        price
    };
    let base = pod.proxy_url(AGENT_PORT);
    let follower = args.verbose.then(|| {
        let dim = std::io::stderr().is_terminal();
        tokio::spawn(follow_log(
            base.clone(),
            token.clone(),
            0,
            None,
            move |line| {
                if dim {
                    eprintln!("\x1b[2m    {line}\x1b[0m");
                } else {
                    eprintln!("    {line}");
                }
            },
        ))
    });
    let waited = tokio::select! {
        result = wait_ready(&base, &token, upload.as_deref(), start) => result,
        _ = tokio::signal::ctrl_c() => Err(CliError::new(Code::InvalidArguments, "interrupted")),
    };
    if let Some(follower) = follower {
        follower.abort();
    }
    let ready = match waited {
        Ok(()) => start_relay(
            &args,
            &pod,
            &token,
            (&gpu_label, price),
            &model,
            Some(created_unix),
        ),
        Err(error) => Err(error),
    };
    if let Err(error) = ready {
        // The full log first: terminating the pod deletes it.
        save_log(&base, &token, &saved_log_path(&pod.id)).await;
        let keep = args.keep_on_error && error.to_string().contains("failed to start");
        if keep {
            eprintln!(
                "keeping pod {} (--keep-on-error); 'mrun cloud down' terminates it",
                pod.id
            );
        } else {
            match client.terminate_pod(&pod.id).await {
                Ok(()) => {
                    eprintln!("terminated pod {}", pod.id);
                    drop_created_volume(&client, created_volume.as_deref()).await;
                }
                Err(e) => eprintln!(
                    "warning: could not terminate pod {}: {e}; run 'mrun cloud down'",
                    pod.id
                ),
            }
        }
        return Err(error);
    }
    info(&format!(
        "ready after {} (${:.3} so far)",
        human_duration(start.elapsed().as_secs()),
        price * start.elapsed().as_secs_f64() / 3600.0
    ));
    Ok(())
}

/// Waits until the agent serves: pod scheduled → agent up (after the upload with
/// `--agent-file`) → loading (bytes on disk) → ready; fails with the agent's log tail.
async fn wait_ready(
    base: &str,
    token: &str,
    upload: Option<&[u8]>,
    start: Instant,
) -> Result<(), CliError> {
    let http = http();
    let elapsed = || human_duration(start.elapsed().as_secs());
    let mut uploaded = upload.is_none();
    let mut said = String::new();
    let mut last_bytes = 0u64;
    let say = |said: &mut String, line: String| {
        if *said != line {
            info(&format!("  [{}] {line}", elapsed()));
            *said = line;
        }
    };
    say(&mut said, "pod scheduled; pulling the image".into());
    loop {
        if start.elapsed() > READY_TIMEOUT {
            return Err(CliError::new(
                Code::ServerUnreachable,
                format!(
                    "the remote model failed to start within {} minutes",
                    READY_TIMEOUT.as_secs() / 60
                ),
            ));
        }
        let health = agent_get(&http, base, "/health", token).await;
        match health {
            Some((200, body)) if body["status"] == "bootstrap" => {
                if !uploaded && let Some(bytes) = upload {
                    say(
                        &mut said,
                        format!("uploading the agent ({:.1} MB)", bytes.len() as f64 / 1e6),
                    );
                    let response = reqwest::Client::builder()
                        .user_agent(mrun_runpod::USER_AGENT)
                        .timeout(Duration::from_secs(600))
                        .build()
                        .unwrap_or_default()
                        .put(format!("{base}/mrun/agent"))
                        .bearer_auth(token)
                        .body(bytes.to_vec())
                        .send()
                        .await;
                    match response {
                        Ok(r) if r.status().is_success() => uploaded = true,
                        Ok(r) => say(
                            &mut said,
                            format!("agent upload answered {}; retrying", r.status()),
                        ),
                        Err(e) => say(&mut said, format!("agent upload failed ({e}); retrying")),
                    }
                } else {
                    say(&mut said, "fetching the agent".into());
                }
            }
            Some((200, body)) if body["phase"].is_string() => {
                let Some((code, status)) = agent_get(&http, base, "/mrun/status", token).await
                else {
                    tokio::time::sleep(POLL).await;
                    continue;
                };
                if code == 401 {
                    return Err(CliError::new(
                        Code::Internal,
                        "the agent refused the pod token",
                    ));
                }
                match status["phase"].as_str() {
                    Some("ready") => {
                        say(
                            &mut said,
                            format!(
                                "ready: {} ({}, context {})",
                                status["model"].as_str().unwrap_or("?"),
                                status["backend"].as_str().unwrap_or("?"),
                                status["max_model_len"]
                            ),
                        );
                        return Ok(());
                    }
                    Some("failed") => {
                        let tail: Vec<String> = status["log_tail"]
                            .as_array()
                            .map(|lines| {
                                lines
                                    .iter()
                                    .filter_map(Value::as_str)
                                    .map(|l| format!("    {l}"))
                                    .collect()
                            })
                            .unwrap_or_default();
                        eprintln!("{}", tail.join("\n"));
                        return Err(CliError::new(
                            Code::Internal,
                            "the remote model failed to start (vLLM's last lines above)",
                        )
                        .hint("--keep-on-error keeps the pod for inspection"));
                    }
                    Some("stopping") => {
                        return Err(CliError::new(
                            Code::Internal,
                            "the remote model failed to start: the pod is stopping",
                        ));
                    }
                    _ => {
                        let bytes = status["weights_bytes"].as_u64().unwrap_or(0);
                        if bytes == 0 {
                            say(&mut said, "agent up; vLLM starting".into());
                        } else if bytes >= last_bytes + 500_000_000 || said.starts_with("agent up")
                        {
                            last_bytes = bytes;
                            say(
                                &mut said,
                                format!("loading: {:.1} GB of weights on disk", bytes as f64 / 1e9),
                            );
                        }
                    }
                }
            }
            _ => {}
        }
        tokio::time::sleep(POLL).await;
    }
}

/// A client for log reads and follows (no total timeout; a follow's keepalives come every 10 s).
fn stream_http() -> reqwest::Client {
    reqwest::Client::builder()
        .user_agent(mrun_runpod::USER_AGENT)
        .connect_timeout(Duration::from_secs(10))
        .read_timeout(Duration::from_secs(60))
        .build()
        .unwrap_or_default()
}

/// One `GET /mrun/logs?offset=` read: the bytes and the start, next and end offsets.
async fn read_log(
    http: &reqwest::Client,
    base: &str,
    token: &str,
    offset: u64,
) -> Result<(Vec<u8>, u64, u64, u64), String> {
    let response = http
        .get(format!("{base}/mrun/logs?offset={offset}"))
        .bearer_auth(token)
        .send()
        .await
        .map_err(|e| e.to_string())?;
    if !response.status().is_success() {
        return Err(format!("the agent answered {}", response.status()));
    }
    let header = |name: &str| {
        response
            .headers()
            .get(name)
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.parse::<u64>().ok())
    };
    let (Some(start), Some(next), Some(end)) = (
        header("x-mrun-log-start"),
        header("x-mrun-log-offset"),
        header("x-mrun-log-end"),
    ) else {
        return Err("the agent has no log route (an older agent?)".into());
    };
    let bytes = response.bytes().await.map_err(|e| e.to_string())?;
    Ok((bytes.to_vec(), start, next, end))
}

/// The log from `offset` to its end (several reads): the bytes, where they start and where
/// they end.
async fn download_log(
    http: &reqwest::Client,
    base: &str,
    token: &str,
    offset: u64,
) -> Result<(Vec<u8>, u64, u64), String> {
    let (mut out, start, mut next, mut end) = read_log(http, base, token, offset).await?;
    while next < end {
        let (bytes, _, n, e) = read_log(http, base, token, next).await?;
        if bytes.is_empty() {
            break;
        }
        out.extend_from_slice(&bytes);
        (next, end) = (n, e);
    }
    Ok((out, start, next))
}

/// Where a failed start's log is saved: `~/.mrun/logs/cloud-<pod id>.log`.
pub fn saved_log_path(pod_id: &str) -> PathBuf {
    let id: String = pod_id
        .chars()
        .filter(|c| c.is_ascii_alphanumeric() || *c == '-')
        .collect();
    crate::cli::mrun_home()
        .join("logs")
        .join(format!("cloud-{id}.log"))
}

/// Writes `bytes` to `path` readable by the user only (0600; the directory is created).
fn write_private(path: &std::path::Path, bytes: &[u8]) -> std::io::Result<()> {
    use std::io::Write;
    #[cfg(unix)]
    use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let mut options = std::fs::OpenOptions::new();
    options.create(true).write(true).truncate(true);
    #[cfg(unix)]
    options.mode(0o600);
    let mut file = options.open(path)?;
    #[cfg(unix)]
    file.set_permissions(std::fs::Permissions::from_mode(0o600))?;
    file.write_all(bytes)
}

/// The last exception lines of a log (the root cause of a failed start), at most `n`.
fn exception_lines(text: &str, n: usize) -> Vec<String> {
    let lines: Vec<&str> = text
        .lines()
        .filter(|l| crate::agent::vllm::is_error_line(l))
        .collect();
    lines[lines.len().saturating_sub(n)..]
        .iter()
        .map(|l| l.to_string())
        .collect()
}

/// After a failed or interrupted start, before the pod goes: downloads the agent's full log to
/// `path` and prints where it is and its last exception lines. Best effort (at most a minute);
/// returns the path when saved.
pub(crate) async fn save_log(base: &str, token: &str, path: &std::path::Path) -> Option<PathBuf> {
    let http = stream_http();
    let downloaded =
        tokio::time::timeout(Duration::from_secs(60), download_log(&http, base, token, 0)).await;
    let bytes = match downloaded {
        Ok(Ok((bytes, ..))) if !bytes.is_empty() => bytes,
        Ok(Ok(_)) => {
            eprintln!("the pod's log is empty (the agent did not start)");
            return None;
        }
        Ok(Err(error)) => {
            eprintln!("could not download the pod's log: {error}");
            return None;
        }
        Err(_) => {
            eprintln!("could not download the pod's log within a minute");
            return None;
        }
    };
    if let Err(error) = write_private(path, &bytes) {
        eprintln!(
            "could not save the pod's log to {}: {error}",
            path.display()
        );
        return None;
    }
    let text = String::from_utf8_lossy(&bytes);
    eprintln!(
        "full log ({:.1} MB): {}",
        bytes.len() as f64 / 1e6,
        path.display()
    );
    let errors = exception_lines(&text, crate::agent::vllm::ERROR_LINES);
    if !errors.is_empty() {
        eprintln!("last exception lines:");
        for line in errors {
            eprintln!("    {line}");
        }
    }
    Some(path.to_path_buf())
}

/// Follows the log from `offset` (`GET /mrun/logs?follow=1`), calling `line` for every line
/// (keepalives skipped); reconnects where it left off. Returns after `give_up` failed
/// connections in a row (never with `None`).
async fn follow_log(
    base: String,
    token: String,
    mut offset: u64,
    give_up: Option<u32>,
    mut line: impl FnMut(&str) + Send,
) {
    let http = stream_http();
    let mut failures = 0u32;
    loop {
        let response = http
            .get(format!("{base}/mrun/logs?follow=1&offset={offset}"))
            .bearer_auth(&token)
            .send()
            .await;
        match response {
            Ok(mut response) if response.status().is_success() => {
                failures = 0;
                if let Some(start) = response
                    .headers()
                    .get("x-mrun-log-start")
                    .and_then(|v| v.to_str().ok())
                    .and_then(|v| v.parse().ok())
                {
                    offset = start;
                }
                let mut pending: Vec<u8> = Vec::new();
                while let Ok(Some(chunk)) = response.chunk().await {
                    pending.extend_from_slice(&chunk);
                    while let Some(i) = pending.iter().position(|&b| b == b'\n') {
                        let raw: Vec<u8> = pending.drain(..=i).collect();
                        // An empty line is a keepalive.
                        if raw.len() > 1 {
                            offset += raw.len() as u64;
                            line(String::from_utf8_lossy(&raw[..raw.len() - 1]).trim_end());
                        }
                    }
                }
            }
            _ => failures += 1,
        }
        if give_up.is_some_and(|n| failures >= n) {
            return;
        }
        tokio::time::sleep(Duration::from_secs(2)).await;
    }
}

/// `mrun cloud logs`: the running pod's log (all, or the last N lines), then with `-f` new
/// lines until Ctrl-C.
async fn logs(args: LogsArgs) -> Result<(), CliError> {
    let client = client()?;
    let pods = mrun_pods(&client).await?;
    let Some(pod) = pods.iter().find(|p| p.is_running()) else {
        return Err(CliError::new(
            Code::InvalidArguments,
            if pods.is_empty() {
                "no mrun pod".to_string()
            } else {
                format!("pod {} is not running", pods[0].id)
            },
        )
        .hint("a failed start's log is saved in ~/.mrun/logs/"));
    };
    let token = pod.env.get("MRUN_POD_TOKEN").cloned().unwrap_or_default();
    let base = pod.proxy_url(AGENT_PORT);
    let http = stream_http();
    let unreachable = |error: String| {
        CliError::new(
            Code::ServerUnreachable,
            format!("cannot read the log of pod {}: {error}", pod.id),
        )
        .hint("the agent may still be starting; 'mrun cloud status' shows its phase")
    };
    let (_, _, end, _) = read_log(&http, &base, &token, u64::MAX)
        .await
        .map_err(unreachable)?;
    let (text, next) = match args.tail {
        Some(n) => tail_lines(&http, &base, &token, end, n)
            .await
            .map_err(unreachable)?,
        None => {
            let (bytes, _, next) = download_log(&http, &base, &token, 0)
                .await
                .map_err(unreachable)?;
            (String::from_utf8_lossy(&bytes).into_owned(), next)
        }
    };
    {
        use std::io::Write;
        let mut out = std::io::stdout().lock();
        let _ = out.write_all(text.as_bytes());
        let _ = out.flush();
    }
    if args.follow {
        let follow = follow_log(base.clone(), token, next, Some(5), |line| {
            println!("{line}")
        });
        tokio::select! {
            () = follow => info("the pod's log is no longer reachable"),
            _ = tokio::signal::ctrl_c() => {}
        }
    }
    Ok(())
}

/// The last `n` lines (up to `end` or a little later): reads a window that grows until it
/// holds them. Returns them and the offset after them.
async fn tail_lines(
    http: &reqwest::Client,
    base: &str,
    token: &str,
    end: u64,
    n: usize,
) -> Result<(String, u64), String> {
    let mut window = (n as u64 * 256).max(64 * 1024);
    loop {
        let from = end.saturating_sub(window);
        let (bytes, start, next) = download_log(http, base, token, from).await?;
        let text = String::from_utf8_lossy(&bytes).into_owned();
        let complete = start > from || from == 0;
        let lines: Vec<&str> = text.lines().collect();
        // The first line of a window is partial unless the window starts at the log's start.
        let usable = if complete {
            lines.len()
        } else {
            lines.len().saturating_sub(1)
        };
        if usable >= n || complete {
            let keep = &lines[lines.len() - usable.min(n)..];
            let mut out = keep.join("\n");
            if !out.is_empty() {
                out.push('\n');
            }
            return Ok((out, next));
        }
        window *= 4;
    }
}

/// What the pod's environment carries (data model: pod env).
struct PodEnv<'a> {
    token: &'a str,
    model: &'a str,
    max_model_len: u32,
    max_num_seqs: u32,
    tool_parser: &'a str,
    reasoning_parser: Option<&'a str>,
    idle_stop: Duration,
    max_runtime: Duration,
    price: f64,
    gpu: &'a str,
    source: &'a AgentSource,
    stub: bool,
    volume: bool,
    /// The model's own vLLM environment and flags (catalog).
    vllm_env: &'a [(&'static str, &'static str)],
    vllm_args: &'a [&'static str],
    /// `--keep-on-error`: a failed pod is not stopped early (only the idle stop applies).
    keep_on_error: bool,
}

/// The pod's environment. It never holds the account key (FR-001): RunPod injects the pod's
/// own scoped key for self-stop.
fn pod_env(spec: &PodEnv) -> BTreeMap<String, String> {
    let mut env = BTreeMap::from([
        ("MRUN_POD_TOKEN".to_string(), spec.token.to_string()),
        ("MRUN_MODEL".to_string(), spec.model.to_string()),
        (
            "MRUN_MAX_MODEL_LEN".to_string(),
            spec.max_model_len.to_string(),
        ),
        (
            "MRUN_MAX_NUM_SEQS".to_string(),
            spec.max_num_seqs.to_string(),
        ),
        ("MRUN_TOOL_PARSER".to_string(), spec.tool_parser.to_string()),
        (
            "MRUN_IDLE_STOP".to_string(),
            format!("{}s", spec.idle_stop.as_secs()),
        ),
        (
            "MRUN_MAX_RUNTIME".to_string(),
            format!("{}s", spec.max_runtime.as_secs()),
        ),
        ("MRUN_COST_PER_HR".to_string(), format!("{}", spec.price)),
        ("MRUN_GPU".to_string(), spec.gpu.to_string()),
        ("MRUN_AGENT_PORT".to_string(), AGENT_PORT.to_string()),
        (
            "MRUN_AGENT_SHA256".to_string(),
            spec.source.sha256().to_string(),
        ),
        (
            "MRUN_BOOTSTRAP".to_string(),
            BOOTSTRAP.replace("@USER_AGENT@", mrun_runpod::USER_AGENT),
        ),
    ]);
    if let Some(parser) = spec.reasoning_parser {
        env.insert("MRUN_REASONING_PARSER".into(), parser.to_string());
    }
    if let AgentSource::Url { url, .. } = spec.source {
        env.insert("MRUN_AGENT_URL".into(), url.clone());
    }
    if spec.stub {
        env.insert("MRUN_AGENT_STUB".into(), "true".into());
    }
    if spec.volume {
        env.insert("HF_HOME".into(), "/workspace/hf".into());
    }
    if spec.keep_on_error {
        env.insert("MRUN_FAILED_STOP".into(), "0s".into());
    }
    for (name, value) in spec.vllm_env {
        env.insert(name.to_string(), value.to_string());
    }
    if !spec.vllm_args.is_empty() {
        env.insert(
            "MRUN_VLLM_ARGS".into(),
            serde_json::to_string(spec.vllm_args).unwrap_or_default(),
        );
    }
    env
}

/// Starts `mrun serve --relay` in the background with the token in its environment only.
fn start_relay(
    args: &UpArgs,
    pod: &Pod,
    token: &str,
    (gpu, price): (&str, f64),
    model: &str,
    since_unix: Option<u64>,
) -> Result<(), CliError> {
    let url = pod.proxy_url(AGENT_PORT);
    let since = since_unix.map(|s| s.to_string());
    let mut child_args: Vec<std::ffi::OsString> = [
        "serve",
        "--relay",
        &url,
        "--port",
        &args.port.to_string(),
        "--relay-pod",
        &pod.id,
        "--relay-gpu",
        gpu,
        "--relay-cost",
        &format!("{price}"),
    ]
    .iter()
    .map(Into::into)
    .collect();
    if let Some(since) = &since {
        child_args.extend(["--relay-since".into(), since.into()]);
    }
    crate::cli::serve::start_detached(
        args.port,
        &child_args,
        &format!("{model} on {gpu}"),
        &[(crate::api::relay::TOKEN_VARIABLE, token)],
    )
    .map_err(|error| match error {
        crate::cli::serve::StartError::Error(error) => error,
        crate::cli::serve::StartError::Exit(code) => CliError::new(
            Code::Internal,
            format!("the local relay exited with code {code}"),
        ),
    })?;
    info(&format!(
        "  remote: {} on pod {} (${price:.2}/h; stops itself after {} idle); 'mrun cloud down' terminates it",
        model,
        pod.id,
        human_duration(args.idle_stop.as_secs())
    ));
    Ok(())
}

async fn down(args: DownArgs) -> Result<(), CliError> {
    let client = client()?;
    if let Some(url) = local_server(args.port)
        && !url.is_empty()
    {
        crate::cli::stop::stop(args.port)?;
        info(&format!(
            "stopped the local relay on 127.0.0.1:{}",
            args.port
        ));
    }
    let pods = mrun_pods(&client).await?;
    if pods.is_empty() {
        info("no mrun pod");
    }
    for pod in &pods {
        client.terminate_pod(&pod.id).await.map_err(api_error)?;
        info(&format!("terminated pod {} ({})", pod.id, pod.name));
    }
    if args.volume {
        let volumes: Vec<_> = client
            .list_volumes()
            .await
            .map_err(api_error)?
            .into_iter()
            .filter(|v| v.name.starts_with(PREFIX))
            .collect();
        for volume in volumes {
            let question = format!(
                "Delete network volume {} ({} GB in {}, the cached weights)?",
                volume.name, volume.size_gb, volume.data_center_id
            );
            if confirm(&question, args.yes)? {
                client.delete_volume(&volume.id).await.map_err(api_error)?;
                info(&format!("deleted volume {}", volume.name));
            }
        }
    }
    let left = mrun_pods(&client).await?;
    if !left.is_empty() {
        return Err(CliError::new(
            Code::Network,
            format!("{} mrun pod(s) still listed", left.len()),
        )
        .hint("run 'mrun cloud down' again or terminate them in the RunPod console"));
    }
    Ok(())
}

async fn status(args: StatusArgs) -> Result<(), CliError> {
    let client = client()?;
    let pods = mrun_pods(&client).await?;
    let http = http();
    let mut out = Vec::new();
    for pod in &pods {
        let token = pod.env.get("MRUN_POD_TOKEN").cloned().unwrap_or_default();
        let agent = if pod.is_running() {
            agent_get(&http, &pod.proxy_url(AGENT_PORT), "/mrun/status", &token)
                .await
                .filter(|(code, _)| *code == 200)
                .map(|(_, body)| body)
        } else {
            None
        };
        out.push(json!({
            "id": pod.id,
            "name": pod.name,
            "state": pod.desired_status,
            "gpu": pod_compute(pod),
            "cost_per_hr": pod.cost_per_hr,
            "status_change": pod.last_status_change,
            "model": pod.env.get("MRUN_MODEL"),
            "url": pod.proxy_url(AGENT_PORT),
            "agent": agent,
        }));
    }
    let relay = local_server(args.port).filter(|u| !u.is_empty());
    if args.json {
        println!(
            "{}",
            serde_json::to_string_pretty(&json!({"pods": out, "relay": relay})).unwrap_or_default()
        );
        return Ok(());
    }
    if out.is_empty() {
        println!("no mrun pod");
    }
    for pod in &out {
        let agent = &pod["agent"];
        println!(
            "pod       {} ({}) {} on {}, ${:.2}/h",
            pod["id"].as_str().unwrap_or("?"),
            pod["name"].as_str().unwrap_or("?"),
            pod["state"].as_str().unwrap_or("?"),
            pod["gpu"].as_str().unwrap_or("GPU unknown"),
            pod["cost_per_hr"].as_f64().unwrap_or(0.0),
        );
        // A stopped pod bills nothing for its GPU; RunPod's own note says when and why it stopped.
        if pod["state"] != "RUNNING"
            && let Some(change) = pod["status_change"].as_str()
        {
            println!("stopped   {change}");
        }
        println!("model     {}", pod["model"].as_str().unwrap_or("?"));
        if agent.is_null() {
            println!("agent     not reachable (starting, stopped or not ready)");
        } else {
            println!(
                "agent     {} ({}), up {}, spend ${:.3}",
                agent["phase"].as_str().unwrap_or("?"),
                agent["backend"].as_str().unwrap_or("?"),
                human_duration(agent["uptime_s"].as_u64().unwrap_or(0)),
                agent["spend_usd"].as_f64().unwrap_or(0.0),
            );
            if let Some(idle) = agent["idle_s"].as_u64() {
                println!(
                    "idle      {} of {} (then the pod stops itself)",
                    human_duration(idle),
                    human_duration(agent["idle_stop_s"].as_u64().unwrap_or(0))
                );
            }
        }
    }
    match relay {
        Some(url) => println!("relay     http://127.0.0.1:{} → {url}", args.port),
        None if !out.is_empty() => {
            println!("relay     not running ('mrun cloud up --model <same model>' attaches)")
        }
        None => {}
    }
    if !out.is_empty() {
        println!("note      {}", crate::api::relay::PRIVACY_NOTE);
    }
    Ok(())
}

async fn gpus(args: ListArgs) -> Result<(), CliError> {
    let client = client()?;
    let mut gpus = client.gpu_types().await.map_err(api_error)?;
    gpus.retain(|g| g.lowest_price().is_some());
    gpus.sort_by(|a, b| {
        a.lowest_price()
            .unwrap_or(f64::MAX)
            .total_cmp(&b.lowest_price().unwrap_or(f64::MAX))
    });
    if args.json {
        let rows: Vec<Value> = gpus
            .iter()
            .map(|g| {
                json!({"id": g.id, "name": g.display_name, "memory_gb": g.memory_gb,
                "secure_per_hr": g.price(CloudType::Secure),
                "community_per_hr": g.price(CloudType::Community),
                "secure_stock": g.stock(CloudType::Secure),
                "community_stock": g.stock(CloudType::Community)})
            })
            .collect();
        println!(
            "{}",
            serde_json::to_string_pretty(&rows).unwrap_or_default()
        );
        return Ok(());
    }
    println!(
        "{:<20} {:>6}  {:<16} {:<16}",
        "GPU", "VRAM", "Secure", "Community"
    );
    let cell = |g: &GpuType, cloud: CloudType| match g.price(cloud) {
        None => "—".to_string(),
        Some(p) => format!(
            "${p:.2}/h ({})",
            g.stock(cloud)
                .map_or("none".to_string(), str::to_ascii_lowercase)
        ),
    };
    for g in &gpus {
        println!(
            "{:<20} {:>4}GB  {:<16} {:<16}",
            g.display_name,
            g.memory_gb,
            cell(g, CloudType::Secure),
            cell(g, CloudType::Community)
        );
    }
    println!("(stock now: high, medium, low, or none available)");
    Ok(())
}

async fn models(args: ListArgs) -> Result<(), CliError> {
    // Prices need the key; without one the list shows VRAM only.
    let gpus = match key() {
        Ok(key) => match Client::new(key) {
            Ok(client) => client.gpu_types().await.unwrap_or_default(),
            Err(_) => Vec::new(),
        },
        Err(_) => Vec::new(),
    };
    let cheapest = |offer: &Offer| {
        let secure = catalog::fitting(offer, &gpus, CloudType::Secure);
        let community = catalog::fitting(offer, &gpus, CloudType::Community);
        (
            secure.first().map(|(g, p)| (g.display_name.clone(), *p)),
            community.first().map(|(g, p)| (g.display_name.clone(), *p)),
        )
    };
    if args.json {
        let rows: Vec<Value> = catalog::OFFERS
            .iter()
            .map(|o| {
                let (secure, community) = cheapest(o);
                json!({"repo": o.repo, "weights_bytes": o.weights_bytes,
                    "context": o.max_model_len, "min_vram_gb": o.min_vram_gib().ceil(),
                    "tool_parser": o.tool_parser, "reasoning_parser": o.reasoning_parser,
                    "secure": secure, "community": community, "note": o.note})
            })
            .collect();
        println!(
            "{}",
            serde_json::to_string_pretty(&rows).unwrap_or_default()
        );
        return Ok(());
    }
    for offer in &catalog::OFFERS {
        let (secure, community) = cheapest(offer);
        let fmt = |c: Option<(String, f64)>| {
            c.map_or("—".to_string(), |(g, p)| format!("{g} ${p:.2}/h"))
        };
        println!(
            "{}\n  {:.1} GB weights, context {}, needs ≥ {:.0} GB; Secure: {}; Community: {}\n  {}",
            offer.repo,
            offer.weights_bytes as f64 / 1e9,
            offer.max_model_len,
            offer.min_vram_gib().ceil(),
            fmt(secure),
            fmt(community),
            offer.note
        );
    }
    if gpus.is_empty() {
        println!("(set RUNPOD_API_KEY to see the cheapest fitting GPU and its price)");
    }
    Ok(())
}

/// Deletes the network volume a failed `cloud up` created (an existing one is never touched).
/// RunPod may still see a just-terminated pod attached, so the delete is retried a few times.
async fn drop_created_volume(client: &Client, id: Option<&str>) {
    let Some(id) = id else { return };
    for attempt in 0..5 {
        match client.delete_volume(id).await {
            Ok(()) => {
                eprintln!("deleted the volume this run created ({id})");
                return;
            }
            Err(e) if attempt == 4 => {
                eprintln!(
                    "warning: could not delete volume {id}: {e}; 'mrun cloud down --volume' deletes it"
                );
            }
            Err(_) => tokio::time::sleep(std::time::Duration::from_secs(5)).await,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn consent() -> Consent<'static> {
        Consent {
            model: "Qwen/Qwen3-Coder-Next-FP8",
            gpu: "NVIDIA RTX PRO 6000, Community Cloud, $1.69/h ($13.52 per 8 h)".into(),
            idle_stop: Duration::from_secs(1800),
            max_runtime: Duration::from_secs(8 * 3600),
            max_model_len: 65_536,
            max_num_seqs: 16,
            host_ram: None,
            image: None,
            volume: None,
            retry: Some((
                Duration::from_secs(3600),
                vec!["RTX PRO 6000 ($1.69/h)".into()],
                true,
            )),
        }
    }

    #[test]
    fn consent_lines_mark_values_from_the_config_file() {
        let none = consent_lines(&consent(), &|_| false);
        assert!(none.iter().all(|l| !l.contains("(config)")), "{none:?}");
        assert_eq!(none[0], "mrun cloud up: Qwen/Qwen3-Coder-Next-FP8");
        assert_eq!(
            none[2],
            "  limits    stops itself after 30m idle, after 8h00m at most"
        );
        let from_file = |arg: &str| matches!(arg, "idle_stop" | "retry");
        let marked = consent_lines(&consent(), &from_file);
        let ends = |prefix: &str| {
            marked
                .iter()
                .find(|l| l.trim_start().starts_with(prefix))
                .unwrap()
                .ends_with("(config)")
        };
        assert!(ends("limits"));
        assert!(ends("retry"));
        assert!(!ends("GPU"));
        assert!(!ends("context"));
        assert!(
            marked
                .last()
                .unwrap()
                .contains("prompts and code are sent to RunPod")
        );
    }

    #[test]
    fn a_gpu_pod_is_never_labelled_cpu() {
        let mut pod = Pod {
            gpu: Some(mrun_runpod::PodGpu {
                id: "NVIDIA RTX PRO 6000 Blackwell Workstation Edition".into(),
                ..Default::default()
            }),
            ..Pod::default()
        };
        assert_eq!(
            pod_compute(&pod),
            "NVIDIA RTX PRO 6000 Blackwell Workstation Edition"
        );
        // Stopped: RunPod omits the GPU; the one `cloud up` chose is in the pod's env.
        pod.gpu = None;
        pod.env.insert(
            "MRUN_GPU".into(),
            "NVIDIA RTX PRO 6000 Blackwell Server Edition".into(),
        );
        assert_eq!(
            pod_compute(&pod),
            "NVIDIA RTX PRO 6000 Blackwell Server Edition"
        );
        pod.env.clear();
        assert_eq!(pod_compute(&pod), "GPU unknown");
        pod.env
            .insert("MRUN_GPU".into(), "CPU (cpu3c, 2 vCPU)".into());
        assert_eq!(pod_compute(&pod), "CPU");
        let cpu = Pod {
            cpu_flavor_id: Some("cpu3c".into()),
            ..Pod::default()
        };
        assert_eq!(pod_compute(&cpu), "CPU");
    }

    #[test]
    fn the_pod_env_carries_the_token_but_never_the_account_key() {
        let source = AgentSource::File {
            bytes: vec![],
            sha256: "ab".repeat(32),
        };
        let env = pod_env(&PodEnv {
            token: "t0ken",
            model: "Qwen/Qwen3-4B-Instruct-2507",
            max_model_len: 65_536,
            max_num_seqs: 16,
            tool_parser: "hermes",
            reasoning_parser: None,
            idle_stop: Duration::from_secs(600),
            max_runtime: Duration::from_secs(3600),
            price: 0.34,
            gpu: "NVIDIA GeForce RTX 4090",
            source: &source,
            stub: false,
            volume: false,
            vllm_env: &[("VLLM_PLE_CPU_OFFLOAD", "1")],
            vllm_args: &["--gpu-memory-utilization", "0.92"],
            keep_on_error: false,
        });
        assert_eq!(env["MRUN_POD_TOKEN"], "t0ken");
        assert_eq!(env["MRUN_IDLE_STOP"], "600s");
        assert_eq!(env["MRUN_MAX_NUM_SEQS"], "16");
        assert_eq!(env["MRUN_AGENT_SHA256"], "ab".repeat(32));
        assert!(!env.contains_key("MRUN_AGENT_URL") && !env.contains_key("MRUN_AGENT_STUB"));
        assert!(!env.contains_key("RUNPOD_API_KEY"));
        // The values never carry a key either (the account key is only inside `Client`).
        let all: String = env.values().cloned().collect();
        assert!(!all.contains("rpa_"), "no RunPod key format in the pod env");
        assert!(env["MRUN_BOOTSTRAP"].contains(mrun_runpod::USER_AGENT));
        assert_eq!(env["VLLM_PLE_CPU_OFFLOAD"], "1");
        assert!(
            !env.contains_key("MRUN_FAILED_STOP"),
            "the agent's default applies"
        );
        assert_eq!(
            env["MRUN_VLLM_ARGS"],
            r#"["--gpu-memory-utilization","0.92"]"#
        );
    }

    const TOKEN: &str = "0123456789abcdef0123456789abcdef";

    /// The log a failed vLLM start leaves: the root cause, then a long traceback.
    fn failed_log() -> String {
        let mut text = String::new();
        let stamp = "2026-10-19T09:02:22.123Z";
        text.push_str(&format!("{stamp} [agent] starting vllm serve x\n"));
        text.push_str(&format!(
            "{stamp} (EngineCore pid=1) ERROR 10-19 09:02:22 [core.py:1] ValueError: 9.4 GiB KV cache is needed\n"
        ));
        for i in 0..300 {
            text.push_str(&format!("{stamp}   File \"/x.py\", line {i}, in f\n"));
        }
        text.push_str(&format!(
            "{stamp} RuntimeError: Engine core initialization failed. See root cause above.\n"
        ));
        text.push_str(&format!("{stamp} [agent] vllm exited (exit status: 1)\n"));
        text
    }

    /// A mock agent whose vLLM failed: `/health` and `/mrun/status` say so, `/mrun/logs` pages
    /// `log` 1000 bytes at a time (token required).
    async fn failed_agent(log: String) -> String {
        use axum::extract::{Query, State};
        use axum::http::{HeaderMap, StatusCode};
        use axum::response::IntoResponse;
        use axum::routing::get;
        async fn logs(
            State(log): State<std::sync::Arc<String>>,
            headers: HeaderMap,
            Query(query): Query<BTreeMap<String, String>>,
        ) -> axum::response::Response {
            if headers.get("authorization").and_then(|v| v.to_str().ok())
                != Some(&format!("Bearer {TOKEN}"))
            {
                return StatusCode::UNAUTHORIZED.into_response();
            }
            let len = log.len() as u64;
            let start = query
                .get("offset")
                .and_then(|o| o.parse::<u64>().ok())
                .unwrap_or(0)
                .min(len);
            let next = (start + 1000).min(len);
            (
                [
                    ("x-mrun-log-start", start.to_string()),
                    ("x-mrun-log-offset", next.to_string()),
                    ("x-mrun-log-end", len.to_string()),
                ],
                log[start as usize..next as usize].to_string(),
            )
                .into_response()
        }
        let app = axum::Router::new()
            .route(
                "/health",
                get(|| async { axum::Json(json!({"status": "failed", "phase": "failed"})) }),
            )
            .route(
                "/mrun/status",
                get(|| async {
                    axum::Json(json!({"phase": "failed", "log_tail": ["RuntimeError: Engine core initialization failed."]}))
                }),
            )
            .route("/mrun/logs", get(logs))
            .with_state(std::sync::Arc::new(log));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await });
        format!("http://{addr}")
    }

    #[tokio::test]
    async fn a_failed_start_saves_the_full_log_before_the_pod_goes() {
        let log = failed_log();
        assert!(log.len() > 10_000, "several pages");
        let base = failed_agent(log.clone()).await;
        let error = wait_ready(&base, TOKEN, None, Instant::now())
            .await
            .unwrap_err();
        assert!(error.to_string().contains("failed to start"), "{error}");
        // What `up` does next, before terminating the pod.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("logs/cloud-pod1.log");
        assert_eq!(save_log(&base, TOKEN, &path).await, Some(path.clone()));
        assert_eq!(
            std::fs::read_to_string(&path).unwrap(),
            log,
            "the whole log"
        );
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = std::fs::metadata(&path).unwrap().permissions().mode();
            assert_eq!(mode & 0o777, 0o600);
        }
        // The root cause is among the exception lines printed, the generic summary is not.
        let errors = exception_lines(&log, crate::agent::vllm::ERROR_LINES);
        assert_eq!(errors.len(), 1, "{errors:?}");
        assert!(errors[0].contains("ValueError: 9.4 GiB KV cache is needed"));
        // A wrong token saves nothing.
        let other = dir.path().join("logs/other.log");
        assert_eq!(save_log(&base, "wrong", &other).await, None);
        assert!(!other.exists());
        // `cloud logs --tail 2`: the last two lines, across pages.
        let http = stream_http();
        let (tail, next) = tail_lines(&http, &base, TOKEN, log.len() as u64, 2)
            .await
            .unwrap();
        let want: Vec<&str> = log.lines().rev().take(2).collect();
        assert_eq!(tail, format!("{}\n{}\n", want[1], want[0]));
        assert_eq!(next, log.len() as u64);
        assert!(saved_log_path("abc123/../x").ends_with("logs/cloud-abc123x.log"));
    }

    #[test]
    fn the_starting_marker_names_model_and_pid_and_goes_with_up() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("cloud-starting-8484.json");
        {
            let _mark = StartingMark::at(path.clone(), "Qwen/Qwen3-Coder-Next-FP8");
            assert_eq!(
                read_starting(&path),
                Some(Starting {
                    model: "Qwen/Qwen3-Coder-Next-FP8".into(),
                    pid: std::process::id(),
                })
            );
        }
        assert!(!path.exists(), "removed when up returns");
        std::fs::write(&path, "{}").unwrap();
        assert_eq!(read_starting(&path), None);
    }

    #[test]
    fn pod_names_are_slugs_with_the_prefix() {
        assert_eq!(slug("Qwen/Qwen3.6-35B-A3B-FP8"), "mrun-qwen3-6-35b-a3b-fp8");
        assert_eq!(slug("stub"), "mrun-stub");
    }

    #[test]
    fn the_bootstrap_checks_the_digest_and_never_needs_the_account_key() {
        assert!(BOOTSTRAP.contains("MRUN_AGENT_SHA256"));
        assert!(BOOTSTRAP.contains("compare_digest"));
        assert!(BOOTSTRAP.contains("@USER_AGENT@"));
        assert!(!BOOTSTRAP.contains("RUNPOD_API_KEY"));
        assert!(VLLM_IMAGE.contains("@sha256:"));
    }

    #[test]
    fn a_too_small_gpu_is_refused_naming_the_smallest_that_fits() {
        let gpus = vec![
            GpuType {
                id: "NVIDIA GeForce RTX 4090".into(),
                display_name: "RTX 4090".into(),
                memory_gb: 24,
                secure_cloud: true,
                community_cloud: true,
                secure_price: 0.89,
                community_price: 0.34,
                ..GpuType::default()
            },
            GpuType {
                id: "NVIDIA L40S".into(),
                display_name: "L40S".into(),
                memory_gb: 48,
                secure_cloud: true,
                community_cloud: true,
                secure_price: 1.09,
                community_price: 0.79,
                ..GpuType::default()
            },
        ];
        let mut args = UpArgs {
            model: "Qwen/Qwen3.6-35B-A3B-FP8".into(),
            gpu: Some("RTX 4090".into()),
            community: false,
            idle_stop: Duration::from_secs(1800),
            max_runtime: Duration::from_secs(28_800),
            max_model_len: None,
            max_num_seqs: 16,
            tool_parser: None,
            reasoning_parser: None,
            port: 8484,
            volume_dc: None,
            retry: None,
            max_price: None,
            any_cloud: false,
            keep_on_error: false,
            verbose: false,
            agent_url: None,
            agent_sha256: None,
            agent_file: None,
            yes: true,
            cpu: false,
            name: None,
        };
        let (offer, len, tool, reasoning) = offer_for(&args);
        assert_eq!(
            (len, tool.as_str(), reasoning.as_deref()),
            (131_072, "qwen3_coder", Some("qwen3"))
        );
        let error = choose_gpu(&args, offer.as_ref(), &gpus, CloudType::Secure).unwrap_err();
        let text = format!("{error:?}");
        assert!(text.contains("L40S"), "{text}");
        args.gpu = None;
        let (gpu, price) = choose_gpu(&args, offer.as_ref(), &gpus, CloudType::Community).unwrap();
        assert_eq!((gpu.display_name.as_str(), price), ("L40S", 0.79));
        args.gpu = Some("H200".into());
        assert!(choose_gpu(&args, offer.as_ref(), &gpus, CloudType::Secure).is_err());
    }

    #[test]
    fn retry_tries_every_fitting_gpu_up_to_the_price_cheapest_first() {
        #[derive(clap::Parser)]
        struct Up {
            #[command(flatten)]
            args: UpArgs,
        }
        let parse = |argv: &[&str]| {
            use clap::Parser;
            Up::try_parse_from(argv).map(|up| up.args)
        };
        let gpu = |name: &str, memory_gb: u32, secure: f64| GpuType {
            id: format!("NVIDIA {name}"),
            display_name: name.into(),
            memory_gb,
            secure_cloud: true,
            secure_price: secure,
            ..GpuType::default()
        };
        let community = GpuType {
            community_cloud: true,
            community_price: 1.69,
            ..gpu("RTX PRO 6000 Blackwell Workstation Edition", 96, 2.59)
        };
        let gpus = vec![
            gpu("H100 NVL", 94, 3.19),
            community,
            gpu("RTX PRO 6000 Blackwell Server Edition", 96, 2.49),
            gpu("L40S", 48, 1.09),
        ];
        let mut args = parse(&[
            "up",
            "--model",
            "primitive-ai/Qwen3.8-Flash-Next-NVFP4",
            "--retry",
            "2h",
            "--max-price",
            "2.6",
        ])
        .unwrap();
        let offer = catalog::find(&args.model);
        let names = |list: Vec<(GpuType, CloudType, f64)>| -> Vec<String> {
            list.into_iter().map(|(g, _, _)| g.display_name).collect()
        };
        // Too small (L40S), not Blackwell (H100 NVL): left out; the rest cheapest first.
        assert_eq!(
            names(retry_candidates(
                &args,
                offer.as_ref(),
                &gpus,
                &[CloudType::Secure],
                2.6
            )),
            [
                "RTX PRO 6000 Blackwell Server Edition",
                "RTX PRO 6000 Blackwell Workstation Edition"
            ]
        );
        assert_eq!(
            retry_candidates(&args, offer.as_ref(), &gpus, &[CloudType::Secure], 2.5).len(),
            1,
            "the price limit holds"
        );
        args.gpu = Some("RTX PRO 6000 Blackwell Workstation Edition".into());
        assert_eq!(
            names(retry_candidates(
                &args,
                offer.as_ref(),
                &gpus,
                &[CloudType::Secure],
                2.6
            )),
            ["RTX PRO 6000 Blackwell Workstation Edition"],
            "a named GPU is the only one tried"
        );
        // Both clouds in one run, cheapest first.
        let both = retry_candidates(
            &args,
            offer.as_ref(),
            &gpus,
            &[CloudType::Secure, CloudType::Community],
            2.5,
        );
        let both: Vec<(CloudType, f64)> = both.into_iter().map(|(_, c, p)| (c, p)).collect();
        assert_eq!(
            both,
            [(CloudType::Community, 1.69)],
            "the named GPU in both clouds, Secure's $2.59 above the limit"
        );
        assert!(
            parse(&["up", "--model", "m", "--max-price", "2"]).is_err(),
            "--max-price needs --retry"
        );
        assert!(parse(&["up", "--model", "m", "--any-cloud"]).is_err());
    }

    #[tokio::test]
    async fn a_volume_created_by_a_failed_up_is_deleted_and_an_existing_one_is_kept() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let calls = std::sync::Arc::new(AtomicUsize::new(0));
        let seen = calls.clone();
        // RunPod may still see the just-terminated pod attached: the first delete is refused.
        let app = axum::Router::new().route(
            "/networkvolumes/{id}",
            axum::routing::delete(move || {
                let n = seen.fetch_add(1, Ordering::SeqCst);
                async move {
                    if n == 0 {
                        axum::http::StatusCode::CONFLICT
                    } else {
                        axum::http::StatusCode::NO_CONTENT
                    }
                }
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await });
        let endpoints = mrun_runpod::Endpoints {
            rest: format!("http://{addr}"),
            ..mrun_runpod::Endpoints::default()
        };
        let client = Client::with_endpoints(ApiKey::new("test"), endpoints).unwrap();
        drop_created_volume(&client, None).await;
        assert_eq!(
            calls.load(Ordering::SeqCst),
            0,
            "an existing volume is never deleted"
        );
        drop_created_volume(&client, Some("vol1")).await;
        assert_eq!(
            calls.load(Ordering::SeqCst),
            2,
            "retried after the refused delete"
        );
    }
}
