//! `mrun doctor`: checks the machine, the models, the server and the clients, each with a fix
//! (004 FR-007, research R6, D037). Exit 7 when any check fails. Checks are pure functions of a
//! [`Context`], so each one is tested with a fake.

use crate::cli::clients::{self, Client};
use crate::cli::server;
use crate::errors::Exit;
use crate::hub::{self, aliases, human_bytes};
use mrun_engine::Binding;
use mrun_engine::budget::{self, DeviceMemory, GIB, Overrides};
use mrun_engine::sysmem::{self, Process};
use std::path::{Path, PathBuf};

#[derive(Debug, clap::Args)]
pub struct DoctorArgs {
    /// Port of the server.
    #[arg(long, env = "MRUN_PORT", default_value_t = server::default_port())]
    pub port: u16,
    /// Print a paste-ready Markdown block for a hardware report (D060): version, chip, memory,
    /// numerics, pulled models, calibration and the findings, with no user paths.
    #[arg(long)]
    pub report: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Status {
    Pass,
    /// Worth knowing, nothing to fix.
    Info,
    Warn,
    Fail,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Finding {
    pub id: &'static str,
    pub status: Status,
    pub message: String,
    pub fix: Option<String>,
}

impl Finding {
    fn pass(id: &'static str, message: impl Into<String>) -> Self {
        Self {
            id,
            status: Status::Pass,
            message: message.into(),
            fix: None,
        }
    }

    fn info(id: &'static str, message: impl Into<String>) -> Self {
        Self {
            status: Status::Info,
            ..Self::pass(id, message)
        }
    }

    fn warn(id: &'static str, message: impl Into<String>, fix: impl Into<String>) -> Self {
        Self {
            id,
            status: Status::Warn,
            message: message.into(),
            fix: Some(fix.into()),
        }
    }

    fn fail(id: &'static str, message: impl Into<String>, fix: impl Into<String>) -> Self {
        Self {
            status: Status::Fail,
            ..Self::warn(id, message, fix)
        }
    }
}

/// Who holds the server port.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PortState {
    Free,
    Mrun,
    Other,
}

/// Everything the checks look at.
#[derive(Clone, Debug)]
pub struct Context {
    pub os: &'static str,
    pub arch: &'static str,
    pub memory: Option<DeviceMemory>,
    /// macOS memory pressure level (1 normal, 2 warning, 4 critical).
    pub pressure: Option<u32>,
    pub low_power: Option<bool>,
    pub port: u16,
    pub port_state: PortState,
    /// `Some(loaded)` when the server answers `/health`.
    pub server: Option<bool>,
    /// The server relays to a remote model (013): its backend and pod.
    pub remote: Option<String>,
    /// `None`: not installed; `Some(None)`: installed, version unreadable.
    pub claude: Option<Option<(u32, u32, u32)>>,
    pub env_api_key: bool,
    /// Claude Code settings files (user and project) whose `env` overrides what `launch` sets,
    /// with the keys.
    pub settings_overrides: Vec<(String, Vec<String>)>,
    pub installed: Vec<Client>,
    /// Names of pulled models.
    pub pulled: Vec<String>,
    pub hf_cache: PathBuf,
    /// Free bytes on the HF cache's disk.
    pub free_disk: Option<u64>,
    pub mrun_home: PathBuf,
    /// Why `MRUN_HOME` is not writable.
    pub home_error: Option<String>,
    /// Why the HF cache (or its nearest existing parent) is not writable.
    pub cache_error: Option<String>,
    /// Why MLX's GPU kernels could not be extracted (D034).
    pub kernels_error: Option<String>,
    /// The build's tensor backend and its kernel mode (016 FR-008, FR-009); `None` without an
    /// engine.
    pub backend: Option<mrun_engine::engine::BackendInfo>,
    /// Other processes, largest resident memory first (D062).
    pub processes: Vec<Process>,
    /// The config file (017): its path, or why it cannot be used.
    pub config: ConfigState,
}

/// The config file as `doctor` sees it (017 FR-009).
#[derive(Clone, Debug, PartialEq)]
pub enum ConfigState {
    /// `MRUN_CONFIG` is empty.
    Disabled,
    /// No file at the default path.
    Absent(PathBuf),
    /// A usable file: path, number of settings, warnings.
    Ok(PathBuf, usize, Vec<String>),
    /// A file that stops every command.
    Error(String),
}

pub fn checks(ctx: &Context) -> Vec<Finding> {
    [
        Some(platform(ctx)),
        Some(memory(ctx)),
        Some(other_processes(ctx)),
        Some(kernels(ctx)),
        Some(config(ctx)),
        Some(low_power(ctx)),
        Some(port(ctx)),
        Some(server_check(ctx)),
        Some(models(ctx)),
        Some(default_model(ctx)),
        Some(disk(ctx)),
        Some(home(ctx)),
        Some(hf_cache(ctx)),
        claude(ctx),
        Some(claude_settings(ctx)),
        Some(api_key(ctx)),
        Some(clients(ctx)),
    ]
    .into_iter()
    .flatten()
    .collect()
}

fn platform(ctx: &Context) -> Finding {
    if ctx.os == "macos" && ctx.arch == "aarch64" {
        Finding::pass("platform", "Apple Silicon Mac")
    } else {
        Finding::fail(
            "platform",
            format!(
                "{} on {}: mrun serves models on Apple Silicon Macs",
                ctx.os, ctx.arch
            ),
            "run mrun on a Mac with an M-series chip",
        )
    }
}

fn memory(ctx: &Context) -> Finding {
    let Some(device) = ctx.memory else {
        return Finding::warn("memory", "physical memory unknown", "report this as a bug");
    };
    let gib = |b: u64| b as f64 / GIB as f64;
    let pressure = match ctx.pressure {
        Some(2) => " (memory pressure: warning)",
        Some(4) => " (memory pressure: critical)",
        _ => "",
    };
    // D062: available vs working set, and which term of the budget binds.
    let available = match device.available {
        Some(bytes) => format!(
            "{:.1} GiB available ({:.1} GiB margin for other apps)",
            gib(bytes),
            gib(budget::margin(device.physical))
        ),
        None => "available memory not read".to_string(),
    };
    match budget::plan(device, Overrides::default()) {
        Ok(plan) => {
            let message = format!(
                "budget {:.1} GiB of {:.0} GiB, bound by {}: {available}, {:.1} GiB working set ({}){pressure}",
                gib(plan.budget),
                gib(device.physical),
                plan.binding,
                gib(device.recommended_working_set),
                hub::memory_source(device)
            );
            if ctx.pressure.is_some_and(|p| p >= 2) {
                Finding::warn(
                    "memory",
                    message,
                    "close memory-hungry apps; mrun sheds its cache under pressure but the model stays",
                )
            } else if plan.binding == Binding::Available {
                Finding::warn(
                    "memory",
                    message,
                    "other apps hold memory mrun would use: quit the largest (see other-processes) and rerun",
                )
            } else {
                Finding::pass("memory", message)
            }
        }
        Err(error) if device.available.is_some() => Finding::fail(
            "memory",
            error
                .to_string()
                .lines()
                .next()
                .unwrap_or_default()
                .to_string(),
            "quit memory-hungry apps or VMs (see other-processes)",
        ),
        Err(error) => Finding::fail("memory", error.to_string(), "use a Mac with more memory"),
    }
}

/// A process this large is worth naming (D062).
const LARGE_PROCESS: u64 = 4 * GIB;
/// A VM host below this holds no guest memory yet (an idle helper process).
const IDLE_VM: u64 = GIB / 4;

/// D062: other processes of at least [`LARGE_PROCESS`] and every VM, which hold memory mrun's
/// budget no longer counts on.
fn other_processes(ctx: &Context) -> Finding {
    let large: Vec<Process> = ctx
        .processes
        .iter()
        .filter(|p| p.resident >= LARGE_PROCESS || (p.is_vm() && p.resident >= IDLE_VM))
        .cloned()
        .collect();
    if large.is_empty() {
        return Finding::pass(
            "other-processes",
            "no other process holds 4 GiB or more, no VM runs",
        );
    }
    let vms = large.iter().filter(|p| p.is_vm()).count();
    let vm_note = match vms {
        0 => String::new(),
        1 => " (one is a VM; its guest can grow to its configured size)".to_string(),
        n => format!(" ({n} are VMs; their guests can grow to their configured sizes)"),
    };
    Finding::warn(
        "other-processes",
        format!(
            "large memory users besides mrun: {}{vm_note}",
            sysmem::describe(&large)
        ),
        "quit what you do not need before serving a model; mrun budgets with what is available at its start",
    )
}

fn kernels(ctx: &Context) -> Finding {
    use mrun_engine::engine::Kernels;
    if let Some(error) = &ctx.kernels_error {
        return Finding::fail(
            "gpu-kernels",
            error.clone(),
            "make ~/Library/Caches/mrun writable and free disk space",
        );
    }
    let Some(info) = ctx.backend else {
        return Finding::pass("gpu-kernels", "no tensor backend in this build");
    };
    let backend = format!("{} {}", info.backend.as_str(), info.version);
    match info.kernels {
        Kernels::Precompiled => Finding::pass(
            "gpu-kernels",
            format!("{backend} GPU kernels ready (kernels: precompiled)"),
        ),
        Kernels::Runtime => Finding::warn(
            "gpu-kernels",
            format!(
                "{backend} compiles its GPU kernels at start (kernels: runtime): this build has no precompiled metallib"
            ),
            "install a release build (it embeds them), or build with the Metal toolchain (`xcrun metal`)",
        ),
        Kernels::Unavailable => Finding::fail(
            "gpu-kernels",
            format!("{backend} was built without Metal kernels (kernels: unavailable)"),
            "build on macOS with the Metal toolchain",
        ),
    }
}

fn low_power(ctx: &Context) -> Finding {
    match ctx.low_power {
        Some(true) => Finding::warn(
            "low-power",
            "Low Power Mode is on: generation runs markedly slower",
            "turn it off in System Settings › Battery",
        ),
        _ => Finding::pass("low-power", "Low Power Mode off"),
    }
}

fn port(ctx: &Context) -> Finding {
    match ctx.port_state {
        PortState::Free => Finding::pass("port", format!("port {} is free", ctx.port)),
        PortState::Mrun => Finding::pass("port", format!("mrun serves port {}", ctx.port)),
        PortState::Other => Finding::fail(
            "port",
            format!("port {} is used by another program", ctx.port),
            "pick another port with --port or MRUN_PORT",
        ),
    }
}

fn server_check(ctx: &Context) -> Finding {
    if let (Some(true), Some(remote)) = (ctx.server, &ctx.remote) {
        // Principle IV: remote mode is stated wherever it is visible.
        return Finding::warn(
            "server",
            format!(
                "the server relays to a remote model ({remote}); prompts and code are sent to RunPod"
            ),
            "'mrun cloud down' ends remote mode",
        );
    }
    match ctx.server {
        Some(true) => Finding::pass("server", "the server is running with a loaded model"),
        Some(false) => Finding::warn(
            "server",
            "the server is still loading its model (or failed to)",
            "check 'mrun ps' or 'mrun logs'",
        ),
        None => Finding::warn(
            "server",
            "no server is running",
            "start one with 'mrun serve --detach', or let 'mrun launch <client>' start it",
        ),
    }
}

/// The per-machine default alias.
fn default_alias(ctx: &Context) -> &'static aliases::Alias {
    aliases::default_alias(ctx.memory)
}

fn models(ctx: &Context) -> Finding {
    if ctx.pulled.is_empty() {
        Finding::warn(
            "models",
            "no model is pulled yet",
            format!("mrun pull {}", default_alias(ctx).name),
        )
    } else {
        Finding::pass("models", format!("pulled: {}", ctx.pulled.join(", ")))
    }
}

fn default_model(ctx: &Context) -> Finding {
    let Some(device) = ctx.memory else {
        return Finding::warn(
            "default-model",
            "cannot pick a default model: physical memory unknown",
            "name a model with --model",
        );
    };
    let alias = default_alias(ctx);
    let idle = alias.fit(device.ignoring_available());
    let fit = alias.fit(device);
    if idle.max_context < aliases::DEFAULT_MIN_CONTEXT {
        return Finding::warn(
            "default-model",
            format!(
                "even {} leaves only {} tokens of context in a budget of {}",
                alias.name,
                idle.max_context,
                human_bytes(idle.budget)
            ),
            "a Mac with more memory (or a raised iogpu.wired_limit_mb) leaves more context",
        );
    }
    if fit.max_context < aliases::DEFAULT_MIN_CONTEXT {
        let leaves = if fit.fits {
            format!("leaves only {} tokens of context", fit.max_context)
        } else {
            "does not fit".to_string()
        };
        return Finding::warn(
            "default-model",
            format!(
                "default model {} {leaves} in the memory available now (budget {})",
                alias.name,
                human_bytes(fit.budget)
            ),
            "quit memory-hungry apps or VMs (see other-processes); 'mrun serve' refuses with E041 until then",
        );
    }
    let preview = alias
        .preview_note()
        .map(|note| format!(" ({note})"))
        .unwrap_or_default();
    Finding::pass(
        "default-model",
        format!(
            "default model {} fits: up to {} tokens of context{preview}",
            alias.name, fit.max_context
        ),
    )
}

fn disk(ctx: &Context) -> Finding {
    let Some(free) = ctx.free_disk else {
        return Finding::warn(
            "disk",
            format!("free space in {} unknown", ctx.hf_cache.display()),
            "check the disk with 'df -h'",
        );
    };
    let alias = default_alias(ctx);
    let needed = if ctx.pulled.iter().any(|name| name == alias.name) {
        0
    } else {
        alias.size + alias.size / 20
    };
    if free < needed {
        Finding::fail(
            "disk",
            format!(
                "{} free in {}; {} needs {}",
                human_bytes(free),
                ctx.hf_cache.display(),
                alias.name,
                human_bytes(needed)
            ),
            "free disk space, or set HF_HUB_CACHE to a larger disk",
        )
    } else {
        Finding::pass(
            "disk",
            format!("{} free in {}", human_bytes(free), ctx.hf_cache.display()),
        )
    }
}

fn home(ctx: &Context) -> Finding {
    match &ctx.home_error {
        None => Finding::pass(
            "mrun-home",
            format!("{} is writable", ctx.mrun_home.display()),
        ),
        Some(error) => Finding::fail(
            "mrun-home",
            format!("{} is not writable: {error}", ctx.mrun_home.display()),
            "fix its permissions, or set MRUN_HOME to a writable directory",
        ),
    }
}

fn hf_cache(ctx: &Context) -> Finding {
    match &ctx.cache_error {
        None => Finding::pass(
            "hf-cache",
            format!("{} is writable", ctx.hf_cache.display()),
        ),
        Some(error) => Finding::fail(
            "hf-cache",
            format!("{} is not writable: {error}", ctx.hf_cache.display()),
            "fix its permissions, or set HF_HUB_CACHE to a writable directory",
        ),
    }
}

/// Only for an installed Claude Code; a missing one is listed by [`clients`].
fn claude(ctx: &Context) -> Option<Finding> {
    let (a, b, c) = clients::CLAUDE_CODE_MIN;
    Some(match ctx.claude? {
        Some(version) if version < clients::CLAUDE_CODE_MIN => Finding::fail(
            "claude-code",
            format!(
                "Claude Code {}.{}.{} is older than {a}.{b}.{c}",
                version.0, version.1, version.2
            ),
            "claude update",
        ),
        Some(version) => Finding::pass(
            "claude-code",
            format!("Claude Code {}.{}.{}", version.0, version.1, version.2),
        ),
        None => Finding::warn(
            "claude-code",
            "Claude Code is installed but its version is unknown",
            format!("make sure it is {a}.{b}.{c} or newer: claude update"),
        ),
    })
}

fn claude_settings(ctx: &Context) -> Finding {
    if ctx.settings_overrides.is_empty() {
        Finding::pass(
            "claude-settings",
            "no Claude Code settings file (user or ./.claude) overrides launch",
        )
    } else {
        Finding::warn(
            "claude-settings",
            format!(
                "{} in \"env\", which beats what 'mrun launch' sets",
                clients::describe_overrides(&ctx.settings_overrides)
            ),
            "remove them there; 'mrun connect claude-code' prints a separate settings file",
        )
    }
}

fn api_key(ctx: &Context) -> Finding {
    if ctx.env_api_key {
        Finding::warn(
            "anthropic-api-key",
            "ANTHROPIC_API_KEY is set in your shell: Claude Code started by hand sends it instead of the local token",
            "use 'mrun launch claude-code' (it empties the key for that process), or unset it",
        )
    } else {
        Finding::pass(
            "anthropic-api-key",
            "no ANTHROPIC_API_KEY in the environment",
        )
    }
}

/// Installed and missing clients, in one line.
fn clients(ctx: &Context) -> Finding {
    let names = |installed: bool| -> Vec<&str> {
        Client::ALL
            .into_iter()
            .filter(|client| ctx.installed.contains(client) == installed)
            .map(Client::binary)
            .collect()
    };
    let (found, missing) = (names(true), names(false));
    if missing.is_empty() {
        return Finding::pass("clients", format!("{} found", found.join(", ")));
    }
    let found = if found.is_empty() {
        "no coding agent found".to_string()
    } else {
        format!("{} found", found.join(", "))
    };
    Finding::info(
        "clients",
        format!("{found}; not installed: {}", missing.join(", ")),
    )
}

/// The real context of this machine.
#[cfg(feature = "engine")]
fn kernels_error() -> Option<String> {
    mrun_engine::tensor::metallib::ensure().err()
}

#[cfg(not(feature = "engine"))]
fn kernels_error() -> Option<String> {
    None
}

pub fn gather(port: u16) -> Context {
    let health = server::request(port, "GET", "/health").ok();
    let mrun_health = health
        .as_ref()
        .is_some_and(|(_, body)| body.get("loaded").is_some());
    let port_state = if std::net::TcpListener::bind(("127.0.0.1", port)).is_ok() {
        PortState::Free
    } else if mrun_health {
        PortState::Mrun
    } else {
        PortState::Other
    };
    let paths = hub::Paths::from_env();
    let claude = crate::cli::which("claude").map(|binary| {
        std::process::Command::new(binary)
            .arg("--version")
            .output()
            .ok()
            .and_then(|o| clients::parse_version(&String::from_utf8_lossy(&o.stdout)))
    });
    Context {
        os: std::env::consts::OS,
        arch: std::env::consts::ARCH,
        memory: hub::device_memory(),
        pressure: sysctl("kern.memorystatus_vm_pressure_level").and_then(|v| v.parse().ok()),
        low_power: command_output("pmset", &["-g"]).map(|text| {
            text.lines()
                .any(|l| l.contains("lowpowermode") && l.trim_end().ends_with('1'))
        }),
        port,
        port_state,
        server: mrun_health.then(|| health.is_some_and(|(_, b)| b["loaded"] == true)),
        remote: mrun_health
            .then(|| crate::cli::server::request(port, "GET", "/mrun/status").ok())
            .flatten()
            .filter(|(_, status)| !status["pod"].is_null())
            .map(|(_, status)| {
                format!(
                    "{} on pod {}",
                    status["backend"].as_str().unwrap_or("a remote backend"),
                    status["pod"]["id"].as_str().unwrap_or("?")
                )
            }),
        claude,
        env_api_key: std::env::var("ANTHROPIC_API_KEY").is_ok_and(|v| !v.is_empty()),
        settings_overrides: clients::claude_settings_overrides(),
        installed: Client::installed(),
        pulled: hub::manifests(&paths)
            .iter()
            .map(|m| m.name().to_string())
            .collect(),
        free_disk: existing_ancestor(&paths.hf_cache).and_then(hub::download::free_space),
        home_error: writable(&paths.mrun_home).err(),
        cache_error: existing_ancestor(&paths.hf_cache)
            .map_or(Err("no existing parent".to_string()), |dir| {
                tempfile::tempfile_in(dir)
                    .map(drop)
                    .map_err(|error| error.to_string())
            })
            .err(),
        kernels_error: kernels_error(),
        backend: crate::cli::version::backend(),
        processes: sysmem::processes(),
        config: config_state(),
        hf_cache: paths.hf_cache,
        mrun_home: paths.mrun_home,
    }
}

fn config_state() -> ConfigState {
    let resolved = crate::config::resolve(&crate::cli::app::cli_command());
    match (&resolved.location, resolved.file) {
        (crate::config::Location::Disabled, _) => ConfigState::Disabled,
        (_, Err(error)) => ConfigState::Error(error.to_string()),
        (location, Ok(file)) => {
            let path = location.path().map(Path::to_path_buf).unwrap_or_default();
            if file.exists {
                ConfigState::Ok(path, file.values.len(), file.warnings)
            } else {
                ConfigState::Absent(path)
            }
        }
    }
}

fn config(ctx: &Context) -> Finding {
    match &ctx.config {
        ConfigState::Disabled => Finding::pass("config", "no config file (MRUN_CONFIG is empty)"),
        ConfigState::Absent(path) => Finding::pass(
            "config",
            format!(
                "no config file at {} (built-in defaults apply)",
                path.display()
            ),
        ),
        ConfigState::Ok(path, n, warnings) if warnings.is_empty() => Finding::pass(
            "config",
            format!(
                "{}: {n} setting{}",
                path.display(),
                if *n == 1 { "" } else { "s" }
            ),
        ),
        ConfigState::Ok(path, n, warnings) => Finding::warn(
            "config",
            format!(
                "{}: {n} setting{}, {} ignored: {}",
                path.display(),
                if *n == 1 { "" } else { "s" },
                warnings.len(),
                warnings.join("; ")
            ),
            "fix or remove the lines named; 'mrun config show' lists every setting",
        ),
        ConfigState::Error(error) => Finding::fail(
            "config",
            error.clone(),
            "fix the config file (every command stops until then), or run with MRUN_CONFIG= to ignore it",
        ),
    }
}

/// `path` or its nearest existing parent (the cache may not exist yet).
fn existing_ancestor(path: &Path) -> Option<&Path> {
    path.ancestors().find(|dir| dir.exists())
}

/// Creates `dir` if needed and writes a temporary file into it.
fn writable(dir: &Path) -> Result<(), String> {
    std::fs::create_dir_all(dir)
        .and_then(|()| tempfile::tempfile_in(dir))
        .map(drop)
        .map_err(|error| error.to_string())
}

fn command_output(program: &str, args: &[&str]) -> Option<String> {
    let output = std::process::Command::new(program)
        .args(args)
        .output()
        .ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).into_owned())
}

fn sysctl(name: &str) -> Option<String> {
    command_output("sysctl", &["-n", name]).map(|s| s.trim().to_string())
}

impl Status {
    fn mark(self) -> &'static str {
        match self {
            Status::Pass => "ok  ",
            Status::Info => "info",
            Status::Warn => "warn",
            Status::Fail => "FAIL",
        }
    }
}

/// What `mrun doctor --report` prints besides the findings (D060).
#[derive(Clone, Debug)]
pub struct Report {
    /// `mrun --version`'s line, numerics included.
    pub version: String,
    /// e.g. `Apple M4 Pro`.
    pub chip: Option<String>,
    /// e.g. `applegpu_g16s` (`cpu` without Metal).
    pub gpu_family: String,
    pub memory: Option<DeviceMemory>,
    /// `sw_vers`: product version and build.
    pub macos: Option<String>,
    pub nax_compiled: bool,
    /// The GPU has NAX (M5), whether or not it is compiled in.
    pub nax_available: bool,
    /// Whether the backend runs its NAX kernels, and why not (`on`, `off (<reason>)`).
    pub nax_running: String,
    pub models: Vec<ReportModel>,
    /// Shown path of the calibration file (home as `~`).
    pub calibration_path: String,
    /// `(key, [(depth, len)])`.
    pub calibration: Vec<(String, Vec<(u32, u32)>)>,
}

#[derive(Clone, Debug)]
pub struct ReportModel {
    pub name: String,
    pub repo: String,
    pub revision: String,
    pub identity: String,
    pub quantization: String,
}

/// `51,539,607,552` (thousands separated by commas).
fn grouped(n: u64) -> String {
    let digits = n.to_string();
    let mut out = String::new();
    for (i, c) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(c);
    }
    out
}

/// Replaces `home` with `~`, then any other `/Users/<name>` with `/Users/…`, so a pasted report
/// names no account.
pub fn redact(text: &str, home: Option<&str>) -> String {
    let mut text = match home.filter(|h| h.len() > 1) {
        Some(home) => text.replace(home.trim_end_matches('/'), "~"),
        None => text.to_string(),
    };
    let mut out = String::with_capacity(text.len());
    while let Some(at) = text.find("/Users/") {
        let (before, rest) = text.split_at(at + "/Users/".len());
        out.push_str(before);
        let end = rest
            .find(|c: char| c == '/' || c.is_whitespace() || c == '`' || c == '|')
            .unwrap_or(rest.len());
        out.push('…');
        text = rest[end..].to_string();
    }
    out.push_str(&text);
    out
}

/// The paste-ready Markdown block, before [`redact`].
pub fn render_report(report: &Report, findings: &[Finding]) -> String {
    use std::fmt::Write;
    let gib = |b: u64| b as f64 / (1u64 << 30) as f64;
    let yes = |b: bool| if b { "yes" } else { "no" };
    let mut out = String::from("### mrun hardware report\n\n| | |\n|---|---|\n");
    let mut row = |name: &str, value: String| {
        let _ = writeln!(out, "| {name} | {} |", value.replace('|', "\\|"));
    };
    row("mrun", format!("`{}`", report.version));
    row(
        "Chip",
        report.chip.clone().unwrap_or_else(|| "unknown".into()),
    );
    row("GPU family", report.gpu_family.clone());
    match report.memory {
        Some(m) => {
            row(
                "Physical memory",
                format!("{} B ({:.0} GiB)", grouped(m.physical), gib(m.physical)),
            );
            row(
                "MLX recommended working set",
                format!(
                    "{} B ({:.2} GiB, {})",
                    grouped(m.recommended_working_set),
                    gib(m.recommended_working_set),
                    hub::memory_source(m)
                ),
            );
        }
        None => row("Physical memory", "unknown".into()),
    }
    row(
        "macOS",
        report.macos.clone().unwrap_or_else(|| "unknown".into()),
    );
    row(
        "NAX",
        format!(
            "available on this GPU: {}; compiled in: {}; running: {}",
            yes(report.nax_available),
            yes(report.nax_compiled),
            report.nax_running
        ),
    );
    out.push_str("\n#### Models\n\n");
    if report.models.is_empty() {
        out.push_str("none pulled\n");
    } else {
        out.push_str(
            "| Model | Repository @ revision | Quantization | Identity |\n|---|---|---|---|\n",
        );
        for m in &report.models {
            let _ = writeln!(
                out,
                "| {} | {}@{} | {} | `{}` |",
                m.name,
                m.repo,
                &m.revision[..m.revision.len().min(12)],
                m.quantization,
                m.identity
            );
        }
    }
    let _ = write!(
        out,
        "\n#### Calibration (`{}`)\n\n",
        report.calibration_path
    );
    if report.calibration.is_empty() {
        out.push_str("none stored\n");
    } else {
        out.push_str("| Key | chunk length by depth |\n|---|---|\n");
        for (key, points) in &report.calibration {
            let points: Vec<String> = points.iter().map(|(d, l)| format!("{d}→{l}")).collect();
            let _ = writeln!(
                out,
                "| {} | {} |",
                key.replace('|', "\\|"),
                points.join(", ")
            );
        }
    }
    out.push_str("\n#### Doctor\n\n```\n");
    for f in findings {
        let _ = writeln!(out, "{} {:<18} {}", f.status.mark(), f.id, f.message);
    }
    out.push_str("```\n");
    out
}

fn quantization_text(q: Option<&hub::Quantization>) -> String {
    match q {
        Some(q) => format!(
            "{}-{}-{}",
            q.mode.as_deref().unwrap_or("affine"),
            q.bits.map_or("?".into(), |b| b.to_string()),
            q.group_size.map_or("?".into(), |g| g.to_string())
        ),
        None => "dense".into(),
    }
}

/// `(gpu family, NAX available, NAX compiled in, NAX running)`.
#[cfg(feature = "engine")]
fn gpu() -> (String, bool, bool, String) {
    (
        mrun_engine::tensor::numerics().gpu_arch,
        mrun_engine::tensor::nax_available(),
        mrun_engine::tensor::NAX_COMPILED,
        mrun_engine::tensor::nax_status(),
    )
}

#[cfg(not(feature = "engine"))]
fn gpu() -> (String, bool, bool, String) {
    ("cpu".into(), false, false, "off (no engine)".into())
}

/// The real report of this machine.
pub fn gather_report(ctx: &Context) -> Report {
    let paths = hub::Paths::from_env();
    let (gpu_family, nax_available, nax_compiled, nax_running) = gpu();
    let macos = command_output("sw_vers", &["-productVersion"]).map(|version| {
        let build = command_output("sw_vers", &["-buildVersion"]).unwrap_or_default();
        format!("{} ({})", version.trim(), build.trim())
    });
    let calibration_path = mrun_engine::calibration::default_path();
    Report {
        version: crate::cli::version::long(),
        chip: sysctl("machdep.cpu.brand_string"),
        gpu_family,
        memory: ctx.memory,
        macos,
        nax_compiled,
        nax_available,
        nax_running,
        models: hub::manifests(&paths)
            .iter()
            .map(|m| ReportModel {
                name: m.name().to_string(),
                repo: m.repo.clone(),
                revision: m.revision.clone(),
                identity: m.identity.clone(),
                quantization: quantization_text(m.quantization.as_ref()),
            })
            .collect(),
        calibration: calibration_path
            .as_deref()
            .map(mrun_engine::calibration::entries)
            .unwrap_or_default(),
        calibration_path: calibration_path.map_or("unknown".into(), |p| p.display().to_string()),
    }
}

pub fn run(args: DoctorArgs) -> u8 {
    let ctx = gather(args.port);
    let findings = checks(&ctx);
    if args.report {
        let home = std::env::var("HOME").ok();
        print!(
            "{}",
            redact(
                &render_report(&gather_report(&ctx), &findings),
                home.as_deref()
            )
        );
        return 0;
    }
    for f in &findings {
        let mark = f.status.mark();
        println!("{mark}  {:<18} {}", f.id, f.message);
        if let Some(fix) = &f.fix {
            println!("      {:<18} fix: {fix}", "");
        }
    }
    if findings.iter().any(|f| f.status == Status::Fail) {
        Exit::DoctorFailures as u8
    } else {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn good() -> Context {
        Context {
            os: "macos",
            arch: "aarch64",
            memory: Some(DeviceMemory {
                physical: 48 << 30,
                recommended_working_set: 37 << 30,
                available: None,
                estimated: false,
            }),
            pressure: Some(1),
            low_power: Some(false),
            port: 8484,
            port_state: PortState::Free,
            server: Some(true),
            remote: None,
            claude: Some(Some((2, 1, 300))),
            env_api_key: false,
            settings_overrides: Vec::new(),
            installed: Client::ALL.to_vec(),
            pulled: vec!["qwen3.6".into()],
            hf_cache: "/h/.cache/huggingface/hub".into(),
            free_disk: Some(100_000_000_000),
            mrun_home: "/h/.mrun".into(),
            home_error: None,
            cache_error: None,
            kernels_error: None,
            backend: Some(mrun_engine::engine::BackendInfo {
                backend: mrun_engine::engine::Backend::Nuvec,
                version: "0.4.1",
                kernels: mrun_engine::engine::Kernels::Precompiled,
            }),
            processes: Vec::new(),
            config: ConfigState::Absent(PathBuf::from("/home/.mrun/config.toml")),
        }
    }

    fn find(ctx: &Context, id: &str) -> Option<Finding> {
        checks(ctx).into_iter().find(|f| f.id == id)
    }

    #[test]
    fn the_kernels_finding_names_the_backend_and_its_kernel_mode() {
        use mrun_engine::engine::{Backend, BackendInfo, Kernels};
        let with = |backend, version, kernels| {
            let mut ctx = good();
            ctx.backend = Some(BackendInfo {
                backend,
                version,
                kernels,
            });
            find(&ctx, "gpu-kernels").unwrap()
        };
        let nuvec = with(Backend::Nuvec, "0.4.1", Kernels::Precompiled);
        assert_eq!(nuvec.status, Status::Pass);
        assert!(
            nuvec.message.contains("nuvec 0.4.1") && nuvec.message.contains("kernels: precompiled"),
            "{}",
            nuvec.message
        );
        let mlx = with(Backend::Mlx, "0.32.4", Kernels::Precompiled);
        assert!(mlx.message.contains("mlx 0.32.4"), "{}", mlx.message);
        let runtime = with(Backend::Nuvec, "0.4.1", Kernels::Runtime);
        assert_eq!(runtime.status, Status::Warn);
        assert!(
            runtime.message.contains("kernels: runtime"),
            "{}",
            runtime.message
        );
        assert_eq!(
            with(Backend::Nuvec, "0.4.1", Kernels::Unavailable).status,
            Status::Fail
        );
        let mut none = good();
        none.backend = None;
        assert_eq!(find(&none, "gpu-kernels").unwrap().status, Status::Pass);
    }

    #[test]
    fn a_healthy_mac_passes_everything() {
        assert!(checks(&good()).iter().all(|f| f.status == Status::Pass));
    }

    #[test]
    fn each_check_reports_its_problem_with_a_fix() {
        type Change = Box<dyn Fn(&mut Context)>;
        let cases: Vec<(&str, Change, Status)> = vec![
            ("platform", Box::new(|c| c.os = "linux"), Status::Fail),
            ("memory", Box::new(|c| c.pressure = Some(4)), Status::Warn),
            (
                "config",
                Box::new(|c| {
                    c.config = ConfigState::Ok(
                        PathBuf::from("/c.toml"),
                        2,
                        vec!["/c.toml:3: unknown section [clould]; did you mean [cloud]?".into()],
                    )
                }),
                Status::Warn,
            ),
            (
                "config",
                Box::new(|c| c.config = ConfigState::Error("/c.toml:1:7: unclosed table".into())),
                Status::Fail,
            ),
            (
                "server",
                Box::new(|c| c.remote = Some("vllm 0.32.0 on pod abc".into())),
                Status::Warn,
            ),
            (
                "memory",
                Box::new(|c| {
                    c.memory = Some(DeviceMemory {
                        physical: 2 << 30,
                        recommended_working_set: 1 << 30,
                        available: None,
                        estimated: false,
                    })
                }),
                Status::Fail,
            ),
            (
                "low-power",
                Box::new(|c| c.low_power = Some(true)),
                Status::Warn,
            ),
            (
                "port",
                Box::new(|c| c.port_state = PortState::Other),
                Status::Fail,
            ),
            (
                "port",
                Box::new(|c| c.port_state = PortState::Mrun),
                Status::Pass,
            ),
            ("server", Box::new(|c| c.server = None), Status::Warn),
            ("models", Box::new(|c| c.pulled.clear()), Status::Warn),
            (
                "default-model",
                Box::new(|c| {
                    c.memory = Some(DeviceMemory {
                        physical: 8 << 30,
                        recommended_working_set: 5 << 30,
                        available: None,
                        estimated: false,
                    })
                }),
                Status::Warn,
            ),
            (
                "disk",
                Box::new(|c| {
                    c.pulled.clear();
                    c.free_disk = Some(1 << 30);
                }),
                Status::Fail,
            ),
            (
                "disk",
                Box::new(|c| c.free_disk = Some(1 << 30)),
                Status::Pass,
            ),
            (
                "mrun-home",
                Box::new(|c| c.home_error = Some("permission denied".into())),
                Status::Fail,
            ),
            (
                "claude-code",
                Box::new(|c| c.claude = Some(Some((2, 1, 200)))),
                Status::Fail,
            ),
            (
                "claude-settings",
                Box::new(|c| {
                    c.settings_overrides = vec![(
                        "./.claude/settings.local.json".into(),
                        vec!["ANTHROPIC_BASE_URL".into()],
                    )]
                }),
                Status::Warn,
            ),
            (
                "hf-cache",
                Box::new(|c| c.cache_error = Some("read-only file system".into())),
                Status::Fail,
            ),
            (
                "anthropic-api-key",
                Box::new(|c| c.env_api_key = true),
                Status::Warn,
            ),
        ];
        for (id, change, expected) in cases {
            let mut ctx = good();
            change(&mut ctx);
            let finding = find(&ctx, id).unwrap();
            assert_eq!(finding.status, expected, "{id}: {}", finding.message);
            if expected != Status::Pass {
                assert!(finding.fix.is_some(), "{id} has a fix");
            }
        }
    }

    #[test]
    fn memory_names_its_source_and_overrides_name_their_file() {
        let mut ctx = good();
        assert!(
            find(&ctx, "memory")
                .unwrap()
                .message
                .contains("(Metal recommended working set)")
        );
        ctx.memory.as_mut().unwrap().estimated = true;
        assert!(find(&ctx, "memory").unwrap().message.contains("(estimated"));
        ctx.settings_overrides = vec![
            (
                "~/.claude/settings.json".into(),
                vec!["ANTHROPIC_MODEL".into()],
            ),
            (
                "./.claude/settings.json".into(),
                vec!["API_TIMEOUT_MS".into()],
            ),
        ];
        let message = find(&ctx, "claude-settings").unwrap().message;
        assert!(
            message.starts_with(
                "~/.claude/settings.json sets ANTHROPIC_MODEL; ./.claude/settings.json sets API_TIMEOUT_MS"
            ),
            "{message}"
        );
    }

    /// D062: the memory line names available vs working set and the binding term; other
    /// apps and VMs are listed, and a default model they squeeze out is a warning.
    #[test]
    fn other_processes_and_available_memory_are_reported() {
        let mut ctx = good();
        let pass = find(&ctx, "memory").unwrap();
        assert!(
            pass.message.contains("bound by working set"),
            "{}",
            pass.message
        );
        assert!(pass.message.contains("available memory not read"));
        ctx.memory.as_mut().unwrap().available = Some(44 << 30);
        let idle = find(&ctx, "memory").unwrap();
        assert_eq!(idle.status, Status::Pass);
        assert!(
            idle.message.contains("44.0 GiB available (6.0 GiB margin"),
            "{}",
            idle.message
        );
        ctx.memory.as_mut().unwrap().available = Some(20 << 30);
        let short = find(&ctx, "memory").unwrap();
        assert_eq!(short.status, Status::Warn);
        assert!(
            short
                .message
                .starts_with("budget 14.0 GiB of 48 GiB, bound by available memory"),
            "{}",
            short.message
        );
        let default = find(&ctx, "default-model").unwrap();
        assert_eq!(default.status, Status::Warn, "{}", default.message);
        assert!(default.message.contains("available now"));
        ctx.memory.as_mut().unwrap().available = Some(4 << 30);
        assert_eq!(find(&ctx, "memory").unwrap().status, Status::Fail);

        assert_eq!(find(&ctx, "other-processes").unwrap().status, Status::Pass);
        let process = |name: &str, gib: u64| Process {
            pid: 1,
            name: name.into(),
            resident: gib << 30,
        };
        ctx.processes = vec![
            process("rust-analyzer", 5),
            process("com.apple.Virtualization.VirtualMachine", 2),
            process("Code Helper", 3),
            process("com.docker.virtualization", 0),
        ];
        let others = find(&ctx, "other-processes").unwrap();
        assert_eq!(others.status, Status::Warn);
        assert_eq!(
            others.message,
            "large memory users besides mrun: rust-analyzer 5.0 GiB, \
             com.apple.Virtualization.VirtualMachine 2.0 GiB (one is a VM; its guest can grow to \
             its configured size)"
        );
        assert!(others.fix.is_some());
    }

    #[test]
    fn missing_clients_are_one_info_line() {
        let mut ctx = good();
        ctx.installed = vec![Client::Codex];
        ctx.claude = None;
        assert!(find(&ctx, "claude-code").is_none());
        let clients = find(&ctx, "clients").unwrap();
        assert_eq!(clients.status, Status::Info);
        assert_eq!(
            clients.message,
            "codex found; not installed: claude, opencode"
        );
        assert!(checks(&ctx).iter().all(|f| f.status != Status::Fail));
    }

    fn report() -> Report {
        Report {
            version: "0.1.0 (abc1234, 2026-10-16) mlx 0.30.0 mlx-c 0.4.0 macos-aarch64 numerics no nax, not available, gpu applegpu_g16s".into(),
            chip: Some("Apple M4 Pro".into()),
            gpu_family: "applegpu_g16s".into(),
            memory: Some(DeviceMemory {
                physical: 51_539_607_552,
                recommended_working_set: 40_200_896_512,
                available: None,
                estimated: false,
            }),
            macos: Some("26.6 (25G5000)".into()),
            nax_compiled: false,
            nax_available: false,
            nax_running: "off".into(),
            models: vec![ReportModel {
                name: "qwen3.5".into(),
                repo: "mlx-community/Qwen3.5-9B-4bit".into(),
                revision: "0123456789abcdef0123456789abcdef01234567".into(),
                identity: "sha256:feed".into(),
                quantization: "affine-4-64".into(),
            }],
            calibration_path: "/Users/alice/.mrun/calibration.toml".into(),
            calibration: vec![(
                "Apple M4 Pro, 48 GiB | qwen3_5 | affine-4-64 | applegpu_g16s/nax-off | sha256:feed"
                    .into(),
                vec![(0, 2048), (65_536, 1536)],
            )],
        }
    }

    #[test]
    fn the_report_names_the_machine_and_no_user() {
        let mut ctx = good();
        ctx.hf_cache = "/Users/alice/.cache/huggingface/hub".into();
        ctx.mrun_home = "/Users/alice/.mrun".into();
        let text = redact(
            &render_report(&report(), &checks(&ctx)),
            Some("/Users/alice"),
        );
        for needle in [
            "### mrun hardware report",
            "| mrun | `0.1.0 (abc1234, 2026-10-16)",
            "| Chip | Apple M4 Pro |",
            "| GPU family | applegpu_g16s |",
            "| Physical memory | 51,539,607,552 B (48 GiB) |",
            "| MLX recommended working set | 40,200,896,512 B (37.44 GiB, Metal recommended working set) |",
            "| macOS | 26.6 (25G5000) |",
            "available on this GPU: no; compiled in: no; running: off",
            "| qwen3.5 | mlx-community/Qwen3.5-9B-4bit@0123456789ab | affine-4-64 | `sha256:feed` |",
            "#### Calibration (`~/.mrun/calibration.toml`)",
            "Apple M4 Pro, 48 GiB \\| qwen3_5",
            "0→2048, 65536→1536",
            "ok   platform",
            "~/.cache/huggingface/hub is writable",
        ] {
            assert!(text.contains(needle), "missing {needle:?} in\n{text}");
        }
        assert!(!text.contains("alice"), "{text}");
    }

    #[test]
    fn redaction_hides_other_accounts_too() {
        assert_eq!(
            redact("/Users/bob/x and /Users/me/y", Some("/Users/me")),
            "/Users/…/x and ~/y"
        );
        assert_eq!(redact("/opt/x", None), "/opt/x");
        let mut empty = report();
        empty.models.clear();
        empty.calibration.clear();
        empty.memory = None;
        let text = render_report(&empty, &[]);
        assert!(text.contains("none pulled") && text.contains("none stored"));
        assert_eq!(grouped(999), "999");
        assert_eq!(grouped(1_000), "1,000");
    }

    #[test]
    fn the_default_model_follows_the_memory() {
        let mut ctx = good();
        ctx.pulled.clear();
        assert_eq!(
            find(&ctx, "models").unwrap().fix.unwrap(),
            "mrun pull qwen3.6"
        );
        ctx.memory = Some(DeviceMemory {
            physical: 24 << 30,
            recommended_working_set: 16 << 30,
            available: None,
            estimated: false,
        });
        assert_eq!(
            find(&ctx, "models").unwrap().fix.unwrap(),
            "mrun pull qwen3.5"
        );
        let message = find(&ctx, "default-model").unwrap().message;
        assert!(
            message.starts_with("default model qwen3.5 fits"),
            "{message}"
        );
        assert!(
            message.ends_with(
                "(qwen3.5 is a preview: below the client gate (Claude Code 27/30, Codex 26/30, \
                 OpenCode 28/30); larger Macs get qwen3.6)"
            ),
            "{message}"
        );
        ctx.memory = good().memory;
        let message = find(&ctx, "default-model").unwrap().message;
        assert!(!message.contains("preview"), "{message}");
    }
}
