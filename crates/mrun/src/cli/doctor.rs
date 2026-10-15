//! `mrun doctor`: checks the machine, the server and the clients, each with a fix (004 FR-007,
//! research R6). Exit 8 when any check fails. Checks are pure functions of a [`Context`], so each
//! one is tested with a fake.

use crate::cli::clients::{self, Client};
use crate::cli::server;
use crate::errors::Exit;
use mrun_engine::budget::{self, DeviceMemory, Overrides};

#[derive(Debug, clap::Args)]
pub struct DoctorArgs {
    /// Port of the server.
    #[arg(long, env = "MRUN_PORT", default_value_t = server::default_port())]
    pub port: u16,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Status {
    Pass,
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
    /// `None`: not installed; `Some(None)`: installed, version unreadable.
    pub claude: Option<Option<(u32, u32, u32)>>,
    pub env_api_key: bool,
    pub codex: bool,
    pub opencode: bool,
}

pub fn checks(ctx: &Context) -> Vec<Finding> {
    vec![
        platform(ctx),
        memory(ctx),
        low_power(ctx),
        port(ctx),
        server_check(ctx),
        claude(ctx),
        api_key(ctx),
        binary(ctx.codex, Client::Codex),
        binary(ctx.opencode, Client::OpenCode),
    ]
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
    let gib = |b: u64| b as f64 / (1u64 << 30) as f64;
    let pressure = match ctx.pressure {
        Some(2) => " (memory pressure: warning)",
        Some(4) => " (memory pressure: critical)",
        _ => "",
    };
    match budget::plan(device, Overrides::default()) {
        Ok(plan) if ctx.pressure.is_some_and(|p| p >= 2) => Finding::warn(
            "memory",
            format!(
                "budget {:.1} GiB of {:.0} GiB{pressure}",
                gib(plan.budget),
                gib(device.physical)
            ),
            "close memory-hungry apps; mrun sheds its cache under pressure but the model stays",
        ),
        Ok(plan) => Finding::pass(
            "memory",
            format!(
                "budget {:.1} GiB of {:.0} GiB",
                gib(plan.budget),
                gib(device.physical)
            ),
        ),
        Err(error) => Finding::fail(
            "memory",
            error.to_string(),
            "lower --reserve-gib or use a Mac with more memory",
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
    match ctx.server {
        Some(true) => Finding::pass("server", "the server is running with a loaded model"),
        Some(false) => Finding::warn(
            "server",
            "the server is still loading its model (or failed to)",
            format!(
                "check 'mrun ps' or {}",
                crate::cli::stop::log_file().display()
            ),
        ),
        None => Finding::warn(
            "server",
            "no server is running",
            "start one with 'mrun serve --detach', or let 'mrun launch <client>' start it",
        ),
    }
}

fn claude(ctx: &Context) -> Finding {
    let (a, b, c) = clients::CLAUDE_CODE_MIN;
    match ctx.claude {
        None => Finding::warn(
            "claude-code",
            "Claude Code is not installed",
            format!(
                "install it if you use it: {}",
                Client::ClaudeCode.install_hint()
            ),
        ),
        Some(Some(version)) if version < clients::CLAUDE_CODE_MIN => Finding::fail(
            "claude-code",
            format!(
                "Claude Code {}.{}.{} is older than {a}.{b}.{c}",
                version.0, version.1, version.2
            ),
            "claude update",
        ),
        Some(Some(version)) => Finding::pass(
            "claude-code",
            format!("Claude Code {}.{}.{}", version.0, version.1, version.2),
        ),
        Some(None) => Finding::warn(
            "claude-code",
            "Claude Code is installed but its version is unknown",
            format!("make sure it is {a}.{b}.{c} or newer: claude update"),
        ),
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

fn binary(installed: bool, client: Client) -> Finding {
    let id = match client {
        Client::Codex => "codex",
        Client::OpenCode => "opencode",
        Client::ClaudeCode => "claude-code",
    };
    if installed {
        Finding::pass(id, format!("{} found", client.binary()))
    } else {
        Finding::warn(
            id,
            format!("{} is not installed", client.binary()),
            format!("install it if you use it: {}", client.install_hint()),
        )
    }
}

/// The real context of this machine.
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
        memory: device_memory(),
        pressure: sysctl("kern.memorystatus_vm_pressure_level").and_then(|v| v.parse().ok()),
        low_power: command_output("pmset", &["-g"]).map(|text| {
            text.lines()
                .any(|l| l.contains("lowpowermode") && l.trim_end().ends_with('1'))
        }),
        port,
        port_state,
        server: mrun_health.then(|| health.is_some_and(|(_, b)| b["loaded"] == true)),
        claude,
        env_api_key: std::env::var("ANTHROPIC_API_KEY").is_ok_and(|v| !v.is_empty()),
        codex: crate::cli::which("codex").is_some(),
        opencode: crate::cli::which("opencode").is_some(),
    }
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

/// Physical memory; on macOS the GPU working set is approximated by physical memory until the
/// MLX engine reports `recommendedMaxWorkingSetSize` (the budget takes the minimum with 75 %).
fn device_memory() -> Option<DeviceMemory> {
    #[cfg(target_os = "linux")]
    {
        DeviceMemory::read_linux().ok()
    }
    #[cfg(not(target_os = "linux"))]
    {
        let physical: u64 = sysctl("hw.memsize")?.parse().ok()?;
        Some(DeviceMemory {
            physical,
            recommended_working_set: physical,
        })
    }
}

pub fn run(args: DoctorArgs) -> u8 {
    let findings = checks(&gather(args.port));
    for f in &findings {
        let mark = match f.status {
            Status::Pass => "ok  ",
            Status::Warn => "warn",
            Status::Fail => "FAIL",
        };
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
                recommended_working_set: 36 << 30,
            }),
            pressure: Some(1),
            low_power: Some(false),
            port: 8484,
            port_state: PortState::Free,
            server: Some(true),
            claude: Some(Some((2, 1, 300))),
            env_api_key: false,
            codex: true,
            opencode: true,
        }
    }

    fn status_of(ctx: &Context, id: &str) -> Status {
        checks(ctx).into_iter().find(|f| f.id == id).unwrap().status
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
                "memory",
                Box::new(|c| {
                    c.memory = Some(DeviceMemory {
                        physical: 4 << 30,
                        recommended_working_set: 1 << 30,
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
            (
                "claude-code",
                Box::new(|c| c.claude = Some(Some((2, 1, 200)))),
                Status::Fail,
            ),
            ("claude-code", Box::new(|c| c.claude = None), Status::Warn),
            (
                "anthropic-api-key",
                Box::new(|c| c.env_api_key = true),
                Status::Warn,
            ),
            ("codex", Box::new(|c| c.codex = false), Status::Warn),
            ("opencode", Box::new(|c| c.opencode = false), Status::Warn),
        ];
        for (id, change, expected) in cases {
            let mut ctx = good();
            change(&mut ctx);
            assert_eq!(status_of(&ctx, id), expected, "{id}");
            let finding = checks(&ctx).into_iter().find(|f| f.id == id).unwrap();
            if expected != Status::Pass {
                assert!(finding.fix.is_some(), "{id} has a fix");
            }
        }
    }
}
