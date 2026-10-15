//! `mrun bench` (005): reproducible measurements of mrun and other local servers.
//!
//! Systems run one at a time, mrun first: on a terminal the harness pauses with "Stop X, start
//! Y, press Enter" between them; without one it goes straight to each system's health check.
//! Bench talks to the systems under test only (FR-011) and writes `result.json` (schema v1)
//! and `report.md` to `--out` (default `./mrun-bench/<timestamp>/`).

pub mod driver;
pub mod machine;
pub mod report;
pub mod result;
pub mod stats;
pub mod suites;
pub mod systems;
pub mod trace;

use crate::api::record::rfc3339_now;
use crate::bench::result::{
    Cell, CellStatus, Comparability, Method, Reps, RunResult, SCHEMA_VERSION, SystemUnderTest,
};
use crate::bench::suites::{Ctx, agent, cancel, not_run, synthetic};
use crate::bench::systems::{System, lossy_kv};
use crate::bench::trace::TraceRecord;
use crate::errors::{CliError, Code};
use std::path::{Path, PathBuf};

/// FR-004 minimums; `--reps` raises, never lowers them.
pub const MIN_REPS: Reps = Reps {
    synthetic: 5,
    cancel: 50,
    agent: 3,
};
const TRACE_DIR: &str = "bench/traces";
const LOW_MEMORY_GAP: u64 = 2 << 30;

#[derive(Clone, Copy, Debug, Eq, PartialEq, clap::ValueEnum)]
pub enum Suite {
    Agent,
    Synthetic,
    Cancel,
    All,
}

#[derive(Debug, clap::Args)]
#[command(args_conflicts_with_subcommands = true)]
pub struct BenchArgs {
    #[command(subcommand)]
    pub command: Option<BenchCommand>,
    /// Which suite to run.
    #[arg(long, value_enum, default_value = "all")]
    pub suite: Suite,
    /// Agent trace: a name under bench/traces/ or a .jsonl file [default: every bundled trace].
    #[arg(long = "trace", value_name = "NAME")]
    pub traces: Vec<String>,
    /// NAME=URL[,apis=chat+messages+responses][,model=M][,profile=P]
    /// [default: mrun at http://127.0.0.1:8484].
    #[arg(long = "system", value_name = "SPEC")]
    pub systems: Vec<String>,
    /// Benchmark several systems one at a time (without --system: every bundled profile).
    #[arg(long)]
    pub compare: bool,
    /// Repetitions (raises the per-suite minimums: synthetic 5, cancel 50, agent 3).
    #[arg(long)]
    pub reps: Option<u32>,
    /// Output directory [default: ./mrun-bench/<timestamp>].
    #[arg(long)]
    pub out: Option<PathBuf>,
}

#[derive(Debug, clap::Subcommand)]
pub enum BenchCommand {
    /// Build an agent trace from `mrun serve --record` output (maintainers).
    #[command(hide = true)]
    Trace {
        /// A `mrun serve --record` JSONL file.
        record: PathBuf,
        /// The trace to write.
        #[arg(short, long)]
        out: PathBuf,
        /// tokenizer.json of the model, for exact prompt and reply counts.
        #[arg(long)]
        tokenizer: Option<PathBuf>,
    },
}

/// What one run does.
#[derive(Debug)]
pub struct Plan {
    pub agent: bool,
    pub synthetic: bool,
    pub cancel: bool,
    pub traces: Vec<(String, Vec<TraceRecord>)>,
    pub systems: Vec<System>,
    pub reps: Reps,
}

impl Plan {
    /// Usage errors are returned as text (exit 2).
    pub fn from_args(args: &BenchArgs) -> Result<Self, String> {
        let all = args.suite == Suite::All;
        let mut systems = if args.systems.is_empty() {
            if args.compare {
                System::all_bundled()
            } else {
                vec![System::parse(&format!("mrun={}", default_url()))?]
            }
        } else {
            args.systems
                .iter()
                .map(|s| System::parse(s))
                .collect::<Result<_, _>>()?
        };
        if systems.len() > 1 && !args.compare {
            return Err("several systems need --compare (they run one at a time)".into());
        }
        // mrun first: its context length is everyone's.
        systems.sort_by_key(|s| !s.is_mrun());
        let agent = all || args.suite == Suite::Agent;
        let traces = if agent {
            load_traces(&args.traces)?
        } else {
            Vec::new()
        };
        let n = args.reps.unwrap_or(0);
        Ok(Self {
            agent,
            synthetic: all || args.suite == Suite::Synthetic,
            cancel: all || args.suite == Suite::Cancel,
            traces,
            systems,
            reps: Reps {
                synthetic: MIN_REPS.synthetic.max(n),
                cancel: MIN_REPS.cancel.max(n),
                agent: MIN_REPS.agent.max(n),
            },
        })
    }
}

fn default_url() -> String {
    format!("http://127.0.0.1:{}", crate::cli::server::default_port())
}

/// `--trace` names (a file, or a name under `bench/traces/`); none means every bundled trace.
fn load_traces(names: &[String]) -> Result<Vec<(String, Vec<TraceRecord>)>, String> {
    let paths: Vec<PathBuf> = if names.is_empty() {
        let mut found: Vec<PathBuf> = std::fs::read_dir(TRACE_DIR)
            .into_iter()
            .flatten()
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|e| e == "jsonl"))
            .collect();
        found.sort();
        found
    } else {
        names
            .iter()
            .map(|n| {
                let direct = PathBuf::from(n);
                if direct.is_file() {
                    Ok(direct)
                } else {
                    let bundled = Path::new(TRACE_DIR).join(format!("{n}.jsonl"));
                    bundled
                        .is_file()
                        .then_some(bundled)
                        .ok_or_else(|| format!("trace '{n}' not found (looked in {TRACE_DIR}/)"))
                }
            })
            .collect::<Result<_, _>>()?
    };
    paths
        .into_iter()
        .map(|p| {
            let name = p
                .file_stem()
                .map_or_else(|| "trace".into(), |s| s.to_string_lossy().into_owned());
            let records = trace::load(&p)?;
            if records.is_empty() {
                return Err(format!("{}: empty trace", p.display()));
            }
            Ok((name, records))
        })
        .collect()
}

/// The cells `system` would have produced, all "not run" for `reason`.
fn not_run_cells(plan: &Plan, system: &System, common: Option<u32>, reason: &str) -> Vec<Cell> {
    let api = system.preferred_api().map_or("chat", |a| a.as_str());
    let mut cells = Vec::new();
    if plan.synthetic {
        for (label, _) in synthetic::sizes(common) {
            for kind in ["prefill", "decode"] {
                let name = format!("synthetic/{kind}-{label}");
                cells.push(not_run(&system.name, &name, api, "tok/s", reason));
            }
        }
    }
    if plan.agent {
        for (name, records) in &plan.traces {
            let api = records.first().map_or(api, |r| r.api.as_str());
            cells.push(not_run(
                &system.name,
                &format!("agent/{name}"),
                api,
                "ms",
                reason,
            ));
        }
    }
    if plan.cancel && system.is_mrun() {
        cells.push(not_run(
            &system.name,
            "cancel/disconnect",
            api,
            "ms",
            reason,
        ));
    }
    cells
}

/// Runs the plan; also returns whether any system was reachable.
pub async fn execute(plan: &Plan, interactive: bool) -> (RunResult, bool) {
    let mut result = RunResult {
        schema_version: SCHEMA_VERSION,
        started_at: rfc3339_now(),
        mrun_version: env!("CARGO_PKG_VERSION").into(),
        machine: machine::machine(),
        method: Method {
            replay: "in-order".into(),
            reps: plan.reps.clone(),
        },
        ..RunResult::default()
    };
    if result.machine.power_source == "battery" {
        result.warnings.push("running on battery".into());
    }
    if result.machine.low_power {
        result.warnings.push("Low Power Mode is on".into());
    }
    if plan.agent && plan.traces.is_empty() {
        result
            .warnings
            .push(format!("no agent traces found in {TRACE_DIR}/"));
    }
    let mut ctx = Ctx::new(interactive);
    let explicit = plan
        .systems
        .iter()
        .map(|s| s.profile.context_length)
        .filter(|&c| c > 0)
        .min();
    let mut mrun_context = None;
    let mut reachable = false;
    let mut mrun_record: Option<SystemUnderTest> = None;
    for (i, system) in plan.systems.iter().enumerate() {
        let common = [explicit, mrun_context].into_iter().flatten().min();
        let hint = system.start_hint(common.unwrap_or(0));
        if interactive {
            if i > 0 {
                let previous = &plan.systems[i - 1].name;
                systems::pause(&format!(
                    "Stop {previous}, start {} ({hint}), then press Enter",
                    system.name
                ));
            } else if system.health(&ctx.client).await.is_err() {
                systems::pause(&format!("Start {} ({hint}), then press Enter", system.name));
            }
        }
        let skip = match system.health(&ctx.client).await {
            Err(error) => Some(error),
            Ok(()) if system.needs_manual_reset() && !interactive => {
                Some("needs a manual restart before each repetition; run from a terminal".into())
            }
            Ok(()) => None,
        };
        if let Some(reason) = skip {
            crate::cli::info(&format!("{}: not run ({reason})", system.name));
            result.systems.push(system.describe("unknown".into(), 0));
            result
                .cells
                .extend(not_run_cells(plan, system, common, &reason));
            continue;
        }
        reachable = true;
        crate::cli::info(&format!("benchmarking {} at {}", system.name, system.url));
        let memory = machine::snapshot(&system.name);
        if let Some(first) = result.machine.per_system.first()
            && first.free_memory_before > memory.free_memory_before + LOW_MEMORY_GAP
        {
            result.warnings.push(format!(
                "free memory before {} is more than 2 GiB below the first system's (is {} still loaded?)",
                system.name, first.system
            ));
        }
        if memory.load_average.is_some_and(|l| l > 2.0) {
            result
                .warnings
                .push(format!("machine under load before {}", system.name));
        }
        result.machine.per_system.push(memory);
        let reported = system.reported_context(&ctx.client).await;
        if system.is_mrun() && mrun_context.is_none() {
            mrun_context = reported;
        }
        let common = [explicit, mrun_context].into_iter().flatten().min();
        let context = match system.profile.context_length {
            0 => reported.or(mrun_context),
            n => Some(n),
        };
        let version = system.version(&ctx.client).await;
        let record = system.describe(version, context.unwrap_or(0));
        if system.is_mrun() && mrun_record.is_none() {
            mrun_record = Some(record.clone());
        }
        result.systems.push(record);
        if plan.synthetic {
            let cells = synthetic::run(&mut ctx, system, context, common, plan.reps.synthetic);
            result.cells.extend(cells.await);
        }
        for (name, records) in &plan.traces {
            let cell = agent::run(&mut ctx, system, name, records, plan.reps.agent).await;
            result.cells.push(cell);
        }
        if plan.cancel && system.is_mrun() {
            let cell = cancel::run(&mut ctx, system, plan.reps.cancel).await;
            result.cells.push(cell);
        }
    }
    result.resets = std::mem::take(&mut ctx.resets);
    finish(&mut result, mrun_record.as_ref());
    (result, reachable)
}

/// Thermal warnings, failed resets and comparability labels.
fn finish(result: &mut RunResult, mrun: Option<&SystemUnderTest>) {
    for cell in &result.cells {
        for state in [&cell.thermal_start, &cell.thermal_end] {
            if !state.is_empty() && state != "nominal" && state != "unknown" {
                let warning = format!(
                    "thermal state '{state}' during {} on {}",
                    cell.cell, cell.system
                );
                if !result.warnings.contains(&warning) {
                    result.warnings.push(warning);
                }
            }
        }
    }
    for reset in result.resets.iter().filter(|r| !r.ok) {
        let warning = format!(
            "{}: cache reset '{}' did not succeed",
            reset.system, reset.step
        );
        if !result.warnings.contains(&warning) {
            result.warnings.push(warning);
        }
    }
    for system in result
        .systems
        .iter()
        .filter(|s| Some(&s.name) != mrun.map(|m| &m.name))
    {
        let kind = match mrun {
            None => "n/a",
            Some(_) if lossy_kv(&system.model.kv_format) => "lossy-kv",
            Some(m)
                if m.model.format != system.model.format
                    || m.model.bits_per_weight != system.model.bits_per_weight =>
            {
                "different-quantization"
            }
            Some(_) => "same-weights",
        };
        let mut names: Vec<&str> = result
            .cells
            .iter()
            .filter(|c| c.system == system.name && c.status != CellStatus::NotRun)
            .map(|c| c.cell.as_str())
            .collect();
        names.dedup();
        for cell in names {
            result.comparability.push(Comparability {
                cell: cell.into(),
                system: system.name.clone(),
                kind: kind.into(),
            });
        }
    }
    result.finished_at = rfc3339_now();
}

/// Writes `result.json` and `report.md` into `dir`.
pub fn write(result: &RunResult, dir: &Path) -> std::io::Result<()> {
    std::fs::create_dir_all(dir)?;
    let json = serde_json::to_string_pretty(result).map_err(std::io::Error::other)?;
    std::fs::write(dir.join("result.json"), json + "\n")?;
    std::fs::write(dir.join("report.md"), report::render(result))
}

fn runtime() -> Result<tokio::runtime::Runtime, CliError> {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|e| CliError::new(Code::Internal, e.to_string()))
}

pub fn run(args: BenchArgs) -> u8 {
    if let Some(BenchCommand::Trace {
        record,
        out,
        tokenizer,
    }) = &args.command
    {
        return crate::cli::exit(build_trace(record, out, tokenizer.as_deref()));
    }
    let plan = match Plan::from_args(&args) {
        Ok(plan) => plan,
        Err(error) => {
            eprintln!("error: {error}\n  hint: see 'mrun bench --help'");
            return 2;
        }
    };
    let runtime = match runtime() {
        Ok(runtime) => runtime,
        Err(error) => return error.report(),
    };
    let (result, reachable) = runtime.block_on(execute(&plan, systems::interactive()));
    let dir = args
        .out
        .unwrap_or_else(|| PathBuf::from("mrun-bench").join(result.started_at.replace(':', "-")));
    if let Err(error) = write(&result, &dir) {
        return CliError::new(
            Code::Internal,
            format!("cannot write {}: {error}", dir.display()),
        )
        .report();
    }
    crate::cli::info(&format!("wrote {}", dir.display()));
    if !reachable {
        let names: Vec<&str> = plan.systems.iter().map(|s| s.name.as_str()).collect();
        return CliError::new(
            Code::ServerUnreachable,
            format!("no system was reachable ({})", names.join(", ")),
        )
        .hint("start the server first ('mrun serve --detach'), or check --system URLs")
        .report();
    }
    0
}

fn build_trace(record: &Path, out: &Path, tokenizer: Option<&Path>) -> Result<(), CliError> {
    let internal = |e: String| CliError::new(Code::Internal, e);
    let text = std::fs::read_to_string(record)
        .map_err(|e| internal(format!("{}: {e}", record.display())))?;
    let tokenizer = tokenizer
        .map(crate::chat::tokenizer::Tokenizer::from_file)
        .transpose()
        .map_err(|e| internal(e.to_string()))?;
    let records = trace::build(&text, tokenizer.as_ref()).map_err(internal)?;
    let lines: Vec<String> = records
        .iter()
        .filter_map(|r| serde_json::to_string(r).ok())
        .collect();
    std::fs::write(out, lines.join("\n") + "\n")
        .map_err(|e| internal(format!("{}: {e}", out.display())))?;
    let conversations: std::collections::HashSet<&str> =
        records.iter().map(|r| r.conversation.as_str()).collect();
    crate::cli::info(&format!(
        "wrote {} requests in {} conversations to {}",
        records.len(),
        conversations.len(),
        out.display()
    ));
    Ok(())
}
