//! `mrun bench` (005): reproducible measurements of mrun and other local servers.
//!
//! Systems run one at a time, mrun first: on a terminal the harness pauses with "Stop X, start
//! Y, press Enter" between them; without one it goes straight to each system's health check.
//! Bench talks to the systems under test only (FR-011) and writes `result.json` (schema v3)
//! and `report.md` to `--out` (default `./mrun-bench/<timestamp>/`).
//!
//! mrun ships no agent traces: the agent suite replays the `--trace` files or directories the
//! user names (built from `mrun serve --record` with `mrun bench trace`). Validity rules (D038)
//! live in `suites::agent` (whole sessions, natural-stop pass), `canary` (truncation) and
//! `stats` (paired ratios, bootstrap intervals); [`finish`] builds the comparisons.

pub mod canary;
pub mod driver;
pub mod machine;
pub mod preflight;
pub mod report;
pub mod result;
pub mod stats;
pub mod suites;
pub mod systems;
pub mod trace;

use crate::api::record::rfc3339_now;
use crate::bench::result::{
    Cell, CellStatus, Comparability, Comparison, Method, Reps, RunResult, SCHEMA_VERSION,
    SystemUnderTest, TurnKind,
};
use crate::bench::suites::{Ctx, agent, cancel, not_run, synthetic};
use crate::bench::systems::{System, lossy_kv};
use crate::bench::trace::TraceRecord;
use crate::errors::{CliError, Code};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// FR-009 minimums; `--reps` raises, never lowers them.
pub const MIN_REPS: Reps = Reps {
    synthetic: 5,
    cancel: 50,
    agent: 3,
};
const NO_TRACE: &str = "the agent suite needs --trace <file|dir>: mrun ships no traces (record a session with 'mrun serve --record' and build one with 'mrun bench trace')";
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
    /// Agent trace: a .jsonl file, or a directory of them (repeatable). Required for the agent
    /// suite; without it `--suite all` skips the agent suite.
    #[arg(long = "trace", value_name = "FILE|DIR")]
    pub traces: Vec<PathBuf>,
    /// NAME=URL[,apis=chat+messages+responses][,model=M][,profile=P]
    /// [default: mrun at http://127.0.0.1:8484].
    #[arg(long = "system", value_name = "SPEC")]
    pub systems: Vec<String>,
    /// Benchmark several systems one at a time (without --system: every bundled profile).
    #[arg(long)]
    pub compare: bool,
    /// Repetitions (raises the per-suite minimums: synthetic 5, cancel 50, agent 3; the synthetic
    /// maximum-context cell runs once).
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
    /// Notes for the result's warnings (a skipped suite).
    pub notes: Vec<String>,
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
        let mut notes = Vec::new();
        let mut agent = all || args.suite == Suite::Agent;
        if agent && args.traces.is_empty() {
            if !all {
                return Err(NO_TRACE.into());
            }
            agent = false;
            notes.push(format!("agent suite skipped: {NO_TRACE}"));
        }
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
            notes,
        })
    }
}

fn default_url() -> String {
    format!("http://127.0.0.1:{}", crate::cli::server::default_port())
}

/// `--trace` paths: `.jsonl` files, or directories whose `.jsonl` files are taken in name order.
/// Paths are used as given (relative to the working directory), never looked up elsewhere.
fn load_traces(paths: &[PathBuf]) -> Result<Vec<(String, Vec<TraceRecord>)>, String> {
    let mut files = Vec::new();
    for path in paths {
        if path.is_dir() {
            let mut found: Vec<PathBuf> = std::fs::read_dir(path)
                .map_err(|e| format!("{}: {e}", path.display()))?
                .flatten()
                .map(|e| e.path())
                .filter(|p| p.is_file() && p.extension().is_some_and(|e| e == "jsonl"))
                .collect();
            if found.is_empty() {
                return Err(format!("{}: no .jsonl traces in it", path.display()));
            }
            found.sort();
            files.extend(found);
        } else if path.is_file() {
            files.push(path.clone());
        } else {
            return Err(format!(
                "trace '{}' is neither a file nor a directory",
                path.display()
            ));
        }
    }
    files
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
            resamples: stats::RESAMPLES,
            seed: stats::SEED,
        },
        warnings: plan.notes.clone(),
        ..RunResult::default()
    };
    if result.machine.power_source == "battery" {
        result.warnings.push("running on battery".into());
    }
    if result.machine.low_power {
        result.warnings.push("Low Power Mode is on".into());
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
    let mrun_weights = mrun_model_bytes();
    for (i, system) in plan.systems.iter().enumerate() {
        let common = [explicit, mrun_context].into_iter().flatten().min();
        let hint = system.start_hint(common.unwrap_or(0));
        let mut refused = None;
        if interactive {
            let up = system.health(&ctx.client).await.is_ok();
            if i > 0 || !up {
                // D062: the previous server has exited and its memory is back before the next
                // one starts.
                if i > 0 {
                    let previous = &plan.systems[i - 1].name;
                    systems::pause(&format!("Stop {previous}, then press Enter"));
                }
                refused = before_start(system, mrun_weights).err();
                if let Some(reason) = &refused {
                    systems::pause(&format!("{reason}. Fix it, then press Enter"));
                    refused = before_start(system, mrun_weights).err();
                }
                if refused.is_none() {
                    systems::pause(&format!("Start {} ({hint}), then press Enter", system.name));
                }
            }
        }
        let skip = match system.health(&ctx.client).await {
            _ if refused.is_some() => refused,
            Err(error) => Some(error),
            Ok(()) if system.needs_manual_reset() && !interactive => {
                Some("needs a manual restart before each repetition; run from a terminal".into())
            }
            Ok(()) if preflight::is_local(system) => {
                preflight::check_servers(&preflight::running_servers(), true).err()
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
        // 013 FR-014: a relay to a remote backend is not the local engine; its cells would mix
        // a rented GPU into a Mac comparison, so it is left out with a warning.
        if system.is_mrun()
            && let Some(backend) = system.remote_backend(&ctx.client).await
        {
            result.warnings.push(format!(
                "{} relays to a remote backend ({backend}); remote mode is not benchmarked",
                system.name
            ));
            continue;
        }
        let version = system.version(&ctx.client).await;
        let mut record = system.describe(version, context.unwrap_or(0));
        if system.is_mrun() {
            (record.lanes, record.numerics) = system.status_facts(&ctx.client).await;
        }
        if system.is_mrun() && mrun_record.is_none() {
            mrun_record = Some(record.clone());
        }
        result.systems.push(record);
        if plan.synthetic && system.runs("synthetic") {
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

/// Bytes of the model mrun serves on this machine (its default alias), the yardstick for the
/// other systems' models (D062).
fn mrun_model_bytes() -> u64 {
    crate::hub::aliases::default_alias(crate::hub::device_memory()).loaded_weights()
}

/// D062: before a local system starts, no known server runs and the available memory holds
/// its model plus the margin.
fn before_start(system: &System, mrun_weights: u64) -> Result<(), String> {
    if !preflight::is_local(system) {
        return Ok(());
    }
    preflight::check_servers(&preflight::running_servers(), false)?;
    let device = crate::hub::device_memory();
    let others = mrun_engine::sysmem::processes();
    preflight::check_memory(
        &system.name,
        preflight::model_bytes(system, mrun_weights),
        device.and_then(|d| d.available),
        device.map_or(0, |d| d.physical),
        &others,
    )
}

/// Thermal warnings, failed resets, comparability labels and comparisons.
fn finish(result: &mut RunResult, mrun: Option<&SystemUnderTest>) {
    for cell in &result.cells {
        if !cell.incomplete.is_empty() {
            result.warnings.push(format!(
                "{} on {}: {} incomplete session(s) excluded",
                cell.cell,
                cell.system,
                cell.incomplete.len()
            ));
        }
        if let Some(check) = cell.truncation_check.as_deref()
            && check.starts_with("unverified")
        {
            let warning = format!("{}: truncation {check}", cell.system);
            if !result.warnings.contains(&warning) {
                result.warnings.push(warning);
            }
        }
    }
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
    for cell in &mut result.cells {
        if let Some(system) = result.systems.iter().find(|s| s.name == cell.system) {
            cell.lossy_kv = system.lossy_kv;
            cell.speculation = Some(system.speculation.clone());
        }
    }
    for system in result
        .systems
        .iter()
        .filter(|s| Some(&s.name) != mrun.map(|m| &m.name))
    {
        let known = |nax: &str| nax != "unknown";
        let kind = match mrun {
            None => "n/a",
            Some(_) if system.lossy_kv || lossy_kv(&system.model.kv_format) => "lossy-kv",
            Some(m)
                if known(&m.numerics.nax)
                    && known(&system.numerics.nax)
                    && (m.numerics.nax != system.numerics.nax
                        || m.numerics.gpu_family != system.numerics.gpu_family
                        || (m.numerics.backend.is_some()
                            && system.numerics.backend.is_some()
                            && m.numerics.backend != system.numerics.backend)) =>
            {
                "different-numerics"
            }
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
    if let Some(mrun) = mrun {
        result.comparisons = comparisons(result, &mrun.name);
    }
    result.finished_at = rfc3339_now();
}

/// `> 1` means mrun is better on a cell measured in `unit`.
fn speedup(unit: &str, mrun: f64, other: f64) -> f64 {
    if unit == "tok/s" {
        mrun / other
    } else {
        other / mrun
    }
}

/// Four decimals: enough for a ratio, and exact through a JSON round trip.
fn round4(x: f64) -> f64 {
    (x * 1e4).round() / 1e4
}

fn verdict(low: f64, high: f64) -> &'static str {
    if low > 1.0 {
        "mrun-faster"
    } else if high < 1.0 {
        "mrun-slower"
    } else {
        "within-noise"
    }
}

/// Per-turn medians of TTFT over a cell's complete sessions, for turns of `kind` (all when
/// `None`).
fn turn_medians(cell: &Cell, kind: Option<TurnKind>) -> BTreeMap<u32, f64> {
    let mut by_turn: BTreeMap<u32, Vec<f64>> = BTreeMap::new();
    for t in cell
        .turns
        .iter()
        .flatten()
        .filter(|t| kind.is_none_or(|k| t.kind == k))
    {
        by_turn.entry(t.n).or_default().push(t.ttft_ms);
    }
    by_turn
        .into_iter()
        .filter_map(|(n, v)| stats::median(&v).map(|m| (n, m)))
        .collect()
}

/// Session times by repetition.
fn session_times(cell: &Cell) -> BTreeMap<u32, f64> {
    cell.session_times.iter().map(|t| (t.rep, t.s)).collect()
}

fn kind_name(kind: TurnKind) -> &'static str {
    match kind {
        TurnKind::New => "new",
        TurnKind::Repeated => "repeated",
        TurnKind::Branching => "branching",
    }
}

/// Paired comparisons of mrun's values with each other system's over the keys (turns or
/// repetitions) every compared system has (FR-006).
#[allow(clippy::too_many_arguments)]
fn paired(
    out: &mut Vec<Comparison>,
    ours: &Cell,
    others: &[&Cell],
    mine: &BTreeMap<u32, f64>,
    theirs: &[BTreeMap<u32, f64>],
    scope: Option<&str>,
    method: &str,
    headline: &dyn Fn(&str) -> bool,
) {
    let common: Vec<u32> = mine
        .keys()
        .copied()
        .filter(|n| theirs.iter().all(|t| t.contains_key(n)))
        .collect();
    for (other, values) in others.iter().zip(theirs) {
        let ratios: Vec<f64> = common
            .iter()
            .map(|n| speedup(&ours.unit, mine[n], values[n]))
            .collect();
        if let Some(i) = stats::paired(&ratios, stats::RESAMPLES, stats::SEED) {
            out.push(Comparison {
                cell: ours.cell.clone(),
                system: other.system.clone(),
                scope: scope.map(Into::into),
                method: method.into(),
                n: ratios.len() as u32,
                speedup: round4(i.estimate),
                ci_low: round4(i.low),
                ci_high: round4(i.high),
                verdict: verdict(i.low, i.high).into(),
                headline: headline(&other.system),
            });
        }
    }
}

/// mrun against every other system on every cell both measured (D038, FR-006). Agent cells
/// pair turns over the turns every system with a complete session completed, overall and per
/// turn kind, and pair session times over the repetitions valid for every system; synthetic
/// cells compare medians. Unverified profiles are compared but never headline.
pub fn comparisons(result: &RunResult, mrun: &str) -> Vec<Comparison> {
    let verified = |name: &str| {
        result
            .systems
            .iter()
            .find(|s| s.name == name)
            .is_none_or(|s| s.verified)
    };
    let usable = |c: &&Cell| c.status == CellStatus::Ok && !c.samples.is_empty();
    let mut out = Vec::new();
    for ours in result
        .cells
        .iter()
        .filter(|c| c.system == mrun)
        .filter(usable)
    {
        let others: Vec<&Cell> = result
            .cells
            .iter()
            .filter(|c| c.system != mrun && c.cell == ours.cell && c.api == ours.api)
            .filter(usable)
            .collect();
        if ours.turns.is_some() {
            let scopes = [
                None,
                Some(TurnKind::New),
                Some(TurnKind::Repeated),
                Some(TurnKind::Branching),
            ];
            for kind in scopes {
                let mine = turn_medians(ours, kind);
                let theirs: Vec<_> = others.iter().map(|c| turn_medians(c, kind)).collect();
                paired(
                    &mut out,
                    ours,
                    &others,
                    &mine,
                    &theirs,
                    kind.map(kind_name),
                    "paired-turns",
                    &verified,
                );
            }
            let mine = session_times(ours);
            let theirs: Vec<_> = others.iter().map(|c| session_times(c)).collect();
            paired(
                &mut out,
                ours,
                &others,
                &mine,
                &theirs,
                Some("session"),
                "paired-sessions",
                &verified,
            );
        } else {
            for other in others {
                let (num, den) = if ours.unit == "tok/s" {
                    (&ours.samples, &other.samples)
                } else {
                    (&other.samples, &ours.samples)
                };
                if let Some(i) = stats::ratio_of_medians(num, den, stats::RESAMPLES, stats::SEED) {
                    out.push(Comparison {
                        cell: ours.cell.clone(),
                        system: other.system.clone(),
                        scope: None,
                        method: "median-ratio".into(),
                        n: ours.samples.len().min(other.samples.len()) as u32,
                        speedup: round4(i.estimate),
                        ci_low: round4(i.low),
                        ci_high: round4(i.high),
                        verdict: verdict(i.low, i.high).into(),
                        headline: verified(&other.system),
                    });
                }
            }
        }
    }
    out
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
            return CliError::new(Code::InvalidArguments, error.to_string())
                .hint("see 'mrun bench --help'")
                .report();
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bench::result::{SessionTime, Turn, TurnClass};

    fn turn(n: u32, kind: TurnKind, rep: u32, ttft_ms: f64) -> Turn {
        Turn {
            n,
            conversation: "c".into(),
            kind,
            ttft_ms,
            tfut_ms: ttft_ms,
            input: 0,
            cached: 0,
            output: 0,
            cause: None,
            message_index: None,
            message_role: None,
            gap: None,
            unreused: None,
            rep: Some(rep),
            class: TurnClass::Main,
            offset_ms: None,
        }
    }

    /// An agent cell: turn 1 `new`, 2 `repeated`, 3 `branching`, each TTFT `scale` × base, over
    /// the given repetitions; `skip` leaves turn 3 out (a system that never completed it).
    fn agent(system: &str, scale: f64, reps: &[u32], skip: bool) -> Cell {
        let mut turns = Vec::new();
        for &rep in reps {
            let jitter = 1.0 + f64::from(rep) * 0.01;
            turns.push(turn(1, TurnKind::New, rep, 1000.0 * scale * jitter));
            turns.push(turn(2, TurnKind::Repeated, rep, 100.0 * scale * jitter));
            if !skip {
                turns.push(turn(3, TurnKind::Branching, rep, 400.0 * scale * jitter));
            }
        }
        Cell {
            system: system.into(),
            cell: "agent/t".into(),
            api: "chat".into(),
            unit: "ms".into(),
            samples: turns.iter().map(|t| t.ttft_ms).collect(),
            session_times: reps
                .iter()
                .map(|&rep| SessionTime {
                    rep,
                    s: 3.0 * scale * (1.0 + f64::from(rep) * 0.01),
                })
                .collect(),
            turns: Some(turns),
            ..Cell::default()
        }
    }

    #[test]
    fn agent_cells_are_compared_overall_per_kind_and_by_session() {
        let result = RunResult {
            cells: vec![
                agent("mrun", 1.0, &[1, 2, 3], false),
                agent("other", 2.0, &[1, 3], true),
            ],
            ..RunResult::default()
        };
        let all = comparisons(&result, "mrun");
        let find = |scope: Option<&str>| {
            all.iter()
                .find(|c| c.scope.as_deref() == scope)
                .unwrap_or_else(|| panic!("{scope:?} in {all:?}"))
        };
        // Turn 3 is not completed by the other system: two turns pair.
        let overall = find(None);
        assert_eq!((overall.n, overall.method.as_str()), (2, "paired-turns"));
        assert!((overall.speedup - 2.0).abs() < 0.05, "{overall:?}");
        assert_eq!(overall.verdict, "mrun-faster");
        assert_eq!(find(Some("new")).n, 1);
        assert_eq!(find(Some("repeated")).n, 1);
        assert!(
            all.iter().all(|c| c.scope.as_deref() != Some("branching")),
            "no branching turn is common"
        );
        // Sessions pair over the repetitions valid for both (1 and 3).
        let session = find(Some("session"));
        assert_eq!((session.n, session.method.as_str()), (2, "paired-sessions"));
        assert!((session.speedup - 2.0).abs() < 0.05);
        // The other way round mrun is slower, and the losing cells name the scope.
        let swapped = RunResult {
            cells: vec![
                agent("mrun", 2.0, &[1, 2, 3], false),
                agent("other", 1.0, &[1, 2, 3], false),
            ],
            ..RunResult::default()
        };
        let mut swapped = swapped;
        swapped.comparisons = comparisons(&swapped, "mrun");
        assert!(
            swapped
                .comparisons
                .iter()
                .all(|c| c.verdict == "mrun-slower")
        );
        let report = report::render(&swapped);
        assert!(
            report.contains("- agent/t (session): other is faster"),
            "{report}"
        );
    }

    #[test]
    fn numerics_and_lossy_kv_label_comparability() {
        let mut result = RunResult {
            systems: vec![
                SystemUnderTest {
                    name: "mrun".into(),
                    numerics: crate::bench::result::Numerics {
                        backend: None,
                        backend_version: None,
                        kernels: None,
                        nax: "on".into(),
                        gpu_family: Some("g17".into()),
                    },
                    ..SystemUnderTest::default()
                },
                SystemUnderTest {
                    name: "other".into(),
                    numerics: crate::bench::result::Numerics {
                        backend: None,
                        backend_version: None,
                        kernels: None,
                        nax: "off".into(),
                        gpu_family: Some("g17".into()),
                    },
                    ..SystemUnderTest::default()
                },
                SystemUnderTest {
                    name: "lossy".into(),
                    lossy_kv: true,
                    speculation: "dflash".into(),
                    ..SystemUnderTest::default()
                },
            ],
            cells: vec![
                agent("other", 1.0, &[1], false),
                agent("lossy", 1.0, &[1], false),
            ],
            ..RunResult::default()
        };
        let mrun = result.systems[0].clone();
        finish(&mut result, Some(&mrun));
        let kind = |system: &str| {
            result
                .comparability
                .iter()
                .find(|c| c.system == system)
                .map(|c| c.kind.clone())
        };
        assert_eq!(kind("other").as_deref(), Some("different-numerics"));
        // Same NAX state and GPU, other backend (MLX vs nuvec round differently; 016).
        let mut backends = result.clone();
        backends.comparability.clear();
        backends.systems[1].numerics = crate::bench::result::Numerics {
            backend: Some("nuvec".into()),
            nax: "on".into(),
            gpu_family: Some("g17".into()),
            ..crate::bench::result::Numerics::default()
        };
        backends.systems[0].numerics.backend = Some("mlx".into());
        let mrun = backends.systems[0].clone();
        finish(&mut backends, Some(&mrun));
        assert!(
            backends
                .comparability
                .iter()
                .any(|c| c.system == "other" && c.kind == "different-numerics")
        );
        assert_eq!(kind("lossy").as_deref(), Some("lossy-kv"));
        let lossy = result.cells.iter().find(|c| c.system == "lossy").unwrap();
        assert!(lossy.lossy_kv);
        assert_eq!(lossy.speculation.as_deref(), Some("dflash"));
    }
}
