//! Engine benchmark (FR-026, R16): prefill and decode speed and cancellation latency at given
//! depths, as JSON with the machine, the model identity and the numerics class.
//!
//! ```text
//! cargo run --release -p mrun-engine --features engine --example bench_engine -- \
//!     <model-dir> [--depths 1024,16384] [--decode 128] [--cancel 50] [--runs 2] \
//!     [--no-store] [--budget-gib 10.67] > result.json
//! ```
//!
//! Per depth `d`:
//! - **prefill**: `d` prompt tokens ([`prompt_ids`], the same ids `scripts/bench_reference.sh`
//!   feeds mlx-lm), all but the last in the engine's own chunk plan (`ModelInfo::chunk_len`),
//!   the last through `forward` + `sample`; tokens per second until the first token is read back,
//!   as mlx-lm measures its prompt speed. Best of `--runs`.
//! - **decode**: `--decode` greedy tokens after that prompt (end tokens ignored), pipelined;
//!   `(n − 1) / time` as mlx-lm reports. Best of `--runs`.
//! - **cancel** (SC-004): at depth `d` (a checkpoint), `--cancel` trials of one chunk each,
//!   cancelled from another thread after a random delay within the chunk's measured duration;
//!   latency = cancel signal to `prefill_chunk` returning with the state as before (ready for
//!   the next prompt). Trials whose chunk finished before the signal are retried (at most
//!   3 × trials in all). p50, p95 and max reported.
//!
//! Depths above the token capacity are skipped and reported as such. `--budget-gib` lowers the
//! budget (a smaller Mac's working set, T062); `--depths ''` only loads and reports the budget.

use std::path::PathBuf;
use std::process::{Command, ExitCode};
use std::sync::mpsc;
use std::time::{Duration, Instant};

use mrun_engine::{CancelToken, Engine, LoadOptions, NuvecEngine, Sampling, calibration, identity};
use serde_json::{Value, json};

/// Prompt token `i` of the benchmark (also in `scripts/bench_reference.py`): spread over ordinary
/// vocabulary ids so no special token appears.
pub fn prompt_ids(n: u32) -> Vec<u32> {
    (0..n).map(|i| 1000 + (i * 7919 + 13) % 30_000).collect()
}

struct Args {
    dir: PathBuf,
    depths: Vec<u32>,
    decode: u32,
    cancel: u32,
    runs: u32,
    store: bool,
    budget: Option<u64>,
}

fn parse() -> Result<Args, String> {
    let mut args = std::env::args().skip(1);
    let mut out = Args {
        dir: PathBuf::new(),
        depths: vec![1024, 16_384],
        decode: 128,
        cancel: 50,
        runs: 2,
        store: true,
        budget: None,
    };
    let num = |v: Option<String>, what: &str| -> Result<u32, String> {
        v.ok_or(format!("{what} needs a value"))?
            .parse()
            .map_err(|e| format!("bad {what}: {e}"))
    };
    let mut dir = None;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--depths" => {
                out.depths = args
                    .next()
                    .ok_or("--depths needs a value")?
                    .split(',')
                    .filter(|d| !d.trim().is_empty())
                    .map(|d| d.trim().parse().map_err(|e| format!("bad depth {d}: {e}")))
                    .collect::<Result<_, _>>()?;
            }
            "--decode" => out.decode = num(args.next(), "--decode")?,
            "--cancel" => out.cancel = num(args.next(), "--cancel")?,
            "--runs" => out.runs = num(args.next(), "--runs")?.max(1),
            "--no-store" => out.store = false,
            "--budget-gib" => {
                let gib: f64 = args
                    .next()
                    .ok_or("--budget-gib needs a value")?
                    .parse()
                    .map_err(|e| format!("bad --budget-gib: {e}"))?;
                out.budget = Some((gib * (1u64 << 30) as f64) as u64);
            }
            other if dir.is_none() && !other.starts_with("--") => dir = Some(PathBuf::from(other)),
            other => return Err(format!("unexpected argument {other}")),
        }
    }
    out.dir = dir.ok_or("usage: bench_engine <model-dir> [--depths a,b] [--decode n] [--cancel n] [--runs n] [--no-store]")?;
    Ok(out)
}

fn sh(cmd: &str, args: &[&str]) -> Option<String> {
    let out = Command::new(cmd).args(args).output().ok()?;
    out.status
        .success()
        .then(|| String::from_utf8_lossy(&out.stdout).trim().to_string())
}

fn machine() -> Value {
    let power = sh("pmset", &["-g", "batt"]).and_then(|t| {
        t.lines()
            .next()
            .and_then(|l| l.split('\'').nth(1).map(str::to_string))
    });
    let mut machine = json!({
        "chip": mrun_engine::tensor::gpu_info_string(c"device_name"),
        "memory_bytes": mrun_engine::tensor::device_memory().map(|m| m.0),
        "recommended_working_set": mrun_engine::tensor::device_memory().map(|m| m.1),
        "macos": sh("sw_vers", &["-productVersion"]),
        "power": power,
    });
    machine["nuvec"] = json!(mrun_engine::tensor::backend_info().version);
    machine
}

/// Prefills `tokens` at `len()` in the engine's chunk plan.
fn prefill(engine: &mut NuvecEngine, tokens: &[u32]) -> Result<(), String> {
    let cancel = CancelToken::new();
    let mut at = 0usize;
    while at < tokens.len() {
        let depth = engine.len();
        let len = engine.info().chunk_len(depth) as usize;
        let end = (at + len).min(tokens.len());
        engine
            .prefill_chunk(&tokens[at..end], &cancel)
            .map_err(|e| e.to_string())?;
        at = end;
    }
    Ok(())
}

/// Prompt and decode speed at `depth`.
fn speed(engine: &mut NuvecEngine, depth: u32, decode: u32) -> Result<(f64, f64), String> {
    let err = |e: mrun_engine::EngineError| e.to_string();
    engine.reset();
    engine.set_limit(depth + decode + 1).map_err(err)?;
    let prompt = prompt_ids(depth);
    let (last, body) = prompt.split_last().ok_or("depth 0")?;
    let start = Instant::now();
    prefill(engine, body)?;
    let step = engine.forward(*last).map_err(err)?;
    let mut token = engine.sample(step, &Sampling::GREEDY, None).map_err(err)?;
    let prompt_tps = f64::from(depth) / start.elapsed().as_secs_f64();
    let start = Instant::now();
    for _ in 1..decode {
        let step = engine.forward(token).map_err(err)?;
        token = engine.sample(step, &Sampling::GREEDY, None).map_err(err)?;
    }
    let decode_tps = f64::from(decode.saturating_sub(1)) / start.elapsed().as_secs_f64();
    engine.reset();
    Ok((prompt_tps, decode_tps))
}

/// Cancel latencies at `depth`, in milliseconds.
fn cancel_latency(engine: &mut NuvecEngine, depth: u32, trials: u32) -> Result<Vec<f64>, String> {
    let err = |e: mrun_engine::EngineError| e.to_string();
    engine.reset();
    let len = engine.info().chunk_len(depth);
    engine.set_limit(depth + len).map_err(err)?;
    let prompt = prompt_ids(depth + len);
    prefill(engine, &prompt[..depth as usize])?;
    let at_depth = engine.checkpoint().map_err(err)?;
    let chunk = &prompt[depth as usize..];
    // The chunk's duration bounds the random delays.
    let start = Instant::now();
    engine
        .prefill_chunk(chunk, &CancelToken::new())
        .map_err(err)?;
    let duration = start.elapsed();
    engine.restore(at_depth.id).map_err(err)?;

    let mut seed = 0x2545_F491_4F6C_DD1Du64 ^ u64::from(depth);
    let mut latencies = Vec::new();
    let mut attempts = 0;
    while latencies.len() < trials as usize && attempts < 3 * trials {
        attempts += 1;
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        let delay = duration.mul_f64((seed % 10_000) as f64 / 10_000.0);
        let cancel = CancelToken::new();
        let (tx, rx) = mpsc::channel();
        let signal = cancel.clone();
        let canceller = std::thread::spawn(move || {
            std::thread::sleep(delay);
            let at = Instant::now();
            signal.cancel();
            let _ = tx.send(at);
        });
        let committed = engine.prefill_chunk(chunk, &cancel).map_err(err)?;
        let returned = Instant::now();
        canceller.join().map_err(|_| "canceller panicked")?;
        let signalled = rx.recv().map_err(|e| e.to_string())?;
        if committed {
            // Finished before the signal: undo and retry.
            engine.restore(at_depth.id).map_err(err)?;
            continue;
        }
        assert_eq!(engine.len(), depth, "a cancelled chunk leaves the state");
        latencies.push(returned.saturating_duration_since(signalled).as_secs_f64() * 1e3);
    }
    engine.drop_checkpoint(at_depth.id);
    engine.reset();
    Ok(latencies)
}

fn percentile(sorted: &[f64], p: f64) -> Option<f64> {
    if sorted.is_empty() {
        return None;
    }
    let rank = ((p / 100.0) * sorted.len() as f64).ceil() as usize;
    Some(sorted[rank.clamp(1, sorted.len()) - 1])
}

fn run() -> Result<(), String> {
    let args = parse()?;
    let cache = calibration::default_path()
        .map(|p| p.with_file_name("identity.toml"))
        .ok_or("no home directory")?;
    let identity = identity::identify(&args.dir, &cache)
        .map_err(|e| e.to_string())?
        .value;
    let loaded = Instant::now();
    let mut engine = NuvecEngine::load(
        &args.dir,
        LoadOptions {
            identity: identity.clone(),
            calibration: if args.store {
                calibration::default_path()
            } else {
                None
            },
            overrides: mrun_engine::budget::Overrides {
                budget: args.budget,
            },
            ..LoadOptions::default()
        },
    )
    .map_err(|e| e.to_string())?;
    let load_s = loaded.elapsed().as_secs_f64();
    let info = engine.info().clone();
    let budget = *engine.budget();
    eprintln!(
        "{} loaded in {load_s:.1} s; numerics {}; capacity {} tokens; time_len {:?}",
        info.family,
        info.numerics,
        budget.kv_capacity_tokens,
        info.time_len.points()
    );

    let mut results = Vec::new();
    for &depth in &args.depths {
        if depth + args.decode + 1 > budget.kv_capacity_tokens || depth == 0 {
            eprintln!(
                "depth {depth}: skipped (capacity {})",
                budget.kv_capacity_tokens
            );
            results.push(json!({"depth": depth, "skipped": "above the token capacity"}));
            continue;
        }
        let mut prompt_tps: f64 = 0.0;
        let mut decode_tps: f64 = 0.0;
        for _ in 0..args.runs {
            let (p, d) = speed(&mut engine, depth, args.decode)?;
            prompt_tps = prompt_tps.max(p);
            decode_tps = decode_tps.max(d);
        }
        let mut cancel = Value::Null;
        if args.cancel > 0 && depth + info.chunk_len(depth) <= budget.kv_capacity_tokens {
            let mut ms = cancel_latency(&mut engine, depth, args.cancel)?;
            ms.sort_by(f64::total_cmp);
            cancel = json!({
                "trials": ms.len(),
                "p50_ms": percentile(&ms, 50.0),
                "p95_ms": percentile(&ms, 95.0),
                "max_ms": ms.last(),
            });
        }
        eprintln!(
            "depth {depth}: prompt {prompt_tps:.1} tok/s, decode {decode_tps:.1} tok/s, cancel {cancel}"
        );
        results.push(json!({
            "depth": depth,
            "chunk_len": info.chunk_len(depth),
            "prompt_tps": prompt_tps,
            "decode_tps": decode_tps,
            "decode_tokens": args.decode,
            "cancel": cancel,
        }));
        // Let the GPU cool down a little between depths.
        std::thread::sleep(Duration::from_millis(500));
    }
    let report = json!({
        "tool": "bench_engine",
        "machine": machine(),
        "model": {
            "dir": args.dir.display().to_string(),
            "identity": identity,
            "family": info.family,
            "numerics": {"nax": info.numerics.nax, "gpu_arch": info.numerics.gpu_arch},
            "time_len": info.time_len.points(),
            "load_s": load_s,
        },
        "budget": {
            "budget": budget.budget,
            "weights": budget.weights,
            "scratch": budget.scratch,
            "scratch_depth": budget.scratch_depth,
            "state_capacity_bytes": budget.state_capacity_bytes,
            "kv_capacity_tokens": budget.kv_capacity_tokens,
        },
        "results": results,
    });
    println!(
        "{}",
        serde_json::to_string_pretty(&report).map_err(|e| e.to_string())?
    );
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("error: {message}");
            ExitCode::FAILURE
        }
    }
}
