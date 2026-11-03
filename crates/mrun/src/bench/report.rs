//! The markdown report of a run (005 contract, D038): Method, Machine, Systems, Summary
//! (headline comparisons), cells per suite, Comparability, Unverified profiles, Warnings, and
//! Losing cells, always present (possibly "none"): every comparison where mrun is slower.
//!
//! The report measures speed and lists quality signals; it never recommends a default model or
//! system. That decision belongs to the client gate (002 pass rate per tier, speed only as a
//! tie-break), and the Method section says so.

use crate::bench::result::{Cell, CellStatus, Comparison, RunResult};
use std::fmt::Write;

fn fmt_value(v: Option<f64>, unit: &str) -> String {
    v.map_or_else(|| "–".into(), |v| format!("{v:.1} {unit}"))
}

fn status_text(cell: &Cell) -> String {
    match cell.status {
        CellStatus::Ok => fmt_value(cell.median, &cell.unit),
        CellStatus::Failed => format!("failed ({})", cell.failures.len()),
        CellStatus::NotRun => "not run".into(),
        CellStatus::NotApplicable => "n/a".into(),
    }
}

/// Comparisons where mrun is slower, headline first; unverified profiles are marked.
pub fn losing_cells(result: &RunResult) -> Vec<&Comparison> {
    let mut losing: Vec<&Comparison> = result
        .comparisons
        .iter()
        .filter(|c| c.speedup < 1.0)
        .collect();
    losing.sort_by_key(|c| !c.headline);
    losing
}

fn comparison_row(out: &mut String, c: &Comparison) {
    let _ = writeln!(
        out,
        "| {} | {} | {} | {} | {:.2}× | {:.2}–{:.2} | {} | {} |",
        c.cell,
        c.scope.as_deref().unwrap_or("all turns"),
        c.system,
        c.n,
        c.speedup,
        c.ci_low,
        c.ci_high,
        c.verdict,
        c.method
    );
}

/// `agent/x (new)`: the cell with its scope when the comparison has one.
fn scoped(c: &Comparison) -> String {
    match &c.scope {
        Some(scope) => format!("{} ({scope})", c.cell),
        None => c.cell.clone(),
    }
}

const COMPARISON_HEADER: &str = "| Cell | Scope | System | n | mrun speed-up | 95 % CI | Verdict | Method |\n| --- | --- | --- | --- | --- | --- | --- | --- |\n";

pub fn render(result: &RunResult) -> String {
    let mut out = String::new();
    let _ = writeln!(out, "# mrun bench — {}\n", result.started_at);
    out.push_str("## Method\n\n");
    let _ = writeln!(
        out,
        "Agent traces are replayed in recorded order with each system's cache on and reset before every repetition. Every request carries the recorded conversation, including the recorded replies (not the system's own), so all systems see identical prompts and reuse is measured up to the previous reply. Every system gets the same context length. Repetitions: synthetic {}, cancel {}, agent {}.\n",
        result.method.reps.synthetic, result.method.reps.cancel, result.method.reps.agent
    );
    out.push_str("Validity: a session (one replay of a trace) with any failed, truncated or ended-early turn (more than 10 % short of the forced length) is excluded as a whole and listed as incomplete. Truncation is detected with a canary line at the start of the prompt, measured in each system's own tokens. Timing uses a forced output length (ignore-EOS where supported; replies that stop short by at most 10 % are counted as short replies); quality signals (malformed tool calls, early EOS) come from a separate natural-stop pass with the client's own limit. Traces with auxiliary requests recorded beside a main turn replay them at their recorded offsets in parallel (concurrent class) and report main and auxiliary TTFT separately.\n\n");
    out.push_str("Splicing: mrun's splice hit rate per client counts natural-stop-pass requests whose current turn holds an earlier assistant reply and for which mrun spliced (its `/mrun/status` `cache.spliced_tokens_total` grew). Under recorded-reply replay a splice needs the recorded reply to equal mrun's own, so the rate is a lower bound.\n\n");
    let _ = writeln!(
        out,
        "Comparisons: mrun's speed-up (> 1: mrun faster). Agent cells take the geometric mean of per-turn TTFT ratios over the turns every compared system completed, overall and per turn kind, and of session-time ratios over the repetitions valid for both; synthetic cells compare medians. Intervals are bootstrap 95 % ({} resamples, seed {}); \"within-noise\" means the interval includes 1.\n",
        result.method.resamples, result.method.seed
    );
    out.push_str("This report measures speed only. It does not choose a default model or system: the per-machine default is decided by the client gate (pass rate per tier, speed only as a tie-break). It is published whatever it shows; no cell blocks a release.\n\n");
    out.push_str("## Machine\n\n");
    let m = &result.machine;
    let _ = writeln!(
        out,
        "{}, {} GiB, macOS {}, power: {}{}\n",
        m.chip,
        m.memory_bytes >> 30,
        m.macos,
        m.power_source,
        if m.low_power {
            ", Low Power Mode on"
        } else {
            ""
        }
    );
    out.push_str("## Systems\n\n| System | Version | Release | Model | Format | Bits/weight | KV | Lossy KV | Speculation | Lanes | Numerics | Context | Flags |\n| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n");
    for s in &result.systems {
        let backend = match (&s.numerics.backend, &s.numerics.backend_version) {
            (Some(b), Some(v)) => format!("{b} {v}, "),
            (Some(b), None) => format!("{b}, "),
            _ => String::new(),
        };
        let numerics = match &s.numerics.gpu_family {
            Some(family) => format!("{}nax {}, {family}", backend, s.numerics.nax),
            None => format!("{}nax {}", backend, s.numerics.nax),
        };
        let _ = writeln!(
            out,
            "| {}{} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |",
            s.name,
            if s.verified { "" } else { " [U]" },
            s.version,
            s.release,
            s.model.name,
            s.model.format,
            s.model
                .bits_per_weight
                .map_or("–".into(), |b| format!("{b:.2}")),
            s.model.kv_format,
            if s.lossy_kv { "yes" } else { "no" },
            s.speculation,
            if s.lanes == 0 {
                "unknown".to_string()
            } else {
                s.lanes.to_string()
            },
            numerics,
            s.context_length,
            s.flags.join(" ")
        );
    }
    if result.systems.iter().any(|s| !s.verified) {
        out.push_str("\n[U]: unverified profile, reported but kept out of the summary.\n");
    }
    out.push_str("\n## Summary\n\n");
    let headline: Vec<&Comparison> = result.comparisons.iter().filter(|c| c.headline).collect();
    if headline.is_empty() {
        out.push_str("no comparisons\n");
    } else {
        out.push_str(COMPARISON_HEADER);
        for c in headline {
            comparison_row(&mut out, c);
        }
    }
    out.push_str("\n## Cells\n\n| Cell | API | System | Lossy KV | Speculation | Median | Min | Max | p95 | Short replies | Truncation check |\n| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n");
    for c in &result.cells {
        let _ = writeln!(
            out,
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |",
            c.cell,
            c.api,
            c.system,
            if c.lossy_kv { "yes" } else { "no" },
            c.speculation.as_deref().unwrap_or("–"),
            status_text(c),
            fmt_value(c.min, &c.unit),
            fmt_value(c.max, &c.unit),
            fmt_value(c.p95, &c.unit),
            c.short_replies,
            c.truncation_check.as_deref().unwrap_or("–")
        );
    }
    agent_section(&mut out, result);
    if !result.comparability.is_empty() {
        out.push_str("\n## Comparability\n\n");
        for c in &result.comparability {
            let _ = writeln!(out, "- {} ({} vs mrun): {}", c.cell, c.system, c.kind);
        }
    }
    let unverified: Vec<&Comparison> = result.comparisons.iter().filter(|c| !c.headline).collect();
    if !unverified.is_empty() {
        out.push_str("\n## Unverified profiles\n\nNot in the summary: these profiles could not be checked against a release.\n\n");
        out.push_str(COMPARISON_HEADER);
        for c in unverified {
            comparison_row(&mut out, c);
        }
    }
    out.push_str("\n## Warnings\n\n");
    if result.warnings.is_empty() {
        out.push_str("none\n");
    } else {
        for w in &result.warnings {
            let _ = writeln!(out, "- {w}");
        }
    }
    out.push_str("\n## Losing cells\n\n");
    let losing = losing_cells(result);
    if losing.is_empty() {
        out.push_str("none\n");
    } else {
        for c in losing {
            let _ = writeln!(
                out,
                "- {}: {}{} is faster, mrun speed-up {:.2}× (95 % CI {:.2}–{:.2}, {})",
                scoped(c),
                c.system,
                if c.headline { "" } else { " [U]" },
                c.speedup,
                c.ci_low,
                c.ci_high,
                c.verdict
            );
        }
    }
    out
}

/// Agent cells: sessions, session time, the concurrent class's main and auxiliary TTFT,
/// quality signals, mrun's splice hit rate per client and the per-turn TTFT of the first
/// complete session with miss causes and unreused tokens.
fn agent_section(out: &mut String, result: &RunResult) {
    let agent: Vec<&Cell> = result
        .cells
        .iter()
        .filter(|c| c.cell.starts_with("agent/") && c.turns.is_some())
        .collect();
    if agent.is_empty() {
        return;
    }
    out.push_str("\n## Agent suite\n");
    for c in agent {
        let complete = c.complete_sessions.unwrap_or(0);
        let _ = writeln!(
            out,
            "\n### {} on {}\n\nComplete sessions {} of {}, session {}",
            c.cell,
            c.system,
            complete,
            complete as usize + c.incomplete.len(),
            fmt_value(c.session_s, "s"),
        );
        for f in &c.incomplete {
            let _ = writeln!(out, "- session {} excluded: {}", f.rep, f.error);
        }
        if c.class.as_deref() == Some("concurrent") {
            let _ = writeln!(
                out,
                "\nConcurrent class: main-turn TTFT {}, auxiliary TTFT {}",
                fmt_value(c.main_ttft, "ms"),
                fmt_value(c.aux_ttft, "ms")
            );
        }
        if let Some(q) = &c.quality {
            let _ = writeln!(
                out,
                "\nQuality (natural-stop pass, {} turns{}): malformed tool calls {}, early EOS {}, length-capped {}",
                q.turns,
                if q.complete { "" } else { ", incomplete" },
                q.malformed_tool_calls,
                q.early_eos,
                q.length_capped
            );
        }
        if let Some(rates) = &c.splice_hit_rate {
            out.push_str(
                "\nSplice hit rate (natural-stop pass, a lower bound under recorded-reply replay):",
            );
            if rates.is_empty() {
                out.push_str(" no request held an earlier reply\n");
            } else {
                out.push('\n');
                for r in rates {
                    let rate = r
                        .rate
                        .map_or("–".to_string(), |rate| format!("{:.0} %", rate * 100.0));
                    let _ = writeln!(
                        out,
                        "- {}: {rate} ({} of {})",
                        r.client, r.spliced, r.eligible
                    );
                }
            }
        }
        let first = c.turns.iter().flatten().filter_map(|t| t.rep).min();
        out.push_str("\n| Turn | Conversation | Kind | Class | TTFT | TFUT | Input | Cached | Output | Cause | Unreused |\n| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n");
        for t in c.turns.iter().flatten().filter(|t| t.rep == first) {
            let class = match (t.class, t.offset_ms) {
                (crate::bench::result::TurnClass::Auxiliary, Some(at)) => {
                    format!("auxiliary +{at:.0} ms")
                }
                (crate::bench::result::TurnClass::Auxiliary, None) => "auxiliary".into(),
                _ => "main".into(),
            };
            let cause = match (&t.cause, t.message_index, &t.message_role) {
                (Some(cause), Some(i), Some(role)) => format!("{cause} @{i} ({role})"),
                (Some(cause), Some(i), None) => format!("{cause} @{i}"),
                (Some(cause), None, _) => cause.clone(),
                _ => "–".into(),
            };
            let cause = match t.gap {
                Some(gap) => format!("{cause}, gap {gap}"),
                None => cause,
            };
            let _ = writeln!(
                out,
                "| {} | {} | {} | {class} | {:.1} ms | {:.1} ms | {} | {} | {} | {cause} | {} |",
                t.n,
                t.conversation,
                format!("{:?}", t.kind).to_lowercase(),
                t.ttft_ms,
                t.tfut_ms,
                t.input,
                t.cached,
                t.output,
                t.unreused.map_or("–".to_string(), |n| n.to_string())
            );
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::bench::result::*;

    fn cell(system: &str, name: &str, unit: &str, samples: &[f64], status: CellStatus) -> Cell {
        let mut c = Cell {
            system: system.into(),
            cell: name.into(),
            api: "chat".into(),
            status,
            samples: samples.to_vec(),
            unit: unit.into(),
            thermal_start: "nominal".into(),
            thermal_end: "nominal".into(),
            ..Cell::default()
        };
        c.summarize();
        c
    }

    pub fn sample() -> RunResult {
        let mut result = RunResult {
            schema_version: SCHEMA_VERSION,
            started_at: "2026-10-15T08:00:00.000Z".into(),
            finished_at: "2026-10-15T09:00:00.000Z".into(),
            mrun_version: "0.1.0".into(),
            machine: Machine {
                chip: "Apple M4 Pro".into(),
                memory_bytes: 48 << 30,
                macos: "15.1".into(),
                power_source: "AC".into(),
                ..Machine::default()
            },
            systems: vec![SystemUnderTest {
                name: "mrun".into(),
                release: "this build".into(),
                verified: true,
                model: ModelUnderTest {
                    name: "qwen3-coder".into(),
                    format: "mlx".into(),
                    bits_per_weight: Some(4.5),
                    kv_format: "bf16".into(),
                },
                ..SystemUnderTest::default()
            }],
            cells: vec![
                cell(
                    "mrun",
                    "synthetic/prefill-8k",
                    "ms",
                    &[120.0, 110.0, 130.0, 125.0, 118.0],
                    CellStatus::Ok,
                ),
                cell(
                    "llama.cpp",
                    "synthetic/prefill-8k",
                    "ms",
                    &[100.0, 101.0, 99.0, 102.0, 98.0],
                    CellStatus::Ok,
                ),
                cell(
                    "mrun",
                    "synthetic/decode-8k",
                    "tok/s",
                    &[80.0, 81.0, 79.0, 82.0, 80.0],
                    CellStatus::Ok,
                ),
                cell(
                    "llama.cpp",
                    "synthetic/decode-8k",
                    "tok/s",
                    &[70.0, 71.0, 69.0, 72.0, 70.0],
                    CellStatus::Ok,
                ),
                cell(
                    "ollama",
                    "agent/opencode-refactor",
                    "s",
                    &[],
                    CellStatus::NotRun,
                ),
            ],
            method: Method {
                replay: "in-order".into(),
                reps: Reps {
                    synthetic: 5,
                    cancel: 50,
                    agent: 3,
                },
                resamples: crate::bench::stats::RESAMPLES,
                seed: crate::bench::stats::SEED,
            },
            ..RunResult::default()
        };
        result.comparisons = crate::bench::comparisons(&result, "mrun");
        result
    }

    #[test]
    fn losing_cells_are_listed_and_never_hidden() {
        let report = render(&sample());
        let losing = report.split("## Losing cells\n\n").nth(1).unwrap();
        assert!(
            losing.starts_with("- synthetic/prefill-8k: llama.cpp is faster, mrun speed-up 0.83×"),
            "{losing}"
        );
        assert!(losing.contains("mrun-slower"));
        assert!(
            !losing.contains("decode-8k"),
            "mrun wins decode (higher is better)"
        );
        assert!(report.contains("| agent/opencode-refactor | chat | ollama | no | – | not run |"));
        let mut none = sample();
        none.cells.retain(|c| c.system == "mrun");
        none.comparisons = crate::bench::comparisons(&none, "mrun");
        assert!(render(&none).ends_with("## Losing cells\n\nnone\n"));
    }

    #[test]
    fn the_report_leaves_the_default_to_the_client_gate() {
        let report = render(&sample());
        assert!(report.contains("It does not choose a default model or system"));
        assert!(report.contains("decided by the client gate"));
        assert!(report.contains("no cell blocks a release"));
        let summary = report.split("## Summary\n\n").nth(1).unwrap();
        assert!(summary.starts_with("| Cell | Scope | System | n | mrun speed-up"));
        assert!(summary.contains("| synthetic/decode-8k | all turns | llama.cpp | 5 | 1.14× |"));
    }

    #[test]
    fn unverified_profiles_stay_out_of_the_summary() {
        let mut result = sample();
        result.systems.push(SystemUnderTest {
            name: "llama.cpp".into(),
            verified: false,
            ..SystemUnderTest::default()
        });
        result.comparisons = crate::bench::comparisons(&result, "mrun");
        assert!(result.comparisons.iter().all(|c| !c.headline));
        let report = render(&result);
        let summary = report.split("## Summary\n\n").nth(1).unwrap();
        assert!(summary.starts_with("no comparisons"));
        assert!(report.contains("## Unverified profiles"));
        assert!(report.contains("- synthetic/prefill-8k: llama.cpp [U] is faster"));
    }
}
