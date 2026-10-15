//! The markdown report of a run (005 contract): Method, Machine, Systems, cells per suite,
//! Warnings, and Losing cells, always present (possibly "none"): every cell where another system
//! beat mrun on the median.

use crate::bench::result::{Cell, CellStatus, RunResult};
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

/// Cells where some system's median beats mrun's on the same cell and API.
pub fn losing_cells(result: &RunResult) -> Vec<(String, String, f64, f64)> {
    let mut losing = Vec::new();
    for ours in result.cells.iter().filter(|c| c.system == "mrun") {
        let (Some(mine), CellStatus::Ok) = (ours.median, ours.status) else {
            continue;
        };
        for theirs in result.cells.iter().filter(|c| {
            c.system != "mrun"
                && c.cell == ours.cell
                && c.api == ours.api
                && c.status == CellStatus::Ok
        }) {
            let Some(other) = theirs.median else { continue };
            let worse = if ours.lower_is_better() {
                mine > other
            } else {
                mine < other
            };
            if worse {
                losing.push((ours.cell.clone(), theirs.system.clone(), mine, other));
            }
        }
    }
    losing
}

pub fn render(result: &RunResult) -> String {
    let mut out = String::new();
    let _ = writeln!(out, "# mrun bench — {}\n", result.started_at);
    out.push_str("## Method\n\n");
    let _ = writeln!(
        out,
        "Agent traces are replayed in recorded order with each system's cache on and reset before every repetition; reuse is measured up to the previous reply. Every system gets the same context length. Repetitions: synthetic {}, cancel {}, agent {}.\n",
        result.method.reps.synthetic, result.method.reps.cancel, result.method.reps.agent
    );
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
    out.push_str("## Systems\n\n| System | Version | Model | Format | Bits/weight | KV | Context | Flags |\n| --- | --- | --- | --- | --- | --- | --- | --- |\n");
    for s in &result.systems {
        let _ = writeln!(
            out,
            "| {} | {} | {} | {} | {} | {} | {} | {} |",
            s.name,
            s.version,
            s.model.name,
            s.model.format,
            s.model
                .bits_per_weight
                .map_or("–".into(), |b| format!("{b:.2}")),
            s.model.kv_format,
            s.context_length,
            s.flags.join(" ")
        );
    }
    out.push_str("\n## Cells\n\n| Cell | API | System | Median | Min | Max | p95 | Early EOS | Malformed calls |\n| --- | --- | --- | --- | --- | --- | --- | --- | --- |\n");
    for c in &result.cells {
        let _ = writeln!(
            out,
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} |",
            c.cell,
            c.api,
            c.system,
            status_text(c),
            fmt_value(c.min, &c.unit),
            fmt_value(c.max, &c.unit),
            fmt_value(c.p95, &c.unit),
            c.early_eos,
            c.malformed_tool_calls
        );
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
        for (cell, system, mine, theirs) in losing {
            let _ = writeln!(out, "- {cell}: {system} {theirs:.1} vs mrun {mine:.1}");
        }
    }
    out
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
        RunResult {
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
            },
            ..RunResult::default()
        }
    }

    #[test]
    fn losing_cells_are_listed_and_never_hidden() {
        let report = render(&sample());
        assert!(
            report.contains(
                "## Losing cells\n\n- synthetic/prefill-8k: llama.cpp 100.0 vs mrun 120.0"
            )
        );
        assert!(
            !report.contains("decode-8k: llama.cpp"),
            "mrun wins decode (higher is better)"
        );
        assert!(report.contains("| agent/opencode-refactor | chat | ollama | not run |"));
        let mut none = sample();
        none.cells.retain(|c| c.system == "mrun");
        assert!(render(&none).ends_with("## Losing cells\n\nnone\n"));
    }
}
