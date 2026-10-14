# Implementation Plan: Benchmark Harness

**Branch**: `005-bench` | **Date**: 2026-10-14 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `specs/005-bench/spec.md`

## Summary

`mrun bench` is an HTTP-only load driver built into the binary. It replays bundled agent traces
and synthetic cells against any system that speaks OpenAI Chat Completions or Anthropic Messages
(mrun, llama.cpp server, Ollama, oMLX, mlx-lm server), measures client-side timings from the
streamed bytes, samples machine state, and writes a versioned JSON result plus a markdown report.
A quality suite runs objective coding tasks; a regression mode compares against a baseline.

## Technical Context

**Language/Version**: Rust 1.94.0 (edition 2024)
**Primary Dependencies**: reqwest 0.13 (streaming HTTP, SSE parsing), tokio 1.53, serde/serde_json,
mrun's own API types (shared with 002) for request building; no competitor SDKs
**Storage**: Bundled suites under `bench/` in the repo (traces as JSONL, tasks as directories);
results to `./mrun-bench/<timestamp>/result.json` and `report.md`
**Testing**: cargo test with a local fake SSE server (axum) emitting scripted timings; statistics
unit tests on fixed samples; schema snapshot test for result JSON
**Target Platform**: macOS (machine-state probes via `sysctl`, `pmset`, `vm_stat`); Linux runs the
driver against fake servers in CI
**Project Type**: CLI subcommand of the `mrun` binary
**Performance Goals**: Client-side timing overhead < 1 ms per event; driver memory < 100 MB
**Constraints**: Systems measured one at a time (FR-007); no uploads (FR-010)
**Scale/Scope**: ~10 agent traces (10–30 turns, up to 60K tokens), ~20 synthetic cells,
~20 quality tasks

## Constitution Check

| Principle | Gate | Status |
| --- | --- | --- |
| I Agent latency | Primary metric is per-turn TTFT and session time on agent traces | Pass |
| II Exactness | Regression mode fails on greedy-output change (conformance set) | Pass |
| III Memory budget | `memory` suite samples process RSS/footprint and system swap | Pass |
| IV Local, private | Traces are reviewed public recordings; no upload | Pass |
| V Own runtime | Talks HTTP only; no engine coupling | Pass |
| VI Simple, DX | One command, sensible default suite, readable report | Pass |

Post-design re-check: Pass.

## Project Structure

### Documentation (this feature)

```text
specs/005-bench/
├── plan.md  research.md  data-model.md  quickstart.md
├── contracts/result-schema.md
└── tasks.md
```

### Source Code (repository root)

```text
crates/mrun/src/bench/
├── mod.rs          # `mrun bench` args, suite selection, orchestration
├── systems.rs      # system-under-test profiles (mrun, llama.cpp, ollama, omlx, mlx-lm)
├── driver.rs       # request replay over HTTP, SSE timing capture
├── suites/
│   ├── agent.rs  synthetic.rs  cancel.rs  memory.rs  quality.rs
├── machine.rs      # chip, memory, macOS, power/thermal, load sampling
├── stats.rs        # median, p95, spread, bootstrap CI
└── report.rs       # JSON result + markdown rendering, regression compare
bench/
├── traces/*.jsonl  # recorded agent sessions (request bodies + tool results + gaps)
├── tasks/<id>/     # quality tasks: repo snapshot, prompt, test command
└── profiles/*.toml # competitor launch notes and recommended flags
crates/mrun/tests/bench_driver.rs   # fake SSE server tests
```

**Structure Decision**: Lives in the `mrun` crate; reuses 002's request types to build requests
for both API styles.

## Complexity Tracking

None.
