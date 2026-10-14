# Implementation Plan: Benchmark Harness

**Branch**: `005-bench` | **Date**: 2026-10-14 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `specs/005-bench/spec.md`

## Summary

`mrun bench` is an HTTP-only load driver built into the binary. It replays bundled agent traces
(teacher-forced history) and synthetic cells against any system speaking OpenAI Chat Completions,
Anthropic Messages or OpenAI Responses (mrun, llama.cpp server, Ollama, oMLX, mlx-lm server),
measures client-side timings from the streamed bytes, samples machine state, and writes a
versioned JSON result plus a markdown report. A sandboxed quality suite runs objective coding
tasks; a regression mode compares against a baseline. It absorbs 001's engine benchmark cells.

## Technical Context

**Language/Version**: Rust 1.94.0 (edition 2024)
**Primary Dependencies**: reqwest 0.13 (streaming HTTP, SSE parsing), tokio 1.53, serde/serde_json,
mrun's own API types (shared with 002) for request building; no competitor SDKs; macOS
`sandbox-exec` (no-network profile) for the quality suite
**Storage**: Bundled suites under `bench/` in the repo (traces as JSONL, tasks as directories);
results to `./mrun-bench/<timestamp>/result.json` and `report.md`
**Testing**: cargo test with a local fake SSE server (axum) emitting scripted timings for all three
API styles; statistics unit tests on fixed samples; result JSON validated against the hand-written
schema; test that the driver contacts only configured hosts
**Target Platform**: macOS (machine-state probes via `sysctl`, `pmset`, `vm_stat`); Linux runs the
driver against fake servers in CI
**Project Type**: CLI subcommand of the `mrun` binary
**Performance Goals**: Client-side timing overhead < 1 ms per event; driver memory < 100 MB
**Constraints**: One system running at a time (FR-008); no other hosts contacted (FR-011)
**Scale/Scope**: ~10 agent traces (10–30 turns, up to 60K tokens), ~20 synthetic cells,
~20 quality tasks

## Constitution Check

| Principle | Gate | Status |
| --- | --- | --- |
| I Agent latency | Per-turn TTFT on teacher-forced agent traces; every cell names model, format and quantization | Pass |
| II Exactness | Quality suite next to speed; greedy conformance stays in 001 `--check` | Pass |
| III Memory budget | `memory` suite samples process RSS/footprint and system swap | Pass |
| IV Local, private | Only open-source-client traces bundled, redacted + secret-scanned; no upload (tested) | Pass |
| V Own runtime | Talks HTTP only; no engine coupling | Pass |
| VI Simple, DX | One command; median/p95/IQR only; hand-written schema | Pass |

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
├── stats.rs        # median, nearest-rank p95, IQR
└── report.rs       # JSON result + markdown rendering, regression compare
bench/
├── traces/*.jsonl  # redacted OpenCode/Codex sessions (request bodies, recorded replies, gaps)
├── result.schema.json  # hand-written, versioned
├── tasks/<id>/     # quality tasks: repo snapshot, prompt, test command
└── profiles/*.toml # competitor launch notes and recommended flags
crates/mrun/tests/bench_driver.rs   # fake SSE server tests
```

**Structure Decision**: Lives in the `mrun` crate; reuses 002's request types to build requests
in all three API styles.

## Complexity Tracking

None.
