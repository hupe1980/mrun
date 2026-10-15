# Implementation Plan: Benchmark Harness

**Branch**: `005-bench` | **Date**: 2026-10-14 (revised 2026-10-15) | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `specs/005-bench/spec.md`

## Summary

`mrun bench` is an HTTP-only load driver built into the binary. It replays bundled agent traces
(recorded requests in order, unchanged) and synthetic cells against any system speaking OpenAI Chat
Completions, Anthropic Messages or OpenAI Responses (mrun, llama.cpp server, Ollama, oMLX, mlx-lm
server). Profiles start, health-check, reset and stop each system so only one is loaded at a time,
with equal context length. It measures client-side timings from the streamed bytes, samples
machine state and per-system memory, and writes a versioned JSON result plus a markdown report. A
regression mode compares against a baseline.

## Technical Context

**Language/Version**: Rust 1.94.0 (edition 2024)
**Primary Dependencies**: reqwest 0.13 (streaming HTTP, SSE parsing), tokio 1.53, serde/serde_json,
jsonschema validation of tool arguments, mrun's own API types (shared with 002) for request
building, objc2-foundation (`NSProcessInfo.thermalState`); no competitor SDKs. `cargo xtask` uses
`tokenizers` for trace building.
**Storage**: Bundled traces and profiles under `bench/`; results to
`./mrun-bench/<timestamp>/result.json` and `report.md`
**Testing**: cargo test with a local fake SSE server (axum) emitting scripted timings for all three
API styles (including pings, comments and role-only chunks); statistics unit tests on fixed
samples; result JSON validated against the hand-written schema; test that the driver contacts only
configured hosts
**Target Platform**: macOS (machine state via `sysctl`, `pmset`, `vm_stat`, `NSProcessInfo`);
driver tests also run on Linux CI against fake servers with a test-only machine-state stub
**Project Type**: CLI subcommand of the `mrun` binary, plus an `xtask` crate for trace building
**Performance Goals**: Client-side timing overhead < 1 ms per event; driver memory < 100 MB
**Constraints**: One system loaded at a time (FR-008); no other hosts contacted (FR-011)
**Scale/Scope**: ~10 agent traces (10–30 turns, up to 60K tokens), 4 synthetic sizes × prefill/decode

## Constitution Check

| Principle | Gate | Status |
| --- | --- | --- |
| I Agent latency | Per-turn TTFT on in-order trace replay with equal cache resets and context; every cell names model, format and bits per weight | Pass |
| II Exactness | Malformed-tool-call and early-EOS counts next to speed; greedy conformance stays in 001 `--check` | Pass |
| III Memory budget | RSS/footprint and swap sampled per system during agent runs | Pass |
| IV Local, private | Only open-source-client traces bundled, redacted + secret-scanned; no upload (tested) | Pass |
| V Own runtime | Talks HTTP only; no engine coupling | Pass |
| VI Simple, DX | One command; min/median/max for small n; no quality sandbox; hand-written schema | Pass |

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
├── systems.rs      # profiles: start/stop/health/reset, context length, pid; one system at a time
├── driver.rs       # request replay over HTTP, SSE timing capture (FR-013)
├── suites/
│   ├── agent.rs  synthetic.rs  cancel.rs
├── machine.rs      # macOS only: chip, memory, macOS, power, thermal, swap, RSS sampling
├── stats.rs        # min/median/max, p95 + IQR for n ≥ 20, throttling slope
└── report.rs       # JSON result + markdown rendering, regression compare
xtask/src/main.rs   # `cargo xtask bench-trace`: order, group, count tokens, redact, secret scan
bench/
├── traces/*.jsonl  # redacted OpenCode/Codex traces
├── result.schema.json  # hand-written, versioned
└── profiles/*.toml # mrun, llama.cpp, ollama, omlx, mlx-lm: commands and flags
crates/mrun/tests/bench_driver.rs   # fake SSE server tests
```

**Structure Decision**: Lives in the `mrun` crate; reuses 002's request types to build requests
in all three API styles. Trace building is a maintainer step in `xtask`, not in the binary.

## Complexity Tracking

None.
