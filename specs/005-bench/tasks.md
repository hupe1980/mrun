---
description: "Tasks for 005 Benchmark Harness"
---

# Tasks: Benchmark Harness

**Input**: [plan.md](plan.md), [spec.md](spec.md), [research.md](research.md), [data-model.md](data-model.md), [contracts/result-schema.md](contracts/result-schema.md)

## Phase 1: Setup

- [ ] T001 Add reqwest 0.13 (stream), schemars to `crates/mrun/Cargo.toml`; `crates/mrun/src/bench/mod.rs` CLI args per contract
- [ ] T002 [P] Fake SSE server for tests in `crates/mrun/tests/bench_driver.rs` (scripted delays, both API styles)

## Phase 2: Foundational

- [ ] T003 `bench/stats.rs`: median, nearest-rank p95, IQR, bootstrap CI; tests on fixed samples
- [ ] T004 `bench/driver.rs`: OpenAI and Anthropic request replay over HTTP, client-side TTFT / first-useful-token / per-token timings from SSE bytes, cached-token extraction; tests against the fake server
- [ ] T005 [P] `bench/machine.rs`: chip, memory, macOS, power source, low power, thermal, swap sampling; Linux fallbacks
- [ ] T006 `bench/report.rs`: RunResult (schema v1, schemars → `bench/result.schema.json` snapshot) + markdown report with "Losing cells" section

## Phase 3: US1 — Measure my workload (P1) 🎯 MVP

- [ ] T007 [US1] `bench/suites/synthetic.rs`: prefill and decode cells at 1K/8K/16K/30K with warm-up and ≥ 5 reps
- [ ] T008 [US1] `serve --record <dir>` debug-only request recorder (002 server) + trace format; `bench/suites/agent.rs` replay with optional think-time gaps
- [ ] T009 [US1] Record and review first traces (Claude Code, Codex, OpenCode on public repos) into `bench/traces/`
- [ ] T010 [P] [US1] `bench/suites/cancel.rs` (disconnect → next first byte) and `bench/suites/memory.rs` (RSS/footprint/swap sampling by PID)

## Phase 4: US2 — Fair comparison (P1)

- [ ] T011 [US2] `bench/systems.rs` + `bench/profiles/{llama.cpp,ollama,omlx,mlx-lm}.toml` with recommended cache flags and version capture; `--compare` sequential runs, randomized cell order, comparability labels
- [ ] T012 [US2] First comparison run on the M4 Pro; publish dated report in `bench/results/` (manual step)

## Phase 5: US3 — Quality (P2)

- [ ] T013 [US3] `bench/suites/quality.rs`: harness-executed tool loop (read/write/run tests) over `bench/tasks/*`, 3 seeds, pass rate per system
- [ ] T014 [US3] Regression mode `--baseline` with thresholds; test catching an injected 10% slowdown and a greedy-output change (SC-004)

## Phase 6: Polish

- [ ] T015 README "Benchmarks" section linking the latest dated report

## Dependencies

T001 → T003/T004 → T006 → suites. Agent suite needs 002 (`serve --record`). Comparison needs a
working mrun server (001 + 002).
