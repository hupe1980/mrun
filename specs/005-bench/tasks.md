---
description: "Tasks for 005 Benchmark Harness"
---

# Tasks: Benchmark Harness

**Input**: [plan.md](plan.md), [spec.md](spec.md), [research.md](research.md), [data-model.md](data-model.md), [contracts/result-schema.md](contracts/result-schema.md)

## Phase 1: Setup

- [ ] T001 Add reqwest 0.13 (stream) to `crates/mrun/Cargo.toml`; `crates/mrun/src/bench/mod.rs` CLI args per contract; `bench/result.schema.json` (hand-written v1)
- [ ] T002 [P] Fake SSE server for tests in `crates/mrun/tests/bench_driver.rs` (scripted delays; Chat, Messages and Responses styles)

## Phase 2: Foundational

- [ ] T003 `bench/stats.rs`: median, nearest-rank p95, IQR, throttling check (> 5% drift); tests on fixed samples
- [ ] T004 `bench/driver.rs`: replay in all three API styles, per-system model name, client-side TTFT / first-useful-token / per-token timings from SSE bytes, cached-token extraction (usage fields, `: mrun` SSE comments when streaming, `x-mrun-*` headers otherwise); tests against the fake server; test that only configured hosts are contacted (FR-011)
- [ ] T005 [P] `bench/machine.rs`: chip, memory, macOS, power source, low power, thermal, free memory and swap per system; Linux fallbacks
- [ ] T006 `bench/report.rs`: RunResult v1 (validated against the schema) + markdown report with Method and "Losing cells" sections

## Phase 3: US1 — Measure my workload (P1) 🎯 MVP

- [ ] T007 [US1] `bench/suites/synthetic.rs`: prefill/decode cells at 1K/8K/16K/30K/64K/max with warm-up, ≥ 5 reps, temperature 0, fixed output, ignore-EOS where supported, early-EOS labels
- [ ] T008 [US1] Trace format + `bench/suites/agent.rs` teacher-forced replay (untimed previous turn with recorded reply, then timed turn), recorded gaps capped at 60 s (`--gaps`, FR-003a), malformed-tool-call and early-EOS checks per system, 3 reps, per-turn records
- [ ] T009 [US1] Record OpenCode and Codex traces on public repos with `mrun serve --record` (002); `scripts/redact_traces.py` (paths, usernames, emails, hosts) + secret scan as a pre-commit check; commit to `bench/traces/` with NOTICE for Codex; at least `agent/opencode-refactor` (20 turns, 30K) used by 003 SC-001
- [ ] T010 [P] [US1] `bench/suites/cancel.rs` (disconnect → next first byte, 50 random points) and `bench/suites/memory.rs` (RSS/footprint/swap by PID)

## Phase 4: US2 — Fair comparison (P1)

- [ ] T011 [US2] `bench/systems.rs` + `bench/profiles/{llama.cpp,ollama,omlx,mlx-lm}.toml` (recommended cache flags, model names, formats, quantization); verify others are stopped via free memory; `--compare` sequential runs; comparability labels
- [ ] T012 [US2] First comparison on the M4 Pro: two back-to-back synthetic runs (SC-001), full agent comparison with wall-time capture (SC-002), SC-005 cells; publish dated report in `bench/results/` (manual step)

## Phase 5: US3 — Quality (P2)

- [ ] T013 [US3] `bench/suites/quality.rs`: harness-executed tool loop over `bench/tasks/*` in a temp copy under `sandbox-exec` (no network), 60 s per command, 10 min per task, 3 seeds; pass rate per system
- [ ] T014 [US3] Regression mode `--baseline` (exit 7); test catching an injected 10% slowdown (SC-004)

## Phase 6: Polish

- [ ] T015 Site page `benchmarks.md` linking the latest dated report and the method

## Dependencies

T001 → T003/T004 → T006 → suites. Agent suite recording needs 002 (`serve --record`). Comparisons
need a working mrun server (001 + 002); SC-005 needs 003.
