---
description: "Tasks for 005 Benchmark Harness"
---

# Tasks: Benchmark Harness

**Input**: [plan.md](plan.md), [spec.md](spec.md), [research.md](research.md), [data-model.md](data-model.md), [contracts/result-schema.md](contracts/result-schema.md)

## Phase 1: Setup

- [ ] T001 Add reqwest 0.13 (stream) to `crates/mrun/Cargo.toml`; `crates/mrun/src/bench/mod.rs` CLI args per contract; `bench/result.schema.json` (hand-written v1)
- [ ] T002 [P] Fake SSE server for tests in `crates/mrun/tests/bench_driver.rs` (scripted delays; Chat, Messages and Responses styles; pings, SSE comments, keepalives, role-only chunks)

## Phase 2: Foundational

- [ ] T003 `bench/stats.rs`: min/median/max; nearest-rank p95 and IQR only for n ≥ 20; throttling slope (> 5%, FR-012); tests on fixed samples
- [ ] T004 `bench/driver.rs`: replay in all three API styles, per-system model name, client-side TTFT per FR-013 (ignoring pings, comments, keepalives, role-only chunks), first-useful-token and per-token timings, cached and prompt token extraction (usage fields, `: mrun` SSE comments when streaming, `x-mrun-*` headers otherwise); "truncated input" check (> 2% from expected, FR-015); tests against the fake server; test that only configured hosts are contacted (FR-011)
- [ ] T005 [P] `bench/machine.rs` (macOS only): chip, memory, macOS, load average before each system, power source, low power, thermal via `NSProcessInfo.thermalState` (powermetrics only when root), free memory and swap per system, RSS/footprint sampling by PID; Linux tests use a stub
- [ ] T006 `bench/report.rs`: RunResult v1 (validated against the schema) + markdown report with Method and "Losing cells" sections

## Phase 3: US1 — Measure my workload (P1) 🎯 MVP

- [ ] T007 [US1] `bench/suites/synthetic.rs`: prefill/decode cells at 1K/8K/32K and the common maximum (smallest context across compared systems), per-repetition random nonce on prefill prompts, warm-up, ≥ 5 reps, temperature 0, fixed output, ignore-EOS where supported, early-EOS labels
- [ ] T008 [US1] Trace format + `bench/suites/agent.rs`: in-order replay of recorded requests unchanged, `max_tokens` = `reply_tokens` with ignore-EOS where supported, session time = Σ(TTFT + decode), per-conversation per-turn records, malformed-tool-call (invalid JSON, undeclared tool, schema failure) and early-EOS counts per system, memory sampling during the run, 3 reps
- [ ] T009 [US1] `cargo xtask bench-trace` (new `xtask` crate): order `mrun serve --record` output by arrival time, group into conversations (`prompt_cache_key`, session headers when recorded, else longest shared prefix), count `prompt_tokens`/`reply_tokens` with the model tokenizer, redact (paths, usernames, emails, hosts) with one deterministic mapping per trace, secret scan; same check as a pre-commit hook on `bench/traces/`. Record OpenCode and Codex traces on public repos; commit with NOTICE for Codex; at least `agent/opencode-refactor` (20 turns, 30K) used by 003 SC-001
- [ ] T010 [P] [US1] `bench/suites/cancel.rs` (mrun only: disconnect → next first byte, 50 random points)

## Phase 4: US2 — Fair comparison (P1)

- [ ] T011 [US2] `bench/systems.rs` + `bench/profiles/{mrun,llama.cpp,ollama,omlx,mlx-lm,sglang-mlx}.toml`: start/stop/health/reset/pid commands (FR-008, FR-014), equal context N = mrun capacity (FR-015), recommended cache flags, model names, formats, bits per weight; one system at a time with free-memory check or "Stop X, start Y, press Enter" pause; reset before each repetition, recorded; unreachable competitor → "not run", exit 5 only when all are unreachable (FR-016); "different quantization" labels
- [ ] T012 [US2] First comparison on the M4 Pro: two back-to-back synthetic runs (SC-001), full agent comparison with start/stop profiles and wall-time capture (SC-002), SC-005 cells; publish dated report in `bench/results/` (manual step)

## Phase 5: US3 — Speed never hides wrong output (P2)

- [ ] T013 *Removed (quality suite moved to Later).*
- [ ] T012a [US2] Default-model decision: agent-suite cells for every 0.1 alias (Qwen3.6, Qwen3-Coder, gpt-oss-20b, Devstral Small 2) on the M4 Pro (all mrun; a `tier: 2` alias cannot be the default); record the outcome in concepts D015 and the 004 alias default before release
- [ ] T014 [US3] Regression mode `--baseline` (exit 7) on median speed and malformed-tool-call/early-EOS counts; tests: injected 15% slowdown detected against a 10% threshold, 0 false alarms on 5 unchanged runs (SC-004); fake-server test that a malformed tool call and early EOS are counted

## Phase 6: Polish

- [ ] T015 Site page `benchmarks.md` linking the latest dated report and the method

## Dependencies

T001 → T003/T004 → T006 → suites. Trace recording needs 002 (`serve --record`, FR-021; timestamps
used only for ordering); cache resets for mrun need 003 (`mrun cache clear`). Comparisons need a
working mrun server (001 + 002); SC-005 needs 003.
