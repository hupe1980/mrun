---
description: "Tasks for 005 Benchmark Harness"
---

# Tasks: Benchmark Harness

**Input**: [plan.md](plan.md), [spec.md](spec.md), [research.md](research.md), [data-model.md](data-model.md), [contracts/result-schema.md](contracts/result-schema.md)

**Scope**: Lean 0.1 (D025, 2026-10-15). Cut tasks keep their IDs and say so.

## Phase 1: Setup

- [ ] T001 Add reqwest 0.13 (stream) to `crates/mrun/Cargo.toml` (already a dev-dependency via 002 T001); `crates/mrun/src/bench/mod.rs` CLI args per contract; `bench/result.schema.json` (hand-written v1)
- [ ] T002 [P] Fake SSE server for tests in `crates/mrun/tests/bench_driver.rs` (scripted delays; Chat, Messages and Responses styles; pings, SSE comments, keepalives, role-only chunks; scripted `x-mrun-cache-*` metadata)

## Phase 2: Foundational

- [ ] T003 `bench/stats.rs`: min/median/max; nearest-rank p95 and IQR only for n ≥ 20; tests on fixed samples. (Throttling slope: removed, 0.1 scope cut, 2026-10-15)
- [ ] T004 `bench/driver.rs`: replay in all three API styles, per-system model name, client-side TTFT per FR-013 (ignoring pings, comments, keepalives, role-only chunks), first-useful-token and per-token timings, cached and prompt token extraction (usage fields, `: mrun` SSE comments when streaming, `x-mrun-*` headers otherwise, incl. cause and message index); "truncated input" check (> 2% from expected, FR-015); tests against the fake server; test that only configured hosts are contacted (FR-011)
- [ ] T005 [P] `bench/machine.rs` (macOS only): chip, memory, macOS, load average before each system, power source, low power, thermal state via `NSProcessInfo.thermalState` at each cell's start and end, free memory and swap per system; Linux tests use a stub. (RSS/footprint sampling: removed, 0.1 scope cut, 2026-10-15)
- [ ] T006 `bench/report.rs`: RunResult v1 (validated against the schema) + markdown report with Method and "Losing cells" sections; insta snapshot tests for every cell status and an empty losing-cells section (SC-003)

## Phase 3: US1 — Measure my workload (P1) 🎯 MVP

- [ ] T007 [US1] `bench/suites/synthetic.rs`: prefill/decode cells at 1K/8K/32K and the common maximum (smallest context across compared systems), per-repetition random nonce on prefill prompts, warm-up, ≥ 5 reps, temperature 0, fixed output, ignore-EOS where supported, early-EOS labels
- [ ] T008 [US1] Trace format + `bench/suites/agent.rs`: in-order replay of recorded requests unchanged, `max_tokens` = `reply_tokens` with ignore-EOS where supported, session time = Σ(TTFT + decode), per-conversation per-turn records with `kind` (`repeated`: appends to the previous request of the same conversation; `branching`: first request after one from a different conversation; data model), malformed-tool-call (invalid JSON, undeclared tool, schema failure) and early-EOS counts per system, `template` re-render counts for mrun (FR-017), 3 reps
- [ ] T009 [US1] `cargo xtask bench-trace` (added to the `xtask` crate created by 004 T005): order `mrun serve --record` output by arrival time, group into conversations (`prompt_cache_key`, session headers when recorded, else longest shared prefix), count `prompt_tokens` and derive `reply_tokens` from the next request's history with the model tokenizer (conversation median for the last request), redact (paths, usernames, emails, hosts) with one deterministic mapping per trace, secret scan; same check as a pre-commit hook on `bench/traces/`. Record OpenCode and Codex traces on public repos; commit with NOTICE for Codex; at least `agent/opencode-refactor` (20 turns, 30K) used by 003 SC-001
- [ ] T010 [P] [US1] `bench/suites/cancel.rs` (mrun only: disconnect → next first byte, 50 random points)

## Phase 4: US2 — Fair comparison (P1)

- [ ] T011 [US2] `bench/systems.rs` + `bench/profiles/{mrun,llama.cpp,ollama,omlx,mlx-lm,splash}.toml`: health and reset (HTTP or manual restart pause, FR-014), equal context N = mrun capacity (FR-015), recommended cache flags, model names, formats, bits per weight, KV format; one system at a time via the "Stop X, start Y, press Enter" pause; free memory before each system with the 2 GiB warning; reset before each repetition, recorded; unreachable competitor → "not run", exit 5 only when all are unreachable (FR-016); "different quantization" and "lossy KV" labels. Splash profile: `splash serve --model mlx-community/Qwen3.6-35B-A3B-4bit`, port 8000, cells `splash-bf16` (`--kv-format bf16`, idle release off) and `splash` (default INT8 KV); verify flag names on the pinned Splash release. (SGLang-MLX profile and start/stop commands: removed, 0.1 scope cut, 2026-10-15)
- [ ] T012 [US2] First comparison on the M4 Pro: two back-to-back synthetic runs (SC-001), agent comparison against all five competitors' profiles (SC-005); publish a dated report in `bench/results/` (manual step)
- [ ] T012a [US2] Default-model decision: agent-suite cells for every 0.1 alias (`qwen3.6`, `qwen3-coder`, `gpt-oss`) on the M4 Pro (all mrun); record the outcome in concepts D015 and the 004 alias default before release

## Phase 5: US3 — Speed never hides wrong output (P2)

- [ ] T013 *Removed (quality suite moved to Later).*
- [ ] T014 *Removed (0.1 scope cut, 2026-10-15)*: regression mode `--baseline` (exit 7). Remaining scope: fake-server test that a malformed tool call and early EOS are counted next to speed

## Phase 6: Polish

- [ ] T015 Site page `benchmarks.md` linking the latest dated report and the method

## Dependencies

T001 → T003/T004 → T006 → suites. Trace recording needs 002 (`serve --record`, FR-021; timestamps
used only for ordering); cache resets for mrun need 003 (`mrun cache clear`); T009 needs 004 T005
(the `xtask` crate). Comparisons need a working mrun server (001 + 002); SC-005 needs 003.
