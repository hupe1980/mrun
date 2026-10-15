---
description: "Tasks for 006 Speculative Decoding"
---

# Tasks: Speculative Decoding

**Input**: [plan.md](plan.md), [spec.md](spec.md), [research.md](research.md), [data-model.md](data-model.md), [contracts/speculation.md](contracts/speculation.md)

**Prerequisites**: 001 (engine, sampler, scratch probe, hybrid state), 002 (metadata), 003
(committed-token recording), 005 (traces, suites, regression threshold) implemented.

## Phase 1: Setup and measure first

- [ ] T001 `crates/mrun/src/spec/index.rs`: suffix automaton over token ids with sentinel-separated sources, incremental extension, longest-suffix lookup with most-recent tie-break; unit tests in `crates/mrun/tests/spec_index.rs`, plus a perf check (index build ≤ 10 ms per 100K tokens, plan Performance Goals)
- [ ] T002 [P] `crates/mrun/src/spec/render.rs`: copy-aware rendering (line-number prefixes `␣*N→`/`␣*N<TAB>` stripped); tests with Claude Code `Read` output
- [ ] T003 `crates/mrun/src/bench/suites/spec.rs` offline cell: reply tokens from the next request's history, simulated drafting, acceptance histogram, `copy_share_16`, `edit-heavy` label; result fields per contract (depends on T001, T002)
- [ ] T004 Run `mrun bench --suite spec --offline` on the bundled traces; record the go/no-go note in `research.md` R8 (FR-001 offline part)

**Checkpoint**: offline acceptance known; continue only if edit-heavy traces exist in the bundle.

## Phase 2: Foundational (engine)

- [ ] T005 `crates/mrun-engine/src/engine.rs`: `SpecCaps`, `DraftStep {id, before, width}`, `VerifyOutcome`, `ProposalId`, `forward_draft`, `accept`, `commit`, `spec_caps`, additive `EngineError::{Cancelled, Unsupported}` per contract; `ScriptedEngine` implementation with per-position token provenance and scripted acceptance (`crates/mrun/src/runtime/scripted.rs`)
- [ ] T006 `crates/mrun-engine/src/sample.rs`: on-device greedy acceptance, point-proposal rejection sampling, residual sampling with the request RNG stream, acceptance ends at the first accepted EOS; `crates/mrun-engine/tests/spec_sampling.rs` chi-square test at a fixed draft-covered position on a tiny CPU model (SC-005)
- [ ] T007 `crates/mrun-engine/src/mlx_engine.rs`: `forward_draft` (N+1 positions), `accept` and `commit(k)` for `qwen3`, `qwen3_moe`, `qwen3_5_moe`; attention rollback by trim; cancel per layer
- [ ] T008 `crates/mrun-engine/src/state.rs`: keep the pre-round recurrent state and the round's per-position GDN inputs (conv input, q/k/v, g, β); on partial acceptance or `commit(k)`, recompute the recurrence over accepted + 1 positions from the pre-round state (R5); tests against a non-speculative run on a tiny random hybrid model (CPU), including EOS and stop-string cuts
- [ ] T009 `crates/mrun-engine/src/budget.rs`: rollback input cache bytes (≤ 32 MiB at N = 64) in the scratch envelope; `max_draft` lowered until it fits (FR-013). (Verify-width probe calibration: removed, see FR-008.)

**Checkpoint**: `verify` is exact on the scripted and tiny models.

## Phase 3: US1 + US2 — copy-aware drafting with a controller (P1) 🎯 MVP

- [ ] T010 [US2] `crates/mrun/src/spec/controller.rs`: Beta acceptance per match bucket, cost curves per context bucket (analytic expert-union prior + EWMA, FR-008), argmax committed/cost over {0} ∪ [N_min, N_max], next depth chosen before `sample`, probing with back-off, server-lifetime state; `crates/mrun/tests/spec_controller.rs` (settles at depth 0 within 64 rounds on non-repetitive text, collapse → 0, deep-context bucket used; perf check: decision < 20 µs)
- [ ] T011 [US1] `crates/mrun/src/spec/mod.rs`: SpecPolicy (mode sources; `grammar`, `seed`, `logprobs`, `limit` → off); index built on a helper thread during prefill
- [ ] T012 [US1] `crates/mrun/src/runtime/generate.rs`: round loop per contract (cap by remaining/reservation, `NeedsMemory` → evict and retry, `pipeline = next_depth == 0`, one-at-a-time emission through stop, UTF-8 hold-back and tool parsers, `commit(k)` on a stop inside the round, cancel per round) (depends on T005, T010, T011)
- [ ] T013 [US1] `crates/mrun/tests/spec_mutations.rs`: ≥ 100 cases (partial/full/zero acceptance, EOS and stop strings inside drafts, hybrid EOS and hybrid stop-string `commit(k)`, limit, cancel mid-round, `NeedsMemory` retry, hybrid rollback) asserting zero rejected tokens in state, store and stream (SC-004)
- [ ] T014 [US1] `crates/mrun-engine/tests/spec_conformance.rs` (macOS): greedy `auto` vs `off` on 001 conformance prompts and teacher-forced traces; report divergences with margins (SC-003)

**Checkpoint**: n-gram speculation works end to end on both models.

## Phase 4: US4 — visibility and control (P2)

- [ ] T015 [P] [US4] `crates/mrun/src/spec/stats.rs` + `crates/mrun/src/api/*`: `spec` metadata (headers, final `: mrun` comment), `x-mrun-speculation: off`, other values ignored (as `x-mrun-cache`); `/mrun/status` `spec` key; conformance tests on all three APIs
- [ ] T016 [P] [US4] `crates/mrun/src/config.rs` + `cli/serve.rs` + `cli/run.rs`: `--speculation auto|off|ngram`, `[speculation] mode`, `MRUN_SPECULATION_MODE`; summary line `accepted/drafted`; `run --check` forces `off` (FR-017); snapshot tests

## Phase 5: US3 — MTP drafting (P2) — Removed

T017–T020 moved to "Follow-up (gated)".

## Phase 6: Gates and polish

- [ ] T021 Online `spec` cells (`off|ngram|auto`) in `bench/suites/spec.rs`; run on the target Mac for both models (FR-001 online part)
- [ ] T022 SC-001: full bench `off` vs `auto` with `--baseline` at the 005 regression threshold (5 repetitions, medians); SC-002: edit-heavy decode ratio ≥ 1.3×; record results and flip the default to `auto` only if both pass
- [ ] T023 [P] Site page section "Speculative decoding" (how it works, how to turn it off, measured numbers)
- [ ] T024 Fold cross-spec changes into the 001–005 contracts and specs/README index (plan "Cross-spec changes")
- [ ] T025 [P] `crates/mrun/src/metrics.rs`: `mrun_spec_drafted_tokens_total`, `mrun_spec_accepted_tokens_total`, `mrun_spec_rounds_total` (label `drafter` = `ngram`) per 008 metrics contract; test with the scripted engine
- [ ] T026 [P] FR-017: `mrun run --check`, 003 exactness tests and `bench --suite agent` comparisons pass `--speculation off`; test that `--check` reports `spec.reason = off`

## Dependencies

T001/T002 → T003 → T004 (gate). T005 → T006/T007 → T008 → T009. T010/T011 → T012 → T013/T014.
US4 (T015/T016) after T012. T021 after T012 → T022. T024 any time after T005; T025/T026 after T015/T016.

## Follow-up (gated): MTP drafting

Not scheduled. Starts only after FG01 meets the gate (spec "Follow-up (gated)").

- [ ] FG01 Measure MTP per-position acceptance offline on the bundled traces with the reference implementation (Ollama MLX runner `mlxrunner/model/qwen3_5`; vLLM Qwen3.5 MTP as cross-check); model the speed-up with FR-008 costs; proceed only if α₁ ≥ 0.8 and ≥ 1.15× on Qwen3.6
- [ ] T017 [US3] `crates/mrun/src/hub/aliases.rs` + `scripts/refresh_aliases.py`: MTP source for `qwen3.6` (repo, revision, per-tensor SHA-256 from a full download)
- [ ] T018 [US3] `crates/mrun/src/hub/mtp.rs`: safetensors index + header range reads, `mtp.*` range fetch, per-tensor verify, shard fallback, 4-bit quantization (gate 8-bit), atomic sidecar write, manifest entry (not part of the cache key), `--no-mtp`; wiremock tests
- [ ] T019 [US3] `crates/mrun-engine/src/models/mtp.rs` + `load.rs`: head forward, 4,096-position window ring, budget accounting; test against the reference implementation's forward on saved tensors
- [ ] T020 [US3] Controller: MTP acceptance per position, drafter choice per round, N_max 4, `draft_mtp` and `ProposalId`, ratio rejection sampling; `--speculation mtp`
