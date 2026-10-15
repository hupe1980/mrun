---
description: "Tasks for 003 Prefix Store"
---

# Tasks: Prefix Store

**Input**: [plan.md](plan.md), [spec.md](spec.md), [research.md](research.md),
[data-model.md](data-model.md), [contracts/cache.md](contracts/cache.md)

## Format: `[ID] [P?] [Story] Description`

## Phase 1: Setup

- [ ] T001 proptest dev-dependency; skeletons `crates/mrun/src/cache/{mod,tree,switch,chunks,evict,miss}.rs`

## Phase 2: Foundational (blocks all stories)

- [ ] T002 Verify 001 primitives against this contract on `ScriptedEngine` and the CPU tiny model: `split_off` then `append(consume)` restores identical state; peak ≤ one layer above steady state; `reserve(keep, total)` is lazy and reports growth before allocating; `checkpoint`/`restore_checkpoint` + re-prefill equals uninterrupted at the same split (needs 001 engine contract tasks)
- [ ] T003 `ScriptedEngine` profiles used by 003 tests: attention and hybrid (recurrent state only at computed positions), per-buffer view counts, peak-byte tracking, configurable capacity (Qwen3-Coder ≈ 105K tokens) (needs 001)
- [ ] T004 `cache/tree.rs`: nodes, live path, segment views over refcounted buffers, metadata edge splits, checkpoint and message marks, evicted counters, merge on commit; invariants 1–6 in `crates/mrun/tests/prefix_tree_props.rs`; lookup microbench ≤ 1 ms at 100K tokens
- [ ] T005 Consume 002 renderer output: message marks with content hashes (reasoning excluded), `system_tools_end`, `generation_start`; tests on golden renders (needs 002 renderer task)

## Phase 3: US1 — Fast repeated turns (P1) 🎯 MVP

- [ ] T006 [US1] `cache/mod.rs` lookup and plan for the live path (extend, trim, full hit at `P − 1`)
- [ ] T007 [US1] `cache/chunks.rs` prefill plan (R4); determinism tests; chunk ends at boundaries
- [ ] T008 [US1] Worker integration in contract order (`runtime/worker.rs`): admission rejection leaves the store unchanged; `initial_need` eviction; growth callback → `evict`; commit of whole chunks on cancel (needs 002 worker)
- [ ] T009 [US1] Usage `cached = min(resume, P − 1)` and processed into 002 usage and metadata; tests for full hit, append, cancel-then-retry

## Phase 4: US2 — Subagents never destroy the main conversation (P1)

- [ ] T010 [US2] `cache/switch.rs`: move out (tail ≥ 256 stored, else discarded), move in whole buffers, split back at the match, Off mode moves the live path out; engine error drops affected state and resets
- [ ] T011 [US2] `crates/mrun/tests/prefix_branches.rs`: main(≥ 80K) → subagent → main and main → title → main at 105K capacity keep 100% of the parent; peak ≤ one layer (SC-002, SC-004 structural)
- [ ] T012 [US2] Mutation suite `crates/mrun/tests/prefix_mutations.rs` (≥ 100 cases, attention + hybrid; asserts no stale reuse, expected cause and divergence, no files written) (SC-003)
- [ ] T013 [P] [US2] macOS `crates/mrun-engine/tests/prefix_equality.rs`: cache-on vs cache-off at the same split, ≥ 30 cases on Qwen3-8B and Qwen3.6, identical ids and logits (SC-005)

## Phase 5: US3 — Hybrid models reuse exactly (P1)

- [ ] T014 [US3] Checkpoint placement (system+tools end, last user/tool start, prompt end), tail-end checkpoint on move-out, cap 8 with middle-first drop; resume from deepest ≤ match with trim and re-prefill of the gap
- [ ] T015 [US3] Hybrid tests on `ScriptedEngine` (append, re-rendered reply, edited last message, retried tool result, round trip) and on Qwen3.6 for SC-006 (needs 001 hybrid model)

## Phase 6: US4 — Know why the cache missed (P1)

- [ ] T016 [P] [US4] `cache/miss.rs`: causes and precedence per data model; one test per cause, including `template` from re-serialized tool arguments (002 FR-004) and `capacity` after pressure
- [ ] T017 [US4] Miss report in 002 metadata (headers, `: mrun` comments), `/mrun/status`, `mrun ps`; `--debug-cache` terminal excerpts refused off-loopback; audit test that no excerpt reaches responses or disk (needs 002 metadata task)

## Phase 7: US5 — The cache never costs memory safety (P2)

- [ ] T018 [P] [US5] `cache/evict.rs`: LRU units (leaf + same-buffer single-child chain), soft pins last, live-path checkpoints after all branches, `clear_cache`; tests that growth-time eviction always succeeds after admission
- [ ] T019 [US5] `runtime/pressure.rs`: macOS source → atomic level read at per-layer check points and per token; warning halves branch bytes, critical evicts all branches and resets an idle live path; injected-event test < 1 s with the running request unaffected
- [ ] T020 [US5] Controls: `--no-cache`, `x-mrun-cache: off`, `mrun cache clear` / `POST /mrun/cache/clear` (live reset when idle); tests

## Phase 8: Polish

- [ ] T021 [P] SC-001 and SC-002 on the target Mac for Qwen3-Coder and Qwen3.6; SC-004 peak measured with MLX peak memory; record in research (needs 005 harness and trace)
- [ ] T022 [P] Site page `site/content/docs/prefix-cache.md` checked against the implementation (moves, checkpoints, causes, controls, shared-cache side channel)

## Dependencies

- 001: engine primitives (T002, T003), hybrid model (T015). 002: renderer marks (T005), worker
  (T008), metadata (T009, T017). 005: harness and trace for T021.
- T002–T005 → T006 → T007 → T008 → T009 (MVP) → T010 → T011/T012/T013; T014 after T010; T015 after
  T014; T016–T020 after T008 in any order; T021–T022 last.

## Parallel opportunities

T013, T016 and T018 touch separate files; T021 and T022 run together.
