---
description: "Tasks for 003 Prefix Cache"
---

# Tasks: Prefix Cache

**Input**: [plan.md](plan.md), [spec.md](spec.md), [research.md](research.md),
[data-model.md](data-model.md), [contracts/cache.md](contracts/cache.md)

## Format: `[ID] [P?] [Story] Description`

## Phase 1: Setup

- [ ] T001 proptest dev-dependency; skeletons `crates/mrun/src/cache/{mod,tree,switch,chunks,evict,miss}.rs`

## Phase 2: Foundational (blocks all stories)

- [ ] T002 Verify 001 primitives against this contract on `ScriptedEngine` and the CPU tiny model: `split_off` (with and without `shrink`) then `append(consume)` restores identical state incl. the segment's recurrent end state; peak ≤ one layer above steady state; `reserve(keep, total)` is lazy and returns `NeedsMemory{bytes}` before allocating; prefill yield + continuation keeps the chunk split; `checkpoint`/`restore_checkpoint` + re-prefill equals uninterrupted at the same split (needs 001 T008a, T011a, T032–T034)
- [ ] T003 Extend 001's `ScriptedEngine` (001 T011a; no duplicate double) with the profiles 003 tests use: positional only, sliding-window (positional, full KV) and position-bound (state only at computed positions), per-buffer view counts, peak-byte tracking, configurable capacity (Qwen3-Coder ≈ 105K tokens), injectable yield
- [ ] T004 `cache/tree.rs`: nodes, live path, segment views over refcounted buffers, metadata edge splits, checkpoint and message marks, evicted counters, merge on commit; one tree per `CacheKey`, a new key discards the old tree (FR-014); in-memory only, no file I/O in `cache/` (FR-013); invariants 1–6 in `crates/mrun/tests/prefix_tree_props.rs`; lookup microbench ≤ 1 ms at 100K tokens
- [ ] T005 Consume 002 renderer output: message marks with content hashes (reasoning excluded), `system_tools_end`, `generation_start`; tests on golden renders (needs 002 T004)

## Phase 3: US1 — Fast repeated turns (P1) 🎯 MVP

- [ ] T006 [US1] `cache/mod.rs` lookup and plan for the live path (extend, trim, full hit at `P − 1`)
- [ ] T007 [US1] `cache/chunks.rs` prefill plan (R4, `chunk_len` floor 128): `PrefillPlan.boundaries` holds only the selected chunk ends; determinism tests; chunk ends at boundaries
- [ ] T008 [US1] `PrefixStore` as the reuse policy of 001's `runtime/generate.rs`, in contract order; replaces 001 Session reuse when the cache is enabled (`serve` without `--no-cache`); 002's `runtime/worker.rs` only dequeues and calls `generate`, applying pending soft pins between requests and at each prefill yield; admission rejection leaves the cache unchanged; `initial_need` eviction (no move-transient term); on `Err(NeedsMemory{bytes})` → `evict(bytes)` and retry the call; pressure → yield → evict → continue with the same plan; commit of whole chunks on cancel (needs 002 T005 and 001 T011b)
- [ ] T009 [US1] Usage `cached = min(resume, P − 1)` and processed into 002 usage and metadata; tests for full hit, append, cancel-then-retry, resume at a checkpoint (position-bound profile) (needs 002 T015, T017)

## Phase 4: US2 — Subagents never destroy the main conversation (P1)

- [ ] T010 [US2] `cache/switch.rs`: move out (tail ≥ 256 stored, else discarded), move in whole buffers, split back at the match, Off mode moves the live path out; engine error drops affected state and resets
- [ ] T011 [US2] `crates/mrun/tests/prefix_branches.rs`: main(≥ 80K) → subagent → main and main → title → main at 105K capacity keep 100% of the parent; peak ≤ one layer (SC-002, SC-004 structural)
- [ ] T012 [US2] Mutation suite `crates/mrun/tests/prefix_mutations.rs` (≥ 100 cases, positional-only, sliding-window and position-bound profiles; asserts no stale reuse, expected cause and divergence, no files written (FR-013), a changed model identity or template file → new key, old tree discarded, cause `model` (FR-014)) (SC-003)
- [ ] T013 [P] [US2] macOS `crates/mrun-engine/tests/prefix_equality.rs`: cache-on vs cache-off at the same split, ≥ 30 cases each on Qwen3-8B, Qwen3.6 and gpt-oss-20b, identical ids and logits (SC-005)

## Phase 5: US3 — Models with position-bound state reuse exactly (P1)

- [ ] T014 [US3] Checkpoint placement, only when `ModelInfo.recurrent_bytes > 0` (no family names in `cache/`) (system+tools end, last user/tool start, `generation_start`); a moved-out tail's end state stays in its 001 segment (restored by `append`, no checkpoint id, not counted); cap 8 keeping the first, the two deepest and the newest at a user (non-tool) message start, dropping the one nearest the midpoint among the rest; resume from deepest ≤ match with trim and re-prefill of the gap
- [ ] T015 [US3] Position-bound tests on `ScriptedEngine` (append, re-rendered reply, edited last message, retried tool result, round trip, new user turn after ≥ 10 tool turns) and on Qwen3.6 for SC-006 (needs 001 T031, T032)

- [ ] T015a [US3] Sliding-window restores (SC-007, US3 scenario 4): `ScriptedEngine` sliding-window profile (no checkpoints planned, gap 0 at any position) and gpt-oss-20b on macOS: divergences and switches at ≥ 20 positions before, at and beyond the window → exact resume, bit-identical logits vs cache-off (needs 001 T042, T045)

## Phase 6: US4 — Know why the cache missed (P1)

- [ ] T016 [P] [US4] `cache/miss.rs`: causes and precedence per data model; one test per cause incl. `system` (changed system prompt or tool list), `template` from re-serialized tool arguments (002 FR-004) and `capacity` after pressure
- [ ] T017 [US4] Miss report in 002 metadata (headers, `: mrun` comments), `/mrun/status`, `mrun ps`; `--debug-cache` terminal excerpts refused off-loopback; audit test that no excerpt reaches responses or disk (needs 002 T015, T017)

## Phase 7: US5 — The cache never costs memory safety (P2)

- [ ] T018 [P] [US5] `cache/evict.rs`: LRU units (leaf + same-buffer single-child chain), soft pins last, live-path checkpoints after all branches, `clear_cache`; tests that growth-time eviction always succeeds after admission
- [ ] T019 [US5] `runtime/pressure.rs`: macOS source → atomic level read at per-layer check points and per token; during prefill a set level requests a yield and is acted on at the next whole-chunk end; warning halves branch bytes, critical evicts all branches and resets an idle live path; injected-event test < 1 s with the running request unaffected
- [ ] T020 [US5] Controls: `--no-cache`, `x-mrun-cache: off`, `mrun cache clear` / `POST /mrun/cache/clear` (live reset when idle); tests

## Phase 8: Polish

- [ ] T021 [P] SC-001 synthetic part (30K conversation, 10 turns; gates M3) and SC-002 on the target Mac for Qwen3-Coder and Qwen3.6; SC-004 peak measured with MLX peak memory; record in research
- [ ] T021a [P] SC-001 trace part (`agent/opencode-refactor` replay), reported with M5 (needs 005 harness and trace)
- [ ] T022 [P] Site page `site/content/docs/prefix-cache.md` checked against the implementation (moves, checkpoints, causes, controls, shared-cache side channel)

## Dependencies

- 001: T002 needs 001 T008a, T011a, T032–T034; T003 extends 001 T011a; T008 needs 001 T011b;
  T015 needs 001 T031/T032; T015a needs 001 T042/T045. 002: T005 needs 002 T004; T008 needs 002 T005; T009/T017 need 002
  T015/T017. 005: harness and trace for T021a.
- T002–T005 → T006 → T007 → T008 → T009 (MVP) → T010 → T011/T012/T013; T014 after T010; T015 after
  T014; T016–T020 after T008 in any order; T021–T022 last; T021a with 005.

## Parallel opportunities

T013, T016 and T018 touch separate files; T021 and T022 run together.
