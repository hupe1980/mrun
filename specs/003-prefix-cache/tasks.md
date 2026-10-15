---
description: "Tasks for 003 Prefix Cache"
---

# Tasks: Prefix Cache

**Input**: [plan.md](plan.md), [spec.md](spec.md), [research.md](research.md), [data-model.md](data-model.md), [contracts/cache.md](contracts/cache.md)

## Phase 1: Setup

- [ ] T001 Add proptest dev-dependency; create `crates/mrun/src/cache/{mod.rs, tree.rs, evict.rs, miss.rs}` skeletons

## Phase 2: Foundational

- [ ] T002 Verify 001's `Engine::snapshot/restore/drop_segment/trim` meet the contract on CPU (tiny model): restore + continue equals uninterrupted within 1e-3; snapshot buffers never alias the active cache; append after restore does not copy the whole buffer (memory counters)
- [ ] T003 `ScriptedEngine` segment tracking (which token ranges each segment holds) for structural tests
- [ ] T004 `cache/tree.rs`: radix tree (insert with edge split by segment copy, longest-match walk incl. live path, pin/unpin, leaf removal, `state_checkpoint` field always None and preserved by all operations — FR-012) + proptest invariants from the data model in `crates/mrun/tests/prefix_tree_props.rs`; lookup microbench (≤ 1 ms at 100K tokens)

## Phase 3: US1 — Fast repeated turns (P1) 🎯 MVP

- [ ] T005 [US1] `cache/mod.rs` lookup / evict_for / prepare / finish / release per contract, live-path materialization on replacement, boundaries from 002's renderer, periodic 4,096-token cuts
- [ ] T006 [US1] Integrate in `runtime/worker.rs` in contract order (lookup → admission → prepare → release → prefill → decode with lazy eviction → finish); tests: admission failure after a hit leaves the store unchanged; restore at exactly-full capacity never exceeds capacity (restore peak checked, partial restore with cause `capacity` when needed, materialization bounded, restored segments unpinned)
- [ ] T007 [US1] Supply cached/processed counts and miss data to 002's usage mapping and mrun metadata (headers / `: mrun` SSE comments, 002 FR-022); `--no-cache` and per-request opt-out
- [ ] T008 [US1] Full-prompt hit reprocesses exactly the last token (FR-004); divergence inside the live path trims without copying (FR-003) — tests

## Phase 4: US2 — Branches share safely (P1)

- [ ] T009 [US2] Mutation suite (≥ 100 cases: append, edit at positions, rewind, interleaved branches, model/template change, cancel mid-prefill) in `crates/mrun/tests/prefix_mutations.rs` asserting structural exactness and in-memory only (no files written)
- [ ] T010 [US2] macOS numerics `crates/mrun-engine/tests/prefix_numerics.rs` on 30 cases (SC-002 thresholds)
- [ ] T011 [US2] Template round-trip tests per model and API: re-rendered history (with reasoning, tool calls) vs generated tokens; mismatches classified `template`

## Phase 5: US3 — Know why the cache missed (P1)

- [ ] T012 [P] [US3] `cache/miss.rs`: closest-prefix tracking with ghost list, text hashes per boundary, classification precedence, volatile pattern table (Claude Code attribution block, timestamps); ≥ 20 injected cases (SC-004)
- [ ] T013 [US3] MissReport in headers, `/mrun/status`, `mrun ps`; `--debug-cache` excerpts on the server terminal only, refused off-loopback; audit test (no excerpts in responses or on disk)

## Phase 6: US4 — Cache never costs memory safety (P2)

- [ ] T014 [P] [US4] `cache/evict.rs`: LRU with turn-type order, never pinned, `clear_cache()` after evicting; tests
- [ ] T015 [US4] `runtime/pressure.rs`: macOS dispatch memory-pressure source → worker queue, handled between steps (warning → 50%, critical → all unpinned); Linux no-op; injected-event test asserting < 1 s and running request unaffected
- [ ] T016 [US4] `mrun cache clear` + `POST /mrun/cache/clear`; running request unaffected — test
- [ ] T016a [US4] Compaction hint (FR-013): on `x-claude-code-context-compacted`, mark the previous live path's segments as first eviction candidates; test with a scripted two-request sequence
- [ ] T016b [US4] Cache hints (FR-014): `cache_control` breakpoints become segment boundaries (via 002's renderer offsets); `prompt_cache_key` session tag: leaves tagged with a different session than the incoming request are evicted before the same session's leaves (FR-007 order); `x-mrun-cache: off` header (FR-009); tests that hints never change matched lengths and that the full eviction order holds

## Phase 7: Polish

- [ ] T017 [P] SC-001 (agent replay 5× TTFT, trace from 005) and SC-003 (restore < 10% of prefill) measured; record in research
- [ ] T018 [P] Site page `site/content/docs/prefix-cache.md` checked against the implementation, incl. the shared-cache side-channel note (FR-011)

## Dependencies

Needs 001 (Engine primitives) and 002 (renderer boundaries, headers, worker). T002–T004 → T005 →
T006 → T007/T008 → US2/US3/US4 in any order. SC-001 needs 005's trace.
