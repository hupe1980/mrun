---
description: "Tasks for 003 Prefix Cache"
---

# Tasks: Prefix Cache

**Input**: [plan.md](plan.md), [spec.md](spec.md), [research.md](research.md), [data-model.md](data-model.md), [contracts/cache.md](contracts/cache.md)

## Phase 1: Setup

- [ ] T001 Add proptest dev-dependency; create `crates/mrun/src/cache/{mod.rs, tree.rs, evict.rs, miss.rs, normalize.rs}` module skeletons

## Phase 2: Foundational

- [ ] T002 `crates/mrun-engine`: implement `Engine::snapshot(range)` (materialized slice copy + eval), `restore(segments, keep)` (copy into fresh step-aligned buffers, trim), `drop_segment`; CPU tests: restore + continue equals uninterrupted run within 1e-3 on tiny model; no aliasing (append after restore does not copy the full buffer — check via memory counters)
- [ ] T003 `ScriptedEngine` snapshot/restore with structural tracking (which token ranges each segment holds)
- [ ] T004 `cache/tree.rs`: radix tree (insert with edge split, longest-match walk, pin/unpin, remove leaf) + proptest invariants in `crates/mrun/tests/prefix_tree_props.rs`

## Phase 3: US1 — Fast repeated turns (P1) 🎯 MVP

- [ ] T005 [US1] `cache/mod.rs` `lookup`/`commit` using renderer boundaries (002 T004) + periodic 4,096-token cuts; hot path when the active cache is a prefix
- [ ] T006 [US1] Integrate in `runtime/worker.rs`: lookup → restore/hot → admission → prefill remainder → commit prompt segments → commit assistant segment on Stop/ToolCalls
- [ ] T007 [US1] Usage reports cached/processed exactly (FR-005) in all three APIs; per-request `"mrun": {"cache": false}` and `--no-cache`
- [ ] T008 [US1] Full-prompt hit reprocesses exactly the last token (FR-004) — test

## Phase 4: US2 — Branches share safely (P1)

- [ ] T009 [US2] Mutation suite (≥ 100 cases: append, edit at positions, rewind, interleaved branches, model/template change) in `crates/mrun/tests/prefix_mutations.rs` asserting structural exactness (reused ranges equal request tokens under the same key)
- [ ] T010 [US2] macOS numerics `crates/mrun-engine/tests/prefix_numerics.rs`: cache-on vs cache-off log-prob |Δ| ≤ 1e-2, greedy agreement ≥ 99% (SC-002)

## Phase 5: US3 — Know why the cache missed (P1)

- [ ] T011 [P] [US3] `cache/miss.rs`: closest-prefix tracking, divergence position, cause classification (new, edit, template, volatile, evicted, capacity, model, disabled); volatile patterns table (Claude Code attribution/billing header, timestamps); tests per cause
- [ ] T012 [US3] Expose MissReport in `usage.mrun.cache` / `response.metadata.mrun.cache`, `/mrun/status`, `mrun ps`; `--debug-cache` excerpts (never written to disk; audit test)
- [ ] T013 [US3] `cache/normalize.rs` opt-in `--normalize claude-code` profile + reporting (FR-013); off by default

## Phase 6: US4 — Cache never costs memory safety (P2)

- [ ] T014 [P] [US4] `cache/evict.rs`: score = recompute seconds / bytes × exp(−idle/10 min); never evict pinned; `evict_for(bytes)` used by admission; tests
- [ ] T015 [US4] `runtime/pressure.rs`: macOS dispatch memory-pressure source (warn → evict to 50%, critical → evict all unpinned), Linux no-op; test via injected events; manual check with `memory_pressure`
- [ ] T016 [US4] `mrun cache clear` + `POST /mrun/cache/clear`; running request unaffected — test

## Phase 7: Polish

- [ ] T017 [P] Measure SC-001 (agent replay 5× TTFT) and SC-003 (restore < 10% of prefill) with `mrun bench`; record in research
- [ ] T018 [P] README section "Why did my cache miss?"

## Dependencies

Needs 001 (engine snapshot API) and 002 (renderer boundaries, usage). T002–T004 → T005 → T006 →
T007/T008 → US2/US3/US4 in any order.
