---
description: "Tasks for 003 Prefix Cache"
---

# Tasks: Prefix Cache

**Input**: [plan.md](plan.md), [spec.md](spec.md), [research.md](research.md),
[data-model.md](data-model.md), [contracts/cache.md](contracts/cache.md)

**Scope**: Lean 0.1 (D025, 2026-10-15). Cut tasks keep their IDs and say so.

## Format: `[ID] [P?] [Story] Description`

## Phase 1: Setup

- [X] T001 proptest dev-dependency; skeletons `crates/mrun/src/cache/{mod,tree,switch,chunks,evict,miss}.rs` *Note*: modules `cache/{mod,chunks,miss}.rs` (tree, switch and eviction live in `mod.rs`: the position-attached tree made separate files unnecessary); a seeded generator replaces proptest (no new dependency).

## Phase 2: Foundational (blocks all stories)

- [ ] T002 Verify 001 primitives against this contract on `ScriptedEngine` and the CPU tiny model: `split_off` (with and without `shrink`) then `append(consume)` restores identical state incl. the segment's recurrent end state; peak ≤ one layer above steady state; `reserve(keep, total)` is lazy and returns `NeedsMemory{bytes}` before allocating; prefill yield + continuation keeps the chunk split; `checkpoint`/`restore_checkpoint` + re-prefill equals uninterrupted at the same split (needs 001 T008a, T011a, T032–T034)
- [ ] T003 Extend 001's `ScriptedEngine` (001 T011a; no duplicate double) with the profiles 003 tests use: positional only, sliding-window (positional, full KV) and position-bound (state only at computed positions), per-buffer view counts, peak-byte tracking, configurable capacity (Qwen3-Coder ≈ 105K tokens), injectable yield
- [X] T004 `cache/tree.rs` structure: nodes, live path, segment views over refcounted buffers, metadata-only edge splits, checkpoint and message marks (start, role); in-memory only, no file I/O in `cache/` (FR-013) *Note*: implemented as a trunk plus stored nodes attached at positions (one segment per node, no views; data model revised 2026-10-15) in `cache/mod.rs`.
- [X] T004a `cache/tree.rs` lifecycle: merge on commit where the live path retraces a stored branch, evicted counters with reasons, one tree per `CacheKey` (a new key discards the old tree, FR-014) *Note*: merge on commit drops stored nodes the trunk retraces entirely; removal records (owner, position, first token) back the `evicted` cause; `PrefixStore::rekey` discards the tree.
- [X] T004b Invariants 1–6 as property tests (7 via the T012 mutation oracle, 8 via T008/T012) in `crates/mrun/tests/prefix_tree_props.rs`; lookup microbench ≤ 1 ms at 100K tokens *Note*: invariants 2, 4, 7 and 8 are asserted after every request of the seeded mutation suite (`tests/prefix_cache.rs`); the 100K-token lookup microbench is still open (T004c).
- [X] T005 Fill 001's `RenderedPrompt` (T011j) from the renderer (message starts and roles, `system_tools_end`, `generation_start`); tests on golden renders (needs 002 T004a) *Note*: done by 002 T004a (`chat/render.rs`).

- [ ] T004c Lookup microbench ≤ 1 ms at 100K tokens (from T004b)
- [ ] T017a Audit test: no prompt text reaches responses, logs or disk (from T017)

## Phase 3: US1 — Fast repeated turns (P1) 🎯 MVP

- [X] T006 [US1] `cache/mod.rs` lookup and plan for the live path (extend, trim, full hit at `P − 1`) *Note*: `PrefixStore::lookup` (depth-first over the trunk and stored nodes).
- [X] T007 [US1] `cache/chunks.rs` prefill plan (R4, `chunk_len` floor 128): `ChunkPlan.boundaries` holds only the selected chunk ends; determinism tests; chunk ends at boundaries *Note*: `cache/chunks.rs`.
- [X] T008 [US1] `PrefixStore` implements the reshaped `ReusePolicy` seam ([contract](contracts/cache.md)): `initial_need`, `evict(bytes, protect) -> u64`, `prepare(engine, &RenderedPrompt, total) -> Prepared` (the policy reserves), `on_pressure(level)` from prefill yields and decode, `commit`, `clear`; replaces 001 `Session` reuse when the cache is enabled (`serve` without `--no-cache`); admission rejection leaves the cache unchanged; on `Err(NeedsMemory{bytes})` → `evict(bytes)` and retry; pressure → yield → evict → continue with the same plan; commit of whole chunks on cancel (needs 002 T005 and the 001 seam change) *Note*: `impl ReusePolicy for PrefixStore`; `serve` builds it through `serve::cache_policy` (`EngineHandle::spawn_with`); splits needing memory evict other branches, else discard; growth `NeedsMemory` is handled by 001's loop.
- [X] T009 [US1] Usage `cached = min(resume, P − 1)` and processed into 002 usage and metadata; tests for full hit, append, cancel-then-retry, resume at a checkpoint (position-bound profile) (needs 002 T015, T017) *Note*: cases `full_hit_and_append_reuse_everything_before_the_new_tokens`, `a_cancelled_prefill_keeps_its_whole_chunks_for_the_retry`, `hybrid_resumes_from_checkpoints_with_a_gap`, and over HTTP `xapi_cached_tokens_resume_p_minus_1`, `cache_report_status_and_clear`.

## Phase 4: US2 — Subagents never destroy the main conversation (P1)

- [X] T010 [US2] `cache/switch.rs`: move out (tail ≥ 256 stored, else discarded), move in whole buffers, split back at the match, Off mode moves the live path out; engine error drops affected state and resets *Note*: in `cache/mod.rs` (`cut_trunk`, `move_in`, split back); engine errors clear the store.
- [ ] T011 [US2] `crates/mrun/tests/prefix_branches.rs`: main(≥ 80K) → subagent → main and main → title → main at 105K capacity keep 100% of the parent; peak ≤ one layer (SC-002, SC-004 structural)
- [X] T012 [US2] Mutation suite `crates/mrun/tests/prefix_mutations.rs` with the research R-mutations coverage on all three profiles: no stale reuse, expected cause, divergence, message index and role, gap; no files written (FR-013); a changed model identity or template file → new key, old tree discarded, cause `model` (FR-014) (SC-003) *Note*: `mutation_suite_attention`/`_hybrid` (8 seeds × 150 requests, edits, subagents, repeats, eviction) plus `a_new_cache_key_discards_the_tree_and_reports_model`; the sliding-window profile joins with T003/T015a.
- [ ] T013 [P] [US2] macOS `crates/mrun/tests/prefix_equality.rs` (feature `mlx`; it drives `PrefixStore`, so it lives in `mrun`, not `mrun-engine`): cache-on vs cache-off at the same split, ≥ 30 cases each on Qwen3-8B, Qwen3.6 and gpt-oss-20b, identical ids and logits (SC-005)

## Phase 5: US3 — Models with position-bound state reuse exactly (P1)

- [X] T014 [US3] Checkpoint placement, only when `ModelInfo.recurrent_bytes > 0` (no family names in `cache/`) (system+tools end, last user/tool start, `generation_start`); a moved-out tail's end state stays in its 001 segment (restored by `append`, no checkpoint id, not counted); cap 8 per path, dropping the shallowest checkpoint other than the first; resume from deepest ≤ match with trim and re-prefill of the gap *Note*: `planned_checkpoints` (only when `recurrent_bytes > 0`), cap 8 dropping the shallowest but the first.
- [ ] T015 [US3] Position-bound tests on `ScriptedEngine` (append, re-rendered reply, edited last message, retried tool result, round trip, new user turn after ≥ 10 tool turns) and on Qwen3.6 for SC-006 (needs 001 T031, T032)
- [ ] T015a [US3] Sliding-window restores (SC-007, US3 scenario 4): `ScriptedEngine` sliding-window profile (no checkpoints planned, gap 0 at any position) and gpt-oss-20b on macOS: divergences and switches at ≥ 20 positions before, at and beyond the window → exact resume, bit-identical logits vs cache-off (needs 001 T042, T045)

## Phase 6: US4 — Know why the cache missed (P1)

- [X] T016 [P] [US4] `cache/miss.rs`: five causes and precedence per data model, divergence with message index and role, gap; one test per cause incl. `changed` for a tool-list edit (index 0), an earlier-message edit and a re-rendered reply (002 FR-004), and `evicted` after pressure *Note*: `cache/miss.rs` with unit tests; `evicted` after pressure is covered by the eviction path, not a dedicated case.
- [X] T017 [US4] Miss report in 002 metadata (headers, `: mrun` comments), `/mrun/status`, `mrun ps`; audit test that no prompt text reaches responses, logs or disk (needs 002 T015, T017). (`--debug-cache`: removed, 0.1 scope cut, 2026-10-15) *Note*: headers and `: mrun` comment fields, `/mrun/status` `cache` and `last`; API message indices come from the decoders (`ChatRequest.api_index`). `mrun ps` is 004; the no-prompt-text audit test is still open (T017a).

## Phase 7: US5 — The cache never costs memory safety (P2)

- [X] T018 [P] [US5] `cache/evict.rs`: LRU units (leaf + same-buffer single-child chain), live-path checkpoints after all branches, MLX `clear_cache`; tests that growth-time eviction always succeeds after admission. (Soft pins: removed, 0.1 scope cut, 2026-10-15) *Note*: LRU over stored leaves in `cache/mod.rs`, then trunk checkpoints (keeping the one the request resumes from); MLX `clear_cache` comes with the MLX engine.
- [ ] T019 [US5] `runtime/pressure.rs`: macOS source calling 001's `PressureGauge::raise` → atomic level read at per-layer check points and per token; during prefill a set level requests a yield and is acted on at the next whole-chunk end; warning halves branch bytes, critical evicts all branches and resets an idle live path; injected-event test < 1 s with the running request unaffected
- [X] T020 [US5] Controls: `--no-cache`, `mrun cache clear` / `POST /mrun/cache/clear` (live reset when idle); tests. (`x-mrun-cache`: removed, 0.1 scope cut, 2026-10-15) *Note*: `--no-cache` (`CacheMode::Off`, cause `disabled`), `POST /mrun/cache/clear` (200 with the report when idle, 202 `after-request` while busy); the `mrun cache clear` command is 004.

## Phase 8: Polish

- [ ] T021 [P] SC-001 synthetic part (30K conversation, 10 turns; gates M3) and SC-002 on the target Mac for Qwen3-Coder and Qwen3.6; SC-004 peak measured with MLX peak memory; record in research
- [ ] T021a [P] SC-001 trace part (`agent/opencode-refactor` replay), reported with M5 (needs 005 harness and trace)
- [X] T022 [P] Site page `site/content/docs/prefix-cache.md` checked against the implementation (moves, checkpoints, causes, controls, shared-cache side channel) *Note*: message indices now name the API array index (system prompt and tools as role `system`); `evicted` lists the short-tail rule.

## Dependencies

- 001: T002 needs 001 T008a, T011a, T032–T034; T003 extends 001 T011a; T008 needs 001 T011b and
  the seam change (contract); T015 needs 001 T031/T032; T015a needs 001 T042/T045. 002: T005
  needs 002 T004a; T008 needs 002 T005; T009/T017 need 002 T015/T017. 005: harness and trace for
  T021a.
- T002–T005 (T004 → T004a → T004b) → T006 → T007 → T008 → T009 (MVP) → T010 → T011/T012/T013;
  T014 after T010; T015 after T014; T016–T020 after T008 in any order; T021–T022 last; T021a
  with 005.

## Parallel opportunities

T013, T016 and T018 touch separate files; T021 and T022 run together.
