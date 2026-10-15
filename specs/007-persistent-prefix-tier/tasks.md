---
description: "Tasks for 007 Persistent Prefix Tier"
---

# Tasks: Persistent Prefix Tier

**Input**: [plan.md](plan.md), [spec.md](spec.md), [research.md](research.md), [data-model.md](data-model.md), [contracts/persist.md](contracts/persist.md)

**Prerequisites**: 003 (live path, branches, checkpoints, miss reports) and 004 (identity, config,
CLI) implemented; 002 `--record` with server-start markers.

## Phase 1: Setup and measure first

- [ ] T001 Add `xxhash-rust`, `libc` to `crates/mrun/Cargo.toml`; config keys and env per data model in `crates/mrun/src/config.rs`; `--persist/--no-persist` in `cli/serve.rs` (off by default, FR-002)
- [ ] T002 [P] `crates/mrun/src/bench/suites/persist.rs`: read/write GB/s (4 MiB blocks, `F_NOCACHE`), restore-30k cell (needs T014 for the restore half; read speed first)
- [ ] T003 [P] `--from RECORD_DIR` attribution: replay arrival order through a simulated LRU memory store with server-start markers; restart/evicted/other tokens; simulated disk tier at the cap
- [ ] T004 Run FR-001 on the target Mac and maintainers' recordings; record the go/no-go note in `research.md` R1

**Checkpoint**: tier value measured.

## Phase 2: Foundational

- [ ] T005 `crates/mrun/src/cache/persist/format.rs`: MRPS encode/decode, 16 KiB-aligned tensor blocks, xxh3 per 4 MiB block, header hash; round-trip and fuzzed-header tests in `crates/mrun/tests/persist_format.rs`
- [ ] T006 `crates/mrun-engine/src/engine.rs` + `state_io.rs`: `export_views`, `import_alloc`, `import_commit`, generation counter; `ScriptedEngine` versions with structural bytes per token
- [ ] T007 `crates/mrun/src/cache/persist/mod.rs`: open (directory create 0700, ownership/permission refusal, backup/Spotlight exclusion, `flock`, temp-file sweep), EntryKey from the loaded model (FR-004, FR-010, FR-014)
- [ ] T008 [P] `crates/mrun/src/cache/persist/index.rs`: rebuild from headers, boundary-hash map, parent chains, LRU by mtime; rebuild-time test for 10,000 entries

**Checkpoint**: entries can be written and indexed in isolation.

## Phase 3: US2 — never reuse the wrong state (P1)

- [ ] T009 [US2] `crates/mrun/src/cache/persist/reader.rs`: full token comparison before use, parent-chain token checks, per-block checksum verification while reading, corrupt → delete + `disk-corrupt` (FR-005, FR-006)
- [ ] T010 [US2] `crates/mrun/tests/persist_identity.rs`: ≥ 100 mutation cases across restarts (weights, tokenizer, template, state format, format version, mrun version, edited tokens, truncated/bit-flipped/foreign files) — zero stale or corrupt loads (SC-002)
- [ ] T011 [US2] `crates/mrun/tests/persist_crash.rs`: crash hook at every write step (before/after fsync, rename, dir sync) × 50 points; restart sees only complete entries (SC-005)

## Phase 4: US1 — resume fast after a restart (P1) 🎯 MVP

- [ ] T012 [US1] `crates/mrun/src/cache/persist/writer.rs`: background writer, jobs from branch move-out and idle live path (2 s), per-block cancel, `quiesce(range)` ≤ one block, atomic publish (temp, fsync, rename, dir fsync) (FR-007)
- [ ] T013 [US1] Engine-thread integration: `quiesce` before trim, growth, move-out, eviction, shutdown; test that the engine never waits more than one block (scripted timing hook)
- [ ] T014 [US1] `crates/mrun/src/cache/mod.rs`: disk candidates in lookup, restore-or-recompute decision (measured read GB/s vs prefill rate), `import_alloc` → read → `import_commit`, hybrid checkpoint resume, fallback on any error; MissReport `source` and `disk-slower` (FR-008, FR-013)
- [ ] T015 [US1] Read-ahead at queue admission: index candidates → `F_RDADVISE` (Linux `posix_fadvise`) (FR-009)
- [ ] T016 [US1] macOS: `persist/restore-30k` for both models; ≥ 20× (SC-001); warm-path TTFT with persistence on vs off within 1% (SC-003)

## Phase 5: US3 — see and control (P2)

- [ ] T017 [P] [US3] `crates/mrun/src/cache/persist/gc.rs`: cap, free-space floor, max age, descendant-first deletion, hourly sweep; `crates/mrun/tests/persist_cap.rs` property test at 3× cap (SC-004)
- [ ] T018 [P] [US3] `crates/mrun/src/cli/cache.rs`: `mrun cache [--json]`, `clear --disk [--model M]`, `clear --all`, server or direct mode with lock; `POST /mrun/cache/clear?tier=`; snapshot tests
- [ ] T019 [US3] `/mrun/status` `cache.disk` TierStats; `doctor` checks (FileVault off with persistence on, permissions, lock holder); `x-mrun-cache-source` header and SSE field

## Phase 6: Polish

- [ ] T020 [P] Site page "Persistent cache": what is stored, where, privacy, how to clear, measured numbers
- [ ] T021 Run quickstart scenarios 1–6 on the target Mac; record results

## Dependencies

T001 → T002/T003 → T004 (gate). T005/T006 → T007 → T008 → T009 → T010/T011. T012 → T013 → T014 →
T015 → T016. US3 after T008 (T017) and T014 (T018/T019). T002's restore half needs T014.
