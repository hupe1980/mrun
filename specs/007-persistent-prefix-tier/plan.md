# Implementation Plan: Persistent Prefix Tier

**Branch**: `007-persistent-prefix-tier` | **Date**: 2026-10-15 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `specs/007-persistent-prefix-tier/spec.md`

## Summary

An opt-in disk tier below 003's memory store. Branch segments and checkpoints of position-bound state are written
once, as exact bytes with their token ids, by a background writer reading immutable host views
handed over by the engine thread; the live path is written after 2 s of idleness. Entries are
self-describing files bound to the full cache key, format and MLX version; startup rebuilds an
in-memory index from headers. Families with position-bound state persist per turn only the end-of-prompt checkpoint (plus
system+tools once); branch entries always have a persisted parent (the live path is written up to
the branch point first). Lookup confirms by full token comparison, and restore reads straight into
budgeted arrays with per-block checksums when the disk match extends the memory match by ≥ 512
tokens; only the deepest checkpoint at or below the match is imported. LRU, a size cap, a free-space floor and max age bound the footprint; `mrun cache` shows
and clears it.

## Technical Context

**Language/Version**: Rust 1.94.0 (edition 2024)
**Primary Dependencies**: `xxhash-rust` (xxh3), `libc` (`flock`, `fcntl F_FULLFSYNC`/`F_NOCACHE`,
`pread`, `fsync` (Linux), `fchmod`), `serde_json` (headers); no MLX calls off the engine thread
**Storage**: `~/Library/Caches/mrun/prefix/<key16>/*.mrps` (`MRUN_CACHE_DIR` overrides;
`$XDG_CACHE_HOME/mrun/prefix` on Linux)
**Testing**: cargo test with `ScriptedEngine` (structural bytes per token) and a temp tier dir:
format round-trip, crash injection (kill between steps via a test hook), corruption, identity
mutation suite (SC-002), cap property test (SC-004); macOS cells via `mrun bench --suite persist`
**Target Platform**: macOS Apple Silicon (target), Linux for portable tests
**Project Type**: Cache subsystem inside the `mrun` binary plus CLI
**Performance Goals**: SC-001 (≥ 20× vs re-prefill at 30K); quiesce ≤ 2 ms; warm-path overhead
≤ 1% (SC-003); index rebuild ≤ 200 ms for 10,000 entries
**Constraints**: Engine-thread wait (including eviction) bounded by one 4 MiB block; restore admitted against the
budget; no request fails because of disk errors
**Scale/Scope**: Up to the cap (32 GiB default), ~10^4 entries

## Constitution Check

| Principle | Gate | Status |
| --- | --- | --- |
| I Agent latency | Restart/eviction misses become ≈ 1 s restores; no warm-path regression (SC-003); numbers from `mrun bench` | Pass |
| II Exactness | Exact bytes; full cache key + format + MLX version; token comparison before use; checksums; mutation suite (SC-002) | Pass |
| III Memory budget | Restores allocate inside the budget via 003 admission | Pass |
| IV Local, private | Opt-in only (v3.2.0 "persistence of inference state requires an explicit opt-in and is bound to the model identity"); 0700/0600; no request text stored; `x-mrun-cache: off` never persisted; no network | Pass |
| V Own runtime | Own format, no third-party cache layer; MLX touched only on the engine thread | Pass |
| VI Simple | One file per entry, index rebuilt from headers, plain LRU, three settings | Pass |

Post-design re-check: Pass. 003 FR-013 ("memory only") is superseded for opted-in servers; see
cross-spec changes.

## Project Structure

### Documentation (this feature)

```text
specs/007-persistent-prefix-tier/
├── plan.md  research.md  data-model.md  quickstart.md  tasks.md
├── contracts/persist.md
└── checklists/requirements.md
```

### Source Code (repository root)

```text
crates/mrun-engine/src/
├── engine.rs            # + export_views(), quiesce generation, import_alloc()/import_seal() (contract)
└── state_io.rs          # [mlx] host views of evaluated arrays; budgeted destination arrays
crates/mrun/src/cache/persist/
├── mod.rs               # Tier: open (lock, perms, sweep temp files), lookup, restore decision
├── format.rs            # MRPS header/blocks/footer encode + decode, xxh3 per 4 MiB block
├── index.rs             # rebuild from headers, boundary-hash map, parent chains, LRU
├── writer.rs            # background writer: jobs, per-block cancel, atomic publish
├── reader.rs            # pread into engine buffers, checksum verify
└── gc.rs                # cap, free-space floor, max age, descendant-first deletion
crates/mrun/src/cache/mod.rs        # + disk candidates in lookup, source in MissReport
crates/mrun/src/cli/cache.rs        # `mrun cache`, `mrun cache clear [--disk|--all] [--model M]`
crates/mrun/src/bench/suites/persist.rs  # FR-001 cells
crates/mrun/tests/
├── persist_format.rs  persist_crash.rs  persist_identity.rs  persist_cap.rs  persist_privacy.rs
```

**Structure Decision**: The tier is part of the prefix store (`cache/`) in the `mrun` crate; the
engine exposes only raw host views and budgeted destination buffers, so all logic is testable on
Linux with the scripted engine.

## Cross-spec changes (to fold into the owning specs)

- 001 engine contract: `export_views`, `import_alloc`, `import_seal` (see contract).
- 002: FR-021 recordings add a `server-start` marker line per serve process (FR-001 attribution);
  metadata `source` field and new causes; `POST /mrun/cache/clear?tier=memory|disk|all`.
- 003: FR-013 becomes "memory only unless 007 persistence is enabled"; miss causes add
  `disk-corrupt`; lookup returns disk candidates.
- 004: CLI contract adds `serve --persist/--no-persist`, `mrun cache`, `mrun cache clear
  --disk|--all|--model`; config `[cache] persist, persist_gib, persist_max_age_days`, env
  `MRUN_CACHE_DIR`; `doctor` checks FileVault, tier permissions and lock.
- 005: suite `persist` (timings with the model loaded); profile reset for mrun clears both tiers
  (`mrun cache clear --all`).
- 008 metrics: `mrun_cache_disk_writes_total`, `mrun_cache_disk_write_bytes_total`,
  `mrun_cache_disk_corrupt_total` (no read-speed gauge).
- Folded by T024.

## Complexity Tracking

None.
