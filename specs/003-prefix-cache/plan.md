# Implementation Plan: Prefix Cache

**Branch**: `003-prefix-cache` | **Date**: 2026-10-14 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `specs/003-prefix-cache/spec.md`

## Summary

A token-level radix tree per model identity whose nodes own immutable KV segments (materialized MLX
copies). Segments are cut at message boundaries reported by the chat renderer (end of system+tools,
end of each user/tool turn, end of each assistant turn) and every 4,096 tokens inside long
messages. Lookup walks tokens, restores the matched segments by copy into the active cache (or
keeps the active cache when the conversation simply continues), trims to the divergence point,
then prefills the rest. Eviction is cost-aware (recompute seconds saved per byte, decayed by idle
time); memory pressure triggers eviction before running work is touched. Every request yields a
`MissReport`. Lives on the engine thread; the `mrun` crate holds the tree logic and talks to
`Engine::snapshot/restore/drop_segment`.

## Technical Context

**Language/Version**: Rust 1.94.0
**Primary Dependencies**: none new (uses 001's `Engine` snapshot API and 002's renderer boundaries);
macOS `dispatch_source_create(DISPATCH_SOURCE_TYPE_MEMORYPRESSURE)` via a tiny FFI shim
**Storage**: In memory only (FR-010)
**Testing**: Property tests (proptest) for the radix tree (insert/lookup/split/evict invariants);
mutation suite (≥100 cases) on `ScriptedEngine` for structural staleness; macOS suite on real
weights for logit tolerance and TTFT (SC-001/002/003)
**Target Platform**: macOS; tree logic and staleness suite on Linux
**Project Type**: Module of the `mrun` crate
**Performance Goals**: Lookup ≤ 1 ms for 100K-token prompts; restore of 30K tokens < 10% of its
prefill time (SC-003); pressure response < 1 s (SC-005)
**Constraints**: Cache bytes ≤ free KV capacity; never evict in-use segments; no hashing-only
matches
**Scale/Scope**: Up to ~1M cached tokens worth of segments on 48 GB with Qwen3-Coder (96 KiB/token)

## Constitution Check

| Principle | Gate | Status |
| --- | --- | --- |
| I Agent latency | TTFT 5× on agent replay; zero-copy hot path | Pass |
| II Exactness | Token-walk matching under full model identity; stored-byte restore; normalization opt-in | Pass |
| III Memory budget | Cache lives in KV capacity; admission evicts first; pressure sheds cache | Pass |
| IV Local, private | In-memory only; excerpts only with `--debug-cache`, never on disk | Pass |
| V Own runtime | Pure Rust tree; MLX only via `Engine` | Pass |
| VI Simple | Whole-segment nodes (no paging, no block tables) | Pass |

Post-design re-check: Pass.

## Project Structure

### Documentation (this feature)

```text
specs/003-prefix-cache/
├── plan.md  research.md  data-model.md  quickstart.md
├── contracts/cache.md
└── tasks.md
```

### Source Code (repository root)

```text
crates/mrun/src/cache/
├── mod.rs          # PrefixStore API used by runtime::worker
├── tree.rs         # radix tree over token ids (edges = token slices)
├── evict.rs        # scoring, pressure handling
├── miss.rs         # MissReport + cause classification
└── normalize.rs    # opt-in client normalization profiles (FR-013)
crates/mrun/src/runtime/pressure.rs   # macOS memory-pressure source (no-op on Linux)
crates/mrun/tests/
├── prefix_tree_props.rs
└── prefix_mutations.rs              # ≥100 structural cases on ScriptedEngine
crates/mrun-engine/tests/prefix_numerics.rs   # macOS: cache-on vs cache-off logits tolerance
```

**Structure Decision**: Tree and policy in `mrun` (testable anywhere); KV bytes stay in
`mrun-engine` addressed by `KvSegmentId`.

## Complexity Tracking

None.
