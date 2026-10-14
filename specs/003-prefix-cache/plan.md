# Implementation Plan: Prefix Cache

**Branch**: `003-prefix-cache` | **Date**: 2026-10-14 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `specs/003-prefix-cache/spec.md`

## Summary

A token-level radix tree per model identity. The active cache is the tree's *live path*: while a
conversation continues, nothing is copied and divergence inside it is handled by trimming. Owned
KV segments (deep copies) are materialized only when the live path is about to be replaced (a
different conversation or branch arrives) and only for its message-boundary prefixes not already
stored. Lookup walks tokens; then admission (with lazy eviction), then restore by copy, then
prefill of the remainder. Every node owns a segment for exactly its edge; edge splits copy slices;
only leaves are evicted (LRU, assistant before user/tool before system). Memory-pressure events
reach the worker between steps and evict before running work is touched. Every request yields a
`MissReport` exposed through 002's mrun metadata (headers or SSE comments) and `mrun ps`.

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
**Performance Goals**: Lookup ≤ 1 ms for 100K-token prompts (microbench); restore of 30K tokens <
10% of its prefill time (SC-003); pressure response < 1 s (SC-005)
**Constraints**: Admission before restore; cache uses only KV capacity not occupied by the active
request; never evict in-use segments; no hashing-only matches
**Scale/Scope**: ≈ 100–110K tokens of KV in total on 48 GB with Qwen3-Coder-30B-A3B (96 KiB/token),
shared by the live conversation and cached segments ([001 R6](../001-engine-core/research.md#r6-memory-budget))

## Constitution Check

| Principle | Gate | Status |
| --- | --- | --- |
| I Agent latency | TTFT 5× on agent replay; zero-copy hot path | Pass |
| II Exactness | Token-walk matching under full model identity; stored-byte restore; no prompt rewriting | Pass |
| III Memory budget | Admission before restore; lazy eviction as output grows; pressure sheds cache + `clear_cache` | Pass |
| IV Local, private | In-memory only; excerpts only on the server terminal with `--debug-cache`, loopback only | Pass |
| V Own runtime | Pure Rust tree; MLX only via `Engine` | Pass |
| VI Simple | Whole-segment nodes, LRU by turn type, no normalization layer (deferred) | Pass |

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
└── miss.rs         # MissReport + cause classification
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
