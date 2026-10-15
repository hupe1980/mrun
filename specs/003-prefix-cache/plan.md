# Implementation Plan: Prefix Cache

**Branch**: `003-prefix-cache` | **Date**: 2026-10-15 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `specs/003-prefix-cache/spec.md`

## Summary

A token radix tree per cache key whose live path is the engine's sequence state. Continuing a
conversation appends; switching moves the old tail out and the matched branch in, one layer at a
time, through 001's `split_off` and `append(consume)`. Segments are refcounted buffer views, so
edge splits are metadata; buffers move whole. Models with position-bound state (linear attention;
0.1: Qwen3.6) capture up to 8 checkpoints per path at planned chunk ends and resume from the deepest
one at or below the match; all other families, gpt-oss sliding-window layers included (full KV
kept), are positional only and resume exactly at the match. Eviction is LRU over
branch units, lazily at the initial allocation and on every `NeedsMemory{bytes}` (evict, retry);
memory pressure is read at every per-layer check point and acted on at the next whole-chunk end
(001 prefill yield). The per-request loop stays in 001's `runtime/generate.rs`, which takes the
`PrefixStore` as its reuse policy. Every request yields a miss report delivered through 002.

## Technical Context

**Language/Version**: Rust 1.94.0
**Primary Dependencies**: none new; 001 `Engine` primitives, 002 renderer marks and metadata;
macOS `dispatch_source_create(DISPATCH_SOURCE_TYPE_MEMORYPRESSURE)` via a small FFI shim
**Storage**: In memory only (FR-013)
**Testing**: proptest for tree invariants; mutation suite (research R-mutations) on `ScriptedEngine` with
positional-only, sliding-window and position-bound profiles and buffer/peak tracking; macOS equality and timing tests on real
weights
**Target Platform**: macOS (Metal); tree, policy and suites run on Linux
**Project Type**: Module of the `mrun` crate
**Performance Goals**: lookup ≤ 1 ms at 100K tokens; SC-001 ≥ 10× TTFT; move of 80K tokens bounded by
memory bandwidth (90K tokens on Qwen3-Coder ≈ 8.2 GiB read + write ≥ 65 ms at 273 GB/s; round
trip ≈ 0.15–0.2 s); pressure response < 1 s
**Constraints**: one copy per unique token; transient ≤ one layer; no hard pins after `prepare`;
no hash-only matches; equality tests at a fixed chunk split
**Scale/Scope**: Qwen3-Coder ≈ 105K tokens of attention state (96 KiB/token); Qwen3.6 ≈ 8 GiB for
attention (≈ 20 KiB/token) plus checkpoints of ≈ 61 MiB each (8 per path ≈ 0.5 GiB); gpt-oss-20b
48 KiB/token, no checkpoints (001 data model)

## Constitution Check

| Principle | Gate | Status |
| --- | --- | --- |
| I Agent latency | SC-001 (10×), SC-002 (subagent round trip keeps the parent), measured with `mrun bench` | Pass |
| II Exactness | Token-walk matching under the full cache key; checkpoints exact; SC-005 bit equality at a fixed split; no prompt rewriting here | Pass |
| III Memory budget | Static admission (001); lazy eviction before every growth; one copy per token; peak ≤ one layer (SC-004); pressure sheds branches first | Pass |
| IV Local, private | In memory only; reports carry positions and roles, never text; side channel documented | Pass |
| V Own runtime | Pure Rust tree; MLX only through `Engine` | Pass |
| VI Simple | Plain LRU, no per-request opt-out, two controls (`--no-cache`, `mrun cache clear`), no tuning knobs | Pass |

Post-design re-check: Pass.

## Project Structure

### Documentation (this feature)

```text
specs/003-prefix-cache/
├── spec.md  plan.md  research.md  data-model.md  quickstart.md  tasks.md
├── contracts/cache.md
└── checklists/requirements.md
```

### Source Code (repository root)

```text
crates/mrun/src/cache/
├── mod.rs        # PrefixStore API (contract)
├── tree.rs       # radix tree, live path, segment views, buffers, merge on commit
├── switch.rs     # plan + prepare: move out/in, split back, checkpoint restore
├── chunks.rs     # prefill plan (R4), checkpoint placement and the 8-per-path cap (R3)
├── evict.rs      # LRU units, pressure levels
└── miss.rs       # MissReport: five causes, divergence, message index and role, gap
crates/mrun/src/runtime/pressure.rs   # macOS memory-pressure source; injectable no-op on Linux
crates/mrun/tests/
├── prefix_tree_props.rs
├── prefix_mutations.rs               # research R-mutations coverage, three profiles
├── prefix_branches.rs                # SC-002 round trips at Qwen3-Coder capacity
└── prefix_equality.rs                # macOS, feature `mlx`: SC-005 (uses PrefixStore, so it lives in `mrun`)
```

**Structure Decision**: Tree and policy live in `mrun` (testable anywhere); state bytes stay in
`mrun-engine`, addressed by segment and checkpoint ids.

## Complexity Tracking

| Item | Why needed | Simpler alternative rejected because |
| --- | --- | --- |
| Buffer views with whole-buffer moves (R2) | Edge splits without copies; no double-held state after partial matches | Copy on split costs a transient of the segment size; view-only moves hold matched state twice |
