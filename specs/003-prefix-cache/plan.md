# Implementation Plan: Prefix Cache

**Branch**: `003-prefix-cache` | **Date**: 2026-10-15 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `specs/003-prefix-cache/spec.md`

## Summary

A token radix tree per cache key whose live path is the engine's sequence state. Continuing a
conversation appends; switching moves the old tail out and the matched branch in, one layer at a
time, through 001's `split_off` and `append(consume)`. Segments are refcounted buffer views, so
edge splits are metadata; buffers move whole. Hybrid models capture up to 8 checkpoints per path at
planned chunk ends and resume from the deepest one at or below the match. Eviction is LRU over
branch units, lazily at the initial allocation and at each growth step; memory pressure is read at
the per-layer check point. Every request yields a miss report delivered through 002.

## Technical Context

**Language/Version**: Rust 1.94.0
**Primary Dependencies**: none new; 001 `Engine` primitives, 002 renderer marks and metadata;
macOS `dispatch_source_create(DISPATCH_SOURCE_TYPE_MEMORYPRESSURE)` via a small FFI shim
**Storage**: In memory only (FR-013)
**Testing**: proptest for tree invariants; mutation suite (≥ 100 cases) on `ScriptedEngine` with
attention and hybrid profiles and buffer/peak tracking; macOS equality and timing tests on real
weights
**Target Platform**: macOS (Metal); tree, policy and suites run on Linux
**Project Type**: Module of the `mrun` crate
**Performance Goals**: lookup ≤ 1 ms at 100K tokens; SC-001 ≥ 10× TTFT; move of 80K tokens bounded by
memory bandwidth (≈ 0.1 s on Qwen3-Coder); pressure response < 1 s
**Constraints**: one copy per unique token; transient ≤ one layer; no hard pins after `prepare`;
no hash-only matches; equality tests at a fixed chunk split
**Scale/Scope**: Qwen3-Coder ≈ 105K tokens of attention state (96 KiB/token); Qwen3.6 ≈ 8 GiB for
attention (≈ 20 KiB/token) plus checkpoints of 60–150 MiB each

## Constitution Check

| Principle | Gate | Status |
| --- | --- | --- |
| I Agent latency | SC-001 (10×), SC-002 (subagent round trip keeps the parent), measured with `mrun bench` | Pass |
| II Exactness | Token-walk matching under the full cache key; checkpoints exact; SC-005 bit equality at a fixed split; no prompt rewriting here | Pass |
| III Memory budget | Static admission (001); lazy eviction before every growth; one copy per token; peak ≤ one layer (SC-004); pressure sheds branches first | Pass |
| IV Local, private | In memory only; excerpts only with `--debug-cache` on loopback, terminal only; side channel documented | Pass |
| V Own runtime | Pure Rust tree; MLX only through `Engine` | Pass |
| VI Simple | Plain LRU, one opt-out header, three controls, no tuning knobs | Pass |

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
├── evict.rs      # LRU units, soft pins, pressure levels
└── miss.rs       # MissReport and cause classification
crates/mrun/src/runtime/pressure.rs   # macOS memory-pressure source; injectable no-op on Linux
crates/mrun/tests/
├── prefix_tree_props.rs
├── prefix_mutations.rs               # ≥ 100 cases, attention + hybrid ScriptedEngine
└── prefix_branches.rs                # SC-002 round trips at Qwen3-Coder capacity
crates/mrun-engine/tests/prefix_equality.rs   # macOS: SC-005
```

**Structure Decision**: Tree and policy live in `mrun` (testable anywhere); state bytes stay in
`mrun-engine`, addressed by segment and checkpoint ids.

## Complexity Tracking

| Item | Why needed | Simpler alternative rejected because |
| --- | --- | --- |
| Buffer views with whole-buffer moves (R2) | Edge splits without copies; no double-held state after partial matches | Copy on split costs a transient of the segment size; view-only moves hold matched state twice |
