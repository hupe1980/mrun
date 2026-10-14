# Research: Prefix Cache

Sources: [literature-study](../../concepts/research/literature-study.md) (A-1..A-9),
[mlx-rust §7.3–7.4](../../concepts/research/mlx-rust.md), [mistralrs-study](../../concepts/research/mistralrs-study.md),
[landscape](../../concepts/research/landscape-2026-10.md).

## R1: Structure

- **Decision**: Radix tree over token ids (SGLang RadixAttention / ChunkAttention idea) with
  whole-segment KV nodes instead of paged blocks.
- **Rationale**: Exact longest-prefix in O(prompt length) with shared storage across branches;
  avoids paged-attention kernels (no new kernels, Principle V); mlx-lm's server trie proves the
  token-trie approach on MLX.
- **Alternatives**: Hash-chained fixed blocks (oMLX, vLLM APC) — needs block concatenation on every
  hit and token confirmation anyway; vLLM CVE-2025-25183 shows hash-only matching is unsafe.
  Whole-conversation entries only (mlx-lm) — wastes memory on branches.

## R2: Segment boundaries and the live path

- **Decision**: The active cache is the live path of the tree. Snapshot boundaries are the
  renderer's message ends (system+tools, each user/tool message, each assistant message) plus every
  4,096 tokens inside long messages. Segments are materialized (deep copy) only when the live path
  is about to be replaced, and only for boundary prefixes not already in the tree.
- **Rationale**: Avoids storing the active conversation twice (analysis T7); agent edits and
  branches happen at message boundaries (literature A-3/A-4; mlx-lm checkpoints at the same
  places); the 4K cap bounds reprocessing when an edit lands mid-message.

## R3: Order of operations

- **Decision**: (1) lookup: longest token match across the live path and the tree; pin matched
  nodes. (2) admission: `prompt + max_output ≤ capacity − pinned(other)`; evict unpinned leaves only
  as much as the initial allocation needs. (3) if the match is on the live path: trim to the
  divergence (zero copy); else materialize the live path's unstored boundaries, then restore the
  matched segments (copy into fresh buffers, then trim to the exact match). (4) prefill the
  remainder (≥ 1 token). (5) during decode, when the active cache grows, evict unpinned leaves on
  demand. Edge split during insert copies the segment into two slices.
- **Rationale**: Constitution III (admission before execution); MLX donation semantics
  ([mlx-rust §7.3](../../concepts/research/mlx-rust.md)); analysis D2, C1, C2, T7.

## R4: Eviction

- **Decision**: LRU over unpinned leaves with turn-type order (assistant before user/tool before
  system), as in mlx-lm's server. Evicting calls `clear_cache()` so memory returns to the OS.
  Pressure events from the dispatch source are queued to the worker and handled between steps:
  warning → evict to 50% of cached bytes, critical → evict all unpinned.
- **Rationale**: Simplest policy that matches agent structure; cost-aware scoring (Marconi) is a
  later optimization if `mrun bench` shows it matters (analysis G3, T8).

## R5: Miss diagnostics

- **Decision**: Nodes store, per boundary, a 64-bit hash of the message text (not the text). A
  bounded ghost list (last 256 evicted paths: token-prefix hash and length) supports `evicted`.
  Classification precedence: `model` (identity changed) > `disabled` > `capacity` (prefix existed
  but could not be kept for this request) > `evicted` > `template` (message-text hashes equal but
  tokens differ) > `volatile` (divergence inside a known volatile pattern, matched on decoded text
  around the divergence) > `edit` > `new`. Excerpts (±40 chars) only with `--debug-cache`, printed
  on the server terminal.
- **Rationale**: Analysis C3; "Don't Break the Cache" (arXiv 2601.06007).

## R6: Client normalization — deferred

- **Decision**: Not built in this feature. `mrun connect` sets `CLAUDE_CODE_ATTRIBUTION_HEADER=0`
  and the attribution block is stable per conversation since Claude Code 2.1.181. Diagnostics will
  show whether residual volatile misses justify an opt-in normalization feature later.

## R7: Numerical oracle

- **Decision**: Structural staleness on `ScriptedEngine` (100+ cases). On macOS, 30 real-weight
  cases: teacher-forced log-probability of the reference next token within 1e-2 of cache-off at
  every position, and identical first 32 greedy tokens for ≥ 95% of cases.
- **Rationale**: MLX selects kernels by shape, so recomputation differs in the last bits
  ([mlx-rust §5](../../concepts/research/mlx-rust.md)); analysis B3.

