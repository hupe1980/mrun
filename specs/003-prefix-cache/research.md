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

## R2: Segment boundaries

- **Decision**: Snapshot at boundaries the renderer reports (end of system+tools block, end of each
  user/tool message, end of each assistant message) plus every 4,096 tokens; never inside the last
  4 tokens of a prompt.
- **Rationale**: Agent edits and branches happen at message boundaries (literature A-3/A-4;
  mlx-lm checkpoints system/user/assistant ends); the 4K cap bounds wasted reprocessing when an
  edit lands mid-message.

## R3: Restore vs hot path

- **Decision**: If the active cache's tokens are a prefix of the request, keep it (zero copy) and
  append. Otherwise restore matched segments into fresh buffers (copy), then trim to the exact
  divergence and prefill the remainder (always ≥ 1 token for next-token logits).
- **Rationale**: MLX donation semantics: shared handles make appends copy whole buffers
  ([mlx-rust §7.3](../../concepts/research/mlx-rust.md)). Copy cost at 96 KiB/token × 30K ≈ 2.8 GiB
  at ~200 GB/s ≈ 15 ms, vs seconds of prefill.

## R4: Eviction

- **Decision**: Score = estimated recompute seconds (tokens / measured prefill tok/s at that depth)
  ÷ bytes × exp(−idle / 10 min); evict lowest score leaf-first; nodes with `in_use > 0` are pinned.
  Admission evicts until the request's reservation fits; pressure warnings evict down to 50% of the
  cache, critical to 0.
- **Rationale**: Marconi-style cost-aware eviction (literature A-3); recency decay keeps hot
  conversations.

## R5: Miss diagnostics

- **Decision**: On every lookup record (closest cached length L, divergence position d, request
  length n). Classify: `new` (L = 0, no shared root), `edit` (divergence inside an earlier message),
  `template` (same message texts, different tokens), `volatile` (divergence inside a known volatile
  pattern, e.g. Claude Code attribution header or timestamps), `evicted` (prefix was cached and
  evicted), `capacity` (could not keep), `model` (identity changed), `disabled`. Text excerpts
  (±40 chars) only with `--debug-cache`.
- **Rationale**: "Don't Break the Cache" (arXiv 2601.06007) and oMLX's experience: volatile client
  content is the main real-world miss cause; competitors don't explain misses.

## R6: Client normalization (opt-in)

- **Decision**: Per-client profiles listing known volatile segments (Claude Code billing/attribution
  header line) that may be removed before rendering when `--normalize claude-code` is set. Off by
  default; reported in responses and `mrun ps`.
- **Rationale**: Exactness principle: changing model input must be explicit. `mrun connect` already
  sets `CLAUDE_CODE_ATTRIBUTION_HEADER=0`, which fixes the main case without normalization.

## R7: Numerical oracle

- **Decision**: Structural staleness tests on `ScriptedEngine`; on macOS, compare cache-on vs
  cache-off next-token log-probs (|Δ| ≤ 1e-2) and greedy agreement (≥ 99%) on the mutation suite.
- **Rationale**: MLX kernel choice depends on shapes, so recomputation can differ in the last bits
  ([mlx-rust §5](../../concepts/research/mlx-rust.md)).
