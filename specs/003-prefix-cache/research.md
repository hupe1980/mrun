# Research: Prefix Cache

Sources: [ARCHITECTURE](../../concepts/ARCHITECTURE.md) (Prefix cache; Engine thread and memory),
[DECISIONS](../../concepts/DECISIONS.md) D004, D017–D019, [competitors-2026-10](../../concepts/research/competitors-2026-10.md),
[arxiv-sota-2026-10](../../concepts/research/arxiv-sota-2026-10.md), [literature-study](../../concepts/research/literature-study.md).

## R1: Structure: live path plus moved branches

- **Decision**: One token radix tree per cache key. The live path is the sequence state itself;
  every other stored conversation is a branch hanging off it, owning segments moved out of the
  sequence state. Switching moves state; it never copies it twice.
- **Rationale**: Contiguous per-layer arrays keep MLX's fused attention fast (D019) but cannot share
  pages, so the only way to keep one copy per unique token is to move tails. Copy-based restore
  (mlx-lm copies the whole KV on every hit, `models/cache.py:1655`) holds matched state twice and
  caps a surviving conversation at about half of capacity: on Qwen3-Coder (~105K tokens) an 80K
  parent could not survive a subagent.
- **Alternatives**: Paged radix tree (SGLang; needs paged kernels, Principle V); hash-chained blocks
  (oMLX, vLLM APC; still needs token confirmation, CVE-2025-25183); snapshot on replacement
  (previous design; doubles memory during restore).

## R2: Moves and segments

- **Decision**: Moving out = `split_off(d)`: per layer, copy the tail into a new buffer and free
  that layer's slack. Moving in = `append(segment, consume = true)` per node: per layer, copy into
  the live arrays and drop the source layer. Segments are refcounted views of per-layer buffers, so
  splitting an edge (a branch forking inside a moved-out tail) is metadata only. A buffer is always
  moved in whole: when the match ends inside a buffer, all of it is appended and the unmatched part
  is split out again (or discarded under 256 tokens), so a partial move never leaves the matched
  part held twice. Tails under 256 tokens are discarded: re-prefilling them costs well under a
  second, and they would add tree churn. At commit, output that retraces a stored branch is merged
  (the branch's overlap view is dropped).
- **Rationale**: Peak stays at one layer above steady state (SC-004). Cost is bandwidth, never
  memory: a 90K-token Qwen3-Coder tail is 90K × 96 KiB ≈ 8.2 GiB read and written, ≥ 65 ms at the
  M4 Pro's 273 GB/s, so a switch round trip (out + in) takes ≈ 0.15–0.2 s. In a switch the
  `split_off` is followed by an `append`, so it skips the prefix shrink (`shrink = false`, 001).
- **Alternative rejected**: moving only the matched views while the remainder keeps the shared
  buffer: on an edit at 60K into a 65K-token branch it would hold ~45K tokens (≈ 4 GB) twice.

## R3: Hybrid checkpoints

- **Decision**: Planned positions: end of system+tools, start of the last user or tool message,
  and `generation_start` (the start of the generation header, where the next turn's re-rendered
  reply diverges). A moved-out tail's end state stays inside its 001 segment and is restored by
  `append`; it is not a checkpoint id and does not count toward the cap. At most 8 per
  root-to-leaf path; the first, the two deepest and the newest checkpoint at a user (non-tool)
  message start are kept, and among the rest the one nearest the path's midpoint is dropped first
  (a long tool loop thus cannot push out the resume point of the next user turn). Resume restores
  the deepest checkpoint ≤ match and re-prefills the gap; attention KV for the gap is trimmed and
  recomputed (bit-identical with the same chunk split). Per prefill call at most 3 are planned
  (001 `MAX_PLAN_CHECKPOINTS`); stored ones beyond the live + 3 are evictable cache.
- **Rationale**: Competitors checkpoint just before the generation tail (Ollama `len-4`, LM Studio
  `total-11`, mlx-lm's think tail) and cap states per path (SGLang `mamba.rs`). Exact options for
  hybrids are checkpoints plus recomputation only; Tail-Replay, DASC, LinearKV, HYPIC and int8 GDN
  sidecars are approximate (D004). A checkpoint is ≈ 61 MiB on Qwen3.6 (001 R4), so 8 per path
  bound a conversation at ≈ 0.5 GiB of checkpoints.
- **Later**: Sparse Prefix Caching (2605.05219) places checkpoints by dynamic programming over
  observed overlap depths; the miss reports collect those depths.

## R4: Prefill plan

- **Decision**: From the resume position s, the next chunk end is the first checkpoint position in
  `(s, s + chunk(s)]`, else the last message boundary in `[s + 256, s + chunk(s)]`, else
  `s + chunk(s)`, where `chunk(depth) = clamp(2048 × 16K / depth, 128, 2048)` (001). The plan is a
  pure function of tokens, boundaries, checkpoints and s; the 001 `PrefillPlan.boundaries` it
  produces holds only these selected chunk ends.
- **Rationale**: Chunk ends at message boundaries cost nothing for attention models and are required
  for hybrid checkpoints (llama.cpp span-aware chunking; Ollama's fixed 8K grid misses). A pure plan
  makes cache-on vs cache-off comparisons reproducible (R9).

## R5: Admission and eviction timing

- **Decision**: (1) lookup; (2) 001's static admission (prompt + clamped output ≤ capacity); (3)
  evict for the initial allocation only: attention growth to `P + min(out, 4K)` beyond
  `allocated()` after the switch, plus planned checkpoints (the one-layer move transient lies in
  001's scratch envelope); (4) switch with the matched nodes pinned; unpin; (5) on
  `Err(NeedsMemory{bytes})` from any engine call the worker calls `evict(bytes)` and retries the
  call. Queued requests' matched branches are soft-pinned (evicted last); the worker applies
  pending soft pins between requests and at each prefill yield.
- **Rationale**: Static admission bounds the request; lazy allocation lets branches use capacity
  reserved for output until the output needs it. Since only the live path's state is unevictable and
  it fits by admission, growth-time eviction cannot fail (Constitution III).

## R6: Eviction policy

- **Decision**: LRU over eviction units. A unit is a branch leaf plus its chain of single-child
  ancestors that are views of the same buffer; its bytes are the buffers it alone references. Units
  freeing nothing are skipped. Order: plain branch units, soft-pinned units, then live-path
  checkpoints (R3 drop order). Evicting increments the fork node's evicted counter (keyed by the
  unit's first token, with the reason) and calls MLX `clear_cache`.
- **Rationale**: "When Fancy Eviction Fails" (2609.28870) and "Where Should the KV Cache Live?"
  (2609.16215) find LRU at the offline optimum for single long sessions (D017).

## R7: Memory pressure

- **Decision**: A macOS `DISPATCH_SOURCE_TYPE_MEMORYPRESSURE` source sets an atomic level; the worker
  reads it at each per-layer check point and between decode tokens, and when idle. During prefill
  a set level requests a yield, so it is acted on at the next whole-chunk end
  (`Prefilled.yielded`, 001); the worker evicts, then continues with the remaining tokens and the
  same plan, so the chunk split is unchanged. Warning: evict branch units until branch bytes are
  halved. Critical: evict all branches and non-live checkpoints;
  if idle, reset the live path. Then `clear_cache`. Linux: no-op source, injectable in tests.
- **Rationale**: The per-layer check point already exists for cancellation, and no second thread
  touches MLX. Acting only at a whole-chunk end keeps every committed chunk and the exact split;
  the response time is bounded by one chunk's duration, which the depth-adaptive chunk length
  keeps roughly constant; the 1 s target is checked by the injected-pressure test (T019).

## R8: Miss classification

- **Decision**: Nodes record message marks `{start, content_hash}` from 002's renderer; content hash
  covers role, visible text and tool calls (name, arguments as a canonical JSON value) and excludes
  reasoning. Precedence: `disabled` > `model` > divergence causes > `checkpoint` > none. For a
  divergence at d against the closest stored path: `evicted` / `capacity` if the request continues
  into a removed child (evicted counter: LRU, clear or tiny-tail discard → `evicted`; memory
  pressure → `capacity`); `template` if every message up to the one containing d has an equal
  content hash; `system` if the divergence lies inside the stored path's system+tools region;
  `edit` if at least one conversation message matched before it (a message changed, removed or
  replaced, including an edited last turn); otherwise `new`. `checkpoint` when there is no divergence cause but resume < match.
- **Rationale**: "Don't Break the Cache" (2601.06007) and oMLX's divergence stats show positions
  alone do not tell users what to fix. Content hashes catch the common template case (Qwen3-Coder
  and Qwen3.6 re-serialize tool arguments, 002 FR-004). The Claude Code attribution nonce is fixed
  in 002 (D021).

## R9: Exactness oracle

- **Decision**: Structural staleness on `ScriptedEngine` (≥ 100 cases). On macOS, cache-on vs
  cache-off with the cache-off run forced to the same chunk split: identical token ids and logits.
- **Rationale**: Prefix-then-suffix prefill differs from one-pass prefill in bf16 rounding (risk
  R04); fixing the split removes that and turns a tolerance test into an equality test.
