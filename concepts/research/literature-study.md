# Literature and Systems Study: Making the Agent Runtime Better

**Feature**: `001-agent-runtime` | **Date**: 2026-10-14 | **Status**: Research input, not a decision.
**Authority**: Written for the superseded 001-agent-runtime spec (2026-10-14); the constitution and current specs govern. Nothing here amends an FR/SC, the
Assumptions' exclusions, or the approved Principle V exception. All benefit figures are the cited
authors' own claims under their own conditions (mostly datacenter GPUs, batched serving); none is
an mrun measurement or a prediction for the M4 Pro.

## 1. Scope and Method

- **Question** (maintainer): "research also arxiv and co — can we make it even better?" Topics are
  the section 2 headings.
- **Method**: Web searches against arXiv listings, conference proceedings (USENIX OSDI/ATC/NSDI,
  MLSys, ICML, NeurIPS, ACL, EuroSys), vendor advisories and upstream repositories, on 2026-10-14.
  Direct fetches of arxiv.org were blocked by this environment's egress policy, so titles, IDs,
  venues and headline numbers were confirmed through search-engine extracts of the primary pages
  (abstract/HTML/proceedings), not by reading full PDFs. Tables and conditions were not re-checked.
- **Evidence marks**: **[V]** ID, title and quoted claim confirmed from a primary listing/abstract in
  this session; **[S]** only secondary sources (blogs, aggregators, vendor docs) found; **[M]** from
  prior knowledge, not re-verified here; **[I]** this study's own inference or arithmetic.
- **mrun context used**: Candle-owned runtime, Qwen3-8B Q4_K_M candidate, 16,384-token context,
  F32 KV as the starting sizing route, one foreground generation, 128-token prefill chunks,
  copied frozen snapshots (at most four), llguidance as the candidate grammar engine (research C-02..C-05).

## 2. Findings by Topic

Each entry: mechanism / reported benefit (author conditions) / exactness / M4 Pro + Candle fit / mrun ids.

### 2.1 Exact prefix and KV reuse

- **RadixAttention / SGLang** — arXiv 2312.07104, NeurIPS 2024 [V]. LRU cache of KV for all
  requests in a radix tree keyed by token sequences; longest-prefix match at request start.
  Benefit: "up to 6.4x higher throughput" across agent, few-shot, JSON, multi-turn workloads on GPUs
  versus contemporary engines (batched; mixes several optimizations). Exact: yes for token-identical
  prefixes. Fit: high for the index; mrun does not need paged kernels for one stream (see 3A-1).
  Maps: FR-006, FR-008, FR-016, SC-002, SC-008.
- **PagedAttention / vLLM APC** — arXiv 2309.06180, SOSP 2023 [V]. Block-table KV ("virtual
  memory" pages) enabling sharing and near-zero fragmentation; automatic prefix caching hashes
  full blocks. Benefit: "2-4x throughput at the same latency" vs FasterTransformer/Orca (batched).
  Exact: yes, if lookup is validated. Caution: **CVE-2025-25183** [V] — prefix-cache keys built on
  Python `hash()` allowed engineered collisions to reuse another prompt's KV (fixed in vLLM 0.7.2).
  This directly supports FR-007's "hash lookup alone MUST NOT replace exact validation". Fit: block
  tables need attention kernels that gather pages (new kernels: excluded); the bookkeeping ideas
  (refcounts, copy-on-write, block-granular accounting) transfer. Maps: FR-007, FR-009, FR-014, SC-003.
- **Prompt Cache** — arXiv 2311.04934, MLSys 2024 [V]. Reuses states of schema-declared "prompt
  modules" even when not a literal prefix; TTFT 8x (GPU) to 60x (CPU). Exact: **no** (modules lack
  preceding context). **CacheBlend** — arXiv 2405.16444, EuroSys 2025 [V/S]: concatenates non-prefix
  chunk KV, recomputes ~15% of tokens; 2.2-3.3x lower TTFT, F1/Rouge-L within 0.02 (RAG). Exact:
  **no**. Both are excluded (approximate prefix matching; FR-006/FR-008).
- **llama.cpp `--cache-reuse` (KV shifting)** [S]. Shifts cached chunks to new positions after a
  middle edit (min chunk `n_cache_reuse`, default reported as 0/off). Exact: **no** (shifted chunks
  never attended to the new middle). Plain `cache_prompt` slot reuse of the common prefix is exact.
- **ChunkAttention** — arXiv 2402.15220, ACL 2024 [V]. Prefix tree of KV chunks + two-phase decode
  kernel; 3.2-4.8x attention-kernel speedup (shared 1-4K system prompts, many sequences). Exact. The
  chunked prefix-tree storage transfers (3A-1); the kernel needs batching. Maps: FR-016, SC-005.
- **Hydragen** — arXiv 2402.05099 [V]. Exact shared-prefix/suffix attention decomposition batching
  queries over the prefix; up to 32x throughput (CodeLlama-13b, large batch). Fit: batching only.
- **Marconi** — arXiv 2411.19379, MLSys 2025 [V]. Prefix caching for hybrid attention/SSM models
  whose recurrent state cannot be rolled back; admission/eviction scores reuse likelihood and FLOPs
  saved per byte, not recency. Benefit: up to 34.4x token hit rate vs prior prefix caches. Exact:
  yes (exact-match only). Fit: hybrids are unsupported now (spec edge cases); the **FLOP-per-byte
  eviction score** transfers to pure attention (3A-3). Maps: FR-016, SC-002, SC-005.
- **CachedAttention/AttentionStore** — arXiv 2403.19708, USENIX ATC 2024 [V; venue M]. Saves
  multi-turn KV to host/disk tiers with layer-wise preload. Benefit: TTFT up to 87-88% lower (version
  dependent). **Pensieve** — arXiv 2312.05516, EuroSys 2025 [V]. GPU/CPU tiered conversation state,
  evict by inactivity and size, recompute dropped parts. Benefit: 1.14-3.0x throughput vs vLLM/TRT-LLM.
  Both exact. Fit: on unified memory, "CPU tier" is the same DRAM, so only an SSD tier adds capacity,
  and that is durable persistence (excluded; SC-009 forbids persisted conversation state). Maps: 3B-3.
- Comparators (FR-022): **mlx-lm server** in-memory `LRUPromptCache`, disk PR #1405 closed unmerged
  [S]; **vllm-mlx** arXiv 2601.19139 [V], MLX prefix caching, 21-87% over llama.cpp on M4 Max.

### 2.2 Agent and tool-loop serving

- **InferCept** — arXiv 2402.01869, ICML 2024 [V]. When generation is intercepted for a tool,
  choose per request among discard (recompute), preserve (keep KV) and swap, minimizing wasted
  memory-time. Reports recomputation after interception is 37-40% of forward time in their setting;
  1.6-2x throughput vs SOTA. Exact: yes. Fit: mrun's "retain the tool-call boundary" is the preserve
  arm; with one foreground slot the cost model is simpler (memory, not GPU-time contention).
  Maps: FR-016, SC-008.
- **Continuum** — arXiv 2511.02230 (ICLR 2026 listing) [V]. Pins a finished tool-calling request's
  KV for a time-to-live (later versions: derived from reload cost and ordering benefit) plus
  program-level FCFS. Evaluated on SWE-Bench and BFCL with Llama-3.1 8B/70B; the "8.18x" figure seen
  is from a third-party summary [S]. Exact: yes. Fit: TTL-based soft pinning maps onto FR-016's
  retained sub-budget. Maps: FR-016, SC-008.
- **KVFlow** — arXiv 2507.07400, NeurIPS 2025 [V]. Agent-step graph predicts which agent runs next;
  eviction by "steps to next use", plus CPU-to-GPU prefetch. Up to 1.83x (single workflow) and 2.19x
  (concurrent) vs SGLang hierarchical radix cache. Exact: yes. Fit: prefetch is moot on unified
  memory; the "next-use" idea is relevant only if OpenCode subagents alternate prefixes.
- **Parrot** arXiv 2405.19888, OSDI 2024 (semantic variables expose app dataflow; needs API/client
  changes) [V]; **Autellix** arXiv 2502.13965 (program-level scheduling, 4-15x program throughput vs
  vLLM; multi-program) [V]; **Conveyor** arXiv 2406.00059 (tool execution during decoding, up to
  38.8% latency; tools are client-owned per FR-005) [V]. Low fit at concurrency one.
- **Workload evidence**: **TraceLab** — arXiv 2606.30560 [V/S]: ~4.3K Claude Code/Codex sessions;
  long contexts, short outputs (reported median ~120K-token prefix vs ~200 output tokens [S]), 95.7%
  cached input overall but ~84.4% on user-initiated rounds because caches expire during human
  pauses; authors caution about generalization. GitHub Copilot trace study (alphaXiv 2608.00101,
  Microsoft) [S]: ~90% hit rate within a turn, ~55% across turn boundaries, ~8% after model switch.
  **"Don't Break the Cache"** — arXiv 2601.06007 [V]: timestamps, session IDs and tool-set changes
  early in the prompt break provider caches; naive full-context caching can even raise latency.
  **TokTier** — arXiv 2607.29678 [V/S]: exact incremental re-tokenization; claims tokenization
  grows to a large share of TTFT at very high hit rates and long (100K+ char) contexts.
  Implication [I]: for mrun the dominant risks are human-pause eviction, early volatile prompt
  content, and template re-rendering differences, not lookup speed. Maps: SC-002, SC-008, FR-019.

### 2.3 KV compression, quantization and eviction (all approximate)

- **KIVI** — arXiv 2402.02750, ICML 2024 [V]: 2-bit KV (keys per-channel, values per-token), "almost
  the same quality", 2.6x less peak memory. **KVQuant** — arXiv 2401.18079, NeurIPS 2024 [V]: pre-RoPE
  per-channel keys, non-uniform types, outliers; <0.1 perplexity loss at 3 bits; custom CUDA kernels.
- **H2O** — arXiv 2306.14048, NeurIPS 2023 [M]; **SnapKV** — arXiv 2404.14469 [V] (92% compression
  at 1024 KV tokens on LongBench, "comparable" quality); **StreamingLLM** — arXiv 2309.17453,
  ICLR 2024 [M] (attention sinks + recent window).
- Exactness: **all approximate**. Eviction methods (H2O, SnapKV, StreamingLLM) additionally make the
  retained KV depend on the query/suffix (SnapKV) or drop middle tokens, so a "prefix snapshot" no
  longer equals the exact state for that token prefix; they are fundamentally incompatible with
  FR-006/FR-008 and silent-truncation rules (FR-016). Quantization is compatible only as a separately
  identified, qualified, reversible precision route (FR-007 "cache precision", Principle II).
- Fit [I]: Metal SDPA paths generally take fp16/bf16/fp32 only (an MLX issue states this for MLX
  [S]); quantized KV would need dequantize-on-read (extra bandwidth/scratch) or new kernels.

### 2.4 Apple Silicon and unified-memory inference

- **Hardware** [S]: M4 Pro memory bandwidth 273 GB/s (256-bit LPDDR5X-8533). Single-stream decode
  is memory-bandwidth bound: each token reads all weights plus the live KV.
- **Arithmetic [I]** (Qwen3-8B: 36 layers, 8 KV heads, head dim 128 [M]): KV is
  2 x 36 x 8 x 128 = 73,728 values/token = **288 KiB/token in F32**, 144 KiB in F16/BF16. At 16,384
  tokens: 4.5 GiB (F32) vs 2.25 GiB (F16). Weights ~5 GB (Q4_K_M, approximate). Decode at a full
  16K context in F32 therefore reads roughly 1.9x the bytes of the weights alone; an upper-bound
  ceiling is ~273/(5+4.8) = ~28 tok/s (F32) vs ~37 tok/s (F16) before any overhead. Four retained
  full-context F32 snapshots would be 18 GiB of a ~30 GiB budget. KV precision is therefore the
  largest single memory and long-context decode lever available without new kernels.
- **Runtime comparisons**: arXiv 2511.05502 [V] (MLX, MLC-LLM, llama.cpp, Ollama, PyTorch MPS on
  M2 Ultra; MLX highest throughput, MLC-LLM better TTFT). **BaseRT** — arXiv 2607.00501 [V]: native
  Metal runtime claims up to 1.56x decode vs llama.cpp and 1.35x vs MLX on M3/M4 Pro via chip-specific
  fused kernels (self-reported). Implication [I]: Candle's stock Metal operators may leave decode
  throughput on the table versus competitors; SC-006 is same-core so unaffected, but FR-022
  competitor comparisons may look unfavorable. Closing that gap means new kernels (excluded).
- Privileged tuning such as raising the GPU wired-memory limit via sysctl is a common community
  recommendation for MLX/llama.cpp [S]; FR-015 forbids it.

### 2.5 Speculative decoding (excluded from this feature)

- Foundations: Leviathan et al. arXiv 2211.17192; Chen et al. arXiv 2302.01318 [M]. Rejection
  sampling preserves the target distribution; greedy verification preserves greedy output **in exact
  arithmetic**. In floating point, verifying k tokens per forward changes kernel shapes, so token IDs
  can differ from one-token decode (see 2.7) [I].
- **Prompt lookup decoding** (Saxena 2023, GitHub, no paper) [S] and **LLMA** arXiv 2304.04487 [M]:
  draft by n-gram match against the prompt. A 2026 study reports n-gram speculation is strongest on
  code editing (local repetition) and weak when overlap is low [S, arXiv 2601.11580].
- **SuffixDecoding** — arXiv 2411.04975, NeurIPS 2025 spotlight [V]: suffix trees over prompts and
  prior outputs, adaptive draft length; evaluated on OpenHands/SWE-Bench; up to 5.3x (arXiv v3) or
  3.9x (NeurIPS version). Model-free, so no extra weights.
- **Medusa** arXiv 2401.10774 [V], **EAGLE** 2401.15077 [M], **EAGLE-3** 2503.01840 [V]: trained
  draft heads; speedup figures seen only in secondary sources [S]. Require per-model head training or
  a trusted published head; more weights and memory.
- **Lookahead decoding** — arXiv 2402.02057, ICML 2024 [V]: Jacobi-style n-gram generation and
  verification; up to 1.8x on MT-bench, up to 4x on code completion with multi-GPU strong scaling.
- **"Speculative Decoding: Performance or Illusion?"** — arXiv 2601.11580, MLSys 2026 [V]: in vLLM,
  verification dominates cost, acceptance varies widely by position/request/dataset, speedups
  shrink with batch size and fall well short of theoretical bounds.
- Fit [I]: batch-1, bandwidth-bound decode on M4 Pro is the favorable regime; coding agents often
  re-emit file content (edits), favoring prompt-lookup/suffix methods. Rollback of rejected tokens
  uses the same exact-prefix truncate primitive mrun needs for FR-008. Maps: SC-006, SC-008 (future).

### 2.6 Constrained decoding / structured output

- **XGrammar** — arXiv 2411.15100 [V]: splits vocabulary into context-independent (precomputed) and
  context-dependent tokens, persistent stack, overlaps mask computation with GPU work; "up to 100x"
  vs prior engines. **XGrammar 2** — arXiv 2601.04426 [V]: dynamic structures for agentic tool
  calling, >6x vs existing engines.
- **llguidance** (GitHub, guidance-ai) [S]: Earley parser over a derivative-based lexer, token-trie
  traversal; project claims ~50 us/token for 128K vocab JSON schemas; independent 2026 papers measure
  73-141 us on Qwen3-8B enums (arXiv 2608.12574 [S]) and 0.8-1.4 ms on SQL/Java grammars (arXiv
  2608.03065 [S]). Near-zero grammar compile cost (lazy).
- **Outlines** arXiv 2307.09702 [M] (FSM index); **SynCode** arXiv 2403.01632, TMLR 2025 [V] (DFA mask store).
- Exactness: masking is exact with respect to the grammar; it changes the sampling distribution by
  design and guarantees structure only (research C-05 already notes this). Fit: CPU mask work can
  overlap GPU forward on unified memory without copies [I]. Maps: FR-005, SC-001, SC-007.

### 2.7 Prefill/decode scheduling, chunking, determinism

- **Sarathi / Sarathi-Serve** — arXiv 2308.16369 [M] / 2403.02310, OSDI 2024 [V]: equal-size prefill
  chunks co-scheduled with decodes ("stall-free"); 2.6x (Mistral-7B, 1xA100) to 5.6x serving
  capacity under tail-latency SLOs. Fit: the throughput result needs concurrent decodes; for one
  foreground request only the **bounded-chunk latency** property transfers (cancellation, SC-004).
- **Determinism** — Thinking Machines, "Defeating Nondeterminism in LLM Inference" (blog, Sep 2025)
  [V title/S content]: logits depend on batch/reduction split; batch-invariant kernels fix it at a
  cost. vLLM's batch-invariance guide lists chunked prefill, prefix caching and preemption as
  unsupported in that mode [S]. Implication [I]: a warm path whose suffix starts at an arbitrary
  token boundary computes different matmul/attention shapes than a cold path chunked from 0, so
  **bitwise-identical token IDs (SC-003 positive replay) are not guaranteed** unless chunk
  boundaries are aligned or the reference is defined as the same chunk schedule (3A-2).

### 2.8 Cancellation, preemption and memory admission

- **FastServe** — arXiv 2305.05920 (NSDI 2026) [V]: token-level preemption, skip-join MLFQ; JCT up
  to 5.1x/6.4x vs Orca. Fit: FR-011 is FIFO with one slot; only per-step cancellation points transfer.
- **vLLM preemption** (PagedAttention paper) [M]: optimistic admission, then swap/recompute
  preemption. Fit [I]: contradicts FR-014 reserve-before-execute; worst-case reservation is right here.
- Metal exposes no kernel-level abort (research D-04 [M, existing]); every system surveyed cancels at
  step/chunk boundaries, consistent with FR-012.

## 3. "Make It Better" Recommendations

Format: effect / correctness-gate risk (Principle II) / evidence needed / informs.

### Tier A — Within current scope and constitution (exact, no new kernels)

- **A-1 Segmented radix prefix index with copy-on-restore.** Index frozen snapshots in a radix tree
  over token IDs (plus FR-007 identity at the root), storing KV in fixed segments (e.g. 128-256
  tokens per layer) so branches share common segments instead of holding up to four full copies.
  Restore = copy matched segments into the request's contiguous active KV (no paged kernel needed).
  Copy cost [I]: ~4.5 GiB F32 at a plausible ~100-200 GB/s effective copy rate is tens of ms, small
  next to an 8K-token prefill, but unmeasured. Effect: more distinct branches per retained byte,
  higher hit rate under FR-016's budget. Risk: low if tokens/identity are compared after lookup
  (FR-007, CVE-2025-25183 lesson) and segments are immutable. Evidence: restore-time and byte
  benchmarks, branch-isolation tests. Informs: research C-02, FR-016 budget, SC-002, SC-003, SC-005.
- **A-2 Chunk-grid-aligned replay for bitwise exactness.** Prefill always on an absolute-position
  grid (multiples of the 128-token chunk). On a hit at position m, restore to floor(m/128)*128 and
  replay the remainder as counted boundary replay (at most 127 tokens). Warm and cold paths then run
  identical operator shapes, making SC-003's identical-token-ID gate attainable without batch-invariant
  kernels. Effect: removes a likely source of spurious SC-003 failures; costs <=127 tokens of prefill.
  Risk: none to correctness; slight SC-002 cost. Evidence: cold vs warm token-ID equality across the
  100 mutation cases with and without alignment. Informs: SC-003, FR-008, FR-019 replay counts.
- **A-3 Cost-aware eviction instead of pure LRU.** Score retained entries by expected saved prefill
  time per byte (Marconi) with recency and a soft TTL protecting the most recent tool-call boundary
  (Continuum/InferCept "preserve"). The single active conversation's tail is the most valuable entry
  during human pauses (TraceLab, Copilot traces). Effect: higher SC-008 benefit at equal budget.
  Risk: none to exactness; must never pin beyond FR-016 pressure rules. Evidence: replay of frozen
  ten-turn loops with pause gaps; hit-rate vs budget curves. Informs: FR-016, SC-008, data model.
- **A-4 Publish the tool-call boundary snapshot.** At a tool-call finish, snapshot prompt + generated
  tokens. Whether the next turn reuses generated tokens depends on the template re-rendering the
  client's echoed call byte-identically (JSON spacing, argument order, stripped thinking).
  Effect: potentially avoids re-prefilling every assistant turn. Risk: none if exact match decides.
  Evidence: measure the first-divergence position between generated and re-rendered tokens on the
  frozen OpenCode traces before building anything. Informs: SC-008, FR-008, US2.
- **A-5 Privacy-safe divergence diagnostics.** Record only the first-divergent token index, prompt
  length and a coarse reason class (identity, template, content) per request. This detects
  "cache-breaking" volatile content (dates, env listings) that the 2601.06007 study found common,
  without logging prompts. Effect: makes SC-002/SC-008 misses actionable. Risk: none; FR-021-safe if
  no token IDs/content are logged. Informs: FR-019, FR-021, SC-009.
- **A-6 F16/BF16 KV as a qualified precision route (not quantization).** Halves KV bytes and
  long-context decode KV traffic (2.4 arithmetic). It is a precision change, so it needs its own
  identity (FR-007) and predeclared numerical/task tolerances (SC-003 precision-dependent routes,
  SC-007); F32 stays the reference and the switch is reversible. Feasibility in stock Candle
  quantized Qwen3 is unverified. Evidence: logit-difference and task-parity runs; memory/decode
  measurements. Informs: plan's open KV dtype decision, FR-014, SC-005, SC-006, SC-007.
- **A-7 Partial-prefill salvage on cancellation.** When a request is cancelled mid-prefill,
  completed whole chunks form a valid exact prefix state; publish it (subject to budget) instead of
  discarding. Users often cancel and resend an edited prompt sharing that prefix. Risk: must be
  synchronized and only published after GPU completion (FR-012, FR-023). Evidence: cancel-then-resend
  traces. Informs: SC-004, SC-002, US3.
- **A-8 Grammar engine hygiene.** Activate tool-call grammars lazily at the tool-call start marker,
  cache compiled grammars keyed by the exact tool-schema set, and compute masks on CPU overlapped
  with the GPU forward (XGrammar technique). Exact by construction. Evidence: per-token mask time
  on the frozen schemas vs decode step time. Informs: research C-05, FR-005, SC-006.
- **A-9 Measure, then freeze, chunk size** (64/128/256/512: prefill rate vs cancel latency; SC-004,
  SC-006) and **tokenization cost** (TokTier-style exact append tokenization only if FR-019 shows
  render/tokenize matters at 16K, which is unlikely [I]).

### Tier B — Candidate follow-up features (spec change required)

- **B-1 Prompt-lookup / suffix speculative decoding** (prompt lookup, SuffixDecoding). Model-free,
  best fit for batch-1 coding edits. Expected: meaningful decode gains on edit-heavy turns, little on
  free text (authors' conditions differ; no M4 Pro data). Risk: must be qualified as non-bitwise-
  identical unless verification is shape-aligned; disable switch required (Principle II). Evidence:
  acceptance-length distribution on frozen OpenCode traces before implementation. Informs: SC-006,
  SC-008. Lifts exclusion: speculative decoding.
- **B-2 KV quantization (Q8 first, then KIVI-style 4/2-bit)** with explicit identity, tolerance
  gates and off-switch. Needs dequantize path or new Metal kernels; check memory/bandwidth net win.
  Informs: FR-007, SC-003, SC-005, SC-007. Lifts exclusion: new kernels (likely).
- **B-3 Opt-in durable prefix persistence** (AttentionStore, Pensieve, LMCache patterns) to SSD for
  restart/idle-unload recovery; encrypted, principal-scoped, clearable. Conflicts with FR-010 and
  SC-009 today; needs privacy design. Informs: FR-010, FR-017, SC-009.
- **B-4 Small-batch continuous batching** for parallel subagents, with shared-prefix attention
  (Hydragen/ChunkAttention) later. Only if traces show concurrent requests matter. Informs: FR-011.
- **B-5** Hybrid/sliding-window models with Marconi-style state checkpoints; **B-6** EAGLE-3/Medusa
  trained-head speculation only if B-1 acceptance proves insufficient.

### Tier C — Rejected or not fitting now

- **H2O, SnapKV, StreamingLLM, PyramidKV-type eviction**: approximate and suffix-dependent; break
  exact prefix semantics and amount to silent context truncation (FR-006, FR-008, FR-016).
- **Prompt Cache, CacheBlend, llama.cpp KV shifting**: non-prefix/approximate reuse; explicitly
  excluded ("approximate prefix matching"). Revisit only as a qualified opt-in mode with quality gates.
- **PagedAttention-style block tables and custom attention kernels, BaseRT-style fused kernels**:
  require new Metal kernels (excluded); A-1 gets the memory benefit for one stream without them.
- **Optimistic admission with swap/recompute preemption** (vLLM, FastServe): conflicts with FR-014
  reserve-before-execute and SC-005; no gain at concurrency one.
- **Disaggregated prefill/decode** (DistServe arXiv 2401.09670, Splitwise 2311.18677 [M]): multi-device only.
- **Parrot semantic variables, Conveyor partial tool execution**: require client or tool changes
  (FR-002, FR-005).
- **Privileged GPU wired-memory tuning**: forbidden by FR-015.

## 4. Open Questions for the Maintainer

1. Is a F16/BF16 KV route acceptable as a separately identified, qualified precision route (A-6),
   or must F32 remain the only first-feature route? This drives FR-016's retained byte budget.
2. May SC-003's positive exact-replay reference be achieved via chunk-aligned boundary replay (A-2),
   counting up to 127 replayed tokens, or must arbitrary-boundary warm paths match cold token IDs?
3. Can frozen interoperability traces include locally captured rendered token streams of the pinned
   OpenCode version (never shipped/logged) to audit volatile early content and tool-call re-render
   divergence (A-4, A-5)? This is the cheapest high-value experiment.
4. Is the "at most four snapshots" limit meant as entries or as bytes? With segmentation (A-1), an
   entry count stops being the right unit.
5. Do OpenCode subagents or parallel sessions occur in your real use? That decides whether B-4
   (batching) or B-1 (speculation) is the better first follow-up.
6. Which follow-up should the next spec target first: B-1 speculation, B-2 KV quantization, or
   B-3 persistence? B-1 has the strongest workload fit; B-3 has the largest privacy surface.
7. Is a hybrid (attention + SSM/linear) coding model likely to become the primary candidate? If so,
   Marconi-style state checkpointing (B-5) should be planned before the cache design ossifies.

Unverified in this session (treat accordingly): all [M] and [S] items; the TraceLab median sizes and
Copilot-trace percentages; EAGLE-3/Medusa speedups; Qwen3-8B geometry and the weight-size figure
used in 2.4; whether stock Candle supports F16 KV for the quantized Qwen3 path; Metal copy rates.
