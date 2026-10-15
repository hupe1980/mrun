# Next Features After M5: Research and Priorities

**Date**: 2026-10-15. **Status**: Research input for the "Later" list in [ROADMAP](../ROADMAP.md).
This is not a decision. **Scope**: speculative decoding, KV quantization, hybrid-attention models,
an SSD prefix tier, parallel sessions, and gaps the roadmap does not list. The target is the best
agent latency on an M4 Pro with 48 GB (273 GB/s), with Qwen3-Coder-30B-A3B 4-bit as the default model.

**Method**: Primary sources were read at these revisions. mlx-lm `ba8a08f` (2026-10-10, v0.32.0).
MLX `bebd642` (v0.32.3). mlx-c v0.7.0. Ollama `292d7ce` (`mlxrunner/`). llama.cpp `23deec0`. oMLX
`6feea9f`. vLLM `e73fa00`. anthropic-sdk-python `2a53eaa`. openai-python `d42accb`. arXiv facts come
from search-engine extracts of the listings, because huggingface.co and arxiv.org fetches are blocked
here. Nothing was built or run on a Mac.

**Markers**: **[V]** verified in source code or docs at the stated revision, or in the arXiv
listing. **[U]** unverified: community or vendor claims, or second-hand configs. **[I]** this
document's own arithmetic or inference.

**Reference arithmetic [I]**: Qwen3-Coder-30B-A3B has 48 layers × 4 KV heads × 128 dims, so its
KV costs 96 KiB per token in bf16. About 10.3 GiB of KV space gives ≈110K tokens. A decode step
reads about 1.9 GB of active weights (3.3B params at about 4.5 bits) plus the live KV. At 100K
tokens the KV is 9.8 GB, so **KV traffic dominates long-context decode**. The bandwidth ceiling is
then ≈23 tok/s, against ≈60+ tok/s at short context. The M4 Max reaches 1,754 tok/s prefill for
Qwen3-30B-A3B 4-bit at 2K tokens [V mlx-lm `BENCHMARKS.md`]. Half the GPU cores puts the M4 Pro at
≈850 tok/s on short prompts, so a cold 30K prompt takes ≈40–60 s [I].

## 1. Speculative decoding

**Mechanism.** A cheap drafter proposes k tokens. The target scores all k+1 positions in one
forward pass and keeps the longest agreeing prefix plus one token of its own. Under greedy decoding,
every emitted token is the target's argmax. Under sampling, rejection sampling preserves the
target's distribution (Leviathan arXiv 2211.17192; Chen arXiv 2302.01318). Rejected positions are
rolled back. For attention KV that is mrun's existing `trim`, which just moves the offset.

Drafter options: **n-gram / prompt lookup** (model-free; matches the last n tokens against the
prompt and output so far). **SuffixDecoding** (arXiv 2411.04975, NeurIPS 2025) [V]: suffix trees over
the prompt and past outputs, adaptive draft length, evaluated on agentic SWE-Bench/OpenHands traces,
up to 5.3× (v3). **MTP heads** shipped inside the checkpoint. **EAGLE-3** (arXiv 2503.01840) [V]: a
trained one-layer drafter reading the target's hidden states. **DFlash** (arXiv 2602.06036, ICML
2026) [V]: a block-diffusion drafter claiming more than 6× "lossless" speedup, 2.5× over EAGLE-3, on GPUs.

**Which Qwen models ship MTP.**
- **Qwen3-Next and Qwen3.5/3.6/3.8 ship MTP.** vLLM has `qwen3_next_mtp.py` and `qwen3_5_mtp.py`,
  and maps `qwen3_5`, `qwen3_5_moe` and their `_text` variants to MTP [V vLLM
  `config/speculative.py`]. Ollama's MLX `qwen3_5` model loads an `MTPHead` "only when its tensors
  are present" [V `mlxrunner/model/qwen3_5/qwen3_5.go:956`]. Nemotron-H also self-drafts [V].
  Qwen3.6-27B declares `mtp_num_hidden_layers: 1` [U: Ollama mirror of config].
- **Qwen3-Coder-30B-A3B (`qwen3_moe`) has no MTP.** There is no `qwen3_moe` MTP class in vLLM [V by
  absence], and no published EAGLE-3 head for it was found. AWS ships P-EAGLE heads [U].
- **mlx-lm strips MTP weights.** `sanitize()` drops `mtp.*` in `qwen3_5.py:354` and
  `qwen3_next.py:489`, and `convert` goes through `load` → `sanitize` [V `utils.py:493`]. So
  **mlx-community conversions made with mlx-lm lack MTP tensors** [I]. Using MTP means mrun converts
  the weights itself or fetches `mtp.*` separately. Community checkpoints missing `mtp.fc.weight`
  have also been reported [U].

**Competitor state.**
- mlx-lm: draft-model speculation only (`speculative_generate_step`); acceptance requires exact
  token equality [V `generate.py:624`]; the server disables batching with a draft model or
  `kv_bits` [V `server.py:366,649`].
- llama.cpp: `draft-simple`, `draft-eagle3`, `draft-mtp`, `draft-dflash` and five n-gram variants
  [V README]. For `ngram-mod` it says "**MoEs require long drafts**" (defaults n_match 24, n_min/n_max
  48/64) and names "rewriting of source code" as the use case [V `docs/speculative.md`].
- Ollama MLX: MTP and DFlash, with a learned **depth controller** maximizing committed(N)/cost(N),
  including depth 0 = plain decode [V `speculate_depth.go`]; MTP is opt-in via `draft_num_predict`
  [V `docs/modelfile.mdx`]. oMLX ships DFlash and MTP and credits MTPLX [V README].

**MoE interaction (key risk).** Verifying k+1 tokens activates the union of their experts.
Cascade (arXiv 2506.20675) [V] measures verification slowdowns of up to 1.5× and turns speculation
off when utility (tokens gained / verify cost) drops below 1. EcoSpec (2607.12696) and MoE-Spec
(2602.16052) [V listings] cap expert scatter. For Qwen3-Coder-30B-A3B [I]: 128 experts with 8
active per layer; plain decode reads ≈1.9 GB, while a 32-token verify can touch most experts
(≈17 GB, ≈65 ms at 273 GB/s). Near-100% acceptance (verbatim copies) gives ≈5× on that stretch, but
an 8-token verify accepting 1 token can be **3–5× slower** than plain decode.

Dense models verify k tokens for roughly the cost of one. MTP on Qwen3.6-27B, whose decode
ceiling is ≈16 tok/s, is the favourable case. Community MLX MTP reports range from 1.6× to 2.6×
(MTPLX; one blog went from 7 to 18 tok/s on an M4 Pro), and one report claims a regression of up
to 25% [all U]. "Performance or Illusion?" (arXiv 2601.11580) [V] warns that acceptance varies
widely by position, request and dataset.

**Benefit on target [I].** Coding agents copy text from context. Claude Code's Edit tool sends
`old_string` copied verbatim from file contents already in the prompt, and file rewrites repeat
whole files. N-gram drafts hit these stretches at very high acceptance. This affects decode time
only, so the win is on edit-heavy turns (hundreds of copied tokens), not on TTFT. The expected
result is 1.5–3× decode on those turns, and about 1.0× elsewhere with a utility controller (never
below 1× with depth 0).

**Exactness / constitution.**
- **Greedy**: each token is the target argmax of a k+1-wide forward, so last bits can differ from
  1-wide decode ([R04](../RISKS.md)); this meets the tolerance definition, not bitwise identity.
- **Prefix store**: accepted tokens' KV is "computed for exactly those tokens", so valid; rejected
  slots are trimmed before anything is committed to the tree.
- **Hybrids**: rollback needs per-draft-position state snapshots (Ollama schedules them at
  [before, before+k) [V `speculate.go:337`]; vLLM keeps an SSM replay buffer [V `config/cache.py`]).
- **MTP heads** have their own KV; Ollama keys its trie with a one-token look-ahead to keep it
  consistent [V `mtp.go`, `prefix_cache.go:184`]: extra prefix-store state.
- **Grammar and memory**: strict-tool masks apply per verified position. N-gram needs no memory;
  MTP needs the head (one layer) plus its KV, budgeted at startup.

**Dependencies**: engine `trim` (001), radix store (003), agent traces (005) to measure acceptance;
MTP also needs hybrid support (§3) and a conversion that keeps `mtp.*`. **Priority**: n-gram/suffix with a utility controller **next**; MTP **next, after §3** (it only pays
on Qwen3.6/3.8, which are hybrids); EAGLE-3/DFlash **later** (per-target trained drafters, none for
the default model).

**Spec outline: "Copy-aware speculative decoding".**
- *Stories*: As a Claude Code user, file edits stream noticeably faster; As a user, I can turn speculation off; As a user, speculation never makes a turn slower.
- *FRs*: Draft from an n-gram index over the current prompt plus output (≥ N-token match, long drafts for MoE); Greedy verification with rejection sampling when temperature > 0; Roll back by trim, and never commit rejected tokens to the store; A per-model utility controller with depth 0, learned per-depth cost, and persistence across requests; Disabled under strict grammar until per-position masking exists; Usage reports drafted and accepted counts.
- *SCs*: Edit-heavy trace decode ≥ 1.5× faster; Non-copy trace decode ≥ 0.98× of baseline; Greedy agreement with the non-speculative path ≥ 99% of tokens on the conformance prompts; Zero rejected tokens ever observed in the store (mutation test).

## 2. KV cache quantization

**Mechanism.** mlx-lm `QuantizedKVCache` stores K and V as `mx.quantize` triplets (packed uint32
plus per-group scales and biases), default group 64, 8 bits [V `cache.py:251`]. Attention goes
through `quantized_scaled_dot_product_attention`: `quantized_matmul(Q,K)` → softmax →
`quantized_matmul(P,V)`. It is **not a fused kernel** and rejects attention sinks [V
`models/base.py:64–131`]. MLX v0.32.3's fused SDPA accepts only f32/f16/bf16 [V
`scaled_dot_product_attention.cpp:1649`]. mlx-lm converts a cache to quantized once
`offset ≥ quantized_kv_start` (default **5000**) [V `generate.py:47,296`]. llama.cpp offers
`-ctk/-ctv` q8_0/q4_0/… [V]; oMLX wraps TurboQuant (mlx-vlm) with its own Metal kernels [V
`turboquant_kv.py`]. mlx-c v0.7.0 exposes `mlx_quantize` and `mlx_quantized_matmul` [V `ops.h`], so
no new kernels are needed.

**Capacity [I]** (Coder-30B, group 64, bf16 scales): 8-bit is 1.0625 B/value, 51 KiB/token, ≈207K
tokens in the same 10.3 GiB; 4-bit is 0.5625 B/value, 27 KiB/token, ≈390K tokens, more than the
model's 256K context [U].

**Decode bandwidth at 100K [I]**: ≈11.7 GB/token in bf16, 7.1 GB in Q8 and 4.7 GB in Q4. The
ceilings are ≈23, ≈38 and ≈58 tok/s. For an A3B MoE this is the **largest long-context decode
lever**. It also doubles how many branches or sessions the prefix store can hold.

**Hidden cost [I].** The unfused path materializes scores of size L×kL per head. A 2,048-token
prefill chunk with 32 query heads at kL=100K needs ≈12.5 GiB of scratch, which breaks D012's
measured-scratch budget. mrun should therefore dequantize one layer's K and V transiently for
prefill (≈200 MB at 100K) and use fused SDPA, and use `quantized_matmul` only at decode (L=1).

**Quality evidence.** KIVI (2402.02750) and KVQuant (2401.18079) [V] show keys need per-channel
treatment and that 2–3 bit is near-lossless only with special handling. vLLM's TurboQuant study
(blog, 2026-05) [U] finds 8-bit keys benign, while 3-bit keys "show meaningful drops, especially on
… very long-context tasks". "Length, Not Bit Width" (NeurIPS 2026) [U] finds int8 and int4 move the
retrieval cliff inward at long context. No study of 8- vs 4-bit KV on long-context *code* was
found: **Q8 is plausible; Q4 is unproven** for 100K-token agent sessions.

**Exactness / constitution.** This is approximate relative to bf16, so it is opt-in under
[D004](../DECISIONS.md). It must be a separate **model identity** component. The radix tree already
roots on "KV dtype" (ARCHITECTURE). The key must include `{bits, group_size, scale dtype, start
policy}`. The start threshold matters because KV computed while attending to a bf16 history differs
from KV computed while attending to a quantized one. Recommendation: quantize from token 0 so the
store has one regime per identity, and drop mlx-lm's 5,000 threshold. The memory budget improves by
construction, since bytes per token are exact.

**Dependencies**: 001 KV layout (quantized variant with offset/trim), 003 identity tuple, 005
quality and capacity cells. **Priority**: **next, behind §3 and §4**; 8-bit only, 4-bit labelled
experimental. If the default moves to a hybrid (§3), only a quarter of layers hold KV and urgency falls.

**Spec outline: "8-bit KV precision route".**
- *Stories*: As a user on 48 GB, I can hold 200K tokens of context; Long sessions decode faster.
- *FRs*: `--kv-bits 8` (group 64), quantized from position 0; The quantization fields are part of the identity, and a cross-precision hit is impossible; Prefill uses per-layer transient dequantization so scratch stays within the measured budget; `doctor` and the budget report token capacity per precision.
- *SCs*: KV capacity ≥ 1.9× bf16; Decode at 100K ≥ 1.4× bf16; Pass-rate delta ≤ 1 point on a long-context coding task set (≥ 50 cases at 32K–128K); Peak scratch at max context within the startup measurement.

## 3. Hybrid-attention models (Gated DeltaNet / Mamba)

**Mechanism.** Qwen3-Next and Qwen3.5/3.6/3.8 interleave 3 Gated DeltaNet layers per full-attention
layer (`full_attention_interval=4`) [V mlx-lm `qwen3_5.py`]. Each GDN layer keeps an
`ArraysCache(size=2)`: a conv state of shape (kernel−1) × conv_dim in the input dtype, and a
recurrent state of shape (B, Hv, Dv, Dk) in **float32** [V `qwen3_5.py:150–197`,
`gated_delta.py:583`]. The recurrent state is **not trimmable**; reuse is possible only at offsets
where a snapshot was captured. Nemotron-H uses Mamba-2 layers the same way [V mlx-lm
`nemotron_h.py`, Ollama `model/nemotron_h`]. mlx-lm's GDN uses a custom Metal kernel via
`mx.fast.metal_kernel`; mlx-c exposes `mlx_fast_metal_kernel_*` [V `fast.h:107`], so mrun can reuse
the kernel source.

**State size [I from U configs].** Qwen3.6-27B has 64 layers, 48 of them GDN, with 48 value heads ×
128 × 128 dims: 3 MiB recurrent state per layer plus 60 KiB conv state, so **≈147 MiB per
checkpoint**. Its attention KV is 16 layers × 4 × 256, or 64 KiB/token. Qwen3.6-35B-A3B (30 GDN
layers, 32 value heads) needs about 61 MiB per checkpoint [U]. A checkpoint every 4,096 tokens at
100K would cost 3.6 GiB, so placement must be sparse and deliberate.

**How competitors reuse prefixes.**
- **mlx-lm** keeps whole-cache entries only at segment ends (system prompt, user context, thinking
  tail) [V `server.py:523–631`]; a non-trimmable cache can use only an exact or shorter entry [V
  `cache.py:1652–1673`].
- **Ollama**'s trie holds whole-state snapshots at node ends; "recurrent caches split their internal
  recurrence at these boundaries" during a single prefill; requests resume at the deepest node ≤
  match holding state, and a capture is scheduled at the match point [V `prefix_cache.go:1–27,323–336`].
- **llama.cpp** keeps `--ctx-checkpoints` (32 per slot) [V]. **vLLM** uses
  `mamba_cache_mode="align"`, caching state at multiples of `mamba_block_size` [V].
- **oMLX** stores GDN "sidecars" on SSD, optionally int8 or RHT-int8. That is lossy and not
  acceptable for mrun [V `boundary_snapshot_store.py`].
- **Marconi** (arXiv 2411.19379, MLSys 2025) [V] admits checkpoints by reuse likelihood and evicts
  by FLOPs saved per byte, reporting up to 34.4× higher token hit rate.

**What mrun's prefix store must add** ([D017](../DECISIONS.md) anticipated this):
1. **Checkpoint objects** on nodes: per-layer conv and recurrent state, fp32, deep-copied, budgeted.
2. **A boundary-aligned chunk planner**: prefill chunk ends land exactly on planned capture offsets
   (end of system+tools, start of the last user turn, end of each assistant turn), so no extra passes.
3. **Lookup** returns the deepest checkpoint ≤ match; the gap is re-prefilled (exact) and the miss
   report states the "checkpoint gap" tokens.
4. **Live-path divergence**: trimming no longer works for GDN layers; restore the nearest checkpoint
   ≤ divergence. The live path keeps the last 2–3 boundary checkpoints, not a periodic grid.
5. **Full-prompt hits**: FR-004 (reprocess the last token) needs a checkpoint at length−1, or stored
   last logits.
6. **Eviction** scoring bytes against prefill seconds saved (Marconi): a 147 MiB checkpoint equals
   ≈2,300 tokens of KV. 7. **Speculative rollback** snapshots (§1).

**Benefit**: Qwen3.6/3.8 are the strongest coders that fit 48 GB. Vendor SWE-bench Verified is
77.2 for 27B and 73.4 for 35B-A3B [U]. The 35B-A3B MoE decodes at ≈85 tok/s on an M4 Pro [U] with
a small KV footprint. This is likely mrun's **next default model**.

**Risk**: new model code (GDN kernel, gated attention, MoE-hybrid) needs an oracle against mlx-lm;
the checkpoint policy is the core algorithm and must be benchmarked on traces. **Dependencies**: 001
(model graph, Metal-kernel binding), 003 (store), 005 (traces). **Priority**: **next, first** (the
strategic model family; unlocks MTP).

**Spec outline: "Hybrid models with state checkpoints".**
- *Stories*: As a user, I run Qwen3.6-35B-A3B/27B with prefix reuse as good as on Coder-30B; Misses explain checkpoint gaps.
- *FRs*: Port `qwen3_5` and `qwen3_5_moe`; Capture checkpoints at planned boundaries during prefill; Resume at the deepest checkpoint ≤ match; Checkpoint bytes are admitted and evicted by the budget; No lossy state compression.
- *SCs*: Greedy matches mlx-lm token-for-token; Repeated-turn TTFT at 30K ≥ 5× cache-off; Re-prefilled checkpoint gap ≤ the last turn's length in ≥ 95% of trace turns; Zero stale reuse in the mutation suite.

## 4. SSD / persistent prefix tier

**Mechanism.** Immutable segments and checkpoints are spilled to disk on eviction or idle, and
reloaded on a match after restart.
- **oMLX** keeps a hot RAM tier plus a cold SSD tier of safetensors blocks, SHA-256 hash-chained and
  keyed by `model_name` (a name, not a content hash); writes are fsynced and atomic; the default
  size is 50% of free disk [V `paged_cache.py:84–115`, `paged_ssd_cache.py`, README].
- **llama.cpp** offers `/slots/{id}?action=save|restore` into `--slot-save-path`, with filename
  validation [V `server-context.cpp:5926`]. It does not bind a content hash of the model [U].
- **Papers**: AttentionStore/CachedAttention (arXiv 2403.19708, ATC 2024) [V], up to 87% lower TTFT;
  Pensieve (2312.05516) [V], multi-turn tiers; LMCache (2510.09665, MLSys 2026) [V], up to 15×
  throughput.
- mlx-c exposes safetensors read and write [V `io.h`]. mrun can also write raw segment bytes from
  host pointers (unified memory).

**Bandwidth vs recompute [I].** M4 Pro internal SSD reads about 5.4 GB/s for a 2 TB unit [U
forum-quoted review]. Smaller SKUs are slower.

| Prompt (Coder-30B) | Restore from SSD | Recompute |
| --- | --- | --- |
| 30K tokens (2.75 GiB) | ≈0.6 s | ≈40–60 s |
| 60K tokens | ≈1.2 s | ≈2–3 min |

Writes are deltas: only new segments, about 190 MB per 2K-token turn, written off the critical
path. SSD wear stays modest if mrun writes on eviction and idle, not on every turn.

**Value for agents.** TraceLab (arXiv 2606.30560) [V/U] attributes most cache loss to human pauses.
Persisting segments covers `claude --resume` the next day, `mrun serve` restarts and upgrades of the
same model, idle weight unloads, and memory-pressure shedding (which becomes spilling, not dropping).

**Safety / constitution.**
- **Identity binding.** Each file header carries the full identity tuple: weights hash, tokenizer,
  template, KV precision, checkpoint format, and the mrun format version. MLX version is
  recommended too, because stored bytes are "exact for those tokens" but a version change alters
  recompute bits. Lookup **compares stored token ids**, never hashes alone (CVE-2025-25183 lesson,
  [literature §2.1](literature-study.md)).
- **Integrity.** Each segment has a checksum. A corrupt or truncated file counts as a miss and is
  deleted.
- **Privacy.** Token ids are the prompt in plaintext-equivalent form, and KV can leak content [I].
  The tier must be **opt-in**, stored under `~/Library/Caches/mrun` with `0700`/`0600` permissions,
  excluded from Time Machine and Spotlight, with a size cap, TTL and `mrun cache clear`. No
  app-level encryption: rely on FileVault, and have `doctor` warn when it is off (per-file keys add
  Keychain complexity for little gain on a single-user Mac) [I]. This conflicts with today's 003
  FR-010 (memory only) and needs a vision note: persistence is a local cache, not a conversation store.
- **Memory.** Loads go into budgeted capacity, as with any restore. Disk use is capped.

**Dependencies**: 003 segments (immutable, per-edge) and 004 content-hash identity. **Priority**:
**next, second**: mrun's biggest real-usage TTFT win after 003, and table stakes (oMLX markets it).

**Spec outline: "Persistent prefix tier".**
- *Stories*: As a user, after a restart or overnight, my resumed Claude Code session's first turn is fast; I can see and clear what is stored.
- *FRs*: Opt-in `--disk-cache[=SIZE]`; Spill on eviction, idle and shutdown; Identity header plus token-id verification on load; Atomic writes with checksums; Restrictive permissions and backup exclusion; LRU/TTL cap; `mrun cache ls|clear`; Miss reports include "disk hit" and "disk identity mismatch".
- *SCs*: Restored 30K-token prefix TTFT ≤ 2 s after restart, versus a cold prefill of ≥ 30 s; Zero cross-identity loads in mutation tests; Disk usage never exceeds the cap; Disk I/O adds no warm-path latency (≤ 1%).

## 5. Parallel sessions / batching

**Mechanism.** mlx-lm `BatchGenerator` does continuous batching, with defaults of decode
concurrency 32 and prompt concurrency 8 [V `server.py:1883–1891`]. `BatchKVCache` left-pads, and
`merge` allocates B × max_length × … and copies each row [V `cache.py:880–1082`]. A 60K-token
session and a 5K-token session therefore both pay 60K, and a shared system prompt is **duplicated
per row**. The server disables batching for draft models, `kv_bits` and seeds [V]. oMLX and
vllm-mlx build on this; vllm-mlx (arXiv 2601.19139) [V] reports 4.3× aggregate throughput at 16
concurrent requests on an M4 Max, a server-throughput regime.

**Target reality [I].** Decode is bandwidth-bound: batch 2–4 raises aggregate tok/s, but each
session's tok/s falls. For MoE, batching scatters experts just like speculation verification, so
gains are smaller than for dense models. Prefill is compute-bound, so concurrent prefills simply
share the GPU. Each session needs its own KV (96 KiB/token); two 60K sessions take 11 GiB, which the
budget must reserve up front. Radix sharing still helps, since each session restores the shared
system+tools prefix by copy, but MLX SDPA cannot share prefix memory across rows without new kernels
(Hydragen/ChunkAttention-style, [literature §2.1](literature-study.md)). Batched forwards are not
bit-identical to single-stream forwards (shapes, padding masks).

Claude Code fans out parallel subagents via the Task tool [U], so concurrency does occur. Its
benefit is overlap. With FIFO, subagent B waits for all of A's turn.

**Exactness / constitution.** Within the tolerance definition, as with speculation, but admission
grows to N reservations with eviction while N streams grow, and it conflicts with ARCHITECTURE's
"one active sequence" (Principle VI). **Cheaper alternative**: turn-level interleaving at chunk and
token boundaries with single-row kernels; agent turns end at tool calls, so B runs while A waits.

**Priority**: **later**. Gate it on 005 traces showing more than 20% of wall time spent queued
behind another session. If adopted, use a fixed small batch (≤ 4) of decode rows with exact per-row
reservation, and never combine it with speculation in v1.

**Spec outline: "Concurrent subagents".**
- *Stories*: As a user running 3 parallel subagents, total wall-clock drops; No session is starved.
- *FRs*: At most `--max-active N` (default 1) sessions; Admission reserves KV per row; Shared prefixes restore from the store; Cancellation is per row; Fairness is FIFO by turn.
- *SCs*: 3-subagent trace wall-clock ≥ 1.3× faster; Single-session latency regression ≤ 5% when N=1 traffic; Zero swap under N × max context.

## 6. Gaps in the roadmap

| Gap | Evidence | Benefit / risk | Priority |
| --- | --- | --- | --- |
| **Anthropic `diagnostics.cache_miss_reason`**: the request carries `diagnostics.previous_message_id`, and the response returns `model_changed`, `system_changed`, `tools_changed`, `messages_changed{cache_missed_input_tokens}`, `previous_message_not_found` or `unavailable` | [V anthropic SDK `types/diagnostics.py`, `cache_miss_*.py`, `beta/message_create_params.py:159`] | This is mrun's miss report (VISION #2) in the client's own wire format, so it is cheap. Responses would need stable message ids, which mrun can derive from token hashes. | **now** (fold into 002/003) |
| **Cache hints**: Anthropic `cache_control` breakpoints and OpenAI `prompt_cache_key` / `prompt_cache_retention` | [V openai-python `completion_create_params.py:192,213`; 002 research notes `cache_control`] | Breakpoints are the client declaring stable prefixes, so mrun should materialize segment boundaries and checkpoints there (critical for §3). The key gives session affinity for eviction. This is exact and needs no new state. | **now** (003 boundary source) |
| **Real context window reporting**, so clients auto-compact before admission fails | oMLX "reporting the model's real context window to auto-compact" [V README]; Codex falls back to 272,000 for unknown slugs [V 002 research] | It avoids hard rejections at the window edge. After compaction, only the system+tools segment hits, which argues for keeping that segment pinned. | **now** (002/004) |
| **Server-side compaction** (`compact_20260112`, `clear_tool_uses` context edits) | [V anthropic SDK beta types] | This is a server-side LLM summarization step, which is out of scope as server-side behaviour. Reject it with a clear error, or ignore the beta header. | **no** |
| **Tool-call-aware early stop**: stop at the close of a tool call when `parallel_tool_calls=false`, or after the last call marker | Qwen3-Coder XML parser [V landscape §2.2] | It saves tens of wasted tokens per tool turn. The risk is cutting legal parallel calls, so it applies only when the request forbids them. | **next** (small) |
| **Observability**: Prometheus `/metrics` (TTFT, prefill/decode tok/s, hit ratio, bytes by tier, evictions) and os_signpost intervals | llama.cpp `/metrics` [V README]; oMLX `server_metrics.py` [V] | It makes `bench` claims checkable in daily use. It costs little and is read-only. | **next** |
| **Partial-prefill salvage on cancel**: publish completed chunks as a segment | [literature A-7](literature-study.md) | Cancel-then-edit-resend is common. The salvage is exact, because chunks completed on the GPU. | **next** (003 follow-up) |
| **Session-aware eviction**: TTL-protect the latest tool-call boundary, and score eviction by FLOPs per byte | Continuum arXiv 2511.02230, Marconi 2411.19379 [V] | It raises the hit rate after human pauses, especially when combined with §4. | **next** |
| **logprobs / top_logprobs** (Chat, Responses) | OpenAI params [V]; Ollama disables speculation when logprobs are requested [V `speculate.go:208`] | Few agents use them. They interact with speculation. | **later** |
| **launchd agent** (`mrun service install`, KeepAlive, optional socket activation for lazy load) and a **Homebrew** formula | oMLX `brew services` [V]; landscape §6 | Zero-touch start after login is expected DX. | **next** (roadmap "Distribution") |
| **Sleep prevention while busy**: an IOPMAssertion during active requests | [I] | Laptop lids and idle sleep kill long agent runs. This is about 30 lines. | **next** |
| **Menu-bar app** | oMLX ships one [V README] | Nice to have. A second UI surface conflicts with the non-goals. | **no** (reconsider after 1.0) |
| **Approximate prefill tricks** (SpecPrefill, CacheBlend, KV eviction) | oMLX `specprefill/` [V]; literature Tier C | They violate D004 exactness. | **no** |

## Prioritized recommendation

1. **now**: fold four small items into 002/003/004: Anthropic `diagnostics.cache_miss_reason`;
   `cache_control` and `prompt_cache_key` as boundary and affinity hints; real context-window
   reporting; rejecting server-side compaction with a clear error.
2. **next, 006 Hybrid models with state checkpoints** (Qwen3.6-35B-A3B first, then 27B): the
   strategic model family, and a prerequisite for MTP. Build the boundary-aligned chunk planner
   and Marconi-style eviction.
3. **next, 007 Persistent prefix tier**: an opt-in, identity-bound, token-verified SSD tier. This
   gives a ~50× faster first turn after a restart or a pause.
4. **next, 008 Copy-aware speculative decoding**: n-gram/suffix drafts with a utility controller,
   tuned for MoE (long drafts only). Add an MTP drafter for Qwen3.6/3.8 once 006 lands.
5. **next, 009 8-bit KV precision route**: opt-in, identity-bound, with per-layer dequantize for
   prefill. Gate it on a long-context coding quality check. 4-bit stays experimental.
6. **next, small items**: `/metrics`, partial-prefill salvage, session-aware eviction, tool-call
   early stop, sleep assertion, launchd and Homebrew.
7. **later**: concurrent sessions (only if traces show queueing), logprobs, EAGLE-3/DFlash (no
   drafter exists for the default model).
8. **no**: lossy state or KV compression by default, server-side compaction, approximate prefill,
   and a menu-bar app before 1.0.

**Measure first** (cheap, in 005), before writing these specs: n-gram acceptance-length
distribution on recorded Claude Code and OpenCode traces; the fraction of wall time spent queued
behind concurrent requests; bytes against prefill seconds per checkpoint boundary on hybrid traces;
SSD read GB/s on the target Mac.
