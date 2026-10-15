# Candle capabilities audit for mrun (Metal, Apple Silicon)

> **Historical**: Candle was dropped on 2026-10-14 (D010). Kept as evidence for that decision and for D012.

- **Audited rev:** `c68b24997319b8d76e5a4e775dab197c7f26322a` (2026-10-02, workspace version 0.11.0, `Cargo.toml:24`).
- **Upstream main (fetched 2026-10-14):** `12bb5d1f` "Pin GitHub Actions to commit SHAs (#4039)". It is the **only** commit after the pinned rev and touches only `.github/workflows/rust-ci.yml`. For the purposes of this audit, the pinned rev is the same as main.
- **Paths** are relative to the candle repo root.
- **Markers:**
  - **[V]** means verified by reading the source at the pinned rev.
  - **[U]** means unverified: an inference, a claim from an issue, or not run on hardware.
- Nothing in this audit was executed on Metal. This container is Linux.

## 1. Models relevant to coding agents

| Model file | GGUF | Metal path | MoE | Sliding / hybrid attention | KV storage |
|---|---|---|---|---|---|
| `models/quantized_qwen3.rs` | yes, `from_gguf` :463 | generic matmul attention :355-377 | no | none | `ConcatKvCache` on GPU :200-204 |
| `models/quantized_qwen3_moe.rs` | yes | **no**: MoE uses `moe_gemm_gguf`, which is CUDA-only (`candle-nn/src/moe.rs:342-354`) | yes | none | `ConcatKvCache` :58,104 |
| `models/qwen3.rs` (safetensors) | no | matmul attention; flash path is CUDA-only :236-238 | no | bails on `use_sliding_window` :126-128 | `ConcatKvCache` :168 |
| `models/qwen3_moe.rs` (safetensors) | no | yes, but the expert loop does a host sync via `to_vec2` :138-139 | yes | none | concat |
| `models/quantized_qwen2.rs` (Qwen2.5-Coder) | yes :181 | matmul + `repeat_kv` :130 | no | none | ad-hoc `Tensor::cat` :115-127 |
| `models/quantized_llama.rs` (Llama, Mistral/Devstral GGUF use arch `llama`) | yes :367 | **SDPA vector for decode** :233-244; matmul for prefill | no | none | ad-hoc cat :220-232 |
| `models/quantized_gemma3.rs` | yes :261 | matmul + `repeat_kv` :225 | no | per-layer sliding mask :152-156, 386-387; full concat KV (no rotation) :210-222 | ad-hoc cat |
| `models/gemma3.rs`, `gemma4/text.rs` | no | matmul; flash is CUDA-only | no | `RotatingKvCache` for sliding layers (`gemma3.rs:192`, `gemma4/text.rs:271`) | `KvCache` / `RotatingKvCache` |
| `models/quantized_glm4.rs` | yes :362 | matmul + `repeat_kv` :263 | no | none | `KvCache::new(2,512)` (grows) :188 |
| `models/quantized_phi3.rs` | yes :223 | matmul (flash is CUDA-only :176-187) | no | ignored (issue #3698) | `KvCache` |
| `models/quantized_mistral.rs` | candle quantized VarBuilder, not plain GGUF :19 | matmul | no | mask-only sliding window :291-295 | cat |
| `models/lfm2.rs`, `quantized_lfm2.rs`, `granitemoehybrid.rs` | lfm2 yes | matmul | granite yes | hybrid conv/SSM + attention | various |
| gpt-oss | **absent**: no file and no `gpt_oss` string anywhere [V]. MXFP4 is also not a `GgmlDType` (`candle-core/src/quantized/mod.rs:285-301`) | – | – | – | – |
| Qwen3.5 / qwen35moe | absent; open issues #3820 and #3865 [V via GitHub] | – | – | – | – |

- **[V]** Supported GGML types are F32, F16, BF16, Q4_0/1, Q5_0/1, Q8_0/1, and Q2K–Q8K (`mod.rs:285-301`).
  - There are **no IQ\* types** (IQ4_XS and similar). Many "UD"/imatrix GGUFs use IQ types for some tensors, so those files will not load.
  - Q8_1 and Q8K have no Metal mm kernel (`candle-metal-kernels/src/kernels/quantized.rs:246-247`).
- **[V]** There is no Metal quantized MoE path. `QTensor::indexed_moe_forward` panics on any non-CUDA platform (`candle-core/src/quantized/mod.rs:762-787`). The only exception is the CPU aarch64 `QMatMul` path at :1060-1120.
  - Consequence: **Qwen3-Coder-30B-A3B GGUF cannot run on Metal** without a new kernel or a per-expert gather loop.

## 2. KV cache facilities (`candle-nn/src/kv_cache.rs`)

- **[V] `Cache` / `KvCache`** (:6-150): preallocated `slice_set` buffer that grows by `max_seq_len` chunks (:58-80).
  - `current_seq_len` is private and has no setter.
  - `reset()` is the only rewind.
  - There is **no truncate**.
- **[V] `RotatingCache` / `RotatingKvCache`** (:154-400): ring buffer for sliding window. Provides `positions()` and `attn_mask()`.
- **[V] `ScatteredKvCache` + `ScatteredCacheBuilder`** (:404-600): per-batch-slot indices and masks. This is a batch-slot design, not paged attention.
- **[V] `ConcatKvCache`** (:634-790, `#[derive(Clone)]` :666):
  - `append` does a full `Tensor::cat` every step, which is O(n) copy per token (:725-748).
  - It exposes `k()`, `v()`, `k_mut()`, `v_mut()`, and `into_inner()`.
  - Truncation is possible by writing `*k_mut() = k.narrow(2, 0, p)?`.
  - Snapshots are cheap: `clone()` shares immutable `Arc` storage, and cat always allocates new storage.
- **[V] `InterleavedKvCache` / `RawInterleavedKvCache`** (:980-1125) are CPU-only f32 fast paths.
- **How `quantized_qwen3` stores KV on Metal** [V]:
  - It uses a `ConcatKvCache(dim=2)` per layer, shape `(B, n_kv, S, D)` (:200-204, 357).
  - KV dtype is **F32**, because activations are F32 throughout:
    - embedding = `QTensor::dequantize`, which returns F32 (:493);
    - the Metal `QMatMul` asserts F32 input and writes F32 output (`candle-core/src/quantized/metal.rs:390`, 318-320, 384-388).
  - K and V are then expanded with `repeat_kv`, which is a `cat` that materializes n_heads copies (:359-360, `utils.rs:91-101`).
- **[V] API exposure:**
  - The `kv_cache` field is private (:168).
  - `AttentionWeights` and `LayerWeights` are private structs, and `ModelWeights.layers` is private (:453).
  - The public API is only `forward(input, offset)` and `clear_kv_cache()` (:538, 563).
  - **No truncate, snapshot, or ownership is exposed, so mrun must fork the model file.**
  - Forking is feasible because every dependency is public: `models::with_tracing::QMatMul` (`models/mod.rs:136`), `quantized_nn::RmsNorm`, `utils::{repeat_kv, build_additive_causal_mask}`, `candle_nn::rotary_emb::rope`, and `candle_nn::ops::sdpa`.
- **[V] `forward` returns logits only for the last position** (:559-560). Per-position logits (needed for speculative decoding or verification) require the fork.

## 3. Attention and matmul on Metal

- **[V] `candle_nn::ops::sdpa`** (`candle-nn/src/ops.rs:1008-1330`) is Metal-only. The CPU path bails (:1029).
  - **Head dims:** 32, 64, 72, 80, 96, 128, 256, 512 (:1079-1086).
  - **Dtypes:** F32, F16, BF16. Q, K and V must have the same dtype (:1114-1125). An F32 + head_dim 512 full kernel is excluded (:1090).
  - **GQA:** native via `gqa_factor` (`kernels/sdpa.rs:103,282,427`). Do not use `repeat_kv`.
  - **Vector kernel** (`q_seq == 1`): switches to a 2-pass kernel when `k_seq >= 1024` (:1131-1132).
  - **Full kernel** (`q_seq > 1`): MLX "steel" attention with tiles bq=32 and bk = 32 (d<128) or 16 (`kernels/sdpa.rs:86-97`).
  - **Causal masking:** in-kernel `do_causal` with `qL_off = kL - qL` (`sdpa.rs:171`, `.metal:2066,2129`). This is correct for chunked prefill against an existing cache, and no mask tensor is needed.
  - **Softcapping:** not supported in the full kernel (:1205-1206).
  - **Additive masks:** must match q dtype and have shape `(B, H, qL, kL)` (:1225-1235).
  - **Sequence length:** no explicit max; it is bounded by memory.
- **[V]/[U] Known SDPA defects:**
  - **#4020 (open):** silently wrong results with non-contiguous q/k/v. Always call `.contiguous()`.
  - **#3522 (open, reported on 0.10.2):** GQA vector decode panic. The current kernel carries `gqa_factor`, so it may already be fixed [U].
  - Past fixes: causal double-apply NaN #3625 and byte offsets #3599 (both in `git log`).
- **[V] Of the models in §1, only `quantized_llama` uses `sdpa`, and only for decode** (`quantized_llama.rs:233`). `quantized_qwen3` on Metal materializes F32 scores of size `H × l × (l+offset)` (:363-374) plus a host-built mask (:543-553). Unchunked long prefill therefore costs O(n²) memory: issue #3705 reports about 10 GB at n≈8.4k for 40 heads.
- **[V] Flash attention:**
  - `candle-flash-attn` is CUDA-only, and every `use_flash_attn` path is behind `#[cfg(feature="flash-attn")]`.
  - CPU flash exists (`candle-nn/src/attention/cpu_flash`).
  - On Metal, the steel SDPA is the flash-style kernel (online softmax).
- **[V] Quantized matmul on Metal** (`candle-core/src/quantized/metal.rs`):
  - When `x.dim(-2) == 1`, it uses `fwd_mv`, which issues **one `kernel_mul_mv_*` dispatch per row** (loop :323-337), so batched decode costs m dispatches.
  - Otherwise it uses `kernel_mul_mm_*_f32` (:344-438, `kernels/quantized.rs:233-245`). This is a true quantized-weight GEMM from ggml, with no dequant + dense GEMM.
  - Activations in and out are F32.
  - The env vars `CANDLE_DEQUANTIZE_ALL` and `CANDLE_DEQUANTIZE_ALL_F16` switch to dequant + dense matmul (`mod.rs:819-853`).
- **[V] Dense matmul** uses the MLX GEMM, and the tile depends on M:
  - `m < 16` selects tile 32×32×16; larger M selects by device class and `B·M·N ≥ 1M` (`kernels/mlx_gemm.rs:70-130`).
  - GEMV is used for M=1 or N=1 (:258-259).
- **[V] Open kernel bug #3863:** the K-quant `mul_mv` kernels have no store guard (`quantized.metal:4667` and similar). Q2K over-dispatches by 2×.
  - It is harmless when the row count is a multiple of 4 (Q3K/Q4K/Q5K) or 2 (Q6K). This holds for Qwen3 dims [U].
  - **Avoid Q2K.**

## 4. Sampling, tokenizer, chat template

- **[V] `generation::LogitsProcessor`** (`candle-transformers/src/generation/mod.rs:10-160`):
  - Strategies: ArgMax, All, TopK, TopP, TopKThenTopP, GumbelSoftmax.
  - Seeded RNG (:26).
  - `sample_f` hook for custom per-token edits (:121).
  - Sampling copies the full softmax to the host, i.e. one `to_vec1` per token (:124-127).
- **[V]** `utils::apply_repeat_penalty` (`utils.rs:68`) also runs on the host.
- **[V] Missing:** min-p, frequency/presence penalties, logit bias, grammar or JSON constrained decoding, stop strings, and logprobs output.
- **[V] Tokenizer:**
  - Examples use the `tokenizers` crate 0.23.1 (`Cargo.toml:95`) with `tokenizer.json` from the HF hub (`examples/quantized-qwen3/main.rs:101-119`).
  - Streaming detokenization is in `candle-examples/src/token_output_stream.rs`. It is an examples crate, not a library API.
  - `candle-core/src/quantized/tokenizer.rs` exists for GGUF-embedded vocabularies [U: scope not audited].
- **[V] Chat templates:**
  - The quantized-qwen3 example hardcodes ChatML (`main.rs:212`).
  - `candle-examples/src/chat_template.rs` (532 lines) uses minijinja 2 to load `tokenizer_config.json` templates. It provides `raise_exception` (:169).
  - Its `Message` has only `role` and `content` (:41-44), so there are **no tool calls or tool definitions**.
  - Whether Qwen3's tool-calling template renders, given filters like `tojson`, is [U].

## 5. Performance signals

- **[U]** No Candle-vs-llama.cpp or Candle-vs-MLX Metal benchmark was found in the repo or on the web. A web search found only MLX vs llama.cpp comparisons ([starmorph guide](https://blog.starmorph.com/blog/apple-silicon-llm-inference-optimization-guide), [Medium: MLX vs llama.cpp](https://medium.com/@andreask_75652/benchmarking-apples-mlx-vs-llama-cpp-bbbebdc18416)). mrun must benchmark Candle itself.
- **[V] GitHub issues found:**
  - **#1939** (open): "Quantized much slower than llama.cpp with same model and settings".
  - **#1757**: slow on an M2 Air.
  - **#3705** (open, 2026-07, M-series 48 GB): `quantized_qwen2` Q4_K_M on Metal produces **deterministic garbage once the KV cache reaches 8,192 positions** (the f32 K tensor is 2^25 bytes). It breaks again past 16,384 even with syncs.
    - The suspected cause is a missed synchronization in candle's untracked-hazard fence map. The workaround is `device.synchronize()` per layer.
    - The issue reports 5.44 tok/s prefill with split-prompt on Qwen2.5-32B, and Metal buffer pool power-of-two buckets that never shrink (peak about 52 GB).
    - It is not fixed at the pinned rev: no related commit is in `git log`.
    - **This applies directly to quantized_qwen3**: same concat F32 KV and same matmul path.
  - **#3707**: CPU quantized prefill had no batch path. It is partly addressed by #4003, which is the pinned head commit.
  - **#2271**: Metal memory leak in matmul (open).
  - **#3698**: BF16 RoPE tables corrupt long prompts in phi3/qwen2/gemma.
- **[V] Precision defect in `quantized_qwen3` RoPE** (inferred from code):
  - The RoPE table is built in `dtype`, which defaults to **F16** unless the non-standard GGUF key `general.dtype` is present (:483-490). The build happens at :111-115.
  - F16 cannot represent integer positions above 2048 exactly, so cos/sin are corrupted at long context.
  - This is the same class of bug as #3698. The numeric impact has not been run [U]. **The fork must build RoPE tables in F32.**
- **[V] Recent Metal work:**
  - concurrent dispatch #3511;
  - inter-encoder sync and gemv #3532;
  - readback race #3595;
  - command-buffer error surfacing #3862;
  - GEMM dynamic tiles #3313.
  - The Metal backend is moving quickly and has recent sync bugs.

## 6. Batch invariance and determinism

- **[V] Decode (M=1) and prefill (M>1) use different quantized kernels** (`mul_mv` vs `mul_mm`, `metal.rs:372-374`). They have different reduction orders, so a token's KV computed during decode will **not** be bit-identical to the same token recomputed in prefill [U on magnitude].
- **[V] Dense GEMM tile choice depends on M** (`m<16` vs larger tiles) and on `B·M·N` (`mlx_gemm.rs:78-85`). For the matmul-attention path, scores and PV depend on chunk size l through tile selection. Whether different `bk` values change the K-reduction order is [U].
- **[U]** Within `mul_mm`, each output row's K-reduction order is likely independent of M (fixed 32-wide K blocks), so prefill chunks of any size ≥2 are probably row-invariant. This is not tested.
- **[V]/[U] SDPA:**
  - In the full kernel, tiles are fixed by head_dim and do not depend on qL. Causal-masked blocks add exact zeros. Per-row output is plausibly invariant to chunk size [U].
  - The vector kernel changes algorithm at `k_seq = 1024` (1-pass to 2-pass). This is deterministic per kv length, but decode results differ from the full kernel.
- **[U]** Concurrent dispatch (#3511) uses no atomics in these kernels, so same inputs plus same shapes should give run-to-run determinism. #3705 shows schedule-dependent corruption is possible.
- **Practical rule:** exact replay holds only for **identical shapes and kernel sequence**. Reusing a prefix is exact only when you reuse the stored KV tensors, never when you recompute them with a different chunking or with decode-versus-prefill.

## 7. What mrun must build itself

- **No new kernels needed:**
  - A forked qwen3 model file with runtime-owned per-layer KV, using F16 or F32 tensors that the runtime manages.
  - Truncate, snapshot, and restore by token position (`narrow` plus Arc-shared clones).
  - F32 RoPE tables.
  - `sdpa` with `do_causal` and contiguous q/k/v for prefill chunks and decode, with no `repeat_kv`.
  - Logits for all positions, if needed.
  - Chunked prefill scheduler and memory accounting (buffer pool growth, #3705).
  - Sampling extras (min-p, penalties, stop strings, logprobs) and grammar or JSON constraints, all on the host.
  - Tool-call chat templating (extend minijinja) and parsing of `<tool_call>` output.
  - Streaming detokenizer (copy `TokenOutputStream`).
  - Prefix-cache keyed by token hash.
- **Would require new kernels (out of scope for the first feature):**
  - Metal quantized MoE (indexed `mul_mv`/`mm`). A per-expert host loop is possible but slow [U].
  - Quantized KV cache (Q8 or Q4 KV).
  - Paged or block-table attention.
  - Batched multi-row `mul_mv`, to replace the per-row dispatch loop.
  - IQ\*/MXFP4 dequant and matmul.
  - Fused RMSNorm+matmul and similar fusions.
  - SDPA with softcapping for prefill.
  - Preallocated in-place KV append without O(n) cat. `slice_set` exists (`Cache::append` :58-80), so this is not a kernel gap, but whether its Metal copy is efficient is [U].

## Implications

1. **Target model first:** **Qwen3 dense GGUF** through a fork of `quantized_qwen3.rs`, starting with Qwen3-8B and then 14B/32B at Q4_K_M or Q8_0.
   - Every op it needs exists on Metal: qmm/qmv for Q4_K and Q8_0, RmsNorm, rope, and sdpa with head_dim 128.
   - Do **not** target Qwen3-Coder-30B-A3B (qwen3moe), gpt-oss, Qwen3.5, or IQ-quant files. They need new kernels or are missing.
   - Qwen2.5-Coder through `quantized_qwen2.rs` is a fallback with the same shape of work.
2. **KV ownership:** **fork the model file** (about 570 lines). Upstream exposes no cache handles (:168, :453).
   - The fork takes `&mut LayerKv` (or a `KvState` with per-layer `(k, v)` tensors) as a forward argument, so the runtime owns truncation and snapshots.
   - Start with ConcatKvCache semantics: immutable tensors give cheap clone snapshots.
   - Move later to a preallocated `slice_set` buffer once correctness is pinned.
3. **KV dtype:**
   - Activations must stay F32 at `QMatMul` boundaries (`metal.rs:390`).
   - Keep **F32 KV for the first feature**. It is bit-faithful to upstream and SDPA needs q/k/v to share a dtype.
   - Budget the memory: for Qwen3-8B (36 layers, 8 kv heads, d=128), F32 KV is about 288 KiB per token, so 32k tokens is about 9 GiB.
   - Treat F16 KV (cast q/k/v to F16 for sdpa, cast output back) as a measured follow-up, because it changes numerics.
4. **Chunk size policy:**
   - Use a fixed prefill chunk size (for example 256 or 512), always ≥2, with the last chunk allowed to be partial.
   - Use `sdpa(do_causal=true)` to avoid the O(n²) mask and scores.
   - Decode is always M=1.
   - Prefix reuse must reuse stored KV, never recompute it.
   - Record chunk size in the cache key and metadata. Determinism tests must compare identical chunk schedules.
5. **Top 5 engineering risks:**
   1. **Metal KV-depth corruption at 8,192 and 16,384** (#3705, open). This is exactly the concat-F32-KV pattern. mrun needs a long-context golden test past 16k on M4 Pro before any claims, and may need per-layer `synchronize()`.
   2. **Upstream numerics bugs inherited by the fork:**
      - F16 RoPE table (`quantized_qwen3.rs:483-490,111-115`);
      - non-contiguous SDPA giving silently wrong output (#4020);
      - K-quant mv store overrun (#3863).
   3. **Performance gap vs llama.cpp/MLX is unmeasured.** The per-row `mul_mv` loop, O(n) cat per token, host-side sampling sync, and non-shrinking buffer pool all cost throughput and memory.
   4. **Batch invariance:** mv vs mm and M-dependent GEMM tiles mean recomputed prefixes are not bit-exact. The replay design must avoid recomputation.
   5. **Scope traps:** MoE coder models, IQ quants, tool-call templates, and constrained decoding all fall outside what Candle provides without kernels or substantial host code. Model choice must be locked to dense Qwen3 GGUF K-quants.
