# Research: Engine Core

Sources: [mlx-rust](../../concepts/research/mlx-rust.md), [candle-capabilities](../../concepts/research/candle-capabilities.md),
[landscape](../../concepts/research/landscape-2026-10.md), [literature](../../concepts/research/literature-study.md).

## R1: Engine library and bindings

- **Decision**: MLX v0.32.4 (2026-10-14; SDPA, gather_qmm, split-K quantized matmul and Metal
  deadlock fixes over 0.32.3) through mlx-c v0.7.0, wrapped by our own `mlx-sys` (vendored,
  hash-checked, bindings checked in) and a ~60-function safe layer in `mrun-engine::mlx`.
- **Rationale**: D010. mlx-rs 0.32 doesn't link on Linux, lacks quant `mode`, `device_info`,
  `metal_is_available`. Our own layer exposes exactly what the two model graphs need.
- **Alternatives**: mlx-rs as-is (rejected above); safemlx (competing runtime, single author);
  Candle (D010: no Metal MoE, corruption >8K, no perf data); llama.cpp FFI (C++ state ownership,
  GGUF-only, earlier rejected design).
- **Spike gate (M0)**: on the M4 Pro, build `mlx-sys` and run: matmul, quantized_matmul (affine
  4-bit), gather_qmm, fast::sdpa, fast::rope, memory APIs, async_eval. Go/no-go within a week.

## R2: Model graphs

- **Decision**: Port mlx-lm `qwen3.py` and `qwen3_moe.py` at commit `ba8a08f` line by line: per-head
  q/k RMSNorm before RoPE, RoPE offset from cache, GQA through `fast::sdpa` with causal mask,
  tied/quantized lm_head; MoE router with 8-bit `mlp.gate`, `softmax(precise)`, `argpartition` top-k,
  `norm_topk_prob`, SwitchGLU via `gather_qmm`, sort-by-expert when `indices.size >= 64`.
- **Rationale**: Exact mlx-lm parity is the conformance oracle (SC-001); same ops, same order, same
  MLX version.
- **Alternatives**: Writing graphs from papers (drift risk); generic model framework (VI).

## R3: Weights and quantization

- **Decision**: Read `config.json`, `quantization {group_size, bits, mode}` with per-path overrides;
  infer mode from scales dtype when absent; load safetensors with MLX's loader (zero-copy mmap where
  MLX supports it); reject unknown `model_type`, unsupported bits/modes with named reasons.
- **Rationale**: Community MLX format (D015); MoE router is 8-bit while experts are 4-bit.

## R4: KV cache

- **Decision**: Per layer K and V `[1, n_kv, cap, head_dim]` in the activation dtype (bf16/f16 per
  config). Allocate `cap = round_up(prompt + min(max_output, 4096), 256)` at admission and grow in
  4,096-token steps, **one layer at a time** (allocate the layer's larger K/V, copy, free the old
  pair before the next layer), so the transient is one layer's K/V (≈ 1/48 of the cache: ≈ 190 MiB
  at 96K tokens on the MoE model) and the whole growth costs ≈ 30 ms at 60K tokens. The scratch
  probe adds this per-layer transient at full capacity to the scratch envelope (Principle III);
  `offset` is the valid length; trim = set `offset`; append = `slice_update` on solely-owned
  buffers. Snapshots use an explicit deep copy (allocate `zeros` of the segment shape, then
  `slice_update` the source range into it, then `eval`), never `slice` or `copy`, because MLX may
  share the source buffer; debug builds assert the snapshot's data pointer differs from the
  source. Restore drops or trims the active cache first, then copies segments into newly
  allocated buffers (no transient doubling).
- **Rationale**: mlx-lm `KVCache` design; MLX donation semantics make shared handles copy on every
  write ([mlx-rust §7.3](../../concepts/research/mlx-rust.md)); analysis finding T5.
- **Sizes**: Qwen3-Coder-30B-A3B: 48 layers × 2 × 4 kv heads × 128 × 2 B = 96 KiB/token.
  Qwen3-8B: 36 × 2 × 8 × 128 × 2 B = 144 KiB/token.

## R5: Prefill chunking and cancellation

- **Decision**: Mirror mlx-lm's schedule: all but the last prompt token in 2,048-token chunks, the
  last token processed by the first decode step. Within a chunk, layers are submitted with
  `async_eval` but the CPU keeps at most two layers in flight: before building layer i+1 it waits
  for layer i−1 to complete, then checks the cancel flag. Cancellation latency is therefore at most
  about two layers of one chunk (2,048-token chunk ≈ 2.7 s at ~768 tok/s on the MoE model → ≈ 56 ms
  per layer, ≈ 115 ms bound; deeper positions are slower but stay under 300 ms). Decode uses
  `async_eval` pipelining with a cancel check per token; when a token mask is required
  (constrained decoding, 002) pipelining is disabled for that request because the mask for step
  n+1 depends on token n.
- **Rationale**: Analysis finding T1: with cancel checks only between chunks, 300 ms would need
  ~150-token chunks, which cuts MoE prefill throughput far below mlx-lm. Per-layer eval keeps
  mlx-lm's chunk size (fair SC-003) and bounds cancellation independently.
- **Risk**: Per-layer eval adds 36–48 sync points per chunk; measured with the engine benchmark
  (FR-017). Fallback: eval every 4 layers (cancel ≈ 225 ms per chunk-layer group on the MoE model).
- **Conformance**: references are generated with the same chunk size and schedule (T026 after T020).

## R6: Memory budget

- **Decision**: `budget = min(0.75 × hw.memsize, device_info.max_recommended_working_set_size) −
  8 GiB` (user can lower; raise only with `--allow-over`). The 8 GiB reserve covers other apps and
  mrun's host heap; host-side work is bounded by input caps (FR-016, 002 FR-009).
  1. Pre-load check from safetensors file sizes (≈ weights) against budget (FR-007).
  2. Load weights; read `get_active_memory` as measured weights.
  3. Scratch: run one 2,048-token chunk at position `P = min(max_position, (budget − weights −
     scratch_estimate) / kv_bytes_per_token)` (`scratch_estimate` = 2 GiB) with a temporary KV of P
     tokens; `scratch = (peak − active_before + kv_bytes_at_P / layers) × 1.25`, which includes the
     per-layer KV growth transient (the only place the margin is applied). Cache in `~/.mrun/probe.toml` keyed by (model
     identity, machine model, macOS, mrun version) so later starts skip it (SC-007).
  4. `kv_capacity_tokens = min(max_position, (budget − weights − scratch) / kv_bytes_per_token)`.
  5. `set_memory_limit(budget)`, `set_cache_limit(scratch)`, `set_wired_limit(min(budget,
     recommended))` for the engine's lifetime; `clear_cache()` after prefill and after evictions.
- **Worked example (M4 Pro 48 GB, Qwen3-Coder-30B-A3B 4-bit)**: budget ≈ min(36, ~36) − 8 = 28 GiB;
  weights ≈ 17 GiB; scratch ≈ 1–2 GiB; KV ≈ 9–10 GiB ≈ 100–110K tokens at 96 KiB/token, shared
  between the active conversation and the prefix cache. Qwen3-8B: weights ≈ 4.3 GiB, memory allows
  ≈ 160K tokens, capped at max_position 40,960.
- **Rationale**: D012; analysis findings T3, T4, C5, F4, A1.

## R7: Sampling

- **Decision**: On-device sampling in MLX ops: temperature scaling, top-k (`argpartition`), top-p
  (sort + cumsum), min-p, categorical with an explicit per-request key derived from `seed`;
  temperature 0 = argmax. An optional token mask (from 002's constrained decoding) is applied as
  `where(mask, logits, -inf)`. Stop sequences are matched on decoded text with a hold-back buffer.
- **Rationale**: Avoid a device→host logits copy per token (Qwen3 vocab 151,936 × 4 B).

## R8: Tokenizer and template

- **Decision**: `tokenizers` 0.23 from `tokenizer.json`; chat template from
  `tokenizer_config.json`/`chat_template.jinja` rendered by minijinja 3 with pycompat; streaming
  detokenization with a UTF-8 hold-back buffer.
- **Rationale**: Same files as mlx-lm; template parity is tested by comparing rendered token IDs with
  mlx-lm's `apply_chat_template` on the reference prompts.

## R9: Threading and layering

- **Decision**: The `Engine` trait and its types live in `mrun-engine` with no MLX dependency;
  `MlxEngine` is behind the `mlx` cargo feature (default on). The engine exposes primitives
  (reserve, prefill, decode, trim, snapshot, restore); the generation loop (stop sequences, UTF-8
  hold-back, events, stats) lives in `mrun::runtime` and is generic over `Engine`, so it is tested
  with `ScriptedEngine` without building MLX (`cargo test -p mrun --no-default-features`). The
  engine is constructed on the worker thread (`EngineHandle::spawn(factory)`), so MLX arrays never
  cross threads and the trait needs no `Send` bound.
- **Rationale**: Analysis findings F1, F3, F6, F12; MLX objects and streams must stay on one thread.


## R10: Model identity without feature 004

- **Decision**: `identity = sha256` over the sorted list of `(relative path, file sha256)` for every
  file that determines output (`*.safetensors`, `config.json`, `tokenizer.json`,
  `tokenizer_config.json`, `chat_template.jinja` if present). In the HF cache, LFS blob file names
  are already their sha256, so weights need no hashing; small files are hashed (milliseconds).
  Other local paths are hashed once and cached in `~/.mrun/identity.toml` keyed by (canonical path,
  size, mtime, inode); `mrun verify <model>` (004) re-hashes everything. A changed hash for a known
  path is reported at load. Feature 004 reuses this exact function.
- **Rationale**: Constitution IV and 003 cache keys need identity from the first feature (analysis
  D1, 004 C2).

## R11: Engine benchmark (until feature 005)

- **Decision**: `crates/mrun-engine/examples/bench_engine.rs` measures prefill tok/s, decode tok/s
  and cancellation latency (50 random interrupt points) at configurable contexts in-process and
  writes JSON with machine info (chip, memory, macOS, power) and model identity. The reference
  numbers come from pinned mlx-lm `mlx_lm.benchmark`/`generate` with `--prefill-step-size 2048` on
  the same weights, run by `scripts/bench_reference.sh`. Feature 005 later absorbs these cells.
- **Rationale**: Constitution I requires reproducible in-repo measurement for 001's gates (analysis D3).
