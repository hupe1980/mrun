# Research: Engine Core

Sources: [mlx-rust](../../concepts/research/mlx-rust.md), [candle-capabilities](../../concepts/research/candle-capabilities.md),
[landscape](../../concepts/research/landscape-2026-10.md), [literature](../../concepts/research/literature-study.md).

## R1: Engine library and bindings

- **Decision**: MLX v0.32.3 through mlx-c v0.7.0, wrapped by our own `mlx-sys` (vendored,
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

- **Decision**: Per layer K and V `[1, n_kv, cap, head_dim]` in the model's activation dtype
  (bf16/f16 per config), grown in 256-position steps up to the request's reservation; `offset` is the
  valid length; trim = set `offset`; append = `slice_update` on solely-owned buffers. Snapshots are
  materialized slice copies; restore copies into fresh buffers.
- **Rationale**: mlx-lm `KVCache` design; MLX donation semantics make shared handles copy on every
  write ([mlx-rust §7.3](../../concepts/research/mlx-rust.md)).
- **Sizes**: Qwen3-Coder-30B-A3B: 48 layers × 2 × 4 kv heads × 128 × 2 B = 96 KiB/token.
  Qwen3-8B: 36 × 2 × 8 × 128 × 2 B = 144 KiB/token.

## R5: Prefill chunking and cancellation

- **Decision**: 512-token chunks (configurable 128–2048); `eval` per chunk, cancel flag checked
  between chunks; `clear_cache` after prefill like mlx-lm. Decode uses `async_eval` pipelining (next
  step graph built while the current one runs) with a cancel check per token.
- **Rationale**: SC-004 (300 ms) vs prefill speed: at ~768 tok/s (30B-A3B on M4 Pro, community
  numbers) a 512 chunk is ~0.67 s worst case; 256 may be required for the MoE model. The chunk size
  is tuned with `mrun bench --suite cancel` and recorded; conformance references are generated
  with the same chunk size.
- **Risk**: Smaller chunks lower prefill speed; SC-003 compares against mlx-lm at its default
  (2048). If both cannot hold, the spec's SC-004 or SC-003 gets an explicit amendment.

## R6: Memory budget

- **Decision**: `budget = min(0.75 × hw.memsize, device_info.max_recommended_working_set_size) −
  8 GiB` (user can lower; raise only with `--allow-over`). Load weights, read `get_active_memory`;
  run one prefill chunk at the maximum capacity position with a scratch KV to read
  `get_peak_memory − active` as scratch; `kv_capacity_tokens = (budget − weights − scratch ×
  1.25) / kv_bytes_per_token`. Set `set_memory_limit(budget)`, `set_cache_limit(scratch)`,
  `set_wired_limit(min(budget, recommended))` for the engine's lifetime.
- **Rationale**: D012; mlx-lm's `wired_limit` pattern; no privileged OS changes.
- **Alternatives**: Per-allocation hooks (proven infeasible in the Candle experiment).

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

## R9: Threading

- **Decision**: `MlxEngine` lives on one dedicated OS thread; `runtime::engine::EngineHandle` sends
  commands (Load, Generate, Cancel, Snapshot/Restore for 003, Stats) over crossbeam channels and
  receives `Event`s on a bounded channel (capacity 64). A consumer that stops reading for the stall
  timeout (default 10 s) cancels the request.
- **Rationale**: MLX objects and streams are not safe to share across threads; matches mlx-lm's
  dedicated generation stream.
