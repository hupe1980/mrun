# Research: Engine Core

Sources: [mlx-rust](../../concepts/research/mlx-rust.md), [next-features §3](../../concepts/research/next-features-2026-10.md),
[competitors](../../concepts/research/competitors-2026-10.md), [fact-check](../../concepts/research/fact-check-2026-10-15.md),
mlx-lm `ba8a08f` source (`models/qwen3.py`, `qwen3_moe.py`, `qwen3_next.py`, `qwen3_5.py`,
`qwen3_5_moe.py`, `gated_delta.py`, `activations.py`, `generate.py`), mlx-c `a341b49` headers.

## R1: Engine library and bindings

- **Decision**: MLX v0.32.4 (`36969c0`) through mlx-c v0.7.0 (`a341b49`), wrapped by our own
  `mlx-sys` (vendored, hash-checked, bindings checked in) and a small safe layer
  `mrun-engine::mlx`.
- **Verified in mlx-c v0.7.0 headers**: `mlx_fast_metal_kernel_new/apply/free` and
  `mlx_fast_metal_kernel_config_*` (grid, thread group, output args, template dtype/int/bool args)
  in `fast.h`; `mlx_compile(res, closure, shapeless)` and `mlx_closure_new_func*` in
  `compile.h`/`closure.h`; `mlx_put_along_axis`, `mlx_conv1d`, `mlx_contiguous`, `mlx_logaddexp`
  (softplus), `mlx_sigmoid`, `mlx_split`, `mlx_argpartition`, `mlx_logsumexp`, `mlx_gather_qmm`,
  `mlx_quantized_matmul` in `ops.h`; `mlx_fast_rms_norm`, `mlx_fast_rope`,
  `mlx_fast_scaled_dot_product_attention`; `mlx_device_info_get/get_size` (`device.h`; Metal keys
  `memory_size`, `max_recommended_working_set_size`, `max_buffer_length`);
  `mlx_stream_new_device` and thread-local streams; memory getters/setters incl.
  `mlx_set_wired_limit`, `mlx_clear_cache`.
- **Rationale**: D010. mlx-rs lacks Linux linking, quant `mode`, `device_info` and metal kernels.
- **Spike gate (M0)**: on the M4 Pro: matmul, 4-bit quantized_matmul, gather_qmm, sdpa, rope, a
  `metal_kernel` (the GDN kernel), `compile`, memory APIs, async_eval. Go/no-go within a week.

## R2: Attention model graphs

- **Decision**: Port mlx-lm `qwen3.py` and `qwen3_moe.py` at `ba8a08f` line by line: per-head q/k
  RMSNorm before RoPE, RoPE offset from the state, GQA through `fast::sdpa` (causal for L > 1),
  tied or quantized lm_head; SwiGLU MLP; MoE router (8-bit `mlp.gate`, `softmax(precise)`,
  `argpartition` top-k, `take_along_axis`, `norm_topk_prob`), SwitchGLU via `gather_qmm` with
  sort-by-expert when `indices.size >= 64`.
- **Rationale**: Same ops in the same order on the same MLX is the conformance oracle (SC-001).

## R3: Weights, quantization and loading

- **Decision**: Read `config.json` (for `qwen3_5_moe` the `text_config` object, or the root when
  absent); `quantization {group_size, bits, mode}` with per-path overrides (router and
  `shared_expert_gate` at 8-bit/64); mode inferred from scales when absent. Load safetensors with
  MLX's loader. Apply mlx-lm `sanitize`: drop `vision_tower`/`model.visual`/`mtp.*`, remap
  `model.language_model.*` → language model, split `experts.gate_up_proj`; for unconverted
  checkpoints (conv1d weight last dim ≠ 1) move the conv1d axis and add 1 to RMSNorm weights.
  `A_log` stays float32 (mlx-lm `cast_predicate`). Reject unknown `model_type`, bits, modes or
  tensor names with named reasons.

## R4: Sequence state and allocation

- **Decision**: Per attention layer K and V `[1, n_kv, cap, head_dim]` in the activation dtype,
  `offset` = valid length, writes with `slice_update` on solely-owned buffers, trim = offset.
  Per linear layer: conv state `[1, kernel − 1, conv_dim]` (input dtype) and recurrent state
  `[1, Hv, Dv, Dk]` float32. Allocation is lazy (FR-010): `reserve(keep, total)` records the
  ceiling; the first growth covers the input + 4,096, later growth +4,096, rounded up to 256 and
  clamped to `min(total, kv_capacity_tokens)`; growth goes one layer at a time (allocate, copy,
  free) so the transient is one layer. Before any allocation the engine compares against
  `state_capacity_bytes − allocated()` and returns `NeedsMemory{bytes}` without side effects.
- **Why `NeedsMemory` and not a reclaim callback**: the evictable state (segments, checkpoints)
  lives in the engine, so a callback invoked during `prefill` would need a second mutable borrow of
  the engine. Checking all memory for a call up front keeps every error side-effect free and the
  retry trivial.
- **Sizes**: Qwen3-8B 36 × 2 × 8 × 128 × 2 B = 144 KiB/token. Qwen3-Coder-30B-A3B 48 × 2 × 4 × 128
  × 2 B = 96 KiB/token. Qwen3.6-35B-A3B: 10 attention layers × 2 × 2 × 256 × 2 B = 20 KiB/token;
  recurrent per linear layer 32 × 128 × 128 × 4 B = 2 MiB + conv 3 × 8,192 × 2 B = 48 KiB, × 30
  ≈ 61.4 MiB per checkpoint (dimensions read from config and checked at load).
- **Rationale**: MLX donation semantics make any extra handle turn each append into a full copy
  ([mlx-rust §7.3](../../concepts/research/mlx-rust.md)); never alias.

## R5: Prefill chunking and cancellation

- **Decision**: Chunk length `clamp(2048 × 16384 / max(depth, 1), 256, 2048)` from the chunk's
  start position (2,048 up to 16K; 1,024 at 32K; 512 at 64K; 256 from 128K), so attention work per
  chunk-layer stays roughly constant and one layer stays near 100 ms at any depth. Chunks also end
  at plan boundaries and checkpoint positions. All but the last prompt token are prefilled; the last
  goes through `forward`, as in mlx-lm. Per layer: build, `eval`, check cancel (one layer in
  flight). On cancel the partial chunk is discarded: attention offset reset to the chunk start, and
  the recurrent arrays from the chunk start (kept alive until commit; MLX arrays are immutable) are
  reinstated.
- **Rationale**: Cancel latency independent of depth; whole chunks are what 003 keeps on cancel.
  Up to 16K the split equals mlx-lm's fixed 2,048 split from position 0, so conformance (≤ 8K
  prompts) and SC-003 (16K) compare like for like.
- **Risk**: per-layer eval adds 36–48 syncs per chunk; measured by the engine bench (FR-018).
  Fallback: eval every 2 layers if overhead exceeds 5% and SC-004 still holds.

## R6: Memory budget

- **Decision**:
  1. `physical = device_info.memory_size` (Linux CPU builds: `/proc/meminfo`),
     `recommended = device_info.max_recommended_working_set_size`.
  2. `reserve = clamp(0.15 × physical, 2 GiB, 8 GiB)`; `budget = min(0.75 × physical,
     recommended) − reserve` (lower freely; higher only with `--allow-over`).
  3. Pre-load check: weights from file sizes + 2 GiB scratch estimate + recurrent state + 4,096
     tokens ≤ budget (FR-007); after load, `get_active_memory` is the measured weights.
  4. Probe: `P = min(max_position, (budget − weights − 2 GiB − R) / kv_bytes_per_token)` where
     `R = recurrent_bytes × (1 + MAX_PLAN_CHECKPOINTS)`. Run one chunk of `chunk_len(P)` at depth P
     and one 2,048-token chunk at depth min(16K, P), each with a temporary state and a live
     recurrent state; `scratch = max(peak − active_before) + one layer of attention state at P`,
     × 1.25 (the only margin). Cache in `~/.mrun/probe.toml` keyed by (identity, machine model,
     macOS, MLX version, mrun version, chunk policy).
  5. `state_capacity_bytes = budget − weights − scratch`; `kv_capacity_tokens = min(max_position,
     P, (state_capacity_bytes − R) / kv_bytes_per_token)`.
  6. `set_memory_limit(budget)`, `set_cache_limit(scratch)`, `set_wired_limit(min(budget,
     recommended))` after `synchronize`; `clear_cache()` after each prefill call and every 256
     decode tokens.
- **Worked example (M4 Pro 48 GB)**: reserve 7.2 GiB, budget ≈ 36 − 7.2 = 28.8 GiB.
  Qwen3.6 4-bit: weights ≈ 20 GiB, scratch ≈ 1–2 GiB → ≈ 7 GiB state → capped at max position
  262,144 (≈ 5 GiB of attention state + 0.25 GiB recurrent). Qwen3-Coder: ≈ 16 GiB weights →
  ≈ 10–11 GiB state ≈ 105–120K tokens. Qwen3-8B: capped at 40,960.

## R7: Forward/sample split, pipelining and greedy parity

- **Decision**: `forward(last)` builds logits for the next position and `async_eval`s them;
  `sample(step, sampling, mask, pipeline)` computes `logprobs = logits − logsumexp(logits)` in the
  logits dtype, applies the mask, samples on device, then (when `pipeline` and the allocation has
  room) enqueues `forward(sampled)` before materializing the token, mirroring mlx-lm
  `generate_step`. Greedy = `argmax(logprobs)`. The mask (002) is computed on the host while the
  next forward runs on the GPU.
- **Consequences**: `active_len` can be one ahead of the fed tokens; the generation loop trims at
  finish. Hybrid rollback of one pending step keeps the pre-step recurrent state alive until
  confirmed.
- **Sampling**: temperature, top-k (`argpartition` + `put_along_axis`), top-p (sort + cumsum),
  min-p, categorical with a per-request key from `seed`, as in mlx-lm `sample_utils.py`.
  `Sampling::validate` rejects NaN/negative temperature, top-p ∉ (0, 1], min-p ∉ [0, 1].

## R8: Tokenizer and template

- **Decision**: `tokenizers` 0.23 (`default-features = false`); `minijinja` 3 + pycompat,
  configured like transformers' environment: `trim_blocks`, `lstrip_blocks`, `raise_exception`,
  and a `tojson(indent=None)` equal to `json.dumps(x, ensure_ascii=False, indent=indent)` (no HTML
  escaping; separators `", "`/`": "`, or `","` before a newline with indent).
- **Verification**: golden fixtures rendered by transformers `apply_chat_template` at the pinned
  version (system, tools, multi-turn, tool calls and results, thinking on/off) for each template;
  text and token IDs must match (SC-009).

## R9: Threading, channels and robustness

- **Decision**: `Engine` and its types live in `mrun-engine` with no MLX; `MlxEngine` behind the
  `mlx` feature. The generation loop lives in `mrun::runtime`, generic over `Engine`, tested with
  `ScriptedEngine`. One worker thread constructs and owns the engine. Commands on an unbounded
  `flume` channel; events on `flume::bounded(64)` (sync `recv` for the CLI, `recv_async` for 002).
  A stalled consumer (send blocked for the stall timeout) is cancelled and never sent to again;
  `Finished` goes on a separate one-slot channel with `try_send` after the event sender is dropped,
  so it is always delivered. Each request runs inside `catch_unwind`; a panic → `Finished{Error}` +
  `reset()`.
- **Rationale**: crossbeam has no async receive; a full channel must not lose `Finished`; MLX
  objects stay on one thread.

## R10: Model identity

- **Decision**: `identity = sha256` over sorted `(relative path, file sha256)` for `*.safetensors`,
  `config.json`, `generation_config.json`, `tokenizer.json`, `tokenizer_config.json`,
  `special_tokens_map.json`, `chat_template.jinja` (those present). HF LFS blob names are the
  weights' sha256; small files are hashed; other paths cached in `~/.mrun/identity.toml` by
  (canonical path, size, mtime, inode). 004 reuses this function.
- **EOS**: union of `config.json` `eos_token_id`, `generation_config.json` `eos_token_id` (int or
  list) and the tokenizer's EOS token (`tokenizer_config.json` / `special_tokens_map.json`). mlx-lm
  lets generation_config replace config before adding the tokenizer EOS; the union is a superset,
  and a test checks it equals mlx-lm's set for the three models.

## R11: Engine benchmark (until 005)

- **Decision**: `crates/mrun-engine/examples/bench_engine.rs`: prefill and decode tok/s at given
  depths, cancel latency over 50 random interrupt points per depth bucket (16K, 64K, max), JSON with
  machine (chip, memory, macOS, power) and model identity. Reference: `scripts/bench_reference.sh`
  runs pinned mlx-lm (`--prefill-step-size 2048`) on the same weights. 005 absorbs these cells.

## R12: Hybrid port (`qwen3_5_moe`, Qwen3.6-35B-A3B)

- **Decision**: Port mlx-lm `qwen3_5.py` (+ `qwen3_5_moe.py` sanitize, `qwen3_next.py` attention,
  MLP, MoE, gated RMSNorm; `gated_delta.py`) line by line.
  - **Layer pattern**: layer i is linear iff `(i + 1) % full_attention_interval != 0` (interval 4):
    layers 3, 7, …, 39 are attention (10), the rest Gated DeltaNet (30). Every layer's MLP is the
    sparse MoE block when `num_experts > 0`.
  - **Gated attention**: `q_proj` outputs `2 × heads × head_dim`, split per head into queries and a
    gate; q/k RMSNorm; RoPE over `head_dim × partial_rotary_factor` dims (0.25), `rope_theta` from
    `rope_parameters` (type `default`; `mrope_section` unused for text); SDPA; `o_proj(out ×
    sigmoid(gate))`.
  - **Gated DeltaNet**: `in_proj_qkv` (→ 2·key_dim + value_dim = conv_dim), `in_proj_z`
    (value_dim), `in_proj_b`, `in_proj_a` (Hv each). Conv: concat `[conv_state, qkv]` along time,
    new conv state = last `kernel − 1` positions (`contiguous`), depthwise `conv1d` (groups =
    conv_dim, no bias, padding 0), SiLU. Split q, k, v into heads; `normalize_qk`: `q = s² ·
    rms_norm(q, eps·s²)`, `k = s · rms_norm(k, eps·s²)`, s = Dk^−½, eps 1e-6. `beta = sigmoid(b)`,
    `g = exp(−exp(A_log) · softplus(a + dt_bias))` (float32). Recurrent update by the Metal
    kernel; output `RMSNormGated(y, z)` = `precise_swiglu(y, z, rms_norm(y, w))` (float32 SiLU,
    cast back), then `out_proj`.
  - **Kernel**: mlx-lm builds it with `mx.fast.metal_kernel`; mrun binds `mlx_fast_metal_kernel_*`
    and embeds the kernel sources verbatim. With batch 1, no mask, scalar gate, Dk = 128, Dv % 8 = 0
    and float32 g/state, mlx-lm uses `gated_delta_step_packed_btree` (grid `(32, Dv/8, Hv)`,
    thread group `(32, 2, 1)`) for prefill and decode; otherwise the generic `gated_delta_step`
    (grid `(32, Dv, Hv)`, thread group `(32, 4, 1)`). Inputs `q, k, v, g, beta, state_in, T`;
    outputs `y [1, T, Hv, Dv]` (input dtype) and `state_out` (state shape, float32); template args
    `InT, StT, Dk, Dv, Hk, Hv`. Port both, plus the ops reference (`gated_delta_ops`) for CPU tests
    and the kernel-vs-ops check on Metal.
  - **MoE**: router softmax(precise) → top-8 → `norm_topk_prob` → SwitchGLU (`gather_qmm`) plus a
    shared expert scaled by `sigmoid(shared_expert_gate(x))`.
- **Rationale**: Bitwise parity needs the same kernel and the same compiled ops (R14). Principle V:
  the kernel is the reference implementation's, ported unchanged, not a new kernel.

## R13: Checkpoints and moves

- **Decision**: A checkpoint deep-copies all conv and recurrent arrays (`zeros` + `slice_update` +
  `eval`, debug pointer assertion). Prefill captures planned checkpoints at chunk ends, so no extra
  passes. `restore_checkpoint` copies (the checkpoint stays for repeated reuse, e.g. a shared
  system prompt). `split_off(at)` per layer: copy the tail into a compact buffer, copy the prefix
  into a buffer of `round_up(at, 256)`, release the old buffer (transient one layer, ≈ cost of one
  read of the state: ≈ 45 ms for 90K tokens on Qwen3-Coder). Hybrids move the live recurrent
  state into the segment (it describes `end`). `append` reverses it, freeing each segment layer as
  it is copied when consuming.
- **Rationale**: D017/D019: state moves, never doubles, so a long conversation survives a subagent.
  Copy-based restore (dropped) capped conversations at half of capacity.

## R14: Compiled ops and fused-op parity

- **Decision**: mlx-lm compiles `swiglu`, `precise_swiglu` and `compute_g` (`mx.compile`,
  shapeless). mrun binds `mlx_compile` with shapeless closures for the same functions, unless an
  on-device test proves the unfused graph bitwise-equal on all three models' shapes (then unfused
  is kept, simpler). The decision is recorded here after T006a.
