# Data Model: Engine Core

## ModelInfo (loaded model)

| Field | Type | Notes |
| --- | --- | --- |
| identity | 64-hex | [R10](research.md#r10-model-identity); same function as 004 |
| family | `qwen3_5_moe` \| `qwen3_moe` \| `gpt_oss` \| `mistral` \| `qwen3` (0.1) | family registry (010) from `config.json` `model_type` (wrapped configs: `text_config.model_type`; `mistral3`/`ministral3` → `mistral`); unknown → refused listing families |
| layers | u32 | all decoder layers |
| layer_kinds | Vec<FullAttention \| SlidingAttention{window} \| Linear> | set by the family module: qwen3_5 linear iff `(i+1) % full_attention_interval != 0`; gpt_oss from `layer_types` (default alternating sliding/full) |
| hidden, heads, kv_heads, head_dim, vocab, max_position | u32 | attention layers |
| linear | Option{k_heads, v_heads, k_dim, v_dim, conv_kernel} | linear-attention families only |
| moe | Option{experts, per_token, norm_topk_prob, shared_expert, bias} | MoE only |
| sinks | bool | per-head attention sinks (gpt_oss) |
| rope | {theta, dims (head_dim × partial_rotary_factor), scaling: None \| Yarn{…}, query_scale: Option{beta, original_max}} | YaRN for gpt_oss and mistral; position-scaled queries for mistral (`llama_4_scaling_beta`) |
| quantization | {bits, group_size, mode: affine \| mxfp4, overrides} | per-path overrides (8-bit routers, MXFP4 experts) |
| dtype | bf16 \| f16 | activations, attention state, conv state |
| kv_bytes_per_token | u64 | positional state: all attention layers (sliding included, full KV kept) × 2 × kv_heads × head_dim × dtype |
| recurrent_bytes | u64 | position-bound state: linear layers × (v_heads·v_dim·k_dim·4 + (kernel−1)·conv_dim·dtype); 0 for families without linear layers |
| eos_tokens | Vec<u32> | union (R10) |

### State kinds and bytes per family (0.1, 4-bit/MXFP4, bf16 state)

| Family (model) | Positional (KV) per token | Position-bound per checkpoint |
| --- | --- | --- |
| `qwen3_5_moe` (Qwen3.6-35B-A3B) | 10 × 2 × 2 × 256 × 2 B = 20 KiB | ≈ 61.4 MiB (30 GDN layers) |
| `qwen3_moe` (Qwen3-Coder-30B-A3B) | 48 × 2 × 4 × 128 × 2 B = 96 KiB | 0 |
| `gpt_oss` (gpt-oss-20b) | 24 × 2 × 8 × 64 × 2 B = 48 KiB (12 sliding + 12 full, full KV kept) | 0 |
| `mistral` (Devstral Small 2 24B) | 40 × 2 × 8 × 128 × 2 B = 160 KiB | 0 |
| `qwen3` (Qwen3-8B) | 36 × 2 × 8 × 128 × 2 B = 144 KiB | 0 |

Dimensions are read from `config.json` and checked at load (gpt-oss and Devstral values to confirm
on the pinned weights).

## MemoryBudget

| Field | Rule |
| --- | --- |
| physical, recommended | `device_info` `memory_size`, `max_recommended_working_set_size` |
| fraction, reserve | 0.75; `clamp(physical ÷ 6, 2 GiB, 8 GiB)` |
| budget | `min(fraction × physical, recommended) − reserve` |
| weights | file sizes before load; measured after |
| scratch | probed peak (two points) + one attention layer at P, × 1.25; cached per key (R6) |
| probed_depth | P |
| state_capacity_bytes | `budget − weights − scratch` |
| kv_capacity_tokens | `min(max_position, P, (state_capacity_bytes − recurrent_bytes × (1 + 3)) / kv_bytes_per_token)` |

Load refused (exit 4) if `weights + scratch + recurrent_bytes + 4,096 × kv_bytes_per_token > budget`.

## Sequence state (engine, live)

| Field | Invariants |
| --- | --- |
| attention[l]: K, V `[1, kv_heads, cap, head_dim]` (positional; sliding layers too) | solely owned; `cap` multiple of 256 or = reserved total; `cap ≤ min(reserved, kv_capacity_tokens)` |
| offset | `≤ cap`; positions `[0, offset)` valid; = active_len |
| linear[l]: conv `[1, kernel−1, conv_dim]`, state `[1, v_heads, v_dim, k_dim]` f32 (position-bound) | describes position `offset`; `None` after `split_off` until restore/append |
| reservation | `total` from `reserve`; growth beyond it is `ReservationExceeded` |
| pending | ≤ 1 pipelined step (token fed, not confirmed) + pre-step position-bound arrays |

Accounting invariant: `live allocation + Σ segments + Σ checkpoints = allocated() ≤
state_capacity_bytes`; the one-layer copy transient (growth, `split_off`, `append`) and the chunk
rollback copy are inside scratch, so no caller budgets them.

`trim(len)` is exact at any `len ≤ offset` for positional state (sliding layers included: the
window is applied when attending, never by dropping KV). `trim(0)` and `reserve(0, …)` always
succeed and reset position-bound state to zeros.

## Checkpoint

`{id, position, bytes = recurrent_bytes}` (≈ 61 MiB on Qwen3.6, R4): deep copy of every linear
layer's conv and recurrent state (all position-bound state) at `position`. Created by `prefill` (plan, at most
`MAX_PLAN_CHECKPOINTS` = 3 per call) or `checkpoint()`; restored by copy; dropped explicitly.
Stored checkpoints beyond the live state + 3 are evictable cache (003). Families without
position-bound state (all 0.1 families except `qwen3_5_moe`): bytes 0.

## Segment

`{id, start, end, bytes}`: attention K/V for `[start, end)` per layer in compact owned buffers,
plus (where present) the position-bound state at `end`, moved out by `split_off`. `append` requires live
`active_len == start`.

## Session (`mrun::runtime`)

The reuse policy of `mrun run`; `runtime/generate.rs` owns the per-request loop and takes either
`Session` or 003's `PrefixStore` (under `serve`).

| Field | Meaning |
| --- | --- |
| committed | prompt + output tokens whose text was fully emitted (the conversation) |
| tokens | tokens the engine state holds; always a prefix of `committed` after a finish; equals the engine's state token for token |
| checkpoints | position-bound families: the two newest `(position, CheckpointId)` of this conversation, planned at `generation_start` |

## Request and events

Request: prompt tokens, `Sampling` (validated), `max_tokens`, `stop` (≤ 4 × 256 B), optional mask
provider (002), `CancelToken` (cancel, plus a yield request that makes `prefill` return after the
current whole chunk with `Prefilled.yielded`).

States: `Admitted → Prefilling → Decoding → Finished(reason)`; any → `Cancelled`; engine error or
panic → `Failed` (state reset).

Events (bounded channel, 64): `Started {prompt, cached, max_tokens}`, `Progress {done, total}`,
`Token {id, text}`. Then exactly one `Finished {reason: Stop{stop} | Length | Cancelled |
Rejected(msg) | Error(msg), prefilled, stats}` on a one-slot channel. `prefilled` = prompt tokens
held in state at the end (whole chunks).

Stats: prompt_tokens, cached_tokens, generated_tokens, max_tokens (applied), ttft (first sampled
token), prefill_tps, decode_tps.
