# Data Model: Engine Core

## ModelInfo (loaded model)

| Field | Type | Notes |
| --- | --- | --- |
| identity | 64-hex | [R10](research.md#r10-model-identity); same function as 004 |
| architecture | `qwen3` \| `qwen3_moe` \| `qwen3_5_moe` | `config.json` `model_type` |
| layers | u32 | all decoder layers |
| layer_kinds | Vec<Attention \| Linear> | qwen3_5: linear iff `(i+1) % full_attention_interval != 0` |
| hidden, heads, kv_heads, head_dim, vocab, max_position | u32 | attention layers |
| linear | Option{k_heads, v_heads, k_dim, v_dim, conv_kernel} | hybrid only |
| moe | Option{experts, per_token, norm_topk_prob, shared_expert} | MoE only |
| rope | {theta, dims (head_dim × partial_rotary_factor)} | |
| quantization | {bits, group_size, mode, overrides} | |
| dtype | bf16 \| f16 | activations, attention state, conv state |
| kv_bytes_per_token | u64 | attention layers × 2 × kv_heads × head_dim × dtype |
| recurrent_bytes | u64 | linear layers × (v_heads·v_dim·k_dim·4 + (kernel−1)·conv_dim·dtype); 0 for attention-only |
| eos_tokens | Vec<u32> | union (R10) |

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
| attention[l]: K, V `[1, kv_heads, cap, head_dim]` | solely owned; `cap` multiple of 256 or = reserved total; `cap ≤ min(reserved, kv_capacity_tokens)` |
| offset | `≤ cap`; positions `[0, offset)` valid; = active_len |
| linear[l]: conv `[1, kernel−1, conv_dim]`, state `[1, v_heads, v_dim, k_dim]` f32 | describes position `offset`; `None` after `split_off` until restore/append |
| reservation | `total` from `reserve`; growth beyond it is `ReservationExceeded` |
| pending | ≤ 1 pipelined step (token fed, not confirmed) + pre-step recurrent arrays |

Accounting invariant: `live allocation + Σ segments + Σ checkpoints = allocated() ≤
state_capacity_bytes`; the one-layer copy transient and chunk rollback copy are inside scratch.

## Checkpoint

`{id, position, bytes = recurrent_bytes}`: deep copy of every linear layer's conv and recurrent
state at `position`. Created by `prefill` (plan) or `checkpoint()`; restored by copy; dropped
explicitly. Attention-only models: bytes 0.

## Segment

`{id, start, end, bytes}`: attention K/V for `[start, end)` per layer in compact owned buffers,
plus (hybrid) the recurrent state at `end`, moved out by `split_off`. `append` requires live
`active_len == start`.

## Session (`mrun::runtime`)

| Field | Meaning |
| --- | --- |
| committed | prompt + output tokens whose text was fully emitted (the conversation) |
| tokens | tokens the engine state holds; always a prefix of `committed` after a finish; equals the engine's state token for token |
| checkpoints | hybrid: the two newest `(position, CheckpointId)` of this conversation |

## Request and events

Request: prompt tokens, `Sampling` (validated), `max_tokens`, `stop` (≤ 4 × 256 B), optional mask
provider (002), `CancelToken`.

States: `Admitted → Prefilling → Decoding → Finished(reason)`; any → `Cancelled`; engine error or
panic → `Failed` (state reset).

Events (bounded channel, 64): `Started {prompt, cached, max_tokens}`, `Progress {done, total}`,
`Token {id, text}`. Then exactly one `Finished {reason: Stop{stop} | Length | Cancelled |
Rejected(msg) | Error(msg), prefilled, stats}` on a one-slot channel. `prefilled` = prompt tokens
held in state at the end (whole chunks).

Stats: prompt_tokens, cached_tokens, generated_tokens, max_tokens (applied), ttft (first sampled
token), prefill_tps, decode_tps.
