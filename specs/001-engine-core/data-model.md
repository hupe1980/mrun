# Data Model: Engine Core

## ModelSpec (loaded model)

| Field | Type | Notes |
| --- | --- | --- |
| identity | 64-hex | from 004 manifest (weights + tokenizer + template hashes) |
| architecture | `qwen3` \| `qwen3_moe` | from `config.json` `model_type` |
| layers, hidden, heads, kv_heads, head_dim, vocab | u32 | config |
| experts, experts_per_token, norm_topk_prob | Option | MoE only |
| rope_theta, rms_eps, tie_embeddings, max_position | | config |
| quantization | {bits, group_size, mode, overrides} | per-path overrides allowed |
| kv_dtype | bf16 \| f16 | activation dtype |
| kv_bytes_per_token | u64 | layers × 2 × kv_heads × head_dim × dtype size |

## MemoryBudget

| Field | Type | Rule |
| --- | --- | --- |
| physical | bytes | `hw.memsize` |
| recommended_working_set | bytes | `device_info` |
| fraction, reserve | f32, bytes | defaults 0.75, 8 GiB |
| budget | bytes | `min(fraction × physical, recommended) − reserve` (≥ 0) |
| weights | bytes | measured after load |
| scratch | bytes | measured worst-case chunk × 1.25 |
| kv_capacity_tokens | u32 | `(budget − weights − scratch) / kv_bytes_per_token` |

Load fails (exit 4) if `weights + scratch + kv_bytes_per_token × min_context (4,096)` > budget.

## Session (one active)

| Field | Type |
| --- | --- |
| tokens | Vec<u32> (rendered history incl. generated) |
| kv | KvCache (per-layer K,V; offset; cap) |
| reservation | tokens reserved = prompt + max_tokens |

State: `Idle → Prefilling(chunk i/n) → Decoding → Finished(reason)`; any state → `Cancelled` on
cancel; `Failed` on engine error (session discarded, engine stays usable).

## KvCache

| Field | Type | Invariants |
| --- | --- | --- |
| keys, values | per-layer arrays `[1, kv_heads, cap, head_dim]` | solely owned by the cache |
| offset | u32 | ≤ cap; tokens `[0, offset)` valid |
| cap | u32 | multiple of 256, ≤ reservation |

Operations: `append(chunk)`, `trim(n)`, `snapshot(range) -> KvSegment` (owned copy),
`restore(&[KvSegment])` (copy into fresh buffers).

## GenerateRequest / Event

Request: `tokens`, `sampling {temperature, top_p, top_k, min_p, seed}`, `max_tokens`,
`stop: Vec<String>`, `mask_provider: Option<..>` (002), `cancel: CancelToken`.

Events: `Started {cached, processed}`, `PrefillProgress {done, total}`, `Token {id, text}`,
`Finished {reason: Stop|Length|Cancelled|Error, stats}` — exactly one `Finished`.

Stats: prompt_tokens, cached_tokens, generated_tokens, ttft_ms, prefill_tps, decode_tps.
