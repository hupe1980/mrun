# Data Model: Engine Core

## ModelInfo (loaded model)

| Field | Type | Notes |
| --- | --- | --- |
| identity | 64-hex | [research R10](research.md#r10-model-identity-without-feature-004); same function as 004 |
| architecture | `qwen3` \| `qwen3_moe` | from `config.json` `model_type` |
| layers, hidden, heads, kv_heads, head_dim, vocab | u32 | config |
| experts, experts_per_token, norm_topk_prob | Option | MoE only |
| rope_theta, rms_eps, tie_embeddings, max_position | | config |
| quantization | {bits, group_size, mode, overrides} | per-path overrides allowed |
| kv_dtype | bf16 \| f16 | activation dtype |
| kv_bytes_per_token | u64 | layers × 2 × kv_heads × head_dim × dtype size |
| template, eos_tokens | | from tokenizer files |

## MemoryBudget

| Field | Type | Rule |
| --- | --- | --- |
| physical | bytes | `hw.memsize` |
| recommended_working_set | bytes | `device_info` |
| fraction, reserve | f32, bytes | defaults 0.75, 8 GiB (other apps + mrun host heap) |
| budget | bytes | `min(fraction × physical, recommended) − reserve` (≥ 0) |
| weights | bytes | estimated from files before load, measured after |
| scratch | bytes | (probed peak + per-layer KV growth transient) × 1.25 (margin applied only here), cached per identity/machine/version |
| kv_capacity_tokens | u32 | `min(max_position, (budget − weights − scratch) / kv_bytes_per_token)` |

Load refused (exit 4) if `weights + scratch + 4,096 × kv_bytes_per_token > budget`.
Worked example: [research R6](research.md#r6-memory-budget).

## KV accounting (shared with 003)

- `active_tokens`: tokens of the active conversation in the active cache (allocated `cap ≥ active`).
- `cached_tokens`: owned segments in the prefix store (003); pinned only while being copied into
  the active cache, otherwise evictable.
- Admission (one request at a time): `prompt + max_output ≤ kv_capacity_tokens`. Memory held by
  cached segments counts as available and is reclaimed on demand: at admission for the initial
  allocation, and later whenever the active cache grows (FR-009). Without 003, `cached = 0`.
- Invariant: `active cap bytes + cached bytes ≤ kv_capacity_bytes` at all times (the one-layer
  growth transient lives in the scratch envelope). During a restore, the matched segments and their
  copy in the active cache coexist, so a restore needs `active allocation + matched bytes ≤
  kv_capacity_bytes`; when that does not fit after evicting other entries, fewer segments are
  restored (003 R3).

## KvCache (active)

| Field | Type | Invariants |
| --- | --- | --- |
| keys, values | per-layer arrays `[1, kv_heads, cap, head_dim]` | solely owned (no other handle) |
| offset | u32 | ≤ cap; tokens `[0, offset)` valid |
| cap | u32 | multiple of 256; grows in 4,096-token steps, one layer at a time (transient ≤ one layer's K/V, inside the scratch envelope) |

## KvSegmentId

Opaque handle to an owned deep copy of a token range of the active cache, with `bytes` known.
Used by 003; dropped explicitly.

## Session and generation (in `mrun::runtime`)

| Field | Type |
| --- | --- |
| tokens | Vec<u32> committed conversation tokens (prompt + shown output) |
| request | prompt tokens, sampling {temperature, top_p, top_k, min_p, seed}, max_output, stop (≤ 4 × 256 B), mask provider (002), cancel token |

States: `Idle → Prefilling(layer/chunk progress) → Decoding → Finished(reason)`; any state →
`Cancelled`; `Failed` on engine error (active cache trimmed to committed tokens or reset). After
any finish the active cache is trimmed to `tokens.len()` (FR-004).

Events: `Started {prompt, cached}`, `Progress {done, total}`, `Token {id, text}`,
`Finished {reason: Stop|Length|Cancelled|Error, prefilled, stats}` — exactly one `Finished`;
`prefilled` reports how far prompt processing got (used by 003 on cancel).

Stats: prompt_tokens, cached_tokens, generated_tokens, ttft_ms, prefill_tps, decode_tps.
