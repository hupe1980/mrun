# Data Model: Prefix Cache

## CacheKey

`{model_identity (004 manifest identity), kv_dtype, template_hash, normalization: Option<profile>}`
— one radix tree per key.

## PrefixNode

| Field | Type | Notes |
| --- | --- | --- |
| edge | Vec<u32> | token slice from parent to this node |
| depth | u32 | tokens from root to end of edge |
| segment | Option<KvSegmentId> | KV bytes for this edge's tokens (owned copy in the engine) |
| bytes | u64 | segment size |
| recompute_s | f32 | estimated prefill time saved |
| last_used | Instant | |
| in_use | u32 | pinned while > 0 |
| boundary | MessageEnd \| Periodic | why the snapshot exists |
| state_checkpoint | Option<StateId> | reserved for hybrid models (FR-012); always None now |

Invariants: children have distinct first tokens; a node with `in_use > 0` and its ancestors are
not evicted; sum of `bytes` ≤ cache capacity = KV capacity − active reservation.

## Lookup result

`{matched: u32, segments: Vec<KvSegmentId>, hot: bool, miss: MissReport}`

## MissReport

| Field | Type |
| --- | --- |
| prompt_tokens | u32 |
| cached_tokens | u32 (reused) |
| closest_cached | u32 (longest prefix ever seen for this key, if still known) |
| divergence | Option<u32> (token position) |
| cause | new \| edit \| template \| volatile \| evicted \| capacity \| model \| disabled \| none |
| normalized | Option<profile> |
| excerpt | Option<String> (only with `--debug-cache`) |

## CacheStats (`mrun ps`, `/mrun/status`)

`{entries, bytes, capacity_bytes, hit_tokens_total, prompt_tokens_total, hit_rate, evictions,
pressure_events, last_miss: MissReport}`
