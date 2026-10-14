# Data Model: Prefix Cache

## CacheKey

`{model_identity (001 R10), kv_dtype, template_hash}` — one radix tree per key.

## PrefixNode

| Field | Type | Notes |
| --- | --- | --- |
| edge | Vec<u32> | token slice from parent to this node |
| depth | u32 | tokens from root to end of edge |
| segment | KvSegmentId | owned KV for exactly this edge's tokens (always present) |
| bytes | u64 | segment size (from `Engine::snapshot`) |
| kind | System \| User \| Tool \| Assistant \| Periodic | eviction order |
| text_hash | Option<u64> | hash of the message text ending at this node (miss classification) |
| last_used | Instant | LRU |
| in_use | u32 | pinned while > 0 (only between lookup and the end of `prepare`) |
| state_checkpoint | Option<StateId> | reserved for hybrid models (FR-012); always None now |

Invariants (property-tested):
- children have distinct first tokens;
- every node on any root path owns a segment for its edge, so restore along a path is contiguous;
- a split at token p of an edge copies the segment into [start, p) and [p, end) and drops the
  original;
- only leaves with `in_use == 0` are evicted; ancestors of pinned nodes are never evicted;
- `sum(bytes) + active_cache_bytes ≤ kv_capacity_bytes`.

## LivePath

The active cache viewed as an unmaterialized path: tokens, boundary positions and kinds. Its
boundary prefixes are materialized into nodes only when the live path is replaced (R2).

## Lookup

`{matched: u32, on_live_path: bool, nodes: Vec<NodeId> (pinned), miss: MissReport}`

## MissReport

| Field | Type |
| --- | --- |
| prompt_tokens | u32 |
| cached_tokens | u32 (reused) |
| closest_cached | u32 (longest prefix known for this key, incl. ghosts) |
| divergence | Option<u32> (token position) |
| cause | new \| edit \| template \| volatile \| evicted \| capacity \| model \| disabled \| none |

## CacheStats (`mrun ps`, `/mrun/status`)

`{segments, bytes, live_tokens, hit_tokens_total, prompt_tokens_total, hit_rate, evictions,
pressure_events, last_miss: MissReport}`
