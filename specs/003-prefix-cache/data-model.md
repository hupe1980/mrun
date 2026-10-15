# Data Model: Prefix Cache

## CacheKey

`{weights_hash, tokenizer_hash, template_hash, state_format}` (001 model identity). One tree per
key; a new key discards the old tree (FR-014).

## Prompt (from 002's renderer)

| Field | Type | Notes |
| --- | --- | --- |
| tokens | Vec<u32> | P = len |
| messages | Vec<MessageMark{start: u32, role, content_hash: u64}> | content hash: role, visible text, tool calls (canonical JSON); reasoning excluded |
| system_tools_end | u32 | end of system + tools region |
| generation_start | u32 | start of the generation header (≤ P − 1) |
| mode | Cached \| Off | `x-mrun-cache: off` or `--no-cache` → Off |

## Node

| Field | Type | Notes |
| --- | --- | --- |
| edge | Vec<u32> | tokens from parent to this node |
| depth | u32 | tokens from root to end of edge |
| state | Live \| Stored(SegmentView) | Live = held by the sequence state |
| checkpoints | Vec<Checkpoint> | positions within this edge |
| messages | Vec<MessageMark> | marks starting within this edge |
| evicted | small map first_token → {count, reason: Lru \| Clear \| Tiny \| Pressure} | removed children |
| last_used | Instant | set on match, move and commit |
| soft_pins | u16 | queued requests matching through this node |

`SegmentView {buffer: BufferId, start, end}`; `Buffer {per-layer arrays, bytes, views: u32}` is
freed when its last view drops.

## Checkpoint

`{pos: u32, id: CheckpointId (001), bytes: u64, kind: SystemTools | LastMessage | PromptEnd | TailEnd}`

## Plan (lookup result)

| Field | Notes |
| --- | --- |
| matched | longest exact match m |
| fork | depth where the matched path leaves the live path |
| move_out | Option<at: u32> (= fork, or divergence inside the live path) and whether the tail is ≥ 256 |
| move_in | nodes from fork to the node containing m; whole buffers |
| split_back | Option<m> when m ends inside a moved-in buffer |
| resume | m (attention) or deepest checkpoint ≤ m (hybrid) |
| checkpoint | Option<CheckpointId> to restore |
| miss | MissReport |

## MissReport

| Field | Type | Notes |
| --- | --- | --- |
| prompt | u32 | P |
| cached | u32 | `min(resume, P − 1)` |
| processed | u32 | `P − cached` |
| divergence | Option<u32> | first differing position against the closest stored path; None when the request extends it |
| gap | u32 | `matched − resume` (0 for attention models) |
| cause | Option<Cause> | None on a full hit |

| Cause | Meaning |
| --- | --- |
| `disabled` | cache off globally or for this request |
| `model` | state existed under a different cache key (model or template file changed) |
| `evicted` | the request continues into a child removed by LRU, `cache clear` or the tiny-tail rule |
| `capacity` | the request continues into a child removed under memory pressure |
| `template` | all messages up to the divergence have equal API content but render to different tokens |
| `system` | the divergence lies inside the stored conversation's system+tools region: the system prompt or tool list changed (for example an MCP server connected) |
| `edit` | at least one conversation message matched before the divergence: a message was changed, removed or replaced (includes an edited or retried last turn) |
| `new` | the request shares at most the system+tools region (a new conversation), or nothing is stored |
| `checkpoint` | hybrid; no divergence cause, but resume < match (gap re-prefilled) |

Precedence: `disabled` > `model` > `evicted`/`capacity` > `template` > `system` > `edit` > `new` >
`checkpoint`; `gap` is reported whatever the cause.

## CacheStats (`/mrun/status`, `mrun ps`)

`{live_tokens, branches, branch_tokens, buffer_bytes, checkpoints, checkpoint_bytes,
prompt_tokens_total, cached_tokens_total, evicted_total, pressure_events, last: MissReport}`

## Invariants (property-tested)

1. Children of a node have distinct first tokens; the live path is one root path.
2. Every stored node's ancestors are stored or live; positions along a path are contiguous.
3. Views of one buffer lie on one root path and are contiguous; an edge split creates views only.
4. `live_bytes + Σ buffer bytes + Σ checkpoint bytes ≤ capacity bytes` before and after every
   operation; during a move the excess is at most one layer of the moved buffer.
5. The live path is never evicted while a request runs; after `prepare` no node is hard-pinned.
6. Checkpoints per root-to-leaf path ≤ 8; every checkpoint position lies on its node's edge.
7. Reused tokens equal the request's tokens under the same key (mutation suite oracle).
8. Committed live path = prompt tokens + emitted output, cut to whole committed chunks on cancel.
