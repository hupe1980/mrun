# Data Model: Prefix Cache

## CacheKey

`{weights_hash, tokenizer_hash, template_hash, state_format}` (001 model identity). One tree per
key; a new key discards the old tree (FR-014).

## Prompt (002's `RenderedPrompt`, the prompt of 001's `GenerateRequest`)

| Field | Type | Notes |
| --- | --- | --- |
| tokens | Vec<u32> | P = len |
| messages | Vec<MessageMark{start: u32, role}> | message starts and roles; each mark maps to the API message it renders (Harmony split calls map to their API message; mid-conversation system messages keep their API index), so reports give API indices |
| system_tools_end | u32 | end of system + tools region |
| generation_start | u32 | start of the generation header (≤ P − 1); checkpoint position for position-bound state |

## PrefixStore configuration

`{key: CacheKey, mode: Cached | Off}`: `--no-cache` → Off for the whole server (no per-request
opt-out).

## Node

| Field | Type | Notes |
| --- | --- | --- |
| start | u32 | position where the node attaches to its parent (the live path or another node) |
| tokens | Vec<u32> | the node's positions `[start, start + len)` |
| state | Live \| Stored(Segment) | Live = the trunk, held by the sequence state; a stored node is exactly one 001 segment |
| checkpoints | Vec<Checkpoint> | positions within this edge |
| messages | Vec<MessageMark> | marks starting within this edge |
| evicted | small map first_token → {count, reason: Lru \| Clear \| Tiny \| Pressure} | removed children |
| last_used | Instant | set on match, move and commit |

A child may attach at any position of its parent, so nothing is ever split for bookkeeping: a
node is split only by a real `split_off` when a request's match ends inside it (revised at
implementation, 2026-10-15; replaces refcounted segment views).

## Checkpoint

`{pos: u32, id: CheckpointId (001), bytes: u64 (≈ 61 MiB on Qwen3.6), kind: SystemTools |
LastMessage | GenerationStart}`. A moved-out tail's end state is not a checkpoint: it stays inside
its 001 segment (the node's segment), is restored by `append`, and does not count
toward the cap of 8.

Drop rule when a path exceeds 8: drop the shallowest checkpoint other than the first.

## Plan (lookup result)

| Field | Notes |
| --- | --- |
| matched | longest exact match m |
| fork | depth where the matched path leaves the live path |
| move_out | Option<at: u32> (= fork, or divergence inside the live path) and whether the tail is ≥ 256 |
| move_in | nodes from fork to the node containing m; whole buffers |
| split_back | Option<m> when m ends inside a moved-in buffer |
| resume | m (positional state only, sliding-window layers included) or deepest checkpoint ≤ m (position-bound state) |
| checkpoint | Option<CheckpointId> to restore |
| miss | MissReport |

## MissReport

| Field | Type | Notes |
| --- | --- | --- |
| prompt | u32 | P |
| cached | u32 | `min(resume, P − 1)` |
| processed | u32 | `P − cached` |
| divergence | Option<u32> | first differing position against the closest stored path; None when the request extends it |
| message | Option<{index: u32, role}> | the API message containing `divergence`: its index in the request's `messages`/`input` array (not the rendered mark index) and role; a top-level system prompt or tool list reports role `system`/`tools` |
| gap | u32 | `matched − resume` (> 0 only with position-bound state: the old `checkpoint` case) |
| cause | Option<Cause> | None on a full hit |

| Cause | Meaning |
| --- | --- |
| `disabled` | `--no-cache` |
| `model` | state existed under a different cache key (model or template file changed) |
| `evicted` | the request continues into a child removed by LRU, `cache clear`, the tiny-tail rule or memory pressure |
| `changed` | the divergence lies inside the stored system+tools region, or at least one conversation message matched before it: a system prompt, tool list or message was changed, removed, replaced or re-rendered by the template (the message index and role say which; 005 labels template re-renders) |
| `new` | the request shares at most the system+tools region with stored state (a new conversation), or nothing is stored |

Precedence: `disabled` > `model` > `evicted` > `changed` > `new`; `gap` is reported whatever the
cause. Lean cut 2026-10-15: `system` and `edit` → `changed`; `capacity` → `evicted`; `checkpoint`
→ `gap > 0`; `template` → 005 diagnostic.

## CacheStats (`/mrun/status`, `mrun ps`)

`{live_tokens, branches, branch_tokens, buffer_bytes, checkpoints, checkpoint_bytes,
prompt_tokens_total, cached_tokens_total, evicted_total, pressure_events, last: MissReport}`

## Invariants (property-tested)

1. Children of a node have distinct first tokens; the live path is one root path.
2. Every stored node's ancestors are stored or live; positions along a path are contiguous.
3. Each stored node is one segment owned by that node alone; a child attaches at a position within
   its parent's range.
4. `live_bytes + Σ buffer bytes + Σ checkpoint bytes ≤ capacity bytes` before and after every
   operation; during a move the excess is at most one layer of the moved buffer.
5. The live path is never evicted while a request runs; after `prepare` no node is pinned.
6. Checkpoints per root-to-leaf path ≤ 8 (segment end states not counted); every checkpoint
   position lies on its node's edge.
7. Reused tokens equal the request's tokens under the same key (mutation suite oracle).
8. Committed live path = prompt tokens + emitted output, cut to whole committed chunks on cancel.
