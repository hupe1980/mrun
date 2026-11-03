+++
title = "Prefix cache"
description = "How mrun reuses model state exactly across turns, subagents and tool calls, for attention and hybrid models, splices the model's own replies back into the next prompt so tool loops always hit, and explains every cache miss."
weight = 40
+++

## What gets reused

When a model reads a prompt it produces state for every token. The next request that starts with
the same tokens can reuse that state instead of computing it again. Reuse is exact: the same token
ids and the same model identity (weights, tokenizer, chat template, state format). Nothing is
approximated. The cache lives in memory only and does not survive a server restart.

- **The live conversation stays in place.** The next turn reuses it without any copy; only the
  new tokens are processed.
- **Other conversations become branches.** When a different request arrives (a subagent, a title
  request, another session, an edited history), the shared beginning stays where it is and the
  rest of the old conversation is *moved* aside as a branch, if it is at least 256 tokens long;
  a shorter tail is dropped. Switching back moves it in again. State is never held twice.
- **Holder chains.** A branch stays usable while its beginning is held by the live conversation or
  by another branch. Example: a subagent shares the system prompt with the main conversation, so
  the main conversation is moved aside from the end of the system prompt. A title request then
  shares nothing, so the live state is moved aside from position 0 as another branch, which now
  holds the system prompt. The main conversation stays reachable through that branch, and its next
  turn moves both back in, in order.
- **Checkpoints for hybrid models.** Linear-attention layers (Qwen3.5, Qwen3.6) keep a
  fixed-size state that exists only for the position it was computed at. While processing a
  prompt, mrun saves checkpoints of it at points that mean the same in every API: the end of the
  system prompt and tools, the start of each user message that is more than tool results, the
  start of the last tool result (a retried tool call resumes there), the start of the reply (the
  next turn re-renders the reply from there) and one token before the end of the prompt (retries
  and regenerations). Checkpoints may use up to a quarter of the state memory. When they would
  exceed it, the checkpoint closest to its predecessor goes first; the system-and-tools
  checkpoint, the newest user message's and the newest reply's are protected as long as the
  budget holds them. An edited or branching conversation resumes from the nearest checkpoint before the
  change.
- **Progress of interrupted prompts.** If a client gives up during a long prompt and retries, mrun
  resumes from the last completed chunk.

## Tool loops always hit

Chat templates re-render an assistant turn differently from what the model generated: Qwen
templates strip whitespace, and re-tokenizing text rarely yields the ids the model sampled.
Without help, every turn of a tool loop would miss at the start of the last reply.

mrun records each finished reply with the part of the cache that holds its tokens: the live
conversation or the branch it was moved to. The record lives exactly as long as those tokens, so
a long subagent loop cannot push out the main conversation's records. It keeps the reply's
canonical form: reasoning and text without surrounding whitespace, tool calls with their
arguments as canonical JSON, so re-serialized arguments still match.

Splicing follows what the template renders:

- **The current turn** (everything after the last real user message, so the whole tool loop):
  each assistant message whose canonical form matches the record at that point is replaced by the
  token ids the model produced, reasoning included. If your client dropped the reasoning, the
  record restores it, because the model saw it when it wrote the reply.
- **Earlier turns** are rendered by the template as it would anyway; Qwen templates drop earlier
  reasoning once a new user message arrives. Where that rendering differs from what the model
  processed, the next request misses there, and the report says `diverged` at that assistant
  message.

Only ids the model itself produced after exactly that prefix are used, so the result stays exact.
If your client edits a reply in the current turn, that reply no longer matches and is rendered by
the template as usual.

Spliced tokens count as ordinary prompt tokens (cached when they match) and are reported per
response in `x-mrun-spliced` and in total as `spliced_tokens_total` in
[`/mrun/status`](@/docs/api.md#mrun-status). `count_tokens` counts the spliced prompt.

Example (M4 Pro, Qwen3.6): a four-turn Chat Completions tool loop hits the cache on every turn
after the first (cached = prompt − about 22 tokens), with 31, 85 and 139 tokens spliced on turns
two to four.

## Reading the numbers

Every response reports how many prompt tokens were reused in each API's own usage field
(`cache_read_input_tokens` for Anthropic, `prompt_tokens_details.cached_tokens` for OpenAI).
mrun also attaches its own report: `x-mrun-cache-*` headers on non-streaming responses and a
`: mrun {…}` comment before the first event of a stream (see [API](@/docs/api.md#mrun-metadata)).
`mrun ps` shows totals and the last miss. The totals are also available from the Prometheus
`/metrics` endpoint, and `mrun ps --watch` shows each running request live.

## Why a miss happened

A miss names one of five causes, the token position where the prompt left the cached state, and
the message that position falls in: its index in your request's `messages` (or `input`) array and
its role, with role `system` for the system prompt and tool list. That tells you what changed
without logging any prompt text.

| Cause | Meaning | What to do |
| --- | --- | --- |
| `new` | The prompt left the cached state before the end of its first conversation message: a new conversation, a subagent, a title request, or a changed system prompt or tool list | Nothing for new conversations; if the system prompt or tools change every turn (message 0, role `system`), check what your client changes there |
| `diverged` | At least one whole conversation message matched, then the prompt differs: an edited, removed or replaced message, or an earlier turn the template rendered differently from what the model processed | Expected after edits and retries, and once per new user message where the template drops earlier reasoning; if it happens every turn at the same message, check what your client changes there |
| `evicted` | The request continues state that was dropped: for memory, under memory pressure, or a tail too short to keep as a branch | More memory, or fewer parallel conversations |
| `gap` | A hybrid model had to resume from a checkpoint more than 256 tokens before the match | Usually after an edit far from a checkpoint; nothing to do |
| `disabled` | The server runs with `--no-cache` | — |

Every report, hit or miss, also gives **`unreused`**: tokens the prompt shares with state mrun
holds or recently dropped that were computed again anyway. It separates a client that changed its
prompt (high `diverged`, low `unreused`) from work the cache could have saved (high `unreused`).
mrun remembers the last 1024 dropped conversations for this, and says when older ones have been
forgotten (`x-mrun-cache-ring-wrapped: true`, `ring_wrapped` in a stream's `: mrun` comment). For
hybrid models the report also gives the gap that was computed again between the restored
checkpoint and the match (`x-mrun-cache-gap`).

## Eviction

Branches share the memory budget with the live conversation. When a request needs room, mrun
drops branches in this order:

1. one-shot or stale branches, least recently used first;
2. **protected** branches: conversations that ran two or more turns and were used within the last
   16 requests, such as the main conversation a subagent returns to. A conversation keeps its turn
   count when it is moved aside and back, so it stays protected over any number of subagent
   round trips;
3. checkpoints, the one closest to its predecessor first, never the one the request resumes from.

The live conversation is never evicted while it runs. Under memory pressure from macOS, branches
go first (see [Memory](@/docs/memory.md#under-pressure)).

## Exactness

A request served from the cache gets bit-identical output to recomputing its prompt from empty
state. The chunk plan (where a prompt is split for processing) depends only on the prompt, never
on what is cached or on other traffic, so a cached and a cold run split it alike.

`--no-cache` output can differ from cached output in the low bits. A cached conversation contains
replies the model computed one token at a time, while `--no-cache` reads them back as part of a
prompt, and MLX's decode and prefill kernels round differently. Neither is approximate.

## Controls

| Control | Effect |
| --- | --- |
| `mrun serve --no-cache` | Disable reuse for the whole server (every miss reports `disabled`) |
| `mrun cache clear` | Drop all cached state of the running server, including the live conversation (after the running request, if any). Prints `cleared N branches (M MiB) and the live conversation`. It also forgets what was evicted before, so the next request reports `new`, not `evicted` |
