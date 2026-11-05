+++
title = "Prefix cache"
description = "How mrun reuses model state exactly across turns, subagents and tool calls, splices the model's own replies back in, and explains every cache miss."
weight = 40
+++

## What gets reused

The next request that starts with the same tokens reuses the state computed for them. Reuse is
exact: the same token ids and the same model identity (weights, tokenizer, chat template, state
format). The cache lives in memory only and does not survive a restart.

- **The live conversation stays in place.** The next turn reuses it without a copy; only new
  tokens are processed.
- **Other conversations become branches.** When a different request arrives (a subagent, a title,
  another session, an edited history), the shared beginning stays and the rest of the old
  conversation is moved aside as a branch if it is at least 256 tokens long; a shorter tail is
  dropped. Switching back moves it in again. State is never held twice.
- **Holder chains.** A branch stays usable while its beginning is held by the live conversation or
  another branch. If a title request shares nothing, the live state is moved aside from position 0
  and the main conversation stays reachable through it; its next turn moves both back in.
- **Checkpoints for hybrid models.** Linear-attention state (Qwen3.5, Qwen3.6) exists only for the
  position it was computed at. mrun saves checkpoints at the end of the system prompt and tools,
  the start of each user message that is more than tool results, the start of the last tool
  result, the start of the reply, and one token before the end of the prompt. Checkpoints use at
  most a quarter of the state memory; when full, the one closest to its predecessor goes first,
  and the system-and-tools, newest user message and newest reply checkpoints are kept while the
  budget allows. An edited conversation resumes from the nearest checkpoint before the change.
- **Interrupted prompts.** A client that gives up during a long prompt and retries resumes from
  the last completed chunk.

## Tool loops always hit

Chat templates re-render an assistant turn differently from what the model generated (Qwen
templates strip whitespace, and re-tokenized text rarely yields the sampled ids). mrun records
each finished reply with the cache state that holds its tokens, for as long as those tokens live.
The record keeps the reply's canonical form: reasoning and text without surrounding whitespace,
tool call arguments as canonical JSON.

- **The current turn** (everything after the last real user message, so the whole tool loop):
  each assistant message that matches its record is replaced by the token ids the model produced,
  reasoning included, also when the client dropped the reasoning.
- **Earlier turns** are rendered by the template; Qwen templates drop earlier reasoning once a new
  user message arrives. Where that differs from what the model processed, the next request misses
  there with `diverged` at that assistant message.

Only ids the model produced after exactly that prefix are used, so the result stays exact. A reply
the client edited is rendered by the template as usual.

Spliced tokens count as ordinary prompt tokens and are reported in `x-mrun-spliced` and as
`spliced_tokens_total` in [`/mrun/status`](@/docs/api.md#mrun-status). `count_tokens` counts the
spliced prompt.

## Reading the numbers

Each response reports reused prompt tokens in its API's usage field (`cache_read_input_tokens`
for Anthropic, `prompt_tokens_details.cached_tokens` for OpenAI), plus mrun's report in
`x-mrun-cache-*` headers or a `: mrun {…}` comment before the first stream event (see
[API](@/docs/api.md#mrun-metadata)). `mrun ps` shows totals and the last miss, `/metrics` the
totals for Prometheus, and `mrun ps --watch` each running request.

## Why a miss happened

A miss names its cause, the token position where the prompt left the cached state, and the
message there: its index in the request's `messages` (or `input`) and its role (`system` for the
system prompt and tools). No prompt text is logged.

| Cause | Meaning | What to do |
| --- | --- | --- |
| `new` | The prompt left the cached state before the end of its first message: a new conversation, a subagent, a title request, or a changed system prompt or tool list | Nothing for new conversations; if it happens every turn at message 0, role `system`, check what your client changes there |
| `diverged` | At least one whole message matched, then the prompt differs: an edited, removed or replaced message, or an earlier turn rendered differently from what the model processed | Expected after edits, retries and once per new user message; if it happens every turn at the same message, check your client |
| `evicted` | The request continues state that was dropped for memory, under memory pressure, or as a tail too short to keep | More memory, or fewer parallel conversations |
| `gap` | A hybrid model resumed from a checkpoint more than 256 tokens before the match | Nothing |
| `disabled` | The server runs with `--no-cache` | — |

Every report also gives **`unreused`**: tokens shared with state mrun holds or recently dropped
that were computed again anyway. A client that changed its prompt shows high `diverged` and low
`unreused`; work the cache could have saved shows high `unreused`. mrun remembers the last 1024
dropped conversations for this and flags when older ones are forgotten
(`x-mrun-cache-ring-wrapped: true`, `ring_wrapped` in the stream comment). For hybrid models,
`x-mrun-cache-gap` gives the tokens recomputed between the restored checkpoint and the match.

## Eviction

Branches share the memory budget with the live conversation. When a request needs room, mrun
drops, in order:

1. one-shot or stale branches, least recently used first;
2. **protected** branches: conversations with two or more turns used within the last 16 requests,
   such as the main conversation a subagent returns to;
3. checkpoints, the one closest to its predecessor first, never the one the request resumes from.

The live conversation is never evicted while it runs. Under memory pressure, branches go first
(see [Memory](@/docs/memory.md#under-pressure)).

## Exactness

A request served from the cache gets bit-identical output to recomputing its prompt from empty
state. The chunk plan depends only on the prompt, never on what is cached or on other traffic.

`--no-cache` output can differ in the low bits: a cached conversation holds replies computed one
token at a time, while `--no-cache` reads them back as a prompt, and decode and prefill kernels
round differently. Neither is approximate.

## Controls

| Control | Effect |
| --- | --- |
| `mrun serve --no-cache` | Disable reuse for the whole server (every miss reports `disabled`) |
| `mrun cache clear` | Drop all cached state, including the live conversation (after the running request). Prints `cleared N branches (M MiB) and the live conversation`; the next request reports `new`, not `evicted` |
