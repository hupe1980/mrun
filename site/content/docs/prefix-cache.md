+++
title = "Prefix cache"
description = "How mrun reuses attention state exactly across turns, branches and tool calls, and how it explains every cache miss."
weight = 40
+++

## What gets reused

When a model reads a prompt it produces attention state — keys and values for every token at
every layer. The next request that starts with the same tokens can reuse that state instead of
computing it again. mrun stores it in a radix tree keyed by token ids:

- **The live conversation** stays in place. The next turn reuses it without any copy.
- **Branches** — subagents, retries, edited history, a second conversation — are kept as snapshots
  of shared segments. Switching back restores the longest matching prefix by copy.
- **Boundaries** are recorded at message ends, so a branch that diverges mid-conversation still
  reuses everything up to the last shared message.

Reuse is exact: the same token ids, the same model content hash, the same rendering. Nothing is
approximated.

## Reading the numbers

Every response reports how many prompt tokens were reused, in each API's own usage field
(`cache_read_input_tokens` for Anthropic, `prompt_tokens_details.cached_tokens` for OpenAI). mrun
also attaches its own report:

- non-streaming responses: `x-mrun-cache-cached`, `x-mrun-cache-processed`, `x-mrun-cache-cause`
  and `x-mrun-cache-divergence` headers;
- streaming responses: a `: mrun {...}` SSE comment line before the first event.

`mrun ps` shows totals and the last miss.

## Why a miss happened

| Cause | Meaning | What to do |
| --- | --- | --- |
| `new` | No earlier request shares a prefix | Nothing; the first turn is always computed |
| `edit` | The conversation changed before the divergence point | Expected after compaction or editing history |
| `template` | Same messages rendered differently, e.g. a changed system prompt or tools | Check client settings that change per session |
| `volatile` | Content that changes on every request appears early in the prompt (timestamps, token counters) | Use `mrun launch`, which turns these off for supported clients |
| `evicted` | The prefix was removed to free memory | More memory, or fewer parallel conversations |
| `capacity` | The prompt was too large to keep | Shorter context or more memory |
| `model` | A different model or revision was loaded | Expected |
| `disabled` | Caching was off for this request | `--no-cache` or `"mrun": {"cache": false}` |

The divergence report gives the token position and the message where the prompt differed from the
best cached candidate. `mrun serve --debug-cache` additionally prints short text excerpts around
the divergence to the server terminal — never to logs or responses.

## Memory and eviction

Cached segments share the memory budget with the active request. When a new request needs room,
mrun evicts the least recently used segments, preferring finished side branches over the main
conversation. A request is never refused while evictable cache remains. If macOS reports memory
pressure, mrun sheds cache first.

## Shared between clients

All clients of one server share the cache. Cached-token counts therefore reveal whether another
client recently sent the same prefix. On your own Mac this doesn't matter; if you expose mrun to
other people with `--host`, keep it in mind, or run with `--no-cache`.

## Client hints

Anthropic `cache_control` breakpoints and OpenAI `prompt_cache_key` are accepted as hints:
breakpoints mark good places to keep a snapshot, and the key groups one session's requests so
eviction prefers other sessions' state. Hints never change what is reused.

Clients can also ask why a turn missed the cache with `diagnostics.previous_message_id`; mrun
answers with the standard `diagnostics.cache_miss_reason` in the response.

## Controls

| Control | Effect |
| --- | --- |
| `mrun serve --no-cache` | Disable reuse for the whole server |
| `"mrun": {"cache": false}` in a Chat Completions or Responses body | Disable reuse for one request |
| `mrun cache clear` | Drop all cached segments of the running server |
| `mrun serve --debug-cache` | Print divergence excerpts on the server terminal (loopback only) |
