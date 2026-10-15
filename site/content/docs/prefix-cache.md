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

Two more things keep the cache warm:

- **Interrupted prompts keep their progress.** If a client gives up during a long prompt and
  retries, mrun resumes from the last completed chunk instead of starting over.
- **Warm-up while tools run.** After a turn that ends in tool calls, the GPU is idle while your
  agent runs them. mrun uses that time to process the conversation exactly as the client will send
  it next (templates often re-render the previous reply slightly differently). A real request
  always takes priority; a wrong guess costs only idle time.

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
| `edit` | The conversation, system prompt or tools changed before the divergence point | Expected after compaction or editing history; otherwise check client settings that change per session |
| `template` | The same messages rendered to different tokens, e.g. earlier reasoning re-rendered differently | Usually a client or template quirk; report it if it repeats |
| `volatile` | Content that changes on every request appears early in the prompt (timestamps, token counters) | Use `mrun launch`, which turns these off for supported clients |
| `evicted` | The prefix was removed to free memory | More memory, or fewer parallel conversations |
| `capacity` | The prompt was too large to keep | Shorter context or more memory |
| `model` | A different model or revision was loaded | Expected |
| `disabled` | Caching was off for this request | `--no-cache` or `"mrun": {"cache": false}` |

The divergence report gives the token position where the prompt differed from the best cached
candidate. `mrun serve --debug-cache` additionally prints short text excerpts around
the divergence to the server terminal — never to logs or responses.

## Memory and eviction

Cached segments share the memory budget with the active request. When a new request needs room,
mrun evicts the least recently used segments: first conversations superseded by a compaction,
then finished subagents and other sessions, then assistant turns before user and tool turns before system prompts. A request is never refused while evictable cache remains. If macOS reports memory
pressure, mrun sheds cache first.

## Shared between clients

All clients of one server share the cache. Cached-token counts therefore reveal whether another
client recently sent the same prefix. On your own Mac this doesn't matter; if you expose mrun to
other people with `--host`, keep it in mind, or run with `--no-cache`.

## Client hints

mrun reads the session and subagent ids that Claude Code, Codex and OpenCode send, plus Anthropic
`cache_control` breakpoints and OpenAI `prompt_cache_key`. Breakpoints mark good places to keep a
snapshot; session ids let eviction drop finished subagents and other sessions before your main
conversation. Hints never change what is reused.

Clients can also ask why a turn missed the cache with `diagnostics.previous_message_id`; mrun
answers with the standard `diagnostics.cache_miss_reason` in the response.

## Controls

| Control | Effect |
| --- | --- |
| `mrun serve --no-cache` | Disable reuse for the whole server |
| `x-mrun-cache: off` request header | Disable reuse for one request (all APIs) |
| `"mrun": {"cache": false}` in a Chat Completions or Responses body | Same, in the body |
| `mrun cache clear` | Drop all cached segments of the running server |
| `mrun serve --no-warmup` | Turn off warm-up while tools run |
| `mrun serve --debug-cache` | Print divergence excerpts on the server terminal (loopback only) |
