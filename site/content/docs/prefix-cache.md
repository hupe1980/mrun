+++
title = "Prefix cache"
description = "How mrun reuses attention state exactly across turns, subagents and tool calls, for attention, sliding-window and hybrid models, and how it explains every cache miss."
weight = 40
+++

## What gets reused

When a model reads a prompt it produces state for every token. The next request that starts with
the same tokens can reuse that state instead of computing it again. mrun keeps:

- **The live conversation in place.** The next turn reuses it without any copy; only the new
  tokens are processed.
- **Branches beside it.** When a different conversation arrives — a subagent, a title request,
  another session, a retry with an edit — the shared beginning stays where it is and the rest of
  the old conversation is *moved* aside. Switching back moves it in again. State is never held
  twice, so on a 48 GB Mac a 90K-token Qwen3-Coder conversation survives a subagent although
  memory holds only about 105K tokens.
- **Checkpoints for position-bound state.** Some layers, such as linear attention, keep a
  fixed-size state that only exists for the position it was computed at. While processing a
  prompt, mrun saves checkpoints at the end of the system prompt and tools, at the start of the
  last user or tool message, and where the reply starts, at most eight per conversation. An edited
  or re-rendered history resumes from the nearest checkpoint before the change.
- **Progress of interrupted prompts.** If a client gives up during a long prompt and retries, mrun
  resumes from the last completed chunk instead of starting over.

Reuse is exact: the same token ids and the same model content hash. Nothing is approximated.

## Reading the numbers

Every response reports how many prompt tokens were reused, in each API's own usage field
(`cache_read_input_tokens` for Anthropic, `prompt_tokens_details.cached_tokens` for OpenAI). mrun
also attaches its own report:

- non-streaming responses: `x-mrun-cache-cached`, `x-mrun-cache-processed`, `x-mrun-cache-cause`,
  `x-mrun-cache-divergence`, `x-mrun-cache-message` and, for models with position-bound state,
  `x-mrun-cache-gap` headers;
- streaming responses: a `: mrun {...}` SSE comment line before the first event.

`mrun ps` shows totals and the last miss.

## Why a miss happened

Every miss report names one of five causes, the token position where the prompt differed from the
best cached candidate, and the index and role of the message that position falls in (message 0
is the system prompt and tools). That tells you what changed without any prompt text.

| Cause | Meaning | What to do |
| --- | --- | --- |
| `new` | The request shares at most the system prompt and tools with stored state (a new conversation), or nothing is stored | Nothing; the first turn is always computed |
| `changed` | Something the cache had already seen differs: the system prompt or tool list (message 0), an earlier message that was edited, removed or replaced, or a message the chat template rendered differently (for example re-serialized tool arguments) | Expected after an MCP server connects, compaction, edits or retries; if it happens every turn at the same message, check what your client changes there |
| `evicted` | The stored state was removed: least recently used, `mrun cache clear`, or memory pressure | More memory, or fewer parallel conversations |
| `model` | A different model, revision or template was loaded | Expected |
| `disabled` | The server runs with `--no-cache` | — |

For models with linear-attention layers, a request may also resume from a checkpoint below the
matching position; the report then gives the gap that was processed again. `mrun bench` counts
how often `changed` comes from template re-rendering alone.

## Memory and eviction

Branches share the memory budget with the live conversation. When a request needs room, mrun
evicts the least recently used branches; the live conversation is never evicted while it runs.
Under memory pressure, branches go first.

## Shared between clients

All clients of one server share the cache. Cached-token counts therefore reveal whether another
client recently sent the same prefix. mrun listens on loopback only, so on your own Mac this
doesn't matter; if other people share your user session, keep it in mind, or run with
`--no-cache`.

## Controls

| Control | Effect |
| --- | --- |
| `mrun serve --no-cache` | Disable reuse for the whole server |
| `mrun cache clear` | Drop all cached state of the running server |
