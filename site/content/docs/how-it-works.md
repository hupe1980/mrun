+++
title = "How it works"
description = "The design of mrun: one engine thread on Apple MLX, an open set of model families, an exact prefix cache that survives subagents, a startup memory budget and interruptible prefill."
weight = 30
+++

## The problem

A coding agent sends the whole conversation on every turn: system instructions, tool schemas,
files it has read, tool results and the history. After a few turns that is 20,000–60,000 tokens,
of which only the last few hundred are new. On a laptop GPU, reading the prompt (prefill) is the
slow part, and re-reading tens of thousands of tokens every turn dominates the wait. Agents also branch:
subagents, title requests and retries arrive between the turns of the main conversation.

Cloud APIs solve this with prompt caching. mrun brings the same idea to your Mac, exactly and with
explanations.

## Request path

1. **Protocol.** The agent calls one of three APIs: Anthropic Messages, OpenAI Chat Completions or
   OpenAI Responses. Each is translated to one internal request: messages, tools, sampling and
   limits.
2. **Rendering.** The model's own chat template turns the conversation into text, which the model's
   tokenizer turns into token ids. The same conversation always yields the same tokens.
3. **Admission.** mrun checks that the prompt fits the memory budget *before* touching anything,
   and clamps the output limit to the space left. Prompts that cannot fit are refused with the
   numbers, and the cache is left intact.
4. **Prefix lookup.** The [prefix cache](@/docs/prefix-cache.md) finds the longest token prefix it
   has already computed. The usual case — the next turn of the current conversation — needs no
   copy at all; a different conversation moves the old one aside and its own state in.
5. **Prefill.** Only the new tokens run through the model, in chunks that end at message
   boundaries and shrink at depth. Cancellation is checked after every layer, so an interrupted
   request stops quickly and keeps the chunks it finished.
6. **Decode.** Tokens stream back. Tool calls are parsed from the model's output format into each
   API's native tool-call structure.
7. **Record.** The finished conversation, including the model's reply, becomes the new reusable
   prefix for the next turn.

## One engine, one thread

All GPU work runs on a single engine thread that owns the model and the attention state. HTTP
handlers only parse, queue and stream. Requests run one at a time in arrival order; others wait
in a bounded queue with keepalives so clients don't time out. Requests run one after another,
which keeps latency predictable and reuse exact. Whether batching would help is measured with
`mrun bench` before it is added (see the [roadmap](@/docs/roadmap.md)).

## MLX from Rust

mrun calls Apple's [MLX](https://github.com/ml-explore/mlx) through its official C API. MLX provides
the Metal kernels — quantized matrix multiplication, fused attention — that make Apple Silicon
fast. mrun owns everything around them: model graphs, attention-state layout, memory accounting,
scheduling and the server. Model families are composed from shared blocks ported from the reference
[mlx-lm](https://github.com/ml-explore/mlx-lm) implementations, and output is checked
token-for-token against them.

## Attention and hybrid models

mrun supports an open set of model families built from shared blocks, and each kind of layer is
cached in the way that keeps reuse exact:

- **Full attention** (Qwen3-Coder) keeps state for every token, so a cached
  prefix can be cut at any position.
- **Sliding-window attention** (gpt-oss) only looks at recent tokens, but mrun keeps the full
  state and applies the window as a mask, so any position stays reusable here too.
- **Linear attention** (the hybrid layers of Qwen3.6) keeps one fixed-size state per layer. That
  makes long context cheap — about 20 KiB per token, against 96 KiB for the full-attention Qwen3-Coder-30B-A3B — but the state exists only
  for the position it was computed at. mrun saves checkpoints of it at the end of the system prompt and
  tools, at the start of the last user or tool message, and where the reply starts, so an edited or branching conversation resumes from the nearest checkpoint
  instead of from the start.

See [Models](@/docs/models.md) for the supported families.

## Design rules

- **Exactness.** A cached prefix is used only if every token and the model identity match. The
  result equals recomputing it, up to the floating-point rounding any chunked prefill has.
- **One GPU thread.** Competing servers that drive MLX from several threads have hit stream errors
  and kernel panics; mrun keeps all GPU work on one thread by design.
- **Memory is a budget, not a hope.** Everything that can grow is accounted for at startup.
- **Standard wire formats.** Response bodies are exactly what each API defines. mrun's extra
  information travels in headers and SSE comments that clients ignore.
- **Local and private.** Loopback by default, no telemetry, no prompt logging.
