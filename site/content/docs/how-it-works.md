+++
title = "How it works"
description = "The design of mrun: one engine thread on Apple MLX, an exact prefix cache, a startup memory budget and interruptible prefill."
weight = 30
+++

## The problem

A coding agent sends the whole conversation on every turn: system instructions, tool schemas,
files it has read, tool results and the history. After a few turns that is 20,000–60,000 tokens,
of which only the last few hundred are new. On a laptop GPU, reading the prompt (prefill) runs at
hundreds to a few thousand tokens per second, so re-reading the full context each turn costs tens
of seconds before the first token appears.

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
   has already computed and restores that attention state. The usual case — the next turn of the
   current conversation — needs no copy at all.
5. **Prefill.** Only the new tokens run through the model, in chunks of 2,048 tokens. Cancellation
   is checked after every layer, so an interrupted request stops within a fraction of a second.
6. **Decode.** Tokens stream back. Tool calls are parsed from the model's output format into each
   API's native tool-call structure.
7. **Record.** The finished conversation, including the model's reply, becomes the new reusable
   prefix for the next turn.

## One engine, one thread

All GPU work runs on a single engine thread that owns the model and the attention state. HTTP
handlers only parse, queue and stream. Requests run one at a time in arrival order; others wait
in a bounded queue with keepalives so clients don't time out. On a single Mac GPU, decode speed is
limited by memory bandwidth, so running requests one after another loses little, keeps latency
predictable and keeps reuse exact. Batching concurrent sessions is on the
[roadmap](@/docs/roadmap.md) if measurements show agents waiting on each other.

## MLX from Rust

mrun calls Apple's [MLX](https://github.com/ml-explore/mlx) through its official C API. MLX provides
the Metal kernels — quantized matrix multiplication, fused attention — that make Apple Silicon
fast. mrun owns everything around them: model graphs, attention-state layout, memory accounting,
scheduling and the server. Model graphs are ported from the reference
[mlx-lm](https://github.com/ml-explore/mlx-lm) implementations, and output is checked
token-for-token against them.

## Design rules

- **Exactness.** A cached prefix is used only if every token and the model identity match. The
  result equals recomputing it, up to the floating-point rounding any chunked prefill has.
- **One GPU thread.** Competing servers that drive MLX from several threads have hit stream errors
  and kernel panics; mrun keeps all GPU work on one thread by design.
- **Memory is a budget, not a hope.** Everything that can grow is accounted for at startup.
- **Standard wire formats.** Response bodies are exactly what each API defines. mrun's extra
  information travels in headers and SSE comments that clients ignore.
- **Local and private.** Loopback by default, no telemetry, no prompt logging.
