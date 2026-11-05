+++
title = "How it works"
description = "The design of mrun: one engine thread on nuvec, segmented tokenization and reply splicing, an exact prefix cache, a startup memory budget and interruptible prefill."
weight = 30
+++

## The problem

A coding agent sends the whole conversation on every turn: instructions, tool schemas, files,
tool results and history. After a few turns that is 15,000–60,000 tokens, of which only the last
few hundred are new. On a laptop GPU, reading the prompt (prefill) is the slow part. Agents also
branch (subagents, titles, retries) and cancel often.

## Request path

1. **Protocol.** Anthropic Messages, OpenAI Chat Completions and OpenAI Responses are each
   translated to one internal request: messages, tools, sampling and limits.
2. **Rendering.** The model's chat template renders the conversation. Template scaffolding is
   tokenized with special tokens recognized, message content without, so a `<|im_end|>` inside a
   file stays text. Message boundary positions are kept for the cache.
3. **Splicing.** Where the current turn contains a reply the model generated earlier, mrun puts
   back the exact token ids it sampled, reasoning included (see
   [Prefix cache](@/docs/prefix-cache.md#tool-loops-always-hit)).
4. **Queue.** One request runs at a time; up to 16 wait. Streaming requests receive keepalives
   and wait as long as needed; non-streaming ones wait at most 240 seconds.
5. **Lookup and admission.** The [prefix cache](@/docs/prefix-cache.md) finds the longest exact
   token prefix it holds. mrun checks that the prompt fits the [memory budget](@/docs/memory.md)
   and clamps the output limit before touching any state; a prompt that cannot fit is refused with
   the numbers.
6. **Switch.** The next turn of the live conversation needs no move. A different conversation moves
   the old tail aside as a branch and the matching branch in; hybrid models restore a checkpoint.
7. **Prefill.** Only the new tokens run through the model, in chunks that end at message
   boundaries and shrink with depth. The chunk plan depends only on the prompt. Cancellation is
   checked after every layer; chunk length is calibrated at startup so one layer takes at most
   250 ms (stored in `~/.mrun/calibration.toml`). An interrupted request keeps the chunks it
   finished.
8. **Decode.** Tokens stream back. A parser turns tool calls and reasoning into each API's native
   structures; a grammar constrains output only for JSON formats and forced tool calls. A loop
   guard stops runaway repetition in text and reasoning, never inside tool-call arguments.
9. **Finish.** The prompt plus the reply stay as the live state for the next turn, and the reply is
   recorded for splicing. A request that fails on its own ends with an error and leaves the cache
   intact.

## One engine, one thread

All GPU work runs on a single engine thread that owns the model, the sequence state and the
prefix cache. HTTP handlers only parse, queue and stream. One sequence runs at a time, which
keeps latency predictable and reuse exact. Moves between live state and cache go one layer at a
time, so state is never held twice.

## nuvec

mrun runs on [nuvec](https://crates.io/crates/nuvec), a pure-Rust port of Apple's
[MLX](https://github.com/ml-explore/mlx). nuvec provides the arrays and the Metal kernels; mrun
owns the model graphs, state layout, memory accounting, scheduling and the server. Model families
are ported from [mlx-lm](https://github.com/ml-explore/mlx-lm) and checked token for token against
it. The Metal kernels are compiled into the binary.

M4 Pro with 48 GB, 4,350-token prompt, best of 2, against mlx-lm 0.32.0:

| Model | Prefill, mrun / mlx-lm | Decode, mrun / mlx-lm |
| --- | --- | --- |
| Qwen3.6-35B-A3B 4-bit | 982 / 938 tok/s | 86.9 / 87.6 tok/s |
| Qwen3-8B 4-bit | 433 / 444 tok/s | 45.7 / 46.4 tok/s |

The first request's time to first token on a short prompt is about 0.13 s.

The M5 Neural Accelerator (NAX) kernels are compiled in but stay off until measured on an M5
(`NUVEC_NAX=on` forces them). Whether they are used is part of the engine identity shown by
`mrun --version` and recorded in bench results.

## Attention and hybrid models

- **Full attention** (Qwen3-8B, Qwen3-Coder) keeps state for every token, so a cached prefix can
  be cut at any position. Qwen3-8B needs 144 KiB per token.
- **Linear attention** (30 of Qwen3.6's 40 layers, 24 of Qwen3.5-9B's 32) keeps one fixed-size
  state per layer. Long context is cheap (about 20 KiB per token for Qwen3.6's attention layers),
  but the state exists only for the position it was computed at, so mrun saves checkpoints and
  resumes from the nearest one.

## Design rules

- **Exactness.** A cached prefix is used only if every token and the model identity match (see
  [Prefix cache](@/docs/prefix-cache.md#exactness)).
- **One GPU thread.** All GPU work happens on one thread.
- **Memory is a budget.** Everything that can grow is accounted for against a budget fixed at
  startup.
- **Standard wire formats.** Response bodies are what each API defines; mrun's extra information
  travels in headers and SSE comments that clients ignore.
- **Local and private.** Loopback only, no telemetry, no prompt logging.
