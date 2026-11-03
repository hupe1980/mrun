+++
title = "How it works"
description = "The design of mrun: one engine thread on Apple MLX, segmented tokenization and reply splicing, an exact prefix cache with flat branches, a startup memory budget and interruptible prefill."
weight = 30
+++

## The problem

A coding agent sends the whole conversation on every turn: system instructions, tool schemas,
files it has read, tool results and the history. After a few turns that is 15,000–60,000 tokens,
of which only the last few hundred are new. On a laptop GPU, reading the prompt (prefill) is the
slow part. Agents also branch: subagents, title requests and retries arrive between the turns of
the main conversation, and clients cancel often.

## Request path

1. **Protocol.** The agent calls one of three APIs: Anthropic Messages, OpenAI Chat Completions or
   OpenAI Responses. Each is translated to one internal request: messages, tools, sampling and
   limits.
2. **Rendering.** The model's own chat template renders the conversation. Template scaffolding is
   tokenized with special tokens recognized, message content without, so a `<|im_end|>` inside a
   file the agent read stays text. The positions of message boundaries are kept for the cache.
3. **Splicing.** Where the current turn contains a reply the model generated earlier, mrun puts
   back the exact token ids it sampled, reasoning included, instead of the template's
   re-rendering; earlier turns are rendered as the template renders them (see
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
   checked after every layer, and chunk length is calibrated at startup so one layer takes at most
   250 ms on this Mac (stored in `~/.mrun/calibration.toml`); an interrupted request keeps the
   chunks it finished.
8. **Decode.** Tokens stream back. A parser turns the model's tool calls and reasoning into each
   API's native structures; a grammar constrains output only for JSON formats and forced tool
   calls. A loop guard stops runaway repetition in text and reasoning, never inside tool-call
   arguments.
9. **Finish.** The prompt plus the reply stay as the live state for the next turn, and the reply is
   recorded with that state for splicing. A request that fails on its own ends with an error and
   leaves the cache intact.

## One engine, one thread

All GPU work runs on a single engine thread that owns the model, the sequence state and the
prefix cache. HTTP handlers only parse, queue and stream. One sequence runs at a time, which
keeps latency predictable and reuse exact.

The engine exposes small primitives (process one chunk, take one decode step, cut, checkpoint,
move a tail out or in); the runtime around it owns the chunk loop, checkpoint placement,
memory-pressure checks and retries after the cache has made room. Moves go one layer at a time,
so state is never held twice.

## MLX from Rust

mrun calls Apple's [MLX](https://github.com/ml-explore/mlx) through its C API (mlx-c), built from
pinned sources and linked statically. MLX provides the Metal kernels (quantized matrix
multiplication, fused attention); mrun owns the model graphs, state layout, memory accounting,
scheduling and the server. Model families are ported from the reference
[mlx-lm](https://github.com/ml-explore/mlx-lm) implementation and checked token for token against
it. The Metal kernel library is embedded in the binary, compressed, and extracted to
`~/Library/Caches/mrun/` on first use.

M4 Pro with 48 GB, 4,350-token prompt, best of 2, against mlx-lm 0.32.0:

| Model | Prefill, mrun / mlx-lm | Decode, mrun / mlx-lm |
| --- | --- | --- |
| Qwen3.6-35B-A3B 4-bit | 982 / 938 tok/s | 86.9 / 87.6 tok/s |
| Qwen3-8B 4-bit | 433 / 444 tok/s | 45.7 / 46.4 tok/s |

The first request's time to first token on a short prompt is about 0.13 s.

MLX compiles its M5 Neural Accelerator kernels (quantized matrix multiplication, prefill
attention, Gated DeltaNet) only for a deployment target of macOS 26.2 or later; a build for an
earlier target runs without them. Whether they are used is part of the engine identity shown by
`mrun --version` and recorded in bench results.

## Attention and hybrid models

- **Full attention** (Qwen3-8B, Qwen3-Coder) keeps state for every token, so a cached prefix can
  be cut at any position. Qwen3-8B needs 144 KiB per token.
- **Linear attention** (30 of Qwen3.6's 40 layers, 24 of Qwen3.5-9B's 32) keeps one fixed-size
  state per layer. That makes long context cheap (about 20 KiB per token for Qwen3.6's remaining
  attention layers), but the state exists only for the position it was computed at, so mrun saves
  checkpoints and resumes from the nearest one.

## Design rules

- **Exactness.** A cached prefix is used only if every token and the model identity match; a
  request served from the cache gets bit-identical output to recomputing its prompt from empty
  state (see [Prefix cache](@/docs/prefix-cache.md#exactness)).
- **One GPU thread.** All MLX work happens on one thread.
- **Memory is a budget.** Everything that can grow is accounted for against a budget fixed at
  startup.
- **Standard wire formats.** Response bodies are what each API defines; mrun's extra information
  travels in headers and SSE comments that clients ignore. (Chat Completions keepalives are
  empty data events, which clients also ignore.)
- **Local and private.** Loopback only, no telemetry, no prompt logging.
