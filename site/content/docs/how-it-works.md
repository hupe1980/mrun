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
   file stays text. Message boundary positions are kept for the cache. Each image becomes its
   tokens; the vision tower encodes it during prefill, and a cached image is not encoded again.
3. **Splicing.** Where the current turn contains a reply the model generated earlier, mrun puts
   back the exact token ids it sampled, reasoning included (see
   [Prefix cache](@/docs/prefix-cache.md#tool-loops-always-hit)).
4. **Queue and lanes.** Up to two requests run at once by default (lanes, see below); up to 16
   more wait. Side requests (titles, summaries) start first. Streaming requests receive keepalives
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

## One engine, one thread, several lanes

All GPU work runs on a single engine thread that owns the model, the sequence state and the
prefix cache. HTTP handlers only parse, queue and stream. Moves between live state and cache go
one layer at a time, so state is never held twice.

The thread runs up to four live sequences, *lanes* (`--lanes`; two when the context holds 32K
tokens for each). Coding agents send side requests beside the main turn and run subagents in
parallel; lanes run them side by side:

- **Decode** steps every decoding lane in one forward pass, so lanes share each read of the
  weights. Each lane's output is bit-identical to running its request alone.
- **Prefill** runs a layer at a time, always for the lane with the fewest prompt tokens left: a
  title that arrives while a long prompt is read takes over at the next layer and starts as soon
  as it would alone, and the long prompt resumes where it paused. Between layers the decoding
  lanes step, with prefill keeping about 80 % of the GPU, so a reply keeps streaming (a token at
  least every ≈ 0.2 s) while a long prompt is read.
- **Admission** reserves each request's prompt and output limit; a request whose reservation does
  not fit beside the running ones waits for a lane.
- **Reuse:** a request goes to the lane that holds the longest part of its prompt. An idle lane
  keeps its conversation, so a second round of subagents finds its prefix on every lane; memory
  pressure sheds stored branches first, then idle lanes, never a running one.
- A client that stops reading pauses only its own lane.

One running request uses exactly the single-sequence path, with speculative decoding.

## nuvec

mrun runs on [nuvec](https://crates.io/crates/nuvec), a pure-Rust port of Apple's
[MLX](https://github.com/ml-explore/mlx). nuvec provides the arrays and the Metal kernels; mrun
owns the model graphs, state layout, memory accounting, scheduling and the server. Model families
are ported from [mlx-lm](https://github.com/ml-explore/mlx-lm) and checked against it
([Benchmarks](@/docs/benchmarks.md)). The Metal kernels are compiled into the binary.

Prefill and decode are within a few percent of mlx-lm 0.32.0 ([Benchmarks](@/docs/benchmarks.md));
the first request's time to first token on a short prompt is about 0.13 s.

On M5-class GPUs the Neural Accelerator (NAX) kernels run where MLX would use them
(`NUVEC_NAX=off` keeps them off). Whether they are used is part of the engine identity shown by
`mrun --version` and recorded in bench results.

## Attention and hybrid models

- **Full attention** (Qwen3-8B, Qwen3-Coder) keeps state for every token, so a cached prefix can
  be cut at any position. Qwen3-8B needs 144 KiB per token.
- **Linear attention** (30 of Qwen3.6's 40 layers, 24 of Qwen3.5-9B's 32) keeps one fixed-size
  state per layer. Long context is cheap (about 20 KiB per token for Qwen3.6's attention layers),
  but the state exists only for the position it was computed at, so mrun saves checkpoints and
  resumes from the nearest one.

## Speculative decoding

Each decode step reads every weight once to produce one token. mrun drafts tokens and checks them
in one pass. Drafts come from two places:

- **Prompt lookup.** Agent replies copy a lot (an `Edit`'s old text, a rewritten file, repeated
  paths). When the reply's last tokens appear earlier, mrun drafts the 7 that followed there. A
  full match goes first, since it costs nothing to make.
- **The DFlash drafter.** A small model pulled with the alias (`z-lab/Qwen3.5-9B-DFlash` for
  `qwen3.5`, `z-lab/Qwen3.6-35B-A3B-DFlash` for `qwen3.6`, quantized to 4 bits at load) reads the
  hidden states of some of the model's layers and proposes the next 3 tokens in one forward pass.
  It keeps the last 4,096 positions of context per lane (about 100 MB), and prefix-cache branches
  keep theirs.

Each position is sampled exactly as plain decoding samples it, with the same random draw, and a
drafted token is kept only where it equals that sample, so the reply is the same token for token
whoever drafted. The pass runs nuvec's row-exact kernels: each row has the bits of a one-token
step; up to 4 rows read each weight once, which is why the drafter proposes 3. A request drafts
only while its speculative rounds, averaged over 4, are faster than its plain steps, and only
while it decodes alone. `mrun ps`, Now and each chat reply show who drafted and the tokens per
pass. Pools decode plainly.

## Design rules

- **Exactness.** A cached prefix is used only if every token and the model identity match (see
  [Prefix cache](@/docs/prefix-cache.md#exactness)).
- **One GPU thread.** All GPU work happens on one thread.
- **Memory is a budget.** Everything that can grow is accounted for against a budget fixed at
  startup.
- **Standard wire formats.** Response bodies are what each API defines; mrun's extra information
  travels in headers and SSE comments that clients ignore.
- **Local and private.** Loopback only unless you opt in to a pool or a rented GPU; no telemetry,
  no prompt logging.
