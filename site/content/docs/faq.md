+++
title = "FAQ"
description = "Common questions about mrun: how it relates to Ollama, llama.cpp, LM Studio and mlx-lm, model support, memory, Linux, privacy and quality."
weight = 120
+++

## Is mrun released?

No. [Build it from source](@/docs/getting-started.md); the [roadmap](@/docs/roadmap.md) lists
what remains before the first release.

## How is mrun different from Ollama, LM Studio, llama.cpp or other MLX servers?

Exact prefix reuse, the three agent APIs, `launch` commands, a native binary and M1 support are
common among local servers, and mrun does not claim them as differences. What it does claim:

- A request served from the cache gets bit-identical output to recomputing its prompt.
- Every miss names its cause and the message where the prompt changed, and every response says
  how much could have been reused.
- Tool loops always hit: mrun splices the model's own token ids back into the next prompt.
- A memory budget that holds.
- An [independent benchmark](@/docs/benchmarks.md) that reports every cell, including losses.

mrun has no speculative decoding, no parallel lanes and no on-disk cache that survives a restart.
Servers with speculative decoding decode faster on Qwen3.6, and servers with lanes answer
parallel requests sooner.

## Why not just use mlx-lm's server?

mlx-lm is the reference implementation for MLX models, and mrun checks its output against it. On
the M4 Pro, mrun's prefill is 92–98 % and its decode 94–108 % of mlx-lm 0.32.0 on Qwen3-8B and
Qwen3.5-9B at 1K and 16K tokens (see [Benchmarks](@/docs/benchmarks.md#measured-on-the-m4-pro-2026-10-16)).
mrun adds what an agent server needs: the Anthropic Messages and OpenAI Responses APIs, admission
control, a bounded queue, a memory budget and an explanation for every cache miss, in one native
binary.

## Does the prefix cache change the model's output?

No. Reuse requires identical token ids and an identical model identity, and a request served
from the cache gets bit-identical output to recomputing its prompt from empty state. Running with `--no-cache` can differ in the low bits, because
MLX rounds a reply computed token by token differently from the same reply read back as a prompt
([details](@/docs/prefix-cache.md#exactness)).

## Which models are supported?

`qwen3.5` (Qwen3.5-9B) and `qwen3.6` (Qwen3.6-35B-A3B); each Mac gets the largest one that leaves
at least 64K tokens of context. `qwen3.5`, the model for Macs where `qwen3.6` does not fit, is a
preview: in the [client gate](@/docs/clients.md#client-gate) it completed 26–28 of 30 runs per
client, below the release bar. Qwen3-8B and the Qwen3-Coder family load by repository id. See
[Models](@/docs/models.md). Larger models run on a rented GPU with
[`mrun cloud`](@/docs/cloud.md).

## What does my Mac need?

Apple Silicon (M1 or later), macOS 15 or newer and 16 GB or more. With 48 GB, Qwen3.6 gets its
full 262K context; on a 16 GB Mac, Qwen3.5-9B gets 84,333 tokens (simulated with `--budget-gib`).
`mrun ls` shows what each alias gets on your Mac.

## Which Macs has mrun been tested on?

One: an M4 Pro with 48 GB. Smaller Macs are simulated on it by lowering the budget to their
working set, which reproduces fit and admission but not a slower GPU's cancellation latency, M5
Neural Accelerator speed or real memory pressure. A
[hardware report](https://github.com/hupe1980/mrun/blob/main/CONTRIBUTING.md#hardware-reports)
(`mrun doctor --report`, optionally with `mrun bench` results) is listed with credit on the
[benchmarks page](@/docs/benchmarks.md#contributed-results).

## Can I use GGUF files?

No. mrun runs MLX-format models from Hugging Face, such as those published by mlx-community.

## Do MCP servers and skills work?

Yes. They run in your agent, which sends their tools and instructions to mrun like any others. See
[MCP and skills](@/docs/mcp-and-skills.md).

## Does it run on Linux, Windows or NVIDIA GPUs?

No. mrun targets Apple Silicon. Linux builds compile and pass the engine-agnostic tests in CI, but
running models on Linux is not supported. A model on an NVIDIA GPU can run remotely with
[`mrun cloud`](@/docs/cloud.md).

## Can several agents use it at once?

Yes. Requests are queued and run one at a time, so a main agent and its subagents are all
served, and the prefix cache keeps each conversation's state as a branch. A subagent or title
request waits behind the main turn; streaming requests never time out while they wait.

## Does mrun send any data anywhere?

No. It listens on loopback only, has no telemetry and does not log prompts. After `mrun pull` it
works offline. Tools run inside your agent, never in the server. The exception is
[`mrun cloud`](@/docs/cloud.md), which is opt-in and sends prompts to a rented RunPod GPU.

## How good is a local model compared to a frontier API?

The supported models are much smaller than frontier models, and mrun does not measure their
coding quality against hosted APIs; judge them on your own tasks. What mrun does measure is
whether each client completes scripted tasks with each model (the
[client gate](@/docs/clients.md#client-gate)).

## What license is mrun under?

MIT or Apache-2.0, at your option. Model weights keep their own licenses.
