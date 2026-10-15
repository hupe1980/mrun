+++
title = "FAQ"
description = "Common questions about mrun: how it compares to Ollama, llama.cpp, LM Studio and mlx-lm, model support, Linux, privacy and quality."
weight = 120
+++

## How is mrun different from Ollama, LM Studio or llama.cpp?

They are general-purpose local model runners. mrun is built for one workload: coding agents that
resend a growing conversation every turn. That shapes everything — an exact prefix cache with miss
explanations, a memory budget sized for long contexts, cancellation that takes effect immediately,
and first-class support for the Anthropic Messages and OpenAI Responses APIs that Claude Code and
Codex use. If you chat with a model occasionally, any of those tools works well. If you run an
agent for hours, mrun is designed for you.

## Why not just use mlx-lm's server?

mlx-lm is the reference implementation for MLX models, and mrun checks its output against it. Its
server is a convenient Python tool; mrun adds the parts an agent server needs: three agent APIs,
admission control, a bounded queue, a memory budget and cache diagnostics, in one native binary.

## Does the prefix cache change the model's output?

No. Reuse requires identical token ids and an identical model. A reused prefix yields the same
attention state as recomputing it, up to floating-point rounding in the GPU kernels.

## Which models are supported?

Qwen3 dense and Qwen3 mixture-of-experts models in MLX format, with Qwen3-Coder-30B-A3B as the
default. More model families follow based on what coding agents actually use; see
[Models](@/docs/models.md) and the [roadmap](@/docs/roadmap.md).

## Can I use GGUF files?

No. mrun runs MLX-format models, which use Apple's fast quantized kernels. Most popular models are
published in MLX format by the mlx-community organization on Hugging Face.

## Does it run on Linux, Windows or NVIDIA GPUs?

Not yet. mrun targets Apple Silicon. Linux builds compile and pass the portable tests; running
models on Linux is evaluated after the first Mac release. On NVIDIA servers, vLLM and SGLang are
the better choice.

## Can several agents use it at once?

Yes. Requests are queued and run one at a time, so a main agent and its subagents all get served,
and the prefix cache keeps each branch's context. Batched decoding for concurrent sessions is on
the roadmap.

## Does mrun send any data anywhere?

No. It listens only on loopback by default, has no telemetry and does not log prompts. After
`mrun pull` it works offline. Tools run inside your agent, never in the server.

## How good is a local model compared to a frontier API?

Smaller. Qwen3-Coder-30B-A3B handles everyday edits, refactors and tool loops well, but it is not a
substitute for the strongest hosted models on hard problems. mrun's goal is to make the local
option fast and dependable, not to claim parity.

## What license is mrun under?

MIT or Apache-2.0, at your option. Model weights keep their own licenses.
