+++
title = "FAQ"
description = "Common questions about mrun: how it compares to Ollama, llama.cpp, LM Studio and mlx-lm, model support, Linux, privacy and quality."
weight = 120
+++

## How is mrun different from Ollama, LM Studio or llama.cpp?

They are general-purpose local model runners. mrun is built for one workload: coding agents that
resend a growing conversation every turn. That shapes everything — an exact prefix cache with miss
explanations, a memory budget sized for long contexts, cancellation designed to free the server within 300 ms (release gate: p95 at 16K and 64K
context for each first-release model on an M4 Pro, 48 GB; not yet measured),
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

mrun supports an open set of model families. The first release covers Qwen3.6-35B-A3B (the
expected default, a hybrid mixture-of-experts model), Qwen3-Coder-30B-A3B and gpt-oss-20b, in MLX
format. Devstral Small 2 is next, then GLM-4.7-Flash, Gemma 4 and Nemotron-H, then other families
based on what coding agents actually use; see [Models](@/docs/models.md) and the
[roadmap](@/docs/roadmap.md).

## Can I use GGUF files?

No. mrun runs MLX-format models, which use Apple's fast quantized kernels. Most popular models are
published in MLX format by the mlx-community organization on Hugging Face.

## Do MCP servers and skills work?

Yes. They run in your agent, which sends their tools and instructions to mrun like any others. See
[MCP and skills](@/docs/mcp-and-skills.md) for keeping large tool sets fast.

## Does it run on Linux, Windows or NVIDIA GPUs?

Not yet. mrun targets Apple Silicon. Linux builds compile and pass the portable tests; running
models on Linux is evaluated after the first Mac release. On NVIDIA servers, vLLM and SGLang are
the better choice.

## Can several agents use it at once?

Yes. Requests are queued and run one at a time, so a main agent and its subagents all get served,
and the prefix cache keeps each branch's context. Batched decoding for concurrent sessions will be
added only if recorded sessions show agents waiting on each other (see the
[roadmap](@/docs/roadmap.md)).

## Does mrun send any data anywhere?

No. It listens on loopback only, has no telemetry and does not log prompts. After
`mrun pull` it works offline. Tools run inside your agent, never in the server.

## How good is a local model compared to a frontier API?

The supported models are much smaller than frontier models, and mrun does not measure their
coding quality against hosted APIs; judge them on your own tasks. mrun's goal is to make the local
option fast and dependable, not to claim parity.

## What license is mrun under?

MIT or Apache-2.0, at your option. Model weights keep their own licenses.
