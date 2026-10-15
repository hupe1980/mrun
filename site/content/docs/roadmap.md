+++
title = "Roadmap"
description = "What mrun ships first and what comes next: engine with hybrid models, agent APIs, prefix cache, developer experience, benchmarks and distribution, then speculative decoding and a persistent cache."
weight = 110
+++

mrun is in active development; nothing is released yet. Each milestone has a measurable gate on
the target Mac (M4 Pro, 48 GB) before it counts as done.

## First release

| Milestone | Outcome |
| --- | --- |
| **Foundation** | MLX bindings proven on a Mac: quantized and mixture-of-experts kernels, attention, custom Metal kernels, memory controls |
| **Engine** | `mrun run` streams Qwen3.6-35B-A3B, Qwen3-Coder-30B-A3B and Qwen3-8B with the memory budget, hybrid-model checkpoints and per-layer cancellation; output matches mlx-lm |
| **Agent APIs** | Claude Code, Codex and OpenCode complete real tool loops, including MCP tools and deferred tool loading; a nightly check runs the latest client releases |
| **Prefix cache** | Exact reuse across turns, subagents and edits for attention and hybrid models, with miss reports |
| **Models and DX** | `pull`, `launch`, `connect`, `doctor`, `ps`; fresh install to working agent with one command |
| **Benchmarks** | `mrun bench` replays recorded agent sessions against llama.cpp, Ollama, oMLX, mlx-lm and SGLang; first dated public comparison, losing cells included |
| **Distribution** | Homebrew, a signed binary, a background service and `/metrics` |

## Next

Each comes after the first release, once measurements on recorded agent sessions justify it:

- **Speculative decoding.** Qwen3.6 ships multi-token prediction heads, and agents copy large parts
  of their context when editing files. Both can speed up generation without changing output. On
  mixture-of-experts models checking a draft costs more, so a controller turns speculation off
  where it does not pay.
- **Persistent cache on SSD**, opt-in, so long sessions survive server restarts.

## Later

- Concurrent sessions with batched decoding, if recordings show agents waiting on each other.
- Linux builds already compile and pass tests in CI; running models on Linux is evaluated after
  the Mac release, without performance commitments.

## Not planned

A web UI, running tools on the server, multi-model routing, multi-GPU or distributed serving,
training or fine-tuning, a custom quantization format, Ollama's native API, server-stored
Responses state, and approximate context reuse.
