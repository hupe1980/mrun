+++
title = "Roadmap"
description = "What mrun ships first and what comes next: engine with an open set of model families, agent APIs, prefix cache, developer experience, benchmarks and distribution, then speculative decoding and a persistent cache."
weight = 110
+++

mrun is in active development; nothing is released yet. Each milestone has a measurable gate on
the target Mac (M4 Pro, 48 GB) before it counts as done.

## First release

| Milestone | Outcome |
| --- | --- |
| **Foundation** | MLX bindings proven on a Mac: quantized and mixture-of-experts kernels, attention, mlx-lm's Metal kernel for the hybrid layers, memory controls |
| **Engine** | `mrun run` streams the first model families — Qwen3.6-35B-A3B, Qwen3-Coder-30B-A3B, gpt-oss-20b, Devstral Small 2 and Qwen3-8B — built from shared blocks, with the memory budget, hybrid-model checkpoints and per-layer cancellation; output matches mlx-lm |
| **Agent APIs** | Claude Code, Codex and OpenCode complete real tool loops, including MCP tools and deferred tool loading; a nightly check runs the latest client releases |
| **Prefix cache** | Exact reuse across turns, subagents and edits for attention, sliding-window and hybrid models, with miss reports |
| **Models and DX** | `pull`, `launch`, `connect`, `doctor`, `ps`; fresh install to working agent with one command |
| **Benchmarks** | `mrun bench` replays recorded agent sessions against llama.cpp, Ollama, oMLX, mlx-lm and SGLang; first dated public comparison, losing cells included |
| **Distribution** | Homebrew, a signed binary, a background service and `/metrics` |

## Next

Each comes after the first release, once measurements on recorded agent sessions justify it:

- **More model families.** GLM-4.7-Flash, Gemma 4 and Nemotron-H, then other families by demand;
  each is one module from the shared blocks plus a check against mlx-lm.
- **Speculative decoding for code edits.** Agents copy large parts of their context when editing
  files; drafting those tokens from the prompt itself and checking them in one pass can speed up
  generation without changing the output beyond the rounding any chunked prefill has. On
  mixture-of-experts models checking a draft costs more, so a controller turns speculation off where it does not pay. Qwen3.6's multi-token prediction heads
  follow only if measurements show they help on this model.
- **Persistent cache on SSD**, opt-in, so long sessions survive server restarts.
- **Image input.** Qwen3.6 can read images; screenshots and image tool results will reach the model
  instead of being replaced by a placeholder, with each image part of the exact cache key.

## Later

- Concurrent sessions with batched decoding, if recordings show agents waiting on each other.
- Linux builds already compile and pass tests in CI; running models on Linux is evaluated after
  the Mac release, without performance commitments.

## Not planned

A web UI, running tools on the server, multi-model routing, multi-GPU or distributed serving,
training or fine-tuning, a custom quantization format, Ollama's native API, server-stored
Responses state, and approximate context reuse.
