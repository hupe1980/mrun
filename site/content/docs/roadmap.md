+++
title = "Roadmap"
description = "What mrun ships first and what comes next: engine, agent APIs, prefix cache, developer experience, benchmarks, then hybrid models, SSD cache, speculative decoding and more."
weight = 110
+++

mrun is in active development; nothing is released yet. Each milestone has a measurable gate on
the target Mac (M4 Pro, 48 GB) before the next one starts.

## First release

| Milestone | Outcome | Gate |
| --- | --- | --- |
| **Engine** | `mrun run` streams Qwen3-8B and Qwen3-Coder-30B-A3B with chunked prefill, per-layer cancellation and the memory budget | Greedy output identical to mlx-lm; at least 90% of its speed; cancellation under 300 ms |
| **Agent APIs** | Claude Code, Codex and OpenCode complete real tool loops | Conformance suite green; 5 of 5 real-client runs per client |
| **Prefix cache** | Exact reuse across turns, branches and tool loops, with miss reports | Repeated turns at 30K context at least 5× faster than without cache; zero stale reuse |
| **Models and DX** | `pull`, `ls`, `rm`, `launch`, `connect`, `doctor`, `ps` | Fresh Mac to working Claude Code session in under 5 minutes |
| **Benchmarks** | `mrun bench` with agent-trace replay | First dated public comparison, losing cells included |

## Next

These come after the first release, each once measurements on real agent traces justify it:

- **Newer models with hybrid attention**, such as Qwen3-Next and later Qwen releases. Their
  recurrent layers can't be trimmed like attention state, so the prefix cache stores state
  checkpoints at message boundaries.
- **Persistent prefix cache on SSD**, opt-in, so long sessions survive server restarts. Reading a
  30K-token prefix from a Mac SSD is expected to take well under a second instead of a minute of
  recomputation.
- **Speculative decoding for code edits.** Agents copy large parts of their context verbatim when
  editing files; drafting tokens from the prompt itself (n-gram lookup) can speed that up. Model
  multi-token-prediction heads follow where available. Mixture-of-experts models need careful
  tuning here, so it stays adaptive and falls back to normal decoding.
- **8-bit attention state** to roughly double context capacity on the same Mac.
- **Native cache-miss reasons** in the Anthropic API format, `/metrics`, and Homebrew packaging.

## Later

- Concurrent sessions with batched decoding, if traces show agents waiting on each other.
- Linux builds already compile and pass tests in CI; running models on Linux is evaluated after
  the Mac release, without performance commitments.

## Not planned

A web UI, running tools on the server, multi-model routing, multi-GPU or distributed serving,
training or fine-tuning, a custom quantization format, Ollama's native API, and server-stored
Responses state.
