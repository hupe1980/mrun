+++
title = "Roadmap"
description = "What is planned for mrun 0.7 and 0.8."
weight = 110
+++

Each release has measurable gates on the target Mac (M4 Pro, 48 GB). Results from other Macs are
welcome as
[hardware reports](https://github.com/hupe1980/mrun/blob/main/CONTRIBUTING.md#hardware-reports)
and never gate a release.

## 0.7: models that fit

| Item | What |
| --- | --- |
| `qwen3.8` | Qwen3.8-27B as the quality option on Macs with 32 GB or more, with its MTP drafter, bit-identical as prompt lookup is |
| Small model | Granite 4, LFM2 and gpt-oss-20b against the `qwen3.5` preview for 16 GB Macs |
| Comparison | The first dated `mrun bench` comparison against llama.cpp, Ollama, oMLX and mlx-lm, published whatever it shows |

## 0.8: bigger Macs, more clients

| Item | What |
| --- | --- |
| 64 GB | Qwen3-Coder-Next (`qwen3_next` family) |
| 96–128 GB | Qwen3.5-122B-A10B (`qwen3_5_moe` family) |
| Restart tier | Exact prefix entries on SSD that survive a server restart |
| Images | Screenshots and image tool results, part of the exact cache key |
| Clients | Kilo Code or Cline, Qwen Code, Goose, Copilot BYOK |
| Service | A background login service; non-loopback addresses with API keys |

## Later

A pool keeps one request per Mac in flight, so several people's agents run at once and a pool's
throughput grows with its Macs.

More model families, ordered by agentic score per GB. An opt-in 8-bit KV cache if a long-context
study shows no quality cost. Shared computation for lanes that receive the same prompt.
Notarization and an SBOM.

## Not planned

A model marketplace, running tools on the server, multi-model routing, tensor-parallel
serving, training or fine-tuning, a custom quantization format, Ollama's native API,
server-stored Responses state, lossy context reuse by default, and performance work on Linux or
CUDA.
