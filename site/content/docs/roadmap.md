+++
title = "Roadmap"
description = "What is planned for mrun 0.8 and 0.9."
weight = 110
+++

Each release has measurable gates on the target Mac (M4 Pro, 48 GB). Results from other Macs are
welcome as
[hardware reports](https://github.com/hupe1980/mrun/blob/main/CONTRIBUTING.md#hardware-reports)
and never gate a release.

## 0.8: the best model that fits, reuse that survives

| Item | What |
| --- | --- |
| `qwen3.8` | Qwen3.8-27B as the quality option on Macs with 32 GB or more, with its MTP drafter, bit-identical as prompt lookup is |
| Restart tier | Exact prefix entries, hybrid states included, on SSD: a restart or a long pause resumes where the agent left off |
| Small model | Granite 4, LFM2 and gpt-oss-20b against the `qwen3.5` preview for 16 GB Macs |
| Comparison | The first dated `mrun bench` comparison against llama.cpp, Ollama, oMLX and mlx-lm on replayed agent sessions, published whatever it shows |

## 0.9: bigger Macs, more clients

| Item | What |
| --- | --- |
| 64 GB | Qwen3-Coder-Next (`qwen3_next` family) |
| 96–128 GB | Qwen3.5-122B-A10B (`qwen3_5_moe` family) |
| Block drafter | A DFlash-style drafter where it beats MTP, still bit-identical |
| Shared prefill | Lanes that receive the same prompt (subagents fanning out) read it once |
| Clients | Kilo Code or Cline, Qwen Code, Goose, Copilot BYOK |
| Service | A background login service; non-loopback addresses with API keys |

## Later

A pool keeps one request per Mac in flight, and uses RDMA over Thunderbolt 5 where both Macs have
it. More model families by agentic score per GB, and FP4 weights (MXFP4, NVFP4) where they keep
quality. An opt-in quantized KV cache, labeled lossy, if a long-context study shows no quality
cost. Notarization and an SBOM.

## Not planned

A model marketplace, running tools on the server, multi-model routing, tensor-parallel serving,
streaming weights from SSD for models larger than memory, the Neural Engine, training or
fine-tuning, a custom quantization format, embeddings or speech, Ollama's native API,
server-stored Responses state, lossy context reuse by default, and performance work on Linux or
CUDA.
