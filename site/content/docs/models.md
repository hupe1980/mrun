+++
title = "Models"
description = "Supported models (Qwen3.6-35B-A3B by default, Qwen3-Coder-30B-A3B, Qwen3-8B), curated aliases, the shared Hugging Face cache and model verification."
weight = 60
+++

## Supported models

| Alias | Model | Architecture | Size | Context on a 48 GB Mac |
| --- | --- | --- | --- | --- |
| `qwen3.6` | Qwen3.6-35B-A3B, 4-bit | Hybrid: 30 linear-attention + 10 attention layers, mixture of experts (3B active) | ≈20 GB | Full 262K |
| `qwen3-coder` | Qwen3-Coder-30B-A3B-Instruct, 4-bit | Attention, mixture of experts (3B active) | 17.2 GB | ≈105K |
| `qwen3-8b` | Qwen3-8B, 4-bit | Attention, dense | 4.6 GB | Full 40K |

**Why Qwen3.6 is the default.** It is the strongest coding model that fits a 48 GB Mac
(Alibaba reports 73.4% on SWE-bench Verified, against about 52% for Qwen3-Coder-30B-A3B). Three of
every four layers use linear attention, which keeps a fixed-size state instead of one entry per
token, so a token of context costs about 20 KiB instead of 96 KiB. The default is re-checked
against newer releases with `mrun bench` before each release.

**Why hybrid models need special caching.** Attention state can be cut at any token, so a cached
prefix of any length is reusable. Linear-attention state only exists for the position it was
computed at. mrun therefore saves checkpoints at message boundaries while it processes a prompt
and resumes from the nearest one; see [Prefix cache](@/docs/prefix-cache.md).

Other Qwen3, Qwen3-MoE and Qwen3.6 checkpoints in MLX format also work by repository name:

```sh
mrun pull mlx-community/Qwen3-14B-4bit
mrun pull mlx-community/Qwen3-14B-4bit@<revision>
```

## Where models live

Models are stored in the shared Hugging Face cache (`HF_HOME` or `HF_HUB_CACHE`; by default
`~/.cache/huggingface/hub`). A model already downloaded by mlx-lm or the `hf` CLI is reused; files
that differ in a newer revision are downloaded again. `mrun ls` also lists compatible models it
finds there, and `mrun pull <repo>` pins them without downloading. mrun keeps its manifests,
aliases and configuration in `~/.mrun` (override with `MRUN_HOME`).

Gated models use `HF_TOKEN` or the token saved by `hf auth login`.

## Identity and verification

Each pulled model has a manifest with the pinned revision and the SHA-256 of every file that
affects output: weights, tokenizer, chat template and generation settings. That identity is part of
every cache key, so a changed file can never reuse stale state. `mrun verify <model>` re-hashes the
files; `mrun show <model>` prints the manifest, architecture, quantization, license and how the
model fits this Mac.

## Managing models

```sh
mrun ls                 # alias, repository, size, fits, max context, status
mrun show qwen3.6
mrun rm qwen3-coder     # warns if other tools use the same revision
```

## Thinking

Models with a thinking mode follow the client: thinking is on when the request asks for it
(Anthropic `thinking`, OpenAI `reasoning_effort` or `reasoning`) and off otherwise. Override with
`mrun serve --thinking on|off`. Qwen3-Coder has no thinking mode.

## Licenses

Model weights keep their own licenses; `mrun pull` shows the license before downloading. The
models listed above are Apache-2.0.
