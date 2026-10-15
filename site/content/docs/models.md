+++
title = "Models"
description = "Supported model families (Qwen3.6, Qwen3-Coder, gpt-oss, Devstral, Qwen3-8B), how model support works, curated aliases, the shared Hugging Face cache and model verification."
weight = 60
+++

## Supported models

mrun supports an open set of model families. The first release covers these:

| Alias | Model | Architecture | Tool format | Size | Context on a 48 GB Mac |
| --- | --- | --- | --- | --- | --- |
| `qwen3.6` | Qwen3.6-35B-A3B, 4-bit | Hybrid: 30 linear-attention + 10 attention layers, mixture of experts (3B active) | Qwen XML | ≈20 GB | Full 262K |
| `qwen3-coder` | Qwen3-Coder-30B-A3B-Instruct, 4-bit | Attention, mixture of experts (3B active) | Qwen XML | 17.2 GB | ≈105K |
| `gpt-oss` | gpt-oss-20b, MXFP4 | Alternating sliding-window and full attention, mixture of experts | Harmony | ≈13 GB | Full 131K |
| `devstral` | Devstral Small 2 24B, 4-bit | Attention, dense | Mistral | ≈14 GB | ≈85K |
| `qwen3-8b` | Qwen3-8B, 4-bit | Attention, dense | Hermes JSON | 4.6 GB | Full 40K |

Context figures are estimates from the [memory budget](@/docs/memory.md); `mrun pull` shows the
exact value for your Mac. After the first release, GLM-4.7-Flash, Gemma 4 and Nemotron-H follow,
then other families as coding agents adopt them (see the [roadmap](@/docs/roadmap.md)). These
later families are planned, not supported yet: until a release lists them here, `mrun pull`
refuses them.

**The default model.** `qwen3.6` is the expected default: Alibaba's model card reports 73.4% on
SWE-bench Verified (full-precision weights, Alibaba's harness; not measured by mrun), and three of
every four layers use linear attention, so a token of context costs about 20 KiB, against 96 KiB
for the full-attention Qwen3-Coder-30B-A3B, and the full 262K window fits a 48 GB Mac. The default is chosen
among the models above by running the `mrun bench` agent suite before each release.

## How model support works

- **Families from shared blocks.** A model family is a small module that combines shared building
  blocks: full, sliding-window and (later) compressed attention; linear attention; dense and
  mixture-of-experts layers. Each family is ported from the reference
  [mlx-lm](https://github.com/ml-explore/mlx-lm) implementation and checked token for token
  against it.
- **Tool-format registry.** Models write tool calls in different formats (Qwen XML, Hermes JSON,
  Harmony, Mistral). A family names its format; mrun parses it into each API's native tool-call
  structure, so your agent sees the same shape whichever model runs.
- **Unknown architectures are refused.** A model whose architecture mrun does not know fails at
  load with the list of supported families, instead of running with wrong output.
- **Every position stays reusable.** Sliding-window layers keep their full attention state and
  apply the window as a mask, so a cached prefix can be cut at any token, as with plain attention.
  Linear-attention state exists only for the position it was computed at, so mrun saves
  checkpoints at the end of the system prompt and tools, at the start of the last user or tool
  message, and where the reply starts, while it processes a prompt, and resumes from the nearest one;
  see [Prefix cache](@/docs/prefix-cache.md).

Other checkpoints of a supported family in MLX format also work by repository name:

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
(Anthropic `thinking`, OpenAI `reasoning_effort` or Responses `reasoning.effort` other than
`none`/`minimal`) and off otherwise. Override with
`mrun serve --thinking on|off`. gpt-oss always reasons: "off" selects low reasoning effort, and
its reasoning is always returned in the API's reasoning field. Qwen3-Coder and Devstral have no
thinking mode; a request for thinking gets a `thinking-unsupported` warning.

## Licenses

Model weights keep their own licenses; `mrun pull` shows the license before downloading. The
models listed above are Apache-2.0; gpt-oss additionally comes with OpenAI's usage policy.
