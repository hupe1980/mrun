+++
title = "Models"
description = "Supported models in the first release (Qwen3.6, Qwen3-Coder, gpt-oss), how model support works, curated aliases, the shared Hugging Face cache and model verification."
weight = 60
+++

## Supported models

mrun supports an open set of model families. The first release (0.1) covers three models:

| Alias | Model | Architecture | Tool format | Size | Context on a 48 GB Mac |
| --- | --- | --- | --- | --- | --- |
| `qwen3.6` | Qwen3.6-35B-A3B, 4-bit | Hybrid: 30 linear-attention + 10 attention layers, mixture of experts (3B active) | Qwen XML | ≈20 GB | Full 262K |
| `qwen3-coder` | Qwen3-Coder-30B-A3B-Instruct, 4-bit | Attention, mixture of experts (3B active) | Qwen XML | 17.2 GB | ≈105K |
| `gpt-oss` | gpt-oss-20b, MXFP4 | Alternating sliding-window and full attention, mixture of experts | Harmony | ≈13 GB | Full 131K |

Context figures are estimates from the [memory budget](@/docs/memory.md); `mrun pull` shows the
exact value for your Mac. Each of these models is tested with Claude Code, Codex and OpenCode
before a release.

After 0.1, Devstral Small 2 (Mistral tool format) comes next, then GLM-4.7-Flash, Gemma 4 and
Nemotron-H, then other families as coding agents adopt them (see the
[roadmap](@/docs/roadmap.md)). These are planned, not supported yet: until a release lists them
here, `mrun pull` refuses them.

**The default model.** `qwen3.6` is the expected default: Alibaba's model card reports 73.4% on
SWE-bench Verified (full-precision weights, Alibaba's harness; not measured by mrun), and three of
every four layers use linear attention, so a token of context costs about 20 KiB, against 96 KiB
for the full-attention Qwen3-Coder-30B-A3B, and the full 262K window fits a 48 GB Mac. The default
is chosen among the models above by running the `mrun bench` agent suite before each release.

## How model support works

- **Families from shared blocks.** A model family is a small module that combines shared building
  blocks: full and sliding-window attention, linear attention, dense and mixture-of-experts
  layers. Each family is ported from the reference [mlx-lm](https://github.com/ml-explore/mlx-lm)
  implementation and checked token for token against it.
- **Tool-format registry.** Models write tool calls in different formats (Qwen XML, Hermes JSON,
  Harmony). Each family names its format, and mrun checks at load that the model's chat template
  uses it; it parses the model's calls into each API's native tool-call structure, so your agent
  sees the same shape whichever model runs.
- **Unknown architectures are refused.** A model whose architecture mrun does not know fails at
  load with the list of supported families, instead of running with wrong output.
- **Every position stays reusable.** Sliding-window layers keep their full attention state and
  apply the window as a mask, so a cached prefix can be cut at any token, as with plain attention.
  Linear-attention state exists only for the position it was computed at, so mrun saves
  checkpoints at the end of the system prompt and tools, at the start of the last user or tool
  message, and where the reply starts, while it processes a prompt, and resumes from the nearest
  one; see [Prefix cache](@/docs/prefix-cache.md).

Other MLX repositories of a supported architecture load by repository name; they are not tested
with agents the way the aliases above are.

```sh
mrun pull <org>/<name>
mrun pull <org>/<name>@<revision>
```

## Where models live

Models are stored in the shared Hugging Face cache (`HF_HOME` or `HF_HUB_CACHE`; by default
`~/.cache/huggingface/hub`). A model already downloaded by mlx-lm or the `hf` CLI is reused:
`mrun pull` of a repository that is already there verifies it and records it without downloading
again. mrun keeps its manifests and aliases in `~/.mrun` (override with `MRUN_HOME`).

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
`none`/`minimal`) and off otherwise. gpt-oss always reasons: "off" selects low reasoning effort,
and its reasoning is always returned in the API's reasoning field. Qwen3-Coder has no thinking
mode and answers without it.

## Licenses

Model weights keep their own licenses; `mrun pull` shows the license before downloading. The
models listed above are Apache-2.0; gpt-oss additionally comes with OpenAI's usage policy.
