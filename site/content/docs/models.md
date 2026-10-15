+++
title = "Models"
description = "Supported model families, curated aliases, the shared Hugging Face cache and how mrun verifies model files."
weight = 60
+++

## Supported models

The first release supports the Qwen3 family in MLX format:

| Alias | Model | Type | Notes |
| --- | --- | --- | --- |
| `qwen3-coder` | `mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit` | Mixture of experts, 3B active | Default; tuned for agentic coding and tool use |
| `qwen3-8b` | `mlx-community/Qwen3-8B-4bit` | Dense | Fits smaller Macs; optional thinking mode |

Other Qwen3 and Qwen3-MoE checkpoints in the MLX community format also work by repository name:

```sh
mrun pull mlx-community/Qwen3-14B-4bit
mrun pull mlx-community/Qwen3-14B-4bit@<revision>
```

Hybrid-attention models such as Qwen3-Next and newer Qwen releases are on the
[roadmap](@/docs/roadmap.md).

## Where models live

Models are stored in the shared Hugging Face cache (`HF_HOME` or `HF_HUB_CACHE`; by default
`~/.cache/huggingface/hub`). A model already downloaded by mlx-lm, LM Studio or another tool using
that cache is reused. mrun keeps its own manifests, aliases and configuration in `~/.mrun`
(override with `MRUN_HOME`).

## Identity and verification

Each pulled model has a manifest with the pinned revision and the SHA-256 of every file. That
content identity is part of every prefix-cache key, so a changed file can never reuse stale state.
`mrun verify <model>` re-hashes the files; `mrun show <model>` prints the manifest, architecture,
quantization, license and how the model fits this Mac.

## Managing models

```sh
mrun ls                 # alias, repository, size, fits, max context, status
mrun show qwen3-coder
mrun rm qwen3-coder     # warns before removing files shared with other tools
```

## Thinking

Models with a thinking mode follow the client: thinking is on when the request asks for it
(Anthropic `thinking`, OpenAI `reasoning_effort` or `reasoning`) and off otherwise. Override with
`mrun serve --thinking on|off`. Qwen3-Coder has no thinking mode.

## Licenses

Model weights keep their own licenses; `mrun pull` shows the license before downloading. The
Qwen3 models listed above are Apache-2.0.
