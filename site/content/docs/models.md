+++
title = "Models"
description = "Supported models and aliases, the per-machine default, where models are stored, verification and thinking."
weight = 60
+++

## Aliases

Each alias is pinned to a Hugging Face revision with the hash of every file `pull` downloads.

| Alias | Model | Architecture | Tool format | Download | Context on a 48 GB Mac |
| --- | --- | --- | --- | --- | --- |
| `qwen3.5` | Qwen3.5-9B, 4-bit (`mlx-community/Qwen3.5-9B-4bit`) | Hybrid: 24 linear-attention (Gated DeltaNet) and 8 attention layers, dense (family `qwen3_5`) | Qwen XML | 6.0 GB | 262,144 (the model's maximum) |
| `qwen3.6` | Qwen3.6-35B-A3B, 4-bit (`mlx-community/Qwen3.6-35B-A3B-4bit`) | Hybrid: 30 linear-attention (Gated DeltaNet) and 10 attention layers, mixture of experts with 3B active (family `qwen3_5_moe`) | Qwen XML | 20.4 GB | 262,144 (the model's maximum) |

**The default is computed per machine**: the largest alias whose memory budget leaves at least
64K tokens (65,536) of context, counting each family's scratch (see
[Memory](@/docs/memory.md#the-budget)). `mrun ls` shows both aliases with the context each gets on
your Mac; `serve`, `run` and `launch` use the default when no model is named.

**`qwen3.5` is a preview.** It completes 26–28 of 30 [client gate](@/docs/clients.md#client-gate)
runs per client (`qwen3.6`: 29–30). Its failures are sessions where the model gets stuck (loops,
runaway reasoning), not server errors. `mrun ls` marks it `preview`, and `mrun show`, `launch` and
`doctor` say so. Expect it to handle short, well-defined edits.

Qwen3-8B is not the small alias: its 40,960-token window barely holds Claude Code's system prompt
and tools, and its state costs 144 KiB per token. Qwen3.5-9B has a 262K window at about 32 KiB per
token plus a fixed linear-attention state.

Greedy output of Qwen3.6 and Qwen3-8B is token-for-token identical to mlx-lm 0.32.0.

## Without an alias

Qwen3-8B (family `qwen3`: dense full attention, Hermes JSON tool calls), the reference model for
parity and tests:

```sh
mrun pull mlx-community/Qwen3-8B-4bit
```

The Qwen3-Coder family (`qwen3_moe`: full attention, mixture of experts, Qwen XML tool calls):

```sh
mrun pull mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit
```

Any other MLX repository whose architecture is `qwen3`, `qwen3_5`, `qwen3_moe` or `qwen3_5_moe`
loads the same way, by `org/name[@revision]` or from a local directory; these are not tested with
coding agents. Other architectures are refused with the list of supported families (E034).

## How model support works

- **A family module per architecture**, ported from
  [mlx-lm](https://github.com/ml-explore/mlx-lm) and checked token for token against it.
- **A chat syntax per model**: its chat template plus one parser for tool calls and reasoning. The
  parser switches modes on special-token ids, never on text, so a `<tool_call>` string inside a
  file stays text. Qwen XML argument values are converted using the tool's JSON schema (booleans,
  numbers, unions). Calls come back in each API's native tool-call structure.
- **Hybrid state stays reusable** through checkpoints; see [Prefix cache](@/docs/prefix-cache.md).

## Where models live

Models are stored in `~/.mrun/models/<org>--<name>/<revision>/` (under `MRUN_HOME`), manifests in
`~/.mrun/manifests`. `pull` realigns each weights file so it can be mapped from disk (it pads the
safetensors header with up to 15 spaces; no tensor byte changes). A loaded model then uses its
files on disk instead of a copy in memory.

A model already in the Hugging Face cache (`HF_HUB_CACHE`, else `$HF_HOME/hub`, by default
`~/.cache/huggingface/hub`) is imported from there instead of downloaded; `pull` then says how much
that cache still holds for it. mrun never writes to the Hugging Face cache. A model that is only in
that cache still runs by name, read into memory instead of mapped.

Hub metadata (revision hashes, file ids) is checked for its expected form before it is used in a
path. Gated models use `HF_TOKEN` or the token saved by `hf auth login`; `HF_ENDPOINT` selects a
mirror.

## Identity and verification

Each pulled model has a manifest with the pinned revision and the hash of every file that affects
output: weights, tokenizer, chat template and generation settings. That identity is part of the
cache key, so a changed file never reuses stale state. `mrun verify <model>` re-hashes the files
(realigned files are hashed without the padding). `mrun show <model>` prints family, architecture,
quantization, license, identity and how the model fits this Mac.

## Managing models

```sh
mrun ls                 # aliases and pulled models: family, size, fit, max context, status
mrun ls --json          # the same as JSON
mrun pull qwen3.6       # -y: download even if it does not fit; --json: JSON lines
mrun show qwen3.6
mrun rm qwen3.5         # asks first, and again if the running server serves it
```

## Thinking

Qwen3, Qwen3.5 and Qwen3.6 are thinking models. On Chat Completions and Responses they think
unless the request sets `reasoning_effort` (or `reasoning.effort`) to `none` or `minimal`; Chat
Completions also accepts `chat_template_kwargs.enable_thinking`. On Anthropic Messages they think
only when `thinking` is enabled or adaptive. `mrun run --think` turns thinking on in the terminal
chat. Qwen3-Coder has no thinking mode.

## Licenses

Model weights keep their own licenses; `mrun pull` shows the license before downloading. Qwen3.5,
Qwen3.6 and Qwen3-8B are Apache-2.0.
