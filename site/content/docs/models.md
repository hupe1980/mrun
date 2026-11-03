+++
title = "Models"
description = "The models mrun supports (Qwen3.5-9B and Qwen3.6-35B-A3B as aliases, Qwen3-8B and Qwen3-Coder by repository id), the per-machine default, the shared Hugging Face cache and model verification."
weight = 60
+++

## Aliases

Each alias is pinned to a Hugging Face revision with the hash of every file `pull` downloads.

| Alias | Model | Architecture | Tool format | Download | Context on a 48 GB Mac |
| --- | --- | --- | --- | --- | --- |
| `qwen3.5` | Qwen3.5-9B, 4-bit (`mlx-community/Qwen3.5-9B-4bit`) | Hybrid: 24 linear-attention (Gated DeltaNet) and 8 attention layers, dense (family `qwen3_5`) | Qwen XML | 6.0 GB | 262,144 (the model's maximum) |
| `qwen3.6` | Qwen3.6-35B-A3B, 4-bit (`mlx-community/Qwen3.6-35B-A3B-4bit`) | Hybrid: 30 linear-attention (Gated DeltaNet) and 10 attention layers, mixture of experts with 3B active (family `qwen3_5_moe`) | Qwen XML | 20.4 GB | 262,144 (the model's maximum) |

**The default is computed per machine**: the largest alias whose memory budget leaves at least
64K tokens (65,536) of context, counting the scratch measured for each family (see
[Memory](@/docs/memory.md#the-budget)). There is no fixed RAM threshold. `mrun ls` shows both
aliases with the context each gets on your Mac; `serve`, `run` and `launch` use the default when
no model is named. Smaller Macs are simulated with `--budget-gib`
([Memory](@/docs/memory.md#smaller-macs)).

**`qwen3.5` is a preview.** It is below the [client gate](@/docs/clients.md#client-gate) with
every client (26–28 of 30 runs; `qwen3.6` completes 29–30). Its failures are sessions where the
9B model gets stuck (loops, runaway reasoning), not server errors. It is the default where
`qwen3.6` does not fit; `mrun ls` marks it `preview`, and `mrun show`, `launch` and `doctor` say
so. Expect it to handle short, well-defined edits.

Qwen3-8B is not the small alias: its 40,960-token window barely holds Claude Code's system prompt,
tools and one large MCP server, and its state costs 144 KiB per token. Qwen3.5-9B has a 262K
window at about 32 KiB per token plus a fixed linear-attention state.

Greedy output of Qwen3.6 and Qwen3-8B is token-for-token identical to mlx-lm 0.32.0 (128 of 128
tokens on a 33-token and a 4,350-token prompt, M4 Pro, 48 GB).

## Without an alias

Qwen3-8B (family `qwen3`: dense full attention, Hermes JSON tool calls) is the reference model for
parity and tests:

```sh
mrun pull mlx-community/Qwen3-8B-4bit
```

The Qwen3-Coder family (`qwen3_moe`: full attention, mixture of experts, Qwen XML tool calls):

```sh
mrun pull mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit
```

Any other MLX repository whose architecture is `qwen3`, `qwen3_5`, `qwen3_moe` or `qwen3_5_moe`
loads the same way, by `org/name[@revision]` or from a local directory. These are not part of the
release tests with coding agents. Any other architecture is refused with the list of supported
families (E034).

## How model support works

- **A family module per architecture**, ported from the reference
  [mlx-lm](https://github.com/ml-explore/mlx-lm) implementation and checked token for token
  against it.
- **A chat syntax per model**: its own chat template plus one parser for tool calls and
  reasoning. The parser switches modes on special-token ids, never on text, so a `<tool_call>`
  string inside a file the model reads stays text. Qwen XML argument values are converted using
  the tool's JSON schema (booleans, numbers, unions), as Qwen's reference parser does. Calls come
  back in each API's native tool-call structure.
- **Hybrid state stays reusable.** Linear-attention state exists only for the position it was
  computed at, so mrun saves checkpoints of it while it processes a prompt and resumes from the
  nearest one; see [Prefix cache](@/docs/prefix-cache.md).

## Where models live

Models are stored in the shared Hugging Face cache (`HF_HUB_CACHE`, else `$HF_HOME/hub`, by
default `~/.cache/huggingface/hub`). A model already downloaded by mlx-lm or the `hf` CLI is
reused: `mrun pull` of a revision that is already there verifies and records it without
downloading again. Metadata from the hub (revision hashes, file ids) is checked for its expected
form before it is used in any path, so a misbehaving mirror cannot make mrun write or remove
files outside the cache. mrun keeps its manifests in `~/.mrun/manifests` (`MRUN_HOME`).

Gated models use `HF_TOKEN` or the token saved by `hf auth login`; `HF_ENDPOINT` selects a
mirror.

## Identity and verification

Each pulled model has a manifest with the pinned revision and the hash of every file that affects
output: weights, tokenizer, chat template and generation settings. That identity is part of the
cache key, so a changed file never reuses stale state. `mrun verify <model>` re-hashes the files;
`mrun show <model>` prints family, architecture, quantization, license, identity and how the model
fits this Mac.

## Managing models

```sh
mrun ls                 # aliases and pulled models: family, size, fit, max context, status
mrun ls --json          # the same as JSON
mrun pull qwen3.6       # -y: download even if it does not fit; --json: JSON lines
mrun show qwen3.6
mrun rm qwen3.5         # asks first; warns that other tools using this revision lose it
```

## Thinking

Qwen3, Qwen3.5 and Qwen3.6 are thinking models. On Chat Completions and Responses they think
unless the request sets `reasoning_effort` (or `reasoning.effort`) to `none` or `minimal`. On
Anthropic Messages they think only when `thinking` is enabled or adaptive. `mrun run --think`
turns thinking on in the terminal chat. Qwen3-Coder has no thinking mode.

## Licenses

Model weights keep their own licenses; `mrun pull` shows the license before downloading. Qwen3.5,
Qwen3.6 and Qwen3-8B are Apache-2.0.
