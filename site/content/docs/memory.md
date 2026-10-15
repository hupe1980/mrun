+++
title = "Memory"
description = "How mrun budgets Apple Silicon unified memory for weights, attention state and scratch, so long agent sessions stay inside a fixed budget."
weight = 50
+++

On a Mac, the CPU and GPU share one pool of memory. A model server that grows without limits
competes with your editor, browser and build tools, and when the pool runs out macOS swaps and
everything slows down. mrun fixes a budget at startup and never exceeds it.

## The budget

```text
budget  = min(75% of physical memory, GPU recommended working set) − reserve
reserve = one sixth of physical memory, between 2 and 8 GiB
context = (budget − weights − scratch − recurrent state) ÷ attention bytes per token
```

- **Weights** are estimated from file sizes before loading (for `pull` and `ls`) and measured after.
- **Recurrent state** (hybrid models only) is the fixed-size state of the linear-attention layers,
  plus room for checkpoints.
- **Scratch** is the temporary memory one prompt chunk needs at the deepest position mrun will
  admit. mrun measures it once per model, machine and settings, and caches the result.
- **Context** is what remains, converted to tokens of attention state and capped by the model's
  maximum position.

`mrun run <model> --info` and `mrun ps` show every term. On a 48 GB Mac the budget is
`min(36, 36) − 8 = 28 GiB`:

| | Qwen3.6-35B-A3B (4-bit) | Qwen3-Coder-30B-A3B (4-bit) |
| --- | --- | --- |
| Weights | ≈19 GiB | ≈16 GiB |
| Attention state per token | ≈20 KiB (10 attention layers) | 96 KiB (48 layers × 4 KV heads × 128 dims × K and V × 2 bytes) |
| Context capacity | The full 262K window | ≈105K tokens |

The context capacity is shared between the current conversation and cached branches such as
subagents.

## Admission

Before a request touches the GPU, mrun checks that its prompt fits, evicting cached branches if
needed, and clamps the output limit to the space left. If the prompt itself cannot fit, the request
is refused immediately with the token counts, in the error format each API defines (for example
`context_length_exceeded`). mrun reports this machine's real context capacity in `GET /v1/models`,
and `mrun launch` passes it to the client, so agents compact their history before reaching the
limit.

Attention state is allocated as the conversation grows, in steps of 4,096 tokens, and never beyond
what the request reserved.

## Tuning

| Setting | Default | Effect |
| --- | --- | --- |
| `--memory-fraction` / `memory.fraction` | 0.75 | Share of physical memory mrun may plan for |
| `--reserve-gib` / `memory.reserve_gib` | ⅙ of RAM, 2–8 GiB | Memory left for other applications |
| `--allow-over` | off | Start even when the model does not fit the budget (at your own risk) |

Raising the fraction above 0.75 helps only up to macOS's recommended GPU working set and leaves
less memory for other apps.

## Under pressure

If macOS reports memory pressure, mrun drops cached branches first, then the state of an idle
conversation. A running request continues. The model stays loaded; it is never unloaded on a
timer, because reloading would throw away the cache.
