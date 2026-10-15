+++
title = "Memory"
description = "How mrun budgets Apple Silicon unified memory for weights, attention state and scratch, so long agent sessions never push your Mac into swap."
weight = 50
+++

On a Mac, the CPU and GPU share one pool of memory. A model server that grows without limits
competes with your editor, browser and build tools, and when the pool runs out macOS swaps and
everything slows down. mrun avoids that by fixing a budget at startup and never exceeding it.

## The budget

```text
budget  = min(75% of physical memory, GPU recommended working set) − reserve (8 GiB)
context = (budget − weights − scratch) ÷ attention bytes per token
```

- **Weights** are measured after loading, not estimated from file sizes.
- **Scratch** is the temporary memory a prefill chunk needs. mrun measures it once per model and
  machine and caches the result.
- **Context** is what remains, converted to tokens of attention state, capped by the model's
  maximum position.

`mrun run <model> --info` and `mrun ps` show every term. A worked example for Qwen3-Coder-30B-A3B
(4-bit) on a 48 GB Mac:

| Term | Value |
| --- | --- |
| Budget | 36 GiB − 8 GiB = 28 GiB |
| Weights | about 16 GiB |
| Scratch | about 1.5 GiB |
| Attention state per token | 96 KiB (48 layers × 4 KV heads × 128 dims × keys and values × 2 bytes) |
| Context capacity | about 100,000–110,000 tokens |

## Admission

Before a request touches the GPU, mrun checks that its prompt plus its output limit fits — after
evicting cached prefixes if needed. If it cannot fit, the request is refused immediately with the
token counts, in the error format each API defines (for example `context_length_exceeded`). The
model reports this machine's real context capacity in `GET /v1/models`, and `mrun launch` passes it
to the client, so agents compact their history before hitting the limit.

Attention state is allocated as the conversation grows, in steps of 4,096 tokens, so a short
conversation does not hold memory for 100,000 tokens.

## Tuning

| Setting | Default | Effect |
| --- | --- | --- |
| `--memory-fraction` / `memory.fraction` | 0.75 | Share of physical memory mrun may plan for |
| `--reserve-gib` / `memory.reserve_gib` | 8 | Memory left for other applications |
| `--allow-over` | off | Start even when the model does not fit the budget (at your own risk) |

On a 24 or 36 GB Mac that runs little else, lowering the reserve to 4 GiB buys a lot of context.
Raising the fraction above 0.75 lets the GPU use more memory than macOS recommends and risks swap.

## Under pressure

If macOS reports memory pressure while mrun is running, mrun drops cached prefixes first. The
active request continues; nothing else is pushed out.
