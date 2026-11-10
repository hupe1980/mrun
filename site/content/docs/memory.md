+++
title = "Memory"
description = "How mrun fits weights, sequence state, cache and scratch into a memory budget fixed at startup."
weight = 50
+++

The CPU and GPU of a Mac share one pool of memory. mrun fixes a budget at startup and keeps
weights, sequence state, cache and scratch inside it.

## The budget

```text
budget  = min(Metal's recommended working set, physical − 2 GiB, available − margin)   (lowered by --budget-gib)
state   = budget − weights − scratch        (live conversation, cached branches, checkpoints)
context = (state − 3 × linear-attention state) ÷ attention bytes per token, capped by the model's maximum
```

- **The working set** is what macOS recommends for the GPU (Metal's device info): about two
  thirds of memory on smaller Macs, more on larger ones. A raised wired-memory limit is honoured,
  but the budget stays at least 2 GiB below physical memory. `available − margin` is explained
  under [Other apps and VMs](#other-apps-and-vms).
- **Weights** are estimated from file sizes before loading and measured after warm-up.
- **Scratch** is the temporary memory a request needs beyond its state:
  `(warm-up peak above the weights + computed term) × 1.25`. The computed term covers the largest
  prompt chunk's activations, expert routing and attention scores, one attention layer copy, one
  linear-attention state copy and nuvec's buffer cache (limited to 256 MiB). Terms that grow with
  position are sized at the depth the state can reach on this Mac. Before a model is loaded
  (`pull`, `ls`, the default alias), the scratch measured for its family is used.
- **State** holds the live conversation, cached branches and checkpoints of the linear-attention
  state. The context leaves room for the live linear-attention state and two checkpoints.

Loading checks the fit three times and refuses with every number: from the file sizes, with the
computed scratch before warm-up, and with the measured numbers after it.

Example, M4 Pro with 48 GB (budget 37.4 GiB):

| | Qwen3.6-35B-A3B (4-bit) | Qwen3-8B (4-bit) |
| --- | --- | --- |
| Weights | 18.2 GiB | 4.3 GiB |
| Scratch | 5.4 GiB | 1.3 GiB |
| State | 13.8 GiB | 31.8 GiB |
| Attention state per token | 20 KiB (10 attention layers) | 144 KiB |
| Context | 262,144 tokens (the model's maximum) | 40,960 tokens (the model's maximum) |

With Qwen3.6, the full 262K window needs about 5 GiB; the rest of the state holds branches and
checkpoints. `mrun run --info` and `mrun ps` show every term on your Mac; `mrun ls` shows the
context each alias would get. The default model is the largest alias that leaves at least 64K
tokens (65,536).

## Other apps and VMs

mrun also reads the memory *available* at startup (free, inactive, purgeable and speculative
pages, as `vm_stat` shows them) and keeps a margin of physical memory ÷ 8, between 2 and 6 GiB,
for other apps.

- If the model's weights, scratch and 64K tokens of context do not fit what is available,
  `mrun serve`, `run` and `launch` refuse before reading the weights (E041), naming the three
  largest other processes and a smaller model that fits:

  ```text
  error[E041]: qwen3.6 needs 25.1 GiB but only 14.2 GiB are available (20.2 GiB free, less a
  6.0 GiB margin for other apps); largest other processes:
  com.apple.Virtualization.VirtualMachine 32.3 GiB, rust-analyzer 4.3 GiB, Code Helper 2.2 GiB
    hint: Quit some, or use qwen3.5.
  ```

- The weights are wired only if they fit what is available; otherwise they load unwired and mrun
  logs one line.
- `mrun ls` uses the same budget, so its FITS and MAX CTX columns reflect the apps running now.
  The default model does not change with other apps.
- `mrun doctor` shows available memory, which term binds, every other process of 4 GiB or more
  and every virtual machine.

A VM's guest memory belongs to one host process (`com.apple.Virtualization.VirtualMachine` for
Apple's framework, Docker Desktop and UTM; `qemu-system-…`). Stop the VM or lower its memory before
serving a large model. Available memory is read once; restart mrun to use memory freed later.

`mrun bench` and the client-gate script check the same before each server: no other inference
server may run, and available memory must hold the next model plus the margin.

## Smaller Macs

Smaller Macs are simulated by setting `--budget-gib` to their working set, which reproduces fit,
the default model and admission:

| Mac | Recommended working set | Default | `qwen3.5` context |
| --- | --- | --- | --- |
| 16 GB | ≈ 10.67 GiB | `qwen3.5` | 84,333 tokens |
| 24 GB | 16 GiB | `qwen3.5` | 235,385 tokens |
| 32 GB | ≈ 21.3 GiB | `qwen3.5` | — |
| 48 GB (M4 Pro, measured) | 37.4 GiB | `qwen3.6` | 262,144 tokens |

Simulation does not show a slower GPU, M5 Neural Accelerator speed or real memory pressure.
`mrun doctor --report` prints what a
[hardware report](https://github.com/hupe1980/mrun/blob/main/CONTRIBUTING.md#hardware-reports)
needs.

## Admission

Before a request touches the GPU, mrun checks that its prompt fits, evicting cached state if
needed, and clamps the output limit to the space left (`x-mrun-max-tokens`). A prompt that cannot
fit is refused immediately with the token counts, in each API's error format (for example
`context_length_exceeded`), and the cache is left intact. `GET /v1/models` reports the context
capacity, and `mrun launch` passes it to the client so agents compact before the limit. `launch`
refuses a model below the client's minimum (64K tokens for Claude Code) and names the budget it
would need.

## Who enforces it

Every allocation is checked against the request's reservation first; an engine step that would
exceed it changes nothing and asks the cache to make room. nuvec's memory limit is set to the
budget as a backstop. mrun clears nuvec's buffer cache after the state grows, after each prompt
chunk, every 256 generated tokens, and after moves and evictions.

## Tuning

| Setting | Default | Effect |
| --- | --- | --- |
| `--budget-gib` / `MRUN_BUDGET_GIB` | `min(working set, physical − 2 GiB, available − margin)` | A smaller budget; a value above the automatic budget is refused |
| `--context` / `MRUN_CONTEXT` | the most that fits | The budget this many tokens need (`64K`); the rest stays free for other apps |

There is no reserve or memory-fraction setting, and no way to exceed the recommended working set.

### Context

`mrun serve --context 64K` sets the budget to what 64K tokens need: the weights, scratch, the
linear-attention state with its checkpoints, and the attention state for 64K tokens. The server
then holds exactly 64K tokens of context. Attention state still grows as conversations do, so
the setting caps memory rather than reserving it. A context that does not fit is refused before
loading (E042), with the largest one that does; mrun never loads less than you asked for.

`mrun ls --context 64K` shows what each model needs for it, and Models in the window offers the
same choice per model with the memory it uses.

## Under pressure

mrun polls macOS's memory-pressure level once a second and acts before a conversation switch,
between prompt chunks, every 64 generated tokens and, while idle, within a second. At *warning*
it drops at least half of the cached branch memory; at *critical* it drops all branches and every
checkpoint except the one the live conversation would resume from. A running request keeps its
live state; an idle server drops that too. The model is never unloaded.
