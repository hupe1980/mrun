+++
title = "Memory"
description = "How mrun budgets Apple Silicon unified memory for weights, sequence state, cache and scratch, so long agent sessions stay inside a budget fixed at startup."
weight = 50
+++

On a Mac, the CPU and GPU share one pool of memory. A model server that grows without limits
competes with your editor, browser and build tools, and when the pool runs out macOS swaps. mrun
fixes a budget at startup and keeps weights, sequence state, cache and scratch inside it.

## The budget

```text
budget  = min(MLX's recommended working set, physical − 2 GiB, available − margin)   (lowered by --budget-gib)
state   = budget − weights − scratch        (live conversation, cached branches, checkpoints)
context = (state − 3 × linear-attention state) ÷ attention bytes per token, capped by the model's maximum
```

- **The budget** is the working set macOS recommends for the GPU, read from MLX's device info.
  It is not a fixed share of RAM: on smaller Macs it is about two thirds of memory, on larger ones
  more. If you have raised the system's wired-memory limit, mrun notices, but the budget still
  stays at least 2 GiB below physical memory. It also never exceeds what is *available* at
  startup minus a margin for other apps (see [Other apps and VMs](#other-apps-and-vms)).
- **Weights** are estimated from file sizes before loading (for `pull`, `ls` and the default
  alias) and measured after warm-up.
- **Scratch** is the temporary memory a request needs beyond its state:
  `(warm-up peak above the weights + computed term) × 1.25`. The computed term is the largest
  prompt chunk's activations, expert routing and attention scores, plus one attention layer (the
  copy made while the state grows or moves), one linear-attention state (the rollback copy) and
  MLX's buffer cache, which mrun limits to 256 MiB. The terms that grow with position are sized
  at the depth the state can actually reach on this Mac, not at the model's maximum, so a small
  Mac does not reserve scratch for a context it could never hold. Before a model is loaded
  (`pull`, `ls`, the default alias) mrun uses the scratch measured for its family.
- **State** holds the live conversation, cached branches and, for hybrid models, checkpoints of
  the linear-attention state. The context is computed after setting aside the live
  linear-attention state and two checkpoints, so even a full context can still checkpoint.

Loading checks the fit three times and refuses with every number: from the file sizes before
reading anything, with the computed scratch before the warm-up reads the weights, and with the
measured numbers after it.

Example, M4 Pro with 48 GB: the recommended working set, and so the budget, is 37.4 GiB.

| | Qwen3.6-35B-A3B (4-bit) | Qwen3-8B (4-bit) |
| --- | --- | --- |
| Weights | 18.2 GiB | 4.3 GiB |
| Scratch | 5.4 GiB | 1.3 GiB |
| State | 13.8 GiB | 31.8 GiB |
| Attention state per token | 20 KiB (10 attention layers) | 144 KiB |
| Context | 262,144 tokens (the model's maximum) | 40,960 tokens (the model's maximum) |

With Qwen3.6, the full 262K window needs about 5 GiB; the rest of the state memory holds branches
and checkpoints. `mrun run --info` and `mrun ps` show every term on your Mac, including the depth
scratch was sized at; `mrun ls` shows the context each alias would get.

The default model follows from the same arithmetic: the largest alias that leaves at least 64K
tokens of context (65,536) on this Mac. There is no RAM-size rule; the measured working set
decides.

## Other apps and VMs

The working set says what the GPU may use on an otherwise idle Mac; it does not know what other
programs already hold. So mrun also reads the memory that is *available* when it starts: free,
inactive, purgeable and speculative pages, which macOS can hand out without swapping (`vm_stat`
shows them). It keeps a margin of physical memory ÷ 8, at least 2 GiB and at most 6 GiB, for the
system and the apps you keep using (the `available − margin` term of [the budget](#the-budget)).

- If the model's weights, scratch and 64K tokens of context do not fit what is available,
  `mrun serve`, `run` and `launch` refuse before reading the weights (error E041), with the
  numbers, the three largest other processes and a smaller model that fits:

  ```text
  error[E041]: qwen3.6 needs 25.1 GiB but only 14.2 GiB are available (20.2 GiB free, less a
  6.0 GiB margin for other apps); largest other processes:
  com.apple.Virtualization.VirtualMachine 32.3 GiB, rust-analyzer 4.3 GiB, Code Helper 2.2 GiB
    hint: Quit some, or use qwen3.5.
  ```

- The weights are pinned in memory (wired) only if they fit what is available; otherwise mrun
  loads them without pinning and logs one line.
- `mrun ls` fits with the same budget, so its FITS and MAX CTX columns reflect the apps running
  now, and a line under the table says how much is available. The default model does not change
  with other apps: it is picked for the Mac, and the refusal names the smaller one.
- `mrun doctor` shows available memory next to the working set, which term binds, and lists
  every other process of 4 GiB or more and every virtual machine.

A VM is the usual large consumer: its guest memory belongs to one host process
(`com.apple.Virtualization.VirtualMachine` for Apple's framework, Docker Desktop and UTM;
`qemu-system-…`) and grows toward the size configured for the guest. Stop the VM, or lower its
memory, before serving a large model. The available figure is read once at startup; mrun does not
grow its budget when you quit apps later, so restart it to use the memory you freed.

`mrun bench` and the client-gate script check the same before each server they wait for: no other
inference server may run, and the available memory must hold the next model plus the margin.

## Smaller Macs

mrun is qualified on one Mac, the M4 Pro with 48 GB. Smaller Macs are simulated there by setting
`--budget-gib` to their working set, which reproduces the fit, the default model, the 64K floor
and admission exactly:

| Mac | Recommended working set | Source | Default | `qwen3.5` context |
| --- | --- | --- | --- | --- |
| 16 GB | ≈ 10.67 GiB | llama.cpp Metal startup logs | `qwen3.5` | 84,333 tokens |
| 24 GB | 16 GiB | ⅔ of memory, as on 16 and 32 GB | `qwen3.5` | 235,385 tokens |
| 32 GB | ≈ 21.3 GiB | llama.cpp Metal startup logs | `qwen3.5` | — |
| 48 GB (M4 Pro) | 37.4 GiB (40,200,896,512 B) | measured | `qwen3.6` | 262,144 tokens |

Simulation does not show a slower GPU's cancellation latency (M1), M5 speed with Neural
Accelerators, or real memory pressure on a small Mac with other apps open. On such a Mac,
`mrun doctor --report` prints what a
[hardware report](https://github.com/hupe1980/mrun/blob/main/CONTRIBUTING.md#hardware-reports)
needs.

## Admission

Before a request touches the GPU, mrun checks that its prompt fits, evicting cached state if
needed, and clamps the output limit to the space left (reported in `x-mrun-max-tokens`). If the
prompt itself cannot fit, the request is refused immediately with the token counts, in the error
format each API defines (for example `context_length_exceeded`), and the cache is left intact.
One request runs at a time. mrun reports this Mac's context capacity in `GET /v1/models`, and `mrun launch`
passes it to the client, so agents compact their history before reaching the limit. `launch`
refuses a model whose capacity is below the client's minimum (64K tokens for Claude Code) and
names the budget it would need.

## Who enforces it

mrun's own accounting enforces the budget: every allocation is checked against the request's
reservation before it happens, and an engine step that would exceed it changes nothing and asks
the cache to make room first. MLX's own memory limit is set to the budget only as a backstop.
Freed memory goes back to the system: mrun clears MLX's buffer cache after the state grows, after
each prompt chunk (also a cancelled one), every 256 generated tokens, and after moves and
evictions.

## Tuning

| Setting | Default | Effect |
| --- | --- | --- |
| `--budget-gib` / `MRUN_BUDGET_GIB` | `min(working set, physical − 2 GiB, available − margin)` | A smaller budget, for example to leave more room for other apps. A value above the automatic budget is refused |

The budget can only be lowered. There is no reserve or memory-fraction setting, and no way to
run above the recommended working set.

## Under pressure

mrun polls macOS's memory-pressure level once a second and reacts when it rises: before a request
switches conversations, between prompt chunks, every 64 generated tokens and, while idle, within a
second. At *warning* it drops at least half of the cached branch memory; at *critical* it drops
all branches and every checkpoint except the one the live conversation would resume from. A
running request continues with its live state; an idle server drops that too. The model
stays loaded; it is never unloaded on a timer, because reloading would throw away the cache.
