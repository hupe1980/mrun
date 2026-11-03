+++
title = "Benchmarks"
description = "How mrun bench measures local inference for coding agents: whole-session replay, paired per-turn comparisons with bootstrap intervals, a truncation canary and dated, complete reports."
weight = 100
+++

`mrun bench` measures what coding agents need: fast responses on turn 30 of a session with
50,000 tokens of context, for mrun and other servers on the same Mac. Comparisons report every
cell, including the ones mrun loses.

## Measured on the M4 Pro (2026-10-16)

All numbers below are from one Mac (M4 Pro, 48 GB, macOS 26).

**Parity with mlx-lm 0.32.0.** Greedy output on 22 prompts up to 8K tokens (256 tokens each):
Qwen3-8B 22 of 22 and Qwen3.5-9B 22 of 22 token-identical.

**Speed against mlx-lm 0.32.0** (mrun / mlx-lm, best of 2, under background load from other
processes):

| Model | Prefill 1K | Prefill 16K | Decode 1K | Decode 16K |
| --- | --- | --- | --- | --- |
| Qwen3-8B 4-bit | 96.8 % | 98.2 % | 99.4 % | 101 % |
| Qwen3.5-9B 4-bit | 91.6 % | 95.7 % | 93.7 % ¹ | 108 % |

¹ Measured under a load average of up to 14; a quiet run gave 104 %.

**Cancellation.** Interrupting prompt processing at random points: p95 ≤ 215 ms until the next
request can start, at every measured depth (gate: 300 ms).

**Exact reuse.** The exactness oracle (a request served from the cache equals recomputing its
prompt with the same chunk plan, bit for bit) passes on Metal for Qwen3-8B (21 cases) and
Qwen3.5-9B (10 cases). Qwen3.6-35B-A3B is not covered.

**Small Macs (simulated).** With `--budget-gib` set to a smaller Mac's working set
([Memory](@/docs/memory.md#smaller-macs)), `qwen3.5` fits 84,333 tokens of context on a 16 GB
Mac and 235,385 on a 24 GB Mac. Simulation reproduces fit and admission, not a slower GPU.

## Quick check against Ollama (2026-10-16)

A quick check, **not** the `mrun bench` comparison: no traces, no bootstrap intervals, one
model, a machine under heavy background load from other processes.

Setup: M4 Pro 48 GB, Qwen3-8B. mrun runs MLX 4-bit weights; Ollama 0.40.4 runs this model on its
llama.cpp engine with Q4_K_M weights (not MLX), so weights and kernels differ. Chat Completions,
greedy, thinking off, 5 repetitions, medians.

| Cell | mrun | Ollama |
| --- | --- | --- |
| Cold time to first token, 3.3K-token prompt | **9.06 s** | 9.46 s |
| Cold time to first token, 13.5K-token prompt | **41.9 s** | 49.7 s |
| Decode at 3.3K | **45.6 tok/s** | 36.1 tok/s |
| Decode at 13.5K | **34.2 tok/s** | 26.4 tok/s |
| Follow-up after a new user message, 3.6K (both recompute ~280 tokens) | 1.23 s | **0.98 s** |
| Follow-up after a new user message, 13.7K | 1.58 s | **1.56 s** |
| 4-round tool loop, time to first token per round | **1.94 / 2.53 / 3.26 / 2.86 s** | 2.42 / 3.16 / 4.05 / 3.92 s |

Losing cells: both follow-ups. At 3.6K mrun is 26 % slower on a short recompute (prefill of a
chunk of a few hundred tokens); at 13.7K the two are within 2 %. In the tool loop mrun reuses
about 45–60 more tokens per round than Ollama: the model's own tool call, spliced back into the
next prompt.

## Contributed results

Results from other Macs (M1, M2, M3, M5, 16–24 GB) are listed here by contributor, chip, memory
and date; see [Hardware reports](https://github.com/hupe1980/mrun/blob/main/CONTRIBUTING.md#hardware-reports).
No contributed results are listed.

## Suites

| Suite | Measures |
| --- | --- |
| `agent` | Replays agent sessions turn by turn: time to first token per turn, session time, and from a separate natural-stop pass the quality signals (malformed tool calls, empty replies). For mrun also each turn's cache miss cause and the message where reuse stopped |
| `synthetic` | Prefill and decode speed at 1K, 8K and 32K tokens and the largest context every compared server supports, each prompt starting with a nonce so no cache can serve it |
| `cancel` | mrun only: time from a disconnect until the next request's first byte |

Free memory, swap, power source, Low Power Mode and thermal state are recorded before each server
runs.

**Agent traces are not bundled.** The agent suite needs `--trace <FILE|DIR>` (a `.jsonl`
trace or a directory of them, repeatable); `--suite all` without one skips the agent suite and
says so in the report. Traces are built from your own sessions recorded with
`mrun serve --record <dir>` (see below).

## Method

- **Same prompts for everyone.** Recorded requests are sent in order, unchanged except for the
  output limit and streaming. The history is the recorded conversation, so every server sees
  byte-identical prompts and can reuse context up to the start of the previous reply.
- **Same work per turn.** In the timing pass each turn generates as many tokens as the recorded
  reply (ignoring end-of-sequence where the server supports it); time to first token is taken at
  the first content event, not at pings or keepalives.
- **Quality from a separate pass.** Forced-length decoding makes a model run past where it would
  stop, so quality signals come from one extra natural-stop replay with the client's own output
  limit (4,096 tokens when the recording named none).
- **Whole sessions or nothing.** A session with any failed or truncated turn is excluded as a
  whole and listed with the reason; skipping a turn would make a session look faster.
- **Truncation canary.** A server that silently truncates a long prompt answers faster. Each
  server first measures, in its own tokens, the size of a short canary line placed at the start of
  the prompt; a turn whose prompt count does not grow by that size when the canary is present was
  truncated. A server that reports no prompt counts is marked unverified.
- **Paired comparisons.** Comparisons use per-turn ratios over turns both servers completed (the
  same turn of the same trace), summarized by their geometric mean with a bootstrap 95 % interval
  (2,000 resamples, fixed seed). The verdict is `mrun-faster` or `mrun-slower` when the interval
  excludes 1, else `within-noise`. Synthetic cells compare medians.
- **Each server in its best exact configuration.** Every server runs with its prompt cache
  enabled as its documentation recommends, in its fastest configuration that keeps state
  lossless; a lossy default such as 8-bit KV runs as an extra, labeled configuration. Caches are
  cleared before each repetition. Minimum repetitions: agent 3, synthetic 5, cancel 50 (`--reps`
  raises them).
- **One server at a time.** On a terminal, `mrun bench` pauses and asks you to stop one server and
  start the next, so none competes for memory.
- **Weights recorded.** Model format, bits per weight and state format are recorded for every
  server; lossy state formats are labeled.
- **Losing cells are always reported.**

## Profiles

Profiles ship inside the binary and hold each server's start command, flags, context length and
how to reset its cache. The profiles for llama.cpp, Ollama, oMLX and mlx-lm are verified against
a pinned release; unverified profiles are reported but kept out of the headline summary.

| Profile | Default URL | APIs | Note |
| --- | --- | --- | --- |
| `mrun` | `http://127.0.0.1:8484` | Chat, Messages, Responses | — |
| `llama.cpp` | `http://127.0.0.1:9931` | Chat | Prompt-cache reuse on, one slot, GGUF weights |
| `ollama` | `http://127.0.0.1:11434` | Chat, Messages, Responses | — |
| `mlx-lm` | `http://127.0.0.1:8080` | Chat | Synthetic suite only |
| `omlx` | `http://127.0.0.1:8000` | Chat, Messages | — |
| `rapid-mlx` | `http://127.0.0.1:8000` | Chat, Messages | **Unverified**; agent suite over Messages |
| `splash` | `http://127.0.0.1:8000` | Chat | **Unverified**. Runs once with its default 8-bit KV (labeled lossy) and once with bf16 KV |

## Running it

```sh
mrun bench --suite synthetic
mrun bench --suite agent --trace ./traces/
mrun bench --compare --trace ./traces/ \
  --system mrun=http://127.0.0.1:8484 \
  --system llama.cpp=http://127.0.0.1:9931,model=qwen3.6
```

| Flag | Meaning |
| --- | --- |
| `--suite <SUITE>` | `agent`, `synthetic`, `cancel` or `all` (default) |
| `--trace <FILE\|DIR>` | Agent trace; required for the agent suite (repeatable) |
| `--system <SPEC>` | `NAME=URL[,apis=chat+messages+responses][,model=M][,profile=P]` (default: mrun at `http://127.0.0.1:8484`) |
| `--compare` | Benchmark several systems one at a time (without `--system`: every bundled profile) |
| `--reps <N>` | Repetitions; raises the per-suite minimums |
| `--out <DIR>` | Output directory (default `./mrun-bench/<timestamp>`) |

Results are written as `result.json` (versioned schema) and `report.md`.

## Your own traces

`mrun serve --record <dir>` records requests and responses as JSON lines; `mrun bench trace
<record.jsonl> --out <trace.jsonl>` turns a recording into a replayable trace (pass
`--tokenizer` for exact token counts). Recordings contain your prompts and code, stay on your
machine and are never uploaded.
