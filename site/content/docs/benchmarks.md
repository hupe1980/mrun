+++
title = "Benchmarks"
description = "Measured results and how mrun bench measures local inference for coding agents: whole-session replay, paired per-turn comparisons and a truncation canary."
weight = 100
+++

`mrun bench` measures fast responses deep into agent sessions, for mrun and other servers on the
same Mac. Comparisons report every cell, including the ones mrun loses.

## Measured results

M4 Pro, 48 GB, macOS 26.

**Parity with mlx-lm 0.32.0.** Greedy output on 22 prompts up to 8K tokens (256 tokens each),
Qwen3.5-9B on nuvec 0.12.0: 9 of 22 identical or parted at an exact tie; 13 part after 41–227
tokens where mlx-lm's top two tokens are one bf16 step apart. Qwen3-8B is not measured on this
nuvec.

**Speed against mlx-lm 0.32.0** (mrun / mlx-lm, best of 2, under background load):

| Model | Prefill 1K | Prefill 16K | Decode 1K | Decode 16K |
| --- | --- | --- | --- | --- |
| Qwen3-8B 4-bit | 96.8 % | 98.2 % | 99.4 % | 101 % |
| Qwen3.5-9B 4-bit | 91.6 % | 95.7 % | 93.7 % ¹ | 108 % |

Qwen3.6-35B-A3B 4-bit on a 4,350-token prompt: prefill 982 / 938 tok/s, decode 86.9 / 87.6 tok/s.

¹ Under a load average of up to 14; 104 % on a quiet machine.

**Cancellation.** Interrupting prompt processing at random points: p95 ≤ 215 ms until the next
request can start, at every measured depth.

**Exact reuse.** A request served from the cache equals recomputing its prompt with the same chunk
plan, bit for bit, for Qwen3-8B (21 cases) and Qwen3.5-9B (10 cases). Qwen3.6-35B-A3B is not
covered.

**Speculative decoding** (`auto` against `off`, greedy, median ratio of 5 alternating pairs,
2026-11-17, under a load average of 20–70 from other work on the Mac):

| Reply | Qwen3.5-9B, DFlash | Qwen3.5-9B, prompt lookup only | Qwen3.6-35B-A3B, DFlash |
| --- | --- | --- | --- |
| Code rewritten with one rename (99 tokens) | 1.77× | 1.42× | 1.63× |
| New code: a date parser with tests (256 tokens) | 1.07× | 0.95× | 1.03× |
| A short story (256 tokens) | 0.97× | 0.96× | 0.94× |
| A short math answer (190–226 tokens) | 1.52× | 0.94× | 1.23× |

The tokens are identical with and without it, greedy and seeded. mrun's verify is row-exact, so
each verified row costs more than in a verify that may round differently: DFlash proposes 3 tokens
per round, where nuvec-lm's inexact verify reaches 2.05× with 7 on the math answer. The MoE model
gains less: a verify pass reads the experts every drafted token routes to.

**Parallel lanes** (Qwen3.6-35B-A3B 4-bit, `--lanes 2` against `--lanes 1`, 2026-11-15):

| Measure | One lane | Two lanes |
| --- | --- | --- |
| Decode at 4K depth, two requests | 80 tok/s | 120 tok/s together (1.5×) |
| First token of a 1.8K-token title request sent during a 19K-token prompt (median of 5) | 28.5 s | 2.1 s (alone: 2.2 s) |
| The 19K-token request itself | 27.8 s | 28.0 s |
| Largest gap between tokens while the other lane reads a long prompt | — | 0.12 s |

Each lane's output is identical to running its request alone. A 30-minute soak of main turns,
subagent fan-outs and titles stayed within the budget, every subagent prefix and main turn
cached.

**Small Macs (simulated).** See [Memory](@/docs/memory.md#smaller-macs).

## Quick check against Ollama

A quick check, **not** the `mrun bench` comparison: no traces, no bootstrap intervals, one model,
heavy background load.

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

mrun loses both follow-ups: 26 % slower on a short recompute at 3.6K, within 2 % at 13.7K. In the
tool loop mrun reuses about 45–60 more tokens per round than Ollama (the model's own tool call,
spliced back into the next prompt).

## Contributed results

Results from other Macs are listed here by contributor, chip and memory; see
[Hardware reports](https://github.com/hupe1980/mrun/blob/main/CONTRIBUTING.md#hardware-reports).
None are listed.

## Suites

| Suite | Measures |
| --- | --- |
| `agent` | Replays agent sessions turn by turn: time to first token per turn, session time, and from a separate natural-stop pass the quality signals (malformed tool calls, empty replies). For mrun also each turn's cache miss cause and the message where reuse stopped |
| `synthetic` | Prefill and decode speed at 1K, 8K and 32K tokens and the largest context every compared server supports, each prompt starting with a nonce so no cache can serve it |
| `cancel` | mrun only: time from a disconnect until the next request's first byte |

Free memory, swap, power source, Low Power Mode and thermal state are recorded before each server
runs.

Agent traces are not bundled. The agent suite needs `--trace <FILE|DIR>` (a `.jsonl` trace or a
directory of them, repeatable); `--suite all` without one skips it and says so in the report. See
[Your own traces](#your-own-traces).

## Method

- **Same prompts for everyone.** Recorded requests are sent in order, unchanged except for the
  output limit and streaming, so every server sees byte-identical prompts.
- **Same work per turn.** In the timing pass each turn generates as many tokens as the recorded
  reply (ignoring end-of-sequence where the server supports it); time to first token is taken at
  the first content event, not at pings or keepalives.
- **Quality from a separate pass.** Quality signals come from one natural-stop replay with the
  client's own output limit (4,096 tokens when the recording named none).
- **Whole sessions or nothing.** A session with any failed or truncated turn is excluded and
  listed with the reason.
- **Truncation canary.** Each server measures, in its own tokens, a short canary line at the start
  of the prompt; a turn whose prompt count does not grow by that size was truncated. A server that
  reports no prompt counts is marked unverified.
- **Paired comparisons.** Comparisons use per-turn ratios over turns both servers completed (the
  same turn of the same trace), summarized by their geometric mean with a bootstrap 95 % interval
  (2,000 resamples, fixed seed). The verdict is `mrun-faster` or `mrun-slower` when the interval
  excludes 1, else `within-noise`. Synthetic cells compare medians.
- **Each server in its best exact configuration.** Prompt cache enabled as documented, fastest
  lossless state; a lossy default such as 8-bit KV runs as an extra, labeled configuration. Caches
  are cleared before each repetition. Minimum repetitions: agent 3, synthetic 5, cancel 50
  (`--reps` raises them); the synthetic maximum-context cell is timed once.
- **One server at a time.** On a terminal, `mrun bench` pauses and asks you to stop one server and
  start the next.
- **Weights recorded.** Model format, bits per weight and state format are recorded for every
  server; lossy state formats are labeled.
- **Losing cells are always reported.**

## Profiles

Profiles ship inside the binary and hold each server's start command, flags, context length and
cache reset. Unverified profiles are reported but kept out of the headline summary.

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
<record.jsonl> --out <trace.jsonl>` turns a recording into a replayable trace (`--tokenizer` for
exact token counts). Recordings contain your prompts and code and stay on your machine.
