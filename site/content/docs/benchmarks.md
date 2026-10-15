+++
title = "Benchmarks"
description = "How mrun measures local inference for coding agents: replayed agent sessions, time to first token per turn, and fair, dated comparisons with llama.cpp, Ollama, oMLX, mlx-lm and Inco Splash."
weight = 100
+++

Published numbers for local inference servers usually measure one long prompt or raw decode speed.
Coding agents need something else: fast responses on turn 30 of a session with 50,000 tokens of
context. `mrun bench` measures that, against mrun and other servers on the same Mac.

No results are published yet. The first dated comparison is a release goal.

## Suites

| Suite | Measures |
| --- | --- |
| `agent` | Replays bundled OpenCode and Codex sessions turn by turn (your own Claude Code sessions can be recorded and replayed locally): time to first token per turn, session time, malformed tool calls, and for mrun the miss causes, including how often the chat template re-rendered history |
| `synthetic` | Prefill and decode speed at 1K, 8K, 32K and the largest context every compared server supports |
| `cancel` | mrun only: time from disconnect until the next request starts |

Free memory, swap, power and thermal state are recorded for every server.

## Method

- **In-order replay.** Requests are sent exactly as the client recorded them. Each server caches
  its own replies, so every server reuses context up to the previous reply in the same way.
- **Same work per turn.** Each turn generates as many tokens as the recorded reply, so session times
  are comparable; time to first token is measured at the first content event, ignoring pings and
  keepalives.
- **Caches on, reset between repetitions.** Every server runs with its prompt cache enabled and
  tuned as its documentation recommends; caches are cleared before each repetition so repetitions
  are independent.
- **Equal context.** Every server gets the same context length. A turn whose reported prompt size
  differs from the recording by more than 2% fails as truncated input, so silent truncation can't
  look fast.
- **One server at a time.** `mrun bench` pauses and asks you to stop one server and start the next,
  so none competes for memory.
- **Weights recorded.** Model file, format, bits per weight and attention-state format are recorded
  for every server; different quantizations and lossy state formats are labeled as such.
- **Competitors.** llama.cpp, Ollama, oMLX, mlx-lm and Inco Splash, each at its latest stable
  release with its prompt cache on. Splash keeps its attention state in 8 bits by default; it runs
  once with full-precision (bf16) state, the setting comparable to mrun, and once with its default,
  labeled as lossy.
- **Losing cells are always reported.** A report without them is not a valid result.

## Running it

```sh
mrun bench --suite agent
mrun bench --suite all --compare \
  --system mrun=http://127.0.0.1:8484,profile=mrun \
  --system llama.cpp=http://127.0.0.1:8080,apis=chat,profile=llama.cpp \
  --system ollama=http://127.0.0.1:11434,apis=chat,profile=ollama
```

Profiles contain each server's flags, context length and how to reset its cache; `mrun bench`
pauses and asks you to switch servers. Results are written as JSON (versioned schema) and a
Markdown report.

## Your own traces

`mrun serve --record <dir>` records your agent sessions as replayable traces. Recordings contain
your prompts and code, stay on your machine and are never uploaded.
