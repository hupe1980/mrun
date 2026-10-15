+++
title = "Benchmarks"
description = "How mrun measures local inference for coding agents: replayed agent sessions, time to first token per turn, and dated comparisons with llama.cpp, Ollama, oMLX and mlx-lm."
weight = 100
+++

Published numbers for local inference servers usually measure one long prompt or raw decode speed.
Coding agents need something else: fast responses on turn 30 of a session with 50,000 tokens of
context. `mrun bench` measures that, against mrun and other servers on the same Mac.

No results are published yet. The first dated comparison is a release goal.

## Suites

| Suite | Measures |
| --- | --- |
| `agent` | Replays recorded Claude Code, Codex and OpenCode sessions turn by turn: time to first token per turn and per session |
| `synthetic` | Prefill and decode speed across context lengths |
| `cancel` | Time from disconnect until the next request starts |
| `memory` | Peak memory and swap during long sessions |
| `quality` | Output agreement with the reference implementation |

## Method

- **Teacher-forced replay.** Each turn sends exactly what the real client sent, including the
  original model replies, so every server sees identical prompts regardless of what it generates.
- **Caches on.** Every competitor runs with its own prompt cache enabled and tuned as its
  documentation recommends. Comparing against a cold server would be meaningless.
- **Same weights where possible**, and the exact model file, format and quantization recorded
  where not.
- **Repetitions** with medians and spread, plus warnings for thermal throttling, low power mode and
  early end-of-sequence.
- **Losing cells are always reported.** A report without them is not a valid result.

## Running it

```sh
mrun bench --suite agent
mrun bench --suite all --compare \
  --system mrun=http://127.0.0.1:8484 \
  --system llama.cpp=http://127.0.0.1:8080,apis=chat \
  --system ollama=http://127.0.0.1:11434,apis=chat
```

Results are written as JSON (versioned schema) and a Markdown report. `--baseline <file>` turns a
run into a regression check that exits with code 7 when thresholds fail.

## Your own traces

`mrun serve --record <dir>` records your agent sessions as replayable traces. Recordings contain
your prompts and code, stay on your machine and are never uploaded.
