+++
title = "CLI reference"
description = "Every mrun command: run, serve, pull, ls, show, rm, verify, launch, connect, doctor, ps, cache, config, bench, with exit codes."
weight = 80
+++

Every command accepts `--help` (with examples) and `-q/--quiet`. Commands that print tables accept
`--json`.

## Commands

| Command | What it does |
| --- | --- |
| `mrun run <model>` | Interactive chat in the terminal. `--think`, sampling flags, `--info` (memory budget), `--check` (compare greedy output with stored reference outputs; non-zero exit on the first mismatch) |
| `mrun serve` | Start the server on `127.0.0.1:8484`. `--detach` runs it in the background, waits until the model is loaded and reports load errors; logs go to `~/.mrun/server.log` |
| `mrun stop` | Stop a detached server |
| `mrun ps` | Loaded model, memory budget, active and queued requests, cache statistics, last miss |
| `mrun pull <alias\|org/name[@rev]> [--yes]` | Show size, license and fit; download, resume and verify |
| `mrun ls` | Local models with size, fit and maximum context |
| `mrun show <model>` | Manifest, architecture, quantization, license, fit breakdown |
| `mrun rm <alias> [--yes]` | Remove a model's pinned revision, warning when other tools use it |
| `mrun verify <model>` | Re-hash every file against the manifest |
| `mrun launch <client> [--model M] [--dry-run] [--yes] [-- args]` | Start Claude Code, Codex or OpenCode configured for mrun; starts the server and offers a download when needed. `--dry-run` prints the settings |
| `mrun connect <client>` | Print persistent client configuration; `--write` applies it, `--undo` restores the backup |
| `mrun doctor` | Check machine, models, server and clients; print a fix for each failure. `--agent <client>` runs a tool-call round trip and a request at 95% of capacity |
| `mrun cache clear` | Clear the running server's prefix cache |
| `mrun config` | Show the effective configuration and where each value came from |
| `mrun bench` | Benchmark mrun and other local servers; see [Benchmarks](@/docs/benchmarks.md) |
| `mrun completions <bash\|zsh\|fish>` | Print a shell completion script |

## `mrun serve` options

| Flag | Default | Meaning |
| --- | --- | --- |
| `--model` | `qwen3.6` | Model to load |
| `--host`, `--port` | `127.0.0.1`, `8484` | Listen address; a non-loopback host requires an API key |
| `--api-key-file` | — | File containing the API key (or set `MRUN_API_KEY`) |
| `--thinking` | `client` | `client`, `on` or `off` |
| `--max-output` | 16384 | Default output limit |
| `--queue`, `--queue-wait-s` | 16, 600 | Queue length and maximum wait |
| `--memory-fraction`, `--reserve-gib` | 0.75, ⅙ of RAM (2–8 GiB) | Memory budget (see [Memory](@/docs/memory.md)) |
| `--allow-over` | off | Start even if the model exceeds the budget |
| `--no-cache` | off | Disable the prefix cache |
| `--debug-cache` | off | Print divergence excerpts on the terminal |
| `--reject-tool-images` | off | Reject images in tool results instead of replacing them |
| `--record <dir>` | off | Record requests as a benchmark trace (stored locally, never uploaded) |

## Exit codes

| Code | Meaning |
| --- | --- |
| 0 | Success |
| 1 | Unexpected error |
| 2 | Usage error |
| 3 | Model not found, unsupported or corrupt |
| 4 | Model or request does not fit the memory budget |
| 5 | Server unreachable |
| 6 | Network or download failure |
| 7 | Benchmark regression threshold exceeded |
| 8 | `mrun doctor` found failures |

## Error messages

Errors name the problem and the next step:

```text
error[E031]: model 'qwen3.66' not found
  hint: did you mean 'qwen3.6'? Run 'mrun ls' to see local models.
```
