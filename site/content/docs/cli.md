+++
title = "CLI reference"
description = "Every mrun command: run, serve, pull, ls, show, rm, verify, launch, connect, doctor, ps, cache, bench, completions, with exit codes."
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
| `mrun connect <client>` | Print the persistent client configuration and the file it belongs in; changes nothing |
| `mrun doctor` | Check the chip and memory budget, Low Power Mode, the port, the server and installed clients; print a fix for each failure |
| `mrun cache clear` | Clear the running server's prefix cache |
| `mrun bench` | Benchmark mrun and other local servers; see [Benchmarks](@/docs/benchmarks.md) |
| `mrun completions <bash\|zsh\|fish>` | Print a shell completion script |
| `mrun --version` | Print the version, commit and MLX versions |

Planned after the first release: a background login service, writing client configuration
(`connect --write`), a configuration file, and listening on addresses other than loopback.

## `mrun serve` options

| Flag | Default | Meaning |
| --- | --- | --- |
| `--model` | `qwen3.6` | Model to load |
| `--port` | `8484` | Port on `127.0.0.1`; mrun listens on loopback only |
| `--max-output` | 16384 | Default output limit for Chat Completions requests |
| `--queue`, `--queue-wait-s` | 16, 600 | Queue length and maximum wait |
| `--memory-fraction`, `--reserve-gib` | 0.75, ⅙ of RAM (2–8 GiB) | Memory budget (see [Memory](@/docs/memory.md)) |
| `--allow-over` | off | Allow a budget above the default formula, still capped by the recommended GPU working set |
| `--no-cache` | off | Disable the prefix cache |
| `--reject-tool-images` | off | Reject images in tool results instead of replacing them |
| `--record <dir>` | off | Record requests as a benchmark trace (stored locally, never uploaded) |

Every flag except `--record` and `--allow-over` can also be set with an environment variable
named after it (`--port` → `MRUN_PORT`, `--reserve-gib` → `MRUN_RESERVE_GIB`); see [Configuration](@/docs/configuration.md).

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
| 7 | Reserved |
| 8 | `mrun doctor` found failures |

## Error messages

Errors name the problem and the next step:

```text
error[E031]: model 'qwen3.66' not found
  hint: did you mean 'qwen3.6'? Run 'mrun ls' to see local models.
```
