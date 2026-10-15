+++
title = "Configuration"
description = "Configure mrun with MRUN_* environment variables and command-line flags."
weight = 90
+++

mrun needs no configuration. When you want to change a default, settings are read in this order,
later sources winning:

1. built-in defaults,
2. `MRUN_*` environment variables,
3. command-line flags.

The first release has no configuration file.

## Environment variables

| Variable | Flag | Default |
| --- | --- | --- |
| `MRUN_SERVER_PORT` | `--port` | 8484 |
| `MRUN_SERVER_QUEUE` | `--queue` | 16 |
| `MRUN_SERVER_QUEUE_WAIT_S` | `--queue-wait-s` | 600 |
| `MRUN_SERVER_MAX_OUTPUT` | `--max-output` | 16384 |
| `MRUN_SERVER_REJECT_TOOL_IMAGES` | `--reject-tool-images` | false |
| `MRUN_MEMORY_FRACTION` | `--memory-fraction` | 0.75 |
| `MRUN_MEMORY_RESERVE_GIB` | `--reserve-gib` | one sixth of physical memory, between 2 and 8 GiB |
| `MRUN_ENGINE_DEFAULT_MODEL` | `--model` | `qwen3.6` |
| `MRUN_CACHE_ENABLED` | `--no-cache` | true |
| `MRUN_LIMITS_BODY_MIB`, `_MESSAGES`, `_TOOLS`, `_SCHEMA_KIB`, `_SCHEMA_DEPTH` | — | 64, 4096, 256, 64, 16 |

## Flag-only settings

`--record` and `--allow-over` exist only as flags, so they are never left on by accident.

## Paths

| Path | Contents | Override |
| --- | --- | --- |
| `~/.mrun` | Manifests, aliases, measured scratch sizes, server log | `MRUN_HOME` |
| `~/.cache/huggingface/hub` | Model files, shared with other tools | `HF_HOME`, `HF_HUB_CACHE` |
