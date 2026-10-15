+++
title = "Configuration"
description = "Configure mrun with ~/.mrun/config.toml, MRUN_* environment variables and command-line flags."
weight = 90
+++

mrun needs no configuration. When you want to change a default, settings are read in this order,
later sources winning:

1. built-in defaults,
2. `~/.mrun/config.toml`,
3. `MRUN_*` environment variables (for example `MRUN_SERVER_PORT=9000`),
4. command-line flags.

`mrun config` prints the effective value of every setting and its source.

## Config file

Every key is optional. The defaults:

```toml
[server]
host = "127.0.0.1"
port = 8484
queue = 16
queue_wait_s = 600
thinking = "client"        # client | on | off
max_output = 16384
reject_tool_images = false

[memory]
fraction = 0.75
# reserve_gib: one sixth of physical memory, between 2 and 8 GiB

[engine]
default_model = "qwen3.6"

[cache]
enabled = true

[limits]
body_mib = 64
messages = 4096
tools = 256
schema_kib = 64
schema_depth = 16
```

## Flag-only settings

`--record`, `--debug-cache` and `--allow-over` exist only as flags, so they are never left on by
accident. The API key is read only from `MRUN_API_KEY` or `--api-key-file`, never from the config
file.

## Paths

| Path | Contents | Override |
| --- | --- | --- |
| `~/.mrun` | Configuration, manifests, aliases, measured scratch sizes | `MRUN_HOME` |
| `~/.cache/huggingface/hub` | Model files, shared with other tools | `HF_HOME`, `HF_HUB_CACHE` |
