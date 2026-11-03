+++
title = "Configuration"
description = "Configure mrun with an optional config file, MRUN_* environment variables and command-line flags, and where it keeps its files."
weight = 90
+++

mrun needs no configuration. When you want to change a default, settings are read in this order,
later sources winning:

1. built-in defaults,
2. the config file: its top-level keys, then the table of the command that runs,
3. `MRUN_*` environment variables,
4. command-line flags.

`mrun config show` prints every setting with the value a command would use and where it comes
from. mrun warns about unknown `MRUN_*` variables and unknown config keys and suggests the closest
known name, so a typo is not ignored silently.

## Config file

The file is optional: without it nothing changes. mrun reads `~/.mrun/config.toml`
(`$MRUN_HOME/config.toml`), or the file named by `MRUN_CONFIG`; `MRUN_CONFIG=` (empty) ignores the
file for one run. mrun never writes it.

```toml
# Shared keys apply to every command with the flag.
port = 8484              # every command with --port
model = "qwen3.6"        # serve, launch

[serve]
queue = 4
no-cache = false

[launch]
model = "qwen3.5"        # overrides the shared model for launch only

[run]
temperature = 0.2

[cloud]                  # mrun cloud up
idle-stop = "30m"
max-runtime = "8h"
retry = "1h"
any-cloud = true
max-price = 2.6

[bench]
reps = 5
```

- **Tables** are the commands with settings: `serve`, `launch`, `run`, `cloud` (`cloud up`) and
  `bench`. **Keys** are the flags' long names without the dashes (`--idle-stop` → `idle-stop`).
- **Shared keys** `port`, `model` and `budget-gib` sit at the top level and apply to every command
  with that flag, as `MRUN_PORT`, `MRUN_MODEL` and `MRUN_BUDGET_GIB` do. A command's table may set
  the same key for that command only.
- **Types** follow the flag: strings for durations, names and paths (`idle-stop = "30m"`, not
  `30`), numbers and booleans unquoted. A list (`["a", "b"]`) only for flags that repeat
  (`bench.system`, `run.stop`).
- **Booleans** can be switched off on the command line: every boolean preference flag takes
  `--flag[=<BOOL>]`, so `--any-cloud=false` overrides `any-cloud = true` in the file, and plain
  `--any-cloud` stays `true`.
- **Dependent settings** count as given: `retry` in the file lets `--max-price` alone on the command
  line work, and `[cloud] model` makes `cloud up`'s `--model` optional.
- **`serve --detach`** passes the values it resolved to the background server, which does not read
  the file again.

Not settable, by design: confirmations and one-shot actions (`yes`, `dry-run`, `write`, `restore`,
`record`, `trace`, `out`, `follow`, `info`), output formats (`json`, `quiet`, `verbose`), remote mode
(`serve --relay`: only `mrun cloud up` starts it), secrets (the RunPod key comes from
`RUNPOD_API_KEY` or `.env`) and addresses (mrun binds to 127.0.0.1 only). Such keys get a warning
like any unknown key.

### Mistakes

An unknown section or key is a warning with the file's line and the closest name; the command
runs:

```text
warning: ~/.mrun/config.toml:11: unknown section [clould]; did you mean [cloud]?
warning: ~/.mrun/config.toml:3: unknown key serve.idle-stop; it belongs to [cloud]
```

An invalid value or a TOML syntax error stops every command before it does anything, except
`mrun config` and `mrun doctor`, which report it (doctor as a failed `config` check).

```text
error[E003]: ~/.mrun/config.toml:2: cloud.idle-stop = 30: expected a string such as "30m"
  hint: fix the config file, or run with MRUN_CONFIG= to ignore it
```

### `mrun config`

| Command | What it does |
| --- | --- |
| `mrun config path` | The file mrun reads, and whether it exists |
| `mrun config show [--json]` | Every setting with its value and source (`default`, `file:<line>`, `env MRUN_*`), as valid TOML |
| `mrun config schema` | A JSON Schema (draft 2020-12) of the file, for editors |

```text
$ MRUN_PORT=9100 mrun config show
# config file: /Users/me/.mrun/config.toml (exists)
port = 9100                 # env MRUN_PORT (file:1 had 8484)
model = "qwen3.6"           # file:2

[serve]
queue = 4                   # file:5
queue-wait-s = 240          # default
...

[cloud]
idle-stop = "30m"           # file:15
max-runtime = "8h"          # file:16
retry = "1h"                # file:17
max-price = 2.6             # file:19 (needs retry)
any-cloud = true            # file:18 (needs retry)
```

For completion and checks while you type (Taplo, Even Better TOML in VS Code), save the schema
and point the file at it:

```sh
mrun config schema > ~/.mrun/config.schema.json
```

```toml
#:schema ./config.schema.json
```

## Environment variables

| Variable | Flag | Default |
| --- | --- | --- |
| `MRUN_HOME` | — | `~/.mrun` |
| `MRUN_CONFIG` | — | `$MRUN_HOME/config.toml`; empty: no config file |
| `MRUN_PORT` | `--port` (`serve`, `launch`, `connect`, `ls`, `ps`, `doctor`, `stop`, `cache clear`, `cloud up`, `cloud down`, `cloud status`) | 8484 |
| `MRUN_MODEL` | `--model` (`serve`, `launch`) | This machine's default alias, or the largest pulled model that fits |
| `MRUN_MAX_OUTPUT` | `serve --max-output` | 32768 (all three APIs, clamped to the context left); at least 1 |
| `MRUN_QUEUE` | `serve --queue` | 16; at least 1 |
| `MRUN_QUEUE_WAIT_S` | `serve --queue-wait-s` | 240 (non-streaming requests only); at least 1 |
| `MRUN_BUDGET_GIB` | `--budget-gib` (`serve`, `run`, `ls`) | MLX's recommended working set; can only be lowered |
| `MRUN_NO_CACHE` | `serve --no-cache` | off |
| `MRUN_REJECT_TOOL_IMAGES` | `serve --reject-tool-images` | off |
| `MRUN_DETACH` | `serve --detach` | off |
| `MRUN_UI_PORT` | `ui --port` | 8485 (the window's own loopback port; the next free one when taken) |
| `MRUN_CALIBRATION_DEBUG` | — | unset; any value prints the per-layer timings of chunk calibration to stderr |

`--record`, `--yes`, `--dry-run`, `--write` and `--restore` exist only as flags (and are not
config keys), so they are never left on by accident. Boolean variables take `true`/`false`,
`1`/`0` or `yes`/`no`.

mrun also reads the Hugging Face variables `HF_HOME`, `HF_HUB_CACHE`, `HF_ENDPOINT` (a mirror) and
`HF_TOKEN` (gated models; otherwise the token saved by `hf auth login`), and Claude Code's
`CLAUDE_CONFIG_DIR` (for `mrun doctor`).

## Paths

| Path | Contents | Override |
| --- | --- | --- |
| `~/.mrun/config.toml` | The optional config file (read only; mrun never writes it) | `MRUN_CONFIG`, `MRUN_HOME` |
| `~/.mrun/manifests` | One manifest per pulled model: pinned revision and file hashes | `MRUN_HOME` |
| `~/.mrun/server.log` | Log of background servers (all ports), moved to `server.log.1` at 10 MB | `MRUN_HOME` |
| `~/.mrun/server-<port>.pid` | Pid of the background server on that port | `MRUN_HOME` |
| `~/.mrun/claude/settings.json` | Claude Code settings file suggested or written by `connect` | `MRUN_HOME` |
| `~/.mrun/backups/<client>/` | Backups made by `connect --write`, used by `connect --restore` | `MRUN_HOME` |
| `~/.mrun/calibration.toml` | Prefill chunk calibration per device and model, reused on later starts | `MRUN_HOME` |
| `~/.mrun/logs/cloud-<pod id>.log` | Full pod log saved by a failed `cloud up` | `MRUN_HOME` |
| `~/.codex/mrun.config.toml` | The Codex profile written by `connect --write` (`launch` passes the same settings as `-c` overrides) | `CODEX_HOME` |
| `~/.cache/huggingface/hub` | Model files, shared with other tools | `HF_HOME`, `HF_HUB_CACHE` |
| `~/Library/Caches/mrun/` | MLX's Metal library, extracted from the binary on first use | — |

When `HOME` is unset, mrun uses the account's home directory, and `./.mrun` if that is unknown
too.

## Build variables

These apply when building from source; see `crates/mrun-mlx-sys/README.md` for the full list.

| Variable | Meaning |
| --- | --- |
| `MLX_SYS_CACHE_DIR` | Where the MLX build is cached (default `~/Library/Caches/mrun-mlx-sys` on macOS) |
| `MLX_SYS_METAL` | `0` builds MLX for the CPU only (tests, machines without Xcode; too slow for real models) |
| `MLX_SYS_SOURCE_DIR` | Directory with the pinned MLX and mlx-c source archives, for offline builds |
| `MLX_SYS_PREBUILT_DIR` | Install prefix of an earlier MLX build; skips fetching and CMake |
