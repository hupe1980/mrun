+++
title = "Getting started"
description = "Build mrun, download a coding model, start the server and launch your coding agent."
weight = 10
+++

## Requirements

- A Mac with Apple Silicon (M1 or later) and macOS 15 or newer.
- 16 GB of memory or more. Each Mac gets the largest alias that leaves at least 64K tokens of
  context: `qwen3.6` (Qwen3.6-35B-A3B, ≈ 20 GB on disk) where it fits, else `qwen3.5`
  (Qwen3.5-9B, ≈ 6 GB), a **preview** that gets stuck more often on harder tasks. `mrun ls` shows
  what each alias gets on your Mac. See [Models](@/docs/models.md#aliases) and
  [Memory](@/docs/memory.md).
- Tested on one Mac (M4 Pro, 48 GB, macOS 26); smaller Macs are simulated with `--budget-gib`. A
  [hardware report](https://github.com/hupe1980/mrun/blob/main/CONTRIBUTING.md#hardware-reports)
  from another Mac helps.
## Install

```sh
brew install hupe1980/tap/mrun
```

### From source

Needs Xcode with the Metal toolchain and Rust. On Xcode 26 the Metal toolchain is a separate
download:

```sh
sudo xcode-select -s /Applications/Xcode.app
xcodebuild -downloadComponent MetalToolchain
cargo install mrun                  # or from a clone: cargo install --path crates/mrun
```

A clean release build takes about 72 s on an M4 Pro. The Metal kernels are compiled into the
`mrun` binary, which needs no other files. Without the Metal toolchain they are compiled at first
start instead, and `mrun doctor` warns and names the command above.

The M5 Neural Accelerator (NAX) kernels are compiled in but stay off until measured on an M5;
`NUVEC_NAX=on` forces them.

## Download a model

```sh
mrun ls               # the aliases, the context each leaves on this Mac and the default
mrun pull qwen3.6     # or qwen3.5, where qwen3.6 leaves less than 64K tokens
```

`pull` shows the model's size, license and whether it fits this Mac, and asks before downloading
a model that does not fit (`-y` skips the question; `--json` prints JSON lines). Downloads resume
after an interruption and are verified against pinned hashes. A model already in the Hugging Face
cache is imported from there. Models are stored in `~/.mrun/models`.

## Try it in the terminal

```sh
mrun run
```

`run` uses the running server's model, else this Mac's default, and opens an interactive chat
(`/clear` starts over, `/info` shows the memory budget, `/exit` quits). `--prompt` answers once
and exits; `--info` prints the memory budget after loading and exits.

## Start the server

```sh
mrun serve --detach
```

The server listens on `http://127.0.0.1:8484`, loopback only. `--detach` returns once the model is
loaded, or reports why loading failed.

| Command | What it does |
| --- | --- |
| `mrun ps` | Model, memory budget, active and queued requests, cache statistics and the last cache miss |
| `mrun logs` | The background server's log |
| `mrun stop` | Stops the server, also while it is generating; with nothing running it says so and exits 0 |
| `mrun ui` | A window with the server's state, a chat and agent setup ([UI](@/docs/ui.md)) |
| `mrun` | Whether a server runs, which models are pulled and the next commands to try |

## Launch your agent

```sh
mrun launch claude-code    # or: codex, opencode
```

`launch` starts the client pointed at mrun, for that process only; your client settings stay
untouched. Without a server it offers, in one confirmation, to pull this Mac's default model and
start the server; a server that is still loading is waited for. It refuses a model that leaves
less context than the client needs (64K tokens for Claude Code) and names the budget that model
would need.

To configure a client permanently, `mrun connect <client>` prints the settings; `--write` writes
them with a backup and `--restore` undoes that. See [Clients](@/docs/clients.md).

## When something is off

```sh
mrun doctor
```

`doctor` checks the platform, GPU kernels, NAX, the memory budget, Low Power Mode, the port, the
server, the pulled models and this Mac's default, free disk space, mrun's state directory and each
installed client, including `env` entries in `~/.claude/settings.json` that would override what
`launch` sets. It prints a fix for every problem and exits with code 7 when a check fails.

`mrun doctor --report` prints the same findings with the chip, memory, macOS version, numerics
class, pulled models and calibration as a Markdown block for an issue. It shows your home
directory as `~` and contains no prompts.
