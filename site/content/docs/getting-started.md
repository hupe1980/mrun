+++
title = "Getting started"
description = "Install mrun, download a coding model, start the server and launch your coding agent against it in a few minutes."
weight = 10
+++

## Requirements

- A Mac with Apple Silicon (M1 or later) and macOS 15 or newer.
- Memory: 48 GB or more for the default model, Qwen3.6-35B-A3B, with its full context; 36 GB
  runs it with about 50K tokens of context. Smaller Macs use Qwen3-8B. `mrun pull` tells you how
  much context a model gets on your Mac before downloading (see [Memory](@/docs/memory.md)).
- About 20 GB of free disk space for the default model.

## Install

Until the first release, build from source with the Rust toolchain pinned in the repository:

```sh
git clone https://github.com/hupe1980/mrun
cd mrun
cargo install --path crates/mrun
```

## Download a model

```sh
mrun pull qwen3.6
```

`pull` shows the model's size, license and whether it fits this Mac (and with how much context)
before downloading. Downloads resume after interruption and are verified against content hashes.
Models live in the shared Hugging Face cache, so a model already downloaded by another MLX tool is
reused instead of downloaded again. See [Models](@/docs/models.md).

## Try it in the terminal

```sh
mrun run qwen3.6
```

`run` opens an interactive chat. Add `--info` to see the memory budget, or `--check` to verify the
model loads and generates without starting a chat.

## Start the server

```sh
mrun serve --detach
```

The server listens on `http://127.0.0.1:8484`, loopback only. `mrun ps` shows the loaded model,
memory, active and queued requests, cache statistics and the last cache miss. `mrun stop` shuts a
detached server down.

## Launch your agent

```sh
mrun launch claude-code    # or: codex, opencode
```

`launch` starts the client already pointed at mrun, for that process only; your global client
settings stay untouched. It also starts the server and offers to download the model if needed, so
on a fresh install this one command is enough. To configure a client permanently, use `mrun connect` (see
[Clients](@/docs/clients.md)).

## When something is off

```sh
mrun doctor
```

`doctor` checks the machine, the model, the server and each installed client, and prints a fix for
every failure.
