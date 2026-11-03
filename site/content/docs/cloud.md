+++
title = "Remote GPU"
description = "Opt-in: run a model too big for your Mac on a rented RunPod GPU, behind the same local APIs, launch and connect, with cost guards that stop a forgotten pod."
weight = 65
+++

`mrun cloud` rents a GPU on [RunPod](https://www.runpod.io), serves the model there with
[vLLM](https://docs.vllm.ai), and keeps everything else local: your agent still talks to
`http://127.0.0.1:8484`, and `mrun launch`, `connect` and `ps` work as with a local model. Use it
for models that do not fit a Mac (Qwen3-Coder-Next needs about 80 GB at FP8) or run slowly on one.

> **Privacy and cost.** Remote mode is opt-in and never the default. Your prompts and code are
> sent to a machine RunPod rents to you (Secure Cloud by default; `--community` uses hosts run by
> third parties). You pay by the hour while the pod runs. mrun shows the price and asks before
> creating anything, the pod stops itself when idle, and `mrun cloud down` terminates it.

## Quick start

```sh
export RUNPOD_API_KEY=...            # or put RUNPOD_API_KEY=... in ./.env
mrun cloud models                    # curated models, the GPU each needs, prices
mrun cloud up --model Qwen/Qwen3.6-35B-A3B-FP8
mrun launch claude-code              # or codex, opencode: as with a local model
mrun cloud down                      # terminate the pod
```

`cloud up` prints the GPU, its price per hour, the idle and runtime limits and the privacy note,
and asks once (`--yes` skips the question). It then creates one pod, shows its progress (pod
scheduled, agent up, weights on disk, ready) and starts a local server that relays to it.
`mrun launch` in another terminal waits for the relay, then starts the client.

## Choosing a GPU

```sh
mrun cloud gpus                      # every GPU type with VRAM and Secure/Community $/h
mrun cloud up --model Qwen/Qwen3-Coder-Next-FP8 --gpu "H100 NVL"
```

Without `--gpu`, mrun picks the cheapest GPU that fits. A model fits when its weights, 64K tokens
of KV cache and about 2 GiB of activations fit in 90 % of the GPU's memory. `cloud up` refuses a
GPU that is too small and names the smallest one that fits. If no GPU of that type is available,
it shows RunPod's message and lists the fitting GPUs in stock in either cloud, cheapest first,
with the flags to rent one (for example `--gpu "RTX PRO 6000 WK" --community`). Stock changes by
the minute.

Example: Qwen3-Coder-Next FP8 on a 96 GB RTX PRO 6000 ($1.69/h Community, $2.59/h Secure) is
ready 9 minutes after `cloud up` without a volume (mostly the 80 GB download), decodes about 190
tokens/s and reaches the first token of a 15K-token prompt in 0.8 s. OpenCode, Claude Code and
Codex pass mrun's client gate against it.

Models outside the catalog work too. Name a GPU with `--gpu`, and set `--max-model-len`,
`--tool-parser` and `--reasoning-parser` as vLLM's documentation says for that model.

## Cost guards

- **Idle stop.** The agent in the pod stops its own pod after 30 minutes without a request
  (`--idle-stop 10m`). This needs nothing from your Mac: closing the laptop is safe.
- **Maximum runtime.** After 8 hours (`--max-runtime 2h`) the pod stops regardless; a running
  response gets up to 60 s to finish.
- **`mrun ps`** shows the pod, the GPU, $/h, the time up, an estimate of the spend so far, and
  how long until the idle stop.
- **`mrun cloud down`** stops the local relay and terminates every pod mrun created. A stopped
  pod is still listed in RunPod (its disk costs a little); `cloud down` removes it.

Weights download every time a new pod starts. To keep them between pods, cache them on a network
volume in one data center: `--volume-dc EU-RO-1`. The pod then runs in that data center, and the
volume costs about $0.07 per GB per month until `mrun cloud down --volume` deletes it. If the
first `cloud up` with a new volume fails, it deletes that volume; an existing volume is kept. Only
a few data centers offer volumes, and a fitting GPU there may be sold out.

## `cloud up` options

| Flag | Default | Effect |
| --- | --- | --- |
| `--model <REPO>` | — | Hugging Face repo to serve (see `mrun cloud models`) |
| `--gpu <TYPE>` | cheapest that fits | GPU type, id or display name (`mrun cloud gpus`) |
| `--community` | off | Rent in Community Cloud (cheaper, hosts run by third parties) |
| `--idle-stop <DURATION>` | `30m` | Stop the pod after this long without a request |
| `--max-runtime <DURATION>` | `8h` | Stop the pod after this long regardless |
| `--max-model-len <N>` | the catalog's, else 65,536 | Context in tokens |
| `--max-num-seqs <N>` | 16 | vLLM's concurrent sequences |
| `--tool-parser <P>` | the catalog's, else `hermes` | vLLM tool-call parser |
| `--reasoning-parser <P>` | the catalog's | vLLM reasoning parser |
| `--volume-dc <DC>` | none | Cache weights on a network volume in this data center |
| `--retry <DURATION>` | no retry | While every fitting GPU is sold out, keep trying this long |
| `--max-price <USD>` | the price shown first | Most $/h a `--retry` may rent at (needs `--retry`) |
| `--any-cloud` | off | With `--retry`: try both clouds, cheapest first (needs `--retry`) |
| `--keep-on-error` | off | Leave a pod whose model failed to start running, for inspection |
| `--agent-url <URL>`, `--agent-sha256 <HEX>` | this version's release asset | Where the pod fetches the agent, and its hash |
| `--agent-file <PATH>` | — | Upload a local Linux build of mrun as the agent |
| `--port <PORT>` | 8484 (`MRUN_PORT`) | Port of the local relay |
| `-v`, `--verbose` | off | Print vLLM's output live while the pod starts |
| `-y`, `--yes` | off | Do not ask |

## Set your cloud policy once

The flags of `cloud up` can live in the `[cloud]` table of the
[config file](@/docs/configuration.md#config-file) (`~/.mrun/config.toml`):

```toml
[cloud]
idle-stop = "30m"
max-runtime = "8h"
retry = "1h"        # keep trying sold-out GPUs for an hour
any-cloud = true    # both clouds, cheapest first (Community hosts are run by third parties)
max-price = 2.6
```

The file cannot skip the confirmation: `cloud up` still shows the GPU, price, limits and retry
list and asks, and lines whose values came from the file end with `(config)`. A flag overrides
the file for one run (`--idle-stop 15m`, `--any-cloud=false`).

## Security

- The pod's port is reachable through RunPod's public HTTPS proxy. Every request must carry a
  random token, created per pod. Requests without it get 401 and reach neither vLLM nor the GPU.
- vLLM listens on the pod's loopback only, with its own random key.
- Your RunPod API key is used on your Mac for the API calls and nothing else: it is never written
  to disk, logged, or sent to the pod. To stop itself, the pod uses the pod-scoped key that RunPod
  gives every pod.

## What differs from local mode

The pod runs vLLM, not mrun's engine. mrun's local-engine guarantees (exact prefix reuse,
splicing, memory budget refusals, miss causes) do not apply; vLLM has its own prefix cache.
`mrun ps` and `/mrun/status` show `backend: vllm <version>` and leave the engine-only fields
empty. The local relay normalizes Claude Code's per-request attribution nonce so the prefix stays
stable, sends keepalives in each API's format during long prefills, and cancels the remote
request when your client disconnects.

## The pod agent

The pod runs a Linux build of mrun (`mrun agent`). By default `cloud up` has the pod fetch this
version's release asset and check its sha256; `--agent-url` with `--agent-sha256` names another.
A development build has no release asset: build the agent on your Mac and let `cloud up` upload
it over the token-checked proxy:

```sh
cargo install cargo-zigbuild && rustup target add x86_64-unknown-linux-gnu
cargo zigbuild --release --target x86_64-unknown-linux-gnu.2.28 --no-default-features -p mrun
mrun cloud up --model Qwen/Qwen3-4B-Instruct-2507 \
  --agent-file target/x86_64-unknown-linux-gnu/release/mrun
```

## Troubleshooting

When vLLM fails to start, the cause is usually one exception line (`ValueError: …`,
`torch.OutOfMemoryError: …`) followed by a long traceback. `cloud up` prints the last lines it
sees and keeps the whole log:

- **Saved log.** On any failed or interrupted start (Ctrl-C included), `cloud up` downloads the
  pod's full log before it terminates the pod, saves it to `~/.mrun/logs/cloud-<pod id>.log`
  (readable by you only), and prints that path and the log's last exception lines.
- **`--verbose` (`-v`).** Prints every line of vLLM's output live while the pod starts, indented
  under the phase lines.
- **`mrun cloud logs`.** The running pod's full log: `--tail 100` (`-n`) for the last 100 lines,
  `-f` to keep following it until Ctrl-C.

The log holds vLLM's startup, loading and error output and the agent's own lines (`[agent] …`:
phases, the self-stop decision), each with a UTC timestamp. vLLM does not log prompt or output
contents by default, and mrun never turns its request logging on. In the pod the log is
`/tmp/mrun/vllm.log`, capped at 50 MB with one rotated file.

**`max_num_seqs (1024) exceeds available Mamba cache blocks`.** Hybrid models (attention plus
Mamba layers, such as Qwen3-Coder-Next) need one Mamba cache block per running request. mrun
starts vLLM with `--max-num-seqs 16`, enough for one coding agent with its parallel subagents. If
a model still fails with this message, lower it: `mrun cloud up --max-num-seqs 8 --model <repo>`.

## Commands

| Command | What it does |
| --- | --- |
| `mrun cloud up --model <repo>` | Create a pod, wait until it serves, relay it on 127.0.0.1 |
| `mrun cloud status [--json]` | The pod, its cost and the agent's phase |
| `mrun cloud logs [-f] [--tail N]` | The running pod's full log (vLLM's output) |
| `mrun cloud down [--volume] [-y]` | Stop the relay, terminate mrun's pods (and delete volumes) |
| `mrun cloud gpus [--json]` | GPU types, VRAM, prices |
| `mrun cloud models [--json]` | Curated models with the cheapest fitting GPU |

See also [CLI](@/docs/cli.md#mrun-cloud) and [Clients](@/docs/clients.md).
