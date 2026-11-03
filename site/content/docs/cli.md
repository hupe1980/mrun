+++
title = "CLI reference"
description = "Every mrun command and flag: run, serve, launch, connect, ps, doctor, stop, logs, cache, pull, ls, show, rm, verify, bench, cloud, config, completions, with exit codes and error codes."
weight = 80
+++

Every command accepts `-h/--help` (with examples), `-q/--quiet` (print only warnings and errors)
and `-V/--version`. This page describes mrun 0.1 as designed on 2026-10-16; a source build may still lag behind in places.

Boolean flags that are preferences (`--no-cache`, `--any-cloud`, …) take an optional value:
`--flag` and `--flag=true` turn them on, `--flag=false` turns them off, overriding the
[config file](@/docs/configuration.md#config-file) and `MRUN_*` variables. `--help` shows them as
`--flag[=<BOOL>]`.

## Commands

| Command | What it does |
| --- | --- |
| `mrun` | Without a command: whether a server runs, the pulled models and the next commands to try |
| `mrun run [MODEL]` | Chat with a local model in the terminal |
| `mrun serve` | Serve the Anthropic Messages, OpenAI Chat Completions and Responses APIs on 127.0.0.1 |
| `mrun launch [CLIENT] [-- ARGS]` | Start a coding agent configured for the local server (this process only); pulls the model and starts the server when needed |
| `mrun connect <CLIENT>` | Print the persistent configuration for a client; `--write` writes it with a backup, `--restore` undoes that |
| `mrun ps` | Show the running server: model, budget, requests, cache and the last miss |
| `mrun doctor` | Check this Mac, the server and the clients; prints a fix for each problem |
| `mrun stop` | Stop a server started with `mrun serve --detach`, also while it generates; exits 0 when nothing runs |
| `mrun logs` | Show the log of servers started in the background |
| `mrun cache clear` | Drop all cached state of the running server, the live conversation included (after the running request, if any); prints `cleared N branches (M MiB) and the live conversation` |
| `mrun pull <MODEL>` | Download a model into the Hugging Face cache, verify it and record it |
| `mrun ls` | List pulled models and the curated aliases with family, size, fit and maximum context on this machine; a preview alias (`qwen3.5`) shows `preview` in STATUS with a note |
| `mrun show <MODEL>` | Show a pulled model: family, architecture, quantization, license, identity and fit (plus a `preview:` line for a preview alias) |
| `mrun rm <MODEL>` | Remove a pulled model's revision from the Hugging Face cache |
| `mrun verify <MODEL>` | Re-hash every file of a pulled model against its manifest |
| `mrun bench` | Measure mrun and, with `--compare`, other local servers; see [Benchmarks](@/docs/benchmarks.md) |
| `mrun cloud up\|down\|status\|logs\|gpus\|models` | Opt-in remote GPU mode: run a model on a rented RunPod GPU behind the same local APIs; see [Remote GPU](@/docs/cloud.md) |
| `mrun ui` | Open mrun's window and menu bar item: status, chat, models and agents; see [The mrun window](@/docs/ui.md) |
| `mrun config path\|show\|schema` | The optional config file: its path, every setting with its value and source, and a JSON Schema for editors; see [Configuration](@/docs/configuration.md#config-file) |
| `mrun completions <SHELL>` | Print a shell completion script (`bash`, `elvish`, `fish`, `powershell`, `zsh`) |

`MODEL` is an alias (`qwen3.5`, `qwen3.6`), a Hugging Face repository id (`org/name`, with
`@revision` where noted) or, for `run` and `serve`, a local directory.

## `mrun run`

```text
mrun run [OPTIONS] [MODEL]
```

| Flag | Default | Meaning |
| --- | --- | --- |
| `MODEL` | The running server's model, else this machine's default | Model to load |
| `--prompt <PROMPT>` | — | One-shot prompt; without it an interactive session starts |
| `--system <SYSTEM>` | — | System prompt |
| `--temperature`, `--top-p`, `--top-k`, `--min-p` | The model's `generation_config.json`, else 0, 1, 0 (off), 0 (off) | Sampling |
| `--seed <SEED>` | — | Sampling seed |
| `--max-tokens <N>` | 2048 | Output limit per reply |
| `--stop <STOP>` | — | Stop sequence (up to 4, each at most 256 bytes) |
| `--think` | off | Let models that support it think before answering |
| `--budget-gib <GIB>` | MLX's recommended working set | A lower memory budget (`MRUN_BUDGET_GIB`); a value above the automatic one is refused |
| `--info` | off | Print the memory budget breakdown after loading and exit |

In an interactive session, `/clear` starts a new conversation, `/info` prints the memory budget
and `/exit` (or `/quit`, `/bye`) quits. Each turn reuses the previous one through the same prefix
cache the server uses.

## `mrun serve`

| Flag | Variable | Default | Meaning |
| --- | --- | --- | --- |
| `--model <MODEL>` | `MRUN_MODEL` | This machine's default alias, or the largest pulled model that fits | Model to serve |
| `--port <PORT>` | `MRUN_PORT` | 8484 | Port on 127.0.0.1 |
| `--max-output <N>` | `MRUN_MAX_OUTPUT` | 32768 | Default output limit for requests that send none, on all three APIs; it is further clamped to the context left after the prompt. At least 1 (0 is a usage error, exit 2) |
| `--queue <N>` | `MRUN_QUEUE` | 16 | Requests that may wait while one generates (0.1 runs one at a time). At least 1 |
| `--queue-wait-s <S>` | `MRUN_QUEUE_WAIT_S` | 240 | Seconds a non-streaming request may wait in the queue; one whose estimated wait is longer is refused at once. Streaming requests receive keepalives and never time out in the queue. At least 1 |
| `--budget-gib <GIB>` | `MRUN_BUDGET_GIB` | MLX's recommended working set | A lower memory budget; a value above the automatic one is refused (see [Memory](@/docs/memory.md#tuning)) |
| `--no-cache` | `MRUN_NO_CACHE` | off | Disable prefix reuse across requests |
| `--reject-tool-images` | `MRUN_REJECT_TOOL_IMAGES` | off | Refuse images inside tool results instead of omitting them |
| `--record <DIR>` | — | off | Record requests and responses as JSON lines into this directory |
| `--detach` | `MRUN_DETACH` | off | Run in the background (stop it with `mrun stop`) |

| `--relay <AGENT_URL>` | `MRUN_RELAY_TOKEN` (the pod token) | off | Relay to a remote mrun agent instead of loading a model; started by `mrun cloud up` ([Remote GPU](@/docs/cloud.md)) |

`--detach` returns once the server serves a loaded model and reports a load failure with the
server's own error. `serve --detach` while a server already runs on that port fails with E003.
Background servers share one log, `~/.mrun/server.log`, which is moved to `server.log.1` when it
reaches 10 MB at the next start; each port has its own pid file (`~/.mrun/server-<port>.pid`).

## `mrun launch`

```text
mrun launch [OPTIONS] [CLIENT] [-- <CLIENT_ARGS>...]
```

| Flag | Default | Meaning |
| --- | --- | --- |
| `CLIENT` | The only one installed | `claude-code`, `codex` or `opencode` |
| `--model <MODEL>` | The running server's, else this machine's default | Model to serve |
| `--dry-run` | off | Print the environment and command instead of starting anything |
| `-y, --yes` | off | Answer yes to confirmations (pulling the model, starting or restarting the server) |
| `--port <PORT>` | 8484 (`MRUN_PORT`) | Port of the server |
| `-- <ARGS>` | — | Arguments passed to the client |

Without a server, `launch` asks once whether to pull the model (if needed) and start the server.
A loading server is waited for; a server with a different model than `--model` is offered a
restart. A model whose context on this Mac is below the client's minimum (65,536 tokens for
Claude Code) is refused with E040 before anything is pulled or started. What each client gets is listed in [Clients](@/docs/clients.md).

## Other commands

| Command | Flags |
| --- | --- |
| `mrun connect <CLIENT>` | `--write` writes the configuration (plan, diff, one confirmation, backup in `~/.mrun/backups/<client>/`, atomic write, check against the server); `--restore` puts the newest backup back; `-y, --yes`; `--port <PORT>` (8484, `MRUN_PORT`). Without a server it prints the values for this Mac's default model. `--write` is not available for `crush` and `zed` (print only). `--json` prints the client's state and the files `--write` would change (or, with `--write`/`--restore`, did change) and `blocks` (`path`, `note`, `text`: the blocks it prints, for every client) |
| `mrun ps` | `--json` prints the status JSON; `--port`. `-w, --watch` keeps polling: it redraws twice a second on a terminal and prints one JSON line per poll otherwise (or with `--json`); Ctrl-C ends it with exit 0 |
| `mrun doctor` | `--port` |
| `mrun stop` | `--port`. Stops a generating server too: new requests are refused, running ones get 5 seconds, then are cancelled. Nothing running is a note, exit 0 |
| `mrun logs` | `-f, --follow` keeps printing new lines; `-n, --lines <N>` lines from the end (50) |
| `mrun cache clear` | — |
| `mrun pull <MODEL>` | `MODEL` is an alias or `org/name[@revision]`. `-y, --yes` downloads even when the model does not fit this machine's memory budget. `--json` prints one JSON line per event (`plan`, `progress`, `done`, `error`; `confirm` without `--yes`). `--dry-run` prints the plan (with `--json`: one `plan` event with the size, `family` and fit) and exits 0 without asking, downloading or writing anything, even for a model that does not fit |
| `mrun ls` | `--json` prints JSON instead of a table, with each model's `fit` (`fits`, `tight`, `wont_fit`), `verdict` and `recommended`; `--budget-gib` shows the verdicts for a smaller budget; `--all` is accepted (the aliases are always listed); `--port <PORT>` the server whose model is marked `serving` (8484, `MRUN_PORT`) |
| `mrun show <MODEL>`, `mrun verify <MODEL>` | `MODEL` is an alias or `org/name[@revision]` of a pulled model |
| `mrun rm <MODEL>` | `-y, --yes` removes without asking. Warns that other tools using the same revision lose it; removing the model the running server serves needs its own confirmation |
| `mrun bench` | `--suite`, `--trace`, `--system`, `--compare`, `--reps`, `--out`; see [Benchmarks](@/docs/benchmarks.md) |
| `mrun ui` | `--browser` opens the window in the default browser; `--background` shows only the menu bar item (used by Open at Login); `--port <PORT>` the window's own loopback port (8485, `MRUN_UI_PORT`; the next free one when taken); `--server-port <PORT>` the server it shows and starts (8484, `MRUN_PORT`). A second `mrun ui` brings the open window forward |
| `mrun config path` | The config file mrun reads (`$MRUN_CONFIG` or `~/.mrun/config.toml`) and whether it exists; works even when the file does not parse |
| `mrun config show` | `--json`. Every setting with the value a command would use and its source (`default`, `file:<line>`, `env MRUN_*`), overridden values and settings that need another; valid TOML |
| `mrun config schema` | A JSON Schema (draft 2020-12) of the config file for editors |

Not in 0.1: a background login service and listening on addresses other than loopback. `--reserve-gib`, `MRUN_RESERVE_GIB` and `--allow-over` were removed on 2026-10-16;
use `--budget-gib` to lower the budget.

## `mrun cloud`

Opt-in remote GPU mode ([Remote GPU](@/docs/cloud.md)); the RunPod key comes from
`RUNPOD_API_KEY` in the environment or `./.env`. Prompts and code are sent to RunPod.

| Command | Flags |
| --- | --- |
| `mrun cloud up --model <REPO>` | `--gpu <NAME>` (the cheapest that fits), `--community`, `--idle-stop <DUR>` (30m), `--max-runtime <DUR>` (8h), `--retry <DUR>` (keep trying sold-out GPUs that long), `--max-price <USD>` (with `--retry`; default the price shown), `--any-cloud` (with `--retry`: both clouds, cheapest first), `--max-model-len <N>`, `--max-num-seqs <N>` (16), `--tool-parser`, `--reasoning-parser`, `--port` (8484), `--volume-dc <DC>`, `--keep-on-error`, `-v, --verbose` (vLLM's log live), `--agent-url <URL> --agent-sha256 <HEX>` or `--agent-file <PATH>`, `-y, --yes`. Every policy flag can live in the `[cloud]` table of the config file |
| `mrun cloud down` | `--volume` also deletes mrun's network volumes, `-y, --yes`, `--port` |
| `mrun cloud status` | `--json`, `--port` |
| `mrun cloud logs` | `-f, --follow`, `-n, --tail <N>`; a failed `up` saves the log to `~/.mrun/logs/cloud-<pod>.log` |
| `mrun cloud gpus`, `mrun cloud models` | `--json` |

One mrun pod per account: `cloud up` with the model of an existing pod attaches to it, and with
another model it refuses. A GPU too small for the model is refused before anything is created
(E040). Any failure after the pod exists terminates it, and so does Ctrl-C.

## Exit codes

| Code | Meaning |
| --- | --- |
| 0 | Success |
| 1 | Unexpected error |
| 2 | Usage error: invalid or contradictory arguments, a confirmation needed without a terminal, a client not installed |
| 3 | Model not found, ambiguous, unsupported or corrupt |
| 4 | Model does not fit the memory budget |
| 5 | Server not running or unreachable |
| 6 | Network or download failure |
| 7 | `mrun doctor` found failures |

## Error codes

Errors name the problem and the next step:

```text
error[E031]: model 'qwen3.7' not found
  hint: did you mean 'qwen3.6'? Run 'mrun ls' to see local models.
```

| Code | Exit | Meaning |
| --- | --- | --- |
| E001 | 1 | Unexpected internal error |
| E002 | 2 | A confirmation is needed and stdin is not a terminal (pass `--yes`) |
| E003 | 2 | The arguments contradict each other or the running server (for example `serve --detach` while a server runs) |
| E010 | 5 | The server is not running, not reachable, failed to start or did not stop in time |
| E020 | 2 | The client program is not installed |
| E031 | 3 | The model is not in the local cache |
| E032 | 3 | A model file does not match its recorded hash, or metadata from the hub or a manifest is not well-formed |
| E033 | 3 | Several cached models match the name |
| E034 | 3 | The architecture or quantization mode is not supported, or the chat template does not render the family's tool-call syntax |
| E040 | 4 | The model does not fit this Mac's memory budget, no alias fits, or its context is below the client's minimum (names the budget needed) |
| E041 | 4 | Other apps or VMs hold the memory the model needs right now; names the largest and a smaller model that fits ([Memory](@/docs/memory.md#other-apps-and-vms)) |
| E050 | 6 | Network or download failure |
| E051 | 6 | A gated model needs a Hugging Face token |
| E052 | 6 | Not enough disk space for the download |
