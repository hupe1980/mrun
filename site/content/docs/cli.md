+++
title = "CLI reference"
description = "Every mrun command and flag, exit codes and error codes."
weight = 80
+++

Every command accepts `-h/--help` (with examples), `-q/--quiet` (only warnings and errors) and
`-V/--version`.

Boolean preference flags (`--no-cache`, `--any-cloud`, …) take an optional value (`--flag[=<BOOL>]`):
`--flag=false` overrides the [config file](@/docs/configuration.md#config-file) and `MRUN_*`
variables.

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
| `mrun pool` | `start [--unencrypted]`, `ls`, `join <pool>`, `confirm <code>`, `admit [<mac>]`, `decline <mac>`, `leave`, `remove <mac>`, `plan <model>`, `serve <model>`, `rejoin`, `replan`; `--json`, `--port`. A member's part of a model shows in `mrun ls` as `layers a–b · for pools`. See [Mac pool](@/docs/pool.md) |
| `mrun stop` | Stop a server started with `mrun serve --detach`, also while it generates; exits 0 when nothing runs |
| `mrun logs` | Show the log of servers started in the background |
| `mrun cache clear` | Drop all cached state of the running server, the live conversation included (after the running request) |
| `mrun pull <MODEL>` | Download a model into mrun's store (or import it from the Hugging Face cache), verify it and record it |
| `mrun ls` | List pulled models and the aliases with family, size, fit, maximum context on this machine and status (`preview` for `qwen3.5`) |
| `mrun show <MODEL>` | Show a pulled model: family, architecture, quantization, license, identity and fit |
| `mrun rm <MODEL>` | Remove a pulled model from mrun's store |
| `mrun verify <MODEL>` | Re-hash every file of a pulled model against its manifest |
| `mrun bench` | Measure mrun and, with `--compare`, other local servers; see [Benchmarks](@/docs/benchmarks.md) |
| `mrun cloud up\|down\|status\|logs\|gpus\|models` | Opt-in: serve a model from a rented RunPod GPU behind the same local APIs; see [Remote GPU](@/docs/cloud.md) |
| `mrun ui` | Open mrun's window and menu bar item; see [The mrun window](@/docs/ui.md) |
| `mrun config path\|show\|schema` | The config file's path, every setting with its source, a JSON Schema; see [Configuration](@/docs/configuration.md#config-file) |
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
| `--budget-gib <GIB>` | Metal's recommended working set | A lower memory budget (`MRUN_BUDGET_GIB`); a value above the automatic one is refused |
| `--context <TOKENS>` | The most that fits | Memory for this many tokens of context, `64K` or `65536` (`MRUN_CONTEXT`); not with `--budget-gib` |
| `--speculation <auto\|off>` | `auto` | Speculative decoding, as for `serve` |
| `--info` | off | Print the memory budget breakdown after loading and exit |

In an interactive session, `/clear` starts a new conversation, `/info` prints the memory budget
and `/exit` (or `/quit`, `/bye`) quits. Turns reuse the prefix cache.

## `mrun serve`

| Flag | Variable | Default | Meaning |
| --- | --- | --- | --- |
| `--model <MODEL>` | `MRUN_MODEL` | This machine's default alias, or the largest pulled model that fits | Model to serve |
| `--port <PORT>` | `MRUN_PORT` | 8484 | Port on 127.0.0.1 |
| `--max-output <N>` | `MRUN_MAX_OUTPUT` | 32768 | Output limit for requests that send none, clamped to the context left after the prompt. At least 1 |
| `--queue <N>` | `MRUN_QUEUE` | 16 | Requests that may wait while one runs. At least 1 |
| `--queue-wait-s <S>` | `MRUN_QUEUE_WAIT_S` | 240 | Seconds a non-streaming request may wait; one whose estimated wait is longer is refused at once. Streaming requests never time out in the queue. At least 1 |
| `--budget-gib <GIB>` | `MRUN_BUDGET_GIB` | Metal's recommended working set | A lower memory budget; a value above the automatic one is refused (see [Memory](@/docs/memory.md#tuning)) |
| `--context <TOKENS>` | `MRUN_CONTEXT` | The most that fits | The budget this many tokens of context need, `64K` or `65536`; refused with E042 when it does not fit; not with `--budget-gib` ([Memory](@/docs/memory.md#context)) |
| `--no-cache` | `MRUN_NO_CACHE` | off | Disable prefix reuse across requests |
| `--speculation <auto\|off>` | `MRUN_SPECULATION` | `auto` | Speculative decoding: drafts from the prompt, keeps only what plain decoding would produce, while it is faster ([How it works](@/docs/how-it-works.md#speculative-decoding)) |
| `--reject-tool-images` | `MRUN_REJECT_TOOL_IMAGES` | off | Refuse images inside tool results instead of omitting them |
| `--record <DIR>` | — | off | Record requests and responses as JSON lines into this directory |
| `--detach` | `MRUN_DETACH` | off | Run in the background (stop it with `mrun stop`) |
| `--relay <AGENT_URL>` | `MRUN_RELAY_TOKEN` (the pod token) | off | Relay to a remote mrun agent instead of loading a model; started by `mrun cloud up` ([Remote GPU](@/docs/cloud.md)) |

`--detach` returns once the model is loaded, or reports the load failure. It fails with E003
while a server runs on that port. Background servers share `~/.mrun/server.log` (moved to
`server.log.1` at 10 MB on the next start); each port has a pid file `~/.mrun/server-<port>.pid`.

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

Without a server, `launch` asks once whether to pull the model (if needed) and start it. A
loading server is waited for; a server with a different model than `--model` is offered a
restart. A model whose context is below the client's minimum (65,536 tokens for Claude Code) is
refused with E040 before anything is pulled or started. See [Clients](@/docs/clients.md).

## Other commands

| Command | Flags |
| --- | --- |
| `mrun connect <CLIENT>` | `--write` writes the configuration (diff, confirmation, backup in `~/.mrun/backups/<client>/`, server check; not for `crush` and `zed`); `--restore` puts the newest backup back; `-y, --yes`; `--port <PORT>` (8484, `MRUN_PORT`). `--json` prints the client's state, the files changed (or that `--write` would change) and `blocks` (`path`, `note`, `text`) |
| `mrun ps` | `--json` prints the status JSON; `--port`. `--sessions` prints the sessions of coding agents and chats with their rows and reuse (`--json`: `/mrun/sessions`). `-w, --watch` keeps polling: it redraws twice a second on a terminal and prints one JSON line per poll otherwise (or with `--json`); Ctrl-C ends it with exit 0 |
| `mrun doctor` | `--port` |
| `mrun stop` | `--port`. Stops a generating server too: new requests are refused, running ones get 5 seconds, then are cancelled. Nothing running is a note, exit 0 |
| `mrun logs` | `-f, --follow` keeps printing new lines; `-n, --lines <N>` lines from the end (50) |
| `mrun cache clear` | — |
| `mrun pull <MODEL>` | `MODEL` is an alias or `org/name[@revision]`. `-y, --yes` downloads even when the model does not fit. `--json` prints one JSON line per event (`plan`, `progress`, `done`, `error`; `confirm` without `--yes`). `--dry-run` prints the plan (with `--json`: one `plan` event with size, `family` and fit) and exits 0 without downloading |
| `mrun ls` | `--json`, with each model's `fit` (`fits`, `tight`, `wont_fit`), `verdict` and `recommended`; `--budget-gib` shows the verdicts for a smaller budget; `--context <TOKENS>` the memory each model needs for that context (rows carry `memory: {fixed, per_token}` and `budget`); `--port <PORT>` the server whose model is marked `serving` (8484, `MRUN_PORT`) |
| `mrun show <MODEL>`, `mrun verify <MODEL>` | `MODEL` is an alias or `org/name[@revision]` of a pulled model |
| `mrun rm <MODEL>` | `-y, --yes` removes without asking. Removing the model the running server serves needs its own confirmation. The Hugging Face cache is not touched |
| `mrun bench` | `--suite`, `--trace`, `--system`, `--compare`, `--reps`, `--out`; see [Benchmarks](@/docs/benchmarks.md) |
| `mrun ui` | `--browser` opens it in the default browser; `--background` shows only the menu bar item (used by Open at Login); `--port <PORT>` the window's loopback port (8485, `MRUN_UI_PORT`; the next free one when taken); `--server-port <PORT>` the server it shows and starts (8484, `MRUN_PORT`). A second `mrun ui` brings the open window forward |
| `mrun config path` | The config file mrun reads and whether it exists; works when the file does not parse |
| `mrun config show` | `--json`. Every setting with its value and source (`default`, `file:<line>`, `env MRUN_*`), as valid TOML |
| `mrun config schema` | A JSON Schema (draft 2020-12) of the config file for editors |

## `mrun cloud`

Opt-in ([Remote GPU](@/docs/cloud.md)); the RunPod key comes from `RUNPOD_API_KEY` in the
environment or `./.env`. Prompts and code are sent to RunPod.

| Command | Flags |
| --- | --- |
| `mrun cloud up --model <REPO>` | `--gpu <NAME>` (the cheapest that fits), `--community`, `--idle-stop <DUR>` (30m), `--max-runtime <DUR>` (8h), `--retry <DUR>` (keep trying sold-out GPUs that long), `--max-price <USD>` (with `--retry`; default the price shown), `--any-cloud` (with `--retry`: both clouds, cheapest first), `--max-model-len <N>`, `--max-num-seqs <N>` (16), `--tool-parser`, `--reasoning-parser`, `--port` (8484), `--volume-dc <DC>`, `--keep-on-error`, `-v, --verbose` (vLLM's log live), `--agent-url <URL> --agent-sha256 <HEX>` or `--agent-file <PATH>`, `-y, --yes`. Policy flags can be set in the config file's `[cloud]` table |
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
| E042 | 4 | The requested `--context` does not fit the budget or the model; names the memory it needs and the largest context that fits |
| E050 | 6 | Network or download failure |
| E051 | 6 | A gated model needs a Hugging Face token |
| E052 | 6 | Not enough disk space for the download |
| E060 | 6 | A Mac of the pool stopped answering; names it |
| E061 | 2 | A pool refused the join, or the codes differ |
| E062 | 4 | The model does not fit the pool; each Mac's numbers |
| E063 | 2 | Another Mac of the pool runs another mrun version |
