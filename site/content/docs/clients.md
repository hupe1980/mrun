+++
title = "Clients"
description = "Connect Claude Code, Codex CLI, OpenCode, Crush, Zed and other OpenAI-compatible coding agents to a local mrun server."
weight = 20
+++

`mrun launch <client>` starts a client configured for mrun, for that one process; your client
settings stay untouched. Without a running server, `launch` asks once whether to pull the model
(this Mac's default unless `--model` names one) and start the server; a server that is still
loading is waited for, and a server with a different model than `--model` is offered a restart.
`-y` answers yes, `--dry-run` prints the environment and command instead of starting anything,
and arguments after `--` go to the client. Without a client name, `launch` picks the only one
installed.

`launch claude-code` refuses a model that would leave less than 65,536 tokens of context on this
Mac, before pulling or starting anything, and names the budget that model would need. For Codex
and OpenCode, `launch` warns below 49,152 tokens.

`mrun connect <client>` prints the equivalent persistent configuration and where it belongs, with
the running server's values (without a server, the values for this Mac's default model). It
writes nothing unless you ask:

- `mrun connect <client> --write` shows a plan and a diff, asks once, backs up the current file
  to `~/.mrun/backups/<client>/`, writes atomically (merging only mrun's keys into an existing
  file), then probes the server at the written address (`/health` and `/v1/models`).
- `mrun connect <client> --restore` puts the newest backup back (or removes a file mrun created),
  with the same diff and confirmation.
- `-y` answers the confirmation; `--json` prints the client's state and the files `--write` would
  change (with `--write` or `--restore`: the files changed) as JSON.

`--write` is available for Claude Code, Codex and OpenCode. Crush and Zed stay print-only: their
files are your own, and mrun cannot merge into them safely.

| Client | Protocol | `launch` | `connect` prints |
| --- | --- | --- | --- |
| Claude Code (2.1.290 or later) | Anthropic Messages | `mrun launch claude-code` | A settings file `~/.mrun/claude/settings.json` for `claude --settings` |
| Codex CLI | OpenAI Responses | `mrun launch codex` | The profile `$CODEX_HOME/mrun.config.toml` (use `codex --profile mrun`) |
| OpenCode | Chat Completions | `mrun launch opencode` | A provider entry for `~/.config/opencode/opencode.json` |
| Crush | Chat Completions | — | `openai-compat` provider commands for `~/.config/crush/crushrc` (print only) |
| Zed | Chat Completions | — | An `openai_compatible` provider for Zed's `settings.json` (print only) |

The same commands work with a model on a rented GPU: after `mrun cloud up`, `launch` and
`connect` use the remote model's id and context as they would a local model's (see
[Remote GPU](@/docs/cloud.md)).

## Client gate

Each client is tested with real tasks before a release: [`scripts/client_gate.py`](https://github.com/hupe1980/mrun/blob/main/scripts/client_gate.py)
starts it through `mrun launch` on 10 scripted read-edit-test tasks
([fixtures](https://github.com/hupe1980/mrun/tree/main/crates/mrun/tests/clients/tasks)), 3 runs each, and a run passes only when the task's own check passes. The
release needs a Wilson 95 % lower bound of at least 0.8 in every cell. Results on the M4 Pro
(48 GB) on 2026-10-16/17 with Claude Code 2.1.292, Codex 0.163.0 and OpenCode 2.0.25:

| Model | Claude Code | Codex | OpenCode |
| --- | --- | --- | --- |
| `qwen3.6` | 30/30 [0.89, 1.00], median 40.6 s | 29/30 [0.83, 0.99], 21.4 s | 30/30 [0.89, 1.00], 20.7 s |
| `qwen3.5` (preview) | 27/30 [0.74, 0.97], 76.2 s | 26/30 [0.70, 0.95], 37.6 s | 28/30 [0.79, 0.98], 38.3 s |

`qwen3.6` meets the gate with all three. `qwen3.5` is below it: every failure is a session in
which the 9B model gets stuck (looping on a regular-expression fix, or reasoning for 13,000–22,000
tokens) until the 600 s timeout; mrun serves 93–99 % of its prompt tokens from the cache. See
[Models](@/docs/models.md#aliases) for what that means for smaller Macs. Run the gate yourself:
`python3 scripts/client_gate.py --model qwen3.6`
([CONTRIBUTING](https://github.com/hupe1980/mrun/blob/main/CONTRIBUTING.md#client-gate)).

## Claude Code

`mrun launch claude-code` passes these settings, for that process only. Each is documented in
Claude Code's environment-variable reference:

| Variable | Value | Why |
| --- | --- | --- |
| `ANTHROPIC_BASE_URL` | `http://127.0.0.1:8484` (never ending in `/v1`) | Send requests to mrun |
| `ANTHROPIC_AUTH_TOKEN` | `mrun` | Claude Code requires a credential; mrun checks none |
| `ANTHROPIC_API_KEY` | empty | A key in your shell would otherwise take precedence |
| `ANTHROPIC_MODEL`, `ANTHROPIC_DEFAULT_MODEL`, `ANTHROPIC_DEFAULT_OPUS_MODEL`, `ANTHROPIC_DEFAULT_SONNET_MODEL`, `ANTHROPIC_DEFAULT_HAIKU_MODEL`, `ANTHROPIC_DEFAULT_FABLE_MODEL`, `CLAUDE_CODE_SUBAGENT_MODEL` | The served model | Every model slot, including subagents and background requests, uses the local model |
| `CLAUDE_CODE_SUBAGENT_MODEL_FORCE` | `1` | Agent definitions that name a Claude model also use the local one |
| `CLAUDE_CODE_USE_BEDROCK`, `CLAUDE_CODE_USE_VERTEX`, `CLAUDE_CODE_USE_FOUNDRY`, `CLAUDE_CODE_USE_ANTHROPIC_AWS`, `CLAUDE_CODE_USE_MANTLE` | `0` | A provider switch in your shell cannot redirect the session |
| `CLAUDE_CODE_ATTRIBUTION_HEADER` | `0` | The attribution block changes per request and would defeat prefix reuse |
| `CLAUDE_CODE_MAX_CONTEXT_TOKENS` | This Mac's context capacity | Claude Code compacts before the server would refuse |
| `CLAUDE_CODE_MAX_OUTPUT_TOKENS` | The server's output limit | Otherwise Claude Code reserves more of the window for output |
| `CLAUDE_CODE_AUTO_COMPACT_WINDOW` | This Mac's context capacity, only when it is at least 100,000 | Claude Code compacts at the real window (100,000 is its minimum) |
| `CLAUDE_CODE_GATEWAY_HINT_HEADERS` | `1`, only when the server records (`serve --record`) | Session and turn ids in headers, to group a recording's requests |
| `CLAUDE_CODE_DISABLE_NONSTREAMING_FALLBACK` | `1` | The fallback re-sends the whole prompt when a stream is slow |
| `CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC` | `1` | No calls to Anthropic's services from this session (this also skips its auto-update check) |
| `DISABLE_ERROR_REPORTING`, `CLAUDE_CODE_DISABLE_FEEDBACK_SURVEY` | `1` | No error reports or feedback surveys about a local model |
| `API_TIMEOUT_MS` | `3000000` (50 minutes) | Long local prompts take longer than hosted APIs |
| `ENABLE_TOOL_SEARCH` | `auto:10` | Behind a custom base URL tool search is off; this defers MCP tools only once their definitions exceed 10 % of the context |

`launch` also passes two arguments:

- `--settings '{"idleCompaction":false}'`. Claude Code otherwise compacts conversations left
  idle, ahead of a hosted cache expiry; mrun's cache does not expire, so compacting would only
  throw away a conversation the cache still holds.
- `--disallowedTools WebSearch`. WebSearch is a tool Anthropic's servers execute, which a local
  server cannot run (`connect` writes it as a permission deny rule).

Not set: `CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY` (Claude Code keeps only model ids
containing `claude` or `anthropic`, so the served model would not appear) and
`CLAUDE_CODE_WORKFLOW_SUBAGENT_MODEL` (undocumented; `CLAUDE_CODE_SUBAGENT_MODEL` covers workflow
agents).

Entries in the `env` block of `~/.claude/settings.json` (or `$CLAUDE_CONFIG_DIR/settings.json`)
can override what `launch` sets. `mrun doctor` warns about each one it finds. `mrun connect
claude-code` prints the same settings as a file to use with `claude --settings`, so your own
settings file stays as it is; mrun never merges them into `~/.claude/settings.json`, which would
redirect every Claude Code session.

## Codex

Codex speaks the OpenAI Responses API. `launch` starts `codex` with `-c` overrides that select a
provider named `mrun` (`wire_api = "responses"`, base URL `http://127.0.0.1:8484/v1`, no
`env_key`) and `web_search = "disabled"`, because Codex otherwise sends a hosted web-search tool a
local server cannot run. The model's real context window is passed as `model_context_window`,
because Codex otherwise assumes 272,000 tokens for unknown models and compacts too late. Your
`config.toml` and every other Codex file stay untouched. `-c` overrides of your own that conflict
with this provider are refused (E003). `mrun connect codex` prints the same setup as a profile,
`$CODEX_HOME/mrun.config.toml`, for `codex --profile mrun`.

Codex sends no output limit, so its requests get the context left after the prompt, at most
32,768 tokens. mrun is stateless: Codex resends the conversation every turn, and the prefix cache
makes that cheap. Requests that depend on server-stored state (`previous_response_id`) are
refused with a clear error.

## OpenCode

`launch` passes the provider only through the `OPENCODE_CONFIG_CONTENT` environment variable, so
no OpenCode file is written: an OpenAI-compatible provider named `mrun` at
`http://127.0.0.1:8484/v1` with the model's context, input and output limits.

- **`--standalone`.** OpenCode 2.x's UI and `opencode run` normally talk to a shared background
  service (`opencode serve --service`) started without mrun's configuration. `launch` starts
  OpenCode with `--standalone` (a private server that reads the configuration); pass
  `--standalone` or `--server` yourself to override.
- **Code Mode off** (`"tools": {"execute": false}`). Code Mode has the model write JavaScript that
  calls OpenCode's tools, which the served models are not trained on. Every direct tool (shell,
  read, edit and the others) stays. MCP servers reach the model through Code Mode by default, so
  give each one `"codemode": false` to expose its tools directly.

`connect opencode` prints the same provider for `~/.config/opencode/opencode.json`; its top-level
`"model"` makes mrun OpenCode's default everywhere, so leave it out to pick mrun with `/models`
instead. After `connect --write`, restart OpenCode's background service (`opencode service
restart`, or quit every OpenCode window) so it reads the new file.

## Crush and Zed

`mrun connect crush` and `mrun connect zed` print a provider entry pointing at
`http://127.0.0.1:8484/v1` with the served model and its context limit, and where it belongs in
the client's configuration (Crush: `provider add` and `model add` lines for `crushrc`; Zed: an
`openai_compatible` entry for `settings.json`). Both are print-only. Neither has a `launch`
command, and neither is part of the release tests.

## Other clients

Any tool that speaks OpenAI Chat Completions or Anthropic Messages can use mrun: set the base URL
to `http://127.0.0.1:8484/v1` (Chat Completions) or `http://127.0.0.1:8484` (Messages) and any API
key value. Unknown model names are served by the loaded model. Set the client's context limit to
the value `mrun ps` shows; most clients otherwise assume far more than fits on a laptop. These
clients are not part of the release tests.
