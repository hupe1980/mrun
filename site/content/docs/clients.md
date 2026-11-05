+++
title = "Clients"
description = "Connect Claude Code, Codex CLI, OpenCode, Crush, Zed and other coding agents to mrun."
weight = 20
+++

`mrun launch <client>` starts a client configured for mrun, for that process only; your client
settings stay untouched. Without a running server, `launch` asks once whether to pull the model
(this Mac's default unless `--model` names one) and start the server; a loading server is waited
for, and a server with a different model than `--model` is offered a restart. `-y` answers yes,
`--dry-run` prints the environment and command, and arguments after `--` go to the client.
Without a client name, `launch` picks the only one installed.

`launch claude-code` refuses a model that leaves less than 65,536 tokens of context, before
pulling or starting anything, and names the budget it would need. For Codex and OpenCode,
`launch` warns below 49,152 tokens.

`mrun connect <client>` prints the persistent configuration and where it belongs, with the running
server's values (else those of this Mac's default model):

- `--write` shows a plan and a diff, asks once, backs up the current file to
  `~/.mrun/backups/<client>/`, writes atomically (merging only mrun's keys), then probes the server
  (`/health` and `/v1/models`). Claude Code, Codex and OpenCode only.
- `--restore` puts the newest backup back (or removes a file mrun created), with the same diff and
  confirmation.
- `-y` answers the confirmation; `--json` prints the client's state and the files changed (or that
  `--write` would change).

| Client | Protocol | `launch` | `connect` prints |
| --- | --- | --- | --- |
| Claude Code (2.1.290 or later) | Anthropic Messages | `mrun launch claude-code` | A settings file `~/.mrun/claude/settings.json` for `claude --settings` |
| Codex CLI | OpenAI Responses | `mrun launch codex` | The profile `$CODEX_HOME/mrun.config.toml` (use `codex --profile mrun`) |
| OpenCode | Chat Completions | `mrun launch opencode` | A provider entry for `~/.config/opencode/opencode.json` |
| Crush | Chat Completions | — | `openai-compat` provider commands for `~/.config/crush/crushrc` (print only) |
| Zed | Chat Completions | — | An `openai_compatible` provider for Zed's `settings.json` (print only) |

After `mrun cloud up`, `launch` and `connect` use the remote model's id and context (see
[Remote GPU](@/docs/cloud.md)).

## Client gate

[`scripts/client_gate.py`](https://github.com/hupe1980/mrun/blob/main/scripts/client_gate.py)
starts each client through `mrun launch` on 10 scripted read-edit-test tasks
([fixtures](https://github.com/hupe1980/mrun/tree/main/crates/mrun/tests/clients/tasks)), 3 runs
each; a run passes when the task's own check passes. A release needs a Wilson 95 % lower bound of
at least 0.8 in every cell. M4 Pro, 48 GB, Claude Code 2.1.292, Codex 0.163.0, OpenCode 2.0.25:

| Model | Claude Code | Codex | OpenCode |
| --- | --- | --- | --- |
| `qwen3.6` | 30/30 [0.89, 1.00], median 40.6 s | 29/30 [0.83, 0.99], 21.4 s | 30/30 [0.89, 1.00], 20.7 s |
| `qwen3.5` (preview) | 27/30 [0.74, 0.97], 76.2 s | 26/30 [0.70, 0.95], 37.6 s | 28/30 [0.79, 0.98], 38.3 s |

`qwen3.5` is below the gate: each failure is a session where the model gets stuck (looping, or
reasoning for 13,000–22,000 tokens) until the 600 s timeout. Run the gate yourself:
`python3 scripts/client_gate.py --model qwen3.6`
([CONTRIBUTING](https://github.com/hupe1980/mrun/blob/main/CONTRIBUTING.md#client-gate)).

## Claude Code

`mrun launch claude-code` sets these variables for that process:

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

- `--settings '{"idleCompaction":false}'`: mrun's cache does not expire, so idle compaction would
  only discard a conversation the cache still holds.
- `--disallowedTools WebSearch`: WebSearch runs on Anthropic's servers (`connect` writes it as a
  permission deny rule).

`CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY` is not set: Claude Code would hide model ids without
`claude` or `anthropic`.

Entries in the `env` block of `~/.claude/settings.json` (or `$CLAUDE_CONFIG_DIR/settings.json`)
can override what `launch` sets; `mrun doctor` warns about each. `mrun connect claude-code`
prints the settings as a separate file for `claude --settings`; mrun never merges them into
`~/.claude/settings.json`, which would redirect every Claude Code session.

## Codex

`launch` starts `codex` with `-c` overrides: a provider named `mrun` (`wire_api = "responses"`,
base URL `http://127.0.0.1:8484/v1`, no `env_key`), `web_search = "disabled"` (a hosted tool), and
the model's context window as `model_context_window` (Codex assumes 272,000 tokens for unknown
models). Your Codex files stay untouched. Your own `-c` overrides that conflict with this provider
are refused (E003). `mrun connect codex` prints the same setup as the profile
`$CODEX_HOME/mrun.config.toml`, for `codex --profile mrun`.

Codex sends no output limit, so its requests get the context left after the prompt, at most
32,768 tokens. Codex resends the conversation every turn; requests that depend on server-stored
state (`previous_response_id`) are refused.

## OpenCode

`launch` passes the configuration through `OPENCODE_CONFIG_CONTENT`, so no OpenCode file is
written: an OpenAI-compatible provider named `mrun` at `http://127.0.0.1:8484/v1` with the model's
context, input and output limits.

- **`--standalone`**: OpenCode starts a private server that reads this configuration instead of
  its shared background service. Pass `--standalone` or `--server` yourself to override.
- **`small_model`**: titles and other side tasks go to the same model without thinking
  (`<model>-no-thinking`, which sends `chat_template_kwargs.enable_thinking: false`), so a title
  cannot think until the output limit while your request waits.
- **Code Mode off** (`"tools": {"execute": false}`): the served models are not trained to write
  JavaScript that calls tools. All direct tools stay. Give each MCP server `"codemode": false` to
  expose its tools directly.

`connect opencode` prints the same provider for `~/.config/opencode/opencode.json`. Its top-level
`"model"` makes mrun the default everywhere; leave it out to pick mrun with `/models`. After
`connect --write`, run `opencode service restart` (or quit every OpenCode window).

## Crush and Zed

`mrun connect crush` and `mrun connect zed` print a provider entry for `http://127.0.0.1:8484/v1`
with the served model and its context limit (Crush: `provider add` and `model add` lines for
`crushrc`; Zed: an `openai_compatible` entry for `settings.json`). Both are print-only, have no
`launch` command and are not in the client gate.

## Other clients

Any tool that speaks OpenAI Chat Completions or Anthropic Messages works: set the base URL to
`http://127.0.0.1:8484/v1` (Chat Completions) or `http://127.0.0.1:8484` (Messages) and any API key.
Unknown model names are served by the loaded model. Set the client's context limit to the value
`mrun ps` shows.
