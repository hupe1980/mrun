+++
title = "Clients"
description = "Connect Claude Code, Codex CLI, OpenCode and any OpenAI-compatible coding agent to a local mrun server."
weight = 20
+++

Every client can be started with `mrun launch <client>`, which configures it for that one process
and leaves your client settings untouched. If no server is running, `launch` starts one; if no
model is downloaded, it offers the default. `--dry-run` prints what it would set.

`mrun connect <client>` prints the equivalent persistent configuration; `--write` applies it
(showing a diff and keeping a backup) and `--undo` restores the backup.

| Client | Protocol | `launch` | `connect --write` edits |
| --- | --- | --- | --- |
| Claude Code (2.1.273+) | Anthropic Messages | `mrun launch claude-code` | `~/.claude/settings.json` |
| Codex CLI | OpenAI Responses | `mrun launch codex` | `~/.codex/mrun.config.toml` (use `codex --profile mrun`) |
| OpenCode | Chat Completions | `mrun launch opencode` | `~/.config/opencode/opencode.json` |

## Claude Code

`mrun launch claude-code` sets:

| Variable | Value | Why |
| --- | --- | --- |
| `ANTHROPIC_BASE_URL` | `http://127.0.0.1:8484` | Send requests to mrun |
| `ANTHROPIC_AUTH_TOKEN` | API key reference, or a placeholder | Claude Code requires one |
| `ANTHROPIC_API_KEY` | empty | A key in your shell would otherwise take precedence |
| `ANTHROPIC_MODEL`, `ANTHROPIC_DEFAULT_{OPUS,SONNET,HAIKU}_MODEL`, `CLAUDE_CODE_SUBAGENT_MODEL` | The served model | Every model slot, including subagents and background tasks, uses the local model |
| `CLAUDE_CODE_MAX_CONTEXT_TOKENS` | This Mac's context capacity | Claude Code compacts before the server would refuse |
| `CLAUDE_CODE_MAX_OUTPUT_TOKENS` | The server's output limit | Otherwise Claude Code reserves 32,000 tokens of the window for output |
| `CLAUDE_CODE_GATEWAY_HINT_HEADERS` | `1` | Sends session headers, which `mrun serve --record` uses to group a recording's requests |
| `CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC` | `1` | Keeps the session local |
| `CLAUDE_CODE_TOTAL_TOKENS_REMINDER` | `off` | Removes a per-turn token count that would break prefix reuse |
| `API_TIMEOUT_MS` | 50 minutes | Long local prompts take longer than hosted APIs |
| `ENABLE_TOOL_SEARCH` | `auto:10` | Loads MCP tool definitions on demand once they exceed 10% of the context (see [MCP and skills](@/docs/mcp-and-skills.md)) |

Claude Code also adds a per-request nonce to its attribution line. mrun replaces just that nonce
with a fixed value, so reuse works even if you start `claude` yourself with only
`ANTHROPIC_BASE_URL` set.

## Codex

Codex speaks the OpenAI Responses API. `launch` selects a provider named `mrun` with
`wire_api = "responses"` through `-c` overrides and passes a model catalog with the real context
window — without it, Codex assumes 272,000 tokens for unknown models and never compacts in time.
`connect --write` creates a separate profile, so your main `config.toml` stays as it is.

mrun is stateless: Codex resends the conversation each turn and the prefix cache makes that cheap.
Requests that depend on server-stored state (`previous_response_id`) are rejected with a clear
error.

## OpenCode

`launch` passes the provider through `OPENCODE_CONFIG_CONTENT`: an OpenAI-compatible provider at
`http://127.0.0.1:8484/v1` with the model's context, input and output limits.

## Other clients

Any tool that speaks OpenAI Chat Completions works — Aider, Continue, Cline, Zed and others. Set
the base URL to `http://127.0.0.1:8484/v1` and any API key value; unknown model names are served
by the loaded model and responses report its real id. Set the client's context limit to the value
`mrun ps` shows; most clients otherwise assume far more than fits on a laptop.
