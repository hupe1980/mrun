+++
title = "Clients"
description = "Connect Claude Code, Codex CLI, OpenCode and any OpenAI-compatible coding agent to a local mrun server."
weight = 20
+++

Every client can be started with `mrun launch <client>`, which sets the configuration for that one
process. `mrun connect <client>` prints the equivalent persistent configuration;
`--write` applies it (showing a diff and keeping a backup) and `--undo` restores the backup.

| Client | Protocol | `launch` | `connect --write` edits |
| --- | --- | --- | --- |
| Claude Code | Anthropic Messages | `mrun launch claude-code` | `~/.claude/settings.json` |
| Codex CLI | OpenAI Responses | `mrun launch codex` | `~/.codex/config.toml` |
| OpenCode | Chat Completions | `mrun launch opencode` | `opencode.json` |

## Claude Code

`mrun launch claude-code` sets:

| Variable | Value | Why |
| --- | --- | --- |
| `ANTHROPIC_BASE_URL` | `http://127.0.0.1:8484` | Send requests to mrun |
| `ANTHROPIC_AUTH_TOKEN` | API key reference, or a placeholder | Claude Code requires one |
| `ANTHROPIC_MODEL`, `ANTHROPIC_DEFAULT_{OPUS,SONNET,HAIKU}_MODEL`, `CLAUDE_CODE_SUBAGENT_MODEL` | The served model | Every model slot, including subagents, uses the local model |
| `CLAUDE_CODE_MAX_CONTEXT_TOKENS`, `CLAUDE_CODE_AUTO_COMPACT_WINDOW` | This Mac's context capacity | Claude Code compacts before the server would refuse |
| `CLAUDE_CODE_ATTRIBUTION_HEADER` | `0` | Removes a per-request header line that would change the prompt |
| `CLAUDE_CODE_TOTAL_TOKENS_REMINDER` | `off` | Removes a per-turn token count that would break prefix reuse |
| `CLAUDE_CODE_GATEWAY_HINT_HEADERS` | `1` | Lets mrun see session and turn boundaries |

The last three matter for speed: any text that changes on every turn early in the prompt forces
the model to re-read everything after it. See [Prefix cache](@/docs/prefix-cache.md).

## Codex

Codex speaks the OpenAI Responses API. mrun adds a provider named `mrun` with
`wire_api = "responses"` and `base_url = "http://127.0.0.1:8484/v1"`. mrun is stateless: Codex
resends the conversation each turn and the prefix cache makes that cheap. Requests that depend on
server-stored state (`previous_response_id`) are rejected with a clear error.

## OpenCode

OpenCode uses Chat Completions through an OpenAI-compatible provider pointing at
`http://127.0.0.1:8484/v1`, with the served model and its context limit.

## Other clients

Any tool that speaks OpenAI Chat Completions works — Aider, Continue, Cline, Zed and others. Set
the base URL to `http://127.0.0.1:8484/v1` and any API key value; unknown model names are served
by the loaded model and responses report its real id.
