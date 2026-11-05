+++
title = "MCP and skills"
description = "How MCP servers, deferred tool loading and skills in Claude Code, Codex and OpenCode work against mrun."
weight = 25
+++

MCP servers and skills run inside your agent, not in mrun. The agent turns them into tool
definitions and messages, and mrun serves those like any others. What you configure in Claude
Code, Codex or OpenCode keeps working.

## MCP tools

| Client | How MCP tools arrive |
| --- | --- |
| Claude Code | Plain tools named `mcp__<server>__<tool>` |
| Codex | A *namespace* per server; mrun flattens them to `<namespace>__<tool>` for the model and restores the namespace on the calls it returns |
| OpenCode | Tools named `<server>_<tool>` |

A few MCP servers can add tens of thousands of tokens to every prompt. The Qwen chat templates
put tool definitions in the system prompt, so the first request after a server connects or
changes its tools is processed from the start (miss cause `new`, divergence in message 0, role
`system`). Later turns are cached as usual.

## Deferred tool loading

With many MCP tools, let the agent load definitions on demand. The agent marks most tools
`defer_loading`; mrun leaves them out of the prompt, and when the agent's tool search returns a
tool (`tool_reference`), mrun inserts its definition at that point in the conversation. The prompt
stays append-only, so nothing earlier is processed again.

Claude Code turns tool search off behind a custom base URL. `mrun launch claude-code` sets
`ENABLE_TOOL_SEARCH=auto:10`: MCP tools stay in the prompt while their definitions are small and
are deferred once they exceed 10 % of the context. Set `ENABLE_TOOL_SEARCH` in the client's
environment to change that.

## Skills

Skills are text the agent manages: a short catalog in the prompt and the full `SKILL.md` when one
is used. Nothing needs configuring on the server. A client that rewrites the system prompt when
its skill catalog changes causes one miss; changes appended to the conversation keep the cache.

## What mrun refuses

Anthropic's `mcp_servers` and `container` fields and server-executed tools ask the server to run
tools. mrun executes nothing, so these requests are refused with an error naming the field.
