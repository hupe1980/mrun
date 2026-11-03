+++
title = "MCP and skills"
description = "How MCP servers, deferred tool loading and skills configured in Claude Code, Codex and OpenCode work against a local mrun server, and how to keep them cache-friendly."
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

Tool schemas are expensive: a few MCP servers can add tens of thousands of tokens to every
prompt. The Qwen chat templates put tool definitions into the system prompt at the start, so the
first request after a server connects or changes its tools is processed from the start (the miss
report says `new`, with the divergence in message 0, role `system`); after that, turns are cached
as usual.

## Deferred tool loading

With many MCP tools, let the agent load definitions on demand. The agent then marks most tools
`defer_loading`; mrun leaves them out of the prompt, and when the agent's tool search returns a
tool (`tool_reference`), mrun inserts its definition at that point in the conversation. The prompt
stays append-only, so nothing earlier is processed again.

Claude Code turns tool search off behind a custom base URL and loads every MCP tool up front.
`mrun launch claude-code` sets `ENABLE_TOOL_SEARCH=auto:10`: MCP tools stay in the prompt (and in
the cached prefix) while their definitions are small, and are deferred only once they exceed 10 %
of the context. Set `ENABLE_TOOL_SEARCH` yourself in the client's environment to change that.

## Skills

Skills are text the agent manages: a short catalog in the prompt and the full `SKILL.md` when one
is used. Nothing needs configuring on the server. A client that rewrites the system prompt when
its skill catalog changes causes one miss from that point; changes appended to the conversation
keep the cache intact.

## What mrun refuses

Some API features ask the *server* to run tools: Anthropic's `mcp_servers` and `container` fields
and server-executed tools. mrun executes nothing, so these requests are refused with an error
naming the field, rather than letting the model believe it has tools that do not exist.
