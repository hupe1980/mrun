+++
title = "MCP and skills"
description = "How MCP servers, deferred tool loading and skills configured in Claude Code, Codex and OpenCode work against a local mrun server, and how to keep them cache-friendly."
weight = 25
+++

MCP servers and skills run inside your agent, not in mrun. The agent turns them into tool
definitions and messages, and mrun serves those like any others. Everything you configure in
Claude Code, Codex or OpenCode keeps working; this page explains what reaches the model and how to
keep it fast.

## MCP tools

| Client | How MCP tools arrive |
| --- | --- |
| Claude Code | Plain tools named `mcp__<server>__<tool>` |
| Codex | A *namespace* per server, plus its built-in sub-agent tools; mrun flattens them to `<namespace>__<tool>` for the model and restores the namespace on the calls it returns |
| OpenCode | Tools named `<server>_<tool>`, by default behind a single code-mode tool |

Tool schemas are expensive: five typical MCP servers add about 55,000 tokens to every prompt. The
chat templates of the supported models put all tool definitions at the top of the prompt, so the
first request after a server connects or changes its tools is processed from the start (the miss
report says `edit`). After that, turns are cached as usual.

## Deferred tool loading

With many MCP tools, let the agent load definitions on demand. The agent then sends most tools
marked `defer_loading`; mrun leaves them out of the prompt, and when the agent's tool search
returns a tool, mrun inserts its definition at that point in the conversation. The prompt stays
append-only, so nothing earlier is processed again.

- **Claude Code** turns tool search off behind a custom base URL. `mrun launch claude-code` sets
  `ENABLE_TOOL_SEARCH=auto:10`, which defers MCP tools once their definitions exceed 10% of the
  context.
- **Codex** uses tool search when the model catalog allows it; `mrun launch codex` enables it.

## Skills

Skills are text the agent manages: a short catalog in the system prompt and the full `SKILL.md`
when one is used. Codex and OpenCode append catalog changes instead of rewriting the start of the
prompt, which keeps the cache intact. Nothing needs configuring on the server.

## What mrun refuses

Some API features ask the *server* to run tools: Anthropic's `mcp_servers`, `container` and
server-side tool search tools. mrun cannot execute anything, so these requests are refused with an
error naming the field, rather than letting the model believe it has tools that do not exist.
