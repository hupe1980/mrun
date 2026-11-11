+++
title = "MCP and skills"
description = "MCP servers, deferred tool loading and skills in your agents, and tools in mrun's own chats."
weight = 25
+++

MCP servers and skills run inside your agent, not in mrun's server. The agent turns them into
tool definitions and messages, and mrun serves those like any others. What you configure in
Claude Code, Codex or OpenCode keeps working. mrun's own chats can use tools too; see
[Tools in mrun's chats](#tools-in-mrun-s-chats).

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

## Tools in mrun's chats

The window's Chat and `mrun run` are MCP clients. Their servers are in `~/.mrun/mcp.json`, in the
`mcpServers` format other clients use (`command`, `args`, `env`, `cwd`, or `url`, `headers`):

```sh
mrun mcp add workspace          # mrun's own tools in this folder
mrun mcp add fetch              # web pages
mrun mcp import claude-code     # or claude-desktop: their servers, names you have are kept
mrun mcp ls                     # each server with its tools, sandbox and errors
mrun run qwen3.6 --mcp workspace
```

In the window, the tools button in the message field turns servers on for one chat and shows
how many prompt tokens their definitions add; Settings › Tools adds, imports and lists them.

<figure class="screenshot"><a href="../../screenshots/tools.png" title="Open full size"><img src="../../screenshots/tools.png" alt="Chat: the model asks to run cargo test in the workspace; Deny, Always Allow and Allow" width="1100" height="720" loading="lazy" decoding="async"></a><figcaption>Every call waits for you: Allow (<kbd>Return</kbd>), Always Allow in this chat (<kbd>⌥Return</kbd>) or Deny (<kbd>Esc</kbd>)</figcaption></figure>

**Each call asks first.** The card says what will happen: the command, the file and its diff, or
the host a call reaches. Deny sends the model an error result. A reply may call tools up to 20
times; then it stops and says so. In `mrun run` the question is `[y]es / [a]lways / [n]o`.

**mrun's own tools.** `workspace` has `read`, `write`, `edit` (replaces one exact string),
`glob`, `grep` and `bash` (120 s, 30,000 characters of output), every path inside its folder.
`fetch` reads a page as text. Both are ordinary MCP servers (`mrun mcp serve workspace --root
DIR`, `mrun mcp serve fetch`), so any agent can use them.

**Sandboxed by default.** Local servers run in a [procage](https://github.com/hupe1980/procage)
sandbox (Seatbelt): their folder (`cwd`, else your home) read-only, secrets and startup files
denied, network only to npm, PyPI and GitHub so `npx` and `uvx` servers can start. A `sandbox`
block widens it:

```json
{"mcpServers": {"notes": {"command": "npx", "args": ["-y", "notes-mcp"],
  "sandbox": {"workspace": "~/notes", "write": true, "hosts": ["api.notes.app"]}}}}
```

When the sandbox stops a call, the card shows what was denied and *Allow …*, which adds it to
the server's block (`mrun mcp allow notes --write ~/notes/x` or `--host api.notes.app`).
`"sandbox": false` runs a server unconfined; agents run all MCP servers that way.

**Remote servers.** A `url` server receives your calls; its host shows on the card and on the
privacy line while a chat uses it. Env values and headers stay in `mcp.json` (mode 0600) and
never appear in the window, logs or `mrun mcp ls`.

## What mrun refuses

Anthropic's `mcp_servers` and `container` fields and server-executed tools ask the server to run
tools. mrun's server executes nothing, so these requests are refused with an error naming the
field.
