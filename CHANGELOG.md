# Changelog

All notable changes to mrun are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/) with 0.x rules: minor versions may break.

## [Unreleased]

### Added

- `mrun serve`: Anthropic Messages (with `count_tokens`), OpenAI Chat Completions and OpenAI
  Responses (stateless) on `127.0.0.1:8484`, with streaming, tool calls, reasoning, keepalives,
  a bounded queue and cancellation on disconnect.
- Tool calls in each model family's own syntax (Hermes JSON, Qwen XML, Harmony) with strict tools,
  `tool_choice` and JSON output enforced by constrained decoding.
- MCP support: Codex namespaces, deferred tool loading and inline tool-search results.
- Prefix cache: exact reuse across turns, subagents and edits; moved branches; checkpoints for
  linear-attention models; a miss report with cause, position and message on every response.
- `mrun launch`, `connect`, `ps`, `doctor`, `stop`, `cache clear`, `completions`;
  `serve --detach`; `--version` with the commit and the vendored MLX versions.
- Idle-sleep prevention while requests run.
