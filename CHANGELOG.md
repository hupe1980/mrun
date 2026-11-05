# Changelog

All notable changes to mrun are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/) with 0.x rules: minor versions may break.

## [Unreleased]

## [0.1.0] - 2026-11-05

The first release.

### Added

- **Server** (`mrun serve`): Anthropic Messages (with `count_tokens`), OpenAI Chat Completions and
  OpenAI Responses (stateless) on `127.0.0.1:8484`: streaming, tool calls, reasoning, structured
  output, keepalives, a bounded queue and cancellation on disconnect. Chat Completions accepts
  `chat_template_kwargs.enable_thinking`, as vLLM and SGLang do.
- **Models**: `qwen3.6` (Qwen3.6-35B-A3B 4-bit), the default where it leaves 64K tokens of
  context, and `qwen3.5` (Qwen3.5-9B 4-bit, a preview) on smaller Macs; Qwen3-8B and the
  Qwen3-Coder family by repo id. Affine 4- and 8-bit quantization. Greedy output matches mlx-lm
  token for token.
- **Engine** on nuvec, a pure-Rust port of MLX: Metal kernels built into the binary, prefill in
  interruptible chunks, pipelined decode, cancellation after every layer.
- **Exact prefix cache** across turns, subagents and side requests, with the model's own reply
  tokens spliced back so tool loops hit. Every response reports cached and processed tokens and,
  on a miss, its cause (`new`, `diverged`, `evicted`, `gap`, `disabled`) and the message where the
  prompt changed.
- **Memory budget** fixed at startup: `min(Metal working set, physical − 2 GiB, available −
  margin)`, covering weights, state, cache and scratch. Requests that cannot fit are refused with
  the numbers; memory pressure sheds cache first.
- **Model store** in `~/.mrun/models`: `pull` shows size, license and fit, resumes downloads,
  verifies every file against the Hub and lays weights out to be mapped from disk. A model the
  Hugging Face cache already holds is imported instead of downloaded; mrun never writes to that
  cache. `--dry-run` and `--json`.
- **CLI**: `launch` (pulls this Mac's default model and starts the server), `connect` (with
  `--write` and `--restore`), `run`, `ls`, `show`, `rm`, `verify`, `ps [--watch]`, `logs`,
  `stop`, `cache clear`, `doctor [--report]`, `config`, `completions`; `--json` where scripts need
  it.
- **Clients**: `launch` and `connect` set up Claude Code, Codex and OpenCode (OpenCode's titles
  run without thinking), and print settings for Crush and Zed.
- **The mrun window** (`mrun ui`): a menu bar item and one window. Now (budget, live requests, a
  reuse ribbon per request), Chat (files, reuse and speed per reply, thinking, export), Models
  (fit verdicts, download, use, remove), Agents (connect in one click), Settings. Every action is
  an `mrun` command (Copy as Command).
- **Remote GPU** (opt-in): `mrun cloud up|down|status|gpus|models|logs` rents a RunPod GPU, runs
  vLLM there and relays it on `127.0.0.1`, so `launch` and `connect` work unchanged. Prompts and
  code are sent to RunPod; the pod stops itself when idle. In the window: Models → On a rented
  GPU, Now's cost card, the spend in the menu bar; `mrun cloud login` keeps the key in the
  Keychain.
- **Config file** `~/.mrun/config.toml`, a table per command; `mrun config show` prints every
  setting and its source.
- **Observability**: `/mrun/status`, `/metrics` (Prometheus) and `mrun ps --watch`.
- **`mrun bench`**: replays agent sessions against mrun and other local servers with their
  caches on, with paired comparisons and a truncation canary.
- One release archive for macOS 15 and later, a Homebrew formula, and three crates on
  crates.io.

[Unreleased]: https://github.com/hupe1980/mrun/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/hupe1980/mrun/releases/tag/v0.1.0
