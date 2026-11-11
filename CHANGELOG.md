# Changelog

All notable changes to mrun are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/) with 0.x rules: minor versions may break.

## [Unreleased]

## [0.4.3] - 2026-11-11

### Fixed

- A chat that was not kept on disk was gone after switching to another place and back; Chat now
  keeps its chats, draft and a streaming reply while you look elsewhere.

## [0.4.2] - 2026-11-10

### Fixed

- Eject, Delete All Chats, Remove key and Stop Pod did nothing in the native window: its webview
  has no `confirm()`. They now ask in a sheet of the window's own.
- Closing the window quits mrun and ejects the model it started, also after the window was
  restarted; a server started in a terminal keeps running.
- The loop guard caught neither a reply that mentioned a tool call while thinking nor a repeated
  paragraph: a call opener inside the reasoning is text now, and repeats of up to 256 tokens end
  the reply.

## [0.4.1] - 2026-11-10

### Added

- Eject in the window: beside the model chip on every place and on the model in use in Models,
  and Stop Serving in the menu bar menu (`mrun stop`).

### Changed

- Quit ejects the model the window started; a server started in a terminal keeps running.
  Without the menu bar item, closing the window quits.
- Settings › Server opens the server log in Console, which follows it live.

## [0.4.0] - 2026-11-10

### Added

- `--context` on `serve` and `run` (`MRUN_CONTEXT`): memory by context length. The budget becomes
  exactly what that many tokens need and the rest stays free; a context that does not fit is
  refused before loading (E042), never shortened. `mrun ls --context` and a per-model picker in
  Models show what each context uses.

### Fixed

- Settings: the Server tab opened nothing; it now lists the server's settings and where each
  comes from.

## [0.3.0] - 2026-11-10

### Added

- Speculative decoding: mrun drafts the next tokens from the prompt and the reply so far and
  checks up to 7 at once, keeping only what plain decoding would produce (bit-identical, greedy
  or seeded). It drafts only while measured faster on the request: 1.43× on a copy-heavy reply
  with `qwen3.6`, unchanged on prose. `--speculation auto|off`; tokens per pass in `mrun ps`,
  `/mrun/status`, `/metrics`, Now and each chat reply.

### Changed

- nuvec 0.8.0.

## [0.2.0] - 2026-11-06

### Added

- Sessions in the window: each coding-agent session as a skyline of its requests (main
  conversation, subagents, side requests), live, with every miss's cause and the text that
  changed. `mrun ps --sessions`, `GET /mrun/sessions` and `GET /mrun/requests/{id}/excerpt`.
- Mac pool (`mrun pool`, Pool in the window): Macs on one network run one model none of them
  fits alone, each a range of layers, with the same replies and reuse as one Mac.
  - Joining with a code both Macs show; a plan per Mac before anything loads.
  - Encrypted traffic between Macs; a host may choose unencrypted for a trusted link.
  - A lost Mac is named within 5 s and the pool serves again on the rest; Rejoin needs no code.
  - A member downloads only its layers.

### Changed

- nuvec 0.7.0.

## [0.1.1] - 2026-11-05

### Fixed

- `mrun` is on crates.io (`cargo install mrun`): 0.1.0 reached GitHub and Homebrew, but its
  crates.io upload failed. The crates are now published with Trusted Publishing.

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

[Unreleased]: https://github.com/hupe1980/mrun/compare/v0.4.3...HEAD
[0.4.3]: https://github.com/hupe1980/mrun/compare/v0.4.2...v0.4.3
[0.4.2]: https://github.com/hupe1980/mrun/compare/v0.4.1...v0.4.2
[0.4.1]: https://github.com/hupe1980/mrun/compare/v0.4.0...v0.4.1
[0.4.0]: https://github.com/hupe1980/mrun/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/hupe1980/mrun/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/hupe1980/mrun/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/hupe1980/mrun/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/hupe1980/mrun/releases/tag/v0.1.0
