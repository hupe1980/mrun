# Feature Specification: Models and CLI Experience

**Feature Branch**: `004-models-cli`

**Created**: 2026-10-14

**Status**: Draft

**Input**: User description: "Best-in-class developer experience: `mrun pull/ls/rm` with content-hash verified models and short aliases, `mrun connect <client>` that configures Claude Code, Codex and OpenCode, `mrun doctor` that diagnoses the machine and setup, `mrun ps` that shows the loaded model, memory and cache, one config file with sane defaults, shell completions; fresh Mac to working agent session in under 5 minutes."

## Clarifications

### Session 2026-10-14

- Q: Where are models stored? → A: In the shared Hugging Face hub cache (`HF_HOME` / `HF_HUB_CACHE`), so models downloaded by other MLX tools are reused; mrun keeps aliases, manifests and config in `~/.mrun` (`MRUN_HOME` overrides). (Maintainer decision.)
- Q: Where is the config file? → A: `~/.mrun/config.toml`; environment variables `MRUN_*` override it; flags override both.
- Q: How do clients get configured? → A: Primary: `mrun launch <client>` sets everything for one process (Ollama-style), leaving global configs untouched. Secondary: `mrun connect` prints or (with `--write`, diff, backup, `--undo`) writes persistent config: Claude Code `~/.claude/settings.json` env, Codex `~/.codex/config.toml` provider with `wire_api = "responses"`, OpenCode `opencode.json`. Revised after analysis on 2026-10-14.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - From zero to a working agent in minutes (Priority: P1)

A developer installs mrun, pulls the recommended coding model, connects their agent and starts
working, without reading documentation beyond the install line.

**Why this priority**: First-run experience decides adoption; competitors set the bar with
one-line pulls and ready-made client integration.

**Independent Test**: On a clean Mac account, time the steps from install to a completed agent
task using only the commands mrun suggests.

**Acceptance Scenarios**:

1. **Given** a fresh install, **When** the developer runs `mrun pull qwen3-coder`, **Then** mrun
   shows size, license and progress, verifies the download and reports where it is stored.
2. **Given** a pulled model and a running server, **When** the developer runs
   `mrun launch claude-code` (or `codex`, `opencode`), **Then** the client starts already pointed
   at mrun with correct model names and context limits, without changing any global config.
3. **Given** the developer prefers persistent setup, **When** they run `mrun connect <client>
   --write`, **Then** mrun shows the diff, backs up the old file, applies it, and `--undo`
   restores it.
4. **Given** mrun is not running, **When** the developer runs `mrun launch` or `connect`, **Then**
   it explains how to start the server (`mrun serve --detach`).

---

### User Story 2 - Manage models confidently (Priority: P1)

The developer lists, removes and inspects models, using short aliases or full names, and always
knows what is on disk and how much it fits in memory.

**Why this priority**: Models are tens of gigabytes; users need clear control over disk and memory.

**Independent Test**: Pull two models, list them with sizes and fit, remove one, and verify disk
usage and listings.

**Acceptance Scenarios**:

1. **Given** pulled models, **When** the developer runs `mrun ls`, **Then** each shows alias, full
   name, size on disk, whether it fits this Mac's budget and maximum context that fits.
2. **Given** a model alias, **When** the developer runs `mrun rm <alias>`, **Then** its files are
   removed after confirmation and no other model is affected.
3. **Given** a corrupted or partially downloaded model, **When** any command uses it, **Then** mrun
   detects the hash mismatch and offers `mrun pull` to repair it.

---

### User Story 3 - Diagnose problems myself (Priority: P2)

When something is slow or failing, `mrun doctor` and `mrun ps` tell the developer what is wrong:
memory budget, other memory-hungry processes, low-power mode, cache hit rates, client config
mistakes such as volatile prompt headers.

**Why this priority**: Local inference performance depends on the machine state; self-service
diagnosis avoids frustration and support load.

**Independent Test**: Introduce known misconfigurations (low-power mode, too-large model, client
configuration with a shell `ANTHROPIC_API_KEY` set) and check that `mrun doctor` reports each with a fix.

**Acceptance Scenarios**:

1. **Given** a running server, **When** the developer runs `mrun ps`, **Then** it shows the loaded
   model, memory budget breakdown, active request, queue length and cache usage and hit rate.
2. **Given** a known misconfiguration, **When** the developer runs `mrun doctor`, **Then** each
   problem is listed with severity and a concrete fix.

### Edge Cases

- Network fails mid-download: the next `mrun pull` resumes and verifies.
- Disk full during pull: stops with the needed and available space; partial files are cleaned.
- Alias collision with a local path: explicit paths always win; ambiguity is reported.
- A model that does not fit the budget: `pull` warns before downloading and requires confirmation.
- `connect --write` on a client config edited by the user: mrun shows a diff and keeps a 0600 backup.
- `rm` of a model whose blobs are also used by another tool's snapshot: shared blobs are kept and
  the user is told.
- A model file edited in place with its size and modification time preserved: detected by
  `mrun verify`, not by fast checks.
- Offline machine: every command except `pull` works without network.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: `mrun pull <alias|repo>` MUST download a model from the Hugging Face hub (honoring
  `HF_ENDPOINT` for mirrors and `HF_TOKEN` for gated models), show size, license and memory fit
  before downloading, resume interrupted downloads, and verify every file's content hash.
- **FR-002**: mrun MUST ship a curated alias list (for example `qwen3-coder` →
  Qwen3-Coder-30B-A3B-Instruct 4-bit, `qwen3-8b` → Qwen3-8B 4-bit) with pinned revisions and hashes.
- **FR-003**: `mrun ls`, `mrun rm`, `mrun show <model>` MUST manage and describe local models,
  including disk size, fit in the current memory budget and context capacity (computed with the
  same budget function as feature 001). `rm` MUST remove only what mrun downloaded (snapshot refs
  and blobs no other snapshot references) and warn before touching anything shared.
- **FR-004**: Models MUST be identified by content hash using feature 001's identity function,
  whether pulled or given as a path. Fast checks (size, modification time) run on every use;
  `mrun verify <model>` re-hashes every file; a mismatch is refused until repaired.
- **FR-005**: `mrun launch <claude-code|codex|opencode> [-- args]` MUST start the client with mrun
  configured for that process only, leaving the user's client files untouched. Values come from the
  running server's `/v1/models` (served id, context window, output limit). Per client (research R5):
  - Claude Code (≥ 2.1.273): base URL, auth token (key by environment reference, otherwise a
    placeholder) with `ANTHROPIC_API_KEY` emptied so a shell key cannot override it; the served id
    in every model slot (`ANTHROPIC_MODEL`, `ANTHROPIC_DEFAULT_{OPUS,SONNET,HAIKU}_MODEL`,
    `CLAUDE_CODE_SUBAGENT_MODEL`); `CLAUDE_CODE_MAX_CONTEXT_TOKENS` = capacity and
    `CLAUDE_CODE_MAX_OUTPUT_TOKENS` = output limit (`CLAUDE_CODE_AUTO_COMPACT_WINDOW` only when
    capacity ≥ 100,000, its minimum); gateway hint headers on; nonessential traffic off; the
    per-turn token reminder off.
  - Codex: `-c` overrides selecting a `mrun` provider (`wire_api = "responses"`) and a model catalog
    file with the real context window, because Codex otherwise assumes 272,000 tokens for unknown
    models and never compacts in time; conflicting user `-c` provider arguments are refused.
  - OpenCode: the whole provider config through `OPENCODE_CONFIG_CONTENT` (highest precedence), with
    `limit.context`, `limit.input` and `limit.output`.

  If no server is running, `launch` starts one (detached) after confirmation; if no model is pulled,
  it offers the default alias with its fit. `--dry-run` prints the environment and arguments instead
  of starting anything. A missing client binary is reported with its install command. Claude Code
  also gets `API_TIMEOUT_MS` raised to 50 minutes, since long local prefills exceed its default.
  `launch` warns when the context capacity is below 48K tokens (agents' system prompts and tool
  schemas alone take 15–25K), naming the larger-memory or smaller-model options.
- **FR-006**: `mrun connect <client>` MUST print the equivalent persistent configuration; with
  `--write` it MUST show a diff, back up the original (mode 0600), write atomically, and
  `mrun connect <client> --undo` MUST restore the backup. Targets: Claude Code
  `~/.claude/settings.json` (`env`), Codex a separate profile `~/.codex/mrun.config.toml` used with
  `codex --profile mrun` (root `config.toml` untouched), OpenCode
  `~/.config/opencode/opencode.json`.
- **FR-007**: `mrun doctor` MUST check: macOS and chip, memory budget and current pressure,
  low-power mode, model integrity, server reachability, port conflicts, free space in the model
  cache, client configuration pitfalls (wrong base URL or wire API, missing context limits, Claude
  Code older than 2.1.273, an `ANTHROPIC_API_KEY` in the environment overriding the gateway token),
  and print fixes. `mrun doctor --agent <client>` additionally runs one tool-call round trip and one
  compaction request against the server and reports whether the client can work with it.
- **FR-008**: `mrun ps` MUST show loaded model, budget breakdown, active and queued requests (with
  prompt-processing progress), cache size and hit rate, recording state and the most recent miss
  explanation.
- **FR-009**: `mrun serve --detach` MUST run the server in the background with logs in `~/.mrun/`,
  and `mrun stop` MUST stop it gracefully.
- **FR-010**: Configuration MUST work with zero files; an optional `~/.mrun/config.toml` and
  `MRUN_*` environment variables override defaults (except settings that are command-line only,
  such as `--record`); `mrun config` prints the effective configuration and each value's source.
- **FR-011**: Every error MUST say what failed and what to do next; exit codes MUST be documented
  and distinct for usage, model, fit, server, network, benchmark-regression and doctor failures.
- **FR-012**: mrun MUST provide shell completions (bash, zsh, fish) and `--help` with examples for
  every command; this CLI contract is the single source for all commands and flags.
- **FR-013**: Only `pull` may use the network; all other commands MUST work offline.

### Key Entities

- **Model Manifest**: Alias, repository, revision, files with hashes and sizes, license, architecture,
  quantization.
- **Client Profile**: Client name, configuration location, settings to write, cache-stability hints.
- **Diagnostic**: Check, status, severity, fix.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: On a clean Mac with a 1 Gbit/s connection, a developer reaches a completed agent task
  in under 5 minutes plus model download time, using only commands mrun suggests.
- **SC-002**: 100% of injected doctor scenarios (at least 10) are detected with the correct fix.
- **SC-003**: Interrupted downloads resume without re-downloading completed bytes in 100% of tests;
  corruption that changes size or modification time is detected on every use, and any corruption
  is detected by `mrun verify`.
- **SC-004**: `mrun launch` and `mrun connect --write` configurations work unmodified for each
  supported client version (pinned): the scripted read-edit-test task completes 5 of 5 runs per
  client through each mechanism.
- **SC-005**: Every CLI error in the error catalog includes a next step (review checklist, 100%).

## Assumptions

- Models are distributed in the community MLX safetensors layout on Hugging Face; mrun does not
  host models.
- Install via a release binary or Homebrew is handled by the distribution milestone; this feature
  covers the CLI after installation.
- Gated models requiring a Hugging Face token use the standard token environment variable.
