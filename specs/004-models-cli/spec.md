# Feature Specification: Models and CLI Experience

**Feature Branch**: `004-models-cli`

**Created**: 2026-10-14

**Status**: Draft

**Input**: User description: "Best-in-class developer experience: `mrun pull/ls/rm` with content-hash verified models and short aliases, `mrun connect <client>` that configures Claude Code, Codex and OpenCode, `mrun doctor` that diagnoses the machine and setup, `mrun ps` that shows the loaded model, memory and cache, one config file with sane defaults, shell completions; fresh Mac to working agent session in under 5 minutes."

## Clarifications

### Session 2026-10-14

- Q: Where are models stored? → A: In the shared Hugging Face hub cache (`HF_HOME` / `HF_HUB_CACHE`), so models downloaded by other MLX tools are reused; mrun keeps aliases, manifests and config in `~/.mrun` (`MRUN_HOME` overrides). (Maintainer decision.)
- Q: Where is the config file? → A: `~/.mrun/config.toml`; environment variables `MRUN_*` override it; flags override both.
- Q: Which client config does `connect` write? → A: Claude Code: environment settings in `~/.claude/settings.json` (`ANTHROPIC_BASE_URL`, model, `CLAUDE_CODE_ATTRIBUTION_HEADER=0`); Codex: a provider profile in `~/.codex/config.toml` with `wire_api = "responses"`; OpenCode: a provider in `opencode.json`. Exact keys are verified against pinned client versions in the plan.

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
2. **Given** a pulled model, **When** the developer runs `mrun connect claude-code` (or `codex`,
   `opencode`), **Then** mrun prints the exact configuration and, with `--write`, applies it after
   showing what will change.
3. **Given** mrun is not running, **When** the developer runs `mrun connect`, **Then** it explains
   how to start the server (`mrun serve`).

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
without the attribution-header setting) and check that `mrun doctor` reports each with a fix.

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
- `connect --write` on a client config edited by the user: mrun shows a diff and keeps a backup.
- Offline machine: every command except `pull` works without network.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: `mrun pull <alias|repo>` MUST download a model from the Hugging Face hub (or a
  configured mirror), show size and license before downloading, resume interrupted downloads, and
  verify every file against pinned content hashes.
- **FR-002**: mrun MUST ship a curated alias list (for example `qwen3-coder` →
  Qwen3-Coder-30B-A3B-Instruct 4-bit, `qwen3-8b` → Qwen3-8B 4-bit) with pinned revisions and hashes.
- **FR-003**: `mrun ls`, `mrun rm`, `mrun show <model>` MUST manage and describe local models,
  including disk size, fit in the current memory budget and context capacity.
- **FR-004**: Models MUST be identified by content hash; a model whose files no longer match is
  refused until repaired.
- **FR-005**: `mrun connect <claude-code|codex|opencode>` MUST print correct client configuration
  for the running server, and with `--write` MUST apply it after showing a diff and saving a backup.
  For Claude Code it MUST include settings that keep prompts cache-stable (attribution header off).
- **FR-006**: `mrun doctor` MUST check: macOS and chip, memory budget and current pressure, low-power
  mode, model integrity, server reachability, client configuration pitfalls, and print fixes.
- **FR-007**: `mrun ps` MUST show loaded model, budget breakdown, active and queued requests, cache
  size and hit rate, and the most recent miss explanation.
- **FR-008**: Configuration MUST work with zero files; an optional config file (one location,
  documented) and environment variables override defaults; `mrun config` prints the effective
  configuration and where each value came from.
- **FR-009**: Every error MUST say what failed and what to do next; exit codes MUST be documented.
- **FR-010**: mrun MUST provide shell completions (bash, zsh, fish) and `--help` with examples for
  every command.
- **FR-011**: Only `pull` may use the network; all other commands MUST work offline.

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
  corrupted files are always detected.
- **SC-004**: `mrun connect` output works unmodified for each supported client version (pinned).
- **SC-005**: Every CLI error in the error catalog includes a next step (review checklist, 100%).

## Assumptions

- Models are distributed in the community MLX safetensors layout on Hugging Face; mrun does not
  host models.
- Install via a release binary or Homebrew is handled by the distribution milestone; this feature
  covers the CLI after installation.
- Gated models requiring a Hugging Face token use the standard token environment variable.
