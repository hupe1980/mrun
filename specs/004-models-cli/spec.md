# Feature Specification: Models and CLI Experience

**Feature Branch**: `004-models-cli`

**Created**: 2026-10-14

**Status**: Draft (revised 2026-10-15: open model architecture, D015)

**Input**: User description: "Best-in-class developer experience: `mrun pull/ls/rm` with content-hash verified models and short aliases, `mrun connect <client>` that configures Claude Code, Codex and OpenCode, `mrun doctor` that diagnoses the machine and setup, `mrun ps` that shows the loaded model, memory and cache, one config file with sane defaults, shell completions; fresh Mac to working agent session in under 5 minutes."

## Clarifications

### Session 2026-10-14

- Q: Where are models stored? → A: In the shared Hugging Face hub cache (`HF_HOME` / `HF_HUB_CACHE`), so models downloaded by other MLX tools are reused; mrun keeps aliases, manifests and config in `~/.mrun` (`MRUN_HOME` overrides). (Maintainer decision.)
- Q: Where is the config file? → A: `~/.mrun/config.toml`; environment variables `MRUN_*` override it; flags override both.
- Q: How do clients get configured? → A: Primary: `mrun launch <client>` sets everything for one process (Ollama-style), leaving global configs untouched. Secondary: `mrun connect` prints or (with `--write`, diff, backup, `--undo`) writes persistent config: Claude Code `~/.claude/settings.json` env (or `--project` for `.claude/settings.local.json`), Codex a separate profile `~/.codex/mrun.config.toml` with a model catalog file (root config untouched), OpenCode `~/.config/opencode/opencode.json`. Revised 2026-10-15 after the competitor study.

### Session 2026-10-15 (open model architecture, D015)

- Q: Which models get aliases? → A: One per release-0.1 family: `qwen3.6` (Qwen3.6-35B-A3B, default candidate), `qwen3-coder` (Qwen3-Coder-30B-A3B), `gpt-oss` (gpt-oss-20b), `devstral` (Devstral Small 2 24B), `qwen3-8b` (Qwen3-8B). Any other repo of a supported family can be pulled by repo id.
- Q: Which is the default? → A: Whichever 0.1 model wins the agent suite of `mrun bench` on the M4 Pro (005 T012a); `qwen3.6` until then.
- Q: What if a repo's architecture is unsupported? → A: `pull` reads `config.json` first and refuses before downloading weights, listing the supported families (001 FR-002).

## User Scenarios & Testing *(mandatory)*

### User Story 1 - From zero to a working agent in minutes (Priority: P1)

A developer installs mrun, pulls the recommended coding model, connects their agent and starts
working, without reading documentation beyond the install line.

**Why this priority**: First-run experience decides adoption; competitors set the bar with
one-line pulls and ready-made client integration.

**Independent Test**: On a clean Mac account, time the steps from install to a completed agent
task using only the commands mrun suggests.

**Acceptance Scenarios**:

1. **Given** a fresh install, **When** the developer runs `mrun pull qwen3.6`, **Then** mrun
   shows size, license and progress, verifies the download and reports where it is stored.
2. **Given** a pulled model and a running server, **When** the developer runs
   `mrun launch claude-code` (or `codex`, `opencode`), **Then** the client starts already pointed
   at mrun with correct model names and context limits, without changing any global config.
3. **Given** the developer prefers persistent setup, **When** they run `mrun connect <client>
   --write`, **Then** mrun shows the diff, backs up the old file, applies it, and `--undo`
   restores it.
4. **Given** mrun is not running, **When** the developer runs `mrun launch`, **Then** it offers to
   start the server in the background and waits for `/health`; `mrun connect` prints
   `mrun serve --detach`.

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
4. **Given** a Hugging Face repo whose `model_type` no family supports, **When** the developer runs
   `mrun pull <repo>`, **Then** mrun refuses before downloading any weights, naming the
   architecture and listing the supported families and their aliases.

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
- Disk full during pull: stops with the needed and available space; partial files are kept for
  resume and reported with their size.
- Alias collision with a local path: explicit paths always win; ambiguity is reported.
- A model that does not fit the budget: `pull` warns before downloading and requires confirmation.
- `connect --write` on a client config edited by the user: mrun shows a diff and keeps a 0600 backup.
- `rm` of a model whose repository and revision another tool also uses (the HF cache records no
  owner): mrun warns that other tools lose that revision; blobs still linked by another revision
  of the same repository are kept.
- A model file edited in place with its size and modification time preserved: detected by
  `mrun verify`, not by fast checks.
- Offline machine: every command except `pull` works without network.
- A repo whose `config.json` is missing or unreadable: `pull` refuses before downloading weights.
- An unpinned repo in the HF cache with an unsupported architecture: `ls` shows it with status
  `unsupported` and its `model_type`, never as runnable.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: `mrun pull <alias|repo>` MUST download a model from the Hugging Face hub (honoring
  `HF_ENDPOINT` for mirrors, `HF_TOKEN` or else `$HF_HOME/token` for gated models, and
  `HF_HUB_OFFLINE=1` by failing fast with exit 6), show size, license and memory fit
  before downloading, resume interrupted downloads, and verify every file's content hash.
- **FR-002**: mrun MUST ship a curated alias list with pinned revisions and hashes, one per
  release-0.1 family: `qwen3.6` → Qwen3.6-35B-A3B 4-bit (`qwen3_5_moe`, default candidate),
  `qwen3-coder` → Qwen3-Coder-30B-A3B-Instruct 4-bit (`qwen3_moe`), `gpt-oss` → gpt-oss-20b MXFP4
  (`gpt_oss`), `devstral` → Devstral Small 2 24B 4-bit (`mistral`), `qwen3-8b` → Qwen3-8B 4-bit
  (`qwen3`). The default alias (used by `launch` when nothing is pulled) is the winner of 005's
  agent-suite decision (005 T012a), `qwen3.6` until then. Each alias records its family, syntax,
  tier and fit numbers and the agent-usable context on the user's Mac. A 0.1 alias whose model
  fails a Tier 1 gate (incl. 010 SC-002, 5 of 5 client runs) ships with `tier: 2` in its
  manifest, is labelled as Tier 2 by `mrun show` and `mrun ls`, and cannot be the default
  (010 FR-016).
- **FR-003**: `mrun ls`, `mrun rm`, `mrun show <model>` MUST manage and describe local models,
  including disk size, fit in the current memory budget and context capacity (computed with the
  same budget function as feature 001, with the family's bytes per token). `ls` shows each model's
  family. `ls` also lists MLX repositories of supported families already in the HF cache (status
  `unpinned`; `mrun pull <repo>` pins them without downloading) and repositories of other
  architectures as `unsupported`. `rm` deletes the
  pinned revision's snapshot and the blobs no other revision of that repository links to, removes
  `refs/*` only when they point to that revision, and warns that other tools using that revision
  lose it.
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
    capacity ≥ 100,000, its minimum); gateway hint headers on (session headers used by
    `--record`); nonessential traffic off; the per-turn token reminder off;
    `ENABLE_TOOL_SEARCH=auto:10` so large MCP tool sets load on demand (002 FR-028).
  - Codex: `-c` overrides selecting a `mrun` provider (`wire_api = "responses"`) and a model catalog
    file with the real context window and `supports_search_tool: true`, because Codex otherwise
    assumes 272,000 tokens for unknown models, never compacts in time and loads every MCP tool
    upfront; conflicting user `-c` provider arguments are refused.
  - OpenCode: the whole provider config through `OPENCODE_CONFIG_CONTENT` (highest precedence), with
    `limit.context`, `limit.input` and `limit.output`.

  If no server is running, `launch` starts one (detached) after confirmation; if no model is pulled,
  it offers the default alias with its size, license and fit. `launch --model M` against a server
  serving another model asks to restart it, or exits 2 with the hint. `--dry-run` prints the environment and arguments instead
  of starting anything. A missing client binary is reported with its install command. Claude Code
  also gets `API_TIMEOUT_MS` raised to 50 minutes, since long local prefills exceed its default.
  `launch` warns when the context capacity is below 48K tokens (agents' system prompts and tool
  schemas alone take 15–25K), naming the larger-memory or smaller-model options.
- **FR-006**: `mrun connect <client>` MUST print the equivalent persistent configuration; with
  `--write` it MUST show a diff, back up the original (mode 0600), write atomically, and
  `mrun connect <client> --undo` MUST restore the backup. Targets: Claude Code
  `~/.claude/settings.json` (`env`; `--project` writes `.claude/settings.local.json` instead, and a
  warning states that the global file redirects all Claude Code use; a keyed server uses
  `apiKeyHelper` reading `MRUN_API_KEY`, never a literal key), Codex a separate profile
  `~/.codex/mrun.config.toml` used with `codex --profile mrun` (root `config.toml` untouched;
  `env_key` only when the server has a key), OpenCode `~/.config/opencode/opencode.json`. Written
  limits reflect the server at write time; `doctor` flags them when they differ from the running
  server's `/v1/models`, and re-running `connect --write` refreshes them.
- **FR-007**: `mrun doctor` MUST check: macOS and chip, memory budget and current pressure,
  low-power mode, model integrity, server reachability, port conflicts, free space in the model
  cache, client configuration pitfalls (wrong base URL or wire API, missing context limits, Claude
  Code older than 2.1.273, an `ANTHROPIC_API_KEY` in the environment overriding the gateway token),
  a template that renders tools after the first user message (`template-tools-late`, 010 FR-014),
  and print fixes. `mrun doctor --agent <client>` additionally runs one tool-call round trip and a request at 95% of capacity
  against the server and reports whether the client can work with it.
- **FR-008**: `mrun ps` MUST show loaded model, budget breakdown, active and queued requests (with
  prompt-processing progress), cache size and hit rate, recording state, the most recent miss
  explanation, and the `template-tools-late` warning when the probe set it (010 FR-014).
- **FR-009**: `mrun serve --detach` MUST run the server in the background with logs in
  `~/.mrun/server.log`, return once `/health` reports the model loaded, and exit 3 or 4 with the
  load error otherwise; `mrun stop` MUST stop it gracefully.
- **FR-010**: Configuration MUST work with zero files; an optional `~/.mrun/config.toml` and
  `MRUN_*` environment variables override defaults (except settings that are command-line only,
  such as `--record`); `mrun config` prints the effective configuration and each value's source.
- **FR-011**: Every error MUST say what failed and what to do next; exit codes MUST be documented
  and distinct for usage, model, fit, server, network, benchmark-regression and doctor failures.
- **FR-012**: mrun MUST provide shell completions (bash, zsh, fish) and `--help` with examples for
  every command; this CLI contract is the single source for all commands and flags.
- **FR-013**: Only `pull` (or a pull the user confirms inside `launch`) may use the network; all
  other commands MUST work offline.
- **FR-014**: `mrun pull` MUST fetch `config.json` (and `text_config` for wrapped configs) before
  any weight file and resolve its family with 001's family registry; an unsupported `model_type`
  or quantization mode MUST be refused before downloading weights (exit 3), naming it and listing
  the supported families with their aliases. The size/fit preview uses the family's bytes per
  token and position-bound bytes.

### Key Entities

- **Model Manifest**: Alias, repository, revision, files with hashes and sizes, license, family,
  architecture (`model_type`), quantization, tier, syntax, fit numbers (state bytes per token,
  checkpoint bytes, capacity on 48 GB).
- **Client Profile**: Client name, configuration location, settings to write, cache-stability hints.
- **Diagnostic**: Check, status, severity, fix.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Excluding installation and the model download, a developer reaches a completed agent
  task within 5 minutes of hands-on time, using only commands mrun suggests.
- **SC-002**: 100% of injected doctor scenarios (at least 10) are detected with the correct fix.
- **SC-003**: Interrupted downloads resume without re-downloading completed bytes in 100% of tests;
  corruption that changes size or modification time is detected on every use, and any corruption
  is detected by `mrun verify`.
- **SC-004**: `mrun launch` and `mrun connect --write` configurations work unmodified for each
  supported client version (pinned): the scripted read-edit-test task completes 5 of 5 runs per
  client through each mechanism.
- **SC-005**: Every CLI error in the error catalog includes a next step (review checklist, 100%).
- **SC-006**: For 100% of unsupported-architecture fixtures (mock hub), `pull` exits 3 having
  downloaded no weight bytes, and the message lists every supported family; every curated alias
  resolves to a repo whose family loads (one pull per alias on the target Mac).

## Assumptions

- Models are distributed in the community MLX safetensors layout on Hugging Face; mrun does not
  host models.
- Install via a release binary or Homebrew is handled by the distribution milestone; this feature
  covers the CLI after installation.
- Gated models requiring a Hugging Face token use the standard token environment variable.
