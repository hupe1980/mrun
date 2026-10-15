# Feature Specification: Models and CLI Experience

**Feature Branch**: `004-models-cli`

**Created**: 2026-10-14

**Status**: Draft (revised 2026-10-15: open model architecture, D015; lean 0.1 cut, D025)

**Input**: User description: "Best-in-class developer experience: `mrun pull/ls/rm` with content-hash verified models and short aliases, `mrun connect <client>` that configures Claude Code, Codex and OpenCode, `mrun doctor` that diagnoses the machine and setup, `mrun ps` that shows the loaded model, memory and cache, one config file with sane defaults, shell completions; fresh Mac to working agent session in under 5 minutes."

## Clarifications

### Session 2026-10-14

- Q: Where are models stored? → A: In the shared Hugging Face hub cache (`HF_HOME` / `HF_HUB_CACHE`), so models downloaded by other MLX tools are reused; mrun keeps aliases and manifests in `~/.mrun` (`MRUN_HOME` overrides). (Maintainer decision.)
- Q: Where is the config file? → A: *Revised 2026-10-15 (lean cut):* none in 0.1. Defaults, `MRUN_*` environment variables and flags (later wins); `~/.mrun/config.toml` and `mrun config` are Later.
- Q: How do clients get configured? → A: Primary: `mrun launch <client>` sets everything for one process (Ollama-style), leaving global configs untouched. Secondary: `mrun connect <client>` prints the equivalent persistent configuration for the user to paste. *Writing it (`--write`, diff, backup, `--undo`, `--project`): Later (lean cut 2026-10-15).*

### Session 2026-10-15 (open model architecture, D015; lean 0.1 cut, D025)

- Q: Which models get aliases? → A: Three: `qwen3.6` (Qwen3.6-35B-A3B, default candidate), `qwen3-coder` (Qwen3-Coder-30B-A3B), `gpt-oss` (gpt-oss-20b). Qwen3-8B is an internal test reference without alias; `devstral` arrives with 0.2. Any other repo of a supported family can be pulled by repo id.
- Q: Which is the default? → A: Whichever alias wins the agent suite of `mrun bench` on the M4 Pro (005 T012a); `qwen3.6` until then. An alias that misses the client gate (002 SC-001) is dropped from the release; there are no tier labels.
- Q: What if a repo's architecture is unsupported? → A: `pull` reads `config.json` first and refuses before downloading weights, listing the supported families (001 FR-002).
- Q: What does `doctor` check in 0.1? → A: Chip and memory budget, low-power mode, port conflict, server reachability and client pitfalls. `doctor --agent`, model integrity, cache free space and `template-tools-late` are Later.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - From zero to a working agent in minutes (Priority: P1)

A developer installs mrun, runs `mrun launch claude-code`, accepts the offered download and
server start, and works, without reading documentation beyond the install line.

**Why this priority**: First-run experience decides adoption; competitors set the bar with
one-line pulls and ready-made client integration.

**Independent Test**: From an empty `~/.mrun` on a clean Mac account, run `mrun launch
claude-code` and count confirmations and commands until a working session (SC-001).

**Acceptance Scenarios**:

1. **Given** a fresh install, **When** the developer runs `mrun pull qwen3.6`, **Then** mrun
   shows size, license and fit, downloads, verifies and reports where it is stored.
2. **Given** a pulled model and a running server, **When** the developer runs
   `mrun launch claude-code` (or `codex`, `opencode`), **Then** the client starts already pointed
   at mrun with correct model names and context limits, without changing any global config.
3. **Given** the developer prefers persistent setup, **When** they run `mrun connect <client>`,
   **Then** mrun prints the configuration and where it belongs, and changes no file.
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
   detects the size or modification-time mismatch (or `mrun verify` the hash mismatch) and offers
   `mrun pull` to repair it.
4. **Given** a Hugging Face repo whose `model_type` no family supports, **When** the developer runs
   `mrun pull <repo>`, **Then** mrun refuses before downloading any weights, naming the
   architecture and listing the supported families and their aliases.

---

### User Story 3 - Diagnose problems myself (Priority: P2)

When something is slow or failing, `mrun doctor` and `mrun ps` tell the developer what is wrong:
memory budget, low-power mode, port conflicts, an unreachable server, cache hit rates and the last
miss, client setup mistakes such as a shell `ANTHROPIC_API_KEY`.

**Why this priority**: Local inference performance depends on the machine state; self-service
diagnosis avoids frustration and support load.

**Independent Test**: Introduce one misconfiguration per doctor check (low-power mode, port
taken, server stopped, old Claude Code, shell `ANTHROPIC_API_KEY`) and check that `mrun doctor`
reports each with a fix.

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
- `rm` of a model whose repository and revision another tool also uses (the HF cache records no
  owner): mrun warns that other tools lose that revision; blobs still linked by another revision
  of the same repository are kept.
- A model file edited in place with its size and modification time preserved: detected by
  `mrun verify`, not by fast checks.
- Offline machine: every command except `pull` works without network.
- A repo whose `config.json` is missing or unreadable: `pull` refuses before downloading weights.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: `mrun pull <alias|repo>` MUST download a model from the Hugging Face hub (honoring
  `HF_ENDPOINT` for mirrors, `HF_TOKEN` or else `$HF_HOME/token` for gated models, and
  `HF_HUB_OFFLINE=1` by failing fast with exit 6), show size, license and memory fit
  before downloading, resume interrupted downloads, and verify every file's content hash.
- **FR-002**: mrun MUST ship a curated alias list with pinned revisions and hashes, one per
  user-facing 0.1 family: `qwen3.6` → Qwen3.6-35B-A3B 4-bit (`qwen3_5_moe`, default candidate),
  `qwen3-coder` → Qwen3-Coder-30B-A3B-Instruct 4-bit (`qwen3_moe`), `gpt-oss` → gpt-oss-20b MXFP4
  (`gpt_oss`). The default alias (used by `launch` when nothing is pulled) is the winner of 005's
  agent-suite decision (005 T012a), `qwen3.6` until then. Each alias records its family, syntax
  and fit numbers and the agent-usable context on the user's Mac. *Removed (0.1 scope cut,
  2026-10-15): the `devstral` (0.2) and `qwen3-8b` (internal reference) aliases and the `tier: 2`
  demotion of an alias that misses a gate (such an alias is dropped from the release instead).*
- **FR-003**: `mrun ls`, `mrun rm`, `mrun show <model>` MUST manage and describe local models
  (those with a manifest), including disk size, fit in the current memory budget and context
  capacity (computed with the same budget function as feature 001, with the family's bytes per
  token). `ls` shows each model's family. `rm` deletes the pinned revision's snapshot and the blobs
  no other revision of that repository links to, removes `refs/*` only when they point to that
  revision, and warns that other tools using that revision lose it. *Removed (0.1 scope cut,
  2026-10-15): listing unpinned and unsupported repositories found in the HF cache.*
- **FR-004**: Models MUST be identified by content hash using feature 001's identity function,
  whether pulled or given as a path. Fast checks (size, modification time) run on every use;
  `mrun verify <model>` re-hashes every file; a mismatch is refused until repaired.
- **FR-005**: `mrun launch <claude-code|codex|opencode> [-- args]` MUST start the client with mrun
  configured for that process only, leaving the user's client files untouched. Values come from the
  running server's `/v1/models` (served id, context window, output limit). Per client (research R5):
  - Claude Code (≥ 2.1.273): base URL, auth token (a placeholder) with `ANTHROPIC_API_KEY` emptied
    so a shell key cannot override it; the served id in every model slot (`ANTHROPIC_MODEL`,
    `ANTHROPIC_DEFAULT_{OPUS,SONNET,HAIKU}_MODEL`, `CLAUDE_CODE_SUBAGENT_MODEL`);
    `CLAUDE_CODE_MAX_CONTEXT_TOKENS` = capacity and `CLAUDE_CODE_MAX_OUTPUT_TOKENS` = output limit
    (`CLAUDE_CODE_AUTO_COMPACT_WINDOW` only when capacity ≥ 100,000, its minimum);
    `CLAUDE_CODE_IDLE_COMPACT_MIN_TOKENS` above capacity, so Claude Code 2.1.299+ never compacts an
    idle conversation (mrun's cache has no expiry, D020); gateway hint headers on (session and
    tool-gap headers used by `--record`); nonessential traffic off; the per-turn token reminder
    off; `ENABLE_TOOL_SEARCH=auto:10` so large MCP tool sets load on demand (002 FR-028).
  - Codex: `-c` overrides selecting a `mrun` provider (`wire_api = "responses"`) and a model catalog
    file with the real context window and `supports_search_tool: true`, because Codex otherwise
    assumes 272,000 tokens for unknown models, never compacts in time and loads every MCP tool
    upfront; `-c web_search="disabled"`, because custom providers send a hosted `web_search` tool
    by default that mrun cannot run; conflicting user `-c` provider arguments are refused.
  - OpenCode: the whole provider config through `OPENCODE_CONFIG_CONTENT` (highest precedence), with
    `limit.context`, `limit.input` and `limit.output`. (OpenCode continues on its own after an
    output-limit stop, up to twice; nothing to set.)

  If no server is running, `launch` starts one (detached) after confirmation; if no model is pulled,
  it offers the default alias with its size, license and fit. `launch --model M` against a server
  serving another model asks to restart it, or exits 2 with the hint. `--dry-run` prints the
  environment and arguments instead of starting anything. A missing client binary is reported with
  its install command. Claude Code also gets `API_TIMEOUT_MS` raised to 50 minutes, since long
  local prefills exceed its default. `launch` warns when the context capacity is below 48K tokens
  (agents' system prompts and tool schemas alone take 15–25K), naming the larger-memory or
  smaller-model options.
- **FR-006**: `mrun connect <client>` MUST print the equivalent persistent configuration (Claude Code
  `env` block for `~/.claude/settings.json`; a Codex profile for `~/.codex/mrun.config.toml` used
  with `codex --profile mrun`; an OpenCode provider for `~/.config/opencode/opencode.json`) with
  the target path and the values of the running server, and change no file. *Removed (0.1 scope
  cut, 2026-10-15; see Later): `--write` with diff, 0600 backup and atomic write, `--undo`,
  `--project`, `apiKeyHelper` for keyed servers, and `doctor`'s stale-limit check.*
- **FR-007**: `mrun doctor` MUST check: chip, macOS and the memory budget (with current pressure),
  low-power mode, port conflicts, server reachability, and client pitfalls (Claude Code older than
  2.1.273, an `ANTHROPIC_API_KEY` in the environment overriding the gateway token, a Codex or
  OpenCode binary missing), and print fixes. *Removed (0.1 scope cut, 2026-10-15; see Later):
  `--agent <client>`, model integrity, cache free space, client-config file checks and
  `template-tools-late` (no 0.1 template renders tools late).*
- **FR-008**: `mrun ps` MUST show loaded model, budget breakdown, active and queued requests (with
  prompt-processing progress), cache size and hit rate, recording state and the most recent miss
  explanation (cause, divergence, message index and role, gap).
- **FR-009**: `mrun serve --detach` MUST run the server in the background with logs in
  `~/.mrun/server.log`, return once `/health` reports the model loaded, and exit 3 or 4 with the
  load error otherwise; `mrun stop` MUST stop it gracefully.
- **FR-010**: Configuration MUST work with zero files: defaults, then `MRUN_<FLAG>` environment
  variables (upper snake case of the flag name, e.g. `MRUN_PORT`), then flags (except the
  command-line-only `--record` and `--allow-over`).
  *Removed (0.1 scope cut, 2026-10-15; see Later): `~/.mrun/config.toml` and `mrun config`.*
- **FR-011**: Every error MUST say what failed and what to do next; exit codes MUST be documented
  and distinct for usage, model, fit, server, network and doctor failures (code 7 is reserved for
  the later bench regression mode).
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
  architecture (`model_type`), quantization, syntax, fit numbers (state bytes per token,
  checkpoint bytes, capacity on 48 GB).
- **Client Profile**: Client name, launch environment and arguments, printed persistent form.
- **Diagnostic**: Check, status, severity, fix.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: From an empty `~/.mrun`, `mrun launch claude-code` reaches a working session with ≤ 2
  confirmations and no other command.
- **SC-002**: 100% of injected doctor scenarios (at least one per FR-007 check) are detected with
  the correct fix.
- **SC-003**: Interrupted downloads resume without re-downloading completed bytes in 100% of tests;
  corruption that changes size or modification time is detected on every use, and any corruption
  is detected by `mrun verify`.
- **SC-004**: Every alias meets the client gate (002 SC-001) through `mrun launch`. Covered by
  002 SC-001 (no separate gate).
- **SC-005**: A snapshot test lists every error in the catalog, and each has a next step.
- **SC-006**: For 100% of unsupported-architecture fixtures (mock hub), `pull` exits 3 having
  downloaded no weight bytes, and the message lists every supported family; every curated alias
  resolves to a repo whose family loads (one pull per alias on the target Mac).

## Assumptions

- Models are distributed in the community MLX safetensors layout on Hugging Face; mrun does not
  host models.
- Install via a release binary or Homebrew is handled by the distribution milestone; this feature
  covers the CLI after installation.
- Gated models requiring a Hugging Face token use the standard token environment variable.

## Later

Moved out of 0.1 on 2026-10-15 (D025); IDs kept: `connect --write/--undo/--project` with diff and
backups and `apiKeyHelper` (FR-006); `doctor --agent`, model-integrity, free-space, client-file and
`template-tools-late` checks (FR-007); `~/.mrun/config.toml` and `mrun config` (FR-010); listing
unpinned and unsupported cached repositories (FR-003); the inode fast check (FR-004); `tier`
labels (FR-002); the `devstral` alias (0.2). The config file returns no later than the first
post-0.1 feature that defines keys for it (006 `[speculation]`, 007 `[cache] persist`, 009
`vision`).
