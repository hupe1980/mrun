---
description: "Tasks for 004 Models and CLI Experience"
---

# Tasks: Models and CLI Experience

**Input**: [plan.md](plan.md), [spec.md](spec.md), [research.md](research.md), [data-model.md](data-model.md), [contracts/cli.md](contracts/cli.md)

**Scope**: Lean 0.1 (D025, 2026-10-15). Cut tasks keep their IDs and say so.

## Phase 1: Setup

- [X] T001 Add hf-hub 1.0, indicatif 0.18, sha2 0.11, toml 1.1 (manifests), clap_complete 4.6, wiremock/insta/tempfile (dev) to `crates/mrun/Cargo.toml` *Note*: no `hf-hub`/`indicatif`/`wiremock`: downloads use `reqwest` directly (Range resume, `.incomplete` blobs, the up-front `config.json` gate) and tests use an axum mock hub; `clap_complete`, `toml`, `sha1` added.
- [X] T002 [P] `crates/mrun/src/config.rs`: defaults < `MRUN_*` < flags (no config file, FR-010), command-line-only settings (`--record`, `--allow-over`), unknown `MRUN_*` names warned with suggestions; tests *Note*: flags carry their `MRUN_*` names (clap `env`); `config.rs` warns about unknown `MRUN_*` names with the closest known one.
- [X] T003 [P] `crates/mrun/src/errors.rs`: error catalog and exit codes 0–8 per contract (7 reserved) + snapshot test listing every error with its next step (FR-011, SC-005) *Note*: `errors.rs` (codes E001–E052, `CliError` with next step, `Exit` 0–8); catalog unit tests.

## Phase 2: Foundational

- [X] T004 `crates/mrun/src/hub/mod.rs`: resolve path → alias → repo id; HF cache resolution (`HF_HUB_CACHE`, `HF_HOME`, `HF_ENDPOINT`); identity via `mrun_engine::identity` (001) *Note*: `hub/mod.rs` (HF cache precedence, `HF_ENDPOINT`, `HF_TOKEN` or the token file, `HF_HUB_OFFLINE`; resolve path → manifest → cache; E031 with did-you-mean, E033 ambiguity).
- [ ] T005 [P] Create the workspace `xtask` crate (`cargo xtask`; 005 T009 adds `bench-trace` to it) with `refresh-aliases`; `hub/aliases.rs`: curated table (`qwen3.6` default, `qwen3-coder`) with pinned revisions, file hashes, family, syntax and fit numbers *Progress*: `hub/aliases.rs` holds `qwen3.6` (default), `qwen3-coder`, `gpt-oss` with a registry-consistency test; all three track `main`, marked `verified: false`, without file hashes or measured fit (the hub is unreachable from the build sandbox); the `xtask refresh-aliases` crate is still open.
- [ ] T005a [P] Alias `gpt-oss` (research R4): resolve the `[U]` repo (`-MXFP4-Q8` quantization block), pin revision and hashes, record family, syntax and fit numbers; default alias field set from 005 T012a. (`devstral`: 0.2; `qwen3-8b`: no alias, 0.1 scope cut, 2026-10-15) *Progress*: entry present, unverified (see T005).
- [X] T006 [P] `hub/verify.rs` + `cli/verify.rs`: fast checks (size, mtime) and full re-hash; test flipping a byte with preserved mtime is caught by `verify` only *Note*: `hub/verify.rs` + `cli/verify.rs`.

## Phase 3: US1 — Zero to working agent (P1) 🎯 MVP

- [X] T007 [US1] `hub/download.rs` + `cli/pull.rs`: size/license/fit preview, disk-space check, hf-hub download with resume, verification, manifest; wiremock tests (resume, corrupt, disk full, gated → `HF_TOKEN` hint, `HF_ENDPOINT` mirror) *Note*: `hub/download.rs` + `cli/pull.rs`; tests in `tests/hub_pull.rs` (resume, wrong hash, repair, disk full, gated, mirror, offline, fit confirmation). One line per file instead of progress bars.
- [X] T008 [US1] `cli/launch.rs`: process-scoped env/flags per client per FR-005 / research R5 (Claude Code env set incl. emptied `ANTHROPIC_API_KEY`, `CLAUDE_CODE_IDLE_COMPACT_MIN_TOKENS` above capacity and `ENABLE_TOOL_SEARCH=auto:10`; Codex `-c` overrides incl. `web_search="disabled"` + model catalog file with `supports_search_tool`; OpenCode `OPENCODE_CONFIG_CONTENT`); start server / offer pull (size, license, fit) with confirmation (≤ 2 confirmations on an empty `~/.mrun`, SC-001); `--model` restart rule; `--dry-run`; missing-client hint; exec the client; tests with a fake server and fake client binary *Note*: `cli/clients.rs` (pure setup per client, unit-tested) + `cli/launch.rs` (server start via `serve --detach` after confirmation or `--yes`, `--dry-run`, served-model check exit 2, small-context warning, Codex reserved `-c` refused, missing binary E020, Claude Code version warning, exec). [U] The Codex model-catalog JSON shape is inferred: verify in 002 T016. Resolving `--model` aliases through the hub table lands with T004/T005.
- [X] T009 [US1] `cli/connect.rs`: print mode only (target path + persistent form, values from the running server); snapshot tests per client; stopped-server detection with `mrun serve --detach` hint. (`--write`, diff, 0600 backup, `--undo`, `--project`: removed, 0.1 scope cut, 2026-10-15) *Note*: `cli/connect.rs` prints the blocks from `clients::persistent` (Claude Code `env`, Codex profile plus its catalog, OpenCode provider); tests in `tests/cli.rs`.
- [X] T010 [US1] `serve --detach` and `cli/stop.rs` (pid file, graceful SIGTERM); tests *Note*: `serve --detach` (own process group, log in `MRUN_HOME/server.log`, pid file, returns on `/health` 200 or with the server's exit code and log tail) and `cli/stop.rs` (SIGTERM via `kill`, waits for `/health` to go away).

## Phase 4: US2 — Manage models (P1)

- [X] T011 [P] [US2] `cli/ls.rs` table/JSON with size, fit, max context (001 budget function), status (fast checks) for models with a manifest. (Listing `unpinned`/`unsupported` cached repos: removed, 0.1 scope cut, 2026-10-15) *Note*: `cli/ls.rs`; models pulled by repo id show alias `-`.
- [X] T012 [P] [US2] `cli/rm.rs` per FR-003/R10 (pinned revision's snapshot, blobs no other revision of the repo links to, refs only when they point to it; warning that other tools lose that revision), `cli/show.rs`; tests with a shared cache where another tool uses the same and a different revision *Note*: `cli/rm.rs`, `cli/show.rs`; shared-cache tests for one and the last revision.
- [X] T013 [US2] Corruption handling on load and in `ls`; repair via `pull` *Note*: fast checks in `ls` and `hub::resolve()`, repair by `pull`; `serve`/`run` call `resolve()` once the MLX load path exists (001 T040c).
- [X] T013a [US2] Architecture gate (FR-014, SC-006): `pull` fetches `config.json` (and wrapped `text_config`) first and resolves the family via 001's registry (001 T040); unsupported → exit 3 (E034) listing families and aliases, no weight bytes downloaded (wiremock asserts); `ls` FAMILY column; `show` family *Note*: config first, family via the registry; unsupported architecture or quantization mode → E034 with nothing downloaded (mock asserts zero weight requests).

## Phase 5: US3 — Diagnose (P2)

- [X] T014 [P] [US3] `cli/doctor.rs` checks per FR-007 (chip/macOS and budget with pressure, low-power mode, port conflict, server reachability, Claude Code ≥ 2.1.273, shell `ANTHROPIC_API_KEY`, missing client binaries); fake-context test per check (SC-002); exit 8 on failures. (`--agent`, integrity, free space, `template-tools-late`: removed, 0.1 scope cut, 2026-10-15) *Note*: `cli/doctor.rs`: pure checks over a `Context` with a fake per check; on macOS the GPU working set is approximated by physical memory until the MLX engine reports it.
- [X] T015 [P] [US3] `cli/ps.rs` against `/mrun/status` (002 schema v1), last miss with cause, divergence, message index and role, gap *Note*: `cli/ps.rs` renders `/mrun/status` (or `--json`).
- [X] T016 [US3] `mrun completions <shell>`, `--help` examples for every command (snapshot tests; latency measured in the benchmark job). (`mrun config`: removed, 0.1 scope cut, 2026-10-15) *Note*: `mrun completions <shell>` (clap_complete); every command has `--help` examples; snapshot tests of help text are not added (the CLI tests assert behavior).

## Phase 6: Polish

- [ ] T017 [P] `crates/mrun/tests/offline.rs`: all commands except `pull` succeed with network denied (FR-013)
- [ ] T018 SC-004 confirmed from 002 T016 (each alias × each client via `mrun launch`, 3 of 3); no separate runs. (`connect --write` runs: removed, 0.1 scope cut, 2026-10-15)
- [ ] T019 Fresh-Mac walkthrough (SC-001: empty `~/.mrun`, `mrun launch claude-code`, count confirmations and commands) and site page `getting-started.md` checked against it

## Dependencies

T001 → T002/T003 → T004 → T005/T006 → T007 → T010 → T008/T009 → 002 T016 → T018. T005a after T005
(default after 005 T012a); T013a after T007 and 001 T040. US2/US3 after T004–T006. `launch`,
`connect` and `ps` need 002's `/v1/models` and `/mrun/status`; fit numbers need 001's budget
function. The serve banner itself is owned by 002 (T012).
