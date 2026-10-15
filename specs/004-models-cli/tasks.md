---
description: "Tasks for 004 Models and CLI Experience"
---

# Tasks: Models and CLI Experience

**Input**: [plan.md](plan.md), [spec.md](spec.md), [research.md](research.md), [data-model.md](data-model.md), [contracts/cli.md](contracts/cli.md)

## Phase 1: Setup

- [ ] T001 Add hf-hub 1.0, indicatif 0.18, sha2 0.11, toml 1.1, similar, clap_complete 4.6, wiremock/insta/tempfile (dev) to `crates/mrun/Cargo.toml`
- [ ] T002 [P] `crates/mrun/src/config.rs`: layered config (defaults < `~/.mrun/config.toml` < `MRUN_*` < flags), command-line-only settings, unknown-key errors with suggestions; tests
- [ ] T003 [P] `crates/mrun/src/errors.rs`: error catalog and exit codes 0–8 per contract + snapshot test listing all errors (FR-011)

## Phase 2: Foundational

- [ ] T004 `crates/mrun/src/hub/mod.rs`: resolve path → alias → repo id; HF cache resolution (`HF_HUB_CACHE`, `HF_HOME`, `HF_ENDPOINT`); identity via `mrun_engine::identity` (001)
- [ ] T005 [P] `hub/aliases.rs`: curated table (`qwen3.6` default, `qwen3-coder`, `qwen3-8b`) with pinned revisions and file hashes; `cargo xtask refresh-aliases`
- [ ] T005a [P] Aliases `gpt-oss` and `devstral` (research R4): resolve the `[U]` repos (Devstral plain `-4bit` vs OptiQ; gpt-oss `-MXFP4-Q8` quantization block), pin revisions and hashes, record family, syntax, tier and fit numbers per alias; default alias field set from 005 T012a
- [ ] T006 [P] `hub/verify.rs` + `cli/verify.rs`: fast checks (size, mtime, inode) and full re-hash; test flipping a byte with preserved mtime is caught by `verify` only

## Phase 3: US1 — Zero to working agent (P1) 🎯 MVP

- [ ] T007 [US1] `hub/download.rs` + `cli/pull.rs`: size/license/fit preview, disk-space check, hf-hub download with resume, verification, manifest; wiremock tests (resume, corrupt, disk full, gated → `HF_TOKEN` hint, `HF_ENDPOINT` mirror)
- [ ] T008 [US1] `cli/launch.rs`: process-scoped env/flags per client per FR-005 / research R5 (Claude Code env set incl. emptied `ANTHROPIC_API_KEY`; `ENABLE_TOOL_SEARCH=auto:10`; Codex `-c` overrides + model catalog file with `supports_search_tool`; OpenCode `OPENCODE_CONFIG_CONTENT`); start server / offer pull (size, license, fit) with confirmation; `--model` restart rule; `--dry-run`; missing-client hint; exec the client; tests with a fake server and fake client binary
- [ ] T009 [US1] `cli/connect.rs`: print mode; `--write` with diff, 0600 backup, atomic write to the FR-006 targets (Codex separate `mrun.config.toml` profile); `--undo`; snapshot tests per client; stopped-server detection with `mrun serve --detach` hint
- [ ] T010 [US1] `serve --detach` and `cli/stop.rs` (pid file, graceful SIGTERM); tests

## Phase 4: US2 — Manage models (P1)

- [ ] T011 [P] [US2] `cli/ls.rs` table/JSON with size, fit, max context (001 budget function), status (fast checks), plus compatible `unpinned` MLX repos found in the HF cache; `pull <repo>` pins them without downloading
- [ ] T012 [P] [US2] `cli/rm.rs` per FR-003/R10 (pinned revision's snapshot, blobs no other revision of the repo links to, refs only when they point to it; warning that other tools lose that revision), `cli/show.rs`; tests with a shared cache where another tool uses the same and a different revision
- [ ] T013 [US2] Corruption handling on load and in `ls`; repair via `pull`
- [ ] T013a [US2] Architecture gate (FR-014, SC-006): `pull` fetches `config.json` (and wrapped `text_config`) first and resolves the family via 001's registry (001 T040); unsupported → exit 3 (E034) listing families and aliases, no weight bytes downloaded (wiremock asserts); `ls` FAMILY column and `unsupported` status for cached repos; `show` family

## Phase 5: US3 — Diagnose (P2)

- [ ] T014 [P] [US3] `cli/doctor.rs` checks registry per FR-007 (incl. Claude Code ≥ 2.1.273, shell `ANTHROPIC_API_KEY`, missing context limits, `--agent <client>` round trip, port conflict, cache free space); fake-context tests for ≥ 10 scenarios (SC-002); exit 8 on failures
- [ ] T015 [P] [US3] `cli/ps.rs` against `/mrun/status` (002 schema v1)
- [ ] T016 [US3] `mrun config`, `mrun completions <shell>`, `--help` examples for every command (snapshot tests; latency measured in the benchmark job)

## Phase 6: Polish

- [ ] T017 [P] `crates/mrun/tests/offline.rs`: all commands except `pull` succeed with network denied (FR-013)
- [ ] T018 Real-client validation (SC-004): Claude Code, Codex and OpenCode (pinned versions) run the scripted read-edit-test task 5× each via `mrun launch` and via `mrun connect --write`; record versions
- [ ] T019 Fresh-Mac walkthrough timing (SC-001) and site page `getting-started.md` checked against it

## Dependencies

T001 → T002/T003 → T004 → T005/T006 → T007 → T008/T009/T010 → T018 (also needs 002's MVP). T005a after T005 (default after 005 T012a); T013a after T007 and 001 T040. US2/US3 after T004–T006. `launch`,
`connect` and `ps` need 002's `/v1/models` and `/mrun/status`; fit numbers need 001's budget
function. The serve banner itself is owned by 002 (T012).
