---
description: "Tasks for 004 Models and CLI Experience"
---

# Tasks: Models and CLI Experience

**Input**: [plan.md](plan.md), [spec.md](spec.md), [research.md](research.md), [data-model.md](data-model.md), [contracts/cli.md](contracts/cli.md)

## Phase 1: Setup

- [ ] T001 Add hf-hub 1.0, indicatif 0.18, sha2 0.11, toml 1.1, similar, clap_complete 4.6, wiremock/insta/tempfile (dev) to `crates/mrun/Cargo.toml`
- [ ] T002 [P] `crates/mrun/src/config.rs`: layered config (defaults < `~/.mrun/config.toml` < `MRUN_*` < flags), unknown-key errors with suggestions; tests
- [ ] T003 [P] `crates/mrun/src/errors.rs`: error catalog (codes, messages, next steps, exit codes) + snapshot test listing all errors (FR-009)

## Phase 2: Foundational

- [ ] T004 `crates/mrun/src/hub/mod.rs`: resolve path → alias → repo id; HF cache path resolution (`HF_HUB_CACHE`, `HF_HOME`)
- [ ] T005 [P] `hub/aliases.rs`: curated table (`qwen3-coder`, `qwen3-8b`) with pinned revisions and file hashes; `scripts/refresh_aliases.py` for maintainers
- [ ] T006 [P] `hub/verify.rs`: sha256 verification with (path, size, mtime) cache; model identity hash; tests

## Phase 3: US1 — Zero to working agent (P1) 🎯 MVP

- [ ] T007 [US1] `hub/download.rs` + `cli/pull.rs`: size/license/fit preview, disk-space check, hf-hub download with resume, verification, manifest write; wiremock tests (resume, corrupt, disk full, gated → `HF_TOKEN` hint)
- [ ] T008 [US1] `cli/connect.rs`: client profiles (Claude Code settings.json env incl. `CLAUDE_CODE_ATTRIBUTION_HEADER=0`; Codex `config.toml` provider `wire_api = "responses"`; OpenCode `opencode.json`), print mode, `--write` with diff + backup + atomic write; snapshot tests; key names verified against 002 research
- [ ] T009 [US1] `serve` startup banner suggests `mrun connect <client>`; `connect` detects a stopped server

## Phase 4: US2 — Manage models (P1)

- [ ] T010 [P] [US2] `cli/ls.rs` table/JSON with size, fit, max context (budget function from 001), status
- [ ] T011 [P] [US2] `cli/rm.rs` (shared blobs safe), `cli/show.rs`
- [ ] T012 [US2] Corruption detection on load and in `ls`; repair via `pull`

## Phase 5: US3 — Diagnose (P2)

- [ ] T013 [P] [US3] `cli/doctor.rs` checks registry: chip/macOS, budget/pressure, low power, model integrity, server reachability, port conflict, client config pitfalls, HF cache free space; fake-context tests for ≥ 10 scenarios (SC-002)
- [ ] T014 [P] [US3] `cli/ps.rs` against `/mrun/status` (model, budget, active/queued, cache, last miss)
- [ ] T015 [US3] `mrun config` (effective values with sources), `mrun completions <shell>`, `--help` examples for every command (snapshot)

## Phase 6: Polish

- [ ] T016 Fresh-Mac walkthrough timing (SC-001) and README "Quick start"

## Dependencies

T001 → T002/T003 → T004 → T005/T006 → T007 → T008/T009. US2/US3 after T004–T006. `ps` needs
002's `/mrun/status`; fit numbers need 001's budget function.
