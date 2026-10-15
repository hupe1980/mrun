# Implementation Plan: Models and CLI Experience

**Branch**: `004-models-cli` | **Date**: 2026-10-14 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `specs/004-models-cli/spec.md`

## Summary

Give mrun an Ollama-grade command line on top of the shared Hugging Face cache: `pull` (resumable,
hash-verified, license shown), `ls/rm/show` with memory fit, `launch` (and a printing `connect`)
for Claude Code, Codex and OpenCode, `doctor` and `ps` diagnostics, `MRUN_*` settings without a
config file (lean 0.1, D025), and shell completions. Models
live in the standard HF hub cache so downloads are shared with mlx-lm and other tools; mrun only
stores aliases and verified manifests under `~/.mrun`.

## Technical Context

**Language/Version**: Rust 1.94.0 (edition 2024), workspace pinned by `rust-toolchain.toml`
**Primary Dependencies**: clap 4.6 + clap_complete 4.6 (CLI), hf-hub 1.0 (HF cache layout and
downloads; async by default), reqwest 0.13 (HEAD/range requests, used through hf-hub), sha2 0.11
(verification), indicatif 0.18 (progress), toml 1.1 + serde (manifests)
**Storage**: HF hub cache (`$HF_HUB_CACHE` or `$HF_HOME/hub` or `~/.cache/huggingface/hub`);
`~/.mrun/` (`MRUN_HOME`) for `manifests/*.toml`, `codex/models.json`, `server.log`, `server.pid`
**Testing**: cargo test; wiremock 0.6 for a fake hub; tempfile for isolated `MRUN_HOME`/`HF_HOME`;
insta 1.49 snapshots for `--help`, `connect` output and error messages
**Target Platform**: macOS on Apple Silicon (primary); Linux compiles and runs these tests
**Project Type**: CLI (part of the single `mrun` binary)
**Performance Goals**: CLI commands other than `pull`, `verify`, `serve` and `launch` finish in under 200 ms (measured in the benchmark job, not asserted in snapshot tests);
`ls` with 20 models under 500 ms (fast checks only; full hashing only in `verify`)
**Constraints**: No network except `pull` (FR-013); never write user client config (FR-006:
`connect` prints); offline-first
**Scale/Scope**: Tens of models, each 5–60 GB

## Constitution Check

| Principle | Gate | Status |
| --- | --- | --- |
| I Agent latency | `launch`/`connect` set cache-stable client settings (token reminder off, hint headers on) and real context limits | Pass |
| II Exactness | Model identity = verified content hashes (feeds 003 cache keys) | Pass |
| III Memory budget | `ls/show/pull/launch` use 001's budget function and the server's capacity | Pass |
| IV Local, private | Only `pull` touches network (offline test); identity for path models too; launch is process-scoped | Pass |
| V Own runtime | No engine types in CLI; fit computed via the runtime budget API | Pass |
| VI Simple, DX | Zero-config defaults, no config file, every error has a next step | Pass |

Post-design re-check: Pass (no violations; no complexity tracking needed).

## Project Structure

### Documentation (this feature)

```text
specs/004-models-cli/
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
├── contracts/cli.md
└── tasks.md
```

### Source Code (repository root)

```text
crates/mrun/src/
├── main.rs              # clap entry, exit codes
├── cli/
│   ├── mod.rs           # command tree, global flags, completions
│   ├── pull.rs  ls.rs  rm.rs  show.rs
│   ├── launch.rs        # process-scoped client start
│   ├── connect.rs       # client profiles, printed persistent form
│   ├── verify.rs        # full re-hash
│   ├── stop.rs          # stop a detached server
│   ├── doctor.rs        # checks registry
│   └── ps.rs            # queries the running server's /mrun/status
├── config.rs            # defaults < MRUN_* env < flags
├── hub/
│   ├── mod.rs           # resolve(alias|repo|path) -> LocalModel
│   ├── aliases.rs       # curated alias table, one per 0.1 family (pinned repo, revision, file hashes, family)
│   ├── download.rs      # hf-hub driven, resumable, progress, disk-space check
│   └── verify.rs        # sha256 verification with size+mtime fast checks
└── errors.rs            # user-facing error catalog (code, message, next step)
crates/mrun/tests/
├── cli_snapshots.rs     # help/connect/errors snapshots
├── hub_pull.rs          # wiremock hub: resume, corrupt, disk full, gated, HF_ENDPOINT
└── offline.rs           # every non-pull command works with network denied
```

**Structure Decision**: The CLI and hub live in the main `mrun` crate (see
[ARCHITECTURE](../../concepts/ARCHITECTURE.md)). `doctor`/`ps` talk to a running server over its
loopback status endpoint; no IPC beyond HTTP.

## Complexity Tracking

None.
