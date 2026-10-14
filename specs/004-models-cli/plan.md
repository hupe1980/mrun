# Implementation Plan: Models and CLI Experience

**Branch**: `004-models-cli` | **Date**: 2026-10-14 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `specs/004-models-cli/spec.md`

## Summary

Give mrun an Ollama-grade command line on top of the shared Hugging Face cache: `pull` (resumable,
hash-verified, license shown), `ls/rm/show` with memory fit, `connect` for Claude Code, Codex and
OpenCode, `doctor` and `ps` diagnostics, one optional TOML config, and shell completions. Models
live in the standard HF hub cache so downloads are shared with mlx-lm and other tools; mrun only
stores aliases and verified manifests under `~/.mrun`.

## Technical Context

**Language/Version**: Rust 1.94.0 (edition 2024), workspace pinned by `rust-toolchain.toml`
**Primary Dependencies**: clap 4.6 + clap_complete 4.6 (CLI), hf-hub 1.0 (HF cache layout and
downloads, tokio API), reqwest 0.13 (HEAD/range requests, used through hf-hub), sha2 0.11
(verification), indicatif 0.18 (progress), toml 1.1 + serde (config), similar (config diffs)
**Storage**: HF hub cache (`$HF_HUB_CACHE` or `$HF_HOME/hub` or `~/.cache/huggingface/hub`);
`~/.mrun/` (`MRUN_HOME`) for `config.toml`, `manifests/*.toml`, `backups/`
**Testing**: cargo test; wiremock 0.6 for a fake hub; tempfile for isolated `MRUN_HOME`/`HF_HOME`;
insta 1.49 snapshots for `--help`, `connect` output and error messages
**Target Platform**: macOS on Apple Silicon (primary); Linux compiles and runs these tests
**Project Type**: CLI (part of the single `mrun` binary)
**Performance Goals**: CLI commands other than `pull` start and finish in under 200 ms; `ls` with
20 models under 500 ms (hashes are cached by file size + mtime, never recomputed per `ls`)
**Constraints**: No network except `pull` (FR-011); never overwrite user client config without a
diff and backup (FR-005); offline-first
**Scale/Scope**: Tens of models, each 5–60 GB

## Constitution Check

| Principle | Gate | Status |
| --- | --- | --- |
| I Agent latency | `connect` sets cache-stable client settings (attribution header off) | Pass |
| II Exactness | Model identity = verified content hashes (feeds 003 cache keys) | Pass |
| III Memory budget | `ls/show/pull` report fit from the same budget function as 001 | Pass |
| IV Local, private | Only `pull` touches network; no telemetry; tokens only from `HF_TOKEN` | Pass |
| V Own runtime | No engine types in CLI; fit computed via the runtime budget API | Pass |
| VI Simple, DX | Zero-config defaults, one config file, every error has a next step | Pass |

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
│   ├── connect.rs       # client profiles, diff + backup writer
│   ├── doctor.rs        # checks registry
│   ├── ps.rs            # queries the running server's /mrun/status
│   └── config_cmd.rs    # `mrun config` (effective config with sources)
├── config.rs            # defaults < config.toml < MRUN_* env < flags
├── hub/
│   ├── mod.rs           # resolve(alias|repo|path) -> LocalModel
│   ├── aliases.rs       # curated alias table (pinned repo, revision, file hashes)
│   ├── download.rs      # hf-hub driven, resumable, progress, disk-space check
│   └── verify.rs        # sha256 verification with size+mtime cache
└── errors.rs            # user-facing error catalog (code, message, next step)
crates/mrun/tests/
├── cli_snapshots.rs     # help/connect/errors snapshots
├── hub_pull.rs          # wiremock hub: resume, corrupt, disk full, gated
└── connect_write.rs     # diff + backup behavior on temp HOME
```

**Structure Decision**: The CLI and hub live in the main `mrun` crate (see
[ARCHITECTURE](../../concepts/ARCHITECTURE.md)). `doctor`/`ps` talk to a running server over its
loopback status endpoint; no IPC beyond HTTP.

## Complexity Tracking

None.
