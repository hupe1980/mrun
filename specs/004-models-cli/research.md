# Research: Models and CLI Experience

## R1: Model storage

- **Decision**: Use the Hugging Face hub cache layout (`models--{org}--{name}/blobs|snapshots|refs`)
  via `hf-hub` 1.0, honoring `HF_HUB_CACHE`, `HF_HOME` and `HF_TOKEN`. mrun keeps a manifest per
  alias in `~/.mrun/manifests/` with the pinned revision and per-file SHA-256.
- **Rationale**: Maintainer decision (spec clarification). Avoids duplicate 18+ GB downloads with
  mlx-lm and other HF tools; blob files are already content-addressed (git/LFS sha256 for LFS
  files), which gives verification for free on large weight files.
- **Alternatives**: Own store under `~/.mrun/models` (duplicates); Ollama-style OCI blobs (not
  compatible with MLX community repos).

## R2: Identity and verification

- **Decision**: Use feature 001's identity function (sha256 over sorted (path, file sha256)); in the
  HF cache, LFS blob names are their sha256, so weights need no re-hashing; downloads are verified
  against the hub's `lfs.oid` as they complete. Fast checks on every use compare (size, mtime,
  inode) with the recorded values; `mrun verify` re-hashes everything. Path-loaded models get the
  same identity (hashed once, cached in `~/.mrun/identity.toml`).
- **Rationale**: Constitution IV; analysis C2/I7: a size+mtime cache cannot catch every in-place
  edit, so the spec promises full detection only through `verify`.

## R3: Resumable download

- **Decision**: hf-hub's tokio API downloads into `.incomplete` blobs; mrun adds range-resume when an
  `.incomplete` file exists (HTTP `Range`), checks free disk space before starting (sum of missing
  file sizes + 5% margin), and deletes partial files on explicit cancel or verify failure.
- **Alternatives**: Shelling out to `huggingface-cli` (extra dependency, worse errors).

## R4: Curated aliases

- **Decision**: Compile-time table in `hub/aliases.rs`: `qwen3-coder` →
  `mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit`, `qwen3-8b` → `mlx-community/Qwen3-8B-4bit`,
  each with pinned revision and file hashes, refreshed by a maintainer script. Unknown names are
  treated as HF repo ids (`org/name[@revision]`), pinned on first pull.
- **Rationale**: Short names are the top DX feature of Ollama; pinning keeps identity exact.

## R5: Client integration (`launch` and `connect`)

- **Decision**: `mrun launch <client>` (primary, Ollama-style) execs the client with process-scoped
  environment and flags; `mrun connect` prints/writes persistent config with diff, 0600 backup and
  `--undo`. Values come from the running server (`/v1/models` context window, served model id).
  - Claude Code: `ANTHROPIC_BASE_URL=http://127.0.0.1:8484`, `ANTHROPIC_AUTH_TOKEN` (env reference
    when a key is configured, otherwise a dummy), `ANTHROPIC_MODEL`,
    `ANTHROPIC_DEFAULT_{OPUS,SONNET,HAIKU}_MODEL`, `CLAUDE_CODE_SUBAGENT_MODEL` (all the served
    id), `CLAUDE_CODE_MAX_CONTEXT_TOKENS` / `CLAUDE_CODE_AUTO_COMPACT_WINDOW` from capacity,
    `CLAUDE_CODE_ATTRIBUTION_HEADER=0`.
  - Codex: provider `mrun` with `base_url`, `wire_api = "responses"`, `env_key` (when keyed),
    `model_context_window` and `model_auto_compact_token_limit` from capacity (output limit from
    `/v1/models` `max_output_tokens`).
  - OpenCode: provider using `@ai-sdk/openai-compatible` with `baseURL`, model
    `limit.context`/`limit.output`.
- **Rationale**: Analysis I3 (all model slots, else background/subagent requests use `claude-*`
  ids), I4 (clients default to 200K–272K context; mrun's capacity is ≈ 100–110K on 48 GB), S4
  (global settings redirect all Claude Code use; launch is scoped). Key names are re-verified
  against pinned client versions in 002 research before release; snapshot tests freeze output.

## R6: Doctor checks

- **Decision**: Registry of checks, each `fn check(&Ctx) -> Finding { id, status, severity, fix }`:
  chip/macOS, memory budget and pressure, low-power mode, model integrity (fast checks), server
  reachability, port conflicts, HF cache free space, client config pitfalls (wrong base URL or wire
  API, missing context limits, Claude Code < 2.1.181 without the attribution setting). Each check is
  unit-tested with a fake context. Exit code 8 when any check fails.

## R7: Config

- **Decision**: Layered: defaults < `~/.mrun/config.toml` < `MRUN_*` env < flags; `mrun config`
  prints each effective value with its source. Unknown keys are errors with a "did you mean".
- **Alternatives**: Multiple config locations (confusing), YAML (no benefit).

## R8: Error catalog

- **Decision**: `errors.rs` enumerates user-facing errors with stable codes (`E0xx`), message and
  next step; exit codes per the [CLI contract](contracts/cli.md#exit-codes). A snapshot test lists
  every error to enforce FR-011.


## R9: Background server

- **Decision**: `mrun serve --detach` forks a background server writing logs to
  `~/.mrun/server.log` and its pid to `~/.mrun/server.pid`; `mrun stop` sends SIGTERM (graceful:
  cancel queue, finish or cancel the active request, exit). launchd integration belongs to the
  distribution milestone.
- **Rationale**: Analysis X2; competitors offer background services (oMLX via brew services, Ollama
  app).

## R10: rm safety in a shared cache

- **Decision**: `rm` deletes the snapshot directory and refs mrun recorded in the manifest, then
  only blobs no remaining snapshot (of any repo in the cache) links to; it lists shared blobs it
  kept.
- **Rationale**: Analysis S7: the HF cache is shared with other tools.
