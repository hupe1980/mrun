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

## R2: Verification

- **Decision**: Verify LFS files against the hub's sha256 (from the API's `lfs.oid`) at download and
  against the pinned manifest at load; small non-LFS files (config, tokenizer, template) are hashed
  and pinned in the manifest. Cache verification results keyed by (path, size, mtime) in
  `~/.mrun/verify-cache.toml` so `ls` stays fast.
- **Rationale**: Constitution IV (content identity); 003 cache keys need weights/tokenizer/template
  hashes.
- **Alternatives**: Re-hash on every load (30 s+ for 18 GB); trust filenames (violates IV).

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

## R5: Client integration (`connect`)

- **Decision**: One profile per client with: config location, keys to set, cache-stability hints,
  and a validation probe.
  - Claude Code: `~/.claude/settings.json` → `env.ANTHROPIC_BASE_URL=http://127.0.0.1:8484`,
    `env.ANTHROPIC_AUTH_TOKEN=mrun`, `env.ANTHROPIC_MODEL=<alias>`,
    `env.CLAUDE_CODE_ATTRIBUTION_HEADER=0`.
  - Codex: `~/.codex/config.toml` → `[model_providers.mrun]` with `base_url`,
    `wire_api = "responses"`, plus `[profiles.mrun]`.
  - OpenCode: `opencode.json` provider using `@ai-sdk/openai-compatible` with `baseURL`.
- **Rationale**: Landscape study: Codex accepts only Responses; Claude Code's per-request
  attribution header breaks exact prefix reuse unless disabled.
- **Validation**: Exact keys are re-verified against the pinned client versions in feature 002's
  research before release; snapshot tests freeze the generated config.
- **Write safety**: Parse, merge, show unified diff, back up to `~/.mrun/backups/<client>-<ts>`,
  write atomically (temp file + rename).

## R6: Doctor checks

- **Decision**: Checks registry, each `fn check(&Ctx) -> Finding { id, status, severity, fix }`:
  chip/macOS (`sysctl machdep.cpu.brand_string`, `sw_vers`), memory budget and pressure
  (`vm.memory_pressure`, `host_statistics64`), low-power mode (`pmset -g`), model integrity
  (verify cache), server reachability (`GET /health`), port conflicts, client config pitfalls
  (missing attribution setting, wrong wire_api, wrong base URL), HF cache location and free space.
- **Alternatives**: A monolithic report (hard to test); each check is unit-tested with a fake
  context.

## R7: Config

- **Decision**: Layered: defaults < `~/.mrun/config.toml` < `MRUN_*` env < flags; `mrun config`
  prints each effective value with its source. Unknown keys are errors with a "did you mean".
- **Alternatives**: Multiple config locations (confusing), YAML (no benefit).

## R8: Error catalog

- **Decision**: `errors.rs` enumerates user-facing errors with stable codes (`E0xx`), message and
  next step; exit codes: 0 ok, 1 generic, 2 usage, 3 model not found/invalid, 4 does not fit,
  5 server unreachable, 6 network/download. A snapshot test lists every error to enforce FR-009.
