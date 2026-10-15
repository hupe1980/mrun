# CLI Contract (single source for all mrun commands)

All commands accept `--help` (with examples) and `-q/--quiet`; tabular commands accept `--json`.

## Exit codes

| Code | Meaning |
| --- | --- |
| 0 | success |
| 1 | unexpected error |
| 2 | usage error |
| 3 | model not found, unsupported or corrupt |
| 4 | does not fit the memory budget |
| 5 | server unreachable |
| 6 | network or download failure |
| 7 | benchmark regression threshold failed (`mrun bench --baseline`) |
| 8 | `mrun doctor` found failures |

## Commands

| Command | Owner | Behavior |
| --- | --- | --- |
| `mrun run <model> [sampling flags] [--think] [--memory-fraction F] [--reserve-gib N] [--allow-over] [--info] [--check]` | 001 | Interactive chat; see [001 contract](../../001-engine-core/contracts/engine.md) |
| `mrun serve [--model M] [--host H] [--port P] [--api-key-file F] [--thinking client\|off\|on] [--max-output N] [--queue N] [--queue-wait-s N] [--memory-fraction F] [--reserve-gib N] [--allow-over] [--no-cache] [--debug-cache] [--reject-tool-images] [--record DIR] [--detach]` | 001/002/003/004 | Start the server (default `127.0.0.1:8484`); prints URL and a `mrun launch` hint |
| `mrun stop` | 004 | Stop a detached server gracefully |
| `mrun ps` | 004 | Model, budget, active (with progress) and queued requests, cache stats, recording flag, last miss |
| `mrun pull <alias\|org/name[@rev]> [--yes]` | 004 | `config.json` first: unsupported architecture refused before any weight (exit 3, supported families listed); size, license, fit; confirm if it won't fit; resumable download; verify; manifest |
| `mrun ls` | 004 | ALIAS, REPO, FAMILY, SIZE, FITS, MAX CTX, STATUS (`ready`, `corrupt`, `unpinned`, `unsupported`) |
| `mrun show <model>` | 004 | Manifest, family, architecture, quantization, license, identity, fit breakdown |
| `mrun rm <alias> [--yes]` | 004 | Delete the pinned revision's snapshot and blobs no other revision links to; remove refs pointing to it; warn that other tools lose that revision |
| `mrun verify <model>` | 004 | Re-hash every file against the recorded identity |
| `mrun launch <claude-code\|codex\|opencode> [--model M] [--dry-run] [--yes] [-- client args]` | 004 | Start the client configured for this process only; starts a server and offers a pull when needed (FR-005) |
| `mrun connect <claude-code\|codex\|opencode> [--write] [--undo] [--project]` | 004 | Print persistent config; write with diff + 0600 backup; restore backup; `--project` targets `.claude/settings.local.json` |
| `mrun doctor [--agent <client>]` | 004 | All checks with fixes (exit 8 on failures); `--agent` runs a tool-call round trip and a request at 95% of capacity |
| `mrun cache clear` | 003 | Clear the prefix cache of the running server |
| `mrun config` | 004 | Effective configuration with sources |
| `mrun bench ...` | 005 | See [005 contract](../../005-bench/contracts/result-schema.md) |
| `mrun completions <bash\|zsh\|fish>` | 004 | Completion script |
| `mrun service install [--preload]` / `uninstall` | 008 | launchd user agent (lazy model load unless `--preload`); see [008 contract](../../008-distribution/contracts/service-and-release.md) |
| `mrun --version` | 008 | Version, commit, MLX and mlx-c versions |

Added by post-0.1 features (listed here so the CLI stays one surface): `serve --speculation
auto\|off\|ngram` (006; `mtp` only after its gated follow-up); `serve --persist` / `--no-persist`,
`mrun cache [--json]` and `mrun cache clear [--disk\|--all] [--model M]` (007).

## Configuration sources

defaults < `~/.mrun/config.toml` < `MRUN_*` environment < flags. Command-line only (no config or
environment equivalent): `--record`, `--debug-cache`, `--allow-over`. The API key comes only from
`MRUN_API_KEY` or `--api-key-file`.

## Model resolution

Existing filesystem path → alias manifest → HF repo id in the cache. Ambiguity is an error listing
candidates. Every resolved model has a content identity (001 R10).

## Error message shape (stderr)

```text
error[E031]: model 'qwen3-codr' not found
  hint: did you mean 'qwen3-coder'? Run 'mrun ls' to see local models.

error[E034]: unsupported architecture 'llama4' in org/model (nothing downloaded)
  supported families: qwen3_5_moe (qwen3.6), qwen3_moe (qwen3-coder), gpt_oss (gpt-oss),
  mistral (devstral), qwen3 (qwen3-8b)
```

Curated aliases: `qwen3.6` (default candidate until 005 T012a decides), `qwen3-coder`, `gpt-oss`,
`devstral`, `qwen3-8b` ([research R4](../research.md#r4-curated-aliases)).
