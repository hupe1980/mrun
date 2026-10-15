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
| `mrun run <model> [sampling flags] [--think] [--chunk N] [--memory-fraction F] [--reserve-gib N] [--allow-over] [--info] [--check]` | 001 | Interactive chat; see [001 contract](../../001-engine-core/contracts/engine.md) |
| `mrun serve [--model M] [--host H] [--port P] [--api-key-file F] [--thinking client\|off\|on] [--max-output N] [--queue N] [--queue-wait-s N] [--chunk N] [--memory-fraction F] [--reserve-gib N] [--allow-over] [--no-cache] [--no-warmup] [--debug-cache] [--reject-tool-images] [--record DIR] [--detach]` | 001/002/003/004 | Start the server (default `127.0.0.1:8484`); prints URL and a `mrun launch` hint |
| `mrun stop` | 004 | Stop a detached server gracefully |
| `mrun ps` | 004 | Model, budget, active (with progress) and queued requests, cache stats, recording flag, last miss |
| `mrun pull <alias\|org/name[@rev]> [--yes]` | 004 | Size, license, fit; confirm if it won't fit; resumable download; verify; manifest |
| `mrun ls` | 004 | ALIAS, REPO, SIZE, FITS, MAX CTX, STATUS |
| `mrun show <model>` | 004 | Manifest, architecture, quantization, license, identity, fit breakdown |
| `mrun rm <alias> [--yes]` | 004 | Remove mrun-created snapshot refs and unshared blobs; warn about shared blobs |
| `mrun verify <model>` | 004 | Re-hash every file against the recorded identity |
| `mrun launch <claude-code\|codex\|opencode> [--model M] [--dry-run] [--yes] [-- client args]` | 004 | Start the client configured for this process only; starts a server and offers a pull when needed (FR-005) |
| `mrun connect <claude-code\|codex\|opencode> [--write] [--undo]` | 004 | Print persistent config; write with diff + 0600 backup; restore backup |
| `mrun doctor [--agent <client>]` | 004 | All checks with fixes (exit 8 on failures); `--agent` runs a tool-call and compaction round trip |
| `mrun cache clear` | 003 | Clear the prefix cache of the running server |
| `mrun config` | 004 | Effective configuration with sources |
| `mrun bench ...` | 005 | See [005 contract](../../005-bench/contracts/result-schema.md) |
| `mrun completions <bash\|zsh\|fish>` | 004 | Completion script |

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
```
