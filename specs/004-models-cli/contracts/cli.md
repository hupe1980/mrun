# CLI Contract (single source for all mrun commands)

All commands accept `--help` (with examples) and `-q/--quiet`; tabular commands accept `--json`.
Lean 0.1 (D025, 2026-10-15): commands and flags marked Later are not in 0.1.

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
| 7 | reserved (bench regression mode, Later) |
| 8 | `mrun doctor` found failures |

## Commands

| Command | Owner | Behavior |
| --- | --- | --- |
| `mrun run <model> [sampling flags] [--think] [--memory-fraction F] [--reserve-gib N] [--allow-over] [--info] [--check]` | 001 | Interactive chat; see [001 contract](../../001-engine-core/contracts/engine.md) |
| `mrun serve [--model M] [--port P] [--max-output N] [--queue N] [--queue-wait-s N] [--memory-fraction F] [--reserve-gib N] [--allow-over] [--no-cache] [--reject-tool-images] [--record DIR] [--detach]` | 001/002/003/004 | Start the server on `127.0.0.1` (default port 8484; loopback only); prints URL and a `mrun launch` hint |
| `mrun stop` | 004 | Stop a detached server gracefully |
| `mrun ps` | 004 | Model, budget, active (with progress) and queued requests, cache stats, recording flag, last miss |
| `mrun pull <alias\|org/name[@rev]> [--yes]` | 004 | `config.json` first: unsupported architecture refused before any weight (exit 3, supported families listed); size, license, fit; confirm if it won't fit; resumable download; verify; manifest |
| `mrun ls` | 004 | ALIAS, REPO, FAMILY, SIZE, FITS, MAX CTX, STATUS (`ready`, `corrupt`) for models with a manifest |
| `mrun show <model>` | 004 | Manifest, family, architecture, quantization, license, identity, fit breakdown |
| `mrun rm <alias> [--yes]` | 004 | Delete the pinned revision's snapshot and blobs no other revision links to; remove refs pointing to it; warn that other tools lose that revision |
| `mrun verify <model>` | 004 | Re-hash every file against the recorded identity |
| `mrun launch <claude-code\|codex\|opencode> [--model M] [--dry-run] [--yes] [-- client args]` | 004 | Start the client configured for this process only; starts a server and offers a pull when needed (FR-005) |
| `mrun connect <claude-code\|codex\|opencode>` | 004 | Print the persistent configuration and its target path; writes nothing |
| `mrun doctor` | 004 | Chip and budget, low-power mode, port, server reachability, client pitfalls, each with a fix (exit 8 on failures) |
| `mrun cache clear` | 003 | Clear the prefix cache of the running server |
| `mrun bench ...` | 005 | See [005 contract](../../005-bench/contracts/result-schema.md) |
| `mrun completions <bash\|zsh\|fish>` | 004 | Completion script |
| `mrun --version` | 008 | Version, commit, MLX and mlx-c versions |

Later (IDs kept in the owning specs): `serve --host H --api-key-file F` (002 FR-012), `serve
--thinking` (002 FR-013), `serve --debug-cache` (003 FR-011), `connect --write/--undo/--project`
(FR-006), `doctor --agent` (FR-007), `mrun config` (FR-010), `mrun service install|uninstall
[--preload]` and `serve --service` (008), `bench --baseline` (005). Post-0.1 features: `serve
--speculation auto\|off\|ngram` (006); `serve --persist` / `--no-persist`, `mrun cache [--json]`
and `mrun cache clear [--disk\|--all] [--model M]` (007).

## Configuration sources

defaults < `MRUN_*` environment (`MRUN_<FLAG>` in upper snake case, for example
`MRUN_SERVER_PORT`, `MRUN_MEMORY_RESERVE_GIB`) < flags. Command-line only (no environment
equivalent): `--record`, `--allow-over`. `MRUN_HOME` sets the state directory. No config file in
0.1.

## Model resolution

Existing filesystem path → alias manifest → HF repo id in the cache. Ambiguity is an error listing
candidates. Every resolved model has a content identity (001 R10).

## Error message shape (stderr)

```text
error[E031]: model 'qwen3-codr' not found
  hint: did you mean 'qwen3-coder'? Run 'mrun ls' to see local models.

error[E034]: unsupported architecture 'llama4' in org/model (nothing downloaded)
  supported families: qwen3_5_moe (qwen3.6), qwen3_moe (qwen3-coder), gpt_oss (gpt-oss), qwen3
```

E034 is the single unsupported-architecture message for every command (`run`, `serve`, `pull`;
referenced by 001 and 010); `(nothing downloaded)` appears only on `pull`.

Curated aliases: `qwen3.6` (default candidate until 005 T012a decides), `qwen3-coder`, `gpt-oss`
([research R4](../research.md#r4-curated-aliases)).
