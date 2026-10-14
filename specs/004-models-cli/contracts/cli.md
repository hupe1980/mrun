# CLI Contract

All commands accept `--help` (with examples), `--json` where output is tabular, and `-q/--quiet`.
Exit codes: 0 ok · 1 error · 2 usage · 3 model not found/invalid · 4 does not fit · 5 server
unreachable · 6 network/download.

| Command | Behavior | Output |
| --- | --- | --- |
| `mrun pull <alias\|org/name[@rev]> [--yes]` | Show size, license, fit; confirm if it won't fit; download (resume); verify; write manifest | Progress bar, final path |
| `mrun ls` | List manifests + verification state | Table: ALIAS, REPO, SIZE, FITS, MAX CTX, STATUS |
| `mrun rm <alias> [--yes]` | Remove snapshot files not referenced by another manifest, then manifest | Freed bytes |
| `mrun show <model>` | Manifest, architecture, quantization, license, fit breakdown | Key/value |
| `mrun run <model>` | Feature 001 | — |
| `mrun serve [--model M] [--host H --port P] [--api-key K]` | Feature 002 | Banner with URL and connect hint |
| `mrun connect <claude-code\|codex\|opencode> [--write]` | Print config for the running server; `--write` shows diff, backs up, writes atomically | Config snippet or diff |
| `mrun doctor` | Run all checks | Findings grouped by severity with fixes; exit 1 if any fail |
| `mrun ps` | Query `/mrun/status` | Model, budget breakdown, active/queued, cache size/hit rate, last miss |
| `mrun cache clear` | Feature 003 | — |
| `mrun config` | Effective config with sources | Table |
| `mrun bench ...` | Feature 005 | — |
| `mrun completions <bash\|zsh\|fish>` | Print completion script | Script |

Model resolution order: existing filesystem path → alias manifest → HF repo id. Ambiguity is an
error listing candidates.

Error message shape (stderr): `error[E031]: model 'qwen3-codr' not found` then
`  hint: did you mean 'qwen3-coder'? Run 'mrun ls' to see local models.`
