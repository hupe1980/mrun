# Data Model: Models and CLI Experience

## ModelManifest (`~/.mrun/manifests/<alias>.toml`)

| Field | Type | Rules |
| --- | --- | --- |
| alias | string | `[a-z0-9][a-z0-9.-]{0,63}`, unique |
| repo | string | HF repo id `org/name` |
| revision | string | 40-hex commit sha (pinned) |
| files | list of {path, size, sha256} | every file needed to load; sha256 64-hex |
| license | string | SPDX id or "other" with URL |
| architecture | string | from `config.json` `model_type` (e.g. `qwen3`, `qwen3_moe`) |
| quantization | {bits, group_size, mode} | from `config.json` |
| identity | string | sha256 over sorted (path, sha256) pairs; used by 001/003 |

States: `missing` → `downloading` → `verifying` → `ready` | `corrupt` (hash mismatch) → (pull) →
`ready`.

## ClientProfile (compiled in)

| Field | Type |
| --- | --- |
| name | `claude-code` \| `codex` \| `opencode` |
| config_path | path template (`~/.claude/settings.json`, ...) |
| format | json \| toml |
| settings | key path → value template (base URL, model, auth token, cache hints) |
| probe | how to verify the client sees mrun (where possible) |

## Diagnostic

| Field | Type |
| --- | --- |
| id | stable string (`memory.pressure`, `client.claude.attribution`, ...) |
| status | ok \| warn \| fail \| skipped |
| severity | info \| warning \| error |
| detail | human text |
| fix | concrete command or setting |

## Config (`~/.mrun/config.toml`)

```toml
[server]  host = "127.0.0.1"  port = 8484  queue = 8  queue_deadline_s = 30
[memory]  fraction = 0.75  reserve_gib = 8  allow_over = false
[engine]  chunk = 512  default_model = "qwen3-coder"
[cache]   enabled = true
```

Every key is optional; `mrun config` shows effective values and sources.
