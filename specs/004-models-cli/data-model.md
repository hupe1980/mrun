# Data Model: Models and CLI Experience

## ModelManifest (`~/.mrun/manifests/<alias>.toml`)

| Field | Type | Rules |
| --- | --- | --- |
| alias | string | `[a-z0-9][a-z0-9.-]{0,63}`, unique |
| repo | string | HF repo id `org/name` |
| revision | string | 40-hex commit sha (pinned) |
| files | list of {path, size, sha256} | every file needed to load; sha256 64-hex |
| license | string | SPDX id or "other" with URL |
| architecture | string | from `config.json` `model_type` (e.g. `qwen3`, `qwen3_moe`, `qwen3_5_moe`) |
| quantization | {bits, group_size, mode} | from `config.json` |
| identity | string | 001 identity function: sha256 over sorted (path, sha256); used by 001/003 |
| fast_check | list of {path, size, mtime} | compared on every use |

States: `missing` → `downloading` → `verifying` → `ready` | `corrupt` (hash mismatch) → (pull) →
`ready`. A compatible MLX repo found in the HF cache without a manifest is listed as `unpinned`;
`mrun pull <repo>` hashes it and writes the manifest without downloading.

## ClientProfile (compiled in)

| Field | Type |
| --- | --- |
| name | `claude-code` \| `codex` \| `opencode` |
| config_path | path template (`~/.claude/settings.json`, ...) |
| format | json \| toml |
| launch_env | env vars and flags for `mrun launch` (base URL, all model slots, context limits, auth reference, cache hints) |
| settings | key path → value template for `connect` (same values, persistent form) |
| probe | how to verify the client sees mrun (where possible) |

## Diagnostic

| Field | Type |
| --- | --- |
| id | stable string (`memory.pressure`, `client.claude.version`, ...) |
| status | ok \| warn \| fail \| skipped |
| severity | info \| warning \| error |
| detail | human text |
| fix | concrete command or setting |

## Config (`~/.mrun/config.toml`)

```toml
[server]
host = "127.0.0.1"
port = 8484
queue = 16
queue_wait_s = 600
thinking = "client"
max_output = 16384
reject_tool_images = false

[memory]
fraction = 0.75
# reserve_gib = clamp(physical ÷ 6, 2, 8)  (computed; 8 on a 48 GB Mac)

[engine]
default_model = "qwen3.6"

[cache]
enabled = true

[limits]
body_mib = 8
messages = 4096
tools = 256
schema_kib = 64
schema_depth = 16
```

Every key is optional; `mrun config` shows effective values and sources. Environment variables
are `MRUN_<SECTION>_<KEY>`; `MRUN_HOME` and `MRUN_API_KEY` are reserved names, not config keys (for example `MRUN_SERVER_PORT`, `MRUN_MEMORY_RESERVE_GIB`).
