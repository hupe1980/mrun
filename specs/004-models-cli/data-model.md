# Data Model: Models and CLI Experience

## ModelManifest (`~/.mrun/manifests/<alias>.toml`)

| Field | Type | Rules |
| --- | --- | --- |
| alias | string | `[a-z0-9][a-z0-9.-]{0,63}`, unique |
| repo | string | HF repo id `org/name` |
| revision | string | 40-hex commit sha (pinned) |
| files | list of {path, size, sha256} | every file needed to load; sha256 64-hex |
| license | string | SPDX id or "other" with URL |
| family | string | 001 family registry: `qwen3_5_moe` \| `qwen3_moe` \| `gpt_oss` \| `qwen3` (0.1) |
| syntax | string | the family's syntax (002 FR-032): `hermes` \| `qwen_xml` \| `harmony` |
| state_bytes_per_token | u64 | positional state per token (001 `kv_bytes_per_token`), measured on the target (010 FR-009) |
| checkpoint_bytes | u64 | position-bound state per checkpoint (001 `recurrent_bytes`); 0 without it |
| capacity_48gb | u32 | token capacity on the 48 GB target (aliases only; recorded at pin time) |
| architecture | string | from `config.json` `model_type` (e.g. `qwen3`, `qwen3_moe`, `qwen3_5_moe`, `gpt_oss`; wrapped `text_config.model_type` recorded too) |
| quantization | {bits, group_size, mode: affine \| mxfp4, overrides} | from `config.json` |
| identity | string | 001 identity function: sha256 over sorted (path, sha256); used by 001/003 |
| fast_check | list of {path, size, mtime} | compared on every use |

States: `missing` → `downloading` → `verifying` → `ready` | `corrupt` (hash mismatch) → (pull) →
`ready`. `mrun pull <repo>` of a repo already in the HF cache hashes it and writes the manifest
without downloading. `pull` of an unsupported architecture stops after `config.json` with no
manifest written. (`unpinned`/`unsupported` listing of cached repos: Later.)

## ClientProfile (compiled in)

| Field | Type |
| --- | --- |
| name | `claude-code` \| `codex` \| `opencode` |
| config_path | path printed by `connect` (`~/.claude/settings.json`, ...) |
| format | json \| toml |
| launch_env | env vars and flags for `mrun launch` (base URL, all model slots, context limits, idle-compaction floor, cache hints; Codex `web_search = "disabled"`) |
| settings | key path → value template printed by `connect` (same values, persistent form) |

## Diagnostic

| Field | Type |
| --- | --- |
| id | stable string (`memory.pressure`, `client.claude.version`, ...) |
| status | ok \| warn \| fail \| skipped |
| severity | info \| warning \| error |
| detail | human text |
| fix | concrete command or setting |

## Settings (no config file in 0.1)

Defaults < `MRUN_*` environment < flags. Settings and defaults: `MRUN_SERVER_PORT` 8484,
`MRUN_SERVER_QUEUE` 16, `MRUN_SERVER_QUEUE_WAIT_S` 600, `MRUN_SERVER_MAX_OUTPUT` 16384,
`MRUN_SERVER_REJECT_TOOL_IMAGES` false, `MRUN_MEMORY_FRACTION` 0.75, `MRUN_MEMORY_RESERVE_GIB`
(computed: clamp(physical ÷ 6, 2, 8)), `MRUN_ENGINE_DEFAULT_MODEL` `qwen3.6`, `MRUN_CACHE_ENABLED`
true, `MRUN_LIMITS_{BODY_MIB, MESSAGES, TOOLS, SCHEMA_KIB, SCHEMA_DEPTH}` 64/4096/256/64/16.
`MRUN_HOME` and (007) `MRUN_CACHE_DIR` are paths. `~/.mrun/config.toml` and `mrun config` are
Later (FR-010).
