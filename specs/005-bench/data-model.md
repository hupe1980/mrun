# Data Model: Benchmark Harness

## Suite and Cell

| Field | Type | Notes |
| --- | --- | --- |
| suite | `agent` \| `synthetic` \| `cancel` | `cancel` runs for mrun only |
| cell | string | e.g. `agent/opencode-refactor`, `synthetic/prefill-32k`, `synthetic/prefill-max` |
| inputs | trace file, prompt length, output length | |
| repetitions | u32 | synthetic ≥ 5 (+ 1 warm-up), cancel 50 points, agent 3 (FR-004) |

## SystemUnderTest

| Field | Type |
| --- | --- |
| name | `mrun` \| `llama.cpp` \| `ollama` \| `omlx` \| `mlx-lm` \| `splash` \| custom |
| version | string (reported by the system or given by the user) |
| endpoint | URL + supported API styles (`chat` \| `messages` \| `responses`) |
| profile | {health?, reset: HTTP call \| manual restart pause} (switching is always a manual pause in 0.1; start/stop commands Later) |
| context_length | u32 (= mrun's capacity, FR-015) |
| flags | string list (from profile, editable) |
| model | {name sent to the system, format (`mlx` \| `gguf`), quantization {bits_per_weight, group_size, scheme}, kv_format (`bf16` \| `f16` \| `int8` \| …), file hashes when local} |

## Trace

JSONL built by `cargo xtask bench-trace` from `mrun serve --record` output, one record per
request in arrival order: `{api, conversation, body, prompt_tokens, reply_tokens, tool_gaps?}`.
`body` is the request exactly as the client sent it (redacted with the trace's single mapping);
`conversation` groups requests (session hint or longest shared prefix); `prompt_tokens` is the
expected count for the truncation check; `reply_tokens` is the reply length used as `max_tokens`,
derived from the next request's history (the conversation's median for its last request);
`tool_gaps` keeps `x-claude-code-prev-tool-durations` when recorded (not replayed). Recorded
timestamps are used only for ordering and are not stored.

## RunResult (JSON, `schema_version: 1`)

| Field | Type |
| --- | --- |
| schema_version | 1 |
| started_at, finished_at | RFC 3339 |
| mrun_version | string |
| machine | {chip, memory_bytes, macos, power_source, low_power, per_system: [{system, free_memory_before, swap}]} |
| systems | SystemUnderTest[] |
| resets | [{system, rep, step, ok}] |
| cells | [{system, cell, api, status: `ok` \| `failed` \| `not-run` \| `n/a`, samples: number[], min, median, max, p95?, iqr?, unit, thermal_start, thermal_end, early_eos: n, malformed_tool_calls: n, template_rerenders?: {turns, tokens}, failures: [{rep, error}], turns?: [{n, conversation, kind: `new` \| `repeated` \| `branching`, ttft_ms, tfut_ms, input, cached, output, cause?, message_index?}], session_s?}] |
| comparability | [{cell, kind: `same-weights` \| `different-quantization` \| `lossy-kv` \| `n/a`}] |
| method | {replay: `in-order`, reps: {synthetic, cancel, agent}} |
| warnings | string[] (battery, thermal state not nominal, load, free memory lower than before the first system) |

Turn kinds (SC-005): `repeated` = a request that appends to the previous request of the same
conversation; `branching` = the first request after a request from a different conversation;
`new` = a conversation's first request (takes precedence). `p95` and `iqr` are present only when n ≥ 20. `failures[].error` includes `truncated input`.

## Baseline (regression mode): Later

Removed from 0.1 (2026-10-15, FR-010). When it returns: a stored RunResult plus thresholds
(`max_speed_regression_pct` default 10 on medians, `max_malformed_tool_calls`, `max_early_eos`).
