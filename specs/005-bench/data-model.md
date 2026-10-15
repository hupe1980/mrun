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
| name | `mrun` \| `llama.cpp` \| `ollama` \| `omlx` \| `mlx-lm` \| custom |
| version | string (reported by the system or given by the user) |
| endpoint | URL + supported API styles (`chat` \| `messages` \| `responses`) |
| profile | {start?, stop?, health?, reset, pid?: command} (no start/stop → manual pause) |
| context_length | u32 (= mrun's capacity, FR-015) |
| flags | string list (from profile, editable) |
| model | {name sent to the system, format (`mlx` \| `gguf`), quantization {bits_per_weight, group_size, scheme}, file hashes when local} |

## Trace

JSONL built by `cargo xtask bench-trace` from `mrun serve --record` output, one record per
request in arrival order: `{api, conversation, body, prompt_tokens, reply_tokens}`. `body` is the
request exactly as the client sent it (redacted with the trace's single mapping); `conversation`
groups requests (session hint or longest shared prefix); `prompt_tokens` is the expected count for
the truncation check; `reply_tokens` is the recorded reply length used as `max_tokens`. Recorded
timestamps are used only for ordering and are not stored.

## RunResult (JSON, `schema_version: 1`)

| Field | Type |
| --- | --- |
| schema_version | 1 |
| started_at, finished_at | RFC 3339 |
| mrun_version | string |
| machine | {chip, memory_bytes, macos, power_source, low_power, thermal_start, thermal_end, per_system: [{system, free_memory_before, free_memory_after_stop, swap, peak_rss, peak_footprint, swap_delta}]} |
| systems | SystemUnderTest[] |
| resets | [{system, rep, step, ok}] |
| cells | [{system, cell, api, status: `ok` \| `failed` \| `not-run` \| `n/a`, samples: number[], min, median, max, p95?, iqr?, unit, early_eos: n, malformed_tool_calls: n, failures: [{rep, error}], turns?: [{n, conversation, ttft_ms, tfut_ms, input, cached, output}], session_s?}] |
| comparability | [{cell, kind: `same-weights` \| `different-quantization` \| `n/a`}] |
| method | {replay: `in-order`, reps: {synthetic, cancel, agent}} |
| warnings | string[] (battery, throttling, load, memory not recovered) |

`p95` and `iqr` are present only when n ≥ 20. `failures[].error` includes `truncated input`.

## Baseline (regression mode)

A stored RunResult plus thresholds: `max_speed_regression_pct` (default 10, on medians),
`max_malformed_tool_calls` and `max_early_eos` (default: baseline counts). Greedy-output
conformance is checked by 001's `mrun run --check`.
