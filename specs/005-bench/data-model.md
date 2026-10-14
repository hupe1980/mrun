# Data Model: Benchmark Harness

## Suite and Cell

| Field | Type | Notes |
| --- | --- | --- |
| suite | `agent` \| `synthetic` \| `cancel` \| `memory` \| `quality` | |
| cell | string | e.g. `agent/claude-code-refactor`, `synthetic/prefill-16k` |
| inputs | trace file, prompt length, output length, task id | |
| repetitions | u32 ≥ 5 | |

## SystemUnderTest

| Field | Type |
| --- | --- |
| name | `mrun` \| `llama.cpp` \| `ollama` \| `omlx` \| `mlx-lm` \| custom |
| version | string (reported by the system or given by the user) |
| endpoint | URL + supported API styles (`chat` \| `messages` \| `responses`) |
| flags | string list (from profile, editable) |
| model | {name sent to the system, format (`mlx` \| `gguf`), quantization {bits, group_size, scheme}, file hashes when local} |
| pid | optional, for memory sampling |

## RunResult (JSON, `schema_version: 1`)

| Field | Type |
| --- | --- |
| schema_version | 1 |
| started_at, finished_at | RFC 3339 |
| mrun_version | string |
| machine | {chip, memory_bytes, macos, power_source, low_power, thermal_start, thermal_end, per_system: [{system, free_memory, swap}]} |
| systems | SystemUnderTest[] |
| cells | [{system, cell, api, samples: number[], median, p95, iqr, unit, early_eos: bool, failures: [{rep, error}], turns?: [{n, ttft_ms, tfut_ms, input, cached, output}]}] |
| quality | [{system, task, seed, passed, duration_s}] |
| comparability | [{cell, kind: `same-weights` \| `quality-matched` \| `n/a`}] |
| method | {replay: `teacher-forced`, reps: {synthetic, cancel, agent, quality_seeds}} |
| warnings | string[] (battery, throttling, load) |

## Baseline (regression mode)

A stored RunResult plus thresholds: `max_speed_regression_pct` (default 10), `max_quality_drop`
(default 1 task of the suite). Greedy-output conformance is checked by 001's `mrun run --check`.
