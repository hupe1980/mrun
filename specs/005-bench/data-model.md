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
| endpoint | URL + API style (`openai` \| `anthropic`) |
| flags | string list (from profile, editable) |
| model | name + file hashes when local files are known |
| pid | optional, for memory sampling |

## RunResult (JSON, `schema_version: 1`)

| Field | Type |
| --- | --- |
| schema_version | 1 |
| started_at, finished_at | RFC 3339 |
| mrun_version | string |
| machine | {chip, memory_bytes, macos, power_source, low_power, thermal_start, thermal_end, swap_start, swap_end} |
| systems | SystemUnderTest[] |
| cells | [{system, cell, samples: number[], median, p95, iqr, ci95: [lo, hi], unit, failures: [{rep, error}]}] |
| quality | [{system, task, seed, passed, duration_s}] |
| comparability | [{cell, kind: `same-weights` \| `quality-matched`}] |
| warnings | string[] (battery, throttling, load) |

## Baseline (regression mode)

A stored RunResult plus thresholds: `max_speed_regression_pct` (default 10),
`conformance_must_match` (true), `max_quality_drop` (default 1 task of the suite).
