# Contract: `GET /metrics`

Prometheus text exposition format 0.0.4 (`Content-Type: text/plain; version=0.0.4`). Same bind,
Host/Origin and API-key rules as every endpoint (002). Label values come only from the fixed sets
below; no prompt, session, header or path data appears. Names are stable within a minor release;
renames are breaking changes listed in the changelog.

Label sets: `api` ∈ {`chat`, `messages`, `responses`}; `outcome` ∈ {`ok`, `cancelled`, `error`,
`rejected`}; `cause` ∈ 003 miss causes; `reason` ∈ {`lru`, `pressure`, `clear`};
`component` as listed.

Before the model is loaded (service mode without `--preload`, D022), only `mrun_build_info`,
`mrun_start_time_seconds`, `mrun_model_loaded` (0), request/queue and power series are exported;
`mrun_model_info`, memory (001) and cache series are absent until loaded.

## Build and model

| Name | Type | Labels | Meaning |
| --- | --- | --- | --- |
| `mrun_build_info` | gauge (1) | `version`, `commit`, `mlx_version` | Build identity |
| `mrun_model_info` | gauge (1) | `model`, `identity` (first 12 hex), `architecture`, `family`, `syntax`, `tier`, `quantization` (e.g. `affine4`, `mxfp4+affine8`) | Loaded model |
| `mrun_start_time_seconds` | gauge | — | Server start (Unix time) |
| `mrun_model_loaded` | gauge | — | 1 once the model is loaded, else 0 |

## Requests and latency

| Name | Type | Labels | Meaning |
| --- | --- | --- | --- |
| `mrun_requests_total` | counter | `api`, `outcome` | Finished requests |
| `mrun_time_to_first_token_seconds` | histogram | `api` | Arrival to first content token (server-side); buckets 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120, 300 |
| `mrun_queue_wait_seconds` | histogram | — | Arrival to start; same buckets |
| `mrun_queue_depth` | gauge | — | Requests waiting |
| `mrun_active_requests` | gauge | — | 0 or 1 |

## Throughput

| Name | Type | Labels | Meaning |
| --- | --- | --- | --- |
| `mrun_prompt_tokens_total` | counter | — | Prompt tokens received |
| `mrun_prefill_tokens_total` | counter | — | Prompt tokens processed (not cached) |
| `mrun_prefill_seconds_total` | counter | — | Time spent in prefill; `rate(tokens)/rate(seconds)` = prefill tok/s |
| `mrun_generated_tokens_total` | counter | — | Output tokens committed |
| `mrun_decode_seconds_total` | counter | — | Time spent decoding |

## Prefix cache (003)

| Name | Type | Labels | Meaning |
| --- | --- | --- | --- |
| `mrun_cache_lookups_total` | counter | — | Cache queries (one per request with cache on) |
| `mrun_cache_hit_tokens_total` | counter | `tier` (`memory`, `disk`) | Prompt tokens reused |
| `mrun_cache_misses_total` | counter | `cause` | Requests whose miss report has a cause other than `none` |
| `mrun_cache_evictions_total` | counter | `reason` | Branches evicted |
| `mrun_cache_bytes` | gauge | `tier` | Bytes held (`disk` only with 007) |
| `mrun_cache_branches` | gauge | — | Stored branches in memory |
| `mrun_cache_live_tokens` | gauge | — | Tokens on the live path |

## Memory (001)

| Name | Type | Labels | Meaning |
| --- | --- | --- | --- |
| `mrun_memory_budget_bytes` | gauge | `component` ∈ {`budget`, `weights`, `scratch`, `reserve`, `state_capacity`} | Startup budget breakdown |
| `mrun_memory_state_bytes` | gauge | `component` ∈ {`active`, `cache`} | Sequence state in use |
| `mrun_state_capacity_tokens` | gauge | — | Context capacity on this machine |
| `mrun_mlx_memory_bytes` | gauge | `kind` ∈ {`active`, `cache`, `peak`} | MLX allocator counters |
| `mrun_memory_pressure_events_total` | counter | `level` ∈ {`warn`, `critical`} | OS pressure signals |

## Power (008)

| Name | Type | Labels | Meaning |
| --- | --- | --- | --- |
| `mrun_sleep_assertion_active` | gauge | — | 1 while the idle-sleep assertion is held |

## Reserved for later features

`mrun_spec_drafted_tokens_total{drafter}`, `mrun_spec_accepted_tokens_total{drafter}`,
`mrun_spec_rounds_total{drafter}` with `drafter` ∈ {`ngram`} (006); `mrun_cache_disk_writes_total`,
`mrun_cache_disk_write_bytes_total`, `mrun_cache_disk_corrupt_total` (007).
