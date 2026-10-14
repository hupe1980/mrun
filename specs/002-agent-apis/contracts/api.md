# Contract: HTTP API

Base URL `http://127.0.0.1:8484`. Wire-level formats (fields, SSE event order, error shapes) follow
the upstream APIs as documented in [research §1](../research.md#1-wire-formats); this contract
lists what mrun serves and where it intentionally differs.

| Method & path | Purpose | Notes |
| --- | --- | --- |
| `GET /health` | Liveness + readiness | `{"status":"ok","model":...}`; 503 while loading |
| `HEAD /api/hello` | Claude Code probe | 200, empty |
| `GET /v1/models` | Served models | `context_window` = this machine's KV capacity |
| `POST /v1/chat/completions` | OpenAI Chat Completions | stream + non-stream, tools, `response_format`, `reasoning_content` |
| `POST /v1/messages` | Anthropic Messages | query string ignored (`?beta=true`); thinking with opaque signatures; `ping` every 10 s of silence |
| `POST /v1/messages/count_tokens` | Anthropic token count | no queue; exact tokens mrun would process |
| `POST /v1/responses` | OpenAI Responses (stateless) | `previous_response_id` and item references → 400; `store:true` accepted, nothing stored |
| `GET /mrun/status` | `mrun ps` data | model, budget, active/queued, cache stats, last miss (003) |
| `POST /mrun/cache/clear` | Feature 003 | |

Cross-cutting:
- **Host/Origin**: only loopback `Host` values are accepted unless `--host` is set; requests with an
  `Origin` header other than the server's own are rejected with 403; no CORS headers are sent.
- **Auth**: none on loopback; with a non-loopback `--host`, an API key from `MRUN_API_KEY` or
  `--api-key-file` is required and checked (constant time) as `Authorization: Bearer` or
  `x-api-key`.
- **Unknown model ids**: served by the loaded model; responses report the real id.
- **Output limit**: absent → server default (16,384); too large → clamped, reported in
  `x-mrun-max-tokens: <effective>`.
- **Response headers** (all APIs): `x-mrun-cache-cached`, `x-mrun-cache-processed`,
  `x-mrun-cache-cause`, `x-mrun-cache-divergence`, `x-mrun-warning` (comma-separated:
  `tool-args-invalid`, `image-omitted`, `max-tokens-clamped`). Bodies stay strictly standard.
- **Disconnect**: dropping the connection cancels queued or running work.
- **Queued streams**: keepalives every 10 s (Anthropic `ping`, SSE comment for OpenAI APIs).

## Status codes

| Condition | Chat Completions | Anthropic Messages | Responses |
| --- | --- | --- | --- |
| Queue full | 429 `rate_limit_exceeded`, `retry-after` | 529 `overloaded_error`, `retry-after` | 429 `rate_limit_exceeded`, `retry-after` |
| Queue deadline (600 s) | 503 `server_busy` | 504 `timeout_error` | 503 `server_busy` |
| Prompt does not fit | 400 `context_length_exceeded` | 400 `invalid_request_error` ("prompt is too long: N tokens > M") | stream: `response.failed` code `context_length_exceeded`; non-stream 400 |
| Unsupported input | 400 `invalid_request_error` with `param` path | 400 `invalid_request_error` | 400 `invalid_request_error` with `param` |
| Host/Origin rejected | 403 | 403 | 403 |
| Bad/missing API key | 401 | 401 `authentication_error` | 401 |

`retry-after` is integer seconds (Claude Code requires it).

## `GET /mrun/status` (schema_version 1)

`{schema_version, version, model: {id, identity, architecture, quantization}, budget: {budget,
weights, scratch, kv_capacity_tokens, kv_bytes_per_token}, active: {id, api, prompt_tokens,
cached, phase, progress} | null, queued: n, cache: CacheStats (003), recording: bool,
normalization: [..]}`
