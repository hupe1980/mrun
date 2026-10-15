# Contract: HTTP API

Base URL `http://127.0.0.1:8484`. Wire-level formats (fields, SSE event order, error shapes) follow
the upstream APIs as documented in [research §1](../research.md#1-wire-formats); this contract
lists what mrun serves and where it intentionally differs.

| Method & path | Purpose | Notes |
| --- | --- | --- |
| `GET /health` | Liveness + readiness | `{"status":"ok","model":...}`; 503 while loading |
| `HEAD /api/hello` | Claude Code probe | 200, empty |
| `GET /v1/models` | Served models | `context_window` and `max_model_len` = this machine's KV capacity (vLLM-style field read by OpenCode's local discovery); `max_output_tokens` = server default |
| `POST /v1/chat/completions` | OpenAI Chat Completions | stream + non-stream, tools, `response_format`, `reasoning_content` |
| `POST /v1/messages` | Anthropic Messages | query string ignored (`?beta=true`); thinking with opaque signatures; `ping` every 10 s of silence; `diagnostics` and `context_management` accepted and ignored |
| `POST /v1/messages/count_tokens` | Anthropic token count | no queue; exact tokens mrun would process |
| `POST /v1/responses` | OpenAI Responses (stateless) | `previous_response_id` and item references → 400; `store:true` accepted, nothing stored; `response.in_progress` every 10 s of silence |
| `GET /mrun/status` | `mrun ps` data | model, budget, active/queued, cache stats, last miss (003) |
| `POST /mrun/cache/clear` | Feature 003 | |

Cross-cutting:
- **Host/Origin**: only loopback `Host` values are accepted unless `--host` is set; requests with an
  `Origin` header other than the server's own are rejected with 403; no CORS headers are sent.
- **Auth**: none on loopback; with a non-loopback `--host`, an API key from `MRUN_API_KEY` or
  `--api-key-file` is required and checked (constant time) as `Authorization: Bearer` or
  `x-api-key`.
- **Unknown model ids**: served by the loaded model; responses report the real id.
- **Output limit**: absent → server default (16,384); too large → clamped to remaining capacity
  (001 FR-009) and reported as `x-mrun-max-tokens` (non-streaming) or in the first `: mrun` comment
  (streaming), with warning `max-tokens-clamped`.
- **mrun metadata** `{cached, processed, cause, divergence, gap, max_tokens, warnings[]}`
  (`gap`: tokens re-prefilled between a hybrid checkpoint and the match, 003) (warnings:
  `tool-args-invalid`, `tool-unknown`, `image-omitted`, `max-tokens-clamped`,
  `strict-unsupported`, `history-args-raw`):
  - non-streaming: headers `x-mrun-cache-cached`, `x-mrun-cache-processed`, `x-mrun-cache-cause`,
    `x-mrun-cache-divergence`, `x-mrun-cache-gap`, `x-mrun-max-tokens`, `x-mrun-warning` (comma-separated);
  - streaming: SSE comment lines `: mrun {json}` before the first content event (cache, limit) and
    before the final event (warnings); clients ignore comments;
  - bodies stay strictly standard; the last request's metadata is in `/mrun/status`.
- **Usage**: `cached = min(matched, P − 1)`; Anthropic `input_tokens` excludes cache reads,
  `cache_creation_input_tokens` is always 0 (FR-007).
- **Ignored fields** (FR-023 removed, FR-024, FR-030): Anthropic `diagnostics` (no
  `diagnostics` in responses) and `context_management`; `cache_control` anywhere;
  `prompt_cache_key`; client session headers (used only by `--record`).
- **Attribution nonce** (FR-025): the `cch=` value of Claude Code's `x-anthropic-billing-header` system text is fixed before rendering; nothing else is rewritten.
- **Progress** (FR-026): Chat keepalive comments carry `: mrun {"queued":n}` or `: mrun {"prefill":[done,total]}`; the same is in `/mrun/status` `active.progress`. Anthropic and Responses keepalives carry none.
- **Codex namespaces** (FR-027): `namespace` tools become functions `<ns>__<tool>`; calls are returned with `name` and `namespace` restored.
- **Deferred tools** (FR-028): `defer_loading` tools are not rendered upfront; `tool_reference` blocks and `tool_search_output` lists render the definition inline in the tool result. Behind a custom base URL Claude Code's tool search is off unless `ENABLE_TOOL_SEARCH` is set; `mrun launch` sets `ENABLE_TOOL_SEARCH=auto:10` (defer when tool definitions exceed 10% of the context window).
- **Rejected server-side features** (FR-029): `mcp_servers`, `container`, `mcp_toolset`, `tool_search_tool_*`, `tool_addition`/`tool_removal` → 400 naming the field.
- **Per-request cache off**: request header `x-mrun-cache: off` on all APIs (003 FR-009).
- **Strict schemas** (FR-004/005): a schema llguidance cannot compile runs unconstrained with warning `strict-unsupported`.
- **`tool_choice: none`**: tools stay rendered; a fixed −inf logit bias on tool-call opener tokens prevents calls (decode stays pipelined).
- **Disconnect** (FR-011): a per-connection close watcher cancels the request in any phase (queued, prefilling, generating, non-streaming) within 300 ms.
- **Queued and prefilling streams** (FR-010): headers at once, then until the first token a keepalive at least every 10 s:

  | API | First event | Keepalive | Why |
  | --- | --- | --- | --- |
  | Responses | `response.created` | `response.in_progress` | Codex's `eventsource-stream` drops SSE comments; 300 s idle timeout (`codex-api/src/sse/responses.rs:557`) |
  | Anthropic | — | `ping` | Claude Code aborts after 300 s without bytes |
  | Chat | — | SSE comment (with progress) | clients ignore comments; bytes keep sockets alive |

## Status codes

Queue holds 16 waiting by default. Before headers are sent the HTTP status applies; afterwards
(streaming requests get headers at once) the error is delivered in-stream and the stream ends.

| Condition | Chat Completions | Anthropic Messages | Responses |
| --- | --- | --- | --- |
| Queue full | 429 `rate_limit_exceeded`, `retry-after` | 529 `overloaded_error`, `retry-after` | 429 `rate_limit_exceeded`, `retry-after` |
| Queue deadline (600 s) | 503 `server_busy` | 504 `timeout_error` | 503 `server_busy` |
| Prompt does not fit | 400 `context_length_exceeded` | 400 `invalid_request_error` ("prompt is too long: N tokens > M") | stream: always `response.failed` code `context_length_exceeded` (Codex maps only this in-stream form to compaction, `codex-api/src/sse/responses_error.rs:48`; an HTTP 400 is a plain error); non-stream 400 |
| Unsupported input | 400 `invalid_request_error` with `param` path | 400 `invalid_request_error` | 400 `invalid_request_error` with `param` |
| Host/Origin rejected | 403 | 403 | 403 |
| Bad/missing API key | 401 | 401 `authentication_error` | 401 |

**Streaming (after headers)**:

| Condition | Chat Completions | Anthropic Messages | Responses |
| --- | --- | --- | --- |
| Queue deadline, engine failure | `data: {"error":{"type":"server_error","code":"server_busy",…}}`, then close (no `[DONE]`) | `event: error` `{"type":"error","error":{"type":"overloaded_error",…}}` | `response.failed`, `error.code` `server_busy` |
| Prompt does not fit (Chat/Anthropic only when found after headers) | `data: {"error":{…,"code":"context_length_exceeded"}}` | `event: error` `invalid_request_error` ("prompt is too long: …") | `response.failed`, `error.code` `context_length_exceeded` (Codex compacts) |

Validation, queue-full, Host/Origin and auth errors happen before headers and always use the status
codes above.

`retry-after` is integer seconds, 1–60 (Claude Code requires it and stops retrying above 60).

## `GET /mrun/status` (schema_version 1)

`{schema_version, version, model: {id, identity, architecture, quantization}, budget: {budget,
weights, scratch, kv_capacity_tokens, kv_bytes_per_token}, active: {id, api, prompt_tokens,
cached, phase, progress} | null, queued: n, cache: CacheStats (003), recording: bool,
last: {cached, processed, cause, divergence, max_tokens, warnings[]} | null}`
