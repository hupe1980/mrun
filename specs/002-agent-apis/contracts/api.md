# Contract: HTTP API

Base URL `http://127.0.0.1:8484`. Wire-level formats (fields, SSE event order, error shapes) follow
the upstream APIs as documented in [research §1](../research.md#1-wire-formats); this contract
lists what mrun serves and where it intentionally differs.

| Method & path | Purpose | Notes |
| --- | --- | --- |
| `GET /health` | Liveness + readiness | `{"status":"ok","model":...,"loaded":bool}`; 503 while loading |
| `HEAD /api/hello` | Claude Code probe | 200, empty |
| `GET /v1/models` | Served models | `context_window` and `max_model_len` = this machine's KV capacity (vLLM-style field read by OpenCode's local discovery); `max_output_tokens` = server default |
| `POST /v1/chat/completions` | OpenAI Chat Completions | stream + non-stream, tools, `response_format`, `reasoning_content` |
| `POST /v1/messages` | Anthropic Messages | query string ignored (`?beta=true`); `anthropic-beta` headers accepted (FR-017); thinking with opaque signatures; `ping` every 10 s of silence; `diagnostics` and `context_management` accepted and ignored |
| `POST /v1/messages/count_tokens` | Anthropic token count | no queue; exact tokens mrun would process |
| `POST /v1/responses` | OpenAI Responses (stateless) | `previous_response_id` and item references → 400; `store:true` accepted, nothing stored; `response.in_progress` every 10 s of silence |
| `GET /mrun/status` | `mrun ps` data | model, budget, active/queued, cache stats, last miss (003) |
| `POST /mrun/cache/clear` | Feature 003 | |

Cross-cutting:
- **Host/Origin**: the server binds loopback only; only loopback `Host` values are accepted;
  requests with an `Origin` header other than the server's own are rejected with 403; no CORS
  headers are sent.
- **Auth**: none (loopback only). Non-loopback binds with API keys are Later (FR-012).
- **Unknown model ids**: served by the loaded model; responses report the real id.
- **Output limit**: absent → server default (Chat 16,384; Responses `capacity − prompt`, FR-019); too large → clamped to remaining capacity
  (001 FR-009) and reported as `x-mrun-max-tokens` (non-streaming) or in the first `: mrun` comment
  (streaming), with warning `max-tokens-clamped`.
- **mrun metadata** `{cached, processed, cause, divergence, message: {index, role}, gap,
  max_tokens, warnings[]}` (causes `new`, `changed`, `evicted`, `model`, `disabled`; `gap`: tokens
  re-prefilled between a hybrid checkpoint and the match, 003; warnings: `tool-args-invalid`,
  `max-tokens-clamped`, `image-omitted`; other conditions go to the server log):
  - non-streaming: headers `x-mrun-cache-cached`, `x-mrun-cache-processed`, `x-mrun-cache-cause`,
    `x-mrun-cache-divergence`, `x-mrun-cache-message` (`<index>:<role>`), `x-mrun-cache-gap`,
    `x-mrun-max-tokens`, `x-mrun-warning` (comma-separated);
  - streaming: SSE comment lines `: mrun {json}` before the first content event (cache, limit) and
    before the final event (warnings); clients ignore comments;
  - bodies stay strictly standard; the last request's metadata is in `/mrun/status`.
- **Usage**: `cached = min(resume, P − 1)`; resume = match (attention) or restored checkpoint position (hybrid), per 003 FR-006; Anthropic `input_tokens` excludes cache reads,
  `cache_creation_input_tokens` is always 0 (FR-007).
- **Ignored fields** (FR-023 removed, FR-024, FR-030): Anthropic `diagnostics` (no
  `diagnostics` in responses) and `context_management`; `cache_control` anywhere;
  `prompt_cache_key`; client session headers (used only by `--record`); `x-mrun-cache` (removed
  2026-10-15; `--no-cache` is the only switch).
- **Attribution nonce** (FR-025): the `cch=` value of Claude Code's `x-anthropic-billing-header` system text is fixed before rendering; nothing else is rewritten.
- **Progress**: `/mrun/status` `active.phase` and `active.progress` (FR-026's keepalive progress is removed).
- **Codex namespaces** (FR-027): `namespace` tools become functions `<ns>__<tool>`; calls are returned with `name` and `namespace` restored.
- **Deferred tools** (FR-028): `defer_loading` tools are not rendered upfront; `tool_reference` blocks and `tool_search_output` lists render the definition inline in the tool result. Behind a custom base URL Claude Code's tool search is off unless `ENABLE_TOOL_SEARCH` is set; `mrun launch` sets `ENABLE_TOOL_SEARCH=auto:10` (defer when tool definitions exceed 10% of the context window).
- **Rejected server-side features** (FR-029): `mcp_servers`, `container`, `mcp_toolset`, `tool_search_tool_*`, `tool_addition`/`tool_removal` → 400 naming the field.
- **Benchmark control**: `x-mrun-ignore-eos: on` makes generation run to the output limit, so `mrun bench` gives every system the same decode work (005 FR-007).
- **Strict schemas** (FR-004/005): a schema llguidance cannot compile runs unconstrained (logged `strict-unsupported`); grammar-forced tokens are appended without sampling.
- **`tool_choice: none`**: tools stay rendered; a fixed −inf logit bias on tool-call opener tokens prevents calls (decode stays pipelined).
- **Disconnect** (FR-011): a per-connection close watcher cancels the request in any phase (queued, prefilling, generating, non-streaming) within 300 ms.
- **Keepalives** (FR-010): streaming responses send headers at once; then, in every phase (queued, prefilling, generating, including while a tool call is buffered), a keepalive whenever 10 s pass without an event:

  | API | First event | Keepalive | Why |
  | --- | --- | --- | --- |
  | Responses | `response.created` | `response.in_progress` | Codex's `eventsource-stream` drops SSE comments; 300 s idle timeout (`codex-api/src/sse/responses.rs:557`) |
  | Anthropic | — | `ping` | Claude Code aborts after 300 s without bytes |
  | Chat | — | SSE comment | clients ignore comments; bytes keep sockets alive |

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

**Streaming (after headers)**:

| Condition | Chat Completions | Anthropic Messages | Responses |
| --- | --- | --- | --- |
| Queue deadline, engine failure | `data: {"error":{"type":"server_error","code":"server_busy",…}}`, then close (no `[DONE]`) | `event: error` `{"type":"error","error":{"type":"overloaded_error",…}}` | `response.failed`, `error.code` `server_busy` |
| Prompt does not fit (Chat/Anthropic only when found after headers) | `data: {"error":{…,"code":"context_length_exceeded"}}` | `event: error` `invalid_request_error` ("prompt is too long: …") | `response.failed`, `error.code` `context_length_exceeded` (Codex compacts) |

Validation, queue-full and Host/Origin errors happen before headers and always use the status
codes above.

`retry-after` is integer seconds, 1–60 (Claude Code requires it and stops retrying above 60).

## `GET /mrun/status` (schema_version 1)

`{schema_version, version, model: {id, identity, architecture, quantization}, budget: {budget,
weights, scratch, kv_capacity_tokens, kv_bytes_per_token}, active: {id, api, prompt_tokens,
cached, phase, progress} | null, queued: n, cache: CacheStats (003), recording: bool,
power: {assertion: bool, since?} (008), last: {cached, processed, cause, divergence, message,
gap, max_tokens, warnings[]} | null}`. Post-0.1 features add `spec` (006) and `cache.disk` (007) as
additive fields.
