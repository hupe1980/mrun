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
- **Auth**: none on loopback; with `--host` non-loopback, `--api-key` required and checked as
  `Authorization: Bearer` or `x-api-key`.
- **Errors**: native shapes per API; `429` (queue full, `retry-after` integer seconds), `408`-style
  queue timeout mapped per API, `400` context too long with token numbers
  (`context_length_exceeded` / `invalid_request_error` / `response.failed` code).
- **Extension fields**: responses carry `mrun` metadata only where clients tolerate unknown fields:
  Chat `usage.mrun`, Responses `response.metadata.mrun`; Anthropic stays strictly standard (the
  same data is in `mrun ps`).
- **Disconnect**: dropping the connection cancels queued or running work.
