+++
title = "API reference"
description = "mrun's HTTP endpoints: OpenAI Chat Completions, Anthropic Messages, OpenAI Responses, model listing, status and error codes."
weight = 70
+++

Base URL: `http://127.0.0.1:8484`. Request and response bodies follow the upstream API
definitions; this page lists what mrun serves and where it deliberately differs.

## Endpoints

| Method and path | Purpose |
| --- | --- |
| `POST /v1/chat/completions` | OpenAI Chat Completions: streaming, tools, `response_format`, `reasoning_content` |
| `POST /v1/messages` | Anthropic Messages: streaming, tools, thinking |
| `POST /v1/messages/count_tokens` | Anthropic token count; exact, never queued |
| `POST /v1/responses` | OpenAI Responses, stateless |
| `GET /v1/models` | The served model with this machine's `context_window` and default `max_output_tokens` |
| `GET /health` | `{"status":"ok","model":...}`; 503 while loading |
| `GET /mrun/status` | Everything `mrun ps` shows, as JSON |
| `POST /mrun/cache/clear` | Clear the prefix cache |

## Behavior worth knowing

- **Any model name works.** The loaded model serves every request; responses report its real id.
- **Output limit.** Requests without one get 16,384 tokens; larger values are clamped to what fits.
- **Stateless Responses.** `previous_response_id` and item references return 400, because nothing
  is stored. Clients resend the conversation, and the prefix cache makes that cheap.
- **Queueing.** One request runs at a time; others wait up to 600 seconds in a queue of 8. Queued
  streams receive keepalives every 10 seconds.
- **Disconnect cancels.** Closing the connection stops queued or running work at once.
- **Invalid tool arguments.** When the model emits tool-call arguments that aren't valid JSON, mrun
  reports them in the way each API allows and adds a `tool-args-invalid` warning.
- **Cache diagnostics.** Anthropic requests with `diagnostics.previous_message_id` receive the
  standard `diagnostics.cache_miss_reason` (`system_changed`, `tools_changed`, `messages_changed`,
  and others).
- **No server-side context editing.** Anthropic `context_management` is rejected with 400; mrun
  never rewrites or summarizes your conversation.
- **Images.** The supported models are text-only. Images inside tool results are replaced with a
  short placeholder and an `image-omitted` warning; `--reject-tool-images` makes them an error.

## mrun metadata

Response bodies stay standard. Additional information travels beside them:

| | Non-streaming | Streaming |
| --- | --- | --- |
| Cache report | `x-mrun-cache-cached`, `x-mrun-cache-processed`, `x-mrun-cache-cause`, `x-mrun-cache-divergence` | `: mrun {...}` SSE comment before the first event |
| Output limit | `x-mrun-max-tokens` | same comment |
| Warnings | `x-mrun-warning` (comma-separated) | `: mrun {...}` comment before the final event |

## Errors

| Condition | Chat Completions / Responses | Anthropic Messages |
| --- | --- | --- |
| Queue full | 429 `rate_limit_exceeded` with `retry-after` | 529 `overloaded_error` with `retry-after` |
| Waited too long in the queue | 503 `server_busy` | 504 `timeout_error` |
| Prompt does not fit | 400 `context_length_exceeded` | 400 `invalid_request_error` ("prompt is too long") |
| Unsupported input | 400 `invalid_request_error` with `param` | 400 `invalid_request_error` |
| Host or Origin rejected | 403 | 403 |
| Missing or wrong API key | 401 | 401 `authentication_error` |

## Security

mrun binds to `127.0.0.1`. It accepts only loopback `Host` headers and rejects browser requests
from other origins, which blocks DNS-rebinding and drive-by requests from web pages. Binding to
another address with `--host` requires an API key (`MRUN_API_KEY` or `--api-key-file`), sent as
`Authorization: Bearer` or `x-api-key`.
