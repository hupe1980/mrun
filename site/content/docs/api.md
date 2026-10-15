+++
title = "API reference"
description = "mrun's HTTP endpoints: OpenAI Chat Completions, Anthropic Messages, OpenAI Responses, model listing, health, status and error codes."
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
| `GET /v1/models` | The served model with this machine's `context_window` (also as `max_model_len`) and default `max_output_tokens` |
| `GET /health` | `{"status":"ok","model":...,"loaded":bool}`; 503 while loading |
| `GET /mrun/status` | Everything `mrun ps` shows, as JSON |
| `POST /mrun/cache/clear` | Clear the prefix cache |

## Behavior worth knowing

- **Any model name works.** The loaded model serves every request; responses report its real id.
- **Output limit.** Chat Completions requests without one get 16,384 tokens; Responses requests
  without one may use all the space left after the prompt (Codex never sends a limit). Larger
  values are clamped to the space left and reported in `x-mrun-max-tokens`, with a
  `max-tokens-clamped` warning when the limit came from your request. Only a prompt that leaves no
  room at all is refused.
- **Stateless Responses.** `previous_response_id` and item references return 400, because nothing
  is stored. Clients resend the conversation, and the prefix cache makes that cheap.
- **Queueing.** One request runs at a time; others wait up to 600 seconds in a queue of 16. While a
  request waits or its prompt is processed, streams receive keepalives every 10 seconds in each
  API's own form (Anthropic `ping`, Responses `response.in_progress`, Chat SSE comments).
- **Disconnect cancels.** Closing the connection stops queued or running work at once.
- **Strict tools and structured output.** Tools marked `strict`, `tool_choice` set to required
  (`any`) or a named tool, and JSON output formats (`response_format`, `text.format`,
  `output_config.format`) are enforced while the model writes: it can only produce text the
  schema allows, and fixed parts such as the call envelope are inserted without sampling. A
  schema the decoder cannot compile runs unconstrained and is logged. `tool_choice: none` keeps
  the model from starting a call at all.
- **Invalid tool arguments.** When the model emits arguments that aren't valid JSON for a
  non-strict tool, mrun passes them on raw (Chat Completions, Responses) or as text (Anthropic,
  which has no raw form) and adds a `tool-args-invalid` warning.
- **Thinking.** Models think only when the request asks for it (Anthropic `thinking`, OpenAI
  `reasoning_effort` or `reasoning.effort` above `minimal`). An Anthropic `budget_tokens` is
  enforced: when it is spent, mrun closes the thinking block and the model answers.
- **MCP and deferred tools.** Codex namespaces are flattened and restored; `defer_loading` tools
  and `tool_reference` results are handled; server-executed tools are refused. See
  [MCP and skills](@/docs/mcp-and-skills.md).
- **No server-side context editing.** Anthropic `context_management` is accepted and ignored; mrun
  never rewrites or summarizes your conversation. Clients compact on their own at the reported
  context window.
- **Images.** mrun 0.1 does not pass images to the model. Images in user messages are rejected
  naming the field; images inside tool results are replaced with a short placeholder and an
  `image-omitted` warning (`--reject-tool-images` makes them an error). Image input is on the
  [roadmap](@/docs/roadmap.md).

## mrun metadata

Response bodies stay standard. Additional information travels beside them:

| | Non-streaming | Streaming |
| --- | --- | --- |
| Cache report | `x-mrun-cache-cached`, `x-mrun-cache-processed`, `x-mrun-cache-cause`, `x-mrun-cache-divergence`, `x-mrun-cache-message` (index and role), `x-mrun-cache-gap` (hybrid models) | `: mrun {...}` SSE comment before the first event |
| Output limit | `x-mrun-max-tokens` | same comment |
| Warnings | `x-mrun-warning` (comma-separated) | `: mrun {...}` comment before the final event |

There are three warnings: `tool-args-invalid`, `max-tokens-clamped` and `image-omitted`. Other
conditions, such as a call to an undeclared tool or a schema the constrained decoder cannot
compile, are written to the server log (without prompt content).

One request header changes generation: `x-mrun-ignore-eos: on` makes the model write until the
output limit, for benchmarks that need a fixed output length.

## Errors

| Condition | Chat Completions / Responses | Anthropic Messages |
| --- | --- | --- |
| Queue full | 429 `rate_limit_exceeded` with `retry-after` (≤ 60 s) | 529 `overloaded_error` with `retry-after` (≤ 60 s) |
| Waited too long in the queue | 503 `server_busy`; once streaming: an in-stream error (`response.failed` for Responses) | 504 `timeout_error`; once streaming: an `error` event |
| Prompt does not fit | 400 `context_length_exceeded`; streaming Responses: an in-stream `response.failed` with that code, which Codex answers by compacting | 400 `invalid_request_error` ("prompt is too long") |
| Unsupported input | 400 `invalid_request_error` with `param` | 400 `invalid_request_error` |
| Host or Origin rejected | 403 | 403 |

## Security

mrun listens on `127.0.0.1` only. It accepts only loopback `Host` headers and rejects browser
requests from other origins, which blocks DNS-rebinding and drive-by requests from web pages. No
API key is needed or checked. Listening on other addresses, with an API key, is planned after the
first release.
