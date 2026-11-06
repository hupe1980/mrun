+++
title = "API reference"
description = "mrun's HTTP endpoints, behaviour, response headers, warnings and error codes."
weight = 70
+++

Base URL: `http://127.0.0.1:8484`. Request and response bodies follow the upstream API
definitions; this page lists what mrun serves and where it differs.

## Endpoints

| Method and path | Purpose |
| --- | --- |
| `POST /v1/messages` | Anthropic Messages: streaming, tools, thinking |
| `POST /v1/messages/count_tokens` | Anthropic token count: exact, never queued |
| `POST /v1/chat/completions` | OpenAI Chat Completions: streaming, tools, `response_format`, `reasoning_content` |
| `POST /v1/responses` | OpenAI Responses, stateless |
| `GET /v1/models` | The served model with this machine's `context_window` and default `max_output_tokens` |
| `GET /v1/models/{id}` | The same for one id, plus the model's sampling `defaults` and whether it thinks (`thinking`) |
| `GET /health` | `{"status":"ok","model":…,"loaded":true}`; 503 with `"status":"loading"` or `"status":"error"` while loading or after a failed load |
| `GET /mrun/status` | Everything `mrun ps` shows, as JSON ([below](#mrun-status)) |
| `GET /mrun/sessions` | Session replay: sessions, rows and the last 2,000 requests, counters only ([below](#sessions)) |
| `GET /mrun/requests/{id}/excerpt` | The text around where a request's prompt changed ([below](#sessions)) |
| `GET /mrun/pool` | This Mac's pool, Macs waiting to join and (`?browse=true`) pools nearby ([Mac pool](@/docs/pool.md)) |
| `GET /mrun/pool/plan?model=` | The plan for a model on the pool's Macs, with each Mac's download |
| `POST /mrun/pool/rejoin` · `POST /mrun/pool/replan` | A member back without a code; the host serves again on the Macs answering |
| `POST /mrun/cache/clear` | Drop all cached state; `200 {"branches_dropped":…,"bytes_freed":…}` ([below](#mrun-status)) |
| `GET /metrics` | Prometheus text format: requests, tokens, cache, queue, memory and latency (`mrun_*`); no content |

Query strings such as `?beta=true` and `anthropic-beta` headers are accepted on every route.

## Behavior worth knowing

- **Any model name works.** The loaded model serves every request, and responses echo the model
  id the request named.
- **Output limit.** Requests without a limit get the context left after the prompt, at most
  32,768 tokens (`--max-output`), on all three APIs. Any limit is clamped to the room left after
  the prompt; the applied value is reported in `x-mrun-max-tokens`, with a `max-tokens-clamped`
  warning when the limit came from the request. Only a prompt that leaves no room at all is
  refused.
- **Stateless Responses.** `previous_response_id`, `conversation` and item references are refused
  with 400; nothing is stored.
- **Queueing.** One request runs at a time; up to 16 others wait (`--queue`). Streaming requests
  wait as long as needed. A non-streaming request waits at most 240 seconds (`--queue-wait-s`) and
  is refused at once when its estimated wait is longer.
- **Keepalives.** Streams receive a keepalive after 10 seconds without a byte: Anthropic `ping`,
  Responses `response.in_progress`, Chat Completions an empty `chat.completion.chunk`
  (`"choices": []`).
- **Disconnect cancels.** Closing the connection stops queued or running work.
- **Repetition penalty.** `repetition_penalty` (above 0, default 1 = off) is accepted on all
  three APIs, as in mlx-lm: logits of the last 20 tokens (prompt and reply) are divided
  (positive) or multiplied (negative) by it before sampling.
- **Structured output and forced tool calls.** JSON output formats (`response_format`,
  `text.format`, `output_config.format`) and `tool_choice` set to required (`any`) or a named tool
  are enforced by masking tokens the grammar does not allow. Tools marked `strict` under
  `tool_choice: auto` are not enforced (the server logs them). `tool_choice: none` keeps the model
  from starting a call.
- **Invalid tool arguments.** When the model emits arguments that are not a valid JSON object
  (or, for a `strict` tool, do not match its schema), mrun passes them on raw (Chat Completions,
  Responses) or as text (Messages, which has no raw form) and adds a `tool-args-invalid` warning.
- **Thinking.** On Chat Completions and Responses a thinking model (Qwen3, Qwen3.5, Qwen3.6)
  thinks by default; `reasoning_effort: "none"` or `"minimal"` (Responses: `reasoning.effort`)
  turns it off. Chat Completions also accepts `chat_template_kwargs: {"enable_thinking": false}`
  (or `true`), which wins over `reasoning_effort`; other `chat_template_kwargs` keys are ignored.
  On Anthropic Messages the model thinks only when `thinking` is enabled or adaptive;
  `budget_tokens` is accepted and not enforced.
- **Runaway repetition.** When the last 512 output tokens repeat with a period of 128 tokens or
  less, generation stops with a `repetition-stopped` warning. The check runs every 16 tokens,
  never inside tool-call arguments, and is off with `x-mrun-ignore-eos: on`.
- **MCP and deferred tools.** Codex namespaces are flattened and restored; `defer_loading` tools
  and `tool_reference` results are handled; server-executed tools (`mcp_servers`, `container`)
  are refused. See [MCP and skills](@/docs/mcp-and-skills.md).
- **Unknown fields are ignored.** Fields mrun does not know are accepted and logged by name (never
  by content); fields that need something mrun cannot do are refused with 400, naming the field.
- **Failures stay local.** A request that fails on its own (for example a grammar that allows no
  token) gets a server error, and the prefix cache is kept; only an engine failure resets it.
- **Shutdown.** On `mrun stop` (SIGTERM) new and queued requests get the overload error, running
  ones get 5 seconds and are then cancelled.
- **No server-side context editing.** Anthropic `context_management` is accepted and ignored;
  clients compact on their own at the reported context window.
- **Images.** Images are not passed to the model. Images in user messages are refused, naming
  the field; images inside tool results are replaced by a placeholder with an `image-omitted`
  warning (`--reject-tool-images` refuses them instead).

## mrun metadata

mrun's own information travels in headers on non-streaming responses and in `: mrun {…}` SSE
comment lines in streams:

| Header | Meaning |
| --- | --- |
| `x-mrun-cache-cached` | Prompt tokens reused from cache |
| `x-mrun-cache-processed` | Prompt tokens computed for this request |
| `x-mrun-max-tokens` | The output limit applied |
| `x-mrun-spliced` | Prompt tokens taken from the model's own earlier replies (only when non-zero) |
| `x-mrun-cache-cause` | Miss cause: `new`, `diverged`, `evicted`, `gap` or `disabled` (absent on a full hit) |
| `x-mrun-cache-divergence` | Token position where the prompt left the cached state |
| `x-mrun-cache-message` | The message containing that position, as `index:role` |
| `x-mrun-cache-gap` | Tokens computed again between a restored checkpoint and the match (hybrid models, only when non-zero) |
| `x-mrun-cache-unreused` | Tokens shared with recently stored or dropped state that were computed again anyway (only when non-zero) |
| `x-mrun-cache-ring-wrapped` | `true` once the record of dropped state has overwritten old entries (absent otherwise) |
| `x-mrun-warning` | Comma-separated in-band warnings |

In a stream, the first comment carries `cached`, `processed`, `max_tokens`, `spliced` and the miss
report (`cause`, `divergence`, `message`, `gap`, `unreused`, `ring_wrapped`); a second comment
with `warnings` comes before the terminal event when there are any. See [Prefix cache](@/docs/prefix-cache.md) for the causes.

Warnings:

| Warning | When |
| --- | --- |
| `tool-args-invalid` | The model wrote tool arguments that are not a valid JSON object (or do not match a `strict` tool's schema) |
| `max-tokens-clamped` | The requested output limit did not fit and was lowered |
| `image-omitted` | An image inside a tool result was replaced by a placeholder |
| `repetition-stopped` | The loop guard ended a repeating output |

Other conditions, such as a call to an undeclared tool or a schema the constrained decoder cannot
compile, are written to the server log without prompt content.

Request header `x-mrun-ignore-eos: on` makes the model write until the output limit and turns
the loop guard off, for benchmarks.

## mrun status

`GET /mrun/status` returns the model, the memory budget (`budget`, `weights`, `scratch`,
`kv_capacity_tokens`, `kv_bytes_per_token`), the queue, the last miss report (`last`) and a `cache`
object:

| Field | Meaning |
| --- | --- |
| `live_tokens` | Tokens of the live conversation |
| `branches`, `branch_tokens`, `buffer_bytes` | Conversations moved aside, their tokens and bytes |
| `checkpoints`, `checkpoint_bytes` | Hybrid-model checkpoints and their bytes |
| `prompt_tokens_total`, `cached_tokens_total` | Prompt tokens since start, and how many were reused |
| `spliced_tokens_total` | Prompt tokens taken from the model's own earlier replies, summed over generated requests (`count_tokens` is not counted) |
| `unreused_tokens_total` | Tokens that could have been reused but were computed again |
| `splice` | Per client (from the `User-Agent`), assistant messages of the current turn seen and spliced |
| `evicted_total` | Dropped paths remembered for the `evicted` cause |
| `pressure_events` | macOS memory-pressure levels acted on |
| `enabled` | `false` under `--no-cache` |

`pool` is this Mac's pool (`null` without one): its role, name, model, `unencrypted`, members
with their state (`joined`, `downloading`, `loading`, `serving`, `lost`) and, on the host,
`degraded` when a Mac stopped answering, `recovery` (`wait`, `smaller`) while it waits for one,
`replanning`, and `queue` (who runs, how many wait per Mac).

`finished` lists the last 20 requests that left the queue, oldest first, each with
`prompt_tokens`, `cached`, `processed`, the miss fields of `last` (`cause`, `divergence`,
`message`, `unreused`), `generated`, `ttft_ms` (from arrival), `decode_tps` and `stop` (`end`,
`stopped`, `error`). It holds counters only, never content.

`POST /mrun/cache/clear` waits for a running request, then drops every branch, checkpoint and
the live conversation and answers `200` with `{"branches_dropped": N, "bytes_freed": B}`
(`mrun cache clear`). The next request reports cause `new`.

## Sessions

`GET /mrun/sessions[?since=<id>]` groups the last 2,000 finished requests into sessions: requests
of one client (from the `User-Agent`), each within 30 minutes of the previous. Each session has
`label`, totals (`requests`, `prompt_tokens`, `cached_tokens`, `attention`: misses worth fixing)
and `rows`: `main`, `sub1`… (subagents), `side` (requests without tools: titles, compaction).
`requests` holds the records after `since`, oldest first, with the fields of `finished` plus
`session`, `row`, `client`, `ended_ms`, `attention` and `excerpt`. Counters only, in memory only.

`GET /mrun/requests/{id}/excerpt` decodes, on each call, the text at the divergence of a request
whose `excerpt` is `true`: `context` (just before it), `before` (the cached text) and `after` (the
sent text), each up to about 200 characters. `404` when the request is gone or its prompt did not
change. Remote mode answers `/mrun/sessions` with `"remote": true` and no sessions.

## Errors

| Condition | Chat Completions / Responses | Anthropic Messages |
| --- | --- | --- |
| Queue full | 429 `rate_limit_exceeded` with `retry-after` (1–60 s) | 529 `overloaded_error` with `retry-after` (1–60 s) |
| Model still loading | 503 `server_busy` | 529 `overloaded_error` |
| Non-streaming request waited 240 s in the queue, or its estimated wait is longer | 503 `server_busy` | 529 `overloaded_error` |
| Prompt does not fit | 400 `context_length_exceeded`; streaming Responses: `response.failed` with that code, which Codex answers by compacting | 400 `invalid_request_error` ("prompt is too long") |
| Server shutting down | 503 `server_busy`; once streaming, an in-stream error | 529 `overloaded_error`; once streaming, an `error` event |
| A request-level or engine failure | 500 `server_error`; once streaming, an in-stream error | 500 `api_error`; once streaming, an `error` event |
| Body over 64 MiB | 413 `request_too_large` | 413 |
| Unsupported input | 400 `invalid_request_error` with `param` | 400 `invalid_request_error` |
| Host or Origin rejected | 403 | 403 |

## Security

mrun listens on `127.0.0.1` only. It accepts only loopback `Host` headers and rejects browser
requests from other origins (DNS rebinding, web pages). No API key is needed or checked.
