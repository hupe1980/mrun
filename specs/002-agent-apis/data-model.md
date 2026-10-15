# Data Model: Agent APIs

Detailed field mapping per API is in [research §1, §5](research.md); this is the internal model.

## ChatRequest

| Field | Type | Source mapping |
| --- | --- | --- |
| api | Chat \| Messages \| Responses | route |
| model | string | `model` (alias or served model id) |
| system | Vec<SystemPart> | Chat `system`/`developer` (leading), Anthropic `system`, Responses leading `developer`/`instructions` |
| messages | Vec<Message> | User, Assistant{parts}, Tool{call_id, content, is_error}, System (mid-conversation) |
| tools | Vec<ToolDef{name, description, schema, strict}> | `tools[].function` / `tools[]` / Responses `function` tools |
| tool_choice | Auto \| None \| Required \| Named | per API; Anthropic `any` → Required. None = fixed −inf bias on tool-call opener tokens (pipelined); Required/Named = grammar |
| parallel_tool_calls | bool | Chat/Responses flag; Anthropic `disable_parallel_tool_use` inverted |
| response_format | Text \| JsonObject \| JsonSchema{schema, strict} | Chat `response_format`, Responses `text.format`, Anthropic `output_config.format` |
| sampling | {temperature, top_p, top_k, min_p, seed} | defaults from model generation_config |
| max_output_tokens | u32 | `max_completion_tokens` > `max_tokens`; `max_tokens`; `max_output_tokens`; absent → server default 16,384; clamped to capacity at admission (001 FR-009), reported as `x-mrun-max-tokens` / `: mrun` comment |
| stop | Vec<String> (≤ 4 × 256 B) | `stop` / `stop_sequences` |
| reasoning | Off \| On{budget} | Anthropic thinking enabled/adaptive; `reasoning_effort` / `reasoning.effort` not in {none, minimal}; a bare `reasoning` object does not enable; overridden by `--thinking` |
| prefill | bool | Anthropic trailing assistant message |
| stream, include_usage | bool | |

Not in the model (FR-030, FR-023 removed): `cache_control`, `prompt_cache_key`, client session
headers and Anthropic `diagnostics` are accepted and dropped by the decoders.

Assistant parts: `Text`, `Reasoning{text}` (restored from signature/encrypted_content; for
Qwen3.6 rendered with the template's preserve-thinking option, FR-013),
`ToolCall{id, name, raw_args}` — raw argument text is kept verbatim. Byte stability of the
re-rendered history depends on the template:

| Template | History arguments | Previous reply re-renders |
| --- | --- | --- |
| Qwen3 (Hermes JSON) | string rendered verbatim | byte-identical |
| Qwen3-Coder, Qwen3.6 (XML) | parsed to an object, rendered via `\|items` and `tojson` | may differ → 003 cause `template` at the reply |

On XML templates, `raw_args` that does not parse as a JSON object is rendered verbatim in place of
the parameter list, with a `history-args-raw` warning (FR-004).

Validation (before queuing), configurable: body ≤ 64 MiB (base64 screenshots in agent history; images are dropped before rendering); messages ≤ 4,096; tools ≤ 256; each schema
≤ 64 KiB and depth ≤ 16; tool result call ids must match a prior tool call; images/audio/files in
user content rejected with the JSON path; images inside tool results replaced by a placeholder
(FR-020). The `model` field is informational when one model is served (FR-018).

## GenEvent (engine → encoders)

`Started{input, cached}` · `ReasoningDelta(text)` · `TextDelta(text)` ·
`ToolStart{idx, id, name}` · `ToolArgsDelta{idx, json}` · `ToolEnd{idx, raw_args, valid}` ·
`Finished{reason, stop_seq, usage}` · `Failed(ApiError)`; exactly one terminal event.

Finish mapping: see [research Design recommendations](research.md#design-recommendations).

## Usage

| Field | Chat | Anthropic | Responses |
| --- | --- | --- | --- |
| input (incl. cached) = P | `prompt_tokens` | `input_tokens + cache_read_input_tokens` | `input_tokens` |
| cached | `prompt_tokens_details.cached_tokens` | `cache_read_input_tokens` | `input_tokens_details.cached_tokens` |
| output | `completion_tokens` | `output_tokens` | `output_tokens` |
| reasoning | `completion_tokens_details.reasoning_tokens` | (in output) | `output_tokens_details.reasoning_tokens` |
| cache writes | — | `cache_creation_input_tokens` = 0 | `cache_write_tokens` omitted |

`cached = min(resume, P − 1)`; resume = match (attention) or restored checkpoint position (hybrid), per 003 FR-006: the last prompt token is always processed. Anthropic
`input_tokens = P − cached`. Writes are reported as 0 because mrun bills nothing and keeps no
separate write tier; clients that price cache writes then estimate zero cost.

## QueueEntry

`{request, rendered_tokens, arrived, deadline (arrived + 600 s, waiting time only), cancel:
CancelToken, events: Sender}`; queue holds 16 waiting. A per-connection close watcher fires
`cancel` whether the request is queued, prefilling, generating or non-streaming. States
`Queued → Running → Done`; `Queued → Expired | Cancelled`; `Running → Cancelled`.

## Record (`--record` only, FR-021)

One JSON line per request: `{request_id, api, arrived, ended (RFC 3339 ms), session: {header:
value} (allow-list: x-claude-code-session-id, thread-id, x-session-id; values ≤ 256 B),
output_tokens, outcome, body}`. Files mode 0600.

## Reasoning round-trip

- Anthropic: thinking block text returned as-is; `signature = "mrun1." + base64(sha256(text))[..16]`;
  any signature accepted on input.
- Responses: reasoning item `{summary: [], encrypted_content: "mrun1." + base64(text)}`; decoded on
  input to restore reasoning text (not encrypted: content stays on the client's machine).

## RenderedPrompt (renderer → 003)

`{tokens, messages: [{start, role, content_hash}], system_tools_end, generation_start}`. Content
hashes cover role, visible text and tool calls as canonical JSON, never reasoning, so 003 can tell
an edit from a template re-render. Render once; derive message starts from the token stream
(`<|im_start|>`, `<tool_response>` special tokens) and check the count against the messages;
prefix rendering only in tests.
