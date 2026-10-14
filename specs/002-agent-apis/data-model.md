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
| tool_choice | Auto \| None \| Required \| Named | per API; Anthropic `any` → Required |
| parallel_tool_calls | bool | Chat/Responses flag; Anthropic `disable_parallel_tool_use` inverted |
| response_format | Text \| JsonObject \| JsonSchema{schema, strict} | Chat `response_format`, Responses `text.format`, Anthropic `output_config.format` |
| sampling | {temperature, top_p, top_k, min_p, seed} | defaults from model generation_config |
| max_output_tokens | Option<u32> | `max_completion_tokens` > `max_tokens`; `max_tokens`; `max_output_tokens` |
| stop | Vec<String> (≤ 4 × 256 B) | `stop` / `stop_sequences` |
| reasoning | Off \| On{budget} | thinking enabled/adaptive; `reasoning_effort` ≠ none; Responses `reasoning.effort` present (a bare `reasoning` object does not enable); overridden by `--thinking` |
| prefill | bool | Anthropic trailing assistant message |
| stream, include_usage | bool | |

Assistant parts: `Text`, `Reasoning{text}` (restored from signature/encrypted_content),
`ToolCall{id, name, raw_args}` — raw argument text is kept verbatim so re-rendered history is
byte-identical (prefix reuse).

Validation (before queuing): body ≤ 8 MiB; messages ≤ 4,096; tools ≤ 128; each schema ≤ 64 KiB and
depth ≤ 16; tool result call ids must match a prior tool call; images/audio/files rejected with the
JSON path.

## GenEvent (engine → encoders)

`Started{input, cached}` · `ReasoningDelta(text)` · `TextDelta(text)` ·
`ToolStart{idx, id, name}` · `ToolArgsDelta{idx, json}` · `ToolEnd{idx, raw_args, valid}` ·
`Finished{reason, stop_seq, usage}` · `Failed(ApiError)`; exactly one terminal event.

Finish mapping: see [research Design recommendations](research.md#design-recommendations).

## Usage

| Field | Chat | Anthropic | Responses |
| --- | --- | --- | --- |
| input (incl. cached) | `prompt_tokens` | `input_tokens + cache_read_input_tokens` | `input_tokens` |
| cached | `prompt_tokens_details.cached_tokens` | `cache_read_input_tokens` | `input_tokens_details.cached_tokens` |
| output | `completion_tokens` | `output_tokens` | `output_tokens` |
| reasoning | `completion_tokens_details.reasoning_tokens` | (in output) | `output_tokens_details.reasoning_tokens` |

## QueueEntry

`{request, rendered_tokens, arrived, deadline (arrived + 30 s), cancel: CancelToken, events: Sender}`;
states `Queued → Running → Done`; `Queued → Expired | Cancelled`.

## Reasoning round-trip

- Anthropic: thinking block text returned as-is; `signature = "mrun1." + base64(sha256(text))[..16]`;
  any signature accepted on input.
- Responses: reasoning item `{summary: [], encrypted_content: "mrun1." + base64(zstd(text))}`;
  decoded on input to restore reasoning text (not encrypted: content stays on the client's machine).
