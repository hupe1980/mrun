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
| max_output_tokens | u32 | `max_completion_tokens` > `max_tokens`; `max_tokens`; `max_output_tokens`; absent → server default 16,384; clamped to capacity (reported) |
| stop | Vec<String> (≤ 4 × 256 B) | `stop` / `stop_sequences` |
| reasoning | Off \| On{budget} | Anthropic thinking enabled/adaptive; `reasoning_effort` / `reasoning.effort` not in {none, minimal}; a bare `reasoning` object does not enable; overridden by `--thinking` |
| prefill | bool | Anthropic trailing assistant message |
| stream, include_usage | bool | |

| cache_hints | {breakpoints: Vec<token offset>, session: Option<u64>} | Anthropic `cache_control` positions (after rendering); hash of `prompt_cache_key`. Passed to 003 (FR-014) |
| diagnostics_prev | Option<Option<String>> | Anthropic `diagnostics.previous_message_id` (`Some(None)` = opted in, first turn) |

Assistant parts: `Text`, `Reasoning{text}` (restored from signature/encrypted_content),
`ToolCall{id, name, raw_args}` — raw argument text is kept verbatim so re-rendered history is
byte-identical (prefix reuse).

Validation (before queuing), configurable: body ≤ 8 MiB; messages ≤ 4,096; tools ≤ 256; each schema
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
| input (incl. cached) | `prompt_tokens` | `input_tokens + cache_read_input_tokens` | `input_tokens` |
| cached | `prompt_tokens_details.cached_tokens` | `cache_read_input_tokens` | `input_tokens_details.cached_tokens` |
| output | `completion_tokens` | `output_tokens` | `output_tokens` |
| reasoning | `completion_tokens_details.reasoning_tokens` | (in output) | `output_tokens_details.reasoning_tokens` |

## QueueEntry

`{request, rendered_tokens, arrived, deadline (arrived + 600 s, waiting time only), cancel: CancelToken, events: Sender}`;
states `Queued → Running → Done`; `Queued → Expired | Cancelled`.

## Reasoning round-trip

- Anthropic: thinking block text returned as-is; `signature = "mrun1." + base64(sha256(text))[..16]`;
  any signature accepted on input.
- Responses: reasoning item `{summary: [], encrypted_content: "mrun1." + base64(text)}`; decoded on
  input to restore reasoning text (not encrypted: content stays on the client's machine).

## Prompt Fingerprint and DiagnosticsIndex (FR-023)

A Prompt Fingerprint is `{model_identity, regions: [{kind: System | Tools | Message(i), start,
end, hash}]}`: token offsets of each rendered region (the renderer emits region boundaries) and a
hash of its tokens. The DiagnosticsIndex is an in-memory LRU of the last 256 `msg_` ids → Prompt
Fingerprint, cleared by `/mrun/cache/clear`; nothing is persisted.

Comparing the current fingerprint with the referenced one yields the first differing region:

| Comparison result | `cache_miss_reason` |
| --- | --- |
| id not in the index | `previous_message_not_found` |
| cache disabled for this request or server | `unavailable` |
| model identity differs | `model_changed` |
| first difference in System | `system_changed` |
| first difference in Tools | `tools_changed` |
| first difference in a message | `messages_changed` |
| no difference but fewer tokens reused than the previous prompt's length (evicted) | `messages_changed` |
| previous prompt fully reused | `diagnostics: null` |

`cache_missed_input_tokens` = previous prompt length − tokens actually reused (from 003's report),
never negative.
