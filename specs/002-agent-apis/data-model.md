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
| max_output_tokens | u32 | `max_completion_tokens` > `max_tokens`; `max_tokens`; `max_output_tokens`; absent → server default 16,384 (Chat) or `capacity − prompt` (Responses; Codex never sends it, FR-019); clamped to capacity at admission (001 FR-009), reported as `x-mrun-max-tokens` / `: mrun` comment |
| stop | Vec<String> (≤ 4 × 256 B) | `stop` / `stop_sequences` |
| reasoning | Off \| On{budget, effort: Option<Low \| Medium \| High>} | Anthropic thinking enabled/adaptive; `reasoning_effort` / `reasoning.effort` not in {none, minimal}; a bare `reasoning` object does not enable. Applied per family (FR-035): `enable_thinking` (Qwen3, Qwen3.6), effort `low\|medium\|high` with Off = `low` (gpt-oss, research §3b mapping), none (Qwen3-Coder → logged `thinking-unsupported`) |
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
| Qwen3-Coder, Qwen3.6 (XML) | parsed to an object, rendered via `\|items` and `tojson` | may differ → 003 cause `changed` at the reply (bench: `template`) |
| gpt-oss (Harmony) | parsed to an object, rendered via `tojson`; one call per message (multi-call history split, logged `history-calls-split`); analysis rendered only inside the current tool loop | may differ → `changed` at the reply; a new user turn drops the loop's analysis → `changed` at its first analysis |

On re-serializing templates (XML, Harmony), `raw_args` that does not parse as a JSON object is
rendered verbatim in place of the parameter list or JSON and logged (`history-args-raw`, FR-004).

## Syntax (registry entry, FR-032)

`SyntaxId` is defined in `mrun-engine` (the family registry names one per family); the parser,
registry entry and history rules live in `mrun/src/chat/syntax/`.

| Field | Hermes JSON | Qwen XML | Harmony |
| --- | --- | --- | --- |
| families | `qwen3` | `qwen3_moe`, `qwen3_5_moe` | `gpt_oss` |
| template check | `<tool_call>\n{"name": "mrun_probe"` | `<function=mrun_probe>` | `to=functions.mrun_probe` |
| message_starts | `<\|im_start\|>`, `<tool_response>` | same (ChatML) | `<\|start\|>` |
| reasoning | `<think>…</think>` (prompt-tail aware) | same (Qwen3.6); none (Qwen3-Coder) | `analysis` channel |
| call opener (`none` bias) | `<tool_call>` | `<tool_call>` | recipient ` to` inside a header |
| arguments | raw JSON, streamed | synthesized, buffered | raw JSON to `<\|call\|>`, streamed |
| parallel calls | yes | yes | no (stops at `<\|call\|>`) |
| thinking control | `enable_thinking` | `enable_thinking` (Qwen3.6) | `reasoning_effort` |
| history (`ChatSyntax::history()`) | arguments verbatim; `reasoning_content` | arguments as object; `reasoning_content`, preserve-thinking (Qwen3.6) | arguments as object, one call per message; `thinking` |

Template check: render the fixed probe conversation (010 R8) and require the family's syntax
signature above in the rendered text; markers must be added tokens. A miss refuses the load.
Mistral (`[TOOL_CALLS]`, `[INST]`/`[TOOL_RESULTS]`/`[SYSTEM_PROMPT]` starts) is added with
Devstral in 0.2.

Validation (before queuing), configurable: body ≤ 64 MiB (base64 screenshots in agent history; images are dropped before rendering); messages ≤ 4,096; tools ≤ 256; each schema
≤ 64 KiB and depth ≤ 16; tool result call ids must match a prior tool call; images/audio/files in
user content rejected with the JSON path; images inside tool results replaced by a placeholder
(FR-020). The `model` field is informational when one model is served (FR-018).

## DecodeHook (per request, engine thread; FR-036)

`on_token(id) -> Option<TokenMask>` (feeds the `SyntaxParser` and, when armed, the llguidance
matcher; returns the mask for the next step), `forced_tokens() -> Vec<u32>` (grammar-forced
tokens appended without sampling), `ignore_eos: bool` (FR-031). Engine token events are lossless:
every sampled id with its text and a separate held-back field; `Stop` carries the EOS token id.

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

One JSON line per request: `{api, arrived (RFC 3339 ms), session: {header: value} (allow-list:
x-claude-code-session-id, thread-id, x-session-id, x-claude-code-prev-tool-durations; values
≤ 256 B), body}`. Files mode 0600. Reply lengths are derived by 005 from the next request.

## Reasoning round-trip

- Anthropic: thinking block text returned as-is; `signature = "mrun1." + base64(sha256(text))[..16]`;
  any signature accepted on input.
- Responses: reasoning item `{summary: [], encrypted_content: "mrun1." + base64(text)}`; decoded on
  input to restore reasoning text (not encrypted: content stays on the client's machine).

## RenderedPrompt (renderer → 003)

`{tokens, messages: [{start, role}], system_tools_end, generation_start}`. It is the prompt of
001's `GenerateRequest`. Render once; message starts are derived from the token stream using the
syntax entry's message-start tokens (ChatML `<|im_start|>`/`<tool_response>`; Harmony
`<|start|>`), and the count is checked against the messages; prefix rendering only in tests.
Each mark maps to the API message it renders (Harmony split calls map to their API message;
mid-conversation system messages keep their API index), so 003 reports the divergence's API
message index and role. (Content hashes were
dropped with the `template` cause, 2026-10-15.)
