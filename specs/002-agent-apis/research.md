# Research: Agent APIs (feature 002)

- **Date:** 2026-10-14. **Markers:** **[V]** = verified in source at the pinned commit or in official docs fetched today; **[U]** = unverified (inference, secondary source, or not exercised against a live client).
- **Pinned sources** (short names used below):
  - `CX` openai/codex `main@ae11049` (2026-10-14; landscape pins rust-v0.162.1) — https://github.com/openai/codex/blob/ae11049/codex-rs/
  - `OC` anomalyco/opencode `dev@13afea6` (2026-10-14, ≈v2.0.29) — https://github.com/anomalyco/opencode/blob/13afea6/
  - `CC` Claude Code **2.1.300** (npm, published 2026-10-14) + docs https://code.claude.com/docs/en/llm-gateway-protocol , /model-config , /errors
  - `LC` llama.cpp `23deec0` — https://github.com/ggml-org/llama.cpp/blob/23deec0/ ; `VL` vLLM `e73fa00` — https://github.com/vllm-project/vllm/blob/e73fa00/ ; `ML` mlx-lm `ba8a08f` — https://github.com/ml-explore/mlx-lm/blob/ba8a08f/
  - `LG` llguidance **1.9.1** (2026-09-30) `6cef6ed` — https://github.com/guidance-ai/llguidance/blob/6cef6ed/
  - `AS` anthropic-sdk-python v1.14.0 `2a53eaa` — https://github.com/anthropics/anthropic-sdk-python/blob/2a53eaa/src/anthropic/types/ ; `OS` openai-python 3.31.0 `d42accb` — https://github.com/openai/openai-python/blob/d42accb/src/openai/types/
  - `OL` ollama `292d7ce` (v0.40.5-rc0) — https://github.com/ollama/ollama/blob/292d7ce/

## 1. Wire formats

### 1a. OpenAI Chat Completions (`POST /v1/chat/completions`)

**Request fields to honor** [V `OS chat/completion_create_params.py`]: `model`, `messages` (roles `system`, `developer`, `user`, `assistant`, `tool`), `tools[{type:"function",function:{name,description,parameters,strict}}]`, `tool_choice` (`"none"|"auto"|"required"|{type:"function",function:{name}}`), `parallel_tool_calls`, `response_format` (`text` | `json_object` | `{type:"json_schema",json_schema:{name,schema,strict}}`), `max_completion_tokens` (preferred) and deprecated `max_tokens`, `stop` (string or array), `temperature`, `top_p`, `seed`, `stream`, `stream_options.include_usage`, `reasoning_effort` (`none|minimal|low|medium|high|xhigh|max` [V `OS shared/reasoning_effort.py`]). Ignore the rest: `store`, `metadata`, `user`, `prompt_cache_key`, `frequency_penalty`/`presence_penalty` (or map them), `cache_control` on tools and content parts.

**Streaming** [V `OS chat/chat_completion_chunk.py`]:
- `delta.tool_calls[]` has `index:int` (required), and `id`, `type:"function"` and `function.name`, which are sent only on the **first** fragment of a call. Later fragments carry only `index` plus `function.arguments` (a raw JSON *substring*).
- `finish_reason` ∈ `stop|length|tool_calls|content_filter|function_call`.
- With `include_usage`, one extra chunk with `choices: []` and `usage` comes before `data: [DONE]`. Every other chunk has `usage: null`.
- `usage` = `{prompt_tokens, completion_tokens, total_tokens, prompt_tokens_details{cached_tokens}, completion_tokens_details{reasoning_tokens}}` [V `OS completion_usage.py`]. Here `prompt_tokens` **includes** cached tokens.
```
data: {"id":"chatcmpl-9","object":"chat.completion.chunk","created":1760400000,"model":"qwen3-coder","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}
data: {...,"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_a1","type":"function","function":{"name":"read_file","arguments":""}}]},"finish_reason":null}]}
data: {...,"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"path\": \"src/"}}]},"finish_reason":null}]}
data: {...,"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}
data: {...,"choices":[],"usage":{"prompt_tokens":812,"completion_tokens":31,"total_tokens":843,"prompt_tokens_details":{"cached_tokens":768}}}
data: [DONE]
```
- **Reasoning field (not in the OpenAI spec):**
  - llama.cpp reads and writes `reasoning_content` [V `LC common/chat.cpp:203,422`].
  - vLLM now emits `reasoning` and still accepts `reasoning_content` on input [V `VL vllm/entrypoints/openai/chat_completion/protocol.py:77,545-561`].
  - mlx-lm emits `reasoning` [V `ML mlx_lm/server.py:1366`].
  - OpenCode reads both, and replays reasoning on assistant history as `reasoning_content` [V `OC packages/ai/src/protocols/openai-chat.ts:152-153,473-484`].
  - → Emit `delta.reasoning_content`. Accept both `reasoning_content` and `reasoning` on input.
- **Errors:** `{"error":{"message","type","param","code"}}`. Context overflow is 400 `type:"invalid_request_error", code:"context_length_exceeded"`; queue full is 429 with `retry-after`. A mid-stream error is sent as `data: {"error":{...}}`, which OpenCode patches its SDK to surface [V `OC patches/@ai-sdk%2Fopenai-compatible@2.0.41.patch`].
- **`response_format` json_schema strict:** the output must be exactly one JSON value that validates. When `strict:false`, OpenAI still steers toward the schema but does not guarantee it [U]. mrun constrains only when `strict:true` or when the type is `json_object` (spec clarification).

### 1b. Anthropic Messages (`POST /v1/messages`, `POST /v1/messages/count_tokens`)

- **Headers** [V `CC` gateway protocol]:
  - Credentials arrive as `x-api-key` (from `ANTHROPIC_API_KEY`) and/or `Authorization: Bearer` (from `ANTHROPIC_AUTH_TOKEN`).
  - `anthropic-version: 2023-06-01`.
  - `anthropic-beta`: a comma list, open-ended. Accept it and ignore it.
  - Also `x-claude-code-session-id` and `x-claude-code-agent-id`.
  - Return a `request-id` header.
  - The path arrives as **`/v1/messages?beta=true`**: match on the path only.
- **Request** [V `AS message_create_params.py`]:
  - `model`, `max_tokens` (required).
  - `messages[]` (roles `user`/`assistant`; content is a string or a list of blocks).
  - `system`: a string **or** an array of `{type:"text",text,cache_control?}`.
  - `tools[{name,description,input_schema,strict?,cache_control?,defer_loading?}]`.
  - `tool_choice`: `{type:auto|any|tool|none, name?, disable_parallel_tool_use?}` [V `AS tool_choice_*.py`].
  - `thinking`: `{type:"enabled",budget_tokens≥1024}` | `{type:"disabled"}` | `{type:"adaptive",display?}` | `{type:"between_tools"}` [V `AS thinking_config_*.py`].
  - `output_config`: `{effort: low|medium|high|xhigh|max, format: json schema}` [V `AS output_config_param.py`].
  - `stop_sequences`, `temperature`, `top_p`, `top_k`, `stream`, `metadata`, and a top-level `cache_control`.
- **Content blocks in:** `text`, `tool_use{id,name,input(object)}`, `tool_result{tool_use_id, content: string | [text|image blocks], is_error?}`, `thinking{thinking,signature}`, `redacted_thinking{data}`, `image`/`document` (reject).
- **Stop reasons:** `end_turn|max_tokens|stop_sequence|tool_use|pause_turn|refusal|model_context_window_exceeded` [V `AS stop_reason.py`].
- **Usage:** `input_tokens`, `cache_read_input_tokens`, `cache_creation_input_tokens`, `output_tokens` [V `AS usage.py`]. **`input_tokens` excludes cache reads and writes** (total input = input + cache_read + cache_creation) [U wording; https://platform.claude.com/docs/en/build-with-claude/prompt-caching]. This is the opposite of OpenAI.
- **Stream sequence** [V https://platform.claude.com/docs/en/build-with-claude/streaming]: `message_start` (Message with empty `content`, `usage.input_tokens` and cache fields) → per block `content_block_start` / `content_block_delta`* / `content_block_stop` → `message_delta{delta:{stop_reason,stop_sequence},usage:{output_tokens,…}}` (usage is **cumulative**) → `message_stop`.
  - Any number of `ping` events may appear.
  - Errors arrive as `event: error` / `data: {"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}`.
  - The deltas are `text_delta{text}`, `input_json_delta{partial_json}`, `thinking_delta{thinking}`, and `signature_delta{signature}`. A thinking block sends exactly one `signature_delta` just before its `content_block_stop`.
```
event: message_start
data: {"type":"message_start","message":{"id":"msg_01","type":"message","role":"assistant","model":"qwen3-coder","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":44,"cache_read_input_tokens":768,"cache_creation_input_tokens":0,"output_tokens":0}}}
event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":"","signature":""}}
event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"Need the file."}}
event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"mrun1.…"}}
event: content_block_stop
data: {"type":"content_block_stop","index":0}
event: content_block_start
data: {"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"toolu_01","name":"Read","input":{}}}
event: content_block_delta
data: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"file_path\": \"/r/a.rs\"}"}}
event: content_block_stop
data: {"type":"content_block_stop","index":1}
event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"tool_use","stop_sequence":null},"usage":{"output_tokens":41}}
event: message_stop
data: {"type":"message_stop"}
```
- **count_tokens:** takes the same body minus `max_tokens`/`stream` and returns `{"input_tokens":N}` [V `AS message_tokens_count.py`].
- **Errors** [V https://platform.claude.com/docs/en/api/errors]:
  - Shape: `{"type":"error","error":{"type","message"},"request_id"}`.
  - Status codes: 400 `invalid_request_error`, 401 `authentication_error`, 403 `permission_error`, 404 `not_found_error`, 413 `request_too_large`, 429 `rate_limit_error`, 500 `api_error`, 504 `timeout_error`, 529 `overloaded_error`.
  - Context overflow: 400 `invalid_request_error` with a message starting "prompt is too long: N tokens > M maximum". Claude Code matches this wording ("Prompt is too long") [U exact match; `CC /errors`].

### 1c. OpenAI Responses, stateless subset (`POST /v1/responses`)

- **Request** [V `OS responses/response_create_params.py`]: `model`, `input` (string or items), `instructions`, `tools`, `tool_choice` (`none|auto|required|{type:"function",name}`), `parallel_tool_calls`, `reasoning{effort,summary}`, `text{format:{type:"json_schema",name,schema,strict}}`, `max_output_tokens`, `store`, `previous_response_id`, `include`, `stream`, `temperature`, `top_p`, `truncation`, `prompt_cache_key`. **There is no stop parameter.**
- **Input items:**
  - `message{role:user|assistant|system|developer, content:[input_text|output_text|input_image|input_file]}`
  - `function_call{call_id,name,arguments(string),id?}`
  - `function_call_output{call_id, output: string | [input_text|input_image…]}`
  - `reasoning{id,summary[],content?[reasoning_text],encrypted_content?}`
  - `custom_tool_call(_output)`, `item_reference` (stateful: reject)
- **Output items:** `message{id,role:"assistant",status,content:[{type:"output_text",text,annotations:[]}]}`, `function_call{id,call_id,name,arguments,status}`, `reasoning{id,summary:[],content:[{type:"reasoning_text",text}],encrypted_content}` [V `OS responses/response_function_tool_call.py`, `response_reasoning_item.py`].
- **Usage:** `{input_tokens, input_tokens_details{cached_tokens,cache_write_tokens}, output_tokens, output_tokens_details{reasoning_tokens}, total_tokens}` [V `OS responses/response_usage.py`]. `input_tokens` includes cached tokens.
- **Events.** Each event carries `sequence_number`. Item events carry `output_index`/`item_id`, and text events also carry `content_index` and `logprobs:[]` [V `OS responses/*_event.py`].
  - Order: `response.created` → `response.in_progress`.
  - Reasoning item: `output_item.added` → `reasoning_text.delta`* → `reasoning_text.done` → `output_item.done`.
  - Message item: `output_item.added` → `content_part.added` → `output_text.delta`* → `output_text.done` → `content_part.done` → `output_item.done`.
  - Function call item: `output_item.added` (with `arguments:""`) → `function_call_arguments.delta`* → `function_call_arguments.done` → `output_item.done`.
  - End: `response.completed` | `response.incomplete` (`incomplete_details.reason:"max_output_tokens"`) | `response.failed` (`response.error{code,message}`). There is also a top-level `error` event.
- **Stateless semantics.** `previous_response_id`, `conversation` and `item_reference` need server state → 400.
  - `store` defaults to true in OpenAI. mrun should accept it, echo `"store": false`, and not persist anything; that does not need stored state.
  - `include:["reasoning.encrypted_content"]` asks for an opaque, client-replayable reasoning blob. This is how stateless (`store:false`) clients keep reasoning across turns [U doc wording; https://platform.openai.com/docs/guides/reasoning].

## 2. What the pinned clients send and need

### Claude Code 2.1.300 (closed source; facts from the official gateway docs)

- **Configuration:**
  - `ANTHROPIC_BASE_URL=http://127.0.0.1:8484`, without `/v1`; Ollama passes its bare host [V `OL cmd/launch/claude.go:69`].
  - Credentials: `ANTHROPIC_AUTH_TOKEN` (sent as Bearer) or `ANTHROPIC_API_KEY` (sent as x-api-key). Ollama sets `ANTHROPIC_API_KEY=` and `ANTHROPIC_AUTH_TOKEN=ollama` [V].
  - Models: `ANTHROPIC_MODEL`, `ANTHROPIC_DEFAULT_{OPUS,SONNET,HAIKU}_MODEL` (Haiku is used for background tasks) and `CLAUDE_CODE_SUBAGENT_MODEL` [V `OL claude.go:172-178`; `CC /model-config`].
  - `CLAUDE_CODE_MAX_CONTEXT_TOKENS` applies directly to unrecognized IDs; `CLAUDE_CODE_AUTO_COMPACT_WINDOW` sets the compaction window [V `CC /model-config`].
- **Endpoints** [V `CC` protocol]:
  - `POST /v1/messages?beta=true`, always streamed in practice [U].
  - `POST /v1/messages/count_tokens` is **optional**. If it is absent, Claude Code falls back to a character estimate.
  - Best-effort `HEAD /api/hello` warm-up probe.
  - `GET /v1/models?limit=1000` only with `CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY=1`, and only ids containing "claude"/"anthropic" are kept. That makes discovery useless for Qwen ids.
- **Body behavior for an unrecognized model ID** ("treated as current models"):
  - `thinking:{type:"adaptive"}` and `output_config.effort`.
  - `context_management` with its beta header.
  - Beta tool fields `strict`/`defer_loading`, and `tool_reference` blocks (tool search).
  - `cache_control` on `system` blocks and on messages, **including `role:"system"` messages appended mid-conversation**.
  - With `CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS=1`, context management, beta tool fields, `output_config.format`/`task_budget` and tool search are dropped. Adaptive thinking and effort remain [V].
- **Thinking signatures** [V]:
  - If the server rejects a signature, Claude Code strips earlier thinking blocks and retries.
  - If the server rejects `thinking` or a mid-conversation system message, Claude Code retries without it.
  - So signatures are not *required* to verify. mrun should emit a non-empty opaque signature and never reject one. llama.cpp emits `""` [V `LC tools/server/server-task.cpp:744,910-919`].
- **Stream needs** [V]:
  - The full event order through `message_delta` + `message_stop`; a clean EOF before that counts as a dropped connection.
  - `ping` events during long silences, so the idle watchdog does not fire.
  - `retry-after` as integer seconds; a value above 60 stops retries.
  - 529/429 are retried up to 10 times (`CLAUDE_CODE_MAX_RETRIES`) [V `CC /errors`].
- **Attribution block.** Since v2.1.181 the system attribution block is **stable per conversation** for custom base URLs, so the prefix cache survives. `CLAUDE_CODE_ATTRIBUTION_HEADER=0` still removes it [V `CC` protocol §System prompt attribution block].
- **Prefill.** Claude ≥4.6 rejects assistant prefill, so Claude Code does not rely on it [V `/errors`]. Supporting prefill is for other Anthropic clients.

### Codex CLI (`wire_api = "responses"` only; `"chat"` is a hard error [V `CX model-provider-info/src/lib.rs:100-130`])

```toml
model = "qwen3-coder"
model_provider = "mrun"
[model_providers.mrun]
name = "mrun"
base_url = "http://127.0.0.1:8484/v1"   # Codex posts to {base_url}/responses [U path join]
wire_api = "responses"
# optional: env_key, http_headers, request_max_retries, stream_idle_timeout_ms, capabilities.remote_compaction
```
- **Request body** [V `CX core/src/client.rs:1040-1093`, `codex-api/src/common.rs:279-305`]: `model`, `stream:true`, `input[]`, `tools`, `tool_choice:"auto"`, `parallel_tool_calls`, `reasoning`, `store:false`, `include:["reasoning.encrypted_content"]`, `prompt_cache_key`, `text` (only for `--output-schema`), `client_metadata`.
- **No `instructions` field.** Base instructions are sent as the first `input` item, a `message` with role **`developer`** and `id:"msg_<uuidv5>"` [V `client.rs:995-1005`, `core/src/context/base_instructions.rs:30`].
- **`reasoning` is always present.** For an unknown model slug the fallback metadata sends `summary:"auto"` and **no effort**, and reports `context_window` 272 000 [V `CX models-manager/src/model_info.rs:98-135`, `client.rs:934-953`]. → The presence of `reasoning` must not turn thinking on; only an `effort` above `none`/`minimal` should.
- **Fallback tools.** Shell type is `UnifiedExec` (function tools). `apply_patch_tool_type: None`, so no freeform `apply_patch` [V `spec_plan.rs:1432`].
  - Tool specs can be `function`, `namespace`, `tool_search`, `web_search` or `custom` [V `CX tools/src/tool_spec.rs:22-53`].
  - → Accept `function`. Drop hosted tools (`web_search`, `tool_search`) with a warning header. Reject `namespace` and `custom` with a named error until they are mapped [U which ones appear with MCP configs].
- **History replay to non-OpenAI providers.** Codex clears `encrypted_content` on everything except `reasoning` items, and **omits reasoning `content` that holds `reasoning_text`** [V `client.rs:1013-1031`, `protocol/src/models.rs:1756-1763`]. → Reasoning survives a turn only through `encrypted_content`.
- **`function_call_output.output`** may be a string or an array of content items, possibly `input_image` (from the view_image tool) [V `models.rs` FunctionCallOutput].
- **SSE consumption** [V `CX codex-api/src/sse/responses.rs:345-506`]:
  - Function calls are taken **only from `response.output_item.done`**. `function_call_arguments.*` events are ignored.
  - An item that fails to deserialize is **silently dropped** (debug log, :359). A reasoning item **must** include `summary: []` [V `models.rs:1082`].
  - `response.completed` needs `response.id`; `usage` is optional but used.
  - `response.incomplete` with a reason other than `interrupted` becomes an error.
  - `response.failed` with `error.code:"context_length_exceeded"` becomes `ContextWindowExceeded`, which triggers compaction [V `sse/responses_error.rs:47-48`].
- **Compaction.** Remote compaction is off for custom providers, so Codex compacts locally with ordinary requests [V `CX model-provider/src/capabilities.rs:31-41`].

### OpenCode (custom provider via `@ai-sdk/openai-compatible`)

```json
{ "provider": { "mrun": { "npm": "@ai-sdk/openai-compatible", "name": "mrun (local)",
    "options": { "baseURL": "http://127.0.0.1:8484/v1" },
    "models": { "qwen3-coder": { "name": "Qwen3-Coder 30B-A3B", "limit": { "context": 65536, "output": 16384 } } } } } }
```
[V `OC packages/web/src/content/docs/providers.mdx:1431-1465`]

- **Runtime library.** In v2 this npm id is mapped to OpenCode's own `@opencode/ai` openai-compatible route, which reuses its OpenAI Chat protocol [V `OC packages/core/src/aisdk-native.ts:73`, `packages/ai/src/protocols/openai-compatible-chat.ts`].
- **Request body** [V `openai-chat.ts:805-850`]:
  - Always `stream:true` and `stream_options:{include_usage:true}`.
  - **`max_completion_tokens`**: unknown providers get this field, not `max_tokens` (:686-715).
  - `tools[].function.strict:false` (:334-343).
  - `tools: []` when there is tool history but no active tools.
  - Also `tool_choice`, `temperature`, `top_p`, `stop`, `seed`, `reasoning_effort` (when configured), and possibly `cache_control` on tools/parts.
- **Parser strictness** [V]:
  - Any content, reasoning or tool delta **after a chunk that carried `finish_reason` is an error** (:1097). The usage-only chunk (`choices:[]`) is fine.
  - An unknown `finish_reason` string is an error (:868-900).
  - Tool deltas with a missing `index` are tolerated, but id and name must arrive before or with the first arguments.
  - Cached tokens are read from `prompt_tokens_details.cached_tokens` (:910-924).

## 3. Qwen3 / Qwen3-Coder templates and tool-call syntax

**Qwen3 (hybrid, e.g. Qwen3-8B)** [V `LC models/templates/Qwen-Qwen3-0.6B.jinja`]:
- **Tools** are JSON lines inside `<tools></tools>` in the system turn. The model outputs a Hermes-style call:
  ```
  <tool_call>
  {"name": "read_file", "arguments": {"path": "src/main.rs"}}
  </tool_call>
  ```
  Multiple calls are separated by `\n`.
- **Tool results:** consecutive `tool` messages merge into one `<|im_start|>user` turn, each wrapped as `\n<tool_response>\n…\n</tool_response>`.
- **Thinking:** the model writes `<think>\n…\n</think>\n\n`. `enable_thinking=false` appends `<think>\n\n</think>\n\n` to the generation prompt.
- **History:** reasoning is re-rendered only for assistant turns **after the last real user query**, i.e. inside the current tool loop. Earlier turns are rendered without it.
- **System messages:** only `messages[0]` is merged into the tools system block. Later system messages are rendered as their own `<|im_start|>system` turns.
- **Arguments in history:** rendered verbatim if they are a string, or via `tojson` if they are an object.
- **Thinking-2507 variants:** the template opens `<think>` itself, so the output contains only `</think>` [V docs https://qwen.readthedocs.io/en/latest/getting_started/quickstart.html; disputed by a vLLM doc note, U]. mlx-lm decides the initial parser state by searching the prompt tail for an open `<think>` [V `ML mlx_lm/server.py:575-581`]. Do the same.

**Qwen3-Coder (30B-A3B-Instruct, no thinking)** [V `LC models/templates/Qwen3-Coder.jinja`]:
- **Tools** render as XML in the system prompt (`<function><name>…<parameters><parameter><name>/<type>/<description>`; extra schema keys are emitted via `tojson`).
- **Default system prompt** when tools are present and there is no system message: "You are Qwen, a helpful AI assistant that can interact with a computer to solve tasks."
- **Output format:**
  ```
  <tool_call>
  <function=read_file>
  <parameter=path>
  src/main.rs
  </parameter>
  </function>
  </tool_call>
  ```
  String values are raw, not quoted. Objects and arrays are JSON.
- **Tool results:** `<|im_start|>user\n<tool_response>\n…\n</tool_response>\n<|im_end|>\n`, with consecutive tool messages merged into one turn.
- **Arguments in history** must be an **object**: the template iterates them with `|items`. OpenAI-style string arguments must therefore be parsed first.

**Reference parsers:**
- **vLLM:**
  - `qwen3_json` is an alias of `hermes_tool_parser.py`, which holds back partial `<tool_call>` suffixes (:125) [V `VL vllm/tool_parsers/__init__.py:193-206`].
  - `qwen3_coder`/`qwen3_xml` use `qwen3_engine_tool_parser.py`, which wraps the single-state-machine engine `vllm/parser/qwen3.py`. That engine handles `<think>` plus XML calls, treats `<tool_call>` as an implicit end of reasoning, absorbs a duplicate `</think>`, recovers a `<function=` without `<tool_call>`, and keeps `_PARTIAL_PARAM_RE` for streaming partial parameters (:41-200) [V].
- **llama.cpp:**
  - `common/parsers/qwen3-coder.cpp` is a PEG parser plus grammar. Its reasoning end tags are `{"\n</think>", "</think>", "<tool_call>"}`.
  - Because Qwen3-Coder sometimes **omits `<tool_call>`**, it also triggers on `<function=NAME>` for each declared tool.
  - Parameters may come in any order (`p.permute`). String parameters run `until "\n</parameter>\n"`; non-string parameters are typed JSON.
  - Detection lives at `common/chat.cpp:1232-1240` [V].
- **mlx-lm:**
  - The parser is inferred from the template text (`tokenizer_utils.py:619-660`).
  - `tool_parsers/qwen3_coder.py` converts parameter values by schema type (string/int/number/bool/object, with `ast.literal_eval` as a fallback); it is adapted from Qwen's official `qwen3coder_tool_parser.py`.
  - `json_tools.py` is a plain `json.loads`.
  - The server buffers a whole tool block and parses it at the end, so arguments are not streamed [V `server.py:58-84`].

**Streaming parse strategy for mrun:**
- Run one incremental state machine per model family: `Content ↔ Reasoning`, `ToolCall{Header, Args}`.
- Fast path on the token id for single added tokens. `<tool_call>`, `</tool_call>`, `<think>` and `</think>` are added tokens in the Qwen vocab [U ids 151657/151658/151667/151668].
- Use text-level matching with a **hold-back buffer** for the multi-token markers `<function=`, `<parameter=`, `\n</parameter>` and `</function>`, and for stop strings.
- **Qwen3 JSON calls:**
  - Emit the name once `"name": "…"` closes.
  - Stream the raw `arguments` substring, tracking JSON depth and string state with a small scanner to find where the object ends.
  - Validate with `serde_json` at `</tool_call>`.
- **Qwen3-Coder XML calls:** synthesize JSON fragments incrementally.
  - At `<parameter=k>` emit `{"k": ` (or `, "k": ` after the first parameter).
  - For a schema `string`, stream JSON-escaped characters, holding back up to `\n</parameter>`.
  - For other types, buffer until close, then emit the parsed JSON value. If it fails to parse, report a parse error and do not repair (FR-004).
  - Unknown types fall back to the string rule.
- **Malformed or truncated calls:** a call that fails to parse becomes an error event; at `Length` it becomes "incomplete". Never report it as a completed call.

## 4. Constrained decoding

- **llguidance 1.9.1** [V `LG parser/Cargo.toml`, CHANGELOG] is pure Rust: crates `llguidance`, `toktrie`, `toktrie_hf_tokenizers`.
  - **API** [V `LG parser/src/matcher.rs:21-326`, `sample_parser/src/minimal.rs`]:
    - `ParserFactory::new_simple(&tok_env)`, created once per tokenizer and shareable via `Arc`.
    - Grammars: `TopLevelGrammar::from_json_schema(v)` or a Lark grammar.
    - `factory.create_parser(g)` → `Matcher::new(parser)`.
    - Per step: `compute_mask()` → `SimpleVob`, then `consume_token`; `rollback(n)`, `compute_ff_tokens()` (forced tokens), `is_accepting()`, `is_stopped()`.
    - `into_cancellable()` and cooperative cancellation are new in 1.9.0/1.9.1.
  - **Lark** supports `%json{schema}` inline, special tokens (`<name>` or `<[id]>`), and `lazy`/`stop`/`suffix` lexemes [V `LG docs/syntax.md:30-240`].
  - **Cost:** about 50 µs average and about 1.5 ms worst-typical per mask at a 128k vocabulary for JSON-schema grammars; under 1% of masks exceed 1 ms. There is no startup precompute [V `LG README.md:78-104`].
  - **Fit with the engine:** compute the mask on a CPU thread while the GPU runs the forward pass, then apply it as an additive −inf bias before sampling.
  - Qwen's vocabulary (about 152k) makes the bitset about 19 KB per step [U timing on M-series].
- **vLLM:**
  - `backend_guidance.py` builds an `LLMatcher` with fast-forward. It pins `llguidance>=1.7,<1.8` and `xgrammar==0.2.8` [V `VL requirements/common.txt:26,29`].
  - Tool calls use **xgrammar structural tags**, including the builtin `qwen_3`/`qwen_3_coder` tags.
  - `--tool-strict-level`: `auto` constrains the call envelope from `tool_choice` and enforces argument schemas **only for `strict:true` tools**; `function` always constrains the envelope; `parameter` treats every tool as strict [V `VL vllm/tool_parsers/tool_strict_level.py`, `structural_tag_registry.py:70-90`]. `auto` matches mrun's clarified policy.
- **llama.cpp:**
  - Its default GBNF grammar is **lazy**: it is armed by trigger words (`<tool_call>`, `<function=NAME>`) when `tool_choice=auto` [V `LC common/parsers/qwen3-coder.cpp:170-183`].
  - A grammar string starting with `%llguidance` routes to llguidance (build flag `LLAMA_LLGUIDANCE`, pinned commit `d795912`), via `llg_new_matcher` / `llg_matcher_compute_mask` [V `LC common/llguidance.cpp`, `common/CMakeLists.txt:160-176`].
- **XGrammar** is the alternative. It is a C++ core with Python bindings and builtin Qwen structural tags.
  - The Rust crate is `xgrammar` 0.5.0 (trymirai bindings, vendors xgrammar v0.2.3, needs CMake and C++17) [V https://docs.rs/crate/xgrammar/0.5.0].
  - llguidance's own comparison says XGrammar's precompute takes seconds and its masks occasionally take 10–100 ms [V `LG README.md:93-94`, vendor claim].
  - → **Use llguidance**: it is pure Rust, has no precompute, and uses the same Lark/JSON dialect for tools and `response_format`.
- **Constraining only the arguments in native syntax.**
  - The default path (non-strict, `tool_choice:auto`) builds no matcher.
  - When the streaming parser sees `<tool_call>` and the request has strict tools (or `required`/named choice), mrun starts a Matcher whose grammar covers **only the call body**, then resumes free decoding after `</tool_call>`. This is the "lazy trigger" from llama.cpp/vLLM, implemented in mrun's parser.
  - `required` and named `tool_choice` start the matcher immediately; it may force the prefix via `compute_ff_tokens`.
  - Sketch for Qwen3-Coder (simplified to required parameters in declared order, then optional; the Lark below is illustrative):
    ```lark
    start: "\n<function=" ( f_read | f_edit ) "</function>\n</tool_call>"
    f_read: "read_file>\n" "<parameter=path>\n" S "\n</parameter>\n" ("<parameter=limit>\n" %json{"type":"integer"} "\n</parameter>\n")?
    S[lazy]: /(.|\n)*/  // terminated by the following "\n</parameter>" literal; enum/pattern → alternation/regex
    ```
  - For Qwen3 JSON: `start: "\n{\"name\": \"read_file\", \"arguments\": " %json{<schema>} "}\n</tool_call>"`.
  - **Caveats [U]:** forcing declared order may push the model slightly out of distribution (llama.cpp uses permutations instead). `<tool_call>` must be addressed as a special-token literal if the tokenizer marks it special; otherwise use a string.

## 5. Translation pitfalls (three APIs → one model)

1. **System content.**
   - Anthropic has a top-level `system` (string or blocks) **and** Claude Code's mid-conversation `role:"system"` messages.
   - OpenAI accepts `system`/`developer` at any position.
   - Responses has `instructions` plus `developer`/`system` input items; Codex uses only developer items.
   - → Keep ordered `SystemPart`s. Join the leading parts deterministically, without reordering (`"\n\n"` between blocks). Render later system parts as system turns, which both Qwen templates support.
   - Ignore `cache_control` everywhere.
2. **Usage semantics differ.**
   - OpenAI `prompt_tokens` and Responses `input_tokens` **include** cached tokens. Anthropic `input_tokens` **excludes** them.
   - Report `cache_creation_input_tokens: 0` until feature 003 defines writes.
   - Reasoning tokens are part of the output tokens in all three APIs.
3. **Max tokens.** Chat: `max_completion_tokens` takes precedence over `max_tokens`. Anthropic: `max_tokens` is required. Responses: `max_output_tokens`. Each bounds reasoning plus answer. Clamp to `capacity − prompt` and error only if fewer than 1 token remains (spec edge case).
4. **Thinking control.**
   - Anthropic `thinking.enabled` and `adaptive` → on; `disabled` and `between_tools` → off. `budget_tokens`: force `</think>` at the budget [U quality].
   - OpenAI `reasoning_effort` / Responses `reasoning.effort`: `none`/`minimal`/absent → off.
   - Claude Code sends `adaptive` on every request, so an "only when requested" policy will think on Claude Code turns. Qwen3-Coder has no thinking, so the primary model is unaffected; for Qwen3 hybrids `mrun connect` can set `MAX_THINKING_TOKENS=0` [U effect behind a base URL].
5. **Reasoning in history.** The Qwen3 template re-renders reasoning only inside the current tool loop. Replay it from Anthropic `thinking` blocks, Chat `reasoning_content`/`reasoning`, and Responses `reasoning.encrypted_content`. Codex never sends `reasoning_text` back, so mrun must put the reasoning text (versioned and base64-encoded) into `encrypted_content`. Signatures and encrypted blobs are opaque, unsigned, and never rejected.
6. **Arguments round-trip.**
   - OpenAI and Responses history carries `arguments` as a JSON string; Anthropic `input` is an object.
   - Qwen3 renders a string verbatim; Qwen3-Coder needs an object.
   - Return the model's **raw** argument text as `arguments` and re-render it unchanged, so the prompt prefix is byte-stable. For Anthropic, also keep `serde_json` key order (`preserve_order`).
   - Invalid JSON arguments in history → 400 naming `messages[i].tool_calls[j]`.
7. **Parallel calls and results.**
   - Anthropic: several `tool_use` blocks in one assistant message; the results come as `tool_result` blocks in the **next user message**, which may also contain text.
   - OpenAI: one `tool` message per call.
   - Responses: `function_call_output` items.
   - Qwen merges consecutive results into one user turn and identifies them **by order, not id**. → Validate that every result id matches an earlier call (spec: unknown id → 400) and render results in the order received.
   - `disable_parallel_tool_use` / `parallel_tool_calls:false` → stop after the first `</tool_call>`.
8. **tool_choice.**

   | API | auto | none | required | named |
   |---|---|---|---|---|
   | Anthropic | `auto` | `none` | `any` | `tool` |
   | Chat | `"auto"` | `"none"` | `"required"` | `{type,function:{name}}` |
   | Responses | `"auto"` | `"none"` | `"required"` | `{type:"function",name}` |

   - Keep tools in the prompt even for `none`, for prefix stability, and mask the `<tool_call>` token instead [U model effect].
   - `required`/named are enforced by grammar.
9. **Ids.** The model emits no ids. Generate `call_…` (Chat), `toolu_…` (Anthropic), and `fc_…` item id + `call_…` call_id (Responses). Ids never enter the Qwen prompt, so they cannot hurt the cache.
10. **Rejections** (name the JSON path):
    - Anthropic `image`/`document` blocks, including inside `tool_result`.
    - OpenAI `image_url`/`input_audio`/`file` parts.
    - Responses `input_image`/`input_file`, including in Codex `function_call_output` arrays.
    - `n>1`, `logprobs`, and `previous_response_id`.
    - Accept and ignore `metadata`, `service_tier`, `context_management`, `output_config.effort` (it maps to thinking), `prompt_cache_key`, and betas.
11. **Stop sequences.**
    - Apply to answer text only, never inside reasoning or a tool call. Hold back partial matches in the stream.
    - Anthropic reports `stop_reason:"stop_sequence"` plus the string. Chat reports `"stop"`. Responses has no stop parameter.
    - The `<|im_end|>` EOS is always active.
12. **Prefill.** An Anthropic trailing assistant message is rendered **without** `<|im_end|>` and without a generation prompt, and its text is continued. In Chat, a trailing assistant message is ordinary history: start a new turn [U OpenAI semantics]. When thinking is on, prefill goes after a closed `</think>` (llama.cpp's continuation logic, `qwen3-coder.cpp:45-60`).
13. **Truncation mid-tool-call.** Chat `finish_reason:"length"` with no further deltas; Anthropic `max_tokens`; Responses `function_call.status:"incomplete"` + `response.incomplete` (Codex surfaces this as an error, which is truthful). Never report the call as complete.
14. **Template rendering fidelity.** mrun renders Jinja in Rust. The `tojson` output (key order, separators `", "`/`": "`, `ensure_ascii=False`) must match HF transformers byte for byte on fixtures, or the prefix cache and model behavior drift [U minijinja defaults].

## Design recommendations

**Internal request model** (`crate::api::model`):
```rust
pub struct ChatRequest {
    pub model: String, pub api: ApiKind, pub request_id: String,
    pub system: Vec<SystemPart>,              // leading system/developer/instructions, in order
    pub messages: Vec<Message>,               // User | Assistant | Tool | System (mid-conversation)
    pub tools: Vec<ToolDef>,                  // name, description, schema: serde_json::Value (preserve_order), strict: bool
    pub tool_choice: ToolChoice,              // Auto | None | Required | Named(String)
    pub parallel_tool_calls: bool,
    pub response_format: ResponseFormat,      // Text | JsonObject | JsonSchema { name, schema, strict }
    pub sampling: Sampling,                   // temperature, top_p, top_k, min_p, seed
    pub max_output_tokens: Option<u32>, pub stop: Vec<String>,
    pub reasoning: ReasoningCtl,              // Off | On { budget: Option<u32> }, plus return: Show | Hide
    pub prefill: bool,                        // Anthropic-only: last assistant message is continued
    pub stream: bool, pub include_usage: bool,
}
pub enum AssistantPart { Text(String), Reasoning { text: String }, ToolCall { id: String, name: String, raw_args: String } }
pub struct ToolResult { pub call_id: String, pub content: String, pub is_error: bool }
```
- Decoders (`chat::decode`, `messages::decode`, `responses::decode`) are pure functions `Json → Result<ChatRequest, ApiError{status, kind, param_path, message}>`.
- Validate before queuing: size, counts, schema depth/size, ids, unsupported modalities.
- `count_tokens` = decode → render → tokenize, with no queue.

**Event model** (engine → encoders):
```rust
enum GenEvent { Started { input: u32, cached: u32 }, ReasoningDelta(String), TextDelta(String),
  ToolStart { idx: u32, id: String, name: String }, ToolArgsDelta { idx: u32, json: String },
  ToolEnd { idx: u32, raw_args: String, valid: bool }, Finished { reason: Finish, stop_seq: Option<String>, usage: Usage }, Failed(ApiError) }
enum Finish { Stop, StopSequence, Length, ContextFull, ToolCalls, Cancelled }
```

| Finish | Chat | Anthropic | Responses |
|---|---|---|---|
| Stop | `stop` | `end_turn` | completed |
| StopSequence | `stop` | `stop_sequence` | completed |
| Length | `length` | `max_tokens` | incomplete/`max_output_tokens` |
| ContextFull | `length` | `model_context_window_exceeded` | incomplete/`max_output_tokens` |
| ToolCalls | `tool_calls` | `tool_use` | completed |

`Cancelled` is never written.

**Per-API encoders.** Each encoder is a state machine over `GenEvent` that owns indices and ids and writes SSE frames. Each handles both streaming and non-streaming by folding the same events.
- **Anthropic:** emits `ping` every 10 s of silence [U interval], a `signature_delta` (`"mrun1."+b64(sha256(thinking))[..16]`) before closing a thinking block, and `cache_read_input_tokens` in `message_start` and `message_delta`.
- **Chat:** emits the usage chunk after the finish chunk, then `[DONE]`, and nothing after `finish_reason`.
- **Responses:** emits `sequence_number`, `output_index` and `content_index`; complete `output_item.done` items (reasoning with `summary:[]` + `encrypted_content`); and errors as `response.failed` with `code` `context_length_exceeded`.

**Tool parser.** Use a `trait ToolSyntax { fn feed(&mut self, tok: TokenId, text: &str) -> SmallVec<ParseEvent>; fn finish(&mut self, cause) -> … }` with impls `HermesJson` (Qwen3) and `QwenXml` (Qwen3-Coder, schema-typed), selected from the template signature the way mlx-lm and llama.cpp do it.
- A shared `HoldBack` handles partial tags and stop strings.
- Reasoning start comes from the prompt tail.
- `<tool_call>` implicitly ends reasoning.
- Accept a missing `<tool_call>` before `<function=NAME>` only for declared names.
- Strict tools and JSON formats arm an llguidance `Matcher` on trigger, using a `ParserFactory` cached per model.

**Conformance test list** (`tests/api_conformance/`, ≥60):
- **chat:**
  - `chat_basic_nonstream`, `chat_basic_stream_done_terminator`, `chat_stream_role_first_chunk`, `chat_include_usage_chunk_empty_choices`
  - `chat_usage_cached_tokens_inclusive`, `chat_tool_call_stream_first_fragment_has_id_name`, `chat_tool_call_args_fragments_concat_valid_json`, `chat_parallel_tool_calls_distinct_index_and_id`
  - `chat_tool_result_roundtrip`, `chat_tool_result_unknown_id_400`, `chat_tool_choice_none_no_calls`, `chat_tool_choice_required_forces_call`
  - `chat_tool_choice_named`, `chat_strict_tool_args_validate_schema`, `chat_nonstrict_invalid_json_reports_parse_error`, `chat_response_format_json_schema_strict`
  - `chat_response_format_json_object`, `chat_reasoning_content_separate_from_content`, `chat_reasoning_effort_none_no_think`, `chat_reasoning_replay_accepts_reasoning_and_reasoning_content`
  - `chat_max_completion_tokens_precedence`, `chat_length_mid_tool_call_not_complete`, `chat_stop_sequence_holdback`, `chat_no_delta_after_finish_reason`
  - `chat_image_part_rejected_with_path`, `chat_unknown_field_ignored`, `chat_context_length_exceeded_400_code`, `chat_empty_tools_array_with_history`
  - `chat_utf8_split_multibyte_stream`
- **messages:**
  - `msg_basic_nonstream`, `msg_stream_full_event_order`, `msg_ping_during_silence`, `msg_query_beta_true_path`
  - `msg_x_api_key_and_bearer_auth`, `msg_system_string_and_blocks`, `msg_midconversation_system_message`, `msg_cache_control_ignored`
  - `msg_usage_input_excludes_cache_read`, `msg_message_delta_usage_cumulative`, `msg_tool_use_input_json_delta`, `msg_parallel_tool_use_blocks`
  - `msg_tool_result_blocks_with_text`, `msg_tool_result_is_error`, `msg_tool_result_image_rejected`, `msg_tool_choice_any_and_tool`
  - `msg_disable_parallel_tool_use`, `msg_thinking_enabled_blocks_with_signature`, `msg_thinking_adaptive_maps_on`, `msg_thinking_disabled_no_think`
  - `msg_signature_not_validated`, `msg_thinking_replay_in_tool_loop`, `msg_assistant_prefill_continues`, `msg_stop_sequence_reports_value`
  - `msg_max_tokens_stop_reason`, `msg_context_overflow_prompt_too_long`, `msg_overloaded_529_retry_after_int`, `msg_stream_error_event_shape`
  - `msg_output_config_format_structured`, `msg_context_management_ignored`, `msg_count_tokens_matches_prompt`, `msg_count_tokens_no_queue`
  - `msg_head_api_hello_ok`
- **responses:**
  - `resp_basic_nonstream`, `resp_stream_event_order_and_sequence_numbers`, `resp_developer_message_as_system`, `resp_function_call_item_done_parsable_by_codex`
  - `resp_function_call_output_string`, `resp_function_call_output_array_text`, `resp_function_call_output_image_rejected`, `resp_reasoning_item_has_summary_array`
  - `resp_encrypted_content_roundtrip_restores_think`, `resp_reasoning_present_without_effort_no_think`, `resp_previous_response_id_rejected`, `resp_item_reference_rejected`
  - `resp_store_true_accepted_not_persisted`, `resp_usage_cached_tokens`, `resp_incomplete_max_output_tokens`, `resp_failed_context_length_exceeded_code`
  - `resp_text_format_json_schema`, `resp_web_search_tool_dropped_with_warning`, `resp_custom_tool_rejected`
- **cross-cutting:**
  - `xapi_same_prompt_same_tokens_all_apis`, `xapi_raw_args_rerender_byte_stable`, `xapi_qwen3_coder_missing_tool_call_tag_recovered`, `xapi_qwen3_coder_param_types_by_schema`
  - `xapi_thinking2507_bare_close_think`, `xapi_queue_full_immediate_error`, `xapi_queue_deadline_timeout`, `xapi_disconnect_cancels_within_one_token`
  - `xapi_disconnect_while_queued_releases_slot`, `xapi_body_over_8mib_413`, `xapi_schema_depth_limit`, `xapi_tool_count_limit`
  - `xapi_nonloopback_requires_key`, `xapi_no_prompt_content_in_logs`

## Sources
- OpenAI: https://platform.openai.com/docs/api-reference/chat/streaming ; https://platform.openai.com/docs/api-reference/responses-streaming ; https://platform.openai.com/docs/guides/reasoning ; SDK types `OS` above.
- Anthropic: https://platform.claude.com/docs/en/build-with-claude/streaming ; https://platform.claude.com/docs/en/api/errors ; https://platform.claude.com/docs/en/build-with-claude/token-counting ; SDK types `AS`.
- Claude Code: https://code.claude.com/docs/en/llm-gateway-protocol ; https://code.claude.com/docs/en/model-config ; https://code.claude.com/docs/en/errors ; https://www.npmjs.com/package/@anthropic-ai/claude-code (2.1.300).
- Codex `CX`, OpenCode `OC`, Ollama `OL`, llama.cpp `LC`, vLLM `VL`, mlx-lm `ML`, llguidance `LG` (commit links in the header); XGrammar Rust crate https://docs.rs/crate/xgrammar/0.5.0 ; Qwen quickstart https://qwen.readthedocs.io/en/latest/getting_started/quickstart.html
