# HTTP Contract: Local Chat Runtime

**Status**: Planned v1 boundary; not a working endpoint or universal OpenAI compatibility claim.
**Client**: OpenCode v1.19.0 at `add9b4ba5adfb895934cb1bf46f590342cec68d2`;
native OpenAI-compatible chat route and retained SDK 2.0.41, as documented in [research](../research.md).
Default origin `http://127.0.0.1:8012`; HTTP/1.1; UTF-8 JSON and SSE; no remote exposure or CORS.

## Access and Endpoints

One trusted OS user; same-user apps are not tenants. Bind only literal loopback addresses.
Reject nonloopback bind configuration in this feature. Reject browser Origin headers unless an
explicit exact loopback-origin allowlist is configured; reject nonlocal Host authorities. Do not
use bearer tokens, user/session headers or cache keys as scope selectors. Ignore bounded client
transport headers without logging them. Disable request-body/native content logs. No outbound fetch.

| Method/path | Behavior |
| --- | --- |
| `GET /healthz` | 200 liveness without loading a model; no private details |
| `GET /v1/models` | Configured admitted model metadata; no load; public ID and capabilities only |
| `POST /v1/chat/completions` | Validate, reserve, queue and generate; stream false by default |
| `GET /v1/runtime/status` | Lifecycle, finite limits, cache/accounting metadata; no private paths |
| `DELETE /v1/runtime/cache` | Advance epoch, invalidate hits; 200 after safe clear; no model unload |
| `POST /v1/runtime/unload` | Drain/cancel queue and active work, synchronize/unload; 200 only after release |

Runtime controls execute through the owner, not concurrently with native mutation. Cache-clear/unload
are idempotent. New generation while draining returns 503. Responses/embeddings/completions,
remote provisioning, audio/images, custom code execution and tenant APIs are unsupported (404).
Discovery uses standard `{object:"list",data:[{id,object:"model",created,owned_by:"mrun"}]}`;
extension capabilities are namespaced `mrun`. A configured but unqualified artifact cannot serve.

## Supported Request Matrix

Unknown top-level, message, tool or semantic nested fields return 400; do not silently ignore options.
Absent means default below; explicit null is rejected except assistant content with tool calls.
Duplicate JSON keys, nonfinite numbers, invalid UTF-8 and trailing JSON are rejected.
Limits and pressure semantics are defined in [data-model.md](../data-model.md).

| Field | Validation and default |
| --- | --- |
| `model` | Required exact admitted ID `qwen3-8b-q4-k-m`; unknown =>404 |
| `messages` | Required 1..512; supported roles system, user, assistant, tool |
| `content` | String; user also ordered `{type:"text",text:string}` parts; no media/native annotations |
| Assistant content | String or null only with one valid historical function tool call |
| `stream` | Boolean, default false |
| `stream_options` | Only `include_usage:boolean`; allowed only with stream=true |
| `max_tokens` | Integer 1..2048; default 2048; prompt+allowance <=16384 after rendering |
| `temperature` | Finite 0..2; default 0.7; zero means greedy |
| `top_p` | Finite (0,1]; default 0.8; applied after temperature |
| `frequency_penalty`, `presence_penalty` | Finite [-2,2]; default zero; request-local history only |
| `seed` | Integer 0..4294967295; absent uses fresh request RNG; never inherited from cache |
| `stop` | String or <=4 strings, each 1..256 UTF-8 bytes; default none |
| `tools` | <=64 distinct function tools; type, function.name/description/parameters/strict only |
| `tool_choice` | auto (default with tools), none, required, or named function; required/name need tools |
| `parallel_tool_calls` | Only false accepted; at most one new tool call per completion |
| `store` | Only false accepted; no persistence |
| `user` | Optional bounded 128-byte attribution string; not logged, not cache authorization |

Reject `n` except 1; reject `max_completion_tokens`, top_k, logprobs/logit_bias, response_format,
reasoning_effort, reasoning_content, adapters, custom templates, prompt_cache_key and thinking
controls. Missing/default non-thinking is selected in the model adapter, not by dropping options.
Reasoning/multimodal/provider variants are disabled in the pinned client configuration.
If a captured released-client request requires any rejected semantic field/schema, interoperability
is blocked until the contract is explicitly reconciled; do not remove the field in a proxy.

Tool names match `[A-Za-z0-9_-]{1,64}`; description <=8 KiB; tool-call IDs <=128 bytes.
Historical assistant calls have type=function, stable id, declared name and JSON arguments;
validate arguments when their tool is declared. If a tool is absent from the current declaration,
reject the history rather than invent its schema. Exactly one tool result must follow each
outstanding call before new user/assistant work; orphan/duplicate/mismatched IDs are errors.
Tool result content is an opaque bounded string, never executed/interpreted as instructions by mrun.
Allow ordered system messages before conversation; require a user or linked tool continuation.
No assistant prefill, developer role, parallel historical calls or repair of malformed history.

## Tool Schema Subset

Use a real JSON Schema validator and backend-native grammar; validate supported keywords before
render/inference. Root parameters must be an object. Accepted keywords: `type`, `properties`,
`required`, boolean `additionalProperties`, `items` with one schema, `enum`, `const`, disjoint-type
`anyOf` (including nullable primitives), `minItems`, `maxItems`, `minLength`, `maxLength`, `minimum`,
`maximum`, and annotations `description`, `title`, `default`, `$schema` (draft 2020-12 only).
Primitive types string, number, integer, boolean and null; nested arrays/objects permitted.
Annotation defaults do not insert argument values. OpenCode's projected built-in schemas must
fit this subset; freeze those exact schemas from public fixtures, not hand-written approximations.

Reject `$ref`/`$defs`, remote/file resolution, recursive schemas, tuple arrays, regex/format,
oneOf/allOf/not/conditionals, patternProperties and schema-valued additionalProperties. Reject anyOf
whose branches overlap by primitive type; no silently altered grammar or discarded constraints.
Optional properties remain optional; undeclared properties follow the explicit flag (default true).
`strict` absent/false/true all retain the same advertised validation; true is not a promise to support
arbitrary schemas. Cap depth/nodes/bytes as in the data model. Native grammar may overaccept a
numeric/string facet only if final validation prevents any invalid completed tool call.

Generate at most one tool call. With auto, use template-native lazy grammar at the tool delimiter;
with required/named, constrain selection appropriately. Run final name/schema validation before
emitting a successful tool terminal. Grammar/schema disagreement, overflow, incomplete JSON or
unknown tool is a generation error. Structural validity does not establish safe tool choice.

## Completion and Streaming

Nonstream response: standard chat.completion with stable opaque id, object, created, model,
one choice index=0, assistant message, finish_reason and usage. Finish reasons: stop for valid text
EOS/stop; tool_calls only for a complete validated call; length only for incomplete text due to token
allowance. Exhaustion inside a tool call is an error, never tool_calls or a successful partial call.
Stop matching holds possible suffix bytes until resolved; generated-token usage includes sampled
tokens consumed in stop detection, not only visible text. UTF-8 output is incremental and valid.

SSE uses `data: <JSON>\n\n`, object chat.completion.chunk, stable response id/model/created,
one choice index=0. First delta declares role assistant. Text deltas are ordered valid UTF-8.
Tool deltas use index=0; first contains stable server-generated id, type=function and name;
later deltas append only argument fragments, never change identity. Partial fragments are not
completed calls. End with exactly one choice finish signal, optional usage-only chunk with
choices=[], then `data: [DONE]\n\n`. No ordinary chunks follow the finish/usage terminal sequence.
Do not send the success finish signal until native completion and validation are known safe.

`usage.prompt_tokens=T`, `completion_tokens=sampled output tokens`, `total_tokens=sum`;
`prompt_tokens_details.cached_tokens=effective_reused`. Runtime status/evidence separately records
new/processed/boundary replay counts and stage times. Tool payload tokens count as output, not SSE
events. Cache hits are not invented evaluated tokens. Nonstream responses always contain usage.

After HTTP headers, failures emit a single `data: {"error":...}\n\n` if the consumer remains writable,
then close without a successful finish or `[DONE]`. If transport has failed, simply close and record
cancelled/failed. The pinned client must observe failure rather than interpret it as successful text
or execute a partial call. Capacity release follows GPU completion, not HTTP close alone.

## Errors

JSON envelope `{error:{message,type,code,param},request_id}`; stable codes, sanitized messages,
no submitted values, tools, secrets, file paths or native traceback. Include limiting resource and
allowed bound in safe numeric metadata. `param` is a field name or null, not caller data.

| HTTP | Codes |
| --- | --- |
| 400 | invalid_request, unsupported_option, invalid_tool_history, unsupported_schema, context_limit |
| 404 | model_not_found, endpoint_not_found |
| 413 | body_limit, headers_limit, schema_limit |
| 429 | queue_full, resource_exhausted; Retry-After=1 where retry may help |
| 503 | queue_timeout, model_unavailable, memory_pressure, backend_failed |
| 500 | invalid_generation, output_limit, internal_error |

Expired queued work is 503 before streaming starts. Cancellation is recorded locally; do not invent
a 499 response to a disconnected socket. Timing and raw status are kept in sanitized evidence.
Actual conformance is established only by the [20 fixtures](qualification.md) and pinned-client run.