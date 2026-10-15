# Feature Specification: Agent APIs

**Feature Branch**: `002-agent-apis`

**Created**: 2026-10-14

**Status**: Draft (revised 2026-10-15 after the third audit; open model architecture, D015; lean 0.1 cut, D025)

**Input**: User description: "`mrun serve` exposes OpenAI Chat Completions, Anthropic Messages (with count_tokens) and stateless OpenAI Responses on loopback, with streaming, tool calls and structured output, so Claude Code, Codex and OpenCode complete real tool loops against a local model. One internal request model; truthful usage and finish reasons; bounded queue; cancellation on disconnect."

## Clarifications

### Session 2026-10-14

- Q: Do models think when the client doesn't ask? → A: No. Thinking is enabled only by the request (Anthropic `thinking`, OpenAI `reasoning_effort` / `reasoning`). (Maintainer decision.)
- Q: How are concurrent requests handled? → A: One generation at a time, FIFO queue (600 s waiting deadline; 16 waiting since 2026-10-15, matching D018); batching is a later feature. (Maintainer decision.)
- Q: Default port? → A: 127.0.0.1:8484 (avoids Ollama 11434, LM Studio 1234, llama.cpp 8080); `--port` and `MRUN_SERVER_PORT` override.
- Q: Are non-strict tool calls constrained? → A: Tool-call syntax is always parsed with the model's native format; argument schemas are enforced by constrained decoding only for strict tools and JSON response formats, keeping the default fast path unconstrained.
- Q: What does `count_tokens` count? → A: Exactly the tokens mrun would process for that request with the loaded model's tokenizer and template.

### Session 2026-10-15 (third audit)

- Anthropic `diagnostics` (old FR-023) is dropped: no supported client sends `diagnostics.previous_message_id`. The field is accepted and ignored.
- Keepalives are per API, because Codex's SSE parser drops comment lines and aborts after 300 s without an event: Responses sends `response.in_progress` events, Anthropic `ping`, Chat SSE comments.
- Client cache hints are gone (003 dropped session and breakpoint hints): `cache_control`, `prompt_cache_key` and client session headers do not affect the cache.
- A strict schema llguidance cannot compile runs unconstrained with a `strict-unsupported` warning instead of failing.
- `tool_choice: none` uses a fixed logit bias, not a grammar mask, so decode stays pipelined.

### Session 2026-10-15 (open model architecture, D015)

- Q: How are tool calls and reasoning parsed across model families? → A: Through a syntax registry: one streaming parser per syntax (0.1: Hermes JSON, Qwen XML, Harmony; Mistral `[TOOL_CALLS]` with 0.2, GLM XML with 010). Each family names one syntax; load checks that the template renders that syntax's markers, else refuses (FR-032). Every FR-004 rule applies to every syntax.
- Q: How is thinking controlled on gpt-oss, which always reasons? → A: By reasoning effort (low/medium/high) mapped from Anthropic `thinking` and OpenAI `reasoning_effort` (research §3b); "off" means `low`. Its reasoning is always returned in the API's reasoning field.
- Q: Models without a thinking mode (Qwen3-Coder)? → A: Served without thinking; a request that asked for it is logged server-side (`thinking-unsupported`).

### Session 2026-10-15 (lean 0.1 cut, D025)

- One client gate (SC-001): each 0.1 alias × each client, started with `mrun launch`, 3 of 3 runs. Pinned client versions run on every PR; the latest releases are checked by hand before a release (Claude Code npm `stable` and `latest`).
- In-band warnings are `tool-args-invalid`, `max-tokens-clamped` and `image-omitted`; every other condition is logged on the server.
- Removed: progress in keepalives (FR-026), the `--thinking` server override, non-loopback binds with API keys (FR-012 second part, Later), the Mistral parser (FR-034, Later, with Devstral in 0.2).
- Codex never sends `max_output_tokens` and, from 0.163, ends the turn on `response.incomplete`: Responses requests without a limit get `capacity − prompt` (FR-019).
- `--record` keeps the request body, arrival time and allow-listed session headers, including Claude Code's `x-claude-code-prev-tool-durations` (FR-021).

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Use a local model from my coding agent (Priority: P1)

A developer starts `mrun serve`, points their agent (OpenCode, Claude Code or Codex) at it, and
completes a real task: the agent reads files, edits code and runs tests through tool calls, all on
the local model.

**Why this priority**: This is the product. Without working agent loops the engine is a demo.

**Independent Test**: Start each client with `mrun launch` (004) against each 0.1 alias, run the
scripted read-edit-test task in a fixture repository and confirm it completes with the tests
passing (SC-001). Before 004 lands, the quickstart's hand-written client configs are used.

**Acceptance Scenarios**:

1. **Given** `mrun serve` and OpenCode configured for an OpenAI-compatible provider, **When** the
   developer asks for a code change, **Then** the agent's tool calls arrive well-formed, tool
   results are accepted, and the session completes.
2. **Given** Claude Code pointed at mrun, **When** the developer runs a task, **Then** streaming,
   tool use, tool results and token counting all work without client-side errors.
3. **Given** Codex pointed at mrun, **When** the developer runs a task, **Then** the Responses
   stream, function calls and function-call outputs work without client-side errors.
4. **Given** Claude Code configured with mrun's capacity, **When** a session grows past that
   capacity, **Then** Claude Code compacts on its own and the session continues.

---

### User Story 2 - Clear, honest responses (Priority: P1)

Every response tells the client exactly what happened: tokens used (including how many came from
cache), why generation stopped, and a specific error when a request cannot be served.

**Why this priority**: Agents budget context from usage numbers and branch on finish reasons; wrong
numbers or vague errors break them in subtle ways.

**Independent Test**: Run the API conformance suite: valid, invalid, truncated, cancelled and
oversize requests across all three APIs, checking status codes, error bodies and usage fields.

**Acceptance Scenarios**:

1. **Given** a completed request, **When** the response ends, **Then** usage reports input, cached
   input and output tokens, and they sum consistently.
2. **Given** a request whose prompt alone does not fit (no room for output), **When** submitted,
   **Then** the API returns the API-appropriate "context too long" error with the token numbers.
3. **Given** an unsupported parameter that changes meaning (for example image input), **When**
   submitted, **Then** the request is rejected naming the parameter; unknown harmless fields are
   ignored.
4. **Given** generation stops at the output limit in the middle of a tool call, **When** the stream
   ends, **Then** the finish reason says length and no incomplete tool call is reported as complete.
5. **Given** a streaming request whose headers are already sent, **When** it fails (for example the
   queue deadline passes), **Then** the client receives the API's in-stream error event.

---

### User Story 3 - Reliable under agent behavior (Priority: P2)

Agents disconnect, retry and fire overlapping requests (subagents). mrun queues fairly, cancels
abandoned work immediately, and never wedges.

**Why this priority**: Real agent traffic includes concurrency and disconnects; a stuck server ends
the session.

**Independent Test**: Fire overlapping requests, disconnect mid-stream, mid-queue, mid-prefill and
during a non-streaming request, and stall a reader; verify queue order, prompt cancellation,
timeouts and that the next request runs normally.

**Acceptance Scenarios**:

1. **Given** a request that is queued, prefilling, generating or non-streaming, **When** the client
   closes the connection, **Then** its work stops within 300 ms and the next queued request starts.
2. **Given** a full queue, **When** another request arrives, **Then** it is rejected immediately
   with a retry-after hint.
3. **Given** a queued request waiting past its deadline (default 600 s), **When** the deadline
   passes, **Then** it is rejected with a timeout error and never starts; while it waits, streaming
   clients receive keepalives that their idle timers count (including Codex, which ignores SSE
   comments).

### Edge Cases

- Tool definitions with deeply nested or very large schemas: rejected above documented limits.
- A strict schema llguidance cannot compile: generated unconstrained; logged (`strict-unsupported`).
- Parallel tool calls requested: the model may emit several; each is reported with a stable id.
- A tool result referencing an unknown call id: rejected with the id named.
- Streaming split inside a multi-byte character or inside tool-call JSON: output stays valid.
- Assistant prefill (Anthropic trailing assistant message): continued, not restarted.
- Thinking/reasoning content from the model: returned in the API's reasoning field or omitted per
  request, never mixed into the answer text.
- `max_tokens` above capacity or absent: clamped / defaulted per FR-019; rejected only when the
  prompt itself does not fit.
- History tool-call arguments that are not a JSON object, sent to a template that re-serializes
  arguments (Qwen3-Coder, Qwen3.6, gpt-oss): rendered as raw text; logged (FR-004).
- gpt-oss emitting a call to a built-in tool (`browser.*`, `python`): not reported as a call;
  logged (`tool-unknown`).
- A history assistant message with several tool calls sent to gpt-oss (whose template renders one
  call per message): split into single-call messages; logged (`history-calls-split`).
- A streaming request queued for more than 300 s: the stream stays alive for every client.
- Images inside tool results (Claude Code reading a PNG, Codex `view_image`): replaced per FR-020.
- A browser page on the same machine calling the server: rejected by the Host/Origin policy.
- Non-loopback bind requested: refused; 0.1 binds loopback only (FR-012).

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: `mrun serve` MUST expose OpenAI Chat Completions (`POST /v1/chat/completions`),
  Anthropic Messages (`POST /v1/messages`, `POST /v1/messages/count_tokens`) and stateless OpenAI
  Responses (`POST /v1/responses`), plus model listing (`GET /v1/models`), health (`GET /health`)
  and server status (`GET /mrun/status`, versioned JSON: model, budget, active/queued requests,
  cache stats, last miss).
- **FR-002**: All three APIs MUST support streaming (server-sent events in each API's native event
  format) and non-streaming responses.
- **FR-003**: All three APIs MUST support tool definitions, tool calls (including parallel calls)
  and tool results, translated through one internal request model so behavior is identical.
  `tool_choice: none` keeps tools rendered (prefix stability) and applies a fixed −inf logit bias
  to the syntax's tool-call opener tokens (Harmony: the recipient token, only inside an assistant
  message header), so decoding stays pipelined; `required` and named choices are
  enforced by grammar.
- **FR-004**: Tool-call arguments MUST be valid JSON matching the declared schema when the client
  requests strict tools (`strict: true` or equivalent), using constrained decoding; tokens the
  grammar forces (scaffold such as `</parameter>\n<parameter=`) are appended without sampling,
  which is exact and saves steps (SchemaFill, delta §5). A strict schema that llguidance cannot
  compile is generated unconstrained and logged (`strict-unsupported`), never rejected. Non-strict
  arguments are never modified. If they are not valid JSON:
  Chat Completions and Responses return the call with its raw arguments string, and such raw
  strings are accepted unvalidated in later history; Anthropic Messages (whose `input` must be an
  object) returns the model's raw call text as a text block instead of a tool call. Either way a
  `tool-args-invalid` warning is reported. When streaming, argument text is forwarded as it arrives
  only where it is the model's raw JSON (Hermes and Harmony syntaxes on Chat/Responses);
  arguments synthesized from other syntaxes (Qwen3-Coder and Qwen3.6 XML) and all Anthropic
  `tool_use` input
  are buffered per call and emitted when the call closes, so the fallback above is always
  possible. The tool-call envelope (for example a missing `<tool_call>` opener before a declared
  function) MAY be recognized, because it does not change argument content. Values are never
  coerced or evaluated (research §3); argument text that contains the syntax's own closing tags (a
  file edit of a parser) MUST round-trip unchanged; a call to an undeclared tool is passed through
  and logged (`tool-unknown`).
  These rules apply to every syntax in the registry (FR-032).
  **History re-rendering is byte-stable per template**: Qwen3 renders string arguments verbatim,
  so the previous reply re-renders exactly. Qwen3-Coder, Qwen3.6 (`|items`, `tojson`) and gpt-oss
  (`tojson`) re-serialize arguments, so the next turn may miss at the previous reply, reported as
  cause `changed` at that message (003; bench diagnoses it as a template re-render). On those
  templates, history arguments that do not parse as a JSON object are rendered verbatim in place of
  the parameter list or JSON and logged (`history-args-raw`).
- **FR-005**: Structured output (JSON schema response format) MUST be supported with the same
  constrained decoding and the same `strict-unsupported` fallback.
- **FR-006**: The Responses API MUST be stateless: `previous_response_id` and item references MUST
  be rejected with a clear error; `store: true` is accepted and nothing is stored.
- **FR-007**: Usage MUST report input, cached input and output tokens with each API's semantics.
  For a prompt of `P` tokens `cached = min(resume, P − 1)`; resume = match (attention) or restored checkpoint position (hybrid), per 003 FR-006; the last prompt token is always processed. Anthropic `input_tokens` excludes cache reads (`input_tokens +
  cache_read_input_tokens = P`) and `cache_creation_input_tokens` is always 0 (mrun has no billed
  cache writes; reporting 0 keeps clients' cost estimates at zero); OpenAI totals include cached
  tokens. Finish reasons MUST follow the mapping table in
  [research](research.md#design-recommendations) (stop, stop sequence, length, context full, tool
  calls); errors are error events, never finish reasons.
- **FR-008**: Errors MUST use each API's native error shape, with a message that names the cause
  and, where possible, the fix. Errors after response headers are sent use the API's in-stream form
  (Anthropic `event: error` with `overloaded_error` for queue failures, Chat `data: {"error":…}`,
  Responses `response.failed`). Status codes follow [contracts/api.md](contracts/api.md#status-codes).
- **FR-009**: Requests MUST be validated before queuing against configurable limits (defaults: body
  64 MiB, since agent histories carry base64 screenshots; 4,096 messages, 256 tools, schema 64 KiB and depth 16); errors name the limit and the
  setting that raises it.
- **FR-010**: The server MUST run one generation at a time with a FIFO queue (default 16 waiting).
  The queue deadline covers waiting time only (default 600 s, configurable). Streaming responses
  MUST send headers at once and, in every phase (queued, prefilling, generating, including while a
  tool call is buffered per FR-004), a keepalive whenever 10 s pass without an event (well under
  the 180 s after which Claude Code before 2.1.299 aborts a silent custom-base-URL stream), in a
  form the client's idle timer counts: Responses `response.created` at once, then
  `response.in_progress` (Codex drops SSE comments and aborts after 300 s without an event);
  Anthropic `ping` (Claude Code aborts after 300 s without bytes); Chat SSE comments. Full-queue
  and deadline errors MUST be immediate and explicit; `retry-after` is integer seconds, at most 60
  (larger values stop Claude Code's retries).
- **FR-011**: Client disconnect MUST cancel work promptly, within 300 ms, whether the request is
  queued, prefilling, generating or non-streaming, and release its reservation. Disconnects are
  detected by watching each connection for close, not only by failed writes.
- **FR-012**: The server MUST bind to loopback only and accept only loopback `Host` headers
  (`127.0.0.1`, `localhost`, `[::1]` with the server port), reject requests carrying a browser
  `Origin` other than the server's own, and send no CORS headers. A non-loopback `--host` MUST be
  refused at startup. *Second part removed (0.1 scope cut, 2026-10-15; see Later): binding
  elsewhere with an API key from `MRUN_API_KEY` or `--api-key-file`, compared in constant time.*
- **FR-013**: Reasoning/thinking output MUST be separated from answer text per each API's
  conventions. Thinking control is per family (FR-035). Thinking is on only when the request asks for it (Anthropic `thinking` enabled or
  adaptive; OpenAI `reasoning_effort` / Responses `reasoning.effort` not in {none, minimal}; a bare
  `reasoning` object does not count). *The `--thinking client|off|on` server override is removed
  (0.1 scope cut, 2026-10-15).* Anthropic `budget_tokens` is honored by closing the thinking block
  when the budget is reached. For Qwen3.6, history is rendered with the template's
  preserve-thinking option (`preserve_thinking`), with reasoning replayed per FR-016, so earlier
  reasoning stays in the prompt and a new user turn does not re-process the previous tool loop
  (its token cost is measured in 005).
- **FR-014**: The server MUST NOT log prompt or completion content by default; request logs contain
  timings, token counts and outcome only.
- **FR-015**: `GET /v1/models` MUST list served models with their context capacity on this machine
  (from the memory budget) and the server's default output limit, not just the model's advertised
  maximum.
- **FR-016**: Reasoning produced in one turn MUST be returnable in the next turn so the model sees
  the same rendered history: as thinking blocks with an opaque signature (Anthropic) or as
  `encrypted_content` on reasoning items (Responses). mrun MUST accept any signature and never
  reject a request because of one.
- **FR-017**: Compatibility probes used by supported clients (for example `HEAD /api/hello`,
  `?beta=true` query strings, `anthropic-beta` headers) MUST be accepted.
- **FR-018**: A request naming a model that is not loaded MUST be served by the loaded model when
  only one model is served (clients send names such as `claude-*` for background and subagent
  work); responses report the actual model id.
- **FR-019**: When a request gives no output limit, mrun MUST apply a server default: 16,384 tokens
  (configurable) for Chat Completions; `capacity − prompt` for Responses, because Codex never sends
  `max_output_tokens` and, from 0.163, ends the turn on `response.incomplete` (a long `apply_patch`
  would otherwise fail). Anthropic requires `max_tokens`. When the requested limit exceeds what
  fits, mrun MUST clamp it to the available
  capacity (001 FR-009) and report the applied limit (`x-mrun-max-tokens` header or `: mrun` SSE
  comment) with a `max-tokens-clamped` warning. Rejection happens only when the prompt itself does
  not fit.
- **FR-020**: Images, audio and files in user messages MUST be rejected naming the JSON path. Images
  inside tool results MUST, by default, be replaced with the text `[image omitted: model has no
  vision input]` and an `image-omitted` warning reported per FR-022, so sessions that read an image
  file stay usable; `--reject-tool-images` restores strict rejection.
- **FR-021**: `mrun serve --record <dir>` (input to feature 005) MAY write, per request: the request
  body, the arrival time and the allow-listed session headers (`x-claude-code-session-id`,
  `thread-id`, `x-session-id`, and `x-claude-code-prev-tool-durations`, which Claude Code sends
  with `CLAUDE_CODE_GATEWAY_HINT_HEADERS=1` and which gives the tool gap). Reply lengths are
  derived by 005 from the next request's history, so nothing about the response is written.
  Recording is a maintainer flag: enabled only on the command line (never via environment), with
  files mode 0600, a startup warning, and a visible flag in `/mrun/status`.
- **FR-022**: Every response MUST report mrun metadata: cached and processed tokens; from feature
  003 the miss cause (`new`, `changed`, `evicted`, `model`, `disabled`), the divergence position
  with its message index and role, and the checkpoint gap; the effective output limit; and the
  warnings `tool-args-invalid`, `max-tokens-clamped` and `image-omitted`. Every other condition
  (`tool-unknown`, `strict-unsupported`, `history-args-raw`, `history-calls-split`,
  `thinking-unsupported`) is written to the server log only, without prompt content (FR-014).
  Non-streaming responses carry it in `x-mrun-*` headers; streaming responses
  carry it in SSE comment lines (`: mrun {json}`) emitted before the first content event and before
  the final event, because headers are sent before the result is known. Bodies stay strictly
  standard. The last request's metadata is also in `/mrun/status`.
- **FR-023**: *Removed 2026-10-15* (Anthropic diagnostics: no supported client sends
  `diagnostics.previous_message_id`). A `diagnostics` request field is accepted and ignored.
- **FR-031**: A request header `x-mrun-ignore-eos: on` MUST make generation continue to the output
  limit (benchmark fairness, 005 FR-007; carried by the request's decode hook); it never changes
  response formats.
- **FR-024**: Anthropic `context_management` (server-side context edits and compaction) MUST be
  accepted and ignored: mrun never rewrites or summarizes client context, and Claude Code sends the
  field by default. Claude Code compacts on its own based on `CLAUDE_CODE_MAX_CONTEXT_TOKENS`,
  which `mrun launch` sets to mrun's capacity (004); it does not read `/v1/models`.
- **FR-025**: Prompts MUST be rendered exactly as sent, with one named exception: the per-request
  nonce in Claude Code's attribution block (a leading system text starting with
  `x-anthropic-billing-header:`, field `cch=`) is replaced by a fixed value of the same length,
  because it changes on every request and would otherwise defeat all prefix reuse for users who
  start Claude Code without `mrun launch` (same approach as llama.cpp; D021). Nothing else in that
  block or elsewhere is altered.
- **FR-026**: *Removed (0.1 scope cut, 2026-10-15)*: progress in Chat keepalive comments.
  `/mrun/status` still shows the active request's phase and prefill progress (FR-001).
- **FR-027**: Responses `namespace` tools (Codex sends MCP and sub-agent tools this way on every
  request) MUST be flattened to functions named `<namespace>__<tool>` and calls MUST be returned
  with `name` and `namespace` restored; history items carrying `namespace` are re-joined the same
  way.
- **FR-028**: Tools with `defer_loading: true` MUST NOT be rendered in the tools region. A
  `tool_reference` block in a tool result (Anthropic) or a `tool_search_output` tool list
  (Responses) MUST be expanded at that position into the tool's rendered definition, so the prefix
  stays append-only; expanded tools are callable afterwards. A reference to an undeclared tool, or a
  request whose tools are all deferred, is 400 with Anthropic's wording.
- **FR-029**: Mid-conversation system messages (Anthropic `role: "system"`, Responses
  `developer`/`system` items) MUST be rendered in place, never hoisted into the first system block.
  Templates that reject non-leading system messages (Qwen3.5/3.6 raise "System message must be at
  the beginning") receive them as a user turn at the same position, so the prompt stays
  append-only. Server-executed features mrun cannot run
  (`mcp_servers`, `container`, `mcp_toolset`, `tool_search_tool_*`, `tool_addition`/`tool_removal`
  blocks) MUST be rejected with 400 naming the field, so the model is never told about tools that
  do not exist.
- **FR-030**: Client cache hints MUST NOT change behaviour: `cache_control` (anywhere) is accepted
  and ignored, and `prompt_cache_key` and client session headers are ignored by the cache (003).
  Session headers are used only for recording (FR-021).

- **FR-032**: Tool-call and reasoning parsing MUST go through a syntax registry
  (`chat/syntax/`): one streaming parser per syntax. Each family names one syntax (`SyntaxId`,
  defined in `mrun-engine`, 001 FR-023); load checks the template renders that syntax's markers,
  else refuses, naming the family, its syntax and the markers that were missing. 0.1 syntaxes:
  Hermes JSON (Qwen3), Qwen XML (Qwen3-Coder, Qwen3.6), Harmony (gpt-oss). No syntax-specific logic
  outside its parser and its registry entry (constitution V). Parsers run on the engine thread
  through the request's decode hook (FR-036).
- **FR-033**: Harmony MUST map `analysis` to reasoning, `final` and recipient-less `commentary` to
  text, and messages addressed `to=functions.NAME` (in the role header or after the channel, with
  any `<|constrain|>` suffix stripped) to tool calls whose arguments are the message body up to
  `<|call|>`; `<|call|>` and `<|return|>` end generation (the engine reports which EOS token
  ended it). Other recipients are not reported as calls (logged, `tool-unknown`). History arguments
  are passed to the template as objects.
- **FR-034**: *Moved to Later (0.1 scope cut, 2026-10-15; with Devstral in 0.2)*: Mistral syntax.
- **FR-035**: Thinking control MUST be per family: Qwen3 and Qwen3.6 via `enable_thinking`;
  gpt-oss via reasoning effort (`low|medium|high` in the template's `Reasoning:` line), mapped from
  OpenAI `reasoning_effort`/`reasoning.effort` and Anthropic `thinking` (budget bands, adaptive
  effort) as in research §3b, with "off" = `low`, its reasoning always returned in the API's
  reasoning field, and Anthropic `budget_tokens` enforced by closing the analysis message;
  Qwen3-Coder has none and logs `thinking-unsupported` when asked.
- **FR-036**: Per-request decode logic MUST run on the engine thread through a decode hook on the
  generation request (001): `on_token(id) -> Option<TokenMask>` feeds the syntax parser and the
  llguidance matcher and returns the next mask, `forced_tokens()` returns tokens the grammar forces
  (appended without sampling, FR-004), and `ignore_eos` implements FR-031. Token events carry every
  sampled id with its text (held-back text as a separate field) and the EOS token in the stop
  reason, so the parser sees every token.

### Key Entities

- **Chat Request**: The internal, API-neutral form: messages (system, user, assistant, tool),
  tools, tool choice, response format, sampling settings, limits, stream flag, reasoning control.
- **Tool Call**: Call id, tool name, arguments, completion status.
- **Syntax**: A registered streaming parser for one tool-call/reasoning format, with its opener
  tokens, argument kind (raw JSON or synthesized), history rendering rule and thinking control.
- **Usage**: Input tokens, cached input tokens, output tokens, reasoning tokens.
- **Queue Entry**: Request, arrival time, deadline, cancellation signal (fired by a connection
  close watcher).
- **Record** (only with `--record`): request body, arrival time, allow-listed session headers.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001** (the client gate, referenced by 004, 008 and 010): Each 0.1 alias (`qwen3.6`,
  `qwen3-coder`, `gpt-oss`) × each client (Claude Code, Codex, OpenCode at pinned versions),
  started via `mrun launch`, completes the scripted task 3 of 3 times with no client-side protocol
  errors; one of the Claude Code runs on `qwen3-coder` crosses the capacity and compacts.
- **SC-002**: The API conformance suite passes 100% and covers, per API: streaming event order and
  terminators; tool-call deltas, parallel calls and tool results; strict tools and JSON formats;
  usage and cached tokens; finish reasons incl. length inside a tool call; errors before and after
  headers; validation limits; keepalives in every phase; disconnect in every phase; and the
  cross-cutting cases of FR-025, FR-027–FR-031 (named list: research, Conformance test list).
- **SC-003**: With strict tools whose schemas compile, 100% of completed tool calls in the
  conformance and client suites validate against their schemas.
- **SC-004**: Serving overhead is small: time to first token through the API is within 50 ms of
  the engine's own time to first token, and generation speed through the API is at least 97% of the
  engine's (measured by this feature's overhead test).
- **SC-005**: After a client disconnect in any phase (queued, prefilling, generating,
  non-streaming), the request's work stops and the next queued request starts within 300 ms (p95).
- **SC-006**: The server binds loopback only, rejects non-loopback `Host` and foreign `Origin`
  headers, and writes no prompt content to disk or logs; `--record` cannot be enabled by the
  environment (verified by an audit test).
- **SC-007**: The syntax conformance cases pass 100%, covering for every 0.1 syntax: markers split
  at every token boundary, parallel calls (Hermes, Qwen XML), values containing the syntax's own
  closing markers, `"null"`/`"true"` kept as strings, truncated calls reported incomplete,
  invalid-argument fallback per API, undeclared tools; and for Harmony: both recipient positions,
  constrain-suffix stripping, built-in recipients, multi-call history split, effort mapping and
  budget closing the analysis message (named list: research §3b). The client part is SC-001.

## Assumptions

- Builds on the engine from feature 001; cross-request prefix reuse (and therefore large cached
  input numbers) arrives with feature 003. This feature reports cached tokens truthfully, which may
  be zero.
- `mrun launch` (feature 004) configures clients, including `CLAUDE_CODE_MAX_CONTEXT_TOKENS`,
  `CLAUDE_CODE_MAX_OUTPUT_TOKENS` and `ENABLE_TOOL_SEARCH=auto:10` for Claude Code; SC-001 runs
  through it. Until 004 lands, client configs are written by hand (quickstart).
- Claude Code's attribution block is stable per conversation for custom base URLs since v2.1.181;
  older versions change the `cch=` nonce every turn, which FR-025 neutralizes server-side, so no
  client setting is needed.
- Vision input is out of scope (see FR-020 for how images are handled).
- Serving overhead (SC-004) is measured by an engine-vs-API timing test in this feature; only
  SC-001 depends on 004 (`mrun launch`).
- Ollama's native `/api` endpoints and stateful Responses storage are non-goals.

## Later

Moved out of 0.1 on 2026-10-15 (D025); IDs kept:

- FR-012 second part: non-loopback binds with an API key (`MRUN_API_KEY`, `--api-key-file`,
  constant-time compare, `Authorization: Bearer` or `x-api-key`).
- FR-026: progress (`queued` position, `prefill` done/total) in Chat keepalive comments.
- FR-034 and the Mistral parser (with Devstral, 0.2): `[TOOL_CALLS]NAME[ARGS]{json}` on token ids,
  parallel calls, literal marker text in arguments round-trips; `[TOOL_CALLS]` bias for `none`.
- `--thinking client|off|on` server override (FR-013, FR-035).
- A nightly matrix against the latest client releases (replaced by pinned smoke on every PR plus a
  manual pre-release check).
