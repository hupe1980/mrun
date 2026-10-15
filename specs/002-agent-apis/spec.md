# Feature Specification: Agent APIs

**Feature Branch**: `002-agent-apis`

**Created**: 2026-10-14

**Status**: Draft

**Input**: User description: "`mrun serve` exposes OpenAI Chat Completions, Anthropic Messages (with count_tokens) and stateless OpenAI Responses on loopback, with streaming, tool calls and structured output, so Claude Code, Codex and OpenCode complete real tool loops against a local model. One internal request model; truthful usage and finish reasons; bounded queue; cancellation on disconnect."

## Clarifications

### Session 2026-10-14

- Q: Do models think when the client doesn't ask? → A: No. Thinking is enabled only by the request (Anthropic `thinking`, OpenAI `reasoning_effort` / `reasoning`). (Maintainer decision.)
- Q: How are concurrent requests handled? → A: One generation at a time, FIFO queue (8 waiting, 600 s waiting deadline, revised after analysis); batching is a later feature. (Maintainer decision.)
- Q: Default port? → A: 127.0.0.1:8484 (avoids Ollama 11434, LM Studio 1234, llama.cpp 8080); `--port` and `MRUN_SERVER_PORT` override.
- Q: Are non-strict tool calls constrained? → A: Tool-call syntax is always parsed with the model's native format; argument schemas are enforced by constrained decoding only for strict tools and JSON response formats, keeping the default fast path unconstrained.
- Q: What does `count_tokens` count? → A: Exactly the tokens mrun would process for that request with the loaded model's tokenizer and template.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Use a local model from my coding agent (Priority: P1)

A developer starts `mrun serve`, points their agent (OpenCode, Claude Code or Codex) at it, and
completes a real task: the agent reads files, edits code and runs tests through tool calls, all on
the local model.

**Why this priority**: This is the product. Without working agent loops the engine is a demo.

**Independent Test**: With `mrun serve` running and each client configured by hand-written config
(as documented in the quickstart), run a scripted read-edit-test task in a fixture repository and
confirm it completes with the tests passing, for each of the three clients.

**Acceptance Scenarios**:

1. **Given** `mrun serve` and OpenCode configured for an OpenAI-compatible provider, **When** the
   developer asks for a code change, **Then** the agent's tool calls arrive well-formed, tool
   results are accepted, and the session completes.
2. **Given** Claude Code pointed at mrun, **When** the developer runs a task, **Then** streaming,
   tool use, tool results and token counting all work without client-side errors.
3. **Given** Codex pointed at mrun, **When** the developer runs a task, **Then** the Responses
   stream, function calls and function-call outputs work without client-side errors.

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
2. **Given** a request whose prompt plus output does not fit, **When** submitted, **Then** the API
   returns the API-appropriate "context too long" error with the token numbers.
3. **Given** an unsupported parameter that changes meaning (for example image input), **When**
   submitted, **Then** the request is rejected naming the parameter; unknown harmless fields are
   ignored.
4. **Given** generation stops at the output limit in the middle of a tool call, **When** the stream
   ends, **Then** the finish reason says length and no incomplete tool call is reported as complete.

---

### User Story 3 - Reliable under agent behavior (Priority: P2)

Agents disconnect, retry and fire overlapping requests (subagents). mrun queues fairly, cancels
abandoned work immediately, and never wedges.

**Why this priority**: Real agent traffic includes concurrency and disconnects; a stuck server ends
the session.

**Independent Test**: Fire overlapping requests, disconnect mid-stream and mid-queue, and stall a
reader; verify queue order, prompt cancellation, timeouts and that the next request runs normally.

**Acceptance Scenarios**:

1. **Given** a request in progress, **When** the client disconnects, **Then** generation stops
   within 300 ms or one token and the next queued request starts.
2. **Given** a full queue, **When** another request arrives, **Then** it is rejected immediately
   with a retry-after hint.
3. **Given** a queued request waiting past its deadline (default 600 s), **When** the deadline
   passes, **Then** it is rejected with a timeout error and never starts; while it waits, streaming
   clients receive keepalives.

### Edge Cases

- Tool definitions with deeply nested or very large schemas: rejected above documented limits.
- Parallel tool calls requested: the model may emit several; each is reported with a stable id.
- A tool result referencing an unknown call id: rejected with the id named.
- Streaming split inside a multi-byte character or inside tool-call JSON: output stays valid.
- Assistant prefill (Anthropic trailing assistant message): continued, not restarted.
- Thinking/reasoning content from the model: returned in the API's reasoning field or omitted per
  request, never mixed into the answer text.
- `max_tokens` above capacity or absent: clamped / defaulted per FR-019; rejected only when the
  prompt itself does not fit.
- Images inside tool results (Claude Code reading a PNG, Codex `view_image`): replaced per FR-020.
- A browser page on the same machine calling the server: rejected by the Host/Origin policy.
- Non-loopback bind requested: refused unless an explicit flag and an API key are configured.

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
- **FR-004**: Tool-call arguments MUST be valid JSON matching the declared schema when the client
  requests strict tools (`strict: true` or equivalent), using constrained decoding. Non-strict
  arguments are never modified. If they are not valid JSON: Chat Completions and Responses return
  the call with its raw arguments string, and such raw strings are accepted unvalidated in later
  history; Anthropic Messages (whose `input` must be an object) returns the model's raw call text
  as a text block instead of a tool call. Either way a `tool-args-invalid` warning is reported per
  FR-022. When streaming, argument text is forwarded as it arrives only where it is the model's raw
  JSON (Qwen3 Hermes syntax on Chat/Responses); arguments synthesized from other syntaxes (Qwen3-Coder
  XML) and all Anthropic `tool_use` input are buffered per call and emitted when the call closes,
  so the fallback above is always possible. The tool-call envelope (for example a missing `<tool_call>` opener before a declared
  function) MAY be recognized, because it does not change argument content.
- **FR-005**: Structured output (JSON schema response format) MUST be supported with the same
  constrained decoding.
- **FR-006**: The Responses API MUST be stateless: `previous_response_id` and item references MUST
  be rejected with a clear error; `store: true` is accepted and nothing is stored.
- **FR-007**: Usage MUST report input, cached input and output tokens with each API's semantics
  (Anthropic `input_tokens` excludes cache reads; OpenAI totals include them;
  `cache_creation_input_tokens` is always 0). Finish reasons MUST follow the mapping table in
  [research](research.md#design-recommendations) (stop, stop sequence, length, context full, tool
  calls); errors are error events, never finish reasons.
- **FR-008**: Errors MUST use each API's native error shape, with a message that names the cause
  and, where possible, the fix. Queue-full and queue-timeout statuses follow
  [contracts/api.md](contracts/api.md#status-codes).
- **FR-009**: Requests MUST be validated before queuing against configurable limits (defaults: body
  8 MiB, 4,096 messages, 256 tools, schema 64 KiB and depth 16); errors name the limit and the
  setting that raises it.
- **FR-010**: The server MUST run one generation at a time with a FIFO queue (default 8 waiting).
  The queue deadline covers waiting time only (default 600 s, configurable); while a request
  waits, streaming responses MUST send keepalives (Anthropic `ping`, SSE comments). Full-queue and
  deadline errors MUST be immediate and explicit.
- **FR-011**: Client disconnect MUST cancel queued or running work promptly (within 300 ms or one
  token) and release its reservation.
- **FR-012**: The server MUST bind to loopback by default and accept only loopback `Host` headers
  (`127.0.0.1`, `localhost`, `[::1]` with the server port), reject requests carrying a browser
  `Origin` other than the server's own, and send no CORS headers. Binding elsewhere MUST require an
  explicit flag and an API key read from `MRUN_API_KEY` or `--api-key-file` (never a command-line
  value), compared in constant time.
- **FR-013**: Reasoning/thinking output MUST be separated from answer text per each API's
  conventions. Thinking is on only when the request asks for it (Anthropic `thinking` enabled or
  adaptive; OpenAI `reasoning_effort` / Responses `reasoning.effort` not in {none, minimal}; a bare
  `reasoning` object does not count). A server setting `--thinking client|off|on` (default
  `client`) overrides clients. Anthropic `budget_tokens` is honored by closing the thinking block
  when the budget is reached.
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
- **FR-019**: When a request gives no output limit, mrun MUST apply a server default (16,384 tokens,
  configurable); when the requested limit exceeds what fits, mrun MUST clamp it to the available
  capacity and report the clamp per FR-022. Rejection happens only when the prompt itself does not
  fit.
- **FR-020**: Images, audio and files in user messages MUST be rejected naming the JSON path. Images
  inside tool results MUST, by default, be replaced with the text `[image omitted: model has no
  vision input]` and an `image-omitted` warning reported per FR-022, so sessions that read an image
  file stay usable; `--reject-tool-images` restores strict rejection.
- **FR-021**: `mrun serve --record <dir>` (debug aid for feature 005) MAY write request bodies, with
  arrival and response-end times, to disk only when given on the command line (never via config
  or environment), only on loopback, with files mode 0600, a startup warning, and a visible flag in `/mrun/status`.
- **FR-022**: Every response MUST report mrun metadata (cached and processed tokens, miss cause and
  divergence from feature 003, effective output limit, warnings such as `tool-args-invalid`,
  `image-omitted`, `max-tokens-clamped`). Non-streaming responses carry it in `x-mrun-*` headers;
  streaming responses carry it in SSE comment lines (`: mrun {json}`) emitted before the first
  content event and before the final event, because headers are sent before the result is known.
  Bodies stay strictly standard. The last request's metadata is also in `/mrun/status`.
- **FR-023**: Anthropic Messages requests carrying `diagnostics.previous_message_id` MUST receive
  the standard `diagnostics.cache_miss_reason` in the response (`model_changed`, `system_changed`,
  `tools_changed`, `messages_changed` with `cache_missed_input_tokens`,
  `previous_message_not_found`, `unavailable`), derived by comparing this prompt's fingerprint with
  that message's (data model: Prompt Fingerprint, DiagnosticsIndex); `diagnostics` is `null` when the
  prefix was fully reused. The index holds the last 256 message ids in memory only.
- **FR-024**: Anthropic `context_management` (server-side context edits and compaction) MUST be
  accepted and ignored: mrun never rewrites or summarizes client context, and Claude Code sends the
  field by default. Clients that depend on the edits keep working because their own compaction
  takes over at the reported context window.

### Key Entities

- **Chat Request**: The internal, API-neutral form: messages (system, user, assistant, tool),
  tools, tool choice, response format, sampling settings, limits, stream flag, reasoning control.
- **Tool Call**: Call id, tool name, arguments, completion status.
- **Usage**: Input tokens, cached input tokens, output tokens, reasoning tokens.
- **Queue Entry**: Request, arrival time, deadline, cancellation signal.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: OpenCode, Claude Code and Codex (pinned versions) each complete the scripted
  read-edit-test task against the primary model with no client-side protocol errors, 5 of 5 runs.
- **SC-002**: The API conformance suite (at least 60 cases across the three APIs: streaming,
  tools, errors, limits, cancellation) passes 100%.
- **SC-003**: With strict tools, 100% of completed tool calls in the conformance and client suites
  validate against their schemas.
- **SC-004**: Serving overhead is small: time to first token through the API is within 50 ms of
  the engine's own time to first token, and generation speed through the API is at least 97% of the
  engine's (measured by this feature's overhead test).
- **SC-005**: After a client disconnect, the next queued request starts within 300 ms (p95).
- **SC-006**: Default configuration exposes nothing beyond loopback, rejects non-loopback `Host`
  and foreign `Origin` headers, and writes no prompt content to disk or logs; `--record` cannot be
  enabled by config or environment (verified by an audit test).

## Assumptions

- Builds on the engine from feature 001; cross-request prefix reuse (and therefore large cached
  input numbers) arrives with feature 003. This feature reports cached tokens truthfully, which may
  be zero.
- `mrun connect <client>` (feature 004) prints client configuration; for testing here, client
  configs are written by hand.
- Claude Code adds an attribution block to its system prompt; since v2.1.181 it is stable per
  conversation for custom base URLs (earlier versions change it every turn). Feature 004's
  `connect`/`launch` set `CLAUDE_CODE_ATTRIBUTION_HEADER=0` as hygiene. This feature does not
  rewrite prompts.
- Vision input is out of scope (see FR-020 for how images are handled).
- Real-client smoke tests (SC-001) use hand-written client configs; serving overhead (SC-004) is
  measured by an engine-vs-API timing test in this feature, so 002's gates do not depend on 004/005.
- Ollama's native `/api` endpoints and stateful Responses storage are non-goals.
