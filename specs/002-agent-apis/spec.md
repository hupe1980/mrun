# Feature Specification: Agent APIs

**Feature Branch**: `002-agent-apis`

**Created**: 2026-10-14

**Status**: Draft

**Input**: User description: "`mrun serve` exposes OpenAI Chat Completions, Anthropic Messages (with count_tokens) and stateless OpenAI Responses on loopback, with streaming, tool calls and structured output, so Claude Code, Codex and OpenCode complete real tool loops against a local model. One internal request model; truthful usage and finish reasons; bounded queue; cancellation on disconnect."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Use a local model from my coding agent (Priority: P1)

A developer starts `mrun serve`, points their agent (OpenCode, Claude Code or Codex) at it, and
completes a real task: the agent reads files, edits code and runs tests through tool calls, all on
the local model.

**Why this priority**: This is the product. Without working agent loops the engine is a demo.

**Independent Test**: With `mrun serve` running and each client configured per `mrun connect`
output, run a scripted read-edit-test task in a fixture repository and confirm it completes with
the tests passing, for each of the three clients.

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
   within one chunk or token and the next queued request starts.
2. **Given** a full queue, **When** another request arrives, **Then** it is rejected immediately
   with a retry-after hint.
3. **Given** a queued request waiting past its deadline, **When** the deadline passes, **Then** it
   is rejected with a timeout error and never starts.

### Edge Cases

- Tool definitions with deeply nested or very large schemas: rejected above documented limits.
- Parallel tool calls requested: the model may emit several; each is reported with a stable id.
- A tool result referencing an unknown call id: rejected with the id named.
- Streaming split inside a multi-byte character or inside tool-call JSON: output stays valid.
- Assistant prefill (Anthropic trailing assistant message): continued, not restarted.
- Thinking/reasoning content from the model: returned in the API's reasoning field or omitted per
  request, never mixed into the answer text.
- `max_tokens` above the model limit: clamped to capacity with a clear error if the prompt does
  not leave room.
- Non-loopback bind requested: refused unless an explicit flag and an API key are configured.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: `mrun serve` MUST expose OpenAI Chat Completions (`POST /v1/chat/completions`),
  Anthropic Messages (`POST /v1/messages`, `POST /v1/messages/count_tokens`) and stateless OpenAI
  Responses (`POST /v1/responses`), plus model listing (`GET /v1/models`) and health
  (`GET /health`).
- **FR-002**: All three APIs MUST support streaming (server-sent events in each API's native event
  format) and non-streaming responses.
- **FR-003**: All three APIs MUST support tool definitions, tool calls (including parallel calls)
  and tool results, translated through one internal request model so behavior is identical.
- **FR-004**: Tool-call arguments MUST be valid JSON matching the declared schema when the client
  requests strict tools (`strict: true` or equivalent), using constrained decoding; non-strict tool
  calls MUST at least be valid JSON or be reported as a parse error, never silently repaired.
- **FR-005**: Structured output (JSON schema response format) MUST be supported with the same
  constrained decoding.
- **FR-006**: The Responses API MUST be stateless: requests that depend on stored server state
  (`previous_response_id`, `store: true` retrieval) MUST be rejected with a clear error.
- **FR-007**: Usage MUST report input, cached input and output tokens accurately in each API's
  fields; finish/stop reasons MUST map exactly (stop, length, tool calls, cancelled, error).
- **FR-008**: Errors MUST use each API's native error shape and status codes, with a message that
  names the cause and, where possible, the fix.
- **FR-009**: Requests MUST be validated before queuing: body size (default 8 MiB), message count,
  tool count and schema size/depth limits, all reported in errors when exceeded.
- **FR-010**: The server MUST run one generation at a time with a FIFO queue (default 8 waiting,
  30 s queue deadline); full-queue and deadline errors MUST be immediate and explicit.
- **FR-011**: Client disconnect MUST cancel queued or running work promptly (within one chunk or
  token) and release its reservation.
- **FR-012**: The server MUST bind to loopback by default; binding elsewhere MUST require an explicit
  flag and an API key, and requests MUST then authenticate.
- **FR-013**: Reasoning/thinking output MUST be separated from answer text according to each API's
  conventions, and thinking MUST be controllable per request where the model supports it.
- **FR-014**: The server MUST NOT log prompt or completion content by default; request logs contain
  timings, token counts and outcome only.
- **FR-015**: `GET /v1/models` MUST list available models with their context capacity on this
  machine (from the memory budget), not just the model's advertised maximum.

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
  engine's.
- **SC-005**: After a client disconnect, the next queued request starts within 300 ms (p95).
- **SC-006**: Default configuration exposes nothing beyond loopback and writes no prompt content to
  disk or logs (verified by an audit test).

## Assumptions

- Builds on the engine from feature 001; cross-request prefix reuse (and therefore large cached
  input numbers) arrives with feature 003. This feature reports cached tokens truthfully, which may
  be zero.
- `mrun connect <client>` (feature 004) prints client configuration; for testing here, client
  configs are written by hand.
- Claude Code adds a per-request attribution header to its system prompt that changes every turn;
  feature 004's `connect` sets `CLAUDE_CODE_ATTRIBUTION_HEADER=0`. This feature does not rewrite
  prompts.
- Image, audio and file inputs are out of scope and rejected.
- Ollama's native `/api` endpoints and stateful Responses storage are non-goals.
