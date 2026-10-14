---
description: "Tasks for 002 Agent APIs"
---

# Tasks: Agent APIs

**Input**: [plan.md](plan.md), [spec.md](spec.md), [research.md](research.md), [data-model.md](data-model.md), [contracts/api.md](contracts/api.md)

**Tests**: Conformance cases from [research](research.md#design-recommendations) run on `ScriptedEngine`.

## Phase 1: Setup

- [ ] T001 Add axum, tower-http, llguidance 1.9.1, toktrie 1.9.1, base64, zstd, ulid to `crates/mrun/Cargo.toml`; `crates/mrun/src/api/mod.rs` router skeleton with `/health`, `HEAD /api/hello`
- [ ] T002 [P] Test harness `crates/mrun/tests/api_conformance/common.rs`: start server on an ephemeral port with `ScriptedEngine` scripts, SSE client helper, insta snapshot helpers

## Phase 2: Foundational

- [ ] T003 `crates/mrun/src/api/model.rs`: ChatRequest, Message, AssistantPart, ToolDef, ToolChoice, ResponseFormat, ReasoningCtl, GenEvent, Finish, Usage, ApiError (per data model)
- [ ] T004 `crates/mrun/src/chat/render.rs`: ChatRequest → template context (tools, tool results, reasoning replay, prefill) → tokens + message boundaries; tests vs fixture token ids
- [ ] T005 `crates/mrun/src/runtime/{queue.rs, worker.rs}`: FIFO (8 / 30 s), cancel on disconnect, retry-after, single worker driving `Engine`, GenEvent fan-out; tests for order, full queue, deadline, disconnect
- [ ] T006 [P] `crates/mrun/src/chat/tools/{mod.rs, holdback.rs, hermes.rs, qwen_xml.rs}`: `ToolSyntax` streaming parsers for Qwen3 and Qwen3-Coder, partial-tag hold-back, raw args preserved, parse-error reporting; unit tests incl. split tags, escaped strings, parallel calls, missing `<tool_call>` recovery for declared names
- [ ] T007 [P] `crates/mrun/src/chat/reasoning.rs`: think-tag splitter (prompt-tail aware), signature and `encrypted_content` codecs; tests
- [ ] T008 `crates/mrun/src/api/sse.rs`: SSE writer, keepalive/ping hook, disconnect detection → cancel token

## Phase 3: US1 — Use a local model from my agent (P1) 🎯 MVP

- [ ] T009 [P] [US1] `api/chat_completions.rs` decode + encode (stream/non-stream, tool call deltas, include_usage chunk, reasoning_content, no content after finish) + conformance `chat_*` cases
- [ ] T010 [P] [US1] `api/messages.rs` decode + encode (full event order, ping, thinking + signature, tool_use input_json_delta, `?beta=true`, x-api-key/bearer) + `count_tokens` + conformance `msg_*` cases
- [ ] T011 [P] [US1] `api/responses.rs` decode + encode (developer message, function_call items, `output_item.done`, reasoning items with `summary: []` + encrypted_content, sequence numbers, `response.failed` codes; `previous_response_id` rejected) + conformance `resp_*` cases
- [ ] T012 [US1] `GET /v1/models` with machine-specific `context_window`; `cli/serve.rs` banner with URL and `mrun connect` hint
- [ ] T013 [US1] `chat/constrain.rs`: llguidance ParserFactory per model (cached), Matcher armed at tool-call trigger for strict tools and for JSON response formats; mask applied via `GenerateRequest.mask_provider`; tests: strict args always validate (SC-003)
- [ ] T014 [US1] Thinking control: request mapping + `--thinking client|off|on`; tests that Codex's bare `reasoning` object does not enable thinking and Claude Code `adaptive` does unless overridden
- [ ] T015 [US1] Real-client smoke runs on macOS (Claude Code 2.1.300, Codex `ae11049`, OpenCode `13afea6`) per [quickstart](quickstart.md); record versions and results in `specs/002-agent-apis/research.md`

## Phase 4: US2 — Clear, honest responses (P1)

- [ ] T016 [US2] Usage semantics per API (Anthropic input excludes cache read; others inclusive) + `cross_*` conformance cases
- [ ] T017 [US2] Error mapping per API (context too long with numbers, unsupported modalities with JSON path, unknown tool result id, limits) + cases
- [ ] T018 [US2] Request validation limits (body 8 MiB via tower-http, messages, tools, schema size/depth) + cases

## Phase 5: US3 — Reliable under agent behavior (P2)

- [ ] T019 [US3] Disconnect mid-queue and mid-stream tests; stalled reader → cancel after 10 s; next request latency (SC-005 measured on macOS)
- [ ] T020 [US3] Non-loopback bind requires `--api-key`; auth tests; audit test that no prompt text reaches logs (SC-006)

## Phase 6: Polish

- [ ] T021 [P] Serving overhead measurement vs engine-only (SC-004) via `mrun bench` cells
- [ ] T022 [P] README "Use with Claude Code / Codex / OpenCode" section

## Dependencies

T001 → T003 → T004/T005 → T009–T011 (parallel) → T012–T015. T006–T008 parallel after T003.
Requires 001's `Engine` trait (001 T010) and chat layer (001 T009); real-model runs need 001 MVP.
