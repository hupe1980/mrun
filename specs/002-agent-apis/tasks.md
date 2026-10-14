---
description: "Tasks for 002 Agent APIs"
---

# Tasks: Agent APIs

**Input**: [plan.md](plan.md), [spec.md](spec.md), [research.md](research.md), [data-model.md](data-model.md), [contracts/api.md](contracts/api.md)

**Tests**: Conformance cases from [research](research.md#design-recommendations) run on `ScriptedEngine`.

## Phase 1: Setup

- [ ] T001 Add axum, tower-http, llguidance 1.9.1, toktrie 1.9.1, base64, ulid, subtle to `crates/mrun/Cargo.toml`; `crates/mrun/src/api/mod.rs` router skeleton with `/health`, `HEAD /api/hello`, Host/Origin guard middleware (FR-012) and tests
- [ ] T002 [P] Test harness `crates/mrun/tests/api_conformance/common.rs`: start server on an ephemeral port with `ScriptedEngine` scripts, SSE client helper, insta snapshot helpers

## Phase 2: Foundational

- [ ] T003 `crates/mrun/src/api/model.rs`: ChatRequest, Message, AssistantPart, ToolDef, ToolChoice, ResponseFormat, ReasoningCtl, GenEvent, Finish, Usage, ApiError (per data model)
- [ ] T004 `crates/mrun/src/chat/render.rs`: ChatRequest → template context (tools, tool results, reasoning replay, prefill) → tokens + message boundaries; tests vs fixture token ids
- [ ] T005 `crates/mrun/src/runtime/{queue.rs, worker.rs}`: FIFO (8 waiting, 600 s wait deadline), keepalives while queued, cancel on disconnect, retry-after, single worker driving `Engine`, GenEvent fan-out; tests for order, full queue, deadline, disconnect, a long generation with a queued streaming request (no client timeout)
- [ ] T006 [P] `crates/mrun/src/chat/tools/{mod.rs, holdback.rs, hermes.rs, qwen_xml.rs}`: `ToolSyntax` streaming parsers for Qwen3 and Qwen3-Coder, partial-tag hold-back, raw args preserved, per-call buffering for XML-synthesized arguments and for Anthropic (FR-004 fallback); unit tests incl. split tags, escaped strings, parallel calls, missing `<tool_call>` recovery for declared names
- [ ] T007 [P] `crates/mrun/src/chat/reasoning.rs`: think-tag splitter (prompt-tail aware), signature and `encrypted_content` codecs; tests
- [ ] T008 `crates/mrun/src/api/sse.rs`: SSE writer, keepalive/ping hook, disconnect detection → cancel token

## Phase 3: US1 — Use a local model from my agent (P1) 🎯 MVP

- [ ] T009 [P] [US1] `api/chat_completions.rs` decode + encode (stream/non-stream, tool call deltas, include_usage chunk, reasoning_content, no content after finish) + conformance `chat_*` cases
- [ ] T010 [P] [US1] `api/messages.rs` decode + encode (full event order, ping, thinking + signature, tool_use input_json_delta, `?beta=true`, x-api-key/bearer) + `count_tokens` + conformance `msg_*` cases
- [ ] T011 [P] [US1] `api/responses.rs` decode + encode (developer message, function_call items, `output_item.done`, reasoning items with `summary: []` + encrypted_content, sequence numbers, `response.failed` codes; `previous_response_id` rejected) + conformance `resp_*` cases
- [ ] T012 [US1] `GET /v1/models` with machine-specific `context_window` and `max_output_tokens`; `GET /mrun/status` (schema v1 per contract) with a conformance case; unknown model ids served by the loaded model (FR-018); `cli/serve.rs` banner with URL and `mrun launch`/`connect` hint
- [ ] T013 [US1] `chat/constrain.rs`: llguidance ParserFactory per model (cached), Matcher armed at tool-call trigger for strict tools and for JSON response formats; the generation loop passes the `TokenMask` to `Engine::decode` (unpipelined while masked); tests: strict args always validate (SC-003)
- [ ] T014 [US1] Thinking control: request mapping (effort none/minimal = off) + `--thinking client|off|on` + Anthropic `budget_tokens` enforcement; tests that Codex's bare `reasoning` object does not enable thinking and Claude Code `adaptive` does unless overridden
- [ ] T015 [US1] Output limit default (16,384) and clamping (FR-019); image-in-tool-result placeholder and `--reject-tool-images` (FR-020); non-strict invalid args policy per API (FR-004); mrun metadata in headers (non-streaming) and `: mrun` SSE comments (streaming) (FR-022); conformance cases for each
- [ ] T016 [US1] Real-client smoke runs on macOS (Claude Code 2.1.300, Codex `ae11049`, OpenCode `13afea6`) per [quickstart](quickstart.md); record versions and results in `specs/002-agent-apis/research.md`

## Phase 4: US2 — Clear, honest responses (P1)

- [ ] T017 [US2] Usage semantics per API (Anthropic input excludes cache read; others inclusive; `cache_creation_input_tokens` = 0) and the cache values in mrun metadata (delivered by T015) + `cross_*` conformance cases
- [ ] T018 [US2] Error mapping per API per the contract status table (queue full, queue deadline, context too long with numbers, unsupported modalities with JSON path, unknown tool result id, limits, auth, Host/Origin) + cases
- [ ] T019 [US2] Request validation limits (body 8 MiB via tower-http, messages, tools, schema size/depth) + cases

## Phase 5: US3 — Reliable under agent behavior (P2)

- [ ] T020 [US3] Disconnect mid-queue and mid-stream tests; stalled reader → cancel after 10 s; next request latency (SC-005 measured on macOS)
- [ ] T021 [US3] Non-loopback bind requires an API key from `MRUN_API_KEY`/`--api-key-file` (constant-time compare); auth tests; `serve --record <dir>` (FR-021: CLI-only, loopback-only, 0600 files, startup warning, shown in status); audit test that no prompt text reaches logs or disk without `--record` and that config/env cannot enable it (SC-006)

## Phase 6: Polish

- [ ] T022 [P] Serving overhead test (SC-004): same prompt through `EngineHandle` directly and through each API, compare TTFT and decode rate (`crates/mrun/tests/overhead.rs`, macOS real model, `#[ignore]`)
- [ ] T023 [P] README "Use with Claude Code / Codex / OpenCode" section

## Dependencies

T001 → T003 → T004/T005 → T009–T011 (parallel) → T012–T016. T006–T008 parallel after T003.
Requires 001's `Engine` trait (001 T008), chat layer (001 T010) and generation loop (001 T011); real-model runs need 001 MVP.
