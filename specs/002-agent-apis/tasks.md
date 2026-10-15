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
- [ ] T004 `crates/mrun/src/chat/render.rs`: ChatRequest → template context (tools, tool results, reasoning replay, prefill) → tokens + `RenderedPrompt` marks for 003 (per message: start offset, role, content hash over role, visible text and canonical tool calls with reasoning excluded; `system_tools_end`; `generation_start`); per-template history arguments (Qwen3 verbatim; XML templates parse to object, non-object raw text rendered in place with `history-args-raw`); tests vs fixture token ids, `xapi_qwen3_raw_args_rerender_byte_stable`, `xapi_qwen_xml_rerender_template_miss_reported`, `xapi_qwen_xml_history_args_nonobject_raw_warning`
- [ ] T005 `crates/mrun/src/runtime/{queue.rs, worker.rs}`: FIFO (16 waiting, 600 s wait deadline), keepalive ticks every 10 s while queued and during prefill (headers sent at once), retry-after clamped to 1–60 s, cancel token checked while queued and per prefill chunk/layer, single worker driving `Engine`, GenEvent fan-out; tests for order, full queue, deadline, cancel in each phase, a long generation with a queued streaming request (no client timeout)
- [ ] T006 [P] `crates/mrun/src/chat/tools/{mod.rs, holdback.rs, hermes.rs, qwen_xml.rs}`: `ToolSyntax` streaming parsers for Qwen3 (Hermes) and Qwen3-Coder/Qwen3.6 (XML), partial-tag hold-back, raw args preserved, per-call buffering for XML-synthesized arguments and for Anthropic (FR-004 fallback); unit tests incl. split tags, escaped strings, parallel calls, missing `<tool_call>` recovery for declared names, argument text containing `</parameter>`/`</tool_call>` round-tripping unchanged, `null` kept as a string for string parameters, non-exact booleans invalid, undeclared tool → `tool-unknown`
- [ ] T007 [P] `crates/mrun/src/chat/reasoning.rs`: think-tag splitter (prompt-tail aware), signature and `encrypted_content` codecs; tests
- [ ] T008 `crates/mrun/src/api/{sse.rs, conn.rs}`: SSE writer with per-API keepalive (Responses `response.created` at once then `response.in_progress`, Anthropic `ping`, Chat comment with progress per FR-026) and in-stream errors after headers (Anthropic `event: error` `overloaded_error`, Chat `data: {"error":…}`, Responses `response.failed`); per-connection close watcher that fires the cancel token in any phase, including non-streaming and queued requests (FR-011), not only on failed writes

## Phase 3: US1 — Use a local model from my agent (P1) 🎯 MVP

- [ ] T009 [P] [US1] `api/chat_completions.rs` decode + encode (stream/non-stream, tool call deltas, include_usage chunk, `prompt_tokens_details.cached_tokens` always present, reasoning_content, no content after finish) + conformance `chat_*` cases
- [ ] T010 [P] [US1] `api/messages.rs` decode + encode (full event order, ping, thinking + signature, tool_use input_json_delta, `?beta=true`, x-api-key/bearer) + `count_tokens` + conformance `msg_*` cases
- [ ] T011 [P] [US1] `api/responses.rs` decode + encode (developer message, function_call items, `output_item.done`, reasoning items with `summary: []` + encrypted_content, sequence numbers, `response.failed` codes; `previous_response_id` rejected) + conformance `resp_*` cases
- [ ] T012 [US1] `GET /v1/models` with machine-specific `context_window`, `max_model_len` and `max_output_tokens`; `GET /mrun/status` (schema v1 per contract) with a conformance case; unknown model ids served by the loaded model (FR-018); `cli/serve.rs` banner with URL and `mrun launch`/`connect` hint
- [ ] T013 [US1] `chat/constrain.rs`: llguidance ParserFactory per model (cached), Matcher armed at tool-call trigger for strict tools and for JSON response formats (grammar covers the answer after reasoning when thinking is on); a schema that fails to compile → unconstrained with `strict-unsupported` warning; `tool_choice: none` → fixed −inf bias on tool-call opener tokens (no matcher, stays pipelined); cancellation handle wired to the request cancel token; the generation loop passes the `TokenMask` to `Engine::decode` (unpipelined while masked); tests: strict args always validate (SC-003), `xapi_strict_schema_uncompilable_unconstrained_warning`, `xapi_tool_choice_none_bias_pipelined`
- [ ] T014 [US1] Thinking control: request mapping (effort none/minimal = off) + `--thinking client|off|on` + Anthropic `budget_tokens` enforcement; tests that Codex's bare `reasoning` object does not enable thinking and Claude Code `adaptive` does unless overridden
- [ ] T015 [US1] Output limit default (16,384) and clamping to remaining capacity (FR-019, 001 FR-009) reported via `x-mrun-max-tokens` / `: mrun` comment; image-in-tool-result placeholder and `--reject-tool-images` (FR-020); non-strict invalid args policy per API (FR-004); XML-template history args that are not a JSON object rendered raw with `history-args-raw` (FR-004); mrun metadata in headers (non-streaming) and `: mrun` SSE comments (streaming) (FR-022); conformance cases for each
- [ ] T016 [US1] Real-client smoke runs on macOS (Claude Code 2.1.300, Codex `ae11049`, OpenCode `13afea6`) per [quickstart](quickstart.md), including the Claude Code run that crosses capacity (with `CLAUDE_CODE_MAX_CONTEXT_TOKENS`/`CLAUDE_CODE_MAX_OUTPUT_TOKENS` set) and a Codex run queued > 300 s; record versions and results in `specs/002-agent-apis/research.md`

## Phase 4: US2 — Clear, honest responses (P1)

- [ ] T017 [US2] Usage semantics per API (cached = `min(matched, P − 1)`; Anthropic input excludes cache read; others inclusive; `cache_creation_input_tokens` = 0) and the cache values in mrun metadata (delivered by T015) + `cross_*` conformance cases
- [ ] T018 [US2] Error mapping per API per both contract status tables, before and after headers (queue full, queue deadline, context too long with numbers, unsupported modalities with JSON path, unknown tool result id, limits, auth, Host/Origin) + cases
- [ ] T019 [US2] Request validation limits (body 64 MiB via tower-http, with a case carrying 5 base64 screenshots in history, messages, tools, schema size/depth) + cases
- [ ] T011a [US1] MCP and skills support (FR-027–FR-029): Codex namespace flattening with alias map and `namespace` restored on streamed and final calls; `defer_loading` tools excluded from the tools region; `tool_reference` / `tool_search_output` expansion rendered inline in the tool result; mid-conversation system/developer messages rendered in place (Qwen3.6 handling per the FR-029 clarification); 400 for server-executed features; `cache_control`, `prompt_cache_key` and session headers ignored (FR-030); conformance cases `resp_namespace_roundtrip`, `msg_tool_reference_inline_no_prefix_change`, `msg_midconversation_system_in_place`, `msg_mcp_servers_rejected`
- [ ] T019c [US2] `x-mrun-ignore-eos: on` (FR-031): loopback-only; generation stops only at the output limit; conformance case and non-loopback-ignored case
- [ ] T019b [US2] Billing-nonce normalization (FR-025): only the `cch=` value in a leading system text starting with `x-anthropic-billing-header:`; byte-length preserved; tests: two requests differing only in the nonce share the whole prefix, any other change is untouched; progress in Chat keepalive comments and `/mrun/status` (FR-026) with `chat_keepalive_comment_progress`
- [ ] T019a [US2] ~~Anthropic `diagnostics` (FR-023)~~ *Removed 2026-10-15.* Remaining scope: `diagnostics` and `context_management` (FR-024) accepted and ignored; cases `msg_diagnostics_ignored`, `msg_context_management_ignored`

## Phase 5: US3 — Reliable under agent behavior (P2)

- [ ] T020 [US3] Disconnect tests in every phase (queued, prefilling, generating, non-streaming) via the close watcher: work stops and the next request starts within 300 ms (`xapi_disconnect_while_queued_prefilling_nonstream_cancels_300ms`); stalled reader → cancel after 10 s; keepalive test `resp_queued_over_300s_eventsource_client_survives` (Rust `eventsource-stream` client with a 300 s idle timeout, paused Tokio clock); SC-005 measured on macOS
- [ ] T021 [US3] Non-loopback bind requires an API key from `MRUN_API_KEY`/`--api-key-file` (constant-time compare); auth tests; `serve --record <dir>` (FR-021: CLI-only, loopback-only, 0600 files, startup warning, shown in status; each record carries the body, allow-listed session headers `x-claude-code-session-id`/`thread-id`/`x-session-id`, RFC 3339 millisecond arrival and response-end times and output token count; `xapi_record_fields_and_session_headers`); audit test that no prompt text reaches logs or disk without `--record` and that config/env cannot enable it (SC-006)

## Phase 6: Polish

- [ ] T022 [P] Serving overhead test (SC-004): same prompt through `EngineHandle` directly and through each API, compare TTFT and decode rate (`crates/mrun/tests/overhead.rs`, macOS real model, `#[ignore]`)
- [ ] T022a [P] Client compatibility matrix in CI: install the latest Claude Code, Codex and OpenCode on Linux nightly, point them at `mrun serve` with `ScriptedEngine` tool-call scripts, run one tool round trip each plus one request held in the queue past the client's idle timeout, and fail on wire errors; pinned versions run on every PR
- [ ] T023 [P] Site page `site/content/docs/clients.md` and `api.md` checked against the implemented behavior

## Dependencies

T001 → T003 → T004/T005 → T009–T011 (parallel) → T012–T016. T006–T008 parallel after T003.
T018 covers in-stream errors, so it depends on T008.
Requires 001's `Engine` trait (001 T008), chat layer (001 T010) and generation loop (001 T011); real-model runs need 001 MVP.
