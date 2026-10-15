# Implementation Plan: Agent APIs

**Branch**: `002-agent-apis` | **Date**: 2026-10-14, revised 2026-10-15 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `specs/002-agent-apis/spec.md`

## Summary

`mrun serve` is an Axum server with three thin decoders (Chat Completions, Anthropic Messages,
Responses) into one `ChatRequest`, one render/tokenize path, a single-worker FIFO queue in front of
the engine, and three encoders that fold one `GenEvent` stream into each API's native SSE or JSON
body. Tool calls and reasoning are parsed by streaming parsers from a syntax registry. The family
names an ordered set of accepted syntaxes; the template probe at load picks the first that matches
(FR-032): Hermes JSON (Qwen3), Qwen XML
(Qwen3-Coder, Qwen3.6), Harmony (gpt-oss: analysis/commentary/final channels, `to=functions.NAME`
calls, reasoning effort) and Mistral `[TOOL_CALLS]` (Devstral Small 2); strict tools and JSON response formats arm an llguidance matcher
that supplies per-step token masks to the engine (uncompilable schemas fall back to unconstrained
with a warning); `tool_choice: none` is a fixed logit bias that keeps decode pipelined. Keepalives
are per API (Responses `response.in_progress`, Anthropic `ping`, Chat comments), and a
per-connection close watcher cancels requests in any phase. Client quirks of Claude Code 2.1.300,
Codex (main `ae11049`) and OpenCode (`13afea6`) are handled explicitly ([research](research.md)).

## Technical Context

**Language/Version**: Rust 1.94.0 (edition 2024)
**Primary Dependencies**: axum 0.8, tokio 1.53, tower-http 0.7 (limits, trace), serde/serde_json
(`preserve_order`), llguidance 1.9.1 + toktrie 1.9.1, minijinja 3 (via 001's chat layer),
ulid (ids), base64, subtle (constant-time key compare)
**Storage**: None (stateless; reasoning round-trips inside client-held `encrypted_content`/signature); `--record <dir>` writes JSON lines only when given on the command line (FR-021)
**Testing**: cargo test with `ScriptedEngine` (deterministic token scripts incl. tool-call text) for
~100 named conformance cases (`crates/mrun/tests/api_conformance/`), insta snapshots of SSE streams,
real-client smoke tests on macOS (Claude Code, Codex, OpenCode pinned) as part of quickstart
**Target Platform**: macOS (serving); Linux runs all conformance tests with `ScriptedEngine`
**Project Type**: HTTP server inside the `mrun` binary
**Performance Goals**: API overhead ≤ 50 ms TTFT, ≥ 97% engine decode speed (SC-004); llguidance
mask ≤ 2 ms typical, computed on CPU while the GPU runs the next step
**Constraints**: Loopback default with Host/Origin checks; no prompt logging; one generation at a
time; queue 16 waiting / 600 s wait deadline
**Scale/Scope**: 1 active + 16 queued requests; configurable limits (bodies 64 MiB, tools 256, schema
depth 16)

## Constitution Check

| Principle | Gate | Status |
| --- | --- | --- |
| I Agent latency | Serving overhead SC-004; cancel on disconnect SC-005 | Pass |
| II Exactness | Usage counts exact (input/cached/output per API semantics); no silent tool repair | Pass |
| III Memory budget | Admission via engine capacity; body/schema caps bound host work | Pass |
| IV Local, private | Loopback default, API key required off-loopback, no content logs, stateless Responses | Pass |
| V Own runtime | Engine types hidden behind `Engine` trait | Pass |
| VI Simple | One internal model, three encoders; no Ollama API, no stored responses | Pass |

Post-design re-check: Pass.

## Project Structure

### Documentation (this feature)

```text
specs/002-agent-apis/
├── plan.md  research.md  data-model.md  quickstart.md
├── contracts/api.md
└── tasks.md
```

### Source Code (repository root)

```text
crates/mrun/src/
├── api/
│   ├── mod.rs            # router, limits, auth, /health, HEAD /api/hello, /v1/models, /mrun/status
│   ├── model.rs          # ChatRequest, Message, ToolDef, GenEvent, Finish, Usage, ApiError
│   ├── chat_completions.rs   # decode + encode (SSE/JSON)
│   ├── messages.rs           # decode + encode + count_tokens
│   ├── responses.rs          # decode + encode (stateless)
│   ├── sse.rs                # SSE writer, per-API keepalive (in_progress / ping / comment), in-stream errors
│   ├── conn.rs               # per-connection close watcher → cancel token (FR-011)
│   └── record.rs             # --record JSON lines (FR-021)
├── chat/
│   ├── render.rs         # ChatRequest -> template context -> tokens (shared with count_tokens)
│   ├── syntax/{mod.rs, registry.rs, holdback.rs, hermes.rs, qwen_xml.rs, harmony.rs, mistral.rs}
│   │                     # registry: family → accepted syntaxes, template probe; qwen_xml serves Qwen3-Coder and Qwen3.6
│   ├── constrain.rs      # llguidance ParserFactory per model, Matcher per request, strict-unsupported fallback,
│   │                     # tool_choice:none opener-token bias
│   └── reasoning.rs      # think-tag split, codecs; thinking control and history rules read from the
│                         # syntax entry (data-model Syntax rows)
├── runtime/
│   ├── queue.rs          # FIFO, deadline, cancel tokens, retry-after
│   └── worker.rs         # dequeues and calls 001 `runtime/generate.rs` (owner of the per-request loop;
│                         # reuse policy: 003 PrefixStore); maps its events to GenEvents
└── cli/serve.rs
crates/mrun/tests/
├── api_conformance/{chat.rs, messages.rs, responses.rs, cross.rs}
└── snapshots/
```

**Structure Decision**: All in the `mrun` crate; the engine is reached only through the `Engine`
trait so every API test runs on Linux with `ScriptedEngine`.

## Complexity Tracking

None.
