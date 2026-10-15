# Implementation Plan: Agent APIs

**Branch**: `002-agent-apis` | **Date**: 2026-10-14 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `specs/002-agent-apis/spec.md`

## Summary

`mrun serve` is an Axum server with three thin decoders (Chat Completions, Anthropic Messages,
Responses) into one `ChatRequest`, one render/tokenize path, a single-worker FIFO queue in front of
the engine, and three encoders that fold one `GenEvent` stream into each API's native SSE or JSON
body. Tool calls are parsed by streaming `ToolSyntax` implementations for Qwen3 (Hermes JSON) and
Qwen3-Coder (XML); strict tools and JSON response formats arm an llguidance matcher that supplies
per-step token masks to the engine. Client quirks of Claude Code 2.1.300, Codex (main `ae11049`)
and OpenCode (`13afea6`) are handled explicitly ([research](research.md)).

## Technical Context

**Language/Version**: Rust 1.94.0 (edition 2024)
**Primary Dependencies**: axum 0.8, tokio 1.53, tower-http 0.7 (limits, trace), serde/serde_json
(`preserve_order`), llguidance 1.9.1 + toktrie 1.9.1, minijinja 3 (via 001's chat layer),
ulid (ids), base64, subtle (constant-time key compare)
**Storage**: None on disk (stateless; reasoning round-trips inside client-held `encrypted_content`/signature); in memory only the DiagnosticsIndex (256 fingerprints, FR-023)
**Testing**: cargo test with `ScriptedEngine` (deterministic token scripts incl. tool-call text) for
~95 named conformance cases (`crates/mrun/tests/api_conformance/`), insta snapshots of SSE streams,
real-client smoke tests on macOS (Claude Code, Codex, OpenCode pinned) as part of quickstart
**Target Platform**: macOS (serving); Linux runs all conformance tests with `ScriptedEngine`
**Project Type**: HTTP server inside the `mrun` binary
**Performance Goals**: API overhead ≤ 50 ms TTFT, ≥ 97% engine decode speed (SC-004); llguidance
mask ≤ 2 ms typical, computed on CPU while the GPU runs the next step
**Constraints**: Loopback default with Host/Origin checks; no prompt logging; one generation at a
time; queue 8 waiting / 600 s wait deadline
**Scale/Scope**: 1 active + 8 queued requests; configurable limits (bodies 8 MiB, tools 256, schema
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
│   ├── diagnostics.rs        # Prompt Fingerprint, DiagnosticsIndex, cache_miss_reason (FR-023)
│   ├── responses.rs          # decode + encode (stateless)
│   └── sse.rs                # SSE writer, ping/keepalive, disconnect detection
├── chat/
│   ├── render.rs         # ChatRequest -> template context -> tokens (shared with count_tokens)
│   ├── tools/{mod.rs, hermes.rs, qwen_xml.rs, holdback.rs}
│   ├── constrain.rs      # llguidance ParserFactory per model, Matcher per request
│   └── reasoning.rs      # think-tag split, signature + encrypted_content codec
├── runtime/
│   ├── queue.rs          # FIFO, deadline, cancel tokens, retry-after
│   └── worker.rs         # pulls from queue, drives Engine, fans out GenEvents
└── cli/serve.rs
crates/mrun/tests/
├── api_conformance/{chat.rs, messages.rs, responses.rs, cross.rs}
└── snapshots/
```

**Structure Decision**: All in the `mrun` crate; the engine is reached only through the `Engine`
trait so every API test runs on Linux with `ScriptedEngine`.

## Complexity Tracking

None.
