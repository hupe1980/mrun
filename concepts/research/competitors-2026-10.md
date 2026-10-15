# Competitor Source Study (2026-10-15)

**Status**: Research input; the adopted items are already folded into specs 001–005.

**Method**: Four parallel studies read source at pinned commits:
- oMLX `958974c`, mlx-lm `ba8a08f`, LM Studio mlx-engine `ab52366`, vllm-mlx `9a939f8`, Ollama `8e4dad4`, exo `bc412e2`, Rapid-MLX `5c70e23`;
- llama.cpp `3e390af`;
- SGLang `3b5ae4e`, vLLM `f46021a`, Dynamo `f5c3659`, llguidance `6cef6ed`, mistral.rs `28ed6da`, LMCache `c3b875a`;
- Codex and OpenCode, both from 2026-10-15;
- the official Claude Code docs: env-vars, model-config and llm-gateway-protocol.

GitHub issue search was blocked (403), so issue-level claims come from web pages and are marked [U]. [V] means verified in source.

## What changes for mrun

| # | Lesson | Evidence | Where it went |
| --- | --- | --- | --- |
| 1 | **Clamp the output limit; refuse only prompts that cannot fit.** Claude Code sends large `max_tokens` on every request; refusing prompt + limit > capacity turns away turns that fit | llama.cpp `server-context.cpp:3577-3585` refuses only `prompt ≥ n_ctx` [V]. mrun code refused (fixed in `runtime/generate.rs`) | 001 FR-009, SC-005 |
| 2 | **Keep completed prefill chunks on cancel.** Client timeouts plus retries restart long prefills forever | Ollama #17839 [U]; fix in `mlxrunner/prefix_cache.go:585-589` [V] | 003 R2 (already FR-004 `finish(prefilled)`); site |
| 3 | **Keep streams alive during prefill.** Claude Code aborts after 300 s without bytes; `retry-after` > 60 stops its retries | Claude Code gateway-protocol doc [V] | 002 FR-010, FR-026 |
| 4 | **Fix Claude Code's attribution nonce server-side** (`cch=` in `x-anthropic-billing-header`), length preserved | llama.cpp `server-chat.cpp:312-346` [V]; vllm-mlx's unanchored version truncated system prompts (#524/#767) [U], hence the narrow anchor | 002 FR-025 |
| 5 | **Warm the next turn while tools run** (render the conversation as the client will resend it, prefill with no output, preempt on arrival) | Dynamo `preprocessor/speculative_prefill.rs` [V] | 003 FR-015 |
| 6 | **Session identity from clients' own headers** (Claude Code session/agent ids, Codex `thread-id`, OpenCode `x-session-id`) for eviction affinity and subagent awareness | Dynamo `protocols/agents.rs:11-20` [V] | 003 FR-014 |
| 7 | **Tool-call parsing that survives code**: end values only at `\n</parameter>\n` + next tag; no coercion (`null` stays a string for string params, exact booleans); never `eval` | vLLM `qwen_coder.rs:273` and SGLang match the first closing tag [V]; SGLang coerces (`qwen3_coder_detector.py:106`) [V]; CVE-2025-9141 from `eval` [U]; llama.cpp `qwen3-coder.cpp` gets it right [V] | 002 FR-004, research §3 |
| 8 | **Client config is version-sensitive.** Codex rejects root `profile =`, needs `<name>.config.toml`, assumes 272K for unknown models; Claude Code needs `ANTHROPIC_API_KEY` emptied and `CLAUDE_CODE_MAX_OUTPUT_TOKENS`; OpenCode takes `OPENCODE_CONFIG_CONTENT` | Codex `config/mod.rs:3458`, `model_info.rs:98-135`; Ollama `cmd/launch/{claude,codex,opencode}.go` [V] | 004 FR-005/006, research R5 |
| 9 | **Test against current client binaries continuously.** Claude Code 2.1.154 broke oMLX (mid-conversation system messages, #1503); LM Studio 0.4.19 broke `/v1/messages` tools (#2164) [U] | Rapid-MLX tier-1 client matrix [V README] | 002 T022a |
| 10 | **Single GPU thread is right.** oMLX #300 (IOGPUMemory panic), vllm-mlx #407 (stream error returned as HTTP 200) [U] | | site how-it-works |
| 11 | **Long-session leaks are real.** Graph inputs kept alive by copies (exo `cache.py:84-106` [V]), Metal buffer limit in recurrent chains (oMLX `scheduler.py:2406` [V]), 120 GB pinned by finished requests (vllm-mlx #442 [U]) | | 001 SC-006 2-hour soak |
| 12 | **Chunk prefill at message boundaries.** Free for attention models, required for hybrid checkpoints | llama.cpp `server-context.cpp:3994-4080` [V]; Ollama's fixed 8K grid misses (#17829) [U] | 003 R2 |
| 13 | **Never shrink or truncate silently.** Ollama truncates history on the GGUF path (`routes.go:3244`, `prompt.go:35-73`) and retries loads with a smaller context unannounced (`sched.go:699-713`) [V]; unloads after 5 min idle | | 001 SC-006 (no idle unload); existing refuse-with-numbers rule |
| 14 | **Real token counts.** Ollama has no `count_tokens` and estimates usage as len/4 (`anthropic.go:1125`) [V] | | already 002 |

## Claims corrected

- **Miss diagnostics are no longer unique.** oMLX computes the first divergence against the last 4 stored sequences and exposes it in stats (`scheduler.py:10005-10100`) [V]. mrun's distinct parts are cause classification, per-response delivery (headers or SSE) and the Anthropic `cache_miss_reason` wire format.
- **A radix cache on MLX exists elsewhere.** SGLang has an opt-in MLX backend with RadixCache (`hardware_backend/mlx/`) [V]. mlx-lm's cache is also exact-token, but it copies the whole KV on every hit (`models/cache.py:1655`) [V]. mrun's zero-copy live path is the measurable difference, and `mrun bench` should show it.
- **"Reuse never changes output" overclaims.** Prefix-then-suffix prefill differs from one-pass prefill in bf16 rounding (mlx-lm #259) [U]. The site now says the result equals recomputation "up to rounding".

## Considered and not adopted

- **Always-on lazy grammar for tool envelopes** (llama.cpp `qwen3-coder.cpp:178-188`, vLLM `ToolStrictLevel::function`). It removes malformed calls, but masked decode is unpipelined and file-writing calls run to thousands of tokens. Revisit with `mrun bench` malformed-call counts (005 FR-003a).
- **`--disallowedTools LSP` in `launch`** (oMLX). It removes a user feature to protect the cache; mrun reports the miss instead.
- **Dynamic memory ceilings** (oMLX `process_memory_enforcer.py`). A static budget plus pressure shedding is simpler and predictable.
- **Approximate reuse** (llama.cpp `--cache-reuse` KV shifting, context shift; SGLang int8 GDN checkpoints): not exact.
- **Server-side Responses store** (vLLM never frees it, `responses/serving.py:165-198`).
- **Multi-node KV routing** (Dynamo, LMCache): no single-user value beyond hit/query counters for a future `/metrics`.

## For later specs

- **006 hybrid models.**
  - Checkpoint at message boundaries and just before the generation tail: Ollama `len-4`, LM Studio `total-11`, and mlx-lm's think tail with a system-prompt boundary diff (`server.py:586-626`).
  - Cap the number of states kept per path (SGLang `mamba.rs:20-75`).
  - Add a miss cause for "no checkpoint at this position".
  - Re-check Qwen3.5+ templates for mid-conversation system messages (LM Studio #2298, Ollama #17778).
- **008 speculative decoding.** llama.cpp's n-gram family (`common/speculative.cpp`), with a per-position acceptance metric.
- **Thinking-off probe.** vLLM renders `reasoning_effort=none` and compares it with other efforts to detect templates that ignore it (`anthropic/serving.py:207-240`). Do this when hybrid thinking models become a default.
- **Reference code.** vLLM `rust/` (winnow streaming parsers) and SGLang `rust/sglang-radix-tree` are Rust references for mrun's parser and tree.
