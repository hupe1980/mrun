# Decisions

Architecture decision records. IDs are stable; superseded entries name their replacement.

| ID | Decision | Status |
| --- | --- | --- |
| D001 | Optimize agent latency (TTFT on repeated turns, tool-loop turnaround), not chat tok/s | Accepted |
| D002 | ~~llama.cpp/Metal behind a C++ shim~~ | Superseded by D003, then D010 |
| D003 | ~~Candle as the tensor core~~ | Superseded by D010 (2026-10-14) |
| D004 | Exact prefix reuse only; approximate methods opt-in and quality-gated | Accepted |
| D005 | ~~Planning stays local (gitignored)~~ | Superseded by D013 |
| D010 | Engine: Apple MLX, called from Rust | Accepted 2026-10-14 |
| D011 | mrun owns the runtime: scheduler, KV state, prefix store, memory budget, APIs | Accepted |
| D012 | Memory budgeted by construction, not per-allocation hooks | Accepted |
| D013 | Specs, concepts and Spec Kit tooling are tracked in git | Accepted |
| D014 | Ship OpenAI Chat Completions, Anthropic Messages and stateless Responses together | Accepted 2026-10-14 |
| D015 | Model format: MLX safetensors (mlx-community layout); Qwen3 family first | Accepted |
| D016 | Drop the qualification-report/evidence-gating machinery; use `mrun bench` results | Accepted |
| D017 | Prefix store is a radix tree of KV snapshots, designed for hybrid-model state too | Accepted |

## D010: Engine is MLX

**Context.** The Candle audit ([candle-capabilities](research/candle-capabilities.md)) found no
Metal quantized MoE path, an open Metal corruption bug past 8,192 KV tokens (#3705), F16 RoPE
tables, and no published Metal performance data. The landscape ([landscape](research/landscape-2026-10.md))
shows the strongest local competitors moving to MLX (Ollama's MLX engine, oMLX, LM Studio).

**Decision.** Build on MLX through Rust bindings (pin and binding layer chosen in the 001 plan
after [mlx-rust](research/mlx-rust.md)). mrun ports model graphs (Qwen3 dense, Qwen3 MoE) to Rust
on MLX ops, owns KV state, and reuses MLX kernels (quantized matmul, gather-qmm, SDPA, RoPE).

**Consequences.** MoE models (Qwen3-Coder-30B-A3B) are reachable without new kernels. Build needs
MLX's C/C++ library; Linux CI runs MLX's CPU backend if the bindings support it, otherwise CI
covers everything except the engine crate. The engine sits behind one internal module boundary,
not a plugin system.

## D011: Own the runtime

The differentiator is the runtime around the engine: exact reuse, explainable misses, memory
admission, cancellation and API fidelity. These are written in Rust in mrun; model graphs are thin
ports of mlx-lm reference implementations, tested against them.

## D012: Budget by construction

The 2026-10-14 Candle experiment ([results](research/candle-budget/results.md)) showed that
per-allocation enforcement inside third-party code is not achievable: op scratch, template
intermediates and grammar construction allocate before any limit trips. mrun instead (1) computes a
budget at startup from physical memory and the GPU working-set recommendation, (2) loads weights,
(3) measures peak scratch with a worst-case prefill chunk, (4) sizes the KV arena and prefix store
from the remainder, (5) admits requests by exact KV bytes, and (6) caps inputs (body, schema,
template) so host-side work stays inside a measured envelope. MLX memory limits and cache limits
back this up at runtime.

## D014: Three APIs at once

Codex CLI only speaks Responses; Claude Code only speaks Anthropic Messages; OpenCode and most
others speak Chat Completions. All three map onto one internal request model (messages, tools,
sampling, stream). Responses is stateless (`store=false` semantics); no server-side conversation
storage (Principle IV).

## D015: Models

- Reference: Qwen3-8B (dense, full attention) for correctness and engineering.
- Primary: Qwen3-Coder-30B-A3B-Instruct, 4-bit (MoE, full attention, ~17–19 GB).
- Next: hybrid-attention coders (Qwen3.6/3.8-27B class) once the prefix store supports state
  checkpoints. Models are pulled explicitly and identified by content hash.

## D016: Drop evidence-gating machinery

The previous design gated serving on hashed qualification reports, staged candidate binaries and
100-case fixture catalogs. That machinery delivered no user value and blocked progress. Correctness
lives in tests; performance lives in reproducible `mrun bench` runs with recorded settings.

## D017: Prefix store design

A radix tree keyed by token IDs, rooted per model identity. Nodes own immutable KV snapshots of
token ranges (segments). Restoring a prefix copies matched segments into the active cache; the hot
path (same conversation continuing) reuses the active cache with zero copy. Every lookup confirms
tokens, not just hashes. For hybrid models, nodes additionally hold recurrent-state checkpoints at
chosen boundaries. Eviction starts as LRU by turn type (assistant, then user/tool, then system);
cost-aware eviction is a later optimization if benchmarks show it matters.
