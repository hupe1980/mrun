# Roadmap

**Updated**: 2026-10-15 (design sprint 3). The spec index ([specs/README](../specs/README.md)) owns
feature order; this page explains the direction, the route and the gates.

## Direction

mrun competes on what an agent user waits for on a Mac, not on chat tok/s. The research of this
sprint ([competitors](research/competitors-2026-10.md), [arxiv-sota](research/arxiv-sota-2026-10.md),
[next-features](research/next-features-2026-10.md)) changes three things:

1. **The best model that fits a 48 GB Mac is now a hybrid** (Qwen3.6-35B-A3B). Hybrid support moves
   from "later" into the first release; it also removes the context squeeze that 8-bit KV would
   have addressed, so 8-bit KV is dropped.
2. **Branching is the real cache problem**, not single-conversation reuse. Subagents, title
   requests and retries must not destroy the main conversation; state moves instead of being
   copied or discarded.
3. **Client fidelity is where competitors break**: wire-format changes, namespaces, deferred tools,
   keepalives, context windows. A nightly client matrix and `mrun launch` that sets everything are
   first-release features.

## First release (0.1)

| Milestone | Outcome | Specs | Gate |
| --- | --- | --- | --- |
| **M0 Foundation** | MLX binding spike on a Mac: load, quantized matmul, gather-qmm, SDPA, RoPE, a custom Metal kernel, memory APIs | 001 phase 1 | Go/no-go within one week |
| **M1 Engine** | `mrun run` streams Qwen3-8B, Qwen3-Coder-30B-A3B and Qwen3.6-35B-A3B with the memory budget, checkpoints and per-layer cancellation | 001 | 001 SC-001…SC-009 |
| **M2 Agent APIs** | Claude Code, Codex and OpenCode complete real tool loops, including MCP tools | 002 | 002 SC-001…SC-006; client matrix green |
| **M3 Prefix store** | Exact reuse across turns and branches for attention and hybrid models, with miss reports | 003 | 003 SC-001…SC-006 |
| **M4 Models & DX** | `pull`, `launch`, `connect`, `doctor`, `ps`; default model confirmed | 004 | 004 SC-001…SC-005 |
| **M5 Bench** | `mrun bench` with in-order trace replay against llama.cpp, Ollama, oMLX, mlx-lm and SGLang-MLX | 005 | First dated public comparison, losing cells included |
| **M6 Distribution** | Homebrew formula, signed and notarized binary, launchd service, `/metrics` | 008 | Fresh Mac: `brew install` → `mrun launch claude-code` works |

M4, M5 and M6 can run in parallel with M3. The binding spike gates everything.

## After 0.1 (each needs its measure-first check)

| Order | Spec | Measure first | Notes |
| --- | --- | --- | --- |
| 1 | **006 Speculative decoding**: MTP heads (Qwen3.6 ships them) and copy-aware n-gram drafts for code edits, with a controller that turns speculation off where it loses | n-gram acceptance and MTP acceptance on recorded traces | On the MoE default, gains come from verbatim-copy stretches and long context; elsewhere expect ≈1.0×. Needs weights with `mtp.*` tensors (mlx-community conversions drop them) |
| 2 | **007 Persistent prefix tier** (opt-in SSD, exact bytes, model-bound, token-verified) | Share of turns missing because of restart or eviction; SSD read speed | Allowed by constitution IV (explicit opt-in, bound to model identity) |
| 3 | Parallel sessions (batched decode) | Share of wall time queued behind other requests in traces | Only if queueing is significant |
| 4 | Linux / CUDA evaluation | — | Linux already builds and runs portable tests; no performance commitments |

## Dropped in this sprint

8-bit KV (hybrid default removes the need), idle warm-up prefill, session and compaction hints for
eviction, `cache_control` boundaries, Anthropic `diagnostics` (no client sends it), quality suite
and gap replay in bench, priming in bench replay, cost-aware eviction for attention models.

## Not planned

See the [vision non-goals](VISION.md#non-goals).
