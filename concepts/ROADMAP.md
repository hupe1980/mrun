# Roadmap

**Updated**: 2026-10-15 (design sprint 3). The spec index ([specs/README](../specs/README.md)) owns
feature order; this page explains the direction, the route and the gates.

## Direction

mrun competes on what an agent user waits for on a Mac, not on chat tok/s. The research of this
sprint ([competitors](research/competitors-2026-10.md), [arxiv-sota](research/arxiv-sota-2026-10.md),
[next-features](research/next-features-2026-10.md)) changes four things:

1. **The best model that fits a 48 GB Mac is now a hybrid** (Qwen3.6-35B-A3B). Hybrid support moves
   from "later" into the first release; it also removes the context squeeze that 8-bit KV would
   have addressed, so 8-bit KV is dropped.
2. **Model support must be open.** Strong coding models come from several families with different
   attention (sliding window, MLA, linear) and tool formats (Harmony, Mistral, GLM). The engine is a
   block library with small family modules and a syntax registry; 0.1 ships four families to prove
   it (spec 010 tracks the family set).
3. **Branching is the real cache problem**, not single-conversation reuse. Subagents, title
   requests and retries must not destroy the main conversation; state moves instead of being
   copied or discarded.
4. **Client fidelity is where competitors break**: wire-format changes, namespaces, deferred tools,
   keepalives, context windows. A nightly client matrix and `mrun launch` that sets everything are
   first-release features.

## First release (0.1)

| Milestone | Outcome | Specs | Gate |
| --- | --- | --- | --- |
| **M0 Foundation** | MLX binding spike on a Mac: load, quantized matmul, gather-qmm, SDPA, RoPE, a custom Metal kernel, memory APIs | 001 phase 1 | Go/no-go within one week |
| **M1 Engine** | `mrun run` streams the 0.1 families (Qwen3.6, Qwen3-Coder, gpt-oss, Devstral, Qwen3-8B) built from the shared block library, with the memory budget, checkpoints and per-layer cancellation | 001, 010 | 001 SC-001…SC-009 (SC-004 max-depth bucket recorded; at risk, 001 R5) |
| **M2 Agent APIs** | Claude Code, Codex and OpenCode complete real tool loops, including MCP tools | 002 | 002 SC-001…SC-006; client matrix green |
| **M3 Prefix cache** | Exact reuse across turns and branches for attention and hybrid models, with miss reports | 003 | 003 SC-001…SC-006; SC-001 synthetic part gates M3; the trace part is reported with M5 |
| **M4 Models & DX** | `pull`, `launch`, `connect`, `doctor`, `ps`; default model confirmed | 004 | 004 SC-001…SC-005 |
| **M5 Bench** | `mrun bench` with in-order trace replay against llama.cpp, Ollama, oMLX, mlx-lm and SGLang-MLX | 005 | First dated public comparison, losing cells included |
| **M6 Distribution** | Homebrew formula, signed and notarized binary, launchd service, `/metrics` | 008 | Fresh Mac: `brew install` → `mrun launch claude-code` works |

M4, M5 and M6 can run in parallel with M3. The binding spike gates everything.

## After 0.1 (each needs its measure-first check)

| Order | Spec | Measure first | Notes |
| --- | --- | --- | --- |
| 1 | **006 Speculative decoding**: copy-aware n-gram/suffix drafts for code edits, with a controller that turns speculation off where it loses; MTP heads only as a gated follow-up (D024) | n-gram acceptance on recorded traces (offline, tokenizer only); MTP: α₁ ≥ 0.8 and modeled ≥ 1.15× before any MTP work | On MoE models, gains come from verbatim-copy stretches and long context; elsewhere expect ≈1.0× |
| 1b | **010 More model families**: GLM-4.7-Flash (MLA), Gemma 4 (sliding window + global), Nemotron-H (Mamba2 hybrid), then by demand | Agent-suite results of candidate models on the target Mac | Each family: one module, a syntax entry if new, a parity test |
| 2 | **007 Persistent prefix tier** (opt-in SSD, exact bytes, model-bound, token-verified) | Share of turns missing because of restart or eviction; SSD read speed | Allowed by constitution IV (explicit opt-in, bound to model identity) |
| 3 | **009 Image input**: screenshots and image tool results for Qwen3.6's vision encoder; exact prefix identity per image | Share of recorded sessions containing images (gate ≥ 5%); whether the pinned repo keeps the vision tower | Until then images in tool results are replaced by a placeholder |
| 4 | Parallel sessions (batched decode) | Share of wall time queued behind other requests in traces | Only if queueing is significant |
| 5 | Linux / CUDA evaluation | — | Linux already builds and runs portable tests; no performance commitments |

## Dropped in this sprint

8-bit KV (hybrid default removes the need), idle warm-up prefill, session and compaction hints for
eviction, `cache_control` boundaries, Anthropic `diagnostics` (no client sends it), quality suite
and gap replay in bench, priming in bench replay, cost-aware eviction for attention models.

## Not planned

See the [vision non-goals](VISION.md#non-goals).
