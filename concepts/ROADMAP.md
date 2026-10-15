# Roadmap

The canonical feature order lives in the [spec index](../specs/README.md). This page explains the
route and the gates between milestones.

## Milestones

| Milestone | Outcome | Specs | Gate to move on |
| --- | --- | --- | --- |
| **M0 Foundation** | Hard cut done; MLX binding spike on a Mac (load, matmul, SDPA, quantized matmul, memory APIs) | 001 (phase 1) | Go/no-go on the binding layer within one week |
| **M1 Engine** | `mrun run` streams Qwen3-8B and Qwen3-Coder-30B-A3B (4-bit) with owned KV cache, chunked prefill, per-layer cancellation and a startup memory budget | 001 | Greedy output matches mlx-lm token-for-token; prefill and decode ≥ 90% of mlx-lm; cancel ≤ 300 ms p95 |
| **M2 Agent APIs** | Claude Code, Codex and OpenCode complete a real tool loop against `mrun serve` (cache off) | 002 | ~95-case conformance suite green; 5/5 real-client runs per client |
| **M3 Prefix store** | Exact radix reuse across turns, branches and tool loops, with miss diagnostics | 003 | Repeated-turn TTFT ≥ 5× faster than cache-off at 30K context; zero stale reuse in the mutation suite |
| **M4 Models & DX** | `mrun pull/ls/rm/launch/connect/doctor/ps`; Qwen3-Coder-30B-A3B as the default model | 004 | Fresh Mac to working Claude Code session in under 5 minutes |
| **M5 Bench** | `mrun bench` with agent-trace replay against llama.cpp, Ollama, oMLX, mlx-lm | 005 | First dated public comparison published, losing cells included |

M4 and M5 can start in parallel with M3. M1's gates use 001's in-repo engine benchmark; 005's HTTP
harness becomes useful once M2 lands.

## After M5: ordered candidates

Sources: [next-features-2026-10](research/next-features-2026-10.md) and
[arxiv-sota-2026-10](research/arxiv-sota-2026-10.md). Each becomes a numbered spec only after its
**measure-first** check, run with 005 on recorded traces, comes back positive.

| Order | Candidate | Measure first | Notes |
| --- | --- | --- | --- |
| 006 | **Hybrid models with state checkpoints** (Qwen3.6-35B-A3B first) | Checkpoint bytes against prefill seconds saved per boundary | Exact checkpoint placement (Sparse Prefix Caching); reject approximate replay schemes. Prerequisite for MTP |
| 007 | **Persistent SSD prefix tier** (opt-in, identity-bound, token-verified, exact bf16) | SSD read GB/s on the target Mac | About 0.5 s reload against tens of seconds of prefill; replaces 003 FR-010 (memory only) |
| 008 | **Copy-aware speculative decoding**: suffix/n-gram drafts that also index rendered tool calls, with a batch-1 MoE controller that can choose no speculation | n-gram acceptance length on Claude Code and OpenCode traces | Realistic gains 1.5–2.3×; MoE verification cost makes the controller mandatory; MTP after 006 |
| 009 | **8-bit attention state** (opt-in) | Long-context coding quality check | About 2× context; per-layer dequantize for prefill; 4-bit stays out |
| — | Small items | — | `/metrics`, keep completed prefill chunks on cancel, tool-call early stop, sleep assertion while busy, launchd service and Homebrew |

Already folded into the M2/M3 specs: Anthropic `diagnostics.cache_miss_reason` (002 FR-023),
accepting and ignoring server-side context management (002 FR-024), cache hints (003 FR-014), recorded tool gaps
in replay (005 FR-003a). Eviction stays plain LRU with turn-type order: two 2026 studies found LRU
near-optimal for single long sessions, so cost-aware scoring waits for hybrid models.

## Later

- **Parallel sessions**: batched decode, only if traces show agents queued behind each other;
  verify-and-rollback determinism then.
- **logprobs**, EAGLE-3 or DFlash drafters (none exists for the default model).
- **Linux / CUDA**: Linux builds compile and pass portable tests from day one (CI). Running models
  on Linux (MLX CPU, MLX CUDA backend) is evaluated only after the Mac release; no performance
  commitments there. Competing with vLLM/SGLang on CUDA servers is not a goal.

## Rejected after research

Non-prefix (position-independent) KV reuse and approximate prefill: not exact. 4-bit KV by
default: no evidence on long-context code. MoE expert offload: the default model fits 48 GB.
Server-side compaction and a menu-bar app before 1.0.

## Explicitly not planned

See the [vision non-goals](VISION.md#non-goals).
