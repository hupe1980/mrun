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

## Later (spec when evidence justifies)

- **Hybrid-attention models** (Qwen3.6/3.8-27B class): state checkpoints in the prefix store.
- **SSD cache tier**: opt-in, content-addressed, model-bound; survives restarts.
- **Speculative decoding**: MTP or prompt-lookup for code edits; opt-in and quality-gated.
- **Parallel sessions**: small-batch decode for concurrent subagents.
- **Distribution**: Homebrew formula, signed release binaries, launchd service.
- **Linux / CUDA**: Linux builds compile and pass portable tests from day one (CI). Running models
  on Linux (MLX CPU, MLX CUDA backend) is evaluated only after the Mac release; no performance
  commitments there. Competing with vLLM/SGLang on CUDA servers is not a goal.

## Explicitly not planned

See the [vision non-goals](VISION.md#non-goals).
