# Decisions

Architecture decision records. IDs are stable. Superseded early records (D002 llama.cpp shim,
D003 Candle, D005 local-only planning, D016 evidence gating) are history: see git and
[candle-capabilities](research/archive/candle-capabilities.md).

| ID | Decision | Status |
| --- | --- | --- |
| D001 | Optimize agent latency (TTFT on repeated and branching turns, tool-loop turnaround), not chat tok/s | Accepted |
| D004 | Exact reuse only; lossy techniques opt-in, separately keyed and quality-gated | Accepted |
| D010 | Engine: Apple MLX through own `mlx-sys` over mlx-c, called from Rust | Accepted 2026-10-14 |
| D011 | mrun owns the runtime: scheduler, sequence state, prefix store, memory budget, APIs | Accepted |
| D012 | Memory budgeted by construction, not per-allocation hooks | Accepted |
| D013 | Specs, concepts and Spec Kit tooling are tracked in git | Accepted |
| D014 | Ship Anthropic Messages, OpenAI Chat Completions and stateless Responses together | Accepted 2026-10-14 |
| D015 | Models: MLX safetensors; default Qwen3.6-35B-A3B (hybrid), Qwen3-Coder-30B-A3B, Qwen3-8B reference | Revised 2026-10-15 |
| D017 | Prefix store: live path plus moved-out branches; checkpoints for hybrids; plain LRU | Revised 2026-10-15 |
| D018 | One active sequence; FIFO queue; batching only if traces show agents waiting on each other | Accepted 2026-10-15 |
| D019 | Contiguous per-layer attention arrays, no paging; state moves between live path and branches | Accepted 2026-10-15 |
| D020 | Resident model, never unloaded on an idle timer | Accepted 2026-10-15 |
| D021 | Fix client-side cache breakers on the server only when narrow and named (attribution nonce) | Accepted 2026-10-15 |

## D010: Engine is MLX

**Context.** Candle had no Metal quantized MoE path and an open Metal corruption bug past 8,192 KV
tokens. The strongest local competitors run on MLX.

**Decision.** Own `mlx-sys` (bindgen over vendored mlx-c v0.7.0, MLX v0.32.4, which fixes
deadlocks and a missing stream in `gather_qmm`) plus a thin safe layer. Model graphs are ports of
mlx-lm reference implementations, tested against them. Every mlx-c function the plan uses has been
checked to exist in v0.7.0.

## D012: Budget by construction

Per-allocation enforcement inside third-party code is not achievable
([experiment](research/archive/candle-budget/results.md)). mrun computes a budget at startup, measures
weights and scratch, sizes attention-state capacity from the remainder, admits requests by their
prompt, clamps output limits to what remains, keeps every allocation within the request's
reservation, and caps inputs so host-side work stays bounded. The reserve for other applications
defaults to one sixth of physical memory, between 2 and 8 GiB (8 GiB on a 48 GB Mac).

## D015: Models

- **Default: Qwen3.6-35B-A3B** (released 2026-04-14, Apache-2.0): 40 layers, 30 Gated DeltaNet and
  10 gated attention, MoE with 256 experts; vendor-reported 73.4% SWE-bench Verified against about
  51.6% for Qwen3-Coder-30B-A3B. Only 10 attention layers, so ≈20 KiB of attention state per token:
  its full 262K window fits a 48 GB Mac, five times Qwen3-Coder's usable context. The default is
  confirmed by `mrun bench` before release, and newer releases are re-checked then.
- **Qwen3-Coder-30B-A3B**: attention-only MoE; simpler cache behaviour; kept as an alternative.
- **Qwen3-8B**: dense reference for correctness tests and 16–24 GB Macs.

## D017: Prefix store

The live path is the active sequence state. Other conversations are branches whose state was moved
out when the live path switched; switching back moves it in again. Moves go one layer at a time, so
state is never held twice and long conversations survive subagents. Hybrid models resume from
checkpoints captured during prefill at message boundaries. Eviction is plain LRU over branch
leaves: two 2026 studies found LRU near-optimal for single long sessions, and turn-type or
session-priority tiers did not pay for themselves. Earlier ideas dropped in the third audit:
copy-based restore (capped conversations at about half of capacity), idle warm-up, session and
compaction hints, `cache_control` boundaries.

## D018: One active sequence

On a single Mac GPU decode is bandwidth-bound, so serial execution loses little and keeps latency
predictable and reuse exact. Parallel subagents queue (default 16 waiting). Batching returns as a
spec only if recorded traces show significant queueing.

## D019: Contiguous attention arrays

Per-layer contiguous arrays keep MLX's fused attention kernels on their fast path and make trims
free. The cost is that branches cannot share pages; moving tails instead of copying them keeps
memory at one copy per unique token.

## D020: Resident model

Unloading after idle (Ollama's default of 5 minutes) throws away the cache; an agent user who steps
away pays a cold reload and a full prefill. mrun keeps the model loaded and gives cache back under
memory pressure instead. Only weights and the active state are wired.

## D021: Narrow server-side fixes for client cache breakers

Claude Code adds a per-request nonce (`cch=`) to an attribution line at the top of the system
prompt; it defeats all reuse for users who start Claude Code without `mrun launch`. mrun replaces
only that value with a fixed one of the same length (as llama.cpp does). Broader prompt rewriting
is rejected: an unanchored version elsewhere truncated users' system prompts.
