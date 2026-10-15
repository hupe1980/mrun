# Decisions

Architecture decision records. IDs are stable. Superseded early records (D002 llama.cpp shim, D006–D009 retired engine and process notes,
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
| D015 | Models: an open set of families (block library + family modules + syntax registry); 0.1: Qwen3.6, Qwen3-Coder, gpt-oss, Devstral, Qwen3-8B reference | Revised 2026-10-15 |
| D017 | Prefix store: live path plus moved-out branches; checkpoints for hybrids; plain LRU | Revised 2026-10-15 |
| D018 | One active sequence; FIFO queue; batching only if traces show agents waiting on each other | Accepted 2026-10-15 |
| D019 | Contiguous per-layer attention arrays, no paging; state moves between live path and branches | Accepted 2026-10-15 |
| D020 | Resident model, never unloaded on an idle timer | Accepted 2026-10-15 |
| D021 | Fix client-side cache breakers on the server only when narrow and named (attribution nonce) | Accepted 2026-10-15 |
| D022 | Login service loads the model lazily on the first request; `--preload` is opt-in | Accepted 2026-10-15 |
| D023 | Persisted prefix entries bound to cache key + on-disk format + MLX version (not mrun version) | Accepted 2026-10-15 |
| D024 | Speculation ships n-gram/suffix drafting first; MTP gated on measured acceptance; sidecar never in the cache key | Accepted 2026-10-15 |

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

- **Open architecture.** mrun supports model *families*, not one model. A family is a small module
  that composes shared blocks (full GQA, sliding-window with sinks and MLA attention; Gated
  DeltaNet and later Mamba2; dense and mixture-of-experts feed-forward), plus a tool-call and
  reasoning syntax from a registry (Hermes JSON, Qwen XML, Harmony, Mistral, GLM XML). Each module
  is a port of the mlx-lm reference with a parity test. Unknown architectures are refused with the
  list of supported families. No plugin system: adding a family is one module and one test.
- **Release 0.1** covers every state kind and the most used tool formats: Qwen3.6-35B-A3B (hybrid
  linear attention; vendor-reported 73.4% SWE-bench Verified; ≈20 KiB per token, so its 262K
  window fits a 48 GB Mac), Qwen3-Coder-30B-A3B (full attention MoE), gpt-oss-20b (sliding window
  with sinks, Harmony format), Devstral Small 2 (dense, Mistral tool format), and Qwen3-8B as the
  dense correctness reference for small Macs.
- **After 0.1** (spec 010): GLM-4.7-Flash (MLA), Gemma 4 (sliding window + global), Nemotron-H
  (Mamba2 hybrid), then families by demand.
- **Default**: whichever 0.1 model wins the agent suite of `mrun bench` on the M4 Pro (task in 005);
  Qwen3.6 is the expected winner.

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

## D022: Lazy model load in the login service

A login service that loads the default model claims ≈20 GB at every login, whether or not an agent
runs that day. `serve --service` (008) creates the engine on the first dequeued request, sends
keepalives or progress to waiting requests while loading, and reports `loaded` in `/health`; the
model then stays resident (D020). Before load, capacity comes from the cached probe or the 004 fit
estimate marked `estimated`, and model and budget metrics are absent (`mrun_model_loaded` 0).
`mrun service install --preload` loads at login instead.

## D023: Persistent tier binding

007 entries are eligible only when the 003 cache key (weights, tokenizer, template, state format),
the on-disk format version and the vendored MLX version all match. Binding to the full mrun version
would discard the tier on every patch release; the MLX version is what can change recompute bits
and layouts. Entries are never converted.

## D024: Speculation scope

006 ships copy-aware n-gram/suffix speculation only: agent edits copy from context, and drafts
cost no model memory. MTP drafting on Qwen3.6 proceeds only after its per-position acceptance is
measured offline with an existing implementation (Ollama's MLX runner MTP module, vLLM as a
cross-check) and shows α₁ ≥ 0.8 and a modeled speed-up ≥ 1.15×. If it ships, the MTP sidecar is
recorded in the manifest and shown by `mrun show`, but never enters the model identity or cache
key: the head does not change target state.
