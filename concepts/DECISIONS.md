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
| D015 | Models: an open set of families (block library + family modules + syntax registry); 0.1: Qwen3.6, Qwen3-Coder, gpt-oss (Qwen3-8B internal reference); Devstral 0.2 | Revised 2026-10-15 (lean 0.1) |
| D017 | Prefix store: live path plus moved-out branches; checkpoints for hybrids; plain LRU; five miss causes | Revised 2026-10-15 (lean 0.1) |
| D018 | One active sequence; FIFO queue; batching only if traces show agents waiting on each other | Accepted 2026-10-15 |
| D019 | Contiguous per-layer attention arrays, no paging; state moves between live path and branches | Accepted 2026-10-15 |
| D020 | Resident model, never unloaded on an idle timer | Accepted 2026-10-15 |
| D021 | Fix client-side cache breakers on the server only when narrow and named (attribution nonce) | Accepted 2026-10-15 |
| D022 | Login service loads the model lazily on the first request; `--preload` is opt-in | Deferred 2026-10-15 (no launchd service in 0.1, D025) |
| D023 | Persisted prefix entries bound to cache key + on-disk format + MLX version (not mrun version) | Accepted 2026-10-15 |
| D024 | Speculation ships n-gram/suffix drafting first; MTP and DFlash2 drafts gated on measured acceptance; sidecar never in the cache key | Revised 2026-10-15 |
| D025 | Lean release 0.1: three aliases, loopback only, five miss causes, three warnings, `connect` prints only, Homebrew tap with an ad-hoc-signed tarball; service, `/metrics`, notarization, SBOM, config file, regression mode deferred | Accepted 2026-10-15 |

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
  DeltaNet and later Mamba2; dense and mixture-of-experts feed-forward), plus an ordered set of
  accepted tool-call and reasoning syntaxes from a registry (Hermes JSON, Qwen XML, Harmony,
  Mistral, GLM XML). At 0.1 each family names exactly one syntax and load checks that the
  template renders its markers, else refuses (ordered syntax sets wait for a model that needs
  them). Each module is a port of the mlx-lm reference with a parity test. Unknown architectures
  are refused with the list of supported families. No plugin system: adding a family is one module
  and one test.
- **Release 0.1** (lean, D025) covers every 0.1 state kind with three aliases: `qwen3.6`
  (Qwen3.6-35B-A3B, hybrid linear attention; vendor-reported 73.4% SWE-bench Verified; ≈20 KiB per
  token, so its 262K window fits a 48 GB Mac), `qwen3-coder` (Qwen3-Coder-30B-A3B, full attention
  MoE) and `gpt-oss` (gpt-oss-20b, sliding window with sinks, Harmony format). Qwen3-8B (`qwen3`)
  is an internal dense test reference: parity and goldens only, no alias, no client or bench cells.
- **0.2** (spec 010): Devstral Small 2 (`mistral`, Mistral tool format). **Later**: GLM-4.7-Flash
  (MLA), Gemma 4 (sliding window + global), Nemotron-H (Mamba2 hybrid), then families by demand.
- **Default**: whichever 0.1 alias wins the agent suite of `mrun bench` on the M4 Pro (task in 005);
  Qwen3.6 is the expected winner. An alias that misses the client gate (002 SC-001) is dropped from
  the release; there are no tier labels in 0.1.

## D017: Prefix store

The live path is the active sequence state. Other conversations are branches whose state was moved
out when the live path switched; switching back moves it in again. Moves go one layer at a time, so
state is never held twice and long conversations survive subagents. Hybrid models resume from
checkpoints captured during prefill at message boundaries (at most 8 per path; the shallowest
other than the first is dropped). Eviction is plain LRU over branch leaves: two 2026 studies found
LRU near-optimal for single long sessions, and turn-type or session-priority tiers did not pay for
themselves. Misses report five causes (`new`, `changed`, `evicted`, `model`, `disabled`) plus the
divergence position, its message index and role, and the checkpoint gap; `template` re-renders are
a bench-side diagnostic. Dropped in the lean cut: soft pins for queued requests, `--debug-cache`,
the `x-mrun-cache` request header. Earlier ideas dropped in the third audit:
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

## D022: Lazy model load in the login service (deferred)

Deferred with the launchd service (D025); kept for the post-0.1 service. A login service that loads the default model claims ≈20 GB at every login, whether or not an agent
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

## D025: Lean release 0.1

**Context.** The 2026-10-15 implementability audit found 0.1 carrying five families, five client
gates, nine miss causes, eight warnings, a persistent-config writer, a login service, metrics and
notarization. Splash (VISION) raised the cost of a late release.

**Decision.** 0.1 ships what an agent user on one Mac needs and what mrun's differentiators rest
on (exact reuse with explained misses, fixed budget, client fidelity, fair benchmark):
- Families and aliases per D015; one syntax per family; Mistral parser and Devstral in 0.2.
- One client gate: each alias × Claude Code, Codex and OpenCode, started with `mrun launch`,
  completes the scripted task 3 of 3 times (002 SC-001). Pinned client versions run on every PR;
  the latest releases (Claude Code npm `stable` and `latest`) are checked by hand before a release.
- Loopback only; non-loopback binds with API keys later.
- Five miss causes (D017); three in-band warnings (`tool-args-invalid`, `max-tokens-clamped`,
  `image-omitted`), everything else logged on the server.
- `mrun connect` prints configuration only; `doctor` checks chip and budget, low-power mode, port,
  server reachability and client pitfalls; no config file (flags and `MRUN_*` only).
- Deferred tool loading and constrained decoding (llguidance) stay.
- Distribution: Homebrew tap and a CI-built, ad-hoc-signed tarball, `--version`, an idle-sleep
  assertion while requests run. Deferred: Developer ID signing and notarization, SBOM, launchd
  service with lazy load (D022), `/metrics`.
- Bench: competitors llama.cpp, Ollama, oMLX, mlx-lm and Splash, started by hand; regression mode,
  automated start/stop, thermal slope and RSS sampling later.

**Consequence.** Every cut keeps its spec ID and is marked "Removed (0.1 scope cut, 2026-10-15)"
or moved to a "Later" section.
