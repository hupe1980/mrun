# mrun Constitution

## Core Principles

### I. Agent Latency Is the Product

mrun MUST optimize what a coding agent waits for: time to first useful token on long, repeated
contexts, tool-loop turnaround, cancellation, and steady decode speed. Every performance claim
MUST name hardware, model, quantization, workload and comparator settings, and MUST be reproducible
with the in-repo benchmark harness. "Best-in-class" is a dated, scoped benchmark result, never a
label.

### II. Exactness Before Speed

Reused inference state MUST be exact: same model, tokenizer, template, KV format and token prefix.
Hash lookups MUST be confirmed by token comparison. Approximate techniques (KV eviction, KV
quantization, speculative decoding, approximate prefix matching) MUST be opt-in, separately
identified in cache keys, and gated by a measured quality check. Silent context truncation,
stale-state reuse and invented usage counts are prohibited.

### III. Memory Is Budgeted by Construction

Large memory (weights, KV arena, prefix cache) MUST be allocated or reserved by mrun against an
explicit budget computed at startup, and admission MUST reserve a request's full prompt plus output
KV before execution. Transient scratch MUST be bounded by measured envelopes from a worst-case
startup profile plus fixed input caps, not by per-allocation hooks into third-party code. Unified
memory MUST NOT be double-counted as CPU plus GPU. Memory pressure MUST shed cache before failing
requests, and failures MUST be explicit.

### IV. Local, Private, Least Authority

Inference MUST work offline after an explicit `mrun pull`. Serving MUST bind to loopback by
default. No telemetry, prompt logging, remote code, or persisted conversation state by default.
Tools run in the client, never in mrun. Models are identified by content hash, never by path or
name alone.

### V. Own the Runtime, Reuse the Kernels

mrun owns scheduling, KV state, memory, APIs and model execution order, built on Apple MLX arrays
and kernels called from Rust. Model graphs are ports of reference implementations (mlx-lm) and are
tested against them. New GPU kernels or forks of upstream libraries require a profile showing the
bottleneck, a reference implementation for correctness tests, and a measured end-to-end win.
Prefer small, upstreamable patches. Engine types MUST NOT leak into the public API.

### VI. Simple Over Clever, Developer Experience First

One binary, sensible defaults, zero required configuration. Clear errors that say what to do next.
Prefer the smallest design that meets the requirement; remove code, options and abstractions that
do not serve a current requirement. No speculative plugin frameworks or multi-backend layers.

## Target and Constraints

- Primary target: Apple Silicon Macs with Metal, qualified first on an M4 Pro with 48 GB unified
  memory. Linux/CPU builds MUST compile and run portable tests; they are not performance targets.
- Rust stable (pinned in `rust-toolchain.toml`), Tokio + Axum for serving, MLX for tensors.
- API compatibility is a product feature: OpenAI Chat Completions, Anthropic Messages and
  stateless OpenAI Responses.
- No automatic privileged OS tuning (for example raising the GPU wired-memory limit).
- Licensing: MIT OR Apache-2.0. Model licenses are shown to the user at pull time.

## Development Workflow

Spec Kit drives features: specify, clarify when needed, plan, tasks, analyze, implement. The spec
index (`specs/README.md`) owns the roadmap and feature order. Each feature owns testable
requirements. Tests ship with the change: positive and rejected paths for cache identity,
admission, cancellation and stream termination. CI runs fmt, clippy and tests on Linux; Metal
correctness and performance run on Apple hardware via `mrun bench` and documented commands.
A feature is done when its acceptance checks pass, not when its tasks are ticked.

## Governance

This constitution governs specs, plans and code reviews. Exceptions MUST be written in the plan
with scope and expiry. Amendments update this file with a semantic version bump (major: principle
removed or redefined; minor: principle added; patch: wording) and reconcile affected specs.
Numerical targets in specs are acceptance gates; changing one after a failed run requires a note
in the spec explaining why.

**Version**: 3.0.0 | **Ratified**: 2026-10-13 | **Last Amended**: 2026-10-14
