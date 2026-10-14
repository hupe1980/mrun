<!--
Sync Impact Report
Version: unratified scaffold -> 1.0.0 (initial project principles)
Replaced: five placeholder principles with six mrun principles.
Added: Target and Constraints; Development Workflow.
Removed: template examples only.
Deferred placeholders: none. No template source files changed.
Remove this temporary report before committing the constitution.
-->
# mrun Constitution

## Core Principles

### I. User Latency Before Language Prestige

mrun MUST optimize successful local coding-agent work, including repeated prompt processing,
time to first useful output, tool handoff, cancellation, and memory pressure. Rust is the runtime
language, not evidence of an inference speedup. Every performance claim MUST name its workload,
hardware, model, quality contract, comparator configuration, and reproducible evidence.

### II. Correctness Gates Every Optimization

An optimization MUST pass state, numerical, protocol, and task-quality checks appropriate to its
scope before becoming a default. Cache reuse MUST preserve exact token-prefix semantics and
compatible inference state. Approximate attention, KV quantization, and speculative decoding
MUST be explicitly qualified and reversible. Silent context truncation, stale state reuse,
invented usage counts, and concealed fallback are prohibited.

### III. Memory Is a Bounded Shared Resource

Weights, active inference state, retained prefixes, scratch space, queues, and streaming buffers
MUST have explicit ownership and finite budgets. Admission MUST reserve prompt and output growth
before execution. Cancellation and eviction MUST respect in-flight GPU lifetimes. Unified memory
MUST NOT be counted as independent CPU and GPU pools. Allocator accounting MUST be accompanied by
process and system measurements; it is not a proof of a hard process-RSS limit.

### IV. Local Privacy and Least Authority

Inference MUST work offline after explicit model provisioning. Default serving MUST bind only to
loopback; cloud fallback, telemetry uploads, remote model code, prompt logging, and persisted
conversation state MUST be disabled by default. Tool execution belongs to the client, not the
runtime. Cache access MUST be scoped to a trusted local principal; client-supplied cache or session
identifiers are not authorization. Network exposure requires an explicit authenticated policy.

### V. Reuse Engines, Own the Agent Contract

mrun MUST first qualify an existing optimized Apple Silicon engine against its requirements.
Native C/C++/Metal dependencies are permitted behind narrow Rust-owned boundaries. Backend types
MUST NOT leak into the public runtime contract. Model-specific state and templates belong to the
model/backend adapter, not HTTP handlers. Custom kernels require a profile showing the bottleneck,
a reference path, and a measured end-to-end win; speculative abstractions are not a substitute.

### VI. Reproducible Evidence and Honest Scope

Tests and benchmarks MUST use pinned, redistributable inputs or documented hash-verified downloads.
Comparisons MUST retain raw observations, failed runs, quality outcomes, and uncertainty, and MUST
enable documented competitor optimizations in the product comparison. Hardware-specific claims
MUST be qualified on that hardware. Unsupported model/backend combinations MUST fail explicitly.
"Best-in-class" MUST remain an aspiration until a dated comparison matrix supports a scoped claim.

## Target and Constraints

- Primary qualification target: Apple M4 Pro with 48 GB unified memory, one interactive local user.
	Exact CPU/GPU configuration, installed memory bytes, macOS, SDK, and power mode MUST be recorded.
- Initial scope: text generation for a small qualified coding-model set and OpenCode-compatible
	streaming and tool-call interchange. General Ollama parity, training, and remote execution are
	not initial requirements.
- Rust orchestration with Tokio/Axum is the intended serving direction. Backend, model artifact,
	quantization, deployment minimum, and Rust minimum MUST be decided in feature planning from
	compatibility and measurement evidence, not guessed in a feature specification.
- No automatic privileged OS tuning, wired-memory-limit changes, or model downloads on inference.
- Preserve the project's MIT/Apache-2.0 licensing intent; audit native dependencies, model licenses,
	and fixture redistribution separately before distribution.

## Development Workflow

Use the official project-local Spec Kit workflow: constitution, then specify, clarify when needed,
plan, tasks, analyze, implement, and converge. Specifications are living contracts; changes MUST
reconcile dependent plans, tasks, tests, and evidence. The project-wide spec index owns scope and
sequence; each numbered feature owns independently testable behavior and explicit requirement IDs.

Planning MUST resolve backend capabilities and numerical tolerances before implementation tasks
are accepted. Implementation MUST add regression tests for positive and rejected paths, especially
cache identity, cancellation, resource cleanup, and stream termination. Portable correctness checks
belong in ordinary CI; physical M4 Pro evidence belongs in explicit hardware qualification lanes.
Feature completion requires passing its acceptance gates, not merely generating tasks or compiling.

## Governance

This constitution governs plans and implementation. Each review MUST check compliance and record
any exception with rationale, scope, expiry, and mitigation. Amendments require maintainer approval,
a dated impact assessment, and reconciliation of affected artifacts. Version changes use semantic
versioning: major for incompatible principle changes, minor for new principles, patch for wording.
Numerical targets in feature specs are proposed acceptance gates, not measured accomplishments;
changing them after a failed run requires review and preservation of the original evidence.

**Version**: 1.0.0 | **Ratified**: 2026-10-13 | **Last Amended**: 2026-10-13
