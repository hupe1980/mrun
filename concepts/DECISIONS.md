# Design Decisions

Stable concepts identifiers record rationale, not task completion. Never reuse or renumber them;
mark superseded decisions and name their replacements. Research D-01 through D-08 are separate
feature-local records; concepts identifiers use D001 format.

## D001: Optimize Coding-Agent Outcomes

**Basis**: Original product request and [product contract](../specs/README.md).
Own repeated-context latency, correct client handoff, safe interruption and shared-memory behavior.
Do not substitute isolated token throughput or language prestige for successful tasks. Separate
runtime/tool clocks and independently test state, numerical and task-quality outcomes.

## D002: Reuse Engines Before New Kernels

**Basis**: [Constitution principle V](../.specify/memory/constitution.md#v-reuse-engines-own-the-agent-contract).
Rust owns orchestration and a narrow backend boundary. Engine replacement or new kernels require
profiling, reference correctness and measured end-to-end benefit. This is not a permanent C++
mandate: Rust-native engines are eligible when evidence favors them.

## D003: Treat the Initial Adapter as Conditional

**Basis**: [Plan](../specs/001-agent-runtime/plan.md) and
[native stop conditions](../specs/001-agent-runtime/contracts/backend.md#stop-conditions).
The plan selects a pinned llama.cpp/Metal spike, not a measured winner. Prove required controls
before production work. Failure requires revised design, not unsafe shims, hidden fallback,
weaker requirements or an automatic second production adapter.

## D004: Separate State Correctness from Speed

**Basis**: [Feature requirements](../specs/001-agent-runtime/spec.md).
Match exact rendered tokens and complete identity, keep shared prefixes immutable, and reset
request-local sampler/grammar/RNG state. Invalidate uncertain native state. Exact-reference token
replay, precision-dependent numerics and coding-task outcomes establish different properties.

## D005: Keep Planning Local

**Basis**: User-directed convention matching billing, sepa and agentplane.
Ignore concepts, numbered specifications, Spec Kit machinery and generated skills while leaving
source and normal GitHub workflows trackable. Public artifacts cannot cite ignored paths.
Local backup/provisioning is explicit: Git does not preserve or restore these notes.

## Unaccepted Product Choice

Whether the first feature must beat strong optimized competitors remains under clarification.
Neither baseline completion nor a new competitor-win gate has been accepted in that discussion.
Current specification gates remain authoritative; do not record a decision by inference.