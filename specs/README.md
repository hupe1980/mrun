# mrun: Spec of Specs

**Created**: 2026-10-13
**Status**: Proposed product baseline; no implementation or measured results
**Owner**: mrun maintainers

## Product Thesis

Build a best-in-class local coding-agent runtime for Apple Silicon, starting with an M4 Pro
with 48 GB unified memory. The product succeeds when a developer completes correct coding tasks
with less waiting and predictable memory use. Rust enables ownership and orchestration control;
inference kernels, memory traffic, model quality, and repeated-context work determine the gains.

The primary optimization unit is the complete agent turn and tool loop, not an isolated tokens/s
number. Prefix reuse, incremental prefill, bounded state, rapid interruption, and warm residency
are the first investment. Decode acceleration follows profiling, not language preference.

## Audience and Scope

Primary user: a developer running OpenCode locally, with long repository instructions, tool schemas,
and a growing conversation. One foreground generation is the initial default. Competing local
requests remain bounded and observable rather than maximizing aggregate throughput.

Initial support is deliberately narrow: text-only models with qualified templates and tool support.
The runtime emits tool calls; the client owns permissions and execution. Offline inference is
required after explicit provisioning. Do not imply that local serving makes client-side tools safe.

Out of initial scope: complete Ollama API/registry parity, an integrated coding agent, web UI,
training, distributed inference, CUDA optimization, universal model support, and automatic cloud
fallback. Other platforms may receive portable correctness tests without performance certification.

## Product Requirements

- **PR-001**: A developer MUST connect an unmodified, pinned OpenCode client using its documented
  local provider configuration and complete streaming text and tool-result continuations.
- **PR-002**: Compatible repeated token prefixes MUST avoid unnecessary recomputation without
  changing conversation meaning or sharing state across untrusted principals.
- **PR-003**: Foreground requests MUST have bounded waiting, cancellation, and output buffering;
  abandoned work MUST release capacity without corrupting reusable state.
- **PR-004**: Model loading and context admission MUST account for total weights, inference state,
  scratch, retained cache, and output growth within one unified-memory budget.
- **PR-005**: Users MUST see truthful supported-model, context, capability, and fallback information.
  Unsupported input MUST fail explicitly rather than silently degrade or truncate.
- **PR-006**: All claimed improvements MUST be reproducible against pinned, appropriately optimized
  comparators and MUST preserve a separately tested task-quality contract.
- **PR-007**: Default runtime behavior MUST remain offline, loopback-only, and free of prompt/state
  persistence, remote code execution, or telemetry export.
- **PR-008**: Later durable-state reuse MUST be optional, compatible, recoverable, access-controlled,
  and demonstrably faster than recomputation for its admitted workloads.
- **PR-009**: Backend or kernel replacements MUST earn adoption through capability qualification,
  numerical/state tests, and measured end-to-end benefit on the target hardware.
- **PR-010**: Releases MUST publish a bounded support matrix, reproducible build/dependency identity,
  license notices, and correctness plus physical-hardware qualification evidence.

## Feature Sequence

Only 001 is specified now. Later identifiers below are reserved roadmap slots, not implemented
features, generated tasks, or promises that all are needed. Reorder by evidence with documented
dependencies; do not create placeholder feature directories.

| Slot | Feature | Priority and dependency | Acceptance direction | Product coverage |
| --- | --- | --- | --- | --- |
| 001 | [Prefix-aware interactive agent runtime](001-agent-runtime/spec.md) | P1; first vertical slice | Real client loop, exact reuse, bounded memory, cancellation, measured baseline | PR-001, PR-002, PR-003, PR-004, PR-005, PR-006, PR-007 |
| 002 | Durable conversation checkpoints | P2; 001 state correctness | Opt-in restore/branch, incompatible/corrupt-state rejection, bounded disk, restore-vs-prefill win | PR-002, PR-007, PR-008 |
| 003 | Workload-aware scheduling | P2; 001 latency evidence | Chunked prefill fairness, foreground p95 protection, no hidden throughput tradeoff | PR-003, PR-004, PR-006 |
| 004 | Quality-preserving decode acceleration | Research; 001 profiling | Speculation/ngram/MTP, KV formats, fusion only where qualified quality and task-time wins exist | PR-006, PR-009 |
| 005 | Backend/model expansion | Evidence-gated; 001 adapter contract | Pinned artifacts, templates, state semantics, architecture-specific regression suite | PR-005, PR-009 |
| 006 | Distribution and operational hardening | Before public runtime release; 001 | Reproducible macOS package, supply-chain review, observability, soak and qualification gates | PR-007, PR-010 |

Benchmarking is mandatory evidence inside every performance feature, not an optional last milestone.
001 can be delivered as an evaluated prototype without declaring public-release hardening complete.

## Architecture Direction

Rust owns request validation, lifecycle, admission, cancellation, bounded streaming, and observability.
Tokio and Axum are the intended serving components. A narrow backend adapter owns tokenizer/template
execution, prefill/decode, model state, and GPU-safe cleanup. Reuse a mature engine first.

Candidate evaluation MUST include llama.cpp/Metal and mistral.rs. Evaluate MLX-based engines as
serious Apple Silicon comparators. Evaluate uzu's conversion/integration cost and Imparo's qualified
model/state support; neither is selected solely because it uses Rust. Choose one initial backend,
not a universal plugin platform. A subprocess adapter may be a baseline harness but is not proof
that fine-grained state, memory, or cancellation control is available.

Custom Metal kernels require a profile showing a material inference bottleneck, reference numerics,
safe fallback, and an end-to-end improvement larger than uncertainty. Metal 4 availability does not
guarantee a gain on an M4 Pro or remove SDK/OS requirements. Avoid new kernel work before qualification.

## Performance and Quality Policy

001's proposed numerical gates and measurement definitions live in its
[feature contract](001-agent-runtime/spec.md) and
[benchmark protocol](001-agent-runtime/benchmark-protocol.md). Do not duplicate thresholds here.
No benchmark has been run. Targets MUST NOT be described as achieved results.

A "best-in-class" claim requires a dated named-comparator matrix on physical M4 Pro hardware,
including current Ollama, llama.cpp, and qualified MLX/Rust-native contenders. The claim MUST name
its model, quantization, context, cache regime, workload, quality, memory, and power conditions.
If a competitor wins a cell, retain it. If artifacts differ, call the result a quality-matched
product comparison, not a same-weights kernel comparison. No universal superiority claim is allowed.

Inspect numerical/state equivalence separately from code-task success. Valid JSON is not necessarily
a correct tool choice. Active MoE parameters are not the total resident weight size. The model's
advertised maximum context is not the runtime's supported memory-safe context.

## Artifact Ownership

- [.specify/memory/constitution.md](../.specify/memory/constitution.md) owns governance.
- This file owns product scope, product requirement IDs, feature sequence, and cross-feature gates.
- Each `specs/NNN-name/spec.md` owns user scenarios, `FR-*`, `SC-*`, scope, and assumptions.
- Feature research owns source provenance and provisional technical decisions, not requirements.
- Benchmark protocols own metric definitions, comparability rules, and experiment procedure.
- Future plans own technical commitments and resolved capability spikes; tasks own implementation
  work linked to requirements. A task checkbox is not evidence of performance acceptance.
- Feature quality checklists review requirements, not implementation completion.

Conflicts MUST be resolved explicitly: governance controls engineering constraints; the index
controls product scope; the numbered feature controls its acceptance behavior. Reconcile all
affected artifacts instead of silently treating newer code as the specification.

## Spec Lifecycle and Gates

1. Specify a bounded user outcome with explicit requirements, edge cases, and measurable gates.
2. Clarify material uncertainties; document conservative defaults. Pin capability evidence in plan.
3. Plan with a constitution check, backend qualification, model identities, contracts, and test design.
4. Generate traceable tasks, analyze cross-artifact coverage, then implement with regression tests.
5. Run portable correctness gates and separately run required physical M4 Pro evidence.
6. Converge only after implementation. Missing hardware evidence means not performance-qualified.
7. Maintain specs as living contracts; amend scope or targets through review, retain superseded evidence.

## Decisions Required During Planning

These are technical qualification work, not unanswered user requirements:

- Pin the initial engine and Rust adapter version after proving prefix/state control, bounded
  memory reporting, GPU cancellation cleanup, tools, and offline loading on the M4 Pro.
- Pin a small dense tool-capable model and a coding-specialist model if it fits. Starting candidates
  are Qwen3-8B and Qwen3-Coder-30B-A3B-Instruct; do not mistake this for a model quality endorsement.
- Record hashes for exact converted artifacts, tokenizer, template, and model license. Resolve
  numerical tolerances per model/precision before the first optimization comparison.
- Record macOS/SDK and Rust minimums actually supported by the chosen engine. Inspect the user's
  exact M4 Pro configuration rather than assuming GPU cores or the usable GPU working set.
- Freeze benchmark fixtures, client version, power conditions, schema subset, and admission policy
  before running acceptance experiments. Tighten or amend draft gates only with review.