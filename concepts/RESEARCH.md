# Research Ownership and Evidence Limits

**Reviewed**: 2026-10-13. This organizes existing research, not a new external retrieval,
literature survey, native experiment or independent benchmark reproduction.

## Source Owners

| Question | Authoritative record |
| --- | --- |
| Engines and documented capabilities | [Candidate matrix](../specs/001-agent-runtime/research.md#backend-and-comparator-matrix) |
| Backend/toolchain/artifact/client pins | [Phase 0 decisions](../specs/001-agent-runtime/research.md#phase-0-decisions-2026-10-13) |
| Model dimensions and memory reasoning | [Sizing review](../specs/001-agent-runtime/research.md#model-candidates-and-memory-reasoning) |
| Native sharing and synchronization | [Native contract](../specs/001-agent-runtime/contracts/backend.md) |
| Pinned client fields and schemas | [HTTP contract](../specs/001-agent-runtime/contracts/http.md) |
| Numerical tolerances | [Numerical decision](../specs/001-agent-runtime/research.md#d-07-numerical-and-quality-contract) |
| Comparative measurement method | [Benchmark protocol](../specs/001-agent-runtime/benchmark-protocol.md) |

Keep immutable revisions, hashes and retrieval observations in those records. Concepts should
link to evidence rather than create another manually maintained dependency manifest.

## Findings That Shape the Design

Optimized native inference already underlies existing products. Porting orchestration does not
establish faster kernels. Prefix reuse is useful but not new. Shared memory still requires
physical-backing accounting, transient allowances and pressure observation. Tool grammars constrain
structure, not correct tool selection or safe execution.

Reviewed native state APIs support a sharing hypothesis, but source inspection cannot prove branch
isolation, numeric parity, cancellation latency or clean unload. Batch shape affects numerics;
a fixed seed does not establish bit-identical outputs. A recommended GPU working set is neither
a hard RSS cap nor a measurement of system availability.

## Changing a Decision

An alternative needs capability evidence, compatible task quality and measured user benefit before
adoption. A failed initial candidate is a valid result. Reconcile research, plan, contracts and
tasks when a backend changes instead of silently retaining its old qualification identity.

Language, custom kernels and acceleration methods are means to outcomes. A full engine rewrite
requires separately reviewed scope and reference behavior. The baseline-versus-competitor-win
question is a product clarification, not something upstream documentation can answer; the user
has not yet accepted either option.