# mrun Architecture Concepts

Internal design notes, not published documentation or evidence of implemented behavior.
This folder, feature specifications and local Spec Kit tooling are gitignored and excluded from
the crate package. Public documentation, source, tests and examples must stand on their own:
they must not link to or require files available only in these internal directories.

## Direction and Evidence

Build a local Rust coding-agent runtime for Apple Silicon, initially an M4 Pro with 48 GB unified
memory. Optimize correct task completion, repeated-context latency, cancellation and bounded
memory, not the language of an existing serving layer. The current crate is a dependency-free
placeholder; there is no inference runtime or measured performance evidence yet.

The planned llama.cpp/Metal integration is an initial candidate, not a proven winner or permanent
engine mandate. Rust owns the serving/state/resource boundary. Existing Rust-native and MLX engines
remain serious comparison candidates. Custom inference work requires a demonstrated bottleneck,
reference correctness and a measured end-to-end benefit. A best-in-class claim requires a scoped,
dated optimized-comparator matrix; cache-off self-comparisons cannot establish leadership.
The first-feature completion bar is under clarification; no unaccepted option is recorded as decided.

## Reading Order

| Document | Owns |
| --- | --- |
| [OVERVIEW.md](OVERVIEW.md) | Product boundary, user value and meaning of best-in-class |
| [ARCHITECTURE.md](ARCHITECTURE.md) | Intended layers, ownership, state and failure boundaries |
| [MARKET_LANDSCAPE.md](MARKET_LANDSCAPE.md) | Candidate roles and defensible differentiation |
| [RESEARCH.md](RESEARCH.md) | Evidence ownership, findings and source-review limits |
| [DECISIONS.md](DECISIONS.md) | Stable rationale and explicitly unaccepted choices |
| [RISKS.md](RISKS.md) | Failure modes, mitigation and discharge evidence |
| [VERIFICATION.md](VERIFICATION.md) | Evidence layers and limits of performance/correctness claims |
| [ROADMAP.md](ROADMAP.md) | Route into spec-owned work, not a second backlog |

## Feature Artifacts

| Document | Owns |
| --- | --- |
| [Spec of specs](../specs/README.md) | Product boundary, roadmap, product requirements and claim policy |
| [Constitution](../.specify/memory/constitution.md) | Engineering principles and governance |
| [First feature](../specs/001-agent-runtime/spec.md) | User outcomes, requirements and proposed acceptance gates |
| [Research](../specs/001-agent-runtime/research.md) | Source provenance, technical choices and qualification limits |
| [Implementation plan](../specs/001-agent-runtime/plan.md) | Intended implementation and capability stop conditions |
| [Data model](../specs/001-agent-runtime/data-model.md) | Ownership, lifecycle, identity and resource budgets |
| [HTTP contract](../specs/001-agent-runtime/contracts/http.md) | Client fields, tools, streaming and errors |
| [CLI contract](../specs/001-agent-runtime/contracts/cli.md) | Local controls and qualification-report authorization |
| [Native contract](../specs/001-agent-runtime/contracts/backend.md) | Backend state, synchronization and safety boundary |
| [Qualification catalog](../specs/001-agent-runtime/contracts/qualification.md) | Distinct state/client cases and quality workloads |
| [Benchmark protocol](../specs/001-agent-runtime/benchmark-protocol.md) | Measurement cohorts, comparisons and evidence disposition |
| [Validation guide](../specs/001-agent-runtime/quickstart.md) | Planned validation workflow; no working inference commands yet |

## Authority and Maintenance

The constitution governs plans. The spec of specs owns roadmap sequencing; numbered specs own
requirements, plans own implementation choices, and tasks own executable work. This index does not
create a second backlog or duplicate acceptance thresholds. Preserve the active feature directory
and existing links; do not relocate specifications merely to make them internal.

Design is not implementation. A checked task is not a performance measurement. Hardware support,
numerical parity, memory safety and comparative speed require identified tests and raw evidence.
Record disagreement and reconcile the owning documents rather than conceal failed qualification.
Keep private prompts, weights and raw unreviewed evidence out of published material.

## Local Spec Kit Workflow

Official Spec Kit was initialized with specify-cli 1.1.3, Copilot skills and Bash scripts.
Generated templates/skills remain project-local; no Git extension is installed. Active feature
routing uses [.specify/feature.json](../.specify/feature.json), independently of Git branch names.

```text
/speckit-specify
/speckit-clarify
/speckit-plan
/speckit-tasks
/speckit-analyze
/speckit-implement
/speckit-converge
```

Clarify outcomes before reconciling downstream plans/tasks. Capability spikes precede production
runtime work; convergence is meaningful only after implementation and required verification.
Use the existing feature rather than regenerate the whole roadmap for every question.

From the repository root, verify CLI availability with `specify check` and inspect active paths:

```sh
.specify/scripts/bash/check-prerequisites.sh --json --paths-only
```

These documents stay on this workstation: .gitignore does not distribute or back them up, and a
fresh clone will not contain the planning workflow. It must be explicitly restored/provisioned
for local planning; normal package builds must not require it.