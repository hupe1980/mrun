# Implementation Plan: Prefix-Aware Interactive Agent Runtime

**Current revision**: 2026-10-14, Candle replan; feature `001-agent-runtime`, actual Git branch `main`.
**Current status**: ERROR: Phase 0 research has unresolved gates; Phase 1 replacement is blocked.
**Authority**: [spec.md](spec.md), including authorization to select modern Rust.

## Current Direction

Candle is the core tensor library for mrun's own inference runtime. mistral.rs is a competitor,
benchmark comparator and learning reference only; it is not an embedded runtime or dependency.
The llama.cpp implementation and design below are historical, not the selected production direction.
Existing compiled foundations and T001..T017 results do not qualify Candle or complete this replan.

## Current Technical Decisions

- **Language/Version**: Rust 1.94.0, edition 2024, is the new planning target and intended minimum.
  Rust 1.85 is superseded. A locked build on 1.94.0 must verify the selected Candle dependency graph;
  source inspection establishes incompatibility with 1.85, not a successful build on 1.94.0.
  Runtime manifests/toolchain migration belongs to implementation after the revised design passes.
- **Tensor foundation**: Evaluate Candle revision `c68b24997319b8d76e5a4e775dab197c7f26322a`
  (workspace version 0.11.0); this is a research candidate, not an accepted dependency pin.
- **Ownership candidate**: Immutable weights, request-local per-layer KV, independently backed frozen
  prompt snapshots, and one serialized GPU owner. Adapt existing Qwen3 model operations; no new kernels.
  Stock model `Clone` and `Tensor::copy()` are not a branch-isolation contract on Metal.
- **Model/profile candidate**: Preserve the official Qwen3-8B Q4_K_M artifact identity from D-02,
  nonthinking, text-only, 16,384 total tokens and at most 2,048 output. Existing FP16 KV and unified
  sequence geometry are not Candle capabilities. Start sizing the reviewed GGUF route as F32 KV;
  dtype, retained-byte budget and bounded prefill must be frozen from a complete allocation design.
- **External contract**: Retain the pinned OpenCode chat route, local privacy, finite HTTP limits,
  tool ownership and all FR-001..FR-023 / SC-001..SC-009 requirements. Tokenizer, template rendering,
  constrained sampling and tool parsing use the candidate stack in research C-05; old native helpers
  are not inherited. Its work/byte-bound enforcement is not established, so the stack is not qualified.
- **Resource policy**: Preserve FR-015, unique-backing accounting and fail-closed missing observations.
  The maintainer approved public-counter-derived admission on 2026-10-14, as specified in research
  C-04. OS severity remains unknown until observed; derived admission is not an invented OS NORMAL.
  Enforced Candle and host-library allocation bounds remain unresolved.
  Neither an observed peak nor the old 2 GiB/25% scratch heuristic proves bounded execution.
- **SC-006 baseline**: The maintainer approved a full-chat standalone Candle same-core direct-serving
  harness on 2026-10-14. Research C-06 freezes matched-path requirements; thresholds are unchanged.
  This is not a pretokenized decode microbenchmark and does not replace optimized engine comparisons.
- **Evidence**: No weights downloaded, Candle build, inference, memory fit, numerical parity,
  cancellation or physical performance qualification performed. Existing backend-bound reports and
  rendered fixtures cannot transfer their eligibility to a changed backend/build/profile.

## Current Constitution Gate

| Principle | Current replan disposition |
| --- | --- |
| I: User latency | Preserved: all latency/comparator gates remain; no measured improvement claimed |
| II: Correctness | Pending design: prove copied backing isolation, interior-prefix restore and numerical parity |
| III: Shared memory | ERROR: pressure policy is approved, but pre-allocation enforcement for Candle and host libraries is not established |
| IV: Local privacy | Preserved: offline, loopback, client-owned tools, no private content diagnostics |
| V: Existing engines | APPROVED EXCEPTION: maintainer authorized Candle model/operator reuse with an mrun-owned runtime on 2026-10-14; scope, expiry and mitigation below |
| VI: Honest evidence | Preserved: pins are candidates, failures remain visible and hardware gates are unrun |

**Approved Principle V exception**: The maintainer explicitly approved the scoped exception on
2026-10-14. Scope: `001-agent-runtime` may reuse Candle's existing model operators while owning
the inference runtime, rather than first integrating/qualifying a complete existing engine.
Rationale: the selected product direction requires ownership of exact state, scheduling and budgets.
Expiry: feature completion or a change of backend direction, whichever comes first; later features
must re-review it. Mitigation: retain matched and optimized competitor comparisons, all correctness,
memory and cancellation gates, private backend types, and the exclusion of new custom kernels.
mistral.rs remains comparator/reference only. This exception does not amend any FR/SC or authorize
production serving, and no other constitutional exception is approved.

The previous PASS table below applies only to the superseded design. There is no post-Phase-1 PASS.

## Current Planning Status

Phase 0 source findings and alternatives are recorded in [research.md](research.md#candle-replan-research-2026-10-14).
Modern Rust, the scoped Principle V exception, derived pressure policy and same-core baseline are
resolved planning choices. The remaining blocker is enforceable allocation/work bounds, not an
unanswered preference question. A future patch must be pinned and reviewed for complete coverage;
the plan cannot treat "implement budget hooks" as evidence that the required capability exists.
Phase 1 is not regenerated while those gates are open. The existing [data model](data-model.md),
[backend contract](contracts/backend.md), [quickstart](quickstart.md), qualification dependencies
and [tasks](tasks.md) require reconciliation before they can guide Candle implementation.
Their old KV cells, sequence IDs, FP16 assumptions, CMake/native helpers and task ordering are not
Candle requirements. Do not implement the superseded backend tasks or run the old guide as a
Candle qualification procedure. No tasks are regenerated by this planning command.

## Resume Conditions

1. Preserve the approved Principle V exception without embedding mistral.rs or authorizing new kernels.
2. Carry the approved derived-pressure policy into the replacement observation contract and tests;
  do not relabel the old missing-signal test as passed or fabricate an OS severity sample.
3. Establish enforceable load/forward/snapshot/staging/pool and tokenizer/template/grammar/parser
  allocation/work bounds. Review and pin the patch and compatible dependency/profile identities.
4. Replace dependent design/contracts/validation guide, recheck the constitution, then regenerate
   and analyze tasks before implementation. Locked toolchain and physical qualification remain
   execution gates, not results of planning.

## Superseded llama.cpp Plan (Historical)

Everything below is preserved historical context, including original status, decisions and test
claims. It is not the active backend plan. The current sections above override it.

**Feature**: `001-agent-runtime` | **Date**: 2026-10-13 | **Spec**: [spec.md](spec.md)
**Git branch**: `main`; no feature branch created. Spec Kit reports the feature identifier as BRANCH.
**Status**: Design/tasks analyzed; T001..T017 validated within their stated scope.
Implementation halted at T018 on 2026-10-14: the required `os_proc_available_memory`
is unavailable on macOS and its SDK semantics differ from the planned OS capacity signal.
See [implementation blocker](research.md#implementation-blocker-mandatory-macos-availability-signal).
No model inference, adapter feasibility or empirical SC qualification has run.
**Input**: [Feature specification](spec.md), governed by the [constitution](../../.specify/memory/constitution.md).

## Summary

Build a Rust-owned, offline, loopback-only coding-agent runtime over one existing Metal engine.
Own exact prefix identity, bounded admission, truthful streams, cancellation, and GPU-safe cleanup.
Native inference performance is not attributed to Rust orchestration. Qualify correctness before
enabling reuse; retain failures and measure the nine specification success criteria separately.

## Technical Context

**Language/Version**: Rust 1.85.0 minimum and initial toolchain, edition 2024; reviewed lock must pass MSRV.
**Primary Dependencies**: Tokio 1.x, Axum 0.8, Tower 0.5, Serde/serde_json 1.x, Clap 4.5,
tracing 0.1, sha2 0.10, jsonschema 0.33; one llama.cpp/Metal adapter at
`45451c366d4256f94fbc8845645e31a6b9155054` behind a private C ABI/C++ shim.
**Storage**: Read-only provisioned model artifacts; volatile prefix state; opt-in public-fixture evidence.
**Testing**: Portable deterministic backend tests and HTTP contracts; explicit physical M4 Pro qualification.
**Target Platform**: Native arm64 M4 Pro, 48 GB unified memory; macOS 15.0 product minimum,
macOS 15+ SDK, Apple Clang C11/C++17 and CMake 3.31.6. No privileged OS tuning.
**Project Type**: Local CLI and OpenAI-compatible chat-completions service, not a tool executor.
**Performance Goals**: SC-001 through SC-009 in [spec.md](spec.md); proposed gates, not achievements.
**Constraints**: One active model/generation; eight queued requests; 30-second queue deadline;
10-second stalled-write deadline; 16,384 total tokens; 2,048 output tokens; in-memory reuse only.
**Scale/Scope**: One trusted OS user; full-attention text-only Qwen3-8B Q4_K_M, official GGUF revision
`7c41481f57cb95916b40956ab2f0b139b296d974`; SHA-256 and size pinned in research D-02.
Larger coding models, thinking, batching, speculation and persistence are deferred.
**Integration**: OpenCode v1.19.0 at `add9b4ba5adfb895934cb1bf46f590342cec68d2`, explicit local
OpenAI-compatible provider. Current native chat protocol reviewed alongside retained SDK 2.0.41.
See [field/schema matrix](contracts/http.md); no Responses API or universal parity claim.
**State Safety**: Unified shared immutable KV, 32768 cells, 5 sequence IDs, 4 retained entries;
128-token synchronized prefill, single-token decode; request-local sampler/grammar; mandatory
final-token boundary replay. Exact CPU and proposed Metal tolerance lanes in research D-07.
**Resource Limits**: [Finite defaults and pressure policy](data-model.md): 64 KiB streaming,
256 KiB generated payload, bounded schemas/render/grammar, OS/device observations, unique backing
ledger and independently sampled footprint. Accounted budgets are not hard RSS guarantees.
**Support Gate**: Candidate-binary capability evidence authorizes cache-off serving. Reuse additionally
requires state/numerical, protocol and SC-007 task-quality passes for the same binary/profile.
Adapter-spike evidence authorizes continued development only; rebuilt binaries require fresh evidence.
Artifact verification and completed planning cannot authorize untested support.

## Constitution Check

| Principle | Pre-research design gate | Post-design review |
| --- | --- | --- |
| I: User latency | PASS: agent turns and first useful output, not language-only claims | PASS: separate useful/generated/tool clocks and strong comparison lanes |
| II: Correctness | PASS: exact token identity; reuse stays gated on qualification | PASS: immutable sharing, full identity, reset sampler, protocol/state/numerical/task-quality gates before default reuse |
| III: Shared memory | PASS: unified ownership, reservation, finite queues and cleanup required | PASS: observations, admission, preallocation leases and pressure handling precede any production serving |
| IV: Local privacy | PASS: loopback, offline, one OS principal, no tool execution or private logs | PASS: local-only controls, no content diagnostics/persistence, traffic audit |
| V: Existing engines | PASS: one existing Metal backend behind Rust-owned boundary | PASS: one pinned backend; no speculative adapter framework or custom kernels |
| VI: Honest evidence | PASS: portable tests separate from hardware evidence; failed gates retained | PASS: pins, named catalogs, raw failures and explicit unrun qualification |

No constitutional exceptions requested. These passes describe the revised design, not the old task
ordering or experimental results. Implementation remains blocked until tasks are regenerated and
reanalyzed against this design. Empirical qualification is not a completed planning gate.

## Phase 0: Research Decisions

Nine choices are resolved in [research.md](research.md), each with decisions, rationale and
alternatives. Backend research was delegated; the client research agent returned no report, so
the client path/versions were verified directly from immutable upstream source. No backend spike,
model fit, client run or speed result is claimed. An unsupported required capability is a stop
condition, not permission to substitute an HTTP wrapper or weaken the specification.
The D-09 revision is a delegated local contract/governance review, not new upstream research or a
native experiment; D-01 through D-08 pins, limits and numerical tolerances remain unchanged.

## Phase 1: Design

Design artifacts:

- [Data model](data-model.md): entities, ownership, state machines and finite resource policy.
- [HTTP contract](contracts/http.md): request/schema matrix, streaming, usage, errors and controls.
- [CLI contract](contracts/cli.md): provisioned-model verification, serving, safe controls and report gate.
- [Native contract](contracts/backend.md): private C/C++ ownership, KV geometry, synchronization/failure.
- [Qualification catalog](contracts/qualification.md): 20 client cases, 100 state cases, coding/replay lanes.
- [Quickstart](quickstart.md): prerequisites and future runnable validation, clearly labeled unimplemented.

### Control Flow

HTTP validates bounded input and schema/history, reserves queue bytes and enters FIFO. The single
native owner coalesces model loading, verifies a fit lease, renders/tokenizes full history, checks
context/output demand, attaches the longest exact eligible prefix, prefills unmatched/boundary
tokens, and publishes only synchronized immutable prompt state. It then samples/parses with fresh
request-local state and sends byte-leased output through the bounded bridge. Disconnect, pressure,
timeout, overflow or native failure triggers one cleanup arbiter; native storage remains alive until
submitted work completes. Streams never label interrupted tools or uncertain native state successful.

### Requirement Ownership

| Requirements | Owning design and validation |
| --- | --- |
| FR-001, FR-018 | CLI/model manifest/native capability gate; offline and unsupported-profile cases |
| FR-002, FR-003, FR-004, FR-005 | HTTP/schema/template boundary; IC-001..020 and captured client tools |
| FR-006, FR-007, FR-008, FR-009, FR-010 | Prefix identity/membership/epochs; P/B/I/S/F catalog |
| FR-011, FR-012, FR-013 | Scheduler/cancellation/byte bridge; queue and 400 cancellation trials |
| FR-014, FR-015, FR-016, FR-017 | Unique ledger/OS pressure/model lifecycle; fit, churn/unload gates |
| FR-019, FR-020, FR-021 | Sanitized evidence, token accounting and independent memory/traffic probes |
| FR-022, FR-023 | Frozen protocol/comparator outcomes and native failure invalidation |

### Validation Sequence and Stop Rules

1. Prepare licensed/hash-verified draft fixture inputs, assertions and seeds. After the native renderer
  exists, seal rendered tokens and full identity; only sealed inputs may enter dependent evidence runs.
2. Run adapter spikes on the pinned native build/model: sharing, boundary logits, geometry, grammar,
  preallocation accounting, fault reporting, unload and physical M4 Pro cancellation. These authorize
  continued implementation only; no production service may accept the spike report.
3. Before the first production inference, implement independent OS/device observations, the automatic
  budget/delta policy, preallocation leases, prompt/output reservation, pressure handling, bounded
  queues/output and safe cancellation. A ledger type or successful spike cannot replace these controls.
4. Build the actual runtime candidate, seal its inputs and execute its own serving capability suite.
  The report binds runtime SHA-256, harness/build identities and model/backend/profile; only this
  report authorizes cache-off serving. Rebuilds invalidate it; requalify before each changed binary
  serves. Startup checks evidence, not destructive experiments on every unchanged invocation.
5. Use the qualification-only candidate path to test reuse and run the complete state/numerical,
  IC-001..020 protocol and SC-007 60-pair task-quality gates before authorizing production/default
  reuse. Its public-fixture loopback endpoint uses the same runtime controls but is available only
  inside the selected harness run; it is not a product flag, arbitrary-prompt mode or force-reuse bypass.
6. Run timing, cancellation, memory churn, replay and matched/optimized comparator lanes on the frozen
  candidate. Preserve failures/inconclusive results and require all SC-001..009 for feature completion.

### Analysis Remediation and Task Handoff

| Finding | Design resolution | Regenerated task handoff (analysis pending) |
| --- | --- | --- |
| C1 | Admission and memory safety precede every production inference | T018..T022 foundation policies precede US1 owner/scheduler integration and T053 execution; exhaustive US4 evidence remains later |
| C2 | Reuse eligibility includes protocol and SC-007 task-quality evidence | T068 runner precedes T070..T072 matching candidate/protocol/state/quality evidence and T073 reuse eligibility |
| I1 | Adapter spikes and exact runtime-binary reports are different evidence stages | T051 refresh machinery runs in T053, T070, T085, T093, T102 and T106 before changed-target serving/evidence |
| A1 | Draft fixtures and renderer-derived sealed fixtures are separate states | T029 seals after verified rendering; T055 seals actual client revisions and refreshes affected evidence |

The regenerated [tasks.md](tasks.md) remains blocked for implementation until `/speckit-analyze`
checks the revised execution order. Task generation completed no implementation or empirical gates.
Base evidence persistence, independent clocks and quality runners are early qualification prerequisites,
not polish work needed after the runs they record. Task generation must make these dependencies acyclic.

The post-design constitution check passes at the design level. All nine empirical success criteria
remain **blocked: not run**. Capability spikes can falsify this choice; a failed capability requires
revised research/design before production tasks. Fixture binary hashes, dependency patch lock,
binary/SDK builds, observed fit bounds and comparator configurations are implementation/evidence
deliverables, not unspecified product behavior or fabricated qualification. No best-in-class claim.

## Project Structure

Planned structure, not existing runtime code:

```text
Cargo.toml
Cargo.lock
rust-toolchain.toml
src/
  main.rs
  lib.rs
  config.rs
  api/
  runtime/
  cache/
  budget/
  backend/
    mod.rs
    llama.rs
  telemetry/
native/
  shim/
  llama.cpp/
examples/
  qualify.rs
tests/
  fixtures/
  support/
  contract.rs
  state.rs
  lifecycle.rs
  privacy.rs
  hardware.rs
  governance.rs
specs/001-agent-runtime/
  plan.md
  research.md
  benchmark-protocol.md
  spec.md
  data-model.md
  quickstart.md
  contracts/
    http.md
    cli.md
    backend.md
    qualification.md
```

**Structure Decision**: One Rust package with a narrow native boundary; no speculative adapter
framework, frontend, or separate services. Public types do not expose native handles. tasks.md
belongs to the later task-generation workflow and is not produced by this command.
Default features contain no native backend; portable tests use deterministic test-only adapters,
never an externally exposed fake model. `metal` enables the macOS native route. A shared qualification
harness backs hardware tests/example execution; fixture checks also run from a clean packaged tree.
Required implementation gates include format, Clippy, locked MSRV checks, all-target feature checks,
portable contracts/governance and explicit hardware acceptance. Native source provisioning must
be documented for clean builds/packages; no dependency on an ignored local checkout is acceptable.

## Complexity Tracking

No constitution violations. Native inference is an allowed dependency, not a Rust-only exception.