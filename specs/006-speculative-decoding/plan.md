# Implementation Plan: Speculative Decoding

**Branch**: `006-speculative-decoding` | **Date**: 2026-10-15 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `specs/006-speculative-decoding/spec.md`

## Summary

Add lossless, target-verified speculation to the decode loop with one drafter: a copy-aware
suffix index over the prompt, its line-number-stripped tool-result rendering and the output (all
models). A per-model controller picks the depth (0 = plain pipelined decode) from measured
acceptance and verify cost (analytic expert-union prior refined by an online EWMA). The engine
verifies N+1 positions in one forward, accepts on device (greedy or point-proposal rejection
sampling, ending at the first accepted EOS), and rolls back rejected positions before the round
ends (attention: trim; hybrids: recompute the linear-attention recurrence from the pre-round state
over the accepted positions from cached per-position inputs), so only committed tokens reach the
stream and the prefix store. Gated on `mrun bench --suite spec`. MTP drafting is a gated follow-up
(spec "Follow-up (gated)").

## Technical Context

**Language/Version**: Rust 1.94.0 (edition 2024)
**Primary Dependencies**: existing `mlx-sys` (no new kernels; the hybrid rollback reruns the
existing GDN recurrence over cached inputs), `tokenizers` for renderings
**Storage**: none (controller state in memory only)
**Testing**: cargo test on `ScriptedEngine` (deterministic drafts and acceptance, mutation suite,
SC-004); sampler statistics test (SC-005) on CPU MLX; macOS conformance and bench cells (SC-001–003)
**Target Platform**: macOS Apple Silicon; Linux builds and runs the scripted tests
**Project Type**: Engine and runtime change inside the single `mrun` binary
**Performance Goals**: SC-001 (no cell slower than `off` beyond the 005 regression threshold), SC-002 (≥ 1.3× decode on edit-heavy
traces); index build ≤ 10 ms per 100K tokens off the engine thread; controller decision < 20 µs
**Constraints**: Verify scratch and the hybrid rollback input cache (≤ 32 MiB at N = 64) inside the measured scratch envelope;
drafts never exceed the reservation; cancellation within one round
**Scale/Scope**: Prompts up to model capacity (262K on Qwen3.6); drafts ≤ 64

## Constitution Check

| Principle | Gate | Status |
| --- | --- | --- |
| I Agent latency | Decode speed on edit-heavy turns; no cell regresses beyond the 005 threshold; claims from `mrun bench` | Pass |
| II Exactness | Target-verified; greedy within chunked-prefill tolerance (SC-003); rejection sampling preserves the distribution (SC-005); rejected tokens never committed (SC-004) | Pass |
| III Memory budget | Verify scratch and rollback input cache in the measured scratch envelope; drafts capped by reservation | Pass |
| IV Local, private | No network access; no persisted speculation state | Pass |
| V Own runtime | No new kernels; MTP follow-up names its reference implementation (Ollama MLX runner, vLLM cross-check) | Pass |
| VI Simple | One controller, one drafter, one mode switch; trained drafters (MTP, DFlash2) only as gated follow-ups | Pass |

Post-design re-check: Pass.

## Project Structure

### Documentation (this feature)

```text
specs/006-speculative-decoding/
├── plan.md  research.md  data-model.md  quickstart.md  tasks.md
├── contracts/speculation.md
└── checklists/requirements.md
```

### Source Code (repository root)

```text
crates/mrun-engine/src/
├── engine.rs              # + forward_draft(), accept(), commit(), SpecCaps, DraftStep, VerifyOutcome (contract)
├── state.rs               # + pre-round recurrent state, per-position GDN input cache, recurrence recompute
├── sample.rs              # + on-device greedy acceptance, point-proposal rejection sampling, EOS stop
└── budget.rs              # + rollback input cache in the scratch envelope, max_draft
crates/mrun/src/
├── spec/
│   ├── mod.rs             # SpecMode, per-request policy (grammar/seed/logprobs/limit → off)
│   ├── index.rs           # suffix automaton over token ids, incremental extension
│   ├── render.rs          # copy-aware rendering (line numbers stripped)
│   ├── controller.rs      # acceptance stats, cost prior + EWMA, depth choice, probing
│   └── stats.rs           # SpecStats for metadata and /mrun/status
├── runtime/generate.rs    # decode loop: rounds, one-at-a-time emission through stop/parsers
├── api/…                  # + spec fields in mrun metadata, x-mrun-speculation header
└── bench/suites/spec.rs   # offline acceptance simulation + online cells
crates/mrun/tests/
├── spec_mutations.rs      # ≥ 100 cases on ScriptedEngine (SC-004)
├── spec_controller.rs     # depth 0 on no-repeat text, collapse/back-off, bucket selection
└── spec_index.rs          # matches, renderings, most-recent tie-break
crates/mrun-engine/tests/
├── spec_sampling.rs       # chi-square test at a fixed draft-covered position, tiny model (SC-005), CPU
└── spec_conformance.rs    # macOS: greedy auto vs off (SC-003)
```

**Structure Decision**: Drafting and control are runtime policy in `mrun` (testable on Linux with
the scripted engine); forward, acceptance and rollback are engine primitives in `mrun-engine`,
reached only through the `Engine` trait.

## Cross-spec changes (to fold into the owning specs)

- 001 engine contract: `forward_draft`, `accept`, `commit`, `spec_caps`, `DraftStep`, `ProposalId`;
  additive `EngineError::{Cancelled, Unsupported}` (see [contract](contracts/speculation.md)).
- 002 metadata: `spec` object; request header `x-mrun-speculation`; `/mrun/status` key `spec`.
- 003: exactness tests run with `--speculation off` (FR-017).
- 004 CLI contract: `serve|run --speculation auto|off|ngram`; `run --check` forces `off`; config key `[speculation] mode` only once 004 FR-010 returns.
- 005: suite `spec`, edit-heavy trace classification, result fields (additive, schema v1);
  `--suite agent` comparisons with `--speculation off`.
- 008 metrics (Later since the lean 0.1 cut): `mrun_spec_*` counters (label `drafter` ∈ {`ngram`}) once `/metrics` lands.
- Folded by T024.

## Complexity Tracking

None.
