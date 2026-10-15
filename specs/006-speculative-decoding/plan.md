# Implementation Plan: Speculative Decoding

**Branch**: `006-speculative-decoding` | **Date**: 2026-10-15 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `specs/006-speculative-decoding/spec.md`

## Summary

Add lossless, target-verified speculation to the decode loop. Two drafters: a copy-aware suffix
index over the prompt, its tool-result renderings and the output (all models), and the MTP head of
Qwen3.6-35B-A3B (fetched as a verified, locally quantized sidecar). A per-model controller picks
drafter and depth (0 = plain pipelined decode) from measured acceptance and per-depth verify cost.
The engine verifies N+1 positions in one forward, accepts on device (greedy or rejection sampling),
and rolls back rejected positions (trim, or recurrent snapshot plus replay on hybrids) before the
round ends, so only committed tokens reach the stream and the prefix store. Gated on `mrun bench
--suite spec`.

## Technical Context

**Language/Version**: Rust 1.94.0 (edition 2024)
**Primary Dependencies**: existing `mlx-sys` (no new kernels; the GDN kernel gains split offsets
already supported by segmenting its call), `tokenizers` for renderings, `reqwest` (range requests
in `pull`), `safetensors` header parsing (hand-written, 8-byte length + JSON)
**Storage**: MTP sidecar in `~/.mrun/sidecars/<base-identity>/mtp.safetensors`; controller
calibration in `~/.mrun/probe.toml` (001's probe cache); nothing else persisted
**Testing**: cargo test on `ScriptedEngine` (deterministic drafts and acceptance, mutation suite,
SC-004); sampler statistics test (SC-005) on CPU MLX; macOS conformance and bench cells (SC-001–003)
**Target Platform**: macOS Apple Silicon; Linux builds and runs the scripted tests
**Project Type**: Engine and runtime change inside the single `mrun` binary
**Performance Goals**: SC-001 (≤ 3% slower on any cell), SC-002 (≥ 1.3× decode on edit-heavy
traces); index build ≤ 10 ms per 100K tokens off the engine thread; controller decision < 20 µs
**Constraints**: Verify scratch and recurrent snapshots inside the measured scratch envelope;
drafts never exceed the reservation; cancellation within one round
**Scale/Scope**: Prompts up to model capacity (262K on Qwen3.6); drafts ≤ 64 (n-gram), ≤ 4 (MTP)

## Constitution Check

| Principle | Gate | Status |
| --- | --- | --- |
| I Agent latency | Decode speed on edit-heavy turns; no cell regresses > 3%; claims from `mrun bench` | Pass |
| II Exactness | Target-verified; greedy within chunked-prefill tolerance (SC-003); rejection sampling preserves the distribution (SC-005); rejected tokens never committed (SC-004) | Pass |
| III Memory budget | MTP weights and window measured at load; verify scratch and snapshots in the probe envelope; drafts capped by reservation | Pass |
| IV Local, private | Only `pull` fetches MTP tensors, pinned and hash-verified; no persisted speculation state besides the probe cache | Pass |
| V Own runtime | MTP head ported from the Qwen3.5/3.6 reference layout and tested against a reference forward; no new kernels | Pass |
| VI Simple | One controller, two drafters, one mode switch; no trained drafters | Pass |

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
├── engine.rs              # + verify(), draft_mtp(), SpecCaps, VerifyOutcome (contract)
├── models/qwen3_5_moe.rs  # + optional MtpHead (fc, norms, one attention+MoE layer, window ring)
├── models/mtp.rs          # MTP head forward, window ring buffer
├── state.rs               # + in-round recurrent snapshots at split offsets, restore + replay
├── sample.rs              # + on-device greedy acceptance, rejection sampling, residual sampling
├── budget.rs              # + verify widths in the scratch probe, cost calibration cache
└── load.rs                # + sidecar loading and identity component
crates/mrun/src/
├── spec/
│   ├── mod.rs             # SpecMode, per-request policy (grammar/seed/limit → off)
│   ├── index.rs           # suffix automaton over token ids, incremental extension
│   ├── render.rs          # copy-aware renderings (line numbers stripped, JSON-escaped)
│   ├── controller.rs      # acceptance stats, cost curves, depth choice, probing
│   └── stats.rs           # SpecStats for metadata and /mrun/status
├── runtime/generate.rs    # decode loop: rounds, one-at-a-time emission through stop/parsers
├── hub/mtp.rs             # range fetch of mtp.* tensors, verify, quantize, sidecar write
├── hub/aliases.rs         # + MTP source (repo, revision, tensor hashes) for qwen3.6
├── api/…                  # + spec fields in mrun metadata, x-mrun-speculation header
└── bench/suites/spec.rs   # offline acceptance simulation + online cells
crates/mrun/tests/
├── spec_mutations.rs      # ≥ 100 cases on ScriptedEngine (SC-004)
├── spec_controller.rs     # depth 0 on no-repeat text, collapse/back-off, bucket selection
└── spec_index.rs          # matches, renderings, most-recent tie-break
crates/mrun-engine/tests/
├── spec_sampling.rs       # chi-square distribution test (SC-005), CPU
└── spec_conformance.rs    # macOS: greedy auto vs off (SC-003)
```

**Structure Decision**: Drafting and control are runtime policy in `mrun` (testable on Linux with
the scripted engine); forward, acceptance and rollback are engine primitives in `mrun-engine`,
reached only through the `Engine` trait.

## Cross-spec changes (to fold into the owning specs)

- 001 engine contract: `verify`, `draft_mtp`, `spec_caps` (see [contract](contracts/speculation.md)).
- 002 metadata: `spec` object; request header `x-mrun-speculation`.
- 004 CLI contract: `serve --speculation`, `pull --no-mtp`; `show` lists the sidecar; config key.
- 005: suite `spec`, edit-heavy trace classification, result fields (additive, schema v1).

## Complexity Tracking

None.
