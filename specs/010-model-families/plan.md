# Implementation Plan: Model Families

**Branch**: `010-model-families` | **Date**: 2026-10-15 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `specs/010-model-families/spec.md`

## Summary

Define the family mechanism that 001, 002 and 004 implement for release 0.1 (block library,
`model_type` registry, per-layer state layout, MXFP4 next to MLX affine, syntax registry with a
template probe, aliases), and own what lies beyond it: the family contract and locality rule,
tiers (Tier 2 loading of any repo of a registered architecture, tiny-model parity per
architecture), the Tier 1 gate across all families, and the post-0.1 families. 0.1 families are
built by 001 (T039–T043a, T045, T046 (+ goldens T010d, parity T027d/e): `qwen3`, `qwen3_moe`, `qwen3_5_moe`, `gpt_oss`, `mistral`), their
syntaxes by 002 (T006a–T006c: Hermes, Qwen XML, Harmony, Mistral), their aliases by 004. After 0.1,
each of `glm4_moe_lite` (MLA), `gemma4` (sliding + global, KV sharing, Gemma syntax) and
`nemotron_h` (Mamba2 hybrid) lands through the same porting checklist, gated by its measure-first
check. Sliding-window layers keep full KV and read a window slice, so every attention-like state
stays positional and 003 needs no per-family logic.

## Technical Context

**Language/Version**: Rust 1.94.0
**Primary Dependencies**: none new; `mlx-sys` (mlx-c v0.7.0 / MLX v0.32.4) with `mode` on
`quantized_matmul`/`gather_qmm` and `sinks` on SDPA (research R6); `minijinja` gets `strftime_now`
**Storage**: Reference outputs and tiny weights under `crates/mrun-engine/tests/families/`;
template goldens under `crates/mrun/tests/templates/<family>/`
**Testing**: Linux CI: tiny random-weight parity per registered architecture on the MLX CPU
backend, parser suites, probe, locality check, layout arithmetic. macOS (`#[ignore]`,
`MRUN_TEST_MODELS`): real-weight parity, cache equality per family, Metal kernels vs ops, bench
cells
**Target Platform**: macOS 15+ Apple Silicon; Linux for portable tests
**Project Type**: Modules of `mrun-engine` (models, state, loader) and `mrun` (chat syntax, hub)
**Performance Goals**: ≥ 90% of mlx-lm decode and 16K prefill per Tier 1 model (SC-005); sliding
layers O(W + L) per chunk
**Constraints**: Parity with mlx-lm `ba8a08f`; no family logic outside its module and syntax
(Principle V); no new kernels beyond mlx-lm's own (Mamba2 decode kernel ported verbatim, like GDN)
**Scale/Scope**: 5 Tier 1 models at 0.1 (5 architectures), 3 after; 6 syntaxes

## Constitution Check

| Principle | Gate | Status |
| --- | --- | --- |
| I Agent latency | SC-005 speed gates per Tier 1 model; window slicing keeps sliding layers O(W); bench cells per family | Pass |
| II Exactness | Greedy parity per family (SC-001); cache equality per state kind incl. off-window restores (SC-003); full KV keeps sliding layers exactly cuttable; parsers never coerce (SC-004) | Pass |
| III Memory budget | Per-layer state layout feeds 001 budget and admission; full-KV cost measured per family (MF-1, FR-009); Gemma decision gated (OQ-1) | Pass |
| IV Local, private | No network; probe and goldens offline; identity unchanged | Pass |
| V Own runtime, reuse kernels | Blocks are mlx-lm ports with parity tests; Mamba2 `ssm_update` kernel is mlx-lm's source ported verbatim (reference kernel, not new); family logic confined to module + syntax, enforced by the locality check | Pass |
| VI Simple | Compile-time registry, no plugin system; options enter blocks only when shared or layout-relevant; unsupported modes refused instead of half-supported | Pass |

Post-design re-check: Pass.

## Project Structure

### Documentation (this feature)

```text
specs/010-model-families/
├── spec.md  plan.md  research.md  data-model.md  quickstart.md  tasks.md
├── contracts/families.md
└── checklists/requirements.md
```

### Source Code (repository root)

```text
crates/mrun-engine/src/
├── engine.rs                 # ModelInfo: family, syntax, per-layer StateLayout (FR-005)
├── load.rs                   # config pair → registry; QuantPlan (affine, mxfp4); pre-load tensor check
├── state.rs                  # allocation per StateLayout (positional widths per layer, windows, none)
└── models/
    ├── mod.rs  registry.rs   # Family trait, lookup, supported()
    ├── blocks/{attention.rs, mla.rs, gated_delta.rs, mamba2.rs, ffn.rs, moe.rs, norm.rs, rope.rs, head.rs}
    ├── qwen3.rs  qwen3_moe.rs  qwen3_5_moe.rs  gpt_oss.rs  mistral.rs   # 0.1 (built by 001)
    └── glm4_moe_lite.rs  gemma4.rs  nemotron_h.rs    # after 0.1, each via the checklist
crates/mrun-engine/src/kernels/{gated_delta.metal.inc, ssm_update.metal.inc}
crates/mrun-engine/tests/families/<family>/           # tiny configs, weights, references
crates/mrun/src/chat/syntax/{mod.rs, probe.rs, holdback.rs, hermes.rs, qwen_xml.rs, harmony.rs, mistral.rs, glm_xml.rs, gemma.rs}
crates/mrun/src/hub/aliases.rs                        # Tier 1 entries (004 format + family fields)
crates/mrun/tests/{templates/<family>/, syntax/<syntax>/}
scripts/{gen_reference.py, gen_tiny_models.py, check_family_diff.sh}
```

**Structure Decision**: Blocks and modules live in `mrun-engine` beside the state they read;
syntaxes live in `mrun::chat::syntax` (002 FR-032) beside the renderer. 010 adds files to these
directories; it creates no parallel structure.

## Relation to other specs

- **001**: FR-002 and FR-023–FR-025 build the 0.1 families on blocks and the registry (T039–T043a, T045, T046 (+ goldens T010d, parity T027d/e)).
  010's contract adds per-layer K/V widths and `None` layers to 001 FR-025's layer kinds (needed
  for MLA and Gemma's shared-KV layers, post-0.1). Reconciled by T003.
- **002**: FR-032–FR-035 own the syntax registry, probe and 0.1 syntaxes; 010 adds GLM XML and
  Gemma under FR-004's rules; 002 FR-032 already holds the ordered set (OQ-7 decided).
- **003**: No change; relies on state kinds through the engine contract. SC-003 adds window cases.
- **004**: FR-002 owns the 0.1 aliases (`gpt-oss`, `devstral`, …) and FR-014 the pull gate; 010
  adds post-0.1 aliases; the `tier`, `syntax` and fit fields are in 004's manifest. Reconciled by T003.
- **005**: One synthetic and one agent cell set per Tier 1 model.
- **ARCHITECTURE / D015**: the syntax list gains Gemma when Gemma 4 lands; the model table gets
  measured numbers.

## Complexity Tracking

| Item | Why needed | Simpler alternative rejected because |
| --- | --- | --- |
| Per-layer `StateLayout` | MLA (different K/V widths), shared-KV layers (none), windows | Uniform `kv_heads × head_dim` cannot describe GLM or Gemma |
| Window slice instead of full-depth mask | Keeps sliding layers O(W + L) per chunk with full KV | Full-depth masked SDPA makes half of gpt-oss's layers cost like full attention at depth |
| Accepted-syntax set + template probe | One architecture ships different syntaxes | A single syntax per family refuses Qwen3-30B-A3B-Instruct-2507 (Hermes) under `qwen3_moe`; substring inference matches comments |
| Mamba2 decode Metal kernel (mlx-lm's) | Parity and speed for Nemotron-H decode | The ops path (`ssm_attn`) is not what mlx-lm runs at L = 1, so parity would fail |
