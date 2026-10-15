---
description: "Tasks for 010 Model Families"
---

# Tasks: Model Families

**Input**: [plan.md](plan.md), [spec.md](spec.md), [research.md](research.md),
[data-model.md](data-model.md), [contracts/families.md](contracts/families.md)

## Format: `[ID] [P?] [Story] Description`

**Scope** (lean 0.1 cut, D025, 2026-10-15): post-0.1. The 0.1 families are built and gated by
other specs: blocks, registry and modules by 001 (T039–T042, T043a, T045, T046), syntaxes and the
template check by 002 (T006a, T006b, T014a), aliases and the pull gate by 004 (T005, T005a,
T013a), the client gate by 002 T016, the default by 005 T012a. Cut tasks keep their IDs. Every
new family follows the porting checklist (research R11): module → syntax → goldens → parity →
alias + fit → client gate (002 SC-001) → cache cases → bench cells; the locality rule is a review
rule.

## Phase 1: Setup

- [ ] T001 *Removed (0.1 scope cut, 2026-10-15)*: `scripts/check_family_diff.sh` and its CI job (FR-019)
- [ ] T002 [P] MF-1: full-KV vs ring state per family on 48/36/24 GB budgets (arithmetic now; allocation check on the target Mac when gpt-oss runs); record in research R3 (OQ-1 stays open until Gemma's turn)
- [ ] T003 Reconcile in the change that lands each piece: 001 FR-025/contract (per-layer K/V widths, `None` layers), 004 manifest (`syntax`, fit fields; `tier` when Tier 2 lands), ARCHITECTURE model table (measured numbers) and syntax list (Mistral, Gemma), D015, site `docs/models.md`

## Phase 2: Foundational

- [ ] T004 Before the first post-0.1 family: reshape 001's registry (`models/registry.rs`, a `Family` struct today) toward the [contract](contracts/families.md): `Family` methods, `StateLayout` per layer incl. differing K/V widths and `None`, `LoadError` set incl. `SyntaxMismatch`
- [ ] T005 [P] `scripts/gen_tiny_models.py` (mlx-lm, fixed seeds; quantized, unquantized and MXFP4 variants; every block option a config enables) and `crates/mrun-engine/tests/families.rs`: greedy ids for 32 steps and CPU logits for every registry key (FR-017, SC-006; with Tier 2)
- [ ] T006 [P] MF-2: window slice vs both references (temporal-order substituted cache; `KVCache` + full-depth mask) on gpt-oss shapes: bit agreement and speed at 16K/64K; record in research R3 (input to 001 T042/T043a, not a gate)
- [ ] T007 *Removed (0.1 scope cut, 2026-10-15)*: Hermes-under-`qwen3_moe` fixture for ordered syntax sets (OQ-7)
- [ ] T007a *Moved to Later (0.1 scope cut, 2026-10-15)*: tool-placement check (`template-tools-late`, FR-014)

## Phase 3: US1 + US2 + US3 — Tier 1 gate for the 0.1 families — Removed

- [ ] T008 *Removed (0.1 scope cut, 2026-10-15)*: gpt-oss checklist (owned by 001 T042/T043a/T027d/T010d, 002 T006b/T014a, 004 T005a, 003 T015a)
- [ ] T009 *Removed (0.1 scope cut, 2026-10-15)*: gpt-oss cache cases (003 T013, T015a)
- [ ] T010 *Moved to Phase 5 (Devstral, 0.2)*
- [ ] T011 *Removed (0.1 scope cut, 2026-10-15)*: Qwen3.6/Qwen3-Coder/Qwen3-8B checklist (001 SC-001, SC-009; 003 T013)
- [ ] T012 *Removed (0.1 scope cut, 2026-10-15)*: client smoke for all families (one gate: 002 SC-001, 002 T016)
- [ ] T013 *Removed (0.1 scope cut, 2026-10-15)*: bench cells per model (005 T012a)

## Phase 4: US4 — Clear answers for unsupported models (P2)

- [ ] T014 [US4] Tier 2: manifest `tier: 2` on pull of a non-alias repo, `mrun show` label, refusal fixtures end to end (unknown pair, unsupported mode, unknown tensor, template-check miss) through `run`, `serve` and `pull` (with 004 T013a) (SC-006)

## Phase 5: US5 — Post-0.1 families (P2)

Block changes land first, separately, with block tests; the family change stays local (review).

### Devstral Small 2 (`mistral`, Mistral syntax), 0.2

- [ ] T010 [US1] Module `models/mistral.rs` (`mistral3`/`ministral3` config, sanitize dropping vision tower and projector, YaRN + position-scaled queries, dense SwiGLU; refused when `tokenizer.json` or a template is missing; 001 R17 design), parity (001 SC-001 extended, reference from 001 T043a's script), `chat/syntax/mistral.rs` per 002 FR-034 (Later section), goldens vs mistral-common and the repo template (OQ-3 decides Tier 1), alias `devstral` with fit numbers (004 research R4), cache cases (003 SC-005), client gate (002 SC-001)

### GLM-4.7-Flash (`glm4_moe_lite`, MLA, GLM XML)

- [ ] T015 [US5] Blocks: `blocks/mla.rs` (latent + RoPE-key storage, expanded prefill, absorbed decode, `embed_q`/`unembed_out` sanitize split), sigmoid-with-bias group-limited router with routed scaling, plain shared expert; block tests vs mlx-lm; MF-5 speed
- [ ] T016 [US5] `chat/syntax/glm_xml.rs`: name until first `<arg_key>`, value end at `</arg_value>` followed by `<arg_key>` or `</tool_call>`, schema strings raw, non-strings JSON only (no `literal_eval`), `<think>` reasoning, history `clear_thinking=false`; parser suite (SC-004)
- [ ] T017 [US5] Module `models/glm4_moe_lite.rs` (first dense layers, MTP layer ignored), goldens, parity, alias `glm-4.7-flash` (OQ-6), cache cases, client gate, bench

### Gemma 4 26B-A4B (`gemma4` + `gemma4_text`, Gemma syntax)

- [ ] T018 [US5] Decide OQ-1 from MF-1 on the target Mac (full KV ≈ 55K capacity vs window ring as position-bound state); record in research and ARCHITECTURE before porting
- [ ] T019 [US5] Blocks: shared-KV layers (`StateLayout::None`), K=V, v-norm without scale, proportional RoPE, GeGLU, logit soft-cap, norm-scaled router with per-expert scale; block tests
- [ ] T020 [US5] `chat/syntax/gemma.rs`: `<|tool_call>call:NAME{…}<tool_call|>`, `<|"|>` strings, bare keys, balanced-brace end, no coercion, buffered per call, thought channel (verify markers on the pinned template); parser suite
- [ ] T021 [US5] Module `models/gemma4.rs` (per-layer input embeddings, vision dropped), goldens, parity (reference per OQ-2 outcome), alias `gemma4`, cache cases incl. p mod 1024 ∈ {0, 1, 1023}, client gate, bench

### Nemotron-H (`nemotron_h`, Mamba2 hybrid, Qwen XML)

- [ ] T022 [US5] Confirm model and dimensions from the pinned config (OQ-5)
- [ ] T023 [US5] Blocks: `blocks/mamba2.rs` (`ssm_attn` prefill, mlx-lm `ssm_update` Metal kernel verbatim + ops reference, gated RMSNorm, `time_step_limit`), relu² `SwitchMLP` with optional latent projection; checkpoint arrays via `StateLayout::PositionBound`; kernel-vs-ops test on macOS
- [ ] T024 [US5] Module `models/nemotron_h.rs` (pattern `M`/`*`/`-`/`E`, MTP dropped), Qwen XML via the template check, goldens, parity, alias `nemotron-3-nano`, hybrid cache cases (restores from every checkpoint kind), client gate, bench

## Phase 6: Polish

- [ ] T025 [P] MF-3: size of Harmony and GLM `template` re-renders on recorded traces (005 FR-017 counts); propose or reject a narrow rendering fix under D021's rule
- [ ] T026 [P] Site page `site/content/docs/models.md`: families, syntaxes, state bytes and capacity per model, what an unsupported-architecture error means

## Dependencies

- 001: T004 before the first Phase 5 family; T006 with 001 T043a. 002: T010 needs 002's Mistral
  parser (FR-034, Later). 003: cache cases extend 003 T013. 004: aliases extend 004 T005. 005:
  bench cells need the harness; T025 needs 005 FR-017.
- After 0.1: T004 → T010 (0.2) → each later family after MF-6 and its block task; T014 with Tier 2;
  T025–T026 last.

## Parallel opportunities

T002, T005 and T006 touch separate files. Post-0.1 block tasks (T015, T019, T023) can run in
parallel.
