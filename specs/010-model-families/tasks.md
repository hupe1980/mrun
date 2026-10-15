---
description: "Tasks for 010 Model Families"
---

# Tasks: Model Families

**Input**: [plan.md](plan.md), [spec.md](spec.md), [research.md](research.md),
[data-model.md](data-model.md), [contracts/families.md](contracts/families.md)

## Format: `[ID] [P?] [Story] Description`

0.1 families are built by other specs: blocks, registry and modules by 001 (T039–T043a, T045, T046 (+ goldens T010d, parity T027d/e)), syntaxes
and probe by 002 (T006a–T006c, T014a), aliases and pull gate by 004 (T005, T013a). Tasks here
own the mechanism's cross-family parts, the Tier 1 gate, Tier 2 and the post-0.1 families. Every
family follows the porting checklist (research R11): module → syntax → goldens → parity → alias +
fit → client smoke → cache cases → bench cells → locality.

## Phase 1: Setup

- [ ] T001 `scripts/check_family_diff.sh` (allowed-path list from the contract) and a CI job on PRs labelled `family:*`; self-test on a fixture diff (FR-019)
- [ ] T002 [P] MF-1: full-KV vs ring state per family on 48/36/24 GB budgets (arithmetic now; allocation check on the target Mac when gpt-oss runs); record in research R3 (OQ-1 stays open until Gemma's turn)
- [ ] T003 Reconcile in the change that lands each piece: 001 FR-025/contract (per-layer K/V widths, `None` layers), 004 manifest (`tier`, `syntax`, fit fields), ARCHITECTURE model table (measured numbers) and syntax list (Gemma), D015, site `docs/models.md`

## Phase 2: Foundational (blocks all stories)

- [ ] T004 Contract conformance of 001's registry and modules (001 T040): `Family` methods per [contract](contracts/families.md), `StateLayout` per layer incl. differing K/V widths and `None`, `LoadError` set incl. `SyntaxMismatch`; review against the locality rule (needs 001 T039–T043)
- [ ] T005 [P] `scripts/gen_tiny_models.py` (mlx-lm, fixed seeds; quantized, unquantized and MXFP4 variants; every block option a config enables) and `crates/mrun-engine/tests/families.rs`: greedy ids for 32 steps and CPU logits for every registry key (FR-017, SC-006)
- [ ] T006 [P] MF-2: window slice vs both references (temporal-order substituted cache; `KVCache` + full-depth mask) on gpt-oss shapes: bit agreement and speed at 16K/64K; record in research R3 (with 001 T043a)
- [ ] T007 Verify 002 T006a with a Qwen3-30B-A3B-Instruct-2507 fixture loading as Hermes and a mismatch fixture (owned by 002) (FR-014; OQ-7)
- [ ] T007a Probe tool-placement check (FR-014): `tools_late` set for a Mistral Small 3.2 template fixture (`[AVAILABLE_TOOLS]` before the last user message) and clear for Ministral 3 and Devstral Small 2507 fixtures; `mrun doctor` and `mrun ps` show `template-tools-late`

## Phase 3: US1 + US2 + US3 — Tier 1 gate for the 0.1 families (P1) 🎯 MVP

One checklist pass per model; items owned elsewhere are verified, not redone.

- [ ] T008 [US1] gpt-oss: module + parity (001 T042, T043a, T027d), Harmony (002 T006b, T014a), goldens (001 T010d), alias `gpt-oss` (004 T005; OQ-6 Q8 vs Q4), fit numbers measured (FR-009, SC-008), MF-4 MXFP4 speed (SC-005)
- [ ] T009 [US3] gpt-oss cache cases (≥ 30): divergences at p mod 128 ∈ {0, 1, 127}, p < 128, p > 256, subagent round trip, cache-on = cache-off (SC-003; with 001 T045, 003 T013)
- [ ] T010 [US1] Devstral: module + parity (001 T043, T043a, T027e), Mistral syntax (002 T006c), goldens vs mistral-common and the repo template (001 T010d; OQ-3 decides Tier 1), alias `devstral` with fit numbers; cache cases (SC-003)
- [ ] T011 [US1] Qwen3.6, Qwen3-Coder, Qwen3-8B: checklist items (parity 001 SC-001, goldens 001 T010c, probe 002 T006a, fit, cache cases 003 T013) recorded per model
- [ ] T012 [US2] Client smoke for all five: 004 read-edit-test task 5/5 in Claude Code, Codex, OpenCode (SC-002; with 002 T016a, 004 SC-004)
- [ ] T013 [P] [US1] Bench cells per Tier 1 model (005 synthetic + agent), SC-005 recorded

## Phase 4: US4 — Clear answers for unsupported models (P2)

- [ ] T014 [US4] Tier 2: manifest `tier: 2` on pull of a non-alias repo, `mrun show` label, refusal fixtures end to end (unknown pair, unsupported mode, unknown tensor, syntax mismatch) through `run`, `serve` and `pull` (with 004 T013a) (SC-006)

## Phase 5: US5 — Post-0.1 families (P2)

Each family starts only after MF-6 (agent suite on the target Mac) selects it. Block changes land
first, separately, with block tests; the family change then passes the locality check (SC-007).

### GLM-4.7-Flash (`glm4_moe_lite`, MLA, GLM XML)

- [ ] T015 [US5] Blocks: `blocks/mla.rs` (latent + RoPE-key storage, expanded prefill, absorbed decode, `embed_q`/`unembed_out` sanitize split), sigmoid-with-bias group-limited router with routed scaling, plain shared expert; block tests vs mlx-lm; MF-5 speed
- [ ] T016 [US5] `chat/syntax/glm_xml.rs`: name until first `<arg_key>`, value end at `</arg_value>` followed by `<arg_key>` or `</tool_call>`, schema strings raw, non-strings JSON only (no `literal_eval`), `<think>` reasoning, history `clear_thinking=false`; parser suite (SC-004)
- [ ] T017 [US5] Module `models/glm4_moe_lite.rs` (first dense layers, MTP layer ignored), goldens, parity, alias `glm-4.7-flash` (OQ-6), cache cases, client smoke, bench, locality

### Gemma 4 26B-A4B (`gemma4` + `gemma4_text`, Gemma syntax)

- [ ] T018 [US5] Decide OQ-1 from MF-1 on the target Mac (full KV ≈ 55K capacity vs window ring as position-bound state); record in research and ARCHITECTURE before porting
- [ ] T019 [US5] Blocks: shared-KV layers (`StateLayout::None`), K=V, v-norm without scale, proportional RoPE, GeGLU, logit soft-cap, norm-scaled router with per-expert scale; block tests
- [ ] T020 [US5] `chat/syntax/gemma.rs`: `<|tool_call>call:NAME{…}<tool_call|>`, `<|"|>` strings, bare keys, balanced-brace end, no coercion, buffered per call, thought channel (verify markers on the pinned template); parser suite
- [ ] T021 [US5] Module `models/gemma4.rs` (per-layer input embeddings, vision dropped), goldens, parity (reference per OQ-2 outcome), alias `gemma4`, cache cases incl. p mod 1024 ∈ {0, 1, 1023}, client smoke, bench, locality

### Nemotron-H (`nemotron_h`, Mamba2 hybrid, Qwen XML)

- [ ] T022 [US5] Confirm model and dimensions from the pinned config (OQ-5)
- [ ] T023 [US5] Blocks: `blocks/mamba2.rs` (`ssm_attn` prefill, mlx-lm `ssm_update` Metal kernel verbatim + ops reference, gated RMSNorm, `time_step_limit`), relu² `SwitchMLP` with optional latent projection; checkpoint arrays via `StateLayout::PositionBound`; kernel-vs-ops test on macOS
- [ ] T024 [US5] Module `models/nemotron_h.rs` (pattern `M`/`*`/`-`/`E`, MTP dropped), Qwen XML via probe, goldens, parity, alias `nemotron-3-nano`, hybrid cache cases (restores from every checkpoint kind), client smoke, bench, locality

## Phase 6: Polish

- [ ] T025 [P] MF-3: size of Harmony and GLM `template` misses on recorded traces; propose or reject a narrow rendering fix under D021's rule
- [ ] T026 [P] Site page `site/content/docs/models.md`: tiers, families, syntaxes, state bytes and capacity per Tier 1 model, what an unsupported-architecture error means

## Dependencies

- 001: T004 needs 001 T039–T043; T006 with 001 T043a; T008–T011 need 001 parity tasks. 002: T007
  needs 002 T006a; T008/T010 need 002 T006b/T006c. 003: cache cases need 003 T013. 004: aliases
  need 004 T005; T014 with 004 T013a. 005: T013 needs the harness.
- T001 → T004–T007 → T008–T013 (MVP gate) → T014; Phase 5 after 0.1, each family after MF-6 and
  its block task; T025–T026 last.

## Parallel opportunities

T005, T006 and T002 touch separate files. T008/T009 and T010 are independent. Post-0.1 block
tasks (T015, T019, T023) can run in parallel.
