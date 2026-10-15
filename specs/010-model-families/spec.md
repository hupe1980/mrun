# Feature Specification: Model Families

**Feature Branch**: `010-model-families`

**Created**: 2026-10-15

**Status**: Draft (revised 2026-10-15, lean 0.1 cut, D025): post-0.1. Devstral Small 2 is 0.2;
GLM-4.7-Flash, Gemma 4 and Nemotron-H follow. The 0.1 family modules (`qwen3_5_moe`, `qwen3_moe`,
`gpt_oss`, and `qwen3` as test reference) are delivered by 001 under this spec's design.

**Input**: "mrun supports an open set of model families, not one model. A family is a small module
that maps `config.json` and weight names onto a shared block library (full GQA, sliding-window
attention with sinks, MLA; Gated DeltaNet and Mamba2; dense SwiGLU/GeGLU and mixture-of-experts
feed-forward with shared experts; norms; RoPE variants incl. YaRN and proportional), declares its
state kinds and names its chat syntax. A registry keyed by `model_type` selects the module; unknown
types are refused with the supported list; a template probe confirms the syntax. Chat syntaxes
(Hermes JSON, Qwen XML, OpenAI Harmony, Mistral `[TOOL_CALLS]`, GLM XML) are streaming parsers that
follow 002's rules. Quantization: MLX affine 4/8-bit and MXFP4. Tier 1 models get a curated alias,
a parity test against mlx-lm and bench cells; Tier 2 architectures load any MLX repo of that
architecture. 0.1 ships Qwen3.6-35B-A3B, Qwen3-Coder-30B-A3B, gpt-oss-20b, Devstral Small 2 and
the Qwen3-8B reference; GLM-4.7-Flash, Gemma 4 and Nemotron-H follow. Adding a family touches only
its module, a syntax entry if new, an alias and tests."

*Input as originally written; superseded by the lean cut (Status, Clarifications): Devstral
Small 2 is 0.2, not 0.1.*

## Clarifications

### Session 2026-10-15

- Q: How are sliding-window layers stored? → A: Decided (ARCHITECTURE, Sequence state): full KV,
  window applied as a mask/slice, so the state stays positional and cuttable at any position. The
  memory cost is measured per family (MF-1); Gemma 4 is the case where it matters (OQ-1).
- Q: What kind of state is MLA? → A: Positional. The compressed latent and the RoPE key are stored
  per token per layer and cut by offset like K/V.
- Q: What does 010 own, given 001, 002 and 004 deliver the 0.1 families? → A: 001 delivers the
  block options, registry and family modules of 0.1 (001 FR-023–FR-025, T039–T043a, T045, T046 (+ goldens T010d, parity T027d/e)), 002 the syntax
  registry, template probe and the 0.1 syntaxes (002 FR-032–FR-035), 004 the aliases and the pull
  gate (004 FR-002, FR-014). 010 owns the complete design they implement (family contract,
  per-layer state layout, tiers, porting checklist, locality rule), Tier 2, the Tier 1 gate across
  all families, and the post-0.1 blocks (MLA, Mamba2, Gemma options), families and syntaxes (GLM
  XML, Gemma). *Revised by the lean cut (session below): Devstral and the Mistral syntax move here
  as 0.2; the cross-family 0.1 gates are deleted in favor of the owning specs' gates.*
- Q: What decides the chat syntax, the family or the template? → A: *Revised by the lean cut
  (2026-10-15):* each family names one syntax; load checks the template renders that syntax's
  markers, else refuses (002 FR-032). Ordered sets (for `qwen3_moe` repos such as
  Qwen3-30B-A3B-Instruct-2507 that use Hermes) return only when a supported model needs them; see
  OQ-7.
- Q: Which quantization formats? → A: MLX affine (4/8-bit, per-path overrides) and MXFP4 (group 32,
  4-bit, uint8 scales, no biases). Unquantized bf16/f16 load. Other modes are refused by name.
  MLX v0.32.4 and mlx-c v0.7.0 support `mode` in `quantized_matmul` and `gather_qmm` (research R6).
- Q: What is a tier? → A: Tier 1: curated alias with pinned revision, parity test on real weights,
  template goldens, client smoke and bench cells. Tier 2: the architecture is registered; any MLX
  repo of it loads after load checks; parity only on a tiny random-weight model. Anything else is
  unsupported and refused.
- Q: Which "Nemotron-H"? → A: `nemotron_h` architecture; Tier 1 candidate NVIDIA Nemotron 3 Nano
  30B-A3B (Mamba2 + MoE + attention). Confirmed by the post-0.1 measure-first check (OQ-5).
- Q: Gemma 4 uses its own tool syntax (`<|tool_call>call:name{…}<tool_call|>`). Allowed? → A: Yes,
  as a sixth syntax entry when Gemma 4 lands (a family may add one syntax entry).

### Session 2026-10-15 (lean 0.1 cut, D025)

- Q: What is 010's scope now? → A: Post-0.1 only, plus the design the 0.1 modules follow. The 0.1
  gates live in their owning specs (parity 001 SC-001, goldens 001 SC-009, cache equality 003
  SC-005/SC-007, speed 001 SC-002/SC-003, client gate 002 SC-001), so 010's duplicates are
  deleted: SC-001, SC-003, SC-005. Also removed: tier demotion of 0.1 aliases (FR-016), the
  locality CI check (FR-019, SC-007), the `--info` accuracy gate (SC-008) and ordered syntax sets.
- Q: Which family comes first after 0.1? → A: Devstral Small 2 (`mistral`, Mistral syntax) in 0.2;
  its design below is unchanged.
- Q: Do non-alias repos load in 0.1? → A: 0.1 loads non-alias repos of the four 0.1
  architectures via 004 FR-001 without a `tier` label; post-0.1 Tier 2 adds the label, refusal
  fixtures and `mrun show` marking (T014).

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Run the strongest model of any supported family (Priority: P1)

A developer pulls `gpt-oss` or, from 0.2, `devstral` (or any alias) and runs it exactly like
the Qwen models: same commands, same memory budget, same exactness.

**Why this priority**: Strong coding models come from several families; tying mrun to one family
loses users the day another family releases a better model.

**Independent Test**: For each 0.1 alias, `mrun run <alias> --check` passes greedy parity, and
`mrun run <alias> --info` reports state bytes per token that match the computed layout.

**Acceptance Scenarios**:

1. **Given** a pulled Tier 1 model, **When** the developer runs it, **Then** it loads through the
   registry, streams output and reports its family, syntax, quantization and state layout.
2. **Given** the reference prompt set, **When** `--check` runs, **Then** greedy token ids equal
   mlx-lm's at the pinned commit for every prompt.
3. **Given** gpt-oss-20b (MXFP4 experts), **When** loaded, **Then** experts run through MXFP4
   `gather_qmm` and the rest through the repo's affine quantization, with no dequantized copy.

---

### User Story 2 - Agents work with every family's tool format (Priority: P1)

Claude Code, Codex and OpenCode complete tool loops against every Tier 1 model, whatever the
model's native syntax (Hermes JSON, Qwen XML, Harmony, Mistral).

**Why this priority**: A model that cannot drive the agent's tools is useless to the user.

**Independent Test**: The 004 scripted read-edit-test task on each Tier 1 model through each client.

**Acceptance Scenarios**:

1. **Given** gpt-oss-20b, **When** it emits `<|channel|>commentary to=functions.read_file
   <|constrain|>json<|message|>{…}<|call|>`, **Then** clients receive one tool call with those
   arguments; analysis-channel text arrives as reasoning, final-channel text as content.
2. **Given** Devstral Small 2 (0.2), **When** it emits `[TOOL_CALLS]read_file[ARGS]{…}` twice,
   **Then** clients receive two tool calls.
3. **Given** any syntax, **When** an argument value contains that syntax's own closing marker,
   **Then** it round-trips unchanged; malformed arguments follow 002 FR-004's raw fallback; values
   are never coerced or evaluated.

---

### User Story 3 - Prefix reuse is exact for every state kind (Priority: P1)

Turns, subagents and edits reuse state exactly on full-attention, sliding-window, MLA and hybrid
models, including restores at positions that are not window-aligned.

**Why this priority**: Reuse is the product (PR-2); a family that breaks it silently is worse than
an unsupported one.

**Independent Test**: 003's equality suite per Tier 1 family, with divergences placed below,
at and across window boundaries.

**Acceptance Scenarios**:

1. **Given** gpt-oss-20b with a 5K-token conversation, **When** a request diverges at position
   p for p ∈ {50, 127, 128, 129, 1000, 4097}, **Then** the cache trims to p and the output equals a
   cache-off run with the same chunk split, bit for bit.
2. **Given** a family with position-bound state (Gated DeltaNet, Mamba2), **When** a request
   diverges, **Then** it resumes from the deepest checkpoint ≤ p (003 FR-004).
3. **Given** any family, **When** `mrun run --info` runs, **Then** positional bytes per token and
   position-bound bytes per checkpoint are reported per state kind.

---

### User Story 4 - Clear answers for models mrun cannot run (Priority: P2)

A developer points mrun at an arbitrary Hugging Face MLX repo. A registered architecture loads
(Tier 2); anything else is refused before any weight is read, naming what is missing.

**Why this priority**: Users try many repos; a crash or garbage output costs trust.

**Independent Test**: Load fixtures: unknown `model_type`, known type with unsupported quantization
mode, unknown tensor names, a tiny random-weight model of every registered architecture.

**Acceptance Scenarios**:

1. **Given** an unknown `model_type`, **When** loading, **Then** mrun exits with code 3 listing the
   supported architectures and the Tier 1 aliases.
2. **Given** a Tier 2 repo, **When** loaded, **Then** it runs and `mrun show` marks it
   `tier 2 (architecture supported, model not validated)`.
3. **Given** a template that does not render the family syntax's markers, **When** loaded,
   **Then** the load is refused naming the family, its syntax and the missing markers (002
   FR-032).

---

### User Story 5 - Adding a family stays local (Priority: P2)

A maintainer ports a new family by following the porting checklist; the change touches only the
family module, one syntax entry if new, one alias and tests.

**Why this priority**: Model releases are monthly; the cost of a port decides how fast mrun follows.

**Independent Test**: Review of the Devstral, GLM-4.7-Flash, Gemma 4 and Nemotron-H changes
against the locality rule (contract; a review rule since the CI check was cut on 2026-10-15).

**Acceptance Scenarios**:

1. **Given** a family that needs a new block, **When** ported, **Then** the block lands first in
   its own change with a block-level test, and the family change stays local.

### Edge Cases

- `config.json` nests the text model (`text_config`, e.g. `mistral3` → `ministral3`, `gemma4` →
  `gemma4_text`): the registry keys on the pair; vision towers and projectors are dropped by the
  family's sanitize; images are 009's concern.
- Same `model_type`, different syntax (Qwen3-30B-A3B-Instruct-2507 Hermes vs Qwen3-Coder XML):
  refused by the template check until ordered syntax sets return (OQ-7).
- Mixed quantization in one repo (MXFP4 experts, affine attention; 8-bit routers): per-path
  overrides from `quantization`; a path with no matching mode is refused at load, naming the path.
- Original (non-MLX) checkpoints, e.g. `openai/gpt-oss-20b` with `_blocks`/`_scales` tensors:
  loaded only where the family's sanitize ports mlx-lm's conversion (Tier 2); else refused.
- Sliding-window layers past the window: KV is kept; attention reads only the last W keys (plus
  sinks); trim to any position is exact.
- Harmony stop tokens `<|call|>` and `<|return|>` are end-of-turn markers, in the EOS set.
- Templates that call `strftime_now` (gpt-oss: current date in the system message): rendered with
  the server's start date (OQ-4), so the date never changes mid-session.
- History re-rendering that differs from generated bytes (Harmony drops earlier analysis once a
  final message follows; `<|return|>` renders as `<|end|>`; GLM drops reasoning before the last
  user turn unless `clear_thinking=false`): reported as 003 cause `changed` (005 counts them as `template` re-renders); mrun passes the
  template options that preserve reasoning where they exist.
- Gemma 4 KV-shared layers have no KV of their own: no state, no bytes.
- An MTP/next-n head in the weights (`mtp.*`, `num_nextn_predict_layers`): ignored, as in 001.
- A model whose maximum position is below its computed capacity: capacity is capped (001 FR-008).

## Requirements *(mandatory)*

### Functional Requirements

Ownership: 001 implements FR-002–FR-008 and FR-010 for the 0.1 block options (via 001
FR-023–FR-025; uniform per-layer widths suffice at 0.1); for the 0.1 families 002 implements
FR-011–FR-015 (via 002 FR-032, FR-033, FR-035) and 004 owns the aliases. 010's own work is
post-0.1: Devstral (0.2), FR-001's later blocks, FR-005's per-layer `StateLayout`, FR-009,
FR-017, FR-018 and Tier 2.

**Family mechanism**

- **FR-001**: The engine MUST provide a block library: attention (full GQA; sliding window with
  optional per-head sinks; gated-output attention; MLA with optional q-LoRA and absorbed decode),
  linear attention (Gated DeltaNet; Mamba2), feed-forward (dense SwiGLU, GeGLU, clamped SwiGLU-OAI;
  MoE with softmax or sigmoid-with-bias routers, top-k, optional renormalization and routed scaling,
  optional shared experts, `gather_qmm` experts), norms (RMSNorm, RMSNorm without scale, gated
  RMSNorm), RoPE (default, linear, YaRN, proportional/partial, Llama-4 position scaling of queries),
  embeddings, tied or separate output head, final logit soft-capping. Each block is a port of the
  mlx-lm code it comes from (research R2) and has a block-level test against it.
- **FR-002**: A family module MUST implement the family contract
  ([contracts/families.md](contracts/families.md)): parse and validate its config (rejecting
  unknown values with named reasons), map and sanitize weights, compose the layer stack from
  blocks, declare per-layer state layout and kind, and name its chat syntax (FR-014). Family
  code
  MUST NOT leak outside its module (Principle V).
- **FR-003**: A registry MUST map `(model_type, text_config.model_type?)` to a family module. An
  unregistered pair MUST be refused before reading weights (exit 3) with the supported
  architectures and Tier 1 aliases listed (message: 004 E034).
- **FR-004**: Loading MUST check, before weights are read: registered architecture, supported
  quantization mode and bits for every path, tensor names and shapes expected by the module
  (after sanitize), and dimensions consistent with the config. Failures name the path or key.
- **FR-005**: `ModelInfo` (001) MUST describe state per layer rather than per model: kind
  (positional, position-bound, none), bytes per token or per checkpoint, and window if any. 001's
  budget, admission and probe use these sums; `mrun run --info` and `mrun show` print them.

**State kinds**

- **FR-006**: Positional state MUST cover full attention, sliding-window attention and MLA.
  Sliding-window layers keep full KV; attention reads keys in `[max(0, q − W + 1), q]` (plus sinks);
  trim, split-off and append work at any position exactly as for full attention.
- **FR-007**: MLA layers MUST store the normalized compressed latent (`kv_lora_rank` wide) and the
  RoPE key (`qk_rope_head_dim` wide) per token, as in mlx-lm; K and V widths may differ per layer.
- **FR-008**: Position-bound state MUST cover Gated DeltaNet (conv + delta state) and Mamba2 (conv
  + SSM state); checkpoints deep-copy all position-bound state (001 FR-019); 003's checkpoint
  policy applies unchanged.
- **FR-009**: The memory cost of each family's state layout MUST be measured on the target Mac and
  recorded with the alias (bytes per token, bytes per checkpoint, capacity on 48 GB), and
  `mrun run --info` MUST match the computed layout within 2%.

**Quantization**

- **FR-010**: The loader MUST support MLX affine quantization (4 and 8 bits, group 32/64/128) with
  per-path overrides, MXFP4 (group 32, 4 bits, uint8 scales, no biases) for dense and
  `gather_qmm` layers, and unquantized bf16/f16. The mode MUST be passed to `quantized_matmul` and
  `gather_qmm` explicitly; when `config.json` omits it, it is inferred as mlx-lm does. Other modes
  (nvfp4, mxfp8, GPTQ/AWQ layouts) MUST be refused by name until a Tier 1 model needs them.

**Chat syntax**

- **FR-011**: A syntax registry MUST provide streaming tool-call and reasoning parsers for: Hermes
  JSON (`<tool_call>{json}</tool_call>`), Qwen XML (`<function=…><parameter=…>`), OpenAI Harmony
  (channels `analysis`/`commentary`/`final`, recipients `to=functions.NAME` in the role or channel
  header, optional `<|constrain|>json`, arguments ending at `<|call|>`), Mistral
  (`[TOOL_CALLS]NAME[ARGS]{json}`, repeated, no end marker), GLM XML
  (`<tool_call>NAME<arg_key>K</arg_key><arg_value>V</arg_value>…</tool_call>`), and, with Gemma 4,
  Gemma (`<|tool_call>call:NAME{…}<tool_call|>` with `<|"|>` string delimiters).
- **FR-012**: Every parser MUST follow 002 FR-004: values end only at the full closing sequence in
  context (code that contains the syntax's own markers round-trips); string values are never
  coerced and non-string values are parsed as JSON only (never `eval`/`literal_eval`); JSON-native
  arguments (Hermes, Harmony, Mistral) stream raw on Chat/Responses, synthesized ones (XML, Gemma)
  are buffered per call; unparseable calls fall back to raw arguments (Chat/Responses) or a text
  block (Messages) with `tool-args-invalid`; undeclared tools pass through with `tool-unknown`;
  a call cut by the output limit is reported incomplete.
- **FR-013**: Reasoning MUST map to 002's reasoning events per syntax: `<think>…</think>` (Qwen,
  GLM, Nemotron), Harmony `analysis` channel, Mistral `[THINK]…[/THINK]`, Gemma's thought channel.
  Harmony `commentary` text without a recipient is content (preamble).
- **FR-014**: At load, a template check MUST render a fixed tool conversation with the model's
  template and require the family syntax's markers in the rendered text and as added tokens; each
  family names one syntax; a miss refuses the load naming the family, its syntax and the missing
  markers (002 FR-032). *Moved to Later (0.1 scope cut, 2026-10-15)*: ordered syntax sets (OQ-7)
  and the tool-placement check (`template-tools-late` in `doctor`/`ps` for templates that render
  tools after the first user message, e.g. Mistral Small 3.2; Ministral 3 and Devstral Small 2507
  put tools at the start, llama.cpp `models/templates`).
- **FR-015**: History rendering MUST pass template options that keep earlier reasoning when the
  template offers them (Qwen3.6 preserve-thinking, GLM `clear_thinking=false`) and MUST map
  reasoning into the field each template reads (`reasoning_content`, Harmony `thinking`, Mistral
  `thinking` blocks).

**Tiers and porting**

- **FR-016**: Tier 1 models MUST have a curated alias (004 FR-002 format) with pinned revision,
  hashes, license, family, syntax and fit numbers: `devstral` (0.2), then `glm-4.7-flash`,
  `gemma4`, `nemotron-3-nano` (each when its family passes its gates). *Removed (0.1 scope cut,
  2026-10-15)*: the `tier: 2` demotion of a 0.1 alias that misses a gate (004 drops it instead).
- **FR-017**: Every registered architecture MUST have a tiny random-weight parity test against
  mlx-lm (CPU, Linux CI), covering every block option its config space enables.
- **FR-018**: A family port MUST complete the porting checklist (tasks): module; syntax entry if
  new; template goldens; parity; alias with fit numbers; the client gate (002 SC-001); bench cells
  (005). A family is Tier 1 only when all items pass.
- **FR-019**: *Removed (0.1 scope cut, 2026-10-15)*: the CI locality check. The locality rule
  ([contracts/families.md](contracts/families.md#locality-rule)) stays a review rule;
  block-library changes are separate changes with their own tests.

### Key Entities

- **Block**: A shared, parameterized layer component ported from mlx-lm.
- **Family module**: Config mapping, sanitize, layer composition, state layout, accepted syntaxes.
- **Registry entry**: `(model_type, text model_type?)` → family, tier, aliases, accepted syntaxes.
- **State layout**: Per layer: kind, widths, window, bytes per token or per checkpoint.
- **Chat syntax**: Streaming tool-call and reasoning parser plus history-rendering options.
- **Template probe result**: Detected syntax, markers found, decision.
- **Tier 1 alias**: Pinned repo, revision, hashes, license, family, syntax, fit numbers.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: *Removed (0.1 scope cut, 2026-10-15)*: duplicate of 001 SC-001 (parity), which
  each new family's module extends.
- **SC-002**: Every new alias meets the client gate (002 SC-001).
- **SC-003**: *Removed (0.1 scope cut, 2026-10-15)*: duplicate of 003 SC-005/SC-007 (cache
  equality), which each new family extends.
- **SC-004**: Each syntax passes 100% of its parser suite: every marker split at every byte
  boundary (text-matched markers) or token boundary (token-id markers), parallel calls where the
  syntax allows them (not Harmony), values containing the syntax's own closing markers, string values
  `"null"`/`"true"` kept as strings, malformed and truncated calls, undeclared tools; and ≥ 10
  template golden fixtures per Tier 1 model render to Hugging Face's text and token ids (001
  SC-009; mistral-common for Devstral, OQ-3).
- **SC-005**: *Removed (0.1 scope cut, 2026-10-15)*: duplicate of 001 SC-002/SC-003 (speed).
- **SC-006**: 100% of load-refusal fixtures (unknown architecture, unsupported mode, unknown
  tensor, shape mismatch) are refused before weights are read with a message naming the cause and
  the supported list; every registered architecture passes its tiny-model parity test in CI.
- **SC-007**: *Removed (0.1 scope cut, 2026-10-15)*: locality CI check (FR-019).
- **SC-008**: *Removed (0.1 scope cut, 2026-10-15)*: `--info` within 2% of measured allocations
  (FR-009 still records measured numbers with each alias).

## Measure first and open questions

| ID | Item | Gate / decision it feeds |
| --- | --- | --- |
| MF-1 | Memory cost of full KV on sliding-window layers, per family, on 48, 36 and 24 GB Macs (computed: gpt-oss 48 vs 24 KiB/token, no capacity loss at 48 GB; Gemma 4 ≈ 220 vs ≈ 20 KiB/token, capacity ≈ 55K vs ≥ 262K) | OQ-1 |
| MF-2 | Sliding-window attention in prefill and decode: key slice `[q − W + 1, q]` vs full-depth mask; speed at 16K/64K and bitwise agreement with the mlx-lm `KVCache` reference | R3 decision; SC-001, SC-005 |
| MF-3 | Size of `template` misses on recorded traces for Harmony (dropped analysis, `<|return|>` → `<|end|>`) and GLM | Whether a narrow server-side rendering fix (D021-style) is worth specifying |
| MF-4 | MXFP4 `gather_qmm` decode/prefill speed vs mlx-lm on gpt-oss-20b | SC-005 |
| MF-5 | MLA absorbed decode vs expanded prefill speed at 32K/128K on GLM-4.7-Flash | SC-005 for GLM |
| MF-6 | Agent-suite results of GLM-4.7-Flash, Gemma 4 26B-A4B, Nemotron 3 Nano on the target Mac (ROADMAP 1b) | Which post-0.1 families become Tier 1 and in which order |

- **OQ-1**: Gemma 4 sliding layers. Full KV costs ≈ 10× its attention state (25 of 30 layers are
  1024-window). Alternative: treat the window ring as position-bound state (≈ 200 MiB per
  checkpoint) under 003's checkpoint policy. Decide after MF-1; the decision for gpt-oss stands.
- **OQ-2**: Does stock mlx-lm with `RotatingKVCache` agree token for token with the full-KV
  reference on gpt-oss-20b (recorded in 001 R16, not a gate; resolved on device by 001 T043a)? If not, the
  same question decides Gemma 4's reference.
- **OQ-3** (decided 2026-10-15): the repo's `chat_template.jinja` renders prompts (ARCHITECTURE, Models: the
  model's own Jinja template renders prompts); goldens compare it against mistral-common, and a mismatch keeps Devstral
  at Tier 2 until the template is fixed upstream or pinned to a corrected revision.
- **OQ-4** (decided 2026-10-15): `strftime_now` returns the date the server started, so the prompt
  stays stable for the server's lifetime; restarting picks up the new date. Documented as a
  deviation from HF rendering.
- **OQ-5**: Nemotron-H Tier 1 model: Nemotron 3 Nano 30B-A3B or the newer Nemotron 3.5 Lightning
  30B-A3B (same `nemotron_h`?); dimensions and Mamba2 state dtype to be read from the pinned config.
- **OQ-6**: Pins for post-0.1 aliases: mlx-community vs lmstudio-community conversions (GLM,
  Gemma, Nemotron), and gpt-oss `MXFP4-Q8` vs `MXFP4-Q4` (non-expert precision; agent-suite result).
- **OQ-7** (decided 2026-10-15, revised by the lean cut the same day): one syntax per family at
  0.1; ordered accepted sets return when a supported model needs a second syntax for one
  architecture (e.g. Qwen3-30B-A3B-Instruct-2507, Hermes, under `qwen3_moe`).

## Assumptions

- 001 implements the engine contract, budget, checkpoints, moves and the 0.1 family modules on
  the block library; 010 defines the layout those modules follow (contract) and adds families.
- 002 owns the syntax registry, the probe, the 0.1 parsers, the API mapping of parser events and
  the FR-004 policy; 010 adds the GLM XML and Gemma parsers under the same rules.
- 003 consumes state kinds only through the engine contract; no cache change per family.
- 004 owns alias format, pulling and the 0.1 aliases; 010 adds post-0.1 alias entries
  (`devstral` first).
- 005 owns bench cells; 010 adds one cell set per new Tier 1 model.
- Reference: mlx-lm `ba8a08f` on MLX v0.32.4 (001). Ports follow that commit; newer upstream
  fixes are taken by re-pinning, not by divergence.
- Vision components (Devstral/Pixtral tower, Gemma 4 vision) are out of scope (009).
