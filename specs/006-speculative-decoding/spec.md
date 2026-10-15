# Feature Specification: Speculative Decoding

**Feature Branch**: `006-speculative-decoding`

**Created**: 2026-10-15

**Status**: Draft (after 0.1; gated by the measure-first check in FR-001)

**Input**: User description: "Lossless target-verified speculation for the default Qwen3.6-35B-A3B (MTP heads; mlx-community conversions strip `mtp.*`, so mrun needs weights with MTP tensors) and copy-aware n-gram/suffix drafting (index the prompt including rendered tool calls and JSON-escaped forms) for all models; a per-request controller that picks draft length including 0 from measured acceptance and per-depth verify cost (MoE verify touches the union of experts; long drafts only); rejected tokens trimmed before anything is committed (the prefix store sees only committed tokens); greedy output identical to non-speculative within the chunked-prefill tolerance; sampling via rejection sampling; disabled under grammar masks initially; usage reports drafted/accepted counts in mrun metadata; measure-first gate: acceptance on recorded traces via `mrun bench`. SC: no slowdown >3% on any bench cell with the controller on; ≥1.3× decode on edit-heavy traces."

## Clarifications

### Session 2026-10-15

- Q: Where do MTP weights come from, given mlx-community conversions drop `mtp.*`? → A: `mrun pull` fetches only the `mtp.*` tensors of the pinned original-precision repository (`Qwen/Qwen3.6-35B-A3B`) by HTTP range requests against its safetensors index, verifies each tensor against a SHA-256 pinned in the alias table, quantizes the head locally with the base model's scheme, and stores it as a sidecar next to the manifest. The base weights stay the pinned mlx-community repository. (Verified in mlx-lm `ba8a08f`: `qwen3_5.py:354` drops every key containing `mtp.`, and `convert` loads through `sanitize`.) See research R1.
- Q: Is speculation on by default? → A: Yes, as `auto`, once the measure-first gate (FR-001) and SC-001/SC-002 pass; until then the default is `off`. Under `auto` the controller decides per round and falls back to plain decode (depth 0) where drafts do not pay. `--speculation off`, the config key and a per-request header turn it off.
- Q: Which requests never speculate? → A: Requests with a grammar mask (strict tools, JSON response formats), requests with an explicit `seed` (seeded output stays reproducible as in 001), requests asking for `logprobs`, and requests whose remaining output limit is 1 token.
- Q: Does the MTP head attend to the whole context? → A: No. Its attention covers a window of the most recent 4,096 positions, kept with the live path only. Drafts are always verified, so the window affects acceptance, never output. See research R4.
- Q: Does speculation keep anything on disk? → A: No. Controller statistics live in memory; per-depth verify costs are calibrated with the 001 scratch probe and cached with it.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - File edits stream faster (Priority: P1)

A Claude Code or OpenCode user asks the agent to edit code. The model's reply copies long
stretches from context (an `Edit` call's `old_string`, a rewritten file, a repeated function).
These stretches stream noticeably faster than plain decoding, with the same output.

**Why this priority**: Copy-heavy tool calls are the longest decode phases in agent sessions and
the only place on an MoE model where speculation reliably pays.

**Independent Test**: Replay the edit-heavy bench traces with `--speculation off` and `auto`;
compare decode tokens per second per turn and confirm identical greedy outputs.

**Acceptance Scenarios**:

1. **Given** a turn whose reply copies text that appears in the prompt, **When** it decodes with
   speculation `auto`, **Then** copied stretches are drafted from the prompt index, verified by the
   target, and the turn's decode rate is higher than with speculation off.
2. **Given** a tool call whose JSON arguments contain a file excerpt that appears raw (with line
   numbers) in an earlier tool result, **When** the model emits it, **Then** the drafter matches
   the escaped, line-number-stripped rendering and drafts it.
3. **Given** greedy sampling, **When** the same request runs with speculation `auto` and `off`,
   **Then** the outputs are identical, except at positions where the non-speculative top-2 logit
   margin is within the chunked-prefill tolerance (SC-003).

---

### User Story 2 - Speculation never makes a turn slower (Priority: P1)

Wherever drafts do not pay (free-form reasoning, adversarial inputs, an MoE verify that touches
many experts), the controller stops drafting, so no workload decodes slower than without
speculation.

**Why this priority**: On a 4-bit MoE a short draft with low acceptance can be several times
slower than plain decode; a default-on feature must not regress any cell.

**Independent Test**: Run every bench cell (synthetic and agent) with speculation `off` and `auto`;
compare medians.

**Acceptance Scenarios**:

1. **Given** text with no exploitable repetition, **When** decoding with `auto`, **Then** the
   controller settles at depth 0 within its warm-up window and decode speed stays within 3% of `off`.
2. **Given** a prompt crafted to make drafts fail, **When** decoding, **Then** acceptance collapse
   drives the controller to depth 0 and probes back off exponentially.
3. **Given** a long-context turn (≥ 64K tokens), **When** decoding with `auto`, **Then** the
   controller uses the verify cost measured at that depth, not a short-context estimate.

---

### User Story 3 - MTP drafting on the default model (Priority: P2)

On Qwen3.6-35B-A3B the model's own multi-token-prediction head drafts short continuations where no
copy exists, so ordinary code generation also gains when the measured cost allows.

**Why this priority**: MTP is the only drafter that helps non-copy text, but its gain on an A3B MoE
is uncertain; n-gram drafting delivers the main value first.

**Independent Test**: Pull the default alias with its MTP sidecar, replay the agent traces with
`--speculation mtp` and `off`, and compare acceptance per position and decode speed.

**Acceptance Scenarios**:

1. **Given** `mrun pull qwen3.6`, **When** it completes, **Then** the MTP sidecar is fetched,
   verified, quantized and recorded in the manifest, and `mrun show` lists it with its size.
2. **Given** a model without MTP tensors, **When** speculation is `auto`, **Then** only the n-gram
   drafter is used and `mrun ps` says why MTP is unavailable.
3. **Given** both drafters available, **When** a round starts, **Then** the controller picks the
   drafter and depth with the highest expected committed tokens per unit of time.

---

### User Story 4 - See and control speculation (Priority: P2)

The user sees per response how many tokens were drafted and accepted, and can turn speculation off
globally or per request.

**Why this priority**: Explainability is a product principle; users must be able to rule out
speculation when diagnosing output differences.

**Independent Test**: Send streaming and non-streaming requests on all three APIs with and without
`x-mrun-speculation: off`; check the mrun metadata and `/mrun/status`.

**Acceptance Scenarios**:

1. **Given** any completed response, **When** inspected, **Then** mrun metadata carries the drafter
   used, drafted, accepted and verify-round counts, or the reason speculation did not run.
2. **Given** `x-mrun-speculation: off`, **When** the request runs, **Then** no drafts are made and
   the metadata reason is `off`.

### Edge Cases

- A draft crosses a stop sequence or EOS: tokens after the stop are discarded and trimmed; output
  and committed tokens equal the non-speculative path.
- A draft is longer than the remaining output limit or reservation: it is capped at
  `remaining − 1` tokens; the reservation from admission is never exceeded.
- A verify forward needs more state memory (growth step or in-round snapshots): the engine reports
  the shortfall before any work and the worker evicts cache first, as for decode.
- Cancellation during a verify round: the round's tokens are discarded, the state is rolled back to
  the last emitted token, and the request finishes as cancelled within one round.
- Hybrid model with partial acceptance: recurrent state is restored from the in-round snapshot at
  or below the accepted length and the gap is re-forwarded (research R5).
- Memory pressure during speculation: in-round snapshots belong to the scratch envelope and are
  freed at round end; pressure handling is unchanged.
- Request with `logprobs`: speculation off, reason `logprobs`.
- The MTP sidecar is missing, corrupt or from another base revision: MTP is unavailable (reason
  shown), the model still loads, n-gram drafting still works.
- Very long prompts (≥ 200K tokens): the drafter index is built off the engine thread during
  prefill; if not ready at the first decode, decoding starts plain and drafting starts when it is.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001 (measure first)**: Before the controller ships enabled by default, `mrun bench --suite
  spec` MUST report, per bundled agent trace: the n-gram acceptance-length distribution simulated
  offline against the recorded replies (tokenizer only, no model), MTP acceptance per draft
  position on the target Mac, and decode speed with speculation `off`, `ngram`, `mtp` and `auto`.
  The default stays `off` until SC-001 and SC-002 pass.
- **FR-002**: The n-gram drafter MUST index, per request, the prompt tokens (which include every
  rendered tool call and tool result) and the tokens emitted so far, and propose the continuation
  of the most recent occurrence of the longest suffix match of at least `min_match` tokens.
- **FR-003**: The drafter MUST also index copy-aware renderings of tool-result and message text:
  line-number prefixes of file-read output removed (`␣*N→` and `␣*N<TAB>` at line start), and, for
  models whose tool-call syntax JSON-escapes argument values, the JSON-escaped form. Renderings are
  tokenized with the model tokenizer and used only for drafting, never rendered into the prompt.
- **FR-004**: For models whose loaded weights include an MTP head, an MTP drafter MUST propose
  tokens autoregressively from the head (one head step per draft token), attending to a window of
  the most recent 4,096 positions.
- **FR-005**: The target MUST verify a draft of length N in one forward over N+1 positions and
  commit the longest accepted prefix plus one target token. Greedy: a draft token is accepted iff it
  equals the target argmax. Sampling (temperature > 0): n-gram drafts are point proposals, accepted
  with probability p(x) and otherwise replaced by a sample from p with x removed and renormalized;
  MTP drafts are accepted with probability min(1, p(x)/q(x)) and otherwise replaced by a sample from
  normalize(max(0, p − q)); p and q are the distributions after the request's temperature, top-k,
  top-p and min-p.
- **FR-006**: Rejected positions MUST be rolled back before the round ends: attention state by
  trim; recurrent state (hybrids) by restoring the in-round snapshot at or below the accepted
  length and re-forwarding the gap. Only committed tokens ever reach the prefix store, the
  conversation and the client.
- **FR-007**: A per-model controller MUST choose, before each round, a drafter and a depth N ∈
  {0} ∪ [N_min, N_max], maximizing expected committed tokens per second, where acceptance is
  measured per drafter (n-gram: by match-length bucket; MTP: per position) and verify cost is
  measured per depth and context-length bucket. Depth 0 is plain pipelined decode. The controller
  MUST probe deeper periodically with exponential back-off and MUST persist its statistics across
  requests in memory for the server's lifetime.
- **FR-008**: Per-depth verify costs MUST be calibrated during 001's scratch probe (widths 1–64 at a
  short and the deepest admissible position) and cached with it; online measurements refine them.
- **FR-009**: Speculation MUST NOT run for requests with a grammar mask, an explicit `seed`,
  `logprobs`, or a remaining output limit of 1; the metadata names the reason (`grammar`, `seed`,
  `logprobs`, `limit`).
- **FR-010**: Speculation mode MUST be configurable: `--speculation auto|off|ngram|mtp`,
  `[speculation] mode` in `config.toml`, `MRUN_SPECULATION_MODE`; per request, header
  `x-mrun-speculation: off` on all three APIs. `auto` is the default once FR-001's gate passes.
- **FR-011**: Every response MUST report in mrun metadata (002 headers or `: mrun` SSE comment
  before the final event): `spec.drafter` (`ngram` | `mtp` | `mixed` | `none`), `spec.drafted`,
  `spec.accepted`, `spec.rounds`, and `spec.reason` when nothing was drafted. Usage token counts are
  unchanged: output tokens count committed tokens only. `/mrun/status` shows per-model controller
  state (current depth per drafter, acceptance, cost curve).
- **FR-012**: Stop sequences, EOS, UTF-8 hold-back and tool-call parsing MUST see accepted tokens
  one at a time in order, exactly as in plain decoding.
- **FR-013**: Draft length MUST be capped so the verify forward never exceeds the request's
  reservation; the verify forward's scratch (including in-round recurrent snapshots) MUST be inside
  the measured scratch envelope (001), or N_max is lowered at load until it is.
- **FR-014**: Cancellation MUST be checked before each round and after each layer of a verify
  forward; a cancelled round emits nothing and leaves the state at the last emitted token.
- **FR-015**: `mrun pull` of an alias that declares an MTP source MUST fetch only `mtp.*` tensors
  from the pinned source revision, verify each against its pinned SHA-256, quantize with the base
  model's scheme, write the sidecar atomically, and record its hash in the manifest; `--no-mtp`
  skips it. The sidecar hash becomes part of the model identity used by 003.
- **FR-016**: The MTP head's weights and its windowed attention state MUST be counted in the memory
  budget at load (weights measured; window state fixed size).

### Key Entities

- **Draft**: Drafter, proposed tokens, proposal distribution (MTP) or point mass (n-gram), match
  length (n-gram).
- **Verify Round**: Draft, accepted length, bonus token, wall time, depth bucket.
- **Draft Index**: Suffix structure over prompt, output and copy-aware renderings of one request.
- **Controller State**: Per model and drafter: acceptance statistics, per-depth cost curve, current
  depth, probe interval.
- **MTP Sidecar**: Quantized head tensors, source repository and revision, per-tensor source
  hashes, sidecar hash.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: With speculation `auto`, no `mrun bench` cell (synthetic prefill and decode at every
  size, every agent trace's TTFT and session time) is more than 3% slower in median than with `off`,
  on Qwen3.6-35B-A3B and Qwen3-Coder-30B-A3B 4-bit on the target Mac.
- **SC-002**: On the edit-heavy traces (bench traces where at least 40% of reply tokens lie in
  matches of ≥ 16 tokens against the prompt), median decode tokens per second with `auto` is at
  least 1.3× that with `off`, on both models.
- **SC-003**: On 001's conformance prompts and the bundled traces (teacher-forced), greedy output
  with `auto` equals `off` token for token in at least 95% of sequences; every divergence occurs at
  a position whose `off` top-2 logit margin is below the chunked-prefill tolerance (log-probability
  difference ≤ 1e-2, as in 003 SC-002).
- **SC-004**: In a mutation suite of at least 100 cases on the scripted engine (partial, full and
  zero acceptance, stops inside drafts, cancellation mid-round, memory shortfalls, hybrid rollback), zero
  rejected tokens are ever observed in the active state after a round, in the prefix store or in
  the client stream.
- **SC-005**: With temperature 0.7, the empirical distribution of 10,000 sampled tokens per test
  prompt under `auto` passes a chi-square test against `off` at p > 0.01 (rejection sampling is
  distribution-preserving).

## Assumptions

- Builds on 001 (engine, sampler, scratch probe, hybrid checkpoints), 002 (metadata, request
  options), 003 (prefix store sees committed tokens only) and 005 (traces, cells).
- The MTP head layout follows Qwen3.5/3.6 (`mtp.fc`, `mtp.pre_fc_norm_{embedding,hidden}`,
  `mtp.layers.0.*`, `mtp.norm`), sharing the base model's embedding and output head (Ollama
  `mlxrunner/model/qwen3_5/qwen3_5.go:1093`). Sizes in research are estimates until measured.
- EAGLE-3, DFlash and other trained drafters are out of scope (no drafter exists for the default
  model). Speculation under grammar masks and concurrent sessions are later work.
