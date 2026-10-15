# Feature Specification: Speculative Decoding

**Feature Branch**: `006-speculative-decoding`

**Created**: 2026-10-15

**Status**: Draft (after 0.1; gated by the measure-first check in FR-001). Scope: copy-aware
n-gram/suffix speculation only; MTP drafting is a gated follow-up (see "Follow-up (gated)").

**Input**: User description: "Lossless target-verified speculation for the default Qwen3.6-35B-A3B (MTP heads; mlx-community conversions strip `mtp.*`, so mrun needs weights with MTP tensors) and copy-aware n-gram/suffix drafting (index the prompt including rendered tool calls and JSON-escaped forms) for all models; a per-request controller that picks draft length including 0 from measured acceptance and per-depth verify cost (MoE verify touches the union of experts; long drafts only); rejected tokens trimmed before anything is committed (the prefix store sees only committed tokens); greedy output identical to non-speculative within the chunked-prefill tolerance; sampling via rejection sampling; disabled under grammar masks initially; usage reports drafted/accepted counts in mrun metadata; measure-first gate: acceptance on recorded traces via `mrun bench`. SC: no slowdown >3% on any bench cell with the controller on; ≥1.3× decode on edit-heavy traces."

## Clarifications

### Session 2026-10-15

- Q: Where do MTP weights come from, given mlx-community conversions drop `mtp.*`? → A: Moved to
  "Follow-up (gated)" (sidecar design kept in research R1 for the follow-up).
- Q: Is speculation on by default? → A: Yes, as `auto`, once the measure-first gate (FR-001) and SC-001/SC-002 pass; until then the default is `off`. Under `auto` the controller decides per round and falls back to plain decode (depth 0) where drafts do not pay. `--speculation off`, the config key and a per-request header turn it off.
- Q: Which requests never speculate? → A: Requests with a grammar mask (strict tools, JSON response formats), requests with an explicit `seed` (seeded output stays reproducible as in 001), requests asking for `logprobs`, and requests whose remaining output limit is 1 token.
- Q: Does the MTP head attend to the whole context? → A: Moved to "Follow-up (gated)".
- Q: Does speculation keep anything on disk? → A: No. Controller statistics and verify costs live
  in memory only; verify cost starts from an analytic prior and is refined online (FR-008).

### Session 2026-10-15 (analyze findings)

- Scope cut: ship n-gram/suffix speculation only; MTP moves to a gated follow-up.
- Hybrid rollback: recompute the linear-attention recurrence from the pre-round state over the
  accepted positions using cached per-position inputs; nothing is re-forwarded (FR-006, research R5).
- Acceptance stops at the first accepted EOS; the runtime can cut a round shorter with `commit(k)`
  for host-side stop strings (FR-005, FR-012).
- Exactness checks (`mrun run --check`, 003 exactness tests, `bench --suite agent` comparisons) run
  with speculation off (FR-017).

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
2. **Given** a reply that copies a file excerpt which appears with line-number prefixes in an
   earlier tool result, **When** the model emits it, **Then** the drafter matches the
   line-number-stripped rendering and drafts it.
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
   controller settles at depth 0 within 64 rounds and no bench cell is slower than `off` beyond the
   005 regression threshold (SC-001).
2. **Given** a prompt crafted to make drafts fail, **When** decoding, **Then** acceptance collapse
   drives the controller to depth 0 and probes back off exponentially.
3. **Given** a long-context turn (≥ 64K tokens), **When** decoding with `auto`, **Then** the
   controller uses the verify cost of that context-length bucket, not a short-context estimate.

---

### User Story 3 - MTP drafting on the default model (Priority: P2) — Removed

Moved to "Follow-up (gated)".

---

### User Story 4 - See and control speculation (Priority: P2)

The user sees per response how many tokens were drafted and accepted, and can turn speculation off
globally or per request.

**Why this priority**: Explainability is a product principle; users must be able to rule out
speculation when diagnosing output differences.

**Independent Test**: Send streaming and non-streaming requests on all three APIs with and without
`x-mrun-speculation: off`; check the mrun metadata and the `spec` key of `/mrun/status`.

**Acceptance Scenarios**:

1. **Given** any completed response, **When** inspected, **Then** mrun metadata carries the drafter
   used, drafted, accepted and verify-round counts, or the reason speculation did not run.
2. **Given** `x-mrun-speculation: off`, **When** the request runs, **Then** no drafts are made and
   the metadata reason is `off`.

### Edge Cases

- A draft contains EOS: acceptance ends at the first accepted EOS; later positions are rejected
  and rolled back.
- A host-side stop string ends inside the accepted tokens: the runtime calls `commit(k)` while
  the round's rollback data still exists, so the committed state ends at the stop on attention and
  hybrid models alike; output and committed tokens equal the non-speculative path.
- A draft is longer than the remaining output limit or reservation: it is capped at
  `remaining − 1` tokens; the reservation from admission is never exceeded.
- A verify forward needs more state memory (growth step or the rollback input cache): the engine
  reports the shortfall before any work and the worker evicts cache first, as for decode.
- Cancellation during a verify round: the round's tokens are discarded, the state is rolled back to
  the last emitted token, and the request finishes as cancelled within one round.
- Hybrid model with partial acceptance: the linear-attention recurrence is recomputed from the
  pre-round state over the accepted positions plus one, from the round's cached per-position
  inputs; attention KV is trimmed; nothing is re-forwarded (research R5).
- Memory pressure during speculation: the rollback input cache belongs to the scratch envelope and
  is freed at round end; pressure handling is unchanged.
- Request with `logprobs`: speculation off, reason `logprobs`.
- Very long prompts (≥ 200K tokens): the drafter index is built off the engine thread during
  prefill; if not ready at the first decode, decoding starts plain and drafting starts when it is.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001 (measure first)**: Before the controller ships enabled by default, `mrun bench --suite
  spec` MUST report, per bundled agent trace: the n-gram acceptance-length distribution simulated
  offline against the recorded replies (tokenizer only, no model), reported per tool-call syntax,
  and decode speed with speculation `off`, `ngram` and `auto`. The default stays `off` until SC-001 and SC-002 pass.
- **FR-002**: The n-gram drafter MUST index, per request, the prompt tokens (which include every
  rendered tool call and tool result) and the tokens emitted so far, and propose the continuation
  of the most recent occurrence of the longest suffix match of at least `min_match` tokens.
- **FR-003**: The drafter MUST also index a copy-aware rendering of tool-result and message text
  with line-number prefixes of file-read output removed (`␣*N→` and `␣*N<TAB>` at line start). The
  rendering is tokenized with the model tokenizer and used only for drafting, never rendered into
  the prompt. For families whose tool-call syntax passes arguments as JSON (Hermes, Harmony;
  Mistral from 0.2), it MUST also index a JSON-escaped rendering of the same text, so copied file content
  inside JSON string arguments matches.
- **FR-004**: Removed (MTP drafter; see "Follow-up (gated)").
- **FR-005**: The target MUST verify a draft of length N in one forward over N+1 positions and
  commit the longest accepted prefix plus one target token. Greedy: a draft token is accepted iff it
  equals the target argmax. Sampling (temperature > 0): drafts are point proposals, accepted with
  probability p(x) and otherwise replaced by a sample from p with x removed and renormalized; p is
  the distribution after the request's temperature, top-k, top-p and min-p. Acceptance ends at the
  first accepted EOS (no position after it is committed).
- **FR-006**: Rejected positions MUST be rolled back before the round ends, per state kind (010
  StateLayout). Positional state (full, sliding-window, MLA): trim. Gated DeltaNet state: the engine keeps the pre-round recurrent state and the round's
  per-position linear-attention inputs (conv input, q/k/v, g, β; ≈ 0.5 MiB per position on
  Qwen3.6-35B-A3B) and, on partial acceptance, recomputes only the recurrence over the accepted
  positions plus one from the pre-round state; nothing is re-forwarded. Mamba2 state (and any state
  kind without a rollback): the engine returns `Unsupported` and the controller stays at depth 0
  for that family. Only committed tokens ever
  reach the prefix store, the conversation and the client.
- **FR-007**: A per-model controller MUST choose a depth N ∈ {0} ∪ [N_min, N_max] maximizing
  expected committed tokens per second, where acceptance is measured by match-length bucket and
  verify cost per depth and context-length bucket. Depth 0 is plain pipelined decode. The next
  round's depth is chosen before `sample`, which pipelines only when it is 0
  (`pipeline = next_depth == 0`). The controller MUST probe deeper periodically with exponential
  back-off and MUST keep its statistics across requests in memory for the server's lifetime.
- **FR-008**: Verify cost per width and context-length bucket MUST start from an analytic prior
  (MoE: decode cost scaled by the expected union of routed experts for the width; dense: decode
  cost, since every position reads the same weights, plus per-position attention cost) and be refined by an
  online EWMA of measured rounds. No probe calibration, nothing persisted.
- **FR-009**: Speculation MUST NOT run for requests with a grammar mask, an explicit `seed`,
  `logprobs`, or a remaining output limit of 1; the metadata names the reason (`grammar`, `seed`,
  `logprobs`, `limit`). Note (SchemaFill 2610.07086, SchemaWeaver 2610.17436; delta §5): under a
  grammar, tokens the grammar forces are already appended without sampling by 002's decode hook
  (002 FR-004, FR-036), which is exact; drafting under masks stays later work.
- **FR-010**: Speculation mode MUST be configurable: `--speculation auto|off|ngram` (`mtp` is added
  by the follow-up), `MRUN_SPECULATION_MODE` (and `[speculation] mode` once the config file
  returns, 004 Later); per request,
  header `x-mrun-speculation: off` on all three APIs. `auto` is the default once FR-001's gate
  passes.
- **FR-011**: Every response MUST report in mrun metadata (002 headers or `: mrun` SSE comment
  before the final event): `spec.drafter` (`ngram` | `none`), `spec.drafted`, `spec.accepted`,
  `spec.rounds`, and `spec.reason` when nothing was drafted (`off` or one of FR-009's reasons). Usage token counts are unchanged:
  output tokens count committed tokens only. `/mrun/status` shows per-model controller state under
  the key `spec` (current depth, acceptance by bucket, cost curve).
- **FR-012**: Stop sequences, EOS, UTF-8 hold-back and tool-call parsing MUST see accepted tokens
  one at a time in order, exactly as in plain decoding. When a stop ends after accepted token k of
  the round, the runtime MUST call `commit(k)` before the round ends, so the engine rolls back to
  that point with the round's rollback data.
- **FR-013**: Draft length MUST be capped so the verify forward never exceeds the request's
  reservation; the verify forward's scratch (including the hybrid rollback input cache, ≤ 32 MiB at
  N = 64) MUST be inside the measured scratch envelope (001), or N_max is lowered at load until it
  is.
- **FR-014**: Cancellation MUST be checked before each round and after each layer of a verify
  forward; a cancelled round emits nothing and leaves the state at the last emitted token.
- **FR-015**: Removed (MTP sidecar pull; see "Follow-up (gated)").
- **FR-016**: Removed (MTP memory accounting; see "Follow-up (gated)").
- **FR-017**: `mrun run --check`, the 003 exactness tests and `mrun bench --suite agent` comparisons
  MUST run with `--speculation off`. With speculation on, output is reproducible only within the
  chunked-prefill tolerance (SC-003).

### Key Entities

- **Draft**: Proposed tokens (point proposals) and match length.
- **DraftStep**: Engine handle for one verify forward: `{id, before, width}` (round id, active
  length before the round, N+1).
- **ProposalId**: Device-side handle of a proposal distribution; unused by n-gram drafts (`None`),
  reserved for the follow-up.
- **Verify Round**: Draft, accepted length, bonus token, wall time, context-length bucket.
- **Draft Index**: Suffix structure over prompt, output and the copy-aware renderings (line-number-stripped;
  JSON-escaped for JSON-argument syntaxes) of one request.
- **Controller State**: Per model: acceptance statistics, per-depth cost curve, current depth,
  probe interval.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: With speculation `auto`, no `mrun bench` cell (synthetic prefill and decode at every
  size, every agent trace's TTFT and session time) is slower than with `off` beyond the 005
  regression threshold (5 repetitions, medians), on every 0.1 alias (Qwen3.6-35B-A3B,
  Qwen3-Coder-30B-A3B, gpt-oss-20b) and on Devstral Small 2 once it ships (0.2) on the target Mac.
- **SC-002**: On the edit-heavy traces (bench traces where at least 40% of reply tokens lie in
  matches of ≥ 16 tokens against the prompt), median decode tokens per second with `auto` is at
  least 1.3× that with `off`, on Qwen3.6-35B-A3B and on the default model if different; other
  models need only SC-001.
- **SC-003**: On 001's conformance prompts and the bundled traces (teacher-forced), greedy output
  with `auto` equals `off` token for token in at least 95% of sequences; every divergence occurs at
  a position whose `off` top-2 logit margin is below the chunked-prefill tolerance (log-probability
  difference ≤ 1e-2, as in 003 SC-002).
- **SC-004**: In a mutation suite of at least 100 cases on the scripted engine (partial, full and
  zero acceptance, EOS and stop strings inside drafts, cancellation mid-round, memory shortfalls,
  hybrid rollback including hybrid EOS and hybrid stop-string `commit(k)` cases), zero rejected
  tokens are ever observed in the active state after a round, in the prefix store or in the client
  stream.
- **SC-005**: With temperature 0.7 on a tiny model on CPU, the empirical distribution of 10,000
  sampled tokens at a fixed draft-covered position under `auto` passes a chi-square test against
  `off` at p > 0.01 (rejection sampling is distribution-preserving).

## Assumptions

- Builds on 001 (engine, sampler, scratch probe, hybrid checkpoints), 002 (metadata, request
  options), 003 (prefix store sees committed tokens only) and 005 (traces, cells, regression
  threshold).
- `Cancelled` and `Unsupported` are additive `EngineError` variants (contract).
- EAGLE-3 is out of scope. Trained DFlash2 drafts exist for Qwen3.6-35B-A3B (Inco Splash ships
  them; license and publication unverified, delta §3.1) and are a measured option in "Follow-up
  (gated)". Speculation under grammar masks and concurrent sessions are later work.
- SC-001 compares `off` and `auto` at the 005 regression threshold; 005's regression mode is Later
  (lean cut 2026-10-15), so it returns with this feature or the comparison is run by hand.

## Follow-up (gated)

MTP drafting on Qwen3.6-35B-A3B (former US3, FR-004, FR-015, FR-016; sidecar pipeline and
4,096-position window ring, research R1 and R4). Not part of this feature.

- **Gate**: measure MTP per-position acceptance offline on the bundled traces with an existing
  implementation first. Reference implementation (constitution V): Ollama's MLX runner MTP module
  (`mlxrunner/model/qwen3_5/qwen3_5.go`, `mtp.go`); vLLM's Qwen3.5 MTP drafter as a cross-check.
  Proceed only if α₁ ≥ 0.8 and the modeled speed-up (research R6 formula with FR-008 costs) is
  ≥ 1.15× on Qwen3.6.
- **If it proceeds**: `--speculation mtp`, ratio rejection sampling min(1, p/q) with residual
  normalize(max(0, p − q)) using `ProposalId`, per-position acceptance in the controller, head
  weights and window in the memory budget.
- **Note (F8)**: the MTP sidecar is not part of the 003 cache key (the head never changes target
  state); it is recorded in the manifest and shown by `mrun show` only.

DFlash2 drafts for Qwen3.6-35B-A3B (measured option, added 2026-10-15 after the delta scan):
Splash's 1.7–2× decode lead on this model comes mostly from them. **Gate**: confirm the drafts'
license and that the weights are published; then measure acceptance offline on the bundled traces
and proceed only under the same α₁ ≥ 0.8 and modeled ≥ 1.15× rule. Like MTP, the drafter is never
part of the cache key, and its state never enters rollback data or checkpoints (FR-006; llama.cpp
`e467e5b` leaked draft KV into checkpoints).
