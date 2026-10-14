# Feature Specification: Prefix-Aware Interactive Agent Runtime

**Feature Directory**: `specs/001-agent-runtime`
**Feature Branch**: None created; feature resolution is independent of Git branches
**Created**: 2026-10-13
**Status**: Draft; Candle-based direction clarified on 2026-10-14; existing backend plan requires revision; not performance-qualified
**Input**: Build an efficient local Rust LLM runtime for an M4 Pro with 48 GB unified memory;
prioritize coding-agent latency, inference-state reuse, memory efficiency, and OpenCode integration.

## Clarifications

### Session 2026-10-14

- Q: What roles should Candle and mistral.rs have in mrun? → A: Candle is the core tensor library for mrun's own runtime; mistral.rs is a competitor and learning reference, not the embedded runtime.
- Q: Must planning retain Rust 1.85? → A: No. A modern Rust version may be selected as needed for Candle; Rust 1.85 must not remain the planned constraint.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Complete a Local Coding Tool Loop (Priority: P1)

A developer selects a provisioned model in their existing coding client, receives streamed output,
lets the client execute a requested tool, and receives a continuation using the tool result.

**Why this priority**: A fast runtime is not useful to an agent if streaming or tool handoff fails.

**Independent Test**: With reuse disabled, use a pinned unmodified client to read a fixture file,
request a correction, apply the client-authorized edit, and run its fixture tests entirely locally.

**Acceptance Scenarios**:

1. **Given** a qualified local model and offline client configuration, **When** the developer asks
   for a text answer, **Then** coherent incremental output and truthful completion status arrive.
2. **Given** declared tools and a tool-capable model, **When** a tool is selected, **Then** the client
   reconstructs one valid call with stable identity, name, and schema-valid arguments; the runtime
   stops at tool handoff and never executes the tool itself.
3. **Given** the client's matching tool result, **When** the client continues, **Then** the model
   sees the result associated with that call and the conversation completes without client patches.
4. **Given** malformed input, an unavailable model, or unsupported options, **When** submitted,
   **Then** the runtime reports a stable actionable error instead of silently ignoring the request.

### User Story 2 - Reuse Context Across Repeated Turns (Priority: P1)

The developer's repeated instructions, tools, and conversation history do not need to be recomputed
on every compatible turn, including a branch or an edited suffix.

**Why this priority**: Repeated long prompts are the central opportunity for agent responsiveness.

**Independent Test**: Replay identical frozen turns with reuse on and off; compare reused/processed
token counts, first useful output latency, generated-output quality, and state correctness.

**Acceptance Scenarios**:

1. **Given** a retained compatible prefix, **When** a new suffix is appended, **Then** only the
   unmatched suffix and any documented boundary replay are processed, and the reuse count is exact.
2. **Given** an earlier conversation edit, **When** another request arrives, **Then** reuse stops
   at the first differing rendered token, even if the visible message text appears similar.
3. **Given** changed weights, tokenizer, template, position settings, adapter, or state format,
   **When** otherwise similar text arrives, **Then** incompatible state is not reused.
4. **Given** two branches or callers, **When** one branch extends or cancels, **Then** the other
   cannot observe mutated state or receive another principal's retained context.

### User Story 3 - Interrupt Without Waiting for Old Work (Priority: P1)

The developer cancels during a long prompt or generation and starts a different request without
waiting for the abandoned request's full output allowance.

**Why this priority**: Rapid direction changes are normal in interactive coding.

**Independent Test**: Cancel queued, prompt-processing, decoding, and slow-consumer requests; inspect
capacity release, next-request latency, bounded buffers, and subsequent state correctness.

**Acceptance Scenarios**:

1. **Given** queued work, **When** its client disconnects or cancels, **Then** no inference begins
   for that request and its reservation is released once.
2. **Given** active work, **When** interrupted, **Then** no further inference work is scheduled for
   it, in-flight work reaches a safe boundary, and the next admitted request can proceed.
3. **Given** a stalled output consumer, **When** its finite write deadline expires, **Then** the
   runtime cancels it rather than buffering indefinitely or pinning foreground capacity forever.
4. **Given** interruption halfway through a tool call, **When** the stream ends, **Then** no partial
   arguments are presented as a successfully completed tool call.

### User Story 4 - Stay Within a Predictable Memory Envelope (Priority: P1)

The developer can keep their editor and tools responsive while the runtime rejects or bounds work
that cannot fit, instead of degrading into avoidable swapping or repeated model reloads.

**Why this priority**: Unified memory serves the OS, editor, tools, weights, and inference state.

**Independent Test**: Configure small finite budgets, grow contexts and distinct prefixes to them,
cancel and unload, inject pressure, and confirm admission, eviction, and recovery.

**Acceptance Scenarios**:

1. **Given** a finite budget, **When** a prompt plus requested output cannot be reserved, **Then**
   it is rejected before inference with the limiting resource and supported allowance.
2. **Given** retained inactive state, **When** new work needs space, **Then** unpinned state is
   evicted; state still referenced by active work is never freed prematurely.
3. **Given** a warm model, **When** compatible turns arrive, **Then** the runtime keeps it loaded
   unless explicit unload, configured idle expiry, failure, or memory pressure requires otherwise.
4. **Given** a completed cancellation or unload, **When** resources are reclaimed, **Then** owned
   reservations and state return to their expected baseline without reference leaks.

### User Story 5 - Understand Where Time and Memory Went (Priority: P2)

A developer or maintainer can distinguish model loading, queueing, prompt processing, generation,
cache benefit, and tool execution without exposing their source code in routine diagnostics.

**Why this priority**: It makes regressions actionable and prevents misleading speed claims.

**Independent Test**: Run frozen cold/warm/hit/miss/tool traces and inspect local measurements;
cross-check first-output timing with an independent client clock and token counts with fixtures.

**Acceptance Scenarios**:

1. **Given** a repeated turn, **When** it completes, **Then** total prompt, reused prompt, processed
   prompt, output, and boundary replay counts reconcile without invented throughput.
2. **Given** routine diagnostics, **When** inspected, **Then** timing and resource metadata are
   available locally but prompts, tool arguments/results, secrets, and absolute repo paths are absent.

### Edge Cases

- Identical complete prompt with no new suffix still needs valid next-token state or explicit
  minimal boundary replay; a zero processed-token count is not infinite prefill throughput.
- Unicode, special tokens, chat-template rewrites, reordered tool definitions, thinking controls,
  or removed reasoning history can change the token prefix even when message structure is similar.
- Changing sampling settings does not authorize reusing RNG, penalties, grammar, or generated
  suffix state from another request; weight-derived prefix compatibility is evaluated separately.
- Empty input, zero output allowance, exact context boundary, oversized output reservation, and
  malformed or deeply nested schemas have explicit rejection/handling before execution.
- Fragmented UTF-8, escaped braces in strings, multiple tool-call fragments, finish signals, and
  usage-only final events must not break stream reconstruction.
- Duplicate/orphan tool results, unknown tool names, incomplete tool output, and unsupported parallel
  tool requests fail clearly; they are not repaired with invented conversation content.
- Cancellation races completion, eviction races active use, failed GPU work leaves uncertain state,
  and a model file replaced at the same path invalidates its old identity.
- Oversubscription, a full queue, a timed-out queue entry, memory-pressure changes after admission,
  and cache-disabled operation preserve finite resource bounds.
- A backend lacking safe rollback must discard uncertain state and cold-prefill the next request;
  recurrent/hybrid/sliding-window architectures are unsupported until separately qualified.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The runtime MUST serve provisioned qualified models locally, without inference-time
  downloads, remote model code, telemetry export, or cloud fallback; default access is loopback only.
- **FR-002**: The developer MUST be able to use an unmodified pinned OpenCode client for streamed
  and nonstreamed text, model discovery, tool handoff, and tool-result continuation.
- **FR-003**: Input handling MUST validate supported fields, roles, limits, and tool-result linkage
  before execution; unsupported semantic options MUST fail explicitly. Planning MUST freeze the
  supported field/schema matrix from the pinned client rather than claim universal compatibility.
- **FR-004**: Streams MUST preserve response and tool-call identity, fragment order, valid text,
  and truthful finish/usage semantics. Exactly one successful terminal completion is allowed;
  interrupted/failed streams MUST NOT be represented as successful full completions.
- **FR-005**: Completed tool calls MUST name a declared tool and satisfy the advertised argument
  schema subset. Unsupported schemas MUST be rejected before inference, and unsuccessful or
  truncated arguments MUST NOT be labeled completed. Tool execution MUST remain with the client.
- **FR-006**: Reuse MUST match the longest eligible exact rendered-token prefix plus compatible
  model and state identity, scoped to the same trusted local principal, not message similarity.
- **FR-007**: Reuse identity MUST include weight content, tokenizer, template and render controls,
  architecture/state version, adapters, positional settings, and cache precision/layout compatibility.
  A hash lookup alone MUST NOT replace exact token/identity validation.
- **FR-008**: Incremental continuation MUST evaluate unmatched tokens only, except explicitly counted
  boundary replay. Editing or branching MUST never reuse state beyond the matching prefix.
- **FR-009**: Shared prefix state MUST be immutable while shared; request-local generated state,
  sampler, grammar, penalties, and random state MUST NOT leak between branches or requests.
- **FR-010**: Users MUST be able to disable reuse and clear retained prefixes. The first feature
  MUST retain state only in memory; restart loses reuse and does not lose the client-owned history.
- **FR-011**: The default MUST allow one active foreground generation and at most eight queued
  requests with a 30-second queue deadline. Queueing is FIFO; caller IDs do not implicitly supersede
  work. Full queues and expired entries MUST fail explicitly and release reservations.
- **FR-012**: Cancellation/disconnection MUST remove queued work, stop scheduling active work at a
  safe boundary, release capacity once, and publish only validated reusable state. GPU-visible
  storage MUST outlive submitted work; interruption MUST NOT claim to preempt an in-flight kernel.
- **FR-013**: Output buffering MUST be finite per request, with a 10-second stalled-write deadline.
  Limit values and overflow behavior MUST be published in the plan and qualified under a slow reader.
- **FR-014**: Admission MUST reserve weights, active state, output growth, transient scratch,
  retained prefixes, and serving buffers against finite configured budgets before execution.
  Model-specific conservative bounds MUST be validated against observed peaks.
- **FR-015**: Initial automatic runtime accounting budget MUST be the lesser of 75% of detected
  physical memory and a pressure-aware available allowance after an 8 GiB host reserve. GPU residency
  MUST additionally respect the device's reported recommended working set where available.
  Users MAY lower the budget; raising it requires explicit opt-in and a warning. No privileged OS
  tuning is permitted. This is admission/accounting policy, not a promised hard process-RSS cap.
- **FR-016**: Inactive state MUST be evictable within a finite retained-state sub-budget whose byte
  value the plan's data model freezes before implementation; active state MUST be
  pinned until safe cleanup. Pressure MUST stop new admissions, evict unpinned state, then fail
  affected work explicitly if safety cannot be restored, without silent context truncation.
- **FR-017**: Model lifecycle MUST avoid duplicate simultaneous loads and unnecessary reloads,
  support explicit unload and a configurable idle policy, and expose load/failure/readiness status.
  Warm interactive mode keeps the model resident until pressure or explicit policy requires unload.
- **FR-018**: Context/output and architecture limits MUST be truthful. Unsupported formats,
  architectures, state operations, or accelerators MUST fail explicitly; fallback must be visible.
- **FR-019**: Local measurements MUST separate queue, load, render/tokenize, cache lookup/restore,
  prefill, first generated output, first useful output, decode, and cleanup; count total/reused/new/
  replayed prompt tokens and generated tokens. Client tool time is separate from runtime inference.
- **FR-020**: Memory measurements MUST report owned budgets and peaks alongside independently
  sampled process and system memory/pressure, without adding overlapping CPU/GPU allocations twice.
- **FR-021**: Default diagnostics MUST omit prompt content, tool data, secrets, and absolute paths.
  Retained prefix state MUST be scoped to a trusted local principal; untrusted sharing is prohibited.
- **FR-022**: Qualification MUST compare reuse on/off, matched baseline engines, and optimized
  product configurations under the frozen benchmark protocol, retaining raw outcomes and failures.
- **FR-023**: Backend errors MUST invalidate uncertain state, preserve process responsiveness where
  safe, and produce a truthful error. Recovery MUST NOT silently resume from an unverified checkpoint.

### Key Entities

- **Qualified Model**: Content identity, tokenizer/template identities, supported state semantics,
  actual context limits, tool/schema capabilities, precision, and deployment requirements.
- **Request**: Trusted local scope, conversation input, output allowance, generation controls,
  lifecycle state, deadline, cancellation, and resource reservation.
- **Prefix Entry**: Valid exact token boundary, compatible inference state, scope, byte accounting,
  usage/eviction metadata, and active-reference lifetime.
- **Resource Envelope**: Physical/device observations, host reserve, configured limits, pinned and
  reclaimable state, transient allowance, and pressure disposition.
- **Run Evidence**: Pinned identities, configuration, client timing, token/memory counters, output
  quality, errors, and comparator conditions. No private prompt capture by default.

## Success Criteria *(mandatory)*

### Measurable Outcomes

All thresholds below are **proposed acceptance gates**, not measurements. Qualify on physical M4 Pro
48 GB hardware under the [frozen protocol](benchmark-protocol.md), initially at concurrency one.
Performance gates apply to the pinned primary model; support and correctness apply to every admitted
model. A failed gate remains visible and prevents performance qualification unless formally amended.

- **SC-001**: All 20 frozen client interoperability cases pass for streaming/nonstreaming,
  tool handoff/result linkage, finish/usage semantics, and explicit rejected inputs, without client
  source changes. At least one real read-edit-test coding task completes offline.
- **SC-002**: For 8,192-token retained-prefix turns with at least 90% prefix overlap and a suffix
  of at most 512 tokens, median first-useful-output latency is at least 50% lower than identical
  cache-off runs, and p95 is at least 30% lower; admission and quality conditions are identical.
- **SC-003**: Over 100 frozen prefix mutation/branch/invalidation cases, no stale or cross-scope
  state is reused; at least 20 positive exact-replay reference cases have identical output token IDs, and
  precision-dependent routes pass predeclared numerical and task-quality tolerances.
- **SC-004**: In 100 cancellation trials per queued/prefill/decode/slow-consumer phase, p95 from
  runtime cancellation observation to safe capacity release is at most 250 ms, with zero leaked
  reservations or corrupt subsequent continuations. Measure disconnect-detection delay separately.
- **SC-005**: During a 30-minute mixed-context/cache-churn run, there are no runtime crashes,
  budget overruns in owned accounting, silent truncations, or runtime-attributable sustained swap
  growth as defined by the protocol; after explicit cache clear and unload, live resource counts return to baseline and
  process footprint is within 512 MiB of its post-initialization unloaded baseline.
- **SC-006**: In matched no-reuse warm runs, median generation throughput is at least 95% of the
  same-core direct-serving baseline for the pinned model (the plan's matched full-chat harness), and median first-generated-output latency is no
  more than 10% worse. A short-prompt and an 8,192-token prompt cell must both pass.
- **SC-007**: Across 20 frozen coding tasks with three paired seeded runs per task, the optimized
  path loses no more than one success out of 60 versus its reference, introduces no new systematic
  failure category, and completed strict tool calls are schema-valid in every admitted test.
  This is a bounded non-regression gate, not proof of general coding ability.
- **SC-008**: The optimized path reduces geometric-mean runtime-controlled time per matched
  ten-turn tool loop by at least 20% versus cache-off mrun. Publish task success and complete
  wall time including tools; do not hide failures or claim tool execution became faster.
- **SC-009**: Routine diagnostics and offline traffic checks across the frozen suite reveal no
  prompt/tool/secret logging, persisted conversation state, or nonloopback runtime network access.

## Assumptions

- One trusted OS user, one active model, one foreground generation, and text-only full-attention
  models are the first scope. Local applications under the same OS identity are not isolated tenants.
- The user provisions models explicitly; paths alone are not model identities. Planning pins actual
  artifacts and proves tool/template/backend compatibility before declaring a support matrix.
- Initial qualified total context is 16,384 tokens with at most 2,048 generated tokens per request;
  prompts must fit the remainder. Wider contexts require separate fit and quality qualification.
- The primary qualification model is selected during planning from the research candidates, not
  presumed best from its parameter count. A second larger coding model is optional if it fits.
- Candle is the selected core tensor library for mrun's own inference runtime, not a requirement
  to embed mistral.rs. mistral.rs remains a competitor, benchmark comparator, and learning reference.
  Planning MUST replace the prior llama.cpp integration direction with a Candle-based design and
  qualify model execution, exact prefix-state control, allocation accounting, and GPU-safe cleanup.
  Modern Rust is authorized where needed; Rust 1.85 is no longer a planning constraint.
  This direction does not mandate new custom kernels, resolve the macOS memory-observation blocker,
  or relax any functional or success criterion.
- Tool-call structural validity does not guarantee correct tool selection or safe execution.
- Durable persistence, approximate prefix matching, automatic conversation compaction, continuous
  batching, speculative decoding, and new Metal kernels are not part of this feature.
- Compatibility boundary and wire schema belong in the plan's contracts; this spec defines outcomes.
  The client/provider version and field matrix must be frozen before the interoperability tests.