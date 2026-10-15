# Feature Specification: Prefix Cache

**Feature Branch**: `003-prefix-cache`

**Created**: 2026-10-14

**Status**: Draft

**Input**: User description: "Exact prefix reuse across turns, branches and tool loops: a radix tree of attention-state snapshots keyed by token IDs per model identity; restore by copy, hot path zero-copy; every response reports cached tokens and every miss explains where and why the prompt diverged; memory pressure sheds cache first; eviction weighs recompute cost saved per byte with recency; designed so hybrid models can store state checkpoints."

## Clarifications

### Session 2026-10-14

- Q: How much memory may the cache use? → A: All attention-state capacity not occupied by the active request's tokens; there is no separate cache budget to tune.
- Q: Do miss explanations include prompt text? → A: Positions, counts and cause only; text excerpts only with `--debug-cache` (loopback only), printed to the server terminal, never on disk or in responses.
- Q: Should mrun strip known volatile client content (opt-in normalization)? → A: Deferred (analysis 2026-10-14): `mrun launch`/`connect` fix the known Claude Code case via environment; revisit if diagnostics show misses they cannot fix.
- Q: How does the cache share memory with large output limits? → A: Cached state may occupy capacity reserved for output until the output needs it, then it is evicted lazily; admission always happens before restore.
- Q: How do concurrent subagents share? → A: Requests are serialized (feature 002); a queued request that shares a prefix with the active one reuses it once that prefix is cached. (Maintainer decision: queue now, batch later.)

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Fast repeated agent turns (Priority: P1)

The developer's agent sends the same long system prompt, tool schemas and history every turn and
appends a little. mrun processes only the new part, so the agent answers quickly even at 30K+
tokens of context.

**Why this priority**: This is the main source of waiting in local agent use and the core of the
product's value.

**Independent Test**: Replay a recorded 20-turn agent session with the cache on and off; compare
time to first token per turn and processed-token counts.

**Acceptance Scenarios**:

1. **Given** a previous turn of the same conversation, **When** the next turn arrives, **Then** only
   the tokens after the longest exactly matching prefix are processed, and usage reports the rest
   as cached.
2. **Given** a request whose whole prompt was seen before, **When** it arrives, **Then** at most the
   last token is reprocessed and the answer starts immediately.
3. **Given** the cache disabled for a request, **When** it runs, **Then** the full prompt is
   processed and the output matches the cached run within the numerical tolerance of SC-002
   (recomputing with different chunking may change the last bits, never the tokens compared).

---

### User Story 2 - Branches and subagents share safely (Priority: P1)

Agents fork conversations: subagents share the parent's system prompt, retries edit the last turn,
and users rewind history. Shared prefixes are reused across all of them without any branch seeing
another's later state.

**Why this priority**: Branching is normal agent behavior; incorrect sharing would corrupt answers
silently, and no sharing wastes most of the benefit.

**Independent Test**: Run the mutation suite: append, edit at various positions, rewind, interleave
two branches, and compare every output against a cache-off run.

**Acceptance Scenarios**:

1. **Given** two conversations with a common prefix, **When** both run, **Then** the common prefix
   is computed once and each conversation's output equals its cache-off output.
2. **Given** a conversation edited at token position N, **When** it is resubmitted, **Then** reuse
   stops at the first differing token and nothing after it is reused.
3. **Given** a different model, tokenizer, chat template or attention-state format, **When**
   otherwise identical text arrives, **Then** no state is reused.

---

### User Story 3 - Know why the cache missed (Priority: P1)

When reuse is lower than expected, the developer sees exactly where the prompt diverged from the
closest cached prompt and a likely cause, and can fix their client setup.

**Why this priority**: Competitors cache too, but misses are invisible. Clients inject volatile
content (timestamps, per-request headers) that silently destroys reuse; explaining it is a clear
differentiator.

**Independent Test**: Send prompts with injected volatile content at known positions and confirm
the reported divergence position and cause category (and, with `--debug-cache`, the excerpt on
the server terminal).

**Acceptance Scenarios**:

1. **Given** a request that reuses less than its closest cached prefix would allow, **When** it
   completes, **Then** the response headers and `mrun ps` report the divergence token position,
   cached vs processed counts and a cause category.
2. **Given** a known volatile pattern (for example a per-request client header in the system
   prompt), **When** detected, **Then** the explanation names it and the documented fix.
3. **Given** debug diagnostics are off (default), **When** a miss is explained, **Then** no prompt
   text appears anywhere; with `--debug-cache` (loopback only), excerpts appear only in the server's
   terminal, never on disk or in HTTP responses.

---

### User Story 4 - Cache never costs memory safety (Priority: P2)

The cache uses spare memory but gives it back first: under pressure or when a request needs room,
the least valuable cached state is evicted before any request fails.

**Why this priority**: The cache must never be the reason the Mac swaps or a request is refused.

**Independent Test**: Fill the cache with many distinct prefixes, then submit a maximum-size
request and simulate memory pressure; verify eviction order and that requests succeed.

**Acceptance Scenarios**:

1. **Given** a full cache, **When** a request needs attention-state capacity, **Then** cached state
   not in use is evicted until the request fits.
2. **Given** OS memory pressure, **When** signaled, **Then** the cache shrinks before any running
   request is affected.
3. **Given** a cached prefix being restored for a request, **When** eviction runs, **Then** that
   prefix is not freed until the copy completes; afterwards it is an ordinary evictable entry.

### Edge Cases

- Identical concurrent requests: the second waits for or reuses the first's prefix rather than
  recomputing it.
- A prefix longer than the remaining capacity: partial reuse is allowed; the rest is processed.
- Cache cleared while a request runs: the running request is unaffected; new requests miss.
- The same text tokenized differently because of template changes: treated as a miss with cause
  "template".
- A model reloaded from changed files: its old cache entries are discarded.
- Processing chunk size changed by configuration: entries stay valid (they hold stored bytes for
  exact tokens); only numerical tolerance tests notice the difference.
- A request whose output reservation needs memory held by cached state: cached state is evicted
  as the output grows; the request never fails because of the cache.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The cache MUST reuse the longest exactly matching token prefix among cached states
  for the same model identity (weights, tokenizer, chat template, attention-state format),
  confirming matches by token comparison, never by hash alone.
- **FR-002**: Reused state MUST be immutable while shared; no request may observe tokens or state
  written by another request beyond the shared prefix.
- **FR-003**: Continuing the same conversation MUST reuse the active state without copying it; when
  the request diverges inside the active state, the active state MUST be trimmed to the divergence
  point (no copy) rather than restored.
- **FR-004**: When the whole prompt is cached, the engine MUST reprocess at most the final token to
  obtain next-token output.
- **FR-005**: Every response MUST report cached and processed prompt tokens; the totals MUST match
  the prompt length exactly (per-API fields owned by feature 002).
- **FR-006**: For every request, mrun MUST record the divergence position against the closest
  cached prompt and classify misses (new, edit, template, volatile, evicted, capacity, model,
  disabled); the result MUST be returned in feature 002's mrun metadata (headers for non-streaming,
  `: mrun` SSE comments for streaming, all three APIs) and shown in `mrun ps`.
- **FR-007**: Cache memory MUST come only from attention-state capacity not needed by the active
  request; admission happens before any restore. Cached state may occupy capacity reserved for a
  request's output until that output needs it, and is then evicted. Eviction removes the least
  recently used unpinned leaves, assistant turns before user/tool turns before system prompts, and
  MUST never evict state in use.
- **FR-008**: Memory pressure signals MUST shrink the cache (and release freed memory to the OS)
  before affecting running requests.
- **FR-009**: Users MUST be able to disable the cache per request and globally, and clear it
  (`mrun cache clear` and an HTTP endpoint).
- **FR-010**: The cache MUST be in memory only in this feature; nothing is persisted to disk.
- **FR-011**: Cache entries are shared by all clients of one server. Because cached-token counts
  reveal whether a prefix was seen before, mrun MUST document this, MUST refuse `--debug-cache`
  (prompt excerpts) on non-loopback binds, and MUST NOT expose excerpts in any HTTP response.
- **FR-012**: The data model MUST allow entries to carry model-state checkpoints at chosen token
  boundaries, so hybrid-attention models can be supported later without redesign.
- **FR-013**: When a client signals that it compacted its context (Claude Code's
  `x-claude-code-context-compacted` hint header), cached state for the superseded conversation MUST
  become the first eviction candidate.

### Key Entities

- **Cache Key**: Model identity (weights hash, tokenizer hash, template hash, state format).
- **Prefix Node**: A token range within a tree of token sequences, holding an immutable state
  segment, byte size, turn type, last use and active-use count.
- **Miss Report**: Divergence position, closest cached length, cause category.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Replaying feature 005's pinned agent trace `agent/opencode-refactor` (20 turns,
  reaching 30K tokens, recorded request parameters) against Qwen3-Coder-30B-A3B 4-bit on the target
  Mac, median time to first token over turns 2–20 is at least 5× lower with the cache than without.
- **SC-002**: In a mutation suite of at least 100 structural cases (append, edit, rewind, branch,
  model or template change), zero stale reuses occur (every reused token range equals the request's
  tokens under the same model identity). On real weights for 30 of those cases, under teacher
  forcing the log-probability of the reference next token differs from cache-off by at most 1e-2
  at every position, and the first 32 greedy tokens are identical for at least 95% of cases.
- **SC-003**: With Qwen3-Coder-30B-A3B 4-bit on the target Mac (M4 Pro, 48 GB), restoring a
  30K-token prefix from cache takes under 10% of the time to compute it.
- **SC-004**: 100% of injected volatile-content cases in the diagnostics suite (at least 20) report
  the correct divergence position and cause category.
- **SC-005**: Under a simulated memory-pressure warning, cached bytes and MLX cache memory drop
  within 1 second and no running request fails.

## Assumptions

- Builds on features 001 (engine, attention state) and 002 (APIs report cached tokens).
- First release supports full-attention models (Qwen3 dense and MoE); hybrid-model checkpoints are
  designed for but implemented later.
- SSD persistence and cross-machine sharing are later features.
- Exactness means reused state holds the stored bytes computed for exactly the same tokens under
  the same model identity; recomputation with different chunking may differ in the last bits.
