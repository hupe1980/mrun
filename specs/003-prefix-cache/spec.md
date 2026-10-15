# Feature Specification: Prefix Cache

**Feature Branch**: `003-prefix-cache`

**Created**: 2026-10-14 · **Revised**: 2026-10-15 (design sprint 3: live path plus moved branches)

**Status**: Draft

**Input**: "Exact prefix reuse across turns, branches and tool loops for attention and hybrid
models. The active sequence state is the live path; other conversations are branches whose state
was moved out, never copied twice, so a subagent or title request never destroys the main
conversation. Hybrid models resume from recurrent-state checkpoints captured during prefill. Plain
LRU eviction; memory pressure sheds branches first. Every request reports cached and processed
tokens, the divergence position and a miss cause."

## Clarifications

### Session 2026-10-14

- Q: How much memory may the cache use? → A: All attention-state capacity not occupied by the
  active request; there is no separate cache budget to tune.
- Q: Do miss reports include prompt text? → A: Positions, counts and cause only. Excerpts only with
  `--debug-cache` (loopback only), printed on the server terminal, never on disk or in responses.
- Q: How do concurrent subagents share? → A: Requests are serialized (002 queue); a queued request
  reuses whatever the requests ahead of it leave behind.

### Session 2026-10-15 (third audit)

- Q: Copy on restore or move? → A: Move. The live path is the sequence state; switching moves the
  old tail out and the matched branch in, one layer at a time. Copy-based restore held matched
  state twice and capped a surviving conversation at about half of capacity.
- Q: What is kept of a short divergent tail? → A: Tails under 256 tokens are discarded, not stored.
- Q: Hybrid models in this feature? → A: Yes (Qwen3.6 is the default model). Checkpoints at planned
  positions; at most 8 per conversation.
- Q: Eviction policy? → A: Plain LRU over branch leaves (two 2026 studies: LRU near-optimal for
  single long sessions); no priority tiers.
- Q: Per-request opt-out? → A: Header `x-mrun-cache: off` only; no request-body field.
- Q: Disk persistence? → A: Not in this feature; opt-in SSD tier is spec 007.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Fast repeated agent turns (Priority: P1)

The agent resends its system prompt, tools and history every turn and appends a little. mrun
processes only the new tokens.

**Why this priority**: Re-reading long prompts is what local agent users wait for.

**Independent Test**: Run a 30K-token conversation for 10 appended turns with the cache on and off;
compare time to first token and processed counts.

**Acceptance Scenarios**:

1. **Given** the previous turn is the live path, **When** the next turn extends it, **Then** no
   state is copied or moved and only the new tokens are processed.
2. **Given** a prompt fully contained in stored state, **When** it arrives, **Then** exactly the
   last prompt token is processed (attention models) and usage reports `P − 1` cached.
3. **Given** a request cancelled during prefill, **When** it is retried, **Then** every whole chunk
   committed before the cancel is reused.
4. **Given** the cache is off for a request, **When** it runs with the same chunk split as a cached
   run, **Then** token ids and logits are identical.

---

### User Story 2 - Subagents and side requests never destroy the main conversation (Priority: P1)

The main conversation is long (80K+ tokens). A subagent (sharing the system prompt), a title or
background request (sharing little or nothing), or a retry with an edit runs in between. When the
main conversation continues, its whole state is still there.

**Why this priority**: Branching is constant in agent use; losing an 80K-token state costs minutes.

**Independent Test**: On Qwen3-Coder capacity (~105K tokens) with a parent of ≥ 80K tokens, run
main → subagent → main and main → title request → main; the returning main turn reuses the whole
parent.

**Acceptance Scenarios**:

1. **Given** a live main conversation of 80K tokens, **When** a subagent sharing its first 15K
   tokens runs, **Then** the main tail is moved out (not copied) and the shared 15K are reused by
   the subagent.
2. **Given** the subagent finished, **When** the main conversation's next turn arrives, **Then** the
   subagent tail is moved out (or discarded if under 256 tokens), the main tail is moved back in and
   all parent tokens count as cached.
3. **Given** any switch, **When** state moves, **Then** memory never exceeds steady state by more
   than one layer's state.
4. **Given** two conversations with a common prefix, **When** both run interleaved, **Then** each
   output equals its cache-off output and no request sees tokens beyond its own match.

---

### User Story 3 - Hybrid models reuse exactly (Priority: P1)

On Qwen3.6 (recurrent layers plus attention), recurrent state exists only at the position it was
computed for. Next turns, edits, retries and switches resume from the nearest checkpoint instead
of from zero.

**Why this priority**: Qwen3.6 is the default model.

**Independent Test**: On a Qwen3.6 conversation, resend with (a) an appended turn, (b) a re-rendered
previous reply, (c) an edited last user message, (d) a subagent round trip; check resume positions
and that outputs equal cache-off runs with the same chunk split.

**Acceptance Scenarios**:

1. **Given** a hybrid prefill, **When** it runs, **Then** checkpoints are captured at the end of
   system+tools, the start of the last user or tool message and the end of the prompt's messages.
2. **Given** a divergence at position d, **When** the request resumes, **Then** it restores the
   deepest checkpoint ≤ d, re-prefills the gap and reports it.
3. **Given** a conversation with 8 checkpoints, **When** another is captured, **Then** a middle one
   is dropped.

---

### User Story 4 - Know why the cache missed (Priority: P1)

Every response says how much was reused and, when less than possible, where and why.

**Why this priority**: Unexplained misses are the top complaint about local servers; cause and
position let users fix their client.

**Independent Test**: The mutation suite injects each cause at known positions; every report names
the injected cause and position.

**Acceptance Scenarios**:

1. **Given** any request, **When** it completes, **Then** 002's mrun metadata and `/mrun/status`
   carry cached, processed, divergence position, checkpoint gap and cause.
2. **Given** a previous reply whose API content is resent unchanged but renders to different tokens,
   **When** the next turn arrives, **Then** the cause is `template` at the reply's position.
3. **Given** `--debug-cache` is off (default), **When** a miss occurs, **Then** no prompt text
   appears anywhere.

---

### User Story 5 - The cache never costs memory safety (Priority: P2)

Stored branches use spare capacity and give it back first, during admission, as the active state
grows, and under OS memory pressure.

**Why this priority**: The cache must never be why a request fails or the Mac swaps.

**Independent Test**: Fill capacity with branches, run a maximum-size request with a large output
limit, inject pressure warnings mid-prefill and mid-decode; no request fails and branches go first.

**Acceptance Scenarios**:

1. **Given** branches filling capacity, **When** an admitted request's state grows, **Then** LRU
   branch leaves are evicted before each growth step and the growth always succeeds.
2. **Given** a pressure warning during a request, **When** the next per-layer check point is
   reached, **Then** branches are evicted and the running request continues.
3. **Given** critical pressure while idle, **When** branches are gone, **Then** the idle live path
   is reset.

### Edge Cases

- Prompt does not fit capacity: rejected by admission (001) before the cache changes anything.
- Identical requests queued back to back: the second reuses the first's committed state.
- A queued request's matched branch is evicted last (soft pin) but stays evictable.
- Retry of an identical request: the output may retrace a stored branch; the overlap is merged at
  commit so the tree keeps one copy.
- Divergence leaves a tail under 256 tokens: discarded; a later request that needs it reports
  `evicted`.
- Divergence inside a stored branch: the whole buffer is moved in and the unmatched part is split
  out again, so nothing is held twice.
- Cancel during a move: the move finishes (bounded by one memory pass), then the cancel applies.
- Hybrid with no checkpoint at or below the match: resume from 0, cause `checkpoint`, gap = match.
- `x-mrun-cache: off` while a main conversation is live: the main state is moved out and kept; the
  uncached request's own state is discarded afterwards.
- `mrun cache clear` during a request: the request is unaffected; branches go now, the live path
  is reset once idle.
- Template file changed and the model reloaded: new cache key; old state discarded; cause `model`.
- Engine error during a move: affected state is dropped and the live path reset; the request
  proceeds from position 0 or fails explicitly; nothing stale is ever reused.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: Reuse MUST require identical token ids under the same cache key (model weights,
  tokenizer, chat template, state format). Matching walks tokens over the live path and stored
  branches; hashes never decide reuse.
- **FR-002**: The live path MUST be the active sequence state. A request extending it MUST append
  without copying or moving state. For attention models a divergence inside it MUST trim exactly.
- **FR-003**: On a switch the shared prefix MUST stay in place; the old tail MUST be moved out into
  a branch (or discarded if under 256 tokens) and the matched branch moved in, one layer at a time.
  No token's state is ever held twice by a move; the transient is at most one layer.
- **FR-004**: For hybrid models, prefill MUST capture recurrent-state checkpoints at the end of
  system+tools, the start of the last user or tool message, and the end of the prompt's messages
  (before the generation header); moving a tail out MUST keep its end state as a checkpoint. At
  most 8 checkpoints per conversation (root-to-leaf path); a middle one is dropped first. A request
  MUST resume from the deepest checkpoint at or below its match and re-prefill the gap.
- **FR-005**: Prefill MUST be planned in chunks that end at message boundaries and at checkpoint
  positions (within 001's chunk-size limit), and only whole chunks are committed. A cancelled
  request MUST keep every whole committed chunk.
- **FR-006**: Usage MUST report `cached = min(resume, P − 1)` and `processed = P − cached`, where
  resume is the match (attention) or the restored checkpoint position (hybrid).
- **FR-007**: Admission (001) checks statically that prompt plus clamped output fits capacity. The
  cache MUST evict only for the initial allocation (`P − resume + min(output limit, 4K)` tokens of
  attention state, the move transient and planned checkpoints), then on demand before each growth
  step the engine reports. After the switch no state is pinned, so decode-time eviction always
  succeeds. Matched prefixes of queued requests MUST be evicted last.
- **FR-008**: Eviction MUST be least-recently-used over branch leaves. The live path in use is never
  evicted; its checkpoints are evicted only after all branches. Freed bytes are counted per buffer.
- **FR-009**: Memory pressure, checked at every per-layer check point and between decode tokens,
  MUST evict branches first, then reset an idle live path, and return freed memory to the OS within
  1 s; a running request is never affected.
- **FR-010**: Every request MUST produce a miss report: cached, processed, divergence position,
  checkpoint gap, and a cause ∈ {`new`, `system`, `edit`, `template`, `evicted`, `capacity`, `checkpoint`,
  `model`, `disabled`} or none on a full hit (definitions in [data-model](data-model.md#missreport)).
  It is delivered through 002's mrun metadata (headers or `: mrun` SSE comments) and in
  `/mrun/status` and `mrun ps`.
- **FR-011**: `--debug-cache` (refused on non-loopback binds) MAY print ±40-character excerpts
  around a divergence on the server terminal only. mrun MUST document that the shared cache lets one
  client infer from cached counts whether a prefix was seen.
- **FR-012**: Controls: `--no-cache` (global), header `x-mrun-cache: off` (per request, all APIs;
  stored conversations survive), `mrun cache clear` / `POST /mrun/cache/clear` (drops branches and
  checkpoints, resets the live path once idle).
- **FR-013**: State MUST stay in memory only; nothing is written to disk.
- **FR-014**: A changed model identity (including a changed template file) MUST use a separate
  cache key; state under the old key is discarded.
- **FR-015**: On finish or cancel, the cache MUST record exactly the committed tokens (prompt and
  emitted output) with message boundaries and content hashes; where the live path retraces a stored
  branch, the overlap is merged.

### Key Entities

- **Cache key**: model identity (weights, tokenizer, template, state format).
- **Live path**: the tree path held by the sequence state.
- **Branch**: a stored subtree hanging off the live path; nodes hold segments (refcounted views of
  per-layer buffers) and checkpoints.
- **Checkpoint**: recurrent state at one position (hybrids only).
- **Miss report**: cached, processed, divergence, gap, cause.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: On the target Mac (M4 Pro, 48 GB), for a 30K-token conversation appending 200–1,000
  tokens per turn, median time to first token over 10 turns is at least 10× lower than cache-off,
  for Qwen3-Coder-30B-A3B and Qwen3.6-35B-A3B, and the same ratio holds for the median turn of
  005's `agent/opencode-refactor` replay.
- **SC-002**: With Qwen3-Coder-30B-A3B at its target capacity (~105K tokens) and a parent of ≥ 80K
  tokens, main → subagent → main and main → title request → main both report the returning turn's
  cached tokens ≥ the parent's committed length (100% kept); Qwen3.6 likewise with gap 0.
- **SC-003**: In a mutation suite of ≥ 100 cases (append, edit, rewind, interleaved branches, tiny
  tails, cancel, eviction, pressure, model and template change, attention and hybrid), zero stale
  reuses occur and every case reports the expected cause and divergence.
- **SC-004**: During a switch moving an 80K-token tail out and back in, peak memory exceeds steady
  state by at most one layer's state.
- **SC-005**: Cache-on output equals cache-off output bit for bit (token ids and logits) when the
  cache-off run uses the same chunk split, on Qwen3-8B and Qwen3.6 for ≥ 30 cases.
- **SC-006**: On Qwen3.6, a divergence inside the previous reply or the last user or tool message
  resumes at or after the start of the message containing the divergence or the message before it.

## Assumptions

- 001 provides: lazy `reserve(keep, total)`, `allocated()`, growth reported before allocation,
  planned prefill returning whole committed chunks and checkpoints, forward/sample split,
  `split_off(at) → Segment`, `append(segment, consume)`, `checkpoint()` /
  `restore_checkpoint(id)`.
- 002 provides message boundaries, per-message content hashes and the generation-header position
  from the renderer, carries the miss report, and sends the `x-mrun-cache` header through.
- One request runs at a time (D018). One model is resident (D020).
- Exactness means identical tokens under one cache key; recomputation with a different chunk split
  may differ in the last bits, so equality tests fix the split.
