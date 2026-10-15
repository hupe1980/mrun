# Feature Specification: Engine Core

**Feature Branch**: `001-engine-core`

**Created**: 2026-10-14 · **Revised**: 2026-10-15 (design sprint 3: hybrid default, reshaped engine contract, audit fixes)

**Status**: Draft

**Input**: User description: "Engine core: run a pulled local coding model on an Apple Silicon Mac from the mrun binary. `mrun run <model>` streams generated text in the terminal; the engine owns the model's sequence state with chunked prefill so a long prompt can be cancelled within one layer; a startup memory budget sizes weights + sequence state + measured scratch and rejects prompts that cannot fit with the numbers; greedy output must match the reference implementation token-for-token; decode speed within 10% of the reference on the same weights. Models: Qwen3-8B (dense reference), Qwen3-Coder-30B-A3B (attention MoE) and Qwen3.6-35B-A3B (hybrid Gated DeltaNet + attention MoE, intended default). The engine exposes the primitives the prefix cache (003) needs: lazy reservation, reported growth, checkpoints of recurrent state, and moving state tails out and back in. Engine runs on a dedicated thread; async callers stream tokens through a bounded channel."

## Clarifications

### Session 2026-10-14

- Q: Should Qwen3 thinking models think by default in `mrun run`? → A: No thinking unless requested (`--think`); Qwen3-Coder has no thinking mode.
- Q: How many conversations run at once? → A: One active sequence; others queue (002). Batching is a later feature (D018).
- Q: Where do models come from? → A: The shared Hugging Face cache (honoring `HF_HOME`), plus explicit local paths. Pulling is 004.

### Session 2026-10-15

- Q: Which models? → A: Qwen3-8B (dense, correctness reference), Qwen3-Coder-30B-A3B (attention MoE), Qwen3.6-35B-A3B (hybrid: 30 Gated DeltaNet + 10 gated attention layers, MoE 256 experts, `model_type` `qwen3_5_moe`). The default is confirmed by `mrun bench` before release (D015).
- Q: What is "greedy parity"? → A: The token chosen is the argmax over log-probabilities (`logits − logsumexp(logits)`) computed in the logits dtype, exactly as mlx-lm `generate_step` does, with the same prefill chunk split. Any fused (compiled) op mlx-lm uses must be bitwise-equal in mrun: either compiled the same way or proven equal unfused.
- Q: What is the sequence state? → A: Attention K/V per attention layer (cut at any position by offset) plus, for hybrid models, the recurrent state (convolution + delta-rule state) per linear layer, which exists only at the position it was computed for. Hybrid models resume earlier positions from checkpoints.
- Q: How large is a prefill chunk? → A: Depth-adaptive: `clamp(2048 × 16384 / depth, 256, 2048)` tokens where depth is the chunk's start position; chunks also end at caller-given boundaries. Cancellation is checked after every layer with one layer in flight. A cancelled partial chunk is discarded; whole chunks are kept.
- Q: When is memory allocated? → A: Lazily and in steps, never above the request's reserved total or the state capacity. Before any allocation the engine reports the bytes it needs so the caller can evict cache first.
- Q: What does a conversation keep after a turn? → A: Exactly its committed tokens: prompt plus emitted output. Held-back, stop-matched and pipelined tokens are trimmed.
- Q: Is the model ever unloaded when idle? → A: No (D020).

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Chat with a local model in the terminal (Priority: P1)

A developer with a supported model on disk runs `mrun run <model>`, types a prompt and sees the
answer stream token by token. Follow-up prompts continue the same conversation.

**Why this priority**: The smallest end-to-end proof that mrun loads and runs a model on the Mac;
every later feature builds on this engine.

**Independent Test**: On an Apple Silicon Mac with any supported model on disk, run
`mrun run <model>`, ask two questions, and observe streamed output, a stats line per answer, and
cached tokens on the second answer.

**Acceptance Scenarios**:

1. **Given** a supported model on disk, **When** the developer runs `mrun run <model>` and enters a
   prompt, **Then** the answer streams until the model stops or reaches the output limit.
2. **Given** an active conversation, **When** the developer enters a follow-up, **Then** the model
   answers with the earlier turns as context and only the new turn is processed (hybrid models:
   plus at most the previous reply, when the template re-renders it differently).
3. **Given** a model name or path that does not exist or is not supported, **When** the developer
   runs `mrun run`, **Then** mrun exits with a message naming the problem and the next step.

---

### User Story 2 - Interrupt instantly at any depth (Priority: P1)

The developer presses Ctrl-C while a long prompt is being processed or while the answer streams.
Work stops almost immediately; completed prompt processing is kept.

**Why this priority**: Agents and humans change direction constantly; waiting for abandoned work is
the most frustrating latency, and it grows with context unless bounded per layer.

**Independent Test**: Submit prompts reaching 16K, 64K and the model's maximum admissible depth,
interrupt during prompt processing and during generation, and measure the time until mrun is
ready for the next prompt.

**Acceptance Scenarios**:

1. **Given** a long prompt being processed at any depth, **When** the developer interrupts,
   **Then** processing stops within 300 ms, no answer is emitted, and every whole chunk already
   processed stays in the conversation state.
2. **Given** an answer streaming, **When** the developer interrupts, **Then** streaming stops after
   at most one more token and the conversation keeps exactly the tokens already shown.
3. **Given** an interrupted request, **When** the developer submits the next prompt, **Then** it
   runs normally and reuses the kept state.

---

### User Story 3 - Never overcommit memory (Priority: P1)

mrun decides at startup how much memory it may use, loads the model only if it fits, refuses
prompts that cannot fit with the numbers, and clamps output limits to what fits.

**Why this priority**: Unified memory is shared with the editor, browser and agent; swapping or an
OS kill destroys the experience.

**Independent Test**: Start mrun with a lowered budget, read the capacity from `--info`, submit
prompts at capacity − 1 and capacity tokens, and an oversized output limit.

**Acceptance Scenarios**:

1. **Given** a model larger than the budget, **When** the developer runs it, **Then** mrun refuses
   before loading, stating weights, budget and how to change the budget.
2. **Given** a loaded model, **When** a prompt leaves no room for one output token, **Then** mrun
   rejects it before processing, stating requested tokens and capacity.
3. **Given** an output limit larger than the remaining capacity, **When** submitted, **Then** the
   limit is clamped, the applied limit is reported, and generation stops with "length" there.
4. **Given** `mrun run --info`, **When** the model loads, **Then** mrun prints weights, scratch,
   recurrent state, state capacity in tokens and bytes, probed depth, reserve and headroom.

---

### User Story 4 - Trustworthy output (Priority: P2)

A maintainer verifies that mrun produces exactly the reference implementation's greedy tokens on
the same weights for every supported model, that chat templates render exactly as Hugging Face
does, and that sampling controls behave as documented.

**Why this priority**: Exactness is the foundation for prefix reuse (003) and fair benchmarks.

**Independent Test**: Run the conformance command per model and the template golden tests.

**Acceptance Scenarios**:

1. **Given** the reference prompt set at temperature 0, **When** the conformance check runs,
   **Then** every generated token ID matches the stored reference output.
2. **Given** template fixtures (system, tools, multi-turn, tool results, thinking on/off),
   **When** rendered, **Then** the text and token IDs equal the Hugging Face rendering.
3. **Given** a fixed seed and sampling settings, **When** the same prompt runs twice, **Then** the
   outputs are identical; invalid settings (NaN or negative temperature, top-p outside (0, 1],
   min-p outside [0, 1]) are rejected up front.
4. **Given** stop sequences, **When** the model emits one, **Then** output ends before it and the
   finish reason is "stop".

---

### User Story 5 - State primitives for branching (Priority: P2)

A maintainer of the prefix cache (003) can move the tail of the conversation state out, restore
recurrent state from a checkpoint, and move a stored tail back in, without ever holding the same
state twice and without changing results.

**Why this priority**: Subagents and retries must not destroy the main conversation (D017); the
prefix cache can only be exact if the engine primitives are.

**Independent Test**: On each model, prefill a conversation with planned checkpoints, split the
tail off, run another prompt, move the tail back, and compare the next-token logits with an
uninterrupted run that used the same chunk split.

**Acceptance Scenarios**:

1. **Given** a live conversation, **When** its tail is split off and appended back, **Then** the
   next-token logits are bit-identical to never having moved it, and peak memory never exceeds
   the state capacity plus one layer.
2. **Given** a hybrid conversation with a checkpoint at position p, **When** state is restored to
   that checkpoint and the gap re-prefilled with the same chunk split, **Then** logits are
   bit-identical to the uninterrupted run.
3. **Given** state growth that needs memory, **When** cache holds evictable state, **Then** the
   engine reports the bytes before allocating, the caller evicts, and the operation succeeds.

### Edge Cases

- Empty prompt or output limit 0: rejected with a clear message.
- Prompt of capacity − 1 tokens: accepted with an output limit of 1; capacity tokens: rejected.
- A stop sequence split across tokens: detected; partial matches are not emitted early; the
  tokens that formed it are not committed.
- Output ending inside a multi-byte character: no invalid text is shown.
- Interrupt arriving as generation completes: exactly one `Finished` is reported.
- A consumer that stops reading: its request is cancelled after the stall timeout; the worker
  never blocks on it again and still delivers `Finished` without waiting.
- A panic inside a request: that request finishes with an error, the state is reset, the worker
  keeps serving.
- Hybrid divergence inside the previous reply: state resumes from the deepest checkpoint at or
  below the divergence and the gap is re-processed.
- Model files changed on disk after load: the loaded weights keep serving; the change is reported
  at next load (identity covers `generation_config.json` and `special_tokens_map.json` too).
- Other applications consume memory after startup: `--info` reports reduced headroom; cache is
  shed first (003).
- Unsupported architecture, quantization or tensor layout: rejected at load with the reason.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: `mrun run <model>` MUST load a supported model from a local path, a Hugging Face cache
  repo id, or (with 004) an alias, and start an interactive streaming chat session.
- **FR-002**: The engine MUST run three architectures in the community MLX quantized format
  (affine 4-bit, with 8-bit router and shared-expert gates): `qwen3` (reference Qwen3-8B),
  `qwen3_moe` (Qwen3-Coder-30B-A3B-Instruct) and `qwen3_5_moe` (Qwen3.6-35B-A3B, text model only;
  vision and `mtp.*` tensors ignored). Other architectures and layouts MUST be rejected with a named
  reason.
- **FR-003**: Every loaded model MUST have a content identity over all files that determine output:
  weights, `config.json`, `generation_config.json`, `tokenizer.json`, `tokenizer_config.json`,
  `special_tokens_map.json`, `chat_template.jinja`. End-of-sequence tokens MUST be the union of the
  EOS ids in `config.json`, `generation_config.json` and the tokenizer's EOS token. A changed
  identity for a known path MUST be reported at load.
- **FR-004**: The engine MUST keep the conversation's sequence state between turns so a follow-up
  processes only new tokens. The runtime MUST track committed tokens (prompt + emitted output) and,
  after every finish (stop, EOS, stop sequence, length, cancel, error), leave the state holding
  exactly the longest committed prefix it can represent: held-back, stop-matched and pipelined
  tokens are trimmed; hybrid models that cannot trim that far resume from the deepest checkpoint
  at or below it. The runtime's view of the state MUST equal the engine's token for token.
- **FR-005**: Prompt processing MUST run in chunks of `clamp(2048 × 16384 / depth, 256, 2048)`
  tokens (depth = chunk start, minimum 1), ending additionally at caller-given boundaries and
  planned checkpoint positions; it MUST keep one layer in flight and check cancellation after
  every layer; a cancelled partial chunk MUST be discarded and only whole chunks reported as
  processed. Generation MUST check cancellation after every token.
- **FR-006**: At startup the engine MUST compute `budget = min(fraction × physical,
  recommended working set) − reserve`, with fraction 0.75 and reserve
  `clamp(physical ÷ 6, 2 GiB, 8 GiB)` by default, physical memory and working set read from the
  MLX device info (`memory_size`, `max_recommended_working_set_size`). Users may lower the budget,
  and raise it only with an explicit flag.
- **FR-007**: Before loading weights the engine MUST estimate weights from file sizes and refuse a
  model whose weights + scratch estimate + recurrent state + 4,096 tokens of attention state
  exceed the budget, stating each number; after loading it MUST verify with measured sizes.
- **FR-008**: The engine MUST measure peak scratch by probe chunks at the deepest admissible
  position and at the largest chunk (including the one-layer growth transient and the recurrent
  rollback copy), cache it per (model identity, machine, macOS, MLX version, mrun version, chunk
  policy), and size state capacity from the remainder. Token capacity MUST NOT exceed the model's
  maximum position or the probed depth.
- **FR-009**: A request MUST be admitted only if its prompt plus one output token fits the token
  capacity (memory held by cached state counts as available) and rejected up front otherwise,
  stating requested tokens and capacity. The output limit MUST be clamped to the remaining
  capacity and the applied limit reported.
- **FR-010**: Sequence-state memory MUST be allocated lazily: attention arrays start at the input
  plus up to 4,096 tokens and grow in 4,096-token steps, one layer at a time, always clamped to the
  request's reserved total and the token capacity. Before any allocation (growth, checkpoint,
  move-in) the engine MUST report the bytes needed without allocating, so the caller evicts
  cached state first; total state bytes MUST never exceed the state capacity.
- **FR-011**: Sampling MUST support temperature, top-p, top-k, min-p, seed, stop sequences and max
  tokens, validated before execution; temperature 0 is greedy as defined in the clarifications.
  An optional token mask (002) MUST be applied between forward and sample.
- **FR-012**: Model work MUST run on one dedicated worker thread that owns all engine state. Events
  MUST flow through a bounded channel usable from sync and async code; a consumer that stops
  reading for the stall timeout (default 10 s) cancels its request and is never waited on again;
  `Finished` MUST always be delivered, exactly once, without blocking; a panic in a request MUST be
  caught, reported as an error finish, and followed by a state reset.
- **FR-013**: Every finished request MUST report prompt tokens, cached tokens, generated tokens,
  applied output limit, time to first token (first sampled token), prompt-processing speed,
  generation speed and finish reason (stop, length, cancelled, rejected, error).
- **FR-014**: A conformance command MUST compare greedy token IDs against stored reference outputs
  for each supported model and report the first mismatch with the log-probability margin there.
- **FR-015**: Streamed text MUST always be valid UTF-8; incomplete characters are held until
  complete.
- **FR-016**: The engine MUST NOT use the network, send telemetry, or write prompts to disk.
- **FR-017**: Input MUST be bounded before tokenization (prompt 8 MiB, at most 4 stop sequences of
  256 bytes each), with errors naming the limit; limits are defined in one place.
- **FR-018**: An in-repo engine benchmark MUST measure prompt-processing speed, generation speed
  and cancellation latency at configurable depths and write results with machine and model
  identity.
- **FR-019**: For hybrid models the engine MUST capture checkpoints (deep copies of all recurrent
  state) at planned positions during prefill and on request at the current position, restore
  recurrent state from a checkpoint (attention state cut to its position), and drop checkpoints.
- **FR-020**: The engine MUST move the tail of the live state into an owned segment and back,
  one layer at a time, freeing the source as it goes, so no state is held twice.
- **FR-021**: Chat templates MUST render as Hugging Face does: `trim_blocks` and `lstrip_blocks`,
  a `tojson` equal to Python `json.dumps` (no HTML escaping, `", "`/`": "` separators, `indent`
  supported, non-ASCII kept), Python string methods and `raise_exception`.
- **FR-022**: The loaded model MUST stay resident; it is never unloaded on an idle timer.

### Key Entities

- **Model**: Loaded weights with configuration, layer kinds, tokenizer, template, EOS set and
  content identity.
- **Sequence state**: Attention K/V per attention layer plus recurrent state per linear layer of
  the live conversation; its length, allocation and reserved total.
- **Checkpoint**: A deep copy of all recurrent state at one position.
- **Segment**: State for positions `[start, end)` moved out of the live sequence (attention tail
  plus, for hybrids, recurrent state at `end`).
- **Session**: Committed tokens of the conversation and the tokens the state holds.
- **Request**: Prompt, sampling, limits, stop sequences, optional mask, cancellation; produces
  events and one `Finished`.
- **Memory budget**: Physical memory, working set, reserve, weights, scratch, recurrent state,
  state capacity in bytes and tokens, probed depth.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: For all three models, greedy outputs match mlx-lm at the pinned commit token for
  token on 100% of the reference set (≥ 20 prompts up to 8K tokens, 256 output tokens), on the same
  weights and chunk split (all but the last prompt token in 2,048-token chunks).
- **SC-002**: Generation speed is ≥ 90% of mlx-lm's on the same weights at 1K and 16K context for
  each model on the target Mac.
- **SC-003**: Prompt-processing speed is ≥ 90% of mlx-lm's (`--prefill-step-size 2048`) at 16K
  context for each model on the target Mac.
- **SC-004**: For each model, interrupting prompt processing at random points makes mrun ready for
  the next prompt within 300 ms, p95 over 50 trials per depth bucket: 16K, 64K and the model's
  maximum admissible depth (buckets above capacity skipped).
- **SC-005**: 100% of admission boundary tests pass: capacity − 1 prompt tokens completes with one
  output token; capacity tokens is rejected before processing with both numbers; an oversized
  output limit is clamped to capacity − prompt and reported.
- **SC-006**: During a 30-minute session alternating 1K and 30K-token prompts, mrun's physical
  footprint stays ≤ budget + 1 GiB and swap-outs do not increase. A 2-hour soak shows no growth in
  MLX active memory or Metal buffer count between the first and last 10 minutes. The model is
  never unloaded on an idle timer.
- **SC-007**: With the model on disk and the probe cached, `mrun run` reaches the first streamed
  token in under 15 s for Qwen3-8B on the target Mac.
- **SC-008**: Moves and checkpoints are exact: after split-off/append and after
  checkpoint-restore + re-prefill (same chunk split), next-token logits are bit-identical to the
  uninterrupted run in 100% of tests on each model; peak sequence-state memory never exceeds the
  state capacity plus one layer.
- **SC-009**: 100% of template golden fixtures (≥ 10 per model family, including tools and tool
  results) render to the same text and token IDs as Hugging Face `apply_chat_template` at the
  pinned version.

## Assumptions

- Target Mac: M4 Pro, 48 GB, current macOS. Other Apple Silicon Macs work but are not
  performance-qualified here.
- The reference is mlx-lm at `ba8a08f` with MLX v0.32.4 on the same weights; its outputs are
  recorded once and stored with the tests.
- Models are on disk in the community MLX layout (`mrun pull` is 004).
- One active sequence; queueing and APIs are 002; storing branches and planning checkpoints across
  conversations is 003. 001 keeps one conversation (with at most two checkpoints for hybrids) and
  provides the primitives.
- Linux builds compile and run engine-independent tests; the MLX engine is validated on macOS.
- Qwen3.6-35B-A3B dimensions (expected: 32 value / 16 key linear heads of 128, conv kernel 4;
  2 KV heads of 256 on attention layers) are read from `config.json` and checked at load.
