# Feature Specification: Engine Core

**Feature Branch**: `001-engine-core`

**Created**: 2026-10-14

**Status**: Draft

**Input**: User description: "Engine core: run a pulled local coding model on an Apple Silicon Mac from the mrun binary. `mrun run <model>` streams generated text in the terminal; the engine owns the model's attention state with chunked prefill so a long prompt can be cancelled within one chunk; a startup memory budget sizes weights + KV + measured scratch and rejects prompts that cannot fit with the numbers; greedy output must match the reference implementation token-for-token; decode speed within 10% of the reference on the same weights. First models: Qwen3-8B-4bit (reference) and Qwen3-Coder-30B-A3B-Instruct-4bit (MoE). Sampling: temperature, top-p, top-k, min-p, seed, stop sequences, max tokens. Engine runs on a dedicated thread; async callers stream tokens through a bounded channel."

## Clarifications

### Session 2026-10-14

- Q: What prompt-processing chunk size meets the 300 ms interruption target? → A: Default 512 tokens (configurable 128–2048); the plan validates it against SC-004 and prompt speed (SC-003) with `mrun bench`.
- Q: Should Qwen3 hybrid models think by default in `mrun run`? → A: No thinking unless requested (`--think`); Qwen3-Coder has no thinking mode. (Maintainer decision.)
- Q: How many conversations run at once? → A: One active generation; others queue (feature 002). Parallel decoding is a later feature. (Maintainer decision.)
- Q: Where do models come from? → A: The shared Hugging Face cache (honoring `HF_HOME`), plus explicit local paths. (Maintainer decision; pulling is feature 004.)

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Chat with a local model in the terminal (Priority: P1)

A developer with a model already on disk runs `mrun run <model>`, types a prompt, and sees the
answer stream token by token. Follow-up prompts continue the same conversation.

**Why this priority**: It is the smallest end-to-end proof that mrun can load and run a model on
the Mac. Every later feature (APIs, prefix reuse, benchmarks) builds on this engine.

**Independent Test**: On an Apple Silicon Mac with the reference model on disk, run
`mrun run <model>`, ask a question, and observe streamed output and a final summary line with
token counts and speeds.

**Acceptance Scenarios**:

1. **Given** a supported model on disk, **When** the developer runs `mrun run <model>` and enters
   a prompt, **Then** the first token appears and the answer streams until the model stops or
   reaches the token limit.
2. **Given** an active conversation, **When** the developer enters a follow-up, **Then** the model
   answers with the earlier turns as context, and only the new turn is processed.
3. **Given** a model name or path that does not exist or is not supported, **When** the developer
   runs `mrun run`, **Then** mrun exits with a message naming the problem and the next step.

---

### User Story 2 - Interrupt instantly (Priority: P1)

The developer presses Ctrl-C while a long prompt is being processed or while the answer streams.
Generation stops almost immediately and the prompt is ready for new input.

**Why this priority**: Agents and humans change direction constantly; waiting for abandoned work
is the most frustrating latency in interactive use.

**Independent Test**: Submit a prompt of at least 16K tokens, interrupt during prompt processing
and again during generation, and measure the time until mrun is ready for the next prompt.

**Acceptance Scenarios**:

1. **Given** a long prompt being processed, **When** the developer interrupts, **Then** processing
   stops after at most one processing chunk and no answer is emitted for that prompt.
2. **Given** an answer streaming, **When** the developer interrupts, **Then** streaming stops after
   at most one more token and the conversation keeps the tokens already shown.
3. **Given** an interrupted request, **When** the developer submits the next prompt, **Then** it
   runs normally and its output is unaffected by the interrupted work.

---

### User Story 3 - Never overcommit memory (Priority: P1)

mrun decides at startup how much memory it may use, loads the model only if it fits, and refuses
prompts that would exceed the attention-state budget, saying how much is needed and available.

**Why this priority**: Unified memory is shared with the editor, browser and agent. Swapping or
being killed by the OS destroys the experience and the user's trust.

**Independent Test**: Start mrun with a deliberately small memory limit, submit prompts just below
and just above the computed token capacity, and confirm acceptance and a clear rejection.

**Acceptance Scenarios**:

1. **Given** a model larger than the memory budget, **When** the developer runs it, **Then** mrun
   refuses before loading, stating model size, budget and how to change the budget.
2. **Given** a loaded model, **When** a prompt plus requested output exceeds the token capacity,
   **Then** mrun rejects it before processing, stating requested tokens and capacity.
3. **Given** a prompt exactly at the capacity, **When** submitted, **Then** it runs to completion.
4. **Given** `mrun run --info`, **When** the model loads, **Then** mrun prints the budget breakdown:
   weights, attention-state capacity in tokens and bytes, scratch reserve, and headroom.

---

### User Story 4 - Trustworthy output (Priority: P2)

A maintainer verifies that mrun produces exactly the same greedy output as the reference
implementation on the same weights, and that sampling controls behave as documented.

**Why this priority**: Exactness is the foundation for prefix reuse (feature 003) and for fair
benchmarks; subtle numerical drift would silently degrade coding quality.

**Independent Test**: Run the conformance command on the reference prompt set for each supported
model and compare token IDs against stored reference outputs.

**Acceptance Scenarios**:

1. **Given** the reference prompt set and temperature 0, **When** the conformance check runs,
   **Then** every generated token ID matches the stored reference output.
2. **Given** a fixed seed and sampling settings, **When** the same prompt runs twice, **Then** the
   outputs are identical.
3. **Given** stop sequences, **When** the model emits one, **Then** output ends before the stop
   sequence and the finish reason says "stop".

### Edge Cases

- Empty prompt, or max tokens of zero: rejected with a clear message.
- Prompt that fills the capacity exactly, leaving room for the requested output: accepted.
- A stop sequence split across two tokens: detected; partial matches are not emitted early.
- Output that ends mid-way through a multi-byte character: no invalid text is shown.
- Interrupt arriving just as generation completes: exactly one outcome is reported.
- Model files changed on disk after load: mrun keeps serving the loaded weights and reports the
  mismatch on next load.
- Other applications consume memory after startup: mrun reports reduced headroom in `--info` and
  later features shed cache first (feature 003).
- An unsupported model architecture or quantization: rejected at load with the reason.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: `mrun run <model>` MUST load a supported model from a local path or a pulled model
  name and start an interactive streaming chat session.
- **FR-002**: The engine MUST support the Qwen3 dense family (reference: Qwen3-8B, 4-bit) and the
  Qwen3 MoE family (primary: Qwen3-Coder-30B-A3B-Instruct, 4-bit) in the standard community
  quantized format, and MUST reject other architectures with a named reason.
- **FR-003**: The engine MUST keep each conversation's attention state between turns so a follow-up
  processes only the new tokens.
- **FR-004**: Prompt processing MUST run in fixed-size chunks (default 512 tokens) so that
  cancellation and memory peaks are bounded per chunk.
- **FR-005**: Cancellation MUST stop prompt processing within one chunk and generation within one
  token, and MUST leave the engine ready for the next request.
- **FR-006**: At startup the engine MUST compute a memory budget as the lesser of a fraction of
  physical memory (default 75%) and the GPU's recommended working set, minus a host reserve
  (default 8 GiB), overridable downward by the user and upward only with an explicit flag.
- **FR-007**: The engine MUST refuse to load a model whose weights plus minimum attention state plus
  scratch reserve exceed the budget, and MUST state each number.
- **FR-008**: The engine MUST measure peak scratch memory with a worst-case processing chunk at
  startup and size attention-state capacity from what remains.
- **FR-009**: Each request MUST reserve attention state for its full prompt plus requested output
  before processing, and MUST be rejected up front if that reservation cannot be made.
- **FR-010**: Sampling MUST support temperature, top-p, top-k, min-p, seed, stop sequences and max
  tokens; temperature 0 MUST be greedy (argmax).
- **FR-011**: The engine MUST run model work off the async runtime on a dedicated worker and deliver
  tokens through a bounded channel; a slow consumer MUST NOT block other engine work indefinitely
  (generation for that request is cancelled after a configurable stall timeout, default 10 s).
- **FR-012**: Every completed request MUST report prompt tokens, generated tokens, time to first
  token, prompt-processing speed, generation speed and finish reason (stop, length, cancelled,
  error).
- **FR-013**: A conformance command MUST compare greedy token IDs against stored reference outputs
  for each supported model and report the first mismatch position.
- **FR-014**: Streamed text MUST always be valid UTF-8; incomplete characters are held until
  complete.
- **FR-015**: The engine MUST NOT download anything, send telemetry, or write prompts to disk.

### Key Entities

- **Model**: A loaded set of weights with its configuration, tokenizer, chat template and content
  hash; identifies everything that determines output.
- **Session**: One conversation's token history and attention state, with its capacity reservation.
- **Request**: A prompt, sampling settings, limits and a cancellation signal; produces a token
  stream and a final report.
- **Memory Budget**: Physical memory, GPU working-set recommendation, reserve, weights, scratch
  reserve and attention-state capacity, computed at startup.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: For both supported models, greedy outputs match the reference implementation token
  for token on 100% of the reference prompt set (at least 20 prompts, up to 8K tokens).
- **SC-002**: Generation speed on the same weights is at least 90% of the reference
  implementation's, measured on the target Mac at 1K and 16K context.
- **SC-003**: Prompt-processing speed on the same weights is at least 90% of the reference
  implementation's at 16K context.
- **SC-004**: Interrupting a 16K-token prompt makes mrun ready for the next prompt within 300 ms
  (p95 over 50 trials) on the target Mac.
- **SC-005**: Prompts at exactly the computed capacity complete; prompts one token over are rejected
  before processing, in 100% of boundary tests.
- **SC-006**: During a 30-minute session alternating 1K and 30K-token prompts, the process never
  exceeds its reported budget by more than the scratch reserve, and the system does not swap
  because of mrun.
- **SC-007**: A developer with the model on disk goes from `mrun run` to the first streamed token in
  under 15 seconds for the 8B model on the target Mac.

## Assumptions

- Target Mac: Apple Silicon M4 Pro with 48 GB unified memory, current macOS. Other Apple Silicon
  Macs should work but are not performance-qualified in this feature.
- The reference implementation is the upstream MLX language-model library at a pinned version,
  run on the same weights; its outputs are recorded once and stored with the tests.
- Models are already on disk (from `mrun pull`, feature 004, or a manual download into the standard
  community layout). Downloading is out of scope here.
- One active generation at a time; concurrency and queuing arrive with the API server (feature 002).
- Tool calling, structured output and the HTTP APIs are feature 002; cross-conversation prefix
  reuse is feature 003. This feature keeps one conversation's state across its own turns only.
- Linux builds compile and run engine-independent tests; the engine itself is validated on macOS.
