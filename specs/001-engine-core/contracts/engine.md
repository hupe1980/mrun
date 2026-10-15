# Contract: Engine trait, generation loop and `mrun run`

## Internal `Engine` trait (`crates/mrun-engine/src/engine.rs`, no MLX dependency)

```rust
pub trait Engine {
    fn info(&self) -> &ModelInfo;
    fn budget(&self) -> &MemoryBudget;

    /// Positions held by the live sequence state (attention offset = position of the
    /// position-bound state, where the family has any),
    /// including a pending pipelined token.
    fn active_len(&self) -> u32;
    /// Bytes of sequence state the engine holds: live allocation + segments + checkpoints.
    /// Invariant: allocated() <= budget().state_capacity_bytes.
    fn allocated(&self) -> u64;

    /// Lazy admission for one request: trim the live state to `keep` (same rules as `trim`) and
    /// record `total` (= prompt + clamped output limit) as the ceiling for every later
    /// allocation. Allocates nothing. Err(Capacity) if total > kv_capacity_tokens.
    /// `reserve(0, …)` always succeeds and resets position-bound state to zeros.
    fn reserve(&mut self, keep: u32, total: u32) -> Result<(), EngineError>;

    /// Append `tokens` at active_len(). Chunks follow `chunk_len(depth)` and end at every
    /// `plan.boundaries` and `plan.checkpoints` position; position-bound state is checkpointed at
    /// each `plan.checkpoints` position (zero-byte checkpoints for families without it). Memory for the whole call (growth + checkpoints) is checked
    /// first: Err(NeedsMemory{bytes}) before any work. One layer in flight; `cancel` checked after
    /// every layer. Returns only whole committed chunks: on cancel, the partial chunk is
    /// discarded (attention offset and position-bound state back to its start). When
    /// `cancel.yield_requested()` is set (read at the same per-layer points), it returns early
    /// after the current whole chunk with `yielded = true`; the caller evicts and calls `prefill`
    /// again with the remaining tokens and the same plan (same chunk ends).
    fn prefill(
        &mut self,
        tokens: &[u32],
        plan: &PrefillPlan,
        cancel: &CancelToken,
        progress: &mut dyn FnMut(u32),
    ) -> Result<Prefilled, EngineError>;

    /// Feed `last` at active_len() and enqueue the logits for the next position. If the pending
    /// pipelined step already fed exactly `last`, it is returned without recomputation; a pending
    /// step for another token is rolled back first. Err(NeedsMemory) before growing.
    fn forward(&mut self, last: u32) -> Result<Step, EngineError>;
    /// Sample from `step` (mask applied as where(mask, logprobs, -inf)). With `pipeline` true and
    /// room in the current allocation, the engine enqueues forward(sampled) before returning the
    /// token, so active_len() exceeds the fed tokens by one until the next forward or a trim.
    fn sample(
        &mut self,
        step: Step,
        sampling: &Sampling,
        mask: Option<&TokenMask>,
        pipeline: bool,
    ) -> Result<u32, EngineError>;
    /// Top-n log-probabilities of `step` (conformance margin; 002 logprobs).
    fn top_logprobs(&mut self, step: &Step, n: usize) -> Result<Vec<(u32, f32)>, EngineError>;

    /// Cut the live state to `len`. Positional state (all attention layers, sliding-window
    /// layers included): offset, exact at any len. Position-bound state: only len ==
    /// active_len() or active_len() - 1 with a pending step; otherwise Err(NotTrimmable) and
    /// nothing changes. `trim(0)` always succeeds and resets position-bound state to zeros.
    fn trim(&mut self, len: u32) -> Result<(), EngineError>;

    /// Deep copy of all position-bound state at active_len() (no pending step). Families without
    /// position-bound state return a zero-byte checkpoint. Err(NeedsMemory) before allocating.
    fn checkpoint(&mut self) -> Result<Checkpoint, EngineError>;
    /// Copy checkpoint `id` into the live position-bound state and cut positional state to its
    /// position.
    /// Requires position <= active_len(). The checkpoint stays stored.
    fn restore_checkpoint(&mut self, id: CheckpointId) -> Result<(), EngineError>;
    fn drop_checkpoint(&mut self, id: CheckpointId);

    /// Move positions [at, active_len()) out, layer by layer: copy the tail into a compact owned
    /// buffer, then (with `shrink`) reallocate the live layer to round_up(at, 256) and release the
    /// old buffer. `shrink = false` when an `append` follows at once (a switch): the live layers
    /// keep their allocation for the append, the prefix copy is skipped, and the segment's bytes
    /// are checked first (NeedsMemory). Families with position-bound state: the live
    /// position-bound state (the tail's end state) moves into the segment, and prefill/forward
    /// return Err(NoRecurrentState) until restore_checkpoint or append supplies one. Sliding-window
    /// layers split like full ones (full KV kept), so `at` may be any position.
    fn split_off(&mut self, at: u32, shrink: bool) -> Result<Segment, EngineError>;
    /// Requires active_len() == segment.start. Copies the segment in layer by layer (growth
    /// reported first); with `consume` each segment layer is freed as soon as it is copied and the
    /// segment's position-bound state is moved, otherwise both are copied and the segment stays.
    fn append(&mut self, segment: SegmentId, consume: bool) -> Result<(), EngineError>;
    fn drop_segment(&mut self, id: SegmentId);

    /// Empty the live state and release its allocation (segments and checkpoints stay).
    fn reset(&mut self);
    /// `reset` plus drop every stored segment and checkpoint (worker recovery after a panic).
    fn reset_all(&mut self);
}

pub fn chunk_len(depth: u32) -> u32; // clamp(2048 * 16384 / max(depth, 1), 128, 2048)
/// Per prefill call; more planned checkpoints in one call → Err(TooManyCheckpoints{planned, max})
/// with nothing changed. Stored checkpoints beyond the live state + 3 are evictable cache (003).
pub const MAX_PLAN_CHECKPOINTS: usize = 3;

pub struct PrefillPlan<'a> { pub boundaries: &'a [u32], pub checkpoints: &'a [u32] } // absolute, ascending
pub struct Prefilled { pub committed: u32, pub checkpoints: Vec<Checkpoint>, pub yielded: bool }
pub struct Step { /* opaque id; at most one current and one pipelined */ }
pub struct Checkpoint { pub id: CheckpointId, pub position: u32, pub bytes: u64 }
/// Allowed next tokens; `from_words` takes a 32-bit-word bitset (llguidance layout).
impl TokenMask { pub fn none_allowed(vocab: u32) -> Self; pub fn from_words(words: &[u32], vocab: u32) -> Self; }
/// Chat syntax of a family (one per family in 0.1), named by the registry in `mrun-engine`;
/// parsers live in `mrun/src/chat/syntax/` (002).
pub enum SyntaxId { Hermes, QwenXml, Harmony }
pub struct Segment { pub id: SegmentId, pub start: u32, pub end: u32, pub bytes: u64 }
pub enum EngineError {
    Capacity { requested: u32, available: u32 },
    NeedsMemory { bytes: u64 },          // shortfall; nothing was allocated or changed
    NotTrimmable { len: u32, active: u32 },
    NoRecurrentState,                    // position-bound state absent (name kept from code)
    ReservationExceeded { needed: u32, reserved: u32 },
    InvalidRange { start: u32, end: u32, len: u32 },
    TooManyCheckpoints { planned: u32, max: u32 },
    UnknownSegment(SegmentId), UnknownCheckpoint(CheckpointId), InvalidStep,
    Backend(String),
}
```

### Guarantees

- **Memory**: every allocation is preceded by a check against `state_capacity_bytes −
  allocated()`; a shortfall returns `NeedsMemory{bytes}` with state unchanged, and the caller drops
  segments or checkpoints and retries. Allocations are clamped to `min(reserved total,
  kv_capacity_tokens)`; the first growth of a request covers the input plus up to 4,096 tokens,
  later growth adds 4,096. The one-layer copy transient of growth, `split_off` and `append` lies
  inside the scratch envelope, so callers never budget it separately (003 `initial_need` does
  not).
- **Cancellation and yield**: at most one layer in flight during prefill; cancel and yield checked
  after each layer and each decode token. A cancel discards the partial chunk; a yield finishes it.
  `Prefilled.committed` counts whole chunks only.
- **Pipelining**: at most one pending step. `trim`, `reserve`, `checkpoint`, `split_off` roll a
  pending step back first (families with position-bound state keep the pre-step copy until the
  step is confirmed).
- **Exactness**: chunk ends depend only on depth and the plan, so two runs with the same tokens and
  plan compute identical splits. Checkpoints, segments and restores never alias live buffers
  (debug builds assert distinct data pointers). Sliding-window layers attend to the last `window`
  positions of the full KV, so a trimmed, moved or restored state computes exactly what an
  uninterrupted run computes at the same chunk split.
- **Family-agnostic**: nothing in this trait names a family. `ModelInfo` carries the family, layer
  kinds and byte costs; callers (budget, 003) use only those (FR-025).
- **Threading**: engines are created on the worker thread (`EngineHandle::spawn(factory)`) and
  never move; no `Send` bound.

### Implementations

`MlxEngine` (feature `mlx`) and `ScriptedEngine` (tests, any OS): deterministic tokens; models
allocation in steps with a byte cost per token and per checkpoint, `NeedsMemory`, reservation
clamping, chunk splits and per-chunk cancel, pipelining (pending step, rollback), position-bound
non-trimmability (configurable profile: positional only, or positional + position-bound), checkpoints and segments with the token ranges they
hold.

## Extensions by later features

Post-0.1 features extend this trait additively, each in its own contract: 006 speculative
decoding (`forward_draft`, `accept`, `draft_mtp`, `spec_caps`;
[contract](../../006-speculative-decoding/contracts/speculation.md)) and 007 persistent tier
(`export_views`, `import_alloc`, `import_seal`;
[contract](../../007-persistent-prefix-tier/contracts/persist.md)). Nothing in 0.1 depends on them.

## Generation loop (`crates/mrun/src/runtime/generate.rs`)

The single owner of the per-request loop, generic over `Engine` and over a reuse policy:
`Session` (001, `mrun run`) or the 003 `PrefixStore` (under `serve`). 002 T005 refactors
`runtime/handle.rs` (no new `worker.rs`): the worker becomes generic over the policy, holds a lazy
`Option<E>` built by the server's factory, takes requests from a FIFO of 16 with a wait deadline,
and handles `Reset` inside `catch_unwind`; it only dequeues and calls `generate`.

```rust
pub struct GenerateRequest {
    pub prompt: RenderedPrompt,
    pub sampling: Sampling, pub max_tokens: u32, pub stop: Vec<String>,
    pub cancel: CancelToken,
    pub hook: Option<Box<dyn DecodeHook>>, // 002 FR-036
}
/// Produced by the renderer (002 T004a); `RenderedPrompt::from_tokens` for plain prompts.
pub struct RenderedPrompt {
    pub tokens: Vec<u32>,
    pub messages: Vec<MessageMark>,   // MessageMark { start: u32, role: Role }
    pub system_tools_end: u32,
    pub generation_start: u32,        // start of the generation header; = tokens.len() if none
}
/// Per-request decode logic on the engine thread (parser, grammar, forced tokens, ignore_eos).
pub trait DecodeHook: Send {
    /// Sees every output token (sampled or forced); returns the mask for the next sampled step.
    fn on_token(&mut self, id: u32) -> Option<TokenMask>;
    /// Tokens to append next without sampling (each still goes through `on_token`).
    fn forced_tokens(&mut self) -> Vec<u32> { Vec::new() }
    /// EOS tokens become ordinary output tokens; only the output limit ends generation.
    fn ignore_eos(&self) -> bool { false }
}
pub enum Event {
    Started { prompt: u32, cached: u32, max_tokens: u32 },
    Progress { done: u32, total: u32 },
    /// One per output token id, in order: `text` is what became visible now (possibly empty),
    /// `held` the bytes of decoded text still held back by stop matching.
    Token { id: u32, text: String, held: u32 },
    /// Held text released at a finish without a stop-sequence match.
    Flush { text: String },
}
pub enum FinishReason {
    /// A stop sequence (`stop`) or the EOS token that ended generation (`eos_token`).
    Stop { stop: Option<String>, eos_token: Option<u32> },
    Length, Cancelled, Rejected(String), Error(String),
}

pub trait ReusePolicy {               // the seam of the 003 contract (authoritative)
    fn initial_need(&self, engine: &dyn Engine, prompt: &RenderedPrompt, total: u32) -> u64;
    fn evict(&mut self, engine: &mut dyn Engine, bytes: u64, protect: Option<&RenderedPrompt>) -> u64;
    fn prepare(&mut self, engine: &mut dyn Engine, prompt: &RenderedPrompt, total: u32)
        -> Result<Prepared, EngineError>;  // the policy reserves
    fn on_pressure(&mut self, engine: &mut dyn Engine, level: Pressure, idle: bool);
    fn commit(&mut self, engine: &mut dyn Engine, prompt: &RenderedPrompt, committed: &[u32],
              target: u32, captured: Vec<Checkpoint>) -> u32;
    fn clear(&mut self, engine: &mut dyn Engine);
}
pub struct Prepared { pub resume: u32, pub chunks: ChunkPlan } // 003 T008 adds `miss`
pub enum Pressure { Warning, Critical }
/// Edge-triggered level shared with the pressure source: `raise` records a level and requests a
/// yield of the running request; the loop `take`s it.
pub struct PressureGauge;
```

1. Validate (empty prompt, limit 0, sampling, stop caps) → `Rejected`. The TTFT clock starts
   when `generate` is called, i.e. at the worker's dequeue.
2. Admission (FR-009): reject unless `prompt + 1 ≤ kv_capacity_tokens`; clamp the limit;
   `total = prompt + limit`. Policy state is unchanged on rejection.
3. `need = initial_need(prompt, total)`; when it exceeds the free bytes
   (`state_capacity_bytes − allocated()`), `evict(need − free, Some(prompt))`.
4. `prepare(prompt, total)`: `Session` takes the common prefix of `session.tokens` and the prompt
   (≤ prompt − 1); models with position-bound state that cannot trim there restore the deepest
   session checkpoint ≤ common; then `reserve(resume, total)`. The plan checkpoints
   `generation_start` (models with position-bound state; the session keeps the two newest).
   `PrefixStore` runs its lookup and switch (003).
5. Prefill all but the last prompt token with `Prepared.chunks`. On `NeedsMemory{bytes}` →
   `evict(bytes, None)` and retry while it frees something (`Session`: the older session
   checkpoint first), else fail with an error. On `yielded`, a pending pressure level is passed to
   `on_pressure(level, idle = false)`; then `prefill` continues with the remaining tokens and the
   same plan.
6. Decode: a pending pressure level → `on_pressure`; `forward(last)` → the next forced token, or
   `sample(step, …, mask from the last on_token, pipeline = remaining > 1)`; EOS (unless
   `ignore_eos`) finishes with `Stop{eos_token}`; otherwise `on_token(id)`, detokenize,
   stop-match, emit `Token{id, text, held}`. TTFT = first sampled or forced token.
7. Finish: committed = prompt + tokens whose text was fully emitted; trim the state to the longest
   committed prefix it holds (position-bound state falls back to a checkpoint); the policy records
   the state tokens (`session.tokens`, or 003 `commit`).

Events go through `flume::bounded(64)`; the worker sends with a stall timeout, then stops sending
`Token`/`Progress` to that consumer and cancels. The event sender is dropped at finish and
`Finished` is sent on its own one-slot channel with `try_send` (never blocks, never lost); the
consumer drains events, then reads `Finished`. The worker wraps each request in `catch_unwind`;
a panic yields `Finished{Error}` and `reset_all()` (the session is cleared with it).

`ScriptedEngine` and the test knobs (`chat::tokenizer::test_support`) are compiled only for unit
tests and with the `test-support` cargo feature of `mrun`, which its integration tests enable.

## CLI: `mrun run`

```text
mrun run <model> [--prompt TEXT] [--system TEXT] [--temperature F] [--top-p F] [--top-k N]
                 [--min-p F] [--seed N] [--max-tokens N] [--stop S]... [--think]
                 [--memory-fraction F] [--reserve-gib N] [--allow-over] [--info] [--check]
```

- `<model>`: local path or HF repo id; aliases arrive with 004. An unsupported `model_type` exits
  3 (message: 004 E034).
- `--think` per family (002 FR-035): Qwen3 and Qwen3.6 set `enable_thinking`; gpt-oss sets
  `RenderOptions.reasoning_effort` `high` (default `low`); Qwen3-Coder prints
  `thinking-unsupported` and runs without thinking.
- Each turn is rendered with and without the generation prompt; `generation_start` is their
  common token prefix (`chat::prompt::generation_start`), so hybrids checkpoint where the next
  turn's re-rendered reply diverges.
- Interactive REPL without `--prompt`; `/clear`, `/info`, `/exit`.
- Stats line per answer: `42 tok · 1.2 s TTFT · 512 tok/s prefill · 61 tok/s decode · 3,104 cached`.
- `--info`: budget breakdown (FR-006/008) incl. current free memory, then exit.
- `--check`: conformance set for this model; non-zero exit at the first mismatch with prompt id,
  position and log-probability margin.
- Input caps (`chat::limits`): prompt ≤ 8 MiB, ≤ 4 stop sequences of ≤ 256 bytes.
- Ctrl-C cancels the current generation; a second Ctrl-C within 1 s exits.
- Exit codes follow the [004 CLI contract](../../004-models-cli/contracts/cli.md).
