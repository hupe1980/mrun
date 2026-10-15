# Contract: Engine trait, generation loop and `mrun run`

## Internal `Engine` trait (`crates/mrun-engine/src/engine.rs`, no MLX dependency)

```rust
pub trait Engine {
    fn info(&self) -> &ModelInfo;
    fn budget(&self) -> &MemoryBudget;

    /// Positions held by the live sequence state (attention offset = recurrent position),
    /// including a pending pipelined token.
    fn active_len(&self) -> u32;
    /// Bytes of sequence state the engine holds: live allocation + segments + checkpoints.
    /// Invariant: allocated() <= budget().state_capacity_bytes.
    fn allocated(&self) -> u64;

    /// Lazy admission for one request: trim the live state to `keep` (same rules as `trim`) and
    /// record `total` (= prompt + clamped output limit) as the ceiling for every later
    /// allocation. Allocates nothing. Err(Capacity) if total > kv_capacity_tokens.
    /// `reserve(0, …)` always succeeds and resets recurrent state to zeros.
    fn reserve(&mut self, keep: u32, total: u32) -> Result<(), EngineError>;

    /// Append `tokens` at active_len(). Chunks follow `chunk_len(depth)` and end at every
    /// `plan.boundaries` and `plan.checkpoints` position; recurrent state is checkpointed at each
    /// `plan.checkpoints` position. Memory for the whole call (growth + checkpoints) is checked
    /// first: Err(NeedsMemory{bytes}) before any work. One layer in flight; `cancel` checked after
    /// every layer. Returns only whole committed chunks: on cancel, the partial chunk is
    /// discarded (attention offset and recurrent state back to its start). When
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

    /// Cut the live state to `len`. Attention: offset. Recurrent: only len == active_len() or
    /// active_len() - 1 with a pending step; otherwise Err(NotTrimmable) and nothing changes.
    /// `trim(0)` always succeeds and resets recurrent state to zeros.
    fn trim(&mut self, len: u32) -> Result<(), EngineError>;

    /// Deep copy of all recurrent state at active_len() (no pending step). Attention-only models
    /// return a zero-byte checkpoint. Err(NeedsMemory) before allocating.
    fn checkpoint(&mut self) -> Result<Checkpoint, EngineError>;
    /// Copy checkpoint `id` into the live recurrent state and cut attention to its position.
    /// Requires position <= active_len(). The checkpoint stays stored.
    fn restore_checkpoint(&mut self, id: CheckpointId) -> Result<(), EngineError>;
    fn drop_checkpoint(&mut self, id: CheckpointId);

    /// Move positions [at, active_len()) out, layer by layer: copy the tail into a compact owned
    /// buffer, then (with `shrink`) reallocate the live layer to round_up(at, 256) and release the
    /// old buffer. `shrink = false` when an `append` follows at once (a switch): the live layers
    /// keep their allocation for the append, the prefix copy is skipped, and the segment's bytes
    /// are checked first (NeedsMemory). Hybrids: the live recurrent state (the tail's end state)
    /// moves into the segment, and prefill/forward return Err(NoRecurrentState) until
    /// restore_checkpoint or append supplies one.
    fn split_off(&mut self, at: u32, shrink: bool) -> Result<Segment, EngineError>;
    /// Requires active_len() == segment.start. Copies the segment in layer by layer (growth
    /// reported first); with `consume` each segment layer is freed as soon as it is copied and the
    /// segment's recurrent state is moved, otherwise both are copied and the segment stays.
    fn append(&mut self, segment: SegmentId, consume: bool) -> Result<(), EngineError>;
    fn drop_segment(&mut self, id: SegmentId);

    /// Empty the live state and release its allocation (segments and checkpoints stay).
    fn reset(&mut self);
}

pub fn chunk_len(depth: u32) -> u32; // clamp(2048 * 16384 / max(depth, 1), 128, 2048)
/// Per prefill call. Stored checkpoints beyond the live state + 3 are evictable cache (003).
pub const MAX_PLAN_CHECKPOINTS: usize = 3;

pub struct PrefillPlan<'a> { pub boundaries: &'a [u32], pub checkpoints: &'a [u32] } // absolute, ascending
pub struct Prefilled { pub committed: u32, pub checkpoints: Vec<Checkpoint>, pub yielded: bool }
pub struct Step { /* opaque id; at most one current and one pipelined */ }
pub struct Checkpoint { pub id: CheckpointId, pub position: u32, pub bytes: u64 }
pub struct Segment { pub id: SegmentId, pub start: u32, pub end: u32, pub bytes: u64 }
pub enum EngineError {
    Capacity { requested: u32, available: u32 },
    NeedsMemory { bytes: u64 },          // shortfall; nothing was allocated or changed
    NotTrimmable { len: u32, active: u32 },
    NoRecurrentState,
    ReservationExceeded { needed: u32, reserved: u32 },
    InvalidRange { start: u32, end: u32, len: u32 },
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
  pending step back first (hybrids keep the pre-step recurrent state until the step is confirmed).
- **Exactness**: chunk ends depend only on depth and the plan, so two runs with the same tokens and
  plan compute identical splits. Checkpoints, segments and restores never alias live buffers
  (debug builds assert distinct data pointers).
- **Threading**: engines are created on the worker thread (`EngineHandle::spawn(factory)`) and
  never move; no `Send` bound.

### Implementations

`MlxEngine` (feature `mlx`) and `ScriptedEngine` (tests, any OS): deterministic tokens; models
allocation in steps with a byte cost per token and per checkpoint, `NeedsMemory`, reservation
clamping, chunk splits and per-chunk cancel, pipelining (pending step, rollback), recurrent
non-trimmability (configurable hybrid mode), checkpoints and segments with the token ranges they
hold.

## Extensions by later features

Post-0.1 features extend this trait additively, each in its own contract: 006 speculative
decoding (`forward_draft`, `accept`, `draft_mtp`, `spec_caps`;
[contract](../../006-speculative-decoding/contracts/speculation.md)) and 007 persistent tier
(`export_views`, `import_alloc`, `import_seal`;
[contract](../../007-persistent-prefix-tier/contracts/persist.md)). Nothing in 0.1 depends on them.

## Generation loop (`crates/mrun/src/runtime/generate.rs`)

The single owner of the per-request loop, generic over `Engine` and over a reuse policy:
`Session` (001, `mrun run`) or the 003 `PrefixStore` (under `serve`).
002's `runtime/worker.rs` only dequeues and calls `generate`.

1. Validate (empty prompt, limit 0, sampling, stop caps) → `Rejected`.
2. Admission (FR-009): reject unless `prompt + 1 ≤ kv_capacity_tokens`; clamp the limit.
3. Reuse (policy): `Session` takes the common prefix of `session.tokens` and prompt
   (≤ prompt − 1); hybrids that cannot trim there restore the deepest session checkpoint ≤ common.
   `PrefixStore` runs its lookup and switch (003). Then `reserve(keep, prompt + limit)`.
4. Prefill all but the last prompt token; hybrids plan a checkpoint at `generation_start` (the
   start of the generation header; the session keeps the two newest). On `NeedsMemory{bytes}` the
   policy evicts (`Session`: drop the older session checkpoint, else fail with an error;
   `PrefixStore`: `evict(bytes)`) and the call is retried. On `yielded` the policy acts on memory
   pressure (and 003 applies pending soft pins), then `prefill` is called again with the remaining
   tokens and the same plan.
5. Decode: `forward(last)` → optional mask → `sample(step, …, pipeline = remaining > 1)`;
   detokenize, stop-match, emit. TTFT = first sampled token.
6. Finish: committed = prompt + tokens whose text was fully emitted; trim the state to the longest
   committed prefix it holds (hybrids fall back to a checkpoint); the policy records the state
   tokens (`session.tokens`, or 003 `commit`).

Events go through `flume::bounded(64)`; the worker sends with a stall timeout, then stops sending
`Token`/`Progress` to that consumer and cancels. The event sender is dropped at finish and
`Finished` is sent on its own one-slot channel with `try_send` (never blocks, never lost); the
consumer drains events, then reads `Finished`. The worker wraps each request in `catch_unwind`;
a panic yields `Finished{Error}` and `reset()`.

## CLI: `mrun run`

```text
mrun run <model> [--prompt TEXT] [--system TEXT] [--temperature F] [--top-p F] [--top-k N]
                 [--min-p F] [--seed N] [--max-tokens N] [--stop S]... [--think]
                 [--memory-fraction F] [--reserve-gib N] [--allow-over] [--info] [--check]
```

- `<model>`: local path or HF repo id; aliases arrive with 004.
- Interactive REPL without `--prompt`; `/clear`, `/info`, `/exit`.
- Stats line per answer: `42 tok · 1.2 s TTFT · 512 tok/s prefill · 61 tok/s decode · 3,104 cached`.
- `--info`: budget breakdown (FR-006/008) incl. current free memory, then exit.
- `--check`: conformance set for this model; non-zero exit at the first mismatch with prompt id,
  position and log-probability margin.
- Input caps (`chat::limits`): prompt ≤ 8 MiB, ≤ 4 stop sequences of ≤ 256 bytes.
- Ctrl-C cancels the current generation; a second Ctrl-C within 1 s exits.
- Exit codes follow the [004 CLI contract](../../004-models-cli/contracts/cli.md).
