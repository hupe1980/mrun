# Contract: Prefix Cache

## Reuse-policy seam (`crates/mrun/src/runtime/generate.rs`, owned by 001)

001's generation loop is generic over a reuse policy: `Session` for `mrun run`, `PrefixStore` under
`serve`. The seam (reshaped 2026-10-15, lean 0.1 cut; 001 adopts it in its contract):

```rust
pub struct GenerateRequest {
    pub prompt: RenderedPrompt,          // 002: tokens, message marks (start, role),
                                         // system_tools_end, generation_start
    pub sampling: Sampling, pub max_tokens: u32, pub stop: Vec<String>,
    pub cancel: CancelToken,
    pub hook: Option<Box<dyn DecodeHook>>, // 002 FR-036: parser, grammar mask, forced tokens, ignore_eos
}

pub trait ReusePolicy {
    /// Bytes to free before `prepare`: attention growth to `P + min(out, 4K)` beyond what the
    /// engine will hold after the switch, plus planned checkpoints. The one-layer move transient
    /// is inside 001's scratch envelope and not counted. Pure (lookup only).
    fn initial_need(&self, engine: &dyn Engine, prompt: &RenderedPrompt, total: u32) -> u64;
    /// Evict LRU units until `bytes` are free; never touches the live path in use or the state
    /// `protect` matches. Returns the bytes freed (0: nothing left to evict).
    fn evict(&mut self, engine: &mut dyn Engine, bytes: u64, protect: Option<&RenderedPrompt>) -> u64;
    /// Lookup, switch (move the old tail out or discard < 256, move matched buffers in, split
    /// back), trim or restore a checkpoint, then `reserve(keep = resume, total)`: the policy
    /// reserves. Nothing is pinned afterwards.
    fn prepare(&mut self, engine: &mut dyn Engine, prompt: &RenderedPrompt, total: u32)
        -> Result<Prepared, EngineError>;
    /// The loop read a memory-pressure level: at a prefill yield (whole-chunk end) or between
    /// decode tokens. Evicts per research R7; resets the live path only when `idle`.
    fn on_pressure(&mut self, engine: &mut dyn Engine, level: Pressure, idle: bool);
    /// Record committed tokens (finish or cancel), message marks and captured checkpoints; merge
    /// overlap with stored branches; enforce ≤ 8 checkpoints per path (drop the shallowest but
    /// the first). Returns the length the state holds.
    fn commit(&mut self, engine: &mut dyn Engine, prompt: &RenderedPrompt, committed: &[u32],
              target: u32, captured: Vec<Checkpoint>) -> u32;
    /// Forget all state (engine error or panic).
    fn clear(&mut self, engine: &mut dyn Engine);
}

pub struct Prepared {
    pub resume: u32,             // tokens the state holds; prefill starts here
    pub chunks: ChunkPlan,       // chunk ends at selected message boundaries (R4) + checkpoint positions
    pub miss: Option<MissReport>,// None for Session
}
```

## `PrefixStore` (`crates/mrun/src/cache/mod.rs`, engine worker thread only)

```rust
impl ReusePolicy for PrefixStore { /* as above */ }
impl PrefixStore {
    fn new(key: CacheKey) -> Self;
    /// Longest exact match over the live path and branches, the switch plan and the miss report.
    /// Pure: changes only `last_used`. Used by `initial_need` and `prepare`.
    fn lookup(&mut self, prompt: &RenderedPrompt) -> Plan;
    /// `mrun cache clear`: drop branches and checkpoints; reset the live path now if idle, else
    /// after the request.
    fn clear_cache(&mut self, idle: bool, engine: &mut dyn Engine);
    fn stats(&self) -> CacheStats;
}
```

## Engine primitives used ([001 contract](../../001-engine-core/contracts/engine.md) is authoritative)

```rust
fn prefill(&mut self, tokens: &[u32], plan: &PrefillPlan, cancel: &CancelToken,
           progress: &mut dyn FnMut(u32)) -> Result<Prefilled, EngineError>;
pub struct Prefilled { pub committed: u32, pub checkpoints: Vec<Checkpoint>, pub yielded: bool }
```

Also `allocated()`, `reserve(keep, total)` (lazy), forward/sample split, `split_off(at, shrink)
→ Segment`, `append(segment, consume)`, `checkpoint()`, `restore_checkpoint(id)`, `trim`, `reset`,
`drop_segment`, `drop_checkpoint`. Every allocating call returns `Err(NeedsMemory{bytes})` with
nothing changed: the loop calls `evict(bytes, None)` and retries the call.

## Loop order per request (001 `generate`)

002's worker (`runtime/handle.rs`) only dequeues and calls `generate` with the server's policy.

1. Validate; admission (001): reject if `P ≥ capacity`; clamp output to `capacity − P`. Cache
   unchanged on rejection.
2. `need = initial_need(prompt, total)`; `evict(need, Some(prompt))` when the engine lacks it.
3. `prepare(prompt, total)` (`split_off(at, shrink = false)` when an `append` follows; mode Off:
   the whole live path is moved out, resume 0).
4. Prefill `prompt[resume..P−1]` with `Prepared.chunks`; on `Err(NeedsMemory{bytes})` →
   `evict(bytes, None)` and retry. The pressure level is read at every per-layer check point;
   when set, the loop requests a yield, and on `Prefilled.yielded` calls `on_pressure(level,
   idle = false)` and continues with the remaining tokens and the same plan.
5. Decode with the request's `DecodeHook`; same `NeedsMemory` handling and `on_pressure` per
   token.
6. `commit` (`--no-cache`: reset afterwards).

## User-visible surface

- Usage: `cached = min(resume, P − 1)` per API (002 data model).
- mrun metadata (002): `x-mrun-cache-cached`, `-processed`, `-cause`, `-divergence`, `-message`
  (`<index>:<role>`), `-gap` headers; the same fields in `: mrun {json}` SSE comments; `last` in
  `/mrun/status`.
- `mrun ps`: CacheStats and the last miss report.
- `mrun cache clear` → `POST /mrun/cache/clear` → `{branches_dropped, bytes_freed, live_reset: "now" | "after-request"}`.
- `--no-cache`. (`x-mrun-cache` and `--debug-cache`: removed, 0.1 scope cut, 2026-10-15.)
