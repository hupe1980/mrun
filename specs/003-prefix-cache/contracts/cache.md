# Contract: Prefix Cache

## Internal API (`crates/mrun/src/cache/mod.rs`, engine worker thread only)

```rust
impl PrefixStore {
    fn new(key: CacheKey) -> Self;
    /// Longest exact match over the live path and branches, the switch plan and the miss report.
    /// Pure: changes only `last_used`.
    fn lookup(&mut self, prompt: &Prompt) -> Plan;
    /// Queued requests: evicted last while waiting; released on dequeue or cancel. Pins requested
    /// at enqueue are pending until the worker applies them (between requests, at each yield).
    fn soft_pin(&mut self, plan: &Plan) -> SoftPin;
    fn release(&mut self, pin: SoftPin);
    /// Bytes to free before `prepare`: attention growth to `P + min(out, 4K)` beyond what the
    /// engine has allocated after the switch, plus planned checkpoints. The one-layer move
    /// transient is inside 001's scratch envelope and not counted.
    fn initial_need(&self, plan: &Plan, out_limit: u32, engine: &dyn Engine) -> u64;
    /// Evict LRU units (R6) until `bytes` are free; never touches the live path in use or `plan`.
    fn evict(&mut self, bytes: u64, protect: Option<&Plan>, engine: &mut dyn Engine) -> u64;
    /// Switch: move the old tail out (or discard < 256), move matched buffers in, split back,
    /// trim or restore checkpoint, `reserve(keep = resume, total)`. Nothing pinned afterwards.
    fn prepare(&mut self, plan: Plan, total: u32, engine: &mut dyn Engine) -> Result<Prepared, EngineError>;
    /// 001 `PrefillPlan` from `resume`: `boundaries` holds only the chunk ends R4 selects
    /// (message boundaries it chose), `checkpoints` the planned checkpoint positions.
    fn prefill_plan(&self, prepared: &Prepared, prompt: &Prompt) -> PrefillPlan;
    /// Record committed tokens (finish or cancel), message marks, captured checkpoints; merge
    /// overlap with stored branches; enforce ≤ 8 checkpoints per path (R3 drop rule; a moved-out
    /// tail's end state lives in its segment and is not counted).
    fn commit(&mut self, prompt: &Prompt, committed: &Committed, engine: &mut dyn Engine);
    fn on_pressure(&mut self, level: Pressure, idle: bool, engine: &mut dyn Engine);
    /// Drop branches and checkpoints; reset the live path now if idle, else after the request.
    fn clear(&mut self, idle: bool, engine: &mut dyn Engine);
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
nothing changed: the worker calls `evict(bytes)` and retries the call.

## Worker order per request

The per-request loop is 001's `runtime/generate.rs`; under `serve` it takes the `PrefixStore` as
its reuse policy in place of 001's `Session`. 002's `runtime/worker.rs` only dequeues and calls
`generate`.

1. On enqueue: `lookup` + `soft_pin` (pending). The worker applies pending soft pins between
   requests and at each prefill yield. On dequeue: `release`, then `lookup` again.
2. Admission (001): reject if `P ≥ capacity`; clamp output to `capacity − P`. Cache unchanged on
   rejection.
3. `evict(initial_need)` protecting the plan.
4. `prepare` (`split_off(at, shrink = false)` when an `append` follows). Mode Off: the whole live
   path is moved out (fork 0), resume 0.
5. Prefill per `prefill_plan`; on `Err(NeedsMemory{bytes})` the worker calls `evict(bytes)` and
   retries the call. The pressure level is read at every per-layer check point; when set, the
   worker requests a yield, and on `Prefilled.yielded` it runs `on_pressure`, applies pending soft
   pins and calls `prefill` again with the remaining tokens and the same plan.
6. Decode: same `NeedsMemory` and pressure handling per token.
7. `commit` (mode Off: discard this request's state afterwards; `--no-cache`: reset).

## User-visible surface

- Usage: `cached = min(resume, P − 1)` per API (002 data model).
- mrun metadata (002): `x-mrun-cache-cached`, `-processed`, `-divergence`, `-gap`, `-cause` headers;
  the same fields in `: mrun {json}` SSE comments; `last` in `/mrun/status`.
- `mrun ps`: CacheStats and the last miss report.
- `mrun cache clear` → `POST /mrun/cache/clear` → `{branches_dropped, bytes_freed, live_reset: "now" | "after-request"}`.
- `--no-cache`; header `x-mrun-cache: off` (all APIs; other values ignored); `--debug-cache`
  (refused with a non-loopback bind; excerpts on the server terminal only).
