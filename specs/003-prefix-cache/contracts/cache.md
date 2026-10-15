# Contract: Prefix Store

## Internal API (`crates/mrun/src/cache/mod.rs`, engine worker thread only)

```rust
impl PrefixStore {
    fn new(key: CacheKey) -> Self;
    /// Longest exact match over the live path and branches, the switch plan and the miss report.
    /// Pure: changes only `last_used`.
    fn lookup(&mut self, prompt: &Prompt) -> Plan;
    /// Queued requests: evicted last while waiting; released on dequeue or cancel.
    fn soft_pin(&mut self, plan: &Plan) -> SoftPin;
    fn release(&mut self, pin: SoftPin);
    /// Bytes to free before `prepare`: attention growth to `P + min(out, 4K)` beyond what the
    /// engine has allocated after the switch, one layer of the largest moved buffer, planned
    /// checkpoints.
    fn initial_need(&self, plan: &Plan, out_limit: u32, engine: &dyn Engine) -> u64;
    /// Evict LRU units (R6) until `bytes` are free; never touches the live path in use or `plan`.
    fn evict(&mut self, bytes: u64, protect: Option<&Plan>, engine: &mut dyn Engine) -> u64;
    /// Switch: move the old tail out (or discard < 256), move matched buffers in, split back,
    /// trim or restore checkpoint, `reserve(keep = resume, total)`. Nothing pinned afterwards.
    fn prepare(&mut self, plan: Plan, total: u32, engine: &mut dyn Engine) -> Result<Prepared, EngineError>;
    /// Chunk ends and checkpoint positions from `resume` (R4).
    fn prefill_plan(&self, prepared: &Prepared, prompt: &Prompt) -> PrefillPlan;
    /// Record committed tokens (finish or cancel), message marks, captured checkpoints; merge
    /// overlap with stored branches; enforce ≤ 8 checkpoints per path.
    fn commit(&mut self, prompt: &Prompt, committed: &Committed, engine: &mut dyn Engine);
    fn on_pressure(&mut self, level: Pressure, idle: bool, engine: &mut dyn Engine);
    /// Drop branches and checkpoints; reset the live path now if idle, else after the request.
    fn clear(&mut self, idle: bool, engine: &mut dyn Engine);
    fn stats(&self) -> CacheStats;
}
```

## Engine primitives used (001 contract is authoritative for names)

`allocated()`, `reserve(keep, total)` (lazy), growth notification before each allocation,
`prefill(tokens, plan, cancel)` returning whole committed chunks and checkpoint ids,
forward/sample split, `split_off(at) → Segment`, `append(segment, consume)`, `checkpoint()`,
`restore_checkpoint(id)`, `trim`, `reset`, drop of segments and checkpoints.

## Worker order per request

1. On enqueue: `lookup` + `soft_pin`. On dequeue: `release`, then `lookup` again.
2. Admission (001): reject if `P ≥ capacity`; clamp output to `capacity − P`. Store unchanged on
   rejection.
3. `evict(initial_need)` protecting the plan.
4. `prepare`. Mode Off: the whole live path is moved out (fork 0), resume 0.
5. Prefill per `prefill_plan`; before each growth the engine reports bytes and the worker calls
   `evict`; pressure level read at every per-layer check point.
6. Decode: same growth and pressure handling per token.
7. `commit` (mode Off: discard this request's state afterwards; `--no-cache`: reset).

## User-visible surface

- Usage: `cached = min(resume, P − 1)` per API (002 data model).
- mrun metadata (002): `x-mrun-cache-cached`, `-processed`, `-divergence`, `-gap`, `-cause` headers;
  the same fields in `: mrun {json}` SSE comments; `last` in `/mrun/status`.
- `mrun ps`: CacheStats and the last miss report.
- `mrun cache clear` → `POST /mrun/cache/clear` → `{branches_dropped, bytes_freed, live_reset: "now" | "after-request"}`.
- `--no-cache`; header `x-mrun-cache: off` (all APIs; other values ignored); `--debug-cache`
  (refused with a non-loopback bind; excerpts on the server terminal only).
