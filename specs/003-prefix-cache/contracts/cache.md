# Contract: Prefix Cache

## Internal API (`crates/mrun/src/cache/mod.rs`, called by `runtime::worker` on the engine thread)

```rust
impl PrefixStore {
    /// Longest exact match over the live path and the tree; pins matched nodes.
    fn lookup(&mut self, key: &CacheKey, tokens: &[u32]) -> Lookup;
    /// Free memory for `tokens` more KV by evicting unpinned leaves; returns tokens freed.
    fn evict_for(&mut self, tokens: u32, engine: &mut dyn Engine) -> u32;
    /// Make the engine's active cache hold exactly `lookup.matched` tokens of this prompt:
    /// trim on the live path, or materialize the live path then restore matched segments.
    fn prepare(&mut self, lookup: &Lookup, engine: &mut dyn Engine) -> Result<(), EngineError>;
    /// Record the new live path (prompt + committed output) with message boundaries from the
    /// renderer (002) after the request finishes. Cancelled requests record up to `prefilled`.
    fn finish(&mut self, key: &CacheKey, tokens: &[u32], boundaries: &[Boundary], prefilled: u32);
    fn release(&mut self, lookup: Lookup);                   // unpin
    fn on_pressure(&mut self, level: Pressure, engine: &mut dyn Engine);
    fn clear(&mut self, engine: &mut dyn Engine);
    fn stats(&self) -> CacheStats;
}
```

Worker order per request: `lookup` → admission (`evict_for` until `prompt + max_output` fits) →
`prepare` → prefill remainder → decode (calling `evict_for` when the active cache grows) →
`finish` → `release`. The finish reason used for recording comes from the worker (002), so tool
calls and stops are both recorded.

## User-visible surface

- Usage fields: cached tokens per API (owned by 002).
- Response headers (all APIs, owned by 002 FR-022): `x-mrun-cache-cached`,
  `x-mrun-cache-processed`, `x-mrun-cache-cause`, `x-mrun-cache-divergence`.
- `mrun ps`: CacheStats and last MissReport.
- `mrun cache clear` → `POST /mrun/cache/clear`.
- Flags: `--no-cache`; per-request `"mrun": {"cache": false}` in Chat/Responses bodies;
  `--debug-cache` (loopback only; excerpts on the server terminal only).
