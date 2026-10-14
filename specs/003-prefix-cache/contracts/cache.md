# Contract: Prefix Cache

## Internal API (`crates/mrun/src/cache/mod.rs`)

```rust
impl PrefixStore {
    fn lookup(&mut self, key: &CacheKey, tokens: &[u32], boundaries: &[u32]) -> Lookup;
    fn commit(&mut self, key: &CacheKey, tokens: &[u32], boundaries: &[u32], engine: &mut dyn Engine);
    fn release(&mut self, lookup: Lookup);                  // unpins
    fn evict_for(&mut self, bytes: u64, engine: &mut dyn Engine) -> u64;
    fn on_pressure(&mut self, level: Pressure, engine: &mut dyn Engine);
    fn clear(&mut self, engine: &mut dyn Engine);
    fn stats(&self) -> CacheStats;
}
```

`commit` runs after prefill (prompt segments) and after generation (assistant segment, when the
request finished with Stop/ToolCalls). Cancelled or failed requests commit only prompt segments
that were fully prefilled.

## User-visible surface

- Usage fields: cached tokens per API (002 data model).
- `usage.mrun.cache` (Chat) / `response.metadata.mrun.cache` (Responses): `{cached, processed,
  cause, divergence}`.
- `mrun ps`: CacheStats and last MissReport.
- `mrun cache clear` → `POST /mrun/cache/clear`.
- Flags: `--no-cache`, per-request `"mrun": {"cache": false}` extension field (Chat/Responses
  bodies), `--debug-cache`, `--normalize <client>`.
