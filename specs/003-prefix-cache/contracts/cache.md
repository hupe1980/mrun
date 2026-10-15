# Contract: Prefix Cache

## Internal API (`crates/mrun/src/cache/mod.rs`, called by `runtime::worker` on the engine thread)

```rust
impl PrefixStore {
    /// Longest exact match over the live path and the tree; pins matched nodes.
    fn lookup(&mut self, key: &CacheKey, tokens: &[u32], hints: &CacheHints) -> Lookup;
    /// Free memory for `tokens` more KV by evicting unpinned leaves; returns tokens freed.
    fn evict_for(&mut self, tokens: u32, engine: &mut dyn Engine) -> u32;
    /// Make the engine's active cache hold exactly `lookup.matched` tokens of this prompt:
    /// trim on the live path, or materialize the live path then restore matched segments.
    fn prepare(&mut self, lookup: &Lookup, engine: &mut dyn Engine) -> Result<(), EngineError>;
    /// Record the new live path (prompt + committed output) with message boundaries from the
    /// renderer (002) after the request finishes. Cancelled requests record up to `prefilled`.
    /// `hints` adds `cache_control` breakpoints as boundaries and tags nodes with the session.
    fn finish(&mut self, key: &CacheKey, tokens: &[u32], boundaries: &[Boundary], hints: &CacheHints, prefilled: u32);
    fn release(&mut self, lookup: Lookup);                   // unpin (called right after prepare)
    fn on_pressure(&mut self, level: Pressure, engine: &mut dyn Engine);
    fn clear(&mut self, engine: &mut dyn Engine);
    fn stats(&self) -> CacheStats;
}
```

Worker order per request: `lookup` → admission (`evict_for` until `prompt + clamped max_output` fits; the clamp uses total capacity, so eviction can always make room) →
`prepare` (bounded materialization, restore) → `release` (segments copied into the active cache are
no longer pinned) → prefill remainder → decode (calling `evict_for` when the active cache grows) →
`finish`. The finish reason used for recording comes from the worker (002), so tool
calls and stops are both recorded.

## User-visible surface

- Usage fields: cached tokens per API (owned by 002).
- mrun metadata (all APIs, owned by 002 FR-022): headers for non-streaming responses, `: mrun`
  SSE comments for streaming responses: cached, processed, cause, divergence.
- `mrun ps`: CacheStats and last MissReport.
- `mrun cache clear` → `POST /mrun/cache/clear`.
- `CacheHints {breakpoints, session}` come from 002's ChatRequest `cache_hints`; they order eviction
  and add boundaries, never change matching.
- Flags: `--no-cache`; per-request `x-mrun-cache: off` (all APIs) or `"mrun": {"cache": false}` in Chat/Responses bodies;
  `--debug-cache` (loopback only; excerpts on the server terminal only).
