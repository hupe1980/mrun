# Contract: Persistent Prefix Tier

## Engine additions (`crates/mrun-engine/src/engine.rs`, no MLX types)

```rust
/// Read-only view of evaluated state bytes, valid until the engine bumps `generation`.
pub struct HostView { pub ptr: *const u8, pub len: usize, pub layer: u32, pub kind: StateKind,
                      pub dtype: DType, pub shape: Vec<u32> }
pub struct ExportedState { pub generation: u64, pub views: Vec<HostView> }

pub trait Engine {
    // … existing methods (001, 003) …
    /// Views of a branch segment or of live-path tokens [start, end) (+ checkpoint at end).
    fn export_views(&mut self, what: ExportSource) -> Result<ExportedState, EngineError>;
    /// Allocate (inside the budget) destination buffers for an entry chain; returns writable
    /// views. Caller fills them, then commits. Dropping without commit frees them.
    fn import_alloc(&mut self, layout: &ImportLayout) -> Result<ImportBuffers, EngineError>;
    /// Make the filled buffers the active state for tokens [0, end) (and checkpoint at end).
    fn import_commit(&mut self, buffers: ImportBuffers, end: u32) -> Result<(), EngineError>;
}
```

Rules: `HostView`s are handed to the writer thread only inside `cache::persist`, which is the only
module allowed `unsafe` reads of them (`#[allow(unsafe_code)]`, documented). The engine bumps the
generation and calls `Tier::quiesce(range)` before trimming, growing, moving or freeing any array a
live view covers.

## Prefix store integration (`crates/mrun/src/cache/mod.rs`)

Worker order per request (extends 003): `lookup` (memory + disk candidates) → admission → if
`choice == disk`: `import_alloc` → reader fills buffers (verify checksums per block) →
`import_commit` → prefill the gap (hybrids: from the checkpoint) → decode → `finish` (memory) →
after 2 s idle: writer job for the live path. Branch move-out (003) enqueues a writer job for the
moved segment. Any reader error before commit drops the buffers and falls back to recompute; the
miss report says why.

## HTTP (extends 002)

- `POST /mrun/cache/clear?tier=memory|disk|all` (default `memory`, as in 003).
- mrun metadata: `source` (`memory` | `disk` | `none`) next to `cached`; non-streaming header
  `x-mrun-cache-source`.
- `/mrun/status` (additive, schema v1): `cache.disk: TierStats` (data model) or `{enabled: false,
  reason?}`.

## CLI (extends 004 contract)

| Command | Behavior |
| --- | --- |
| `mrun serve --persist` / `--no-persist` | Enable or disable the tier (FR-002) |
| `mrun cache [--json]` | Location, cap, used, entries, per-model usage; memory stats when a server runs |
| `mrun cache clear` | Memory tier of the running server (003, unchanged) |
| `mrun cache clear --disk [--model M]` | Disk entries (all, or one model's keys); via server or directly with the lock |
| `mrun cache clear --all` | Both tiers |

Exit codes: 0 success; 5 when `clear` without `--disk` finds no server; 1 when the lock is held by
an unreachable process (message names the pid).

## Bench (extends 005, additive)

`mrun bench --suite persist [--from RECORD_DIR]`: cells `persist/read-gbps`, `persist/write-gbps`,
`persist/restore-30k` (restore vs re-prefill seconds, ratio), and with `--from`:
`persist/attribution` (`{restart_tokens, evicted_tokens, other_tokens, disk_served_tokens_at_cap}`).
