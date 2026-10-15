# Contract: Persistent Prefix Tier

## Engine additions (`crates/mrun-engine/src/engine.rs`, no MLX types)

```rust
/// Read-only view of evaluated state bytes, valid until the engine bumps `generation`.
pub struct HostView { pub ptr: *const u8, pub len: usize, pub layer: u32, pub kind: StateKind,
                      pub dtype: DType, pub shape: Vec<u32> }
pub struct ExportedState { pub generation: u64, pub views: Vec<HostView> }
pub enum ExportSource { Segment(SegmentId), Checkpoint(CheckpointId), Live { start: u32, end: u32 } }
pub enum ImportKind { Segment { start: u32, end: u32 }, Checkpoint { position: u32 } }

pub trait Engine {
    // … existing methods (001: split_off/append, checkpoint/restore_checkpoint, …) …
    /// Views of a stored segment, a stored checkpoint, or live positions [start, end).
    fn export_views(&mut self, what: ExportSource) -> Result<ExportedState, EngineError>;
    /// Allocate owned buffers for a segment or checkpoint, counted in allocated() like any other
    /// segment: Err(NeedsMemory{bytes}) before allocating. Returns writable host views.
    fn import_alloc(&mut self, kind: ImportKind) -> Result<ImportBuffers, EngineError>;
    /// Turn filled buffers into an ordinary Segment or Checkpoint; dropping ImportBuffers
    /// without sealing frees them.
    fn import_seal(&mut self, buffers: ImportBuffers) -> Result<Imported, EngineError>;
}
pub enum Imported { Segment(Segment), Checkpoint(Checkpoint) }
```

Imported segments and checkpoints are indistinguishable from ones made by `split_off` and
`checkpoint`, so 003 restores them with its existing `append(segment, consume: true)` and
`restore_checkpoint` path; no new restore logic exists in the engine.

## Prefix store integration (`crates/mrun/src/cache/mod.rs`)

Worker order per request (extends 003): `lookup` (memory + disk candidates) → admission → if
`choice == disk` (disk match ≥ memory match + 512 tokens): for each entry's attention segment in
the chain, and only the deepest checkpoint at or below the match, `import_alloc` (on `NeedsMemory`,
evict first) → reader fills buffers (checksums per block) → `import_seal` → 003
`append(consume: true)` / `restore_checkpoint` → prefill the gap → decode → `finish` (memory) →
after 2 s idle: writer job for the live path (split at persisted checkpoints). Branch move-out
(003) first enqueues the live path up to the branch point `at` (if not yet persisted), then the
moved segment, so the branch entry has a persisted parent. Families with position-bound state write per turn only the
end-of-prompt checkpoint, plus the system+tools checkpoint once. Nothing is written for requests
with `x-mrun-cache: off`. Quiesce before any mutation (including eviction) waits at most one 4 MiB
block. Any reader error before sealing drops the buffers and falls back to recompute; the
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

`mrun bench --suite persist [--from RECORD_DIR]` (all timings with the model loaded): cells `persist/read-gbps`, `persist/write-gbps`,
`persist/restore-30k` (restore vs re-prefill seconds, ratio), and with `--from`:
`persist/attribution` (`{restart_tokens, evicted_tokens, other_tokens, disk_served_tokens_at_cap}`).
