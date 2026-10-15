# Contract: Speculative Decoding

## Engine additions (`crates/mrun-engine/src/engine.rs`, no MLX types)

```rust
pub struct SpecCaps {
    /// MTP head loaded (sidecar present and verified).
    pub mtp: bool,
    /// Largest draft the measured scratch envelope admits (FR-013).
    pub max_draft: u32,
}

pub struct VerifyOutcome {
    pub accepted: u32,   // 0..=draft.len()
    pub next: u32,       // target token after the accepted prefix
    pub replayed: u32,   // hybrid rollback replay length (R5)
}

pub trait Engine {
    // … existing methods (001: forward/sample split, trim, checkpoints, split_off/append) …
    fn spec_caps(&self) -> SpecCaps;
    /// Feed `last` followed by `draft` (N+1 positions) at active_len(), with no pending
    /// pipelined step (a pending one is rolled back first, as in `forward`). Memory for growth and
    /// in-round snapshots is checked first: Err(NeedsMemory{bytes}) before any work, state
    /// unchanged. Cancel is checked after every layer; on cancel the state returns to `before`
    /// and Err(Cancelled) is returned.
    fn forward_draft(&mut self, last: u32, draft: &[u32], cancel: &CancelToken)
        -> Result<DraftStep, EngineError>;
    /// Accept on device (greedy compare; rejection sampling with `proposal` = None for point
    /// drafts, Some for MTP drafts), roll back rejected positions (attention: trim; recurrent:
    /// restore the in-round snapshot ≤ accepted + 1 and replay), free in-round snapshots, and
    /// leave active_len() == before + accepted + 1. One host sync.
    fn accept(&mut self, step: DraftStep, proposal: Option<ProposalId>, sampling: &Sampling)
        -> Result<VerifyOutcome, EngineError>;
    /// Draft up to `n` tokens with the MTP head from the current position; returns tokens and a
    /// device-side proposal handle consumed by the next `accept`. Err(Unsupported) without a head.
    fn draft_mtp(&mut self, n: u32, sampling: &Sampling) -> Result<(Vec<u32>, ProposalId), EngineError>;
}
```

Guarantees:
- After `accept` returns, no state for rejected positions remains. `ScriptedEngine` records the
  token at every state position so tests can assert this (SC-004).
- The forward/accept split mirrors 001's forward/sample split, leaving room for per-position grammar
  masks later (not in this feature).
- `forward_draft` with an empty draft followed by `accept` equals `forward` + `sample(pipeline:
  false)`; depth 0 uses the ordinary pipelined `forward`/`sample` path.

## Runtime loop (`crates/mrun/src/runtime/generate.rs`)

Per round: `controller.choose()` → depth 0: pipelined `forward`/`sample` → else build draft (index
lookup or `draft_mtp`), cap at `min(depth, remaining − 1, caps.max_draft)`, `forward_draft` (on
`NeedsMemory`, evict and retry, as for decode), `accept`, then feed `accepted` draft tokens and
`next` one at a time through stop detection, UTF-8 hold-back and tool parsers; on a stop inside
the round, `trim` to the stop and finish. `controller.observe(round)`.

## HTTP surface (extends 002)

- Request header `x-mrun-speculation: off` (all three APIs). Other values → 400 naming the header.
- mrun metadata adds `spec`:
  - non-streaming headers: `x-mrun-spec-drafter`, `x-mrun-spec-drafted`, `x-mrun-spec-accepted`,
    `x-mrun-spec-rounds`, `x-mrun-spec-reason` (only when nothing was drafted);
  - streaming: in the final `: mrun {json}` comment, `"spec": {"drafter": "ngram", "drafted": 412,
    "accepted": 377, "rounds": 19}`.
- `/mrun/status` (additive, schema v1): `speculation: {mode, drafters: {ngram: true, mtp: false,
  mtp_reason?: "no-sidecar"}, controller: {ngram: {depth, acceptance_by_bucket}, mtp: {depth,
  acceptance_by_position}}, totals: {drafted, accepted, rounds}}`.

## CLI and config (extends 004 contract)

| Surface | Behavior |
| --- | --- |
| `mrun serve --speculation auto\|off\|ngram\|mtp` | Mode; config `[speculation] mode`, env `MRUN_SPECULATION_MODE` |
| `mrun run --speculation …` | Same for the REPL; summary line adds `· 377/412 accepted` |
| `mrun pull <alias> [--no-mtp]` | Fetches the MTP sidecar when the alias declares one (FR-015) |
| `mrun show <model>` | Lists `mtp: ready (0.47 GB, source Qwen/Qwen3.6-35B-A3B@<rev>)` or the reason it is absent |
| `mrun bench --suite spec [--offline]` | FR-001 cells; `--offline` needs only the tokenizer |

## Bench result (extends 005 schema v1, additive)

Cells `spec/<trace>/offline` (`samples` = acceptance lengths; `copy_share_16` = share of reply
tokens in matches ≥ 16) and `spec/<trace>/<mode>` (decode tok/s per turn, `drafted`, `accepted`).
A trace is labelled `edit-heavy` when `copy_share_16 ≥ 0.40`.
