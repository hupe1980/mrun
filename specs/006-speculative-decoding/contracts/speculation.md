# Contract: Speculative Decoding

## Engine additions (`crates/mrun-engine/src/engine.rs`, no MLX types)

```rust
pub struct SpecCaps {
    /// Largest draft the measured scratch envelope admits (FR-013).
    pub max_draft: u32,
}

/// Handle for one verify forward; consumed by `accept`.
pub struct DraftStep {
    pub id: u64,      // round id; a stale or foreign handle is rejected
    pub before: u32,  // active_len() before the round
    pub width: u32,   // N + 1 positions fed
}

/// Device-side proposal distribution handle. Unused by n-gram drafts (always `None`);
/// reserved for the MTP follow-up.
pub struct ProposalId(u64);

pub struct VerifyOutcome {
    pub accepted: u32,   // 0..=draft.len(); stops at the first accepted EOS
    pub next: u32,       // target token after the accepted prefix (none committed after EOS)
    pub recomputed: u32, // hybrid: positions whose recurrence was recomputed (R5)
}

pub enum EngineError {
    // … existing 001 variants …
    Cancelled,    // additive: cancel observed inside forward_draft
    Unsupported,  // additive: operation not available for this model/state
}

pub trait Engine {
    // … existing methods (001: forward/sample split, trim, checkpoints, split_off/append) …
    fn spec_caps(&self) -> SpecCaps;
    /// Feed `last` followed by `draft` (N+1 positions) at active_len(), with no pending
    /// pipelined step (a pending one is rolled back first, as in `forward`). Memory for growth and
    /// the rollback input cache is checked first: Err(NeedsMemory{bytes}) before any work, state
    /// unchanged. Hybrids keep the pre-round recurrent state and the per-position GDN inputs
    /// (conv input, q/k/v, g, β). Cancel is checked after every layer; on cancel the state returns
    /// to `before` and Err(Cancelled) is returned.
    fn forward_draft(&mut self, last: u32, draft: &[u32], cancel: &CancelToken)
        -> Result<DraftStep, EngineError>;
    /// Accept on device (greedy compare, or point-proposal rejection sampling; `proposal` is
    /// always None in this feature), end at the first accepted EOS, roll back rejected positions
    /// (attention: trim; recurrent: recompute the recurrence over accepted + 1 positions from the
    /// pre-round state), and leave active_len() == before + accepted + 1. Rollback data stays
    /// until the next `forward_draft`, `forward` or `commit`. One host sync.
    fn accept(&mut self, step: &DraftStep, proposal: Option<ProposalId>, sampling: &Sampling)
        -> Result<VerifyOutcome, EngineError>;
    /// Cut the round to its first `k` committed positions (k ≤ accepted + 1), used when a
    /// host-side stop string ends inside the round; same rollback as `accept`, then frees the
    /// round's rollback data. Err(Unsupported) when no round's rollback data exists.
    fn commit(&mut self, step: &DraftStep, k: u32) -> Result<(), EngineError>;
}
```

Guarantees:
- After `accept` (and `commit`, if called) returns, no state for rejected positions remains.
  `ScriptedEngine` records the token at every state position so tests can assert this (SC-004).
- The forward/accept split mirrors 001's forward/sample split, leaving room for per-position grammar
  masks later (not in this feature).
- `forward_draft` with an empty draft followed by `accept` equals `forward` + `sample(pipeline:
  false)`; depth 0 uses the ordinary pipelined `forward`/`sample` path.
- `Cancelled` and `Unsupported` are additive `EngineError` variants (fold into the 001 contract).

## Runtime loop (`crates/mrun/src/runtime/generate.rs`)

Per round, with depth N chosen at the end of the previous round: depth 0: `forward`, then
`controller.choose()` for the next round, then `sample(pipeline = next_depth == 0)` → else build
the draft by index lookup, cap at `min(N, remaining − 1, caps.max_draft)`, `forward_draft` (on
`NeedsMemory`, evict and retry, as for decode), `accept`, then feed `accepted` draft tokens and
`next` one at a time through stop detection, UTF-8 hold-back and tool parsers; on a stop after
the round's k-th token, `commit(k)` and finish. `controller.observe(round)`.

## HTTP surface (extends 002)

- Request header `x-mrun-speculation: off` (all three APIs). Other values → 400 naming the header.
- mrun metadata adds `spec`:
  - non-streaming headers: `x-mrun-spec-drafter`, `x-mrun-spec-drafted`, `x-mrun-spec-accepted`,
    `x-mrun-spec-rounds`, `x-mrun-spec-reason` (only when nothing was drafted);
  - streaming: in the final `: mrun {json}` comment, `"spec": {"drafter": "ngram", "drafted": 412,
    "accepted": 377, "rounds": 19}`.
- `/mrun/status` (additive, schema v1): key `spec`: `{mode, controller: {depth,
  acceptance_by_bucket, cost_by_bucket}, totals: {drafted, accepted, rounds}}`.

## CLI and config (extends 004 contract)

| Surface | Behavior |
| --- | --- |
| `mrun serve --speculation auto\|off\|ngram` | Mode; config `[speculation] mode`, env `MRUN_SPECULATION_MODE` (`mtp` added by the follow-up) |
| `mrun run --speculation …` | Same for the REPL; summary line adds `· 377/412 accepted` |
| `mrun run --check` | Always runs with speculation `off` (FR-017) |
| `mrun bench --suite spec [--offline]` | FR-001 cells; `--offline` needs only the tokenizer |

## Bench result (extends 005 schema v1, additive)

Cells `spec/<trace>/offline` (`samples` = acceptance lengths; `copy_share_16` = share of reply
tokens in matches ≥ 16) and `spec/<trace>/<mode>` with mode ∈ {`off`, `ngram`, `auto`} (decode
tok/s per turn, `drafted`, `accepted`). A trace is labelled `edit-heavy` when
`copy_share_16 ≥ 0.40`. `--suite agent` comparisons run with `--speculation off` (FR-017).
