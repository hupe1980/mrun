# Data Model: Speculative Decoding

## SpecMode

`auto | off | ngram` (`mtp` added by the follow-up). Source order as 004: default (`off` until the FR-001 gate passes, then
`auto`) < `[speculation] mode` < `MRUN_SPECULATION_MODE` < `--speculation` < request header
`x-mrun-speculation: off` (can only turn off). `mrun run --check` forces `off` (FR-017).

## SpecPolicy (per request)

| Field | Type | Rule |
| --- | --- | --- |
| mode | SpecMode | effective mode |
| reason_off | Option<`off` \| `grammar` \| `seed` \| `logprobs` \| `limit`> | set when no drafting can happen |

## DraftIndex (per request, helper thread builds, engine thread queries)

| Field | Type | Notes |
| --- | --- | --- |
| sources | list of {kind: prompt \| output \| rendering, tokens} | renderings separated by a sentinel id outside the vocabulary |
| automaton | suffix automaton over all sources | extended per committed output token |
| ready | bool | false until built; the controller uses depth 0 meanwhile |

Lookup → `Option<{match_len: u32, continuation: &[u32]}>` (most recent occurrence of the longest
suffix with `match_len ≥ min_match`).

## Draft

`{tokens: Vec<u32> (1..=N_max), match_len: u32}` (point proposals).

## DraftStep (engine handle)

`{id: u64, before: u32 (active_len before the round), width: u32 (N + 1)}`. Valid until the next
`forward_draft`, `forward` or `commit`; holds the hybrid rollback data (pre-round recurrent state,
per-position GDN inputs ≈ 0.5 MiB/position, ≤ 32 MiB at N = 64, in the scratch envelope).

## ProposalId

Opaque device-side proposal handle; always `None` for n-gram drafts; reserved for the MTP follow-up.

## VerifyOutcome (engine → runtime)

`{accepted: u32 (0..=N, ends at the first accepted EOS), next: u32 (target token after the
accepted prefix), seconds: f32, recomputed: u32 (hybrid recurrence positions recomputed)}`.
Postcondition: `engine.active_len() == before + accepted + 1` (or `before + k` after `commit(k)`).

## ControllerState (per model, in memory for the server lifetime)

| Field | Type | Notes |
| --- | --- | --- |
| acc_ngram | Beta(a, b) per match-length bucket {4–7, 8–15, 16–31, 32+} | per-position acceptance |
| cost | seconds per (width 1..=65, context bucket {<8K, 8–32K, 32–128K, ≥128K}) | analytic expert-union prior + EWMA (α = 0.1) |
| decode_cost | seconds per token at depth 0 per depth bucket | pipelined plain decode |
| scheduled | depth | next round's choice, made before `sample` |
| probe_interval, probe_since | u32 | 4 → ×2 up to 512 while probes change nothing |

## SpecStats (per request → metadata; aggregated in `/mrun/status`)

`{drafter: ngram | none, drafted: u32, accepted: u32, rounds: u32, reason?: string}`.
Invariant: `accepted ≤ drafted`; output tokens in usage = committed tokens.

## MtpSidecar — Removed

Moved to the MTP follow-up (spec "Follow-up (gated)"). When it returns, the sidecar is recorded in
the manifest and shown by `mrun show` only; it is never part of the model identity or cache key.
