# Data Model: Speculative Decoding

## SpecMode

`auto | off | ngram | mtp`. Source order as 004: default (`off` until the FR-001 gate passes, then
`auto`) < `[speculation] mode` < `MRUN_SPECULATION_MODE` < `--speculation` < request header
`x-mrun-speculation: off` (can only turn off).

## SpecPolicy (per request)

| Field | Type | Rule |
| --- | --- | --- |
| mode | SpecMode | effective mode |
| reason_off | Option<`off` \| `grammar` \| `seed` \| `limit` \| `no-drafter` \| `logprobs`> | set when no drafting can happen |
| drafters | {ngram: bool, mtp: bool} | mtp only when the sidecar is loaded |

## DraftIndex (per request, helper thread builds, engine thread queries)

| Field | Type | Notes |
| --- | --- | --- |
| sources | list of {kind: prompt \| output \| rendering, tokens} | renderings separated by a sentinel id outside the vocabulary |
| automaton | suffix automaton over all sources | extended per committed output token |
| ready | bool | false until built; the controller uses depth 0 meanwhile |

Lookup → `Option<{match_len: u32, continuation: &[u32]}>` (most recent occurrence of the longest
suffix with `match_len ≥ min_match`).

## Draft

`{drafter: ngram | mtp, tokens: Vec<u32> (1..=N_max), match_len: Option<u32>, q: Option<device
distribution handle>}`

## VerifyOutcome (engine → runtime)

`{accepted: u32 (0..=N), next: u32 (target token after the accepted prefix), seconds: f32,
replayed: u32 (hybrid rollback replay length)}`. Postcondition: `engine.active_len() ==
before + accepted + 1`.

## ControllerState (per model, in memory for the server lifetime)

| Field | Type | Notes |
| --- | --- | --- |
| acc_ngram | Beta(a, b) per match-length bucket {4–7, 8–15, 16–31, 32+} | per-position acceptance |
| acc_mtp | Beta(a, b) per position 1..=4 | |
| cost | seconds per (width 1..=65, depth bucket {<8K, 8–32K, 32–128K, ≥128K}) | probe prior + EWMA (α = 0.1) |
| decode_cost | seconds per token at depth 0 per depth bucket | pipelined plain decode |
| scheduled | {drafter, depth} | next round's choice |
| probe_interval, probe_since | u32 | 4 → ×2 up to 512 while probes change nothing |

## SpecStats (per request → metadata; aggregated in `/mrun/status`)

`{drafter: ngram | mtp | mixed | none, drafted: u32, accepted: u32, rounds: u32, reason?: string}`.
Invariant: `accepted ≤ drafted`; output tokens in usage = committed tokens.

## MtpSidecar (manifest extension, `~/.mrun/manifests/<alias>.toml`)

| Field | Type |
| --- | --- |
| source_repo, source_revision | string, 40-hex |
| tensors | list of {name, dtype, shape, sha256 (source bytes)} |
| quantization | {bits: 4, group_size: 64, overrides: {gate: 8}} |
| path | `~/.mrun/sidecars/<base-identity>/mtp.safetensors` |
| sha256 | sidecar file hash; added to the model identity as `mtp=<sha256>` |

States: `absent → fetching → verifying → quantizing → ready | corrupt`; `corrupt` or `absent` →
MTP unavailable, model loads without it.
