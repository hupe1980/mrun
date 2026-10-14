# Qualification Contract and Fixture Catalog

**Candle replan notice (2026-10-14)**: Case catalogs remain useful, but native geometry/adapter stages below assume llama.cpp. Superseded for Candle by the
[active plan](../plan.md#current-planning-status) and pending replacement in task T122; do not implement
from it. Retained as historical llama.cpp design context.

**Status**: Case identities and assertions frozen by this plan; fixture bytes/hashes not generated.
No case has run. Materialize redistributable inputs/expected results, licenses and SHA-256 manifest
before capability/acceptance runs; missing entries block the gate. Changes require reviewed versioning.
Timing/sample rules remain in [benchmark-protocol.md](../benchmark-protocol.md), numerical policy in
[research D-07](../research.md), and wire rules in [http.md](http.md).

## Fixture Preparation and Sealing

Prepare draft source inputs/assertions/licenses/seed/tool limits independently. After the verified
renderer exists, derive exact token fixtures and full render/model identity, then seal the manifest
revision with both source and rendered hashes. Primitive renderer construction probes may use draft
inputs but cannot yield dependent gate passes. Sealed fixture verification precedes adapter evidence
that depends on rendering, candidate capability, correctness, quality and timing runs.
Later released-client schema/request captures form a new sealed revision and invalidate affected
evidence; preserve previous manifests and outcomes. Do not fabricate captures or tokens to seal early.

## Staged Evidence and Candidate Runs

The future `examples/qualify.rs` interface has the following suites; none is implemented today.

| Suite | Prerequisite and purpose | Output/authorization |
| --- | --- | --- |
| adapter | Pinned native primitives and sealed inputs for dependent probes | Adapter feasibility evidence only; continued development, never production serving |
| seal | Verified renderer, draft catalog and selected candidate | Versioned source/rendered fixture manifest; no inference authorization |
| capability | Safely admitted runtime candidate and sealed catalog | Exact-binary capability report permits cache-off serving |
| reuse | Matching candidate capabilities, sealed public state/client/coding fixtures | State/numerical, IC-001..020 and SC-007 evidence; all must pass for production reuse |
| acceptance | Same frozen candidate and required prior evidence | All SC-001..009 and comparator outcomes; only passing gates confer their recorded status |

`--runtime-binary PATH` is required for seal/capability/reuse/acceptance, not adapter-only probes.
Hash and exercise the selected runtime executable through a private run-scoped harness channel,
recording the harness identity separately. Merely hashing a binary while running different library
code is insufficient. The candidate channel accepts bounded selected-fixture operations; it cannot
mint production reports, select private workspaces, or enable arbitrary production reuse. Live client
quality uses an isolated loopback endpoint bound to the selected public fixture workspace/client run,
with normal validation, budgets, pressure handling, output bounds, synchronization and cleanup.
This channel is not a product `serve` option or force-reuse bypass and closes with the harness run.
The harness observes and verifies evidence before writing the candidate report.

Candidate-only reuse breaks the quality bootstrap cycle: measure the experimental optimized path
against cache off, then permit public/default reuse only after matching protocol/state/numerical and
SC-007 passes. Timing gates cannot replace quality. Evidence writers, independent clocks and paired
coding runners must exist before these suites execute, not after their authorization checkpoint.

Every binary/build/profile/model/render change invalidates candidate evidence. Rebuild, rerun
capabilities and affected reuse/acceptance suites before a changed binary serves; never transfer an
adapter pass or rewrite an old report hash. Use the bounded [report model](../data-model.md#qualificationreport).
All case counts/tolerances/SC thresholds below remain unchanged and empirically unrun.

## Twenty Interoperability Cases

| ID | Distinct fixture and required assertion |
| --- | --- |
| IC-001 | Model discovery returns exact admitted ID without loading |
| IC-002 | Nonstream text, one choice, reconciled usage and stop |
| IC-003 | Stream text with role-first, ordered deltas and single finish |
| IC-004 | UTF-8 multibyte token fragmented across native bytes |
| IC-005 | Tool arguments contain escaped braces/quotes; fragment assembly exact |
| IC-006 | Completed named tool has stable ID, declared name and valid schema |
| IC-007 | Tool result continuation matches preceding ID and sees result |
| IC-008 | Required tool choice succeeds; undeclared named choice rejected |
| IC-009 | tool_choice=none emits text and never tool_calls |
| IC-010 | Usage-only final chunk with empty choices; replay excluded from cached count |
| IC-011 | Text output allowance emits length, not stop |
| IC-012 | Tool JSON exhausts allowance; client observes failure and executes no partial call |
| IC-013 | Truncated/duplicate-key/invalid UTF-8 request rejected before backend |
| IC-014 | Unknown model/semantic option rejected with stable code |
| IC-015 | Orphan/duplicate tool result rejected before backend |
| IC-016 | Recursive/remote-ref/overdepth schema rejected before backend |
| IC-017 | Nine waiting requests: ninth rejected, eight preserve FIFO |
| IC-018 | Queue expiry releases lease and never starts inference |
| IC-019 | Streaming disconnect/slow reader cancels without success terminal |
| IC-020 | Unmodified pinned OpenCode reads, edits and tests a public fixture offline |

IC-001..019 include both raw HTTP assertions and native client/protocol reconstruction where applicable;
IC-020 captures sanitized actual requests/tool schemas and verifies the entire read-edit-test task.
Raw fixtures cannot substitute for the pinned client run. Disabled reasoning/media/parallel variants
are disclosed; every captured supported field is checked against the matrix without proxy rewriting.

## One Hundred State Cases

Each row has a distinct fixture mutation/control, not repeated happy-path runs under new names.
Exact positives compare CPU reference output token IDs with identical fixed execution shape and
reset sampler; Metal variation also needs D-07 checks. Every case asserts reservation/pin cleanup.

### Positive Exact Replay (P-001 through P-020)

| ID | Fixture |
| --- | --- |
| P-001 | One-token identical prompt with boundary replay |
| P-002 | 127-token identical prompt before chunk boundary |
| P-003 | 128-token identical prompt at chunk boundary |
| P-004 | 129-token identical prompt after chunk boundary |
| P-005 | 512-token identical prompt |
| P-006 | 8192-token identical prompt |
| P-007 | 14336-token identical prompt plus 2048 output reserve |
| P-008 | Cached prefix plus one appended token |
| P-009 | Cached prefix plus 32-token suffix |
| P-010 | Cached prefix plus 128-token suffix |
| P-011 | Cached prefix plus 512-token suffix |
| P-012 | Longest match chooses a deeper retained entry |
| P-013 | Shortened prompt fully contained in retained prefix |
| P-014 | Unicode input with exact token-boundary matching |
| P-015 | Embedded special-token text with identical escaping |
| P-016 | Exact unchanged ordered tool definitions |
| P-017 | Matched tool result and identical continuation history |
| P-018 | Seed reset yields same positive continuation |
| P-019 | Identical stop configuration preserves output/usage |
| P-020 | Disable then reenable retention produces cold/reference parity |

### Edit and Branch (B-001 through B-020)

| ID | Mutation; reuse must stop at first differing token |
| --- | --- |
| B-001 | Change first system token |
| B-002 | Edit middle of system message |
| B-003 | Append system suffix |
| B-004 | Edit first user message |
| B-005 | Edit middle user message |
| B-006 | Replace final user message |
| B-007 | Remove earlier assistant message |
| B-008 | Change historical assistant content |
| B-009 | Change tool description |
| B-010 | Reorder tool declarations |
| B-011 | Change tool parameter type |
| B-012 | Edit tool result |
| B-013 | Shorten history at a completed turn |
| B-014 | Branch A appends then B extends old immutable prefix |
| B-015 | Cancel A while B's retained prefix survives |
| B-016 | A chooses tool then B generates text from old prefix |
| B-017 | Leading whitespace changes tokenization |
| B-018 | Unicode normalization changes token bytes |
| B-019 | JSON argument property order changes rendered history |
| B-020 | Hash-bucket collision cannot override exact token comparison |

### Compatibility Identity (I-001 through I-020)

| ID | Changed identity; old state must not be reused |
| --- | --- |
| I-001 | Weight content at same pathname |
| I-002 | Weight revision/quantization |
| I-003 | Tokenizer vocabulary |
| I-004 | Tokenizer merge rules |
| I-005 | Template source |
| I-006 | Template variant |
| I-007 | BOS policy |
| I-008 | EOS policy |
| I-009 | Thinking/render controls |
| I-010 | Frozen template-time input |
| I-011 | Architecture |
| I-012 | State ABI |
| I-013 | Backend commit |
| I-014 | Backend build/kernel configuration |
| I-015 | Position origin |
| I-016 | RoPE base |
| I-017 | RoPE scaling |
| I-018 | Adapter identity |
| I-019 | K/V precision |
| I-020 | K/V layout/physical pool epoch |

Unsupported identity variations use fake-backend rejection tests; they do not imply additional model
support. Real model replacement/backend reload tests remain mandatory for the admitted profile.

### Scope and Request Isolation (S-001 through S-020)

| ID | Probe; no cross-scope or request-local mutable state reuse |
| --- | --- |
| S-001 | Different trusted scope ID in deterministic backend harness |
| S-002 | Same tokens in second process boot epoch |
| S-003 | Forged user field cannot select scope |
| S-004 | Forged session header cannot select scope |
| S-005 | Forged cache key rejected, never authorizes lookup |
| S-006 | Scope hash collision still compares full identity |
| S-007 | RNG seed A does not become B's RNG state |
| S-008 | Omitted seed creates fresh RNG |
| S-009 | Temperature differs without inherited sampler |
| S-010 | Top-p differs without inherited sampler |
| S-011 | Frequency penalty starts with current request history |
| S-012 | Presence penalty starts with current request history |
| S-013 | Grammar cursor from A does not affect B |
| S-014 | Named tool selection does not leak to auto |
| S-015 | Required selection does not leak to none |
| S-016 | Stop matcher carry does not leak across requests |
| S-017 | Partial UTF-8 carry does not leak |
| S-018 | Tool ID/argument buffer does not leak |
| S-019 | Generated suffix never becomes another prompt checkpoint |
| S-020 | Metadata diagnostics contain no private prompt/token/path data |

### Eviction, Cancellation, and Failure (F-001 through F-020)

| ID | Fault/race and expected safety |
| --- | --- |
| F-001 | Fifth entry evicts least recently used unpinned entry |
| F-002 | Retained-cell limit evicts before attaching new demand |
| F-003 | Pinned membership not freed during eviction |
| F-004 | Clear while decode active blocks new hits and releases safely |
| F-005 | Explicit unload drains queued and active work |
| F-006 | Queue-full rejection releases body/reservation |
| F-007 | Queue expiry releases once |
| F-008 | Queued disconnect never begins native work |
| F-009 | Prefill cancellation stops at synchronized chunk boundary |
| F-010 | Decode cancellation stops scheduling next token |
| F-011 | Slow-reader timeout/overflow cancels bounded bridge |
| F-012 | Cancel races successful terminal arbitration |
| F-013 | Partial prefill backend failure publishes no uncertain entry |
| F-014 | Decode fault invalidates uncertain context |
| F-015 | Grammar/parser failure cannot finish a partial tool |
| F-016 | Load failure has no duplicate load or leaked model lease |
| F-017 | Pressure stops admission then evicts/unloads safely |
| F-018 | Native allocation exceeds reservation and fails closed |
| F-019 | Repeated clear/unload returns live counts to baseline |
| F-020 | Restart loses cache and preserves client-owned history |

## Coding and Replay Catalog

Twenty public fixture tasks, each with objective supplied tests and bounded tools/wall time:
C-001 off-by-one loop; C-002 empty-input handling; C-003 UTF-8 slicing; C-004 integer overflow;
C-005 stable sorting; C-006 deduplication; C-007 date boundary; C-008 path validation;
C-009 JSON missing field; C-010 escaped string parser; C-011 error propagation;
C-012 cancellation cleanup; C-013 resource leak; C-014 queue fairness; C-015 retry limit;
C-016 HTTP status mapping; C-017 CLI argument validation; C-018 numerical rounding;
C-019 configuration default; C-020 multi-file API migration. Run three paired seeds per task,
same time/tool limits, paired reference/optimized paths, and retain all 60 outcomes per path.
Task fixture languages/tools and expected tests must be recorded before execution; no private repos.

Ten distinct ten-turn replay traces: R-001 Rust function edit, R-002 TypeScript API edit,
R-003 Python parser repair, R-004 C boundary fix, R-005 JSON configuration change,
R-006 shell validation change, R-007 Rust async cleanup, R-008 TypeScript tool schema change,
R-009 Python filesystem fix, R-010 mixed read-edit-test task. Freeze exact rendered inputs,
recorded tools and expected valid-output checks; do not count tool time as runtime-controlled time.

## Evidence Gates

| Spec gate | Required lane |
| --- | --- |
| SC-001 | IC-001..020 and offline pinned-client task |
| SC-002 | 8192-prefix hit cells, matched cache-off control, protocol statistics |
| SC-003 | P/B/I/S/F catalog, exact and Metal numerical lanes |
| SC-004 | 100 trials each queued/prefill/decode/slow reader, actual release clock |
| SC-005 | 30-minute churn, reservations/footprint/swap and safe unload |
| SC-006 | Same pinned backend direct-serving short and 8192 cells |
| SC-007 | C-001..020, 60 paired outcomes and exact completed schema validity |
| SC-008 | R-001..010, paired replay loop times and live tool/wall times |
| SC-009 | Sanitized logs/artifact audit and independent offline traffic observation |

Do not mark any lane passed merely because inputs, tasks or reports were generated. Fixture capture
must fail if the expected counts/licenses/hashes are missing; feature completion requires all spec
gates, and best-in-class additionally requires optimized competitor comparison. Raw evidence lives
under ignored benchmark-results/, with only reviewed sanitized artifacts published.