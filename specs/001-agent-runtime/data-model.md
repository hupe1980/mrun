# Data Model: Interactive Runtime

**Status**: Planned entities and invariants; no implementation or qualified model yet.
**Authority**: [spec.md](spec.md), [research decisions](research.md), [HTTP contract](contracts/http.md).
All sizes use bytes or binary KiB/MiB/GiB; tokens are rendered backend token IDs, not characters.

## QualifiedModel

- `model_id`: stable public name `qwen3-8b-q4-k-m`; one configured model initially.
- `artifact`: repository, immutable revision, filename, byte length, SHA-256, license.
- `identity`: weight/tokenizer/template hashes, render-control digest, architecture and state ABI,
  backend commit/build configuration, FP16 KV layout, positions/RoPE, adapters (empty initially).
- `capabilities`: full attention, text-only, non-thinking, single tool call, grammar/schema subset,
  context 16,384 and output maximum 2,048. Report only capabilities actually qualified.
- `native_owner`: private owner-thread handle; not serializable or exposed through public Rust types.
- `fit_profile`: observed weight/KV/scratch/host peaks and conservative admitted bounds.

Hash the opened file before load; verify metadata and template against the selected profile. Model
files must remain unchanged while mapped: reject changed stat identity and invalidate on reload;
do not trust pathname alone. Operator-provisioned files must not be writable by untrusted users.

Lifecycle: `Unloaded -> Loading -> Ready -> Draining -> Unloaded`; load failures enter `Failed`.
One shared loading operation prevents duplicate concurrent loads. Failed requires explicit retry
load after invalidation. Pressure enters Draining, cancels work safely, then unloads. Idle expiry
is disabled by default; an explicit positive idle timeout unloads only after no active/queued work.

## Request

- `request_id`: server-generated opaque ID; independent of user/tool/session identifiers.
- `scope`: server OS uid and process boot epoch. Loopback is one trusted-user boundary, not
  tenant authentication; caller-supplied `user`, session, cache keys or headers cannot select scope.
- `conversation`: bounded validated messages, tools and matched tool results.
- `generation`: output allowance, seed, temperature, top-p, penalties and stop strings.
- `rendered_tokens`: full rerender of the request, plus exact identity used for matching.
- `reservation`: uniquely owned ledger lease for queued bytes, active growth and serving space.
- `deadline`: queue-entry instant +30 seconds; stalled-write interval 10 seconds.
- `cancel`: request-local signal polled at native safe boundaries.
- `stream`: fixed-byte-budget delivery state, UTF-8 carry, stop matcher, stable tool ID.
- `terminal`: at most one disposition: stopped, tool handoff, length, cancelled, failed, rejected.

State transitions:

```text
Received -> Validated -> Queued -> Preparing -> Prefill -> Decode -> Completing -> Released
Received/Validated -> Rejected
Queued -> Expired/Cancelled -> Released
Preparing/Prefill/Decode/Completing -> Cancelling/Failed -> Synchronizing -> Released
```

Only the scheduler transfers the foreground slot. Cancellation races completion through a single
terminal arbiter; all paths consume the lease once. Queue cancellation prevents native execution.
Client disconnect is separate from cancellation observation latency. Validation does not run
unbounded inference/rendering on async executor threads. Requests use FIFO, never implicit replacement.

## PrefixEntry

- `entry_id`, `scope`, full model/state identity; hash is only a candidate lookup key.
- Exact token IDs, contiguous valid positions starting at zero, `valid_prompt_length`.
- Native retained sequence ID; publication requires synchronized successful prompt computation.
- Unique pool-cell set/charge; entry references do not multiply shared backing allocations.
- Last-use order, pin count, generation/epoch and eviction state.

Longest eligible prefix matches exact tokens and identity within the same scope. Let `T` be total
prompt tokens and `M` be the longest matching valid prompt prefix, or zero on a miss:

```text
effective_reused = min(M, T - 1)
new_prompt = T - M
boundary_replay = M - effective_reused
processed_prompt = new_prompt + boundary_replay
T = effective_reused + processed_prompt
```

Reject an empty rendered prompt before this arithmetic. When `M=T`, final-token replay is one.
Never report the replay token as cached in wire usage. Freeze checkpoints after successful prefill,
before generation, and preserve them while active decoding appends a separate sequence membership.
One active sequence and four retained IDs require a free retained ID or eviction before publication.
If cache retention cannot fit, finish the request without publishing a new entry; report that outcome.

Lifecycle: `Candidate -> Synchronized -> Retained -> Pinned -> Retained -> Evicting -> Removed`.
Shared token positions are immutable. No position shifting, in-place modification of shared KV,
host/device snapshot serialization, sampler cloning or generated-suffix publication. Clear advances
the cache epoch immediately, blocks new hits and defers physical membership release until safe.
Evicting a retained membership releases cell eligibility, not preallocated device buffer bytes.
Failed/uncertain GPU work invalidates affected state; if the context's integrity is uncertain,
invalidate every entry and teardown rather than cold-prefill in a corrupt context.

## ResourceEnvelope and Reservation

Observations: physical memory, OS additional available memory, pressure severity, Metal recommended
working set/current allocations, process footprint/RSS, swap/pageout samples with timestamps.
Store GPU recommendation separately from OS availability. Missing observations block load/admission.

At startup/load compute the automatic owned-accounting budget as the lesser of 75% physical bytes
and available bytes after an 8 GiB host reserve. Users can lower it. Raising it requires explicit
opt-in and a local warning; GPU working-set checks still apply. While loaded, retain that selected
ceiling and independently recheck additional available capacity before every allocation/admission;
do not subtract the runtime's existing allocations twice from an available-memory observation.
Pressure or decreased availability stops new admissions before eviction or synchronized unload.
None of these policies guarantees a hard OS process-RSS cap.

Ledger tracks unique backing allocation IDs, owned bytes, committed/reserved bytes, maximum demand,
sharing/pins and release state. CPU views and GPU views of one backing have one charge. Weights,
the full physical KV pool, scratch, native host allocations and serving allocations are charged
before allocation, not retroactively inferred from RSS. Reservation is an allocation ceiling:
growth exceeding it fails safely rather than reallocating unnoticed. Before measurements, reserve
scratch at least the greater of 2 GiB or 25% of weight-plus-KV backing, plus enumerated native
host/serving demand; qualification may require a larger recorded conservative fit profile.
Unobserved/unbounded native allocations block qualification.

### Finite Defaults

| Resource | Bound and action |
| --- | --- |
| Active/queued | 1/8; full queue rejected; waiting raw bodies <=16 MiB |
| Request body/headers | 2 MiB/32 KiB; reject before unbounded parsing |
| HTTP connections | 16; excess refused; 5-second headers, 30-second body receive deadline |
| Messages/tools | 512/64; schema depth 8, JSON nesting 32 |
| Schemas | 32 KiB per tool, 256 KiB total, 4,096 schema nodes total |
| Render/tokenize temporaries | 64 MiB per active request; reject overflow |
| Grammar/parser working storage | 16 MiB per active request; reject unsupported/excess complexity |
| Generated/tool/nonstream payload | 256 KiB per request; failure on overflow, never successful truncation |
| Streaming queue | 64 KiB total serialized bytes per request, including event envelopes |
| Native-to-async bridge | 16 event slots, additionally bounded by that same byte lease |
| Stop strings | At most 4, each <=256 UTF-8 bytes; matcher/carry included in output lease |
| KV | 32,768 physical cells, FP16; 5 sequence IDs, no context shifting |
| Retention | 4 entries, <=16,384 unique retained cells, <=4 MiB metadata |
| Prefill/decode | <=128 tokens / 1 token per synchronized call |
| Diagnostics | 1 MiB bounded metadata ring; no prompt/tool/path data |
| Warm residency | No default idle expiry; explicit unload/pressure/failure allowed |

Budget includes active/receiving requests in addition to queued bodies. Before body reads allocate
bounded slots and leases; a 16-connection pool cannot bypass the eight-request inference queue.
Native callbacks never block forever on HTTP consumers: cancel after the 10-second no-progress
deadline or bounded-queue overflow, wake the owner and synchronize. Backpressure must not allocate
an extra unbounded token or tool buffer. Partial UTF-8 bytes and tool arguments count toward limits.

Pressure order: reject new admissions; evict unpinned checkpoints; if physical capacity is still
unsafe cancel affected active work, synchronize, teardown context/model and expose unavailable
status. No silent context shortening. Resume only after fresh observations and explicit safe reload.

## RunEvidence

Manifest includes artifact/backend/client/build identities, toolchain/SDK/macOS/hardware, limits,
template/control/token fixture hashes, sampling/batch settings, licenses, conditions and seeds.
Per-run records contain stage timestamps, prompt/new/replay/reused/output tokens, pool/ledger peaks,
independent memory samples, terminal outcome and quality checks. Never persist private prompt/token
arrays, tool data, credentials or absolute paths in normal diagnostics. Public fixture payloads
may appear only in explicitly selected qualification results after review.

Gate disposition is `pass`, `fail`, `blocked`, or `inconclusive`. Missing evidence means blocked.
Client tool time, runtime-controlled loop time and wall time are separate. Hardware results require
the [benchmark protocol](benchmark-protocol.md); portable fake-backend tests do not replace them.
An explicit bounded local qualification report references public-evidence hashes and exact
binary/model/backend/profile identities. Candidate-binary capability gates authorize cache-off serving;
state/numerical, IC-001..020 protocol and SC-007 task-quality passes additionally authorize reuse.
Adapter-spike evidence authorizes continued development only. All nine acceptance gates are still required for feature completion/performance
qualification. A successful model hash check cannot authorize inference by itself.

## QualificationReport

- `schema_version`: exactly 1; unknown versions and fields rejected, no permissive fallback.
- `stage`: adapter_spike, candidate_capability, or candidate_reuse; adapter_spike never authorizes serving.
- `subject`: exact runtime executable SHA-256 for candidate stages, backend/model/profile/render identity,
  fixture manifest revision/hash and build configuration. Adapter stages have no runtime authorization.
- `harness`: independent executable/build identity and run-scoped target binding; hashing the harness
  cannot substitute for exercising the selected runtime candidate.
- `records`: evidence artifact hashes, gate dispositions and required case/trial coverage. Pass without
  complete matching records is invalid. Reports retain failure/blocked/inconclusive gates.
- `eligibility`: cache-off serving only after candidate capabilities pass; production reuse only after
  capabilities, state/numerical, IC-001..020 and SC-007 pass for the same subject/profile.

Parse limits: 1 MiB before parsing, JSON depth 16, 65,536 nodes, 1,024 evidence references and
4 KiB per string. Reject duplicate keys and trailing data. Bulk raw records live in separately hashed
public evidence artifacts; verify them before deriving eligibility, not just the summary declarations.
Reports are trusted local operator evidence, not cryptographic certification or private conversation state.

Lifecycle: draft -> complete -> validated -> eligible; any binary/build/model/profile/render identity
change or affected fixture revision makes evidence stale. Rebuild requires fresh candidate qualification
before serving; never edit an old report's hash to transfer its pass. Keep historical failed evidence.
An unchanged binary may reuse verified matching evidence at startup. Performance-qualified status still
requires all SC-001..009; a capability-only report cannot imply it.

## FixtureManifest

Source stage records distinct case IDs, licensed input bytes, expected assertions, languages/tools,
ceilings and paired seeds. Draft preparation needs no renderer and cannot invent rendered tokens.
Rendered stage derives exact token arrays and tokenizer/template/render/model identities only after
the verified native renderer exists. Seal both source and rendered hashes before any dependent run;
the loader rejects draft, missing or mismatched records. Later public client captures create a new
sealed revision; retain old manifests and rerun affected gates rather than mutate evidence in place.

These entities do not relax production admission: independent observations, conservative allocation
leases, full prompt/output reservation and pressure handling precede even the first cache-off request.
Candidate-only qualification may exercise reuse before eligibility, but never bypass resource/lifetime
controls or expose a product force-reuse mode. See [qualification lifecycle](contracts/qualification.md#staged-evidence-and-candidate-runs).