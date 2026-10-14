# Native Backend Contract

**Status**: Planned narrow ownership boundary; no binding, build or spike has run.
Pin and source rationale: [research](../research.md). Entities and limits: [data-model](../data-model.md).

## Boundary and Operations

One private llama.cpp adapter and C ABI shim around pinned C/C++ common helpers. Public Rust types
contain owned IDs/metadata, never `llama_context`, pointers, C++ objects or device handles.
Unsafe code is confined to this boundary; every shim operation returns explicit status and owns
its allocations. Catch C++ exceptions before crossing C ABI. Native fatal/driver faults that cannot
be caught safely are qualification blockers, not fictional Rust Result recoveries.

| Operation | Preconditions and outcome |
| --- | --- |
| Inspect | Verified immutable GGUF; owned metadata/template/tokenizer identity |
| Load | Unique owner and full resource lease; model/context ready or explicit failure |
| Render/tokenize | Fully validated history/tools; owned exact tokens and parser/grammar plan |
| Attach prefix | Same identity/scope, synchronized retained entry; active sequence membership |
| Evaluate | <=128 prefill or one decode token, explicit positions; logits and actual counters |
| Publish | Successful synchronized prompt boundary; immutable retained sequence and pool charge |
| Sample/parse | Request-local sampler/grammar/RNG; stable tool plan and bounded fragments |
| Synchronize | Wait before mutation/release; no claim of GPU kernel preemption |
| Release | Remove active membership once; free only unreferenced safe storage |
| Clear/unload | Invalidate epoch, drain work, synchronize and teardown ownership |

Planned source is explicitly provisioned under `native/llama.cpp` at
`45451c366d4256f94fbc8845645e31a6b9155054`; build verifies the commit and rejects dirty source.
No build-time fetch hidden in Cargo/build.rs. Compile native arm64 with Metal and common Jinja/chat
helpers, no server HTTP stack, no curl/download facilities. CPU reference is a test route, not silent
serving fallback. Set GPU offload explicitly and fail if the qualified Metal route is unavailable.
Disable native content/path logging before model load, including parser-debug output.

## Geometry and Immutable Sharing

Set `n_ctx=32768`, `n_seq_max=5`, `kv_unified=true`, `n_batch=n_ubatch=128`, FP16 K/V,
full attention/no sliding or RoPE changes. Native `llama_n_ctx`, `llama_n_ctx_seq`, sequence count,
batch sizes and actual metadata must be read back and asserted before readiness. With this pin,
unified KV sets per-sequence context to the physical context instead of dividing by five; the Rust
admission contract still caps any request at 16384 total tokens. Native defaults are not accepted.

Use `llama_get_memory`, `llama_memory_seq_cp` and `llama_memory_seq_rm` only under the owner.
Same-stream copy adds sequence membership; share [0,effective_reused) and evaluate the remainder.
Retained sequence positions cannot be edited. Every release/eviction waits for `llama_synchronize`.
No `seq_add`, `seq_div`, global `seq_keep`, serialized checkpoints or generated-suffix retention.
Verify the actual native cell graph against the unique-byte ledger; native allocation geometry is
not inferred only from parameter requests. Remove on failure may be insufficient: teardown any
context whose validity cannot be established, preserving GPU-visible lifetime first.

## Cancellation and Backpressure

The owner checks request cancellation/pressure before render, before each native call and after
each synchronized completion. No new native call follows observed cancellation. The Metal route
cannot depend on the CPU-only abort callback. Measure call duration and observation-to-safe-release
on the required hardware; 128-token chunks are not evidence for the 250 ms gate.
Slow readers share one byte lease between bridge and HTTP queues. Full capacity pauses further
generation only while safe/bounded; overflow or ten seconds without write progress cancels.
The backend thread must remain wakeable while the consumer fails. Synchronized cleanup publishes
only independently known valid prompt checkpoints; failure cannot promote partially written state.

## Tools and Numerics

Use `common_chat_templates_apply` and `common_chat_session`/native parser helpers with an explicitly
frozen non-thinking render plan. Grammar is request-local, instantiated through backend facilities
and reset on every request. The HTTP subset is validated with a standard Rust JSON Schema library
after parsing. Never label an invalid/truncated/unknown call successful, even if native grammar
accepted it. Required/named/none selection must pass positive and rejected tool fixtures.

Copy borrowed logits to bounded owned storage for numerical evidence only; do not keep borrowed
native pointers across another evaluate or owner operation. Exact CPU token replay and Metal
teacher-forced tolerance lanes are separate, as defined in research D-07. Record precision, flash
attention choice, thread counts and shader/build configuration identically for paired lanes.
Flash attention is explicitly off initially; no auto-routing between numerical paths in a run.

## Stop Conditions

Before production-path implementation, capability spikes must establish boundary replay, immutable
branch isolation, geometry, preallocation enforcement, bounded render/grammar compilation,
asynchronous fault disposition, safe unload, and cancellation on physical M4 Pro. A failed or
unobservable requirement blocks backend qualification and requires revised design/selection.
Do not add a second production adapter, privileged OS tuning, or hard-exit "cleanup" to hide failure.
Portable fake-backend tests can establish Rust state-machine behavior, not these native capabilities.