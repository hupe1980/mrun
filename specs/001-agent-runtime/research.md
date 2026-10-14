# Research: Apple Silicon Agent Inference

## Candle Replan Research (2026-10-14)

**Status**: Source research recorded; Phase 0 incomplete. No Candle build or hardware qualification.
These decisions supersede conflicting backend/toolchain assumptions in the historical research below.
The maintainer selected Candle as the core tensor library for mrun's own runtime and explicitly
authorized modern Rust. mistral.rs remains a competitor/comparator/reference, never an embedded runtime.

### C-01: Modern Rust and Candidate Revision

**Decision**: Select Rust 1.94.0 / edition 2024 as the planning target and intended minimum;
validate the locked dependency graph on that compiler before accepting the backend pin. Do not
retain 1.85 as a constraint or add a compatibility fork merely to preserve it. Evaluate Candle
`c68b24997319b8d76e5a4e775dab197c7f26322a` (workspace version 0.11.0), not an unpinned branch.
The version string does not identify the release-tag commit. This is a candidate, not qualification.

**Rationale**: Candidate quantized core calls `usize::is_multiple_of`, stabilized in Rust 1.87;
mandatory `zip` 8.6.0 declares Rust 1.88. Thus 1.85 is incompatible with the examined source.
Neither lower bound proves the complete MSRV. The modern target is authorized, but no dependency
resolution/build has run. Existing runtime manifests/toolchain files are not migrated by planning.

**Alternatives considered**: Candle 0.10.0 (`153cadd27933d1f03982622ebdbe382de4eddd84`),
0.10.1 (`904bf22e63168d3e3cf712058ed9ae65fd5a6176`) and 0.10.2
(`7c7a8c570e6a16ea24cc30a7501c8ffbbdb51680`) also use the newer integer API.
0.9.1 (`cd96fa80da255e34f7b16b4ff98b6a31d557201b`) lacks the examined Qwen3 model
implementations and does not establish locked 1.85 compatibility. Backporting solely for the old
compiler adds maintenance without a current requirement. mistral.rs's compiler requirement is
competitor provenance, not a reason to embed it.

Sources: [Rust 1.94](https://blog.rust-lang.org/2026/03/05/Rust-1.94.0/),
[Candle manifest](https://github.com/huggingface/candle/blob/c68b24997319b8d76e5a4e775dab197c7f26322a/Cargo.toml),
[mandatory core dependencies](https://github.com/huggingface/candle/blob/c68b24997319b8d76e5a4e775dab197c7f26322a/candle-core/Cargo.toml),
[quantized source](https://github.com/huggingface/candle/blob/c68b24997319b8d76e5a4e775dab197c7f26322a/candle-core/src/quantized/mod.rs),
[integer stabilization](https://doc.rust-lang.org/std/primitive.usize.html#method.is_multiple_of),
[zip 8.6.0 manifest](https://docs.rs/crate/zip/8.6.0/source/Cargo.toml).

### C-02: Weights, Mutable Active KV and Frozen Snapshots

**Decision**: Candidate architecture separates immutable weights from request-local per-layer KV,
behind one serialized GPU owner. Adapt the existing Qwen3 model operations rather than write
new tensor kernels. Fixed-capacity private active KV and separately allocated frozen prompt
snapshots are a design candidate, not yet a qualified public API or allocation profile.

**Rationale**: Stock quantized `ModelWeights` combines weights with private attention caches.
Public load/forward/clear methods do not expose per-layer import, truncate or export operations.
`Tensor::clone`, views and `detach` share storage. On the reviewed Metal path `Tensor::copy()`
reaches a storage clone that shares an `Arc<Buffer>`; it is not a deep-copy guarantee.
`force_contiguous()` allocates/copies, while `slice_set(&self, ...)` can mutate storage without
enforcing uniqueness. The runtime must prohibit mutable access to shared snapshots, synchronize
copies, and prove independent backing plus unchanged branch logits/tokens in regression tests.
Stock cache helpers can grow; their initial sequence length is not a hard allocation cap.

Restore arbitrary interior exact-prefix boundaries, not just chunk endpoints. Restore through
`min(longest_match, prompt_length - 1)` and count the last-token replay when logits are needed.
Never share sampler, grammar, RNG, penalties or generated suffix state. Limit retained snapshots
by actual backing bytes as well as at most four entries; no four-full-context fit guarantee.

**Alternatives considered**: Cloning the stock model does not establish immutable sharing;
recomputing every branch fails the reuse contract; importing llama.cpp sequence IDs/cell geometry
into Candle invents capabilities. Copied snapshots cost memory/restore time and must pass SC-002,
SC-005 and SC-008; their apparent simplicity is not a performance result.

Sources at the candidate revision:
[quantized Qwen3](https://github.com/huggingface/candle/blob/c68b24997319b8d76e5a4e775dab197c7f26322a/candle-transformers/src/models/quantized_qwen3.rs),
[tensor ownership](https://github.com/huggingface/candle/blob/c68b24997319b8d76e5a4e775dab197c7f26322a/candle-core/src/tensor.rs),
[mutation/cat](https://github.com/huggingface/candle/blob/c68b24997319b8d76e5a4e775dab197c7f26322a/candle-core/src/tensor_cat.rs),
[Metal storage](https://github.com/huggingface/candle/blob/c68b24997319b8d76e5a4e775dab197c7f26322a/candle-core/src/metal_backend/mod.rs).

### C-03: GPU Completion and Allocation Enforcement

**Decision**: Cancellation stops new scheduling between bounded calls; storage and reservations
outlive submitted GPU work. Require completion/error observation before reuse or release. A
complete conservative allocation manifest plus enforced pre-allocation reservations is required
for model load, forward, snapshot/restore, readback/staging and retained pools.

**Rationale**: Candidate completion code checks command errors after waiting; older examined paths
lack equivalent post-wait checks. This is source evidence, not injected-error or driver-hang proof.
On uncertain completion invalidate state, retain potentially live backing and mark the owner
unavailable; never call that successful capacity release. Finite call sizes do not prove SC-004.

The Metal allocator rounds new pooled buffers to powers of two, may reuse larger buffers and
retains shared/private pools. Logical tensor bytes, CPU Arc counts and sampled peaks are not
complete backing/GPU-lifetime accounting. A public hard-budget allocation hook was not established.
The old 2 GiB/25% scratch minimum cannot authorize a Candle forward. Bound all possible backing
growth conservatively without assuming temporary reuse, or select an audited enforcement path.
This remains a design blocker, not a task to discover after production admission is enabled.

For the reviewed Qwen3 dimensions, 16,384-token KV alone is 4.5 GiB F32 or 2.25 GiB F16.
Budget F32 initially for the examined GGUF route; FP16 needs explicit conversion/numerical review.
Dequantized embeddings are approximately 2.32 GiB. A 128-token, 32-head attention score at 16K
is approximately 256 MiB F32 before probabilities, GQA copies, masks, MLP and old/new KV overlap.
These are logical sizing examples, not fit bounds or measurements of the pinned artifact.

**Alternatives considered**: New custom kernels are not authorized by this research. Empirical
high-water marks alone miss unseen shapes. Charging every tensor view double-counts aliases;
charging only live logical tensors misses retained pool capacity and staging.

Sources: [commands](https://github.com/huggingface/candle/blob/c68b24997319b8d76e5a4e775dab197c7f26322a/candle-metal-kernels/src/metal/commands.rs),
[allocator](https://github.com/huggingface/candle/blob/c68b24997319b8d76e5a4e775dab197c7f26322a/candle-core/src/metal_backend/device.rs),
[GGUF Metal math](https://github.com/huggingface/candle/blob/c68b24997319b8d76e5a4e775dab197c7f26322a/candle-core/src/quantized/metal.rs),
[model dimensions](https://huggingface.co/Qwen/Qwen3-8B/blob/b968826d9c46dd6066d109eabc6255188de91218/config.json).

### C-04: Approved Public-Counter-Derived Admission Policy

**Decision**: On 2026-10-14 the maintainer explicitly approved conservative admission from public
VM counters and a live pressure observer while leaving OS severity unknown until an event arrives.
Preserve FR-015 and the historical T018 failure as evidence, not a test passed by this design.
Replace the old observation contract explicitly during the later design/implementation; do not
fabricate `Severity::Nominal` or remove completeness checks without replacing their source contract.
Reject the historical
`os_proc_available_memory()` selection: unavailable on macOS, with different semantics.
Select public `host_statistics64(HOST_VM_INFO64)` / `host_page_size()` as a deliberately
conservative additional-allocation allowance, not an exact system available-memory metric.

**Rationale**: A conservative candidate is `(free_count - speculative_count) * page_size`, using
checked arithmetic and validated counts. XNU explicitly includes speculative pages in free pages;
never add them again. Do not add inactive/purgeable/compressor/file-backed categories as though
they were disjoint immediately reclaimable capacity. Retain the initial ceiling
`min(floor(0.75 * physical), max(0, allowance - 8 GiB))` and independent Metal recommendation.
Keep that initial owned ceiling while warm; freshly check outstanding unmaterialized reservations
plus new backing growth, not already resident weights a second time.

Public dispatch pressure events report changes: absence of an event does not establish initial
normal pressure. Registration acknowledgment establishes a registered observer, not OS severity.
Require acknowledgment within 2 seconds, then four valid VM samples spanning at least 3.3 seconds,
separated by at least 1.1 seconds. Bootstrap requires sufficient free-minus-speculative headroom,
no pageout/swapout increases and no warning/critical event. Its result is `DerivedAdmit`, never an
observed OS NORMAL. Use the minimum bootstrap free value for the initial ceiling.

Required sources are physical bytes/page size, validated VM counters, registered/live observer,
independent process footprint/RSS and swap samples, and supported Metal observations from the
actual inference device. Missing, invalid or stale required sources yield `Unknown` and no admission.
OS severity may separately be unknown with a healthy observer; that is exposed explicitly.
Poll dynamic sources and observer-queue heartbeats every 100 ms; acquisition-start/heartbeat age
must be at most 250 ms. XNU can cache VM samples for a one-second window: use an explicit 1.25-second
effective-age uncertainty allowance, not a fictional kernel timestamp. These durations are chosen
policy bounds requiring target-OS qualification, not Apple freshness/delivery guarantees.

Track observation debt for newly materialized backing within that uncertainty window. Outstanding
additional demand plus new backing demand must fit `max(0, fresh_free - 8 GiB - debt)`. Never charge
the same physical obligation both as debt and as outstanding unmaterialized demand. Keep existing
resident weights only in the owned ceiling, not subtracted again from fresh free headroom.

Warning/critical events, growing pageouts/swapouts or exhausted headroom enter `Hold`: stop new
admissions, retire unpinned state from bounded ownership metadata without reading cache contents,
then cancel/synchronize affected work if safety cannot be restored. Preserve adverse event latches
across sleep, observer replacement and counter regression. Recovery needs 10 seconds of valid,
sufficient-headroom observations with unchanged counters; a latched adverse OS event additionally
needs an observed NORMAL event. NORMAL alone cannot authorize admission. Sleep/discontinuity
invalidates samples/observer epoch and repeats bootstrap. Queue heartbeats prove local liveness,
not guaranteed kernel event delivery. Neither sampling nor the host reserve eliminates OS races.

RSS/footprint and swap/pageout observations remain independent evidence/vetoes, not authorization
bytes. Thirty-two GiB remains a synthetic 48/40 GiB fixture result, not a physical-host entitlement.

**Alternatives considered**: Reject RSS-derived headroom, unconditional 75% allocation, treating
Metal's recommendation as free RAM, private APIs and availability-annotation bypasses. A broader
reclaimability estimator needs its own reviewed policy/evidence. This conservative policy may
reject usable capacity; it is not a promised host fit, process-RSS cap or completed qualification.

Sources: [VM counters](https://github.com/apple-oss-distributions/xnu/blob/xnu-12377.1.9/osfmk/mach/vm_statistics.h),
[statistics caching](https://github.com/apple-oss-distributions/xnu/blob/xnu-12377.1.9/osfmk/kern/host.c),
[pressure implementation](https://github.com/apple-oss-distributions/xnu/blob/xnu-12377.1.9/bsd/kern/kern_memorystatus_notify.c),
[public registration and pressure contract](https://github.com/apple-oss-distributions/libdispatch/blob/libdispatch-1542.0.4/dispatch/source.h),
[public dispatch observer](https://developer.apple.com/documentation/dispatch/dispatchsource/makeMemoryPressureSource(eventMask:queue:)),
[Metal recommendation](https://developer.apple.com/documentation/metal/mtldevice/recommendedmaxworkingsetsize).

### C-05: Candidate Tokenizer, Template and Tool-Decoding Stack

**Decision**: Evaluate tokenizers 0.23.1 (`7f1623b90b5adfb9bc327d4c3468d2f70bbce262`),
MiniJinja/minijinja-contrib 2.24.0 (`0ca749f7ba507514fa6b052c74130ae6ae472e03`) and
LLGuidance/toktrie 1.9.1 (`f0971424ec072d3e4d4196bcc7f31a2f60527df9`). Retain the existing
jsonschema 0.33 / serde_json validation direction rather than downgrade dependencies merely because
an older source tag was reviewed. Freeze a reviewed transitive lock and licenses before acceptance.
The first two libraries are Apache-2.0; LLGuidance/toktrie are MIT. Rust 1.94 graph compatibility
has not been built. These are sourced candidates, not a qualified bounded stack.

**Rationale**: Candle's public GGUF tokenizer reconstruction recognizes the Qwen2 pre-tokenizer,
but fallback metadata, special-token flags and NFC normalization create parity risks. Prefer a
separately provisioned original tokenizer at model revision
`b968826d9c46dd6066d109eabc6255188de91218`, with an independently established SHA-256, bounded
byte input and exact GGUF vocabulary/merge/special-token parity tests. No auxiliary hash is invented
and no download occurs during inference. Verify the original tokenizer-config template against the
GGUF template and freeze nonthinking/generation controls, whitespace, ordering and JSON escaping.

MiniJinja uses only the pinned template, builtins/serde/json/fuel and reviewed Python-compatible
methods. Disable loaders/debug output/arbitrary templates. Candidate limits: fuel 1,000,000,
recursion 32 and a bounded captured-output writer. These do not alone bound compilation, filter
allocations or tokenizer work. Compile only the provisioned template, not request-provided code.

Use a narrow raw-token-byte `TokenizerEnv` bridge to LLGuidance, not its stock HF bridge tied to
a different tokenizer version/special-token policy. Verify EOS/control/padding masks. Request-local
matcher order is compute mask, apply penalties and forbidden-token logits, sample, consume token.
Disable forcing/backtracking initially. Candle's post-softmax `sample_f` callback bypasses greedy
sampling, so it cannot be the sole grammar mask. Use the existing sampler after pre-softmax masking.

Candidate LLGuidance limits: initial lexer fuel 100,000, step lexer fuel 20,000, step items 4,096,
row items 512, lexer states 1,024 and grammar size 16,384; disable large-lexeme precomputation,
verbose errors, downloading helpers and cross-request grammar caching. These are complexity limits,
not evidence that the advertised 16 MiB grammar/parser budget is enforced.

Model-format tools use one `<tool_call>` JSON envelope, not wire-format OpenAI JSON. A bounded
format adapter recognizes delimiters; standard serde visitors enforce depth/nodes/duplicates before
collection growth and jsonschema validates the completed arguments. Buffer a call within the 256 KiB
output bound until it closes and validates, then emit stable ID/name and ordered argument fragments.
Text remains incremental. Auto permits text or one constrained tool; required/named constrain the
matching declaration; none masks all tool routes. No successful terminal for partial/invalid calls.
Preserve the entire existing HTTP schema subset; do not weaken optional/additional-property or
facet semantics to fit a compiler. Tool executors remain exclusively client-owned.

**Alternatives considered**: Reusing llama.cpp common would preserve the superseded dependency;
hand-writing a tokenizer or grammar engine adds unnecessary correctness risk. Lossy UTF-8 token
decoding corrupts trie bytes. Final schema validation alone does not provide constrained generation.
Timeouts, limited output writers and sampled heap peaks do not prove finite internal allocation.

Sources: [GGUF tokenizer](https://github.com/huggingface/candle/blob/c68b24997319b8d76e5a4e775dab197c7f26322a/candle-core/src/quantized/tokenizer.rs),
[tokenizers manifest](https://github.com/huggingface/tokenizers/blob/7f1623b90b5adfb9bc327d4c3468d2f70bbce262/tokenizers/Cargo.toml),
[MiniJinja manifest](https://github.com/mitsuhiko/minijinja/blob/0ca749f7ba507514fa6b052c74130ae6ae472e03/minijinja/Cargo.toml),
[Python methods](https://github.com/mitsuhiko/minijinja/blob/0ca749f7ba507514fa6b052c74130ae6ae472e03/minijinja-contrib/src/pycompat.rs),
[LLGuidance manifest](https://github.com/guidance-ai/llguidance/blob/f0971424ec072d3e4d4196bcc7f31a2f60527df9/parser/Cargo.toml),
[parser limits](https://github.com/guidance-ai/llguidance/blob/f0971424ec072d3e4d4196bcc7f31a2f60527df9/parser/src/api.rs),
[grammar syntax](https://github.com/guidance-ai/llguidance/blob/f0971424ec072d3e4d4196bcc7f31a2f60527df9/docs/syntax.md),
[Candle sampler](https://github.com/huggingface/candle/blob/c68b24997319b8d76e5a4e775dab197c7f26322a/candle-transformers/src/generation/mod.rs).

### C-06: Approved Same-Core Direct-Serving Baseline

**Decision**: On 2026-10-14 the maintainer explicitly accepted a full-chat standalone Candle
same-core serving harness as SC-006's direct baseline for mrun's owned runtime. It is independently
identified and accepts complete chat requests and produces real completion streams. No threshold
or short/8,192-token cell is changed. No performance outcome is claimed.

**Rationale**: Match weights, rendered token IDs, KV precision, exact operators, prefill chunks,
synchronization/readback, sampler/penalties/grammar, termination, threading and instrumentation.
Both paths are warm, no-reuse and non-speculative. Record request-origin and first-payload timing
boundaries, including direct/HTTP transport differences. Retain the frozen paired observations,
sessions and confidence gates in the benchmark protocol. Optimized mistral.rs and other compatible
engines remain separate full-product comparison lanes with their documented optimizations enabled.

**Alternatives considered**: A pretokenized decode-only inner loop removes chat/grammar/stream work
and cannot stand in for serving. An unrelated engine cannot measure same-core orchestration overhead.
Neither baseline is allowed to hide failed tasks or trade away the quality/resource contract.

### Remaining Phase 0 Gates

- **RESOLVED / Principle V**: On 2026-10-14 the maintainer explicitly approved a scoped exception
   for `001-agent-runtime` to use Candle's existing model operators with an mrun-owned runtime.
   It expires at feature completion or backend-direction change. Competitor comparisons, private
   backend types, all memory/correctness/cancellation gates and the no-new-kernels scope remain.
   See the [approved exception](plan.md#current-constitution-gate); no FR/SC is relaxed.
- **RESOLVED / FR-015 policy**: Derived admission, distinct OS severity, bootstrap/recovery,
   freshness uncertainty and observation debt are approved in C-04; hardware evidence remains unrun.
- **RESOLVED / SC-006 interpretation**: Full-chat same-core serving baseline approved in C-06;
   benchmark protocol and later harness must be reconciled before any comparison can count.
- **NEEDS CLARIFICATION / FR-014 allocation capability**: No reviewed, pinned patch establishes
   complete pre-allocation enforcement across the candidate Candle paths. An allocation-budget API
   must cover core pools, lower-level raw/data/private constructors, device RNG initialization,
   quantized uploads/dequantization, readback, GGUF metadata and CPU collections. Foreign/no-copy
   imports must be forbidden initially or use a reviewed budgeted-owner API. Lower-level multi-block
   sort scratch bypasses core pools. Leases must follow backing through pool/residency retention and
   GPU command completion, not expire with a tensor or wrapper clone. Post-allocation `allocatedSize`
   does not establish the conservative pre-allocation capacity bound. Pipeline/command metadata,
   driver-facing backing rules and complete host-load overlap still need bounded treatment.
- **NEEDS CLARIFICATION / FR-002..FR-005 host-library bounds**: C-05 selects a candidate stack,
   but lexer/template fuel, grammar-state limits and output writers do not enforce 64 MiB render/
   tokenize or 16 MiB grammar/parser byte ceilings. A reviewed conservative pre-allocation/work
   contract or enforceable library allocation path is required before Phase 0 can close.

The smallest candidate change is a reviewable allocation/ownership patch in Candle core and Metal
plumbing, not new kernels. A pre-initialization device-wide fallible budget acquisition would return
leases attached to unique backing and retained by commands until safe completion; pooled reuse
keeps its original charge. This describes the missing contract, not an existing accepted patch.
Do not claim capability merely by adding a later task named "implement allocator hooks".

Allocation inventory sources: [core allocator](https://github.com/huggingface/candle/blob/c68b24997319b8d76e5a4e775dab197c7f26322a/candle-core/src/metal_backend/device.rs),
[raw constructors](https://github.com/huggingface/candle/blob/c68b24997319b8d76e5a4e775dab197c7f26322a/candle-metal-kernels/src/metal/device.rs),
[sort scratch](https://github.com/huggingface/candle/blob/c68b24997319b8d76e5a4e775dab197c7f26322a/candle-metal-kernels/src/kernels/sort.rs),
[buffer ownership](https://github.com/huggingface/candle/blob/c68b24997319b8d76e5a4e775dab197c7f26322a/candle-metal-kernels/src/metal/buffer.rs),
[residency ownership](https://github.com/huggingface/candle/blob/c68b24997319b8d76e5a4e775dab197c7f26322a/candle-metal-kernels/src/metal/residency_set.rs),
[GGUF loader](https://github.com/huggingface/candle/blob/c68b24997319b8d76e5a4e775dab197c7f26322a/candle-core/src/quantized/gguf_file.rs),
[Metal resource size](https://developer.apple.com/documentation/metal/mtlresource/allocatedsize).

After these are resolved, freeze the dependency/profile pins, replace Phase 1 artifacts, recheck
the constitution, then regenerate/analyze tasks. No implementation or dependency upgrade is
performed by this source research. Full model/cancellation/quality/performance evidence remains unrun.

## Historical Research and Superseded Backend Decisions

The remainder preserves prior research and implementation evidence. Backend selection, toolchain,
native helpers, KV geometry and memory-API decisions below are superseded, not current Candle support.

**Reviewed**: 2026-10-13
**Evidence level**: Primary-source review plus the bounded implementation checks below;
no model download, inference benchmark or numerical qualification performed
**Decision status**: Implementation path selected below; production support remains qualification-gated

## Implementation Blocker: Mandatory macOS Availability Signal

Observed 2026-10-14 on the physical M4 Pro 48 GiB host, macOS 26.6.2,
macOS SDK 26.5, Apple Clang 21.0.0, pinned CMake 3.31.6 and Rust 1.85.0.
T012..T017 tests-first foundations and the private no-model native boundary compile;
strict all-target Metal Clippy and the content-discarding log/null-input probe pass.
These results are not model inference, fault recovery or adapter qualification.

T018 is blocked: the SDK's `usr/include/os/proc.h` declaration is explicitly
`API_UNAVAILABLE(macos)`. Apple Clang rejects `os_proc_available_memory()` for
the planned macOS target. The same header describes a process dirty-memory limit
minus dirty footprint, not system-wide additional available memory. Thus both its
platform availability and the frozen interpretation need reviewed correction.

The private observation bridge now leaves that signal missing; the Rust loader
rejects incomplete observations before model allocation. The focused physical test
compiles and fails with `os_proc_available_memory unavailable`; independent timestamp
and missing/unknown-signal tests pass. No replacement signal, private API, availability
override, RSS-derived authorization or weakened missing-observation gate was introduced.

Review D-04, the resource data model, T018/T019 and affected contracts for a supported
macOS capacity signal and its units, freshness and conservative interpretation.
Regenerate/analyze the affected plan/tasks after review. T018 remains unchecked,
dependent implementation is halted, and every empirical SC gate remains unpassed.

## Recommendation

Proceed with a narrow Rust-owned interactive runtime over a qualified existing Metal engine.
Measure agent turns and memory before investing in new kernels. Prefix caching is necessary but
not novel; the opportunity is robust state identity, bounded residency, interruptibility, and
better latency/quality tradeoffs on an actual developer workload.

Do not optimize for the largest model that barely loads. Optimize successful tasks per unit of
waiting within a safe memory envelope. Model quality, template correctness, reasoning policy,
and tool reliability can outweigh a modest decode-throughput advantage.

## Checked Claims

| Claim from the proposal | Finding | Primary evidence and limit |
| --- | --- | --- |
| Porting Go to Rust produces the main gain | Unsupported as a performance claim; inference and weight/state traffic remain | [Ollama source](https://github.com/ollama/ollama), [llama.cpp](https://github.com/ggml-org/llama.cpp); no controlled port experiment |
| Ollama already uses native accelerated inference | Supported; current tree also includes MLX paths | [Ollama source](https://github.com/ollama/ollama); pin runner choice, not just product version |
| Prefix reuse is an opportunity | Supported but already available in competitors | [llama.cpp server](https://github.com/ggml-org/llama.cpp/blob/efe3626b119ff81033e309c3ba07a770bef2c546/tools/server/README.md), [MLX LM](https://github.com/ml-explore/mlx-lm); compare caches enabled |
| Unified memory eliminates all copying | Not established; shared memory does not eliminate conversions, scratch, or duplicate buffers | [Apple device memory API](https://developer.apple.com/documentation/metal/mtldevice/recommendedmaxworkingsetsize); measure actual resident resources |
| A model's maximum context fits in 48 GB | False as a general inference; weights, KV, scratch, OS and editor all need space | [Qwen3-Coder card](https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct); card itself suggests shorter contexts for OOM |
| Grammar enforcement makes tools correct | Structural validity only; semantics and client permissions remain separate | [XGrammar](https://github.com/mlc-ai/xgrammar), [OpenCode providers](https://opencode.ai/docs/providers/) |
| Published Rust-native benchmark wins establish mrun superiority | No; they are author-reported for particular models/hardware/settings | [Imparo](https://github.com/zeraix/imparo); narrow samples, routes, and quality limitations are disclosed |

The pasted ELEKS service advertisement and decorative images provide no inference-engine evidence
and are not used as technical sources. No assessment of the vendor's services is implied.

## Backend and Comparator Matrix

| Candidate | Documented strengths | Qualification risks | Proposed role |
| --- | --- | --- | --- |
| llama.cpp / Metal | Mature GGUF baseline; optimized Apple Silicon; prefix reuse, unified KV, constraints, speculation, state save/restore | Rust FFI/ownership, batch-dependent numerics, actual memory accounting and safe interruption need proving | Mandatory baseline and first integration candidate |
| mistral.rs | Rust SDK; Metal; documented paged attention/prefix cache, batching and strict tool support | Model/precision/state API support differs; built-in agent execution must remain disabled; broad CUDA results do not establish M4 Pro speed | Mandatory Rust-native candidate |
| MLX LM | Apple Silicon ecosystem; prompt caching, configurable prefill chunks, quantization | Python API is not a drop-in Rust engine; format differences require quality-matched product comparison; rotating KV may reduce quality | Mandatory compatible MLX comparator; possible future adapter |
| uzu | Native Rust with Metal/CPU, structured output, tool examples, documented local server | Own converted format via lalamo; capability/state controls and conversion provenance need validation | Conditional Apple-first candidate/comparator |
| Imparo | Rust Metal/CPU/CUDA, persistent paged state; documented agent-oriented design | Narrow qualified architectures; public README says serial generation; author timing/quality limits; compatibility with selected coding model unproven | Research candidate/comparator if common model is admitted |
| SGLang | Agent-oriented serving; documented Apple Metal/MLX platform support | Platform/model-specific qualification; large server feature set need not fit the local product | Additional compatible comparator and scheduling design reference |

Sources: [llama.cpp](https://github.com/ggml-org/llama.cpp),
[mistral.rs](https://github.com/EricLBuehler/mistral.rs),
[MLX LM](https://github.com/ml-explore/mlx-lm), [uzu](https://github.com/trymirai/uzu),
[Imparo](https://github.com/zeraix/imparo), [SGLang](https://github.com/sgl-project/sglang).
Repository READMEs describe capabilities, not independently verified performance.

For every candidate, run the same capability spike: offline load; exact rendered token inspection;
warm continuation; prefix hit/miss/branch; state invalidation; total memory envelope; safe cancellation
during prefill/decode; tool schema/stream reconstruction; unload/reload; failure recovery. A backend
that cannot expose these controls cannot satisfy the feature by wrapping its HTTP server alone.

Prefer the smallest integration that meets these gates. Do not implement multiple production adapters
just to finish evaluation. A rejected candidate and its evidence are a useful result.

## Model Candidates and Memory Reasoning

- [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B): documented 8.2B dense parameters, GQA,
  tool use, thinking/nonthinking controls, Apache-2.0. A small qualification candidate, not a promise
  of strong coding-task success. The model card warns against greedy decoding in thinking mode.
- [Qwen3-Coder-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct):
  documented 30.5B total / 3.3B active parameters, 48 layers, four KV heads, nonthinking only,
  native 262,144 context and a special function-call format, Apache-2.0. All total weights, not only
  active experts, contribute to resident capacity. Its advertised context is not our initial limit.

A 30.5B model at idealized four bits/weight is about 15.25 billion bytes (14.20 GiB) for weights
alone, excluding scales, mixed precision, metadata, conversion overhead, scratch, and caches.
This is a sizing lower-bound illustration, not an artifact-size or fit guarantee.

For a conventional full-attention model, approximate KV bytes as
`2 * layers * tokens * KV_heads * head_dimension * bytes_per_element * independent_sequences`.
Read dimensions from actual artifact metadata; add allocator alignment and scratch. Quantized,
sliding-window, recurrent, hybrid, cross-layer-shared, or branched states need their own accounting.
Do not extrapolate that formula to all architectures or count shared pages once per reference.

Use exact model revisions and conversion hashes in the plan. Start at 16k total context, then extend
only after observed memory and long-context quality pass. Avoid automatic global wired-memory tuning.
Apple's recommended working set is advisory performance guidance, not the same as all physical RAM.

## Opportunities Beyond a Service Rewrite

1. Exact-prefix reuse with model/template/state identity and branch-safe sharing. Tokenize the full
   rendered request; reasoning-history removal and tool changes often break naive append assumptions.
2. Chunked prefill with cancellation checkpoints. Benchmark latency/memory tradeoffs rather than
   maximizing prompt tokens/s with an uninterruptible large chunk.
3. Unified residency accounting and demand-shaped context allocation. Avoid reserving every model's
   advertised maximum and duplicating host/GPU representations without evidence.
4. Cached grammar compilation and backend-native constraints. Evaluate an existing engine facility
   or [XGrammar](https://github.com/mlc-ai/xgrammar), including its documented community Rust bindings,
   rather than inventing a JSON parser/grammar decoder. Bound schema compile time and memory.
5. Adaptive speculation only later: cheap ngram/self speculation may suit repeated code; draft-model
   or MTP methods require model support, exact verification/rollback, acceptance-rate evidence,
   extra-memory accounting, and unchanged output-distribution/quality contracts.
6. KV precision, attention kernels, quantized matrix multiplication, fusion, and fewer GPU dispatches
   only after profiling. A speedup that changes quality or trades latency for throughput is a separate
   operating point, not a free improvement. Persisting state can cost more than recomputing it.

Apple documents Metal profiling/validation tooling and Metal 4 availability on Apple Silicon:
[Metal](https://developer.apple.com/metal/). This establishes tooling/hardware direction, not kernel
speed, deployment minimum, or model support. Use Instruments to distinguish dispatch overhead,
bandwidth, compute, and host synchronization before choosing a kernel project.

## Source Provenance and Double-Check Limits

Reviewed official project pages, llama.cpp's detailed server contract, Qwen's original model cards,
OpenCode's provider instructions, and Apple's memory documentation. Key cross-checks:

- llama.cpp's detailed server documentation confirms prefix reuse and explicitly warns that
  different batching paths may not produce bit-identical logits; a fixed seed alone is not proof.
- OpenCode documents `@ai-sdk/openai-compatible` for chat completions and a configurable local
  base URL; selecting the generic OpenAI provider can imply a different endpoint contract.
- Qwen's cards confirm active-vs-total MoE size and different thinking/tool-template semantics.
- Apple's API describes a recommended allocation threshold as an approximation, not a hard limit.
- Imparo's public documentation bounds its serial serving and published quality/performance claims.

Observed repository heads on review (provenance only, **not selected dependency pins**):

| Project | Observed revision |
| --- | --- |
| Spec Kit | `449da1c443996f34c535b3db9c2724165fcac2e2` |
| llama.cpp | `45451c366d4256f94fbc8845645e31a6b9155054` |
| mistral.rs | `28ed6daca455b4ae063896014d1ea698b912c539` |
| uzu | `e61e355a1caa14710db279bdfcc4fa27dab1cacf` |
| Imparo | `d4f6ad53023ee1f4b34fe89d4ebac9f168c58b41` |
| MLX LM | `ba8a08fd2ed72bbc53de84f89933f91fe4743e4a` |
| Ollama | `41503eb272d609125c8e84451d92c3546c5bcb30` |

Live documentation and pages can change. Before implementation, retrieve pinned source/docs, verify
licenses at those pins, and record actual binaries, artifact hashes, toolchain, and capability results.
No author speedup number is adopted as a project target or independently reproduced here.

## Spec Kit Findings

Official [Spec Kit repository](https://github.com/github/spec-kit) and
[existing-project guide](https://github.github.io/spec-kit/guides/existing-projects.html)
describe in-place initialization and constitution/specify/plan/tasks/implement/converge.
The installed CLI is 1.1.3; its help and bundled files were checked directly. Initialization used
`specify init --here --force --non-interactive --integration copilot --script sh` after confirming
there were no colliding managed files. Installed Copilot skills use hyphenated `speckit-*` names.
The git extension is optional and was not installed. Active feature routing uses `feature.json`.

## Planning Exit Evidence

The plan MUST resolve engine/model/artifact versions, numerical tolerances, wire/field/schema
contracts, memory-measurement method, bounded cancellation chunks, buffer sizes, SDK/OS/Rust minimums,
and the frozen fixture manifests. If capabilities cannot satisfy this spec, amend the feature openly
or reject the backend; do not mark unrun spikes as passed. No tasks or implementation are implied.

## Phase 0 Decisions (2026-10-13)

The following resolve planning choices, not experimental acceptance. No native build, model binary
download, client run, or physical-hardware measurement was performed. Source-level capability
evidence cannot establish the cancellation, quality, memory, or performance gates.

### D-01: Backend and Toolchain

**Decision**: One in-process llama.cpp/Metal adapter at
`45451c366d4256f94fbc8845645e31a6b9155054`, behind a small project-owned C ABI shim around C/C++
APIs. Rust 1.85.0 minimum, edition 2024; native arm64 macOS 15.0 product minimum, macOS 15+ SDK,
Apple Clang with C11/C++17, and CMake 3.31.6. Embed shader source using upstream's normal path;
record runtime shader compilation in cold-load time. Do not assume a packaged metallib.

**Rationale**: The pinned APIs expose sequence membership, synchronization, tokenization, native
grammar sampling, Jinja tool templates, and output parsing. macOS 15 is a deliberate product floor
for the residency-set path, not a claim that every upstream configuration requires it. Rust 1.85
introduces edition 2024. Keep native handles private and constrain unsafe code to the adapter.

**Alternatives considered**: A subprocess server hides state ownership and resource controls;
pure Rust kernels duplicate optimized work; mistral.rs/MLX/uzu remain comparison or later-adapter
candidates, not parallel production implementations.

Sources: [Rust 1.85](https://blog.rust-lang.org/2025/02/20/Rust-1.85.0/),
[pinned header](https://github.com/ggml-org/llama.cpp/blob/45451c366d4256f94fbc8845645e31a6b9155054/include/llama.h),
[Metal build](https://github.com/ggml-org/llama.cpp/blob/45451c366d4256f94fbc8845645e31a6b9155054/ggml/src/ggml-metal/CMakeLists.txt),
[chat helpers](https://github.com/ggml-org/llama.cpp/blob/45451c366d4256f94fbc8845645e31a6b9155054/common/chat.h).

### D-02: Primary Artifact and Rendering

**Decision**: `Qwen/Qwen3-8B-GGUF`, revision
`7c41481f57cb95916b40956ab2f0b139b296d974`, file `Qwen3-8B-Q4_K_M.gguf`,
5,027,783,488 bytes, SHA-256
`d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785`.
Public API model ID: `qwen3-8b-q4-k-m`. FP16 KV, normal artifact positions/RoPE, no adapters,
16,384 total tokens and 2,048 output maximum, `enable_thinking=false`. Artifact is Apache-2.0;
audit the pinned native dependency and fixture licenses before distribution.

**Rationale**: A small official full-attention artifact makes isolation/fit qualification tractable.
The revision-specific Hugging Face API independently supplied the size and LFS SHA-256; no weights
were downloaded. Read tokenizer and Jinja template bytes from the verified GGUF and hash them at
load. The artifact hash transitively pins those bytes; record the extracted hashes in run evidence.
Use `common_chat_templates_apply`/native parser and grammar helpers, not a hand-written Qwen parser
or the limited `llama_chat_apply_template` as general Jinja support. Freeze render time/kwargs,
BOS/EOS handling, and tool ordering; never let wall-clock template variables vary accidentally.

**Alternatives considered**: Qwen3-Coder-30B-A3B may improve coding quality but remains a later,
separately qualified model. Larger context and thinking are excluded from this initial matrix.

Sources: [artifact manifest](https://huggingface.co/api/models/Qwen/Qwen3-8B-GGUF/revision/7c41481f57cb95916b40956ab2f0b139b296d974?blobs=true),
[original configuration](https://huggingface.co/Qwen/Qwen3-8B/blob/b968826d9c46dd6066d109eabc6255188de91218/config.json).

### D-03: Shared Checkpoints and Boundary Replay

**Decision**: Explicit `kv_unified=true`, 32,768 physical KV cells, five sequence IDs: one active
and four retained. Retain at most four immutable prompt checkpoints with at most 16,384 unique
retained cells. `llama_memory_seq_cp` shares membership within the unified stream; do not serialize
state, duplicate full snapshots, shift positions, or use global sequence keep. Remove only the
active membership after GPU-safe cleanup. Retain validated prompt state, never request-local
sampler/grammar/RNG or generated suffixes. CPU reference tests use the same ownership contract.

**Rationale**: The pin's same-stream sequence copy adds membership rather than copying tensors;
cross-stream copy has different costs/constraints. Exact tokens and complete identity are checked
after lookup. Always evaluate the final prompt token for logits: effective reuse is
`min(longest_eligible_prefix, prompt_tokens - 1)`. If an identical prompt matched completely,
count that last token as one boundary replay, not a cache miss or infinite prefill throughput.

**Alternatives considered**: Host state blobs add memory and restore costs; ON_DEVICE snapshots
copy storage and have invalidation rules. Both are excluded from the first serving path.

Sources: [KV implementation](https://github.com/ggml-org/llama.cpp/blob/45451c366d4256f94fbc8845645e31a6b9155054/src/llama-kv-cache.cpp),
[state implementation](https://github.com/ggml-org/llama.cpp/blob/45451c366d4256f94fbc8845645e31a6b9155054/src/llama-context.cpp).

### D-04: Scheduling, Cancellation, and Failure

**Decision**: A dedicated owner thread performs all native operations; async HTTP tasks send
bounded commands and cancellation signals. Start with 128-token prefill calls and one-token
decode calls, `n_batch=n_ubatch=128`. Synchronize and check cancellation before scheduling another
call. One foreground slot, eight FIFO waiting entries, 30-second queue expiry, no superseding by
client ID. All terminal paths converge on one idempotent cleanup/reservation release.

**Rationale**: The abort callback is described as CPU-only, and Metal does not expose that generic
abort hook. No kernel-preemption promise is possible. Partial/fatal backend work must not publish
uncertain state. Synchronization has no public error return; detect/report available backend faults
and never treat unobservable asynchronous failure as verified recovery. Fault handling that cannot
preserve process safety requires a controlled unavailable state/restart, not a false success.

**Alternatives considered**: Large prefill batches risk cancellation latency; thread-per-request
native contexts duplicate memory; forced context destruction during GPU work is unsafe.
128 is a selected starting bound, not evidence for SC-004. Reduce it only through recorded,
pre-acceptance capability experiments; freeze the final value before timing comparisons.

### D-05: Accounting and Finite Limits

**Decision**: Use the limits and pressure procedure in [data-model.md](data-model.md). Observe
physical memory with `sysctl hw.memsize`, pressure-aware available allowance with
`os_proc_available_memory()`, and `recommendedMaxWorkingSetSize`/`currentAllocatedSize` separately.
Admission ceiling is `min(0.75 * physical_bytes, max(0, available_bytes - 8 GiB))` sampled before
load and each admission. Treat available bytes as additional capacity, not a total-process ceiling:
existing live allocations plus a proposed delta must fit the physical cap, while that delta must
fit the available allowance. Respect the device recommendation independently for GPU-backed bytes.
Missing/invalid observations fail closed; do not substitute the GPU recommendation for OS availability.

**Rationale**: FP16 KV sizing from the pinned original model config is 144 KiB/cell and 4.5 GiB
for this pool, before alignment/scratch. Verify actual GGUF dimensions and allocator sizes at load.
Count unique backing allocations once. The backend's C++ memory-breakdown helper is WIP and omits
some serving/host allocations; supplement it with a pre-allocation ledger and conservative scratch
allowance validated against peaks. Native preallocation must be gated before it occurs.
Eviction frees pool cells but not the preallocated buffer; reclaim physical memory by synchronized
context teardown/unload. Sample `proc_pid_rusage` physical footprint, RSS, system swap/pressure,
and Metal observations at 100 ms, preserving allocator high-water marks between samples.

**Alternatives considered**: Adding CPU and GPU totals double-counts unified buffers; RSS alone
misses transient ownership; OS tuning and swap-as-capacity violate the product policy.

Sources: [Metal device observations](https://github.com/ggml-org/llama.cpp/blob/45451c366d4256f94fbc8845645e31a6b9155054/ggml/src/ggml-metal/ggml-metal-device.m),
[memory helper](https://github.com/ggml-org/llama.cpp/blob/45451c366d4256f94fbc8845645e31a6b9155054/src/llama-ext.h),
[Apple available memory](https://developer.apple.com/documentation/os/os_proc_available_memory()),
[Apple working set](https://developer.apple.com/documentation/metal/mtldevice/recommendedmaxworkingsetsize).

### D-06: Pinned Client and Wire Boundary

**Decision**: OpenCode release `v1.19.0`, source commit
`add9b4ba5adfb895934cb1bf46f590342cec68d2`. Configure a local provider named `mrun`, text-only,
non-reasoning, tool-capable, base URL `http://127.0.0.1:8012/v1`, explicit model limits.
Freeze fields, error semantics and schema subset in [HTTP contract](contracts/http.md).

**Rationale**: This release has a native `@opencode-ai/llm` OpenAI-compatible route that delegates
to its OpenAI chat protocol: `/chat/completions`, `stream=true`, `include_usage=true`, `max_tokens`,
temperature/top-p/penalties/seed/stop, optional store/reasoning fields and tools. It also retains
`@ai-sdk/openai-compatible` 2.0.41, `ai` 6.0.168 and an SDK patch in its lockfile. The installed SDK
alone is not proof of the active request path. Both source mappings were reviewed; acceptance
must capture sanitized public-fixture requests from the released binary, including its tool schemas.
The package manifest reports an internal version distinct from the public release; pin release
and commit, not that internal version. Explicit client model configuration does not establish
that it calls model discovery, so test `/v1/models` separately.

**Alternatives considered**: The generic OpenAI provider can route to Responses rather than chat;
claiming universal OpenAI/Ollama parity is unnecessary. New client versions require contract review.

Sources at this immutable commit:
[release](https://github.com/anomalyco/opencode/releases/tag/v1.19.0),
[native protocol](https://github.com/anomalyco/opencode/blob/add9b4ba5adfb895934cb1bf46f590342cec68d2/packages/llm/src/protocols/openai-chat.ts),
[compatible route](https://github.com/anomalyco/opencode/blob/add9b4ba5adfb895934cb1bf46f590342cec68d2/packages/llm/src/protocols/openai-compatible-chat.ts),
[provider configuration](https://github.com/anomalyco/opencode/blob/add9b4ba5adfb895934cb1bf46f590342cec68d2/packages/core/src/v1/config/provider.ts),
[SDK wrapper](https://github.com/anomalyco/opencode/blob/add9b4ba5adfb895934cb1bf46f590342cec68d2/packages/core/src/plugin/provider/openai-compatible.ts),
[lockfile](https://github.com/anomalyco/opencode/blob/add9b4ba5adfb895934cb1bf46f590342cec68d2/bun.lock),
[versioned SDK mapping](https://unpkg.com/@ai-sdk/openai-compatible@2.0.41/dist/index.mjs).

### D-07: Numerical and Quality Contract

**Decision**: At least 20 positive exact-reference fixtures require identical output token IDs with
fixed execution shape, single-thread CPU reference, greedy selection and reset sampler state.
That lane does not claim Metal determinism. For FP16 KV Metal shape variations, compare raw finite
logits on identical teacher-forced continuations: centered-logit RMSE <=0.01 and softmax total
variation <=0.01 at every checked position, computed in float64 over the complete vocabulary.
Center each vector by its own arithmetic mean; TV is half the sum of absolute probability differences.
Compare warm reuse with cold recomputation, including boundary positions and every forced decode
position. Free-running sampled output need not be byte-identical. Native format/token/parser
agreement, exact schema validity and SC-007's 60 paired coding outcomes remain independent gates.

**Rationale**: Batch-shape changes can alter logits. These tolerances are proposed preregistered
limits, not upstream accuracy guarantees. Freeze before experiments; record any failure and require
reviewed amendment rather than silently widening tolerances. No quantized KV or speculation initially.

**Alternatives considered**: Seed-only equality is false evidence; text similarity misses state
corruption; numerical closeness alone does not prove coding/tool quality.

### D-08: Rust Libraries and Qualification Sequence

**Decision**: Tokio 1.x, Axum 0.8, Tower 0.5, Serde/serde_json 1.x, Clap 4.5, tracing 0.1,
SHA-256 via sha2 0.10, and jsonschema 0.33 with remote/file reference resolution disabled.
Use upstream native grammar facilities for generation and the Rust validator for final arguments;
do not implement a custom JSON Schema validator. Exact patch versions/checksums belong in the
implementation's reviewed Cargo.lock and MSRV gate, not invented in a planning-only repository.
Use a small audited native shim; no pretense that C++ common helpers are stable public C APIs.

**Rationale**: These libraries fit the narrow serving boundary. Validate the lock against Rust 1.85
and reject dependency choices that raise the declared minimum without review. No async runtime
reentrancy, blocking inference on Tokio workers, or unbounded native logging callbacks.

**Alternatives considered**: XGrammar adds an integration while the chosen backend already supplies
grammar/template tooling; introduce it only if a required supported schema cannot be qualified.

### D-09: Staged Qualification and Production Prerequisites

**Decision**: Resolve the four analysis findings through sequencing and authorization, preserving
D-01..D-08 and every specification threshold. Production memory observations/admission/preallocation,
prompt/output reservation and pressure handling precede any serving, including the cache-off MVP.
Adapter spikes authorize continued development only. A separately built runtime candidate requires
capability evidence bound to its exact executable SHA-256, harness/build, backend/model and profile.
Rebuild or identity change invalidates authorization; the selected binary must be requalified before
serving again. Startup verifies a matching report, not a destructive suite on every unchanged launch.

Production/default reuse additionally requires state/numerical, IC-001..020 protocol and SC-007
task-quality passes. The internal qualification harness can exercise candidate reuse on sealed public
fixtures before that report exists, with the same resource/lifetime/validation controls and no product
force-reuse switch. An isolated run-scoped loopback endpoint permits unmodified-client quality tests;
it never authorizes arbitrary production requests or transfers provisional pass status to serving.

Prepare draft fixture source inputs/licenses/assertions first; after the verified renderer exists,
seal token arrays, render/model identities and hashes before dependent runs. Later client captures
create a new sealed manifest revision, retaining the old version and invalidating affected evidence.

The future harness requires `--runtime-binary PATH` for runtime capability/reuse/acceptance suites;
record and exercise that target rather than just hash the qualification example. Adapter-only probes
carry no serving authorization. Freeze version-1 reports at 1 MiB, JSON depth 16, 65,536 nodes,
1,024 evidence references and 4 KiB individual strings; bulk records stay in hashed evidence artifacts.

**Rationale**: Constitution II/III require safe admission and quality before default optimization.
Exact-binary matching otherwise invalidates early reports after implementation. Draft/sealed fixture
states and candidate-only evidence break renderer and reuse bootstrap cycles without weakening gates.
Evidence writers, independent clocks and quality runners must exist before the runs they record.

**Alternatives considered**: An unbudgeted MVP, numerical-only reuse eligibility, carrying spike
reports across rebuilds, operator-declared passes, product force-reuse flags and invented token fixtures
are rejected. A separate production backend, new kernels or revised performance targets are unnecessary.

**Evidence level**: Delegated local design/constitution review on 2026-10-13, not a new external-source
retrieval, artifact download, source build or empirical validation. Reuse/performance remains unqualified.

## Planning Disposition

Planning choices are resolved. Backend qualification is **blocked: not run**, including shared-state
isolation, actual allocation peaks/preallocation enforcement, asynchronous failure observability,
template/schema agreement, client request capture, numerical gates and the 250 ms cancellation gate.
The design below is conditional on those proofs. Capability spikes must precede production-path
implementation; a failed capability stops the adapter decision and requires revised research/design,
not weakened feature requirements. All SC-001 through SC-009 remain blocked until evidence exists.
Fixture cases are defined in [qualification contract](contracts/qualification.md); their actual
input files, expected outputs, licenses and hashes must be materialized and frozen before execution.