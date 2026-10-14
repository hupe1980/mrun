# mistral.rs Learning-Reference Study

**Status**: Learning reference only, 2026-10-14. mistral.rs is a competitor, benchmark comparator
and design reference for mrun. It is **not** a dependency, embedded runtime or source of copied code,
and this study is not a dependency decision (written against the superseded Candle plan; see [DECISIONS](../DECISIONS.md)).
**Method**: Static reading of the source only. Nothing was built, run or benchmarked. Statements
about runtime behavior are inferred from code and docs and are **unverified** until exercised.
Paths are relative to the mistral.rs repository root. `core/` abbreviates `mistralrs-core/src/`.

## 1. Identity

| Item | Value |
| --- | --- |
| Repository | https://github.com/EricLBuehler/mistral.rs |
| Commit / date | `28ed6daca455b4ae063896014d1ea698b912c539`, 2026-10-12 10:48:32 -0400 |
| Workspace version | 0.9.4, edition 2021, `rust-version = "1.94"` (`Cargo.toml:22-31`) |
| License | MIT, (c) 2024 Eric Buehler (`LICENSE`) |
| Candle pin | upstream `huggingface/candle` git rev `66a8cf184a5a519671454066b1b9efd446ec9f5c`, version 0.11.0, for candle-core/nn/metal-kernels/flash-attn-v3 (`Cargo.toml:42-45`). There is no `[patch]` section, so this is not a fork |
| Host libs | tokenizers 0.21.4, minijinja/contrib 2.14.0 (`pycompat`), llguidance 1.2.0 (`lark`), toktrie 1.4.0, toktrie_hf_tokenizers 1.2.0 (`Cargo.toml:115-147`) |

mrun's candidate is Candle `c68b24997319b8d76e5a4e775dab197c7f26322a` (also 0.11.0), with
tokenizers 0.23.1, MiniJinja 2.24.0 and LLGuidance 1.9.1 (research C-05). Which Candle rev is older,
and how far apart the two are, is **unverified**. mistral.rs shows that an upstream, unpatched
0.11.0 Candle can carry a full production runtime. It adds its own Metal kernels in its own crates
(section 3h) and does not patch Candle.

## 2. Architecture Overview

- **Layers**: server (HTTP/OpenAI/Anthropic/MCP), engine (queue, scheduler, tool loop, sessions)
  and pipeline (model, tokenizer, quantization, attention). The design doc is
  `docs/src/content/docs/developer/architecture.md:8-26`. Each loaded model gets one engine thread.
- **Request intake**: the bounded mpsc channel has capacity `DEFAULT_ENGINE_REQUEST_QUEUE_CAPACITY
  = 10_000` (`core/lib.rs:47`, created at `core/lib.rs:1178`). The engine drains it into an
  `AdmissionQueue` sized to the channel capacity (`core/engine/mod.rs:1122-1128`). Classes are
  Workload, OrderedControl, BypassControl and Shutdown (`core/engine/admission.rs:4-9`).
  `pop_admissible` admits by free sequence slots (`admission.rs:112-120`). Abandoned requests
  (closed response channel) are dropped before dispatch (`core/engine/mod.rs:828-848,1184`).
- **Engine loop**: `Engine::run` (`core/engine/mod.rs:1115-1161`). It holds one global RNG,
  `Isaac64Rng::seed_from_u64(SEED)` with `SEED = 0` (`core/engine/mod.rs:149,1120`). Each
  iteration calls `scheduler.schedule()` (`:1403`), then runs completion (decode) and prompt
  batches through `Pipeline::step` with explicit `CacheInstruction`s In/Out/Reset/Nothing
  (`core/pipeline/mod.rs:1324-1333,1971`).
- **Scheduler**: on non-paged builds (Metal default), FCFS bucketed by sequence length. Waiting
  sequences gain urgency (`core/scheduler/default_scheduler.rs:79-158`). The paged scheduler
  (`core/paged_attention/scheduler.rs`, 4,865 lines) does token-budgeted continuous batching.
- **Request/sequence model**: `Request::Normal` becomes one or more `Sequence`s (n choices form
  a group). Each `Sequence` owns its token list, per-layer `KvCache` copies (non-paged), sampler,
  optional per-request seeded RNG (`core/sequence.rs:759,922,1459-1463`), a tool-call state
  machine and a responder `Sender<Response>`.
- **HTTP/OpenAI**: `mistralrs-server-core` (axum) provides chat completions, completions,
  Responses, Anthropic Messages and MCP. Each request has a response channel buffer of 10,000
  messages (`mistralrs-server-core/src/handler_core.rs:22,400-404`). SSE emits chunks, an error
  event on failure, then `[DONE]` (`mistralrs-server-core/src/chat_completion.rs:387-440`).

## 3. Topic-by-Topic Comparison

### (a) Prefix caching and exact token-prefix reuse (FR-006..FR-010)

**How mistral.rs does it**: There are two independent mechanisms.
1. *Non-paged (the Metal default)*: `PrefixCacheManagerV2` (`core/prefix_cacher.rs:178-189`) is
   an `IndexMap<CacheKey, CacheElement>`. The key is the complete token vector plus an optional
   adapter generation (`:51-56`). Lookup is a linear scan for the longest shared token prefix
   (`:711-845`, `shared_prefix_len` at `:36-42`) with exact token comparison, not hashing. The
   match is clamped to KV coverage and to rotating-cache rewindability (`:833-845`). An exact full
   match returns `None`, so a fully cached prompt is re-prefilled in full (`:847-850`).
   `set_len(match_len)` truncates logically (`:873-880`). Finished sequences are inserted whole,
   prompt plus generated tokens (`:365-401`, called from `core/pipeline/sampling.rs:149-186,500-503`).
   A send-failure cancel sets `Done(Canceled)` without inserting (`sampling.rs:486-496`).
   Eviction is count-based (`--prefix-cache-n`, default 16; `mistralrs-cli/src/args/mod.rs:607-611`):
   the oldest entries on the device are dropped, not offloaded (`prefix_cacher.rs:405-469`).
   There are no byte budget and no recency update on hit (`IndexMap` keeps insertion order).
2. *Paged*: a vLLM-v1-style chain hash per full block (`core/paged_attention/block_hash.rs:1-10,
   126-146,232-263`) using a 64-bit `DefaultHasher`. Extra keys are multimodal hashes and the
   adapter generation. `CacheSalt` exists but is `#[allow(dead_code)]` (`:59-62`). Lookup trusts
   the hash and never compares stored tokens (`core/paged_attention/kv_cache_manager.rs:182-221`,
   `block_pool.rs:790-817`). The hit is capped at `num_tokens - 1` so that the last token is always
   recomputed for logits (`kv_cache_manager.rs:194-196`). Blocks are ref-counted, and free
   zero-ref blocks are evicted LRU from a free list (`block_pool.rs:7,319,853-900`).
- **Identity/invalidation**: neither key contains weight content, tokenizer, template/render
  controls, KV dtype/layout or principal. Identity relies on one cache per loaded pipeline;
  `evict_all_caches` runs on reload/re-ISQ paths (`core/utils/mod.rs:136,263`).
- **Branch isolation**: in the non-paged path, a hit clones `KvCache` structs that share Candle
  storage (`kv_cache/single_cache.rs:5-21`). `append` writes in place with `slice_set`
  (`single_cache.rs:102-137`). Isolation depends on `clone_in_cache` allocating a fresh batched
  tensor and copying (`kv_cache/mod.rs:486-560`). That runs only when the batch composition
  changes (`engine/mod.rs:1421-1426`) or the prompt has a cached offset (`:1507`). This invariant
  is implicit. Paged blocks are immutable once full and hashed, so writes go only to private tail
  blocks (by construction; **unverified** by test here).

**Fit with mrun**: Exact token comparison in the non-paged path matches FR-006. Hash-only paged
lookup violates FR-007 ("a hash lookup alone MUST NOT replace exact token/identity validation"),
and the dead `CacheSalt` falls short of FR-021 principal scoping. The keys omit most FR-007
identity fields. Count-based eviction does not satisfy FR-016's byte sub-budget. Implicit
copy-on-batch-change isolation is weaker than FR-009 requires. The `n-1` boundary recompute is a
legitimate "explicitly counted boundary replay" per FR-008, but mrun must count it in FR-019.
Re-prefilling a full exact match wastes work on SC-002 repeat turns.
**Recommendation**: *Adapt* the longest-exact-prefix scan and the ref-counted immutable-block idea.
*Avoid* hash-only lookup, count-based eviction, implicit storage sharing and dead salt keys.

### (b) KV cache layout, paged attention on Metal, snapshot/copy semantics

**How mistral.rs does it**: The non-paged `SingleCache` preallocates `[.., capacity, ..]` and grows
in steps of `CACHE_GROW_SIZE = 512` tokens. Each step allocates a new zeroed tensor and copies, so
growth briefly needs about twice the old capacity (`kv_cache/single_cache.rs:102-130`,
`kv_cache/mod.rs:426`). Snapshots record only `current_seq_len`, a logical rollback with no
backing copy (`single_cache.rs:6-8,58-78`). Paged attention has Metal kernels for
`reshape_and_cache`, `copy_blocks`, `gather_kv_cache`, paged attention and FP8 scales
(`mistralrs-paged-attn/src/metal/kernels/*.metal`). However, `--paged-attn auto` **disables it on
Metal** (`mistralrs-cli/src/args/paged_attn.rs:20,66-82`;
`docs/.../guides/perf/paged-attention.mdx`). When forced on with unified memory, the KV pool is
capped to the tokens needed for `max_num_tokens` or the model context
(`core/paged_attention/mod.rs:573-590`). Supported block sizes are 8/16/32 (doc). On Metal the
`--dtype auto` setting picks BF16 if a probe matmul succeeds (`core/utils/normal.rs:121-170`), so
KV is BF16.
**Fit with mrun**: plan.md starts the Candle GGUF route at F32 KV. For Qwen3-8B (36 layers,
8 KV heads, head_dim 128) that is about 288 KiB/token, or about 4.5 GiB at 16,384 tokens, against
about 2.25 GiB for BF16. This arithmetic comes from the published config and is **unverified**
against the artifact. mrun's FR-009 calls for independently backed frozen snapshots (plan
"Ownership candidate"). mistral.rs's length-only snapshots are not independent. The 2x transient
from grow-and-copy must be counted under FR-014.
**Recommendation**: *Adapt* preallocated per-request KV with explicit capacity, but reserve the full
capacity at admission instead of growing in 512-token steps. *Learn from* paged Metal kernels
only as reference. mrun forbids new kernels, and mistral.rs itself does not default to paged on
Metal. *Avoid* treating a length-only snapshot as a branch-isolation contract.

### (c) Memory planning and admission (FR-014..FR-016)

**How mistral.rs does it**: `resource_plan.rs` only splits a paged-KV budget across several models
in proportion to `max_num_seqs` (`core/resource_plan.rs:35-112`). At load time, automatic device
mapping estimates weights, max activation and KV for `max_seq_len * max_batch_size` (defaults
4096 x 1) against `MemoryUsage::available()`, layer by layer
(`core/pipeline/loaders/auto_device_map.rs:198-260,380-470`). On Metal, the budget is
`max(recommendedMaxWorkingSetSize, sysctl floor)`. The floor comes from spawning
`sysctl -n iogpu.wired_limit_mb`, or 2/3 of RAM (≤36 GiB) or 3/4 (>36 GiB), and `allocated` is
`currentAllocatedSize` (`core/utils/memory_usage.rs:127-150,401-426`). On unified memory the CPU is
not added as a fallback device (`auto_device_map.rs:383-392`). Admission at request time counts
sequences, not bytes (`engine/admission.rs:39-41,112-120`). The non-paged KV grows lazily
during execution.
**Fit with mrun**: The load-time fit check is a useful precedent for FR-015, which also requires
respecting the recommended working set. Allocations are **not bounded before execution**: no
per-request byte reservation exists, and activation size is a heuristic estimate. FR-014 is
therefore unmet by design. Taking `max()` with a sysctl floor can exceed Metal's own
recommendation, while FR-015 asks for the lesser of the limits. Reading a tunable sysctl is fine,
but mrun must not suggest raising it (no privileged tuning).
**Recommendation**: *Adapt* the per-layer weight/activation/KV accounting decomposition (useful
input to T114/T117). *Avoid* `max()` budgets, sequence-count-only admission and post-hoc
`currentAllocatedSize` as an admission signal (plan rejects post-allocation authorization).

### (d) Scheduling, queueing, cancellation and stream termination (FR-011..FR-013)

**How mistral.rs does it**: The defaults are `--max-seqs 32` (`mistralrs-cli/src/args/mod.rs:582-586`)
and a pending queue of up to 10,000. When the queue is full, the engine stops draining the channel
(`engine/mod.rs:870-886`) and senders then block. There is no explicit rejection and no queue
deadline (none found in `engine/` or `server-core`). Cancellation: a closed response channel marks
running and waiting sequences `Done(Canceled)` at the next schedule
(`default_scheduler.rs:333-338`, called at `engine/mod.rs:760,1192`). A failed chunk send cancels
the sequence (`pipeline/sampling.rs:486-496`). A global `TERMINATE_ALL_NEXT_STEP` flag also exists
(`engine/mod.rs:150-152`). The engine `await`s chunk sends on the 10,000-slot per-request channel
(`core/sequence.rs:2516-2533`), with no stalled-write deadline. SSE keep-alive is 10 s
(`mistralrs-server-core/src/streaming.rs:126`). On error, the stream emits an error event and
then `[DONE]` (`chat_completion.rs:411-433`).
**Fit with mrun**: Cancelling at the step boundary matches FR-012. Missing queue deadlines,
implicit backpressure instead of explicit "full" errors, a 10k-message channel instead of finite
byte-bounded buffering, and no 10 s stalled-write deadline conflict with FR-011/FR-013. `[DONE]`
after an error event risks a client reading the stream as a normal end (FR-004). Behavior with the
pinned OpenCode client is **unverified**.
**Recommendation**: *Adopt* "check cancellation at each scheduling boundary; cancel on send
failure". *Avoid* unbounded-in-practice channels and implicit backpressure. *Adapt* the admission
class split (bypass control vs. ordered workload) for mrun's FIFO with a 1 active + 8 queued cap.

### (e) Tool calling, grammar-constrained decoding and streaming tool calls (FR-002..FR-005)

**How mistral.rs does it**: It uses llguidance with a toktrie built from the HF tokenizer and fixes
up special-token markers (`core/pipeline/llg.rs:9-58`). User constraints (regex, Lark, JSON
Schema) map to a `TopLevelGrammar` (`llg.rs:60-77`). Tool grammars compose a Lark wrapper with a
`@json_body` JSON-Schema subgrammar (`core/tools/grammar.rs:15-33`). Arguments are constrained only
for `strict: true` tools; non-strict tools are parsed after generation
(`docs/.../guides/agents/tool-calling-basics.mdx:363-379`). For Qwen, the grammar activates
lazily once `<tool_call>` appears (`core/tools/parsers/qwen.rs:26-27,34-50,64-70`,
`tools/state.rs:152-161`). `required` tool choice forces the grammar after a deadline of
max_gen/4, clamped to 1024..4096 tokens (`tools/state.rs:11-13,163-184`). Lenient parsing
includes `fix_broken_json` string surgery (`core/tools/mod.rs:92-107`) and schema-based argument
coercion (`tools/mod.rs:332-373`). While streaming, text is held back while it "could be" a tool
call, and complete calls are emitted when parsed (`tools/state.rs:246-298`). Argument fragments
do not appear to stream incrementally (**unverified**). Unknown OpenAI request fields are
accepted and ignored (`docs/.../reference/openai-compatibility.md:40`).
**Fit with mrun**: Lazy trigger-activated grammar fits the model's native `<tool_call>` envelope
(research C-05). Constraining only `strict` tools, lenient JSON repair and silently ignoring fields
conflict with FR-003/FR-005 ("unsupported... MUST fail explicitly"; arguments must satisfy the
schema subset). Holding the call until it is complete matches C-05's "buffer a call until it
closes and validates". The llguidance build carries no bounds (no fuel/limits set in `llg.rs`),
unlike C-05's candidate limits.
**Recommendation**: *Adopt* llguidance with a Lark wrapper plus a JSON-Schema subgrammar, and lazy
activation on the tool-call delimiter. *Adapt* the toktrie special-token fixup with a narrow
bridge (C-05). *Avoid* JSON repair, post-hoc coercion, non-strict defaults and ignoring unknown
fields.

### (f) GGUF/Qwen3 loading and chat templates

**How mistral.rs does it**: The GGUF `qwen3` architecture maps onto the shared normal model through
bindings (`core/gguf/normal_registry.rs:48`, `core/models/qwen3.rs`;
`docs/.../reference/gguf-support.md:25`). The tokenizer is rebuilt from GGUF metadata and
**refuses to guess** when `tokenizer.ggml.pre` is missing (`core/gguf/gguf_tokenizer.rs:308-330`,
with qwen2 regex). `--tok-model-id` overrides it with original assets (serve.md:27). Quantized
matmul on Metal uses Candle `QMatMul` (`mistralrs-quant/src/gguf/mod.rs:20,45`). Templates use
MiniJinja with `pycompat` unknown-method callback, `lstrip/trim_blocks`, and custom
`raise_exception`/`tojson`/`strftime_now` (`core/pipeline/chat_template.rs:723-726,820-822`).
Templates are **rewritten** by string/regex substitutions before compile (`:799-812`,
`normalize_minijinja_compatibility` at `:477`). No fuel or recursion limit is set (none found).
Thinking defaults to **on** for chat completions unless `enable_thinking`/`reasoning_effort`
disables it (`openai-compatibility.md:52`).
**Fit with mrun**: Refusing unknown pre-tokenizers supports FR-018. Template rewriting changes the
effective template, so for FR-007 the identity must hash the post-normalization text and
render controls. Unbounded MiniJinja conflicts with T115/T118. Default-on thinking diverges from
mrun's nonthinking profile.
**Recommendation**: *Adopt* "fail on missing tokenizer metadata" and original-tokenizer parity.
*Avoid* silent template rewriting. If compatibility shims are needed, version them into the
template identity.

### (g) Telemetry and privacy defaults (FR-001, FR-019..FR-021)

**How mistral.rs does it**: The server binds to `0.0.0.0` by default
(`mistralrs-cli/src/args/server.rs:16`) and does not validate API keys (coding-agents guide).
Prometheus `/metrics` is on by default but pull-only (`mistralrs-server-core/src/metrics.rs:1,105-115,
150-183`), and no push telemetry exporter dependency was found. Logging all requests and
responses is opt-in via `--log <file>` (`mistralrs-cli/src/args/mod.rs:563-566`). Inference-time
network access exists: request media URLs (`server-core/src/media_source.rs:87-88,197`) and web
search (`core/search/mod.rs:254`), both through `fetch_limited` with a public-only redirect policy
(`core/remote_fetch.rs:9-30`). Model loading uses HF Hub unless `HF_HUB_OFFLINE=1`
(`docs/.../reference/environment-variables.md:16`). Server-side tools (code execution, shell,
MCP, `--tool-dispatch-url`) exist but are opt-in (`server.rs:30-40`). Remote model code (Python
`trust_remote_code`) was not found; models are Rust. On Metal, the memory probe spawns `sysctl`
(`memory_usage.rs:405-411`).
**Fit with mrun**: The `0.0.0.0` default and media/web fetch paths conflict with FR-001 (loopback,
no inference-time downloads). Pull-only metrics without prompt content is compatible with FR-021
in spirit.
**Recommendation**: *Adopt* opt-in content logging and pull-only metrics with bounded label sets.
*Avoid* non-loopback default binding and any URL-fetching input path.

### (h) Metal kernels and what Candle upstream lacks

**How mistral.rs does it**: For Qwen3 (head_dim 128), attention on Metal goes to upstream
`candle_nn::ops::sdpa` with `do_causal` (`core/attention/mod.rs:403,457-475`), so Candle already
supplies the main fused attention. mistral.rs adds its own kernels in its crates:
- `mistralrs-quant/src/metal_kernels/`: `rmsnorm_residual`, `fused_glu`, rotary q/k with
  positions, `topk_logits` (GPU top-k before sampling; `core/sampler.rs:1650`), sort/argsort/scan,
  softcap, sinks SDPA, a DK=512 flash-attn port from llama.cpp (`core/attention/mod.rs:405-452`),
  and AFQ/MXFP4/FP8/HQQ/BnB quant kernels (`metal_kernels/mod.rs:176-4336`).
- `mistralrs-paged-attn/src/metal/kernels/`: paged attention and block copy/gather.
- `core/metal/kernels/`: GDN/SSM for hybrid models (not relevant to Qwen3-8B).
These are the gaps mistral.rs chose to fill. Whether each one is missing from Candle at mrun's
pinned rev, and how much it speeds up Qwen3-8B Q4_K_M, is **unverified**.
**Fit with mrun**: plan.md forbids new kernels. Fused RMSNorm+residual, fused GLU, rotary and GPU
top-k explain part of any throughput gap measured in SC-006/SC-007 optimized-competitor lanes.
**Recommendation**: *Avoid* porting kernels in this feature. *Record* these as named explanations
for performance gaps and as candidates for a later, separately approved feature.

## 4. Lessons for mrun (prioritized)

1. **Exact tokens plus a full identity tuple, never hash-only** (FR-006/FR-007, T122 data model).
   Use an mistral.rs-style token comparison, keyed by an explicit identity tuple: weights,
   tokenizer, post-normalization template, render controls, KV dtype/layout, architecture version,
   adapters and principal. A hash index is permitted for lookup but must be verified against the
   tokens. Write the T122 Prefix Entry to make the paged hash-only path impossible.
2. **Make isolation explicit, not a side effect of batching** (FR-009, T119). mistral.rs isolation
   relies on `clone_in_cache` copying when the batch changes. mrun should copy frozen snapshots
   into independent backing at publish time and prove it with the T119 "independent snapshot
   backing" probe. Do not rely on Candle `Clone`, which shares storage.
3. **Reserve KV capacity up front; no 512-token grow-and-copy** (FR-014, T114/T117). Inventory
   the grow/copy transient as a reachable allocation path. mrun's single-active profile should
   reserve 16,384-token capacity at admission.
4. **Byte-budgeted LRU retained state, not entry counts** (FR-016, T122). Freeze the retained-state
   sub-budget in bytes, touch on hit (mistral.rs's `IndexMap` never does) and evict unpinned
   entries first.
5. **Reuse a full exact match** (FR-008, SC-002). mistral.rs re-prefills when the whole prompt
   matches. mrun should replay only the boundary token needed for logits and count it as
   "replayed" in FR-019 metrics, as the paged `n-1` cap does implicitly.
6. **Watch generated-token re-render mismatch** (FR-006, SC-002). mistral.rs caches prompt plus
   generated tokens. Agent turns re-render the assistant message through the template, and token
   identity holds only if render(history) reproduces the generated IDs. Add T122/SC-003 cases for
   Qwen3 tool-call and whitespace re-render.
7. **Re-decide F32 vs BF16 KV with numbers** (plan "Model/profile candidate", T122). mistral.rs
   runs BF16 on Metal. F32 KV doubles retained bytes (about 4.5 vs 2.25 GiB at 16K for Qwen3-8B,
   unverified) and changes how many prefixes FR-016 can retain.
8. **Use the lesser memory limit and enforce bytes at admission** (FR-014/FR-015, T117/T121).
   Reuse the weight/activation/KV decomposition from `auto_device_map.rs`, but take
   `min(75% RAM, available - 8 GiB, recommendedMaxWorkingSetSize)` rather than mistral.rs's `max()`.
9. **Explicit queue semantics** (FR-011/FR-013, T122 http contract). Use explicit "queue full" and
   30 s deadline errors, per-request byte-bounded output buffers with a 10 s stalled-write
   deadline, and no `[DONE]` after an error event unless the pinned OpenCode client is shown to
   treat it as failure.
10. **Request-local RNG and grammar state** (FR-009). mistral.rs shares one engine RNG (seed 0)
    across unseeded requests. mrun should derive request-local RNG state and keep the llguidance
    matcher per request (`llg.rs` creates a matcher per constraint, which is a good pattern).
11. **Strict-by-default tool grammar with lazy delimiter activation** (FR-005, T115/T118). Adopt
    the Lark wrapper plus `@json_body` JSON-Schema composition and `<tool_call>` trigger, with C-05
    limits set. Reject non-strict, repair-based or coerced arguments instead of fixing them.
12. **Comparator config is part of the evidence** (FR-022, SC-006/SC-007). Record the mistral.rs
    commit, Candle rev, features, flags, dtype and thinking setting in the qualification report,
    and keep its documented optimizations on in the optimized lane (section 5).

## 5. Comparator Notes (all **unverified until run**)

The goal is a fair optimized-competitor lane for Qwen3-8B Q4_K_M GGUF on M4 Pro Metal. This lane
is separate from the C-06 same-core baseline. Build at the pinned commit with
`cargo build --release --locked --features metal -p mistralrs-cli` (pattern from
`docs/.../developer/from-source.md:29`, feature from `reference/cargo-features.md:17`). Proposed
launch, with no network access. The `-f` file is the same D-02 artifact as mrun; use
`--prefix-cache-n 0` for the reuse-off lane:

```bash
HF_HUB_OFFLINE=1 mistralrs serve --host 127.0.0.1 -p 1234 --no-ui \
  -f /models/Qwen3-8B-Q4_K_M.gguf \
  --max-seqs 1 --max-seq-len 16384 --max-batch-size 1 \
  --prefix-cache-n 16
```

- **Prefix cache**: on Metal the default is non-paged, so reuse is the sequence-level
  `PrefixCacheManagerV2` (exact token prefix, count-evicted). Run two optimized variants:
  (A) the default `--paged-attn auto`, which is off on Metal, and (B) `--paged-attn on
  --pa-context-len 16384 --pa-block-size 32`, which enables block prefix sharing and the Metal
  paged kernels. Report both, and do not choose the faster one after observing results. Whether
  (B) supports this GGUF on Metal is **unverified**.
- **Tokenizer/template**: supply the same original tokenizer/template via `--tok-model-id <local
  dir>` or `--jinja-explicit` if GGUF metadata diverges. Record the rendered token IDs and compare
  them with mrun's (C-06 requires matched rendered tokens in matched lanes).
- **Thinking**: mistral.rs enables thinking by default. To match mrun's nonthinking profile, send
  `enable_thinking: false` or `reasoning_effort: "off"`, or use a nonthinking template. Note that
  the pinned OpenCode client may not send either field.
- **Precision**: `--dtype auto` gives BF16 activations/KV on Metal. Record this, and either pin
  `--dtype` to mrun's KV dtype for a matched cell or report it as a documented difference.
- **Privacy**: bind to loopback, leave `--log` unset, enable no MCP/search/code tools, and send no
  remote media.
- **Measurement**: use the frozen benchmark protocol and client, not `mistralrs bench`. Record
  warm/cold state, sampling seed (per-request `seed` to avoid the shared RNG), commit, Candle rev,
  features and all flags above with the raw outcomes (FR-022).
