# Architecture

**Status**: Target design for features 001–005 (2026-10-14). Nothing below is implemented yet.

## Shape

One binary, two crates plus a sys crate:

```text
crates/
├── mlx-sys/     # bindgen over vendored mlx-c v0.7.0 + MLX v0.32.4 (cmake); checked-in bindings
├── mrun-engine/ # Engine trait + identity + budget (no MLX); behind feature `mlx`: safe MLX layer,
│                # model graphs (qwen3, qwen3_moe), KV cache, sampler, loader, scratch probe
└── mrun/        # the binary: cli, config, hub, chat (tokenizer, template, tool parsing,
                 # constraints), runtime (worker, queue, admission), cache (prefix store),
                 # api (chat completions, messages, responses), bench
```

`mrun` talks to the engine through one primitive trait, `Engine` (reserve, prefill, decode, trim,
snapshot, restore), implemented by `MlxEngine` and by `ScriptedEngine` (a deterministic test double
for API, cache and scheduler tests on any OS, built with `--no-default-features`). The generation
loop (stops, UTF-8, events) lives in `mrun`. This is a test seam, not a plugin system (Principle VI).

## Request flow

```text
client ──HTTP──▶ api (parse: chat|messages|responses ─▶ ChatRequest)
                 │ validate limits, map tools, reasoning control
                 ▼
          chat::render (template ─▶ token ids) ── count_tokens answers here
                 ▼
          runtime::queue (FIFO, 8 waiting, 600 s wait deadline, keepalives, cancel on disconnect)
                 ▼  (one at a time)
          engine worker thread (owns MLX, the active sequence and the prefix store)
             1. prefix lookup: longest exact token match (pins nodes)
             2. admission: prompt + max_tokens must fit; evict unpinned cache as needed
             3. prepare: trim the live path (zero copy) or restore matched segments (copy)
             4. prefill remaining tokens in 2,048-token chunks (≤ 2 layers in flight, cancel check per layer)
             5. decode: sample on device (+ grammar mask if strict; unpipelined then),
                async-eval pipelined, cancel check per token, lazy cache eviction as KV grows
             6. finish: trim to committed tokens; record the live path and its boundaries
                 ▼ bounded channel of Events (token text, tool deltas, usage, finish)
          api encoder (per API SSE format) ──▶ client
```

## Engine thread and memory

- MLX objects are not shared across threads: one engine thread owns the device, weights, the active
  KV cache and the prefix store, and a single GPU stream. Async code sends commands over a channel
  and receives events over a bounded channel (stall timeout cancels).
- **Budget by construction** (D012): `budget = min(fraction × physical, recommended_working_set) −
  reserve`. Load weights (measured with `get_active_memory`), run one worst-case prefill chunk at
  maximum context to measure peak scratch (`get_peak_memory`), then size KV capacity from the
  remainder. `set_memory_limit` and `set_cache_limit` back the budget; `set_wired_limit` is set to
  the budget for the engine's lifetime (no privileged tuning).
- **KV cache**: per layer, K and V arrays allocated at admission (prompt + min(output, 4K), rounded
  to 256) and grown in 4,096-token steps one layer at a time (transient ≤ one layer, budgeted); `offset` marks the valid length; trim = move `offset`.
  Appends use in-place slice updates on solely-owned buffers.
- **Snapshots** are explicit deep copies (MLX may share buffers on `slice`/`copy`, and a shared
  buffer makes every later append copy the whole cache). Restore drops the active cache first,
  then copies segments into fresh buffers.

## Prefix store (003)

Radix tree over token ids, one tree per model identity (weights, tokenizer, template, KV dtype).
The active cache is the tree's live path: a continuing conversation copies nothing, and a
divergence inside it is handled by trimming. Segments (deep copies, one per node edge) are
materialized at message boundaries (end of system+tools, each user/tool turn, each assistant turn,
every 4,096 tokens) only when the live path is replaced by another conversation or branch.
Lookup walks tokens (never hashes alone). Eviction is LRU over unpinned leaves, assistant turns
before user/tool turns before system prompts; cached state may occupy capacity reserved for output
and is evicted lazily as output grows. Memory pressure (dispatch source) shrinks the store first.
Every lookup produces a miss report: divergence token, closest cached length, cause.

Exactness definition: a reused segment holds the stored bytes computed for exactly those tokens
under the same model identity. Recomputing the same tokens with different chunking may differ in
the last bits (MLX selects kernels by shape), so tests compare cache-on to cache-off with a logit
tolerance and greedy agreement, while staleness tests are structural (token prefix equality).

## Chat layer

- Tokenizer: `tokenizers` (HF `tokenizer.json`); template: `minijinja` with the model's own
  `chat_template`, rendered with tools and reasoning flags; only templates from pulled, hashed
  models are rendered.
- Tool calls: streaming parsers for the model's native syntax (Qwen3 `<tool_call>{json}`,
  Qwen3-Coder XML `<function=…><parameter=…>`). Strict tools and JSON response formats use
  `llguidance` masks (decode is not pipelined while a mask is active).
- One internal `ChatRequest` / `Event` model; three thin encoders map it to Chat Completions,
  Anthropic Messages and Responses wire formats.

## Testing strategy

- Linux CI (every PR): fmt, clippy, unit and integration tests with `ScriptedEngine`; `mlx-sys`
  and `mrun-engine` build with MLX's CPU backend and run tiny random-weight model tests.
- macOS CI: build only (hosted runners lack a reliable Metal GPU for MLX).
- Apple hardware (developer Mac / self-hosted): logits oracle against mlx-lm on pinned weights,
  greedy conformance, engine benchmark and `mrun bench` cells.

## What is deliberately absent

No paged attention, no continuous batching, no multi-model router, no plugin system, no on-disk
cache, no server-side tools. Each has a roadmap slot if evidence demands it.
