# Architecture

**Status**: Target design for features 001–005 (2026-10-14). Nothing below is implemented yet.

## Shape

One binary, two crates plus a sys crate:

```text
crates/
├── mlx-sys/     # bindgen over vendored mlx-c v0.7.0 + MLX v0.32.3 (cmake); checked-in bindings
├── mrun-engine/ # safe MLX layer (~60 fns), model graphs (qwen3, qwen3_moe), KV cache, sampler,
│                # weight loading, memory probe. Everything MLX lives here, on one thread.
└── mrun/        # the binary: cli, config, hub, chat (tokenizer, template, tool parsing,
                 # constraints), runtime (worker, queue, admission), cache (prefix store),
                 # api (chat completions, messages, responses), bench
```

`mrun` talks to the engine through one internal trait, `Engine`, implemented by `MlxEngine` and by
`ScriptedEngine` (a deterministic test double for API, cache and scheduler tests on any OS). This
is a test seam, not a plugin system (Principle VI).

## Request flow

```text
client ──HTTP──▶ api (parse: chat|messages|responses ─▶ ChatRequest)
                 │ validate limits, map tools, reasoning control
                 ▼
          chat::render (template ─▶ token ids) ── count_tokens answers here
                 ▼
          runtime::queue (FIFO, 8 waiting, 30 s deadline, cancel on disconnect)
                 ▼  (one at a time)
          engine worker thread (owns MLX, the active sequence and the prefix store)
             1. prefix lookup: longest exact token match ─▶ restore (copy) or hot path
             2. admission: reserve KV for prompt + max_tokens, evicting cache if needed
             3. prefill remaining tokens in 512-token chunks (cancel check per chunk)
             4. decode: sample on device (+ grammar mask if strict), async-eval pipelined,
                cancel check per token, publish snapshot boundaries for the prefix store
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
- **KV cache**: per layer, preallocated K and V arrays grown in 256-token steps up to the session's
  reservation; `offset` marks the valid length; trim = move `offset`. Appends use in-place slice
  updates on solely-owned buffers.
- **Snapshots** are materialized slice copies (MLX copies a shared buffer on write, so sharing would
  make every later append copy the whole cache). Restore copies into the active buffers.

## Prefix store (003)

Radix tree over token ids, one tree per model identity (weights, tokenizer, template, KV dtype).
Nodes hold immutable KV segments covering token ranges, created at message boundaries (end of
system prompt, end of each user/tool turn, end of each assistant turn) and at most every 4,096
tokens. Lookup walks tokens (never hashes alone), restores the matched segments into the active
cache, then prefills the remainder. The hot path (same conversation continuing) keeps the active
cache and only appends. Eviction scores segments by recompute seconds saved per byte, decayed by
age, and never evicts segments referenced by the running request. Memory pressure (dispatch source)
shrinks the store first. Every lookup produces a miss report: divergence token, closest cached
length, cause.

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
  `llguidance` masks computed on the CPU while the GPU runs the next step.
- One internal `ChatRequest` / `Event` model; three thin encoders map it to Chat Completions,
  Anthropic Messages and Responses wire formats.

## Testing strategy

- Linux CI (every PR): fmt, clippy, unit and integration tests with `ScriptedEngine`; `mlx-sys`
  and `mrun-engine` build with MLX's CPU backend and run tiny random-weight model tests.
- macOS (nightly or on demand): Metal build, logits oracle against mlx-lm on pinned weights,
  conformance (greedy tokens), `mrun bench` cells.

## What is deliberately absent

No paged attention, no continuous batching, no multi-model router, no plugin system, no on-disk
cache, no server-side tools. Each has a roadmap slot if evidence demands it.
