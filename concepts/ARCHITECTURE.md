# Architecture

**Status**: Target design for the first release (2026-10-15, after the third audit). Only the
portable foundation (engine trait, identity, chat layer, generation loop) is implemented.

## Shape

One binary, two crates plus a sys crate:

```text
crates/
├── mlx-sys/     # bindgen over vendored mlx-c v0.7.0 + MLX v0.32.4 (cmake); checked-in bindings
├── mrun-engine/ # Engine trait + identity + budget (no MLX); behind feature `mlx`: safe MLX layer,
│                # model graphs (qwen3, qwen3_moe, qwen3_5_moe hybrid), sequence state, sampler,
│                # loader, scratch probe
└── mrun/        # the binary: cli, config, hub, chat (tokenizer, template, tool parsing,
                 # constraints), runtime (worker, queue, admission), cache (prefix cache),
                 # api (chat completions, messages, responses), bench
```

`mrun` talks to the engine through one primitive trait, `Engine`, implemented by `MlxEngine` and
by `ScriptedEngine`, a deterministic double that models allocation, growth, pipelining and
checkpoints so API, cache and scheduler logic is tested on Linux. It is a test seam, not a plugin
system.

## Models and sequence state

| Model | Layers | Attention state per token | Fixed recurrent state | Role |
| --- | --- | --- | --- | --- |
| Qwen3.6-35B-A3B, 4-bit (≈20 GB) | 30 Gated DeltaNet + 10 gated attention, MoE 256 experts | ≈20 KiB | ≈61 MiB per checkpoint | Default (confirmed by `mrun bench` before release) |
| Qwen3-Coder-30B-A3B, 4-bit (17.2 GB) | 48 attention, MoE 128 experts | 96 KiB | none | Attention-only alternative |
| Qwen3-8B, 4-bit | 36 attention, dense | 144 KiB | none | Reference for correctness; small Macs |

The **sequence state** of the active conversation is the attention KV of every attention layer
(contiguous per-layer arrays; `offset` marks the valid length) plus, for hybrid models, the
recurrent state of every linear-attention layer (convolution and delta-rule state at the current
position). Attention KV can be cut at any position. Recurrent state exists only at the position it
was computed for, so a hybrid model resumes from a **checkpoint**: a copy of all recurrent state
taken at a planned position during prefill.

Worked budget on the target (M4 Pro, 48 GB): `min(0.75 × 48, 36) − 8 = 28 GiB`. Qwen3.6 4-bit
leaves ≈6–7 GiB after weights and scratch, more than its 262K window needs (≈5 GiB). Qwen3-Coder
leaves ≈10 GiB, ≈105K tokens. Clients assume larger windows (Claude Code 200K, Codex 272K for
unknown models), so mrun reports its real capacity and `mrun launch` passes it on.

## Request flow

```text
client ──HTTP──▶ api (Chat | Messages | Responses ─▶ ChatRequest; namespaces flattened,
                 │      deferred tools held back, limits checked)
                 ▼
          chat::render (model template, HF-compatible Jinja ─▶ tokens + message boundaries)
                 ▼                                           count_tokens answers here
          runtime::queue (FIFO, keepalives: SSE comments, Anthropic ping, Responses in_progress)
                 ▼  one request at a time
          engine worker thread (owns MLX, the sequence state and the prefix cache)
             1. lookup: longest exact token match against the live path and stored branches
             2. admit: the prompt must fit; the output limit is clamped to what remains
             3. switch: keep the shared prefix in place, move the old tail out, move the
                matched branch in (hybrids: resume from the deepest checkpoint ≤ match)
             4. prefill the rest in chunks ending at message boundaries; chunk size shrinks with
                depth so one layer stays short; one layer in flight, cancel checked per layer;
                checkpoints captured at planned boundaries; only whole chunks are committed
             5. decode: forward and sample split (grammar mask computed between them), pipelined
                one token ahead; cancel checked per token; cache evicted on demand as state grows
             6. finish: keep exactly the committed tokens (prompt + emitted output)
                 ▼ bounded async channel of events; `Finished` always delivered
          api encoder (per API) ──▶ client
```

## Engine thread and memory

- One engine thread owns the device, weights, sequence state, prefix cache and one GPU stream.
  Competing servers that drive MLX from several threads have hit stream errors and kernel panics.
- **Budget by construction**: `budget = min(fraction × physical, recommended working set) −
  reserve`, with the reserve defaulting to `clamp(physical ÷ 6, 2 GiB, 8 GiB)`. Weights are
  measured after load; scratch is measured by probe chunks at two points, the deepest admissible
  position and the largest chunk (including the one-layer copy transient of growth and moves),
  cached per model identity, machine, macOS, MLX version, mrun version and chunk policy (001
  FR-008); capacity is sized from the remainder and never exceeds the probed depth. MLX memory
  and cache limits back the budget.
- **Allocation never exceeds the reservation**: per-layer attention arrays start at the prompt
  plus up to 4K tokens of output and grow in 4K steps, both clamped to the request's reserved
  total. Before any allocation the engine returns `NeedsMemory{bytes}` with nothing changed; the
  worker evicts cache and retries the call.
- **Cancellation bound**: one layer in flight; prefill chunk = `clamp(2048 × 16K / depth, 128,
  2048)` tokens so a layer stays near 100 ms up to 128K and ≤ ~200 ms at 262K (cancel at maximum
  depth is at risk against the 300 ms target, 001 SC-004). A cancelled prefill keeps every whole
  chunk; a yield (memory pressure) ends prefill after the current whole chunk and the worker
  continues with the same plan.

## Prefix cache

One tree of token sequences per model identity (weights, tokenizer, template, state format). The
**live path** is the active sequence state itself; every other stored conversation is a **branch**
hanging off it.

- **Continue** (next turn of the live conversation): nothing moves; the new tokens are appended.
- **Switch** (a subagent, a title request, another session, a retry with an edit): the shared
  prefix stays in place. The old tail is **moved** out into a branch segment, one layer at a time
  (copy the tail, then free that layer's slack), and the requested branch is moved in the same way.
  State is never held twice, so a 90K-token conversation survives a subagent on a 105K-token
  machine; moving its 8.2 GiB tail out and back costs ≈ 0.15–0.2 s of bandwidth. Tiny tails
  (under 256 tokens) are discarded instead.
- **Hybrid models**: checkpoints are captured during prefill at the end of system+tools, at the
  start of the last user or tool message, and at the end of the prompt's messages, before the
  generation header (`generation_start`, where the next turn's re-rendered reply diverges). At
  most 8 per conversation (≈ 0.5 GiB); the first, the two deepest and the newest at a user message
  start are kept, and the one nearest the midpoint is dropped first. A moved-out tail keeps its
  end state inside its segment. A switch or edit resumes from the deepest checkpoint at or below
  the match and re-prefills the gap.
- **Eviction**: least recently used branch leaves; the live path is never evicted while in use.
  Under memory pressure, branches go first, then an idle live path.
- **Miss report** on every request: cached and processed tokens, divergence position, and a cause
  (`new`, `system`, `edit`, `template`, `evicted`, `capacity`, `checkpoint`, `model`, `disabled`).

Exactness: reuse requires identical tokens and model identity. Reused state equals recomputation
up to the floating-point rounding any chunked prefill has; tests compare cache-on to cache-off
runs that use the same chunk split, so results must match bit for bit.

## Chat layer

- Tokenizer: `tokenizers` (HF `tokenizer.json`). Templates: `minijinja` configured like HF
  (`trim_blocks`, `lstrip_blocks`, a `tojson` without HTML escaping), golden-tested against HF
  renderings with tools.
- Tool calls: streaming parsers for the model's native syntax (Qwen3 JSON, Qwen3-Coder and Qwen3.6
  XML). Values end only at the full closing sequence, are never coerced or evaluated, and code
  containing the syntax's own tags round-trips. Strict tools and JSON response formats use
  `llguidance` masks computed between forward and sample.
- One internal `ChatRequest` / event model; three thin encoders.

## What is deliberately absent

No paged attention, no continuous batching, no multi-model router, no plugin system, no on-disk
cache (an opt-in SSD tier is spec 007), no server-side tools, no speculative work while idle.
