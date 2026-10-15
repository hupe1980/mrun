# Research: Model Families

Sources: [ARCHITECTURE](../../concepts/ARCHITECTURE.md) (Models; Sequence state),
[DECISIONS](../../concepts/DECISIONS.md) D010, D015, D019, [landscape §4](../../concepts/research/landscape-2026-10.md),
[mlx-rust](../../concepts/research/mlx-rust.md), 001 [research](../001-engine-core/research.md)
R2–R4, R12–R14, 002 [research](../002-agent-apis/research.md) §3. Code read at: mlx-lm `ba8a08f`
(`models/{gpt_oss,ministral3,mistral3,glm4_moe_lite,mla,gemma4_text,nemotron_h,ssm,base,cache,
rope_utils,switch_layers,activations}.py`, `tool_parsers/*`, `tokenizer_utils.py`), mlx-c v0.7.0
(`a341b49`), MLX v0.32.4 (`36969c0`), llama.cpp `common/parsers/{gpt-oss,ministral3,gemma4,
qwen3-coder}.cpp` and `models/templates/*`, vLLM `tool_parsers/`, `parser/{harmony,nemotron_v3}.py`.

Markers: [V] verified in code or primary source; [U] secondary source or unverified; [E] computed
estimate.

## R1: Registry and family contract

- **Decision**: A compile-time table maps `(model_type, text_config.model_type?)` to a family
  module. A module exposes: `parse_config`, `sanitize`, `build` (compose blocks), `state_layout`,
  `syntaxes` (accepted set, default first), `tiny_configs` (for Tier 2 tests). 001 T040 builds the
  table for 0.1; module names follow 001 (`mistral` for `mistral3` + `ministral3`). No dynamic loading, no traits for
  third parties (Principle VI).
- **Keys at 0.1**: `qwen3`, `qwen3_moe`, `qwen3_5_moe`, `gpt_oss`, `mistral3`+`ministral3`. Tier 2
  candidates for the same modules: `qwen3_5` (dense Qwen3.6-27B), bare `ministral3`. After
  0.1: `glm4_moe_lite`, `gemma4`+`gemma4_text` (and bare `gemma4_text`), `nemotron_h`.
- **Rationale**: mlx-lm selects by `model_type` too (`models/<type>.py`); nested configs
  (`mistral3` dispatches to `ministral3`, `mistral4` or `llama` by `text_config.model_type` [V
  `mistral3.py`]) need the pair, and refusing `mistral3`+`llama` keeps Tier 2 honest.
- **Alternatives**: Plugin trait objects (rejected: Principle VI, no current requirement);
  architecture inferred from tensor names (fragile).

## R2: Block library, mapped to the families

| Block | Options | Used by (mlx-lm source) |
| --- | --- | --- |
| GQA attention | q/k RMSNorm, biases, gated output (`q_proj` → q + gate), sinks, window, partial rotary, Llama-4 query scaling, explicit scale, KV-less (shared) layers, K=V | qwen3, qwen3_moe, qwen3_5 (gated), gpt_oss (bias, sinks, window), ministral3 (Llama-4 scaling, optional window), gemma4_text (q/k norm, v-norm without scale, scale 1.0, K=V on global layers, KV sharing) |
| MLA | q-LoRA or plain q, `kv_lora_rank`, `qk_nope`/`qk_rope`/`v_head` dims, `embed_q`/`unembed_out` multi-linear, absorbed path at L = 1 | glm4_moe_lite [V `glm4_moe_lite.py:125-173`, `mla.py`] |
| Gated DeltaNet | conv kernel, heads, Metal kernel | qwen3_5 (001 R12) |
| Mamba2 | heads, head dim, groups, state size, conv kernel, `time_step_limit`, gated RMSNorm; prefill `ssm_attn` (ops), decode `ssm_update_kernel` (Metal) | nemotron_h [V `nemotron_h.py`, `ssm.py:219-260`] |
| Dense FFN | SwiGLU, GeGLU (`gelu_approx`), SwiGLU-OAI (gate ≤ 7, up ∈ [−7, 7], α 1.702, `(up + 1)`), squared-ReLU MLP (Nemotron `-` blocks) | all; gpt_oss experts use SwiGLU-OAI [V `activations.py`] |
| MoE | router: softmax→top-k (qwen), top-k→softmax (gpt_oss), sigmoid + correction bias, group-limited top-k, routed scaling (glm, nemotron), norm + per-expert scale (gemma4); shared experts (gated: qwen3_5; plain: glm, nemotron, gemma4); `SwitchGLU` (gated) or `SwitchMLP` (ungated, relu², optional latent projection: nemotron) via `gather_qmm`, sort by expert when `indices.size ≥ 64` | qwen3_moe, qwen3_5_moe, gpt_oss, glm4_moe_lite, gemma4_text, nemotron_h |
| Norms | RMSNorm, RMSNorm without scale, gated RMSNorm (`precise_swiglu`) | all |
| RoPE | default/linear, YaRN (`factor`, `original_max_position_embeddings`, `beta_fast/slow`, `mscale*`), proportional (partial rotation), Llama-4 attention scaling `1 + β·log(1 + ⌊pos / orig_max⌋)` | gpt_oss (YaRN), ministral3 (YaRN + Llama-4 β) [V `ministral3.py:42-53`], gemma4 (proportional on global layers) [V `rope_utils.py`] |
| Output | tied or separate head; final logit soft-cap `tanh(x/c)·c` | gemma4 (c = 30) |

- **Rule**: an option enters the block library when a second family needs it or when it changes the
  state layout (sinks, window, KV sharing, MLA widths). Single-family arithmetic (Gemma per-layer
  input embeddings, Gemma router scaling) stays in the family module.
- **Compiled ops**: mlx-lm compiles `swiglu`, `swiglu_oai`, `geglu`, `logit_softcap`,
  `precise_swiglu` (shapeless). 001 R14 applies per op: bind `mlx_compile` or prove unfused
  bitwise-equal on the family's shapes.

## R3: Sliding-window layers keep full KV (decided) and read only the window

- **Decision**: Every attention layer stores full KV (001 R4 layout). A sliding layer with window W
  at query position q attends to keys `[max(0, q − W + 1), q]`. Implementation: slice the key/value
  range to the window (plus the chunk) and apply the causal-window mask inside the slice, so a
  sliding layer costs O(W + L) per chunk, not O(depth). Sinks (per-head logits) go to
  `fast::scaled_dot_product_attention`'s `sinks` argument [V mlx-c `fast.h:195-205`].
- **Why not mlx-lm's rotating buffer**: `RotatingKVCache` keeps `max_size` (+ chunk − 1) entries in
  rotated order and is trimmable only while `offset < max_size` [V `cache.py:423-555`]; earlier
  positions are unrecoverable, which breaks 003's arbitrary-position restore and move.
- **Parity reference** (decided in 001 R16): the reference process substitutes a temporal-order
  full-KV windowed cache (same slice as mrun) for `RotatingKVCache`; stock mlx-lm must agree token
  for token. Rotated decode order is the reason: once the ring wraps, SDPA sums keys in rotated
  order and may differ in the last bits. A second candidate, mlx-lm's own `KVCache` with a
  full-depth window mask (`create_causal_mask(N, offset, window_size)` [V `base.py`,
  `cache.py:141-151`]), needs no substitution but attends over all keys, so its bits may differ
  from the slice too; MF-2 records both comparisons. Logits equality is required bit for bit only
  mrun-vs-mrun (003 SC-005).
- **Memory cost [E]**:

| Model | Sliding layers | Full-KV state per token | Ring alternative | Capacity on 48 GB [E] |
| --- | --- | --- | --- | --- |
| gpt-oss-20b | 12 of 24, W = 128 | 24 × 2 × 8 × 64 × 2 B = 48 KiB | 24 KiB/token + 3 MiB | ≈ 14.7 GiB free → > 300K, capped at max position 131,072: no loss |
| Gemma 4 26B-A4B | 25 of 30, W = 1024 | 25 × 2 × 8 × 256 × 2 B + 5 × 2 × 2 × 512 × 2 B ≈ 220 KiB | ≈ 20 KiB/token + 200 MiB | ≈ 11.7 GiB free → ≈ 55K full-KV vs ≥ 262K ring |
| Devstral Small 2 | none in config [U] | 40 × 2 × 8 × 128 × 2 B = 160 KiB | — | ≈ 11.9 GiB free → ≈ 78K |

  Free memory = 28 GiB budget − weights (repo bytes) − ≈ 2 GiB scratch (001 probe measures it).
  Gemma head dims (256 local, 512 global) are mlx-lm defaults [U for 26B-A4B]; KV heads 8 local /
  2 global [U config via search]. Hence OQ-1.

## R4: MLA is positional state

- **Decision**: Per MLA layer store `kv_latent [1, 1, T, kv_lora_rank]` (after `kv_a_layernorm`) as
  "K" and `k_pe [1, 1, T, qk_rope_head_dim]` (after RoPE) as "V", exactly what mlx-lm's `KVCache`
  holds [V `glm4_moe_lite.py:143-150`]. Prefill (L > 1) expands K/V through `embed_q`/
  `unembed_out`; decode (L = 1) uses the absorbed form (q projected into latent space, output
  unembedded) [V `:160-170`]. The RoPE term enters SDPA as an additive mask of `q_pe · k_pe`.
- **Cost [E]**: GLM-4.7-Flash 47 layers × (512 + 64) × 2 B = 54,144 B ≈ 52.9 KiB/token; ≈ 10.3 GiB
  free on 48 GB → ≈ 200K, near its 202,752 maximum position.
- **Engine impact**: K and V widths differ per layer (001's uniform `kv_heads × head_dim` becomes a
  per-layer layout, FR-005). Trim, split and append are unchanged.

## R5: Mamba2 is position-bound state

- **Decision**: Port `NemotronHMamba2Mixer` and `ssm.py` (`ssm_attn` for prefill, the
  `ssm_update` Metal kernel for decode, as mlx-lm dispatches [V `ssm.py:219-260`]). State per layer:
  conv state `[1, kernel − 1, conv_dim]` and SSM state `[1, heads, head_dim, state]`. Checkpoints
  copy both (001 FR-019). Nemotron blocks are single-mixer residual blocks (`M`, `*`, `-`, `E`) [V].
- **Cost [E, dims U]**: Nemotron 3 Nano 30B-A3B: 52 layers = 23 Mamba2 + 23 MoE + 6 attention,
  2 KV heads [U secondary]. Attention ≈ 6 × 2 × 2 × 128 × 2 B = 6 KiB/token; SSM ≈ 23 × 64 × 64 ×
  128 × 4 B ≈ 46 MiB per checkpoint if float32. Read from the pinned config at port time (OQ-5).
- **Note**: `ssm_attn` results depend on the chunk split like any chunked prefill; 003's equality
  tests fix the split.

## R6: Quantization formats

- **mlx-c v0.7.0** [V `mlx/c/ops.h:539-552, 915-933`]: `mlx_quantized_matmul`, `mlx_gather_qmm`
  and `mlx_quantize` take `const char* mode`; biases may be null.
- **MLX v0.32.4** [V `mlx/primitives.h:155`, `backend/metal/kernels/fp_quantized.metal` at tag
  v0.32.4]: `QuantizationMode {Affine, Mxfp4, Mxfp8, Nvfp4}`; Metal instantiates `mxfp4` (group 32,
  4 bits) for qmv/qmm/split-k/aligned and `fp_gather_qmm_rhs`, plus `fp_gather_qmv/qmm` kernels; SDPA
  supports `sinks` [V `scaled_dot_product_attention.cpp`]. mlx-rs hard-codes affine
  ([mlx-rust](../../concepts/research/mlx-rust.md) §3), which own `mlx-sys` avoids.
- **Decision**: `QuantSpec {mode, bits, group_size}` per path from `config.json` `quantization`
  (with per-path overrides); mode inferred from scales dtype/shape when absent (uint8 scales, no
  biases → mxfp4 when bits 4 and group 32). Refuse other modes by name. gpt-oss routers are 8-bit
  affine (mlx-lm `quant_predicate`) [V `gpt_oss.py`].
- **Repos [V download sizes from rapid-mlx `model_sizes.json`]**: `mlx-community/gpt-oss-20b-MXFP4-Q8`
  12.10 GB, `…-MXFP4-Q4` 11.21 GB (experts MXFP4; the suffix is the precision of other layers [U]).

## R7: Chat syntaxes

| Syntax | Families | Tool call | Reasoning | Sources |
| --- | --- | --- | --- | --- |
| Hermes JSON | Qwen3-8B, Qwen3-30B-A3B-Instruct (Tier 2) | `<tool_call>\n{"name":…,"arguments":{…}}\n</tool_call>` | `<think>` | 002 §3 [V] |
| Qwen XML | Qwen3-Coder, Qwen3.6, Nemotron 3 | `<tool_call>\n<function=N>\n<parameter=K>\nV\n</parameter>\n</function>\n</tool_call>` | `<think>` | 002 §3; Nemotron template [V llama.cpp `NVIDIA-Nemotron-3-Nano-30B-A3B-BF16.jinja:92,148`]; vLLM `nemotron_v3` reuses `qwen3_config` [V] |
| Harmony | gpt-oss | `<|start|>assistant<|channel|>commentary to=functions.N <|constrain|>json<|message|>{…}<|call|>`; recipient may sit in the role header | `analysis` channel; `final` = answer; stray `<|channel|>commentary` prefix tolerated | llama.cpp `gpt-oss.cpp` [V]; OpenAI cookbook [U] |
| Mistral | Devstral Small 2, Ministral 3 | `[TOOL_CALLS]N[ARGS]{json}` repeated, no end marker (ends at next `[TOOL_CALLS]` or EOS); legacy `[TOOL_CALLS][{…}]` | `[THINK]…[/THINK]` (reasoning models) | llama.cpp `ministral3.cpp`, Ministral-3 template `:104-112` [V]; mlx-lm `tool_parsers/mistral.py` [V]; Devstral Small 2 template [U, OQ-3] |
| GLM XML | GLM-4.7-Flash | `<tool_call>N<arg_key>K</arg_key><arg_value>V</arg_value>…</tool_call>`; string V raw, others `tojson` | `<think>…</think>`; turns before the last user message render `</think>` only unless `clear_thinking=false` | llama.cpp `GLM-4.7-Flash.jinja` [V] |
| Gemma | Gemma 4 | `<|tool_call>call:N{k:<|"|>str<|"|>,n:1}<tool_call|>` | thought channel [U, verify at port] | mlx-lm `tool_parsers/gemma4.py` [V]; llama.cpp `gemma4.cpp` |

- **Parsing rules (all)**: 002 FR-004 and 002 research §3 (hold-back buffers for multi-token
  markers, token-id fast path for added tokens, values end only at the full closing sequence in
  context, no coercion). Per syntax:
  - Harmony: the `<|call|>` token ends the arguments, so closing-marker ambiguity cannot occur in
    the value; arguments are raw JSON (streamed on Chat/Responses). A recipient other than
    `functions.*` (built-in `browser.*`, `python`) → `tool-unknown`. `<|call|>` and `<|return|>`
    join the EOS set; the generation loop treats `<|call|>` as a tool-call finish.
  - Mistral: the JSON object's end is found by the depth/string scanner (002 §3 Hermes rule);
    text after it before the next `[TOOL_CALLS]` is an error-tolerant gap reported as content.
  - GLM: a value ends at `</arg_value>` followed by `<arg_key>` or `</tool_call>`; schema string →
    raw text; otherwise JSON, failure → FR-004 fallback. mlx-lm and vLLM fall back to
    `ast.literal_eval` [V `glm47.py:_deserialize`]; mrun does not.
  - Gemma: convert `<|"|>`-delimited strings and bare keys to JSON without coercion; buffered per
    call; a string may contain `}` or `<tool_call|>` text, so the call ends only at a balanced
    brace followed by `<tool_call|>`.
- **History stability** (feeds 003 `template` cause, MF-3): Harmony template drops a tool turn's
  analysis once a later final message exists and renders the last final as `<|return|>`
  (llama.cpp replaces it with `<|end|>` for continued inference, issue 15417) [V template
  `:277-311`]; GLM drops earlier reasoning unless `clear_thinking=false`; Qwen XML and GLM
  re-serialize non-string arguments with `tojson`.
- **Template functions**: gpt-oss calls `strftime_now("%Y-%m-%d")` and reads `reasoning_effort`
  [V template `:201-206`]; minijinja gets `strftime_now` (local date, OQ-4).

## R8: Template probe

- **Decision** (002 FR-032 owns the probe; 010 adds the accepted set): Render a fixed conversation (system, user, assistant with one call `mrun_probe(a =
  "x")`, tool result, generation prompt) with the model's template and tools; match the text
  against each registered syntax's signature (`<tool_call>\n{"name": "mrun_probe"`,
  `<function=mrun_probe>`, `to=functions.mrun_probe`, `[TOOL_CALLS]mrun_probe[ARGS]`,
  `<tool_call>mrun_probe<arg_key>`, `<|tool_call>call:mrun_probe{`), and check that its markers
  are added tokens. Rendering errors (e.g. templates that `raise_exception` on tools) count as no
  match.
- **Rationale**: mlx-lm infers from template substrings (`_infer_tool_parser` [V
  `tokenizer_utils.py:619-660`]), which can match text in comments; rendering tests what the model
  will see. Same `model_type` with different syntaxes (Qwen3-30B-A3B-Instruct-2507 Hermes vs
  Qwen3-Coder XML, both `qwen3_moe`) makes a single syntax per family refuse valid Tier 2 repos,
  hence an ordered accepted set (OQ-7).

## R9: Models and repos

| Alias | Repo (pin at port) | Arch | Size | Facts |
| --- | --- | --- | --- | --- |
| `gpt-oss` (004) | `mlx-community/gpt-oss-20b-MXFP4-Q8` [V search + sizes] | `gpt_oss` | 12.10 GB | 24 layers alternating sliding (128) / full, 64 q heads, 8 KV heads, head dim 64, 32 experts top-4 [V config copies via search], sinks, biases, YaRN [V mlx-lm], Apache-2.0 [U] |
| `devstral` (004) | `mlx-community/Devstral-Small-2-24B-Instruct-2512-4bit` [V listed with size in rapid-mlx; OptiQ-4bit variant V search] | `mistral3` + `ministral3` | 15.14 GB | 40 layers, 32 q / 8 KV heads [U], YaRN + Llama-4 query scaling [V mlx-lm; model card via search], vision tower dropped, 256K context [U, sources disagree with 128K] |
| `glm-4.7-flash` | `mlx-community/GLM-4.7-Flash-4bit` [V sizes] | `glm4_moe_lite` | 16.87 GB | 47 layers, MLA q-LoRA 768, kv-LoRA 512, rope 64, nope 192, v 256 [V config copies + mlx-lm], 64 routed + 1 shared expert, top-4, first layer dense, sigmoid router |
| `gemma4` | `mlx-community/gemma-4-26b-a4b-it-4bit` [V sizes] | `gemma4` + `gemma4_text` | 15.37 GB | 30 layers: 25 sliding (1024) + 5 global, last layer global; 8 / 2 KV heads; 128 experts top-8 + shared [V search: model card, config]; K=V on global layers, proportional RoPE, soft-cap 30 [V mlx-lm defaults] |
| `nemotron-3-nano` | `mlx-community/NVIDIA-Nemotron-3-Nano-30B-A3B-4bit` [V search] | `nemotron_h` | ≈ 17.8 GB (lmstudio 4-bit [V sizes]) | 23 Mamba2 + 23 MoE + 6 attention [U]; Qwen XML tools and `<think>` [V template]; vLLM uses `qwen3_coder` [V recipe] |

## R10: Parity tests

- **Tier 1**: 001 R7/SC-001 procedure per model: `scripts/gen_reference.py` runs pinned mlx-lm on
  the same weights and chunk split, stores token ids and top-n logprobs. For sliding families it
  substitutes a temporal-order full-KV windowed cache for `RotatingKVCache` in the reference
  process and also records a stock run for token agreement (001 R16, T043a).
- **Tier 2 / every architecture**: a tiny random-weight config per registered key (2–4 layers,
  every block option on, hidden 64–128, quantized and unquantized variants, MXFP4 included) saved
  by mlx-lm with fixed seed; mrun on the MLX CPU backend matches token ids greedily for 32 steps
  and logits within the CPU kernels' bitwise result. Runs in Linux CI (001 plan: CPU backend).
  Metal-only kernels (GDN, Mamba2 decode) compare against their ops references on CPU and against
  each other on macOS (001 R12 pattern).

## R11: Porting checklist (one family)

1. Module (config, sanitize, build, state layout, default syntax) + tiny-model parity.
2. Syntax entry if new + parser suite + probe signature.
3. Template goldens (≥ 10 fixtures; HF `apply_chat_template` or mistral-common).
4. Real-weight parity (`--check`) on the pinned repo.
5. Alias with pinned revision, hashes, license and fit numbers (state bytes, capacity on 48 GB).
6. Client smoke: 004 read-edit-test task in Claude Code, Codex, OpenCode (5 of 5 each).
7. Cache exactness cases (003 SC-005 style) for the family's state kinds.
8. Bench cells (005 synthetic + agent suite) recorded.
9. Locality check green.

## Sources (accessed 2026-10-15)

- mlx-lm `ba8a08f`, mlx-c v0.7.0, MLX v0.32.4, llama.cpp, vLLM: local checkouts listed above.
- rapid-mlx `rapid_mlx/model_sizes.json` (download sizes per repo).
- Web search: gpt-oss-20b config mirrors; GLM-4.7-Flash config copies (Geodd, Intel AutoRound);
  google/gemma-4-26B-A4B-it config and model card, Kaitchup Gemma 4 analysis; vLLM Nemotron 3 Nano
  recipe; mlx-community model listings (gpt-oss MXFP4-Q8/Q4, Nemotron 3 Nano 4-bit, Devstral
  OptiQ-4bit); OpenAI Harmony cookbook. Hugging Face itself was not reachable from this session;
  every pin and config value marked [U] is re-read from the pinned revision in the port task.
