# Data Model: Model Families

## RegistryEntry (compile-time table, `mrun-engine/src/models/registry.rs`)

| Field | Type | Notes |
| --- | --- | --- |
| model_type | `&'static str` | `config.json` `model_type` |
| text_model_type | `Option<&'static str>` | `text_config.model_type` when the config nests the text model (`mistral3` → `ministral3`, `gemma4` → `gemma4_text`) |
| family | `&'static dyn Family` | the module (contract) |
| release | `Release` | `V0_1` \| `After0_1`; only families whose module ships are in the table |
| tier | `1 \| 2` | highest tier of any model of this architecture; per-model tier is in the 004 manifest |
| aliases | `&'static [&'static str]` | Tier 1 aliases of this architecture (listed in 004 E034) |

Lookup is exact on the pair; a nested config whose text type is unregistered is refused
(`UnsupportedArchitecture { model_type, text_model_type, supported }`).

## FamilyConfig (per module)

Parsed from `config.json` (or its `text_config`) with unknown fields ignored and known fields
validated; every value the module uses is typed. Shared fields every module fills:

| Field | Type | Notes |
| --- | --- | --- |
| layers | `Vec<LayerPlan>` | one per decoder layer |
| hidden, vocab | u32 | |
| max_position | u32 | caps capacity (001 FR-008) |
| eos_extra | `Vec<u32>` | family-specific end tokens joined to the 001 EOS union (Harmony `<|call|>`, `<|return|>`) |
| quant | `QuantPlan` | below |

## LayerPlan

`{ mixer: Mixer, ffn: Option<Ffn>, norms: NormPlan }`

- `Mixer = Gqa(GqaSpec) | Mla(MlaSpec) | GatedDeltaNet(GdnSpec) | Mamba2(Mamba2Spec) | None`
  (`None`: Nemotron FFN-only blocks).
- `GqaSpec { q_heads, kv_heads, head_dim, qk_norm, v_norm, bias, gated_output, sinks: bool,
  window: Option<u32>, rope: RopeSpec, query_scaling: Option<Llama4Scaling>, scale: f32,
  kv: OwnKv | SharedFrom(layer) , k_eq_v: bool }`
- `MlaSpec { heads, q_lora_rank: Option<u32>, kv_lora_rank, qk_nope_dim, qk_rope_dim, v_head_dim,
  rope: RopeSpec }`
- `GdnSpec` as 001 R12; `Mamba2Spec { heads, head_dim, groups, state, conv_kernel,
  time_step_limit }`
- `Ffn = Dense { act: SwiGlu | GeGlu | SwiGluOai | Relu2, inter } | Moe(MoeSpec)`
- `MoeSpec { experts, top_k, router: SoftmaxTopK | TopKSoftmax | SigmoidBias { groups, topk_groups }
  | NormScaled, norm_topk: bool, routed_scaling: f32, shared: Option<SharedExpert { inter, gated }>,
  expert_act, latent: Option<u32> }`
- `RopeSpec = Default { theta, dims } | Linear { factor } | Yarn { factor, orig_max, beta_fast,
  beta_slow, mscale, mscale_all_dim } | Proportional { theta, rotated_dims, factor }`

## StateLayout (per layer; replaces 001 `ModelInfo.kv_heads/head_dim`) (post-0.1; at 0.1 001's `layer_kinds` + `kv_bytes_per_token`)

| Variant | Fields | Bytes |
| --- | --- | --- |
| `Positional` | `k_shape [heads, width]`, `v_shape [heads, width]`, dtype, `window: Option<u32>` | per token: `(k_heads·k_width + v_heads·v_width) · dtype` |
| `PositionBound` | arrays (`conv [k−1, conv_dim]`, `state [..]`) with dtypes | per checkpoint: sum of arrays |
| `None` | — | 0 (shared-KV and FFN-only layers) |

`ModelInfo` (which has `family: String` since 001 T040a) gains `syntax: SyntaxId`, `state: Vec<StateLayout>`,
`kv_bytes_per_token = Σ Positional`, `recurrent_bytes = Σ PositionBound`. `is_hybrid()` = any
`PositionBound`. 003 depends only on these sums and `is_hybrid()`.

Examples (computed; 003 and 004 fit use the measured values):

| Model | Positional per token | Position-bound per checkpoint |
| --- | --- | --- |
| Qwen3-8B | 144 KiB | 0 |
| Qwen3-Coder-30B-A3B | 96 KiB | 0 |
| Qwen3.6-35B-A3B | 20 KiB | ≈ 61 MiB |
| gpt-oss-20b | 48 KiB (12 layers windowed) | 0 |
| Devstral Small 2 | 160 KiB | 0 |
| GLM-4.7-Flash | ≈ 52.9 KiB (MLA latent 512 + rope 64) | 0 |
| Gemma 4 26B-A4B | ≈ 220 KiB (OQ-1) | 0 |
| Nemotron 3 Nano | ≈ 6 KiB | ≈ 46 MiB (dims U) |

## QuantPlan

`default: QuantSpec`, `overrides: Map<path, QuantSpec | Unquantized>`;
`QuantSpec { mode: Affine | Mxfp4, bits: 4 | 8, group_size: 32 | 64 | 128 }`. Validation:
`Mxfp4 ⇒ bits 4, group 32, uint8 scales, no biases`; `Affine ⇒ biases present`. Any other mode in
the file → `UnsupportedQuantization { path, mode }`.

## SyntaxId and ChatSyntax (`mrun/src/chat/syntax/`)

`SyntaxId = Hermes | QwenXml | Harmony | Mistral | GlmXml | Gemma`. Each entry:

| Field | Notes |
| --- | --- |
| probe_signature | text expected in the probe rendering (research R8) |
| markers | added tokens that must exist in the vocabulary |
| parser | streaming state machine → `ParseEvent` (contract) |
| args_mode | `RawJson` (stream on Chat/Responses) \| `Synthesized` (buffer per call) |
| reasoning | `ThinkTags` \| `HarmonyChannels` \| `MistralThink` \| `GemmaThought` \| `None` |
| history | template variables mrun sets (`preserve_thinking`, `clear_thinking=false`, `reasoning_effort`) and the message field reasoning goes into |

## ProbeResult

`{ detected: Option<SyntaxId>, accepted: &[SyntaxId], used: SyntaxId, tools_late: bool }`: shown by `mrun show` and
`/mrun/status`. `detected ∉ accepted` ⇒ `SyntaxMismatch` load error. Accepted sets at 0.1:
`qwen3` {Hermes}, `qwen3_moe` {QwenXml, Hermes}, `qwen3_5_moe` {QwenXml}, `gpt_oss` {Harmony},
`mistral` {Mistral}; after 0.1: `glm4_moe_lite` {GlmXml}, `gemma4` {Gemma}, `nemotron_h`
{QwenXml}.

## Tier1Alias (extends 004 ModelManifest)

Adds `family`, `syntax`, `tier: 1`, `state_bytes_per_token`, `checkpoint_bytes`,
`capacity_48gb`, `parity_ref` (path of the stored reference outputs). Tier 2 models get a manifest
on pull with `tier: 2` and no parity reference; a 0.1 alias that misses a Tier 1 gate keeps its
alias with `tier: 2` (FR-016).

## ParityFixture

`{ arch_key, config (tiny), seed, weights (mlx-lm saved), prompts, expected_ids, expected_top_logprobs }`
under `crates/mrun-engine/tests/families/<family>/`.
