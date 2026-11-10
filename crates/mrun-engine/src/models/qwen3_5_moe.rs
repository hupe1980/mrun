//! `qwen3_5_moe` (Qwen3.5/3.6 MoE, e.g. Qwen3.6-35B-A3B) and `qwen3_5` (dense, e.g.
//! Qwen3.5-9B): a port of mlx-lm 0.32.0 `mlx_lm/models/qwen3_5_moe.py` → `qwen3_5.py` (text
//! model only).
//!
//! A hybrid decoder: every `full_attention_interval`-th layer is gated full attention (q/gate
//! split, q/k RMSNorm, partial RoPE, `o_proj(attn · σ(gate))`); the others are Gated DeltaNet
//! layers (fused q/k/v projection → causal depthwise conv1d over the last `K − 1` inputs plus the
//! chunk → SiLU → L2-normalized q/k → gated delta rule → gated RMSNorm with `z` → `out_proj`).
//! Every layer's MLP is the sparse MoE with a sigmoid-gated shared expert (a dense MLP when the
//! config has no experts, as in `qwen3_5`). The vision tower and multi-token-prediction weights
//! are dropped. Every size the graph needs is read from the config or refused by name.

use serde_json::Value;

use super::common::{self, Attention, AttentionConfig, Head, Mlp, int};
use super::moe::{Ffn, MoeConfig, SparseMoe};
use super::{LoadError, Model, Part, activations, gated_delta};
use crate::engine::LayerKind;
use crate::load::{Linear, Quantization, RmsNorm, Weights, invalid};
use crate::state::{LayerState, Recurrent};
use crate::tensor::{self, Array, Dtype, fast};

/// Linear-attention dimensions.
#[derive(Clone, Copy, Debug)]
struct DeltaConfig {
    k_heads: i32,
    v_heads: i32,
    k_dim: i32,
    v_dim: i32,
    kernel: i32,
    eps: f32,
}

impl DeltaConfig {
    fn key_dim(&self) -> i32 {
        self.k_heads * self.k_dim
    }

    fn value_dim(&self) -> i32 {
        self.v_heads * self.v_dim
    }

    fn conv_dim(&self) -> i32 {
        2 * self.key_dim() + self.value_dim()
    }
}

/// mlx-lm `GatedDeltaNet`.
#[derive(Debug)]
struct GatedDeltaNet {
    in_proj_qkv: Linear,
    in_proj_z: Linear,
    in_proj_b: Linear,
    in_proj_a: Linear,
    out_proj: Linear,
    /// Depthwise kernel `[conv_dim, K, 1]`.
    conv: Array,
    dt_bias: Array,
    a_log: Array,
    /// Gated RMSNorm weight over `v_dim`.
    norm: RmsNorm,
    cfg: DeltaConfig,
}

impl GatedDeltaNet {
    fn load(
        weights: &mut Weights,
        prefix: &str,
        quant: &Quantization,
        hidden: i32,
        cfg: DeltaConfig,
    ) -> Result<Self, LoadError> {
        let p = |name: &str| format!("{prefix}.{name}");
        let linear =
            |w: &mut Weights, name: &str, in_dims: i32| Linear::load(w, &p(name), quant, in_dims);
        let conv = weights.take(&p("conv1d.weight"))?;
        if conv.shape() != [cfg.conv_dim(), cfg.kernel, 1] {
            return Err(invalid(
                "weights",
                format!(
                    "{} has shape {:?}, expected [{}, {}, 1]",
                    p("conv1d.weight"),
                    conv.shape(),
                    cfg.conv_dim(),
                    cfg.kernel
                ),
            ));
        }
        Ok(Self {
            in_proj_qkv: linear(weights, "in_proj_qkv", hidden)?,
            in_proj_z: linear(weights, "in_proj_z", hidden)?,
            in_proj_b: linear(weights, "in_proj_b", hidden)?,
            in_proj_a: linear(weights, "in_proj_a", hidden)?,
            out_proj: linear(weights, "out_proj", cfg.value_dim())?,
            conv,
            dt_bias: weights.take(&p("dt_bias"))?,
            a_log: weights.take(&p("A_log"))?,
            norm: RmsNorm::load(weights, &p("norm"), cfg.eps)?,
            cfg,
        })
    }

    /// The layer over `x` `[1, S, hidden]`, advancing `state` (zeros when unset) by `S`.
    fn forward(&self, x: &Array, state: &mut Recurrent) -> tensor::Result<Array> {
        self.forward_split(x, state, &[], &mut Vec::new(), false)
    }

    /// [`Self::forward`] with the recurrence run in segments ending at the chunk-relative
    /// positions `splits` (ascending, inside `(0, S)`), threading the state from one segment to
    /// the next (003 SC-009, ollama's snapshot splits): the projections and the convolution run
    /// once over the whole chunk, unchanged. The state after each split (conv tail and delta
    /// state) is pushed to `captured`. With `rows` (a verify) the convolution runs once per
    /// position over its own `K` inputs, the call a decode step makes, so each row has its bits.
    fn forward_split(
        &self,
        x: &Array,
        state: &mut Recurrent,
        splits: &[i32],
        captured: &mut Vec<Recurrent>,
        rows: bool,
    ) -> tensor::Result<Array> {
        let c = &self.cfg;
        let len = x.dim(1);
        let qkv = self.in_proj_qkv.forward(x)?;
        let z = tensor::reshape(&self.in_proj_z.forward(x)?, &[1, len, c.v_heads, c.v_dim])?;
        let b = self.in_proj_b.forward(x)?;
        let a = self.in_proj_a.forward(x)?;

        let conv_state = match state.conv.take() {
            Some(conv) => conv,
            None => tensor::zeros(&[1, c.kernel - 1, c.conv_dim()], x.dtype())?,
        };
        let conv_input = tensor::concatenate(&[&conv_state, &qkv], 1)?;
        // The next call's convolution needs the last K − 1 inputs.
        let total = conv_input.dim(1);
        state.conv = Some(tensor::contiguous(&tensor::slice(
            &conv_input,
            &[0, total - (c.kernel - 1), 0],
            &[1, total, c.conv_dim()],
        )?)?);
        let conv = |input: &Array| tensor::conv1d(input, &self.conv, 1, 0, 1, c.conv_dim());
        let conv_out = if rows {
            let outs = (0..len)
                .map(|i| {
                    conv(&tensor::slice(
                        &conv_input,
                        &[0, i, 0],
                        &[1, i + c.kernel, c.conv_dim()],
                    )?)
                })
                .collect::<tensor::Result<Vec<_>>>()?;
            tensor::concatenate(&outs.iter().collect::<Vec<_>>(), 1)?
        } else {
            conv(&conv_input)?
        };
        let conv_out = activations::silu(&conv_out)?;

        let key_dim = c.key_dim();
        let parts = tensor::split_sections(&conv_out, &[key_dim, 2 * key_dim], -1)?;
        let heads = |t: &Array, h: i32, d: i32| tensor::reshape(t, &[1, len, h, d]);
        let q = heads(&parts[0], c.k_heads, c.k_dim)?;
        let k = heads(&parts[1], c.k_heads, c.k_dim)?;
        let v = heads(&parts[2], c.v_heads, c.v_dim)?;
        let (q, k) = gated_delta::normalize_qk(&q, &k, f64::from(c.k_dim).powf(-0.5), 1e-6)?;
        let (out, next) = if splits.is_empty() {
            gated_delta::update(
                &q,
                &k,
                &v,
                &a,
                &b,
                &self.a_log,
                &self.dt_bias,
                state.state.as_ref(),
            )?
        } else {
            // `x[:, from..to]` of a `[1, S, ...]` array.
            let part = |t: &Array, from: i32, to: i32| {
                let shape = t.shape();
                let mut start = vec![0; shape.len()];
                let mut stop = shape.clone();
                start[1] = from;
                stop[1] = to;
                tensor::slice(t, &start, &stop)
            };
            let mut carried = state.state.clone();
            let mut outs = Vec::with_capacity(splits.len() + 1);
            let mut from = 0;
            for &to in splits.iter().chain([&len]) {
                let (out, next) = gated_delta::update(
                    &part(&q, from, to)?,
                    &part(&k, from, to)?,
                    &part(&v, from, to)?,
                    &part(&a, from, to)?,
                    &part(&b, from, to)?,
                    &self.a_log,
                    &self.dt_bias,
                    carried.as_ref(),
                )?;
                if to < len {
                    // The conv tail at `to`: the K − 1 inputs before it.
                    let conv = tensor::contiguous(&tensor::slice(
                        &conv_input,
                        &[0, to, 0],
                        &[1, to + c.kernel - 1, c.conv_dim()],
                    )?)?;
                    captured.push(Recurrent {
                        conv: Some(conv),
                        state: Some(next.clone()),
                    });
                }
                outs.push(out);
                carried = Some(next);
                from = to;
            }
            let refs: Vec<&Array> = outs.iter().collect();
            (
                tensor::concatenate(&refs, 1)?,
                carried.expect("one segment"),
            )
        };
        state.state = Some(next);

        // Gated RMSNorm (mlx-lm `Qwen3NextRMSNormGated`): silu(z) · rms_norm(out) in f32.
        let normed = fast::rms_norm(&out, Some(&self.norm.weight), self.norm.eps)?;
        let out = activations::precise_swiglu(&out, &z, &normed)?;
        self.out_proj
            .forward(&tensor::reshape(&out, &[1, len, c.value_dim()])?)
    }

    /// Bytes of this layer's recurrent state: the conv tail in `dtype` and the f32 delta state.
    fn state_bytes(&self, dtype: Dtype) -> u64 {
        let c = &self.cfg;
        let conv = (c.kernel - 1) as u64 * c.conv_dim() as u64 * dtype.size();
        let delta = c.v_heads as u64 * c.v_dim as u64 * c.k_dim as u64 * 4;
        conv + delta
    }

    /// Activation bytes per token, f32 at worst: the projection, conv input and output, the
    /// normalized q/k/v, `z`, the output and its gated norm.
    fn scratch_per_token(&self) -> u64 {
        let c = &self.cfg;
        (4 * c.conv_dim() as u64 + 4 * c.value_dim() as u64) * 4
    }
}

#[derive(Debug)]
enum Mixer {
    Linear(GatedDeltaNet),
    Attention(Attention),
}

#[derive(Debug)]
struct Layer {
    input_layernorm: RmsNorm,
    mixer: Mixer,
    post_attention_layernorm: RmsNorm,
    ffn: Ffn,
}

/// The feed-forward block of every layer.
#[derive(Clone, Copy, Debug)]
enum FfnConfig {
    Moe(MoeConfig),
    /// Dense MLP of this intermediate width.
    Dense(i32),
}

#[derive(Debug)]
pub struct Qwen3_5Moe {
    head: Option<Head>,
    /// The first layer of this part (026).
    first: usize,
    layers: Vec<Layer>,
    kinds: Vec<LayerKind>,
    dtype: Dtype,
    hidden: u64,
}

/// An optional field: `default` when absent (meaning the feature is off, or the registry's own
/// default).
fn int_or(config: &Value, key: &str, default: i32) -> Result<i32, LoadError> {
    match config.get(key) {
        None | Some(Value::Null) => Ok(default),
        Some(_) => int(config, key),
    }
}

/// mlx-lm `Model.sanitize` + `TextModel.sanitize` (+ the MoE variant's fused-expert split), so
/// names match `language_model.model.…` whether the checkpoint is a converted MLX one or a
/// Hugging Face one.
fn sanitize(weights: &mut Weights, layers: usize) -> Result<(), LoadError> {
    weights.rename(|name| {
        if name.starts_with("vision_tower") || name.starts_with("model.visual") {
            None
        } else if let Some(rest) = name.strip_prefix("model.language_model") {
            Some(format!("language_model.model{rest}"))
        } else if name.starts_with("language_model.") {
            Some(name.to_string())
        } else {
            Some(format!("language_model.{name}"))
        }
    });
    for l in 0..layers {
        let prefix = format!("language_model.model.layers.{l}.mlp");
        let Some(gate_up) = weights.take_opt(&format!("{prefix}.experts.gate_up_proj")) else {
            continue;
        };
        let halves = tensor::split(&gate_up, 2, -2)
            .map_err(|e| invalid("weights", format!("{prefix}.experts.gate_up_proj: {e}")))?;
        let [gate, up]: [Array; 2] = halves
            .try_into()
            .map_err(|_| invalid("weights", "gate_up_proj does not split in two"))?;
        weights.insert(format!("{prefix}.switch_mlp.gate_proj.weight"), gate);
        weights.insert(format!("{prefix}.switch_mlp.up_proj.weight"), up);
        let down = weights.take(&format!("{prefix}.experts.down_proj"))?;
        weights.insert(format!("{prefix}.switch_mlp.down_proj.weight"), down);
    }
    // Hugging Face checkpoints keep the conv kernel as [C, 1, K] and store the decoder norms as
    // offsets from 1 (`(1 + w) · x`); converted MLX checkpoints already hold [C, K, 1] and w.
    let is_conv = |name: &str, t: &Array| name.contains("conv1d.weight") && t.dim(-1) != 1;
    let unsanitized = weights.any(is_conv);
    weights.map(is_conv, |t| tensor::transpose(t, &[0, 2, 1]))?;
    if unsanitized {
        const NORMS: [&str; 5] = [
            ".input_layernorm.weight",
            ".post_attention_layernorm.weight",
            "model.norm.weight",
            ".q_norm.weight",
            ".k_norm.weight",
        ];
        weights.map(
            |name, t| t.shape().len() == 1 && NORMS.iter().any(|s| name.ends_with(s)),
            |t| tensor::add(t, &tensor::scalar(1.0, t.dtype())?),
        )?;
    }
    Ok(())
}

impl Qwen3_5Moe {
    /// `config` is the text config (`text_config`, or the root of an unwrapped config).
    pub fn load(
        root: &Value,
        config: &Value,
        mut weights: Weights,
        part: Part,
    ) -> Result<Self, LoadError> {
        // Per-path quantization lives at the root of a wrapped config.
        let quant = Quantization::from_config(root)?;
        let layers = int(config, "num_hidden_layers")? as usize;
        let hidden = int(config, "hidden_size")?;
        let heads = int(config, "num_attention_heads")?;
        let eps = config["rms_norm_eps"].as_f64().unwrap_or(1e-6) as f32;
        let head_dim = match config.get("head_dim") {
            None | Some(Value::Null) => hidden / heads.max(1),
            Some(_) => int(config, "head_dim")?,
        };
        // mlx-lm `TextModelArgs.__post_init__`: RoPE settings come from `rope_parameters`
        // (absent: its default, theta 1e5 and a quarter of each head rotated) unless that is
        // null or empty, then from the top-level fields.
        let rope = match config.get("rope_parameters") {
            None => Some(&Value::Null),
            Some(r) if r.as_object().is_some_and(|o| !o.is_empty()) => Some(r),
            Some(_) => None,
        };
        let field = |key: &str, default: f64| match rope {
            Some(r) => r[key].as_f64().unwrap_or(default),
            None => config[key].as_f64().unwrap_or(default),
        };
        let (partial, theta) = (
            field("partial_rotary_factor", 0.25),
            field("rope_theta", 1e5),
        );
        let scaling = match rope {
            Some(r) => r,
            None => &config["rope_scaling"],
        };
        let attn = AttentionConfig {
            hidden,
            heads,
            kv_heads: int(config, "num_key_value_heads")?,
            head_dim,
            rope_dims: (f64::from(head_dim) * partial) as i32,
            rope_theta: theta as f32,
            rope_scale: common::rope_scale(Some(scaling))?,
            eps,
            gated: true,
        };
        let delta = DeltaConfig {
            k_heads: int(config, "linear_num_key_heads")?,
            v_heads: int(config, "linear_num_value_heads")?,
            k_dim: int(config, "linear_key_head_dim")?,
            v_dim: int(config, "linear_value_head_dim")?,
            kernel: int(config, "linear_conv_kernel_dim")?,
            eps,
        };
        if delta.k_heads < 1 || delta.v_heads % delta.k_heads != 0 || delta.kernel < 2 {
            return Err(invalid(
                "linear_num_value_heads",
                "must be a multiple of linear_num_key_heads (and the conv kernel at least 2)",
            ));
        }
        // No experts (`qwen3_5`): a dense MLP of `intermediate_size`.
        let experts = int_or(config, "num_experts", 0)?;
        let ffn = if experts > 0 {
            let moe = MoeConfig {
                hidden,
                experts,
                top_k: int(config, "num_experts_per_tok")?,
                intermediate: int(config, "moe_intermediate_size")?,
                norm_topk_prob: config["norm_topk_prob"].as_bool().unwrap_or(true),
                shared_intermediate: Some(int(config, "shared_expert_intermediate_size")?),
            };
            if moe.top_k < 1 || moe.top_k > experts {
                return Err(invalid(
                    "num_experts_per_tok",
                    format!("must be in 1..={experts}"),
                ));
            }
            FfnConfig::Moe(moe)
        } else {
            FfnConfig::Dense(int(config, "intermediate_size")?)
        };
        // The registry's default when `layer_types` is absent; it checks the two agree.
        let interval = int_or(config, "full_attention_interval", 4)?.max(1) as usize;

        sanitize(&mut weights, layers)?;
        let tied = config["tie_word_embeddings"].as_bool().unwrap_or(false);
        let head = part
            .head
            .then(|| {
                Head::load(
                    &mut weights,
                    "language_model.model",
                    "language_model.lm_head",
                    &quant,
                    hidden,
                    eps,
                    tied,
                )
            })
            .transpose()?;
        let mut stack = Vec::with_capacity(layers);
        let kinds: Vec<LayerKind> = (0..layers)
            .map(|i| {
                if (i + 1) % interval != 0 {
                    LayerKind::Linear
                } else {
                    LayerKind::FullAttention
                }
            })
            .collect();
        for i in (0..layers).filter(|&i| part.contains(i)) {
            let p = format!("language_model.model.layers.{i}");
            let norm = |w: &mut Weights, name: &str| RmsNorm::load(w, &format!("{p}.{name}"), eps);
            let mixer = if kinds[i] == LayerKind::Linear {
                Mixer::Linear(GatedDeltaNet::load(
                    &mut weights,
                    &format!("{p}.linear_attn"),
                    &quant,
                    hidden,
                    delta,
                )?)
            } else {
                Mixer::Attention(Attention::load(
                    &mut weights,
                    &format!("{p}.self_attn"),
                    &quant,
                    attn,
                )?)
            };
            let ffn = match ffn {
                FfnConfig::Moe(moe) => Ffn::Moe(SparseMoe::load(
                    &mut weights,
                    &format!("{p}.mlp"),
                    &quant,
                    moe,
                )?),
                FfnConfig::Dense(width) => Ffn::Dense(Mlp::load(
                    &mut weights,
                    &format!("{p}.mlp"),
                    &quant,
                    hidden,
                    width,
                )?),
            };
            stack.push(Layer {
                input_layernorm: norm(&mut weights, "input_layernorm")?,
                mixer,
                post_attention_layernorm: norm(&mut weights, "post_attention_layernorm")?,
                ffn,
            });
        }
        common::ensure_consumed(&weights)?;
        Ok(Self {
            dtype: common::part_dtype(head.as_ref(), stack.first().map(|l| &l.input_layernorm))?,
            head,
            first: part.first,
            layers: stack,
            kinds,
            hidden: hidden as u64,
        })
    }
}

impl Qwen3_5Moe {
    /// A decoder layer with its linear mixer split at `splits`, the convolution per row when
    /// `rows` ([`Model::layer_split`], [`Model::layer_verify`]).
    #[allow(clippy::too_many_arguments)]
    fn layer_with(
        &self,
        i: usize,
        x: &Array,
        state: &mut LayerState,
        offset: i32,
        splits: &[i32],
        captured: &mut Vec<Recurrent>,
        rows: bool,
    ) -> tensor::Result<Array> {
        let layer = &self.layers[i - self.first];
        let h = layer.input_layernorm.forward(x)?;
        let r = match &layer.mixer {
            Mixer::Linear(gdn) => {
                gdn.forward_split(&h, state.recurrent()?, splits, captured, rows)?
            }
            Mixer::Attention(attn) => attn.forward(&h, state.kv()?, offset)?,
        };
        let x = tensor::add(x, &r)?;
        let h = layer.post_attention_layernorm.forward(&x)?;
        tensor::add(&x, &layer.ffn.forward(&h)?)
    }
}

impl Model for Qwen3_5Moe {
    fn layer_kinds(&self) -> &[LayerKind] {
        &self.kinds
    }

    fn state_dtype(&self) -> Dtype {
        self.dtype
    }

    fn recurrent_bytes(&self) -> u64 {
        self.layers
            .iter()
            .map(|l| match &l.mixer {
                Mixer::Linear(gdn) => gdn.state_bytes(self.dtype),
                Mixer::Attention(_) => 0,
            })
            .sum()
    }

    fn embed(&self, ids: &Array) -> tensor::Result<Array> {
        common::head(&self.head)?.embed.forward(ids)
    }

    fn layer(
        &self,
        i: usize,
        x: &Array,
        state: &mut LayerState,
        offset: i32,
    ) -> tensor::Result<Array> {
        let layer = &self.layers[i - self.first];
        let h = layer.input_layernorm.forward(x)?;
        let r = match &layer.mixer {
            Mixer::Linear(gdn) => gdn.forward(&h, state.recurrent()?)?,
            Mixer::Attention(attn) => attn.forward(&h, state.kv()?, offset)?,
        };
        let x = tensor::add(x, &r)?;
        let h = layer.post_attention_layernorm.forward(&x)?;
        tensor::add(&x, &layer.ffn.forward(&h)?)
    }

    fn logits(&self, x: &Array) -> tensor::Result<Array> {
        common::head(&self.head)?.logits(x)
    }

    fn logits_rows(&self, x: &Array) -> tensor::Result<Array> {
        common::head(&self.head)?.rows(x)
    }

    fn hidden(&self) -> i32 {
        self.hidden as i32
    }

    fn layer_split(
        &self,
        i: usize,
        x: &Array,
        state: &mut LayerState,
        offset: i32,
        splits: &[i32],
        captured: &mut Vec<Recurrent>,
    ) -> tensor::Result<Array> {
        self.layer_with(i, x, state, offset, splits, captured, false)
    }

    fn layer_verify(
        &self,
        i: usize,
        x: &Array,
        state: &mut LayerState,
        offset: i32,
        captured: &mut Vec<Recurrent>,
    ) -> tensor::Result<Array> {
        let splits: Vec<i32> = (1..x.dim(1)).collect();
        self.layer_with(i, x, state, offset, &splits, captured, true)
    }

    fn chunk_scratch(&self, len: u32, depth: u32) -> u64 {
        let mut mixer = 0;
        let mut ffn = 0;
        for layer in &self.layers {
            mixer = mixer.max(match &layer.mixer {
                Mixer::Linear(gdn) => u64::from(len) * gdn.scratch_per_token(),
                Mixer::Attention(attn) => attn.chunk_scratch(len, depth),
            });
            ffn = ffn.max(layer.ffn.scratch_per_token());
        }
        u64::from(len) * (6 * self.hidden * 4 + ffn) + mixer
    }
}
