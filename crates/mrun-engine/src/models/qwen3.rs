//! `qwen3` (Qwen3 dense, e.g. Qwen3-8B): a port of mlx-lm `mlx_lm/models/qwen3.py`.
//!
//! Pre-norm decoder: RMSNorm → GQA attention with per-head q/k RMSNorm and RoPE → residual →
//! RMSNorm → SwiGLU MLP → residual; final RMSNorm and an untied or tied output head.

use serde_json::Value;

use super::common::{self, Attention, AttentionConfig, Head, Mlp, int, num};
use super::{LoadError, Model};
use crate::engine::LayerKind;
use crate::load::{Quantization, RmsNorm, Weights, invalid};
use crate::state::LayerState;
use crate::tensor::{self, Array, Dtype};

#[derive(Debug)]
struct Layer {
    input_layernorm: RmsNorm,
    attention: Attention,
    post_attention_layernorm: RmsNorm,
    mlp: Mlp,
}

#[derive(Debug)]
pub struct Qwen3 {
    head: Head,
    layers: Vec<Layer>,
    kinds: Vec<LayerKind>,
    dtype: Dtype,
    hidden: u64,
}

/// Attention settings of the Qwen3 dense and MoE configs: `rope_parameters` (transformers ≥ 5)
/// or top-level `rope_theta`/`rope_scaling`. mlx-lm requires a theta and ignores any scaling
/// (plain `nn.RoPE`), so a scaling other than `default` is refused rather than silently dropped.
pub fn attention_config(config: &Value) -> Result<AttentionConfig, LoadError> {
    let heads = int(config, "num_attention_heads")?;
    let hidden = int(config, "hidden_size")?;
    let head_dim = config["head_dim"]
        .as_i64()
        .map_or(hidden / heads.max(1), |d| d as i32);
    let rope = config.get("rope_parameters").filter(|r| r.is_object());
    let rope_theta = rope
        .and_then(|r| r["rope_theta"].as_f64())
        .or_else(|| config["rope_theta"].as_f64())
        .ok_or_else(|| invalid("rope_theta", "is missing"))? as f32;
    if common::rope_scale(rope.or(config.get("rope_scaling")))? != 1.0 {
        return Err(invalid(
            "rope_scaling",
            "only the default RoPE is supported",
        ));
    }
    Ok(AttentionConfig {
        hidden,
        heads,
        kv_heads: int(config, "num_key_value_heads")?,
        head_dim,
        rope_dims: head_dim,
        rope_theta,
        rope_scale: 1.0,
        eps: num(config, "rms_norm_eps")? as f32,
        gated: false,
    })
}

impl Qwen3 {
    pub fn load(config: &Value, mut weights: Weights) -> Result<Self, LoadError> {
        let quant = Quantization::from_config(config)?;
        let layers = int(config, "num_hidden_layers")? as usize;
        let attn = attention_config(config)?;
        let intermediate = int(config, "intermediate_size")?;
        let tied = config["tie_word_embeddings"].as_bool().unwrap_or(false);
        let head = Head::load(
            &mut weights,
            "model",
            "lm_head",
            &quant,
            attn.hidden,
            attn.eps,
            tied,
        )?;
        let mut stack = Vec::with_capacity(layers);
        for i in 0..layers {
            let p = format!("model.layers.{i}");
            let norm =
                |w: &mut Weights, name: &str| RmsNorm::load(w, &format!("{p}.{name}"), attn.eps);
            stack.push(Layer {
                input_layernorm: norm(&mut weights, "input_layernorm")?,
                attention: Attention::load(&mut weights, &format!("{p}.self_attn"), &quant, attn)?,
                post_attention_layernorm: norm(&mut weights, "post_attention_layernorm")?,
                mlp: Mlp::load(
                    &mut weights,
                    &format!("{p}.mlp"),
                    &quant,
                    attn.hidden,
                    intermediate,
                )?,
            });
        }
        weights.retain(|name| !name.contains("rotary_emb.inv_freq"));
        common::ensure_consumed(&weights)?;
        Ok(Self {
            dtype: head.norm.weight.dtype(),
            head,
            layers: stack,
            kinds: vec![LayerKind::FullAttention; layers],
            hidden: attn.hidden as u64,
        })
    }
}

impl Model for Qwen3 {
    fn layer_kinds(&self) -> &[LayerKind] {
        &self.kinds
    }

    fn state_dtype(&self) -> Dtype {
        self.dtype
    }

    fn embed(&self, ids: &Array) -> tensor::Result<Array> {
        self.head.embed.forward(ids)
    }

    fn layer(
        &self,
        i: usize,
        x: &Array,
        state: &mut LayerState,
        offset: i32,
    ) -> tensor::Result<Array> {
        let layer = &self.layers[i];
        let h = layer.input_layernorm.forward(x)?;
        let x = tensor::add(x, &layer.attention.forward(&h, state.kv()?, offset)?)?;
        let h = layer.post_attention_layernorm.forward(&x)?;
        tensor::add(&x, &layer.mlp.forward(&h)?)
    }

    fn logits(&self, x: &Array) -> tensor::Result<Array> {
        self.head.logits(x)
    }

    fn chunk_scratch(&self, len: u32, depth: u32) -> u64 {
        // One layer is live at a time: the residual stream and norms (6 × hidden, f32 at
        // worst), attention and the MLP.
        let layer = &self.layers[0];
        u64::from(len) * (6 * self.hidden * 4 + layer.mlp.scratch_per_token())
            + layer.attention.chunk_scratch(len, depth)
    }
}
