//! `qwen3_moe` (Qwen3-MoE, e.g. Qwen3-Coder-30B-A3B): a port of mlx-lm 0.32.0
//! `mlx_lm/models/qwen3_moe.py`.
//!
//! The `qwen3` decoder with a sparse MoE block in place of the MLP on every
//! `decoder_sparse_step`-th layer not listed in `mlp_only_layers`. mlx-community conversions
//! quantize the router (`mlp.gate`) to 8 bits; the per-layer settings come from the config or
//! the packed shapes (`Quantization::resolve`).

use serde_json::Value;

use super::common::{self, Attention, Head, Mlp, int};
use super::moe::{Ffn, MoeConfig, SparseMoe};
use super::qwen3::attention_config;
use super::{LoadError, Model, Part};
use crate::engine::LayerKind;
use crate::load::{Quantization, RmsNorm, Weights, invalid};
use crate::state::LayerState;
use crate::tensor::{self, Array, Dtype};

#[derive(Debug)]
struct Layer {
    input_layernorm: RmsNorm,
    attention: Attention,
    post_attention_layernorm: RmsNorm,
    ffn: Ffn,
}

#[derive(Debug)]
pub struct Qwen3Moe {
    head: Option<Head>,
    /// The first layer of this part (026).
    first: usize,
    layers: Vec<Layer>,
    kinds: Vec<LayerKind>,
    dtype: Dtype,
    hidden: u64,
}

/// mlx-lm `sanitize`: unquantized Hugging Face checkpoints store each expert separately; stack
/// them into `switch_mlp.{gate,up,down}_proj` `[E, out, in]`.
fn stack_experts(weights: &mut Weights, layers: usize, experts: i32) -> Result<(), LoadError> {
    for l in 0..layers {
        let prefix = format!("model.layers.{l}.mlp");
        if !weights.contains(&format!("{prefix}.experts.0.up_proj.weight")) {
            continue;
        }
        for name in ["up_proj", "down_proj", "gate_proj"] {
            let parts = (0..experts)
                .map(|e| weights.take(&format!("{prefix}.experts.{e}.{name}.weight")))
                .collect::<Result<Vec<_>, _>>()?;
            let refs: Vec<&Array> = parts.iter().collect();
            let stacked = tensor::stack(&refs, 0)
                .map_err(|e| invalid("weights", format!("{prefix}.experts: {e}")))?;
            weights.insert(format!("{prefix}.switch_mlp.{name}.weight"), stacked);
        }
    }
    Ok(())
}

impl Qwen3Moe {
    pub fn load(config: &Value, mut weights: Weights, part: Part) -> Result<Self, LoadError> {
        let quant = Quantization::from_config(config)?;
        let layers = int(config, "num_hidden_layers")? as usize;
        let attn = attention_config(config)?;
        let moe = MoeConfig {
            hidden: attn.hidden,
            experts: int(config, "num_experts")?,
            top_k: int(config, "num_experts_per_tok")?,
            intermediate: int(config, "moe_intermediate_size")?,
            norm_topk_prob: config["norm_topk_prob"]
                .as_bool()
                .ok_or_else(|| invalid("norm_topk_prob", "is missing"))?,
            shared_intermediate: None,
        };
        if moe.top_k < 1 || moe.top_k > moe.experts {
            return Err(invalid(
                "num_experts_per_tok",
                format!("must be in 1..={}", moe.experts),
            ));
        }
        let sparse_step = config["decoder_sparse_step"].as_i64().unwrap_or(1).max(1) as usize;
        let mlp_only: Vec<usize> = config["mlp_only_layers"]
            .as_array()
            .map(|l| {
                l.iter()
                    .filter_map(|v| v.as_u64())
                    .map(|v| v as usize)
                    .collect()
            })
            .unwrap_or_default();
        stack_experts(&mut weights, layers, moe.experts)?;
        let tied = config["tie_word_embeddings"].as_bool().unwrap_or(false);
        let head = part
            .head
            .then(|| {
                Head::load(
                    &mut weights,
                    "model",
                    "lm_head",
                    &quant,
                    attn.hidden,
                    attn.eps,
                    tied,
                )
            })
            .transpose()?;
        let mut stack = Vec::with_capacity(layers);
        for i in (0..layers).filter(|&i| part.contains(i)) {
            let p = format!("model.layers.{i}");
            let norm =
                |w: &mut Weights, name: &str| RmsNorm::load(w, &format!("{p}.{name}"), attn.eps);
            let sparse = !mlp_only.contains(&i) && moe.experts > 0 && (i + 1) % sparse_step == 0;
            let ffn = if sparse {
                Ffn::Moe(SparseMoe::load(
                    &mut weights,
                    &format!("{p}.mlp"),
                    &quant,
                    moe,
                )?)
            } else {
                Ffn::Dense(Mlp::load(
                    &mut weights,
                    &format!("{p}.mlp"),
                    &quant,
                    attn.hidden,
                    int(config, "intermediate_size")?,
                )?)
            };
            stack.push(Layer {
                input_layernorm: norm(&mut weights, "input_layernorm")?,
                attention: Attention::load(&mut weights, &format!("{p}.self_attn"), &quant, attn)?,
                post_attention_layernorm: norm(&mut weights, "post_attention_layernorm")?,
                ffn,
            });
        }
        weights.retain(|name| !name.contains("rotary_emb.inv_freq"));
        common::ensure_consumed(&weights)?;
        Ok(Self {
            dtype: common::part_dtype(head.as_ref(), stack.first().map(|l| &l.input_layernorm))?,
            head,
            first: part.first,
            layers: stack,
            kinds: vec![LayerKind::FullAttention; layers],
            hidden: attn.hidden as u64,
        })
    }
}

impl Model for Qwen3Moe {
    fn layer_kinds(&self) -> &[LayerKind] {
        &self.kinds
    }

    fn state_dtype(&self) -> Dtype {
        self.dtype
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
        let x = tensor::add(x, &layer.attention.forward(&h, state.kv()?, offset)?)?;
        let h = layer.post_attention_layernorm.forward(&x)?;
        tensor::add(&x, &layer.ffn.forward(&h)?)
    }

    fn logits(&self, x: &Array) -> tensor::Result<Array> {
        common::head(&self.head)?.logits(x)
    }

    fn hidden(&self) -> i32 {
        self.hidden as i32
    }

    fn chunk_scratch(&self, len: u32, depth: u32) -> u64 {
        let ffn = self
            .layers
            .iter()
            .map(|l| l.ffn.scratch_per_token())
            .max()
            .unwrap_or(0);
        let attention = self
            .layers
            .first()
            .map_or(0, |l| l.attention.chunk_scratch(len, depth));
        u64::from(len) * (6 * self.hidden * 4 + ffn) + attention
    }
}
