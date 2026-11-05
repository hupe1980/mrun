//! Blocks the Qwen families share: config access, GQA attention with per-head q/k RMSNorm and
//! RoPE (optionally partial and output-gated, as in Qwen3-Next/Qwen3.5), the SwiGLU MLP, and the
//! embedding plus output head.

use serde_json::Value;

use super::LoadError;
use super::activations;
use crate::load::{Embedding, Linear, Quantization, RmsNorm, Weights, invalid};
use crate::state::LayerKv;
use crate::tensor::{self, Array, fast};

/// A required number from a config object.
pub fn num(config: &Value, key: &str) -> Result<f64, LoadError> {
    config[key]
        .as_f64()
        .ok_or_else(|| invalid(key, "is missing or not a number"))
}

/// A required integer from a config object.
pub fn int(config: &Value, key: &str) -> Result<i32, LoadError> {
    config[key]
        .as_i64()
        .and_then(|n| i32::try_from(n).ok())
        .ok_or_else(|| invalid(key, "is missing or not an integer"))
}

/// RoPE scale of a `rope_scaling`/`rope_parameters` object as mlx-lm `initialize_rope` reads it:
/// `default` → 1, `linear` → 1 / factor; other types are not supported by these families.
pub fn rope_scale(scaling: Option<&Value>) -> Result<f32, LoadError> {
    let Some(scaling) = scaling.filter(|s| s.is_object()) else {
        return Ok(1.0);
    };
    let kind = scaling["type"]
        .as_str()
        .or_else(|| scaling["rope_type"].as_str())
        .unwrap_or("default");
    match kind {
        "default" => Ok(1.0),
        "linear" => Ok(1.0 / num(scaling, "factor")? as f32),
        other => Err(invalid(
            "rope_scaling",
            format!("type {other} is not supported"),
        )),
    }
}

/// Attention shape and RoPE settings.
#[derive(Clone, Copy, Debug)]
pub struct AttentionConfig {
    pub hidden: i32,
    pub heads: i32,
    pub kv_heads: i32,
    pub head_dim: i32,
    /// Leading features of each head that RoPE rotates (`head_dim` unless partial).
    pub rope_dims: i32,
    pub rope_theta: f32,
    pub rope_scale: f32,
    pub eps: f32,
    /// `q_proj` also emits a per-feature output gate (Qwen3-Next `Qwen3NextAttention`).
    pub gated: bool,
}

#[derive(Debug)]
pub struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    cfg: AttentionConfig,
    scale: f32,
}

/// The score buffer of a chunk's attention may hold at most this many bytes (D059, FR-017):
/// nuvec takes its tiled kernel instead of materialising larger scores. At 16 heads, 2,048
/// queries and 8,192 keys in bf16 it is exactly reached.
pub const SCORE_BLOCK_BYTES: u64 = 512 << 20;
#[doc(hidden)]
pub const SCORE_BLOCK_BYTES_DEFAULT: u64 = SCORE_BLOCK_BYTES;

thread_local! {
    static SCORE_LIMIT: std::cell::Cell<u64> = const { std::cell::Cell::new(SCORE_BLOCK_BYTES) };
}

/// Overrides [`SCORE_BLOCK_BYTES`] on this thread (tests force small blocks).
#[doc(hidden)]
pub fn set_score_block_bytes(bytes: u64) {
    SCORE_LIMIT.with(|limit| limit.set(bytes));
}

impl Attention {
    pub fn load(
        weights: &mut Weights,
        prefix: &str,
        quant: &Quantization,
        cfg: AttentionConfig,
    ) -> Result<Self, LoadError> {
        let p = |name: &str| format!("{prefix}.{name}");
        let (hidden, inner) = (cfg.hidden, cfg.heads * cfg.head_dim);
        Ok(Self {
            q_proj: Linear::load(weights, &p("q_proj"), quant, hidden)?,
            k_proj: Linear::load(weights, &p("k_proj"), quant, hidden)?,
            v_proj: Linear::load(weights, &p("v_proj"), quant, hidden)?,
            o_proj: Linear::load(weights, &p("o_proj"), quant, inner)?,
            q_norm: RmsNorm::load(weights, &p("q_norm"), cfg.eps)?,
            k_norm: RmsNorm::load(weights, &p("k_norm"), cfg.eps)?,
            cfg,
            scale: (cfg.head_dim as f32).powf(-0.5),
        })
    }

    /// Attention over `x` `[1, L, hidden]` at absolute position `offset`, writing K/V to `kv`.
    pub fn forward(&self, x: &Array, kv: &mut LayerKv, offset: i32) -> tensor::Result<Array> {
        let c = &self.cfg;
        let len = x.dim(1);
        let heads = |y: &Array, n: i32| tensor::reshape(y, &[1, len, n, -1]);
        let to_bhld = |y: &Array| tensor::transpose(y, &[0, 2, 1, 3]);
        let (queries, gate) = if c.gated {
            // [1, L, H, 2·D] split into the queries and the output gate, per head.
            let mut parts = tensor::split(&heads(&self.q_proj.forward(x)?, c.heads)?, 2, -1)?;
            let gate = parts.pop().expect("two parts");
            let queries = parts.pop().expect("two parts");
            (queries, Some(tensor::reshape(&gate, &[1, len, -1])?))
        } else {
            (heads(&self.q_proj.forward(x)?, c.heads)?, None)
        };
        let q = to_bhld(&self.q_norm.forward(&queries)?)?;
        let k = to_bhld(
            &self
                .k_norm
                .forward(&heads(&self.k_proj.forward(x)?, c.kv_heads)?)?,
        )?;
        let v = to_bhld(&heads(&self.v_proj.forward(x)?, c.kv_heads)?)?;
        let rope = |y: &Array| {
            fast::rope(
                y,
                c.rope_dims,
                false,
                Some(c.rope_theta),
                c.rope_scale,
                offset,
                None,
            )
        };
        let (q, k) = (rope(&q)?, rope(&k)?);
        let (keys, values) = kv.update(&k, &v, offset)?;
        let mask = if len > 1 {
            fast::Mask::Causal
        } else {
            fast::Mask::None
        };
        let limit = SCORE_LIMIT.with(std::cell::Cell::get);
        let o = fast::scaled_dot_product_attention(&q, &keys, &values, self.scale, mask, limit)?;
        let o = tensor::reshape(&to_bhld(&o)?, &[1, len, -1])?;
        let o = match gate {
            Some(gate) => tensor::multiply(&o, &tensor::sigmoid(&gate)?)?,
            None => o,
        };
        self.o_proj.forward(&o)
    }

    /// Activation bytes of one chunk of `len` tokens at `depth` (f32 at worst): projections, and
    /// the attention's scratch as nuvec plans it for the kernel it will run (D059, FR-017).
    pub fn chunk_scratch(&self, len: u32, depth: u32) -> u64 {
        let c = &self.cfg;
        let q_width = c.heads * c.head_dim * if c.gated { 2 } else { 1 };
        let qkv = (q_width + 2 * c.kv_heads * c.head_dim + c.heads * c.head_dim) as u64;
        let keys = depth + len;
        let planned = fast::attention_scratch_bytes(
            [1, c.heads, len as i32, c.head_dim],
            [1, c.kv_heads, keys as i32, c.head_dim],
            tensor::Dtype::Bfloat16,
            if len > 1 {
                fast::Mask::Causal
            } else {
                fast::Mask::None
            },
            SCORE_LIMIT.with(std::cell::Cell::get),
        );
        let scores = planned.unwrap_or(0);
        u64::from(len) * 2 * qkv * 4 + scores
    }
}

/// SwiGLU MLP: `down(swiglu(gate(x), up(x)))`.
#[derive(Debug)]
pub struct Mlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    intermediate: i32,
}

impl Mlp {
    pub fn load(
        weights: &mut Weights,
        prefix: &str,
        quant: &Quantization,
        hidden: i32,
        intermediate: i32,
    ) -> Result<Self, LoadError> {
        let p = |name: &str| format!("{prefix}.{name}");
        Ok(Self {
            gate_proj: Linear::load(weights, &p("gate_proj"), quant, hidden)?,
            up_proj: Linear::load(weights, &p("up_proj"), quant, hidden)?,
            down_proj: Linear::load(weights, &p("down_proj"), quant, intermediate)?,
            intermediate,
        })
    }

    pub fn forward(&self, x: &Array) -> tensor::Result<Array> {
        let act = activations::swiglu(&self.gate_proj.forward(x)?, &self.up_proj.forward(x)?)?;
        self.down_proj.forward(&act)
    }

    /// Activation bytes per token (gate, up and their product, f32 at worst).
    pub fn scratch_per_token(&self) -> u64 {
        3 * self.intermediate as u64 * 4
    }
}

/// Token embedding, final norm and output head (tied to the embedding when absent).
#[derive(Debug)]
pub struct Head {
    pub embed: Embedding,
    pub norm: RmsNorm,
    pub lm_head: Option<Linear>,
}

impl Head {
    /// Loads `<prefix>.embed_tokens`, `<prefix>.norm` and `<head>` (unless `tied`, in which case
    /// a stored head is dropped as mlx-lm's `sanitize` does).
    pub fn load(
        weights: &mut Weights,
        prefix: &str,
        head: &str,
        quant: &Quantization,
        hidden: i32,
        eps: f32,
        tied: bool,
    ) -> Result<Self, LoadError> {
        if tied {
            weights.retain(|name| !name.starts_with(&format!("{head}.")));
        }
        Ok(Self {
            embed: Embedding::load(weights, &format!("{prefix}.embed_tokens"), quant, hidden)?,
            norm: RmsNorm::load(weights, &format!("{prefix}.norm"), eps)?,
            lm_head: if tied {
                None
            } else {
                Some(Linear::load(weights, head, quant, hidden)?)
            },
        })
    }

    /// Hidden states `[1, L, hidden]` → logits of the last position `[1, vocab]`.
    pub fn logits(&self, x: &Array) -> tensor::Result<Array> {
        let (len, hidden) = (x.dim(1), x.dim(2));
        let last = tensor::slice(x, &[0, len - 1, 0], &[1, len, hidden])?;
        let h = self.norm.forward(&last)?;
        let logits = match &self.lm_head {
            Some(head) => head.forward(&h)?,
            None => self.embed.as_linear(&h)?,
        };
        let vocab = logits.dim(-1);
        tensor::reshape(&logits, &[1, vocab])
    }
}

/// Fails on tensors no layer took (after the family dropped what it ignores).
pub fn ensure_consumed(weights: &Weights) -> Result<(), LoadError> {
    match weights.remaining().first() {
        Some(name) => Err(invalid("weights", format!("unexpected tensor {name}"))),
        None => Ok(()),
    }
}
