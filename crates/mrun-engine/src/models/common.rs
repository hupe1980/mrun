//! Blocks the Qwen families share: config access, GQA attention with per-head q/k RMSNorm and
//! RoPE (optionally partial and output-gated, as in Qwen3-Next/Qwen3.5), the SwiGLU MLP, and the
//! embedding plus output head.

use serde_json::Value;

use super::activations;
use super::{LoadError, Pos, Run};
use crate::load::{Embedding, Linear, Quantization, RmsNorm, Weights, invalid};
use crate::state::LayerKv;
use crate::tensor::{self, Array, Dtype, fast};

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
    /// Interleaved M-RoPE sections (t, h, w) of image rows (009); `None` for text-only families.
    pub mrope: Option<[i32; 3]>,
}

/// The axis (0 = t, 1 = h, 2 = w) each rotary frequency of an image row takes its position
/// from: Qwen3-VL's interleaved layout, `[t, h, w, t, h, w, …]` within the h and w sections.
pub fn mrope_axes(sections: [i32; 3], freqs: usize) -> Vec<usize> {
    (0..freqs)
        .map(|j| {
            let j32 = j as i32;
            if j % 3 == 1 && j32 < sections[1] * 3 {
                1
            } else if j % 3 == 2 && j32 < sections[2] * 3 {
                2
            } else {
                0
            }
        })
        .collect()
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

    /// Attention over `x` `[1, L, hidden]` at `pos`, writing K/V to `kv`.
    pub fn forward(&self, x: &Array, kv: &mut LayerKv, pos: Pos<'_>) -> tensor::Result<Array> {
        let projected = self.project(x)?;
        let o = self.core(&projected, kv, pos)?;
        self.o_proj.forward(&o)
    }

    /// One decode step of several lanes (011): row `b` of `x` `[1, B, hidden]` attends over
    /// `kvs[b]` at `positions[b]`. The projections run on every row at once, the attention per
    /// lane.
    pub fn forward_lanes(
        &self,
        x: &Array,
        kvs: &mut [&mut LayerKv],
        positions: &[Pos<'_>],
    ) -> tensor::Result<Array> {
        let [q, k, v] = self.project(x)?;
        let (q, k, v) = (
            super::per_row(&q, |_, r| Ok(r.clone()))?,
            super::per_row(&k, |_, r| Ok(r.clone()))?,
            super::per_row(&v, |_, r| Ok(r.clone()))?,
        );
        let mut rows = Vec::with_capacity(kvs.len());
        for (b, kv) in kvs.iter_mut().enumerate() {
            let row = [q[b].clone(), k[b].clone(), v[b].clone()];
            rows.push(self.core(&row, kv, positions[b])?);
        }
        self.o_proj
            .forward(&tensor::concatenate(&rows.iter().collect::<Vec<_>>(), 1)?)
    }

    /// The q, k and v projections of `x` `[1, L, hidden]` (q with its gate when gated).
    fn project(&self, x: &Array) -> tensor::Result<[Array; 3]> {
        Ok([
            self.q_proj.forward(x)?,
            self.k_proj.forward(x)?,
            self.v_proj.forward(x)?,
        ])
    }

    /// Attention of projected `[q, k, v]` `[1, L, ·]` at `offset`: norms, RoPE, the K/V
    /// update, the attention and the output gate; `[1, L, heads · head_dim]` before `o_proj`.
    fn core(
        &self,
        [q, k, v]: &[Array; 3],
        kv: &mut LayerKv,
        pos: Pos<'_>,
    ) -> tensor::Result<Array> {
        let c = &self.cfg;
        let len = q.dim(1);
        let heads = |y: &Array, n: i32| tensor::reshape(y, &[1, len, n, -1]);
        let to_bhld = |y: &Array| tensor::transpose(y, &[0, 2, 1, 3]);
        let (queries, gate) = if c.gated {
            // [1, L, H, 2·D] split into the queries and the output gate, per head.
            let mut parts = tensor::split(&heads(q, c.heads)?, 2, -1)?;
            let gate = parts.pop().expect("two parts");
            let queries = parts.pop().expect("two parts");
            (queries, Some(tensor::reshape(&gate, &[1, len, -1])?))
        } else {
            (heads(q, c.heads)?, None)
        };
        let q = to_bhld(&self.q_norm.forward(&queries)?)?;
        let k = to_bhld(&self.k_norm.forward(&heads(k, c.kv_heads)?)?)?;
        let v = to_bhld(&heads(v, c.kv_heads)?)?;
        let (q, k) = (self.rope(&q, pos)?, self.rope(&k, pos)?);
        let (keys, values) = kv.update(&k, &v, pos.at)?;
        let mask = if len > 1 {
            fast::Mask::Causal
        } else {
            fast::Mask::None
        };
        let limit = SCORE_LIMIT.with(std::cell::Cell::get);
        let o = fast::scaled_dot_product_attention(&q, &keys, &values, self.scale, mask, limit)?;
        let o = tensor::reshape(&to_bhld(&o)?, &[1, len, -1])?;
        match gate {
            Some(gate) => tensor::multiply(&o, &tensor::sigmoid(&gate)?),
            None => Ok(o),
        }
    }

    /// RoPE of `y` `[1, H, L, D]` at `pos`: text rows with the fused kernel (bit for bit the
    /// text-only path, since it is elementwise per position), image rows with M-RoPE.
    fn rope(&self, y: &Array, pos: Pos<'_>) -> tensor::Result<Array> {
        let c = &self.cfg;
        let text = |y: &Array, offset: i32| {
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
        if pos.runs.is_empty() {
            return text(y, pos.at + pos.delta);
        }
        let shape = y.shape();
        let mut parts = Vec::with_capacity(pos.runs.len());
        let mut row = 0;
        for run in pos.runs {
            let n = run.rows();
            let piece =
                tensor::slice(y, &[0, 0, row, 0], &[shape[0], shape[1], row + n, shape[3]])?;
            parts.push(match *run {
                Run::Text { offset, .. } => text(&piece, offset)?,
                Run::Image {
                    first, width, base, ..
                } => self.mrope(&piece, first, width, base)?,
            });
            row += n;
        }
        tensor::concatenate(&parts.iter().collect::<Vec<_>>(), 2)
    }

    /// Interleaved M-RoPE of image rows `y` `[1, H, n, D]` (009 FR-005): the image's token `k`
    /// (`first + row`) sits at (t, h, w) = (base, base + k / width, base + k % width); each
    /// frequency takes its axis from [`mrope_axes`]; halves rotated as `fast::rope` does.
    fn mrope(&self, y: &Array, first: i32, width: i32, base: i32) -> tensor::Result<Array> {
        let c = &self.cfg;
        let sections = c.mrope.unwrap_or([c.rope_dims / 2, 0, 0]);
        let half = (c.rope_dims / 2) as usize;
        let axes = mrope_axes(sections, half);
        let n = y.dim(2);
        let mut cos = Vec::with_capacity(n as usize * half);
        let mut sin = Vec::with_capacity(n as usize * half);
        for row in 0..n {
            let k = first + row;
            let at = [base, base + k / width.max(1), base + k % width.max(1)];
            for (j, &axis) in axes.iter().enumerate() {
                let inv = (c.rope_theta as f64).powf(-2.0 * j as f64 / c.rope_dims as f64);
                let angle = (f64::from(at[axis]) * f64::from(c.rope_scale) * inv) as f32;
                cos.push(angle.cos());
                sin.push(angle.sin());
            }
        }
        let shape = [1, 1, n, half as i32];
        let (cos, sin) = (Array::from_f32(&cos, &shape), Array::from_f32(&sin, &shape));
        let dtype = y.dtype();
        let d = y.dim(3);
        let h = y.dim(1);
        let part = |from: i32, to: i32| tensor::slice(y, &[0, 0, 0, from], &[1, h, n, to]);
        let x1 = tensor::astype(&part(0, half as i32)?, Dtype::Float32)?;
        let x2 = tensor::astype(&part(half as i32, c.rope_dims)?, Dtype::Float32)?;
        let o1 = tensor::subtract(&tensor::multiply(&x1, &cos)?, &tensor::multiply(&x2, &sin)?)?;
        let o2 = tensor::add(&tensor::multiply(&x2, &cos)?, &tensor::multiply(&x1, &sin)?)?;
        let mut out = vec![tensor::astype(&o1, dtype)?, tensor::astype(&o2, dtype)?];
        if c.rope_dims < d {
            out.push(part(c.rope_dims, d)?);
        }
        tensor::concatenate(&out.iter().collect::<Vec<_>>(), 3)
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

    /// Hidden states `[1, L, hidden]` → logits of every position `[L, vocab]` (a verify's rows).
    pub fn rows(&self, x: &Array) -> tensor::Result<Array> {
        let h = self.norm.forward(x)?;
        let logits = match &self.lm_head {
            Some(head) => head.forward(&h)?,
            None => self.embed.as_linear(&h)?,
        };
        let vocab = logits.dim(-1);
        tensor::reshape(&logits, &[x.dim(1), vocab])
    }
}

/// Fails on tensors no layer took (after the family dropped what it ignores).
/// The head of a part that has it; a stage of a pool without it cannot embed or give logits.
pub fn head(head: &Option<Head>) -> tensor::Result<&Head> {
    head.as_ref()
        .ok_or_else(|| tensor::Error("this part of the model has no embeddings or head".into()))
}

/// The dtype of a part's weights: the head's norm, else its first layer's (a part without either
/// loads nothing and is refused).
pub fn part_dtype(head: Option<&Head>, first: Option<&RmsNorm>) -> Result<Dtype, LoadError> {
    head.map(|h| &h.norm)
        .or(first)
        .map(|norm| norm.weight.dtype())
        .ok_or_else(|| invalid("layers", "the part holds no layers"))
}

pub fn ensure_consumed(weights: &Weights) -> Result<(), LoadError> {
    match weights.remaining().first() {
        Some(name) => Err(invalid("weights", format!("unexpected tensor {name}"))),
        None => Ok(()),
    }
}
