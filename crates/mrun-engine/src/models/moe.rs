//! Sparse mixture-of-experts block of the Qwen MoE families: a port of mlx-lm 0.32.0
//! `Qwen3MoeSparseMoeBlock` / `Qwen3NextSparseMoeBlock` and `switch_layers.SwitchGLU`.
//!
//! Router: `softmax(gate(x), precise)`, the top-k experts by `argpartition`, their scores
//! (renormalized when `norm_topk_prob`). Experts: one gathered matmul per projection over the
//! selected experts' weights (`gather_qmm`, or `gather_mm` for dense checkpoints); with 64 or
//! more (token, expert) pairs the pairs are sorted by expert first, as mlx-lm does, so each
//! expert's weights are read once. An optional shared expert is added under a sigmoid gate.

use super::LoadError;
use super::activations;
use super::common::Mlp;
use crate::load::{Linear, Quantization, Weights, invalid};
use crate::tensor::{self, Array};

/// mlx-lm `SwitchGLU` sorts the pairs at this many (token, expert) indices.
const SORT_AT: usize = 64;

/// The stacked weights of one projection across experts `[E, out, in]` (packed when quantized).
#[derive(Debug)]
enum SwitchLinear {
    Dense(Array),
    Quantized(tensor::Quantized),
}

impl SwitchLinear {
    fn load(
        weights: &mut Weights,
        path: &str,
        quant: &Quantization,
        in_dims: i32,
    ) -> Result<Self, LoadError> {
        match Linear::load(weights, path, quant, in_dims)? {
            Linear::Dense { weight, bias: None } => Ok(Self::Dense(weight)),
            Linear::Quantized { w, bias: None } => Ok(Self::Quantized(w)),
            _ => Err(invalid("weights", format!("{path}.bias is not supported"))),
        }
    }

    /// `x [..., 1, in]` through expert `indices[...]` → `[..., 1, out]`.
    fn forward(&self, x: &Array, indices: &Array, sorted: bool) -> tensor::Result<Array> {
        match self {
            Self::Dense(weight) => {
                let wt = tensor::transpose(weight, &[0, 2, 1])?;
                tensor::gather_mm(x, &wt, indices, sorted)
            }
            Self::Quantized(w) => tensor::gather_qmm(x, w, indices, sorted),
        }
    }
}

#[derive(Debug)]
struct SwitchGlu {
    gate_proj: SwitchLinear,
    up_proj: SwitchLinear,
    down_proj: SwitchLinear,
}

impl SwitchGlu {
    /// `x [1, L, D]`, `indices [1, L, k]` → `[1, L, k, D]`.
    fn forward(&self, x: &Array, indices: &Array) -> tensor::Result<Array> {
        let x = tensor::expand_dims(x, &[-2, -3])?;
        let shape = indices.shape();
        let sort = indices.size() >= SORT_AT;
        let (x, idx, inverse) = if sort {
            // mlx-lm `_gather_sort`: flatten the pairs, order them by expert, gather each pair's
            // token row (`order // k`), and remember the inverse order to scatter back.
            let k = shape[shape.len() - 1];
            let flat = tensor::reshape(indices, &[-1])?;
            let order = tensor::argsort(&flat, 0)?;
            let inverse = tensor::argsort(&order, 0)?;
            let rows = tensor::reshape(&x, &[-1, 1, x.dim(-1)])?;
            let token = tensor::floor_divide(&order, &tensor::scalar(k as f32, order.dtype())?)?;
            let x = tensor::take(&rows, &token, 0)?;
            (x, tensor::take(&flat, &order, 0)?, Some(inverse))
        } else {
            (x, indices.clone(), None)
        };
        let up = self.up_proj.forward(&x, &idx, sort)?;
        let gate = self.gate_proj.forward(&x, &idx, sort)?;
        let y = self
            .down_proj
            .forward(&activations::swiglu(&gate, &up)?, &idx, sort)?;
        let y = match inverse {
            Some(inverse) => {
                let y = tensor::take(&y, &inverse, 0)?;
                let mut unflat = shape.clone();
                unflat.extend([1, y.dim(-1)]);
                tensor::reshape(&y, &unflat)?
            }
            None => y,
        };
        tensor::squeeze(&y, &[-2])
    }
}

/// Router settings.
#[derive(Clone, Copy, Debug)]
pub struct MoeConfig {
    pub hidden: i32,
    pub experts: i32,
    pub top_k: i32,
    pub intermediate: i32,
    pub norm_topk_prob: bool,
    /// Shared expert width (Qwen3-Next/Qwen3.5); `None` for Qwen3-MoE.
    pub shared_intermediate: Option<i32>,
}

#[derive(Debug)]
pub struct SparseMoe {
    gate: Linear,
    switch: SwitchGlu,
    shared: Option<(Mlp, Linear)>,
    cfg: MoeConfig,
}

impl SparseMoe {
    /// Loads `<prefix>.gate`, `<prefix>.switch_mlp.*` and, with a shared expert,
    /// `<prefix>.shared_expert.*` and `<prefix>.shared_expert_gate`.
    pub fn load(
        weights: &mut Weights,
        prefix: &str,
        quant: &Quantization,
        cfg: MoeConfig,
    ) -> Result<Self, LoadError> {
        let p = |name: &str| format!("{prefix}.{name}");
        let switch = |w: &mut Weights, name: &str, in_dims: i32| {
            SwitchLinear::load(w, &p(&format!("switch_mlp.{name}")), quant, in_dims)
        };
        let shared = match cfg.shared_intermediate {
            Some(width) => Some((
                Mlp::load(weights, &p("shared_expert"), quant, cfg.hidden, width)?,
                Linear::load(weights, &p("shared_expert_gate"), quant, cfg.hidden)?,
            )),
            None => None,
        };
        Ok(Self {
            gate: Linear::load(weights, &p("gate"), quant, cfg.hidden)?,
            switch: SwitchGlu {
                gate_proj: switch(weights, "gate_proj", cfg.hidden)?,
                up_proj: switch(weights, "up_proj", cfg.hidden)?,
                down_proj: switch(weights, "down_proj", cfg.intermediate)?,
            },
            shared,
            cfg,
        })
    }

    pub fn forward(&self, x: &Array) -> tensor::Result<Array> {
        let k = self.cfg.top_k;
        let gates = tensor::softmax(&self.gate.forward(x)?, -1, true)?;
        // `argpartition(gates, kth=-k)[..., -k:]`: the k largest, in partition order.
        let experts = gates.dim(-1);
        let parts = tensor::argpartition(&gates, experts - k, -1)?;
        let inds = tensor::slice(&parts, &[0, 0, experts - k], &[1, x.dim(1), experts])?;
        let mut scores = tensor::take_along_axis(&gates, &inds, -1)?;
        if self.cfg.norm_topk_prob {
            scores = tensor::divide(&scores, &tensor::sum(&scores, -1, true)?)?;
        }
        let y = self.switch.forward(x, &inds)?;
        let y = tensor::sum(
            &tensor::multiply(&y, &tensor::expand_dims(&scores, &[-1])?)?,
            -2,
            false,
        )?;
        match &self.shared {
            Some((expert, gate)) => {
                let shared =
                    tensor::multiply(&tensor::sigmoid(&gate.forward(x)?)?, &expert.forward(x)?)?;
                tensor::add(&y, &shared)
            }
            None => Ok(y),
        }
    }

    /// Activation bytes per token, f32 at worst: router logits, the gathered input rows and
    /// each selected expert's gate, up, product and output, the shared expert.
    pub fn scratch_per_token(&self) -> u64 {
        let c = &self.cfg;
        let k = c.top_k as u64;
        let routed = 2 * c.experts as u64 + k * (2 * c.hidden as u64 + 3 * c.intermediate as u64);
        let shared = self
            .shared
            .as_ref()
            .map_or(0, |(mlp, _)| mlp.scratch_per_token() / 4);
        (routed + shared) * 4
    }
}

/// A decoder layer's feed-forward block: the dense MLP or the sparse MoE.
#[derive(Debug)]
#[expect(
    clippy::large_enum_variant,
    reason = "built once per layer and never moved on a hot path"
)]
pub enum Ffn {
    Dense(Mlp),
    Moe(SparseMoe),
}

impl Ffn {
    pub fn forward(&self, x: &Array) -> tensor::Result<Array> {
        match self {
            Self::Dense(mlp) => mlp.forward(x),
            Self::Moe(moe) => moe.forward(x),
        }
    }

    pub fn scratch_per_token(&self) -> u64 {
        match self {
            Self::Dense(mlp) => mlp.scratch_per_token(),
            Self::Moe(moe) => moe.scratch_per_token(),
        }
    }
}
