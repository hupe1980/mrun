//! The Qwen3-VL vision tower (009) of the `qwen3_5` and `qwen3_5_moe` repos: a port of mlx-vlm
//! 0.7.12's `qwen3_vl/vision.py`. One image (one frame) in, one row per merged 2 × 2 patch out,
//! as wide as the text model's hidden state.
//!
//! Patches come in the processor's order, merge blocks first (block row, block column, then the
//! 2 × 2 inside), each patch row laid out `(t, y, x, c)` — the channels-last layout the
//! checkpoint's Conv3d weight has, so the patch embedding is one matmul.

use serde_json::Value;

use super::LoadError;
use super::common::{SCORE_BLOCK_BYTES, int};
use crate::engine::CancelToken;
use crate::load::{Linear, Quantization, Weights, invalid};
use crate::tensor::{self, Array, Dtype, fast};

/// Geometry of the tower (`vision_config`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VisionConfig {
    pub hidden: i32,
    pub heads: i32,
    pub depth: i32,
    pub intermediate: i32,
    pub patch: i32,
    pub temporal: i32,
    pub merge: i32,
    pub out_hidden: i32,
    /// Side of the learned position grid (`sqrt(num_position_embeddings)`).
    pub grid_side: i32,
}

impl VisionConfig {
    pub fn from_config(v: &Value) -> Result<Self, LoadError> {
        let positions = int(v, "num_position_embeddings")?;
        let side = (f64::from(positions)).sqrt() as i32;
        if side * side != positions {
            return Err(invalid(
                "vision_config",
                "num_position_embeddings is not a square",
            ));
        }
        let cfg = Self {
            hidden: int(v, "hidden_size")?,
            heads: int(v, "num_heads")?,
            depth: int(v, "depth")?,
            intermediate: int(v, "intermediate_size")?,
            patch: int(v, "patch_size")?,
            temporal: int(v, "temporal_patch_size")?,
            merge: int(v, "spatial_merge_size")?,
            out_hidden: int(v, "out_hidden_size")?,
            grid_side: side,
        };
        if v["deepstack_visual_indexes"]
            .as_array()
            .is_some_and(|d| !d.is_empty())
        {
            return Err(invalid(
                "vision_config",
                "deepstack features are not supported",
            ));
        }
        if cfg.hidden % cfg.heads != 0 || (cfg.hidden / cfg.heads) % 4 != 0 {
            return Err(invalid(
                "vision_config",
                "head size must be a multiple of 4",
            ));
        }
        Ok(cfg)
    }

    /// Pixels per merged token side (32 for Qwen3.x).
    pub fn token_px(&self) -> i32 {
        self.patch * self.merge
    }

    /// Values in one patch row (`t · p · p · 3`).
    pub fn patch_values(&self) -> i32 {
        self.temporal * self.patch * self.patch * 3
    }

    fn head_dim(&self) -> i32 {
        self.hidden / self.heads
    }
}

#[derive(Debug)]
struct Norm {
    weight: Array,
    bias: Array,
}

impl Norm {
    fn load(weights: &mut Weights, path: &str) -> Result<Self, LoadError> {
        Ok(Self {
            weight: weights.take(&format!("{path}.weight"))?,
            bias: weights.take(&format!("{path}.bias"))?,
        })
    }

    fn forward(&self, x: &Array) -> tensor::Result<Array> {
        fast::layer_norm(x, Some(&self.weight), Some(&self.bias), 1e-6)
    }
}

#[derive(Debug)]
struct Block {
    norm1: Norm,
    norm2: Norm,
    qkv: Linear,
    proj: Linear,
    fc1: Linear,
    fc2: Linear,
}

/// The tower's weights.
#[derive(Debug)]
pub struct VisionTower {
    cfg: VisionConfig,
    /// The Conv3d as a matmul weight `[hidden, t·p·p·3]`.
    patch: Array,
    patch_bias: Array,
    pos_embed: Array,
    blocks: Vec<Block>,
    merger_norm: Norm,
    merger_fc1: Linear,
    merger_fc2: Linear,
    dtype: Dtype,
    bytes: u64,
}

/// Names the tower's tensors may carry: mlx-community's `vision_tower.` or Hugging Face's
/// `model.visual.`.
pub fn is_vision_tensor(name: &str) -> bool {
    name.starts_with("vision_tower.") || name.starts_with("model.visual.")
}

impl VisionTower {
    /// Takes the tower's tensors out of `weights` (the text model never sees them).
    pub fn load(weights: &mut Weights, v: &Value) -> Result<Self, LoadError> {
        let cfg = VisionConfig::from_config(v)?;
        weights.rename(|name| {
            Some(match name.strip_prefix("model.visual.") {
                Some(rest) => format!("vision_tower.{rest}"),
                None => name.to_string(),
            })
        });
        let none = Quantization::default();
        let p = |name: &str| format!("vision_tower.{name}");
        let raw = weights.take(&p("patch_embed.proj.weight"))?;
        let shape = raw.shape();
        // Channels-last [out, t, p, p, c] (MLX); a PyTorch [out, c, t, p, p] is moved.
        let raw = if shape.len() == 5 && shape[1] == 3 && shape[4] != 3 {
            tensor::transpose(&raw, &[0, 2, 3, 4, 1]).map_err(|e| invalid("weights", e.0))?
        } else {
            raw
        };
        let patch = tensor::reshape(&raw, &[cfg.hidden, cfg.patch_values()])
            .map_err(|e| invalid("weights", format!("patch_embed: {}", e.0)))?;
        let dtype = patch.dtype();
        let patch_bias = weights.take(&p("patch_embed.proj.bias"))?;
        let pos_embed = weights.take(&p("pos_embed.weight"))?;
        let mut blocks = Vec::with_capacity(cfg.depth as usize);
        for i in 0..cfg.depth {
            let b = |name: &str| p(&format!("blocks.{i}.{name}"));
            blocks.push(Block {
                norm1: Norm::load(weights, &b("norm1"))?,
                norm2: Norm::load(weights, &b("norm2"))?,
                qkv: Linear::load(weights, &b("attn.qkv"), &none, cfg.hidden)?,
                proj: Linear::load(weights, &b("attn.proj"), &none, cfg.hidden)?,
                fc1: Linear::load(weights, &b("mlp.linear_fc1"), &none, cfg.hidden)?,
                fc2: Linear::load(weights, &b("mlp.linear_fc2"), &none, cfg.intermediate)?,
            });
        }
        let merged = cfg.hidden * cfg.merge * cfg.merge;
        let tower = Self {
            merger_norm: Norm::load(weights, &p("merger.norm"))?,
            merger_fc1: Linear::load(weights, &p("merger.linear_fc1"), &none, merged)?,
            merger_fc2: Linear::load(weights, &p("merger.linear_fc2"), &none, merged)?,
            cfg,
            patch,
            patch_bias,
            pos_embed,
            blocks,
            dtype,
            bytes: 0,
        };
        Ok(Self {
            bytes: tower.parameters() * dtype.size(),
            ..tower
        })
    }

    pub fn config(&self) -> VisionConfig {
        self.cfg
    }

    /// Bytes of the tower's weights.
    pub fn bytes(&self) -> u64 {
        self.bytes
    }

    fn parameters(&self) -> u64 {
        let c = &self.cfg;
        let (h, i) = (c.hidden as u64, c.intermediate as u64);
        let merged = h * (c.merge * c.merge) as u64;
        let block = 4 * h + (3 * h * h + 3 * h) + (h * h + h) + (h * i + i) + (i * h + h);
        h * c.patch_values() as u64
            + h
            + (c.grid_side * c.grid_side) as u64 * h
            + c.depth as u64 * block
            + 2 * h
            + (merged * merged + merged)
            + (merged * c.out_hidden as u64 + c.out_hidden as u64)
    }

    /// Activation bytes of encoding an image of `patches` patches (f32 at worst), the attention's
    /// own scratch included: the peak of one block plus the merger.
    pub fn scratch(&self, patches: u64) -> u64 {
        let c = &self.cfg;
        let (h, i) = (c.hidden as u64, c.intermediate as u64);
        // f32 at worst: a block's projections, MLP and residuals, then the attention's scores
        // (tiled above the score limit), plus the patch rows themselves.
        let block = patches * (6 * h + 2 * i) * 4;
        let heads = c.heads as u64;
        let attention = (heads * patches * patches * 4).min(SCORE_BLOCK_BYTES);
        let merger = patches * h * 4 * 2;
        let input = patches * c.patch_values() as u64 * 4;
        block.max(merger) + attention + input
    }

    /// The features of one image `[tokens, out_hidden]`: `patches` `[gh · gw, t·p·p·3]` (bf16 or
    /// f32, in [-1, 1]) of a grid `gh` × `gw` patches (multiples of the merge size). `None` when
    /// cancelled (checked after every block).
    pub fn encode(
        &self,
        patches: &Array,
        gh: i32,
        gw: i32,
        cancel: &CancelToken,
    ) -> tensor::Result<Option<Array>> {
        self.encode_traced(patches, gh, gw, cancel, &mut Vec::new())
    }

    /// [`Self::encode`] keeping the hidden state after the embedding and after every block
    /// (comparison with the reference).
    #[doc(hidden)]
    pub fn encode_traced(
        &self,
        patches: &Array,
        gh: i32,
        gw: i32,
        cancel: &CancelToken,
        stages: &mut Vec<Array>,
    ) -> tensor::Result<Option<Array>> {
        let c = &self.cfg;
        let n = gh * gw;
        let x = tensor::astype(patches, self.dtype)?;
        let x = tensor::matmul(&x, &tensor::transpose(&self.patch, &[1, 0])?)?;
        let x = tensor::add(&x, &self.patch_bias)?;
        let mut x = tensor::add(&x, &self.positions(gh, gw)?)?;
        stages.push(x.clone());
        let (cos, sin) = self.rotary(gh, gw);
        let hd = c.head_dim();
        let cos = Array::from_f32(&cos, &[n, 1, hd]);
        let sin = Array::from_f32(&sin, &[n, 1, hd]);
        for block in &self.blocks {
            let h = block.norm1.forward(&x)?;
            let qkv = tensor::reshape(&block.qkv.forward(&h)?, &[n, 3, c.heads, hd])?;
            let part = |k: i32| {
                tensor::slice(&qkv, &[0, k, 0, 0], &[n, k + 1, c.heads, hd])
                    .and_then(|y| tensor::reshape(&y, &[n, c.heads, hd]))
            };
            let rotate = |y: &Array| -> tensor::Result<Array> {
                // `x · cos + rotate_half(x) · sin` in f32, back to the tower's dtype.
                let half = hd / 2;
                let a = tensor::slice(y, &[0, 0, 0], &[n, c.heads, half])?;
                let b = tensor::slice(y, &[0, 0, half], &[n, c.heads, hd])?;
                let rotated = tensor::concatenate(&[&tensor::negative(&b)?, &a], 2)?;
                let out = tensor::add(
                    &tensor::multiply(y, &cos)?,
                    &tensor::multiply(&rotated, &sin)?,
                )?;
                tensor::astype(&out, self.dtype)
            };
            let bhld = |y: &Array| {
                tensor::transpose(y, &[1, 0, 2]).and_then(|y| tensor::expand_dims(&y, &[0]))
            };
            let q = bhld(&rotate(&part(0)?)?)?;
            let k = bhld(&rotate(&part(1)?)?)?;
            let v = bhld(&part(2)?)?;
            let scale = (hd as f32).powf(-0.5);
            let o = fast::scaled_dot_product_attention(
                &q,
                &k,
                &v,
                scale,
                fast::Mask::None,
                SCORE_BLOCK_BYTES,
            )?;
            // [1, H, n, d] → [n, H · d]
            let o = tensor::reshape(&tensor::transpose(&o, &[0, 2, 1, 3])?, &[n, c.hidden])?;
            x = tensor::add(&x, &block.proj.forward(&o)?)?;
            let h = block.norm2.forward(&x)?;
            let h = gelu_tanh(&block.fc1.forward(&h)?)?;
            x = tensor::add(&x, &block.fc2.forward(&h)?)?;
            stages.push(x.clone());
            tensor::async_eval(&[&x])?;
            if cancel.is_cancelled() {
                return Ok(None);
            }
        }
        let merged = c.hidden * c.merge * c.merge;
        let x = self.merger_norm.forward(&x)?;
        let x = tensor::reshape(&x, &[n / (c.merge * c.merge), merged])?;
        let x = gelu(&self.merger_fc1.forward(&x)?)?;
        let out = self.merger_fc2.forward(&x)?;
        out.eval()?;
        Ok(Some(out))
    }

    /// The learned position embedding interpolated bilinearly to `gh` × `gw` and put in merge
    /// order (mlx-vlm `fast_pos_embed_interpolate`).
    fn positions(&self, gh: i32, gw: i32) -> tensor::Result<Array> {
        let side = self.cfg.grid_side;
        let lin = |count: i32| -> Vec<f32> {
            if count == 1 {
                return vec![0.0];
            }
            let step = (side - 1) as f32 / (count - 1) as f32;
            (0..count).map(|i| i as f32 * step).collect()
        };
        let (hs, ws) = (lin(gh), lin(gw));
        let m = self.cfg.merge;
        let mut idx = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
        let mut wts = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
        for (y, x) in merge_order(gh, gw, m) {
            let (h, w) = (hs[y as usize], ws[x as usize]);
            let (hf, wf) = (h as i32, w as i32);
            let (hc, wc) = ((hf + 1).min(side - 1), (wf + 1).min(side - 1));
            let (dh, dw) = (h - hf as f32, w - wf as f32);
            let corners = [
                (hf * side + wf, (1.0 - dh) * (1.0 - dw)),
                (hf * side + wc, (1.0 - dh) * dw),
                (hc * side + wf, dh * (1.0 - dw)),
                (hc * side + wc, dh * dw),
            ];
            for (k, (i, wt)) in corners.into_iter().enumerate() {
                idx[k].push(i as u32);
                wts[k].push(wt);
            }
        }
        let n = gh * gw;
        let mut sum: Option<Array> = None;
        for k in 0..4 {
            let rows = tensor::take(&self.pos_embed, &Array::from_u32(&idx[k], &[n]), 0)?;
            let wt = tensor::astype(&Array::from_f32(&wts[k], &[n, 1]), self.dtype)?;
            let term = tensor::multiply(&rows, &wt)?;
            sum = Some(match sum {
                None => term,
                Some(s) => tensor::add(&s, &term)?,
            });
        }
        Ok(sum.expect("four corners"))
    }

    /// cos and sin `[n, head_dim]` of the 2-D rotary embedding in merge order: the first half of
    /// each head's features rotates with the patch row, the second with its column.
    fn rotary(&self, gh: i32, gw: i32) -> (Vec<f32>, Vec<f32>) {
        let dim = self.cfg.head_dim() / 2;
        let inv: Vec<f32> = (0..dim / 2)
            .map(|j| 1.0 / 10000f32.powf((2 * j) as f32 / dim as f32))
            .collect();
        let hd = self.cfg.head_dim() as usize;
        let mut cos = Vec::with_capacity((gh * gw) as usize * hd);
        let mut sin = Vec::with_capacity((gh * gw) as usize * hd);
        for (y, x) in merge_order(gh, gw, self.cfg.merge) {
            let angles: Vec<f32> = inv
                .iter()
                .map(|f| y as f32 * f)
                .chain(inv.iter().map(|f| x as f32 * f))
                .collect();
            // Tiled twice: halves of the head share the angles (`rotate_half`).
            for _ in 0..2 {
                cos.extend(angles.iter().map(|a| a.cos()));
                sin.extend(angles.iter().map(|a| a.sin()));
            }
        }
        (cos, sin)
    }
}

/// Patch (row, column) of each position in merge order: merge blocks row by row, the
/// `merge × merge` patches of a block row by row.
pub fn merge_order(gh: i32, gw: i32, merge: i32) -> impl Iterator<Item = (i32, i32)> {
    (0..gh / merge).flat_map(move |br| {
        (0..gw / merge).flat_map(move |bc| {
            (0..merge).flat_map(move |r| (0..merge).map(move |c| (br * merge + r, bc * merge + c)))
        })
    })
}

/// `nn.GELU(approx="tanh")`.
fn gelu_tanh(x: &Array) -> tensor::Result<Array> {
    let d = x.dtype();
    let s = |v: f32| tensor::scalar(v, d);
    let cube = tensor::multiply(&tensor::multiply(x, x)?, x)?;
    let inner = tensor::multiply(
        &s((2.0 / std::f32::consts::PI).sqrt())?,
        &tensor::add(x, &tensor::multiply(&s(0.044715)?, &cube)?)?,
    )?;
    let one = tensor::add(&s(1.0)?, &tensor::tanh(&inner)?)?;
    tensor::multiply(&tensor::multiply(&s(0.5)?, x)?, &one)
}

/// `nn.GELU()` (exact): `x · (1 + erf(x / √2)) / 2`.
fn gelu(x: &Array) -> tensor::Result<Array> {
    let d = x.dtype();
    let s = |v: f32| tensor::scalar(v, d);
    let e = tensor::erf(&tensor::divide(x, &s(std::f32::consts::SQRT_2)?)?)?;
    tensor::divide(&tensor::multiply(x, &tensor::add(&s(1.0)?, &e)?)?, &s(2.0)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn merge_order_walks_blocks_first() {
        let order: Vec<_> = merge_order(2, 4, 2).collect();
        assert_eq!(
            order,
            [
                (0, 0),
                (0, 1),
                (1, 0),
                (1, 1),
                (0, 2),
                (0, 3),
                (1, 2),
                (1, 3)
            ]
        );
    }
}
