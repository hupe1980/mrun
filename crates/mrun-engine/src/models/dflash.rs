//! DFlash drafter (030): a small Qwen3-style stack that reads the target's hidden states after a
//! few of its layers and proposes a block of tokens in one forward pass. A port of nuvec-lm
//! 0.14.0's `DFlashDrafter` (DFlash v1, itself checked against z-lab's MLX reference), on mrun's
//! own target graph.
//!
//! - **Context** ("KV injection"): for each position the target computes, `h =
//!   hidden_norm(fc(features))`, and every drafter layer stores `k = rope(k_norm(k_proj(h)))` and
//!   `v = v_proj(h)` at that position. No queries run for context positions.
//! - **Draft**: the block `[anchor, mask × n]` at the anchor's position is embedded with the
//!   target's embedding and attends to the context and to itself (bidirectional, causal in
//!   sliding layers as z-lab's reference); the drafter's final norm and the target's output
//!   projection (without its final norm) give logits, and rows 1..=n their argmax.
//!
//! The drafter only proposes: 006's verify keeps what plain decoding samples, so its state may be
//! stale, partial or empty without changing any output (030 FR-003). [`DraftContext`] keeps at
//! most the last window of positions and repairs itself: a write below its end cuts it there, a
//! write past its end restarts it.

use std::path::Path;

use serde_json::Value;

use super::{LoadError, Model};
use crate::load::{Linear, Quantization, RmsNorm, Weights, invalid};
use crate::tensor::{self, Array, Dtype, fast};

/// Positions a context holds beyond its window before it compacts (nuvec's bounded window cache).
pub const SLACK: u32 = 256;
/// Context kept by a drafter without sliding layers.
const DEFAULT_WINDOW: u32 = 4096;
/// Bits and group size of projections quantized at load (nuvec-lm and dflash-mlx).
const QUANT_BITS: i32 = 4;
const QUANT_GROUP: i32 = 64;

/// A drafter's `config.json` (DFlash v1, `architectures: ["DFlashDraftModel"]`).
#[derive(Clone, Debug, PartialEq)]
pub struct DFlashConfig {
    pub hidden: i32,
    pub intermediate: i32,
    pub layers: usize,
    pub heads: i32,
    pub kv_heads: i32,
    pub head_dim: i32,
    pub eps: f32,
    pub rope_theta: f32,
    pub rope_scale: f32,
    /// Per layer: its sliding window, `None` for full attention.
    pub windows: Vec<Option<u32>>,
    /// Per layer: whether the block attends to itself causally.
    pub causal: Vec<bool>,
    pub block_size: usize,
    /// The target layers whose outputs the drafter reads, in feature order.
    pub target_layer_ids: Vec<usize>,
    pub mask_token: u32,
    pub vocab: u32,
}

impl DFlashConfig {
    /// Parses a DFlash v1 config; DFlash settings are read from `dflash_config`, then the top
    /// level (as z-lab's loader).
    pub fn from_config(c: &Value) -> Result<Self, LoadError> {
        let arch = c["architectures"][0].as_str().unwrap_or("");
        match arch {
            "DFlashDraftModel" => {}
            "DFlash2DraftModel" | "Qwen3DSparkModel" => {
                return Err(invalid(
                    "architectures",
                    format!("{arch} is not supported yet (DFlash v1 only)"),
                ));
            }
            other => {
                return Err(invalid(
                    "architectures",
                    format!("{other:?} is not a DFlash drafter"),
                ));
            }
        }
        let int = |k: &str| {
            c[k].as_i64()
                .filter(|&v| v > 0)
                .ok_or_else(|| invalid(k, "is missing or not positive"))
        };
        let setting = |k: &str| {
            let d = &c["dflash_config"];
            if d[k].is_null() { &c[k] } else { &d[k] }
        };
        let layers = int("num_hidden_layers")? as usize;
        let types: Vec<String> = match c["layer_types"].as_array() {
            Some(t) => t
                .iter()
                .map(|v| v.as_str().unwrap_or("").to_owned())
                .collect(),
            None => vec!["full_attention".into(); layers],
        };
        if types.len() != layers {
            return Err(invalid("layer_types", "does not match num_hidden_layers"));
        }
        let window = c["sliding_window"].as_u64().map(|w| w as u32);
        let explicit_causal = c["is_causal"].as_bool();
        let mut windows = Vec::with_capacity(layers);
        let mut causal = Vec::with_capacity(layers);
        for t in &types {
            let sliding = match t.as_str() {
                "full_attention" => false,
                "sliding_attention" => true,
                other => {
                    return Err(invalid(
                        "layer_types",
                        format!("{other:?} is not supported"),
                    ));
                }
            };
            if sliding && window.is_none() {
                return Err(invalid("sliding_window", "is missing for sliding layers"));
            }
            windows.push(window.filter(|_| sliding));
            causal.push(explicit_causal.unwrap_or(sliding));
        }
        let rope = c["rope_theta"]
            .as_f64()
            .or_else(|| c["rope_parameters"]["rope_theta"].as_f64())
            .unwrap_or(10_000.0);
        let scaling = if c["rope_parameters"].is_object() {
            Some(&c["rope_parameters"])
        } else {
            c.get("rope_scaling")
        };
        let target_layer_ids = setting("target_layer_ids")
            .as_array()
            .ok_or_else(|| invalid("target_layer_ids", "is missing"))?
            .iter()
            .map(|v| v.as_u64().map(|v| v as usize))
            .collect::<Option<Vec<_>>>()
            .filter(|ids| !ids.is_empty() && ids.windows(2).all(|w| w[0] < w[1]))
            .ok_or_else(|| invalid("target_layer_ids", "must be ascending layer indices"))?;
        Ok(Self {
            hidden: int("hidden_size")? as i32,
            intermediate: int("intermediate_size")? as i32,
            layers,
            heads: int("num_attention_heads")? as i32,
            kv_heads: int("num_key_value_heads")? as i32,
            head_dim: int("head_dim")? as i32,
            eps: c["rms_norm_eps"].as_f64().unwrap_or(1e-6) as f32,
            rope_theta: rope as f32,
            rope_scale: super::common::rope_scale(scaling)?,
            windows,
            causal,
            block_size: setting("block_size").as_u64().unwrap_or(16).max(2) as usize,
            target_layer_ids,
            mask_token: setting("mask_token_id")
                .as_u64()
                .ok_or_else(|| invalid("mask_token_id", "is missing"))?
                as u32,
            vocab: int("vocab_size")? as u32,
        })
    }

    /// Positions a context keeps: the largest window less one (the block's anchor is the
    /// window's last position).
    pub fn keep(&self) -> u32 {
        self.windows
            .iter()
            .flatten()
            .copied()
            .max()
            .unwrap_or(DEFAULT_WINDOW)
            .saturating_sub(1)
            .max(1)
    }

    /// Bytes of one context position over every layer (keys and values, bf16).
    pub fn bytes_per_position(&self) -> u64 {
        self.layers as u64 * 2 * self.kv_heads as u64 * self.head_dim as u64 * 2
    }

    /// Bytes of one lane's context at its largest (the window plus the slack).
    pub fn context_bytes(&self) -> u64 {
        u64::from(self.keep() + SLACK) * self.bytes_per_position()
    }
}

/// What a drafter must agree with: its target's width, vocabulary and layer count.
#[derive(Clone, Copy, Debug)]
pub struct Target {
    pub hidden: i32,
    pub vocab: u32,
    pub layers: usize,
}

#[derive(Debug)]
struct Layer {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

/// A loaded DFlash drafter for one target.
#[derive(Debug)]
pub struct DFlash {
    config: DFlashConfig,
    fc: Linear,
    hidden_norm: RmsNorm,
    norm: RmsNorm,
    layers: Vec<Layer>,
    bytes: u64,
}

impl DFlash {
    /// Loads the drafter in `dir` for `target`: its config must match the target's width,
    /// vocabulary and layers; bf16 projections are quantized here (affine 4-bit, group 64), one
    /// tensor at a time, so the full-precision file is never held at once.
    pub fn load(dir: &Path, target: Target) -> Result<Self, LoadError> {
        let path = dir.join("config.json");
        let config: Value = std::fs::read_to_string(&path)
            .map_err(|e| e.to_string())
            .and_then(|t| serde_json::from_str(&t).map_err(|e| e.to_string()))
            .map_err(|e| invalid("config.json", format!("{}: {e}", path.display())))?;
        let c = DFlashConfig::from_config(&config)?;
        if c.hidden != target.hidden {
            return Err(invalid(
                "hidden_size",
                format!("{} does not match the target's {}", c.hidden, target.hidden),
            ));
        }
        if c.vocab != target.vocab {
            return Err(invalid(
                "vocab_size",
                format!("{} does not match the target's {}", c.vocab, target.vocab),
            ));
        }
        if let Some(i) = c.target_layer_ids.iter().find(|&&i| i >= target.layers) {
            return Err(invalid(
                "target_layer_ids",
                format!("layer {i} of a {}-layer target", target.layers),
            ));
        }
        if c.mask_token >= c.vocab {
            return Err(invalid("mask_token_id", "is outside the vocabulary"));
        }
        let quant = Quantization::from_config(&config)?;
        let mut weights = Weights::load(dir, |_| true)?;
        // Some exports keep the transformers prefix.
        weights.rename(|name| Some(name.strip_prefix("model.").unwrap_or(name).to_owned()));
        let mut bytes = 0u64;
        let mut linear = |weights: &mut Weights, name: &str, in_dims: i32| {
            let quantized = weights.contains(&format!("{name}.scales"));
            let layer = if quantized {
                Linear::load(weights, name, &quant, in_dims)?
            } else if in_dims % QUANT_GROUP != 0 {
                // Rows that do not split into groups stay dense (tiny test drafters).
                Linear::load(weights, name, &quant, in_dims)?
            } else {
                let weight = weights.take(&format!("{name}.weight"))?;
                let bias = weights.take_opt(&format!("{name}.bias"));
                let w = tensor::quantize(&weight, QUANT_GROUP, QUANT_BITS)
                    .map_err(|e| invalid("weights", format!("{name}: {e}")))?;
                drop(weight);
                Linear::Quantized { w, bias }
            };
            bytes += linear_bytes(&layer);
            Ok::<_, LoadError>(layer)
        };
        let taps = (c.target_layer_ids.len() as i32) * c.hidden;
        let fc = linear(&mut weights, "fc", taps)?;
        let inner = c.heads * c.head_dim;
        let mut layers = Vec::with_capacity(c.layers);
        for i in 0..c.layers {
            let p = |s: &str| format!("layers.{i}.{s}");
            let (q_proj, k_proj, v_proj, o_proj) = (
                linear(&mut weights, &p("self_attn.q_proj"), c.hidden)?,
                linear(&mut weights, &p("self_attn.k_proj"), c.hidden)?,
                linear(&mut weights, &p("self_attn.v_proj"), c.hidden)?,
                linear(&mut weights, &p("self_attn.o_proj"), inner)?,
            );
            let gate_proj = linear(&mut weights, &p("mlp.gate_proj"), c.hidden)?;
            let up_proj = linear(&mut weights, &p("mlp.up_proj"), c.hidden)?;
            let down_proj = linear(&mut weights, &p("mlp.down_proj"), c.intermediate)?;
            layers.push(Layer {
                q_proj,
                k_proj,
                v_proj,
                o_proj,
                gate_proj,
                up_proj,
                down_proj,
                q_norm: RmsNorm::load(&mut weights, &p("self_attn.q_norm"), c.eps)?,
                k_norm: RmsNorm::load(&mut weights, &p("self_attn.k_norm"), c.eps)?,
                input_layernorm: RmsNorm::load(&mut weights, &p("input_layernorm"), c.eps)?,
                post_attention_layernorm: RmsNorm::load(
                    &mut weights,
                    &p("post_attention_layernorm"),
                    c.eps,
                )?,
            });
        }
        let hidden_norm = RmsNorm::load(&mut weights, "hidden_norm", c.eps)?;
        let norm = RmsNorm::load(&mut weights, "norm", c.eps)?;
        // Copies of the target's embedding or head some exports carry are not used.
        weights.retain(|name| !name.starts_with("embed_tokens.") && !name.starts_with("lm_head."));
        super::common::ensure_consumed(&weights)?;
        let norm_bytes = |n: &RmsNorm| n.weight.size() as u64 * n.weight.dtype().size();
        bytes += norm_bytes(&hidden_norm) + norm_bytes(&norm);
        for l in &layers {
            for n in [
                &l.q_norm,
                &l.k_norm,
                &l.input_layernorm,
                &l.post_attention_layernorm,
            ] {
                bytes += norm_bytes(n);
            }
        }
        Ok(Self {
            config: c,
            fc,
            hidden_norm,
            norm,
            layers,
            bytes,
        })
    }

    pub fn config(&self) -> &DFlashConfig {
        &self.config
    }

    /// Bytes of the loaded weights.
    pub fn bytes(&self) -> u64 {
        self.bytes
    }

    /// The target layers whose outputs feed the drafter.
    pub fn taps(&self) -> &[usize] {
        &self.config.target_layer_ids
    }

    /// An empty context for this drafter.
    pub fn context(&self) -> DraftContext {
        DraftContext::new(
            self.config.layers,
            self.config.keep(),
            self.config.bytes_per_position(),
        )
    }

    /// `[1, L, kv_heads, head_dim]`-shaped heads of `y` `[1, L, n · head_dim]`, normed by `norm`,
    /// as `[1, n, L, head_dim]`.
    fn heads(&self, y: &Array, n: i32, norm: Option<&RmsNorm>) -> tensor::Result<Array> {
        let y = tensor::reshape(y, &[1, y.dim(1), n, self.config.head_dim])?;
        let y = match norm {
            Some(norm) => norm.forward(&y)?,
            None => y,
        };
        tensor::transpose(&y, &[0, 2, 1, 3])
    }

    fn rope(&self, y: &Array, at: u32) -> tensor::Result<Array> {
        let c = &self.config;
        fast::rope(
            y,
            c.head_dim,
            false,
            Some(c.rope_theta),
            c.rope_scale,
            at as i32,
            None,
        )
    }

    /// Feeds the target's `features` `[1, n, taps · hidden]` of positions `[at, at + n)` into
    /// `ctx`. Only the last window of them is computed: earlier ones would be dropped at once.
    pub fn extend(&self, ctx: &mut DraftContext, features: &Array, at: u32) -> tensor::Result<()> {
        let n = features.dim(1) as u32;
        if n == 0 {
            return Ok(());
        }
        let skip = n.saturating_sub(ctx.keep);
        let features = if skip > 0 {
            let shape = features.shape();
            tensor::slice(features, &[0, skip as i32, 0], &[1, shape[1], shape[2]])?
        } else {
            features.clone()
        };
        let (at, n) = (at + skip, n - skip);
        let c = &self.config;
        let h = self.hidden_norm.forward(&self.fc.forward(&features)?)?;
        let mut rows = Vec::with_capacity(self.layers.len());
        for layer in &self.layers {
            let k = self.heads(&layer.k_proj.forward(&h)?, c.kv_heads, Some(&layer.k_norm))?;
            let k = tensor::astype(&self.rope(&k, at)?, Dtype::Bfloat16)?;
            let v = self.heads(&layer.v_proj.forward(&h)?, c.kv_heads, None)?;
            rows.push((k, tensor::astype(&v, Dtype::Bfloat16)?));
        }
        ctx.write(at, n, rows)
    }

    /// Up to `n` draft tokens after `anchor` at position `at`, from the context below `at`.
    /// Empty without context.
    pub fn draft(
        &self,
        ctx: &DraftContext,
        model: &dyn Model,
        anchor: u32,
        at: u32,
        n: usize,
    ) -> tensor::Result<Vec<u32>> {
        let c = &self.config;
        let n = n.min(c.block_size - 1);
        let context = ctx.rows_below(at);
        if n == 0 || context == 0 {
            return Ok(Vec::new());
        }
        let first = ctx.start;
        let b = n as i32 + 1;
        let mut ids = vec![c.mask_token; n + 1];
        ids[0] = anchor;
        let mut x = model.embed(&Array::from_u32(&ids, &[1, b]))?;
        let scale = (c.head_dim as f32).powf(-0.5);
        for (l, layer) in self.layers.iter().enumerate() {
            let h = layer.input_layernorm.forward(&x)?;
            let q = self.heads(&layer.q_proj.forward(&h)?, c.heads, Some(&layer.q_norm))?;
            let q = self.rope(&q, at)?;
            let k = self.heads(&layer.k_proj.forward(&h)?, c.kv_heads, Some(&layer.k_norm))?;
            let k = tensor::astype(&self.rope(&k, at)?, Dtype::Bfloat16)?;
            let v = tensor::astype(
                &self.heads(&layer.v_proj.forward(&h)?, c.kv_heads, None)?,
                Dtype::Bfloat16,
            )?;
            let (ck, cv) = ctx.layer(l, context)?;
            let keys = tensor::concatenate(&[&ck, &k], 2)?;
            let values = tensor::concatenate(&[&cv, &v], 2)?;
            let q = tensor::astype(&q, Dtype::Bfloat16)?;
            let o = match block_mask(c.windows[l], c.causal[l], first, context, at, b as u32) {
                Some(mask) => {
                    fast::scaled_dot_product_attention_masked(&q, &keys, &values, scale, &mask)?
                }
                None => fast::scaled_dot_product_attention(
                    &q,
                    &keys,
                    &values,
                    scale,
                    fast::Mask::None,
                    u64::MAX,
                )?,
            };
            let o = tensor::reshape(&tensor::transpose(&o, &[0, 2, 1, 3])?, &[1, b, -1])?;
            let o = tensor::astype(&layer.o_proj.forward(&o)?, x.dtype())?;
            x = tensor::add(&x, &o)?;
            let h = layer.post_attention_layernorm.forward(&x)?;
            let mlp = super::activations::swiglu(
                &layer.gate_proj.forward(&h)?,
                &layer.up_proj.forward(&h)?,
            )?;
            let mlp = tensor::astype(&layer.down_proj.forward(&mlp)?, x.dtype())?;
            x = tensor::add(&x, &mlp)?;
        }
        let x = self.norm.forward(&x)?;
        let rows = tensor::slice(&x, &[0, 1, 0], &[1, b, x.dim(2)])?;
        let logits = model.head_raw(&rows)?;
        tensor::argmax(&logits, -1, false)?.to_vec_u32()
    }
}

fn linear_bytes(layer: &Linear) -> u64 {
    let size = |a: &Array| a.size() as u64 * a.dtype().size();
    match layer {
        Linear::Dense { weight, bias } => size(weight) + bias.as_ref().map_or(0, size),
        Linear::Quantized { w, bias } => w.bytes() + bias.as_ref().map_or(0, size),
    }
}

/// The mask of a block of `b` rows at `at` over `context` context positions from `first`, then
/// the block itself: context keys within the layer's window of each row, block keys before or at
/// the row when causal. `None` when every key is allowed.
fn block_mask(
    window: Option<u32>,
    causal: bool,
    first: u32,
    context: u32,
    at: u32,
    b: u32,
) -> Option<Array> {
    let in_window = |row: u32, pos: u32| window.is_none_or(|w| at + row - pos < w);
    let all = !causal && (0..b).all(|row| in_window(row, first));
    if all {
        return None;
    }
    let keys = context + b;
    let mut mask = Vec::with_capacity((b * keys) as usize);
    for row in 0..b {
        for key in 0..keys {
            mask.push(if key < context {
                in_window(row, first + key)
            } else {
                !causal || key - context <= row
            });
        }
    }
    Some(Array::from_bool(&mask, &[b as i32, keys as i32]))
}

/// One lane's (or segment's) drafter context: per layer the K/V `[1, kv_heads, cap, head_dim]` of
/// positions `[start, end)` (row `i` is position `start + i`), at most `keep` of them after a
/// write, with [`SLACK`] rows of room so most writes need no copy.
#[derive(Debug)]
pub struct DraftContext {
    start: u32,
    end: u32,
    keep: u32,
    /// Bytes of one position over every layer.
    per_position: u64,
    layers: Vec<Option<(Array, Array)>>,
}

impl DraftContext {
    fn new(layers: usize, keep: u32, per_position: u64) -> Self {
        Self {
            start: 0,
            end: 0,
            keep,
            per_position,
            layers: (0..layers).map(|_| None).collect(),
        }
    }

    /// The first position [`DraftContext::split_off`] at `at` moves.
    fn tail_start(&self, at: u32) -> u32 {
        at.max(self.start).max(self.end.saturating_sub(self.keep))
    }

    /// Bytes [`DraftContext::split_off`] at `at` would move out.
    pub fn tail_bytes(&self, at: u32) -> u64 {
        u64::from(self.end.saturating_sub(self.tail_start(at))) * self.per_position
    }

    /// Positions held.
    pub fn len(&self) -> u32 {
        self.end - self.start
    }

    pub fn is_empty(&self) -> bool {
        self.start == self.end
    }

    /// The first and end position held.
    pub fn span(&self) -> (u32, u32) {
        (self.start, self.end)
    }

    /// Bytes held by the buffers.
    pub fn bytes(&self) -> u64 {
        let size = |a: &Array| a.size() as u64 * a.dtype().size();
        self.layers
            .iter()
            .flatten()
            .map(|(k, v)| size(k) + size(v))
            .sum()
    }

    /// Drops every position.
    pub fn clear(&mut self) {
        for layer in &mut self.layers {
            *layer = None;
        }
        self.start = 0;
        self.end = 0;
    }

    /// Positions held below `at` (a draft at `at` reads them).
    fn rows_below(&self, at: u32) -> u32 {
        self.end.min(at).saturating_sub(self.start)
    }

    /// Layer `l`'s first `rows` keys and values.
    fn layer(&self, l: usize, rows: u32) -> tensor::Result<(Array, Array)> {
        let (k, v) = self.layers[l]
            .as_ref()
            .ok_or_else(|| tensor::Error("an empty drafter context".into()))?;
        let shape = k.shape();
        let stop = [shape[0], shape[1], rows as i32, shape[3]];
        Ok((
            tensor::slice(k, &[0, 0, 0, 0], &stop)?,
            tensor::slice(v, &[0, 0, 0, 0], &stop)?,
        ))
    }

    /// Writes per-layer `[k, v]` `[1, kv_heads, n, head_dim]` of positions `[at, at + n)`: cuts
    /// the context to `at` when it holds positions at or past it, restarts it at `at` when it
    /// ends before `at` (a gap) or starts after it, and keeps the last `keep` positions.
    fn write(&mut self, at: u32, n: u32, rows: Vec<(Array, Array)>) -> tensor::Result<()> {
        debug_assert!(n <= self.keep);
        if self.is_empty() || at < self.start || at > self.end {
            self.clear();
            self.start = at;
            self.end = at;
        }
        self.end = at;
        // Positions kept from before: as many of the last ones as fit beside the new rows.
        let kept = self.len().min(self.keep - n);
        let cap = self.keep + SLACK;
        let index = self.len();
        let moved = index + n > cap || self.layers.iter().any(Option::is_none);
        let drop = self.len() - kept;
        for (layer, (k, v)) in self.layers.iter_mut().zip(rows) {
            let shape = k.shape();
            let (heads, dim) = (shape[1], shape[3]);
            let full = [1, heads, cap as i32, dim];
            let (mut keys, mut values) = match layer.take() {
                Some((old_k, old_v)) if !moved => (old_k, old_v),
                old => {
                    let fresh = |old: Option<&Array>| -> tensor::Result<Array> {
                        let buf = tensor::zeros(&full, k.dtype())?;
                        match old.filter(|_| kept > 0) {
                            Some(old) => {
                                let piece = tensor::slice(
                                    old,
                                    &[0, 0, drop as i32, 0],
                                    &[1, heads, (drop + kept) as i32, dim],
                                )?;
                                let to = [1, heads, kept as i32, dim];
                                tensor::slice_update(&buf, &piece, &[0, 0, 0, 0], &to)
                            }
                            None => Ok(buf),
                        }
                    };
                    let (ok, ov) = match &old {
                        Some((k, v)) => (Some(k), Some(v)),
                        None => (None, None),
                    };
                    (fresh(ok)?, fresh(ov)?)
                }
            };
            let from = if moved { kept } else { index } as i32;
            let to = [1, heads, from + n as i32, dim];
            keys = tensor::slice_update(&keys, &k, &[0, 0, from, 0], &to)?;
            values = tensor::slice_update(&values, &v, &[0, 0, from, 0], &to)?;
            // Queued, not awaited: decode keeps its pipeline (the step's token is read back while
            // this runs).
            tensor::async_eval(&[&keys, &values])?;
            *layer = Some((keys, values));
        }
        if moved {
            self.start += drop;
        }
        self.end = at + n;
        Ok(())
    }

    /// Moves positions `[at, end)` out (at most `keep`), leaving `[start, at)`.
    pub fn split_off(&mut self, at: u32) -> tensor::Result<DraftContext> {
        let mut tail = DraftContext::new(self.layers.len(), self.keep, self.per_position);
        if at >= self.end || self.is_empty() {
            return Ok(tail);
        }
        // At most the window: a context holds more only until its next compaction.
        let from = self.tail_start(at);
        let (lo, hi) = (from - self.start, self.end - self.start);
        tail.start = from;
        tail.end = self.end;
        for (mine, theirs) in self.layers.iter().zip(tail.layers.iter_mut()) {
            if let Some((k, v)) = mine {
                let shape = k.shape();
                let (a, b) = ([0, 0, lo as i32, 0], [1, shape[1], hi as i32, shape[3]]);
                let k = tensor::deep_copy(&tensor::slice(k, &a, &b)?)?;
                let v = tensor::deep_copy(&tensor::slice(v, &a, &b)?)?;
                tensor::eval(&[&k, &v])?;
                *theirs = Some((k, v));
            }
        }
        if at <= self.start {
            self.clear();
        } else {
            self.end = at;
        }
        Ok(tail)
    }

    /// Appends `tail` (from [`DraftContext::split_off`]) when it starts where this context ends;
    /// otherwise this context becomes `tail`'s positions.
    pub fn append(&mut self, tail: DraftContext) -> tensor::Result<()> {
        if tail.is_empty() {
            return Ok(());
        }
        if self.is_empty() || self.end != tail.start {
            self.clear();
        }
        let (start, n) = (tail.start, tail.len());
        let rows = tail
            .layers
            .into_iter()
            .map(|l| l.ok_or_else(|| tensor::Error("a drafter context without layers".into())))
            .collect::<tensor::Result<Vec<_>>>()?;
        self.write(start, n.min(self.keep), rows)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> Value {
        serde_json::json!({
            "architectures": ["DFlashDraftModel"],
            "hidden_size": 64, "num_hidden_layers": 2, "num_attention_heads": 4,
            "num_key_value_heads": 2, "head_dim": 16, "vocab_size": 100, "intermediate_size": 96,
            "layer_types": ["sliding_attention", "full_attention"], "sliding_window": 8,
            "rope_parameters": {"rope_theta": 10000000, "rope_type": "default"},
            "dflash_config": {"block_size": 16, "mask_token_id": 99, "target_layer_ids": [0, 1]}
        })
    }

    #[test]
    fn parses_z_lab_configs() {
        let c = DFlashConfig::from_config(&config()).unwrap();
        assert_eq!(c.windows, [Some(8), None]);
        assert_eq!(c.causal, [true, false]);
        assert_eq!((c.mask_token, c.block_size, c.rope_theta), (99, 16, 1e7));
        assert_eq!(c.target_layer_ids, [0, 1]);
        assert_eq!(c.keep(), 7);
        assert_eq!(c.bytes_per_position(), 2 * 2 * 2 * 16 * 2);
    }

    #[test]
    fn refuses_other_drafters_and_broken_configs() {
        for (key, value) in [
            ("architectures", serde_json::json!(["DFlash2DraftModel"])),
            ("architectures", serde_json::json!(["Qwen3ForCausalLM"])),
            ("layer_types", serde_json::json!(["full_attention"])),
            ("sliding_window", Value::Null),
        ] {
            let mut c = config();
            c[key] = value;
            assert!(DFlashConfig::from_config(&c).is_err(), "{key}");
        }
        let mut c = config();
        c["dflash_config"]["target_layer_ids"] = serde_json::json!([]);
        assert!(DFlashConfig::from_config(&c).is_err());
    }

    /// Per-layer rows whose values are their positions (every element of position `p` is `p`).
    fn rows(at: u32, n: u32) -> Vec<(Array, Array)> {
        (0..2)
            .map(|_| {
                let data: Vec<f32> = (0..2)
                    .flat_map(|_| (at..at + n).flat_map(|p| std::iter::repeat_n(p as f32, 4)))
                    .collect();
                // f32: bf16 cannot hold every position (257 rounds to 256).
                let a = Array::from_f32(&data, &[1, 2, n as i32, 4]);
                (a.clone(), a)
            })
            .collect()
    }

    /// The positions a context holds, read back from its values.
    fn held(ctx: &DraftContext) -> Vec<u32> {
        if ctx.is_empty() {
            return Vec::new();
        }
        let (k, _) = ctx.layer(1, ctx.len()).unwrap();
        let k = tensor::slice(&k, &[0, 0, 0, 0], &[1, 1, ctx.len() as i32, 1]).unwrap();
        k.to_vec_f32().unwrap().iter().map(|&p| p as u32).collect()
    }

    #[test]
    fn a_context_keeps_its_window_and_repairs_itself() {
        tensor::use_device(tensor::Device::Cpu);
        let mut ctx = DraftContext::new(2, 7, 32);
        ctx.write(0, 5, rows(0, 5)).unwrap();
        assert_eq!(held(&ctx), [0, 1, 2, 3, 4]);
        for p in 5..300 {
            ctx.write(p, 1, rows(p, 1)).unwrap();
            assert!(ctx.len() <= 7 + SLACK);
            assert_eq!(*held(&ctx).last().unwrap(), p);
        }
        // Past the slack it compacted to the window and kept every value.
        let all = held(&ctx);
        assert_eq!(all, (300 - all.len() as u32..300).collect::<Vec<_>>());
        // A write below the end cuts there (a rolled-back step).
        ctx.write(290, 1, rows(290, 1)).unwrap();
        assert_eq!(ctx.span().1, 291);
        assert_eq!(*held(&ctx).last().unwrap(), 290);
        // A gap restarts.
        ctx.write(400, 2, rows(400, 2)).unwrap();
        assert_eq!(held(&ctx), [400, 401]);
        // Writes fill the slack before compacting.
        ctx.write(402, 7, rows(402, 7)).unwrap();
        assert_eq!(held(&ctx), (400..409).collect::<Vec<_>>());
    }

    #[test]
    fn split_and_append_move_positions_intact() {
        tensor::use_device(tensor::Device::Cpu);
        let mut ctx = DraftContext::new(2, 7, 32);
        ctx.write(10, 6, rows(10, 6)).unwrap();
        let tail = ctx.split_off(13).unwrap();
        assert_eq!(held(&ctx), [10, 11, 12]);
        assert_eq!(held(&tail), [13, 14, 15]);
        ctx.append(tail).unwrap();
        assert_eq!(held(&ctx), (10..16).collect::<Vec<_>>());
        // A tail that does not continue the context replaces it.
        let mut other = DraftContext::new(2, 7, 32);
        other.write(50, 2, rows(50, 2)).unwrap();
        let tail = other.split_off(0).unwrap();
        assert!(other.is_empty());
        ctx.append(tail).unwrap();
        assert_eq!(held(&ctx), [50, 51]);
        // Splitting past the end moves nothing.
        assert!(ctx.split_off(60).unwrap().is_empty());
    }

    #[test]
    fn masks_follow_the_window_and_the_block() {
        tensor::use_device(tensor::Device::Cpu);
        // Full attention, not causal: no mask.
        assert!(block_mask(None, false, 0, 5, 5, 3).is_none());
        // Sliding window 4 over context positions 0..5, block at 5..8, causal.
        let mask = block_mask(Some(4), true, 0, 5, 5, 3).unwrap();
        let m: Vec<f32> = tensor::astype(&mask, Dtype::Float32)
            .unwrap()
            .to_vec_f32()
            .unwrap();
        let row = |r: usize| {
            m[r * 8..(r + 1) * 8]
                .iter()
                .map(|&x| x as u8)
                .collect::<Vec<_>>()
        };
        assert_eq!(row(0), [0, 0, 1, 1, 1, 1, 0, 0]);
        assert_eq!(row(2), [0, 0, 0, 0, 1, 1, 1, 1]);
    }
}
