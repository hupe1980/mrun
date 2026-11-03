//! Loading a checkpoint for the MLX engine: `*.safetensors` into a weight map, the quantization
//! settings from `config.json` (with per-path overrides), and the quantized or dense linear and
//! embedding layers the family modules compose.

use std::collections::HashMap;
use std::fs;
use std::path::Path;

use serde_json::Value;

use crate::models::LoadError;
use crate::tensor::{self, Array, Dtype};

/// Bit widths of the affine quantization mrun runs.
pub const BITS: [i32; 2] = [4, 8];
/// Group sizes of the affine quantization mrun runs.
pub const GROUP_SIZES: [i32; 3] = [32, 64, 128];

/// Affine quantization of one layer (scales + biases; the only mode mrun runs).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct QParams {
    pub group_size: i32,
    pub bits: i32,
}

impl QParams {
    /// Refuses what mrun does not run: bit widths other than 4 or 8, group sizes other than 32,
    /// 64 or 128.
    fn checked(self, path: &str) -> Result<Self, LoadError> {
        if !BITS.contains(&self.bits) || !GROUP_SIZES.contains(&self.group_size) {
            return Err(invalid(
                "quantization",
                format!(
                    "{path}: {} bits with group size {} is not supported (affine 4 or 8 bits, \
                     groups of 32, 64 or 128)",
                    self.bits, self.group_size
                ),
            ));
        }
        Ok(self)
    }
}

/// `config.json` `quantization`: per-path settings (`false` = not quantized). The top-level
/// group size and bits only fill in what a path entry leaves out: mlx-lm infers every unnamed
/// path from its shapes ([`Quantization::resolve`]).
#[derive(Clone, Debug, Default)]
pub struct Quantization {
    overrides: HashMap<String, Option<QParams>>,
}

impl Quantization {
    /// Reads the settings; a `mode` other than `affine` (mxfp4, nvfp4, mxfp8, ...) is refused by
    /// name here, before any tensor is read.
    pub fn from_config(config: &Value) -> Result<Self, LoadError> {
        let Some(q) = config
            .get("quantization")
            .or_else(|| config.get("quantization_config"))
            .and_then(Value::as_object)
        else {
            return Ok(Self::default());
        };
        let params = |path: &str, v: &Value, base: Option<QParams>| {
            if let Some(mode) = v.get("mode").and_then(Value::as_str)
                && mode != "affine"
            {
                return Err(invalid(
                    "quantization",
                    format!("{path}: mode {mode} is not supported (only affine)"),
                ));
            }
            let group_size = v["group_size"]
                .as_i64()
                .map(|g| g as i32)
                .or(base.map(|b| b.group_size));
            let bits = v["bits"]
                .as_i64()
                .map(|b| b as i32)
                .or(base.map(|b| b.bits));
            Ok(match (group_size, bits) {
                (Some(group_size), Some(bits)) => Some(QParams { group_size, bits }),
                _ => None,
            })
        };
        let root = Value::Object(q.clone());
        let default = params("quantization", &root, None)?;
        let mut overrides = HashMap::new();
        for (key, value) in q {
            match value {
                Value::Object(_) => {
                    if let Some(named) = params(key, value, default)? {
                        overrides.insert(key.clone(), Some(named.checked(key)?));
                    }
                }
                Value::Bool(false) => {
                    overrides.insert(key.clone(), None);
                }
                _ => {}
            }
        }
        Ok(Self { overrides })
    }

    /// Parameters of the quantized layer at `path` (without the `.weight` suffix) whose
    /// unquantized input width is `in_dims`, as mlx-lm 0.32.0 `load_model` decides them: a path
    /// named in the config wins (`false` there means not quantized: an error, as the checkpoint
    /// has scales); any other path is inferred from the packed shapes (`infer_quant_config`),
    /// since the default can be wrong (Qwen3.6 lists its 8-bit routers only under
    /// `quantization_config`, which mlx-lm ignores). Only shapes are read, never tensor data.
    pub fn resolve(
        &self,
        path: &str,
        weight: &Array,
        scales: &Array,
        in_dims: i32,
    ) -> Result<QParams, LoadError> {
        if let Some(named) = self.overrides.get(path) {
            return named.ok_or_else(|| {
                invalid(
                    "quantization",
                    format!("{path} is quantized but the config marks it unquantized"),
                )
            });
        }
        infer(path, weight, scales, in_dims)
    }
}

/// mlx-lm `infer_quant_config`: group size and bits from the packed weight and scales, both of
/// which must divide exactly. Microscaling formats keep their scales as `uint8` and are refused.
fn infer(path: &str, weight: &Array, scales: &Array, in_dims: i32) -> Result<QParams, LoadError> {
    let bad = |what: String| invalid("quantization", format!("{path}: {what}"));
    if scales.dtype() == Dtype::Uint8 {
        return Err(bad(
            "uint8 scales (mxfp4, nvfp4 or mxfp8) are not supported (only affine)".into(),
        ));
    }
    let groups = scales.dim(-1);
    if in_dims <= 0 || groups <= 0 || in_dims % groups != 0 {
        return Err(bad(format!(
            "{groups} scale groups do not divide the input width {in_dims}"
        )));
    }
    let packed = i64::from(weight.dim(-1)) * 32;
    if packed % i64::from(in_dims) != 0 {
        return Err(bad(format!(
            "{} packed words do not hold whole {in_dims}-wide rows",
            weight.dim(-1)
        )));
    }
    QParams {
        group_size: in_dims / groups,
        bits: (packed / i64::from(in_dims)) as i32,
    }
    .checked(path)
}

/// All tensors of a checkpoint, taken out by name as the model is built.
#[derive(Debug, Default)]
pub struct Weights {
    tensors: HashMap<String, Array>,
}

/// Whether no model family reads tensor `name`: the vision tower (until image input, 009), MTP
/// heads and rotary frequency tables (mlx-lm's `sanitize` drops them too). Such tensors are not
/// loaded at all: nuvec reads unaligned files (every mlx-community checkpoint) at load, so a
/// tensor dropped after loading would still be read (0.83 GiB on Qwen3.6-35B-A3B).
pub fn unused_tensor(name: &str) -> bool {
    name.starts_with("vision_tower")
        || name.starts_with("model.visual")
        || name.contains("mtp.")
        || name.contains("rotary_emb.inv_freq")
}

impl Weights {
    /// Loads the tensors of every `*.safetensors` file in `dir` that `keep` accepts. MLX reads
    /// their data on first evaluation; nuvec maps aligned files and reads unaligned ones (every
    /// mlx-community checkpoint) here, in parallel.
    pub fn load(dir: &Path, keep: impl Fn(&str) -> bool) -> Result<Self, LoadError> {
        let mut files: Vec<_> = fs::read_dir(dir)
            .map_err(|e| invalid("weights", format!("cannot read {}: {e}", dir.display())))?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|e| e == "safetensors"))
            .collect();
        files.sort();
        if files.is_empty() {
            return Err(invalid("weights", "no *.safetensors files"));
        }
        let mut tensors = HashMap::new();
        for file in files {
            let loaded = tensor::load_safetensors_filtered(&file, &keep)
                .map_err(|e| invalid("weights", format!("{}: {e}", file.display())))?;
            tensors.extend(loaded);
        }
        Ok(Self { tensors })
    }

    pub fn contains(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }

    pub fn take(&mut self, name: &str) -> Result<Array, LoadError> {
        self.tensors
            .remove(name)
            .ok_or_else(|| invalid("weights", format!("missing tensor {name}")))
    }

    pub fn take_opt(&mut self, name: &str) -> Option<Array> {
        self.tensors.remove(name)
    }

    pub fn insert(&mut self, name: String, tensor: Array) {
        self.tensors.insert(name, tensor);
    }

    /// Renames every tensor through `rename`; `None` drops it.
    pub fn rename(&mut self, rename: impl Fn(&str) -> Option<String>) {
        self.tensors = std::mem::take(&mut self.tensors)
            .into_iter()
            .filter_map(|(name, tensor)| rename(&name).map(|name| (name, tensor)))
            .collect();
    }

    /// Whether any tensor satisfies `test`.
    pub fn any(&self, test: impl Fn(&str, &Array) -> bool) -> bool {
        self.tensors.iter().any(|(name, tensor)| test(name, tensor))
    }

    /// Replaces each tensor whose name satisfies `select` with `map(tensor)`.
    pub fn map(
        &mut self,
        select: impl Fn(&str, &Array) -> bool,
        map: impl Fn(&Array) -> tensor::Result<Array>,
    ) -> Result<(), LoadError> {
        for (name, tensor) in &mut self.tensors {
            if select(name, tensor) {
                *tensor = map(tensor).map_err(|e| invalid("weights", format!("{name}: {e}")))?;
            }
        }
        Ok(())
    }

    /// Drops tensors the family does not use (e.g. vision towers); returns how many.
    pub fn retain(&mut self, keep: impl Fn(&str) -> bool) -> usize {
        let before = self.tensors.len();
        self.tensors.retain(|name, _| keep(name));
        before - self.tensors.len()
    }

    /// Names left after the model took its tensors (reported as unused).
    pub fn remaining(&self) -> Vec<&str> {
        let mut names: Vec<&str> = self.tensors.keys().map(String::as_str).collect();
        names.sort_unstable();
        names
    }
}

/// `y = x Wᵀ (+ b)`, dense or quantized.
#[derive(Debug)]
pub enum Linear {
    Dense {
        weight: Array,
        bias: Option<Array>,
    },
    Quantized {
        w: tensor::Quantized,
        bias: Option<Array>,
    },
}

impl Linear {
    /// Takes `<path>.weight` (+ `scales`, `biases`, `bias`) from `weights`; `in_dims` is the
    /// unquantized input width.
    pub fn load(
        weights: &mut Weights,
        path: &str,
        quant: &Quantization,
        in_dims: i32,
    ) -> Result<Self, LoadError> {
        let weight = weights.take(&format!("{path}.weight"))?;
        let bias = weights.take_opt(&format!("{path}.bias"));
        match weights.take_opt(&format!("{path}.scales")) {
            Some(scales) => {
                let q = quant.resolve(path, &weight, &scales, in_dims)?;
                let biases = weights.take_opt(&format!("{path}.biases"));
                let w = tensor::Quantized::new(weight, scales, biases, q.group_size, q.bits)
                    .map_err(|e| invalid("weights", format!("{path}: {e}")))?;
                Ok(Self::Quantized { w, bias })
            }
            None => Ok(Self::Dense { weight, bias }),
        }
    }

    pub fn forward(&self, x: &Array) -> tensor::Result<Array> {
        let (y, bias) = match self {
            Self::Dense { weight, bias } => {
                let wt = tensor::transpose(weight, &[1, 0])?;
                (tensor::matmul(x, &wt)?, bias)
            }
            Self::Quantized { w, bias } => (tensor::quantized_matmul(x, w)?, bias),
        };
        match bias {
            Some(b) => tensor::add(&y, b),
            None => Ok(y),
        }
    }
}

/// Token embedding table, dense or quantized; also usable as the tied output projection.
#[derive(Debug)]
pub enum Embedding {
    Dense(Array),
    Quantized(tensor::Quantized),
}

impl Embedding {
    pub fn load(
        weights: &mut Weights,
        path: &str,
        quant: &Quantization,
        hidden: i32,
    ) -> Result<Self, LoadError> {
        match Linear::load(weights, path, quant, hidden)? {
            Linear::Dense { weight, .. } => Ok(Self::Dense(weight)),
            Linear::Quantized { w, .. } => Ok(Self::Quantized(w)),
        }
    }

    /// Rows for `ids` (`[1, L]` integer array) → `[1, L, hidden]`.
    pub fn forward(&self, ids: &Array) -> tensor::Result<Array> {
        match self {
            Self::Dense(weight) => tensor::take(weight, ids, 0),
            Self::Quantized(w) => tensor::dequantize(&w.take(ids)?),
        }
    }

    /// `x Eᵀ` (tied output head).
    pub fn as_linear(&self, x: &Array) -> tensor::Result<Array> {
        match self {
            Self::Dense(weight) => tensor::matmul(x, &tensor::transpose(weight, &[1, 0])?),
            Self::Quantized(w) => tensor::quantized_matmul(x, w),
        }
    }
}

/// RMSNorm weight with its epsilon.
#[derive(Debug)]
pub struct RmsNorm {
    pub weight: Array,
    pub eps: f32,
}

impl RmsNorm {
    pub fn load(weights: &mut Weights, path: &str, eps: f32) -> Result<Self, LoadError> {
        Ok(Self {
            weight: weights.take(&format!("{path}.weight"))?,
            eps,
        })
    }

    pub fn forward(&self, x: &Array) -> tensor::Result<Array> {
        tensor::fast::rms_norm(x, Some(&self.weight), self.eps)
    }
}

/// The weights estimate of the budget: what loading `dir` without the [`unused_tensor`]s will
/// hold, as the backend reports it from the headers ([`tensor::weights_footprint`]), else the
/// bytes of all `*.safetensors` files. Symlinks are followed: Hugging Face snapshots link to
/// blobs.
pub fn weights_bytes(dir: &Path) -> u64 {
    tensor::weights_footprint(dir, |name| !unused_tensor(name)).unwrap_or_else(|| files_bytes(dir))
}

/// Bytes of all `*.safetensors` files in `dir`.
fn files_bytes(dir: &Path) -> u64 {
    fs::read_dir(dir)
        .map(|entries| {
            entries
                .filter_map(Result::ok)
                .map(|e| e.path())
                .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
                .filter_map(|p| fs::metadata(p).ok())
                .map(|m| m.len())
                .sum()
        })
        .unwrap_or(0)
}

pub(crate) fn invalid(key: &str, reason: impl Into<String>) -> LoadError {
    LoadError::InvalidConfig {
        key: key.to_string(),
        reason: reason.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unused_tensors_are_the_ones_no_family_reads() {
        for name in [
            "vision_tower.blocks.0.attn.qkv.weight",
            "model.visual.patch_embed.proj.weight",
            "mtp.layers.0.mlp.gate.weight",
            "model.layers.0.self_attn.rotary_emb.inv_freq",
        ] {
            assert!(unused_tensor(name), "{name}");
        }
        for name in [
            "language_model.model.layers.0.self_attn.q_proj.weight",
            "model.embed_tokens.weight",
            "lm_head.weight",
        ] {
            assert!(!unused_tensor(name), "{name}");
        }
    }
    use serde_json::json;

    #[test]
    fn named_paths_win_and_others_are_inferred_from_shapes() {
        crate::tensor::use_device(crate::tensor::Device::Cpu);
        let q = Quantization::from_config(&json!({
            "quantization": {
                "group_size": 64, "bits": 4,
                "model.layers.0.mlp.gate": {"group_size": 32, "bits": 8},
                "model.layers.1.mlp.gate": false
            }
        }))
        .unwrap();
        // A 16 × 128 layer: 4-bit packs 16 words per row, 8-bit 32.
        let packed = |words: i32| Array::from_u32(&vec![0; 16 * words as usize], &[16, words]);
        let scales = |groups: i32| Array::from_f32(&vec![1.0; 16 * groups as usize], &[16, groups]);
        let named = q
            .resolve("model.layers.0.mlp.gate", &packed(32), &scales(4), 128)
            .unwrap();
        assert_eq!((named.bits, named.group_size), (8, 32));
        // Not named: inferred even where the default (4 bits, 64) disagrees.
        let inferred = q
            .resolve("model.layers.2.mlp.gate", &packed(32), &scales(2), 128)
            .unwrap();
        assert_eq!((inferred.bits, inferred.group_size), (8, 64));
        let default = q.resolve("x", &packed(16), &scales(2), 128).unwrap();
        assert_eq!((default.bits, default.group_size), (4, 64));
        assert!(
            q.resolve("model.layers.1.mlp.gate", &packed(16), &scales(2), 128)
                .is_err()
        );
    }

    #[test]
    fn only_affine_4_and_8_bits_in_groups_of_32_to_128_load() {
        crate::tensor::use_device(crate::tensor::Device::Cpu);
        let q = Quantization::default();
        let packed = |words: i32| Array::from_u32(&vec![0; 16 * words as usize], &[16, words]);
        let scales = |groups: i32| Array::from_f32(&vec![1.0; 16 * groups as usize], &[16, groups]);
        // 3 bits, group 16 (8 groups), and a width the packing does not divide.
        assert!(q.resolve("a", &packed(12), &scales(2), 128).is_err());
        assert!(q.resolve("b", &packed(16), &scales(8), 128).is_err());
        assert!(q.resolve("c", &packed(15), &scales(2), 100).is_err());
        // Microscaling: uint8 scales.
        let e8 = Array::from_u8(&[0; 16 * 4], &[16, 4]);
        assert!(q.resolve("d", &packed(16), &e8, 128).is_err());
        // Named modes and settings are refused from the config alone.
        for config in [
            json!({"quantization": {"group_size": 32, "bits": 4, "mode": "mxfp4"}}),
            json!({"quantization": {"group_size": 64, "bits": 4, "x": {"mode": "nvfp4"}}}),
            json!({"quantization": {"group_size": 64, "bits": 4, "x": {"bits": 2}}}),
        ] {
            match Quantization::from_config(&config) {
                Err(LoadError::InvalidConfig { key, .. }) => assert_eq!(key, "quantization"),
                other => panic!("{config}: {other:?}"),
            }
        }
    }

    /// A `.safetensors` file whose data starts at an odd offset, so no f32 tensor is aligned
    /// (as MLX writes them): `(name, f32 count)` in file order.
    fn misaligned(path: &Path, tensors: &[(&str, usize)]) {
        let mut header = serde_json::Map::new();
        let mut at = 0;
        for &(name, n) in tensors {
            header.insert(
                name.into(),
                json!({"dtype": "F32", "shape": [n], "data_offsets": [at, at + 4 * n]}),
            );
            at += 4 * n;
        }
        let mut text = Value::Object(header).to_string();
        while (8 + text.len()) % 4 != 1 {
            text.push(' ');
        }
        let mut bytes = (text.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(text.as_bytes());
        bytes.resize(bytes.len() + at, 0);
        fs::write(path, bytes).unwrap();
    }

    #[test]
    fn the_weights_estimate_counts_what_the_load_keeps() {
        let dir = tempfile::tempdir().unwrap();
        let w = Array::from_f32(&[1.0; 1024], &[1024]);
        tensor::save_safetensors(&dir.path().join("a.safetensors"), &[("model.w", &w)]).unwrap();
        misaligned(
            &dir.path().join("b.safetensors"),
            &[("model.x", 1024), ("mtp.layers.0.w", 64 * 1024)],
        );
        let files = files_bytes(dir.path());
        let estimate = weights_bytes(dir.path());
        // At most one page of rounding per file.
        assert!(estimate <= files + 2 * 16_384, "{estimate} vs {files}");
        if tensor::weights_footprint(dir.path(), |_| true).is_some() {
            // The dropped 256 KiB `mtp.` tensor of the misaligned file is not read.
            assert!(estimate + 200 * 1024 < files, "{estimate} vs {files}");
        } else {
            assert_eq!(estimate, files);
        }
    }
}
