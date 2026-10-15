//! Family registry: `config.json` `model_type` (and, for wrapped configs, `text_config.model_type`)
//! to a family, plus the per-layer state layout and byte costs derived from the config. Runs
//! before any weights are read.

use super::LoadError;
use crate::engine::{LayerKind, ModelInfo};
use serde_json::Value;

/// Chat syntax of a family's template: tool calls, reasoning and message framing. The parsers
/// live in `mrun`; the registry names one per family.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SyntaxId {
    Hermes,
    QwenXml,
    Harmony,
}

/// What a layer's sequence state is.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StateKind {
    /// K/V per position; can be cut at any position.
    Positional,
    /// Recurrent state describing one position; restored only from checkpoints.
    PositionBound,
}

impl StateKind {
    pub fn of(kind: LayerKind) -> Self {
        match kind {
            LayerKind::FullAttention | LayerKind::SlidingAttention { .. } => Self::Positional,
            LayerKind::Linear => Self::PositionBound,
        }
    }
}

/// Layer kinds when `layer_types` is absent.
#[derive(Clone, Copy, Debug)]
enum DefaultLayers {
    AllFull,
    /// Linear unless `(i + 1) % full_attention_interval == 0`.
    LinearWithFullInterval,
    /// Sliding, full, sliding, full, ...
    AlternatingSliding,
}

#[derive(Debug)]
pub struct Family {
    /// Registry name (`ModelInfo.family`).
    pub name: &'static str,
    /// Curated alias shown in the unsupported-architecture message.
    pub alias: Option<&'static str>,
    /// Accepted `(model_type, text_config.model_type)` pairs.
    keys: &'static [(&'static str, Option<&'static str>)],
    pub syntax: SyntaxId,
    /// End tokens joined to the EOS union, by token text.
    pub eos_extra: &'static [&'static str],
    default_layers: DefaultLayers,
}

/// Registered families, in the order the unsupported-architecture message lists them.
static FAMILIES: [Family; 4] = [
    Family {
        name: "qwen3_5_moe",
        alias: Some("qwen3.6"),
        keys: &[
            ("qwen3_5_moe", None),
            ("qwen3_5_moe", Some("qwen3_5_moe_text")),
        ],
        syntax: SyntaxId::QwenXml,
        eos_extra: &[],
        default_layers: DefaultLayers::LinearWithFullInterval,
    },
    Family {
        name: "qwen3_moe",
        alias: Some("qwen3-coder"),
        keys: &[("qwen3_moe", None)],
        syntax: SyntaxId::QwenXml,
        eos_extra: &[],
        default_layers: DefaultLayers::AllFull,
    },
    Family {
        name: "gpt_oss",
        alias: Some("gpt-oss"),
        keys: &[("gpt_oss", None)],
        syntax: SyntaxId::Harmony,
        eos_extra: &["<|call|>", "<|return|>"],
        default_layers: DefaultLayers::AlternatingSliding,
    },
    Family {
        name: "qwen3",
        alias: None,
        keys: &[("qwen3", None)],
        syntax: SyntaxId::Hermes,
        eos_extra: &[],
        default_layers: DefaultLayers::AllFull,
    },
];

/// The family registered for this `(model_type, text_config.model_type)` pair.
pub fn lookup(
    model_type: &str,
    text_model_type: Option<&str>,
) -> Result<&'static Family, LoadError> {
    FAMILIES
        .iter()
        .find(|family| family.keys.contains(&(model_type, text_model_type)))
        .ok_or_else(|| LoadError::UnsupportedArchitecture {
            model_type: model_type.to_string(),
            text_model_type: text_model_type.map(str::to_string),
        })
}

/// Supported families with their aliases, e.g. `qwen3_moe (qwen3-coder), qwen3`.
pub fn supported() -> String {
    FAMILIES
        .iter()
        .map(|family| match family.alias {
            Some(alias) => format!("{} ({alias})", family.name),
            None => family.name.to_string(),
        })
        .collect::<Vec<_>>()
        .join(", ")
}

/// A family plus the state layout its `config.json` describes.
#[derive(Clone, Debug)]
pub struct FamilyConfig {
    pub family: &'static Family,
    pub layer_kinds: Vec<LayerKind>,
    pub kv_heads: u32,
    pub head_dim: u32,
    pub vocab: u32,
    pub max_position: u32,
    /// K and V bytes for one token across all attention layers (sliding ones keep full K/V).
    pub kv_bytes_per_token: u64,
    /// Recurrent bytes across all linear layers (one checkpoint); 0 without linear layers.
    pub recurrent_bytes: u64,
}

impl FamilyConfig {
    /// Resolves the family of a parsed `config.json` and reads its layout.
    pub fn from_config(config: &Value) -> Result<Self, LoadError> {
        let text_config = config.get("text_config").filter(|text| text.is_object());
        let model_type = config
            .get("model_type")
            .and_then(Value::as_str)
            .ok_or_else(|| invalid("model_type", "is missing"))?;
        let text_model_type = text_config.and_then(|text| text["model_type"].as_str());
        let family = lookup(model_type, text_model_type)?;
        let cfg = Fields(text_config.unwrap_or(config));

        let layers = cfg.u32("num_hidden_layers")?;
        let layer_kinds = layer_kinds(&cfg, family.default_layers, layers)?;
        let heads = cfg.u32("num_attention_heads")?;
        let kv_heads = cfg.u32("num_key_value_heads")?;
        let head_dim = match cfg.opt_u32("head_dim")? {
            Some(dim) => dim,
            None => cfg.u32("hidden_size")? / heads.max(1),
        };
        let dtype = dtype_bytes(config, &cfg)?;
        let attention = layer_kinds
            .iter()
            .filter(|&&kind| StateKind::of(kind) == StateKind::Positional)
            .count() as u64;
        let linear = layer_kinds.len() as u64 - attention;
        let recurrent_bytes = if linear > 0 {
            let (k_heads, v_heads) = (
                cfg.u32("linear_num_key_heads")?,
                cfg.u32("linear_num_value_heads")?,
            );
            let (k_dim, v_dim) = (
                cfg.u32("linear_key_head_dim")?,
                cfg.u32("linear_value_head_dim")?,
            );
            let kernel = cfg.u32("linear_conv_kernel_dim")?;
            let conv_dim = u64::from(2 * k_heads * k_dim + v_heads * v_dim);
            // f32 recurrent state plus the convolution tail in the activation dtype.
            let state = u64::from(v_heads) * u64::from(v_dim) * u64::from(k_dim) * 4;
            let conv = u64::from(kernel.saturating_sub(1)) * conv_dim * dtype;
            linear * (state + conv)
        } else {
            0
        };
        Ok(Self {
            family,
            layer_kinds,
            kv_heads,
            head_dim,
            vocab: cfg.u32("vocab_size")?,
            max_position: cfg.u32("max_position_embeddings")?,
            kv_bytes_per_token: attention * 2 * u64::from(kv_heads) * u64::from(head_dim) * dtype,
            recurrent_bytes,
        })
    }

    pub fn state_kinds(&self) -> impl Iterator<Item = StateKind> + '_ {
        self.layer_kinds.iter().map(|&kind| StateKind::of(kind))
    }

    pub fn model_info(&self, identity: String, eos_tokens: Vec<u32>) -> ModelInfo {
        ModelInfo {
            identity,
            family: self.family.name.to_string(),
            layer_kinds: self.layer_kinds.clone(),
            kv_heads: self.kv_heads,
            head_dim: self.head_dim,
            vocab: self.vocab,
            max_position: self.max_position,
            kv_bytes_per_token: self.kv_bytes_per_token,
            recurrent_bytes: self.recurrent_bytes,
            eos_tokens,
        }
    }
}

fn invalid(key: &str, reason: impl Into<String>) -> LoadError {
    LoadError::InvalidConfig {
        key: key.to_string(),
        reason: reason.into(),
    }
}

/// Typed access to the (text) config object.
struct Fields<'a>(&'a Value);

impl Fields<'_> {
    fn opt_u32(&self, key: &str) -> Result<Option<u32>, LoadError> {
        match self.0.get(key) {
            None | Some(Value::Null) => Ok(None),
            Some(value) => value
                .as_u64()
                .and_then(|n| u32::try_from(n).ok())
                .map(Some)
                .ok_or_else(|| {
                    invalid(key, format!("must be a non-negative integer, got {value}"))
                }),
        }
    }

    fn u32(&self, key: &str) -> Result<u32, LoadError> {
        self.opt_u32(key)?.ok_or_else(|| invalid(key, "is missing"))
    }
}

fn layer_kinds(
    cfg: &Fields,
    default: DefaultLayers,
    layers: u32,
) -> Result<Vec<LayerKind>, LoadError> {
    let sliding = || {
        cfg.u32("sliding_window")
            .map(|window| LayerKind::SlidingAttention { window })
    };
    if let Some(types) = cfg.0.get("layer_types").and_then(Value::as_array) {
        if types.len() != layers as usize {
            return Err(invalid(
                "layer_types",
                format!("has {} entries for {layers} layers", types.len()),
            ));
        }
        return types
            .iter()
            .map(|kind| match kind.as_str() {
                Some("full_attention") => Ok(LayerKind::FullAttention),
                Some("sliding_attention") => sliding(),
                Some("linear_attention") => Ok(LayerKind::Linear),
                _ => Err(invalid("layer_types", format!("has unknown entry {kind}"))),
            })
            .collect();
    }
    match default {
        DefaultLayers::AllFull => Ok(vec![LayerKind::FullAttention; layers as usize]),
        DefaultLayers::LinearWithFullInterval => {
            let interval = cfg.opt_u32("full_attention_interval")?.unwrap_or(4);
            if interval == 0 {
                return Err(invalid("full_attention_interval", "must be at least 1"));
            }
            Ok((1..=layers)
                .map(|i| {
                    if i % interval == 0 {
                        LayerKind::FullAttention
                    } else {
                        LayerKind::Linear
                    }
                })
                .collect())
        }
        DefaultLayers::AlternatingSliding => {
            let window = sliding()?;
            Ok((0..layers)
                .map(|i| {
                    if i % 2 == 0 {
                        window
                    } else {
                        LayerKind::FullAttention
                    }
                })
                .collect())
        }
    }
}

/// Bytes per element of the activation dtype (the K/V and convolution state dtype).
fn dtype_bytes(root: &Value, cfg: &Fields) -> Result<u64, LoadError> {
    let name = ["torch_dtype", "dtype"]
        .iter()
        .find_map(|key| cfg.0[key].as_str().or_else(|| root[key].as_str()));
    match name {
        None | Some("bfloat16" | "float16") => Ok(2),
        Some("float32") => Ok(4),
        Some(other) => Err(invalid("torch_dtype", format!("{other} is not supported"))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    const KIB: u64 = 1024;

    fn qwen3_8b() -> Value {
        json!({
            "model_type": "qwen3", "num_hidden_layers": 36, "num_attention_heads": 32,
            "num_key_value_heads": 8, "head_dim": 128, "hidden_size": 4096,
            "vocab_size": 151936, "max_position_embeddings": 40960, "torch_dtype": "bfloat16"
        })
    }

    fn qwen3_coder() -> Value {
        json!({
            "model_type": "qwen3_moe", "num_hidden_layers": 48, "num_attention_heads": 32,
            "num_key_value_heads": 4, "head_dim": 128, "hidden_size": 2048,
            "vocab_size": 151936, "max_position_embeddings": 262144, "torch_dtype": "bfloat16",
            "num_experts": 128, "num_experts_per_tok": 8
        })
    }

    fn qwen3_6_text() -> Value {
        json!({
            "model_type": "qwen3_5_moe_text", "num_hidden_layers": 40,
            "num_attention_heads": 16, "num_key_value_heads": 2, "head_dim": 256,
            "hidden_size": 2048, "vocab_size": 248320, "max_position_embeddings": 262144,
            "full_attention_interval": 4, "linear_num_key_heads": 16,
            "linear_num_value_heads": 32, "linear_key_head_dim": 128,
            "linear_value_head_dim": 128, "linear_conv_kernel_dim": 4, "dtype": "bfloat16"
        })
    }

    fn gpt_oss_20b() -> Value {
        let layer_types = ["sliding_attention", "full_attention"].repeat(12);
        json!({
            "model_type": "gpt_oss", "num_hidden_layers": 24, "num_attention_heads": 64,
            "num_key_value_heads": 8, "head_dim": 64, "hidden_size": 2880,
            "vocab_size": 201088, "max_position_embeddings": 131072, "sliding_window": 128,
            "layer_types": layer_types
        })
    }

    fn resolve(config: Value) -> FamilyConfig {
        FamilyConfig::from_config(&config).unwrap()
    }

    #[test]
    fn dense_and_moe_qwen3_are_all_full_attention() {
        let qwen3 = resolve(qwen3_8b());
        assert_eq!(qwen3.family.name, "qwen3");
        assert_eq!(qwen3.layer_kinds, vec![LayerKind::FullAttention; 36]);
        assert_eq!(qwen3.kv_bytes_per_token, 144 * KIB);
        assert_eq!((qwen3.recurrent_bytes, qwen3.max_position), (0, 40960));
        assert_eq!(qwen3.family.syntax, crate::SyntaxId::Hermes);

        let coder = resolve(qwen3_coder());
        assert_eq!(coder.family.name, "qwen3_moe");
        assert_eq!(coder.kv_bytes_per_token, 96 * KIB);
        assert_eq!(coder.family.syntax, SyntaxId::QwenXml);
        assert!(
            coder
                .state_kinds()
                .all(|kind| kind == StateKind::Positional)
        );
    }

    #[test]
    fn hybrid_layout_follows_the_full_attention_interval_wrapped_or_not() {
        let wrapped = resolve(json!({"model_type": "qwen3_5_moe", "text_config": qwen3_6_text()}));
        let mut root = qwen3_6_text();
        root["model_type"] = json!("qwen3_5_moe");
        let root = resolve(root);
        for cfg in [&wrapped, &root] {
            assert_eq!(cfg.family.name, "qwen3_5_moe");
            let full: Vec<_> = (0..40)
                .filter(|&i| cfg.layer_kinds[i] == LayerKind::FullAttention)
                .collect();
            assert_eq!(full, [3, 7, 11, 15, 19, 23, 27, 31, 35, 39]);
            assert_eq!(cfg.kv_bytes_per_token, 20 * KIB);
            // 30 GDN layers × (32·128·128·4 + 3·8192·2) ≈ 61.4 MiB.
            assert_eq!(cfg.recurrent_bytes, 64_389_120);
            assert_eq!(
                cfg.state_kinds()
                    .filter(|&kind| kind == StateKind::PositionBound)
                    .count(),
                30
            );
        }
        assert_eq!(wrapped.family.syntax, SyntaxId::QwenXml);
        let info = wrapped.model_info("id".into(), vec![1]);
        assert!(info.is_hybrid());
        assert_eq!(info.family, "qwen3_5_moe");
    }

    #[test]
    fn explicit_layer_types_override_the_interval() {
        let mut text = qwen3_6_text();
        text["num_hidden_layers"] = json!(2);
        text["layer_types"] = json!(["full_attention", "linear_attention"]);
        let cfg = resolve(json!({"model_type": "qwen3_5_moe", "text_config": text}));
        assert_eq!(
            cfg.layer_kinds,
            [LayerKind::FullAttention, LayerKind::Linear]
        );
    }

    #[test]
    fn gpt_oss_alternates_sliding_and_full_layers_with_full_kv() {
        let explicit = resolve(gpt_oss_20b());
        let mut implicit = gpt_oss_20b();
        implicit.as_object_mut().unwrap().remove("layer_types");
        let implicit = resolve(implicit);
        for cfg in [&explicit, &implicit] {
            assert_eq!(
                cfg.layer_kinds[0],
                LayerKind::SlidingAttention { window: 128 }
            );
            assert_eq!(cfg.layer_kinds[1], LayerKind::FullAttention);
            assert_eq!(cfg.layer_kinds.len(), 24);
            assert_eq!(cfg.kv_bytes_per_token, 48 * KIB);
            assert_eq!(cfg.recurrent_bytes, 0);
        }
        assert_eq!(explicit.family.syntax, SyntaxId::Harmony);
        assert_eq!(explicit.family.eos_extra, ["<|call|>", "<|return|>"]);
    }

    #[test]
    fn unknown_and_unregistered_architectures_are_refused_with_the_supported_list() {
        let devstral = json!({
            "model_type": "mistral3",
            "text_config": {"model_type": "ministral3", "num_hidden_layers": 40}
        });
        for (config, arch) in [
            (json!({"model_type": "llama4"}), "llama4"),
            (devstral, "mistral3+ministral3"),
            (
                json!({"model_type": "qwen3_5_moe", "text_config": {"model_type": "other"}}),
                "qwen3_5_moe+other",
            ),
        ] {
            let error = FamilyConfig::from_config(&config).unwrap_err();
            assert!(matches!(error, LoadError::UnsupportedArchitecture { .. }));
            assert_eq!(
                error.report("org/model"),
                format!(
                    "unsupported architecture '{arch}' in org/model\n  supported families: \
                     qwen3_5_moe (qwen3.6), qwen3_moe (qwen3-coder), gpt_oss (gpt-oss), qwen3"
                )
            );
        }
    }

    #[test]
    fn malformed_configs_name_the_key() {
        let mut missing = qwen3_8b();
        missing
            .as_object_mut()
            .unwrap()
            .remove("num_key_value_heads");
        let mut wrong_types = gpt_oss_20b();
        wrong_types["layer_types"] = json!(["full_attention"]);
        let mut no_type = qwen3_8b();
        no_type.as_object_mut().unwrap().remove("model_type");
        for (config, key) in [
            (missing, "num_key_value_heads"),
            (wrong_types, "layer_types"),
            (no_type, "model_type"),
        ] {
            match FamilyConfig::from_config(&config) {
                Err(LoadError::InvalidConfig { key: found, .. }) => assert_eq!(found, key),
                other => panic!("{key}: {other:?}"),
            }
        }
    }
}
