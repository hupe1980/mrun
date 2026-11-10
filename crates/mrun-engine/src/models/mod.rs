//! Model families: the registry from `config.json` to a family, and (with `engine`) the graphs.

#[cfg(feature = "engine")]
pub(crate) mod activations;
#[cfg(feature = "engine")]
mod common;
#[cfg(feature = "engine")]
pub(crate) mod gated_delta;
#[cfg(feature = "engine")]
mod moe;
#[cfg(feature = "engine")]
pub mod qwen3;
#[cfg(feature = "engine")]
pub mod qwen3_5_moe;
#[cfg(feature = "engine")]
pub mod qwen3_moe;
pub mod registry;

#[cfg(feature = "engine")]
#[doc(hidden)]
pub use common::{SCORE_BLOCK_BYTES_DEFAULT, set_score_block_bytes};
pub use registry::{Family, FamilyConfig, SyntaxId};

#[cfg(feature = "engine")]
pub use graph::{Model, build};

#[cfg(feature = "engine")]
mod graph {
    use super::{LoadError, Part};
    use crate::engine::LayerKind;
    use crate::load::Weights;
    use crate::state::LayerState;
    use crate::tensor::{self, Array, Dtype};
    use serde_json::Value;

    /// A family's graph, driven layer by layer by the engine (at most two layers in flight
    /// during prefill, so cancellation is checked after every layer).
    pub trait Model {
        /// Kind of each decoder layer (`Kv` state for attention, `Recurrent` for linear).
        fn layer_kinds(&self) -> &[LayerKind];
        fn layers(&self) -> usize {
            self.layer_kinds().len()
        }
        /// dtype the family stores attention state in (drives `kv_bytes_per_token`).
        fn state_dtype(&self) -> Dtype;
        /// Bytes of the recurrent state of all linear layers (one position).
        fn recurrent_bytes(&self) -> u64 {
            0
        }
        /// Token ids `[1, L]` → hidden states `[1, L, hidden]`.
        fn embed(&self, ids: &Array) -> tensor::Result<Array>;
        /// Decoder layer `i` over `x` at absolute position `offset`, updating its state.
        fn layer(
            &self,
            i: usize,
            x: &Array,
            state: &mut LayerState,
            offset: i32,
        ) -> tensor::Result<Array>;
        /// [`Model::layer`] with linear layers' recurrence run in segments ending at the
        /// chunk-relative positions `splits` (003 SC-009), pushing the recurrent state after each
        /// split to `captured`; attention layers ignore the splits. Families without linear
        /// layers have no position-bound state to capture.
        fn layer_split(
            &self,
            i: usize,
            x: &Array,
            state: &mut LayerState,
            offset: i32,
            splits: &[i32],
            captured: &mut Vec<crate::state::Recurrent>,
        ) -> tensor::Result<Array> {
            let _ = (splits, captured);
            self.layer(i, x, state, offset)
        }
        /// [`Model::layer`] for a verify pass (006 FR-001): every row computed as the decode step
        /// at its position computes it, the state after each position but the last pushed to
        /// `captured` (linear layers; attention layers push nothing).
        fn layer_verify(
            &self,
            i: usize,
            x: &Array,
            state: &mut LayerState,
            offset: i32,
            captured: &mut Vec<crate::state::Recurrent>,
        ) -> tensor::Result<Array> {
            let splits: Vec<i32> = (1..x.dim(1)).collect();
            self.layer_split(i, x, state, offset, &splits, captured)
        }
        /// Hidden states `[1, L, hidden]` → logits of the last position `[1, vocab]`.
        fn logits(&self, x: &Array) -> tensor::Result<Array>;
        /// Hidden states `[1, L, hidden]` → logits of every position `[L, vocab]`.
        fn logits_rows(&self, x: &Array) -> tensor::Result<Array>;
        /// Width of the hidden state between layers.
        fn hidden(&self) -> i32;
        /// Analytic scratch of one chunk of `len` tokens at `depth` (activations, MoE routing,
        /// unfused attention scores; D029); the engine maximizes it over the chunk schedule.
        fn chunk_scratch(&self, len: u32, depth: u32) -> u64;
    }

    /// Builds the graph of `family` from `config.json` (`text` is its text config) and the
    /// checkpoint's weights, `part` of its layers only.
    pub fn build(
        family: &str,
        root: &Value,
        text: &Value,
        weights: Weights,
        part: Part,
    ) -> Result<Box<dyn Model>, LoadError> {
        match family {
            "qwen3" => Ok(Box::new(super::qwen3::Qwen3::load(text, weights, part)?)),
            "qwen3_moe" => Ok(Box::new(super::qwen3_moe::Qwen3Moe::load(
                text, weights, part,
            )?)),
            // `qwen3_5` is the same graph with a dense MLP (no experts in its config).
            "qwen3_5_moe" | "qwen3_5" => Ok(Box::new(super::qwen3_5_moe::Qwen3_5Moe::load(
                root, text, weights, part,
            )?)),
            other => Err(LoadError::InvalidConfig {
                key: "model_type".into(),
                reason: format!("'{other}' is not a family this build runs"),
            }),
        }
    }
}

/// The layers one engine runs (026): all of them, or a contiguous range on one Mac of a pool.
/// The part with the head also embeds the tokens and turns the last hidden state into logits.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Part {
    pub first: usize,
    /// One past the last layer.
    pub end: usize,
    pub head: bool,
}

impl Part {
    pub fn whole(layers: usize) -> Self {
        Self {
            first: 0,
            end: layers,
            head: true,
        }
    }

    pub fn contains(&self, layer: usize) -> bool {
        (self.first..self.end).contains(&layer)
    }

    /// Whether the part reads tensor `name`: its own layers', and the others only with the head.
    pub fn keeps(&self, name: &str) -> bool {
        match layer_index(name) {
            Some(i) => self.contains(i),
            None => self.head,
        }
    }
}

/// The decoder layer a tensor belongs to (`….layers.<i>.…`), if any.
fn layer_index(name: &str) -> Option<usize> {
    let rest = &name[name.find(".layers.")? + ".layers.".len()..];
    rest.split('.').next()?.parse().ok()
}

/// Why a model cannot be loaded; all are refused before any weights are read.
#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
pub enum LoadError {
    #[error("{}", unsupported(model_type, text_model_type.as_deref(), None))]
    UnsupportedArchitecture {
        model_type: String,
        text_model_type: Option<String>,
    },
    #[error("invalid config.json: {key} {reason}")]
    InvalidConfig { key: String, reason: String },
}

impl LoadError {
    /// The message naming the model (`org/name` or a path), as the CLI prints it.
    pub fn report(&self, model: &str) -> String {
        match self {
            Self::UnsupportedArchitecture {
                model_type,
                text_model_type,
            } => unsupported(model_type, text_model_type.as_deref(), Some(model)),
            other => other.to_string(),
        }
    }
}

fn unsupported(model_type: &str, text_model_type: Option<&str>, model: Option<&str>) -> String {
    let arch = match text_model_type {
        Some(text) => format!("{model_type}+{text}"),
        None => model_type.to_string(),
    };
    let location = model
        .map(|model| format!(" in {model}"))
        .unwrap_or_default();
    format!(
        "unsupported architecture '{arch}'{location}\n  supported families: {}",
        registry::supported()
    )
}
