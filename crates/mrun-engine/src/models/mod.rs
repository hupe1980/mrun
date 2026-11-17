//! Model families: the registry from `config.json` to a family, and (with `engine`) the graphs.

#[cfg(feature = "engine")]
pub(crate) mod activations;
#[cfg(feature = "engine")]
mod common;
#[cfg(feature = "engine")]
pub mod dflash;
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
pub mod vision;

#[cfg(feature = "engine")]
#[doc(hidden)]
pub use common::{SCORE_BLOCK_BYTES_DEFAULT, set_score_block_bytes};
pub use registry::{Family, FamilyConfig, SyntaxId};

#[cfg(feature = "engine")]
pub(crate) use graph::per_row;
#[cfg(feature = "engine")]
pub use graph::{Model, build};

/// Where a chunk's rows sit (009): row `r` is KV slot `at + r`. Without `runs` every row is text
/// at rotary position `at + r + delta`; with `runs` each run says how its rows rotate.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Pos<'a> {
    /// The KV slot of the first row.
    pub at: i32,
    /// Rotary position minus slot of text rows when `runs` is empty (images before shorten it).
    pub delta: i32,
    /// The chunk's rows in order, when any is an image row; empty for text-only chunks.
    pub runs: &'a [Run],
}

impl Pos<'static> {
    /// Text at slot `at`, rotary position `at`.
    pub fn at(at: i32) -> Self {
        Self {
            at,
            delta: 0,
            runs: &[],
        }
    }
}

impl Pos<'_> {
    /// The same rows `by` slots later (a later piece of the same chunk).
    pub fn shifted(self, by: i32) -> Pos<'static> {
        assert!(self.runs.is_empty(), "image chunks are not split");
        Pos {
            at: self.at + by,
            delta: self.delta,
            runs: &[],
        }
    }
}

/// Consecutive rows of a chunk with one rotation rule.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Run {
    /// `len` text rows; the first at rotary position `offset`.
    Text { len: i32, offset: i32 },
    /// `len` rows of an image whose merged grid is `width` wide; the run's first row is the
    /// image's token `first`, and every row of the image has temporal position `base`, height
    /// `base + k / width` and width `base + k % width` for its token `k`.
    Image {
        len: i32,
        first: i32,
        width: i32,
        base: i32,
    },
}

impl Run {
    /// Rows of the run.
    pub fn rows(&self) -> i32 {
        match *self {
            Run::Text { len, .. } | Run::Image { len, .. } => len,
        }
    }
}

#[cfg(feature = "engine")]
mod graph {
    use super::{LoadError, Part, Pos};
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
        /// Decoder layer `i` over `x` at `pos`, updating its state.
        fn layer(
            &self,
            i: usize,
            x: &Array,
            state: &mut LayerState,
            pos: Pos<'_>,
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
            pos: Pos<'_>,
            splits: &[i32],
            captured: &mut Vec<crate::state::Recurrent>,
        ) -> tensor::Result<Array> {
            let _ = (splits, captured);
            self.layer(i, x, state, pos)
        }
        /// [`Model::layer`] for a verify pass (006 FR-001): every row computed as the decode step
        /// at its position computes it, the state after each position but the last pushed to
        /// `captured` (linear layers; attention layers push nothing).
        fn layer_verify(
            &self,
            i: usize,
            x: &Array,
            state: &mut LayerState,
            pos: Pos<'_>,
            captured: &mut Vec<crate::state::Recurrent>,
        ) -> tensor::Result<Array> {
            let splits: Vec<i32> = (1..x.dim(1)).collect();
            self.layer_split(i, x, state, pos, &splits, captured)
        }
        /// Decoder layer `i` for one decode step of several lanes (011): row `b` of `x`
        /// `[1, B, hidden]` is lane `b`'s token at `positions[b]` with its state `states[b]`. Each
        /// row equals [`Model::layer`] on that lane alone, bit for bit, under
        /// `tensor::with_row_exact`. The default runs the rows one by one; families share their
        /// weight passes across the rows.
        fn layer_lanes(
            &self,
            i: usize,
            x: &Array,
            states: &mut [&mut LayerState],
            positions: &[Pos<'_>],
        ) -> tensor::Result<Array> {
            let rows = per_row(x, |b, row| self.layer(i, row, states[b], positions[b]))?;
            tensor::concatenate(&rows.iter().collect::<Vec<_>>(), 1)
        }
        /// Hidden states `[1, L, hidden]` → logits of the last position `[1, vocab]`.
        fn logits(&self, x: &Array) -> tensor::Result<Array>;
        /// Hidden states `[1, L, hidden]` → logits of every position `[L, vocab]`.
        fn logits_rows(&self, x: &Array) -> tensor::Result<Array>;
        /// Hidden states `[1, L, hidden]` → logits `[L, vocab]` through the output projection
        /// alone, without the final norm (a DFlash drafter's head, 030).
        fn head_raw(&self, x: &Array) -> tensor::Result<Array>;
        /// Width of the hidden state between layers.
        fn hidden(&self) -> i32;
        /// Analytic scratch of one chunk of `len` tokens at `depth` (activations, MoE routing,
        /// unfused attention scores; D029); the engine maximizes it over the chunk schedule.
        fn chunk_scratch(&self, len: u32, depth: u32) -> u64;
    }

    /// `f(b, row)` for each row `b` of `x` `[1, B, …]`, as a `[1, 1, …]` array.
    pub(crate) fn per_row(
        x: &Array,
        mut f: impl FnMut(usize, &Array) -> tensor::Result<Array>,
    ) -> tensor::Result<Vec<Array>> {
        let shape = x.shape();
        (0..x.dim(1))
            .map(|b| {
                let mut start = vec![0; shape.len()];
                let mut stop = shape.clone();
                start[1] = b;
                stop[1] = b + 1;
                f(b as usize, &tensor::slice(x, &start, &stop)?)
            })
            .collect()
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
