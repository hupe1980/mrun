//! Sequence state of the engine, one [`LayerState`] per layer (D028):
//!
//! - `Kv`: an attention layer's K/V buffers `[1, kv_heads, capacity, head_dim]`, allocated
//!   lazily and resized one layer at a time, so a resize or move never holds more than one extra
//!   layer. Positions `[0, len)` hold the live sequence; data at or beyond `len` is garbage until
//!   written. Writes use `slice_update`, so a buffer with no other reference is updated in
//!   place (on the CPU and the GPU).
//! - `Recurrent`: a linear layer's convolution tail `[1, K − 1, conv_dim]` and delta state
//!   `[1, Hv, Dv, Dk]` (f32), describing exactly one position. The model replaces both arrays on
//!   every call (it never writes into them), so keeping the old handles keeps the old state:
//!   that is how a cancelled chunk and a discarded pipelined step roll back, and checkpoints
//!   share the handles instead of copying the buffers.
//!
//! After each moved or resized layer the freed buffer is returned to the system (`clear_cache`),
//! so nuvec's buffer cache does not keep memory the accounting already released (D029).

use crate::tensor::{self, Array, Dtype};

/// One attention layer's K and V buffers (`None` until the first growth).
#[derive(Debug, Default)]
pub struct LayerKv {
    buffers: Option<(Array, Array)>,
}

impl LayerKv {
    /// A layer allocated for `capacity` positions of zeros (timing at depth, calibration).
    pub fn zeroed(
        kv_heads: u32,
        head_dim: u32,
        capacity: u32,
        dtype: Dtype,
    ) -> tensor::Result<Self> {
        let shape = [1, kv_heads as i32, capacity as i32, head_dim as i32];
        let keys = tensor::zeros(&shape, dtype)?;
        let values = tensor::zeros(&shape, dtype)?;
        tensor::eval(&[&keys, &values])?;
        Ok(Self {
            buffers: Some((keys, values)),
        })
    }

    /// Writes `k`, `v` (`[1, H, L, D]`) at `offset` and returns views of positions
    /// `[0, offset + L)` for attention. The capacity must already cover `offset + L`.
    pub fn update(&mut self, k: &Array, v: &Array, offset: i32) -> tensor::Result<(Array, Array)> {
        let (keys, values) = self
            .buffers
            .take()
            .ok_or_else(|| tensor::Error("attention state written before allocation".into()))?;
        let shape = k.shape();
        let (heads, len, dim) = (shape[1], shape[2], shape[3]);
        let end = offset + len;
        let (start, stop) = ([0, 0, offset, 0], [1, heads, end, dim]);
        let keys = tensor::slice_update(&keys, k, &start, &stop)?;
        let values = tensor::slice_update(&values, v, &start, &stop)?;
        let view = |a: &Array| tensor::slice(a, &[0, 0, 0, 0], &[1, heads, end, dim]);
        let out = (view(&keys)?, view(&values)?);
        self.buffers = Some((keys, values));
        Ok(out)
    }
}

/// One linear layer's recurrent state; `None` arrays are zeros (the state before position 0).
#[derive(Clone, Debug, Default)]
pub struct Recurrent {
    pub conv: Option<Array>,
    pub state: Option<Array>,
}

impl Recurrent {
    fn arrays(&self) -> impl Iterator<Item = &Array> {
        self.conv.iter().chain(&self.state)
    }
}

/// The recurrent state of every linear layer, in layer order.
pub type RecurrentState = Vec<Recurrent>;

/// Evaluates `state`, so a stored copy of its handles holds data rather than a graph.
/// Checkpoints share the live handles instead of copying them: the model replaces the arrays on
/// every call and never writes into them, and nuvec (as MLX) donates an input buffer to an output
/// only when nothing else references it.
pub fn eval(state: &[Recurrent]) -> tensor::Result<()> {
    tensor::eval(&state.iter().flat_map(Recurrent::arrays).collect::<Vec<_>>())
}

/// A layer's sequence state.
#[derive(Debug)]
pub enum LayerState {
    Kv(LayerKv),
    Recurrent(Recurrent),
}

impl LayerState {
    pub fn kv(&mut self) -> tensor::Result<&mut LayerKv> {
        match self {
            Self::Kv(kv) => Ok(kv),
            Self::Recurrent(_) => Err(tensor::Error("attention on a linear layer's state".into())),
        }
    }

    pub fn recurrent(&mut self) -> tensor::Result<&mut Recurrent> {
        match self {
            Self::Recurrent(r) => Ok(r),
            Self::Kv(_) => Err(tensor::Error(
                "linear attention on an attention layer's state".into(),
            )),
        }
    }

    /// The arrays a layer's output depends on, for evaluation with it.
    pub fn arrays(&self) -> Vec<&Array> {
        match self {
            Self::Kv(kv) => kv.buffers.iter().flat_map(|(k, v)| [k, v]).collect(),
            Self::Recurrent(r) => r.arrays().collect(),
        }
    }
}

/// Positions `[start, end)` of every attention layer detached from the live state (`None` for
/// an empty range).
#[derive(Debug)]
pub struct KvTail {
    layers: Vec<Option<(Array, Array)>>,
}

/// Per-layer state with one shared attention capacity.
#[derive(Debug)]
pub struct SeqState {
    pub layers: Vec<LayerState>,
    /// The images of the sequence (009), set by the runtime per request.
    pub media: Media,
    capacity: u32,
    kv_heads: i32,
    head_dim: i32,
    dtype: Dtype,
}

impl SeqState {
    /// `linear[i]`: layer `i` holds recurrent rather than attention state.
    pub fn new(linear: &[bool], kv_heads: u32, head_dim: u32, dtype: Dtype) -> Self {
        Self {
            layers: linear
                .iter()
                .map(|&linear| {
                    if linear {
                        LayerState::Recurrent(Recurrent::default())
                    } else {
                        LayerState::Kv(LayerKv::default())
                    }
                })
                .collect(),
            media: Media::default(),
            capacity: 0,
            kv_heads: kv_heads as i32,
            head_dim: head_dim as i32,
            dtype,
        }
    }

    pub fn capacity(&self) -> u32 {
        self.capacity
    }

    /// An empty state of the same layout (another lane's, 011).
    pub fn fresh(&self) -> Self {
        Self {
            layers: self
                .layers
                .iter()
                .map(|l| match l {
                    LayerState::Recurrent(_) => LayerState::Recurrent(Recurrent::default()),
                    LayerState::Kv(_) => LayerState::Kv(LayerKv::default()),
                })
                .collect(),
            media: Media::default(),
            capacity: 0,
            kv_heads: self.kv_heads,
            head_dim: self.head_dim,
            dtype: self.dtype,
        }
    }

    fn kv_layers(&mut self) -> impl Iterator<Item = &mut LayerKv> {
        self.layers.iter_mut().filter_map(|layer| match layer {
            LayerState::Kv(kv) => Some(kv),
            LayerState::Recurrent(_) => None,
        })
    }

    fn recurrent_layers(&mut self) -> impl Iterator<Item = &mut Recurrent> {
        self.layers.iter_mut().filter_map(|layer| match layer {
            LayerState::Recurrent(r) => Some(r),
            LayerState::Kv(_) => None,
        })
    }

    /// Handles of the recurrent state (shared buffers, no copy).
    pub fn recurrent(&mut self) -> RecurrentState {
        self.recurrent_layers().map(|r| r.clone()).collect()
    }

    /// Replaces the recurrent state, dropping the current one.
    pub fn set_recurrent(&mut self, state: RecurrentState) {
        debug_assert_eq!(state.len(), self.recurrent_layers().count());
        for (layer, r) in self.recurrent_layers().zip(state) {
            *layer = r;
        }
    }

    /// Moves the recurrent state out, leaving zeros.
    pub fn take_recurrent(&mut self) -> RecurrentState {
        self.recurrent_layers().map(std::mem::take).collect()
    }

    fn range(&self, start: u32, end: u32) -> ([i32; 4], [i32; 4]) {
        (
            [0, 0, start as i32, 0],
            [1, self.kv_heads, end as i32, self.head_dim],
        )
    }

    /// A buffer of `capacity` positions holding `keep` positions of `old` and, at `at`, `extra`.
    ///
    /// Zeros, then the kept prefix and the extra positions written in place: one buffer on
    /// either backend (nuvec donates a base pending in the same evaluation since its D-066, which
    /// closed mrun's N10b).
    fn rebuilt(
        &self,
        old: Option<&Array>,
        keep: u32,
        capacity: u32,
        extra: Option<(u32, &Array)>,
    ) -> tensor::Result<Array> {
        let shape = [1, self.kv_heads, capacity as i32, self.head_dim];
        let mut buf = tensor::zeros(&shape, self.dtype)?;
        if let Some(old) = old
            && keep > 0
        {
            let (a, b) = self.range(0, keep);
            buf = tensor::slice_update(&buf, &tensor::slice(old, &a, &b)?, &a, &b)?;
        }
        if let Some((at, part)) = extra {
            let (a, b) = self.range(at, at + part.dim(2) as u32);
            buf = tensor::slice_update(&buf, part, &a, &b)?;
        }
        Ok(buf)
    }

    /// Positions of each attention layer, by index into `layers`.
    fn kv_indices(&self) -> Vec<usize> {
        (0..self.layers.len())
            .filter(|&i| matches!(self.layers[i], LayerState::Kv(_)))
            .collect()
    }

    fn take_buffers(&mut self, i: usize) -> Option<(Array, Array)> {
        match &mut self.layers[i] {
            LayerState::Kv(kv) => kv.buffers.take(),
            LayerState::Recurrent(_) => None,
        }
    }

    fn put_buffers(&mut self, i: usize, buffers: Option<(Array, Array)>) {
        if let LayerState::Kv(kv) = &mut self.layers[i] {
            kv.buffers = buffers;
        }
    }

    /// Resizes every attention layer to `capacity` positions (0 releases them), keeping
    /// `[0, keep)`, one layer at a time.
    pub fn resize(&mut self, keep: u32, capacity: u32) -> tensor::Result<()> {
        debug_assert!(keep <= capacity);
        for i in self.kv_indices() {
            let old = self.take_buffers(i);
            if capacity == 0 {
                drop(old);
                tensor::memory::clear_cache();
                continue;
            }
            let keys = self.rebuilt(old.as_ref().map(|b| &b.0), keep, capacity, None)?;
            let values = self.rebuilt(old.as_ref().map(|b| &b.1), keep, capacity, None)?;
            tensor::eval(&[&keys, &values])?;
            drop(old);
            self.put_buffers(i, Some((keys, values)));
            tensor::memory::clear_cache();
        }
        self.capacity = capacity;
        Ok(())
    }

    /// Releases every attention buffer and zeroes the recurrent state.
    pub fn clear(&mut self) {
        for kv in self.kv_layers() {
            kv.buffers = None;
        }
        for r in self.recurrent_layers() {
            *r = Recurrent::default();
        }
        self.capacity = 0;
        tensor::memory::clear_cache();
    }

    /// Moves attention positions `[at, len)` out, layer by layer: copy the layer's tail, shrink
    /// the layer to `capacity` (0 releases it; the same capacity keeps the buffer), release the
    /// old buffer, next layer.
    pub fn detach(&mut self, at: u32, len: u32, capacity: u32) -> tensor::Result<KvTail> {
        let mut layers = Vec::new();
        for i in self.kv_indices() {
            let old = self.take_buffers(i);
            let tail = match &old {
                Some((keys, values)) if at < len => {
                    let (a, b) = self.range(at, len);
                    Some((
                        tensor::deep_copy(&tensor::slice(keys, &a, &b)?)?,
                        tensor::deep_copy(&tensor::slice(values, &a, &b)?)?,
                    ))
                }
                Some(_) => None,
                None if at < len => {
                    return Err(tensor::Error("detach from unallocated state".into()));
                }
                None => None,
            };
            let kept = if capacity == 0 {
                None
            } else if capacity == self.capacity {
                // The capacity stays: positions past `at` are garbage until written again.
                old.clone()
            } else {
                let k = self.rebuilt(old.as_ref().map(|b| &b.0), at, capacity, None)?;
                let v = self.rebuilt(old.as_ref().map(|b| &b.1), at, capacity, None)?;
                Some((k, v))
            };
            let outputs: Vec<&Array> = tail.iter().chain(&kept).flat_map(|(k, v)| [k, v]).collect();
            tensor::eval(&outputs)?;
            if let (Some((tail_k, _)), Some((keys, _))) = (&tail, &old) {
                debug_assert_ne!(tail_k.buffer_id()?, keys.buffer_id()?);
            }
            drop(old);
            self.put_buffers(i, kept);
            layers.push(tail);
            tensor::memory::clear_cache();
        }
        self.capacity = capacity;
        Ok(KvTail { layers })
    }

    /// Moves `tail` back in at `at`, layer by layer, growing the layer to `capacity` when needed
    /// and freeing each tail layer as soon as it is copied.
    pub fn attach(&mut self, at: u32, tail: KvTail, capacity: u32) -> tensor::Result<()> {
        debug_assert!(capacity >= self.capacity);
        for (i, part) in self.kv_indices().into_iter().zip(tail.layers) {
            if capacity == 0 {
                // Nothing allocated and nothing to attach.
                continue;
            }
            let old = self.take_buffers(i);
            let buffers = match (old, &part) {
                (Some(old), None) if capacity == self.capacity => old,
                (Some((k, v)), Some((part_k, part_v))) if capacity == self.capacity => {
                    let (a, b) = self.range(at, at + part_k.dim(2) as u32);
                    (
                        tensor::slice_update(&k, part_k, &a, &b)?,
                        tensor::slice_update(&v, part_v, &a, &b)?,
                    )
                }
                (old, part) => {
                    let extra_k = part.as_ref().map(|(k, _)| (at, k));
                    let extra_v = part.as_ref().map(|(_, v)| (at, v));
                    (
                        self.rebuilt(old.as_ref().map(|b| &b.0), at, capacity, extra_k)?,
                        self.rebuilt(old.as_ref().map(|b| &b.1), at, capacity, extra_v)?,
                    )
                }
            };
            tensor::eval(&[&buffers.0, &buffers.1])?;
            drop(part);
            self.put_buffers(i, Some(buffers));
            tensor::memory::clear_cache();
        }
        self.capacity = capacity;
        Ok(())
    }
}

/// One image of a sequence (009): its tokens `[start, start + len)` (a merged grid `height` ×
/// `width`, row-major) and, until they are prefilled, its features `[len, hidden]`.
#[derive(Debug, Clone)]
pub struct MediaSpan {
    pub start: u32,
    pub height: u32,
    pub width: u32,
    /// The runtime's key for the image.
    pub key: u64,
    pub features: Option<Array>,
}

impl MediaSpan {
    /// Tokens of the image.
    pub fn tokens(&self) -> u32 {
        self.height * self.width
    }

    pub fn end(&self) -> u32 {
        self.start + self.tokens()
    }

    /// Rotary positions the image takes (t = 1): its largest extent.
    fn extent(&self) -> i32 {
        self.height.max(self.width) as i32
    }
}

/// The images of a sequence, in order, and the rotary positions they imply (M-RoPE): text after
/// an image continues at the image's largest position + 1, so its rotary position is its slot
/// plus the deltas of the images before it.
#[derive(Debug, Clone, Default)]
pub struct Media {
    pub spans: Vec<MediaSpan>,
}

impl Media {
    /// Rotary position minus slot of a text token at `at` (images ending at or before it).
    pub fn delta(&self, at: u32) -> i32 {
        self.spans
            .iter()
            .filter(|s| s.end() <= at)
            .map(|s| s.extent() - s.tokens() as i32)
            .sum()
    }

    /// The rotation runs of rows `[at, at + len)`: empty when no image overlaps them (every row
    /// is text at `slot + delta(at)`).
    pub fn runs(&self, at: u32, len: u32) -> Vec<crate::models::Run> {
        use crate::models::Run;
        let end = at + len;
        if !self.spans.iter().any(|s| s.start < end && s.end() > at) {
            return Vec::new();
        }
        let mut runs = Vec::new();
        let mut row = at;
        for s in self.spans.iter().filter(|s| s.start < end && s.end() > at) {
            if s.start > row {
                runs.push(Run::Text {
                    len: (s.start - row) as i32,
                    offset: row as i32 + self.delta(row),
                });
                row = s.start;
            }
            let stop = s.end().min(end);
            runs.push(Run::Image {
                len: (stop - row) as i32,
                first: (row - s.start) as i32,
                width: s.width as i32,
                base: s.start as i32 + self.delta(s.start),
            });
            row = stop;
        }
        if row < end {
            runs.push(Run::Text {
                len: (end - row) as i32,
                offset: row as i32 + self.delta(row),
            });
        }
        runs
    }

    /// The image rows of `[at, at + len)`: `(row in the chunk, span, first token, count)`.
    pub fn rows(&self, at: u32, len: u32) -> Vec<(u32, usize, u32, u32)> {
        let end = at + len;
        self.spans
            .iter()
            .enumerate()
            .filter(|(_, s)| s.start < end && s.end() > at)
            .map(|(i, s)| {
                let from = s.start.max(at);
                let to = s.end().min(end);
                (from - at, i, from - s.start, to - from)
            })
            .collect()
    }

    /// Drops the features of images whose tokens all lie before `len` (FR-004).
    pub fn committed(&mut self, len: u32) {
        for s in &mut self.spans {
            if s.end() <= len {
                s.features = None;
            }
        }
    }
}

#[cfg(test)]
mod media_tests {
    use super::*;
    use crate::models::Run;

    fn span(start: u32, height: u32, width: u32) -> MediaSpan {
        MediaSpan {
            start,
            height,
            width,
            key: u64::from(start),
            features: None,
        }
    }

    #[test]
    fn positions_follow_m_rope() {
        // 5 text tokens, a 2 × 3 image (6 tokens, extent 3), then text.
        let m = Media {
            spans: vec![span(5, 2, 3)],
        };
        assert_eq!(m.delta(5), 0);
        assert_eq!(
            m.delta(11),
            3 - 6,
            "text after the image continues at 5 + 3"
        );
        assert!(m.runs(0, 5).is_empty());
        assert!(m.runs(11, 4).is_empty());
        assert_eq!(
            m.runs(3, 10),
            vec![
                Run::Text { len: 2, offset: 3 },
                Run::Image {
                    len: 6,
                    first: 0,
                    width: 3,
                    base: 5
                },
                Run::Text { len: 2, offset: 8 },
            ]
        );
        // A chunk ending inside the image, and the next one.
        assert_eq!(
            m.runs(7, 2),
            vec![Run::Image {
                len: 2,
                first: 2,
                width: 3,
                base: 5
            }]
        );
        assert_eq!(m.rows(3, 10), vec![(2, 0, 0, 6)]);
        assert_eq!(m.rows(7, 2), vec![(0, 0, 2, 2)]);
    }

    #[test]
    fn two_images_add_their_deltas() {
        let m = Media {
            spans: vec![span(1, 2, 2), span(8, 1, 4)],
        };
        assert_eq!(m.delta(5), 2 - 4);
        // The 1 × 4 image's extent equals its tokens: no change after it.
        assert_eq!(m.delta(12), 2 - 4);
        let runs = m.runs(5, 3);
        assert_eq!(runs, Vec::new(), "text between the images rotates at 5 - 2");
    }
}
