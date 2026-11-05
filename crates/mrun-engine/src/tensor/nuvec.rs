//! The tensor surface of [`super`] on nuvec, the pure-Rust MLX port (D065), implemented on
//! `nuvec::Array`.
//!
//! nuvec operations return arrays, not results: a failed operation gives a failed array that
//! carries the error and the caller's location. Every function here returns `Result` and calls
//! [`nv::Array::check`] on what it builds (it never evaluates), so a failure surfaces at the op
//! that caused it. Shapes and axes are `i32`. Arrays are `Send` (nuvec's are); the engine still
//! keeps them on its worker thread, whose default stream follows [`use_device`].

use std::cell::{Cell, RefCell};
use std::ffi::CStr;
use std::path::Path;
use std::sync::OnceLock;

use ::nuvec as nv;
use nv::{DType, ops};

/// A nuvec failure: nuvec's message.
#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
#[error("nuvec: {0}")]
pub struct Error(pub String);

impl From<nv::Error> for Error {
    fn from(error: nv::Error) -> Self {
        Self(error.to_string())
    }
}

pub type Result<T> = std::result::Result<T, Error>;

/// `a`, or the error it carries (never evaluates).
fn ok(a: nv::Array) -> Result<Array> {
    Ok(Array(a.check()?))
}

/// `arrays`, or the first error one of them carries (never evaluates).
fn ok_all(arrays: Vec<nv::Array>) -> Result<Vec<Array>> {
    arrays.into_iter().map(ok).collect()
}
thread_local! {
    static DEVICE: Cell<Device> = const { Cell::new(Device::Default) };
    /// The error a compiled graph returned (nuvec's `compile` sees only its own error type).
    static GRAPH_ERROR: RefCell<Option<String>> = const { RefCell::new(None) };
}

/// Where this thread's operations run.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Device {
    /// GPU when available, else CPU.
    Default,
    Cpu,
}

/// Selects the device for operations on this thread (nuvec's per-thread default stream); call
/// before the first operation.
pub fn use_device(device: Device) {
    DEVICE.with(|d| d.set(device));
    let target = match device {
        Device::Cpu => nv::Device::cpu(),
        Device::Default => nv::default_device(),
    };
    nv::set_default_stream(nv::Stream::default_for(target));
}

/// Whether a Metal device is present (asked once).
pub fn metal_available() -> bool {
    static AVAILABLE: OnceLock<bool> = OnceLock::new();
    *AVAILABLE.get_or_init(nv::metal::is_available)
}

/// Whether this thread's operations run on the GPU (Metal present and not [`Device::Cpu`]).
pub fn on_gpu() -> bool {
    DEVICE.with(Cell::get) != Device::Cpu && metal_available()
}

/// Element types used by mrun.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Dtype {
    Bool,
    Uint8,
    Uint32,
    Int32,
    Float16,
    Float32,
    Bfloat16,
    /// Another nuvec dtype: an index into [`OTHER`].
    Other(u32),
}

/// nuvec dtypes mrun does not name, by their [`Dtype::Other`] code.
const OTHER: [DType; 7] = [
    DType::U16,
    DType::U64,
    DType::I8,
    DType::I16,
    DType::I64,
    DType::F64,
    DType::Complex64,
];

impl Dtype {
    /// Bytes per element.
    pub fn size(self) -> u64 {
        match self {
            Self::Bool | Self::Uint8 => 1,
            Self::Float16 | Self::Bfloat16 => 2,
            Self::Uint32 | Self::Int32 | Self::Float32 => 4,
            Self::Other(code) => OTHER.get(code as usize).map_or(8, |d| d.size() as u64),
        }
    }

    fn raw(self) -> Result<DType> {
        Ok(match self {
            Self::Bool => DType::Bool,
            Self::Uint8 => DType::U8,
            Self::Uint32 => DType::U32,
            Self::Int32 => DType::I32,
            Self::Float16 => DType::F16,
            Self::Float32 => DType::F32,
            Self::Bfloat16 => DType::BF16,
            Self::Other(code) => *OTHER
                .get(code as usize)
                .ok_or_else(|| Error(format!("unknown dtype code {code}")))?,
        })
    }

    fn from_raw(raw: DType) -> Self {
        match raw {
            DType::Bool => Self::Bool,
            DType::U8 => Self::Uint8,
            DType::U32 => Self::Uint32,
            DType::I32 => Self::Int32,
            DType::F16 => Self::Float16,
            DType::F32 => Self::Float32,
            DType::BF16 => Self::Bfloat16,
            other => Self::Other(
                OTHER
                    .iter()
                    .position(|&d| d == other)
                    .map_or(u32::MAX, |i| i as u32),
            ),
        }
    }
}

/// A nuvec array (reference-counted: `Clone` shares the computation and its data). Lazy until
/// evaluated.
#[derive(Clone)]
pub struct Array(nv::Array);

impl std::fmt::Debug for Array {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Array({:?}, {:?})", self.shape(), self.dtype())
    }
}

fn raw(arrays: &[&Array]) -> Vec<nv::Array> {
    arrays.iter().map(|a| a.0.clone()).collect()
}

fn wrap(arrays: Vec<nv::Array>) -> Vec<Array> {
    arrays.into_iter().map(Array).collect()
}

impl Array {
    /// The caller guarantees `data.len()` equals the shape's element count (as with MLX, whose
    /// constructor asserts it); a violation is a programming error and panics.
    fn from_data<T: nv::Element>(data: &[T], shape: &[i32]) -> Self {
        match nv::Array::from_slice(data, shape).check() {
            Ok(a) => Self(a),
            Err(e) => panic!("data does not match shape {shape:?}: {e}"),
        }
    }

    pub fn from_u8(data: &[u8], shape: &[i32]) -> Self {
        Self::from_data(data, shape)
    }

    pub fn from_u32(data: &[u32], shape: &[i32]) -> Self {
        Self::from_data(data, shape)
    }

    pub fn from_f32(data: &[f32], shape: &[i32]) -> Self {
        Self::from_data(data, shape)
    }

    pub fn from_bool(data: &[bool], shape: &[i32]) -> Self {
        Self::from_data(data, shape)
    }

    pub fn scalar_f32(value: f32) -> Self {
        Self(nv::Array::from(value))
    }

    pub fn scalar_i32(value: i32) -> Self {
        Self(nv::Array::from(value))
    }

    pub fn shape(&self) -> Vec<i32> {
        self.0.shape().to_vec()
    }

    pub fn dim(&self, axis: i32) -> i32 {
        let shape = self.shape();
        let axis = if axis < 0 {
            shape.len() as i32 + axis
        } else {
            axis
        };
        shape[axis as usize]
    }

    pub fn dtype(&self) -> Dtype {
        Dtype::from_raw(self.0.dtype())
    }

    pub fn size(&self) -> usize {
        self.0.size()
    }

    /// Evaluates the array (blocking).
    pub fn eval(&self) -> Result<()> {
        Ok(self.0.eval()?)
    }

    /// The single element as `u32` (evaluates first).
    pub fn item_u32(&self) -> Result<u32> {
        Ok(self.0.astype(DType::U32).item::<u32>()?)
    }

    /// The single element as `f32` (evaluates first).
    pub fn item_f32(&self) -> Result<f32> {
        Ok(self.0.astype(DType::F32).item::<f32>()?)
    }

    /// All elements as `f32`, row-major (evaluates first).
    pub fn to_vec_f32(&self) -> Result<Vec<f32>> {
        Ok(self.0.astype(DType::F32).to_vec::<f32>()?)
    }

    /// All elements as `u32`, row-major (evaluates first).
    pub fn to_vec_u32(&self) -> Result<Vec<u32>> {
        Ok(self.0.astype(DType::U32).to_vec::<u32>()?)
    }

    /// The identity of the evaluated array's storage, for aliasing checks in debug builds:
    /// equal for arrays that share a buffer (a view and its base), different otherwise.
    pub fn buffer_id(&self) -> Result<usize> {
        self.eval()?;
        self.0
            .buffer_id()
            .ok_or_else(|| Error("buffer_id: the array holds no data".into()))
    }
}

/// Evaluates `arrays` (blocking).
pub fn eval(arrays: &[&Array]) -> Result<()> {
    Ok(nv::eval(&raw(arrays))?)
}

/// Starts evaluating `arrays` and returns without waiting for the GPU.
pub fn async_eval(arrays: &[&Array]) -> Result<()> {
    Ok(nv::async_eval(&raw(arrays))?)
}

// ---- Shape and type ----

pub fn reshape(a: &Array, shape: &[i32]) -> Result<Array> {
    ok(a.0.reshape(shape))
}

pub fn transpose(a: &Array, axes: &[i32]) -> Result<Array> {
    ok(a.0.transpose(axes))
}

pub fn astype(a: &Array, dtype: Dtype) -> Result<Array> {
    ok(a.0.astype(dtype.raw()?))
}

pub fn contiguous(a: &Array) -> Result<Array> {
    ok(a.0.contiguous())
}

fn check_bounds(start: &[i32], stop: &[i32]) -> Result<()> {
    if start.len() == stop.len() {
        Ok(())
    } else {
        Err(Error(format!(
            "slice: {} starts and {} stops",
            start.len(),
            stop.len()
        )))
    }
}

/// One range per axis, for nuvec's runtime-rank indexes (Python bounds; mrun's are in range).
fn ranges(start: &[i32], stop: &[i32]) -> Result<Vec<std::ops::Range<i32>>> {
    check_bounds(start, stop)?;
    Ok(start.iter().zip(stop).map(|(&s, &e)| s..e).collect())
}

/// `a[start..stop]` per axis (unit strides; a view).
pub fn slice(a: &Array, start: &[i32], stop: &[i32]) -> Result<Array> {
    ok(a.0.i(&ranges(start, stop)?))
}

/// `a` with `a[start..stop] = update`.
pub fn slice_update(a: &Array, update: &Array, start: &[i32], stop: &[i32]) -> Result<Array> {
    ok(a.0.slice_update(&update.0, &ranges(start, stop)?))
}

/// `a` with size-1 axes inserted at `axes` (positions in the result, as MLX).
pub fn expand_dims(a: &Array, axes: &[i32]) -> Result<Array> {
    ok(a.0.expand_dims(axes))
}

/// `a` without the size-1 axes `axes` (none removed when `axes` is empty, as MLX).
pub fn squeeze(a: &Array, axes: &[i32]) -> Result<Array> {
    if axes.is_empty() {
        return Ok(a.clone());
    }
    ok(a.0.squeeze(axes))
}

/// Splits `a` along `axis` before each of `indices`.
pub fn split_sections(a: &Array, indices: &[i32], axis: i32) -> Result<Vec<Array>> {
    ok_all(ops::split_at(&a.0, indices, axis))
}

/// Splits `a` along `axis` into `parts` equal parts.
pub fn split(a: &Array, parts: i32, axis: i32) -> Result<Vec<Array>> {
    ok_all(ops::split(&a.0, parts, axis))
}

/// Each element of `a` repeated `repeats` times along `axis`.
pub fn repeat(a: &Array, repeats: i32, axis: i32) -> Result<Array> {
    ok(ops::repeat(&a.0, repeats, axis))
}

pub fn stack(arrays: &[&Array], axis: i32) -> Result<Array> {
    ok(ops::stack(&raw(arrays), axis))
}

pub fn zeros(shape: &[i32], dtype: Dtype) -> Result<Array> {
    ok(ops::zeros(shape, dtype.raw()?))
}

pub fn concatenate(arrays: &[&Array], axis: i32) -> Result<Array> {
    ok(ops::concatenate(&raw(arrays), axis))
}

/// A copy that never shares its buffer with `a` (nuvec's `copy`, MLX's `mx.copy`: no
/// simplification or in-place update makes it share memory with `a`).
pub fn deep_copy(a: &Array) -> Result<Array> {
    ok(a.0.copy())
}

/// The scalar `value` in `dtype` (via f32, as MLX converts Python floats), built as a constant
/// of that dtype: the same bits as casting an f32 scalar, without a cast node per call.
pub fn scalar(value: f32, dtype: Dtype) -> Result<Array> {
    let a = match dtype {
        Dtype::Float32 => nv::Array::from(value),
        Dtype::Bfloat16 => nv::Array::from(nv::bf16::from_f32(value)),
        Dtype::Float16 => nv::Array::from(nv::f16::from_f32(value)),
        Dtype::Int32 => nv::Array::from(value as i32),
        Dtype::Uint32 => nv::Array::from(value as u32),
        _ => return astype(&Array::scalar_f32(value), dtype),
    };
    ok(a)
}

// ---- Arithmetic ----

macro_rules! binary {
    ($name:ident) => {
        pub fn $name(a: &Array, b: &Array) -> Result<Array> {
            ok(ops::$name(&a.0, &b.0))
        }
    };
}

macro_rules! unary {
    ($name:ident) => {
        pub fn $name(a: &Array) -> Result<Array> {
            ok(ops::$name(&a.0))
        }
    };
}

binary!(add);
binary!(subtract);
binary!(multiply);
binary!(divide);
binary!(greater_equal);
binary!(logaddexp);
unary!(sigmoid);
unary!(exp);
unary!(log);
unary!(negative);

pub fn matmul(a: &Array, b: &Array) -> Result<Array> {
    ok(a.0.matmul(&b.0))
}

/// `floor(a / b)` in the promoted dtype (an integer division for integer operands).
pub fn floor_divide(a: &Array, b: &Array) -> Result<Array> {
    ok(a.0.floor_divide(&b.0))
}

pub fn where_(cond: &Array, x: &Array, y: &Array) -> Result<Array> {
    ok(ops::where_(&cond.0, &x.0, &y.0))
}

// ---- Indexing and reductions ----

pub fn take(a: &Array, indices: &Array, axis: i32) -> Result<Array> {
    ok(ops::take(&a.0, &indices.0, axis))
}

pub fn take_along_axis(a: &Array, indices: &Array, axis: i32) -> Result<Array> {
    ok(ops::take_along_axis(&a.0, &indices.0, axis))
}

pub fn put_along_axis(a: &Array, indices: &Array, values: &Array, axis: i32) -> Result<Array> {
    ok(ops::put_along_axis(&a.0, &indices.0, &values.0, axis))
}

pub fn argmax(a: &Array, axis: i32, keepdims: bool) -> Result<Array> {
    ok(if keepdims {
        ops::argmax_keepdims(&a.0, axis)
    } else {
        ops::argmax(&a.0, axis)
    })
}

pub fn argsort(a: &Array, axis: i32) -> Result<Array> {
    ok(ops::argsort(&a.0, axis))
}

/// Indices that partition `a` around `kth` along `axis` (a valid partition; the element order
/// within the two sides may differ from MLX's).
pub fn argpartition(a: &Array, kth: i32, axis: i32) -> Result<Array> {
    ok(ops::argpartition(&a.0, kth, axis))
}

/// Cumulative sum along `axis` (`reverse`: from the end; `inclusive`: including the element).
pub fn cumsum(a: &Array, axis: i32, reverse: bool, inclusive: bool) -> Result<Array> {
    let options = nv::ScanOptions {
        reverse,
        exclusive: !inclusive,
    };
    ok(ops::cumsum_with(&a.0, axis, options))
}

macro_rules! reduction {
    ($name:ident, $keep:ident) => {
        pub fn $name(a: &Array, axis: i32, keepdims: bool) -> Result<Array> {
            ok(if keepdims {
                ops::$keep(&a.0, axis)
            } else {
                ops::$name(&a.0, axis)
            })
        }
    };
}

reduction!(sum, sum_keepdims);
reduction!(logsumexp, logsumexp_keepdims);
reduction!(max, max_keepdims);

pub fn sort(a: &Array, axis: i32) -> Result<Array> {
    ok(ops::sort(&a.0, axis))
}

/// Softmax along `axis`. nuvec's softmax is always MLX's `precise=True` (half precision
/// computed in f32 and rounded once), so `precise` changes nothing here.
pub fn softmax(a: &Array, axis: i32, precise: bool) -> Result<Array> {
    let _ = precise;
    ok(ops::softmax(&a.0, axis))
}

// ---- Quantization ----

/// One affine-quantized weight (`nuvec::Quantized`): packed codes, scales and biases with their
/// group size and bits, built once at load. nuvec requires affine biases (as MLX), so a
/// checkpoint without them gets zeros shaped like the scales here, once (`code · scale + 0`).
#[derive(Clone, Debug)]
pub struct Quantized(nv::Quantized);

impl Quantized {
    pub fn new(
        weight: Array,
        scales: Array,
        biases: Option<Array>,
        group_size: i32,
        bits: i32,
    ) -> Result<Self> {
        let biases = match biases {
            Some(b) => b.0,
            None => ops::zeros(scales.0.shape(), scales.0.dtype()).check()?,
        };
        Ok(Self(nv::Quantized::new(
            weight.0,
            scales.0,
            Some(biases),
            nv::Quant::affine(group_size, bits),
        )))
    }

    /// The rows `ids` of weights, scales and biases (embedding lookups).
    pub fn take(&self, ids: &Array) -> Result<Self> {
        let rows = |a: &nv::Array| ok(ops::take(a, &ids.0, 0)).map(|a| a.0);
        Ok(Self(nv::Quantized::new(
            rows(&self.0.weight)?,
            rows(&self.0.scales)?,
            self.0.biases.as_ref().map(rows).transpose()?,
            self.0.quant,
        )))
    }
}

/// `x @ wᵀ` with an affine-quantized `w`.
pub fn quantized_matmul(x: &Array, w: &Quantized) -> Result<Array> {
    ok(x.0.quantized_matmul(&w.0, true))
}

/// `w` restored, in its scales' dtype.
pub fn dequantize(w: &Quantized) -> Result<Array> {
    ok(w.0.dequantize(None))
}

/// `x @ w[rhs_indices]ᵀ` per batch element with affine-quantized expert weights `w`
/// (`[E, out, in·bits/32]`); `sorted_indices` promises ascending indices (faster kernel).
pub fn gather_qmm(
    x: &Array,
    w: &Quantized,
    rhs_indices: &Array,
    sorted_indices: bool,
) -> Result<Array> {
    ok(x.0.gather_qmm(
        &w.0,
        None,
        Some(&rhs_indices.0),
        nv::GatherQmm::default()
            .transpose(true)
            .sorted_indices(sorted_indices),
    ))
}

/// `a @ b[rhs_indices]` per batch element (dense expert weights).
pub fn gather_mm(a: &Array, b: &Array, rhs_indices: &Array, sorted_indices: bool) -> Result<Array> {
    ok(ops::gather_mm(
        &a.0,
        &b.0,
        None,
        Some(&rhs_indices.0),
        sorted_indices,
    ))
}

/// 1-D convolution of `input` `[N, L, C_in]` with `weight` `[C_out, K, C_in / groups]`.
pub fn conv1d(
    input: &Array,
    weight: &Array,
    stride: i32,
    padding: i32,
    dilation: i32,
    groups: i32,
) -> Result<Array> {
    ok(ops::conv1d(
        &input.0, &weight.0, stride, padding, dilation, groups,
    ))
}

// ---- Compilation ----

/// A graph-building function for [`Compiled`]: arrays in, arrays out.
pub type Graph = fn(&[Array]) -> Result<Vec<Array>>;

type CompiledFn = Box<dyn Fn(&[nv::Array]) -> nv::Result<Vec<nv::Array>>>;

/// A function compiled by nuvec (`nuvec::compile_flat_with`, as `mx.compile`): traced once per
/// input signature (at most 64 kept), or once per dtype and rank when `shapeless`, its
/// element-wise chains fused on the GPU. A shapeless graph must not reshape, slice or attend
/// (nuvec refuses those on the first call, naming the op).
pub struct Compiled {
    f: CompiledFn,
}

impl std::fmt::Debug for Compiled {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Compiled")
    }
}

impl Compiled {
    /// Compiles `graph`; `shapeless` reuses one trace for every input shape.
    pub fn new(graph: Graph, shapeless: bool) -> Result<Self> {
        let options = nv::compile::Options::default().shapeless(shapeless);
        let f = nv::compile::compile_flat_with(options, move |inputs: &[nv::Array]| {
            let inputs: Vec<Array> = inputs.iter().cloned().map(Array).collect();
            match graph(&inputs) {
                Ok(outputs) => Ok(outputs.into_iter().map(|a| a.0).collect()),
                Err(Error(message)) => {
                    GRAPH_ERROR.with(|slot| *slot.borrow_mut() = Some(message.clone()));
                    Err(nv::Error::InvalidArgument(message))
                }
            }
        });
        Ok(Self { f: Box::new(f) })
    }

    pub fn call(&self, inputs: &[&Array]) -> Result<Vec<Array>> {
        GRAPH_ERROR.with(|slot| slot.borrow_mut().take());
        match (self.f)(&raw(inputs))
            .and_then(|outputs| outputs.into_iter().map(nv::Array::check).collect())
        {
            Ok(outputs) => Ok(wrap(outputs)),
            // The graph's own error, not nuvec's wrapping of it.
            Err(e) => Err(GRAPH_ERROR
                .with(|slot| slot.borrow_mut().take())
                .map_or_else(|| Error::from(e), Error)),
        }
    }
}

// ---- Random ----

pub fn random_key(seed: u64) -> Result<Array> {
    ok(nv::random::key(seed))
}

/// Samples indices from unnormalized log-probabilities along `axis`.
pub fn categorical(logits: &Array, axis: i32, key: &Array) -> Result<Array> {
    ok(nv::random::categorical(&logits.0, axis, &key.0))
}

/// Fused kernels (`nuvec::fast`, as `mlx.core.fast`).
pub mod fast {
    use super::*;

    /// RMS normalization over the last axis, scaled by `weight` when given.
    pub fn rms_norm(x: &Array, weight: Option<&Array>, eps: f32) -> Result<Array> {
        ok(nv::fast::rms_norm(&x.0, weight.map(|w| &w.0), eps))
    }

    /// `x + residual` and its RMS norm in one kernel (nuvec-lm's residual fusion): returns the
    /// sum (the residual stream) and the normalised sum, bit for bit equal to [`super::add`] then
    /// [`rms_norm`] (nuvec asserts this with zero tolerance in F32, F16 and BF16).
    pub fn add_rms_norm(
        x: &Array,
        residual: &Array,
        weight: Option<&Array>,
        eps: f32,
    ) -> Result<(Array, Array)> {
        let (h, y) = nv::fast::add_rms_norm(&x.0, &residual.0, weight.map(|w| &w.0), eps);
        Ok((ok(h)?, ok(y)?))
    }

    /// mlx-lm's gated delta rule (`gated_delta_update` without mask): `q, k [B, T, Hk, Dk]`,
    /// `v [B, T, Hv, Dv]`, `g, beta [B, T, Hv]`, `state [B, Hv, Dv, Dk]`; returns `y [B, T, Hv,
    /// Dv]` (in `q`'s dtype) and the new state (in `state`'s dtype). nuvec runs mlx-lm's generic
    /// or packed kernel on the GPU for `Dk` a multiple of 32, else the steps as operations.
    pub fn gated_delta(
        q: &Array,
        k: &Array,
        v: &Array,
        g: &Array,
        beta: &Array,
        state: &Array,
    ) -> Result<(Array, Array)> {
        let (y, s) = nv::fast::gated_delta(&q.0, &k.0, &v.0, &g.0, &beta.0, &state.0);
        Ok((ok(y)?, ok(s)?))
    }

    /// Rotary embedding of the last `dims` features starting at position `offset`.
    pub fn rope(
        x: &Array,
        dims: i32,
        traditional: bool,
        base: Option<f32>,
        scale: f32,
        offset: i32,
        freqs: Option<&Array>,
    ) -> Result<Array> {
        let offset = nv::Array::from(offset);
        ok(nv::fast::rope(
            &x.0,
            dims,
            traditional,
            base,
            scale,
            &offset,
            freqs.map(|f| &f.0),
        ))
    }

    /// Attention mask of [`scaled_dot_product_attention`].
    #[derive(Clone, Copy, Debug)]
    pub enum Mask {
        None,
        /// Causal, aligned to the end of the keys.
        Causal,
    }

    /// The options for `q` and `k` shaped `[B, H, L, D]`. A cap below one query row's scores
    /// (`B · H · L_k`, 4 bytes at most per score) is raised to it: where nuvec has no tiled
    /// kernel it runs blocks of at least one row.
    fn options(
        q: &[i32],
        k: &[i32],
        mask: Mask,
        max_score_bytes: u64,
    ) -> nv::fast::AttentionOptions<'static> {
        let options = match mask {
            Mask::None => nv::fast::AttentionOptions::default(),
            Mask::Causal => nv::fast::AttentionOptions::causal(),
        };
        let dim = |s: &[i32], i: usize| s.get(i).map_or(1, |&d| d.max(1) as u64);
        let row = dim(q, 0) * dim(q, 1) * dim(k, 2) * 4;
        let cap = max_score_bytes.max(row);
        options.max_score_bytes(usize::try_from(cap).unwrap_or(usize::MAX))
    }

    /// The scratch bytes of an attention call with these `[B, H, L, D]` shapes (values shaped
    /// like keys): nuvec's plan for the kernel it will run. `None` when the shapes are invalid.
    pub fn attention_scratch_bytes(
        q: [i32; 4],
        k: [i32; 4],
        dtype: Dtype,
        mask: Mask,
        max_score_bytes: u64,
    ) -> Option<u64> {
        let plan = nv::fast::plan_attention(
            &q,
            &k,
            &k,
            dtype.raw().ok()?,
            &options(&q, &k, mask, max_score_bytes),
        )
        .ok()?;
        Some(plan.scratch_bytes as u64)
    }

    /// `softmax(scale · q kᵀ + mask) v`, its score buffer within `max_score_bytes`.
    pub fn scaled_dot_product_attention(
        q: &Array,
        k: &Array,
        v: &Array,
        scale: f32,
        mask: Mask,
        max_score_bytes: u64,
    ) -> Result<Array> {
        ok(nv::fast::scaled_dot_product_attention(
            &q.0,
            &k.0,
            &v.0,
            scale,
            &options(q.0.shape(), k.0.shape(), mask, max_score_bytes),
        ))
    }
}

/// Memory counters and limits (bytes) of nuvec's GPU allocator; zero and no-ops without Metal.
///
/// nuvec maps checkpoint weights instead of copying them; `active` and `peak` count those
/// mappings (`*_with_mapped`), which is what MLX's active and peak memory mean (MLX copies every
/// weight). The memory limit bounds the same sum and is a hard refusal in nuvec (a guideline in
/// MLX): mrun sets it to its budget, which already includes the weights.
pub mod memory {
    use super::nv::memory;

    pub fn active() -> u64 {
        memory::active_memory() as u64
    }

    pub fn peak() -> u64 {
        memory::peak_memory() as u64
    }

    /// [`active`] without memory-mapped files: bytes the allocator holds (data read or computed).
    pub fn active_allocated() -> u64 {
        memory::stats().active as u64
    }

    /// [`peak`] without memory-mapped files.
    pub fn peak_allocated() -> u64 {
        memory::stats().peak as u64
    }

    pub fn reset_peak() {
        memory::reset_peak_memory();
    }

    pub fn clear_cache() {
        memory::clear_cache();
    }

    /// Sets the limit and returns the previous one.
    pub fn set_memory_limit(limit: u64) -> u64 {
        memory::set_memory_limit(usize::try_from(limit).unwrap_or(usize::MAX)) as u64
    }

    pub fn set_cache_limit(limit: u64) -> u64 {
        memory::set_cache_limit(usize::try_from(limit).unwrap_or(usize::MAX)) as u64
    }

    /// Sets the wired limit and returns the previous one; a limit nuvec refuses (above the
    /// recommended working set) leaves it unchanged and returns the current one.
    pub fn set_wired_limit(limit: u64) -> u64 {
        let bytes = usize::try_from(limit).unwrap_or(usize::MAX);
        memory::set_wired_limit(bytes).unwrap_or_else(|_| memory::stats().wired_limit) as u64
    }
}

/// `(memory_size, max_recommended_working_set_size)` of the GPU, when there is one (Apple
/// Silicon's GPU memory is the physical memory).
pub fn device_memory() -> Option<(u64, u64)> {
    let info = nv::metal::device_info()?;
    Some((info.physical_memory, info.recommended_working_set))
}

/// A string entry of the GPU's device info (`architecture`, `device_name`); `None` without
/// Metal or for another key.
pub fn gpu_info_string(key: &CStr) -> Option<String> {
    let info = nv::metal::device_info()?;
    match key.to_bytes() {
        b"architecture" => Some(info.architecture.clone()),
        b"device_name" => Some(info.name.clone()),
        _ => None,
    }
}

/// The numerics class of this process (FR-027, D044; 016 FR-008, FR-016): backend nuvec, whether
/// nuvec runs its NAX kernels (they change the bits), and the GPU architecture (`cpu` without
/// Metal). Asked once.
pub fn numerics() -> crate::engine::Numerics {
    static NUMERICS: OnceLock<crate::engine::Numerics> = OnceLock::new();
    NUMERICS
        .get_or_init(|| crate::engine::Numerics {
            nax: nv::metal::nax().is_active(),
            gpu_arch: gpu_info_string(c"architecture").unwrap_or_else(|| "cpu".into()),
        })
        .clone()
}

/// Whether this GPU has NAX (M5 Neural Accelerators: GPU generation 17 or later), used or not
/// (`mrun doctor --report`; [`nax_status`] says whether nuvec uses it). `false` without Metal.
pub fn nax_available() -> bool {
    nv::metal::device_info()
        .and_then(|info| info.generation)
        .is_some_and(|generation| generation >= 17)
}

/// The backend of this build (016 FR-008): nuvec, its version and its kernel mode (FR-009: a
/// precompiled `nuvec.metallib`, or shader sources compiled at start).
pub fn backend_info() -> crate::engine::BackendInfo {
    use crate::engine::Kernels;
    crate::engine::BackendInfo {
        version: nv::VERSION,
        kernels: match nuvec_kernels::MODE {
            nuvec_kernels::MetallibMode::Precompiled => Kernels::Precompiled,
            nuvec_kernels::MetallibMode::RuntimeSource => Kernels::Runtime,
            _ => Kernels::Unavailable,
        },
    }
}

/// Whether nuvec runs its NAX kernels, and why not (`mrun doctor --report`): `on`, `on (forced)`
/// (`NUVEC_NAX=on`), or `off (<reason>)` (`off (no metal)` without a GPU).
pub fn nax_status() -> String {
    use nv::metal::{NaxReason, NaxStatus};
    match nv::metal::nax() {
        NaxStatus::Active { forced: false } => "on".into(),
        NaxStatus::Active { forced: true } => "on (forced)".into(),
        NaxStatus::Off(reason) => {
            let reason = match reason {
                NaxReason::NoDevice => return "off (no metal)".into(),
                NaxReason::Override => "NUVEC_NAX=off",
                NaxReason::Os => "macOS before 26.2",
                NaxReason::Generation => "not an M5-class GPU",
                NaxReason::NoLibrary => "no NAX kernels in this build",
                NaxReason::Unmeasured => "not yet measured on this GPU",
                NaxReason::PipelineError => "pipeline error",
                _ => "unknown reason",
            };
            format!("off ({reason})")
        }
        _ => "unknown".into(),
    }
}

/// Whether this build has NAX kernels (nuvec's NAX metallib, or its sources).
pub const NAX_COMPILED: bool = !matches!(nuvec_kernels::NAX_MODE, nuvec_kernels::NaxMode::Absent);

/// Loads every tensor of a `.safetensors` file: mapped when every tensor is aligned to its
/// element size (data read on use), else read here in parallel (mlx-community files).
pub fn load_safetensors(path: &Path) -> Result<Vec<(String, Array)>> {
    load_safetensors_filtered(path, |_| true)
}

/// The tensors of a `.safetensors` file that `keep` accepts; the others are never read.
pub fn load_safetensors_filtered(
    path: &Path,
    keep: impl Fn(&str) -> bool,
) -> Result<Vec<(String, Array)>> {
    let file = nv::io::load_safetensors_filtered(path, keep)?;
    Ok(file
        .tensors
        .into_iter()
        .map(|(name, a)| (name, Array(a)))
        .collect())
}

/// The bytes a filtered load of `dir`'s `*.safetensors` files will hold (aligned files mapped
/// whole, page-rounded; misaligned ones read for the tensors `keep` accepts), from the headers
/// alone. `None` when a header cannot be read.
pub fn weights_footprint(dir: &Path, keep: impl Fn(&str) -> bool) -> Option<u64> {
    let fp = nv::io::safetensors_footprint_filtered(dir, keep).ok()?;
    Some(fp.mapped + fp.copied)
}

/// Writes `tensors` to a `.safetensors` file (tiny test checkpoints).
pub fn save_safetensors(path: &Path, tensors: &[(&str, &Array)]) -> Result<()> {
    let tensors: Vec<(&str, &nv::Array)> = tensors.iter().map(|&(n, a)| (n, &a.0)).collect();
    Ok(nv::io::save_safetensors(path, &tensors, &[])?)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cpu() {
        use_device(Device::Cpu);
    }

    #[test]
    fn errors_become_results() {
        cpu();
        let a = Array::from_f32(&[1.0, 2.0, 3.0], &[3]);
        let b = Array::from_f32(&[1.0, 2.0], &[2]);
        let err = add(&a, &b).and_then(|c| c.eval().map(|()| c)).unwrap_err();
        assert!(
            err.0.contains("broadcast") || err.0.contains("shape"),
            "{err}"
        );
        assert_eq!(add(&a, &a).unwrap().to_vec_f32().unwrap(), [2.0, 4.0, 6.0]);
    }

    #[test]
    fn slices_and_updates() {
        cpu();
        let a = Array::from_f32(&[0.0; 6], &[2, 3]);
        let u = Array::from_f32(&[7.0, 8.0], &[1, 2]);
        let b = slice_update(&a, &u, &[1, 1], &[2, 3]).unwrap();
        assert_eq!(b.to_vec_f32().unwrap(), [0.0, 0.0, 0.0, 0.0, 7.0, 8.0]);
        let s = slice(&b, &[1, 0], &[2, 2]).unwrap();
        assert_eq!(s.shape(), [1, 2]);
        assert_eq!(s.to_vec_f32().unwrap(), [0.0, 7.0]);
    }

    #[test]
    fn deep_copy_does_not_alias() {
        cpu();
        let a = Array::from_f32(&[-0.0, 1.5, f32::NAN], &[3]);
        let b = deep_copy(&a).unwrap();
        assert_ne!(a.buffer_id().unwrap(), b.buffer_id().unwrap());
        let v = b.to_vec_f32().unwrap();
        assert!(v[0] == 0.0 && v[0].is_sign_negative());
        assert_eq!(v[1], 1.5);
        assert!(v[2].is_nan());
    }

    #[test]
    fn argmax_and_logsumexp() {
        cpu();
        let a = Array::from_f32(&[0.1, 3.0, -1.0, 2.0], &[1, 4]);
        assert_eq!(argmax(&a, -1, false).unwrap().item_u32().unwrap(), 1);
        let lse = logsumexp(&a, -1, true).unwrap().item_f32().unwrap();
        let expect = [0.1f32, 3.0, -1.0, 2.0]
            .iter()
            .map(|x| x.exp())
            .sum::<f32>()
            .ln();
        assert!((lse - expect).abs() < 1e-5);
    }

    #[test]
    fn quantized_matmul_matches_dequantized() {
        cpu();
        let q: Vec<u32> = (0..2 * 8)
            .map(|i| 0x7654_3210u32.rotate_left(i * 4))
            .collect();
        let w = Array::from_u32(&q, &[2, 8]);
        let scales = Array::from_f32(&[0.5, 0.25], &[2, 1]);
        let biases = Array::from_f32(&[-1.0, 0.5], &[2, 1]);
        let x = Array::from_f32(
            &(0..64).map(|i| i as f32 / 64.0).collect::<Vec<_>>(),
            &[1, 64],
        );
        let wq = Quantized::new(w.clone(), scales.clone(), Some(biases), 64, 4).unwrap();
        let y = quantized_matmul(&x, &wq).unwrap();
        let dense = dequantize(&wq).unwrap();
        let reference = matmul(&x, &transpose(&dense, &[1, 0]).unwrap()).unwrap();
        let (y, r) = (y.to_vec_f32().unwrap(), reference.to_vec_f32().unwrap());
        for (a, b) in y.iter().zip(&r) {
            assert!((a - b).abs() < 1e-3, "{y:?} vs {r:?}");
        }
        // Rows of a quantized table, as an embedding looks them up.
        let row = dequantize(&wq.take(&Array::from_u32(&[1], &[1])).unwrap()).unwrap();
        assert_eq!(row.to_vec_f32().unwrap(), dense.to_vec_f32().unwrap()[64..]);
        // Without biases: `code · scale`.
        let plain = dequantize(&Quantized::new(w, scales, None, 64, 4).unwrap()).unwrap();
        let codes: Vec<f32> = q[..8]
            .iter()
            .flat_map(|word| (0..8).map(move |i| ((word >> (4 * i)) & 0xf) as f32 * 0.5))
            .collect();
        assert_eq!(plain.to_vec_f32().unwrap()[..64], codes[..]);
    }

    #[test]
    fn buffer_ids_see_shared_storage() {
        cpu();
        let a = add(
            &Array::from_f32(&[1.0; 6], &[2, 3]),
            &Array::scalar_f32(1.0),
        )
        .unwrap();
        let row = slice(&a, &[1, 0], &[2, 3]).unwrap();
        assert_eq!(row.buffer_id().unwrap(), a.buffer_id().unwrap());
        let copy = deep_copy(&a).unwrap();
        assert_ne!(copy.buffer_id().unwrap(), a.buffer_id().unwrap());
    }

    #[test]
    fn scalars_equal_their_casts_bit_for_bit() {
        cpu();
        // Includes bf16/f16 rounding ties and values that are not representable.
        for value in [
            0.0,
            1.0,
            -2.5,
            0.088_388_35,
            1.0 / 3.0,
            1.003_906_3,
            65_519.0,
            7.0,
        ] {
            for dtype in [
                Dtype::Float32,
                Dtype::Bfloat16,
                Dtype::Float16,
                Dtype::Int32,
                Dtype::Uint32,
            ] {
                let direct = scalar(value, dtype).unwrap();
                let cast = astype(&Array::scalar_f32(value), dtype).unwrap();
                assert_eq!(direct.dtype(), dtype);
                let bits = |a: &Array| match dtype {
                    Dtype::Int32 | Dtype::Uint32 => a.to_vec_f32().unwrap()[0].to_bits(),
                    _ => astype(a, Dtype::Float32).unwrap().to_vec_f32().unwrap()[0].to_bits(),
                };
                assert_eq!(bits(&direct), bits(&cast), "{value} {dtype:?}");
            }
        }
    }

    #[test]
    fn sdpa_causal_attends_to_the_past_only() {
        cpu();
        let q = Array::from_f32(&[0.0; 2], &[1, 1, 2, 1]);
        let k = Array::from_f32(&[0.0; 2], &[1, 1, 2, 1]);
        let v = Array::from_f32(&[1.0, 3.0], &[1, 1, 2, 1]);
        let o = fast::scaled_dot_product_attention(&q, &k, &v, 1.0, fast::Mask::Causal, u64::MAX)
            .unwrap();
        assert_eq!(o.to_vec_f32().unwrap(), [1.0, 2.0]);
    }

    #[test]
    fn shape_ops() {
        cpu();
        let a = Array::from_f32(&(0..12).map(|i| i as f32).collect::<Vec<_>>(), &[2, 6]);
        let parts = split_sections(&a, &[2, 5], 1).unwrap();
        let shapes: Vec<_> = parts.iter().map(Array::shape).collect();
        assert_eq!(shapes, [vec![2, 2], vec![2, 3], vec![2, 1]]);
        assert_eq!(parts[2].to_vec_f32().unwrap(), [5.0, 11.0]);
        let halves = split(&a, 2, 1).unwrap();
        assert_eq!(
            halves[1].to_vec_f32().unwrap(),
            [3.0, 4.0, 5.0, 9.0, 10.0, 11.0]
        );
        let e = expand_dims(&a, &[-2, -3]).unwrap();
        assert_eq!(e.shape(), [2, 1, 1, 6]);
        assert_eq!(squeeze(&e, &[1, 2]).unwrap().shape(), [2, 6]);
        let r = repeat(&Array::from_f32(&[1.0, 2.0], &[1, 2, 1]), 2, 1).unwrap();
        assert_eq!(r.to_vec_f32().unwrap(), [1.0, 1.0, 2.0, 2.0]);
        let s = stack(&[&parts[2], &parts[2]], 0).unwrap();
        assert_eq!(s.shape(), [2, 2, 1]);
        let q = floor_divide(
            &Array::from_u32(&[7, 8, 9], &[3]),
            &scalar(4.0, Dtype::Uint32).unwrap(),
        );
        assert_eq!(q.unwrap().to_vec_u32().unwrap(), [1, 2, 2]);
        let t = transpose(&a, &[-1, 0]).unwrap();
        assert_eq!(t.shape(), [6, 2]);
    }

    #[test]
    fn floor_divide_rounds_down() {
        cpu();
        let a = Array::from_f32(&[7.0, -7.0, 7.0, -7.0, 6.0], &[5]);
        let b = Array::from_f32(&[2.0, 2.0, -2.0, -2.0, 3.0], &[5]);
        let q = astype(&a, Dtype::Int32)
            .and_then(|a| floor_divide(&a, &astype(&b, Dtype::Int32)?)?.to_vec_f32());
        assert_eq!(q.unwrap(), [3.0, -4.0, -4.0, 3.0, 2.0]);
        let big: Vec<u32> = (0..64).map(|i| (1 << 23) + i * 8191).collect();
        let k = scalar(8.0, Dtype::Uint32).unwrap();
        let got = floor_divide(&Array::from_u32(&big, &[64]), &k)
            .unwrap()
            .to_vec_u32()
            .unwrap();
        let want: Vec<u32> = big.iter().map(|v| v / 8).collect();
        assert_eq!(got, want);
    }

    #[test]
    fn precise_softmax_rounds_once() {
        cpu();
        let a = astype(&Array::from_f32(&[0.5, 1.0, 2.0], &[1, 3]), Dtype::Bfloat16).unwrap();
        let p = softmax(&a, -1, true).unwrap();
        assert_eq!(p.dtype(), Dtype::Bfloat16);
        let sum: f32 = p.to_vec_f32().unwrap().iter().sum();
        assert!((sum - 1.0).abs() < 1e-2, "{sum}");
    }

    #[test]
    fn depthwise_conv1d() {
        cpu();
        let x = Array::from_f32(&[1.0, 10.0, 2.0, 20.0, 3.0, 30.0], &[1, 3, 2]);
        let w = Array::from_f32(&[1.0, 2.0, 0.5, -1.0], &[2, 2, 1]);
        let y = conv1d(&x, &w, 1, 0, 1, 2).unwrap();
        assert_eq!(y.shape(), [1, 2, 2]);
        assert_eq!(y.to_vec_f32().unwrap(), [5.0, -15.0, 8.0, -20.0]);
    }

    #[test]
    fn gathered_matmuls_pick_each_rows_expert() {
        cpu();
        let eye: Vec<f32> = (0..64 * 64)
            .map(|i| f32::from(u8::from(i % 65 == 0)))
            .collect();
        let experts: Vec<f32> = (0..3)
            .flat_map(|e| eye.iter().map(move |v| v * (e + 1) as f32))
            .collect();
        let w = Array::from_f32(&experts, &[3, 64, 64]);
        let x = Array::from_f32(
            &(0..2 * 64).map(|i| (i % 7) as f32).collect::<Vec<_>>(),
            &[2, 1, 64],
        );
        let idx = Array::from_u32(&[2, 0], &[2]);
        let dense = gather_mm(&x, &transpose(&w, &[0, 2, 1]).unwrap(), &idx, true).unwrap();
        assert_eq!(dense.shape(), [2, 1, 64]);
        let d = dense.to_vec_f32().unwrap();
        let xs = x.to_vec_f32().unwrap();
        assert_eq!(d[5], 3.0 * xs[5]);
        assert_eq!(d[64 + 5], xs[64 + 5]);
        let wq = Quantized(ops::quantize(&w.0, nv::Quant::affine(64, 8)));
        let q = gather_qmm(&x, &wq, &idx, false).unwrap();
        for (a, b) in q.to_vec_f32().unwrap().iter().zip(&d) {
            assert!((a - b).abs() < 1e-3, "{a} vs {b}");
        }
    }

    #[test]
    fn compiled_graphs_run_and_report_errors() {
        cpu();
        let graph: Graph = |a| Ok(vec![multiply(&add(&a[0], &a[1])?, &a[1])?]);
        let compiled = Compiled::new(graph, true).unwrap();
        let x = Array::from_f32(&[1.0, 2.0], &[2]);
        let y = Array::from_f32(&[3.0, 4.0], &[2]);
        let out = compiled.call(&[&x, &y]).unwrap();
        assert_eq!(out[0].to_vec_f32().unwrap(), [12.0, 24.0]);
        let z = Array::from_f32(&[1.0; 3], &[3]);
        assert_eq!(
            compiled.call(&[&z, &z]).unwrap()[0].to_vec_f32().unwrap(),
            [2.0; 3]
        );
        let failing: Graph = |a| Ok(vec![add(&a[0], &a[1])?]);
        let failing = Compiled::new(failing, false).unwrap();
        assert!(failing.call(&[&x, &z]).is_err());
        let refusing: Graph = |_| Err(Error("refused by the graph".into()));
        let refusing = Compiled::new(refusing, false).unwrap();
        let error = refusing.call(&[&x]).unwrap_err();
        assert_eq!(error.0, "refused by the graph");
    }

    #[test]
    fn safetensors_round_trip() {
        cpu();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("t.safetensors");
        let a = Array::from_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let b = Array::from_u32(&[7], &[1]);
        save_safetensors(&path, &[("a", &a), ("b", &b)]).unwrap();
        let loaded = load_safetensors(&path).unwrap();
        let names: Vec<&str> = loaded.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(names, ["a", "b"]);
        assert_eq!(loaded[0].1.to_vec_f32().unwrap(), [1.0, 2.0, 3.0, 4.0]);
        assert_eq!(loaded[1].1.dtype(), Dtype::Uint32);
    }

    #[test]
    fn filtered_loads_skip_tensors() {
        cpu();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("t.safetensors");
        let a = Array::from_f32(&[1.0, 2.0], &[2]);
        save_safetensors(&path, &[("keep.weight", &a), ("vision_tower.w", &a)]).unwrap();
        let loaded = load_safetensors_filtered(&path, |name| !name.starts_with("vision")).unwrap();
        let names: Vec<&str> = loaded.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(names, ["keep.weight"]);
    }

    #[test]
    fn runtime_rank_slices() {
        cpu();
        let a = Array::from_f32(&(0..24).map(|i| i as f32).collect::<Vec<_>>(), &[2, 3, 4]);
        let s = slice(&a, &[1, 0, 2], &[2, 2, 4]).unwrap();
        assert_eq!(s.shape(), [1, 2, 2]);
        assert_eq!(s.to_vec_f32().unwrap(), [14.0, 15.0, 18.0, 19.0]);
        let u = Array::from_f32(&[-1.0; 4], &[1, 2, 2]);
        let b = slice_update(&a, &u, &[1, 0, 2], &[2, 2, 4]).unwrap();
        assert_eq!(
            slice(&b, &[1, 0, 2], &[2, 2, 4])
                .unwrap()
                .to_vec_f32()
                .unwrap(),
            [-1.0; 4]
        );
        assert!(
            slice(&a, &[0, 0], &[1]).is_err(),
            "starts and stops must pair up"
        );
    }

    #[test]
    fn the_backend_reports_nuvec_its_version_and_kernel_mode() {
        use crate::engine::Kernels;
        let info = backend_info();
        assert_eq!(info.version, nv::VERSION);
        // FR-009: release builds set NUVEC_REQUIRE_METALLIB=1, which makes nuvec's build fail
        // rather than fall back to runtime-compiled sources.
        if option_env!("NUVEC_REQUIRE_METALLIB") == Some("1") {
            assert_eq!(info.kernels, Kernels::Precompiled);
        }
        if cfg!(target_os = "macos") {
            assert_ne!(info.kernels, Kernels::Unavailable);
        }
    }

    #[test]
    fn numerics_name_the_backend() {
        let n = numerics();
        assert!(!n.nax);
        assert!(n.key().ends_with("/nuvec"), "{n}");
    }

    #[test]
    fn dtypes_round_trip() {
        for d in [
            Dtype::Bool,
            Dtype::Uint8,
            Dtype::Uint32,
            Dtype::Int32,
            Dtype::Float16,
            Dtype::Float32,
            Dtype::Bfloat16,
        ] {
            assert_eq!(Dtype::from_raw(d.raw().unwrap()), d);
            assert_eq!(d.size(), d.raw().unwrap().size() as u64);
        }
        let other = Dtype::from_raw(DType::I64);
        assert_eq!(other.size(), 8);
        assert_eq!(other.raw().unwrap(), DType::I64);
        assert!(Dtype::Other(99).raw().is_err());
    }
}
