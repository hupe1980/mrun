//! Thin safe layer over `mlx-sys` (mlx-c): an owning [`Array`], the ops the model families and
//! the sampler use, `fast::*`, evaluation, memory and device queries, and safetensors loading.
//!
//! mlx-c reports failures through a global error handler; ours stores the message in a
//! thread-local slot and every wrapper turns a non-zero return code into [`Error`]. Arrays are
//! `!Send`: all MLX state lives on the engine's worker thread, which also owns the thread's
//! default stream (GPU when MLX was built with Metal and a device is present, else CPU).

#![allow(unsafe_code)]

use std::cell::{Cell, RefCell};
use std::ffi::{CStr, CString, c_char, c_int, c_void};
use std::path::Path;
use std::sync::Once;

use mlx_sys as sys;

mod gated_delta;
pub mod metallib;

/// An MLX failure: the message mlx-c passed to the error handler.
#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
#[error("mlx: {0}")]
pub struct Error(pub String);

pub type Result<T> = std::result::Result<T, Error>;

thread_local! {
    static LAST_ERROR: RefCell<Option<String>> = const { RefCell::new(None) };
    static STREAM: RefCell<Option<OwnedStream>> = const { RefCell::new(None) };
    static DEVICE: Cell<Device> = const { Cell::new(Device::Default) };
}

static HANDLER: Once = Once::new();

thread_local! {
    /// The error a compiled graph returned, which mlx-c replaces with a generic message.
    static GRAPH_ERROR: RefCell<Option<String>> = const { RefCell::new(None) };
}

unsafe extern "C" fn record_error(msg: *const c_char, _data: *mut c_void) {
    let text = if msg.is_null() {
        "unknown error".to_string()
    } else {
        // SAFETY: mlx-c passes a NUL-terminated message valid for the duration of the call.
        unsafe { CStr::from_ptr(msg) }
            .to_string_lossy()
            .into_owned()
    };
    LAST_ERROR.with(|slot| *slot.borrow_mut() = Some(text));
}

/// Installs the error handler (before any other mlx-c call, [`metallib::ensure`] included).
fn install_handler() {
    HANDLER.call_once(|| {
        // SAFETY: installs a handler with no user data; it only writes a thread-local.
        unsafe { sys::mlx_set_error_handler(Some(record_error), std::ptr::null_mut(), None) };
    });
}

fn init() {
    install_handler();
    // A failure is kept: `MlxEngine::load` reports it before the first GPU operation.
    let _ = metallib::ensure();
}

fn check(code: c_int) -> Result<()> {
    if code == 0 {
        return Ok(());
    }
    let message = LAST_ERROR
        .with(|slot| slot.borrow_mut().take())
        .unwrap_or_else(|| format!("mlx-c call failed with code {code}"));
    Err(Error(message))
}

/// Where this thread's operations run.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Device {
    /// GPU when available, else CPU.
    Default,
    Cpu,
}

/// Selects the device for operations on this thread; call before the first operation.
pub fn use_device(device: Device) {
    DEVICE.with(|d| d.set(device));
    STREAM.with(|s| s.borrow_mut().take());
}

/// A stream this thread created on its device; freed when the thread ends or the device changes.
struct OwnedStream(sys::mlx_stream);

impl Drop for OwnedStream {
    fn drop(&mut self) {
        // SAFETY: the stream was created by `stream()` and is freed exactly once.
        unsafe { sys::mlx_stream_free(self.0) };
    }
}

/// Whether MLX was built with Metal and a Metal device is present (asked once).
pub fn metal_available() -> bool {
    static AVAILABLE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *AVAILABLE.get_or_init(|| {
        init();
        if !sys::METAL {
            return false;
        }
        let mut available = false;
        // SAFETY: writes one bool.
        unsafe { sys::mlx_metal_is_available(&mut available) == 0 && available }
    })
}

/// Whether this thread's operations run on the GPU (Metal present and not [`Device::Cpu`]).
pub fn on_gpu() -> bool {
    DEVICE.with(Cell::get) != Device::Cpu && metal_available()
}

/// This thread's stream: a dedicated stream on the GPU when Metal is available (CPU otherwise or
/// when [`Device::Cpu`] was selected), as mlx-lm's `new_thread_local_stream` does. Separate
/// threads never share a stream.
fn stream() -> sys::mlx_stream {
    init();
    STREAM.with(|slot| {
        if let Some(stream) = &*slot.borrow() {
            return stream.0;
        }
        let cpu = DEVICE.with(Cell::get) == Device::Cpu || !metal_available();
        let kind = if cpu {
            sys::mlx_device_type__MLX_CPU
        } else {
            sys::mlx_device_type__MLX_GPU
        };
        // SAFETY: creates a device handle and a new stream on it; the device handle is freed
        // here, the stream by `OwnedStream`.
        let stream = unsafe {
            let device = sys::mlx_device_new_type(kind, 0);
            let stream = sys::mlx_stream_new_device(device);
            sys::mlx_device_free(device);
            stream
        };
        *slot.borrow_mut() = Some(OwnedStream(stream));
        stream
    })
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
    Other(u32),
}

impl Dtype {
    /// Bytes per element.
    pub fn size(self) -> u64 {
        match self {
            Self::Bool | Self::Uint8 => 1,
            Self::Float16 | Self::Bfloat16 => 2,
            Self::Uint32 | Self::Int32 | Self::Float32 => 4,
            Self::Other(_) => 8,
        }
    }

    fn raw(self) -> sys::mlx_dtype {
        match self {
            Self::Bool => sys::mlx_dtype__MLX_BOOL,
            Self::Uint8 => sys::mlx_dtype__MLX_UINT8,
            Self::Uint32 => sys::mlx_dtype__MLX_UINT32,
            Self::Int32 => sys::mlx_dtype__MLX_INT32,
            Self::Float16 => sys::mlx_dtype__MLX_FLOAT16,
            Self::Float32 => sys::mlx_dtype__MLX_FLOAT32,
            Self::Bfloat16 => sys::mlx_dtype__MLX_BFLOAT16,
            Self::Other(raw) => raw,
        }
    }

    fn from_raw(raw: sys::mlx_dtype) -> Self {
        match raw {
            sys::mlx_dtype__MLX_BOOL => Self::Bool,
            sys::mlx_dtype__MLX_UINT8 => Self::Uint8,
            sys::mlx_dtype__MLX_UINT32 => Self::Uint32,
            sys::mlx_dtype__MLX_INT32 => Self::Int32,
            sys::mlx_dtype__MLX_FLOAT16 => Self::Float16,
            sys::mlx_dtype__MLX_FLOAT32 => Self::Float32,
            sys::mlx_dtype__MLX_BFLOAT16 => Self::Bfloat16,
            other => Self::Other(other),
        }
    }
}

/// An owned MLX array handle (reference-counted on the MLX side; `Clone` shares the data).
pub struct Array {
    raw: sys::mlx_array,
}

impl std::fmt::Debug for Array {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Array({:?}, {:?})", self.shape(), self.dtype())
    }
}

impl Drop for Array {
    fn drop(&mut self) {
        // SAFETY: `raw` is owned by this value and freed exactly once.
        unsafe { sys::mlx_array_free(self.raw) };
    }
}

impl Clone for Array {
    fn clone(&self) -> Self {
        let mut out = Self::null();
        // SAFETY: `set` takes another reference to the same array.
        unsafe { sys::mlx_array_set(&mut out.raw, self.raw) };
        out
    }
}

impl Array {
    /// A null handle: "no array" for optional mlx-c arguments and the slot an op writes into.
    /// Never handed out: every public `Array` holds data.
    fn null() -> Self {
        init();
        // SAFETY: returns a handle with a null context.
        Self {
            raw: unsafe { sys::mlx_array_new() },
        }
    }

    fn from_data<T: Copy>(data: &[T], shape: &[i32], dtype: Dtype) -> Self {
        init();
        let size: i64 = shape.iter().map(|&d| i64::from(d)).product();
        assert_eq!(
            size as usize,
            data.len(),
            "data does not match shape {shape:?}"
        );
        // SAFETY: mlx copies `data`, whose length matches `shape`; both outlive the call.
        Self {
            raw: unsafe {
                sys::mlx_array_new_data(
                    data.as_ptr().cast(),
                    shape.as_ptr(),
                    shape.len() as c_int,
                    dtype.raw(),
                )
            },
        }
    }

    pub fn from_u8(data: &[u8], shape: &[i32]) -> Self {
        Self::from_data(data, shape, Dtype::Uint8)
    }

    pub fn from_u32(data: &[u32], shape: &[i32]) -> Self {
        Self::from_data(data, shape, Dtype::Uint32)
    }

    pub fn from_f32(data: &[f32], shape: &[i32]) -> Self {
        Self::from_data(data, shape, Dtype::Float32)
    }

    pub fn from_bool(data: &[bool], shape: &[i32]) -> Self {
        Self::from_data(data, shape, Dtype::Bool)
    }

    pub fn scalar_f32(value: f32) -> Self {
        init();
        // SAFETY: creates a new scalar array.
        Self {
            raw: unsafe { sys::mlx_array_new_float32(value) },
        }
    }

    pub fn scalar_i32(value: i32) -> Self {
        init();
        // SAFETY: creates a new scalar array.
        Self {
            raw: unsafe { sys::mlx_array_new_int(value) },
        }
    }

    pub fn shape(&self) -> Vec<i32> {
        // SAFETY: the shape pointer is valid for `ndim` elements while `self` lives.
        unsafe {
            let ndim = sys::mlx_array_ndim(self.raw);
            let ptr = sys::mlx_array_shape(self.raw);
            if ndim == 0 || ptr.is_null() {
                return Vec::new();
            }
            std::slice::from_raw_parts(ptr, ndim).to_vec()
        }
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
        // SAFETY: reads metadata of a live array.
        Dtype::from_raw(unsafe { sys::mlx_array_dtype(self.raw) })
    }

    pub fn size(&self) -> usize {
        // SAFETY: reads metadata of a live array.
        unsafe { sys::mlx_array_size(self.raw) }
    }

    /// Evaluates the array (blocking).
    pub fn eval(&self) -> Result<()> {
        // SAFETY: evaluates a live array.
        check(unsafe { sys::mlx_array_eval(self.raw) })
    }

    /// The single element as `u32` (evaluates first).
    pub fn item_u32(&self) -> Result<u32> {
        let a = astype(self, Dtype::Uint32)?;
        let mut out = 0u32;
        // SAFETY: writes one element of an evaluated scalar.
        check(unsafe { sys::mlx_array_item_uint32(&mut out, a.raw) })?;
        Ok(out)
    }

    /// The single element as `f32` (evaluates first).
    pub fn item_f32(&self) -> Result<f32> {
        let a = astype(self, Dtype::Float32)?;
        let mut out = 0f32;
        // SAFETY: writes one element of an evaluated scalar.
        check(unsafe { sys::mlx_array_item_float32(&mut out, a.raw) })?;
        Ok(out)
    }

    /// All elements as `f32`, row-major (evaluates first).
    pub fn to_vec_f32(&self) -> Result<Vec<f32>> {
        let a = contiguous(&astype(self, Dtype::Float32)?)?;
        a.eval()?;
        // SAFETY: an evaluated contiguous f32 array holds `size` elements at `data`.
        unsafe {
            let ptr = sys::mlx_array_data_float32(a.raw);
            if ptr.is_null() {
                return Err(Error("array has no data".into()));
            }
            Ok(std::slice::from_raw_parts(ptr, a.size()).to_vec())
        }
    }

    /// All elements as `u32`, row-major (evaluates first).
    pub fn to_vec_u32(&self) -> Result<Vec<u32>> {
        let a = contiguous(&astype(self, Dtype::Uint32)?)?;
        a.eval()?;
        // SAFETY: an evaluated contiguous u32 array holds `size` elements at `data`.
        unsafe {
            let ptr = sys::mlx_array_data_uint32(a.raw);
            if ptr.is_null() {
                return Err(Error("array has no data".into()));
            }
            Ok(std::slice::from_raw_parts(ptr, a.size()).to_vec())
        }
    }

    /// The identity of the evaluated array's storage (its data address), for aliasing checks in
    /// debug builds: equal for arrays that share a buffer.
    pub fn buffer_id(&self) -> Result<usize> {
        self.eval()?;
        // SAFETY: reads the data pointer of an evaluated array without dereferencing it.
        Ok(unsafe { sys::mlx_array_data_uint8(self.raw) } as usize)
    }
}

/// Runs an mlx-c op that writes one array.
fn op(f: impl FnOnce(*mut sys::mlx_array, sys::mlx_stream) -> c_int) -> Result<Array> {
    LAST_ERROR.with(|slot| slot.borrow_mut().take());
    let mut out = Array::null();
    check(f(&mut out.raw, stream()))?;
    Ok(out)
}

struct VectorArray(sys::mlx_vector_array);

impl VectorArray {
    fn new(arrays: &[&Array]) -> Self {
        // SAFETY: a new vector; each append takes a reference to a live array.
        unsafe {
            let vec = sys::mlx_vector_array_new();
            for a in arrays {
                sys::mlx_vector_array_append_value(vec, a.raw);
            }
            Self(vec)
        }
    }
}

impl VectorArray {
    fn empty() -> Self {
        // SAFETY: a new empty vector, freed on drop.
        Self(unsafe { sys::mlx_vector_array_new() })
    }

    /// The arrays of the vector, each an owned reference.
    fn to_arrays(&self) -> Vec<Array> {
        // SAFETY: reads `size` elements of a live vector; each `get` takes a new reference.
        unsafe {
            (0..sys::mlx_vector_array_size(self.0))
                .map(|i| {
                    let mut out = Array::null();
                    sys::mlx_vector_array_get(&mut out.raw, self.0, i);
                    out
                })
                .collect()
        }
    }
}

/// Runs an mlx-c op that writes a vector of arrays.
fn op_vec(
    f: impl FnOnce(*mut sys::mlx_vector_array, sys::mlx_stream) -> c_int,
) -> Result<Vec<Array>> {
    LAST_ERROR.with(|slot| slot.borrow_mut().take());
    let mut out = VectorArray::empty();
    check(f(&mut out.0, stream()))?;
    Ok(out.to_arrays())
}

impl Drop for VectorArray {
    fn drop(&mut self) {
        // SAFETY: frees the vector created in `new`.
        unsafe { sys::mlx_vector_array_free(self.0) };
    }
}

/// Evaluates `arrays` (blocking).
pub fn eval(arrays: &[&Array]) -> Result<()> {
    init();
    let vec = VectorArray::new(arrays);
    // SAFETY: evaluates live arrays.
    check(unsafe { sys::mlx_eval(vec.0) })
}

/// Schedules evaluation of `arrays` and returns at once.
pub fn async_eval(arrays: &[&Array]) -> Result<()> {
    init();
    let vec = VectorArray::new(arrays);
    // SAFETY: schedules evaluation of live arrays.
    check(unsafe { sys::mlx_async_eval(vec.0) })
}

fn optional_int(value: Option<i32>) -> sys::mlx_optional_int {
    sys::mlx_optional_int {
        value: value.unwrap_or(0),
        has_value: value.is_some(),
    }
}

// ---- Shape and type ----

pub fn reshape(a: &Array, shape: &[i32]) -> Result<Array> {
    // SAFETY: valid handles and a shape slice that outlives the call.
    op(|r, s| unsafe { sys::mlx_reshape(r, a.raw, shape.as_ptr(), shape.len(), s) })
}

pub fn transpose(a: &Array, axes: &[i32]) -> Result<Array> {
    // SAFETY: valid handles and an axes slice that outlives the call.
    op(|r, s| unsafe { sys::mlx_transpose_axes(r, a.raw, axes.as_ptr(), axes.len(), s) })
}

pub fn astype(a: &Array, dtype: Dtype) -> Result<Array> {
    // SAFETY: valid handles.
    op(|r, s| unsafe { sys::mlx_astype(r, a.raw, dtype.raw(), s) })
}

pub fn contiguous(a: &Array) -> Result<Array> {
    // SAFETY: valid handles.
    op(|r, s| unsafe { sys::mlx_contiguous(r, a.raw, false, s) })
}

/// `a[start..stop]` per axis (unit strides).
pub fn slice(a: &Array, start: &[i32], stop: &[i32]) -> Result<Array> {
    let strides = vec![1; start.len()];
    // SAFETY: valid handles and slices that outlive the call.
    op(|r, s| unsafe {
        sys::mlx_slice(
            r,
            a.raw,
            start.as_ptr(),
            start.len(),
            stop.as_ptr(),
            stop.len(),
            strides.as_ptr(),
            strides.len(),
            s,
        )
    })
}

/// `a` with `a[start..stop] = update`.
pub fn slice_update(a: &Array, update: &Array, start: &[i32], stop: &[i32]) -> Result<Array> {
    let strides = vec![1; start.len()];
    // SAFETY: valid handles and slices that outlive the call.
    op(|r, s| unsafe {
        sys::mlx_slice_update(
            r,
            a.raw,
            update.raw,
            start.as_ptr(),
            start.len(),
            stop.as_ptr(),
            stop.len(),
            strides.as_ptr(),
            strides.len(),
            s,
        )
    })
}

/// `a` with size-1 axes inserted at `axes`.
pub fn expand_dims(a: &Array, axes: &[i32]) -> Result<Array> {
    // SAFETY: valid handles and an axes slice that outlives the call.
    op(|r, s| unsafe { sys::mlx_expand_dims_axes(r, a.raw, axes.as_ptr(), axes.len(), s) })
}

/// `a` without the size-1 axes `axes`.
pub fn squeeze(a: &Array, axes: &[i32]) -> Result<Array> {
    // SAFETY: valid handles and an axes slice that outlives the call.
    op(|r, s| unsafe { sys::mlx_squeeze_axes(r, a.raw, axes.as_ptr(), axes.len(), s) })
}

/// Splits `a` along `axis` before each of `indices`.
pub fn split_sections(a: &Array, indices: &[i32], axis: i32) -> Result<Vec<Array>> {
    // SAFETY: valid handles and an index slice that outlives the call.
    op_vec(|r, s| unsafe {
        sys::mlx_split_sections(r, a.raw, indices.as_ptr(), indices.len(), axis, s)
    })
}

/// Splits `a` along `axis` into `parts` equal parts.
pub fn split(a: &Array, parts: i32, axis: i32) -> Result<Vec<Array>> {
    // SAFETY: valid handles.
    op_vec(|r, s| unsafe { sys::mlx_split(r, a.raw, parts, axis, s) })
}

/// Each element of `a` repeated `repeats` times along `axis`.
pub fn repeat(a: &Array, repeats: i32, axis: i32) -> Result<Array> {
    // SAFETY: valid handles.
    op(|r, s| unsafe { sys::mlx_repeat_axis(r, a.raw, repeats, axis, s) })
}

pub fn stack(arrays: &[&Array], axis: i32) -> Result<Array> {
    let vec = VectorArray::new(arrays);
    // SAFETY: valid handles.
    op(|r, s| unsafe { sys::mlx_stack_axis(r, vec.0, axis, s) })
}

pub fn zeros(shape: &[i32], dtype: Dtype) -> Result<Array> {
    // SAFETY: a shape slice that outlives the call.
    op(|r, s| unsafe { sys::mlx_zeros(r, shape.as_ptr(), shape.len(), dtype.raw(), s) })
}

pub fn concatenate(arrays: &[&Array], axis: i32) -> Result<Array> {
    let vec = VectorArray::new(arrays);
    // SAFETY: valid handles.
    op(|r, s| unsafe { sys::mlx_concatenate_axis(r, vec.0, axis, s) })
}

/// A copy that never shares its buffer with `a` (`a × 1`, exact for every value including
/// −0 and NaN); MLX's `copy` may share buffers.
pub fn deep_copy(a: &Array) -> Result<Array> {
    let one = astype(&Array::scalar_f32(1.0), a.dtype())?;
    multiply(a, &one)
}

/// The scalar `value` in `dtype` (via f32, as MLX converts Python floats).
pub fn scalar(value: f32, dtype: Dtype) -> Result<Array> {
    astype(&Array::scalar_f32(value), dtype)
}

// ---- Arithmetic ----

macro_rules! binary {
    ($name:ident, $sys:ident) => {
        pub fn $name(a: &Array, b: &Array) -> Result<Array> {
            // SAFETY: valid handles.
            op(|r, s| unsafe { sys::$sys(r, a.raw, b.raw, s) })
        }
    };
}

macro_rules! unary {
    ($name:ident, $sys:ident) => {
        pub fn $name(a: &Array) -> Result<Array> {
            // SAFETY: valid handles.
            op(|r, s| unsafe { sys::$sys(r, a.raw, s) })
        }
    };
}

binary!(add, mlx_add);
binary!(subtract, mlx_subtract);
binary!(multiply, mlx_multiply);
binary!(divide, mlx_divide);
binary!(matmul, mlx_matmul);
binary!(greater_equal, mlx_greater_equal);
binary!(floor_divide, mlx_floor_divide);
binary!(logaddexp, mlx_logaddexp);
unary!(sigmoid, mlx_sigmoid);
unary!(exp, mlx_exp);
unary!(log, mlx_log);
unary!(negative, mlx_negative);

pub fn where_(cond: &Array, x: &Array, y: &Array) -> Result<Array> {
    // SAFETY: valid handles.
    op(|r, s| unsafe { sys::mlx_where(r, cond.raw, x.raw, y.raw, s) })
}

// ---- Indexing and reductions ----

pub fn take(a: &Array, indices: &Array, axis: i32) -> Result<Array> {
    // SAFETY: valid handles.
    op(|r, s| unsafe { sys::mlx_take_axis(r, a.raw, indices.raw, axis, s) })
}

pub fn take_along_axis(a: &Array, indices: &Array, axis: i32) -> Result<Array> {
    // SAFETY: valid handles.
    op(|r, s| unsafe { sys::mlx_take_along_axis(r, a.raw, indices.raw, axis, s) })
}

pub fn put_along_axis(a: &Array, indices: &Array, values: &Array, axis: i32) -> Result<Array> {
    // SAFETY: valid handles.
    op(|r, s| unsafe { sys::mlx_put_along_axis(r, a.raw, indices.raw, values.raw, axis, s) })
}

pub fn argmax(a: &Array, axis: i32, keepdims: bool) -> Result<Array> {
    // SAFETY: valid handles.
    op(|r, s| unsafe { sys::mlx_argmax_axis(r, a.raw, axis, keepdims, s) })
}

pub fn argsort(a: &Array, axis: i32) -> Result<Array> {
    // SAFETY: valid handles.
    op(|r, s| unsafe { sys::mlx_argsort_axis(r, a.raw, axis, s) })
}

pub fn argpartition(a: &Array, kth: i32, axis: i32) -> Result<Array> {
    // SAFETY: valid handles.
    op(|r, s| unsafe { sys::mlx_argpartition_axis(r, a.raw, kth, axis, s) })
}

/// Cumulative sum along `axis` (`reverse`: from the end; `inclusive`: including the element).
pub fn cumsum(a: &Array, axis: i32, reverse: bool, inclusive: bool) -> Result<Array> {
    let none = sys::mlx_optional_dtype {
        value: 0,
        has_value: false,
    };
    // SAFETY: valid handles.
    op(|r, s| unsafe { sys::mlx_cumsum_axis(r, a.raw, axis, reverse, inclusive, none, s) })
}

pub fn sum(a: &Array, axis: i32, keepdims: bool) -> Result<Array> {
    // SAFETY: valid handles.
    op(|r, s| unsafe { sys::mlx_sum_axis(r, a.raw, axis, keepdims, s) })
}

pub fn sort(a: &Array, axis: i32) -> Result<Array> {
    // SAFETY: valid handles.
    op(|r, s| unsafe { sys::mlx_sort_axis(r, a.raw, axis, s) })
}

pub fn logsumexp(a: &Array, axis: i32, keepdims: bool) -> Result<Array> {
    // SAFETY: valid handles.
    op(|r, s| unsafe { sys::mlx_logsumexp_axis(r, a.raw, axis, keepdims, s) })
}

pub fn max(a: &Array, axis: i32, keepdims: bool) -> Result<Array> {
    // SAFETY: valid handles.
    op(|r, s| unsafe { sys::mlx_max_axis(r, a.raw, axis, keepdims, s) })
}

pub fn softmax(a: &Array, axis: i32, precise: bool) -> Result<Array> {
    // SAFETY: valid handles.
    op(|r, s| unsafe { sys::mlx_softmax_axis(r, a.raw, axis, precise, s) })
}

// ---- Quantization ----

/// The only quantization mode mrun runs (`load` refuses the others by name).
const AFFINE: &CStr = c"affine";

/// One affine-quantized weight: packed codes, scales and biases with their group size and bits,
/// built once at load. MLX 0.32.4 requires affine biases, so a checkpoint without them gets
/// zeros shaped like the scales here, once (`code · scale + 0`; nuvec keeps them absent).
#[derive(Clone, Debug)]
pub struct Quantized {
    weight: Array,
    scales: Array,
    biases: Option<Array>,
    group_size: i32,
    bits: i32,
}

impl Quantized {
    pub fn new(
        weight: Array,
        scales: Array,
        biases: Option<Array>,
        group_size: i32,
        bits: i32,
    ) -> Result<Self> {
        let biases = match biases {
            Some(b) => b,
            None => zeros(&scales.shape(), scales.dtype())?,
        };
        Ok(Self {
            weight,
            scales,
            biases: Some(biases),
            group_size,
            bits,
        })
    }

    /// The rows `ids` of weights, scales and biases (embedding lookups).
    pub fn take(&self, ids: &Array) -> Result<Self> {
        Ok(Self {
            weight: take(&self.weight, ids, 0)?,
            scales: take(&self.scales, ids, 0)?,
            biases: self.biases.as_ref().map(|b| take(b, ids, 0)).transpose()?,
            ..*self
        })
    }
}

/// `x @ wᵀ` with an affine-quantized `w`.
pub fn quantized_matmul(x: &Array, w: &Quantized) -> Result<Array> {
    let none = Array::null();
    let biases = w.biases.as_ref().unwrap_or(&none);
    // SAFETY: valid handles (a null bias handle means none) and a static mode string.
    op(|r, s| unsafe {
        sys::mlx_quantized_matmul(
            r,
            x.raw,
            w.weight.raw,
            w.scales.raw,
            biases.raw,
            true,
            optional_int(Some(w.group_size)),
            optional_int(Some(w.bits)),
            AFFINE.as_ptr(),
            s,
        )
    })
}

/// `w` restored, in its scales' dtype.
pub fn dequantize(w: &Quantized) -> Result<Array> {
    let none = sys::mlx_optional_dtype {
        value: 0,
        has_value: false,
    };
    let global_scale = Array::null();
    let no_biases = Array::null();
    let biases = w.biases.as_ref().unwrap_or(&no_biases);
    // SAFETY: valid handles (null bias/scale handles mean none) and a static mode string.
    op(|r, s| unsafe {
        sys::mlx_dequantize(
            r,
            w.weight.raw,
            w.scales.raw,
            biases.raw,
            optional_int(Some(w.group_size)),
            optional_int(Some(w.bits)),
            AFFINE.as_ptr(),
            global_scale.raw,
            none,
            s,
        )
    })
}

/// `x @ w[rhs_indices]ᵀ` per batch element with affine-quantized expert weights `w`
/// (`[E, out, in·bits/32]`); `sorted_indices` promises ascending indices (faster kernel).
pub fn gather_qmm(
    x: &Array,
    w: &Quantized,
    rhs_indices: &Array,
    sorted_indices: bool,
) -> Result<Array> {
    let none = Array::null();
    let biases = w.biases.as_ref().unwrap_or(&none);
    // SAFETY: valid handles (null biases and lhs indices mean none) and a static mode string.
    op(|r, s| unsafe {
        sys::mlx_gather_qmm(
            r,
            x.raw,
            w.weight.raw,
            w.scales.raw,
            biases.raw,
            none.raw,
            rhs_indices.raw,
            true,
            optional_int(Some(w.group_size)),
            optional_int(Some(w.bits)),
            AFFINE.as_ptr(),
            sorted_indices,
            s,
        )
    })
}

/// `a @ b[rhs_indices]` per batch element (dense expert weights).
pub fn gather_mm(a: &Array, b: &Array, rhs_indices: &Array, sorted_indices: bool) -> Result<Array> {
    let none = Array::null();
    // SAFETY: valid handles (null lhs indices mean none).
    op(|r, s| unsafe {
        sys::mlx_gather_mm(
            r,
            a.raw,
            b.raw,
            none.raw,
            rhs_indices.raw,
            sorted_indices,
            s,
        )
    })
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
    // SAFETY: valid handles.
    op(|r, s| unsafe {
        sys::mlx_conv1d(
            r, input.raw, weight.raw, stride, padding, dilation, groups, s,
        )
    })
}

// ---- Compilation ----

/// A graph-building function for [`Compiled`]: arrays in, arrays out.
pub type Graph = fn(&[Array]) -> Result<Vec<Array>>;

/// A function compiled by MLX (`mx.compile`): traced once per input signature, its elementwise
/// ops fused into one kernel. mlx-lm compiles some activations; fusion can change the rounding
/// of intermediates (a fused kernel keeps them in registers), so parity requires compiling the
/// same functions. `!Send` like [`Array`].
pub struct Compiled {
    raw: sys::mlx_closure,
}

unsafe extern "C" fn run_graph(
    out: *mut sys::mlx_vector_array,
    input: sys::mlx_vector_array,
    payload: *mut c_void,
) -> c_int {
    // SAFETY: `payload` is the `Graph` function pointer `Compiled::new` stored.
    let graph: Graph = unsafe { std::mem::transmute::<*mut c_void, Graph>(payload) };
    let inputs = VectorArray(input);
    let result =
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| graph(&inputs.to_arrays())));
    // The input vector belongs to mlx-c.
    std::mem::forget(inputs);
    let message = match result {
        Ok(Ok(outputs)) => {
            let refs: Vec<&Array> = outputs.iter().collect();
            let vec = VectorArray::new(&refs);
            // SAFETY: `out` is the vector mlx-c handed in; `set` copies the handles.
            return unsafe { sys::mlx_vector_array_set(out, vec.0) };
        }
        Ok(Err(e)) => e.0,
        Err(_) => "a compiled graph panicked".into(),
    };
    // mlx-c reports a generic failure through the handler; `Compiled::call` prefers this one.
    LAST_ERROR.with(|slot| *slot.borrow_mut() = Some(message.clone()));
    GRAPH_ERROR.with(|slot| *slot.borrow_mut() = Some(message));
    1
}

impl std::fmt::Debug for Compiled {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Compiled")
    }
}

impl Compiled {
    /// Compiles `graph`; `shapeless` reuses one trace for every input shape.
    pub fn new(graph: Graph, shapeless: bool) -> Result<Self> {
        init();
        // SAFETY: the closure stores a plain function pointer (no destructor needed); the
        // uncompiled closure is freed once the compiled one holds its own reference.
        unsafe {
            let plain = sys::mlx_closure_new_func_payload(
                Some(run_graph),
                graph as *const () as *mut c_void,
                None,
            );
            let mut raw = sys::mlx_closure_new();
            let code = sys::mlx_compile(&mut raw, plain, shapeless);
            sys::mlx_closure_free(plain);
            if let Err(e) = check(code) {
                sys::mlx_closure_free(raw);
                return Err(e);
            }
            Ok(Self { raw })
        }
    }

    pub fn call(&self, inputs: &[&Array]) -> Result<Vec<Array>> {
        let input = VectorArray::new(inputs);
        let mut out = VectorArray::empty();
        LAST_ERROR.with(|slot| slot.borrow_mut().take());
        GRAPH_ERROR.with(|slot| slot.borrow_mut().take());
        // Tracing builds ops on this thread's stream.
        stream();
        // SAFETY: applies a live closure to a live vector; `out` receives the results.
        check(unsafe { sys::mlx_closure_apply(&mut out.0, self.raw, input.0) }).map_err(|e| {
            GRAPH_ERROR
                .with(|slot| slot.borrow_mut().take())
                .map_or(e, Error)
        })?;
        Ok(out.to_arrays())
    }
}

impl Drop for Compiled {
    fn drop(&mut self) {
        // SAFETY: frees the closure created in `new` exactly once.
        unsafe { sys::mlx_closure_free(self.raw) };
    }
}

// ---- Random ----

pub fn random_key(seed: u64) -> Result<Array> {
    init();
    let mut out = Array::null();
    // SAFETY: writes one new array.
    check(unsafe { sys::mlx_random_key(&mut out.raw, seed) })?;
    Ok(out)
}

/// Samples indices from unnormalized log-probabilities along `axis`.
pub fn categorical(logits: &Array, axis: i32, key: &Array) -> Result<Array> {
    // SAFETY: valid handles.
    op(|r, s| unsafe { sys::mlx_random_categorical(r, logits.raw, axis, key.raw, s) })
}

/// Fused kernels (`mlx.core.fast`).
pub mod fast {
    use super::*;

    /// RMS normalization over the last axis, scaled by `weight` when given.
    pub fn rms_norm(x: &Array, weight: Option<&Array>, eps: f32) -> Result<Array> {
        let none = Array::null();
        let weight = weight.unwrap_or(&none);
        // SAFETY: valid handles (a null weight means none).
        op(|r, s| unsafe { sys::mlx_fast_rms_norm(r, x.raw, weight.raw, eps, s) })
    }

    /// `x + residual` and its RMS norm: [`super::add`] then [`rms_norm`] (MLX has no fused
    /// kernel; the nuvec backend fuses them). Returns the sum and the normalised sum.
    pub fn add_rms_norm(
        x: &Array,
        residual: &Array,
        weight: Option<&Array>,
        eps: f32,
    ) -> Result<(Array, Array)> {
        let h = super::add(x, residual)?;
        let y = rms_norm(&h, weight, eps)?;
        Ok((h, y))
    }

    /// A template argument of a [`MetalKernel`].
    #[derive(Clone, Copy, Debug)]
    pub(super) enum TemplateArg {
        Dtype(Dtype),
        Int(i32),
    }

    /// Launch configuration of a [`MetalKernel`] (`mx.fast.metal_kernel(...)(...)` arguments).
    #[derive(Clone, Debug)]
    pub(super) struct KernelLaunch<'a> {
        pub template: &'a [(&'a str, TemplateArg)],
        pub grid: [i32; 3],
        pub threadgroup: [i32; 3],
        pub outputs: &'a [(&'a [i32], Dtype)],
    }

    /// A custom Metal kernel (`mx.fast.metal_kernel`): MLX writes the signature from the input
    /// and output names (a 0-d input becomes a `constant` scalar), `source` is the body. Rows are
    /// made contiguous; outputs are not atomic and not initialized.
    pub(super) struct MetalKernel {
        raw: sys::mlx_fast_metal_kernel,
    }

    impl std::fmt::Debug for MetalKernel {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str("MetalKernel")
        }
    }

    impl MetalKernel {
        pub(in super::super) fn new(
            name: &str,
            inputs: &[&str],
            outputs: &[&str],
            source: &str,
        ) -> Self {
            init();
            let strings = |names: &[&str]| {
                // SAFETY: a new vector; each append copies a NUL-terminated name.
                unsafe {
                    let vec = sys::mlx_vector_string_new();
                    for name in names {
                        let name = CString::new(*name).expect("kernel argument name");
                        sys::mlx_vector_string_append_value(vec, name.as_ptr());
                    }
                    vec
                }
            };
            let (name, source) = (
                CString::new(name).expect("kernel name"),
                CString::new(source).expect("kernel source"),
            );
            // SAFETY: mlx-c copies the strings; the name vectors are freed after the call.
            unsafe {
                let (inputs, outputs) = (strings(inputs), strings(outputs));
                let raw = sys::mlx_fast_metal_kernel_new(
                    name.as_ptr(),
                    inputs,
                    outputs,
                    source.as_ptr(),
                    c"".as_ptr(),
                    true,
                    false,
                );
                sys::mlx_vector_string_free(inputs);
                sys::mlx_vector_string_free(outputs);
                Self { raw }
            }
        }

        pub(in super::super) fn apply(
            &self,
            inputs: &[&Array],
            launch: &KernelLaunch<'_>,
        ) -> Result<Vec<Array>> {
            let names: Vec<CString> = launch
                .template
                .iter()
                .map(|(name, _)| CString::new(*name).expect("template name"))
                .collect();
            let input = VectorArray::new(inputs);
            // SAFETY: the config is created, filled and freed here; names outlive the calls.
            unsafe {
                let config = sys::mlx_fast_metal_kernel_config_new();
                for (shape, dtype) in launch.outputs {
                    sys::mlx_fast_metal_kernel_config_add_output_arg(
                        config,
                        shape.as_ptr(),
                        shape.len(),
                        dtype.raw(),
                    );
                }
                let [g0, g1, g2] = launch.grid;
                sys::mlx_fast_metal_kernel_config_set_grid(config, g0, g1, g2);
                let [t0, t1, t2] = launch.threadgroup;
                sys::mlx_fast_metal_kernel_config_set_thread_group(config, t0, t1, t2);
                for (name, (_, arg)) in names.iter().zip(launch.template) {
                    match *arg {
                        TemplateArg::Dtype(dtype) => {
                            sys::mlx_fast_metal_kernel_config_add_template_arg_dtype(
                                config,
                                name.as_ptr(),
                                dtype.raw(),
                            )
                        }
                        TemplateArg::Int(value) => {
                            sys::mlx_fast_metal_kernel_config_add_template_arg_int(
                                config,
                                name.as_ptr(),
                                value,
                            )
                        }
                    };
                }
                let result = op_vec(|r, s| {
                    sys::mlx_fast_metal_kernel_apply(r, self.raw, input.0, config, s)
                });
                sys::mlx_fast_metal_kernel_config_free(config);
                result
            }
        }
    }

    impl Drop for MetalKernel {
        fn drop(&mut self) {
            // SAFETY: frees the kernel created in `new` exactly once.
            unsafe { sys::mlx_fast_metal_kernel_free(self.raw) };
        }
    }

    /// mlx-lm's gated delta rule (`gated_delta_update` without mask): `q, k [B, T, Hk, Dk]`,
    /// `v [B, T, Hv, Dv]`, `g, beta [B, T, Hv]`, `state [B, Hv, Dv, Dk]`; returns `y [B, T, Hv,
    /// Dv]` (in `q`'s dtype) and the new state. mlx-lm's kernels on Metal, else its per-step
    /// reference ([`super::gated_delta`]).
    pub fn gated_delta(
        q: &Array,
        k: &Array,
        v: &Array,
        g: &Array,
        beta: &Array,
        state: &Array,
    ) -> Result<(Array, Array)> {
        super::gated_delta::gated_delta(q, k, v, g, beta, state)
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
        let base = sys::mlx_optional_float {
            value: base.unwrap_or(0.0),
            has_value: base.is_some(),
        };
        let none = Array::null();
        let freqs = freqs.unwrap_or(&none);
        // SAFETY: valid handles (freqs may be null).
        op(|r, s| unsafe {
            sys::mlx_fast_rope(
                r,
                x.raw,
                dims,
                traditional,
                base,
                scale,
                offset,
                freqs.raw,
                s,
            )
        })
    }

    /// Attention mask of [`scaled_dot_product_attention`].
    #[derive(Clone, Copy, Debug)]
    pub enum Mask {
        None,
        /// Causal, aligned to the end of the keys.
        Causal,
    }

    /// Whether the attention below keeps its score buffer within `max_score_bytes` by itself:
    /// not MLX's (it has no such bound; mrun runs query blocks instead, D059).
    pub const CAPS_SCORES: bool = false;

    /// The scratch bytes of an attention call with these `[B, H, L, D]` shapes, when the backend
    /// can tell: MLX cannot (`None`; the model's estimate applies).
    pub fn attention_scratch_bytes(
        q: [i32; 4],
        k: [i32; 4],
        dtype: Dtype,
        mask: Mask,
        max_score_bytes: u64,
    ) -> Option<u64> {
        let _ = (q, k, dtype, mask, max_score_bytes);
        None
    }

    /// `softmax(scale · q kᵀ + mask) v`; `max_score_bytes` is ignored (see [`CAPS_SCORES`]).
    pub fn scaled_dot_product_attention(
        q: &Array,
        k: &Array,
        v: &Array,
        scale: f32,
        mask: Mask,
        max_score_bytes: u64,
    ) -> Result<Array> {
        let _ = max_score_bytes;
        let none = Array::null();
        let mode = match mask {
            Mask::None => c"",
            Mask::Causal => c"causal",
        };
        // SAFETY: valid handles (null mask array and sinks mean none) and a static mode string.
        op(|r, s| unsafe {
            sys::mlx_fast_scaled_dot_product_attention(
                r,
                q.raw,
                k.raw,
                v.raw,
                scale,
                mode.as_ptr(),
                none.raw,
                none.raw,
                false,
                s,
            )
        })
    }
}

/// Memory counters and limits (bytes).
pub mod memory {
    use super::*;

    fn read(f: unsafe extern "C" fn(*mut usize) -> c_int) -> u64 {
        init();
        let mut out = 0usize;
        // SAFETY: writes one usize.
        unsafe { f(&mut out) };
        out as u64
    }

    pub fn active() -> u64 {
        read(sys::mlx_get_active_memory)
    }

    pub fn peak() -> u64 {
        read(sys::mlx_get_peak_memory)
    }

    /// [`active`] without memory-mapped files: the same in MLX, which maps nothing.
    pub fn active_allocated() -> u64 {
        active()
    }

    /// [`peak`] without memory-mapped files: the same in MLX, which maps nothing.
    pub fn peak_allocated() -> u64 {
        peak()
    }

    pub fn reset_peak() {
        init();
        // SAFETY: no arguments.
        unsafe { sys::mlx_reset_peak_memory() };
    }

    pub fn clear_cache() {
        init();
        // SAFETY: no arguments.
        unsafe { sys::mlx_clear_cache() };
    }

    /// Sets the limit and returns the previous one.
    pub fn set_memory_limit(limit: u64) -> u64 {
        init();
        let mut previous = 0usize;
        // SAFETY: writes one usize.
        unsafe { sys::mlx_set_memory_limit(&mut previous, limit as usize) };
        previous as u64
    }

    pub fn set_cache_limit(limit: u64) -> u64 {
        init();
        let mut previous = 0usize;
        // SAFETY: writes one usize.
        unsafe { sys::mlx_set_cache_limit(&mut previous, limit as usize) };
        previous as u64
    }

    pub fn set_wired_limit(limit: u64) -> u64 {
        init();
        let mut previous = 0usize;
        // SAFETY: writes one usize.
        unsafe { sys::mlx_set_wired_limit(&mut previous, limit as usize) };
        previous as u64
    }
}

/// `(memory_size, max_recommended_working_set_size)` of the default device, when it reports
/// them (Metal devices do; the CPU device does not).
pub fn device_memory() -> Option<(u64, u64)> {
    init();
    // Without its kernels MLX aborts while opening the Metal device; the load reports why.
    if metallib::ensure().is_err() {
        return None;
    }
    // SAFETY: device and info handles are created and freed here; keys are NUL-terminated.
    unsafe {
        let mut device = sys::mlx_device_new();
        if sys::mlx_get_default_device(&mut device) != 0 {
            sys::mlx_device_free(device);
            return None;
        }
        let mut info = sys::mlx_device_info_new();
        let ok = sys::mlx_device_info_get(&mut info, device) == 0;
        let mut size = 0usize;
        let mut working = 0usize;
        let found = ok
            && sys::mlx_device_info_get_size(&mut size, info, c"memory_size".as_ptr()) == 0
            && sys::mlx_device_info_get_size(
                &mut working,
                info,
                c"max_recommended_working_set_size".as_ptr(),
            ) == 0;
        sys::mlx_device_info_free(info);
        sys::mlx_device_free(device);
        LAST_ERROR.with(|slot| slot.borrow_mut().take());
        found.then_some((size as u64, working as u64))
    }
}

/// A string entry of the GPU's MLX device info (`architecture`, `device_name`); `None` without
/// Metal or when the key is missing.
pub fn gpu_info_string(key: &std::ffi::CStr) -> Option<String> {
    if !metal_available() || metallib::ensure().is_err() {
        return None;
    }
    // SAFETY: device and info handles are created and freed here; the string is owned by `info`
    // and copied before it is freed.
    unsafe {
        let device = sys::mlx_device_new_type(sys::mlx_device_type__MLX_GPU, 0);
        let mut info = sys::mlx_device_info_new();
        let mut out = None;
        if sys::mlx_device_info_get(&mut info, device) == 0 {
            let mut value: *const std::os::raw::c_char = std::ptr::null();
            if sys::mlx_device_info_get_string(&mut value, info, key.as_ptr()) == 0
                && !value.is_null()
            {
                out = Some(
                    std::ffi::CStr::from_ptr(value)
                        .to_string_lossy()
                        .into_owned(),
                );
            }
        }
        sys::mlx_device_info_free(info);
        sys::mlx_device_free(device);
        LAST_ERROR.with(|slot| slot.borrow_mut().take());
        out
    }
}

/// The numerics class of this process (FR-027, D044): whether MLX's NAX kernels are compiled in
/// (`cfg(mlx_nax)`) and available on this GPU, and the GPU architecture from MLX's device info
/// (`cpu` without Metal). Asked once.
pub fn numerics() -> crate::engine::Numerics {
    static NUMERICS: std::sync::OnceLock<crate::engine::Numerics> = std::sync::OnceLock::new();
    NUMERICS
        .get_or_init(|| {
            let Some(arch) = gpu_info_string(c"architecture") else {
                return crate::engine::Numerics {
                    nax: false,
                    gpu_arch: "cpu".into(),
                    backend: crate::engine::Backend::Mlx,
                };
            };
            let mut nax = false;
            // SAFETY: writes one bool.
            let ok = unsafe { sys::mlx_metal_is_nax_available(&mut nax) == 0 };
            LAST_ERROR.with(|slot| slot.borrow_mut().take());
            crate::engine::Numerics {
                nax: ok && nax && cfg!(mlx_nax),
                gpu_arch: arch,
                backend: crate::engine::Backend::Mlx,
            }
        })
        .clone()
}

/// Whether this GPU has NAX (M5 Neural Accelerators), compiled in or not (`mrun doctor
/// --report`); `false` without Metal.
pub fn nax_available() -> bool {
    if !metal_available() || metallib::ensure().is_err() {
        return false;
    }
    let mut nax = false;
    // SAFETY: writes one bool.
    let ok = unsafe { sys::mlx_metal_is_nax_available(&mut nax) == 0 };
    LAST_ERROR.with(|slot| slot.borrow_mut().take());
    ok && nax
}

/// Whether this build compiled MLX's NAX kernels (SDK and deployment target 26.2 or later).
pub const NAX_COMPILED: bool = cfg!(mlx_nax);

/// Whether MLX runs its NAX kernels (`mrun doctor --report`): `on`, or `off (<reason>)`.
pub fn nax_status() -> String {
    if numerics().nax {
        "on".into()
    } else if !NAX_COMPILED {
        "off (no NAX kernels in this build)".into()
    } else if nax_available() {
        "off".into()
    } else {
        "off (not an M5-class GPU)".into()
    }
}

/// The bytes a filtered load will hold: MLX cannot tell from the headers (`None`; the budget
/// counts the files).
pub fn weights_footprint(_dir: &Path, _keep: impl Fn(&str) -> bool) -> Option<u64> {
    None
}

/// The backend of this build (016 FR-008): MLX, its vendored version, and its kernels, a
/// `mlx.metallib` compiled at build time and embedded (D034).
pub fn backend_info() -> crate::engine::BackendInfo {
    crate::engine::BackendInfo {
        backend: crate::engine::Backend::Mlx,
        version: sys::MLX_VERSION,
        kernels: crate::engine::Kernels::Precompiled,
    }
}

/// The tensors of a `.safetensors` file that `keep` accepts (lazily, so the others cost nothing
/// but their header entry).
pub fn load_safetensors_filtered(
    path: &Path,
    keep: impl Fn(&str) -> bool,
) -> Result<Vec<(String, Array)>> {
    let mut tensors = load_safetensors(path)?;
    tensors.retain(|(name, _)| keep(name));
    Ok(tensors)
}

/// Loads every tensor of a `.safetensors` file (lazily: data is read on evaluation).
pub fn load_safetensors(path: &Path) -> Result<Vec<(String, Array)>> {
    init();
    let file =
        CString::new(path.as_os_str().as_encoded_bytes()).map_err(|e| Error(e.to_string()))?;
    // SAFETY: the maps are created by mlx-c and freed here; keys are copied before the
    // iterator moves on; each value is taken by reference into an owned `Array`.
    unsafe {
        let mut tensors = sys::mlx_map_string_to_array_new();
        let mut metadata = sys::mlx_map_string_to_string_new();
        let cpu = sys::mlx_default_cpu_stream_new();
        let code = sys::mlx_load_safetensors(&mut tensors, &mut metadata, file.as_ptr(), cpu);
        sys::mlx_stream_free(cpu);
        let result = check(code).map(|()| {
            let mut out = Vec::new();
            let it = sys::mlx_map_string_to_array_iterator_new(tensors);
            loop {
                let mut key: *const c_char = std::ptr::null();
                let mut value = Array::null();
                if sys::mlx_map_string_to_array_iterator_next(&mut key, &mut value.raw, it) != 0
                    || key.is_null()
                {
                    break;
                }
                out.push((CStr::from_ptr(key).to_string_lossy().into_owned(), value));
            }
            sys::mlx_map_string_to_array_iterator_free(it);
            out
        });
        sys::mlx_map_string_to_array_free(tensors);
        sys::mlx_map_string_to_string_free(metadata);
        LAST_ERROR.with(|slot| slot.borrow_mut().take());
        result
    }
}

/// Writes `tensors` to a `.safetensors` file (tiny test checkpoints).
pub fn save_safetensors(path: &Path, tensors: &[(&str, &Array)]) -> Result<()> {
    init();
    let file =
        CString::new(path.as_os_str().as_encoded_bytes()).map_err(|e| Error(e.to_string()))?;
    let keys: Vec<CString> = tensors
        .iter()
        .map(|(name, _)| CString::new(*name).map_err(|e| Error(e.to_string())))
        .collect::<Result<_>>()?;
    // SAFETY: the maps are created and freed here; keys are NUL-terminated and outlive the
    // inserts; values are live arrays (the map takes its own references).
    unsafe {
        let map = sys::mlx_map_string_to_array_new();
        let metadata = sys::mlx_map_string_to_string_new();
        for (key, (_, array)) in keys.iter().zip(tensors) {
            sys::mlx_map_string_to_array_insert(map, key.as_ptr(), array.raw);
        }
        let code = sys::mlx_save_safetensors(file.as_ptr(), map, metadata);
        sys::mlx_map_string_to_array_free(map);
        sys::mlx_map_string_to_string_free(metadata);
        check(code)
    }
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
        // The next call succeeds and does not see the old message.
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
        // A 2×64 weight quantized with group 64, 4 bits: w = scale·q + bias.
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
    fn sdpa_causal_attends_to_the_past_only() {
        cpu();
        // One head, two positions, values 1 and 3: position 0 sees only itself.
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
    }

    #[test]
    fn depthwise_conv1d() {
        cpu();
        // Two channels, kernel 2, groups = channels: y[t, c] = Σ_k x[t + k, c] · w[c, k].
        let x = Array::from_f32(&[1.0, 10.0, 2.0, 20.0, 3.0, 30.0], &[1, 3, 2]);
        let w = Array::from_f32(&[1.0, 2.0, 0.5, -1.0], &[2, 2, 1]);
        let y = conv1d(&x, &w, 1, 0, 1, 2).unwrap();
        assert_eq!(y.shape(), [1, 2, 2]);
        assert_eq!(y.to_vec_f32().unwrap(), [5.0, -15.0, 8.0, -20.0]);
    }

    #[test]
    fn gathered_matmuls_pick_each_rows_expert() {
        cpu();
        // Expert e scales by e + 1 (identity weights, in = out = 64 for one quantization group).
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
        // The same through 8-bit affine quantization: scale·q + bias represents 0..3 exactly.
        let none = Array::null();
        let mut packed = VectorArray::empty();
        // SAFETY: quantizes a live array into a new vector (a null global scale means none).
        check(unsafe {
            sys::mlx_quantize(
                &mut packed.0,
                w.raw,
                optional_int(Some(64)),
                optional_int(Some(8)),
                c"affine".as_ptr(),
                none.raw,
                stream(),
            )
        })
        .unwrap();
        let [wq, scales, biases]: [Array; 3] = packed.to_arrays().try_into().unwrap();
        let wq = Quantized::new(wq, scales, Some(biases), 64, 8).unwrap();
        let q = gather_qmm(&x, &wq, &idx, false).unwrap();
        for (a, b) in q.to_vec_f32().unwrap().iter().zip(&d) {
            assert!((a - b).abs() < 1e-3, "{a} vs {b}");
        }
    }

    #[test]
    fn the_backend_reports_mlx_and_its_vendored_version() {
        let info = backend_info();
        assert_eq!(info.backend, crate::engine::Backend::Mlx);
        assert_eq!(info.version, sys::MLX_VERSION);
        assert_eq!(info.kernels, crate::engine::Kernels::Precompiled);
        assert_eq!(
            numerics().class(),
            if numerics().nax { "nax" } else { "plain" }
        );
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
        // Shapeless: another shape reuses the trace.
        let z = Array::from_f32(&[1.0; 3], &[3]);
        assert_eq!(
            compiled.call(&[&z, &z]).unwrap()[0].to_vec_f32().unwrap(),
            [2.0; 3]
        );
        let failing: Graph = |a| Ok(vec![add(&a[0], &a[1])?]);
        let failing = Compiled::new(failing, false).unwrap();
        assert!(failing.call(&[&x, &z]).is_err());
        // The graph's own error, not mlx-c's generic one.
        let refusing: Graph = |_| Err(Error("refused by the graph".into()));
        let refusing = Compiled::new(refusing, false).unwrap();
        let error = refusing.call(&[&x]).unwrap_err();
        assert_eq!(error.0, "refused by the graph");
    }
}
