//! The gated delta rule on MLX (`tensor::fast::gated_delta`): a port of mlx-lm 0.32.0
//! `mlx_lm/models/gated_delta.py` for the shapes mrun runs (one sequence, no padding mask,
//! scalar per-head gates). nuvec ships the same kernels and its own fallback, so this module
//! exists only for the MLX backend (spec 016 FR-014) and goes with it at the cut-over.
//!
//! On Metal mlx-lm runs a custom kernel whenever `Dk` is a positive multiple of 32: the packed
//! `gated_delta_step_packed_btree` kernel when `Dk == 128`, `Dv % 8 == 0` and `g` and the state
//! are f32 (enabled unless `MLX_GDN_PACKED=0`; Qwen3.6 takes this path), else the generic
//! `gated_delta_step` kernel. The kernel bodies in `kernels/` are copied verbatim from mlx-lm
//! (the generic one as its f-string renders without mask and with scalar gates), with the same
//! names, input order, template arguments, grids and thread groups. Elsewhere (CPU, other `Dk`)
//! the compiled per-step ops reference runs, as in mlx-lm.

use std::cell::OnceCell;
use std::sync::OnceLock;

use super::fast::{KernelLaunch, MetalKernel, TemplateArg};
use crate::tensor::{self, Array, Compiled, Dtype, Graph};

const GENERIC_SOURCE: &str = include_str!("kernels/gated_delta_step.metal");
const PACKED_SOURCE: &str = include_str!("kernels/gated_delta_step_packed_btree.metal");
const INPUTS: [&str; 7] = ["q", "k", "v", "g", "beta", "state_in", "T"];
const OUTPUTS: [&str; 2] = ["y", "state_out"];

thread_local! {
    static GENERIC: OnceCell<MetalKernel> = const { OnceCell::new() };
    static PACKED: OnceCell<MetalKernel> = const { OnceCell::new() };
    static STEP: OnceCell<Compiled> = const { OnceCell::new() };
}

/// The recurrence on the path mlx-lm takes for these inputs.
pub fn gated_delta(
    q: &Array,
    k: &Array,
    v: &Array,
    g: &Array,
    beta: &Array,
    state: &Array,
) -> tensor::Result<(Array, Array)> {
    match path(k.dim(-1), v.dim(-1), g.dtype(), state.dtype()) {
        Path::Ops => ops(q, k, v, g, beta, state),
        kernel => run_kernel(kernel, q, k, v, g, beta, state),
    }
}

/// Which implementation runs the recurrence.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Path {
    /// Compiled per-step ops (CPU, or `Dk` not a multiple of 32).
    Ops,
    /// One SIMD-group per value row.
    Generic,
    /// Eight value rows per SIMD-group (`Dk == 128`).
    Packed,
}

/// The path mlx-lm 0.32.0 takes for these inputs on this thread's device.
pub fn path(dk: i32, dv: i32, g: Dtype, state: Dtype) -> Path {
    if !tensor::on_gpu() || dk < 32 || dk % 32 != 0 {
        return Path::Ops;
    }
    // Read once: this runs for every linear layer of every step.
    static PACKED_DISABLED: OnceLock<bool> = OnceLock::new();
    let packed_disabled =
        *PACKED_DISABLED.get_or_init(|| std::env::var("MLX_GDN_PACKED").is_ok_and(|v| v == "0"));
    if dk == 128
        && dv % 8 == 0
        && g == Dtype::Float32
        && state == Dtype::Float32
        && !packed_disabled
    {
        Path::Packed
    } else {
        Path::Generic
    }
}

/// mlx-lm `gated_delta_ops`: the step function over every position.
pub fn ops(
    q: &Array,
    k: &Array,
    v: &Array,
    g: &Array,
    beta: &Array,
    state: &Array,
) -> tensor::Result<(Array, Array)> {
    let (len, hk) = (q.dim(1), q.dim(2));
    let hv = v.dim(-2);
    let (q, k) = if hv / hk > 1 {
        (
            tensor::repeat(q, hv / hk, -2)?,
            tensor::repeat(k, hv / hk, -2)?,
        )
    } else {
        (q.clone(), k.clone())
    };
    // `x[:, t]` of a `[B, T, ...]` array.
    let at = |x: &Array, t: i32| {
        let shape = x.shape();
        let mut start = vec![0; shape.len()];
        let mut stop = shape.clone();
        start[1] = t;
        stop[1] = t + 1;
        tensor::squeeze(&tensor::slice(x, &start, &stop)?, &[1])
    };
    let mut state = state.clone();
    let mut ys = Vec::with_capacity(len as usize);
    for t in 0..len {
        let (y, next) = step(
            &at(&q, t)?,
            &at(&k, t)?,
            &at(v, t)?,
            &at(g, t)?,
            &at(beta, t)?,
            &state,
        )?;
        ys.push(y);
        state = next;
    }
    let refs: Vec<&Array> = ys.iter().collect();
    Ok((tensor::stack(&refs, 1)?, state))
}

/// mlx-lm `_gated_delta_step_ops` (compiled per shape, no mask): one recurrent step on
/// `q, k [B, H, Dk]`, `v [B, H, Dv]`, `g, beta [B, H]`, `state [B, H, Dv, Dk]`; returns
/// `y [B, H, Dv]` in q's dtype and the new state.
fn step(
    q: &Array,
    k: &Array,
    v: &Array,
    g: &Array,
    beta: &Array,
    state: &Array,
) -> tensor::Result<(Array, Array)> {
    let graph: Graph = |a| {
        let (q, k, v, g, beta, state) = (&a[0], &a[1], &a[2], &a[3], &a[4], &a[5]);
        let decay = tensor::expand_dims(g, &[-1, -2])?;
        let state = tensor::multiply(state, &decay)?;
        let k_row = tensor::expand_dims(k, &[-2])?;
        let kv_mem = tensor::sum(&tensor::multiply(&state, &k_row)?, -1, false)?;
        let delta = tensor::multiply(
            &tensor::subtract(v, &kv_mem)?,
            &tensor::expand_dims(beta, &[-1])?,
        )?;
        let state = tensor::add(
            &state,
            &tensor::multiply(&k_row, &tensor::expand_dims(&delta, &[-1])?)?,
        )?;
        let q_row = tensor::expand_dims(q, &[-2])?;
        let y = tensor::sum(&tensor::multiply(&state, &q_row)?, -1, false)?;
        Ok(vec![tensor::astype(&y, q.dtype())?, state])
    };
    let mut out = STEP.with(|cell| {
        if cell.get().is_none() {
            // A second initialisation cannot race: the cell is thread-local.
            let _ = cell.set(Compiled::new(graph, false)?);
        }
        cell.get()
            .expect("initialised above")
            .call(&[q, k, v, g, beta, state])
    })?;
    let state = out.pop();
    match (out.pop(), state) {
        (Some(y), Some(state)) => Ok((y, state)),
        _ => Err(tensor::Error("gated delta step returned too little".into())),
    }
}

/// mlx-lm `gated_delta_kernel` (`kernel` = Generic or Packed).
pub fn run_kernel(
    kernel: Path,
    q: &Array,
    k: &Array,
    v: &Array,
    g: &Array,
    beta: &Array,
    state: &Array,
) -> tensor::Result<(Array, Array)> {
    let (batch, len, hk, dk) = (k.dim(0), k.dim(1), k.dim(2), k.dim(3));
    let (hv, dv) = (v.dim(2), v.dim(3));
    let (slot, name, source, grid, threadgroup) = match kernel {
        Path::Packed => (
            &PACKED,
            "gated_delta_step_packed_btree",
            PACKED_SOURCE,
            [32, dv / 8, batch * hv],
            [32, 2, 1],
        ),
        Path::Generic => (
            &GENERIC,
            "gated_delta_step",
            GENERIC_SOURCE,
            [32, dv, batch * hv],
            [32, 4, 1],
        ),
        Path::Ops => return Err(tensor::Error("the ops path has no kernel".into())),
    };
    let y_shape = [batch, len, hv, dv];
    let state_shape = state.shape();
    let launch = KernelLaunch {
        template: &[
            ("InT", TemplateArg::Dtype(q.dtype())),
            ("StT", TemplateArg::Dtype(state.dtype())),
            ("Dk", TemplateArg::Int(dk)),
            ("Dv", TemplateArg::Int(dv)),
            ("Hk", TemplateArg::Int(hk)),
            ("Hv", TemplateArg::Int(hv)),
        ],
        grid,
        threadgroup,
        outputs: &[(&y_shape, q.dtype()), (&state_shape, state.dtype())],
    };
    // `T` is a Python int in mlx-lm, i.e. a 0-d int32 array: a `constant int&` in the kernel.
    let t = Array::scalar_i32(len);
    let mut out = slot.with(|cell| {
        cell.get_or_init(|| MetalKernel::new(name, &INPUTS, &OUTPUTS, source))
            .apply(&[q, k, v, g, beta, state, &t], &launch)
    })?;
    let state = out.pop();
    match (out.pop(), state) {
        (Some(y), Some(state)) => Ok((y, state)),
        _ => Err(tensor::Error(
            "gated delta kernel returned too little".into(),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::activations;
    use crate::models::gated_delta::normalize_qk;
    use crate::tensor::{Device, use_device};

    fn noise(seed: &mut u64, shape: &[i32], scale: f32) -> Array {
        let n: i32 = shape.iter().product();
        let data: Vec<f32> = (0..n)
            .map(|_| {
                *seed = seed
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                ((*seed >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0) * scale
            })
            .collect();
        Array::from_f32(&data, shape)
    }

    /// Normalized inputs for `T` positions: `(q, k, v, g, beta)`.
    fn inputs(t: i32, hk: i32, hv: i32, dk: i32, dv: i32, dtype: Dtype) -> [Array; 5] {
        let mut seed = 11;
        let cast = |a: Array| tensor::astype(&a, dtype).unwrap();
        let q = cast(noise(&mut seed, &[1, t, hk, dk], 1.0));
        let k = cast(noise(&mut seed, &[1, t, hk, dk], 1.0));
        let (q, k) = normalize_qk(&q, &k, f64::from(dk).powf(-0.5), 1e-6).unwrap();
        let v = cast(noise(&mut seed, &[1, t, hv, dv], 1.0));
        let a = noise(&mut seed, &[1, t, hv], 2.0);
        let g = activations::compute_g(
            &noise(&mut seed, &[hv], 1.0),
            &cast(a),
            &cast(noise(&mut seed, &[hv], 1.0)),
        )
        .unwrap();
        let beta = tensor::sigmoid(&cast(noise(&mut seed, &[1, t, hv], 3.0))).unwrap();
        [q, k, v, g, beta]
    }

    fn max_diff(a: &Array, b: &Array) -> f32 {
        let (a, b) = (a.to_vec_f32().unwrap(), b.to_vec_f32().unwrap());
        a.iter()
            .zip(&b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f32::max)
    }

    #[test]
    fn the_cpu_takes_the_ops_path() {
        use_device(Device::Cpu);
        assert_eq!(path(128, 128, Dtype::Float32, Dtype::Float32), Path::Ops);
    }

    #[test]
    fn ops_carry_the_state_across_calls() {
        use_device(Device::Cpu);
        let [q, k, v, g, beta] = inputs(6, 2, 4, 32, 16, Dtype::Float32);
        let zero = tensor::zeros(&[1, 4, 16, 32], Dtype::Float32).unwrap();
        let (y, state) = ops(&q, &k, &v, &g, &beta, &zero).unwrap();
        // Positions 0..3 then 3..6 continue exactly.
        let part = |x: &Array, a: i32, b: i32| {
            let mut start = vec![0; x.shape().len()];
            let mut stop = x.shape();
            start[1] = a;
            stop[1] = b;
            tensor::slice(x, &start, &stop).unwrap()
        };
        let parts = |a, b| [&q, &k, &v, &g, &beta].map(|x| part(x, a, b));
        let [q1, k1, v1, g1, b1] = parts(0, 3);
        let (y1, s1) = ops(&q1, &k1, &v1, &g1, &b1, &zero).unwrap();
        let [q2, k2, v2, g2, b2] = parts(3, 6);
        let (y2, s2) = ops(&q2, &k2, &v2, &g2, &b2, &s1).unwrap();
        assert_eq!(max_diff(&state, &s2), 0.0);
        assert_eq!(max_diff(&part(&y, 0, 3), &y1), 0.0);
        assert_eq!(max_diff(&part(&y, 3, 6), &y2), 0.0);
    }

    #[test]
    fn kernels_match_the_ops_and_packed_matches_generic_bitwise() {
        if !tensor::metal_available() {
            return;
        }
        use_device(Device::Default);
        for (dk, dtype) in [
            (128, Dtype::Bfloat16),
            (128, Dtype::Float32),
            (64, Dtype::Bfloat16),
        ] {
            let [q, k, v, g, beta] = inputs(9, 2, 4, dk, 32, dtype);
            let zero = tensor::zeros(&[1, 4, 32, dk], Dtype::Float32).unwrap();
            let (y_ops, s_ops) = ops(&q, &k, &v, &g, &beta, &zero).unwrap();
            let (y_gen, s_gen) = run_kernel(Path::Generic, &q, &k, &v, &g, &beta, &zero).unwrap();
            let tolerance = if dtype == Dtype::Float32 { 1e-5 } else { 2e-2 };
            assert!(max_diff(&y_ops, &y_gen) < tolerance, "dk {dk} {dtype:?}");
            assert!(max_diff(&s_ops, &s_gen) < 1e-4, "dk {dk} {dtype:?}");
            if dk == 128 {
                let (y_pack, s_pack) =
                    run_kernel(Path::Packed, &q, &k, &v, &g, &beta, &zero).unwrap();
                assert_eq!(max_diff(&y_gen, &y_pack), 0.0, "{dtype:?}");
                assert_eq!(max_diff(&s_gen, &s_pack), 0.0, "{dtype:?}");
            }
        }
    }

    #[test]
    fn gated_delta_step_updates_the_state() {
        use_device(Device::Cpu);
        // One head, Dk = Dv = 1: state' = s·g + k·(v − s·g·k)·β, y = state'·q.
        let one = |v: f32| Array::from_f32(&[v], &[1, 1, 1]);
        let (y, state) = step(
            &one(2.0),
            &one(0.5),
            &one(3.0),
            &Array::from_f32(&[0.5], &[1, 1]),
            &Array::from_f32(&[0.8], &[1, 1]),
            &Array::from_f32(&[4.0], &[1, 1, 1, 1]),
        )
        .unwrap();
        let s = 4.0f32 * 0.5;
        let s = s + 0.5 * (3.0 - s * 0.5) * 0.8;
        assert!((state.item_f32().unwrap() - s).abs() < 1e-6);
        assert!((y.item_f32().unwrap() - 2.0 * s).abs() < 1e-6);
    }
}
