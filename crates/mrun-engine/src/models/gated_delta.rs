//! The gated delta rule of Gated DeltaNet layers: mlx-lm 0.32.0 `mlx_lm/models/gated_delta.py`
//! for the shapes mrun runs (one sequence, no padding mask, scalar per-head gates). The gates
//! and the q/k normalization are model numerics and live here; the recurrence itself is one
//! backend operation, `tensor::fast::gated_delta` (mlx-lm's kernels on the GPU, the per-step
//! reference elsewhere).

use crate::models::activations;
use crate::tensor::fast;
use crate::tensor::{self, Array, Dtype};

/// mlx-lm `normalize_qk`: L2-normalizes q and k and folds the `Dk^−0.5` readout scale into q.
/// The reference eps is added to `sum(x²)`; `rms_norm` adds it to `mean(x²)`, hence the scaled
/// eps. The scales are Python floats, i.e. rounded to the activation dtype.
pub fn normalize_qk(
    q: &Array,
    k: &Array,
    inv_scale: f64,
    eps: f64,
) -> tensor::Result<(Array, Array)> {
    let rms_eps = (eps * inv_scale * inv_scale) as f32;
    let q = tensor::multiply(
        &tensor::scalar((inv_scale * inv_scale) as f32, q.dtype())?,
        &fast::rms_norm(q, None, rms_eps)?,
    )?;
    let k = tensor::multiply(
        &tensor::scalar(inv_scale as f32, k.dtype())?,
        &fast::rms_norm(k, None, rms_eps)?,
    )?;
    Ok((q, k))
}

/// mlx-lm `gated_delta_update` without mask: `q, k [1, T, Hk, Dk]` (normalized), `v [1, T, Hv,
/// Dv]`, `a, b [1, T, Hv]`, the state `[1, Hv, Dv, Dk]` (`None`: zeros, f32). Returns `y [1, T,
/// Hv, Dv]` and the new state.
#[allow(clippy::too_many_arguments)]
pub fn update(
    q: &Array,
    k: &Array,
    v: &Array,
    a: &Array,
    b: &Array,
    a_log: &Array,
    dt_bias: &Array,
    state: Option<&Array>,
) -> tensor::Result<(Array, Array)> {
    let beta = tensor::sigmoid(b)?;
    let g = activations::compute_g(a_log, a, dt_bias)?;
    let (dk, dv, hv) = (k.dim(-1), v.dim(-1), v.dim(-2));
    let state = match state {
        Some(state) => state.clone(),
        None => tensor::zeros(&[q.dim(0), hv, dv, dk], Dtype::Float32)?,
    };
    fast::gated_delta(q, k, v, &g, &beta, &state)
}

#[cfg(test)]
mod tests {
    use super::*;
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

    /// Positions `[a, b)` of a `[1, T, ...]` array.
    fn part(x: &Array, a: i32, b: i32) -> Array {
        let mut start = vec![0; x.shape().len()];
        let mut stop = x.shape();
        start[1] = a;
        stop[1] = b;
        tensor::slice(x, &start, &stop).unwrap()
    }

    fn bits(a: &Array) -> Vec<u32> {
        let a = tensor::astype(a, Dtype::Float32).unwrap();
        a.to_vec_f32()
            .unwrap()
            .iter()
            .map(|v| v.to_bits())
            .collect()
    }

    /// Chunked (3 + 3) equals unchunked (6) bit for bit, on the backend's kernel (GPU, `Dk`
    /// 128: the packed path) and on its fallback (CPU), in f32 and bf16 activations.
    #[test]
    fn chunked_updates_equal_one_update() {
        let devices: &[Device] = if tensor::metal_available() {
            &[Device::Cpu, Device::Default]
        } else {
            &[Device::Cpu]
        };
        for &device in devices {
            use_device(device);
            for (dk, dtype) in [
                (128, Dtype::Bfloat16),
                (128, Dtype::Float32),
                (64, Dtype::Float32),
            ] {
                let mut seed = 7;
                let cast = |a: Array| tensor::astype(&a, dtype).unwrap();
                let (hk, hv, dv) = (2, 4, 32);
                let q = cast(noise(&mut seed, &[1, 6, hk, dk], 1.0));
                let k = cast(noise(&mut seed, &[1, 6, hk, dk], 1.0));
                let (q, k) = normalize_qk(&q, &k, f64::from(dk).powf(-0.5), 1e-6).unwrap();
                let v = cast(noise(&mut seed, &[1, 6, hv, dv], 1.0));
                let a = cast(noise(&mut seed, &[1, 6, hv], 2.0));
                let b = cast(noise(&mut seed, &[1, 6, hv], 3.0));
                let a_log = noise(&mut seed, &[hv], 1.0);
                let dt_bias = cast(noise(&mut seed, &[hv], 1.0));
                let run = |lo: i32, hi: i32, state: Option<&Array>| {
                    let [q, k, v, a, b] = [&q, &k, &v, &a, &b].map(|x| part(x, lo, hi));
                    update(&q, &k, &v, &a, &b, &a_log, &dt_bias, state).unwrap()
                };
                let (y, state) = run(0, 6, None);
                let (y1, s1) = run(0, 3, None);
                let (y2, s2) = run(3, 6, Some(&s1));
                let what = format!("{device:?} dk {dk} {dtype:?}");
                assert_eq!(state.dtype(), Dtype::Float32, "{what}");
                assert_eq!(bits(&state), bits(&s2), "{what}");
                assert_eq!(bits(&part(&y, 0, 3)), bits(&y1), "{what}");
                assert_eq!(bits(&part(&y, 3, 6)), bits(&y2), "{what}");
            }
        }
    }
}
