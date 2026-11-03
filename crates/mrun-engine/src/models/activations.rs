//! The functions mlx-lm 0.32.0 runs through `mx.compile`, compiled the same way.
//!
//! Compilation is not just a speed-up: a fused kernel can round differently from the same ops run
//! one by one (measured on MLX 0.32.4: `compute_g` on f32 inputs on Metal and CPU, `swiglu` on
//! f16 on CPU), so parity requires compiling exactly what mlx-lm compiles. mlx-lm nests compiled
//! functions (`swiglu` calls the compiled `nn.silu`); one flat compile of the whole body is
//! bitwise identical to the nesting (checked for every dtype on both devices), so each function
//! here is a single trace. Each thread compiles its own copy on first use (MLX arrays and the
//! stream they trace on are per thread).

use std::cell::OnceCell;
use std::thread::LocalKey;

use crate::tensor::{self, Array, Compiled, Dtype, Graph};

thread_local! {
    static SILU: OnceCell<Compiled> = const { OnceCell::new() };
    static SWIGLU: OnceCell<Compiled> = const { OnceCell::new() };
    static PRECISE_SWIGLU: OnceCell<Compiled> = const { OnceCell::new() };
    static COMPUTE_G: OnceCell<Compiled> = const { OnceCell::new() };
}

fn call(
    slot: &'static LocalKey<OnceCell<Compiled>>,
    graph: Graph,
    shapeless: bool,
    inputs: &[&Array],
) -> tensor::Result<Vec<Array>> {
    slot.with(|cell| {
        if cell.get().is_none() {
            // A second initialisation cannot race: the cell is thread-local.
            let _ = cell.set(Compiled::new(graph, shapeless)?);
        }
        cell.get().expect("initialised above").call(inputs)
    })
}

fn one(outputs: Vec<Array>) -> tensor::Result<Array> {
    outputs
        .into_iter()
        .next()
        .ok_or_else(|| tensor::Error("compiled function returned nothing".into()))
}

/// `x · σ(x)`.
fn silu_of(x: &Array) -> tensor::Result<Array> {
    tensor::multiply(x, &tensor::sigmoid(x)?)
}

/// `nn.silu` (compiled, shapeless).
pub fn silu(x: &Array) -> tensor::Result<Array> {
    one(call(&SILU, |a| Ok(vec![silu_of(&a[0])?]), true, &[x])?)
}

/// mlx-lm `swiglu(gate, x) = silu(gate) · x` (compiled, shapeless).
pub fn swiglu(gate: &Array, x: &Array) -> tensor::Result<Array> {
    let graph: Graph = |a| Ok(vec![tensor::multiply(&silu_of(&a[0])?, &a[1])?]);
    one(call(&SWIGLU, graph, true, &[gate, x])?)
}

/// mlx-lm `precise_swiglu(h, gate, x)`: `silu(gate) · x` in f32, cast to `h`'s dtype.
pub fn precise_swiglu(h: &Array, gate: &Array, x: &Array) -> tensor::Result<Array> {
    let graph: Graph = |a| {
        let gate = silu_of(&tensor::astype(&a[1], Dtype::Float32)?)?;
        let x = tensor::astype(&a[2], Dtype::Float32)?;
        Ok(vec![tensor::astype(
            &tensor::multiply(&gate, &x)?,
            a[0].dtype(),
        )?])
    };
    one(call(&PRECISE_SWIGLU, graph, true, &[h, gate, x])?)
}

/// mlx-lm `compute_g`: the per-head decay `exp(−exp(A_log) · softplus(a + dt_bias))`, with
/// `softplus(x) = logaddexp(x, 0)` (`nn.softplus`); f32 because `A_log` is cast to f32.
pub fn compute_g(a_log: &Array, a: &Array, dt_bias: &Array) -> tensor::Result<Array> {
    let graph: Graph = |v| {
        let decay = tensor::exp(&tensor::astype(&v[0], Dtype::Float32)?)?;
        let x = tensor::add(&v[1], &v[2])?;
        let softplus = tensor::logaddexp(&x, &tensor::scalar(0.0, x.dtype())?)?;
        let rate = tensor::multiply(&tensor::negative(&decay)?, &softplus)?;
        Ok(vec![tensor::exp(&rate)?])
    };
    one(call(&COMPUTE_G, graph, true, &[a_log, a, dt_bias])?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::{Device, use_device};

    #[test]
    fn compiled_activations_compute_their_formulas() {
        use_device(Device::Cpu);
        let g = Array::from_f32(&[-2.0, 0.0, 1.5], &[3]);
        let x = Array::from_f32(&[1.0, 2.0, -3.0], &[3]);
        let sig = |v: f32| 1.0 / (1.0 + (-v).exp());
        let expect: Vec<f32> = [(-2.0f32, 1.0f32), (0.0, 2.0), (1.5, -3.0)]
            .iter()
            .map(|&(g, x)| g * sig(g) * x)
            .collect();
        for (got, want) in [
            (swiglu(&g, &x).unwrap(), &expect),
            (precise_swiglu(&x, &g, &x).unwrap(), &expect),
        ] {
            for (a, b) in got.to_vec_f32().unwrap().iter().zip(want) {
                assert!((a - b).abs() < 1e-5, "{a} vs {b}");
            }
        }
        let s = silu(&g).unwrap().to_vec_f32().unwrap();
        assert!((s[2] - 1.5 * sig(1.5)).abs() < 1e-6);
        // g = exp(−exp(A) · softplus(a + dt)).
        let out = compute_g(
            &Array::from_f32(&[0.5], &[1]),
            &Array::from_f32(&[0.25], &[1]),
            &Array::from_f32(&[-1.0], &[1]),
        )
        .unwrap()
        .item_f32()
        .unwrap();
        let want = (-(0.5f32.exp()) * (1.0 + (-0.75f32).exp()).ln()).exp();
        assert!((out - want).abs() < 1e-6, "{out} vs {want}");
    }
}
