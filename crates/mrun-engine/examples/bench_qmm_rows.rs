//! Batched quantized-matmul bandwidth probe (011 T001a, research R16): times MLX's
//! `quantized_matmul` and `gather_qmm` on the weight shapes of real models with `[1, B, hidden]`
//! inputs at B = 1, 2, 4, and models the aggregate decode ratio of B lanes.
//!
//! ```text
//! cargo run --release -p mrun-engine --features mlx --example bench_qmm_rows -- \
//!     [--step-ms qwen3=21.9,qwen3.5=17.0] [--reps 15]
//! ```
//!
//! The weights are synthetic (4-bit affine, group size 64, as the checkpoints; 8-bit routers for
//! `qwen3.6`), so no file is read: kernel time does not depend on the values. Per shape the
//! table gives the time per call (median of `--reps` runs of 16 independent calls evaluated
//! together, so launch overhead is amortized as in a decode step's command buffer) and the
//! effective weight bandwidth. Per model, `weights(B)` is one decode step's weight passes (every
//! projection of every layer plus the head; for MoE the experts B rows select, distinct per row).
//! With a measured one-lane step time `T1` (`--step-ms`, from `bench_engine` decode), the
//! per-lane rest is `T1 − weights(1)` and the modeled aggregate ratio at B lanes is
//! `B · T1 / (weights(B) + B · (T1 − weights(1)))` (1.0 = no gain, B = perfect sharing).

use std::collections::BTreeMap;
use std::time::Instant;

use mrun_engine::tensor::{self, Array, Dtype};

const GROUP: i32 = 64;

/// One weight matrix of a decode step.
#[derive(Clone, Copy)]
struct Proj {
    name: &'static str,
    out: i32,
    inp: i32,
    bits: i32,
    /// How often a decode step runs it.
    count: u32,
    /// `Some((experts, top_k))`: a `gather_qmm` over stacked experts.
    experts: Option<(i32, i32)>,
}

const fn p(name: &'static str, out: i32, inp: i32, count: u32) -> Proj {
    Proj {
        name,
        out,
        inp,
        bits: 4,
        count,
        experts: None,
    }
}

fn models() -> Vec<(&'static str, Vec<Proj>)> {
    // Qwen3-8B: 36 attention layers, hidden 4096, 32/8 heads × 128, MLP 12,288.
    let qwen3 = vec![
        p("q_proj", 4096, 4096, 36),
        p("k_proj", 1024, 4096, 36),
        p("v_proj", 1024, 4096, 36),
        p("o_proj", 4096, 4096, 36),
        p("gate_proj", 12_288, 4096, 36),
        p("up_proj", 12_288, 4096, 36),
        p("down_proj", 4096, 12_288, 36),
        p("lm_head", 151_936, 4096, 1),
    ];
    // Qwen3.5-9B: 24 Gated DeltaNet + 8 gated attention layers, hidden 4096, dense MLP 12,288.
    let qwen3_5 = vec![
        p("in_proj_qkv", 8192, 4096, 24),
        p("in_proj_z", 4096, 4096, 24),
        p("in_proj_b", 32, 4096, 24),
        p("in_proj_a", 32, 4096, 24),
        p("out_proj", 4096, 4096, 24),
        p("q_proj (gated)", 8192, 4096, 8),
        p("k_proj", 1024, 4096, 8),
        p("v_proj", 1024, 4096, 8),
        p("o_proj", 4096, 4096, 8),
        p("gate_proj", 12_288, 4096, 32),
        p("up_proj", 12_288, 4096, 32),
        p("down_proj", 4096, 12_288, 32),
        p("lm_head", 248_320, 4096, 1),
    ];
    // Qwen3.6-35B-A3B: 30 Gated DeltaNet + 10 gated attention layers, hidden 2048, 256 experts
    // top-8 (512) plus a shared expert (512) under an 8-bit gate, 8-bit router.
    let moe = |name, out, inp| Proj {
        name,
        out,
        inp,
        bits: 4,
        count: 40,
        experts: Some((256, 8)),
    };
    let eight = |name, out, inp| Proj {
        name,
        out,
        inp,
        bits: 8,
        count: 40,
        experts: None,
    };
    let qwen3_6 = vec![
        p("in_proj_qkv", 8192, 2048, 30),
        p("in_proj_z", 4096, 2048, 30),
        p("in_proj_b", 32, 2048, 30),
        p("in_proj_a", 32, 2048, 30),
        p("out_proj", 2048, 4096, 30),
        p("q_proj (gated)", 8192, 2048, 10),
        p("k_proj", 512, 2048, 10),
        p("v_proj", 512, 2048, 10),
        p("o_proj", 2048, 4096, 10),
        eight("router", 256, 2048),
        eight("shared_expert_gate", 1, 2048),
        moe("experts.gate_proj", 512, 2048),
        moe("experts.up_proj", 512, 2048),
        moe("experts.down_proj", 2048, 512),
        p("shared.gate_proj", 512, 2048, 40),
        p("shared.up_proj", 512, 2048, 40),
        p("shared.down_proj", 2048, 512, 40),
        p("lm_head", 248_320, 2048, 1),
    ];
    vec![("qwen3", qwen3), ("qwen3.5", qwen3_5), ("qwen3.6", qwen3_6)]
}

/// Synthetic packed weights `[experts?, out, in × bits / 32]` with bf16 scales and biases.
fn weights(p: &Proj) -> tensor::Result<tensor::Quantized> {
    let e = p.experts.map_or(1, |(e, _)| e);
    let packed = p.inp * p.bits / 32;
    let n = (e * p.out * packed) as usize;
    let data: Vec<u32> = (0..n as u32)
        .map(|i| i.wrapping_mul(2_654_435_761) ^ 0x5bd1_e995)
        .collect();
    let groups = p.inp / GROUP;
    let shape = |last: i32| -> Vec<i32> {
        if p.experts.is_some() {
            vec![e, p.out, last]
        } else {
            vec![p.out, last]
        }
    };
    let w = Array::from_u32(&data, &shape(packed));
    drop(data);
    let s = (e * p.out * groups) as usize;
    let scales = tensor::astype(
        &Array::from_f32(&vec![0.01; s], &shape(groups)),
        Dtype::Bfloat16,
    )?;
    let biases = tensor::astype(
        &Array::from_f32(&vec![-0.05; s], &shape(groups)),
        Dtype::Bfloat16,
    )?;
    tensor::eval(&[&w, &scales, &biases])?;
    tensor::Quantized::new(w, scales, Some(biases), GROUP, p.bits)
}

/// Bytes one call reads for `rows` rows: the weight matrix (for experts, the distinct experts
/// the rows select, `top_k` each, disjoint across rows).
fn bytes(p: &Proj, rows: i32) -> f64 {
    let matrix = f64::from(p.out) * f64::from(p.inp) * f64::from(p.bits) / 8.0
        + 2.0 * f64::from(p.out) * f64::from(p.inp / GROUP) * 2.0;
    match p.experts {
        Some((e, k)) => matrix * f64::from((rows * k).min(e)),
        None => matrix,
    }
}

/// Median milliseconds per call of `p` at `rows` rows.
fn time(p: &Proj, w: &tensor::Quantized, rows: i32, reps: usize) -> tensor::Result<f64> {
    const CALLS: usize = 16;
    let xs: Vec<Array> = (0..CALLS)
        .map(|c| {
            let data: Vec<f32> = (0..rows * p.inp)
                .map(|i| ((i as f32 + c as f32 * 0.37) * 0.001).sin())
                .collect();
            let x = Array::from_f32(&data, &[1, rows, p.inp]);
            tensor::astype(&x, Dtype::Bfloat16)
        })
        .collect::<tensor::Result<_>>()?;
    let refs: Vec<&Array> = xs.iter().collect();
    tensor::eval(&refs)?;
    let indices = p.experts.map(|(e, k)| {
        // Distinct experts per row, spread over the range.
        let ids: Vec<u32> = (0..rows * k).map(|i| ((i * 37 + 5) % e) as u32).collect();
        Array::from_u32(&ids, &[1, rows, k])
    });
    let call = |x: &Array| -> tensor::Result<Array> {
        match &indices {
            None => tensor::quantized_matmul(x, w),
            Some(idx) => {
                let x = tensor::expand_dims(x, &[-2, -3])?;
                tensor::gather_qmm(&x, w, idx, false)
            }
        }
    };
    let mut runs = Vec::with_capacity(reps);
    for rep in 0..reps + 2 {
        let start = Instant::now();
        let outs: Vec<Array> = xs.iter().map(&call).collect::<tensor::Result<_>>()?;
        let refs: Vec<&Array> = outs.iter().collect();
        tensor::eval(&refs)?;
        let ms = start.elapsed().as_secs_f64() * 1e3 / CALLS as f64;
        if rep >= 2 {
            runs.push(ms);
        }
    }
    runs.sort_by(f64::total_cmp);
    Ok(runs[runs.len() / 2])
}

fn run() -> Result<(), String> {
    let mut step_ms: BTreeMap<String, f64> = BTreeMap::new();
    let mut reps = 15;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--step-ms" => {
                for pair in args.next().ok_or("--step-ms needs a value")?.split(',') {
                    let (model, ms) = pair.split_once('=').ok_or("--step-ms model=ms,...")?;
                    step_ms.insert(
                        model.into(),
                        ms.parse().map_err(|e| format!("{pair}: {e}"))?,
                    );
                }
            }
            "--reps" => {
                reps = args
                    .next()
                    .ok_or("--reps needs a value")?
                    .parse()
                    .map_err(|e| format!("--reps: {e}"))?;
            }
            other => return Err(format!("unexpected argument {other}")),
        }
    }
    tensor::metallib::ensure()?;
    if !tensor::metal_available() {
        return Err("needs Metal".into());
    }
    println!(
        "{} · {}",
        tensor::gpu_info_string(c"device_name").unwrap_or_default(),
        tensor::numerics()
    );
    for (model, projs) in models() {
        println!("\n### {model}\n");
        println!(
            "| weight (out × in, bits) | per step | B=1 ms | GB/s | B=2 ms | GB/s | B=4 ms | GB/s |"
        );
        println!("|---|---|---|---|---|---|---|---|");
        let mut step = [0.0f64; 3];
        for p in &projs {
            let w = weights(p).map_err(|e| e.0)?;
            let mut cells = Vec::new();
            for (i, rows) in [1, 2, 4].into_iter().enumerate() {
                let ms = time(p, &w, rows, reps).map_err(|e| e.0)?;
                step[i] += ms * f64::from(p.count);
                cells.push(format!("{ms:.4} | {:.0}", bytes(p, rows) / ms / 1e6));
            }
            let kind = match p.experts {
                Some((e, k)) => format!(", {e} experts top-{k}"),
                None => String::new(),
            };
            println!(
                "| {} ({} × {}, {}-bit{kind}) | {}× | {} |",
                p.name,
                p.out,
                p.inp,
                p.bits,
                p.count,
                cells.join(" | ")
            );
            drop(w);
            tensor::memory::clear_cache();
        }
        println!(
            "\nweights per decode step: B=1 {:.2} ms, B=2 {:.2} ms ({:.2}×), B=4 {:.2} ms ({:.2}×)",
            step[0],
            step[1],
            step[1] / step[0],
            step[2],
            step[2] / step[0]
        );
        if let Some(&t1) = step_ms.get(model) {
            let rest = (t1 - step[0]).max(0.0);
            let ratio = |b: f64, w: f64| b * t1 / (w + b * rest);
            println!(
                "measured 1-lane step {t1:.2} ms (rest {rest:.2} ms per lane): modeled aggregate \
                 decode ratio B=2 {:.2}×, B=4 {:.2}×",
                ratio(2.0, step[1]),
                ratio(4.0, step[2])
            );
        }
    }
    Ok(())
}

fn main() -> std::process::ExitCode {
    match run() {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("error: {message}");
            std::process::ExitCode::FAILURE
        }
    }
}
