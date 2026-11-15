//! 009 SC-001: mrun's vision tower against mlx-vlm 0.7.12 on the same patches.
//!
//! `MRUN_VISION_REFS="<model dir>=<reference.safetensors>,…"` (references from
//! `scripts/gen_vision_reference.py`); without it the test is skipped.
#![cfg(feature = "engine")]

use mrun_engine::engine::CancelToken;
use mrun_engine::load::Weights;
use mrun_engine::models::vision::{VisionTower, is_vision_tensor};
use mrun_engine::tensor::{self, Array};
use std::path::Path;

fn tower(dir: &Path) -> VisionTower {
    let config: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("config.json")).unwrap()).unwrap();
    let mut weights = Weights::load(dir, is_vision_tensor).unwrap();
    VisionTower::load(&mut weights, &config["vision_config"]).unwrap()
}

#[test]
#[ignore = "needs MRUN_VISION_REFS (model dirs and mlx-vlm references)"]
fn the_tower_matches_mlx_vlm() {
    let Ok(refs) = std::env::var("MRUN_VISION_REFS") else {
        eprintln!("MRUN_VISION_REFS not set: skipped");
        return;
    };
    for pair in refs.split(',').filter(|p| !p.is_empty()) {
        let (dir, reference) = pair.split_once('=').expect("<model dir>=<reference>");
        let tensors = tensor::load_safetensors(Path::new(reference)).unwrap();
        let get =
            |name: &str| -> Array { tensors.iter().find(|(n, _)| n == name).unwrap().1.clone() };
        let grid = get("grid").to_vec_u32().unwrap();
        let tower = tower(Path::new(dir));
        let ours = tower
            .encode(
                &get("patches"),
                grid[0] as i32,
                grid[1] as i32,
                &CancelToken::new(),
            )
            .unwrap()
            .expect("not cancelled");
        let ours = tensor::astype(&ours, tensor::Dtype::Float32)
            .unwrap()
            .to_vec_f32()
            .unwrap();
        let theirs = get("features").to_vec_f32().unwrap();
        assert_eq!(ours.len(), theirs.len(), "{dir}");
        let dot: f64 = ours
            .iter()
            .zip(&theirs)
            .map(|(a, b)| f64::from(*a) * f64::from(*b))
            .sum();
        let norm = |v: &[f32]| v.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>().sqrt();
        let cosine = dot / (norm(&ours) * norm(&theirs));
        let scale = theirs.iter().fold(0f32, |m, x| m.max(x.abs()));
        let worst = ours
            .iter()
            .zip(&theirs)
            .fold(0f32, |m, (a, b)| m.max((a - b).abs()));
        eprintln!("{dir}: cosine {cosine:.6}, max abs diff {worst:.4} (scale {scale:.2})");
        // bf16 over 27 blocks of different kernels: the embedding matches exactly and block 1
        // to 0.99999; the drift that remains is rounding (0.99954 on Qwen3.6, 0.99840 on
        // Qwen3.5, whose last block has activations above 1,000).
        assert!(cosine >= 0.998, "{dir}: cosine {cosine}");
        assert!(
            worst <= scale / 8.0,
            "{dir}: max abs diff {worst} at scale {scale}"
        );
    }
}

#[test]
#[ignore = "debugging aid: MRUN_VISION_STAGES=<model dir>=<stages.safetensors>"]
fn stages_against_mlx_vlm() {
    let Ok(pair) = std::env::var("MRUN_VISION_STAGES") else {
        return;
    };
    let (dir, reference) = pair.split_once('=').unwrap();
    let tensors = tensor::load_safetensors(Path::new(reference)).unwrap();
    let tower = tower(Path::new(dir));
    let (gh, gw, n) = (8, 12, 96);
    let width = 2 * 16 * 16 * 3;
    let mine: Vec<f32> = (0..n)
        .flat_map(|i| (0..width).map(move |j| (i as f32 * 0.37 + j as f32 * 0.011).sin()))
        .collect();
    let patches = Array::from_f32(&mine, &[n, width]);
    let mut stages = Vec::new();
    tower
        .encode_traced(&patches, gh, gw, &CancelToken::new(), &mut stages)
        .unwrap();
    for (k, ours) in stages.iter().enumerate() {
        let theirs = tensors
            .iter()
            .find(|(name, _)| *name == format!("s{k}"))
            .unwrap()
            .1
            .to_vec_f32()
            .unwrap();
        let ours = tensor::astype(ours, tensor::Dtype::Float32)
            .unwrap()
            .to_vec_f32()
            .unwrap();
        let dot: f64 = ours
            .iter()
            .zip(&theirs)
            .map(|(a, b)| f64::from(*a) * f64::from(*b))
            .sum();
        let norm = |v: &[f32]| v.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>().sqrt();
        let worst = ours
            .iter()
            .zip(&theirs)
            .fold(0f32, |m, (a, b)| m.max((a - b).abs()));
        eprintln!(
            "stage {k}: cosine {:.6} max diff {worst:.4}",
            dot / (norm(&ours) * norm(&theirs))
        );
    }
}

/// 009 SC-002: encoding a Claude-size screenshot (1,568 × 992 → 31 × 49 tokens) and the cap.
#[test]
#[ignore = "timing on the target Mac: MRUN_VISION_REFS (model dirs)"]
fn encoding_time() {
    let Ok(refs) = std::env::var("MRUN_VISION_REFS") else {
        return;
    };
    for pair in refs.split(',').filter(|p| !p.is_empty()) {
        let dir = pair.split_once('=').map_or(pair, |(d, _)| d);
        let tower = tower(Path::new(dir));
        for (gh, gw) in [(62, 98), (80, 128)] {
            let width = 2 * 16 * 16 * 3;
            let n = gh * gw;
            let data: Vec<f32> = (0..n * width)
                .map(|i| ((i % 97) as f32 / 48.0) - 1.0)
                .collect();
            let patches = Array::from_f32(&data, &[n, width]);
            let _ = tower.encode(&patches, gh, gw, &CancelToken::new()).unwrap();
            let mut best = f64::MAX;
            for _ in 0..3 {
                let t = std::time::Instant::now();
                let _ = tower.encode(&patches, gh, gw, &CancelToken::new()).unwrap();
                best = best.min(t.elapsed().as_secs_f64());
            }
            eprintln!("{dir}: {} tokens encoded in {best:.3} s", n / 4);
        }
    }
}
