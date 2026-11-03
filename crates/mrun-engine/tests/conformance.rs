//! Greedy parity with mlx-lm 0.32.0 on real models (SC-001, US4), against the references
//! `scripts/gen_reference.py` writes to `tests/reference/<model>.json`.
//!
//! ```text
//! MRUN_TEST_MODELS=qwen3,qwen3.5 cargo test -p mrun-engine --features mlx --release \
//!     --test conformance -- --ignored --nocapture
//! ```
//!
//! Every prompt is prefilled in the engine's own chunk plan (`ModelInfo::chunk_len`, the
//! calibrated split mrun uses; `MRUN_CONFORMANCE_SPLIT=2048` uses mlx-lm's fixed split instead),
//! its last token fed by `forward`, then 256 greedy tokens are sampled (pipelined, as the runtime
//! does). A prompt passes when every token matches, or when the first divergence is a numerical
//! tie: the reference's top-2 margin is below 1e-3 and mrun's token is the reference's second
//! choice (Clarifications). The report names the first divergence and its margin. The engine's
//! end tokens must include mlx-lm's set (extra ones may not occur in the references), and
//! references from another numerics class
//! (FR-027) are refused.

#![cfg(feature = "engine")]

mod common;

use mrun_engine::{Engine, Sampling};
use serde::Deserialize;

/// Tie margin (Clarifications).
const TIE: f32 = 1e-3;
/// A reference must hold at least this many prompts (SC-001; T072's minimum count).
const MIN_PROMPTS: usize = 20;

#[derive(Deserialize)]
struct Case {
    prompt: Vec<u32>,
    tokens: Vec<u32>,
    top2: Vec<[(u32, f32); 2]>,
}

#[derive(Deserialize)]
struct Numerics {
    gpu_arch: String,
    nax: bool,
}

#[derive(Deserialize)]
struct Reference {
    eos: Vec<u32>,
    numerics: Numerics,
    cases: Vec<Case>,
}

enum Outcome {
    Match,
    Tie {
        at: usize,
        margin: f32,
    },
    Diverged {
        at: usize,
        got: u32,
        want: u32,
        margin: f32,
    },
}

#[test]
#[ignore = "needs real weights: MRUN_TEST_MODELS"]
fn greedy_output_matches_mlx_lm() {
    let split = std::env::var("MRUN_CONFORMANCE_SPLIT")
        .ok()
        .map(|s| s.parse::<u32>().expect("MRUN_CONFORMANCE_SPLIT"));
    let mut compared = 0;
    let mut failures = Vec::new();
    for (name, dir) in common::real_models() {
        let path = common::fixture("../reference").join(format!("{name}.json"));
        let text = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("{}: {e} (run scripts/gen_reference.py)", path.display()));
        let reference: Reference = serde_json::from_str(&text).unwrap();
        assert!(
            reference.cases.len() >= MIN_PROMPTS,
            "{name}: {} prompts, SC-001 needs {MIN_PROMPTS}",
            reference.cases.len()
        );
        let mut engine = common::load_real(&dir);
        let info = engine.info().clone();
        assert_eq!(
            (&reference.numerics.gpu_arch, reference.numerics.nax),
            (&info.numerics.gpu_arch, info.numerics.nax),
            "{name}: the reference was generated on another numerics class"
        );
        // mrun's end tokens include mlx-lm's. mrun also reads a wrapped config's
        // `text_config.eos_token_id` (Qwen3.5: `<|endoftext|>`), which mlx-lm 0.32.0 ignores; such
        // an extra token must not occur in any reference output (mrun would stop there).
        let mut eos = info.eos_tokens.clone();
        eos.sort_unstable();
        let missing: Vec<u32> = reference
            .eos
            .iter()
            .copied()
            .filter(|t| !eos.contains(t))
            .collect();
        assert!(
            missing.is_empty(),
            "{name}: mlx-lm's end tokens {missing:?} are missing"
        );
        let extra: Vec<u32> = eos
            .iter()
            .copied()
            .filter(|t| !reference.eos.contains(t))
            .collect();
        if !extra.is_empty() {
            eprintln!("{name}: extra end tokens beyond mlx-lm's: {extra:?}");
        }
        for (i, case) in reference.cases.iter().enumerate() {
            assert!(
                !case.tokens.iter().any(|t| extra.contains(t)),
                "{name} prompt {i}: the reference generates an end token mlx-lm does not stop at"
            );
        }
        eprintln!(
            "{name}: {} prompts, split {}",
            reference.cases.len(),
            split.map_or(format!("engine {:?}", info.time_len.points()), |s| s
                .to_string())
        );

        let (mut matched, mut ties) = (0, 0);
        for (i, case) in reference.cases.iter().enumerate() {
            let outcome = run(&mut engine, case, split);
            let len = case.prompt.len();
            match outcome {
                Outcome::Match => matched += 1,
                Outcome::Tie { at, margin } => {
                    ties += 1;
                    eprintln!("  prompt {i} ({len} tokens): tie at {at} (margin {margin:.2e})");
                }
                Outcome::Diverged {
                    at,
                    got,
                    want,
                    margin,
                } => {
                    let message = format!(
                        "{name} prompt {i} ({len} tokens): diverged at output {at}: {got} instead \
                         of {want} (reference margin {margin:.3e})"
                    );
                    eprintln!("  {message}");
                    failures.push(message);
                }
            }
            compared += 1;
        }
        eprintln!(
            "{name}: {matched} identical, {ties} tie divergences, {} failures of {}",
            reference.cases.len() - matched - ties,
            reference.cases.len()
        );
    }
    assert!(compared >= MIN_PROMPTS, "only {compared} prompts compared");
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

fn run(engine: &mut mrun_engine::MlxEngine, case: &Case, split: Option<u32>) -> Outcome {
    engine.reset();
    let n = case.tokens.len() as u32;
    engine.set_limit(case.prompt.len() as u32 + n).unwrap();
    let (last, body) = case.prompt.split_last().unwrap();
    common::prefill_plan(engine, body, split);
    let mut step = engine.forward(*last).unwrap();
    for (at, (&want, top2)) in case.tokens.iter().zip(&case.top2).enumerate() {
        let got = engine.sample(step, &Sampling::GREEDY, None).unwrap();
        if got != want {
            let margin = top2[0].1 - top2[1].1;
            if margin < TIE && got == top2[1].0 {
                return Outcome::Tie { at, margin };
            }
            return Outcome::Diverged {
                at,
                got,
                want,
                margin,
            };
        }
        if at + 1 == case.tokens.len() {
            break;
        }
        step = engine.forward(got).unwrap();
    }
    engine.reset();
    Outcome::Match
}
