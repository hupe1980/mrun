//! Greedy-token agreement between two engine builds (spec 016 T018, US2 AS3), e.g. before and
//! after a nuvec update.
//!
//! ```text
//! # with one build: greedy tokens and top-2 gaps for the client-gate prompts
//! cargo run --release -p mrun --example token_agreement -- record <model-dir> <a.json> [tokens]
//! # with the other build: the same prompts, fed `a.json`'s tokens (teacher forcing)
//! cargo run --release -p mrun --example token_agreement \
//!     -- follow <model-dir> <a.json> <report.json>
//! ```
//!
//! Prompts are the client-gate tasks (`tests/clients/tasks/*/PROMPT.txt`), each as one user
//! message through the model's chat template. `follow` feeds the recorded tokens, so every
//! position is compared, not only those before the first divergence: a position agrees when
//! this build's greedy choice is the recorded token. A disagreement is a near-tie when the top-2
//! log-probability gap is at most one bf16 step in either build (spec 022 FR-004), the step taken
//! at the smallest logit magnitude seen ([`bf16_step`]), so the rule can only count fewer ties,
//! never more. Passes at 95 % agreement (FR-004); prints the agreement and every disagreement
//! that is not a near-tie with both builds' gaps.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use mrun::chat::template::{ChatTemplate, RenderOptions};
use mrun::chat::tokenizer::Tokenizer;
use mrun_engine::{CancelToken, Engine, LoadOptions, NuvecEngine, Sampling, TokenMask};
use serde_json::{Value, json};

const PREFILL_STEP: usize = 2048;

fn prompts() -> Result<Vec<(String, String)>, String> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/clients/tasks");
    let mut tasks: Vec<PathBuf> = std::fs::read_dir(&dir)
        .map_err(|e| format!("{}: {e}", dir.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.join("PROMPT.txt").is_file())
        .collect();
    tasks.sort();
    tasks
        .iter()
        .map(|task| {
            let text = std::fs::read_to_string(task.join("PROMPT.txt"))
                .map_err(|e| format!("{}: {e}", task.display()))?;
            let name = task.file_name().unwrap_or_default().to_string_lossy();
            Ok((name.into_owned(), text.trim().to_string()))
        })
        .collect()
}

/// The prompt's token ids: one user message through the chat template.
fn encode(dir: &Path, prompt: &str) -> Result<Vec<u32>, String> {
    let template = ChatTemplate::from_model_dir(dir).map_err(|e| e.to_string())?;
    let tokenizer = Tokenizer::from_file(&dir.join("tokenizer.json")).map_err(|e| e.to_string())?;
    let text = template
        .render(
            &json!([{ "role": "user", "content": prompt }]),
            &RenderOptions {
                add_generation_prompt: true,
                enable_thinking: true,
                tools: None,
                preserve_thinking: false,
            },
        )
        .map_err(|e| e.to_string())?;
    tokenizer.encode(&text).map_err(|e| e.to_string())
}

fn load(dir: &Path) -> Result<NuvecEngine, String> {
    NuvecEngine::load(
        dir,
        LoadOptions {
            identity: "token_agreement".into(),
            ..LoadOptions::default()
        },
    )
    .map_err(|e| e.to_string())
}

/// One bf16 step at the smallest logit magnitude behind `gaps`: the largest power of two (from
/// 2^-10 to 2^3) that divides every finite, nonzero gap.
fn bf16_step(gaps: impl IntoIterator<Item = f32>) -> f32 {
    let gaps: Vec<f32> = gaps
        .into_iter()
        .filter(|g| g.is_finite() && *g > 0.0)
        .collect();
    (-10..=3)
        .rev()
        .map(|k| 2f32.powi(k))
        .find(|step| gaps.iter().all(|g| (g / step).fract() == 0.0))
        .unwrap_or(2f32.powi(-10))
}

/// The gap between the two most likely tokens of `top` (log-probabilities).
fn gap(top: &[(u32, f32)]) -> f32 {
    match top {
        [(_, a), (_, b), ..] => a - b,
        _ => f32::INFINITY,
    }
}

/// Prefills all prompt tokens but the last, then feeds the last: the first decode step.
fn start(engine: &mut NuvecEngine, ids: &[u32], total: u32) -> Result<mrun_engine::Step, String> {
    let error = |e: mrun_engine::EngineError| e.to_string();
    engine.trim(0).map_err(error)?;
    engine.set_limit(total).map_err(error)?;
    let (last, body) = ids.split_last().ok_or("empty prompt")?;
    let cancel = CancelToken::new();
    for chunk in body.chunks(PREFILL_STEP) {
        engine.prefill_chunk(chunk, &cancel).map_err(error)?;
    }
    engine.forward(*last).map_err(error)
}

fn record(dir: &Path, out: &Path, tokens: u32) -> Result<(), String> {
    let error = |e: mrun_engine::EngineError| e.to_string();
    let mut engine = load(dir)?;
    let eos = engine.info().eos_tokens.clone();
    let mut runs = Vec::new();
    for (name, prompt) in prompts()? {
        let ids = encode(dir, &prompt)?;
        let mut step = start(&mut engine, &ids, ids.len() as u32 + tokens)?;
        let (mut out_ids, mut gaps) = (Vec::new(), Vec::new());
        loop {
            gaps.push(gap(&engine.top_logprobs(&step, 2).map_err(error)?));
            let token = engine
                .sample(step, &Sampling::GREEDY, None)
                .map_err(error)?;
            out_ids.push(token);
            if out_ids.len() as u32 >= tokens || eos.contains(&token) {
                break;
            }
            step = engine.forward(token).map_err(error)?;
        }
        eprintln!(
            "{name}: {} prompt tokens, {} generated",
            ids.len(),
            out_ids.len()
        );
        runs.push(json!({ "task": name, "prompt": ids, "tokens": out_ids, "gaps": gaps }));
    }
    let record = json!({
        "numerics": mrun_engine::tensor::numerics().to_string(),
        "model": dir.display().to_string(),
        "runs": runs,
    });
    std::fs::write(
        out,
        serde_json::to_string(&record).map_err(|e| e.to_string())?,
    )
    .map_err(|e| format!("{}: {e}", out.display()))
}

fn follow(dir: &Path, recorded: &Path, report: &Path) -> Result<bool, String> {
    let error = |e: mrun_engine::EngineError| e.to_string();
    let text =
        std::fs::read_to_string(recorded).map_err(|e| format!("{}: {e}", recorded.display()))?;
    let record: Value = serde_json::from_str(&text).map_err(|e| e.to_string())?;
    let mut engine = load(dir)?;
    let vocab = engine.info().vocab;
    let ids = |v: &Value| -> Vec<u32> {
        v.as_array()
            .map(|a| {
                a.iter()
                    .filter_map(|x| x.as_u64().map(|x| x as u32))
                    .collect()
            })
            .unwrap_or_default()
    };
    let all_gaps = record["runs"]
        .as_array()
        .into_iter()
        .flatten()
        .flat_map(|run| run["gaps"].as_array().cloned().unwrap_or_default())
        .filter_map(|g| g.as_f64().map(|g| g as f32));
    let near_tie = bf16_step(all_gaps);
    let (mut agree, mut total, mut near_ties) = (0usize, 0usize, 0usize);
    let (mut tasks, mut real) = (Vec::new(), Vec::new());
    for run in record["runs"].as_array().ok_or("no runs in the record")? {
        let name = run["task"].as_str().unwrap_or("?");
        let (prompt, tokens) = (ids(&run["prompt"]), ids(&run["tokens"]));
        let gaps_a: Vec<f32> = run["gaps"]
            .as_array()
            .map(|a| {
                a.iter()
                    .map(|g| g.as_f64().unwrap_or(f64::INFINITY) as f32)
                    .collect()
            })
            .unwrap_or_default();
        let mut step = start(&mut engine, &prompt, (prompt.len() + tokens.len()) as u32)?;
        let mut same = 0;
        for (i, &want) in tokens.iter().enumerate() {
            let top = engine.top_logprobs(&step, 2).map_err(error)?;
            let (best, gap_b) = (top.first().map_or(u32::MAX, |t| t.0), gap(&top));
            if best == want {
                same += 1;
            } else {
                let gap_a = gaps_a.get(i).copied().unwrap_or(f32::INFINITY);
                if gap_a.min(gap_b) <= near_tie {
                    near_ties += 1;
                } else {
                    real.push(json!({ "task": name, "position": i, "recorded": want,
                        "this_build": best, "gap_recorded": gap_a, "gap_this_build": gap_b }));
                }
            }
            if i + 1 == tokens.len() {
                break;
            }
            let mut words = vec![0u32; vocab.div_ceil(32) as usize];
            words[(want / 32) as usize] |= 1 << (want % 32);
            let mask = TokenMask::from_words(&words, vocab);
            engine
                .sample(step, &Sampling::GREEDY, Some(&mask))
                .map_err(error)?;
            step = engine.forward(want).map_err(error)?;
        }
        eprintln!("{name}: {same}/{} positions agree", tokens.len());
        tasks.push(json!({ "task": name, "agree": same, "positions": tokens.len() }));
        agree += same;
        total += tokens.len();
    }
    let rate = agree as f64 / total.max(1) as f64;
    // FR-004: the agreement bar; the divergences are listed, not gated (one step can only be
    // bounded from below here, see `bf16_step`).
    let pass = rate >= 0.95;
    let summary = json!({
        "recorded": record["numerics"], "this_build": mrun_engine::tensor::numerics().to_string(),
        "model": dir.display().to_string(), "agreement": rate, "positions": total,
        "bf16_step": near_tie, "near_ties": near_ties, "other_divergences": real, "tasks": tasks, "pass": pass,
    });
    std::fs::write(
        report,
        serde_json::to_string_pretty(&summary).map_err(|e| e.to_string())?,
    )
    .map_err(|e| format!("{}: {e}", report.display()))?;
    println!(
        "agreement {:.2} % over {total} positions; {near_ties} near-ties, {} other divergences: {}",
        100.0 * rate,
        summary["other_divergences"].as_array().map_or(0, Vec::len),
        if pass { "pass" } else { "FAIL" }
    );
    Ok(pass)
}

fn run() -> Result<bool, String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args
        .iter()
        .map(String::as_str)
        .collect::<Vec<_>>()
        .as_slice()
    {
        ["record", dir, out, rest @ ..] => {
            let tokens = match rest.first() {
                Some(n) => n
                    .parse()
                    .map_err(|e| format!("bad token count {n:?}: {e}"))?,
                None => 256,
            };
            record(Path::new(dir), Path::new(out), tokens).map(|()| true)
        }
        ["follow", dir, recorded, report] => {
            follow(Path::new(dir), Path::new(recorded), Path::new(report))
        }
        _ => Err(
            "usage: token_agreement record <model-dir> <out.json> [tokens] | \
                  follow <model-dir> <recorded.json> <report.json>"
                .into(),
        ),
    }
}

fn main() -> ExitCode {
    match run() {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::from(1),
        Err(message) => {
            eprintln!("error: {message}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::bf16_step;

    #[test]
    fn the_step_is_the_finest_power_of_two_dividing_every_gap() {
        // Logits in [16, 32): gaps in steps of 0.125.
        assert_eq!(bf16_step([0.125, 0.25, 0.375, 1.0, 3.5]), 0.125);
        // One logit in [8, 16) makes the step finer.
        assert_eq!(bf16_step([0.125, 0.0625, 0.5]), 0.0625);
        // [32, 64): 0.25.
        assert_eq!(bf16_step([0.25, 0.75, 2.0]), 0.25);
        // Zero, infinite and NaN gaps say nothing about the step.
        assert_eq!(bf16_step([0.0, f32::INFINITY, f32::NAN, 0.5, 1.5]), 0.5);
        assert_eq!(bf16_step([]), 8.0);
    }
}
