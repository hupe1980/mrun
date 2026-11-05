//! Greedy generation from token ids, for parity and speed checks against mlx-lm.
//!
//! ```text
//! cargo run --release -p mrun-engine --features engine --example generate -- \
//!     <model-dir> <ids> [max-tokens] [--cpu]
//! ```
//!
//! `<ids>` is a file of token ids separated by commas or whitespace, or such a list inline. No
//! chat template is applied: feed the ids mlx-lm printed. Like mlx-lm's `generate_step`, all
//! prompt tokens but the last are prefilled (in chunks of 2048, mlx-lm's default
//! `prefill_step_size`), the last is fed by `forward`, and decoding is pipelined. Generation
//! stops after `max-tokens` (default 64) or at an end token; prints the ids, then the prompt and
//! generation speeds measured as mlx-lm does (prompt: until the first token is read back).

use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use mrun_engine::{CancelToken, Engine, LoadOptions, NuvecEngine, Sampling};

const PREFILL_STEP: usize = 2048;

fn parse_ids(arg: &str) -> Result<Vec<u32>, String> {
    let text = std::fs::read_to_string(arg).unwrap_or_else(|_| arg.to_string());
    text.split(|c: char| c == ',' || c.is_whitespace())
        .filter(|s| !s.is_empty())
        .map(|s| s.parse().map_err(|e| format!("bad token id {s:?}: {e}")))
        .collect()
}

fn run() -> Result<(), String> {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let cpu = args.iter().any(|a| a == "--cpu");
    args.retain(|a| a != "--cpu");
    let [dir, ids, rest @ ..] = args.as_slice() else {
        return Err("usage: generate <model-dir> <ids> [max-tokens] [--cpu]".into());
    };
    let prompt = parse_ids(ids)?;
    let max_tokens: u32 = match rest.first() {
        Some(n) => n
            .parse()
            .map_err(|e| format!("bad max-tokens {n:?}: {e}"))?,
        None => 64,
    };
    let (body, last) = match prompt.split_last() {
        Some((last, body)) => (body, *last),
        None => return Err("the prompt is empty".into()),
    };

    let loaded = Instant::now();
    let mut engine = NuvecEngine::load(
        &PathBuf::from(dir),
        LoadOptions {
            identity: "generate".into(),
            cpu,
            ..LoadOptions::default()
        },
    )
    .map_err(|e| e.to_string())?;
    let budget = engine.budget();
    let gib = |b: u64| b as f64 / (1u64 << 30) as f64;
    eprintln!(
        "loaded {} in {:.1} s: budget {:.1} GiB = weights {:.1} + scratch {:.1} + state {:.1} \
         ({} tokens)",
        engine.info().family,
        loaded.elapsed().as_secs_f64(),
        gib(budget.budget),
        gib(budget.weights),
        gib(budget.scratch),
        gib(budget.state_capacity_bytes),
        budget.kv_capacity_tokens
    );
    let error = |e: mrun_engine::EngineError| e.to_string();
    engine
        .set_limit(prompt.len() as u32 + max_tokens)
        .map_err(error)?;

    let start = Instant::now();
    let cancel = CancelToken::new();
    for chunk in body.chunks(PREFILL_STEP) {
        engine.prefill_chunk(chunk, &cancel).map_err(error)?;
    }
    let step = engine.forward(last).map_err(error)?;
    let mut token = engine
        .sample(step, &Sampling::GREEDY, None)
        .map_err(error)?;
    let prompt_time = start.elapsed().as_secs_f64();

    let eos = engine.info().eos_tokens.clone();
    let mut out = vec![token];
    let decode = Instant::now();
    while (out.len() as u32) < max_tokens && !eos.contains(&token) {
        let step = engine.forward(token).map_err(error)?;
        token = engine
            .sample(step, &Sampling::GREEDY, None)
            .map_err(error)?;
        out.push(token);
    }
    let decode_time = decode.elapsed().as_secs_f64();

    let ids: Vec<String> = out.iter().map(u32::to_string).collect();
    println!("{}", ids.join(","));
    eprintln!(
        "prompt: {} tokens, {:.1} tokens-per-sec",
        prompt.len(),
        prompt.len() as f64 / prompt_time
    );
    eprintln!(
        "generation: {} tokens, {:.1} tokens-per-sec",
        out.len(),
        (out.len() - 1) as f64 / decode_time
    );
    eprintln!(
        "peak memory: {:.2} GB",
        mrun_engine::tensor::memory::peak() as f64 / 1e9
    );
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("error: {message}");
            ExitCode::FAILURE
        }
    }
}
