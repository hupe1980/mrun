//! Synthetic prefill and decode cells (005 FR-001, FR-004, FR-007, FR-014): prompts of 1K, 8K
//! and 32K tokens and the common maximum context, each starting with a per-repetition nonce so
//! no cache can serve it, one warm-up then the repetitions; temperature 0 and a fixed output
//! length with ignore-EOS where supported. One request per repetition gives both the prefill
//! rate (processed prompt tokens / TTFT) and the decode rate.
//!
//! Prompts are common English words, one token each in the usual BPE vocabularies; a one-word
//! calibration request measures the system's template overhead so the expected prompt count is
//! exact enough for the truncation check (FR-015).

use crate::api::model::ApiKind;
use crate::bench::driver::{self, Outcome};
use crate::bench::machine;
use crate::bench::result::{Cell, CellStatus, Failure};
use crate::bench::suites::{Ctx, Rng, not_run};
use crate::bench::systems::System;
use serde_json::{Value, json};

/// Fixed output length of every repetition.
pub const DECODE_TOKENS: u32 = 128;
const NONCE_WORDS: u32 = 8;
const WORDS: [&str; 32] = [
    "the", "of", "and", "to", "in", "is", "that", "for", "it", "as", "was", "with", "be", "by",
    "on", "not", "he", "this", "are", "or", "his", "from", "at", "which", "but", "have", "an",
    "had", "they", "you", "were", "their",
];

/// Cell sizes: 1K, 8K and 32K that fit, plus the common maximum (`None` when unknown).
pub fn sizes(common_context: Option<u32>) -> Vec<(String, Option<u32>)> {
    let max = common_context.map(|c| c.saturating_sub(DECODE_TOKENS + 64));
    let mut out: Vec<(String, Option<u32>)> = [(1, 1024), (8, 8192), (32, 32_768)]
        .into_iter()
        .map(|(k, n)| (format!("{k}k"), Some(n)))
        .collect();
    out.push(("max".into(), max));
    out
}

fn prompt(rng: &mut Rng, words: u32) -> String {
    let nonce = (0..NONCE_WORDS).map(|_| WORDS[rng.below(WORDS.len() as u64) as usize]);
    let filler = (0..words).map(|i| WORDS[(i as usize * 7 + 3) % WORDS.len()]);
    nonce.chain(filler).collect::<Vec<_>>().join(" ")
}

fn body(api: ApiKind, text: &str) -> Value {
    match api {
        ApiKind::Responses => json!({"input": text, "temperature": 0}),
        _ => json!({"messages": [{"role": "user", "content": text}], "temperature": 0}),
    }
}

/// Sends a fresh prompt of a nonce and `words` filler words.
async fn send(
    ctx: &mut Ctx,
    system: &System,
    api: ApiKind,
    words: u32,
    max: u32,
) -> Result<Outcome, String> {
    let text = prompt(&mut ctx.rng, words);
    let request = driver::prepare(
        api,
        &body(api, &text),
        Some(&system.model),
        max,
        system.ignore_eos(),
    );
    driver::send(
        &ctx.client,
        &system.url,
        api,
        &request,
        system.ignore_eos(),
        None,
    )
    .await
}

/// Runs every size for `system`; `context` is its context length.
pub async fn run(
    ctx: &mut Ctx,
    system: &System,
    context: Option<u32>,
    common: Option<u32>,
    reps: u32,
) -> Vec<Cell> {
    let mut cells = Vec::new();
    let Some(api) = system.preferred_api() else {
        return cells;
    };
    let name = |kind: &str, label: &str| format!("synthetic/{kind}-{label}");
    // Template overhead: a one-word prompt after the nonce.
    let overhead = match send(ctx, system, api, 1, 1).await {
        Ok(o) => o.prompt_tokens.map(|p| p.saturating_sub(NONCE_WORDS + 1)),
        Err(error) => {
            for (label, _) in sizes(common) {
                for (kind, unit) in [("prefill", "tok/s"), ("decode", "tok/s")] {
                    let mut cell = not_run(
                        &system.name,
                        &name(kind, &label),
                        api.as_str(),
                        unit,
                        &error,
                    );
                    cell.status = CellStatus::Failed;
                    cells.push(cell);
                }
            }
            return cells;
        }
    };
    for (label, size) in sizes(common) {
        let (prefill_name, decode_name) = (name("prefill", &label), name("decode", &label));
        let fits = size.filter(|&n| context.is_none_or(|c| n + DECODE_TOKENS <= c));
        let Some(size) = fits else {
            let reason = if size.is_none() {
                "the common context length is unknown"
            } else {
                "exceeds the context length"
            };
            cells.push(not_run(
                &system.name,
                &prefill_name,
                api.as_str(),
                "tok/s",
                reason,
            ));
            cells.push(not_run(
                &system.name,
                &decode_name,
                api.as_str(),
                "tok/s",
                reason,
            ));
            continue;
        };
        let words = size.saturating_sub(overhead.unwrap_or(0) + NONCE_WORDS);
        let expected = if overhead.is_some() { size } else { 0 };
        let template = Cell {
            system: system.name.clone(),
            api: api.as_str().into(),
            unit: "tok/s".into(),
            thermal_start: machine::thermal(),
            ..Cell::default()
        };
        let mut prefill = Cell {
            cell: prefill_name,
            ..template.clone()
        };
        let mut decode = Cell {
            cell: decode_name,
            ..template
        };
        // Warm-up, not recorded.
        let _ = send(ctx, system, api, words, DECODE_TOKENS).await;
        for rep in 1..=reps {
            ctx.reset(system, rep).await;
            match send(ctx, system, api, words, DECODE_TOKENS).await {
                Ok(out) => {
                    if driver::truncated(out.prompt_tokens, expected) {
                        prefill.failures.push(Failure {
                            rep,
                            error: "truncated input".into(),
                        });
                    }
                    if let (Some(ttft), Some(p)) = (out.ttft_ms, out.prompt_tokens) {
                        let processed = p - out.cached_tokens.unwrap_or(0).min(p);
                        prefill
                            .samples
                            .push(f64::from(processed) * 1000.0 / ttft.max(0.001));
                    }
                    if let Some(rate) = out.decode_tok_s() {
                        decode.samples.push(rate);
                    }
                    if out.early_eos(DECODE_TOKENS) {
                        decode.early_eos += 1;
                    }
                }
                Err(error) => {
                    prefill.failures.push(Failure {
                        rep,
                        error: error.clone(),
                    });
                    decode.failures.push(Failure { rep, error });
                }
            }
        }
        for mut cell in [prefill, decode] {
            cell.thermal_end = machine::thermal();
            if !cell.failures.is_empty() || cell.samples.is_empty() {
                cell.status = CellStatus::Failed;
            }
            cell.summarize();
            cells.push(cell);
        }
    }
    cells
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sizes_and_prompts() {
        let s = sizes(Some(16_384));
        assert_eq!(s.len(), 4);
        assert_eq!(s[3], ("max".into(), Some(16_384 - DECODE_TOKENS - 64)));
        assert_eq!(sizes(None)[3].1, None);
        let mut rng = Rng::new(7);
        let a = prompt(&mut rng, 100);
        let b = prompt(&mut rng, 100);
        assert_eq!(a.split(' ').count(), 108);
        assert_ne!(a, b, "a fresh nonce per repetition");
        assert_eq!(
            a.split(' ').skip(8).collect::<Vec<_>>(),
            b.split(' ').skip(8).collect::<Vec<_>>()
        );
    }
}
