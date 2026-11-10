//! Speculative decoding is exact on real models (006 SC-001): a copy-heavy request decodes the
//! same tokens with speculation off and on, greedy and seeded, through the runtime's loop on
//! the GPU, and speculation drafts and keeps tokens on the way.
//!
//! Ignored by default (a few minutes on the GPU; the models must be pulled):
//! `cargo test -p mrun --test speculation_exact -- --ignored` (`MRUN_SPEC_EXACT_MODELS`,
//! comma-separated, default `qwen3.5,qwen3.6`).

#![cfg(feature = "engine")]

use mrun::cache::{CacheMode, PrefixStore};
use mrun::chat::prompt::RenderedPrompt;
use mrun::runtime::generate::{Event, EventSink, GenerateRequest, Stats, generate};
use mrun::runtime::pressure::PressureGauge;
use mrun_engine::{Engine, Sampling};

/// Collects the output token ids.
struct Tokens(Vec<u32>);

impl EventSink for Tokens {
    fn send(&mut self, event: Event) -> bool {
        if let Event::Token { id, .. } = event {
            self.0.push(id);
        }
        true
    }
}

const SOURCE: &str = r#"fn parse(line: &str) -> Option<(String, u32)> {
    let (name, count) = line.split_once('=')?;
    let count = count.trim().parse().ok()?;
    Some((name.trim().to_string(), count))
}

fn total(lines: &[&str]) -> u32 {
    lines.iter().filter_map(|l| parse(l)).map(|(_, c)| c).sum()
}
"#;

/// A copy-heavy request: rewrite the code with one rename.
const COPY: &str =
    "Repeat this Rust code exactly, renaming `total` to `sum_counts`, and nothing else:";
/// A request that copies nothing.
const PROSE: &str =
    "Write a short story about a lighthouse keeper who finds a message in a bottle.";

fn prompt(model: &mrun::cli::model::OpenModel, ask: &str) -> Vec<u32> {
    let source = if ask == COPY { SOURCE } else { "" };
    let text = format!(
        "<|im_start|>user\n{ask}\n\n{source}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    );
    model.tokenizer.encode(&text).unwrap()
}

/// A model loaded once (a fresh 18 GB load per run would wait for the last one's memory).
struct Loaded {
    model: mrun::cli::model::OpenModel,
    engine: mrun_engine::NuvecEngine,
}

fn load(name: &str) -> Loaded {
    let model = mrun::cli::model::open(name).expect("pull the model first");
    let factory = mrun::cli::model::engine_factory(&model, mrun::cli::model::overrides(None));
    let engine = factory().unwrap();
    assert!(engine.max_draft() > 0, "{name}: the GPU verifies");
    Loaded { model, engine }
}

/// One request from empty state (no reuse between runs).
fn decode(
    loaded: &mut Loaded,
    ask: &str,
    sampling: Sampling,
    speculate: bool,
) -> (Vec<u32>, Stats) {
    loaded.engine.reset_all();
    let mut store = PrefixStore::new(CacheMode::Off);
    let prompt = prompt(&loaded.model, ask);
    let mut request = GenerateRequest::new(RenderedPrompt::from_tokens(prompt), 256);
    request.sampling = sampling;
    request.speculate = speculate;
    let mut tokens = Tokens(Vec::new());
    let finished = generate(
        &mut loaded.engine,
        &mut store,
        &mut loaded.model.tokenizer.stream(),
        request,
        &mut tokens,
        &PressureGauge::default(),
    );
    (tokens.0, finished.stats)
}

#[test]
#[ignore = "a few minutes on the GPU with pulled models (the Mac may be shared)"]
fn speculation_decodes_what_plain_decoding_decodes() {
    let names =
        std::env::var("MRUN_SPEC_EXACT_MODELS").unwrap_or_else(|_| "qwen3.5,qwen3.6".into());
    let seeded = Sampling {
        temperature: 0.6,
        top_p: 0.95,
        top_k: 20,
        seed: Some(11),
        ..Sampling::GREEDY
    };
    for name in names.split(',') {
        let mut loaded = load(name);
        for sampling in [Sampling::GREEDY, seeded] {
            let (plain, _) = decode(&mut loaded, COPY, sampling, false);
            let (spec, stats) = decode(&mut loaded, COPY, sampling, true);
            assert_eq!(spec, plain, "{name} ({sampling:?})");
            assert!(stats.accepted > 0, "{name}: nothing kept ({stats:?})");
            eprintln!(
                "{name} {}: {} tokens, {} of {} drafted kept in {} passes",
                if sampling.is_greedy() {
                    "greedy"
                } else {
                    "seeded"
                },
                spec.len(),
                stats.accepted,
                stats.drafted,
                stats.passes
            );
        }
    }
}

/// SC-002/SC-003: decode speed with speculation off and on, on a copy-heavy and a prose reply
/// (median of 3 runs each; the same tokens either way).
#[test]
#[ignore = "about 10 minutes on the GPU with pulled models (the Mac must be idle)"]
fn speculation_speed() {
    let names =
        std::env::var("MRUN_SPEC_EXACT_MODELS").unwrap_or_else(|_| "qwen3.5,qwen3.6".into());
    for name in names.split(',') {
        let mut loaded = load(name);
        for (label, ask) in [("copy", COPY), ("prose", PROSE)] {
            let mut median = |speculate: bool| {
                let mut runs: Vec<(f64, Vec<u32>, Stats)> = (0..3)
                    .map(|_| {
                        let (tokens, stats) = decode(&mut loaded, ask, Sampling::GREEDY, speculate);
                        (stats.decode_tps, tokens, stats)
                    })
                    .collect();
                runs.sort_by(|a, b| a.0.total_cmp(&b.0));
                runs.swap_remove(1)
            };
            let (plain, plain_tokens, _) = median(false);
            let (spec, spec_tokens, stats) = median(true);
            assert_eq!(spec_tokens, plain_tokens, "{name} {label}");
            eprintln!(
                "{name} {label}: {} tokens, plain {plain:.1} tok/s, speculative {spec:.1} tok/s \
                 ({:.2}×), {} of {} drafted kept in {} passes",
                spec_tokens.len(),
                spec / plain,
                stats.accepted,
                stats.drafted,
                stats.passes
            );
        }
    }
}
