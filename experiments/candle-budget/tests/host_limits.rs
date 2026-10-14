//! T118 host-library limit probes at the research C-05 pins. Targets (not established
//! capabilities): render/tokenize 64 MiB per active request, grammar/parser 16 MiB.
//! Run with `--nocapture` to see the measured figures recorded in `results.md`.

mod support;

use llguidance::api::{ParserLimits, TopLevelGrammar};
use llguidance::{Matcher, ParserFactory};
use std::cell::Cell;
use std::fmt;
use toktrie::ApproximateTokEnv;

#[global_allocator]
static ALLOCATOR: support::Counting = support::Counting;

const MIB: u64 = 1024 * 1024;
const RENDER_TARGET: u64 = 64 * MIB;
const GRAMMAR_TARGET: u64 = 16 * MIB;

const CHATML: &str = "{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}<|im_start|>assistant\n";

/// Output sink that fails once `limit` bytes would be exceeded.
struct Capped {
    out: String,
    limit: usize,
}

impl fmt::Write for Capped {
    fn write_str(&mut self, text: &str) -> fmt::Result {
        if self.out.len() + text.len() > self.limit {
            return Err(fmt::Error);
        }
        self.out.push_str(text);
        Ok(())
    }
}

#[derive(Debug, PartialEq)]
enum RenderError {
    ContextTooLarge,
    Engine,
}

/// mrun-side bound: cap serialized context bytes before rendering, then render with fuel
/// into a capped sink. Only the context cap acts before any engine allocation.
fn render_bounded(
    source: &str,
    context: &serde_json::Value,
    context_cap: usize,
    output_cap: usize,
    fuel: u64,
    entered: &Cell<bool>,
) -> Result<String, RenderError> {
    let context_bytes = serde_json::to_vec(context)
        .map_err(|_| RenderError::Engine)?
        .len();
    if context_bytes > context_cap {
        return Err(RenderError::ContextTooLarge);
    }
    entered.set(true);
    let mut environment = minijinja::Environment::new();
    environment.set_fuel(Some(fuel));
    let template = environment
        .template_from_str(source)
        .map_err(|_| RenderError::Engine)?;
    let mut sink = Capped {
        out: String::new(),
        limit: output_cap,
    };
    let rendered = template.render(context).map_err(|_| RenderError::Engine)?;
    fmt::Write::write_str(&mut sink, &rendered).map_err(|_| RenderError::Engine)?;
    Ok(sink.out)
}

fn messages(count: usize, bytes: usize) -> serde_json::Value {
    let content = "a".repeat(bytes);
    serde_json::json!({
        "messages": (0..count)
            .map(|i| serde_json::json!({"role": if i % 2 == 0 { "user" } else { "assistant" }, "content": content}))
            .collect::<Vec<_>>()
    })
}

#[test]
fn template_context_cap_rejects_before_engine_entry() {
    let _serial = support::serial();
    let entered = Cell::new(false);
    let context = messages(4, 1024);
    let result = render_bounded(CHATML, &context, 1024, 1 << 20, 100_000, &entered);
    assert_eq!(result, Err(RenderError::ContextTooLarge));
    assert!(!entered.get());
}

#[test]
fn lawful_chatml_render_stays_within_render_target() {
    let _serial = support::serial();
    let entered = Cell::new(false);
    let context = messages(512, 2048);
    let (result, observed) = support::measure(|| {
        render_bounded(CHATML, &context, 2 << 20, 4 << 20, 10_000_000, &entered)
    });
    let rendered = result.expect("lawful render");
    println!(
        "render lawful: context≈1 MiB output={} B peak={} B ({:.1}x output)",
        rendered.len(),
        observed.peak,
        observed.peak as f64 / rendered.len() as f64
    );
    assert!(observed.peak < RENDER_TARGET);
}

/// Decisive counterexample: neither fuel nor a capped sink bounds an intermediate value the
/// engine materializes in one instruction, so arbitrary templates cannot meet the 64 MiB target.
#[test]
fn template_repeat_allocates_past_target_despite_fuel_and_capped_sink() {
    let _serial = support::serial();
    let entered = Cell::new(false);
    let (result, observed) = support::measure(|| {
        render_bounded(
            "{{ 'x' * 100000000 }}",
            &serde_json::json!({}),
            1024,
            64 * 1024,
            100,
            &entered,
        )
    });
    assert!(result.is_err(), "output cap still rejects the result");
    println!("render repeat counterexample: peak={} B", observed.peak);
    assert!(observed.peak > RENDER_TARGET);
}

fn byte_level_tokenizer() -> tokenizers::Tokenizer {
    use tokenizers::models::bpe::BPE;
    use tokenizers::pre_tokenizers::byte_level::ByteLevel;
    let vocab: tokenizers::models::bpe::Vocab = ByteLevel::alphabet()
        .into_iter()
        .enumerate()
        .map(|(id, ch)| (ch.to_string(), id as u32))
        .collect();
    let model = BPE::builder()
        .vocab_and_merges(vocab, Vec::new())
        .build()
        .unwrap();
    let mut tokenizer = tokenizers::Tokenizer::new(model);
    tokenizer.with_pre_tokenizer(Some(ByteLevel::new(false, true, true)));
    tokenizer
}

/// Measures worst-case byte-level amplification (one token per input byte, no merges) and
/// checks that an input cap derived from it keeps encoding under the 64 MiB target.
#[test]
fn tokenizer_amplification_bounds_the_input_cap() {
    let _serial = support::serial();
    let tokenizer = byte_level_tokenizer();
    let probe = "a b ".repeat(16 * 1024); // 64 KiB, two-byte pre-tokens: adversarial for offsets
    let (encoding, observed) =
        support::measure(|| tokenizer.encode(probe.as_str(), false).unwrap());
    let ratio = observed.peak.div_ceil(probe.len() as u64);
    let input_cap = RENDER_TARGET / ratio;
    println!(
        "tokenize: input={} B tokens={} peak={} B amplification={}x derived input cap={} B",
        probe.len(),
        encoding.len(),
        observed.peak,
        ratio,
        input_cap
    );
    drop(encoding);
    let capped = "a b ".repeat((input_cap / 4) as usize);
    let (encoding, observed) =
        support::measure(|| tokenizer.encode(capped.as_str(), false).unwrap());
    println!(
        "tokenize at cap: input={} B peak={} B",
        capped.len(),
        observed.peak
    );
    assert!(
        observed.peak <= RENDER_TARGET,
        "amplification is not linear at the cap"
    );
    drop(encoding);
}

fn factory(limits: ParserLimits) -> ParserFactory {
    let mut factory = ParserFactory::new_simple(&ApproximateTokEnv::single_byte_env()).unwrap();
    factory.set_stderr_log_level(0);
    *factory.limits_mut() = limits;
    factory
}

fn tool_schema(properties: usize) -> serde_json::Value {
    let properties: serde_json::Map<_, _> = (0..properties)
        .map(|i| {
            (
                format!("p{i}"),
                serde_json::json!({"type": "string", "maxLength": 64}),
            )
        })
        .collect();
    serde_json::json!({"type": "object", "properties": properties, "additionalProperties": false})
}

#[test]
fn lawful_tool_grammar_constructs_within_grammar_target() {
    let _serial = support::serial();
    let factory = factory(ParserLimits::default());
    let (parser, observed) = support::measure(|| {
        factory.create_parser(TopLevelGrammar::from_json_schema(tool_schema(16)))
    });
    parser.expect("lawful schema");
    println!("grammar lawful (16 properties): peak={} B", observed.peak);
    assert!(observed.peak < GRAMMAR_TARGET);
}

/// Decisive counterexample: construction limits count symbols and fuel, not bytes. The
/// oversized schema is rejected, but only after allocating past the 16 MiB target.
#[test]
fn oversized_tool_grammar_is_rejected_only_after_exceeding_target() {
    let _serial = support::serial();
    let limits = ParserLimits {
        max_grammar_size: 50_000,
        ..ParserLimits::default()
    };
    let factory = factory(limits);
    let schema = tool_schema(20_000);
    let (parser, observed) = support::measure(|| {
        factory.create_parser(TopLevelGrammar::from_json_schema(schema.clone()))
    });
    println!(
        "grammar oversized (20,000 properties): rejected={} peak={} B",
        parser.is_err(),
        observed.peak
    );
    assert!(
        parser.is_err(),
        "symbol limit must reject the oversized schema"
    );
    assert!(observed.peak > GRAMMAR_TARGET);
}

/// Matcher state grows with consumed lexemes. Measure growth per generated byte inside one
/// string value and project it to the 256 KiB generated-payload ceiling.
#[test]
fn matcher_growth_per_generated_byte_is_measured() {
    let _serial = support::serial();
    let factory = factory(ParserLimits::default());
    let grammar = TopLevelGrammar::from_json_schema(serde_json::json!({"type": "string"}));
    let mut matcher = Matcher::new(factory.create_parser(grammar));
    let quote = u32::from(b'"');
    let letter = u32::from(b'a');
    matcher.consume_token(quote).unwrap();
    let generated: u64 = 16 * 1024;
    let ((), observed) = support::measure(|| {
        for _ in 0..generated {
            matcher.consume_token(letter).unwrap();
        }
    });
    let per_byte = observed.peak.div_ceil(generated);
    let projected = per_byte * 256 * 1024;
    println!(
        "matcher growth: {generated} B peak={} B ≈{per_byte} B/byte, projected at 256 KiB payload={projected} B",
        observed.peak
    );
    assert!(!matcher.is_error());
}
