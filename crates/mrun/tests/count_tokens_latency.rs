//! SC-009 (002 T074): counting a 60K-token conversation whose prefix was rendered before takes
//! ≤ 50 ms, because the segment memo re-encodes only the new turns (FR-046).
//!
//! The portable case runs everywhere on the BPE fixture with a generous bound; the precise one
//! needs the real Qwen3.5 tokenizer and template in the Hugging Face cache and a release build:
//! `cargo test --release -p mrun --test count_tokens_latency -- --ignored`.

use mrun::api::model::{ApiKind, AssistantPart, ChatRequest, Message};
use mrun::chat::render::{RenderContext, render};
use mrun::chat::template::ChatTemplate;
use mrun::chat::tokenizer::Tokenizer;
use mrun::chat::tokenizer::test_support::bpe_tokenizer;
use mrun_engine::SyntaxId;
use std::time::{Duration, Instant};

/// An agent session of `turns` user/assistant pairs, each with a few KB of code-like text.
fn session(turns: usize) -> ChatRequest {
    let mut request = ChatRequest::new(ApiKind::Messages);
    request.system = vec!["You are a coding agent.".into()];
    for i in 0..turns {
        let code: String = (0..40)
            .map(|line| {
                format!("    let value_{i}_{line} = compute({line}, \"a.rs\"); // step {line}\n")
            })
            .collect();
        request.messages.push(Message::user_text(format!(
            "Read file {i}.rs and explain:\n{code}"
        )));
        request
            .messages
            .push(Message::Assistant(vec![AssistantPart::Text(format!(
                "File {i}.rs computes values in order. Both files are fine. Now b.rs.\n{code}"
            ))]));
    }
    request.messages.push(Message::user_text("thanks"));
    request
}

/// Renders `request` once (warming the memo), appends a turn and returns the second render's
/// duration and token count.
fn follow_up(
    tokenizer: &Tokenizer,
    template: &ChatTemplate,
    syntax: SyntaxId,
    turns: usize,
) -> (Duration, u32) {
    let ctx = RenderContext {
        template,
        tokenizer,
        syntax,
        supports_thinking: true,
    };
    let mut request = session(turns);
    render(&request, &ctx).unwrap();
    request
        .messages
        .push(Message::Assistant(vec![AssistantPart::Text(
            "You're welcome.".into(),
        )]));
    request
        .messages
        .push(Message::user_text("one more question"));
    let started = Instant::now();
    let rendered = render(&request, &ctx).unwrap();
    (started.elapsed(), rendered.prompt.len())
}

#[test]
fn count_tokens_shared_prefix_reuses_the_memo() {
    let tokenizer = bpe_tokenizer();
    let template =
        ChatTemplate::new(include_str!("../src/chat/testdata/qwen3_5.jinja").into()).unwrap();
    let hits = tokenizer.memo_hits();
    let (elapsed, tokens) = follow_up(&tokenizer, &template, SyntaxId::QwenXml, 60);
    assert!(tokens > 20_000, "{tokens}");
    assert!(
        tokenizer.memo_hits() > hits + 100,
        "the shared turns came from the memo"
    );
    // Generous: debug builds on shared CI machines; the precise bound is the ignored case.
    assert!(
        elapsed < Duration::from_secs(5),
        "{elapsed:?} for {tokens} tokens"
    );
}

/// The cached `mlx-community/Qwen3.5-9B-4bit` snapshot, if present.
fn qwen3_5_dir() -> Option<std::path::PathBuf> {
    let home = std::env::var("HOME").ok()?;
    let snapshots = std::path::Path::new(&home)
        .join(".cache/huggingface/hub/models--mlx-community--Qwen3.5-9B-4bit/snapshots");
    let dir = std::fs::read_dir(snapshots).ok()?.next()?.ok()?.path();
    dir.join("tokenizer.json").exists().then_some(dir)
}

#[test]
#[ignore = "needs the cached Qwen3.5 tokenizer and a release build"]
fn xapi_count_tokens_60k_shared_prefix_50ms() {
    let dir = qwen3_5_dir().expect("mlx-community/Qwen3.5-9B-4bit in the Hugging Face cache");
    let tokenizer = Tokenizer::from_file(&dir.join("tokenizer.json")).unwrap();
    let template = ChatTemplate::from_model_dir(&dir).unwrap();
    // Grow the session until it holds 60K tokens.
    let mut turns = 40;
    loop {
        let (elapsed, tokens) = follow_up(&tokenizer, &template, SyntaxId::QwenXml, turns);
        if tokens >= 60_000 {
            eprintln!("count_tokens follow-up: {tokens} tokens in {elapsed:?}");
            assert!(
                elapsed <= Duration::from_millis(50),
                "{elapsed:?} for {tokens} tokens"
            );
            return;
        }
        turns = turns * 60_000 / tokens as usize + 1;
    }
}
