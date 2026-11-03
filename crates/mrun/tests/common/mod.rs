//! Shared helpers: run requests on a `ScriptedEngine` and check the state invariants.

#![allow(dead_code)]

use mrun::cache::{CacheMode, PrefixStore};
use mrun::chat::prompt::RenderedPrompt;
use mrun::chat::tokenizer::Tokenizer;
use mrun::chat::tokenizer::test_support::byte_tokenizer;
use mrun::runtime::generate::{Event, EventSink, Finished, GenerateRequest, generate};
use mrun::runtime::pressure::PressureGauge;
use mrun::runtime::scripted::ScriptedEngine;
use mrun_engine::Engine;
use std::sync::LazyLock;

pub static TOKENIZER: LazyLock<Tokenizer> = LazyLock::new(byte_tokenizer);

/// Byte-level token IDs of `text`.
pub fn ids(text: &str) -> Vec<u32> {
    TOKENIZER.encode(text).unwrap()
}

pub fn request(prompt: Vec<u32>, max_tokens: u32) -> GenerateRequest {
    GenerateRequest::new(RenderedPrompt::from_tokens(prompt), max_tokens)
}

/// The prefix store `mrun run` and `mrun serve` use.
pub fn session() -> PrefixStore {
    PrefixStore::new(CacheMode::Cached)
}

/// Runs one request and asserts that the store mirrors the engine state afterwards.
pub fn run(
    engine: &mut ScriptedEngine,
    session: &mut PrefixStore,
    request: GenerateRequest,
    sink: &mut dyn EventSink,
) -> Finished {
    let pressure = PressureGauge::default();
    let finished = generate(
        engine,
        session,
        &mut TOKENIZER.stream(),
        request,
        sink,
        &pressure,
    );
    assert_eq!(
        engine.tokens(),
        session.live_tokens(),
        "the store mirrors the state"
    );
    assert!(!engine.has_pending(), "no pending step after a finish");
    assert!(engine.allocated() <= engine.budget().state_capacity_bytes);
    finished
}

/// Runs one request with a collecting sink.
pub fn run_collect(
    engine: &mut ScriptedEngine,
    session: &mut PrefixStore,
    request: GenerateRequest,
) -> (Vec<Event>, Finished) {
    let mut events = Vec::new();
    let finished = run(engine, session, request, &mut events);
    (events, finished)
}

pub fn text(events: &[Event]) -> String {
    events
        .iter()
        .filter_map(|event| match event {
            Event::Token { text, .. } | Event::Flush { text } => Some(text.as_str()),
            _ => None,
        })
        .collect()
}

/// `(prompt, cached, max_tokens)` of a `Started` event.
pub fn started(event: &Event) -> (u32, u32, u32) {
    match event {
        Event::Started {
            prompt,
            cached,
            max_tokens,
            ..
        } => (*prompt, *cached, *max_tokens),
        other => panic!("not a Started event: {other:?}"),
    }
}
