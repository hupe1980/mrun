//! Shared helpers: run requests on a `ScriptedEngine` and check the state invariants.

#![allow(dead_code)]

use mrun::chat::prompt::RenderedPrompt;
use mrun::chat::tokenizer::Tokenizer;
use mrun::chat::tokenizer::test_support::byte_tokenizer;
use mrun::runtime::generate::{
    Event, EventSink, Finished, GenerateRequest, PressureGauge, generate,
};
use mrun::runtime::scripted::ScriptedEngine;
use mrun::runtime::session::Session;
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

/// Runs one request and asserts that the session mirrors the engine state afterwards.
pub fn run(
    engine: &mut ScriptedEngine,
    session: &mut Session,
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
        session.tokens(),
        "session mirrors the state"
    );
    assert!(!engine.has_pending(), "no pending step after a finish");
    assert!(session.committed().starts_with(session.tokens()));
    assert!(engine.allocated() <= engine.budget().state_capacity_bytes);
    finished
}

/// Runs one request with a collecting sink.
pub fn run_collect(
    engine: &mut ScriptedEngine,
    session: &mut Session,
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
