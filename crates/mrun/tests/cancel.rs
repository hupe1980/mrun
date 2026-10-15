//! Cancellation in every phase of a request on `ScriptedEngine`: the request finishes once with
//! `Cancelled` and the state holds exactly the committed tokens.

mod common;

use common::{TOKENIZER, ids, request, run, run_collect, text};
use mrun::runtime::generate::{Event, EventSink, FinishReason};
use mrun::runtime::handle::EngineHandle;
use mrun::runtime::scripted::ScriptedEngine;
use mrun::runtime::session::Session;
use mrun_engine::CancelToken;
use std::sync::Arc;
use std::time::Duration;

const RB: u64 = 1_000;

/// Cancels once `after` events have arrived.
struct CancelAfter {
    events: Vec<Event>,
    cancel: CancelToken,
    after: usize,
}

impl EventSink for CancelAfter {
    fn send(&mut self, event: Event) -> bool {
        self.events.push(event);
        if self.events.len() == self.after {
            self.cancel.cancel();
        }
        true
    }
}

#[test]
fn cancel_before_start_processes_nothing_new() {
    let mut engine = ScriptedEngine::new(ids("abc"), 10_000);
    let mut session = Session::default();
    run_collect(&mut engine, &mut session, request(ids("hello"), 2));
    let cached = session.tokens().to_vec();
    engine.processed.clear();

    let mut prompt = cached.clone();
    prompt.extend(ids(" more words"));
    let req = request(prompt, 10);
    req.cancel.cancel();
    let (events, finished) = run_collect(&mut engine, &mut session, req);
    assert_eq!(finished.reason, FinishReason::Cancelled);
    assert_eq!(finished.prefilled, cached.len() as u32);
    assert!(engine.processed.is_empty());
    assert!(text(&events).is_empty());
    assert_eq!(session.tokens(), cached);
}

#[test]
fn cancel_mid_chunk_discards_the_partial_chunk_and_keeps_whole_ones() {
    let prompt = ids(&"p".repeat(5_000));
    for hybrid in [false, true] {
        let mut engine = if hybrid {
            ScriptedEngine::new_hybrid(ids("never"), 10_000, RB)
        } else {
            ScriptedEngine::new(ids("never"), 10_000)
        };
        // Four layers per chunk: layer 7 lies inside the second 2,048-token chunk.
        engine.cancel_after_layers = Some(7);
        let mut session = Session::default();
        let (events, finished) = run_collect(&mut engine, &mut session, request(prompt.clone(), 5));
        assert_eq!(finished.reason, FinishReason::Cancelled, "hybrid {hybrid}");
        assert_eq!(finished.prefilled, 2_048);
        assert_eq!(engine.chunks, [(0, 2_048)]);
        assert_eq!(session.tokens(), &prompt[..2_048]);
        assert!(text(&events).is_empty());
    }
}

#[test]
fn cancel_on_a_chunk_boundary_keeps_that_chunk() {
    let prompt = ids(&"p".repeat(5_000));
    let mut engine = ScriptedEngine::new(ids("never"), 10_000);
    // Layer 8 is the last layer of the second chunk: it completes.
    engine.cancel_after_layers = Some(8);
    let mut session = Session::default();
    let (_, finished) = run_collect(&mut engine, &mut session, request(prompt.clone(), 5));
    assert_eq!(finished.reason, FinishReason::Cancelled);
    assert_eq!(finished.prefilled, 4_096);
    assert_eq!(session.tokens(), &prompt[..4_096]);
}

#[test]
fn cancel_after_prefill_before_the_first_token() {
    let mut engine = ScriptedEngine::new(ids("abc"), 10_000);
    let mut session = Session::default();
    let req = request(ids("question"), 10);
    // Events: Started, Progress (the whole prompt but its last token).
    let mut sink = CancelAfter {
        events: Vec::new(),
        cancel: req.cancel.clone(),
        after: 2,
    };
    let finished = run(&mut engine, &mut session, req, &mut sink);
    assert_eq!(finished.reason, FinishReason::Cancelled);
    assert_eq!(finished.prefilled, 7);
    assert_eq!(finished.stats.generated_tokens, 0);
    assert_eq!(session.tokens(), &ids("question")[..7]);
}

#[test]
fn cancel_during_decode_trims_the_pending_step() {
    for hybrid in [false, true] {
        let script = ids("abcdefgh");
        let mut engine = if hybrid {
            ScriptedEngine::new_hybrid(script, 1_000, RB)
        } else {
            ScriptedEngine::new(script, 1_000)
        };
        let mut session = Session::default();
        let mut req = request(ids("q?"), 50);
        req.generation_start = Some(1);
        // Events: Started, Progress, then tokens; cancel at the third token.
        let mut sink = CancelAfter {
            events: Vec::new(),
            cancel: req.cancel.clone(),
            after: 5,
        };
        let finished = run(&mut engine, &mut session, req, &mut sink);
        assert_eq!(finished.reason, FinishReason::Cancelled, "hybrid {hybrid}");
        assert_eq!(text(&sink.events), "abc");
        assert_eq!(session.committed(), ids("q?abc"));
        // "c" was fed only as a pending pipelined step and is rolled back, recurrent state too.
        assert_eq!(session.tokens(), ids("q?ab"), "hybrid {hybrid}");
        assert_eq!(engine.processed.last(), Some(&ids("c")[0]));
    }
}

#[test]
fn cancel_racing_the_last_token_still_finishes_once() {
    let mut engine = ScriptedEngine::new(ids("abcdef"), 1_000);
    let mut session = Session::default();
    let req = request(ids("q"), 3);
    // Cancel together with the third (last allowed) token.
    let mut sink = CancelAfter {
        events: Vec::new(),
        cancel: req.cancel.clone(),
        after: 4,
    };
    let finished = run(&mut engine, &mut session, req, &mut sink);
    assert_eq!(finished.reason, FinishReason::Length);
    assert_eq!(text(&sink.events), "abc");
    assert_eq!(session.committed(), ids("qabc"));

    // A cancel that arrives after the finish does not affect the next request.
    let mut next = session.committed().to_vec();
    next.extend(ids("!"));
    let (_, finished) = run_collect(&mut engine, &mut session, request(next, 1));
    assert_eq!(finished.reason, FinishReason::Length);
}

#[test]
fn cancel_from_another_thread_through_the_handle() {
    let script = ids(&"z".repeat(5_000));
    let handle = EngineHandle::spawn(
        move || Ok(ScriptedEngine::new(script, 100_000)),
        Arc::new(TOKENIZER.clone()),
        Duration::from_secs(10),
    )
    .unwrap();
    let req = request(ids("go"), 5_000);
    let cancel = req.cancel.clone();
    let generation = handle.generate(req).unwrap();
    let mut tokens = 0;
    for event in generation.events.iter() {
        if matches!(event, Event::Token { .. }) {
            tokens += 1;
            if tokens == 10 {
                cancel.cancel();
            }
        }
    }
    let finished = generation.finished.recv().unwrap();
    assert_eq!(finished.reason, FinishReason::Cancelled);
    assert!(finished.stats.generated_tokens < 5_000);
    assert!(
        generation.finished.try_recv().is_err(),
        "exactly one Finished"
    );

    let (_, next) = handle
        .generate(request(ids("again"), 2))
        .unwrap()
        .wait()
        .unwrap();
    assert_eq!(next.reason, FinishReason::Length);
}
