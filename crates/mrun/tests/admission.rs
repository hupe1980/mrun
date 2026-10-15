//! Admission at the capacity boundary on `ScriptedEngine`: a prompt is admitted only with room
//! for one output token, the output limit is clamped to what remains, and cached state counts as
//! available.

mod common;

use common::{ids, request, run_collect};
use mrun::runtime::generate::{Event, FinishReason};
use mrun::runtime::scripted::{KV_BYTES, ScriptedEngine};
use mrun::runtime::session::Session;
use mrun_engine::Engine;

const CAPACITY: u32 = 600;
const RB: u64 = 1_000;

fn prompt(len: u32) -> Vec<u32> {
    ids(&"t".repeat(len as usize))
}

#[test]
fn capacity_minus_one_prompt_tokens_completes_with_one_token() {
    let mut engine = ScriptedEngine::new(ids("abc"), CAPACITY);
    let (events, finished) = run_collect(
        &mut engine,
        &mut Session::default(),
        request(prompt(CAPACITY - 1), 100),
    );
    assert_eq!(finished.reason, FinishReason::Length);
    assert_eq!(finished.stats.generated_tokens, 1);
    assert_eq!(finished.stats.max_tokens, 1);
    assert_eq!(
        events[0],
        Event::Started {
            prompt: CAPACITY - 1,
            cached: 0,
            max_tokens: 1,
            miss: None
        }
    );
}

#[test]
fn capacity_prompt_tokens_are_rejected_with_both_numbers() {
    let mut engine = ScriptedEngine::new(ids("abc"), CAPACITY);
    let (events, finished) = run_collect(
        &mut engine,
        &mut Session::default(),
        request(prompt(CAPACITY), 1),
    );
    match &finished.reason {
        FinishReason::Rejected(message) => {
            assert!(message.contains(&(CAPACITY + 1).to_string()), "{message}");
            assert!(message.contains(&CAPACITY.to_string()), "{message}");
        }
        other => panic!("{other:?}"),
    }
    assert!(events.is_empty());
    assert!(engine.processed.is_empty(), "rejected before processing");
    assert_eq!(engine.allocated(), 0);
}

#[test]
fn oversized_output_limit_is_clamped_and_reported() {
    let mut engine = ScriptedEngine::new(ids(&"o".repeat(1_000)), CAPACITY);
    let (events, finished) = run_collect(
        &mut engine,
        &mut Session::default(),
        request(prompt(500), 10_000),
    );
    assert_eq!(finished.stats.max_tokens, CAPACITY - 500);
    assert_eq!(finished.stats.generated_tokens, CAPACITY - 500);
    assert_eq!(finished.reason, FinishReason::Length);
    assert!(matches!(
        events[0],
        Event::Started {
            max_tokens: 100,
            ..
        }
    ));
}

#[test]
fn cached_state_counts_as_available() {
    // Room for the live state and two recurrent copies (the live one and one checkpoint).
    let mut engine = ScriptedEngine::new_hybrid(ids(&"r".repeat(1_000)), CAPACITY, RB)
        .with_state_capacity_bytes(u64::from(CAPACITY) * KV_BYTES + 2 * RB);
    let mut session = Session::default();
    let first = prompt(100);
    let mut req = request(first.clone(), 2);
    req.prompt.generation_start = 90;
    let (_, finished) = run_collect(&mut engine, &mut session, req);
    assert_eq!(finished.reason, FinishReason::Length);
    assert_eq!(session.checkpoints().len(), 1);

    // The next turn fills the capacity; it needs the bytes the old checkpoint holds.
    let mut next = session.committed().to_vec();
    next.extend(prompt(CAPACITY - 1 - next.len() as u32));
    let mut req = request(next.clone(), 50);
    req.prompt.generation_start = CAPACITY - 10;
    let (events, finished) = run_collect(&mut engine, &mut session, req);
    assert_eq!(finished.reason, FinishReason::Length);
    assert_eq!(finished.stats.generated_tokens, 1);
    assert_eq!(
        events[0],
        Event::Started {
            prompt: CAPACITY - 1,
            cached: 101,
            max_tokens: 1,
            miss: None,
        }
    );
    // The old checkpoint was evicted (a session otherwise keeps two).
    let positions: Vec<_> = session.checkpoints().iter().map(|c| c.position).collect();
    assert_eq!(positions, [CAPACITY - 10]);
}
