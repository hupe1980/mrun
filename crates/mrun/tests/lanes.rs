//! Parallel lanes (011) on `ScriptedEngine` through the engine worker: a side request streams
//! while the main turn reads its prompt (US1), subagents run at once and their second fan-out
//! finds the prefix on both lanes while the main conversation stays cached (US2), reservations
//! that do not fit wait (US3), and a cancelled, failed or stalled lane ends alone (US4).

mod common;

use common::{TOKENIZER, ids};
use mrun::chat::prompt::RenderedPrompt;
use mrun::chat::tokenizer::test_support::byte_tokenizer;
use mrun::runtime::generate::{Event, FinishReason, Finished, GenerateRequest};
use mrun::runtime::handle::{EngineHandle, Generation};
use mrun::runtime::scripted::ScriptedEngine;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// What a consumer saw of one request: its events with their arrival times, and `Finished`.
struct Seen {
    events: Vec<(Instant, Event)>,
    finished: Finished,
    at: Instant,
}

impl Seen {
    fn first_token(&self) -> Option<Instant> {
        self.events
            .iter()
            .find(|(_, e)| matches!(e, Event::Token { .. }))
            .map(|(t, _)| *t)
    }

    fn started(&self) -> (u32, u32) {
        self.events
            .iter()
            .find_map(|(_, e)| match e {
                Event::Started { cached, lane, .. } => Some((*cached, *lane)),
                _ => None,
            })
            .expect("started")
    }

    fn tokens(&self) -> usize {
        self.events
            .iter()
            .filter(|(_, e)| matches!(e, Event::Token { .. }))
            .count()
    }
}

/// Reads `generation` on its own thread (after `delay`, a slow client).
fn consume(generation: Generation, delay: Duration) -> std::thread::JoinHandle<Seen> {
    std::thread::spawn(move || {
        std::thread::sleep(delay);
        let events = generation
            .events
            .iter()
            .map(|e| (Instant::now(), e))
            .collect();
        let finished = generation.finished.recv().expect("finished");
        Seen {
            events,
            finished,
            at: Instant::now(),
        }
    })
}

fn spawn(engine: ScriptedEngine, stall: Duration) -> EngineHandle {
    EngineHandle::spawn(move || Ok(engine), Arc::new(byte_tokenizer()), stall).unwrap()
}

fn request(prompt: Vec<u32>, max_tokens: u32) -> GenerateRequest {
    GenerateRequest::new(RenderedPrompt::from_tokens(prompt), max_tokens)
}

fn text(len: usize, c: char) -> Vec<u32> {
    ids(&c.to_string().repeat(len))
}

#[test]
fn a_side_request_streams_while_the_main_turn_reads_its_prompt() {
    let mut engine = ScriptedEngine::new(ids("main reply"), 100_000).with_lanes(2);
    // Each chunk layer takes a while: the main prompt (12 chunks) reads for ≈ 0.5 s.
    engine.layer_delay = Some(Duration::from_millis(10));
    let handle = spawn(engine, Duration::from_secs(10));
    assert_eq!(handle.lanes(), 2);
    let main = handle.generate(request(text(24_000, 'm'), 8)).unwrap();
    let main = consume(main, Duration::ZERO);
    std::thread::sleep(Duration::from_millis(60));
    let mut side = request(text(100, 's'), 4);
    side.auxiliary = true;
    let side = consume(handle.generate(side).unwrap(), Duration::ZERO);
    let (main, side) = (main.join().unwrap(), side.join().unwrap());
    assert_eq!(side.finished.reason, FinishReason::Length);
    assert!(
        side.at < main.first_token().expect("main decodes"),
        "the side request finished before the main turn's first token"
    );
    assert_ne!(main.started().1, side.started().1, "two lanes");
    assert_eq!(main.finished.stats.generated_tokens, 8);
}

#[test]
fn subagents_run_at_once_and_reuse_their_prefix_on_both_lanes() {
    // A reply long enough that the first subagent still decodes when the second arrives.
    let engine = ScriptedEngine::new(ids(&"done ".repeat(16)), 100_000).with_lanes(2);
    let handle = spawn(engine, Duration::from_secs(10));
    let mut main_prompt = text(2_000, 'M');
    let main = handle
        .generate(request(main_prompt.clone(), 4))
        .unwrap()
        .wait()
        .unwrap();
    assert_eq!(main.1.reason, FinishReason::Length);
    let main_reply: Vec<u32> = main
        .0
        .iter()
        .filter_map(|e| match e {
            Event::Token { id, .. } => Some(*id),
            _ => None,
        })
        .collect();

    let prefix = text(1_200, 'P');
    let fan_out = |tasks: [char; 2]| {
        let runs: Vec<_> = tasks
            .iter()
            .map(|&t| {
                let mut prompt = prefix.clone();
                prompt.extend(text(50, t));
                consume(
                    handle.generate(request(prompt, 64)).unwrap(),
                    Duration::ZERO,
                )
            })
            .collect();
        runs.into_iter()
            .map(|r| r.join().unwrap())
            .collect::<Vec<_>>()
    };
    let first = fan_out(['a', 'b']);
    let lanes: Vec<u32> = first.iter().map(|s| s.started().1).collect();
    assert_ne!(
        lanes[0], lanes[1],
        "both subagents ran at once, one per lane"
    );
    let second = fan_out(['c', 'd']);
    for s in &second {
        assert!(
            s.started().0 >= prefix.len() as u32,
            "the second fan-out reuses the subagent prefix: {:?}",
            s.started()
        );
    }
    // The main conversation continues where it was.
    main_prompt.extend(&main_reply[..main_reply.len() - 1]);
    main_prompt.extend(text(20, 'n'));
    let back = handle
        .generate(request(main_prompt, 2))
        .unwrap()
        .wait()
        .unwrap();
    let cached = back
        .0
        .iter()
        .find_map(|e| match e {
            Event::Started { cached, .. } => Some(*cached),
            _ => None,
        })
        .unwrap();
    assert!(
        cached >= 2_000,
        "the main conversation is still cached: {cached}"
    );
}

#[test]
fn a_request_whose_reservation_does_not_fit_waits_for_a_lane() {
    let mut engine = ScriptedEngine::new(ids(&"r".repeat(400)), 1_000).with_lanes(2);
    engine.step_delay = Some(Duration::from_millis(5));
    let handle = spawn(engine, Duration::from_secs(10));
    // 600 + 300 reserved; the second needs 100 + 50 more than the 1,000 tokens hold.
    let big = consume(
        handle.generate(request(text(600, 'b'), 300)).unwrap(),
        Duration::ZERO,
    );
    std::thread::sleep(Duration::from_millis(30));
    let small = consume(
        handle.generate(request(text(100, 's'), 50)).unwrap(),
        Duration::ZERO,
    );
    let (big, small) = (big.join().unwrap(), small.join().unwrap());
    let small_started = small.events[0].0;
    assert!(
        small_started >= big.at - Duration::from_millis(5),
        "the small request started only once the big one ended"
    );
    // One that fits starts beside it.
    let mut engine = ScriptedEngine::new(ids(&"r".repeat(400)), 1_000).with_lanes(2);
    engine.step_delay = Some(Duration::from_millis(5));
    let handle = spawn(engine, Duration::from_secs(10));
    let big = consume(
        handle.generate(request(text(600, 'b'), 100)).unwrap(),
        Duration::ZERO,
    );
    std::thread::sleep(Duration::from_millis(30));
    let small = consume(
        handle.generate(request(text(100, 's'), 50)).unwrap(),
        Duration::ZERO,
    );
    let (big, small) = (big.join().unwrap(), small.join().unwrap());
    assert!(small.at < big.at, "it ran beside the big one");
}

#[test]
fn a_cancelled_lane_ends_alone_and_the_other_streams_on() {
    let mut engine = ScriptedEngine::new(ids(&"x".repeat(80)), 100_000).with_lanes(2);
    engine.step_delay = Some(Duration::from_millis(2));
    let handle = spawn(engine, Duration::from_secs(10));
    let keep = consume(
        handle.generate(request(text(300, 'k'), 60)).unwrap(),
        Duration::ZERO,
    );
    let stop = handle.generate(request(text(300, 'c'), 60)).unwrap();
    let ticket = stop.ticket.clone();
    let stop = consume(stop, Duration::ZERO);
    std::thread::sleep(Duration::from_millis(40));
    ticket.cancel();
    let (keep, stop) = (keep.join().unwrap(), stop.join().unwrap());
    assert_eq!(stop.finished.reason, FinishReason::Cancelled);
    assert_eq!(keep.finished.reason, FinishReason::Length);
    assert_eq!(keep.tokens(), 60);
}

#[test]
fn a_client_that_stops_reading_does_not_stall_the_other_lane() {
    let engine = ScriptedEngine::new(ids(&"y".repeat(400)), 100_000).with_lanes(2);
    let handle = spawn(engine, Duration::from_millis(300));
    // Its client never reads: 400 tokens overflow the event channel.
    let stalled = handle.generate(request(text(200, 'q'), 400)).unwrap();
    let reader = consume(
        handle.generate(request(text(200, 'r'), 300)).unwrap(),
        Duration::ZERO,
    );
    let reader = reader.join().unwrap();
    assert_eq!(reader.finished.reason, FinishReason::Length);
    assert_eq!(reader.tokens(), 300);
    // The stalled request is cancelled after the stall timeout, and the lane frees.
    let finished = stalled
        .finished
        .recv_timeout(Duration::from_secs(5))
        .unwrap();
    assert_eq!(finished.reason, FinishReason::Cancelled);
    drop(stalled.events);
    let (_, after) = handle
        .generate(request(text(10, 'z'), 2))
        .unwrap()
        .wait()
        .unwrap();
    assert_eq!(after.reason, FinishReason::Length);
}

#[test]
fn one_lane_is_the_engine_calls_of_0_5() {
    // `--lanes 1`: every chunk whole, never a batched or resumed call.
    let engine = ScriptedEngine::new(ids("abc"), 100_000);
    let handle = spawn(engine, Duration::from_secs(10));
    assert_eq!(handle.lanes(), 1);
    let a = consume(
        handle.generate(request(text(5_000, 'a'), 3)).unwrap(),
        Duration::ZERO,
    );
    let b = consume(
        handle.generate(request(text(100, 'b'), 3)).unwrap(),
        Duration::ZERO,
    );
    let (a, b) = (a.join().unwrap(), b.join().unwrap());
    assert!(
        b.events[0].0 >= a.at - Duration::from_millis(5),
        "served in turn"
    );
    let _ = &*TOKENIZER;
}
