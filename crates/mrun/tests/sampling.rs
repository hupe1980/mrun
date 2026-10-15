//! Request validation and stop handling on `ScriptedEngine`: invalid sampling is rejected before
//! any work; stop sequences and EOS end with `Stop` and leave no stop tokens in the state.

mod common;

use common::{ids, request, run_collect, text};
use mrun::runtime::generate::FinishReason;
use mrun::runtime::scripted::ScriptedEngine;
use mrun::runtime::session::Session;
use mrun_engine::Sampling;

const RB: u64 = 1_000;

#[test]
fn invalid_sampling_is_rejected_before_any_work() {
    for (sampling, needle) in [
        (
            Sampling {
                temperature: f32::NAN,
                ..Sampling::GREEDY
            },
            "temperature",
        ),
        (
            Sampling {
                temperature: -0.5,
                ..Sampling::GREEDY
            },
            "temperature",
        ),
        (
            Sampling {
                temperature: f32::INFINITY,
                ..Sampling::GREEDY
            },
            "temperature",
        ),
        (
            Sampling {
                top_p: 0.0,
                ..Sampling::GREEDY
            },
            "top_p",
        ),
        (
            Sampling {
                top_p: 1.5,
                ..Sampling::GREEDY
            },
            "top_p",
        ),
        (
            Sampling {
                min_p: -0.1,
                ..Sampling::GREEDY
            },
            "min_p",
        ),
        (
            Sampling {
                min_p: 2.0,
                ..Sampling::GREEDY
            },
            "min_p",
        ),
    ] {
        let mut engine = ScriptedEngine::new(ids("abc"), 1_000);
        let mut req = request(ids("q"), 5);
        req.sampling = sampling;
        let (events, finished) = run_collect(&mut engine, &mut Session::default(), req);
        assert!(
            matches!(&finished.reason, FinishReason::Rejected(message) if message.contains(needle)),
            "{sampling:?}: {:?}",
            finished.reason
        );
        assert!(events.is_empty());
        assert!(engine.processed.is_empty());
        assert_eq!(finished.prefilled, 0);
    }
}

#[test]
fn valid_sampling_settings_are_accepted() {
    let sampling = Sampling {
        temperature: 0.7,
        top_p: 0.9,
        top_k: 20,
        min_p: 0.05,
        seed: Some(42),
    };
    let mut engine = ScriptedEngine::new(ids("ok"), 1_000);
    let mut req = request(ids("q"), 5);
    req.sampling = sampling;
    let (events, finished) = run_collect(&mut engine, &mut Session::default(), req);
    assert_eq!(finished.reason, FinishReason::Stop { stop: None });
    assert_eq!(text(&events), "ok");
}

#[test]
fn invalid_stop_lists_are_rejected() {
    for stop in [
        vec!["s".to_string(); 5],
        vec![String::new()],
        vec!["x".repeat(257)],
    ] {
        let mut req = request(ids("q"), 5);
        req.stop = stop;
        let (_, finished) = run_collect(
            &mut ScriptedEngine::new(ids("abc"), 1_000),
            &mut Session::default(),
            req,
        );
        assert!(matches!(finished.reason, FinishReason::Rejected(_)));
    }
}

#[test]
fn stop_sequence_split_across_tokens_ends_with_its_text_trimmed() {
    let mut req = request(ids("q"), 50);
    req.stop = vec!["END".into(), "never".into()];
    let mut engine = ScriptedEngine::new(ids("one ENtwo END three"), 1_000);
    let mut session = Session::default();
    let (events, finished) = run_collect(&mut engine, &mut session, req);
    assert_eq!(
        finished.reason,
        FinishReason::Stop {
            stop: Some("END".into())
        }
    );
    assert_eq!(text(&events), "one ENtwo ");
    assert_eq!(session.committed(), ids("qone ENtwo "));
    assert_eq!(session.tokens(), ids("qone ENtwo "));
    assert_eq!(
        finished.stats.generated_tokens,
        ids("one ENtwo END").len() as u32
    );
}

#[test]
fn eos_ends_with_stop_and_is_not_kept() {
    let mut engine = ScriptedEngine::new(ids("done"), 1_000);
    let mut session = Session::default();
    let (_, finished) = run_collect(&mut engine, &mut session, request(ids("q"), 50));
    assert_eq!(finished.reason, FinishReason::Stop { stop: None });
    assert_eq!(finished.stats.generated_tokens, 4);
    assert_eq!(session.tokens(), ids("qdone"));
}

#[test]
fn hybrid_stop_falls_back_to_the_generation_checkpoint() {
    let prompt = ids("<user>hi<asst>");
    let mut req = request(prompt.clone(), 50);
    req.stop = vec!["</x>".into()];
    req.generation_start = Some(8);
    let mut engine = ScriptedEngine::new_hybrid(ids("reply</x>tail"), 1_000, RB);
    let mut session = Session::default();
    let (events, finished) = run_collect(&mut engine, &mut session, req);
    assert_eq!(
        finished.reason,
        FinishReason::Stop {
            stop: Some("</x>".into())
        }
    );
    assert_eq!(text(&events), "reply");
    let mut committed = prompt.clone();
    committed.extend(ids("reply"));
    assert_eq!(session.committed(), committed);
    // Recurrent state cannot be cut inside the reply: back to the checkpoint.
    assert_eq!(session.tokens(), &prompt[..8]);
}
