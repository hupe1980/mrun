//! The generation loop, generic over [`Engine`]: admission, prefill of all but the
//! last prompt token, decode with stop and UTF-8 handling, exactly one `Finished` event.

use crate::chat::stop::StopMatcher;
use crate::chat::tokenizer::StreamDecoder;
use mrun_engine::{CancelToken, Engine, EngineError, Sampling};
use std::time::{Duration, Instant};

/// Turns generated token ids into text without splitting characters.
pub trait Detokenizer {
    fn step(&mut self, id: u32) -> Result<Option<String>, String>;
}

impl Detokenizer for StreamDecoder<'_> {
    fn step(&mut self, id: u32) -> Result<Option<String>, String> {
        StreamDecoder::step(self, id).map_err(|error| error.to_string())
    }
}

/// Receives events; returns `false` when the consumer is gone (the request is then cancelled).
pub trait EventSink {
    fn send(&mut self, event: Event) -> bool;
}

impl EventSink for Vec<Event> {
    fn send(&mut self, event: Event) -> bool {
        self.push(event);
        true
    }
}

#[derive(Clone, Debug)]
pub struct GenerateRequest {
    /// Fully rendered prompt tokens (the whole conversation).
    pub prompt: Vec<u32>,
    pub sampling: Sampling,
    pub max_tokens: u32,
    pub stop: Vec<String>,
    pub cancel: CancelToken,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum FinishReason {
    /// End-of-sequence token or a stop sequence (`stop` holds the sequence).
    Stop {
        stop: Option<String>,
    },
    Length,
    Cancelled,
    /// Refused before any processing (capacity, invalid input).
    Rejected(String),
    /// The engine failed; the session was reset.
    Error(String),
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Stats {
    pub prompt_tokens: u32,
    pub cached_tokens: u32,
    pub generated_tokens: u32,
    /// Output limit actually applied: the requested one, clamped to the remaining capacity.
    pub max_tokens: u32,
    pub ttft: Option<Duration>,
    pub prefill_tps: f64,
    pub decode_tps: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Event {
    Started {
        prompt: u32,
        cached: u32,
    },
    Progress {
        done: u32,
        total: u32,
    },
    Token {
        id: u32,
        text: String,
    },
    Finished {
        reason: FinishReason,
        /// Prompt tokens held in the attention state when the request ended.
        prefilled: u32,
        stats: Stats,
    },
}

/// The conversation state mirrored from the engine: exactly the tokens in its active cache.
#[derive(Debug, Default)]
pub struct Session {
    kv_tokens: Vec<u32>,
}

impl Session {
    pub fn tokens(&self) -> &[u32] {
        &self.kv_tokens
    }

    pub fn clear<E: Engine>(&mut self, engine: &mut E) {
        engine.reset();
        self.kv_tokens.clear();
    }
}

/// Runs one request to completion and sends exactly one `Finished` event.
pub fn generate<E: Engine>(
    engine: &mut E,
    session: &mut Session,
    detok: &mut dyn Detokenizer,
    request: GenerateRequest,
    sink: &mut dyn EventSink,
) {
    let mut stats = Stats {
        prompt_tokens: request.prompt.len() as u32,
        ..Stats::default()
    };
    let mut prefilled = 0;
    let reason = run(
        engine,
        session,
        detok,
        &request,
        sink,
        &mut stats,
        &mut prefilled,
    );
    if matches!(reason, FinishReason::Error(_)) {
        session.clear(engine);
        prefilled = 0;
    }
    sink.send(Event::Finished {
        reason,
        prefilled,
        stats,
    });
}

fn run<E: Engine>(
    engine: &mut E,
    session: &mut Session,
    detok: &mut dyn Detokenizer,
    request: &GenerateRequest,
    sink: &mut dyn EventSink,
    stats: &mut Stats,
    prefilled: &mut u32,
) -> FinishReason {
    let prompt = &request.prompt;
    if prompt.is_empty() {
        return FinishReason::Rejected("the prompt is empty".into());
    }
    if request.max_tokens == 0 {
        return FinishReason::Rejected("max tokens must be at least 1".into());
    }
    let mut matcher = match StopMatcher::new(request.stop.clone()) {
        Ok(matcher) => matcher,
        Err(error) => return FinishReason::Rejected(error.to_string()),
    };
    // Refuse only a prompt that leaves no room for one output token; otherwise clamp the
    // output limit to what fits, so large client defaults never turn away a turn that fits.
    let capacity = engine.budget().kv_capacity_tokens;
    let Some(room) = u32::try_from(prompt.len())
        .ok()
        .and_then(|len| capacity.checked_sub(len))
        .filter(|room| *room > 0)
    else {
        return FinishReason::Rejected(
            EngineError::Capacity {
                requested: (prompt.len() as u64 + 1).min(u64::from(u32::MAX)) as u32,
                available: capacity,
            }
            .to_string(),
        );
    };
    let max_tokens = request.max_tokens.min(room);
    stats.max_tokens = max_tokens;

    // Admit before touching the cache, so a rejected request leaves the conversation intact.
    match engine.reserve(prompt.len() as u32 + max_tokens) {
        Ok(()) => {}
        Err(error @ EngineError::Capacity { .. }) => {
            return FinishReason::Rejected(error.to_string());
        }
        Err(error) => return FinishReason::Error(error.to_string()),
    }

    // Reuse the longest exact prefix, always leaving the last prompt token to process.
    let common = session
        .kv_tokens
        .iter()
        .zip(prompt)
        .take_while(|(a, b)| a == b)
        .count()
        .min(prompt.len() - 1);
    engine.trim(common as u32);
    session.kv_tokens.truncate(common);
    *prefilled = common as u32;
    stats.cached_tokens = common as u32;
    if !sink.send(Event::Started {
        prompt: prompt.len() as u32,
        cached: common as u32,
    }) {
        return FinishReason::Cancelled;
    }

    let started = Instant::now();
    let pending = &prompt[common..prompt.len() - 1];
    let total_pending = pending.len() as u32;
    let mut consumer_gone = false;
    let done = match engine.prefill(pending, &request.cancel, &mut |done| {
        if !consumer_gone
            && !sink.send(Event::Progress {
                done,
                total: total_pending,
            })
        {
            consumer_gone = true;
            request.cancel.cancel();
        }
    }) {
        Ok(done) => done as usize,
        Err(error) => return FinishReason::Error(error.to_string()),
    };
    session.kv_tokens.extend_from_slice(&pending[..done]);
    *prefilled = (common + done) as u32;
    if done < pending.len() {
        return FinishReason::Cancelled;
    }

    let eos = &engine.info().eos_tokens.clone();
    let mut last = prompt[prompt.len() - 1];
    let mut decode_started = None;
    loop {
        if request.cancel.is_cancelled() {
            return FinishReason::Cancelled;
        }
        let next = match engine.decode(last, &request.sampling, None) {
            Ok(next) => next,
            Err(error) => return FinishReason::Error(error.to_string()),
        };
        session.kv_tokens.push(last);
        if decode_started.is_none() {
            // The first decode step processes the last prompt token.
            *prefilled = session.kv_tokens.len() as u32;
            let elapsed = started.elapsed().as_secs_f64();
            if elapsed > 0.0 {
                stats.prefill_tps = (prompt.len() - common) as f64 / elapsed;
            }
            decode_started = Some(Instant::now());
        }
        if eos.contains(&next) {
            flush(&mut matcher, next, sink, stats, started);
            return finish_decode(stats, decode_started, FinishReason::Stop { stop: None });
        }
        stats.generated_tokens += 1;
        let text = match detok.step(next) {
            Ok(text) => text,
            Err(error) => return FinishReason::Error(error),
        };
        if let Some(text) = text {
            let fed = matcher.feed(&text);
            if !fed.emit.is_empty() {
                stats.ttft.get_or_insert_with(|| started.elapsed());
                if !sink.send(Event::Token {
                    id: next,
                    text: fed.emit,
                }) {
                    return FinishReason::Cancelled;
                }
            }
            if let Some(stop) = fed.stopped {
                return finish_decode(
                    stats,
                    decode_started,
                    FinishReason::Stop { stop: Some(stop) },
                );
            }
        }
        if stats.generated_tokens >= max_tokens {
            flush(&mut matcher, next, sink, stats, started);
            return finish_decode(stats, decode_started, FinishReason::Length);
        }
        last = next;
    }
}

fn flush(
    matcher: &mut StopMatcher,
    id: u32,
    sink: &mut dyn EventSink,
    stats: &mut Stats,
    started: Instant,
) {
    let rest = matcher.finish();
    if !rest.is_empty() {
        stats.ttft.get_or_insert_with(|| started.elapsed());
        sink.send(Event::Token { id, text: rest });
    }
}

fn finish_decode(
    stats: &mut Stats,
    decode_started: Option<Instant>,
    reason: FinishReason,
) -> FinishReason {
    if let Some(start) = decode_started {
        let elapsed = start.elapsed().as_secs_f64();
        if elapsed > 0.0 && stats.generated_tokens > 1 {
            stats.decode_tps = f64::from(stats.generated_tokens - 1) / elapsed;
        }
    }
    reason
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chat::tokenizer::Tokenizer;
    use crate::chat::tokenizer::test_support::byte_tokenizer;
    use crate::runtime::scripted::ScriptedEngine;

    struct Run {
        events: Vec<Event>,
        engine: ScriptedEngine,
        session: Session,
    }

    fn ids(tokenizer: &Tokenizer, text: &str) -> Vec<u32> {
        tokenizer.encode(text).unwrap()
    }

    fn request(prompt: Vec<u32>, max_tokens: u32) -> GenerateRequest {
        GenerateRequest {
            prompt,
            sampling: Sampling::GREEDY,
            max_tokens,
            stop: Vec::new(),
            cancel: CancelToken::new(),
        }
    }

    fn run_with(engine: ScriptedEngine, session: Session, request: GenerateRequest) -> Run {
        let tokenizer = byte_tokenizer();
        let mut engine = engine;
        let mut session = session;
        let mut events = Vec::new();
        generate(
            &mut engine,
            &mut session,
            &mut tokenizer.stream(),
            request,
            &mut events,
        );
        assert_eq!(
            engine.active_tokens(),
            session.tokens(),
            "session mirrors the engine cache"
        );
        Run {
            events,
            engine,
            session,
        }
    }

    fn finished(events: &[Event]) -> (&FinishReason, u32, &Stats) {
        let finished: Vec<_> = events
            .iter()
            .filter_map(|event| match event {
                Event::Finished {
                    reason,
                    prefilled,
                    stats,
                } => Some((reason, *prefilled, stats)),
                _ => None,
            })
            .collect();
        assert_eq!(finished.len(), 1, "exactly one Finished: {events:?}");
        assert!(matches!(events.last(), Some(Event::Finished { .. })));
        finished[0]
    }

    fn text(events: &[Event]) -> String {
        events
            .iter()
            .filter_map(|event| match event {
                Event::Token { text, .. } => Some(text.as_str()),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn streams_text_until_eos_and_reports_stats() {
        let tokenizer = byte_tokenizer();
        let engine = ScriptedEngine::new(ids(&tokenizer, "Hi €!"), 1_000);
        let run = run_with(
            engine,
            Session::default(),
            request(ids(&tokenizer, "hello"), 50),
        );
        assert_eq!(text(&run.events), "Hi €!");
        let (reason, prefilled, stats) = finished(&run.events);
        assert_eq!(*reason, FinishReason::Stop { stop: None });
        assert_eq!(prefilled, 5);
        assert_eq!(stats.prompt_tokens, 5);
        assert_eq!(stats.generated_tokens, 7);
        assert!(stats.ttft.is_some());
        assert_eq!(
            run.events[0],
            Event::Started {
                prompt: 5,
                cached: 0
            }
        );
    }

    #[test]
    fn follow_up_turn_processes_only_new_tokens() {
        let tokenizer = byte_tokenizer();
        let engine = ScriptedEngine::new(ids(&tokenizer, "ok"), 1_000);
        let first = run_with(
            engine,
            Session::default(),
            request(ids(&tokenizer, "abc"), 10),
        );
        let mut prompt = first.session.tokens().to_vec();
        prompt.extend(ids(&tokenizer, "ok xyz"));
        let mut engine = first.engine;
        engine.processed.clear();
        let second = run_with(engine, first.session, request(prompt.clone(), 10));
        // The session holds "abc" plus the generated "ok" (the EOS token is never fed).
        assert_eq!(
            second.events[0],
            Event::Started {
                prompt: prompt.len() as u32,
                cached: 5
            }
        );
        assert_eq!(second.engine.processed.len(), prompt.len() - 5);
    }

    #[test]
    fn edited_history_reuses_only_the_common_prefix() {
        let tokenizer = byte_tokenizer();
        let engine = ScriptedEngine::new(Vec::new(), 1_000);
        let first = run_with(
            engine,
            Session::default(),
            request(ids(&tokenizer, "abcdef"), 1),
        );
        let second = run_with(
            first.engine,
            first.session,
            request(ids(&tokenizer, "abXdef"), 1),
        );
        assert_eq!(
            second.events[0],
            Event::Started {
                prompt: 6,
                cached: 2
            }
        );
    }

    #[test]
    fn identical_prompt_reprocesses_only_the_last_token() {
        let tokenizer = byte_tokenizer();
        let engine = ScriptedEngine::new(Vec::new(), 1_000);
        let prompt = ids(&tokenizer, "same prompt");
        let first = run_with(engine, Session::default(), request(prompt.clone(), 1));
        let mut engine = first.engine;
        engine.processed.clear();
        let second = run_with(engine, first.session, request(prompt.clone(), 1));
        assert_eq!(second.engine.processed, [*prompt.last().unwrap()]);
    }

    #[test]
    fn stop_sequence_ends_generation_without_showing_it() {
        let tokenizer = byte_tokenizer();
        let engine = ScriptedEngine::new(ids(&tokenizer, "answer</done>more"), 1_000);
        let mut req = request(ids(&tokenizer, "q"), 50);
        req.stop = vec!["</done>".into()];
        let run = run_with(engine, Session::default(), req);
        assert_eq!(text(&run.events), "answer");
        let (reason, _, _) = finished(&run.events);
        assert_eq!(
            *reason,
            FinishReason::Stop {
                stop: Some("</done>".into())
            }
        );
    }

    #[test]
    fn output_limit_ends_with_length() {
        let tokenizer = byte_tokenizer();
        let engine = ScriptedEngine::new(ids(&tokenizer, "abcdefgh"), 1_000);
        let run = run_with(engine, Session::default(), request(ids(&tokenizer, "q"), 3));
        assert_eq!(text(&run.events), "abc");
        let (reason, _, stats) = finished(&run.events);
        assert_eq!(*reason, FinishReason::Length);
        assert_eq!(stats.generated_tokens, 3);
    }

    #[test]
    fn output_limit_is_clamped_to_capacity() {
        let tokenizer = byte_tokenizer();
        let prompt = ids(&tokenizer, "0123456789");
        let run = run_with(
            ScriptedEngine::new(ids(&tokenizer, "xyz"), 12),
            Session::default(),
            request(prompt, 100),
        );
        let (reason, _, stats) = finished(&run.events);
        assert_eq!(*reason, FinishReason::Length);
        assert_eq!(stats.max_tokens, 2);
        assert_eq!(stats.generated_tokens, 2);
    }

    #[test]
    fn prompt_without_room_for_output_is_rejected() {
        let tokenizer = byte_tokenizer();
        let prompt = ids(&tokenizer, "0123456789");
        let at = run_with(
            ScriptedEngine::new(ids(&tokenizer, "xyz"), 11),
            Session::default(),
            request(prompt.clone(), 3),
        );
        assert_eq!(*finished(&at.events).0, FinishReason::Length);
        let over = run_with(
            ScriptedEngine::new(ids(&tokenizer, "xyz"), 10),
            Session::default(),
            request(prompt, 3),
        );
        let (reason, prefilled, _) = finished(&over.events);
        assert!(matches!(reason, FinishReason::Rejected(message) if message.contains("11 tokens")));
        assert_eq!(prefilled, 0);
        assert_eq!(
            over.engine.processed.len(),
            0,
            "nothing processed when rejected"
        );
    }

    #[test]
    fn rejected_request_keeps_the_cached_conversation() {
        let tokenizer = byte_tokenizer();
        let first = run_with(
            ScriptedEngine::new(Vec::new(), 20),
            Session::default(),
            request(ids(&tokenizer, "abcdefgh"), 1),
        );
        let cached = first.session.tokens().to_vec();
        let rejected = run_with(
            first.engine,
            first.session,
            request(ids(&tokenizer, "0123456789abcdefghijk"), 1),
        );
        assert!(matches!(
            finished(&rejected.events).0,
            FinishReason::Rejected(_)
        ));
        assert_eq!(rejected.session.tokens(), cached);
    }

    #[test]
    fn invalid_requests_are_rejected_up_front() {
        let tokenizer = byte_tokenizer();
        for (prompt, max_tokens) in [(Vec::new(), 5), (ids(&tokenizer, "q"), 0)] {
            let run = run_with(
                ScriptedEngine::new(Vec::new(), 100),
                Session::default(),
                request(prompt, max_tokens),
            );
            assert!(matches!(finished(&run.events).0, FinishReason::Rejected(_)));
        }
        let mut req = request(ids(&tokenizer, "q"), 5);
        req.stop = vec!["s".into(); 5];
        let run = run_with(
            ScriptedEngine::new(Vec::new(), 100),
            Session::default(),
            req,
        );
        assert!(matches!(finished(&run.events).0, FinishReason::Rejected(_)));
    }

    #[test]
    fn cancel_during_prefill_keeps_processed_prefix_consistent() {
        let tokenizer = byte_tokenizer();
        let mut engine = ScriptedEngine::new(ids(&tokenizer, "never"), 10_000);
        engine.cancel_granularity = 100;
        let prompt = ids(&tokenizer, &"x".repeat(1_000));
        let req = request(prompt, 10);
        let cancel = req.cancel.clone();
        let mut session = Session::default();
        let mut events = Vec::new();
        struct CancelAfterProgress<'a>(&'a mut Vec<Event>, CancelToken);
        impl EventSink for CancelAfterProgress<'_> {
            fn send(&mut self, event: Event) -> bool {
                if matches!(event, Event::Progress { done: 200, .. }) {
                    self.1.cancel();
                }
                self.0.push(event);
                true
            }
        }
        generate(
            &mut engine,
            &mut session,
            &mut tokenizer.stream(),
            req,
            &mut CancelAfterProgress(&mut events, cancel),
        );
        let (reason, prefilled, _) = finished(&events);
        assert_eq!(*reason, FinishReason::Cancelled);
        assert_eq!(prefilled, 200);
        assert_eq!(engine.active_tokens(), session.tokens());
        assert!(text(&events).is_empty());
    }

    #[test]
    fn consumer_gone_cancels() {
        struct Gone(usize);
        impl EventSink for Gone {
            fn send(&mut self, event: Event) -> bool {
                if let Event::Finished { reason, .. } = event {
                    assert_eq!(reason, FinishReason::Cancelled);
                    self.0 += 1;
                }
                false
            }
        }
        let tokenizer = byte_tokenizer();
        let mut engine = ScriptedEngine::new(ids(&tokenizer, "abc"), 100);
        let mut sink = Gone(0);
        generate(
            &mut engine,
            &mut Session::default(),
            &mut tokenizer.stream(),
            request(ids(&tokenizer, "q"), 10),
            &mut sink,
        );
        assert_eq!(sink.0, 1);
    }
}
