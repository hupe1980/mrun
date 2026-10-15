//! The per-request generation loop, generic over [`Engine`] and a [`ReusePolicy`]: admission,
//! reuse, prefill of all but the last prompt token, pipelined decode with stop and UTF-8
//! handling, and a finish that leaves the state holding committed tokens only.

use crate::chat::stop::StopMatcher;
use crate::chat::tokenizer::StreamDecoder;
use mrun_engine::{CancelToken, Checkpoint, Engine, EngineError, PrefillPlan, Sampling};
use std::time::{Duration, Instant};

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
    /// Position where the generation header starts; hybrids checkpoint there for the next turn.
    pub generation_start: Option<u32>,
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
    /// The engine failed; the state was reset.
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
        max_tokens: u32,
    },
    Progress {
        done: u32,
        total: u32,
    },
    Token {
        id: u32,
        text: String,
    },
}

/// The single final report of a request.
#[derive(Clone, Debug, PartialEq)]
pub struct Finished {
    pub reason: FinishReason,
    /// Prompt tokens held in the state when the request ended.
    pub prefilled: u32,
    pub stats: Stats,
}

/// Owned form of a [`PrefillPlan`].
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ChunkPlan {
    pub boundaries: Vec<u32>,
    pub checkpoints: Vec<u32>,
}

impl ChunkPlan {
    pub fn as_plan(&self) -> PrefillPlan<'_> {
        PrefillPlan {
            boundaries: &self.boundaries,
            checkpoints: &self.checkpoints,
        }
    }
}

/// Decides which state a request reuses and what is kept after it.
pub trait ReusePolicy {
    /// Leaves the engine holding a prefix of the prompt of at most `prompt.len() − 1` tokens
    /// and returns its length.
    fn prepare(
        &mut self,
        engine: &mut dyn Engine,
        request: &GenerateRequest,
    ) -> Result<u32, EngineError>;
    /// Chunk ends and checkpoints for prefilling the prompt from `keep`.
    fn plan(&self, engine: &dyn Engine, request: &GenerateRequest, keep: u32) -> ChunkPlan;
    /// Frees cached state after `NeedsMemory{bytes}`; `false` when nothing is left to free.
    fn evict(&mut self, engine: &mut dyn Engine, bytes: u64) -> bool;
    /// Prefill paused for memory pressure; it resumes when this returns.
    fn on_yield(&mut self, _engine: &mut dyn Engine) {}
    /// Records a finished request. The engine holds `committed[..target]` (plus possibly
    /// uncommitted tokens beyond it); the policy cuts it to a prefix of `committed`, takes
    /// ownership of `captured` and returns the length the state holds.
    fn commit(
        &mut self,
        engine: &mut dyn Engine,
        committed: &[u32],
        target: u32,
        captured: Vec<Checkpoint>,
    ) -> u32;
    /// Forgets all state (after an engine error or a panic).
    fn clear(&mut self, engine: &mut dyn Engine);
}

/// Progress of one request, used to decide what is committed at the finish.
#[derive(Debug, Default)]
struct Turn {
    /// Prompt tokens confirmed in the state.
    prefilled: u32,
    /// The last prompt token has been fed.
    decoding: bool,
    /// Output tokens fed and confirmed.
    forwarded: u32,
    /// Sampled output tokens (EOS excluded).
    output: Vec<u32>,
    /// Byte offset in the decoded text where each output token's text ends, once complete.
    text_ends: Vec<usize>,
    decoded_bytes: usize,
    emitted_bytes: usize,
    captured: Vec<Checkpoint>,
}

impl Turn {
    /// Output tokens whose text was fully emitted.
    fn committed_output(&self) -> usize {
        self.text_ends
            .iter()
            .take_while(|&&end| end <= self.emitted_bytes)
            .count()
    }
}

/// Runs one request to completion; the result is its only `Finished`.
pub fn generate<E: Engine, P: ReusePolicy>(
    engine: &mut E,
    policy: &mut P,
    detok: &mut StreamDecoder<'_>,
    request: GenerateRequest,
    sink: &mut dyn EventSink,
) -> Finished {
    let mut stats = Stats {
        prompt_tokens: request.prompt.len() as u32,
        ..Stats::default()
    };
    let mut turn = Turn::default();
    let reason = match run(engine, policy, detok, &request, sink, &mut stats, &mut turn) {
        Ok(reason) => reason,
        Err(message) => FinishReason::Error(message),
    };
    let prefilled = match reason {
        FinishReason::Rejected(_) => 0,
        FinishReason::Error(_) => {
            for checkpoint in turn.captured {
                engine.drop_checkpoint(checkpoint.id);
            }
            policy.clear(engine);
            0
        }
        _ => {
            let prompt = &request.prompt;
            let shown = turn.committed_output();
            let mut committed = prompt.clone();
            committed.extend_from_slice(&turn.output[..shown]);
            let target = if turn.decoding {
                prompt.len() as u32 + turn.forwarded.min(shown as u32)
            } else {
                turn.prefilled
            };
            let reached = policy.commit(engine, &committed, target, turn.captured);
            reached.min(prompt.len() as u32)
        }
    };
    Finished {
        reason,
        prefilled,
        stats,
    }
}

/// Retries `op` after the policy evicts, while `NeedsMemory` can be satisfied.
fn with_memory<E: Engine, P: ReusePolicy, T>(
    engine: &mut E,
    policy: &mut P,
    mut op: impl FnMut(&mut E) -> Result<T, EngineError>,
) -> Result<T, String> {
    loop {
        match op(engine) {
            Err(EngineError::NeedsMemory { bytes }) if policy.evict(engine, bytes) => {}
            result => return result.map_err(|error| error.to_string()),
        }
    }
}

/// Returns the finish reason, or `Err` with a message when the engine failed.
fn run<E: Engine, P: ReusePolicy>(
    engine: &mut E,
    policy: &mut P,
    detok: &mut StreamDecoder<'_>,
    request: &GenerateRequest,
    sink: &mut dyn EventSink,
    stats: &mut Stats,
    turn: &mut Turn,
) -> Result<FinishReason, String> {
    let prompt = &request.prompt;
    let cancel = &request.cancel;
    if prompt.is_empty() {
        return Ok(FinishReason::Rejected("the prompt is empty".into()));
    }
    if request.max_tokens == 0 {
        return Ok(FinishReason::Rejected(
            "max tokens must be at least 1".into(),
        ));
    }
    if let Err(message) = request.sampling.validate() {
        return Ok(FinishReason::Rejected(message));
    }
    let mut matcher = match StopMatcher::new(request.stop.clone()) {
        Ok(matcher) => matcher,
        Err(error) => return Ok(FinishReason::Rejected(error.to_string())),
    };
    // Refuse only a prompt that leaves no room for one output token; clamp the output limit.
    let capacity = engine.budget().kv_capacity_tokens;
    let Some(room) = u32::try_from(prompt.len())
        .ok()
        .and_then(|len| capacity.checked_sub(len))
        .filter(|room| *room > 0)
    else {
        let requested = (prompt.len() as u64 + 1).min(u64::from(u32::MAX)) as u32;
        let error = EngineError::Capacity {
            requested,
            available: capacity,
        };
        return Ok(FinishReason::Rejected(error.to_string()));
    };
    let len = prompt.len() as u32;
    let limit = request.max_tokens.min(room);
    stats.max_tokens = limit;

    let keep = policy
        .prepare(engine, request)
        .map_err(|error| error.to_string())?;
    turn.prefilled = keep;
    engine
        .reserve(keep, len + limit)
        .map_err(|error| error.to_string())?;
    stats.cached_tokens = keep;
    if !sink.send(Event::Started {
        prompt: len,
        cached: keep,
        max_tokens: limit,
    }) {
        return Ok(FinishReason::Cancelled);
    }

    let started = Instant::now();
    let plan = policy.plan(engine, request, keep);
    let pending = &prompt[keep as usize..prompt.len() - 1];
    let total = pending.len() as u32;
    let mut done = 0;
    let mut consumer_gone = false;
    while done < total {
        let filled = with_memory(engine, policy, |engine| {
            engine.prefill(
                &pending[done as usize..],
                &plan.as_plan(),
                cancel,
                &mut |n| {
                    if !consumer_gone
                        && !sink.send(Event::Progress {
                            done: done + n,
                            total,
                        })
                    {
                        consumer_gone = true;
                        cancel.cancel();
                    }
                },
            )
        })?;
        done += filled.committed;
        turn.prefilled = keep + done;
        turn.captured.extend(filled.checkpoints);
        if filled.yielded {
            cancel.clear_yield();
            policy.on_yield(engine);
        } else if done < total {
            return Ok(FinishReason::Cancelled);
        }
    }

    let eos = engine.info().eos_tokens.clone();
    let mut last = prompt[prompt.len() - 1];
    let mut decode_started = None;
    loop {
        if cancel.is_cancelled() {
            return Ok(FinishReason::Cancelled);
        }
        let step = with_memory(engine, policy, |engine| engine.forward(last))?;
        if turn.decoding {
            turn.forwarded += 1;
        } else {
            turn.decoding = true;
            turn.prefilled = len;
        }
        let pipeline = limit - stats.generated_tokens > 1;
        let next = engine
            .sample(step, &request.sampling, None, pipeline)
            .map_err(|error| error.to_string())?;
        if decode_started.is_none() {
            // Time to first token: the first sampled token, even if its text is held back.
            let elapsed = started.elapsed();
            stats.ttft = Some(elapsed);
            if elapsed.as_secs_f64() > 0.0 {
                stats.prefill_tps = f64::from(len - keep) / elapsed.as_secs_f64();
            }
            decode_started = Some(Instant::now());
        }
        if eos.contains(&next) {
            flush(&mut matcher, next, sink, turn);
            return Ok(finish_decode(
                stats,
                decode_started,
                FinishReason::Stop { stop: None },
            ));
        }
        stats.generated_tokens += 1;
        turn.output.push(next);
        if let Some(text) = detok.step(next).map_err(|error| error.to_string())? {
            turn.decoded_bytes += text.len();
            let ends = turn.output.len() - turn.text_ends.len();
            turn.text_ends
                .extend(std::iter::repeat_n(turn.decoded_bytes, ends));
            let fed = matcher.feed(&text);
            if !emit(sink, next, fed.emit, turn) {
                return Ok(FinishReason::Cancelled);
            }
            if let Some(stop) = fed.stopped {
                return Ok(finish_decode(
                    stats,
                    decode_started,
                    FinishReason::Stop { stop: Some(stop) },
                ));
            }
        }
        if stats.generated_tokens >= limit {
            flush(&mut matcher, next, sink, turn);
            return Ok(finish_decode(stats, decode_started, FinishReason::Length));
        }
        last = next;
    }
}

/// Sends `text` (if any); `false` when the consumer is gone.
fn emit(sink: &mut dyn EventSink, id: u32, text: String, turn: &mut Turn) -> bool {
    if text.is_empty() {
        return true;
    }
    let bytes = text.len();
    let delivered = sink.send(Event::Token { id, text });
    if delivered {
        turn.emitted_bytes += bytes;
    }
    delivered
}

fn flush(matcher: &mut StopMatcher, id: u32, sink: &mut dyn EventSink, turn: &mut Turn) {
    emit(sink, id, matcher.finish(), turn);
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
    use crate::runtime::scripted::{KV_BYTES, ScriptedEngine};
    use crate::runtime::session::Session;

    const RB: u64 = 1_000;

    struct Run {
        events: Vec<Event>,
        finished: Finished,
        engine: ScriptedEngine,
        session: Session,
    }

    fn ids(text: &str) -> Vec<u32> {
        byte_tokenizer().encode(text).unwrap()
    }

    fn request(prompt: Vec<u32>, max_tokens: u32) -> GenerateRequest {
        GenerateRequest {
            prompt,
            sampling: Sampling::GREEDY,
            max_tokens,
            stop: Vec::new(),
            cancel: CancelToken::new(),
            generation_start: None,
        }
    }

    fn generate_into(
        engine: &mut ScriptedEngine,
        session: &mut Session,
        tokenizer: &Tokenizer,
        request: GenerateRequest,
        sink: &mut dyn EventSink,
    ) -> Finished {
        let finished = generate(engine, session, &mut tokenizer.stream(), request, sink);
        assert_eq!(
            engine.tokens(),
            session.tokens(),
            "session mirrors the state"
        );
        assert!(!engine.has_pending());
        assert!(session.committed().starts_with(session.tokens()));
        finished
    }

    fn run_with(engine: ScriptedEngine, session: Session, request: GenerateRequest) -> Run {
        let (mut engine, mut session) = (engine, session);
        let mut events = Vec::new();
        let finished = generate_into(
            &mut engine,
            &mut session,
            &byte_tokenizer(),
            request,
            &mut events,
        );
        Run {
            events,
            finished,
            engine,
            session,
        }
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

    fn concat(parts: &[&[u32]]) -> Vec<u32> {
        parts.concat()
    }

    #[test]
    fn streams_text_until_eos_and_reports_stats() {
        let run = run_with(
            ScriptedEngine::new(ids("Hi €!"), 1_000),
            Session::default(),
            request(ids("hello"), 50),
        );
        assert_eq!(text(&run.events), "Hi €!");
        let Finished {
            reason,
            prefilled,
            stats,
        } = &run.finished;
        assert_eq!(*reason, FinishReason::Stop { stop: None });
        assert_eq!(*prefilled, 5);
        assert_eq!((stats.prompt_tokens, stats.generated_tokens), (5, 7));
        assert_eq!(stats.max_tokens, 50);
        assert!(stats.ttft.is_some());
        assert_eq!(
            run.events[0],
            Event::Started {
                prompt: 5,
                cached: 0,
                max_tokens: 50
            }
        );
        // The pipelined EOS step was trimmed; prompt and reply stay.
        assert_eq!(run.session.tokens(), ids("helloHi €!"));
    }

    #[test]
    fn follow_up_turn_processes_only_new_tokens() {
        let first = run_with(
            ScriptedEngine::new(ids("ok"), 1_000),
            Session::default(),
            request(ids("abc"), 10),
        );
        let prompt = concat(&[first.session.tokens(), &ids(" xyz")]);
        let mut engine = first.engine;
        engine.processed.clear();
        let second = run_with(engine, first.session, request(prompt.clone(), 10));
        assert_eq!(
            second.events[0],
            Event::Started {
                prompt: prompt.len() as u32,
                cached: 5,
                max_tokens: 10
            }
        );
        assert!(second.engine.processed.starts_with(&prompt[5..]));
    }

    #[test]
    fn edited_history_reuses_only_the_common_prefix() {
        let first = run_with(
            ScriptedEngine::new(Vec::new(), 1_000),
            Session::default(),
            request(ids("abcdef"), 1),
        );
        let second = run_with(first.engine, first.session, request(ids("abXdef"), 1));
        assert_eq!(
            second.events[0],
            Event::Started {
                prompt: 6,
                cached: 2,
                max_tokens: 1
            }
        );
    }

    #[test]
    fn identical_prompt_reprocesses_only_the_last_token() {
        let prompt = ids("same prompt");
        let first = run_with(
            ScriptedEngine::new(Vec::new(), 1_000),
            Session::default(),
            request(prompt.clone(), 1),
        );
        let mut engine = first.engine;
        engine.processed.clear();
        let second = run_with(engine, first.session, request(prompt.clone(), 1));
        assert_eq!(second.engine.processed, [*prompt.last().unwrap()]);
    }

    #[test]
    fn stop_sequence_is_neither_shown_nor_committed() {
        let mut req = request(ids("q"), 50);
        req.stop = vec!["</done>".into()];
        let run = run_with(
            ScriptedEngine::new(ids("answer</done>more"), 1_000),
            Session::default(),
            req,
        );
        assert_eq!(text(&run.events), "answer");
        assert_eq!(
            run.finished.reason,
            FinishReason::Stop {
                stop: Some("</done>".into())
            }
        );
        assert_eq!(run.session.committed(), ids("qanswer"));
        assert_eq!(run.session.tokens(), ids("qanswer"));
    }

    #[test]
    fn output_limit_ends_with_length_and_keeps_the_fed_tokens() {
        let run = run_with(
            ScriptedEngine::new(ids("abcdefgh"), 1_000),
            Session::default(),
            request(ids("q"), 3),
        );
        assert_eq!(text(&run.events), "abc");
        assert_eq!(run.finished.reason, FinishReason::Length);
        assert_eq!(run.finished.stats.generated_tokens, 3);
        assert_eq!(run.session.committed(), ids("qabc"));
        // The last sampled token was never fed.
        assert_eq!(run.session.tokens(), ids("qab"));
    }

    #[test]
    fn incomplete_character_is_neither_shown_nor_committed() {
        let run = run_with(
            ScriptedEngine::new(ids("a€"), 1_000),
            Session::default(),
            request(ids("q"), 2),
        );
        assert_eq!(text(&run.events), "a");
        assert_eq!(run.session.committed(), ids("qa"));
    }

    #[test]
    fn output_limit_is_clamped_to_capacity() {
        let run = run_with(
            ScriptedEngine::new(ids("xyz"), 12),
            Session::default(),
            request(ids("0123456789"), 100),
        );
        assert_eq!(run.finished.reason, FinishReason::Length);
        assert_eq!(run.finished.stats.max_tokens, 2);
        assert_eq!(run.finished.stats.generated_tokens, 2);
    }

    #[test]
    fn prompt_without_room_for_output_is_rejected() {
        let prompt = ids("0123456789");
        let at = run_with(
            ScriptedEngine::new(ids("xyz"), 11),
            Session::default(),
            request(prompt.clone(), 3),
        );
        assert_eq!(at.finished.reason, FinishReason::Length);
        let over = run_with(
            ScriptedEngine::new(ids("xyz"), 10),
            Session::default(),
            request(prompt, 3),
        );
        assert!(
            matches!(&over.finished.reason, FinishReason::Rejected(message) if message.contains("11 tokens"))
        );
        assert_eq!(over.finished.prefilled, 0);
        assert!(over.engine.processed.is_empty(), "nothing processed");
    }

    #[test]
    fn rejected_request_keeps_the_cached_conversation() {
        let first = run_with(
            ScriptedEngine::new(Vec::new(), 20),
            Session::default(),
            request(ids("abcdefgh"), 1),
        );
        let cached = first.session.tokens().to_vec();
        let rejected = run_with(
            first.engine,
            first.session,
            request(ids("0123456789abcdefghijk"), 1),
        );
        assert!(matches!(
            rejected.finished.reason,
            FinishReason::Rejected(_)
        ));
        assert_eq!(rejected.session.tokens(), cached);
    }

    #[test]
    fn invalid_requests_are_rejected_up_front() {
        let mut bad_stop = request(ids("q"), 5);
        bad_stop.stop = vec!["s".into(); 5];
        let mut bad_sampling = request(ids("q"), 5);
        bad_sampling.sampling.top_p = 0.0;
        for req in [
            request(Vec::new(), 5),
            request(ids("q"), 0),
            bad_stop,
            bad_sampling,
        ] {
            let run = run_with(
                ScriptedEngine::new(Vec::new(), 100),
                Session::default(),
                req,
            );
            assert!(matches!(run.finished.reason, FinishReason::Rejected(_)));
        }
    }

    #[test]
    fn cancel_mid_chunk_keeps_whole_chunks() {
        let mut engine = ScriptedEngine::new_hybrid(ids("never"), 10_000, RB);
        // Four layers per chunk: layer 6 is inside the second chunk.
        engine.cancel_after_layers = Some(6);
        let prompt = ids(&"x".repeat(1_000));
        let mut req = request(prompt.clone(), 10);
        req.generation_start = Some(400);
        let run = run_with(engine, Session::default(), req);
        assert_eq!(run.finished.reason, FinishReason::Cancelled);
        assert_eq!(run.finished.prefilled, 400);
        assert_eq!(run.session.tokens(), &prompt[..400]);
        assert_eq!(run.session.checkpoints().len(), 1);
        assert!(text(&run.events).is_empty());
    }

    #[test]
    fn cancel_during_decode_keeps_exactly_the_shown_tokens() {
        struct CancelAfterTokens(Vec<Event>, CancelToken);
        impl EventSink for CancelAfterTokens {
            fn send(&mut self, event: Event) -> bool {
                self.0.push(event);
                if self.0.len() == 3 {
                    self.1.cancel();
                }
                true
            }
        }
        let tokenizer = byte_tokenizer();
        let req = request(ids("q"), 50);
        let mut sink = CancelAfterTokens(Vec::new(), req.cancel.clone());
        let mut engine = ScriptedEngine::new(ids("abcdef"), 1_000);
        let mut session = Session::default();
        let finished = generate_into(&mut engine, &mut session, &tokenizer, req, &mut sink);
        assert_eq!(finished.reason, FinishReason::Cancelled);
        assert_eq!(text(&sink.0), "ab");
        assert_eq!(session.committed(), ids("qab"));
        // "b" was only fed as a pending pipelined step.
        assert_eq!(session.tokens(), ids("qa"));
    }

    #[test]
    fn consumer_gone_cancels() {
        struct Gone;
        impl EventSink for Gone {
            fn send(&mut self, _: Event) -> bool {
                false
            }
        }
        let tokenizer = byte_tokenizer();
        let finished = generate_into(
            &mut ScriptedEngine::new(ids("abc"), 100),
            &mut Session::default(),
            &tokenizer,
            request(ids("q"), 10),
            &mut Gone,
        );
        assert_eq!(finished.reason, FinishReason::Cancelled);
    }

    #[test]
    fn needs_memory_without_cache_to_evict_fails_and_resets() {
        let first = run_with(
            ScriptedEngine::new(ids("ok"), 10_000).with_state_capacity_bytes(300 * KV_BYTES),
            Session::default(),
            request(ids("abc"), 5),
        );
        assert_eq!(first.finished.reason, FinishReason::Stop { stop: None });
        let failed = run_with(
            first.engine,
            first.session,
            request(ids("abcdefghijklmnop"), 1_000),
        );
        assert!(
            matches!(&failed.finished.reason, FinishReason::Error(message) if message.contains("bytes")),
            "{:?}",
            failed.finished.reason
        );
        assert_eq!(failed.finished.prefilled, 0);
        assert!(failed.session.tokens().is_empty());
        assert_eq!(failed.engine.allocated(), 0);
    }

    #[test]
    fn needs_memory_evicts_the_older_session_checkpoint() {
        // Room for the live state plus two checkpoints, not three.
        let mut engine = ScriptedEngine::new_hybrid(Vec::new(), 1_000, RB)
            .with_state_capacity_bytes(1_000 + 3 * RB);
        let mut session = Session::default();
        let tokenizer = byte_tokenizer();
        let mut prompt = Vec::new();
        for turn in 0..3 {
            let start = prompt.len() as u32 + 3;
            prompt.extend(ids("user"));
            prompt.extend(ids("|reply"));
            let mut req = request(prompt.clone(), 1);
            req.generation_start = Some(start);
            let finished =
                generate_into(&mut engine, &mut session, &tokenizer, req, &mut Vec::new());
            assert_eq!(
                finished.reason,
                FinishReason::Stop { stop: None },
                "turn {turn}"
            );
            assert!(session.checkpoints().len() <= 2);
        }
        assert_eq!(engine.checkpoint_count(), 2);
        let positions: Vec<_> = session.checkpoints().iter().map(|c| c.position).collect();
        assert_eq!(positions, [13, 23]);
    }

    #[test]
    fn hybrid_stop_sequence_falls_back_to_the_generation_checkpoint() {
        let prompt = ids("<sys><user>hi<asst>");
        let start = 15;
        let mut req = request(prompt.clone(), 50);
        req.stop = vec!["</done>".into()];
        req.generation_start = Some(start);
        let run = run_with(
            ScriptedEngine::new_hybrid(ids("answer</done>"), 1_000, RB),
            Session::default(),
            req,
        );
        assert_eq!(text(&run.events), "answer");
        assert_eq!(run.session.committed(), concat(&[&prompt, &ids("answer")]));
        assert_eq!(run.session.tokens(), &prompt[..start as usize]);
    }

    #[test]
    fn hybrid_next_turn_resumes_from_the_deepest_checkpoint() {
        let first_prompt = ids("<user>hi<asst>");
        let mut req = request(first_prompt.clone(), 50);
        req.generation_start = Some(8);
        let first = run_with(
            ScriptedEngine::new_hybrid(ids("hello there"), 1_000, RB),
            Session::default(),
            req,
        );
        assert_eq!(first.session.tokens(), ids("<user>hi<asst>hello there"));

        // The template re-renders the reply differently: divergence inside it.
        let prompt = ids("<user>hi<asst>hello world<user>more<asst>");
        let mut req = request(prompt.clone(), 1);
        req.generation_start = Some(prompt.len() as u32 - 6);
        let mut engine = first.engine;
        engine.processed.clear();
        let second = run_with(engine, first.session, req);
        assert_eq!(
            second.events[0],
            Event::Started {
                prompt: prompt.len() as u32,
                cached: 8,
                max_tokens: 1
            }
        );
        assert!(second.engine.processed.starts_with(&prompt[8..]));
        assert_eq!(second.finished.reason, FinishReason::Stop { stop: None });
        let positions: Vec<_> = second
            .session
            .checkpoints()
            .iter()
            .map(|c| c.position)
            .collect();
        assert_eq!(positions, [8, prompt.len() as u32 - 6]);
    }

    #[test]
    fn yield_resumes_prefill_with_identical_chunks() {
        let prompt = ids(&"y".repeat(500));
        let chunks = |yield_after: Option<u64>| {
            let mut engine = ScriptedEngine::new_hybrid(Vec::new(), 1_000, RB);
            engine.yield_after_layers = yield_after;
            let mut req = request(prompt.clone(), 1);
            req.generation_start = Some(200);
            let run = run_with(engine, Session::default(), req);
            assert_eq!(run.finished.reason, FinishReason::Stop { stop: None });
            run.engine.chunks
        };
        let straight = chunks(None);
        assert_eq!(straight, [(0, 200), (200, 499)]);
        assert_eq!(chunks(Some(2)), straight);
    }
}
