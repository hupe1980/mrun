//! SC-003 (003 T020, FR-016, D051) with the real Qwen3-8B chat template and BPE tokenizer
//! (T007, T041; never a byte tokenizer or a hand-written template): 20-turn tool loops with
//! thinking on, off, and with a client that drops reasoning, then a new user message; a 30-reply
//! subagent loop between two main turns. Every loop turn's cached tokens are the previous prompt
//! plus its generated ids; the turn after the new user message is `diverged` at the first
//! re-rendered assistant message. On `ScriptedEngine` and on `NuvecEngine` with a tiny `qwen3`
//! using the real vocabulary, its replies produced through the real decode path by a one-token
//! mask (T009); on `NuvecEngine` every turn is also checked by the SC-005 oracle. `cache.splice`
//! (FR-020) counts seen and spliced messages per client.
//!
//! Skips with a message when Qwen3-8B-4bit's `tokenizer.json` is not in the Hugging Face cache.

#[path = "fixtures/support.rs"]
mod support;

use mrun::api::model::{
    ApiKind, AssistantPart, ChatRequest, Message, ReasoningCtl, ToolDef, ToolResult,
};
use mrun::cache::miss::{Cause, MissReport};
use mrun::cache::{CacheMode, PrefixStore};
use mrun::chat::prompt::{RenderedPrompt, Role};
use mrun::chat::render::{RenderContext, render};
use mrun::chat::splice::{self, ClientSplices, Reply, ReplyParts, client_class};
use mrun::chat::template::ChatTemplate;
use mrun::chat::tokenizer::Tokenizer;
use mrun::chat::tokenizer::test_support::sampled_ids;
use mrun::runtime::generate::{DecodeHook, Event, FinishReason, GenerateRequest, generate};
use mrun::runtime::pressure::PressureGauge;
use mrun::runtime::scripted::{SCRIPT_EOS, ScriptedEngine};
use mrun_engine::{Engine, SyntaxId};
use serde_json::json;
use std::sync::PoisonError;

/// One reply the model writes: its parts as the client sees them.
#[derive(Clone)]
struct Said {
    reasoning: String,
    text: String,
    call: Option<(String, String)>,
}

impl Said {
    fn call(reasoning: &str, path: &str) -> Self {
        Self {
            reasoning: reasoning.into(),
            text: String::new(),
            call: Some(("read".into(), format!("{{\"path\": \"{path}\"}}"))),
        }
    }

    fn answer(reasoning: &str, text: &str) -> Self {
        Self {
            reasoning: reasoning.into(),
            text: text.into(),
            call: None,
        }
    }

    /// What the model generates (Qwen3 Hermes syntax).
    fn output(&self) -> String {
        let mut out = String::new();
        if !self.reasoning.is_empty() {
            out.push_str(&format!("<think>\n{}\n</think>\n\n", self.reasoning));
        }
        out.push_str(&self.text);
        if let Some((name, args)) = &self.call {
            if !self.text.is_empty() {
                out.push('\n');
            }
            out.push_str(&format!(
                "<tool_call>\n{{\"name\": \"{name}\", \"arguments\": {args}}}\n</tool_call>"
            ));
        }
        out
    }

    fn parts(&self) -> ReplyParts {
        ReplyParts {
            reasoning: self.reasoning.clone(),
            text: self.text.clone(),
            calls: self.call.iter().cloned().collect(),
        }
    }

    fn message(&self, drop_reasoning: bool) -> Message {
        let mut parts = Vec::new();
        if !self.reasoning.is_empty() && !drop_reasoning {
            parts.push(AssistantPart::Reasoning(self.reasoning.clone()));
        }
        if !self.text.is_empty() {
            parts.push(AssistantPart::Text(self.text.clone()));
        }
        if let Some((name, args)) = &self.call {
            parts.push(AssistantPart::ToolCall {
                id: "call_0".into(),
                name: name.clone(),
                raw_args: args.clone(),
            });
        }
        Message::Assistant(parts)
    }
}

/// How a server produces a reply on its engine.
trait Backend {
    type E: Engine;
    fn engine(&mut self) -> &mut Self::E;
    /// The EOS that ends a reply and the decode hook producing `script`.
    fn reply(&mut self, script: &[u32]) -> (u32, Option<Box<dyn DecodeHook>>);
    /// After a turn: checks it against the oracle when the backend can.
    fn check(&mut self, _prompt: &RenderedPrompt, _store: &PrefixStore, _resume: u32, _what: &str) {
    }
}

/// `ScriptedEngine`: the whole script is set up front.
struct Scripted(ScriptedEngine);

impl Backend for Scripted {
    type E = ScriptedEngine;
    fn engine(&mut self) -> &mut ScriptedEngine {
        &mut self.0
    }
    fn reply(&mut self, _script: &[u32]) -> (u32, Option<Box<dyn DecodeHook>>) {
        (SCRIPT_EOS, None)
    }
}

struct Server<B: Backend> {
    template: ChatTemplate,
    tokenizer: Tokenizer,
    backend: B,
    store: PrefixStore,
    client: String,
    checked: usize,
}

/// What one turn reported.
struct Turn {
    prompt: RenderedPrompt,
    generated: Vec<u32>,
    miss: MissReport,
    spliced: u32,
}

/// The ids the model "samples" for a reply: not its re-tokenization (D051).
fn script(tokenizer: &Tokenizer, said: &Said) -> Vec<u32> {
    sampled_ids(tokenizer, &said.output())
}

impl<B: Backend> Server<B> {
    fn new(tokenizer: Tokenizer, backend: B) -> Self {
        Self {
            template: ChatTemplate::new(support::QWEN3_TEMPLATE.into()).unwrap(),
            tokenizer,
            backend,
            store: PrefixStore::new(CacheMode::Cached),
            client: client_class(Some("claude-cli/2.1.3 (external, cli)")),
            checked: 0,
        }
    }

    /// Renders, splices, generates and records like `mrun serve` does.
    fn turn(&mut self, request: &ChatRequest, said: &Said) -> Turn {
        let rendered = render(
            request,
            &RenderContext {
                template: &self.template,
                tokenizer: &self.tokenizer,
                syntax: SyntaxId::Hermes,
                supports_thinking: true,
            },
        )
        .unwrap();
        let ledger = self.store.ledger();
        let mut prompt = rendered.prompt;
        let spliced = {
            let mut ledger = ledger.lock().unwrap_or_else(PoisonError::into_inner);
            let spliced = splice::splice(
                &ledger,
                &mut prompt,
                &rendered.replies,
                &request.messages,
                &request.tools,
            );
            ledger.note_spliced(spliced.tokens);
            ledger.note_client(&self.client, &spliced);
            spliced
        };
        if std::env::var_os("SPLICE_DEBUG").is_some() {
            let live = self.store.live_tokens();
            eprintln!(
                "before: prompt {} live {} branches {:?} tail prompt {:?} live {:?}",
                prompt.len(),
                live.len(),
                self.store.branches(),
                &prompt.tokens[prompt.len().saturating_sub(8) as usize..],
                &live[live.len().saturating_sub(8)..]
            );
        }
        let ids = script(&self.tokenizer, said);
        let (eos, hook) = self.backend.reply(&ids);
        let mut events = Vec::new();
        let mut generate_request = GenerateRequest::new(prompt.clone(), 500);
        generate_request.hook = hook;
        let finished = generate(
            self.backend.engine(),
            &mut self.store,
            &mut self.tokenizer.stream(),
            generate_request,
            &mut events,
            &PressureGauge::default(),
        );
        assert_eq!(
            finished.reason,
            FinishReason::Stop {
                stop: None,
                eos_token: Some(eos)
            }
        );
        let generated: Vec<u32> = events
            .iter()
            .filter_map(|e| match e {
                Event::Token { id, .. } => Some(*id),
                _ => None,
            })
            .collect();
        assert_eq!(generated, ids, "the scripted reply");
        self.store.assert_invariants(self.backend.engine());
        let resume = finished.stats.cached_tokens;
        self.backend.check(
            &prompt,
            &self.store,
            resume,
            &format!("turn {}", self.checked),
        );
        self.checked += 1;
        ledger
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .record(Reply {
                prompt: prompt.tokens.clone(),
                generation_start: prompt.generation_start,
                generated: generated.clone(),
                eos,
                canonical: said.parts().canonical(&request.tools),
            });
        Turn {
            prompt,
            generated,
            miss: finished.stats.miss.unwrap(),
            spliced: spliced.messages,
        }
    }

    fn splice_counts(&self) -> ClientSplices {
        let ledger = self.store.ledger();
        let ledger = ledger.lock().unwrap_or_else(PoisonError::into_inner);
        ledger.clients()[&self.client]
    }
}

fn request(system: &str, task: &str, thinking: bool) -> ChatRequest {
    let mut request = ChatRequest::new(ApiKind::Messages);
    request.system = vec![system.into()];
    request.tools = vec![ToolDef {
        name: "read".into(),
        description: "Read a file".into(),
        schema: json!({"type": "object", "properties": {"path": {"type": "string"}}}),
        ..Default::default()
    }];
    if thinking {
        request.reasoning = ReasoningCtl::On { budget: None };
    }
    request.messages = vec![Message::user_text(task)];
    request
}

fn result(content: &str) -> Message {
    Message::Tool(ToolResult {
        call_id: "call_0".into(),
        content: content.into(),
        is_error: false,
        images_omitted: 0,
    })
}

/// The replies of a 20-call tool loop, its answer, and the answer to a new user message.
fn loop_replies(thinking: bool) -> Vec<Said> {
    let think = |t: &str| {
        if thinking {
            t.to_string()
        } else {
            String::new()
        }
    };
    let mut replies: Vec<Said> = (0..20)
        .map(|i| {
            Said::call(
                &think(&format!("Step {i}: read the next file.")),
                &format!("src/{i}.rs"),
            )
        })
        .collect();
    replies.push(Said::answer(&think("Done."), "All files are fine."));
    replies.push(Said::answer(&think("Now c."), "c.rs is fine too."));
    replies
}

/// A 20-call tool loop, an answer, and a new user message (SC-003).
fn loop_then_new_message<B: Backend>(server: &mut Server<B>, thinking: bool, drop_reasoning: bool) {
    let replies = loop_replies(thinking);
    let mut request = request(
        "You are a coding agent.",
        "Check the files in src/.",
        thinking,
    );
    let mut previous: Option<Turn> = None;
    let before = server.splice_counts_or_default();
    for (i, said) in replies[..21].iter().enumerate() {
        let turn = server.turn(&request, said);
        if let Some(before) = &previous {
            assert_eq!(
                turn.miss.cause, None,
                "thinking {thinking}, drop {drop_reasoning}, turn {i}: {:?}",
                turn.miss
            );
            let reused = before.prompt.len() + before.generated.len() as u32;
            assert_eq!(
                turn.miss.cached, reused,
                "turn {i}: previous prompt + generated ids"
            );
            assert_eq!(turn.miss.unreused, 0, "turn {i}");
            assert_eq!(turn.spliced, i as u32, "turn {i}");
        }
        request.messages.push(said.message(drop_reasoning));
        if said.call.is_some() {
            request.messages.push(result("fn main() {}"));
        }
        previous = Some(turn);
    }
    // A new user message: the template re-renders earlier turns, so the first assistant message
    // diverges from what the model processed.
    request.messages.push(Message::user_text("And c.rs?"));
    let turn = server.turn(&request, &replies[21]);
    assert_eq!(turn.spliced, 0);
    assert_eq!(turn.miss.cause, Some(Cause::Diverged), "{:?}", turn.miss);
    let at = turn.miss.message.unwrap();
    assert_eq!((at.index, at.role), (2, Role::Assistant), "{:?}", turn.miss);
    let first_reply = turn.prompt.messages[2].start;
    assert!(turn.miss.cached >= first_reply, "{:?}", turn.miss);
    // FR-020: every loop turn's assistant messages were seen, and all of them spliced.
    let after = server.splice_counts_or_default();
    let seen: u64 = (0..21).sum();
    assert_eq!(after.seen - before.seen, seen);
    assert_eq!(
        after.spliced - before.spliced,
        seen,
        "a 100 % splice hit rate"
    );
}

/// Main, a 30-reply subagent loop, main again: main stays spliced (records move with its branch).
fn subagent_loop_between_main_turns<B: Backend>(server: &mut Server<B>) {
    let mut replies = vec![Said::call("Look at main.rs.", "main.rs")];
    replies.extend((0..30).map(|i| Said::call(&format!("Sub {i}."), &format!("s{i}.rs"))));
    replies.push(Said::answer("Read it.", "main.rs is fine."));
    let task = "Review the crate and report what main.rs does. ".repeat(40);
    let mut main = request("You are a coding agent.", &task, true);
    let first = server.turn(&main, &replies[0]);
    main.messages.push(replies[0].message(false));
    main.messages.push(result("fn main() {}"));
    let mut sub = request("You are a search subagent.", "Find things.", true);
    for said in &replies[1..31] {
        let turn = server.turn(&sub, said);
        sub.messages.push(said.message(false));
        sub.messages.push(result("found"));
        assert_ne!(turn.miss.cause, Some(Cause::Evicted));
    }
    let back = server.turn(&main, &replies[31]);
    assert_eq!(back.spliced, 1, "main's record moved with its branch");
    assert_eq!(back.miss.cause, None, "{:?}", back.miss);
    assert_eq!(
        back.miss.cached,
        first.prompt.len() + first.generated.len() as u32
    );
}

impl<B: Backend> Server<B> {
    fn splice_counts_or_default(&self) -> ClientSplices {
        let ledger = self.store.ledger();
        let ledger = ledger.lock().unwrap_or_else(PoisonError::into_inner);
        ledger
            .clients()
            .get(&self.client)
            .copied()
            .unwrap_or_default()
    }
}

/// A `ScriptedEngine` server whose script holds `replies` in order.
fn scripted(tokenizer: Tokenizer, replies: &[Said]) -> Server<Scripted> {
    let mut all = Vec::new();
    for said in replies {
        all.extend(script(&tokenizer, said));
        all.push(SCRIPT_EOS);
    }
    let engine = ScriptedEngine::new(all, 200_000).with_vocab(tokenizer.vocab_size());
    Server::new(tokenizer, Scripted(engine))
}

#[test]
fn scripted_tool_loops_hit_with_the_real_tokenizer() {
    let Some(tokenizer) = support::qwen3_tokenizer() else {
        return;
    };
    for (thinking, drop) in [(true, false), (false, false), (true, true)] {
        let mut server = scripted(tokenizer.clone(), &loop_replies(thinking));
        loop_then_new_message(&mut server, thinking, drop);
        let counts = server.splice_counts();
        assert!(counts.seen > 0 && counts.spliced == counts.seen);
    }
}

#[test]
fn scripted_subagent_loop_leaves_main_spliced_with_the_real_tokenizer() {
    let Some(tokenizer) = support::qwen3_tokenizer() else {
        return;
    };
    let mut replies = vec![Said::call("Look at main.rs.", "main.rs")];
    replies.extend((0..30).map(|i| Said::call(&format!("Sub {i}."), &format!("s{i}.rs"))));
    replies.push(Said::answer("Read it.", "main.rs is fine."));
    let mut server = scripted(tokenizer, &replies);
    subagent_loop_between_main_turns(&mut server);
}

#[cfg(feature = "engine")]
mod engine {
    use super::*;
    use mrun_engine::NuvecEngine;
    use mrun_engine::testing::QWEN3_VOCAB;
    use support::{IM_END, Recorder, ScriptHook, Served, oracle, tiny};

    /// A tiny `qwen3` with the real vocabulary; every turn checked by the oracle.
    struct Tiny {
        engine: Recorder<NuvecEngine>,
        fresh: NuvecEngine,
        _dir: Option<tempfile::TempDir>,
        script: Vec<u32>,
    }

    impl Tiny {
        fn new() -> Self {
            let tiny::Tiny { engine, dir } = tiny::qwen3(QWEN3_VOCAB, None);
            let fresh = mrun_engine::testing::load_cpu(dir.as_ref().unwrap().path(), None).unwrap();
            Self {
                engine: Recorder::new(engine),
                fresh,
                _dir: dir,
                script: Vec::new(),
            }
        }
    }

    impl Backend for Tiny {
        type E = Recorder<NuvecEngine>;
        fn engine(&mut self) -> &mut Recorder<NuvecEngine> {
            &mut self.engine
        }
        fn reply(&mut self, script: &[u32]) -> (u32, Option<Box<dyn DecodeHook>>) {
            self.script = script.to_vec();
            let vocab = self.engine.info().vocab;
            (
                IM_END,
                Some(Box::new(ScriptHook::new(script.to_vec(), IM_END, vocab))),
            )
        }
        fn check(&mut self, prompt: &RenderedPrompt, store: &PrefixStore, resume: u32, what: &str) {
            let (rows, sampled) = self.engine.take();
            let served = Served {
                prompt: prompt.clone(),
                resume,
                schedule: store.schedule().to_vec(),
                live: store.live_tokens().to_vec(),
                rows,
                sampled,
                script: Some((self.script.clone(), IM_END)),
            };
            assert!(oracle(&mut self.fresh, &served, what), "{what}: checked");
        }
    }

    #[test]
    fn tiny_qwen3_tool_loops_hit_through_the_real_decode_path() {
        let Some(tokenizer) = support::qwen3_tokenizer() else {
            return;
        };
        // One server per loop: the loops share their prefix, and a later loop would reuse an
        // earlier one's state beyond its own previous turn.
        let mut checked = 0;
        for (thinking, drop) in [(true, false), (false, false), (true, true)] {
            let mut server = Server::new(tokenizer.clone(), Tiny::new());
            loop_then_new_message(&mut server, thinking, drop);
            checked += server.checked;
        }
        assert_eq!(checked, 66);
        eprintln!("{checked} turns bit-identical to recomputation");
    }

    #[test]
    fn tiny_qwen3_subagent_loop_leaves_main_spliced() {
        let Some(tokenizer) = support::qwen3_tokenizer() else {
            return;
        };
        let mut server = Server::new(tokenizer, Tiny::new());
        subagent_loop_between_main_turns(&mut server);
    }
}
