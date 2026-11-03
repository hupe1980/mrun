//! Reply splicing v2 end to end (003 FR-016, D051, SC-003 on `ScriptedEngine`): the real Qwen3
//! chat template and a BPE tokenizer (sampled ids differ from re-tokenization), the prefix
//! store holding the reply records, and the generation loop. Tool loops always hit; a new user
//! message renders earlier turns with the template and reports `diverged` there; a subagent
//! loop between two main turns leaves main's records on main's branch.

use mrun::api::model::{
    ApiKind, AssistantPart, ChatRequest, Message, ReasoningCtl, ToolDef, ToolResult,
};
use mrun::cache::miss::{Cause, MissReport};
use mrun::cache::{CacheMode, PrefixStore};
use mrun::chat::prompt::{RenderedPrompt, Role};
use mrun::chat::render::{RenderContext, render};
use mrun::chat::splice::{self, Reply, ReplyParts};
use mrun::chat::template::ChatTemplate;
use mrun::chat::tokenizer::Tokenizer;
use mrun::chat::tokenizer::test_support::{bpe_tokenizer, sampled_ids};
use mrun::runtime::generate::{Event, GenerateRequest, generate};
use mrun::runtime::pressure::PressureGauge;
use mrun::runtime::scripted::{SCRIPT_EOS, ScriptedEngine};
use mrun_engine::SyntaxId;
use serde_json::json;
use std::sync::PoisonError;

/// The real Qwen3-8B template (T007 fixture, pinned revision in its header).
const QWEN3: &str = include_str!("fixtures/qwen3-8b/chat_template.jinja");

/// One reply the model writes: its parts as the client sees them.
#[derive(Clone)]
struct Said {
    reasoning: String,
    text: String,
    /// `(name, arguments)`.
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

    /// What the model generates (Qwen3 Hermes syntax); thinking off adds no think block (the
    /// generation header holds the empty one).
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

    /// The assistant message a client sends back.
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

struct Server {
    template: ChatTemplate,
    tokenizer: Tokenizer,
    engine: ScriptedEngine,
    store: PrefixStore,
}

/// What one turn reported.
struct Turn {
    prompt: RenderedPrompt,
    generated: Vec<u32>,
    miss: MissReport,
    spliced: u32,
}

impl Server {
    fn new(replies: &[Said]) -> Self {
        let tokenizer = bpe_tokenizer();
        let mut script = Vec::new();
        for said in replies {
            script.extend(sampled_ids(&tokenizer, &said.output()));
            script.push(SCRIPT_EOS);
        }
        Self {
            template: ChatTemplate::new(QWEN3.into()).unwrap(),
            engine: ScriptedEngine::new(script, 200_000),
            store: PrefixStore::new(CacheMode::Cached),
            tokenizer,
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
        let spliced = splice::splice(
            &ledger.lock().unwrap_or_else(PoisonError::into_inner),
            &mut prompt,
            &rendered.replies,
            &request.messages,
            &request.tools,
        );
        let mut events = Vec::new();
        let finished = generate(
            &mut self.engine,
            &mut self.store,
            &mut self.tokenizer.stream(),
            GenerateRequest::new(prompt.clone(), 500),
            &mut events,
            &PressureGauge::default(),
        );
        let generated: Vec<u32> = events
            .iter()
            .filter_map(|e| match e {
                Event::Token { id, .. } => Some(*id),
                _ => None,
            })
            .collect();
        assert_eq!(
            self.tokenizer.inner().decode(&generated, false).unwrap(),
            said.output()
        );
        ledger
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .record(Reply {
                prompt: prompt.tokens.clone(),
                generation_start: prompt.generation_start,
                generated: generated.clone(),
                eos: SCRIPT_EOS,
                canonical: said.parts().canonical(&request.tools),
            });
        Turn {
            prompt,
            generated,
            miss: finished.stats.miss.unwrap(),
            spliced: spliced.messages,
        }
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

/// A 20-call tool loop, an answer, and a new user message.
fn loop_then_new_message(thinking: bool, drop_reasoning: bool) {
    let think = |t: &str| {
        if thinking {
            t.to_string()
        } else {
            String::new()
        }
    };
    let mut replies: Vec<Said> = (0..20)
        .map(|i| Said::call(&think(&format!("Step {i}.")), &format!("{i}.rs")))
        .collect();
    replies.push(Said::answer(&think("Done."), "All files are fine."));
    replies.push(Said::answer(&think("Now c."), "c.rs is fine too."));
    let mut server = Server::new(&replies);
    let mut request = request("You are a coding agent.", "check the files", thinking);
    let mut previous: Option<Turn> = None;
    for (i, said) in replies[..21].iter().enumerate() {
        let turn = server.turn(&request, said);
        if let Some(before) = &previous {
            // Tool loops always hit: everything the previous turn processed and generated.
            assert_eq!(turn.miss.cause, None, "turn {i}: {:?}", turn.miss);
            let reused = before.prompt.len() + before.generated.len() as u32;
            assert_eq!(turn.miss.cached, reused, "turn {i}");
            assert_eq!(turn.miss.unreused, 0, "turn {i}");
            assert_eq!(turn.spliced, i as u32, "turn {i}");
        }
        request.messages.push(said.message(drop_reasoning));
        if said.call.is_some() {
            request.messages.push(result("ok"));
        }
        previous = Some(turn);
    }
    // A new user message: earlier turns get the template's rendering. With thinking on Qwen3
    // drops their reasoning, with thinking off their empty think blocks: either way the first
    // assistant message re-renders differently from what the model processed.
    request.messages.push(Message::user_text("and c.rs?"));
    let turn = server.turn(&request, &replies[21]);
    assert_eq!(turn.spliced, 0);
    assert_eq!(turn.miss.cause, Some(Cause::Diverged), "{:?}", turn.miss);
    let at = turn.miss.message.unwrap();
    assert_eq!((at.index, at.role), (2, Role::Assistant), "{:?}", turn.miss);
    let first_reply = turn.prompt.messages[2].start;
    assert!(turn.miss.cached >= first_reply, "{:?}", turn.miss);
}

#[test]
fn qwen3_tool_loop_thinking_on() {
    loop_then_new_message(true, false);
}

#[test]
fn qwen3_tool_loop_thinking_off() {
    loop_then_new_message(false, false);
}

#[test]
fn qwen3_tool_loop_client_drops_reasoning() {
    loop_then_new_message(true, true);
}

#[test]
fn a_thirty_reply_subagent_loop_leaves_main_spliced() {
    // The v1 ledger kept 8 replies per server; records now live with main's branch.
    let mut replies = vec![Said::call("Look at main.rs.", "main.rs")];
    replies.extend((0..30).map(|i| Said::call(&format!("Sub {i}."), &format!("s{i}.rs"))));
    replies.push(Said::answer("Read it.", "main.rs is fine."));
    let mut server = Server::new(&replies);
    let task = "x".repeat(400);
    let mut main = request("You are a coding agent.", &task, true);
    let first = server.turn(&main, &replies[0]);
    main.messages.push(replies[0].message(false));
    main.messages.push(result("fn main() {}"));
    // A subagent with its own system prompt runs a 30-call loop.
    let mut sub = request("You are a search subagent.", "find things", true);
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
    assert!(server.store.stats().records > 0);
}

#[test]
fn a_dropped_path_takes_its_records() {
    let replies = [Said::call("Look.", "a.rs"), Said::call("Again.", "b.rs")];
    let mut server = Server::new(&replies);
    let mut request = request("You are a coding agent.", &"y".repeat(400), true);
    server.turn(&request, &replies[0]);
    request.messages.push(replies[0].message(false));
    request.messages.push(result("ok"));
    let ledger = server.store.ledger();
    assert_eq!(ledger.lock().unwrap().len(), 1);
    // Memory pressure drops the live state of an idle server: the record goes with it.
    server.store.on_pressure(
        &mut server.engine,
        mrun::runtime::pressure::Pressure::Critical,
        true,
    );
    assert_eq!(ledger.lock().unwrap().len(), 0);
    assert_eq!(server.store.stats().records, 0);
    let turn = server.turn(&request, &replies[1]);
    assert_eq!(turn.spliced, 0, "nothing to splice from");
}
