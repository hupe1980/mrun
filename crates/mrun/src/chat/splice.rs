//! Token-in/token-out reply splicing v2 (D051, 003 FR-016).
//!
//! Chat templates re-render an earlier assistant turn differently from what the model generated
//! (Qwen3 drops empty think blocks and earlier reasoning, strips whitespace; BPE re-tokenization
//! differs from sampled ids), so the next request would miss the cache at the start of that
//! reply. A finished reply is recorded on the cache path that holds its tokens (the prefix
//! store's trunk or a branch) as its generation start and end positions and its canonical form;
//! the ids are the path's own tokens, and the record lives exactly as long as they do.
//!
//! Only the current turn (the assistant messages after the last user message) is spliced, oldest
//! first: a message is replaced by the path's tokens from its generation header to the reply's
//! end when the request's tokens before it equal the path's, a record starts there and the
//! canonical forms agree (a message without reasoning takes the record's, which restores it).
//! A message that does not splice keeps the template's rendering and the walk goes on: a later
//! record splices when its path holds exactly the tokens before it (a path re-prefilled from the
//! template after its records were lost, the Claude Code case). Earlier turns keep the template's
//! rendering, which is what the model was trained on. Only tokens the model itself produced after
//! exactly that prefix are used, so the result is exact.
//!
//! The [`Ledger`] is shared between the api layer (which splices while rendering) and the prefix
//! store (which owns the records): the store publishes the paths that carry records and adopts
//! replies recorded since its last request.

use crate::api::model::{AssistantPart, Message, ToolDef};
use crate::chat::prompt::RenderedPrompt;
use crate::chat::render::ReplySpan;
use serde_json::Value;
use std::collections::BTreeMap;
use std::sync::Arc;

/// Replies recorded but not yet placed on a path, at most (the store adopts them at its next
/// request).
const PENDING: usize = 64;

/// A reply's canonical form: reasoning, and text with tool calls.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Canonical {
    reasoning: Arc<str>,
    rest: Arc<str>,
}

impl Canonical {
    /// A message the client sent back (`sent`) is this reply: equal forms, or the same text and
    /// calls with the reasoning omitted (restored from the record, the model saw it).
    pub fn accepts(&self, sent: &Canonical) -> bool {
        self.rest == sent.rest && (sent.reasoning.is_empty() || self.reasoning == sent.reasoning)
    }
}

/// A recorded reply on a path: `tokens[start..end]` of that path are its generation header and
/// generated ids; the EOS that ended it is not part of the path.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Record {
    pub start: u32,
    pub end: u32,
    pub eos: u32,
    pub canonical: Canonical,
}

/// A path holding records, as the prefix store published it.
#[derive(Clone, Debug)]
pub struct HeldPath {
    pub tokens: Arc<Vec<u32>>,
    pub records: Vec<Record>,
}

/// One finished reply, as the api layer reports it.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Reply {
    /// The prompt it answered, up to and including the generation header.
    pub prompt: Vec<u32>,
    /// Start of the generation header in `prompt`.
    pub generation_start: u32,
    /// Generated ids, without the ending EOS.
    pub generated: Vec<u32>,
    /// The EOS token that ended it.
    pub eos: u32,
    pub canonical: Canonical,
}

/// What [`splice`] did.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct Spliced {
    /// Tokens taken from earlier replies (their generation headers and generated ids).
    pub tokens: u32,
    /// Assistant messages spliced.
    pub messages: u32,
    /// Assistant messages of the current turn (spliceable ones), spliced or not (FR-020).
    pub seen: u32,
}

/// Current-turn assistant messages one client sent and how many were spliced (FR-020).
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, serde::Serialize)]
pub struct ClientSplices {
    pub seen: u64,
    pub spliced: u64,
}

/// The client class of a `User-Agent`: its first product token's name, lowercased (`claude-cli`
/// for `claude-cli/2.1.3 (external, cli)`), else `other`. Only the name is kept, no version or
/// comment.
pub fn client_class(user_agent: Option<&str>) -> String {
    let name = user_agent
        .and_then(|ua| ua.split_whitespace().next())
        .and_then(|product| product.split('/').next())
        .map(str::trim)
        .filter(|name| {
            !name.is_empty()
                && name.len() <= 64
                && name
                    .chars()
                    .all(|c| c.is_ascii_alphanumeric() || "-_.".contains(c))
        });
    name.map_or_else(|| "other".into(), str::to_ascii_lowercase)
}

/// Reply records: published by the prefix store, read while rendering.
#[derive(Debug, Default)]
pub struct Ledger {
    paths: Vec<HeldPath>,
    /// Replies recorded since the store last adopted: `(prompt + generated, record)`.
    pending: Vec<(Vec<u32>, Record)>,
    /// Tokens spliced into generated prompts since start.
    spliced_total: u64,
    /// Per client class (FR-020), for generated requests.
    clients: BTreeMap<String, ClientSplices>,
}

impl Ledger {
    /// Records a finished reply; the store places it on the path holding its tokens (or drops
    /// it when none does).
    pub fn record(&mut self, reply: Reply) {
        let mut tokens = reply.prompt;
        tokens.extend_from_slice(&reply.generated);
        let record = Record {
            start: reply.generation_start,
            end: tokens.len() as u32,
            eos: reply.eos,
            canonical: reply.canonical,
        };
        self.pending
            .retain(|(t, r)| !(*t == tokens && *r == record));
        if self.pending.len() == PENDING {
            self.pending.remove(0);
        }
        self.pending.push((tokens, record));
    }

    /// Replies recorded since the last call (the store adopts them).
    pub fn take_pending(&mut self) -> Vec<(Vec<u32>, Record)> {
        std::mem::take(&mut self.pending)
    }

    /// Replaces the published paths (the store's view after a change).
    pub fn publish(&mut self, paths: Vec<HeldPath>) {
        self.paths = paths;
    }

    /// Records on published paths plus pending ones.
    pub fn len(&self) -> usize {
        self.paths.iter().map(|p| p.records.len()).sum::<usize>() + self.pending.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Counts tokens spliced into a prompt that is generated from.
    pub fn note_spliced(&mut self, tokens: u32) {
        self.spliced_total += u64::from(tokens);
    }

    pub fn spliced_total(&self) -> u64 {
        self.spliced_total
    }

    /// A generated request's splice: its tokens (`spliced_tokens_total`) and its client's counts.
    pub fn note_turn(&mut self, client: &str, spliced: &Spliced) {
        self.note_spliced(spliced.tokens);
        self.note_client(client, spliced);
    }

    /// Counts one generated request's splice outcome for its client class (FR-020; a bounded
    /// number of classes, the rest under `other`).
    pub fn note_client(&mut self, client: &str, spliced: &Spliced) {
        const MAX_CLIENTS: usize = 32;
        let key = if self.clients.contains_key(client) || self.clients.len() < MAX_CLIENTS {
            client
        } else {
            "other"
        };
        let entry = self.clients.entry(key.to_string()).or_default();
        entry.seen += u64::from(spliced.seen);
        entry.spliced += u64::from(spliced.messages);
    }

    /// `/mrun/status` `cache.splice`: `{<client>: {seen, spliced}}`.
    pub fn clients(&self) -> &BTreeMap<String, ClientSplices> {
        &self.clients
    }

    /// Every path with records: published ones, then each pending reply as its own path.
    fn sources(&self) -> impl Iterator<Item = (&[u32], &[Record])> {
        self.paths
            .iter()
            .map(|p| (p.tokens.as_slice(), p.records.as_slice()))
            .chain(
                self.pending
                    .iter()
                    .map(|(tokens, record)| (tokens.as_slice(), std::slice::from_ref(record))),
            )
    }
}

/// A reply as the client sends it back: reasoning, text and calls `(name, arguments)`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ReplyParts {
    pub reasoning: String,
    pub text: String,
    pub calls: Vec<(String, String)>,
}

impl ReplyParts {
    pub fn of_message(parts: &[AssistantPart]) -> Self {
        let mut reply = Self::default();
        for part in parts {
            match part {
                AssistantPart::Text(text) => reply.text.push_str(text),
                AssistantPart::Reasoning(text) => reply.reasoning.push_str(text),
                AssistantPart::ToolCall { name, raw_args, .. } => {
                    reply.calls.push((name.clone(), raw_args.clone()));
                }
            }
        }
        reply
    }

    /// Insensitive to how a client re-serializes the reply (surrounding whitespace, JSON
    /// formatting and key order of arguments, top-level arguments equal to the default the
    /// tool's schema declares — Claude Code stores `Edit` calls with `"replace_all": false`
    /// added), sensitive to any other edit.
    pub fn canonical(&self, tools: &[ToolDef]) -> Canonical {
        let calls: Vec<Value> = self
            .calls
            .iter()
            .map(|(name, args)| {
                let args = without_defaults(args, tools.iter().find(|t| t.name == *name));
                Value::from(vec![name.clone(), canonical_json(&args)])
            })
            .collect();
        let rest = Value::from(vec![Value::from(self.text.trim()), Value::Array(calls)]);
        Canonical {
            reasoning: self.reasoning.trim().into(),
            rest: rest.to_string().into(),
        }
    }
}

/// `raw` without the top-level arguments whose value is the default `tool`'s schema declares
/// for them (a client may fill those in when it stores a call); other text is unchanged.
fn without_defaults<'a>(raw: &'a str, tool: Option<&ToolDef>) -> std::borrow::Cow<'a, str> {
    let Some(properties) = tool.and_then(|t| t.schema["properties"].as_object()) else {
        return raw.into();
    };
    if !properties.values().any(|p| p.get("default").is_some()) {
        return raw.into();
    }
    let Ok(Value::Object(mut args)) = serde_json::from_str::<Value>(raw) else {
        return raw.into();
    };
    args.retain(|key, value| properties.get(key).and_then(|p| p.get("default")) != Some(value));
    Value::Object(args).to_string().into()
}

/// `raw` as compact JSON with sorted keys and integral numbers without a fraction; text that is
/// not JSON stays as it is.
pub fn canonical_json(raw: &str) -> String {
    fn write(value: &Value, out: &mut String) {
        match value {
            Value::Object(map) => {
                let mut entries: Vec<(&String, &Value)> = map.iter().collect();
                entries.sort_by(|a, b| a.0.cmp(b.0));
                out.push('{');
                for (i, (key, item)) in entries.into_iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    out.push_str(&Value::from(key.as_str()).to_string());
                    out.push(':');
                    write(item, out);
                }
                out.push('}');
            }
            Value::Array(items) => {
                out.push('[');
                for (i, item) in items.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    write(item, out);
                }
                out.push(']');
            }
            Value::Number(n) => match n.as_f64() {
                Some(f) if n.is_f64() && f.fract() == 0.0 && f.abs() < 9.0e15 => {
                    out.push_str(&(f as i64).to_string());
                }
                _ => out.push_str(&n.to_string()),
            },
            other => out.push_str(&other.to_string()),
        }
    }
    match serde_json::from_str::<Value>(raw) {
        Ok(value) => {
            let mut out = String::new();
            write(&value, &mut out);
            out
        }
        Err(_) => raw.to_string(),
    }
}

/// Splices the current turn's assistant messages, oldest first, with the recorded tokens of a
/// path the request follows; moves the message marks with them. A message without a matching
/// record keeps its rendering.
pub fn splice(
    ledger: &Ledger,
    prompt: &mut RenderedPrompt,
    replies: &[ReplySpan],
    messages: &[Message],
    tools: &[ToolDef],
) -> Spliced {
    // The current turn: after the last user message (tool results are `Message::Tool`).
    let turn_start = messages
        .iter()
        .rposition(|m| matches!(m, Message::User(_)))
        .map_or(0, |u| u + 1);
    let seen = replies.iter().filter(|s| s.message >= turn_start).count() as u32;
    if ledger.is_empty() || seen == 0 {
        return Spliced {
            seen,
            ..Spliced::default()
        };
    }
    let tokens = &prompt.tokens;
    let sources: Vec<(&[u32], &[Record])> = ledger.sources().collect();
    let mut out: Vec<u32> = Vec::new();
    // Request tokens before `q` are in `out` (spliced).
    let mut q = 0usize;
    let mut cuts = Vec::new();
    let mut spliced = Spliced {
        seen,
        ..Spliced::default()
    };
    // `(source, n)`: that source's path starts with `out[..n]`.
    let mut verified: Option<(usize, usize)> = None;
    for span in replies.iter().filter(|s| s.message >= turn_start) {
        let (start, end_of_turn) = (span.start as usize, span.end_of_turn as usize);
        if start < q || end_of_turn < start || end_of_turn > tokens.len() {
            break;
        }
        let Some(Message::Assistant(parts)) = messages.get(span.message) else {
            break;
        };
        let form = ReplyParts::of_message(parts).canonical(tools);
        let at = out.len() + (start - q);
        let gap = &tokens[q..start];
        let found = sources.iter().enumerate().find_map(|(i, (path, records))| {
            let record = records
                .iter()
                .find(|r| r.start as usize == at && r.canonical.accepts(&form))?;
            if path.len() < record.end as usize {
                return None;
            }
            let done = match verified {
                Some((v, n)) if v == i && n == out.len() => true,
                _ => path[..out.len()] == out[..],
            };
            (done && path[out.len()..at] == *gap).then_some((i, record.end as usize))
        });
        // No record (lost with its path, or never made): the message keeps the template's
        // rendering and the walk goes on; a later record still needs a path holding exactly
        // that rendering before it.
        let Some((source, end)) = found else {
            continue;
        };
        let path = sources[source].0;
        out.extend_from_slice(gap);
        out.extend_from_slice(&path[at..end]);
        verified = Some((source, out.len()));
        cuts.push((span.start, span.end_of_turn, end as u32));
        spliced.tokens += (end - at) as u32;
        spliced.messages += 1;
        q = end_of_turn;
    }
    if cuts.is_empty() {
        return Spliced {
            seen,
            ..Spliced::default()
        };
    }
    out.extend_from_slice(&tokens[q..]);
    remap(prompt, &cuts, out);
    spliced
}

/// Moves the marks of `prompt` through `cuts` and makes `tokens` its tokens: a position after a
/// cut moves with it, one inside a replaced message is gone.
fn remap(prompt: &mut RenderedPrompt, cuts: &[(u32, u32, u32)], tokens: Vec<u32>) {
    let map = |pos: u32| -> Option<u32> {
        for &(start, end, spliced_end) in cuts.iter().rev() {
            if pos >= end {
                return Some(spliced_end + (pos - end));
            }
            if pos > start {
                return None;
            }
        }
        Some(pos)
    };
    prompt.messages = std::mem::take(&mut prompt.messages)
        .into_iter()
        .filter_map(|mut mark| {
            mark.start = map(mark.start)?;
            Some(mark)
        })
        .collect();
    prompt.generation_start = map(prompt.generation_start).unwrap_or(tokens.len() as u32);
    prompt.system_tools_end = map(prompt.system_tools_end).unwrap_or(prompt.system_tools_end);
    prompt.tokens = tokens;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::model::GenEvent;
    use crate::api::model::{ApiKind, ChatRequest, ReasoningCtl, ToolDef, ToolResult};
    use crate::chat::render::{RenderContext, Rendered, render};
    use crate::chat::syntax::registry::ChatSyntax;
    use crate::chat::syntax::{ParseEnd, ParseOut, ParserConfig, Specials};
    use crate::chat::template::ChatTemplate;
    use crate::chat::tokenizer::Tokenizer;
    use crate::chat::tokenizer::test_support::{bpe_tokenizer, sampled_ids};
    use mrun_engine::SyntaxId;
    use serde_json::json;
    use std::sync::Arc;

    const QWEN3: &str = include_str!("testdata/qwen3.jinja");

    /// The real Qwen3 template with a BPE tokenizer: sampled ids differ from re-tokenization.
    struct Model {
        template: ChatTemplate,
        tokenizer: Tokenizer,
        eos: u32,
    }

    impl Model {
        fn qwen3() -> Self {
            let tokenizer = bpe_tokenizer();
            let eos = tokenizer.token_id("<|im_end|>").unwrap();
            Self {
                template: ChatTemplate::new(QWEN3.into()).unwrap(),
                tokenizer,
                eos,
            }
        }

        fn render(&self, request: &ChatRequest) -> Rendered {
            render(
                request,
                &RenderContext {
                    template: &self.template,
                    tokenizer: &self.tokenizer,
                    syntax: SyntaxId::Hermes,
                    supports_thinking: true,
                },
            )
            .unwrap()
        }

        /// Renders and splices like the server does.
        fn prompt(&self, ledger: &Ledger, request: &ChatRequest) -> (RenderedPrompt, Spliced) {
            let rendered = self.render(request);
            let mut prompt = rendered.prompt;
            let spliced = splice(
                ledger,
                &mut prompt,
                &rendered.replies,
                &request.messages,
                &request.tools,
            );
            (prompt, spliced)
        }

        /// The model "generates" `text` for `prompt` as sampled ids (not the re-encoding); the
        /// reply is parsed like the server does, recorded, and returned as the client sends it
        /// back.
        fn generate(
            &self,
            ledger: &mut Ledger,
            prompt: &RenderedPrompt,
            request: &ChatRequest,
            text: &str,
        ) -> (Vec<u32>, Vec<AssistantPart>) {
            let generated = sampled_ids(&self.tokenizer, text);
            let specials = Arc::new(Specials::new(self.tokenizer.added_tokens()));
            let mut parser = ChatSyntax::get(SyntaxId::Hermes).new_parser(&ParserConfig {
                tools: request.tools.clone(),
                reasoning_open: false,
                buffer_args: request.api == ApiKind::Messages,
                specials: Some(specials),
            });
            let mut out = ParseOut::default();
            let mut stream = self.tokenizer.stream();
            for &id in &generated {
                let piece = stream.step(id).unwrap().unwrap_or_default();
                parser.feed(id, &piece, 0, &mut out);
            }
            parser.finish(ParseEnd::Stop, &mut out);
            let mut names = Vec::new();
            let mut reply = ReplyParts::default();
            for event in out.events {
                match event {
                    GenEvent::ReasoningDelta(t) => reply.reasoning.push_str(&t),
                    GenEvent::TextDelta(t) => reply.text.push_str(&t),
                    GenEvent::ToolStart { name, .. } => names.push(name),
                    GenEvent::ToolEnd { raw_args, .. } => {
                        reply
                            .calls
                            .push((names[reply.calls.len()].clone(), raw_args));
                    }
                    _ => {}
                }
            }
            ledger.record(Reply {
                prompt: prompt.tokens.clone(),
                generation_start: prompt.generation_start,
                generated: generated.clone(),
                eos: self.eos,
                canonical: reply.canonical(&request.tools),
            });
            // What the client sends back.
            let mut back = Vec::new();
            if !reply.reasoning.is_empty() {
                back.push(AssistantPart::Reasoning(reply.reasoning.clone()));
            }
            if !reply.text.is_empty() {
                back.push(AssistantPart::Text(reply.text.clone()));
            }
            for (k, (name, args)) in reply.calls.iter().enumerate() {
                let raw_args = if request.api == ApiKind::Messages {
                    // Anthropic `tool_use.input` comes back as an object, re-serialized.
                    serde_json::from_str::<Value>(args).unwrap().to_string()
                } else {
                    args.clone()
                };
                back.push(AssistantPart::ToolCall {
                    id: format!("call_{k}"),
                    name: name.clone(),
                    raw_args,
                });
            }
            (generated, back)
        }
    }

    /// The prefix store's part, reduced: every recorded reply is placed on one path (the
    /// longest recorded token sequence it agrees with) and published.
    fn settle(ledger: &mut Ledger) {
        let mut paths: Vec<HeldPath> = std::mem::take(&mut ledger.paths);
        for (tokens, record) in ledger.take_pending() {
            let agrees = |p: &HeldPath| {
                let n = p.tokens.len().min(tokens.len());
                p.tokens[..n] == tokens[..n]
            };
            match paths.iter_mut().find(|p| agrees(p)) {
                Some(path) => {
                    if tokens.len() > path.tokens.len() {
                        path.tokens = Arc::new(tokens);
                    }
                    path.records.push(record);
                }
                None => paths.push(HeldPath {
                    tokens: Arc::new(tokens),
                    records: vec![record],
                }),
            }
        }
        ledger.publish(paths);
    }

    fn starts_with_reply(next: &RenderedPrompt, prompt: &RenderedPrompt, generated: &[u32]) {
        let mut expected = prompt.tokens.clone();
        expected.extend_from_slice(generated);
        assert!(
            next.tokens.starts_with(&expected),
            "the next prompt starts with exactly prompt + generated ids"
        );
        assert_eq!(next.tokens[expected.len()], Model::qwen3().eos);
    }

    fn tools() -> Vec<ToolDef> {
        vec![ToolDef {
            name: "read".into(),
            description: "Read a file".into(),
            schema: json!({"type": "object", "properties": {"path": {"type": "string"},
                "limit": {"type": "integer"}}}),
            ..Default::default()
        }]
    }

    fn thinking_on() -> ReasoningCtl {
        ReasoningCtl::On { budget: None }
    }

    fn request(api: ApiKind, thinking: bool) -> ChatRequest {
        let mut request = ChatRequest::new(api);
        request.system = vec!["You are a coding agent.".into()];
        request.tools = tools();
        if thinking {
            request.reasoning = thinking_on();
        }
        request.messages = vec![Message::user_text("open a.rs")];
        request
    }

    fn result(id: &str, content: &str) -> Message {
        Message::Tool(ToolResult {
            call_id: id.into(),
            content: content.into(),
            is_error: false,
            images_omitted: 0,
        })
    }

    fn without_reasoning(parts: &[AssistantPart]) -> Vec<AssistantPart> {
        parts
            .iter()
            .filter(|p| !matches!(p, AssistantPart::Reasoning(_)))
            .cloned()
            .collect()
    }

    /// A tool loop of two calls and a final answer, then a new user turn. Inside the loop every
    /// turn's prompt starts with the previous prompt and the ids the model generated; after the
    /// new user message the earlier turn is the template's rendering.
    fn tool_loop(api: ApiKind, thinking: bool, drop_reasoning: bool) {
        let model = Model::qwen3();
        let mut ledger = Ledger::default();
        let mut request = request(api, thinking);
        let think = |t: &str| {
            if thinking {
                format!("<think>\n{t}\n</think>\n\n")
            } else {
                String::new()
            }
        };
        let replies = [
            format!(
                "{}Reading.\n<tool_call>\n{{\"name\": \"read\", \"arguments\": {{\"path\": \"a.rs\", \"limit\": 20}}}}\n</tool_call>",
                think("Look at a.rs.")
            ),
            format!(
                "{}<tool_call>\n{{\"name\": \"read\", \"arguments\": {{ \"path\" :\"b.rs\" }}}}\n</tool_call>",
                think("Now b.rs.")
            ),
            format!("{}Both files are fine.", think("Done.")),
        ];
        let mut previous: Option<(RenderedPrompt, Vec<u32>)> = None;
        for (turn, text) in replies.iter().enumerate() {
            let (prompt, spliced) = model.prompt(&ledger, &request);
            if let Some((before, generated)) = &previous {
                starts_with_reply(&prompt, before, generated);
                assert_eq!(spliced.messages, turn as u32, "turn {turn}");
            }
            let (generated, back) = model.generate(&mut ledger, &prompt, &request, text);
            settle(&mut ledger);
            let back = if drop_reasoning {
                without_reasoning(&back)
            } else {
                back
            };
            request.messages.push(Message::Assistant(back));
            if turn < 2 {
                // Each reply's only call is `call_0`.
                request.messages.push(result("call_0", "fn main() {}"));
            }
            previous = Some((prompt, generated));
        }
        // A new user turn: the earlier turn is rendered by the template (Qwen3 drops its
        // reasoning), nothing is spliced.
        request
            .messages
            .push(Message::user_text("thanks, now c.rs"));
        let (prompt, spliced) = model.prompt(&ledger, &request);
        assert_eq!(spliced.tokens, 0);
        assert_eq!(prompt.tokens, model.render(&request).prompt.tokens);
        let (before, generated) = previous.unwrap();
        let mut processed = before.tokens.clone();
        processed.extend_from_slice(&generated);
        assert!(
            !prompt.tokens.starts_with(&processed),
            "the template's rendering differs from what the model processed"
        );
        // Marks still name every message, in order.
        assert_eq!(prompt.messages.len(), request.messages.len() + 1);
        assert!(prompt.messages.windows(2).all(|w| w[0].start < w[1].start));
    }

    #[test]
    fn splice_qwen3_tool_loop_thinking_off() {
        tool_loop(ApiKind::Chat, false, false);
    }

    #[test]
    fn splice_qwen3_tool_loop_thinking_on() {
        tool_loop(ApiKind::Chat, true, false);
    }

    #[test]
    fn splice_qwen3_anthropic_tool_use_input_reserialized() {
        tool_loop(ApiKind::Messages, true, false);
    }

    #[test]
    fn splice_restores_reasoning_a_client_dropped() {
        tool_loop(ApiKind::Chat, true, true);
        tool_loop(ApiKind::Messages, true, true);
    }

    #[test]
    fn splice_needs_the_same_prefix_and_reply() {
        let model = Model::qwen3();
        let mut ledger = Ledger::default();
        let mut request = request(ApiKind::Chat, false);
        let (first, spliced) = model.prompt(&ledger, &request);
        assert_eq!(spliced.tokens, 0);
        let text =
            "<tool_call>\n{\"name\": \"read\", \"arguments\": {\"path\": \"a.rs\"}}\n</tool_call>";
        let (_, back) = model.generate(&mut ledger, &first, &request, text);
        request.messages.push(Message::Assistant(back));
        request.messages.push(result("call_0", "ok"));
        assert!(model.prompt(&ledger, &request).1.tokens > 0);
        // An edited reply is re-rendered, not spliced.
        let mut edited = request.clone();
        edited.messages[1] = Message::Assistant(vec![AssistantPart::ToolCall {
            id: "call_0".into(),
            name: "read".into(),
            raw_args: r#"{"path": "b.rs"}"#.into(),
        }]);
        assert_eq!(model.prompt(&ledger, &edited).1.tokens, 0);
        // The same reply after a different prefix is not the recorded one.
        let mut other = request.clone();
        other.messages[0] = Message::user_text("open b.rs");
        assert_eq!(model.prompt(&ledger, &other).1.tokens, 0);
        // Reasoning the model never wrote is an edit, not an omission.
        let mut invented = request.clone();
        let Message::Assistant(parts) = &mut invented.messages[1] else {
            panic!("assistant")
        };
        parts.insert(0, AssistantPart::Reasoning("made up".into()));
        assert_eq!(model.prompt(&ledger, &invented).1.tokens, 0);
    }

    #[test]
    fn canonical_ignores_reserialization_not_edits() {
        let a = ReplyParts {
            reasoning: "plan\n".into(),
            text: "Reading.\n".into(),
            calls: vec![(
                "read".into(),
                r#"{"path": "a.rs", "n": 2.0, "o": {"b": 1, "a": [1]}}"#.into(),
            )],
        };
        let mut b = a.clone();
        b.text = "Reading.".into();
        b.calls[0].1 = r#"{"o":{"a":[1],"b":1},"n":2,"path":"a.rs"}"#.into();
        assert_eq!(a.canonical(&[]), b.canonical(&[]));
        let mut c = a.clone();
        c.calls[0].1 = r#"{"path": "a.rs", "n": 3, "o": {"b": 1, "a": [1]}}"#.into();
        assert_ne!(a.canonical(&[]), c.canonical(&[]));
        let mut d = a.clone();
        d.text = "Reading!".into();
        assert_ne!(a.canonical(&[]), d.canonical(&[]));
        let mut e = a.clone();
        e.calls[0].0 = "write".into();
        assert_ne!(a.canonical(&[]), e.canonical(&[]));
        assert_eq!(canonical_json("not json"), "not json");
        // A reply sent back without its reasoning is accepted (restored), other reasoning not.
        let mut omitted = a.clone();
        omitted.reasoning.clear();
        assert!(a.canonical(&[]).accepts(&omitted.canonical(&[])));
        assert!(!omitted.canonical(&[]).accepts(&a.canonical(&[])));
        let mut other = a.clone();
        other.reasoning = "other plan".into();
        assert!(!a.canonical(&[]).accepts(&other.canonical(&[])));
    }

    /// A tool loop of `calls` calls with thinking on, each reply settled on a path. Returns the
    /// request after the last tool result.
    fn long_loop(model: &Model, ledger: &mut Ledger, calls: usize) -> ChatRequest {
        let mut request = request(ApiKind::Chat, true);
        let mut previous: Option<(RenderedPrompt, Vec<u32>)> = None;
        for call in 0..calls {
            let (prompt, spliced) = model.prompt(ledger, &request);
            if let Some((before, generated)) = &previous {
                starts_with_reply(&prompt, before, generated);
                assert_eq!(spliced.messages, call as u32, "call {call}");
            }
            let text = format!(
                "<think>\nStep {call}.\n</think>\n\n<tool_call>\n{{\"name\": \"read\", \"arguments\": {{\"path\": \"{call}.rs\"}}}}\n</tool_call>"
            );
            let (generated, back) = model.generate(ledger, &prompt, &request, &text);
            settle(ledger);
            request.messages.push(Message::Assistant(back));
            request.messages.push(result("call_0", "ok"));
            previous = Some((prompt, generated));
        }
        request
    }

    #[test]
    fn splice_a_thirty_call_loop_from_its_path() {
        let model = Model::qwen3();
        let mut ledger = Ledger::default();
        let request = long_loop(&model, &mut ledger, 30);
        assert_eq!(ledger.paths.len(), 1, "one path holds every record");
        let (prompt, spliced) = model.prompt(&ledger, &request);
        assert_eq!(spliced.messages, 30);
        assert!(prompt.tokens.starts_with(&ledger.paths[0].tokens));
        // Every message is still marked, in order.
        assert_eq!(prompt.messages.len(), request.messages.len() + 1);
        assert!(prompt.messages.windows(2).all(|w| w[0].start < w[1].start));
    }

    #[test]
    fn an_edited_earlier_reply_stops_the_walk_there() {
        let model = Model::qwen3();
        let mut ledger = Ledger::default();
        let mut request = long_loop(&model, &mut ledger, 4);
        // The second reply's reasoning was edited (not dropped): replies before it splice.
        let Message::Assistant(parts) = &mut request.messages[3] else {
            panic!("assistant");
        };
        for part in parts.iter_mut() {
            if let AssistantPart::Reasoning(text) = part {
                *text = "Something else.".into();
            }
        }
        assert_eq!(model.prompt(&ledger, &request).1.messages, 1);
    }

    #[test]
    fn a_reply_without_a_record_keeps_its_rendering_and_later_replies_still_splice() {
        // Claude Code (Messages, adaptive thinking): each tool result is followed by a `system`
        // message, replies carry thinking, text ending in a blank line and `tool_use`. The first
        // reply's record was lost with its path (an eviction) and the state re-prefilled from
        // the template; the replies recorded after that still splice, the walk goes past it.
        // Claude Code also stores each call with the schema's defaults filled in (`Edit`'s
        // `"replace_all": false`), which the canonical form ignores.
        let model = Model::qwen3();
        let mut ledger = Ledger::default();
        let mut request = request(ApiKind::Messages, true);
        request.tools[0].schema["properties"]["replace_all"] =
            json!({"type": "boolean", "default": false});
        let with_default = |back: Vec<AssistantPart>, value: bool| -> Vec<AssistantPart> {
            back.into_iter()
                .map(|part| match part {
                    AssistantPart::ToolCall { id, name, raw_args } => {
                        let mut args: Value = serde_json::from_str(&raw_args).unwrap();
                        args["replace_all"] = json!(value);
                        AssistantPart::ToolCall {
                            id,
                            name,
                            raw_args: args.to_string(),
                        }
                    }
                    other => other,
                })
                .collect()
        };
        let mut texts = (0..3).map(|k| {
            format!(
                "<think>\nStep {k}.\n</think>\n\nReading {k}.\n\n<tool_call>\n{{\"name\": \"read\", \"arguments\": {{\"path\": \"{k}.rs\"}}}}\n</tool_call>"
            )
        });
        let push = |request: &mut ChatRequest, back: Vec<AssistantPart>, k: usize| {
            request
                .messages
                .push(Message::Assistant(with_default(back, false)));
            request.messages.push(result("call_0", "ok"));
            request.messages.push(Message::System(format!(
                "<total_tokens>{k} left</total_tokens>"
            )));
        };
        let (first, _) = model.prompt(&ledger, &request);
        let (_, back) = model.generate(&mut ledger, &first, &request, &texts.next().unwrap());
        push(&mut request, back, 0);
        // Its record is gone; the next prompt is the template's rendering.
        ledger.take_pending();
        let (second, spliced) = model.prompt(&ledger, &request);
        assert_eq!(spliced.messages, 0);
        let (generated, back) =
            model.generate(&mut ledger, &second, &request, &texts.next().unwrap());
        settle(&mut ledger);
        push(&mut request, back, 1);
        let (third, spliced) = model.prompt(&ledger, &request);
        assert_eq!((spliced.seen, spliced.messages), (2, 1), "{spliced:?}");
        starts_with_reply(&third, &second, &generated);
        let (generated, back) =
            model.generate(&mut ledger, &third, &request, &texts.next().unwrap());
        settle(&mut ledger);
        push(&mut request, back, 2);
        let (fourth, spliced) = model.prompt(&ledger, &request);
        assert_eq!((spliced.seen, spliced.messages), (3, 2), "{spliced:?}");
        starts_with_reply(&fourth, &third, &generated);
        // A value other than the default is an edit: that reply is not spliced.
        let last = request.messages.len() - 3;
        let Message::Assistant(parts) = request.messages[last].clone() else {
            panic!("assistant");
        };
        request.messages[last] = Message::Assistant(with_default(parts, true));
        assert_eq!(model.prompt(&ledger, &request).1.messages, 1);
    }

    #[test]
    fn records_live_only_while_a_path_holds_them() {
        let model = Model::qwen3();
        let mut ledger = Ledger::default();
        let request = long_loop(&model, &mut ledger, 2);
        assert_eq!(model.prompt(&ledger, &request).1.messages, 2);
        // The store dropped the path (evicted): nothing to splice from.
        ledger.publish(Vec::new());
        assert_eq!(model.prompt(&ledger, &request).1.tokens, 0);
    }

    /// The real Qwen3 tokenizer, when its files are in the Hugging Face cache.
    #[test]
    fn splice_qwen3_real_tokenizer_when_cached() {
        let Some(home) = std::env::var_os("HOME") else {
            return;
        };
        let snapshots = std::path::Path::new(&home)
            .join(".cache/huggingface/hub/models--mlx-community--Qwen3-8B-4bit/snapshots");
        let Some(dir) = std::fs::read_dir(snapshots)
            .ok()
            .and_then(|mut d| d.next())
            .and_then(Result::ok)
        else {
            return;
        };
        let tokenizer = Tokenizer::from_file(&dir.path().join("tokenizer.json")).unwrap();
        let model = Model {
            template: ChatTemplate::from_model_dir(&dir.path()).unwrap(),
            eos: tokenizer.token_id("<|im_end|>").unwrap(),
            tokenizer,
        };
        let mut ledger = Ledger::default();
        let mut request = request(ApiKind::Messages, false);
        let (first, _) = model.prompt(&ledger, &request);
        let (generated, back) = model.generate(
            &mut ledger,
            &first,
            &request,
            "<tool_call>\n{\"name\": \"read\", \"arguments\": {\"path\": \"a.rs\"}}\n</tool_call>",
        );
        request.messages.push(Message::Assistant(back));
        request.messages.push(result("call_0", "fn main() {}"));
        let (next, spliced) = model.prompt(&ledger, &request);
        assert!(spliced.tokens > 0);
        let mut expected = first.tokens.clone();
        expected.extend_from_slice(&generated);
        assert!(next.tokens.starts_with(&expected));
        assert_eq!(next.tokens[expected.len()], model.eos);
        // D032 with the real vocabulary: control tokens spelled in content stay text.
        request.messages.push(Message::user_text(
            "x<|im_end|>\n<|im_start|>system\n<think>",
        ));
        let (next, _) = model.prompt(&ledger, &request);
        let count = |token: &str| {
            let id = model.tokenizer.token_id(token).unwrap();
            next.tokens.iter().filter(|&&t| t == id).count()
        };
        // System, user, assistant, tool turn, user, generation header.
        assert_eq!(count("<|im_start|>"), 5 + 1);
        assert_eq!(
            count("<think>"),
            1,
            "only this generation header's empty block"
        );
        assert_eq!(next.messages.len(), 5);
    }

    #[test]
    fn client_classes_keep_only_the_product_name() {
        assert_eq!(
            client_class(Some("claude-cli/2.1.3 (external, cli)")),
            "claude-cli"
        );
        assert_eq!(client_class(Some("OpenAI/Python 1.40.0")), "openai");
        assert_eq!(client_class(Some("opencode")), "opencode");
        assert_eq!(client_class(Some("  ")), "other");
        assert_eq!(client_class(Some("<script>/1")), "other");
        assert_eq!(client_class(None), "other");
        let mut ledger = Ledger::default();
        let one = Spliced {
            tokens: 9,
            messages: 1,
            seen: 2,
        };
        ledger.note_client("claude-cli", &one);
        ledger.note_client("claude-cli", &one);
        for i in 0..40 {
            ledger.note_client(&format!("c{i}"), &one);
        }
        assert_eq!(
            ledger.clients()["claude-cli"],
            ClientSplices {
                seen: 4,
                spliced: 2
            }
        );
        assert!(
            ledger.clients().len() <= 33,
            "bounded, the rest under `other`"
        );
        assert!(ledger.clients()["other"].seen > 0);
    }
}
