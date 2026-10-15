//! `ChatRequest` → template context → prompt text → tokens with message marks (002 T004,
//! T004a, T004b). Shared by generation and `count_tokens`.
//!
//! History rules depend on the family's syntax: Hermes keeps argument strings verbatim (the
//! previous reply re-renders byte for byte); Qwen XML and Harmony templates re-serialize
//! arguments from an object, so they are parsed first and may re-render differently. Harmony
//! renders one call per assistant message, so multi-call history is split.

use crate::api::model::{
    ApiError, AssistantPart, ChatRequest, Effort, ErrorKind, Message, ReasoningCtl, ToolDef,
    UserPart,
};
use crate::chat::prompt::{MessageMark, RenderedPrompt, Role};
use crate::chat::template::{ChatTemplate, RenderOptions};
use crate::chat::tokenizer::Tokenizer;
use mrun_engine::SyntaxId;
use serde_json::{Map, Value, json};
use std::collections::HashSet;

/// What the renderer needs to know about the loaded model.
#[derive(Debug)]
pub struct RenderContext<'a> {
    pub template: &'a ChatTemplate,
    pub tokenizer: &'a Tokenizer,
    pub syntax: SyntaxId,
    /// The family has a thinking mode (Qwen3, Qwen3.6; gpt-oss always reasons).
    pub supports_thinking: bool,
}

/// Server-log entries from rendering (never in-band).
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum RenderLog {
    /// History arguments that are not a JSON object, rendered raw (message index).
    HistoryArgsRaw(u32),
    /// A multi-call assistant message split into single-call messages (message index).
    HistoryCallsSplit(u32),
    /// Thinking was requested from a family without a thinking mode.
    ThinkingUnsupported,
    /// Message marks could not be aligned with the messages; the prompt has none.
    MarksMismatch,
}

#[derive(Debug)]
pub struct Rendered {
    pub prompt: RenderedPrompt,
    /// The prompt ends inside an open reasoning block (the parser starts in reasoning).
    pub reasoning_open: bool,
    pub logs: Vec<RenderLog>,
}

/// One message as the template sees it, with the API message it came from.
struct Entry {
    value: Value,
    role: Role,
    message: Option<u32>,
}

/// Private-use sentinel standing in for raw history arguments until after rendering.
const RAW_KEY: char = '\u{E000}';

pub fn render(request: &ChatRequest, ctx: &RenderContext<'_>) -> Result<Rendered, ApiError> {
    let mut logs = Vec::new();
    let mut raws: Vec<String> = Vec::new();
    let entries = entries(request, ctx.syntax, &mut raws, &mut logs)?;
    let messages = Value::Array(entries.iter().map(|e| e.value.clone()).collect());

    let thinking = request.reasoning.is_on();
    if thinking && !ctx.supports_thinking && ctx.syntax != SyntaxId::Harmony {
        logs.push(RenderLog::ThinkingUnsupported);
    }
    // Deferred tools stay out of the tools region (FR-028): their definitions appear inline
    // in the tool result that returned them, so the prefix stays append-only.
    let rendered: Vec<Value> = request
        .tools
        .iter()
        .filter(|t| !t.deferred)
        .map(ToolDef::definition)
        .collect();
    let tools = (!rendered.is_empty()).then_some(Value::Array(rendered));
    let mut options = RenderOptions {
        add_generation_prompt: !request.prefill,
        enable_thinking: thinking && ctx.supports_thinking,
        tools,
        reasoning_effort: (ctx.syntax == SyntaxId::Harmony)
            .then(|| harmony_effort(request.reasoning).to_string()),
        preserve_thinking: ctx.syntax == SyntaxId::QwenXml && ctx.supports_thinking,
    };
    let render = |options: &RenderOptions| {
        ctx.template
            .render(&messages, options)
            .map(|text| restore_raw(text, &raws, ctx.syntax))
            .map_err(|e| ApiError::new(ErrorKind::InvalidRequest, e.to_string()))
    };

    let mut text = render(&options)?;
    let header_at = if request.prefill {
        let Some(Message::Assistant(parts)) = request.messages.last() else {
            unreachable!("decoders set prefill only for a trailing assistant message")
        };
        let continued = parts
            .iter()
            .filter_map(|p| match p {
                AssistantPart::Text(t) => Some(t.as_str()),
                _ => None,
            })
            .collect::<String>();
        let end = text
            .rfind(continued.trim_end())
            .map(|at| at + continued.trim_end().len())
            .ok_or_else(|| {
                ApiError::invalid(
                    "the chat template does not render the assistant prefill",
                    format!("messages[{}]", request.messages.len() - 1),
                )
            })?;
        text.truncate(end);
        end
    } else {
        options.add_generation_prompt = false;
        let without = render(&options)?;
        common_prefix(&without, &text)
    };

    let (tokens, offsets) = ctx
        .tokenizer
        .encode_with_offsets(&text)
        .map_err(|e| ApiError::new(ErrorKind::Internal, e.to_string()))?;
    let token_at = |byte: usize| offsets.partition_point(|&(start, _)| start < byte) as u32;
    let generation_start = token_at(header_at).min(tokens.len() as u32);

    let marks = match align_marks(&text[..header_at], ctx.syntax, &entries) {
        Some(marks) => marks
            .into_iter()
            .map(|(byte, role, message)| MessageMark {
                start: token_at(byte),
                role,
                message,
            })
            .collect(),
        None => {
            logs.push(RenderLog::MarksMismatch);
            Vec::new()
        }
    };
    let system_tools_end = marks
        .iter()
        .find(|m: &&MessageMark| m.role != Role::System)
        .map_or(
            if marks.is_empty() {
                0
            } else {
                generation_start
            },
            |m| m.start,
        );
    let reasoning_open = reasoning_open(&text[header_at..], ctx.syntax);
    Ok(Rendered {
        prompt: RenderedPrompt {
            tokens,
            messages: marks,
            system_tools_end,
            generation_start,
        },
        reasoning_open,
        logs,
    })
}

/// gpt-oss reasoning effort (research §3b): off → low.
fn harmony_effort(reasoning: ReasoningCtl) -> &'static str {
    match reasoning {
        ReasoningCtl::Off => "low",
        ReasoningCtl::On {
            effort: Some(effort),
            ..
        } => match effort {
            Effort::Low => "low",
            Effort::Medium => "medium",
            Effort::High => "high",
        },
        ReasoningCtl::On {
            budget: Some(budget),
            ..
        } => match budget {
            0..8_192 => "low",
            8_192..24_576 => "medium",
            _ => "high",
        },
        ReasoningCtl::On { .. } => "medium",
    }
}

fn entries(
    request: &ChatRequest,
    syntax: SyntaxId,
    raws: &mut Vec<String>,
    logs: &mut Vec<RenderLog>,
) -> Result<Vec<Entry>, ApiError> {
    let mut out = Vec::new();
    if !request.system.is_empty() {
        out.push(Entry {
            value: json!({"role": "system", "content": request.system.join("\n\n")}),
            role: Role::System,
            message: Some(0),
        });
    }
    // API message indices count the leading system block as message 0 when present.
    let base = u32::from(!request.system.is_empty());
    let mut calls: HashSet<&str> = HashSet::new();
    let mut i = 0;
    while i < request.messages.len() {
        let index = base + i as u32;
        match &request.messages[i] {
            Message::User(parts) => {
                let text = parts
                    .iter()
                    .map(|UserPart::Text(t)| t.as_str())
                    .collect::<Vec<_>>()
                    .join("\n");
                out.push(entry("user", text, Role::User, index));
            }
            Message::System(text) => {
                // Qwen XML templates refuse non-leading system messages: render a user turn
                // at the same position (FR-029).
                if syntax == SyntaxId::QwenXml {
                    out.push(entry("user", text.clone(), Role::User, index));
                } else {
                    out.push(entry("system", text.clone(), Role::System, index));
                }
            }
            Message::Tool(result) => {
                if !calls.contains(result.call_id.as_str()) {
                    return Err(ApiError::invalid(
                        format!(
                            "tool result '{}' does not match any earlier tool call",
                            result.call_id
                        ),
                        format!("messages[{i}]"),
                    ));
                }
                out.push(tool_entry(&result.call_id, &result.content, index));
            }
            Message::Assistant(parts) => {
                let mut content = String::new();
                let mut reasoning = String::new();
                let mut tool_calls = Vec::new();
                for part in parts {
                    match part {
                        AssistantPart::Text(t) => content.push_str(t),
                        AssistantPart::Reasoning(r) => reasoning.push_str(r),
                        AssistantPart::ToolCall { id, name, raw_args } => {
                            calls.insert(id);
                            let arguments = history_args(raw_args, syntax, raws, logs, index);
                            tool_calls.push(json!({"id": id, "type": "function",
                                "function": {"name": name, "arguments": arguments}}));
                        }
                    }
                }
                let reasoning_key = if syntax == SyntaxId::Harmony {
                    "thinking"
                } else {
                    "reasoning_content"
                };
                let assistant = |content: &str, reasoning: &str, calls: Vec<Value>| {
                    let mut m = Map::new();
                    m.insert("role".into(), "assistant".into());
                    m.insert("content".into(), content.into());
                    if !reasoning.is_empty() {
                        m.insert(reasoning_key.into(), reasoning.into());
                    }
                    if !calls.is_empty() {
                        m.insert("tool_calls".into(), Value::Array(calls));
                    }
                    Entry {
                        value: Value::Object(m),
                        role: Role::Assistant,
                        message: Some(index),
                    }
                };
                if syntax == SyntaxId::Harmony && tool_calls.len() > 1 {
                    // One call per message, each followed by its result.
                    logs.push(RenderLog::HistoryCallsSplit(index));
                    let mut results: Vec<(usize, &str, &str)> = Vec::new();
                    let mut j = i + 1;
                    while let Some(Message::Tool(r)) = request.messages.get(j) {
                        results.push((j, r.call_id.as_str(), r.content.as_str()));
                        j += 1;
                    }
                    for (k, call) in tool_calls.into_iter().enumerate() {
                        let id = call["id"].as_str().unwrap_or_default().to_string();
                        let (c, r) = if k == 0 {
                            (content.as_str(), reasoning.as_str())
                        } else {
                            ("", "")
                        };
                        out.push(assistant(c, r, vec![call]));
                        if let Some(pos) = results.iter().position(|(_, cid, _)| *cid == id) {
                            let (at, cid, text) = results.remove(pos);
                            out.push(tool_entry(cid, text, base + at as u32));
                        }
                    }
                    for (at, cid, text) in results {
                        out.push(tool_entry(cid, text, base + at as u32));
                    }
                    i = j;
                    continue;
                }
                out.push(assistant(&content, &reasoning, tool_calls));
            }
        }
        i += 1;
    }
    Ok(out)
}

fn entry(role: &str, content: String, kind: Role, index: u32) -> Entry {
    Entry {
        value: json!({"role": role, "content": content}),
        role: kind,
        message: Some(index),
    }
}

fn tool_entry(call_id: &str, content: &str, index: u32) -> Entry {
    Entry {
        value: json!({"role": "tool", "tool_call_id": call_id, "content": content}),
        role: Role::Tool,
        message: Some(index),
    }
}

/// History arguments per syntax: Hermes keeps the string; XML and Harmony need an object, and
/// a non-object stands in as a sentinel replaced by the raw text after rendering.
fn history_args(
    raw: &str,
    syntax: SyntaxId,
    raws: &mut Vec<String>,
    logs: &mut Vec<RenderLog>,
    index: u32,
) -> Value {
    if syntax == SyntaxId::Hermes {
        return Value::String(raw.to_string());
    }
    match serde_json::from_str::<Value>(raw) {
        Ok(Value::Object(object)) => Value::Object(object),
        _ => {
            logs.push(RenderLog::HistoryArgsRaw(index));
            let key = format!("{RAW_KEY}{}", raws.len());
            raws.push(raw.to_string());
            json!({ key: "" })
        }
    }
}

/// Replaces raw-argument sentinels with the raw text: Qwen XML renders the sentinel as an empty
/// parameter, Harmony as a one-key JSON object.
fn restore_raw(mut text: String, raws: &[String], syntax: SyntaxId) -> String {
    for (n, raw) in raws.iter().enumerate() {
        let key = format!("{RAW_KEY}{n}");
        let patterns = match syntax {
            SyntaxId::QwenXml => vec![
                (
                    format!("<parameter={key}>\n\n</parameter>\n"),
                    format!("{raw}\n"),
                ),
                (
                    format!("<parameter={key}>\n</parameter>\n"),
                    format!("{raw}\n"),
                ),
            ],
            _ => vec![
                (format!("{{\"{key}\": \"\"}}"), raw.clone()),
                (format!("{{\"{key}\":\"\"}}"), raw.clone()),
            ],
        };
        for (from, to) in patterns {
            if text.contains(&from) {
                text = text.replacen(&from, &to, 1);
                break;
            }
        }
    }
    text
}

fn common_prefix(a: &str, b: &str) -> usize {
    let mut n = a.bytes().zip(b.bytes()).take_while(|(x, y)| x == y).count();
    while !b.is_char_boundary(n) {
        n -= 1;
    }
    n
}

/// The generation prompt opens a reasoning block that it does not close.
fn reasoning_open(tail: &str, syntax: SyntaxId) -> bool {
    match syntax {
        SyntaxId::Harmony => false,
        _ => match (tail.rfind("<think>"), tail.rfind("</think>")) {
            (Some(open), Some(close)) => open > close,
            (Some(_), None) => true,
            _ => false,
        },
    }
}

/// Message starts in `text` (before the generation header), aligned with the entries:
/// `(byte offset, role, API message)`. `None` when they cannot be aligned.
fn align_marks(
    text: &str,
    syntax: SyntaxId,
    entries: &[Entry],
) -> Option<Vec<(usize, Role, Option<u32>)>> {
    let raw = raw_marks(text, syntax);
    let mut out: Vec<(usize, Role, Option<u32>)> = Vec::new();
    let mut j = 0;
    for (k, &(byte, role)) in raw.iter().enumerate() {
        // Leading system blocks beyond the request's own are inserted by the template (default
        // system prompt, Harmony's system block before the developer message).
        let inserted = role == Role::System
            && j == 0
            && (entries.first().is_none_or(|e| e.role != Role::System)
                || raw.get(k + 1).is_some_and(|next| next.1 == Role::System));
        if inserted {
            out.push((byte, role, None));
        } else if j < entries.len() && role == entries[j].role {
            out.push((byte, role, entries[j].message));
            j += 1;
        } else if role == Role::Assistant && out.last().is_some_and(|m| m.1 == Role::Assistant) {
            // A second rendered message of the same assistant turn (Harmony analysis + call).
        } else {
            return None;
        }
    }
    (j == entries.len()).then_some(out)
}

/// Every message start with the role the text shows.
fn raw_marks(text: &str, syntax: SyntaxId) -> Vec<(usize, Role)> {
    let mut marks = Vec::new();
    match syntax {
        SyntaxId::Harmony => {
            for (at, _) in text.match_indices("<|start|>") {
                let rest = &text[at + "<|start|>".len()..];
                let role = if rest.starts_with("user") {
                    Role::User
                } else if rest.starts_with("assistant") {
                    Role::Assistant
                } else if rest.starts_with("functions.") {
                    Role::Tool
                } else {
                    Role::System
                };
                marks.push((at, role));
            }
        }
        SyntaxId::Hermes | SyntaxId::QwenXml => {
            const START: &str = "<|im_start|>";
            const TOOL: &str = "<tool_response>";
            let mut at = 0;
            while let Some(next) = next_of(text, at, &[START, TOOL]) {
                if text[next..].starts_with(START) {
                    let rest = &text[next + START.len()..];
                    let mut skip = START.len();
                    let role = if let Some(user) = rest.strip_prefix("user") {
                        // A merged tool turn opens with a tool response right after the header.
                        let gap = user.len() - user.trim_start_matches('\n').len();
                        if user[gap..].starts_with(TOOL) {
                            skip += "user".len() + gap + TOOL.len();
                            Role::Tool
                        } else {
                            Role::User
                        }
                    } else if rest.starts_with("assistant") {
                        Role::Assistant
                    } else {
                        Role::System
                    };
                    marks.push((next, role));
                    at = next + skip;
                } else {
                    marks.push((next, Role::Tool));
                    at = next + TOOL.len();
                }
            }
        }
    }
    marks
}

fn next_of(text: &str, from: usize, needles: &[&str]) -> Option<usize> {
    needles
        .iter()
        .filter_map(|n| text[from..].find(n).map(|i| from + i))
        .min()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::model::{ApiKind, ToolDef, ToolResult};
    use crate::chat::tokenizer::test_support::byte_tokenizer;

    /// Qwen3-style ChatML: string arguments verbatim, tool results merged into a user turn.
    const HERMES: &str = r#"{%- if tools %}<|im_start|>system
{% if messages[0].role == 'system' %}{{ messages[0].content }}

{% endif %}<tools>{% for t in tools %}
{{ t | tojson }}{% endfor %}
</tools><|im_end|>
{% elif messages[0].role == 'system' %}<|im_start|>system
{{ messages[0].content }}<|im_end|>
{% endif %}
{%- for m in messages %}
{%- if m.role == 'system' and loop.first %}
{%- elif m.role == 'user' or m.role == 'system' %}<|im_start|>{{ m.role }}
{{ m.content }}<|im_end|>
{% elif m.role == 'assistant' %}<|im_start|>assistant
{% if m.reasoning_content %}<think>
{{ m.reasoning_content }}
</think>

{% endif %}{{ m.content }}{% for c in m.tool_calls or [] %}
<tool_call>
{"name": "{{ c.function.name }}", "arguments": {% if c.function.arguments is string %}{{ c.function.arguments }}{% else %}{{ c.function.arguments | tojson }}{% endif %}}
</tool_call>{% endfor %}<|im_end|>
{% elif m.role == 'tool' %}{% if loop.first or loop.previtem.role != 'tool' %}<|im_start|>user{% endif %}
<tool_response>
{{ m.content }}
</tool_response>{% if loop.last or loop.nextitem.role != 'tool' %}<|im_end|>
{% endif %}{% endif %}
{%- endfor %}
{%- if add_generation_prompt %}<|im_start|>assistant
{% if not enable_thinking %}<think>

</think>

{% endif %}{% endif %}"#;

    /// Qwen3-Coder-style XML: arguments iterated as an object, default system prompt.
    const XML: &str = r#"{%- if messages[0].role != 'system' %}<|im_start|>system
You are Qwen.<|im_end|>
{% endif %}
{%- for m in messages %}
{%- if m.role == 'assistant' %}<|im_start|>assistant
{{ m.content }}{% for c in m.tool_calls or [] %}
<tool_call>
<function={{ c.function.name }}>
{% for k, v in c.function.arguments | items %}<parameter={{ k }}>
{{ v if v is string else v | tojson }}
</parameter>
{% endfor %}</function>
</tool_call>{% endfor %}<|im_end|>
{% elif m.role == 'tool' %}{% if loop.previtem.role != 'tool' %}<|im_start|>user{% endif %}
<tool_response>
{{ m.content }}
</tool_response>{% if loop.last or loop.nextitem.role != 'tool' %}<|im_end|>
{% endif %}{% else %}<|im_start|>{{ m.role }}
{{ m.content }}<|im_end|>
{% endif %}
{%- endfor %}
{%- if add_generation_prompt %}<|im_start|>assistant
{% endif %}"#;

    /// gpt-oss-style Harmony: one call per message, arguments via tojson.
    const HARMONY: &str = r#"<|start|>system<|message|>Reasoning: {{ reasoning_effort }}<|end|>
{%- for m in messages %}
{%- if m.role == 'assistant' %}
{%- if m.thinking %}<|start|>assistant<|channel|>analysis<|message|>{{ m.thinking }}<|end|>{% endif %}
{%- if m.tool_calls %}<|start|>assistant to=functions.{{ m.tool_calls[0].function.name }}<|channel|>commentary json<|message|>{{ m.tool_calls[0].function.arguments | tojson }}<|call|>
{%- else %}<|start|>assistant<|channel|>final<|message|>{{ m.content }}<|end|>{% endif %}
{%- elif m.role == 'tool' %}<|start|>functions.x to=assistant<|channel|>commentary<|message|>{{ m.content | tojson }}<|end|>
{%- else %}<|start|>{{ m.role }}<|message|>{{ m.content }}<|end|>{% endif %}
{%- endfor %}
{%- if add_generation_prompt %}<|start|>assistant{% endif %}"#;

    fn ctx<'a>(
        template: &'a ChatTemplate,
        tokenizer: &'a Tokenizer,
        syntax: SyntaxId,
    ) -> RenderContext<'a> {
        RenderContext {
            template,
            tokenizer,
            syntax,
            supports_thinking: syntax != SyntaxId::Harmony,
        }
    }

    fn text(tokenizer: &Tokenizer, ids: &[u32]) -> String {
        let mut stream = tokenizer.stream();
        ids.iter()
            .filter_map(|&id| stream.step(id).unwrap())
            .collect()
    }

    fn call(id: &str, args: &str) -> AssistantPart {
        AssistantPart::ToolCall {
            id: id.into(),
            name: "read".into(),
            raw_args: args.into(),
        }
    }

    fn result(id: &str, content: &str) -> Message {
        Message::Tool(ToolResult {
            call_id: id.into(),
            content: content.into(),
            is_error: false,
            images_omitted: 0,
        })
    }

    fn tool_loop() -> ChatRequest {
        let mut request = ChatRequest::new(ApiKind::Chat);
        request.system = vec!["Be brief.".into()];
        request.tools = vec![ToolDef {
            name: "read".into(),
            description: "Read a file".into(),
            schema: json!({"type": "object", "properties": {"path": {"type": "string"}}}),
            strict: false,
            ..Default::default()
        }];
        request.messages = vec![
            Message::user_text("open it"),
            Message::Assistant(vec![
                AssistantPart::Text("Sure.".into()),
                call("c1", r#"{"path":"a.rs"}"#),
            ]),
            result("c1", "fn main() {}"),
            Message::user_text("thanks"),
        ];
        request
    }

    fn roles(rendered: &Rendered) -> Vec<(Role, Option<u32>)> {
        rendered
            .prompt
            .messages
            .iter()
            .map(|m| (m.role, m.message))
            .collect()
    }

    #[test]
    fn hermes_marks_follow_api_messages() {
        let tokenizer = byte_tokenizer();
        let template = ChatTemplate::new(HERMES.into()).unwrap();
        let rendered = render(&tool_loop(), &ctx(&template, &tokenizer, SyntaxId::Hermes)).unwrap();
        assert_eq!(
            roles(&rendered),
            vec![
                (Role::System, Some(0)),
                (Role::User, Some(1)),
                (Role::Assistant, Some(2)),
                (Role::Tool, Some(3)),
                (Role::User, Some(4)),
            ]
        );
        let prompt = &rendered.prompt;
        let all = text(&tokenizer, &prompt.tokens);
        assert!(all.contains(r#""arguments": {"path":"a.rs"}}"#), "{all}");
        let tail = text(
            &tokenizer,
            &prompt.tokens[prompt.generation_start as usize..],
        );
        assert_eq!(tail, "<|im_start|>assistant\n<think>\n\n</think>\n\n");
        assert_eq!(prompt.system_tools_end, prompt.messages[1].start);
        assert!(!rendered.reasoning_open);
        for mark in &prompt.messages {
            let at = text(&tokenizer, &prompt.tokens[mark.start as usize..]);
            assert!(at.starts_with("<|im_start|>"), "{at}");
        }
    }

    #[test]
    fn xapi_qwen3_raw_args_rerender_byte_stable() {
        // The model's own bytes, odd spacing included, come back unchanged.
        let tokenizer = byte_tokenizer();
        let template = ChatTemplate::new(HERMES.into()).unwrap();
        let mut request = tool_loop();
        request.messages[1] = Message::Assistant(vec![call("c1", r#"{ "path" :  "a.rs" }"#)]);
        let rendered = render(&request, &ctx(&template, &tokenizer, SyntaxId::Hermes)).unwrap();
        let all = text(&tokenizer, &rendered.prompt.tokens);
        assert!(all.contains(r#"{"name": "read", "arguments": { "path" :  "a.rs" }}"#));
    }

    #[test]
    fn thinking_opens_the_reasoning_block() {
        let tokenizer = byte_tokenizer();
        let template = ChatTemplate::new(
            "{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}\
             {% if add_generation_prompt %}<|im_start|>assistant\n{% if enable_thinking %}<think>\n{% endif %}{% endif %}"
                .into(),
        )
        .unwrap();
        let mut request = ChatRequest::new(ApiKind::Messages);
        request.messages = vec![Message::user_text("hi")];
        request.reasoning = ReasoningCtl::On {
            budget: None,
            effort: None,
        };
        let rendered = render(&request, &ctx(&template, &tokenizer, SyntaxId::Hermes)).unwrap();
        assert!(rendered.reasoning_open);
        request.reasoning = ReasoningCtl::Off;
        let rendered = render(&request, &ctx(&template, &tokenizer, SyntaxId::Hermes)).unwrap();
        assert!(!rendered.reasoning_open);
    }

    #[test]
    fn xml_default_system_and_merged_tool_results() {
        let tokenizer = byte_tokenizer();
        let template = ChatTemplate::new(XML.into()).unwrap();
        let mut request = tool_loop();
        request.system.clear();
        request.messages[1] = Message::Assistant(vec![call("c1", "{}"), call("c2", "{}")]);
        request.messages.insert(3, result("c2", "two"));
        let rendered = render(&request, &ctx(&template, &tokenizer, SyntaxId::QwenXml)).unwrap();
        assert_eq!(
            roles(&rendered),
            vec![
                (Role::System, None),
                (Role::User, Some(0)),
                (Role::Assistant, Some(1)),
                (Role::Tool, Some(2)),
                (Role::Tool, Some(3)),
                (Role::User, Some(4)),
            ]
        );
    }

    #[test]
    fn xapi_qwen_xml_rerender_changed_reported() {
        // Arguments are re-serialized from an object: the model's bytes are not kept.
        let tokenizer = byte_tokenizer();
        let template = ChatTemplate::new(XML.into()).unwrap();
        let mut request = tool_loop();
        request.messages[1] =
            Message::Assistant(vec![call("c1", r#"{"path": "a.rs", "n":  [1,2]}"#)]);
        let rendered = render(&request, &ctx(&template, &tokenizer, SyntaxId::QwenXml)).unwrap();
        let all = text(&tokenizer, &rendered.prompt.tokens);
        assert!(
            all.contains(
                "<parameter=path>\na.rs\n</parameter>\n<parameter=n>\n[1, 2]\n</parameter>"
            ),
            "{all}"
        );
        assert!(rendered.logs.is_empty());
    }

    #[test]
    fn xapi_qwen_xml_history_args_nonobject_raw_logged() {
        let tokenizer = byte_tokenizer();
        let template = ChatTemplate::new(XML.into()).unwrap();
        let mut request = tool_loop();
        request.messages[1] = Message::Assistant(vec![call("c1", "<<not json>>")]);
        let rendered = render(&request, &ctx(&template, &tokenizer, SyntaxId::QwenXml)).unwrap();
        let all = text(&tokenizer, &rendered.prompt.tokens);
        assert!(
            all.contains("<function=read>\n<<not json>>\n</function>"),
            "{all}"
        );
        assert_eq!(rendered.logs, vec![RenderLog::HistoryArgsRaw(2)]);
    }

    #[test]
    fn qwen_xml_midconversation_system_becomes_user_turn() {
        let tokenizer = byte_tokenizer();
        let template = ChatTemplate::new(XML.into()).unwrap();
        let mut request = tool_loop();
        request
            .messages
            .push(Message::System("Now be verbose.".into()));
        let rendered = render(&request, &ctx(&template, &tokenizer, SyntaxId::QwenXml)).unwrap();
        let last = rendered.prompt.messages.last().unwrap();
        assert_eq!((last.role, last.message), (Role::User, Some(5)));
    }

    #[test]
    fn syn_harmony_history_multi_call_split_logged() {
        let tokenizer = byte_tokenizer();
        let template = ChatTemplate::new(HARMONY.into()).unwrap();
        let mut request = tool_loop();
        request.messages[1] = Message::Assistant(vec![
            AssistantPart::Reasoning("look at both".into()),
            call("c1", r#"{"path":"a"}"#),
            call("c2", r#"{"path":"b"}"#),
        ]);
        request.messages.insert(3, result("c2", "B"));
        request.messages[2] = result("c1", "A");
        let rendered = render(&request, &ctx(&template, &tokenizer, SyntaxId::Harmony)).unwrap();
        assert_eq!(rendered.logs, vec![RenderLog::HistoryCallsSplit(2)]);
        let all = text(&tokenizer, &rendered.prompt.tokens);
        let a = all.find(r#"{"path": "a"}"#).unwrap();
        let ra = all.find(r#""A""#).unwrap();
        let b = all.find(r#"{"path": "b"}"#).unwrap();
        assert!(a < ra && ra < b, "{all}");
        assert!(all.contains("Reasoning: low"));
        assert_eq!(
            roles(&rendered),
            vec![
                (Role::System, None),
                (Role::System, Some(0)),
                (Role::User, Some(1)),
                (Role::Assistant, Some(2)),
                (Role::Tool, Some(3)),
                (Role::Assistant, Some(2)),
                (Role::Tool, Some(4)),
                (Role::User, Some(5)),
            ]
        );
    }

    /// Qwen3.5/3.6-style history: reasoning is rendered only after the last user query unless
    /// `preserve_thinking` is set.
    const QWEN36: &str = r#"{%- set ns = namespace(last_user=-1) %}
{%- for m in messages %}{% if m.role == 'user' %}{% set ns.last_user = loop.index0 %}{% endif %}{% endfor %}
{%- for m in messages %}
{%- if m.role == 'assistant' %}<|im_start|>assistant
{% if m.reasoning_content and (preserve_thinking or loop.index0 > ns.last_user) %}<think>
{{ m.reasoning_content }}
</think>

{% endif %}{{ m.content }}{% for c in m.tool_calls or [] %}
<tool_call>
<function={{ c.function.name }}>
{% for k, v in c.function.arguments | items %}<parameter={{ k }}>
{{ v }}
</parameter>
{% endfor %}</function>
</tool_call>{% endfor %}<|im_end|>
{% elif m.role == 'tool' %}<|im_start|>user{{ '\n' }}<tool_response>
{{ m.content }}
</tool_response><|im_end|>
{% else %}<|im_start|>{{ m.role }}
{{ m.content }}<|im_end|>
{% endif %}
{%- endfor %}
{%- if add_generation_prompt %}<|im_start|>assistant
<think>
{% endif %}"#;

    #[test]
    fn qwen36_preserve_thinking_keeps_the_tool_loop_prefix() {
        let tokenizer = byte_tokenizer();
        let template = ChatTemplate::new(QWEN36.into()).unwrap();
        let loop_turns = vec![
            Message::user_text("fix it"),
            Message::Assistant(vec![
                AssistantPart::Reasoning("Look at a.rs first.".into()),
                call("c1", r#"{"path":"a.rs"}"#),
            ]),
            result("c1", "fn main() {}"),
            Message::Assistant(vec![
                AssistantPart::Reasoning("It is fine.".into()),
                AssistantPart::Text("Done.".into()),
            ]),
        ];
        let mut first = ChatRequest::new(ApiKind::Messages);
        first.messages = loop_turns[..3].to_vec();
        let mut second = ChatRequest::new(ApiKind::Messages);
        second.messages = loop_turns.clone();
        second.messages.push(Message::user_text("thanks, now b.rs"));
        for (preserve, extends) in [(true, true), (false, false)] {
            let context = RenderContext {
                supports_thinking: preserve,
                ..ctx(&template, &tokenizer, SyntaxId::QwenXml)
            };
            let a = render(&first, &context).unwrap().prompt;
            let b = render(&second, &context).unwrap().prompt;
            let head = &a.tokens[..a.generation_start as usize];
            assert_eq!(b.tokens.starts_with(head), extends, "preserve={preserve}");
        }
    }

    #[test]
    fn syn_harmony_effort_mapping_openai() {
        use crate::api::chat_completions::openai_effort;
        let effort = |e: Option<&str>| harmony_effort(openai_effort(e));
        assert_eq!(effort(None), "low");
        assert_eq!(effort(Some("none")), "low");
        assert_eq!(effort(Some("minimal")), "low");
        assert_eq!(effort(Some("low")), "low");
        assert_eq!(effort(Some("medium")), "medium");
        assert_eq!(effort(Some("high")), "high");
        assert_eq!(effort(Some("xhigh")), "high");
    }

    #[test]
    fn syn_harmony_effort_mapping_anthropic_budget_and_adaptive() {
        let on = |budget, effort| ReasoningCtl::On { budget, effort };
        assert_eq!(harmony_effort(ReasoningCtl::Off), "low");
        assert_eq!(harmony_effort(on(None, Some(Effort::High))), "high");
        assert_eq!(harmony_effort(on(Some(4_000), None)), "low");
        assert_eq!(harmony_effort(on(Some(10_000), None)), "medium");
        assert_eq!(harmony_effort(on(Some(30_000), None)), "high");
        assert_eq!(harmony_effort(on(None, None)), "medium");
    }

    #[test]
    fn chat_tool_result_unknown_id_400() {
        let tokenizer = byte_tokenizer();
        let template = ChatTemplate::new(HERMES.into()).unwrap();
        let mut request = tool_loop();
        request.messages[2] = result("nope", "x");
        let error = render(&request, &ctx(&template, &tokenizer, SyntaxId::Hermes)).unwrap_err();
        assert_eq!(error.param.as_deref(), Some("messages[2]"));
    }

    #[test]
    fn msg_assistant_prefill_continues() {
        let tokenizer = byte_tokenizer();
        let template = ChatTemplate::new(HERMES.into()).unwrap();
        let mut request = ChatRequest::new(ApiKind::Messages);
        request.messages = vec![
            Message::user_text("Say JSON"),
            Message::Assistant(vec![AssistantPart::Text("{\"a\":".into())]),
        ];
        request.prefill = true;
        let rendered = render(&request, &ctx(&template, &tokenizer, SyntaxId::Hermes)).unwrap();
        let all = text(&tokenizer, &rendered.prompt.tokens);
        assert!(all.ends_with("<|im_start|>assistant\n{\"a\":"), "{all}");
        assert_eq!(rendered.prompt.generation_start, rendered.prompt.len());
    }

    #[test]
    fn thinking_unsupported_is_logged() {
        let tokenizer = byte_tokenizer();
        let template = ChatTemplate::new(XML.into()).unwrap();
        let mut request = tool_loop();
        request.reasoning = ReasoningCtl::On {
            budget: None,
            effort: None,
        };
        let mut context = ctx(&template, &tokenizer, SyntaxId::QwenXml);
        context.supports_thinking = false;
        let rendered = render(&request, &context).unwrap();
        assert!(rendered.logs.contains(&RenderLog::ThinkingUnsupported));
    }
}
