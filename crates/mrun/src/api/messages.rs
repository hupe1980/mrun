//! Anthropic Messages (`POST /v1/messages`, `POST /v1/messages/count_tokens`): decoder and
//! encoders (002 research §1b).
//!
//! Tool calls are buffered per call (the parser's `buffer_args`), so a call whose arguments do
//! not parse can still become a plain text block instead of a broken `tool_use` (FR-004).

use crate::api::decode::{
    DecodeOptions, arr, check_counts, check_schema, declared, function_tool, ignore_unknown,
    join_text_parts, obj, opt_bool, opt_f32, opt_str, opt_u32, raw_args, stops, str_field,
    text_part, tool_image,
};
use crate::api::handler::{self, Api, StreamEncode};
use crate::api::model::{
    ApiError, ApiKind, AssistantPart, ChatRequest, ErrorKind, Finish, GenEvent, Message,
    ReasoningCtl, ResponseFormat, ToolChoice, ToolDef, ToolResult, Usage,
};
use crate::api::sse::{self, new_id};
use crate::chat::reasoning;
use crate::cli::serve::ServerState;
use axum::body::Bytes;
use axum::extract::State;
use axum::extract::rejection::BytesRejection;
use axum::response::{IntoResponse, Response};
use serde_json::{Value, json};

/// Top-level fields mrun reads or knowingly ignores (Claude Code's included, D055).
pub const KNOWN: &[&str] = &[
    "model",
    "messages",
    "system",
    "max_tokens",
    "metadata",
    "stop_sequences",
    "repetition_penalty",
    "stream",
    "temperature",
    "top_p",
    "top_k",
    "tools",
    "tool_choice",
    "thinking",
    "output_config",
    "context_management",
    "service_tier",
    "mcp_servers",
    "container",
];
/// `output_config` fields: `format` is enforced; `effort` and `task_budget` are accepted and
/// ignored (D047).
pub const KNOWN_OUTPUT_CONFIG: &[&str] = &["format"];

pub fn decode(body: &Value, options: DecodeOptions) -> Result<ChatRequest, ApiError> {
    obj(body, "body")?;
    ignore_unknown(body, KNOWN, "");
    ignore_unknown(
        &body["output_config"],
        KNOWN_OUTPUT_CONFIG,
        "output_config.",
    );
    // Server-executed features (FR-029): mrun runs nothing on the client's behalf.
    for field in ["mcp_servers", "container"] {
        if !body[field].is_null() {
            return Err(ApiError::invalid(
                format!(
                    "{field} asks the server to run tools, which mrun does not; configure MCP servers in the client"
                ),
                field,
            ));
        }
    }
    check_counts(body, "messages")?;
    let mut request = ChatRequest::new(ApiKind::Messages);
    request.model = opt_str(body, "model", "body")?
        .unwrap_or_default()
        .to_string();
    match &body["system"] {
        Value::Null => {}
        Value::String(s) => request.system.push(normalize_billing_nonce(s)),
        Value::Array(blocks) => {
            for (i, block) in blocks.iter().enumerate() {
                let text = str_field(block, "text", &format!("system[{i}]"))?;
                request.system.push(if i == 0 {
                    normalize_billing_nonce(text)
                } else {
                    text.to_string()
                });
            }
        }
        _ => {
            return Err(ApiError::invalid(
                "system must be a string or blocks",
                "system",
            ));
        }
    }

    if let Some(tools) = body.get("tools").filter(|t| !t.is_null()) {
        for (i, tool) in arr(tools, "tools")?.iter().enumerate() {
            let path = format!("tools[{i}]");
            if let Some(kind) = tool["type"].as_str()
                && kind != "custom"
            {
                return Err(ApiError::invalid(
                    format!(
                        "server tool '{kind}' is not supported: mrun runs no server-executed tools"
                    ),
                    format!("{path}.type"),
                ));
            }
            request
                .tools
                .push(function_tool(tool, &path, "input_schema", None)?);
        }
    }
    if !request.tools.is_empty() && request.tools.iter().all(|t| t.deferred) {
        return Err(ApiError::invalid(
            "All tools have defer_loading set. At least one tool must be non-deferred.",
            "tools",
        ));
    }
    let messages = arr(&body["messages"], "messages")?;
    if messages.is_empty() {
        return Err(ApiError::invalid("messages must not be empty", "messages"));
    }
    for (i, message) in messages.iter().enumerate() {
        if i > 0 {
            request.label_new(i as u32 - 1);
        }
        let path = format!("messages[{i}]");
        obj(message, &path)?;
        match str_field(message, "role", &path)? {
            "user" => user(
                &message["content"],
                &path,
                options,
                &request.tools,
                &mut request.messages,
            )?,
            "assistant" => request
                .messages
                .push(assistant(&message["content"], &path)?),
            "system" => {
                let text = blocks_text(&message["content"], &path)?;
                request.messages.push(Message::System(text));
            }
            other => {
                return Err(ApiError::invalid(
                    format!("unsupported role '{other}'"),
                    format!("{path}.role"),
                ));
            }
        }
    }
    request.label_new(messages.len().saturating_sub(1) as u32);
    request.prefill = matches!(request.messages.last(), Some(Message::Assistant(_)));

    let choice = &body["tool_choice"];
    request.tool_choice = match choice["type"].as_str() {
        None | Some("auto") => ToolChoice::Auto,
        Some("any") => ToolChoice::Required,
        Some("none") => ToolChoice::None,
        Some("tool") => {
            let name = ToolChoice::Named(str_field(choice, "name", "tool_choice")?.to_string());
            declared(&request, &name, "tool_choice.name")?;
            name
        }
        Some(_) => {
            return Err(ApiError::invalid(
                "unknown tool_choice type",
                "tool_choice.type",
            ));
        }
    };
    request.parallel_tool_calls = !choice["disable_parallel_tool_use"]
        .as_bool()
        .unwrap_or(false);
    request.reasoning = thinking(body);
    if let Some(format) = body["output_config"].get("format").filter(|f| !f.is_null()) {
        let schema = format.get("schema").cloned().unwrap_or(Value::Null);
        check_schema(&schema, "output_config.format.schema")?;
        request.response_format = ResponseFormat::JsonSchema {
            name: "output".into(),
            schema,
            strict: true,
        };
    }
    // Anthropic requires `max_tokens`; a missing one gets the server default (leniency).
    request.max_output_tokens = opt_u32(body, "max_tokens")?;
    request.stop = stops(body, "stop_sequences")?;
    request.sampling.temperature = opt_f32(body, "temperature")?;
    request.sampling.top_p = opt_f32(body, "top_p")?;
    request.sampling.top_k = opt_u32(body, "top_k")?;
    request.sampling.repetition_penalty = opt_f32(body, "repetition_penalty")?;
    request.stream = opt_bool(body, "stream")?.unwrap_or(false);
    Ok(request)
}

/// FR-025: Claude Code's attribution block carries a per-request nonce (`cch=`) that would
/// defeat all prefix reuse. Its value is replaced by zeros of the same length; nothing else
/// changes, and only a leading system text starting with the attribution header qualifies.
pub fn normalize_billing_nonce(text: &str) -> String {
    const HEADER: &str = "x-anthropic-billing-header:";
    if !text.starts_with(HEADER) {
        return text.to_string();
    }
    let line_end = text.find('\n').unwrap_or(text.len());
    let Some(at) = text[..line_end].find("cch=") else {
        return text.to_string();
    };
    let start = at + "cch=".len();
    let end = text[start..line_end]
        .find(|c: char| c == ';' || c.is_whitespace())
        .map_or(line_end, |i| start + i);
    let mut out = String::with_capacity(text.len());
    out.push_str(&text[..start]);
    out.extend(std::iter::repeat_n('0', text[start..end].chars().count()));
    out.push_str(&text[end..]);
    out
}

/// `thinking`: enabled (with a budget, not enforced) or adaptive turn reasoning on; disabled or
/// absent off.
fn thinking(body: &Value) -> ReasoningCtl {
    match body["thinking"]["type"].as_str() {
        Some("enabled") => ReasoningCtl::On {
            budget: body["thinking"]["budget_tokens"]
                .as_u64()
                .map(|b| b.min(u64::from(u32::MAX)) as u32),
        },
        Some("adaptive") => ReasoningCtl::On { budget: None },
        _ => ReasoningCtl::Off,
    }
}

/// Text of a string or text blocks (`cache_control` ignored).
fn blocks_text(content: &Value, path: &str) -> Result<String, ApiError> {
    join_text_parts(
        content,
        path,
        |block| text_part(block, &["text"]),
        "only text blocks are supported here",
    )
}

/// A user message: tool results become tool messages in order; text around them stays user
/// text at its position. Adjacent text blocks are joined with a newline, as llama.cpp joins
/// typed text parts (Claude Code sends `<system-reminder>` blocks before the prompt).
fn user(
    content: &Value,
    path: &str,
    options: DecodeOptions,
    tools: &[ToolDef],
    out: &mut Vec<Message>,
) -> Result<(), ApiError> {
    let Value::Array(blocks) = content else {
        out.push(Message::User(blocks_text(content, path)?));
        return Ok(());
    };
    let mut text = String::new();
    for (j, block) in blocks.iter().enumerate() {
        let bpath = format!("{path}.content[{j}]");
        match block["type"].as_str() {
            Some("text") => {
                if !text.is_empty() {
                    text.push('\n');
                }
                text.push_str(block["text"].as_str().unwrap_or_default());
            }
            Some("tool_result") => {
                if !text.is_empty() {
                    out.push(Message::User(std::mem::take(&mut text)));
                }
                out.push(Message::Tool(tool_result(block, &bpath, options, tools)?));
            }
            _ => {
                return Err(ApiError::invalid(
                    "only text and tool_result blocks are supported in user messages",
                    bpath,
                ));
            }
        }
    }
    if !text.is_empty() {
        out.push(Message::User(text));
    }
    Ok(())
}

fn tool_result(
    block: &Value,
    path: &str,
    options: DecodeOptions,
    tools: &[ToolDef],
) -> Result<ToolResult, ApiError> {
    let call_id = str_field(block, "tool_use_id", path)?.to_string();
    let mut content = String::new();
    let mut images_omitted = 0;
    match &block["content"] {
        Value::Null => {}
        Value::String(s) => content.push_str(s),
        Value::Array(parts) => {
            for (k, part) in parts.iter().enumerate() {
                let ppath = format!("{path}.content[{k}]");
                match part["type"].as_str() {
                    Some("text") => content.push_str(part["text"].as_str().unwrap_or_default()),
                    Some("image") => {
                        content.push_str(tool_image(options, &ppath)?);
                        images_omitted += 1;
                    }
                    // A tool search result (FR-028): the referenced tool's definition is
                    // rendered here, so the tools region and the cached prefix stay unchanged.
                    Some("tool_reference") => {
                        let name = str_field(part, "tool_name", &ppath)?;
                        let tool = tools.iter().find(|t| t.name == name).ok_or_else(|| {
                            ApiError::invalid(
                                format!("Tool reference '{name}' not found in available tools"),
                                &ppath,
                            )
                        })?;
                        if !content.is_empty() && !content.ends_with('\n') {
                            content.push('\n');
                        }
                        content.push_str(&tool.definition().to_string());
                    }
                    _ => {
                        return Err(ApiError::invalid(
                            "unsupported tool_result content block",
                            ppath,
                        ));
                    }
                }
            }
        }
        _ => {
            return Err(ApiError::invalid(
                "tool_result content must be a string or blocks",
                format!("{path}.content"),
            ));
        }
    }
    Ok(ToolResult {
        call_id,
        content,
        is_error: block["is_error"].as_bool().unwrap_or(false),
        images_omitted,
    })
}

fn assistant(content: &Value, path: &str) -> Result<Message, ApiError> {
    let Value::Array(blocks) = content else {
        return Ok(Message::Assistant(vec![AssistantPart::Text(blocks_text(
            content, path,
        )?)]));
    };
    let mut parts = Vec::new();
    for (j, block) in blocks.iter().enumerate() {
        let bpath = format!("{path}.content[{j}]");
        match block["type"].as_str() {
            Some("text") => parts.push(AssistantPart::Text(
                block["text"].as_str().unwrap_or_default().to_string(),
            )),
            // Any signature is accepted and never checked (FR-016).
            Some("thinking") => parts.push(AssistantPart::Reasoning(
                block["thinking"].as_str().unwrap_or_default().to_string(),
            )),
            Some("redacted_thinking") => {}
            Some("tool_use") => parts.push(AssistantPart::ToolCall {
                id: str_field(block, "id", &bpath)?.to_string(),
                name: str_field(block, "name", &bpath)?.to_string(),
                raw_args: raw_args(&block["input"]),
            }),
            _ => {
                return Err(ApiError::invalid(
                    "unsupported assistant content block",
                    bpath,
                ));
            }
        }
    }
    Ok(Message::Assistant(parts))
}

pub fn stop_reason(finish: Finish) -> &'static str {
    match finish {
        Finish::Stop => "end_turn",
        Finish::StopSequence => "stop_sequence",
        Finish::Length | Finish::Repetition => "max_tokens",
        Finish::ContextFull => "model_context_window_exceeded",
        Finish::ToolCalls => "tool_use",
    }
}

/// Anthropic usage: `input_tokens` excludes cache reads.
fn usage(usage: Usage) -> Value {
    json!({
        "input_tokens": usage.input - usage.cached.min(usage.input),
        "cache_read_input_tokens": usage.cached,
        "cache_creation_input_tokens": 0,
        "output_tokens": usage.output,
    })
}

/// Anthropic error object; `overloaded_error` goes with status 529.
pub fn error_body(error: &ApiError) -> Value {
    let kind = match error.kind {
        ErrorKind::InvalidRequest | ErrorKind::ContextLengthExceeded => "invalid_request_error",
        ErrorKind::QueueFull
        | ErrorKind::QueueTimeout
        | ErrorKind::NotReady
        | ErrorKind::Overloaded => "overloaded_error",
        ErrorKind::PayloadTooLarge => "request_too_large",
        ErrorKind::Forbidden => "permission_error",
        ErrorKind::Unauthorized => "authentication_error",
        ErrorKind::Internal => "api_error",
    };
    json!({"type": "error", "error": {"type": kind, "message": error.message}})
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Open {
    None,
    Thinking,
    Text,
}

/// Streaming encoder: `message_start`, content blocks, `message_delta`, `message_stop`. It
/// also folds the events into the non-streaming message, block for block.
#[derive(Debug)]
pub struct StreamEncoder {
    id: String,
    model: String,
    index: u32,
    open: Open,
    /// Text of the open thinking or text block.
    open_text: String,
    /// A buffered call: id and name until its end decides tool_use or text.
    call: Option<(String, String)>,
    valid_calls: u32,
    /// Closed blocks, as the non-streaming body lists them.
    content: Vec<Value>,
    finish: Option<(Finish, Option<String>, Usage)>,
    done: bool,
}

impl StreamEncoder {
    pub fn new(model: &str) -> Self {
        Self {
            id: new_id("msg_"),
            model: model.to_string(),
            index: 0,
            open: Open::None,
            open_text: String::new(),
            call: None,
            valid_calls: 0,
            content: Vec::new(),
            finish: None,
            done: false,
        }
    }

    fn close(&mut self, frames: &mut Vec<String>) {
        let text = std::mem::take(&mut self.open_text);
        match self.open {
            Open::None => return,
            Open::Thinking => {
                let signature = reasoning::signature(&text);
                frames.push(sse::event("content_block_delta", &json!({"type": "content_block_delta",
                    "index": self.index, "delta": {"type": "signature_delta", "signature": signature}})));
                self.content
                    .push(json!({"type": "thinking", "thinking": text, "signature": signature}));
            }
            Open::Text => self.content.push(json!({"type": "text", "text": text})),
        }
        frames.push(sse::event(
            "content_block_stop",
            &json!({"type": "content_block_stop", "index": self.index}),
        ));
        self.index += 1;
        self.open = Open::None;
    }

    fn ensure(&mut self, kind: Open, frames: &mut Vec<String>) {
        if self.open == kind {
            return;
        }
        self.close(frames);
        let block = match kind {
            Open::Thinking => json!({"type": "thinking", "thinking": "", "signature": ""}),
            _ => json!({"type": "text", "text": ""}),
        };
        frames.push(sse::event(
            "content_block_start",
            &json!({"type": "content_block_start",
            "index": self.index, "content_block": block}),
        ));
        self.open = kind;
    }

    /// Text into a text block (opened if needed).
    fn text(&mut self, text: &str, frames: &mut Vec<String>) {
        self.ensure(Open::Text, frames);
        self.open_text.push_str(text);
        frames.push(sse::event(
            "content_block_delta",
            &json!({"type": "content_block_delta",
            "index": self.index, "delta": {"type": "text_delta", "text": text}}),
        ));
    }

    /// `tool_use` only when a call was shown as one (FR-004).
    fn stop_reason(&self, reason: Finish) -> &'static str {
        if reason == Finish::ToolCalls && self.valid_calls == 0 {
            stop_reason(Finish::Stop)
        } else {
            stop_reason(reason)
        }
    }
}

impl StreamEncode for StreamEncoder {
    fn encode(&mut self, event: &GenEvent) -> Vec<String> {
        let mut frames = Vec::new();
        if self.done {
            return frames;
        }
        match event {
            GenEvent::Started { input, cached, .. } => {
                let usage = usage(Usage {
                    input: *input,
                    cached: *cached,
                    output: 0,
                    reasoning: 0,
                });
                frames.push(sse::event(
                    "message_start",
                    &json!({"type": "message_start",
                    "message": {"id": self.id, "type": "message", "role": "assistant",
                        "model": self.model, "content": [], "stop_reason": null,
                        "stop_sequence": null, "usage": usage}}),
                ));
            }
            GenEvent::ReasoningDelta(text) => {
                self.ensure(Open::Thinking, &mut frames);
                self.open_text.push_str(text);
                frames.push(sse::event(
                    "content_block_delta",
                    &json!({"type": "content_block_delta",
                    "index": self.index, "delta": {"type": "thinking_delta", "thinking": text}}),
                ));
            }
            GenEvent::TextDelta(text) => self.text(text, &mut frames),
            GenEvent::ToolStart { id, name, .. } => {
                self.call = Some((id.clone(), name.clone()));
            }
            GenEvent::ToolArgsDelta { .. } => {}
            GenEvent::ToolEnd {
                raw_args, valid, ..
            } => {
                let Some((id, name)) = self.call.take() else {
                    return frames;
                };
                if !*valid {
                    // FR-004: an unparseable call is shown as text, never as a broken tool_use.
                    self.text(raw_args, &mut frames);
                    return frames;
                }
                self.close(&mut frames);
                self.valid_calls += 1;
                frames.push(sse::event(
                    "content_block_start",
                    &json!({"type": "content_block_start",
                    "index": self.index, "content_block": {"type": "tool_use", "id": id,
                        "name": name, "input": {}}}),
                ));
                frames.push(sse::event(
                    "content_block_delta",
                    &json!({"type": "content_block_delta",
                    "index": self.index, "delta": {"type": "input_json_delta",
                        "partial_json": raw_args}}),
                ));
                frames.push(sse::event(
                    "content_block_stop",
                    &json!({"type": "content_block_stop", "index": self.index}),
                ));
                let input = serde_json::from_str::<Value>(raw_args).unwrap_or(json!({}));
                self.content
                    .push(json!({"type": "tool_use", "id": id, "name": name, "input": input}));
                self.index += 1;
            }
            GenEvent::Finished {
                reason,
                stop_seq,
                usage: u,
            } => {
                self.close(&mut frames);
                self.finish = Some((*reason, stop_seq.clone(), *u));
                frames.push(sse::event(
                    "message_delta",
                    &json!({"type": "message_delta",
                    "delta": {"stop_reason": self.stop_reason(*reason), "stop_sequence": stop_seq},
                    "usage": usage(*u)}),
                ));
                frames.push(sse::event("message_stop", &json!({"type": "message_stop"})));
                self.done = true;
            }
            GenEvent::Failed(error) => {
                frames.push(sse::event("error", &error_body(error)));
                self.done = true;
            }
        }
        frames
    }

    fn keepalive(&mut self) -> String {
        sse::event("ping", &json!({"type": "ping"}))
    }

    fn snapshot(&self) -> Value {
        let (reason, stop_seq, u) =
            self.finish
                .clone()
                .unwrap_or((Finish::Stop, None, Usage::default()));
        json!({
            "id": self.id,
            "type": "message",
            "role": "assistant",
            "model": self.model,
            "content": self.content,
            "stop_reason": self.stop_reason(reason),
            "stop_sequence": stop_seq,
            "usage": usage(u),
        })
    }
}

/// `POST /v1/messages`
#[derive(Debug)]
pub struct Messages;

impl Api for Messages {
    const KIND: ApiKind = ApiKind::Messages;
    type Encoder = StreamEncoder;

    fn decode(body: &Value, options: DecodeOptions) -> Result<ChatRequest, ApiError> {
        decode(body, options)
    }

    fn encoder(model: &str, _: &ChatRequest) -> StreamEncoder {
        StreamEncoder::new(model)
    }

    fn error_body(error: &ApiError) -> Value {
        error_body(error)
    }
}

/// `POST /v1/messages/count_tokens`: exactly the tokens mrun would process; no queue.
pub async fn count_tokens(
    State(state): State<ServerState>,
    bytes: Result<Bytes, BytesRejection>,
) -> Response {
    let result = async {
        let request = decode(
            &handler::parse(&handler::body(bytes)?)?,
            state.decode_options(),
        )?;
        let loaded = handler::ready(&state)?;
        let prepared = crate::api::run::prepare(&loaded, &request).await?;
        Ok::<_, ApiError>(prepared.rendered.prompt.len())
    }
    .await;
    match result {
        Ok(tokens) => axum::Json(json!({"input_tokens": tokens})).into_response(),
        Err(error) => handler::error::<Messages>(&error),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn decode_ok(body: Value) -> ChatRequest {
        decode(&body, DecodeOptions::default()).unwrap()
    }

    #[test]
    fn repetition_penalty_is_read_and_checked() {
        let body = |penalty: Value| json!({"max_tokens": 8, "messages": [{"role": "user", "content": "hi"}], "repetition_penalty": penalty});
        let request = decode_ok(body(json!(1.2)));
        assert_eq!(request.sampling.repetition_penalty, Some(1.2));
        assert!(KNOWN.contains(&"repetition_penalty"));
        for bad in [json!(0), json!(-0.5)] {
            let error = decode_ok(body(bad))
                .sampling
                .resolve_valid(mrun_engine::Sampling::GREEDY)
                .unwrap_err();
            assert_eq!(error.param.as_deref(), Some("repetition_penalty"));
            assert_eq!(error.status(ApiKind::Messages).as_u16(), 400);
        }
        let error = decode(&body(json!("high")), DecodeOptions::default()).unwrap_err();
        assert_eq!(error.param.as_deref(), Some("repetition_penalty"));
    }

    fn frames_of(encoder: &mut StreamEncoder, events: &[GenEvent]) -> Vec<(String, Value)> {
        events
            .iter()
            .flat_map(|e| encoder.encode(e))
            .map(|f| {
                let mut lines = f.lines();
                let name = lines
                    .next()
                    .unwrap()
                    .trim_start_matches("event: ")
                    .to_string();
                let data = lines.next().unwrap().trim_start_matches("data: ");
                (name, serde_json::from_str(data).unwrap())
            })
            .collect()
    }

    #[test]
    fn billing_nonce_is_normalized_and_nothing_else() {
        let a = "x-anthropic-billing-header: cc_version=2.1.300.a1; cc_entrypoint=cli; cch=9f3c1;\nYou are Claude Code.";
        let b = "x-anthropic-billing-header: cc_version=2.1.300.a1; cc_entrypoint=cli; cch=0b7e2;\nYou are Claude Code.";
        let (na, nb) = (normalize_billing_nonce(a), normalize_billing_nonce(b));
        assert_eq!(na, nb);
        assert_eq!(na.len(), a.len());
        assert!(na.contains("cch=00000;"));
        // Any other change stays visible; texts without the header are untouched.
        let c = b.replace("cli", "sdk");
        assert_ne!(normalize_billing_nonce(&c), na);
        let plain = "cch=12345 in a normal prompt";
        assert_eq!(normalize_billing_nonce(plain), plain);
        // Two requests differing only in the nonce share the whole system text.
        let decode_system = |text: &str| {
            decode_ok(json!({"system": [{"type": "text", "text": text}, {"type": "text", "text": "cch=1"}],
                "messages": [{"role": "user", "content": "hi"}]})).system
        };
        assert_eq!(decode_system(a), decode_system(b));
        assert_eq!(decode_system(a)[1], "cch=1");
    }

    #[test]
    fn msg_system_string_and_blocks() {
        let request = decode_ok(json!({"max_tokens": 10,
            "system": [{"type": "text", "text": "A", "cache_control": {"type": "ephemeral"}},
                       {"type": "text", "text": "B"}],
            "messages": [{"role": "user", "content": "hi"}]}));
        assert_eq!(request.system, vec!["A", "B"]);
        assert!(!request.prefill);
    }

    #[test]
    fn msg_tool_result_blocks_with_text_and_errors() {
        let request = decode_ok(json!({"messages": [
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": [
                {"type": "thinking", "thinking": "hm", "signature": "anything"},
                {"type": "tool_use", "id": "toolu_1", "name": "Read", "input": {"p": 1}}]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "toolu_1", "is_error": true,
                 "content": [{"type": "text", "text": "no"},
                             {"type": "image", "source": {"type": "base64", "data": "AA"}}]},
                {"type": "text", "text": "try again"}]}]}));
        assert_eq!(request.messages.len(), 4);
        let Message::Tool(result) = &request.messages[2] else {
            panic!()
        };
        assert!(result.is_error);
        assert_eq!(result.images_omitted, 1);
        assert!(result.content.starts_with("no[image omitted"));
        assert_eq!(request.messages[3], Message::user_text("try again"));
        // msg_reject_tool_images_flag
        let strict = DecodeOptions {
            reject_tool_images: true,
        };
        let body = json!({"messages": [{"role": "user", "content": [{"type": "tool_result",
            "tool_use_id": "t", "content": [{"type": "image"}]}]}]});
        let error = decode(&body, strict).unwrap_err();
        assert_eq!(
            error.param.as_deref(),
            Some("messages[0].content[0].content[0]")
        );
    }

    #[test]
    fn msg_user_text_blocks_join_with_a_newline() {
        let request = decode_ok(json!({"messages": [{"role": "user", "content": [
            {"type": "text", "text": "<system-reminder>r</system-reminder>"},
            {"type": "text", "text": "Fix the bug."}]}]}));
        assert_eq!(
            request.messages,
            [Message::user_text(
                "<system-reminder>r</system-reminder>\nFix the bug."
            )]
        );
    }

    #[test]
    fn non_streaming_body_is_the_stream_snapshot() {
        // An invalid call after text merges into the open text block in both forms.
        let mut encoder = StreamEncoder::new("m");
        for event in [
            GenEvent::Started {
                input: 5,
                cached: 2,
                max_tokens: 100,
                miss: None,
            },
            GenEvent::ReasoningDelta("hm".into()),
            GenEvent::TextDelta("Trying: ".into()),
            GenEvent::ToolStart {
                idx: 0,
                id: "toolu_1".into(),
                name: "Edit".into(),
            },
            GenEvent::ToolEnd {
                idx: 0,
                raw_args: "{broken".into(),
                valid: false,
            },
            GenEvent::Finished {
                reason: Finish::ToolCalls,
                stop_seq: None,
                usage: Usage {
                    input: 5,
                    cached: 2,
                    output: 4,
                    reasoning: 1,
                },
            },
        ] {
            encoder.encode(&event);
        }
        let body = encoder.snapshot();
        assert_eq!(body["content"].as_array().unwrap().len(), 2, "{body}");
        assert_eq!(body["content"][0]["type"], "thinking");
        assert_eq!(body["content"][1]["text"], "Trying: {broken");
        assert_eq!(body["stop_reason"], "end_turn");
        assert_eq!(body["usage"]["input_tokens"], 3);
        assert!(body["id"].as_str().unwrap().starts_with("msg_"));
    }

    #[test]
    fn msg_thinking_modes_and_choices() {
        let base = json!({"messages": [{"role": "user", "content": "x"}],
            "tools": [{"name": "Read", "input_schema": {"type": "object"}}]});
        let mut body = base.clone();
        body["thinking"] = json!({"type": "enabled", "budget_tokens": 2048});
        body["tool_choice"] = json!({"type": "any", "disable_parallel_tool_use": true});
        let request = decode_ok(body);
        assert_eq!(request.reasoning, ReasoningCtl::On { budget: Some(2048) });
        assert_eq!(request.tool_choice, ToolChoice::Required);
        assert!(!request.parallel_tool_calls);
        let mut body = base.clone();
        body["thinking"] = json!({"type": "adaptive"});
        assert!(decode_ok(body).reasoning.is_on());
        let mut body = base;
        body["thinking"] = json!({"type": "disabled"});
        body["diagnostics"] = json!({"x": 1});
        body["context_management"] = json!({"edits": []});
        assert_eq!(decode_ok(body).reasoning, ReasoningCtl::Off);
    }

    #[test]
    fn msg_server_tools_and_mcp_servers_rejected() {
        let body = json!({"messages": [{"role": "user", "content": "x"}],
            "tools": [{"type": "web_search_20250305", "name": "web_search"}]});
        assert!(decode(&body, DecodeOptions::default()).is_err());
        let body = json!({"messages": [{"role": "user", "content": "x"}],
            "mcp_servers": [{"url": "https://x"}]});
        assert_eq!(
            decode(&body, DecodeOptions::default())
                .unwrap_err()
                .param
                .as_deref(),
            Some("mcp_servers")
        );
    }

    #[test]
    fn msg_stream_full_event_order_with_thinking_and_tool_use() {
        let mut encoder = StreamEncoder::new("m");
        let frames = frames_of(
            &mut encoder,
            &[
                GenEvent::Started {
                    input: 50,
                    cached: 40,
                    max_tokens: 100,
                    miss: None,
                },
                GenEvent::ReasoningDelta("Need it.".into()),
                GenEvent::ToolStart {
                    idx: 0,
                    id: "toolu_1".into(),
                    name: "Read".into(),
                },
                GenEvent::ToolArgsDelta {
                    idx: 0,
                    json: "{\"p\": 1}".into(),
                },
                GenEvent::ToolEnd {
                    idx: 0,
                    raw_args: "{\"p\": 1}".into(),
                    valid: true,
                },
                GenEvent::Finished {
                    reason: Finish::ToolCalls,
                    stop_seq: None,
                    usage: Usage {
                        input: 50,
                        cached: 40,
                        output: 7,
                        reasoning: 3,
                    },
                },
            ],
        );
        let names: Vec<&str> = frames.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(
            names,
            vec![
                "message_start",
                "content_block_start",
                "content_block_delta",
                "content_block_delta",
                "content_block_stop",
                "content_block_start",
                "content_block_delta",
                "content_block_stop",
                "message_delta",
                "message_stop",
            ]
        );
        // msg_usage_input_excludes_cache_read
        assert_eq!(frames[0].1["message"]["usage"]["input_tokens"], 10);
        assert_eq!(
            frames[0].1["message"]["usage"]["cache_read_input_tokens"],
            40
        );
        assert_eq!(frames[3].1["delta"]["type"], "signature_delta");
        assert_eq!(frames[5].1["content_block"]["type"], "tool_use");
        assert_eq!(frames[6].1["delta"]["partial_json"], "{\"p\": 1}");
        assert_eq!(frames[8].1["delta"]["stop_reason"], "tool_use");
        assert_eq!(frames[8].1["usage"]["output_tokens"], 7);
    }

    #[test]
    fn msg_nonstrict_invalid_args_as_text() {
        let mut encoder = StreamEncoder::new("m");
        let frames = frames_of(
            &mut encoder,
            &[
                GenEvent::Started {
                    input: 5,
                    cached: 0,
                    max_tokens: 100,
                    miss: None,
                },
                GenEvent::ToolStart {
                    idx: 0,
                    id: "toolu_1".into(),
                    name: "Edit".into(),
                },
                GenEvent::ToolEnd {
                    idx: 0,
                    raw_args: "{broken".into(),
                    valid: false,
                },
                GenEvent::Finished {
                    reason: Finish::ToolCalls,
                    stop_seq: None,
                    usage: Usage::default(),
                },
            ],
        );
        assert_eq!(frames[1].1["content_block"]["type"], "text");
        assert_eq!(frames[2].1["delta"]["text"], "{broken");
        assert_eq!(frames[4].1["delta"]["stop_reason"], "end_turn");
    }

    #[test]
    fn msg_stream_error_event_shape() {
        let mut encoder = StreamEncoder::new("m");
        let frames = frames_of(
            &mut encoder,
            &[GenEvent::Failed(ApiError::new(
                ErrorKind::QueueTimeout,
                "waited",
            ))],
        );
        assert_eq!(frames[0].0, "error");
        assert_eq!(frames[0].1["error"]["type"], "overloaded_error");
    }
}
