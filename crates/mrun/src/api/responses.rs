//! OpenAI Responses, stateless subset (`POST /v1/responses`): decoder and encoders (002 research
//! §1c, §2 Codex).
//!
//! Nothing is stored: `previous_response_id`, `conversation` and `item_reference` are refused,
//! and reasoning survives a turn only through `encrypted_content` (FR-016).

use crate::api::chat_completions::{error_body, openai_effort};
use crate::api::collect::{Block, Collected};
use crate::api::decode::{
    DecodeOptions, arr, check_limits, check_schema, check_tool, obj, opt_bool, opt_f32, opt_str,
    opt_u32, str_field, tool_image,
};
use crate::api::handler::{self, ApiShape, StreamEncode};
use crate::api::model::{
    ApiError, ApiKind, AssistantPart, ChatRequest, ErrorKind, Finish, GenEvent, Message,
    NAMESPACE_SEP, ResponseFormat, ToolChoice, ToolDef, ToolResult, Usage, UserPart,
};
use crate::api::sse::{self, new_id};
use crate::chat::reasoning;
use crate::cli::serve::ServerState;
use axum::body::Bytes;
use axum::extract::State;
use axum::http::HeaderMap;
use axum::response::Response;
use serde_json::{Value, json};
use std::collections::HashMap;

pub fn decode(body: &Value, options: DecodeOptions) -> Result<ChatRequest, ApiError> {
    obj(body, "body")?;
    let mut request = ChatRequest::new(ApiKind::Responses);
    request.model = opt_str(body, "model", "body")?
        .unwrap_or_default()
        .to_string();
    for key in ["previous_response_id", "conversation"] {
        if !body[key].is_null() {
            return Err(ApiError::invalid(
                format!("{key} needs stored state; mrun serves stateless requests only"),
                key,
            ));
        }
    }
    if let Some(instructions) = opt_str(body, "instructions", "body")?
        && !instructions.is_empty()
    {
        request.system.push(instructions.to_string());
    }

    if let Some(tools) = body.get("tools").filter(|t| !t.is_null()) {
        for (i, tool) in arr(tools, "tools")?.iter().enumerate() {
            let path = format!("tools[{i}]");
            obj(tool, &path)?;
            match str_field(tool, "type", &path)? {
                "function" => request.tools.push(function_tool(tool, &path, None)?),
                // MCP and sub-agent tools (FR-027): flattened to `<namespace>__<tool>`.
                "namespace" => {
                    let namespace = str_field(tool, "name", &path)?;
                    for (j, inner) in arr(&tool["tools"], &format!("{path}.tools"))?
                        .iter()
                        .enumerate()
                    {
                        let ipath = format!("{path}.tools[{j}]");
                        request
                            .tools
                            .push(function_tool(inner, &ipath, Some(namespace))?);
                    }
                }
                // Codex's client-executed tool search: the model calls it like a function; the
                // call goes back as a `tool_search_call` item (FR-028).
                "tool_search" => request.tools.push(ToolDef {
                    name: TOOL_SEARCH.into(),
                    description: tool["description"]
                        .as_str()
                        .unwrap_or("Search the available tools by keyword; matching tool definitions are returned and can be called afterwards.")
                        .to_string(),
                    schema: match &tool["parameters"] {
                        Value::Null => json!({"type": "object",
                            "properties": {"query": {"type": "string"}}, "required": ["query"]}),
                        schema => schema.clone(),
                    },
                    ..Default::default()
                }),
                web if web.starts_with("web_search") => {
                    // A hosted tool mrun cannot run: dropped so the request still works.
                    eprintln!("mrun: tool-dropped (web_search)");
                }
                other => {
                    return Err(ApiError::invalid(
                        format!("tool type '{other}' is not supported; only function, namespace and tool_search tools are"),
                        format!("{path}.type"),
                    ));
                }
            }
        }
    }

    match &body["input"] {
        Value::String(text) => request.messages.push(Message::user_text(text.clone())),
        Value::Array(items) => decode_items(items, options, &mut request)?,
        Value::Null => {}
        _ => {
            return Err(ApiError::invalid(
                "input must be a string or an array",
                "input",
            ));
        }
    }
    if request.messages.is_empty() && request.system.is_empty() {
        return Err(ApiError::invalid("input must not be empty", "input"));
    }

    request.tool_choice = match &body["tool_choice"] {
        Value::Null => ToolChoice::Auto,
        Value::String(s) => match s.as_str() {
            "auto" => ToolChoice::Auto,
            "none" => ToolChoice::None,
            "required" => ToolChoice::Required,
            _ => return Err(ApiError::invalid("unknown tool_choice", "tool_choice")),
        },
        choice if choice["type"] == "function" => {
            ToolChoice::Named(str_field(choice, "name", "tool_choice")?.to_string())
        }
        _ => {
            return Err(ApiError::invalid(
                "tool_choice must be auto, none, required or a function",
                "tool_choice",
            ));
        }
    };
    if let ToolChoice::Named(name) = &request.tool_choice
        && request.tool(name).is_none()
    {
        return Err(ApiError::invalid(
            format!("tool_choice names an undeclared tool '{name}'"),
            "tool_choice",
        ));
    }
    request.parallel_tool_calls = opt_bool(body, "parallel_tool_calls")?.unwrap_or(true);
    request.response_format = match &body["text"]["format"] {
        Value::Null => ResponseFormat::Text,
        format => match format["type"].as_str() {
            Some("text") => ResponseFormat::Text,
            Some("json_object") => ResponseFormat::JsonObject,
            Some("json_schema") => {
                let schema = format["schema"].clone();
                check_schema(&schema, "text.format.schema")?;
                ResponseFormat::JsonSchema {
                    name: format["name"].as_str().unwrap_or("response").to_string(),
                    schema,
                    strict: format["strict"].as_bool().unwrap_or(false),
                }
            }
            _ => {
                return Err(ApiError::invalid(
                    "unsupported text.format type",
                    "text.format.type",
                ));
            }
        },
    };
    request.max_output_tokens = opt_u32(body, "max_output_tokens")?;
    request.sampling.temperature = opt_f32(body, "temperature")?;
    request.sampling.top_p = opt_f32(body, "top_p")?;
    // Codex always sends `reasoning` (without an effort for unknown models): only an effort
    // turns thinking on.
    request.reasoning = match &body["reasoning"] {
        Value::Null => openai_effort(None),
        reasoning => {
            obj(reasoning, "reasoning")?;
            openai_effort(opt_str(reasoning, "effort", "reasoning")?)
        }
    };
    request.stream = opt_bool(body, "stream")?.unwrap_or(false);
    check_limits(&request)?;
    Ok(request)
}

/// Input items in order; consecutive assistant output (reasoning, messages, function calls)
/// forms one assistant turn.
fn decode_items(
    items: &[Value],
    options: DecodeOptions,
    request: &mut ChatRequest,
) -> Result<(), ApiError> {
    let mut assistant: Vec<AssistantPart> = Vec::new();
    let mut assistant_start = 0u32;
    let flush = |assistant: &mut Vec<AssistantPart>, messages: &mut Vec<Message>| {
        // Reasoning alone is no turn: there is nothing to attach it to.
        if assistant
            .iter()
            .any(|p| !matches!(p, AssistantPart::Reasoning(_)))
        {
            messages.push(Message::Assistant(std::mem::take(assistant)));
        }
        assistant.clear();
    };
    for (i, item) in items.iter().enumerate() {
        label(request, i.saturating_sub(1) as u32, assistant_start);
        let assistant_was_empty = assistant.is_empty();
        let path = format!("input[{i}]");
        obj(item, &path)?;
        let kind = match opt_str(item, "type", &path)? {
            Some(kind) => kind,
            None if !item["role"].is_null() => "message",
            None => return Err(ApiError::invalid("input item has no type", path)),
        };
        match kind {
            "message" => {
                let role = str_field(item, "role", &path)?;
                match role {
                    "assistant" => {
                        let text = content_text(&item["content"], &path, false)?;
                        if !text.is_empty() {
                            assistant.push(AssistantPart::Text(text));
                        }
                    }
                    "user" | "system" | "developer" => {
                        flush(&mut assistant, &mut request.messages);
                        let text = content_text(&item["content"], &path, role == "user")?;
                        if role == "user" {
                            request
                                .messages
                                .push(Message::User(vec![UserPart::Text(text)]));
                        } else if request.messages.is_empty() {
                            request.system.push(text);
                        } else {
                            request.messages.push(Message::System(text));
                        }
                    }
                    other => {
                        return Err(ApiError::invalid(
                            format!("unsupported role '{other}'"),
                            format!("{path}.role"),
                        ));
                    }
                }
            }
            "function_call" => {
                let raw_args = match &item["arguments"] {
                    Value::String(s) => s.clone(),
                    Value::Null => "{}".to_string(),
                    other => other.to_string(),
                };
                let name = str_field(item, "name", &path)?;
                let name = match opt_str(item, "namespace", &path)? {
                    Some(namespace) => format!("{namespace}{NAMESPACE_SEP}{name}"),
                    None => name.to_string(),
                };
                assistant.push(AssistantPart::ToolCall {
                    id: str_field(item, "call_id", &path)?.to_string(),
                    name,
                    raw_args,
                });
            }
            "tool_search_call" => {
                let raw_args = match &item["arguments"] {
                    Value::String(s) => s.clone(),
                    Value::Null => "{}".to_string(),
                    other => other.to_string(),
                };
                assistant.push(AssistantPart::ToolCall {
                    id: str_field(item, "call_id", &path)?.to_string(),
                    name: TOOL_SEARCH.to_string(),
                    raw_args,
                });
            }
            // The tools a search returned: their definitions are rendered here, in the tool
            // result, and they are callable from now on without entering the tools region.
            "tool_search_output" => {
                flush(&mut assistant, &mut request.messages);
                let call_id = str_field(item, "call_id", &path)?.to_string();
                let mut found = Vec::new();
                for (j, tool) in arr(&item["tools"], &format!("{path}.tools"))?
                    .iter()
                    .enumerate()
                {
                    let tpath = format!("{path}.tools[{j}]");
                    match tool["type"].as_str() {
                        Some("namespace") => {
                            let namespace = str_field(tool, "name", &tpath)?;
                            for (k, inner) in arr(&tool["tools"], &format!("{tpath}.tools"))?
                                .iter()
                                .enumerate()
                            {
                                found.push(function_tool(
                                    inner,
                                    &format!("{tpath}.tools[{k}]"),
                                    Some(namespace),
                                )?);
                            }
                        }
                        _ => found.push(function_tool(tool, &tpath, None)?),
                    }
                }
                let mut content = String::new();
                for mut tool in found {
                    tool.deferred = true;
                    content.push_str(&tool.definition().to_string());
                    content.push('\n');
                    if !request.tools.iter().any(|t| t.name == tool.name) {
                        request.tools.push(tool);
                    }
                }
                request.messages.push(Message::Tool(ToolResult {
                    call_id,
                    content: content.trim_end().to_string(),
                    is_error: false,
                    images_omitted: 0,
                }));
            }
            "reasoning" => {
                if let Some(text) = reasoning_text(item) {
                    assistant.push(AssistantPart::Reasoning(text));
                }
            }
            "function_call_output" => {
                flush(&mut assistant, &mut request.messages);
                let call_id = str_field(item, "call_id", &path)?.to_string();
                let (content, images_omitted) = output_content(&item["output"], &path, options)?;
                request.messages.push(Message::Tool(ToolResult {
                    call_id,
                    content,
                    is_error: false,
                    images_omitted,
                }));
            }
            "item_reference" => {
                return Err(ApiError::invalid(
                    "item_reference needs stored state; send the item itself",
                    path,
                ));
            }
            other => {
                return Err(ApiError::invalid(
                    format!("input item type '{other}' is not supported"),
                    format!("{path}.type"),
                ));
            }
        }
        if assistant_was_empty && !assistant.is_empty() {
            assistant_start = i as u32;
        }
    }
    flush(&mut assistant, &mut request.messages);
    label(
        request,
        items.len().saturating_sub(1) as u32,
        assistant_start,
    );
    Ok(())
}

/// API indices for messages added since the last call: an assistant turn takes the index of
/// its first item, everything else `index`.
fn label(request: &mut ChatRequest, index: u32, assistant_start: u32) {
    for k in request.api_index.len()..request.messages.len() {
        let i = match request.messages[k] {
            Message::Assistant(_) => assistant_start,
            _ => index,
        };
        request.api_index.push(i);
    }
}

/// A function tool, flattened into `namespace` when it came from one.
fn function_tool(tool: &Value, path: &str, namespace: Option<&str>) -> Result<ToolDef, ApiError> {
    let name = str_field(tool, "name", path)?;
    let def = ToolDef {
        name: match namespace {
            Some(namespace) => format!("{namespace}{NAMESPACE_SEP}{name}"),
            None => name.to_string(),
        },
        description: opt_str(tool, "description", path)?
            .unwrap_or_default()
            .to_string(),
        schema: match &tool["parameters"] {
            Value::Null => json!({"type": "object", "properties": {}}),
            schema => schema.clone(),
        },
        strict: tool["strict"].as_bool().unwrap_or(false),
        deferred: tool["defer_loading"].as_bool().unwrap_or(false),
        namespace: namespace.map(str::to_string),
    };
    check_tool(&def, path)?;
    Ok(def)
}

/// Reasoning text of a replayed item: from mrun's `encrypted_content`, else from
/// `reasoning_text` content; another provider's blob is dropped.
fn reasoning_text(item: &Value) -> Option<String> {
    if let Some(text) = item["encrypted_content"]
        .as_str()
        .and_then(reasoning::decode_content)
    {
        return Some(text);
    }
    let text: String = item["content"]
        .as_array()?
        .iter()
        .filter(|p| p["type"] == "reasoning_text")
        .filter_map(|p| p["text"].as_str())
        .collect();
    (!text.is_empty()).then_some(text)
}

/// Message content: a string, or text parts joined. Media in user content is refused with its
/// path (FR-020).
fn content_text(content: &Value, path: &str, user: bool) -> Result<String, ApiError> {
    match content {
        Value::Null => Ok(String::new()),
        Value::String(s) => Ok(s.clone()),
        Value::Array(parts) => {
            let mut text = String::new();
            for (j, part) in parts.iter().enumerate() {
                match part["type"].as_str() {
                    Some("input_text" | "output_text" | "text") => {
                        text.push_str(part["text"].as_str().unwrap_or_default());
                    }
                    Some("refusal") if !user => {
                        text.push_str(part["refusal"].as_str().unwrap_or_default());
                    }
                    _ => {
                        return Err(ApiError::invalid(
                            "only text content is supported (the model has no vision input)",
                            format!("{path}.content[{j}]"),
                        ));
                    }
                }
            }
            Ok(text)
        }
        _ => Err(ApiError::invalid(
            "content must be a string or an array",
            format!("{path}.content"),
        )),
    }
}

/// `function_call_output.output`: a string, or text parts joined with images replaced per
/// FR-020.
fn output_content(
    output: &Value,
    path: &str,
    options: DecodeOptions,
) -> Result<(String, u32), ApiError> {
    match output {
        Value::Null => Ok((String::new(), 0)),
        Value::String(s) => Ok((s.clone(), 0)),
        Value::Array(parts) => {
            let mut text = String::new();
            let mut images = 0;
            for (j, part) in parts.iter().enumerate() {
                let part_path = format!("{path}.output[{j}]");
                match part["type"].as_str() {
                    Some("input_text" | "output_text" | "text") => {
                        text.push_str(part["text"].as_str().unwrap_or_default());
                    }
                    Some("input_image") => {
                        text.push_str(tool_image(options, &part_path)?);
                        images += 1;
                    }
                    _ => {
                        return Err(ApiError::invalid(
                            "only text and image output parts are supported",
                            part_path,
                        ));
                    }
                }
            }
            Ok((text, images))
        }
        _ => Err(ApiError::invalid(
            "output must be a string or an array",
            format!("{path}.output"),
        )),
    }
}

pub fn usage(usage: Usage) -> Value {
    json!({
        "input_tokens": usage.input,
        "input_tokens_details": {"cached_tokens": usage.cached},
        "output_tokens": usage.output,
        "output_tokens_details": {"reasoning_tokens": usage.reasoning},
        "total_tokens": usage.input + usage.output,
    })
}

/// `error.code` of a failed response.
fn error_code(error: &ApiError) -> &'static str {
    match error.kind {
        ErrorKind::ContextLengthExceeded => "context_length_exceeded",
        ErrorKind::QueueTimeout | ErrorKind::Overloaded | ErrorKind::NotReady => "server_busy",
        ErrorKind::QueueFull => "rate_limit_exceeded",
        ErrorKind::InvalidRequest | ErrorKind::PayloadTooLarge | ErrorKind::Forbidden => {
            "invalid_request"
        }
        ErrorKind::Internal => "server_error",
    }
}

fn truncated(finish: Finish) -> bool {
    matches!(finish, Finish::Length | Finish::ContextFull)
}

fn status(complete: bool) -> &'static str {
    if complete { "completed" } else { "incomplete" }
}

fn reasoning_item(id: &str, text: &str) -> Value {
    json!({"type": "reasoning", "id": id, "summary": [],
        "content": [{"type": "reasoning_text", "text": text}],
        "encrypted_content": reasoning::encode_content(text)})
}

fn output_text(text: &str) -> Value {
    json!({"type": "output_text", "text": text, "annotations": []})
}

fn message_item(id: &str, text: &str, complete: bool) -> Value {
    json!({"type": "message", "id": id, "role": "assistant", "status": status(complete),
        "content": [output_text(text)]})
}

/// How calls go back to the client: namespaced tools split into `namespace` and `name`
/// (FR-027), and calls of Codex's client-executed `tool_search` as `tool_search_call` items.
#[derive(Clone, Debug, Default)]
pub struct Naming {
    namespaced: HashMap<String, (String, String)>,
    tool_search: bool,
}

/// Name of the synthetic tool standing for Codex's `tool_search` (client-executed).
pub const TOOL_SEARCH: &str = "tool_search";

impl Naming {
    pub fn of(request: &ChatRequest) -> Self {
        let mut naming = Self::default();
        for tool in &request.tools {
            if let Some((namespace, name)) = tool.split_name() {
                naming
                    .namespaced
                    .insert(tool.name.clone(), (namespace.to_string(), name.to_string()));
            }
            naming.tool_search |= tool.name == TOOL_SEARCH && tool.namespace.is_none();
        }
        naming
    }

    fn item(&self, id: &str, call_id: &str, name: &str, arguments: &str, status: &str) -> Value {
        if self.tool_search && name == TOOL_SEARCH {
            let arguments: Value = serde_json::from_str(arguments).unwrap_or_else(|_| json!({}));
            return json!({"type": "tool_search_call", "id": id, "call_id": call_id,
                "execution": "client", "arguments": arguments, "status": status});
        }
        let mut item = json!({"type": "function_call", "id": id, "call_id": call_id,
            "name": name, "arguments": arguments, "status": status});
        if let Some((namespace, tool)) = self.namespaced.get(name) {
            item["name"] = tool.as_str().into();
            item["namespace"] = namespace.as_str().into();
        }
        item
    }
}

fn call_item(
    naming: &Naming,
    id: &str,
    call_id: &str,
    name: &str,
    arguments: &str,
    complete: bool,
) -> Value {
    naming.item(id, call_id, name, arguments, status(complete))
}

/// The response object of bodies and lifecycle events.
fn response_object(
    id: &str,
    created_at: u64,
    model: &str,
    status: &str,
    output: Vec<Value>,
    usage: Option<Usage>,
) -> Value {
    json!({
        "id": id,
        "object": "response",
        "created_at": created_at,
        "status": status,
        "error": null,
        "incomplete_details": if status == "incomplete" {
            json!({"reason": "max_output_tokens"})
        } else {
            Value::Null
        },
        "model": model,
        "output": output,
        "store": false,
        "usage": usage.map(self::usage),
    })
}

/// Non-streaming body.
pub fn response(model: &str, collected: &Collected, naming: &Naming) -> Value {
    let (finish, _, usage) =
        collected
            .finish
            .clone()
            .unwrap_or((Finish::Stop, None, Usage::default()));
    let complete = !truncated(finish);
    let last = collected.blocks.len().saturating_sub(1);
    let output = collected
        .blocks
        .iter()
        .enumerate()
        .map(|(i, block)| {
            // Only the item open when the limit was reached is incomplete.
            let item_complete = complete || i != last;
            match block {
                Block::Reasoning(text) => reasoning_item(&new_id("rs_"), text),
                Block::Text(text) => message_item(&new_id("msg_"), text, item_complete),
                Block::Call(call) => call_item(
                    naming,
                    &new_id("fc_"),
                    &call.id,
                    &call.name,
                    &call.raw_args,
                    item_complete,
                ),
            }
        })
        .collect();
    response_object(
        &new_id("resp_"),
        sse::unix_now(),
        model,
        status(complete),
        output,
        Some(usage),
    )
}

/// The output item being streamed.
#[derive(Debug)]
enum Open {
    Reasoning {
        id: String,
        text: String,
    },
    Message {
        id: String,
        text: String,
    },
    Call {
        id: String,
        call_id: String,
        name: String,
        args: String,
        /// Argument deltas were sent.
        streamed: bool,
        /// `ToolEnd` arrived; `output_item.done` waits for the next event, which tells whether
        /// the call was cut by the output limit.
        ended: bool,
    },
}

/// Streaming encoder: `response.created` with the headers, then item events with increasing
/// `sequence_number`s, and one terminal `response.completed`, `response.incomplete` or
/// `response.failed`.
#[derive(Debug)]
pub struct StreamEncoder {
    id: String,
    created_at: u64,
    model: String,
    seq: u64,
    output: Vec<Value>,
    open: Option<Open>,
    done: bool,
    naming: Naming,
}

impl StreamEncoder {
    pub fn new(model: &str, naming: Naming) -> Self {
        Self {
            naming,
            id: new_id("resp_"),
            created_at: sse::unix_now(),
            model: model.to_string(),
            seq: 0,
            output: Vec::new(),
            open: None,
            done: false,
        }
    }

    fn event(&mut self, name: &str, mut body: Value) -> String {
        body["type"] = name.into();
        body["sequence_number"] = self.seq.into();
        self.seq += 1;
        sse::event(name, &body)
    }

    fn object(&self, status: &str, usage: Option<Usage>) -> Value {
        response_object(
            &self.id,
            self.created_at,
            &self.model,
            status,
            self.output.clone(),
            usage,
        )
    }

    fn index(&self) -> usize {
        self.output.len()
    }

    /// Closes the open item; `complete: false` when the output limit cut it.
    fn close(&mut self, complete: bool, frames: &mut Vec<String>) {
        let Some(open) = self.open.take() else { return };
        let index = self.index();
        let item = match open {
            Open::Reasoning { id, text } => {
                frames.push(self.event(
                    "response.reasoning_text.done",
                    json!({"item_id": id, "output_index": index, "content_index": 0,
                        "text": text}),
                ));
                reasoning_item(&id, &text)
            }
            Open::Message { id, text } => {
                frames.push(self.event(
                    "response.output_text.done",
                    json!({"item_id": id, "output_index": index, "content_index": 0,
                        "text": text, "logprobs": []}),
                ));
                frames.push(self.event(
                    "response.content_part.done",
                    json!({"item_id": id, "output_index": index, "content_index": 0,
                        "part": output_text(&text)}),
                ));
                message_item(&id, &text, complete)
            }
            Open::Call {
                id,
                call_id,
                name,
                args,
                ended,
                ..
            } => {
                if !ended {
                    frames.push(self.event(
                        "response.function_call_arguments.done",
                        json!({"item_id": id, "output_index": index, "arguments": args}),
                    ));
                }
                call_item(&self.naming, &id, &call_id, &name, &args, complete && ended)
            }
        };
        frames.push(self.event(
            "response.output_item.done",
            json!({"output_index": index, "item": item}),
        ));
        self.output.push(item);
    }

    /// Frames for one event; nothing after the terminal event.
    pub fn encode(&mut self, event: &GenEvent) -> Vec<String> {
        if self.done {
            return Vec::new();
        }
        let mut frames = Vec::new();
        // An ended call is closed by whatever comes next, unless that is the end of the output.
        if matches!(self.open, Some(Open::Call { ended: true, .. }))
            && !matches!(event, GenEvent::Finished { .. } | GenEvent::Started { .. })
        {
            self.close(true, &mut frames);
        }
        match event {
            GenEvent::Started { .. } => {}
            GenEvent::ReasoningDelta(delta) => {
                if delta.is_empty() {
                    return frames;
                }
                if !matches!(self.open, Some(Open::Reasoning { .. })) {
                    self.close(true, &mut frames);
                    let id = new_id("rs_");
                    let index = self.index();
                    frames.push(self.event(
                        "response.output_item.added",
                        json!({"output_index": index, "item": {"type": "reasoning", "id": id,
                            "summary": [], "content": []}}),
                    ));
                    self.open = Some(Open::Reasoning {
                        id,
                        text: String::new(),
                    });
                }
                let index = self.index();
                if let Some(Open::Reasoning { id, text }) = &mut self.open {
                    text.push_str(delta);
                    let id = id.clone();
                    frames.push(self.event(
                        "response.reasoning_text.delta",
                        json!({"item_id": id, "output_index": index, "content_index": 0,
                            "delta": delta}),
                    ));
                }
            }
            GenEvent::TextDelta(delta) => {
                if delta.is_empty() {
                    return frames;
                }
                if !matches!(self.open, Some(Open::Message { .. })) {
                    self.close(true, &mut frames);
                    let id = new_id("msg_");
                    let index = self.index();
                    frames.push(self.event(
                        "response.output_item.added",
                        json!({"output_index": index, "item": {"type": "message", "id": id,
                            "role": "assistant", "status": "in_progress", "content": []}}),
                    ));
                    frames.push(self.event(
                        "response.content_part.added",
                        json!({"item_id": id, "output_index": index, "content_index": 0,
                            "part": output_text("")}),
                    ));
                    self.open = Some(Open::Message {
                        id,
                        text: String::new(),
                    });
                }
                let index = self.index();
                if let Some(Open::Message { id, text }) = &mut self.open {
                    text.push_str(delta);
                    let id = id.clone();
                    frames.push(self.event(
                        "response.output_text.delta",
                        json!({"item_id": id, "output_index": index, "content_index": 0,
                            "delta": delta, "logprobs": []}),
                    ));
                }
            }
            GenEvent::ToolStart {
                id: call_id, name, ..
            } => {
                self.close(true, &mut frames);
                let id = new_id("fc_");
                let index = self.index();
                let item = self.naming.item(&id, call_id, name, "", "in_progress");
                frames.push(self.event(
                    "response.output_item.added",
                    json!({"output_index": index, "item": item}),
                ));
                self.open = Some(Open::Call {
                    id,
                    call_id: call_id.clone(),
                    name: name.clone(),
                    args: String::new(),
                    streamed: false,
                    ended: false,
                });
            }
            GenEvent::ToolArgsDelta { json: delta, .. } => {
                let index = self.index();
                if let Some(Open::Call {
                    id, args, streamed, ..
                }) = &mut self.open
                {
                    args.push_str(delta);
                    *streamed = true;
                    let id = id.clone();
                    frames.push(self.event(
                        "response.function_call_arguments.delta",
                        json!({"item_id": id, "output_index": index, "delta": delta}),
                    ));
                }
            }
            GenEvent::ToolEnd { raw_args, .. } => {
                let index = self.index();
                if let Some(Open::Call {
                    id,
                    args,
                    streamed,
                    ended,
                    ..
                }) = &mut self.open
                {
                    // The model's raw text is the arguments, valid or not; buffered arguments
                    // go out in one delta.
                    let send_delta = !*streamed && !raw_args.is_empty();
                    *args = raw_args.clone();
                    *ended = true;
                    let id = id.clone();
                    if send_delta {
                        frames.push(self.event(
                            "response.function_call_arguments.delta",
                            json!({"item_id": id, "output_index": index, "delta": raw_args}),
                        ));
                    }
                    frames.push(self.event(
                        "response.function_call_arguments.done",
                        json!({"item_id": id, "output_index": index, "arguments": raw_args}),
                    ));
                }
            }
            GenEvent::Finished { reason, usage, .. } => {
                self.done = true;
                let complete = !truncated(*reason);
                self.close(complete, &mut frames);
                let (name, status) = if complete {
                    ("response.completed", "completed")
                } else {
                    ("response.incomplete", "incomplete")
                };
                let response = self.object(status, Some(*usage));
                frames.push(self.event(name, json!({"response": response})));
            }
            GenEvent::Failed(error) => {
                self.done = true;
                let mut response = self.object("failed", None);
                response["error"] = json!({"code": error_code(error), "message": error.message});
                frames.push(self.event("response.failed", json!({"response": response})));
            }
        }
        frames
    }
}

impl StreamEncode for StreamEncoder {
    fn encode(&mut self, event: &GenEvent) -> Vec<String> {
        StreamEncoder::encode(self, event)
    }

    /// Codex's SSE client drops comments, so the keepalive is a real event.
    fn keepalive(&mut self) -> String {
        let response = self.object("in_progress", None);
        self.event("response.in_progress", json!({"response": response}))
    }

    fn open(&mut self) -> Vec<String> {
        let response = self.object("in_progress", None);
        vec![
            self.event("response.created", json!({"response": response.clone()})),
            self.event("response.in_progress", json!({"response": response})),
        ]
    }
}

pub const SHAPE: ApiShape = ApiShape {
    api: ApiKind::Responses,
    error_body,
};

/// `POST /v1/responses`
pub async fn handle(State(state): State<ServerState>, headers: HeaderMap, body: Bytes) -> Response {
    let flags = handler::observe(&state, ApiKind::Responses, &headers, &body);
    let mut request =
        match handler::parse(&body).and_then(|value| decode(&value, state.decode_options())) {
            Ok(request) => request,
            Err(error) => return SHAPE.error(&error),
        };
    request.ignore_eos = flags.ignore_eos;
    let stream = request.stream;
    let naming = Naming::of(&request);
    let turn = match handler::start(&state, request).await {
        Ok(turn) => turn,
        // Codex compacts only on the in-stream form of this error (contract: status codes).
        Err(error) if stream && error.kind == ErrorKind::ContextLengthExceeded => {
            let model = handler::ready(&state)
                .map(|loaded| loaded.model_id.clone())
                .unwrap_or_default();
            return handler::stream_error_response(StreamEncoder::new(&model, naming), error);
        }
        Err(error) => return SHAPE.error(&error),
    };
    if stream {
        let encoder = StreamEncoder::new(&turn.model, naming);
        return handler::stream_response(turn, encoder);
    }
    let model = turn.model.clone();
    match handler::collect(turn).await {
        Ok((collected, warnings)) => {
            handler::json_response(response(&model, &collected, &naming), &collected, &warnings)
        }
        Err(error) => SHAPE.error(&error),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn decode_ok(body: Value) -> ChatRequest {
        decode(&body, DecodeOptions::default()).unwrap()
    }

    fn decode_err(body: Value) -> ApiError {
        decode(&body, DecodeOptions::default()).unwrap_err()
    }

    /// `(event name, data)` of each frame.
    fn events(frames: &[String]) -> Vec<(String, Value)> {
        frames
            .iter()
            .map(|f| {
                let mut lines = f.lines();
                let name = lines.next().unwrap().strip_prefix("event: ").unwrap();
                let data = lines.next().unwrap().strip_prefix("data: ").unwrap();
                (name.to_string(), serde_json::from_str(data).unwrap())
            })
            .collect()
    }

    fn usage3() -> Usage {
        Usage {
            input: 10,
            cached: 4,
            output: 3,
            reasoning: 1,
        }
    }

    #[test]
    fn decode_maps_items_into_turns() {
        let encrypted = reasoning::encode_content("plan");
        let request = decode_ok(json!({
            "model": "gpt-x",
            "instructions": "Base.",
            "input": [
                {"type": "message", "role": "developer", "content": [
                    {"type": "input_text", "text": "Dev."}]},
                {"role": "user", "content": "hi"},
                {"type": "reasoning", "id": "rs_1", "summary": [], "encrypted_content": encrypted},
                {"type": "message", "role": "assistant", "content": [
                    {"type": "output_text", "text": "Looking."}]},
                {"type": "function_call", "id": "fc_1", "call_id": "call_1", "name": "shell",
                 "arguments": "{\"cmd\":\"ls\"}"},
                {"type": "function_call_output", "call_id": "call_1", "output": "a.rs"},
                {"type": "reasoning", "summary": [], "encrypted_content": "gAAAA-foreign"},
                {"type": "message", "role": "assistant", "content": "Done."},
                {"type": "message", "role": "system", "content": "mid"},
            ],
            "store": true, "include": ["reasoning.encrypted_content"],
            "prompt_cache_key": "k", "max_output_tokens": 50, "temperature": 0.2,
            "reasoning": {"effort": "high", "summary": "auto"},
            "text": {"format": {"type": "json_schema", "name": "x",
                "schema": {"type": "object"}, "strict": true}},
        }));
        assert_eq!(request.api, ApiKind::Responses);
        assert_eq!(request.system, vec!["Base.", "Dev."]);
        assert_eq!(
            request.messages,
            vec![
                Message::user_text("hi"),
                Message::Assistant(vec![
                    AssistantPart::Reasoning("plan".into()),
                    AssistantPart::Text("Looking.".into()),
                    AssistantPart::ToolCall {
                        id: "call_1".into(),
                        name: "shell".into(),
                        raw_args: "{\"cmd\":\"ls\"}".into(),
                    },
                ]),
                Message::Tool(ToolResult {
                    call_id: "call_1".into(),
                    content: "a.rs".into(),
                    is_error: false,
                    images_omitted: 0,
                }),
                Message::Assistant(vec![AssistantPart::Text("Done.".into())]),
                Message::System("mid".into()),
            ]
        );
        assert_eq!(request.max_output_tokens, Some(50));
        assert!(request.reasoning.is_on());
        assert!(matches!(
            request.response_format,
            ResponseFormat::JsonSchema { strict: true, .. }
        ));
    }

    #[test]
    fn string_input_and_no_limit() {
        let request = decode_ok(json!({"input": "hello"}));
        assert_eq!(request.messages, vec![Message::user_text("hello")]);
        assert_eq!(request.max_output_tokens, None, "FR-019: capacity − prompt");
        assert!(!request.stream);
    }

    #[test]
    fn resp_reasoning_present_without_effort_no_think() {
        for reasoning in [
            json!({"summary": "auto"}),
            json!({"effort": "minimal"}),
            json!(null),
        ] {
            let request = decode_ok(json!({"input": "x", "reasoning": reasoning}));
            assert!(!request.reasoning.is_on(), "{reasoning}");
        }
    }

    #[test]
    fn resp_previous_response_id_and_item_reference_rejected() {
        let error = decode_err(json!({"input": "x", "previous_response_id": "resp_1"}));
        assert_eq!(error.param.as_deref(), Some("previous_response_id"));
        let error = decode_err(json!({"input": [{"role": "user", "content": "x"},
            {"type": "item_reference", "id": "msg_1"}]}));
        assert_eq!(error.param.as_deref(), Some("input[1]"));
    }

    #[test]
    fn user_media_rejected_with_path() {
        for kind in ["input_image", "input_file"] {
            let error = decode_err(json!({"input": [{"role": "user", "content": [
                {"type": "input_text", "text": "see"}, {"type": kind}]}]}));
            assert_eq!(
                error.param.as_deref(),
                Some("input[0].content[1]"),
                "{kind}"
            );
        }
    }

    #[test]
    fn resp_function_call_output_array_and_images() {
        let body = json!({"input": [
            {"role": "user", "content": "look"},
            {"type": "function_call", "call_id": "c", "name": "view", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "c", "output": [
                {"type": "input_text", "text": "shot: "},
                {"type": "input_image", "image_url": "data:image/png;base64,AA"}]}]});
        let request = decode_ok(body.clone());
        let Message::Tool(result) = &request.messages[2] else {
            panic!("{:?}", request.messages)
        };
        assert_eq!(
            result.content,
            format!("shot: {}", crate::api::decode::IMAGE_PLACEHOLDER)
        );
        assert_eq!(result.images_omitted, 1);
        // resp_reject_tool_images_flag
        let error = decode(
            &body,
            DecodeOptions {
                reject_tool_images: true,
            },
        )
        .unwrap_err();
        assert_eq!(error.param.as_deref(), Some("input[2].output[1]"));
    }

    #[test]
    fn tools_function_web_search_custom() {
        let request = decode_ok(json!({"input": "x", "tools": [
            {"type": "function", "name": "shell", "description": "run",
             "parameters": {"type": "object", "properties": {}}, "strict": false},
            {"type": "web_search"}, {"type": "web_search_preview"}],
            "tool_choice": {"type": "function", "name": "shell"}, "parallel_tool_calls": false}));
        assert_eq!(request.tools.len(), 1);
        assert_eq!(request.tools[0].name, "shell");
        assert_eq!(request.tool_choice, ToolChoice::Named("shell".into()));
        assert!(!request.parallel_tool_calls);
        for kind in ["custom", "file_search"] {
            let error = decode_err(json!({"input": "x", "tools": [{"type": kind, "name": "t"}]}));
            assert_eq!(error.param.as_deref(), Some("tools[0].type"), "{kind}");
            assert!(error.message.contains(kind));
        }
    }

    #[test]
    fn resp_namespace_roundtrip() {
        let body = json!({"input": [
            {"role": "user", "content": "list issues"},
            {"type": "function_call", "call_id": "c1", "namespace": "mcp__github",
             "name": "list_issues", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "c1", "output": "[]"}],
            "tools": [{"type": "namespace", "name": "mcp__github", "description": "GitHub",
                "tools": [{"type": "function", "name": "list_issues",
                    "parameters": {"type": "object", "properties": {}}}]}]});
        let request = decode_ok(body);
        assert_eq!(request.tools[0].name, "mcp__github__list_issues");
        let Message::Assistant(parts) = &request.messages[1] else {
            panic!()
        };
        assert!(
            matches!(&parts[0], AssistantPart::ToolCall { name, .. } if name == "mcp__github__list_issues")
        );
        // Calls go back split into namespace and name, streamed and final.
        let naming = Naming::of(&request);
        let item = call_item(
            &naming,
            "fc_1",
            "call_2",
            "mcp__github__list_issues",
            "{}",
            true,
        );
        assert_eq!(
            (item["namespace"].as_str(), item["name"].as_str()),
            (Some("mcp__github"), Some("list_issues"))
        );
        let mut encoder = StreamEncoder::new("m", naming);
        let frames = encoder.encode(&GenEvent::ToolStart {
            idx: 0,
            id: "call_2".into(),
            name: "mcp__github__list_issues".into(),
        });
        let added = &events(&frames)[0].1;
        assert_eq!(added["item"]["namespace"], "mcp__github");
        assert_eq!(added["item"]["name"], "list_issues");
    }

    #[test]
    fn tool_search_call_and_output_expand_deferred_tools() {
        let body = json!({"input": [
            {"role": "user", "content": "find a tool"},
            {"type": "tool_search_call", "call_id": "s1", "execution": "client",
             "arguments": {"query": "issues"}},
            {"type": "tool_search_output", "call_id": "s1", "execution": "client",
             "tools": [{"type": "namespace", "name": "mcp__github", "tools": [
                {"type": "function", "name": "list_issues", "defer_loading": true,
                 "parameters": {"type": "object", "properties": {}}}]}]}],
            "tools": [{"type": "tool_search", "execution": "client"},
                {"type": "function", "name": "shell", "parameters": {"type": "object"}}]});
        let request = decode_ok(body);
        let found = request.tool("mcp__github__list_issues").unwrap();
        assert!(found.deferred, "callable but never in the tools region");
        let Message::Tool(result) = &request.messages[2] else {
            panic!()
        };
        assert!(
            result
                .content
                .contains("\"name\":\"mcp__github__list_issues\""),
            "{}",
            result.content
        );
        let naming = Naming::of(&request);
        let item = call_item(
            &naming,
            "ts_1",
            "s2",
            TOOL_SEARCH,
            "{\"query\":\"pr\"}",
            true,
        );
        assert_eq!(item["type"], "tool_search_call");
        assert_eq!(item["arguments"], json!({"query": "pr"}));
        assert_eq!(item["execution"], "client");
    }

    #[test]
    fn non_streaming_items_and_usage() {
        let mut collected = Collected::default();
        for event in [
            GenEvent::ReasoningDelta("hm".into()),
            GenEvent::TextDelta("Calling.".into()),
            GenEvent::ToolStart {
                idx: 0,
                id: "call_9".into(),
                name: "shell".into(),
            },
            GenEvent::ToolEnd {
                idx: 0,
                raw_args: "{\"cmd\":\"ls\"}".into(),
                valid: true,
            },
            GenEvent::Finished {
                reason: Finish::ToolCalls,
                stop_seq: None,
                usage: usage3(),
            },
        ] {
            collected.push(event);
        }
        let body = response("m", &collected, &Naming::default());
        assert_eq!(body["object"], "response");
        assert_eq!(body["status"], "completed");
        assert!(body["incomplete_details"].is_null());
        let output = body["output"].as_array().unwrap();
        assert_eq!(output[0]["type"], "reasoning");
        assert_eq!(output[0]["summary"], json!([]));
        assert_eq!(
            reasoning::decode_content(output[0]["encrypted_content"].as_str().unwrap()).unwrap(),
            "hm"
        );
        assert_eq!(output[1]["content"][0]["text"], "Calling.");
        assert_eq!(output[1]["content"][0]["annotations"], json!([]));
        assert_eq!(output[2]["call_id"], "call_9");
        assert!(output[2]["id"].as_str().unwrap().starts_with("fc_"));
        assert_eq!(output[2]["status"], "completed");
        assert_eq!(body["usage"]["input_tokens_details"]["cached_tokens"], 4);
        assert_eq!(
            body["usage"]["output_tokens_details"]["reasoning_tokens"],
            1
        );
        assert_eq!(body["usage"]["total_tokens"], 13);
    }

    #[test]
    fn truncated_call_is_incomplete() {
        let mut collected = Collected::default();
        for event in [
            GenEvent::ToolStart {
                idx: 0,
                id: "call_1".into(),
                name: "shell".into(),
            },
            GenEvent::ToolEnd {
                idx: 0,
                raw_args: "{\"cmd\":".into(),
                valid: false,
            },
            GenEvent::Finished {
                reason: Finish::Length,
                stop_seq: None,
                usage: usage3(),
            },
        ] {
            collected.push(event);
        }
        let body = response("m", &collected, &Naming::default());
        assert_eq!(body["status"], "incomplete");
        assert_eq!(body["incomplete_details"]["reason"], "max_output_tokens");
        assert_eq!(body["output"][0]["status"], "incomplete");
        assert_eq!(body["output"][0]["arguments"], "{\"cmd\":");

        let mut encoder = StreamEncoder::new("m", Naming::default());
        let mut frames = Vec::new();
        for event in [
            GenEvent::ToolStart {
                idx: 0,
                id: "call_1".into(),
                name: "shell".into(),
            },
            GenEvent::ToolEnd {
                idx: 0,
                raw_args: "{\"cmd\":".into(),
                valid: false,
            },
            GenEvent::Finished {
                reason: Finish::Length,
                stop_seq: None,
                usage: usage3(),
            },
        ] {
            frames.extend(encoder.encode(&event));
        }
        let events = events(&frames);
        let done = events
            .iter()
            .find(|(n, _)| n == "response.output_item.done")
            .unwrap();
        assert_eq!(done.1["item"]["status"], "incomplete");
        let (name, last) = events.last().unwrap();
        assert_eq!(name, "response.incomplete");
        assert_eq!(last["response"]["output"][0]["status"], "incomplete");
    }

    #[test]
    fn stream_event_order_and_sequence_numbers() {
        let mut encoder = StreamEncoder::new("m", Naming::default());
        let mut frames = encoder.open();
        frames.push(encoder.keepalive());
        for event in [
            GenEvent::Started {
                input: 10,
                cached: 4,
                max_tokens: 100,
                miss: None,
            },
            GenEvent::ReasoningDelta("a".into()),
            GenEvent::ReasoningDelta("b".into()),
            GenEvent::TextDelta("Hi".into()),
            GenEvent::ToolStart {
                idx: 0,
                id: "call_1".into(),
                name: "shell".into(),
            },
            GenEvent::ToolArgsDelta {
                idx: 0,
                json: "{}".into(),
            },
            GenEvent::ToolEnd {
                idx: 0,
                raw_args: "{}".into(),
                valid: true,
            },
            GenEvent::ToolStart {
                idx: 1,
                id: "call_2".into(),
                name: "shell".into(),
            },
            GenEvent::ToolEnd {
                idx: 1,
                raw_args: "not json".into(),
                valid: false,
            },
            GenEvent::Finished {
                reason: Finish::ToolCalls,
                stop_seq: None,
                usage: usage3(),
            },
            GenEvent::TextDelta("late".into()),
        ] {
            frames.extend(encoder.encode(&event));
        }
        let events = events(&frames);
        let names: Vec<&str> = events.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(
            names,
            [
                "response.created",
                "response.in_progress",
                "response.in_progress",
                "response.output_item.added",
                "response.reasoning_text.delta",
                "response.reasoning_text.delta",
                "response.reasoning_text.done",
                "response.output_item.done",
                "response.output_item.added",
                "response.content_part.added",
                "response.output_text.delta",
                "response.output_text.done",
                "response.content_part.done",
                "response.output_item.done",
                "response.output_item.added",
                "response.function_call_arguments.delta",
                "response.function_call_arguments.done",
                "response.output_item.done",
                "response.output_item.added",
                "response.function_call_arguments.delta",
                "response.function_call_arguments.done",
                "response.output_item.done",
                "response.completed",
            ]
        );
        for (i, (name, data)) in events.iter().enumerate() {
            assert_eq!(data["type"], name.as_str());
            assert_eq!(data["sequence_number"], i as u64);
        }
        // resp_function_call_item_done_parsable_by_codex: complete items with call_id.
        let calls: Vec<&Value> = events
            .iter()
            .filter(|(n, d)| {
                n == "response.output_item.done" && d["item"]["type"] == "function_call"
            })
            .map(|(_, d)| &d["item"])
            .collect();
        assert_eq!(calls[0]["call_id"], "call_1");
        assert_eq!(calls[0]["arguments"], "{}");
        assert_eq!(calls[1]["arguments"], "not json", "invalid args raw");
        let completed = &events.last().unwrap().1["response"];
        assert_eq!(completed["status"], "completed");
        assert_eq!(completed["output"].as_array().unwrap().len(), 4);
        assert_eq!(completed["usage"]["input_tokens"], 10);
        assert_eq!(completed["output"][0]["summary"], json!([]));
    }

    #[test]
    fn resp_stream_failed_codes() {
        for (kind, code) in [
            (ErrorKind::ContextLengthExceeded, "context_length_exceeded"),
            (ErrorKind::QueueTimeout, "server_busy"),
            (ErrorKind::Overloaded, "server_busy"),
        ] {
            let mut encoder = StreamEncoder::new("m", Naming::default());
            let frames = encoder.encode(&GenEvent::Failed(ApiError::new(kind, "x")));
            let (name, data) = &events(&frames)[0];
            assert_eq!(name, "response.failed");
            assert_eq!(data["response"]["status"], "failed");
            assert_eq!(data["response"]["error"]["code"], code);
            assert!(encoder.encode(&GenEvent::TextDelta("x".into())).is_empty());
        }
    }
}
