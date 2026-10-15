//! OpenAI Chat Completions (`POST /v1/chat/completions`): decoder and encoders (002 research
//! §1a).

use crate::api::collect::Collected;
use crate::api::decode::{
    DecodeOptions, arr, check_limits, check_schema, check_tool, obj, opt_bool, opt_f32, opt_str,
    opt_u32, stops, str_field, tool_image,
};
use crate::api::handler::{self, ApiShape, StreamEncode};
use crate::api::model::{
    ApiError, ApiKind, AssistantPart, ChatRequest, Effort, ErrorKind, Finish, GenEvent, Message,
    ReasoningCtl, ResponseFormat, ToolChoice, ToolDef, ToolResult, Usage, UserPart,
};
use crate::api::sse::{self, new_id};
use crate::cli::serve::ServerState;
use axum::body::Bytes;
use axum::extract::State;
use axum::http::HeaderMap;
use axum::response::Response;
use serde_json::{Value, json};
use std::collections::HashMap;

pub fn decode(body: &Value, options: DecodeOptions) -> Result<ChatRequest, ApiError> {
    obj(body, "body")?;
    let mut request = ChatRequest::new(ApiKind::Chat);
    request.model = opt_str(body, "model", "body")?
        .unwrap_or_default()
        .to_string();

    let messages = arr(&body["messages"], "messages")?;
    for (i, message) in messages.iter().enumerate() {
        let path = format!("messages[{i}]");
        obj(message, &path)?;
        let role = str_field(message, "role", &path)?;
        match role {
            "system" | "developer" => {
                let text = text_content(&message["content"], &path)?;
                if request.messages.is_empty() {
                    request.system.push(text);
                } else {
                    request.messages.push(Message::System(text));
                }
            }
            "user" => {
                let text = text_content(&message["content"], &path)?;
                request
                    .messages
                    .push(Message::User(vec![UserPart::Text(text)]));
            }
            "assistant" => request.messages.push(assistant(message, &path)?),
            "tool" => {
                let call_id = str_field(message, "tool_call_id", &path)?.to_string();
                let (content, images_omitted) = tool_content(&message["content"], &path, options)?;
                request.messages.push(Message::Tool(ToolResult {
                    call_id,
                    content,
                    is_error: false,
                    images_omitted,
                }));
            }
            other => {
                return Err(ApiError::invalid(
                    format!("unsupported role '{other}'"),
                    format!("{path}.role"),
                ));
            }
        }
    }
    if request.messages.is_empty() && request.system.is_empty() {
        return Err(ApiError::invalid("messages must not be empty", "messages"));
    }

    if let Some(tools) = body.get("tools").filter(|t| !t.is_null()) {
        for (i, tool) in arr(tools, "tools")?.iter().enumerate() {
            let path = format!("tools[{i}]");
            if tool["type"] != "function" {
                return Err(ApiError::invalid(
                    "only function tools are supported",
                    format!("{path}.type"),
                ));
            }
            let function = &tool["function"];
            obj(function, &format!("{path}.function"))?;
            let def = ToolDef {
                name: str_field(function, "name", &format!("{path}.function"))?.to_string(),
                description: opt_str(function, "description", &path)?
                    .unwrap_or_default()
                    .to_string(),
                schema: match &function["parameters"] {
                    Value::Null => json!({"type": "object", "properties": {}}),
                    schema => schema.clone(),
                },
                strict: function["strict"].as_bool().unwrap_or(false),
                ..Default::default()
            };
            check_tool(&def, &path)?;
            request.tools.push(def);
        }
    }
    request.tool_choice = match &body["tool_choice"] {
        Value::Null => ToolChoice::Auto,
        Value::String(s) => match s.as_str() {
            "auto" => ToolChoice::Auto,
            "none" => ToolChoice::None,
            "required" => ToolChoice::Required,
            _ => return Err(ApiError::invalid("unknown tool_choice", "tool_choice")),
        },
        choice => ToolChoice::Named(
            str_field(&choice["function"], "name", "tool_choice.function")?.to_string(),
        ),
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
    request.response_format = match &body["response_format"] {
        Value::Null => ResponseFormat::Text,
        format => match format["type"].as_str() {
            Some("text") => ResponseFormat::Text,
            Some("json_object") => ResponseFormat::JsonObject,
            Some("json_schema") => {
                let spec = &format["json_schema"];
                let schema = spec["schema"].clone();
                check_schema(&schema, "response_format.json_schema.schema")?;
                ResponseFormat::JsonSchema {
                    name: spec["name"].as_str().unwrap_or("response").to_string(),
                    schema,
                    strict: spec["strict"].as_bool().unwrap_or(false),
                }
            }
            _ => {
                return Err(ApiError::invalid(
                    "unsupported response_format type",
                    "response_format.type",
                ));
            }
        },
    };
    request.max_output_tokens = match opt_u32(body, "max_completion_tokens")? {
        Some(n) => Some(n),
        None => opt_u32(body, "max_tokens")?,
    };
    request.stop = stops(body, "stop")?;
    request.sampling.temperature = opt_f32(body, "temperature")?;
    request.sampling.top_p = opt_f32(body, "top_p")?;
    request.sampling.top_k = opt_u32(body, "top_k")?;
    request.sampling.min_p = opt_f32(body, "min_p")?;
    request.sampling.seed = body["seed"].as_u64();
    request.reasoning = openai_effort(opt_str(body, "reasoning_effort", "body")?);
    request.stream = opt_bool(body, "stream")?.unwrap_or(false);
    request.include_usage = body["stream_options"]["include_usage"]
        .as_bool()
        .unwrap_or(false);
    check_limits(&request)?;
    Ok(request)
}

/// OpenAI `reasoning_effort` / Responses `reasoning.effort`: `none` and `minimal` mean off.
pub fn openai_effort(effort: Option<&str>) -> ReasoningCtl {
    let effort = match effort {
        None | Some("none" | "minimal") => return ReasoningCtl::Off,
        Some("low") => Effort::Low,
        Some("medium") => Effort::Medium,
        Some(_) => Effort::High,
    };
    ReasoningCtl::On {
        budget: None,
        effort: Some(effort),
    }
}

/// String content, or text parts joined; images and other media are refused with their path.
fn text_content(content: &Value, path: &str) -> Result<String, ApiError> {
    match content {
        Value::Null => Ok(String::new()),
        Value::String(s) => Ok(s.clone()),
        Value::Array(parts) => {
            let mut text = String::new();
            for (j, part) in parts.iter().enumerate() {
                match part["type"].as_str() {
                    Some("text") => text.push_str(part["text"].as_str().unwrap_or_default()),
                    _ => {
                        return Err(ApiError::invalid(
                            "only text content is supported",
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

/// Tool message content: text parts joined, images replaced per FR-020.
fn tool_content(
    content: &Value,
    path: &str,
    options: DecodeOptions,
) -> Result<(String, u32), ApiError> {
    let Value::Array(parts) = content else {
        return Ok((text_content(content, path)?, 0));
    };
    let mut text = String::new();
    let mut images = 0;
    for (j, part) in parts.iter().enumerate() {
        match part["type"].as_str() {
            Some("text") => text.push_str(part["text"].as_str().unwrap_or_default()),
            Some("image_url") => {
                text.push_str(tool_image(options, &format!("{path}.content[{j}]"))?);
                images += 1;
            }
            _ => {
                return Err(ApiError::invalid(
                    "only text content is supported",
                    format!("{path}.content[{j}]"),
                ));
            }
        }
    }
    Ok((text, images))
}

fn assistant(message: &Value, path: &str) -> Result<Message, ApiError> {
    let mut parts = Vec::new();
    let reasoning = opt_str(message, "reasoning_content", path)?
        .or(opt_str(message, "reasoning", path)?)
        .unwrap_or_default();
    if !reasoning.is_empty() {
        parts.push(AssistantPart::Reasoning(reasoning.to_string()));
    }
    let text = text_content(&message["content"], path)?;
    if !text.is_empty() {
        parts.push(AssistantPart::Text(text));
    }
    if let Some(calls) = message.get("tool_calls").filter(|c| !c.is_null()) {
        for (k, call) in arr(calls, &format!("{path}.tool_calls"))?
            .iter()
            .enumerate()
        {
            let cpath = format!("{path}.tool_calls[{k}]");
            let function = &call["function"];
            let raw_args = match &function["arguments"] {
                Value::String(s) => s.clone(),
                Value::Null => "{}".to_string(),
                other => other.to_string(),
            };
            parts.push(AssistantPart::ToolCall {
                id: str_field(call, "id", &cpath)?.to_string(),
                name: str_field(function, "name", &format!("{cpath}.function"))?.to_string(),
                raw_args,
            });
        }
    }
    Ok(Message::Assistant(parts))
}

pub fn finish_reason(finish: Finish) -> &'static str {
    match finish {
        Finish::Stop | Finish::StopSequence => "stop",
        Finish::Length | Finish::ContextFull => "length",
        Finish::ToolCalls => "tool_calls",
    }
}

pub fn usage(usage: Usage) -> Value {
    json!({
        "prompt_tokens": usage.input,
        "completion_tokens": usage.output,
        "total_tokens": usage.input + usage.output,
        "prompt_tokens_details": {"cached_tokens": usage.cached},
        "completion_tokens_details": {"reasoning_tokens": usage.reasoning},
    })
}

pub fn error_body(error: &ApiError) -> Value {
    let (kind, code) = match error.kind {
        ErrorKind::InvalidRequest | ErrorKind::PayloadTooLarge => ("invalid_request_error", None),
        ErrorKind::ContextLengthExceeded => {
            ("invalid_request_error", Some("context_length_exceeded"))
        }
        ErrorKind::QueueFull => ("rate_limit_exceeded", Some("rate_limit_exceeded")),
        ErrorKind::QueueTimeout | ErrorKind::Overloaded | ErrorKind::NotReady => {
            ("server_error", Some("server_busy"))
        }
        ErrorKind::Forbidden => ("forbidden", None),
        ErrorKind::Internal => ("server_error", None),
    };
    json!({"error": {"message": error.message, "type": kind, "param": error.param, "code": code}})
}

/// Non-streaming body.
pub fn response(model: &str, collected: &Collected) -> Value {
    let (finish, _, usage_) =
        collected
            .finish
            .clone()
            .unwrap_or((Finish::Stop, None, Usage::default()));
    let text = collected.text();
    let reasoning = collected.reasoning();
    let calls: Vec<Value> = collected
        .calls()
        .map(|c| {
            json!({"id": c.id, "type": "function",
                "function": {"name": c.name, "arguments": c.raw_args}})
        })
        .collect();
    let mut message = json!({"role": "assistant",
        "content": if text.is_empty() && !calls.is_empty() { Value::Null } else { text.into() }});
    if !reasoning.is_empty() {
        message["reasoning_content"] = reasoning.into();
    }
    if !calls.is_empty() {
        message["tool_calls"] = Value::Array(calls);
    }
    json!({
        "id": new_id("chatcmpl-"),
        "object": "chat.completion",
        "created": sse::unix_now(),
        "model": model,
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason(finish)}],
        "usage": usage(usage_),
    })
}

/// Streaming encoder: one `chat.completion.chunk` per delta, the role in the first chunk,
/// nothing after `finish_reason` except the usage chunk and `[DONE]`.
#[derive(Debug)]
pub struct StreamEncoder {
    id: String,
    created: u64,
    model: String,
    include_usage: bool,
    /// Call index → whether arguments were streamed.
    streamed_args: HashMap<u32, bool>,
    done: bool,
}

impl StreamEncoder {
    pub fn new(model: &str, include_usage: bool) -> Self {
        Self {
            id: new_id("chatcmpl-"),
            created: sse::unix_now(),
            model: model.to_string(),
            include_usage,
            streamed_args: HashMap::new(),
            done: false,
        }
    }

    fn chunk(&self, delta: Value, finish: Option<&str>) -> String {
        sse::data(&json!({
            "id": self.id, "object": "chat.completion.chunk", "created": self.created,
            "model": self.model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }))
    }

    /// Frames for one event; `[DONE]` follows the terminal event.
    pub fn encode(&mut self, event: &GenEvent) -> Vec<String> {
        if self.done {
            return Vec::new();
        }
        match event {
            GenEvent::Started { .. } => {
                vec![self.chunk(json!({"role": "assistant", "content": ""}), None)]
            }
            GenEvent::TextDelta(text) => vec![self.chunk(json!({"content": text}), None)],
            GenEvent::ReasoningDelta(text) => {
                vec![self.chunk(json!({"reasoning_content": text}), None)]
            }
            GenEvent::ToolStart { idx, id, name } => {
                self.streamed_args.insert(*idx, false);
                vec![self.chunk(
                    json!({"tool_calls": [{"index": idx, "id": id, "type": "function",
                        "function": {"name": name, "arguments": ""}}]}),
                    None,
                )]
            }
            GenEvent::ToolArgsDelta { idx, json } => {
                self.streamed_args.insert(*idx, true);
                vec![self.chunk(
                    json!({"tool_calls": [{"index": idx, "function": {"arguments": json}}]}),
                    None,
                )]
            }
            GenEvent::ToolEnd { idx, raw_args, .. } => {
                // Buffered arguments (synthesized or a fallback) go out in one fragment.
                if self.streamed_args.get(idx) == Some(&false) && !raw_args.is_empty() {
                    vec![self.chunk(
                        json!({"tool_calls": [{"index": idx, "function": {"arguments": raw_args}}]}),
                        None,
                    )]
                } else {
                    Vec::new()
                }
            }
            GenEvent::Finished { reason, usage, .. } => {
                self.done = true;
                let mut frames = vec![self.chunk(json!({}), Some(finish_reason(*reason)))];
                if self.include_usage {
                    frames.push(sse::data(&json!({
                        "id": self.id, "object": "chat.completion.chunk",
                        "created": self.created, "model": self.model,
                        "choices": [], "usage": self::usage(*usage),
                    })));
                }
                frames.push("data: [DONE]\n\n".to_string());
                frames
            }
            GenEvent::Failed(error) => {
                // After headers: an error object, then close without `[DONE]`.
                self.done = true;
                vec![sse::data(&error_body(error))]
            }
        }
    }
}

impl StreamEncode for StreamEncoder {
    fn encode(&mut self, event: &GenEvent) -> Vec<String> {
        StreamEncoder::encode(self, event)
    }

    fn keepalive(&mut self) -> String {
        sse::comment("keepalive")
    }
}

pub const SHAPE: ApiShape = ApiShape {
    api: ApiKind::Chat,
    error_body,
};

/// `POST /v1/chat/completions`
pub async fn handle(State(state): State<ServerState>, headers: HeaderMap, body: Bytes) -> Response {
    let flags = handler::observe(&state, ApiKind::Chat, &headers, &body);
    let mut request =
        match handler::parse(&body).and_then(|value| decode(&value, state.decode_options())) {
            Ok(request) => request,
            Err(error) => return SHAPE.error(&error),
        };
    request.ignore_eos = flags.ignore_eos;
    let stream = request.stream;
    let include_usage = request.include_usage;
    let turn = match handler::start(&state, request).await {
        Ok(turn) => turn,
        Err(error) => return SHAPE.error(&error),
    };
    if stream {
        let encoder = StreamEncoder::new(&turn.model, include_usage);
        return handler::stream_response(turn, encoder);
    }
    let model = turn.model.clone();
    match handler::collect(turn).await {
        Ok((collected, warnings)) => {
            handler::json_response(response(&model, &collected), &collected, &warnings)
        }
        Err(error) => SHAPE.error(&error),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frames_json(frames: &[String]) -> Vec<Value> {
        frames
            .iter()
            .filter_map(|f| f.strip_prefix("data: "))
            .filter(|f| !f.starts_with("[DONE]"))
            .map(|f| serde_json::from_str(f.trim()).unwrap())
            .collect()
    }

    #[test]
    fn decode_maps_roles_tools_and_limits() {
        let body = json!({
            "model": "claude-x",
            "messages": [
                {"role": "developer", "content": "Be brief."},
                {"role": "user", "content": [{"type": "text", "text": "hi"}]},
                {"role": "assistant", "content": null, "reasoning": "think",
                 "tool_calls": [{"id": "c1", "type": "function",
                    "function": {"name": "read", "arguments": "{\"p\":1}"}}]},
                {"role": "tool", "tool_call_id": "c1", "content": "ok"},
                {"role": "system", "content": "mid"},
            ],
            "tools": [{"type": "function", "function": {"name": "read", "parameters": {"type": "object"}}}],
            "max_tokens": 10, "max_completion_tokens": 20,
            "reasoning_effort": "minimal", "stream": true,
            "stream_options": {"include_usage": true}, "temperature": 0.5,
            "unknown_field": 1
        });
        let request = decode(&body, DecodeOptions::default()).unwrap();
        assert_eq!(request.system, vec!["Be brief."]);
        assert_eq!(request.messages.len(), 4);
        assert_eq!(request.messages[3], Message::System("mid".into()));
        assert_eq!(
            request.messages[1],
            Message::Assistant(vec![
                AssistantPart::Reasoning("think".into()),
                AssistantPart::ToolCall {
                    id: "c1".into(),
                    name: "read".into(),
                    raw_args: "{\"p\":1}".into()
                }
            ])
        );
        assert_eq!(request.max_output_tokens, Some(20));
        assert_eq!(request.reasoning, ReasoningCtl::Off);
        assert!(request.stream && request.include_usage);
        assert_eq!(request.sampling.temperature, Some(0.5));
    }

    #[test]
    fn chat_image_part_rejected_with_path() {
        let body = json!({"messages": [{"role": "user", "content": [
            {"type": "text", "text": "see"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA"}}]}]});
        let error = decode(&body, DecodeOptions::default()).unwrap_err();
        assert_eq!(error.param.as_deref(), Some("messages[0].content[1]"));
    }

    #[test]
    fn chat_tool_call_stream_first_fragment_has_id_name() {
        let mut encoder = StreamEncoder::new("m", true);
        let mut frames = Vec::new();
        for event in [
            GenEvent::Started {
                input: 5,
                cached: 2,
                max_tokens: 9,
                miss: None,
            },
            GenEvent::ToolStart {
                idx: 0,
                id: "call_1".into(),
                name: "read".into(),
            },
            GenEvent::ToolArgsDelta {
                idx: 0,
                json: "{\"p\":".into(),
            },
            GenEvent::ToolArgsDelta {
                idx: 0,
                json: "1}".into(),
            },
            GenEvent::ToolEnd {
                idx: 0,
                raw_args: "{\"p\":1}".into(),
                valid: true,
            },
            GenEvent::Finished {
                reason: Finish::ToolCalls,
                stop_seq: None,
                usage: Usage {
                    input: 5,
                    cached: 2,
                    output: 4,
                    reasoning: 0,
                },
            },
            GenEvent::TextDelta("late".into()),
        ] {
            frames.extend(encoder.encode(&event));
        }
        assert_eq!(frames.last().unwrap(), "data: [DONE]\n\n");
        let chunks = frames_json(&frames);
        assert_eq!(chunks[0]["choices"][0]["delta"]["role"], "assistant");
        let first = &chunks[1]["choices"][0]["delta"]["tool_calls"][0];
        assert_eq!(
            (first["id"].as_str(), first["function"]["name"].as_str()),
            (Some("call_1"), Some("read"))
        );
        let args: String = chunks
            .iter()
            .filter_map(|c| {
                c["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"].as_str()
            })
            .collect();
        assert_eq!(args, "{\"p\":1}");
        assert_eq!(chunks[4]["choices"][0]["finish_reason"], "tool_calls");
        // chat_include_usage_chunk_empty_choices, chat_usage_cached_tokens_inclusive
        assert_eq!(chunks[5]["choices"], json!([]));
        assert_eq!(chunks[5]["usage"]["prompt_tokens"], 5);
        assert_eq!(
            chunks[5]["usage"]["prompt_tokens_details"]["cached_tokens"],
            2
        );
        assert_eq!(chunks.len(), 6, "nothing after the finish");
    }

    #[test]
    fn buffered_arguments_go_out_at_the_end() {
        let mut encoder = StreamEncoder::new("m", false);
        encoder.encode(&GenEvent::ToolStart {
            idx: 0,
            id: "c".into(),
            name: "edit".into(),
        });
        let frames = encoder.encode(&GenEvent::ToolEnd {
            idx: 0,
            raw_args: "{\"path\": \"a\"}".into(),
            valid: true,
        });
        let chunk = &frames_json(&frames)[0];
        assert_eq!(
            chunk["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"],
            "{\"path\": \"a\"}"
        );
    }

    #[test]
    fn chat_stream_error_after_headers() {
        let mut encoder = StreamEncoder::new("m", false);
        let frames = encoder.encode(&GenEvent::Failed(ApiError::new(
            ErrorKind::QueueTimeout,
            "waited too long",
        )));
        assert_eq!(frames.len(), 1);
        let body = &frames_json(&frames)[0];
        assert_eq!(body["error"]["code"], "server_busy");
    }

    #[test]
    fn non_streaming_body() {
        let mut collected = Collected::default();
        for event in [
            GenEvent::ReasoningDelta("hm".into()),
            GenEvent::TextDelta("Hello".into()),
            GenEvent::Finished {
                reason: Finish::Length,
                stop_seq: None,
                usage: Usage {
                    input: 3,
                    cached: 0,
                    output: 2,
                    reasoning: 1,
                },
            },
        ] {
            collected.push(event);
        }
        let body = response("m", &collected);
        assert_eq!(body["choices"][0]["message"]["content"], "Hello");
        assert_eq!(body["choices"][0]["message"]["reasoning_content"], "hm");
        assert_eq!(body["choices"][0]["finish_reason"], "length");
        assert_eq!(body["usage"]["total_tokens"], 5);
    }
}
