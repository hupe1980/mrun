//! OpenAI Chat Completions (`POST /v1/chat/completions`): decoder and encoders (002 research
//! §1a).

use crate::api::decode::{
    DecodeOptions, arr, check_counts, function_tool, join_text_parts, obj, openai_tool_choice,
    opt_bool, opt_f32, opt_str, opt_u32, raw_args, response_format, stops, str_field, text_part,
    tool_image,
};
use crate::api::handler::{Api, StreamEncode};
use crate::api::model::{
    ApiError, ApiKind, AssistantPart, ChatRequest, ErrorKind, Finish, GenEvent, Message,
    ReasoningCtl, ToolResult, Usage,
};
use crate::api::sse::{self, new_id};
use serde_json::{Value, json};
use std::collections::HashMap;

/// Top-level fields mrun reads or knowingly ignores; any other is logged `ignored-field`.
pub const KNOWN: &[&str] = &[
    "model",
    "messages",
    "max_tokens",
    "max_completion_tokens",
    "min_p",
    "parallel_tool_calls",
    "reasoning_effort",
    "chat_template_kwargs",
    "repetition_penalty",
    "stop",
    "stream",
    "stream_options",
    "temperature",
    "top_k",
    "top_p",
    "tools",
    "tool_choice",
    "response_format",
    "seed",
    "user",
    "metadata",
    "store",
    "service_tier",
    "prompt_cache_key",
];

pub fn decode(body: &Value, options: DecodeOptions) -> Result<ChatRequest, ApiError> {
    obj(body, "body")?;
    crate::api::decode::ignore_unknown(body, KNOWN, "");
    check_counts(body, "messages")?;
    let mut request = ChatRequest::new(ApiKind::Chat);
    request.model = opt_str(body, "model", "body")?
        .unwrap_or_default()
        .to_string();

    let messages = arr(&body["messages"], "messages")?;
    for (i, message) in messages.iter().enumerate() {
        if i > 0 {
            request.label_new(i as u32 - 1);
        }
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
                request.messages.push(Message::User(text));
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
    request.label_new(messages.len().saturating_sub(1) as u32);
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
            let tool = function_tool(function, &format!("{path}.function"), "parameters", None)?;
            request.tools.push(tool);
        }
    }
    request.tool_choice = openai_tool_choice(&request, &body["tool_choice"])?;
    request.parallel_tool_calls = opt_bool(body, "parallel_tool_calls")?.unwrap_or(true);
    let format = &body["response_format"];
    request.response_format = response_format(format, &format["json_schema"], "response_format")?;
    request.max_output_tokens = match opt_u32(body, "max_completion_tokens")? {
        Some(n) => Some(n),
        None => opt_u32(body, "max_tokens")?,
    };
    request.stop = stops(body, "stop")?;
    request.sampling.temperature = opt_f32(body, "temperature")?;
    request.sampling.top_p = opt_f32(body, "top_p")?;
    request.sampling.top_k = opt_u32(body, "top_k")?;
    request.sampling.min_p = opt_f32(body, "min_p")?;
    request.sampling.repetition_penalty = opt_f32(body, "repetition_penalty")?;
    request.sampling.seed = body["seed"].as_u64();
    request.reasoning = openai_effort(opt_str(body, "reasoning_effort", "body")?);
    if let Some(thinking) = enable_thinking(&body["chat_template_kwargs"])? {
        request.reasoning = thinking;
    }
    request.stream = opt_bool(body, "stream")?.unwrap_or(false);
    request.include_usage = body["stream_options"]["include_usage"]
        .as_bool()
        .unwrap_or(false);
    Ok(request)
}

/// OpenAI `reasoning_effort` / Responses `reasoning.effort`: `none` and `minimal` mean off, any
/// other level on (the level itself is not used). Absent means the model's own default, which is
/// on for every thinking family (D063): OpenAI-API clients such as OpenCode never send the field,
/// and a thinking model without its reasoning writes markedly worse code. The template still
/// renders thinking only where the family supports it.
pub fn openai_effort(effort: Option<&str>) -> ReasoningCtl {
    match effort {
        Some("none" | "minimal") => ReasoningCtl::Off,
        None | Some(_) => ReasoningCtl::On { budget: None },
    }
}

/// `chat_template_kwargs.enable_thinking` (023), the switch vLLM and SGLang take for Qwen chat
/// templates: `false` turns thinking off and `true` on, over `reasoning_effort` (OpenCode adds
/// `reasoning_effort: "low"` to the small-model requests it sends this with). Other keys are
/// ignored: mrun renders its own templates.
fn enable_thinking(kwargs: &Value) -> Result<Option<ReasoningCtl>, ApiError> {
    if kwargs.is_null() {
        return Ok(None);
    }
    obj(kwargs, "chat_template_kwargs")?;
    crate::api::decode::ignore_unknown(kwargs, &["enable_thinking"], "chat_template_kwargs.");
    match &kwargs["enable_thinking"] {
        Value::Null => Ok(None),
        Value::Bool(true) => Ok(Some(ReasoningCtl::On { budget: None })),
        Value::Bool(false) => Ok(Some(ReasoningCtl::Off)),
        _ => Err(ApiError::invalid(
            "chat_template_kwargs.enable_thinking must be a boolean",
            "chat_template_kwargs.enable_thinking",
        )),
    }
}

/// String content, or text parts joined; images and other media are refused with their path.
fn text_content(content: &Value, path: &str) -> Result<String, ApiError> {
    join_text_parts(
        content,
        path,
        |part| text_part(part, &["text"]),
        "only text content is supported",
    )
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
            parts.push(AssistantPart::ToolCall {
                id: str_field(call, "id", &cpath)?.to_string(),
                name: str_field(function, "name", &format!("{cpath}.function"))?.to_string(),
                raw_args: raw_args(&function["arguments"]),
            });
        }
    }
    Ok(Message::Assistant(parts))
}

pub fn finish_reason(finish: Finish) -> &'static str {
    match finish {
        Finish::Stop | Finish::StopSequence => "stop",
        Finish::Length | Finish::ContextFull | Finish::Repetition => "length",
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

/// OpenAI error object: `type` and `code` as OpenAI reports them.
pub fn error_body(error: &ApiError) -> Value {
    let (kind, code) = match error.kind {
        ErrorKind::InvalidRequest => ("invalid_request_error", None),
        ErrorKind::PayloadTooLarge => ("invalid_request_error", Some("request_too_large")),
        ErrorKind::Forbidden => ("invalid_request_error", Some("forbidden")),
        ErrorKind::Unauthorized => ("invalid_request_error", Some("invalid_api_key")),
        ErrorKind::ContextLengthExceeded => {
            ("invalid_request_error", Some("context_length_exceeded"))
        }
        ErrorKind::QueueFull => ("requests", Some("rate_limit_exceeded")),
        ErrorKind::QueueTimeout | ErrorKind::NotReady | ErrorKind::Overloaded => {
            ("server_error", Some("server_busy"))
        }
        ErrorKind::Internal => ("server_error", None),
    };
    json!({"error": {"message": error.message, "type": kind, "param": error.param, "code": code}})
}

/// Streaming encoder: one `chat.completion.chunk` per delta, the role in the first chunk,
/// nothing after `finish_reason` except the usage chunk and `[DONE]`. With `include_usage`
/// every other chunk carries `"usage": null`, as OpenAI's do. It also folds the events into the
/// non-streaming `chat.completion`.
#[derive(Debug)]
pub struct StreamEncoder {
    id: String,
    created: u64,
    model: String,
    include_usage: bool,
    /// Call index → whether arguments were streamed.
    streamed_args: HashMap<u32, bool>,
    done: bool,
    text: String,
    reasoning: String,
    /// `(index, id, name, arguments)` in order.
    calls: Vec<(u32, String, String, String)>,
    finish: Option<(Finish, Usage)>,
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
            text: String::new(),
            reasoning: String::new(),
            calls: Vec::new(),
            finish: None,
        }
    }

    fn envelope(&self, choices: Value) -> Value {
        json!({
            "id": self.id, "object": "chat.completion.chunk", "created": self.created,
            "model": self.model, "choices": choices,
        })
    }

    fn chunk(&self, delta: Value, finish: Option<&str>) -> String {
        let mut chunk =
            self.envelope(json!([{"index": 0, "delta": delta, "finish_reason": finish}]));
        if self.include_usage {
            chunk["usage"] = Value::Null;
        }
        sse::data(&chunk)
    }

    fn call(&mut self, idx: u32) -> Option<&mut (u32, String, String, String)> {
        self.calls.iter_mut().rev().find(|c| c.0 == idx)
    }
}

impl StreamEncode for StreamEncoder {
    /// Frames for one event; `[DONE]` follows the terminal event.
    fn encode(&mut self, event: &GenEvent) -> Vec<String> {
        if self.done {
            return Vec::new();
        }
        match event {
            GenEvent::Started { .. } => {
                vec![self.chunk(json!({"role": "assistant", "content": ""}), None)]
            }
            GenEvent::TextDelta(text) => {
                self.text.push_str(text);
                vec![self.chunk(json!({"content": text}), None)]
            }
            GenEvent::ReasoningDelta(text) => {
                self.reasoning.push_str(text);
                vec![self.chunk(json!({"reasoning_content": text}), None)]
            }
            GenEvent::ToolStart { idx, id, name } => {
                self.streamed_args.insert(*idx, false);
                self.calls
                    .push((*idx, id.clone(), name.clone(), String::new()));
                vec![self.chunk(
                    json!({"tool_calls": [{"index": idx, "id": id, "type": "function",
                        "function": {"name": name, "arguments": ""}}]}),
                    None,
                )]
            }
            GenEvent::ToolArgsDelta { idx, json } => {
                self.streamed_args.insert(*idx, true);
                if let Some(call) = self.call(*idx) {
                    call.3.push_str(json);
                }
                vec![self.chunk(
                    json!({"tool_calls": [{"index": idx, "function": {"arguments": json}}]}),
                    None,
                )]
            }
            GenEvent::ToolEnd { idx, raw_args, .. } => {
                if let Some(call) = self.call(*idx) {
                    call.3.clone_from(raw_args);
                }
                // Buffered arguments go out in one fragment.
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
                self.finish = Some((*reason, *usage));
                let mut frames = vec![self.chunk(json!({}), Some(finish_reason(*reason)))];
                if self.include_usage {
                    let mut chunk = self.envelope(json!([]));
                    chunk["usage"] = self::usage(*usage);
                    frames.push(sse::data(&chunk));
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

    /// A data event, not an SSE comment: OpenCode resets its timeout only on parsed events
    /// (D047). An empty `choices` chunk, like OpenAI's usage chunk.
    fn keepalive(&mut self) -> String {
        let mut chunk = self.envelope(json!([]));
        if self.include_usage {
            chunk["usage"] = Value::Null;
        }
        sse::data(&chunk)
    }

    fn snapshot(&self) -> Value {
        let (finish, totals) = self.finish.unwrap_or((Finish::Stop, Usage::default()));
        let calls: Vec<Value> = self
            .calls
            .iter()
            .map(|(_, id, name, args)| {
                json!({"id": id, "type": "function",
                    "function": {"name": name, "arguments": args}})
            })
            .collect();
        let content = if self.text.is_empty() && !calls.is_empty() {
            Value::Null
        } else {
            self.text.clone().into()
        };
        let mut message = json!({"role": "assistant", "content": content});
        if !self.reasoning.is_empty() {
            message["reasoning_content"] = self.reasoning.clone().into();
        }
        if !calls.is_empty() {
            message["tool_calls"] = Value::Array(calls);
        }
        json!({
            "id": self.id,
            "object": "chat.completion",
            "created": self.created,
            "model": self.model,
            "choices": [{"index": 0, "message": message,
                "finish_reason": finish_reason(finish)}],
            "usage": usage(totals),
        })
    }
}

/// `POST /v1/chat/completions`
#[derive(Debug)]
pub struct Chat;

impl Api for Chat {
    const KIND: ApiKind = ApiKind::Chat;
    type Encoder = StreamEncoder;

    fn decode(body: &Value, options: DecodeOptions) -> Result<ChatRequest, ApiError> {
        decode(body, options)
    }

    fn encoder(model: &str, request: &ChatRequest) -> StreamEncoder {
        StreamEncoder::new(model, request.include_usage)
    }

    fn error_body(error: &ApiError) -> Value {
        error_body(error)
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
    fn repetition_penalty_is_read_and_checked() {
        let body = |penalty: Value| json!({"messages": [{"role": "user", "content": "hi"}], "repetition_penalty": penalty});
        let request = decode_ok(body(json!(1.2)));
        assert_eq!(request.sampling.repetition_penalty, Some(1.2));
        assert!(KNOWN.contains(&"repetition_penalty"));
        for bad in [json!(0), json!(-0.5)] {
            let error = decode_ok(body(bad))
                .sampling
                .resolve_valid(mrun_engine::Sampling::GREEDY)
                .unwrap_err();
            assert_eq!(error.param.as_deref(), Some("repetition_penalty"));
            assert_eq!(error.status(ApiKind::Chat).as_u16(), 400);
        }
        let error = decode(&body(json!("high")), DecodeOptions::default()).unwrap_err();
        assert_eq!(error.param.as_deref(), Some("repetition_penalty"));
    }

    fn decode_ok(body: Value) -> ChatRequest {
        decode(&body, DecodeOptions::default()).unwrap()
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
                    speculation: None,
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
        // chat_include_usage_null_on_chunks
        assert!(
            chunks[..5]
                .iter()
                .all(|c| c["usage"].is_null() && c.get("usage").is_some())
        );
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
    fn keepalive_is_a_data_event_with_no_choices() {
        let mut encoder = StreamEncoder::new("m", true);
        let frame = encoder.keepalive();
        let chunk = &frames_json(&[frame])[0];
        assert_eq!(chunk["object"], "chat.completion.chunk");
        assert_eq!(chunk["choices"], json!([]));
        assert_eq!(chunk["model"], "m");
        assert!(chunk["created"].is_u64() && chunk["id"].is_string());
    }

    #[test]
    fn non_streaming_body() {
        let mut encoder = StreamEncoder::new("m", false);
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
                    speculation: None,
                },
            },
        ] {
            encoder.encode(&event);
        }
        let body = encoder.snapshot();
        assert_eq!(body["choices"][0]["message"]["content"], "Hello");
        assert_eq!(body["choices"][0]["message"]["reasoning_content"], "hm");
        assert_eq!(body["choices"][0]["finish_reason"], "length");
        assert_eq!(body["usage"]["total_tokens"], 5);
    }

    #[test]
    fn absent_reasoning_effort_keeps_the_models_thinking_on() {
        // D063: OpenCode and other OpenAI-API clients never send reasoning_effort.
        let on = |body| {
            decode(&body, DecodeOptions::default())
                .unwrap()
                .reasoning
                .is_on()
        };
        assert!(on(json!({"messages": [{"role": "user", "content": "x"}]})));
        assert!(on(
            json!({"messages": [{"role": "user", "content": "x"}], "reasoning_effort": "low"})
        ));
        assert!(!on(
            json!({"messages": [{"role": "user", "content": "x"}], "reasoning_effort": "none"})
        ));
        assert!(!on(
            json!({"messages": [{"role": "user", "content": "x"}], "reasoning_effort": "minimal"})
        ));
    }

    #[test]
    fn enable_thinking_switches_thinking_over_reasoning_effort() {
        let body = |kwargs: Value| {
            json!({"messages": [{"role": "user", "content": "x"}],
                   "reasoning_effort": "low", "chat_template_kwargs": kwargs})
        };
        let on = |kwargs| {
            decode(&body(kwargs), DecodeOptions::default())
                .unwrap()
                .reasoning
                .is_on()
        };
        // OpenCode's small model: `reasoning_effort: "low"` and the switch off.
        assert!(!on(json!({"enable_thinking": false})));
        assert!(on(json!({"enable_thinking": true})));
        // Without the switch (or with other keys only) reasoning_effort decides, as before.
        assert!(on(json!({})));
        assert!(on(json!({"add_generation_prompt": true})));
        let error = decode(
            &body(json!({"enable_thinking": "no"})),
            DecodeOptions::default(),
        )
        .unwrap_err();
        assert_eq!(
            error.param.as_deref(),
            Some("chat_template_kwargs.enable_thinking")
        );
        let error = decode(&body(json!(true)), DecodeOptions::default()).unwrap_err();
        assert_eq!(error.param.as_deref(), Some("chat_template_kwargs"));
    }
}
