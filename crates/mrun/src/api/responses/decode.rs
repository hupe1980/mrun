//! Responses request decoding: input items into turns, tools (functions, namespaces, Codex's
//! client-executed tool search) and options.

use super::naming::TOOL_SEARCH;
use crate::api::chat_completions::openai_effort;
use crate::api::decode::{
    DecodeOptions, arr, check_counts, function_tool, join_text_parts, obj, openai_tool_choice,
    opt_bool, opt_f32, opt_str, opt_u32, raw_args, response_format, str_field, text_part,
    tool_image,
};
use crate::api::model::{
    ApiError, ApiKind, AssistantPart, ChatRequest, Message, NAMESPACE_SEP, ToolDef, ToolResult,
};
use crate::chat::reasoning;
use serde_json::{Value, json};

/// Top-level fields mrun reads, rejects or knowingly ignores (Codex's included, FR-042); any
/// other is logged `ignored-field`.
pub const KNOWN: &[&str] = &[
    "model",
    "input",
    "instructions",
    "max_output_tokens",
    "parallel_tool_calls",
    "repetition_penalty",
    "stream",
    "temperature",
    "top_p",
    "tools",
    "tool_choice",
    "reasoning",
    "text",
    "previous_response_id",
    "conversation",
    "prompt_cache_key",
    "client_metadata",
    "stream_options",
    "service_tier",
    "include",
    "store",
    "metadata",
    "user",
];

pub fn decode(body: &Value, options: DecodeOptions) -> Result<ChatRequest, ApiError> {
    obj(body, "body")?;
    crate::api::decode::ignore_unknown(body, KNOWN, "");
    check_counts(body, "input")?;
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
                "function" => request
                    .tools
                    .push(function_tool(tool, &path, "parameters", None)?),
                // MCP and sub-agent tools (FR-027): flattened to `<namespace>__<tool>`.
                "namespace" => {
                    let namespace = str_field(tool, "name", &path)?;
                    for (j, inner) in arr(&tool["tools"], &format!("{path}.tools"))?
                        .iter()
                        .enumerate()
                    {
                        let ipath = format!("{path}.tools[{j}]");
                        request.tools.push(function_tool(
                            inner,
                            &ipath,
                            "parameters",
                            Some(namespace),
                        )?);
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

    request.tool_choice = openai_tool_choice(&request, &body["tool_choice"])?;
    request.parallel_tool_calls = opt_bool(body, "parallel_tool_calls")?.unwrap_or(true);
    let format = &body["text"]["format"];
    request.response_format = response_format(format, format, "text.format")?;
    request.max_output_tokens = opt_u32(body, "max_output_tokens")?;
    request.sampling.temperature = opt_f32(body, "temperature")?;
    request.sampling.top_p = opt_f32(body, "top_p")?;
    request.sampling.repetition_penalty = opt_f32(body, "repetition_penalty")?;
    // Codex always sends `reasoning` (without an effort for unknown models): no effort means the
    // model's default, thinking on (D063); `none`/`minimal` turn it off.
    request.reasoning = match &body["reasoning"] {
        Value::Null => openai_effort(None),
        reasoning => {
            obj(reasoning, "reasoning")?;
            openai_effort(opt_str(reasoning, "effort", "reasoning")?)
        }
    };
    request.stream = opt_bool(body, "stream")?.unwrap_or(false);
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
                            request.messages.push(Message::User(text));
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
                let raw_args = raw_args(&item["arguments"]);
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
                let raw_args = raw_args(&item["arguments"]);
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
                                    "parameters",
                                    Some(namespace),
                                )?);
                            }
                        }
                        _ => found.push(function_tool(tool, &tpath, "parameters", None)?),
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
    join_text_parts(
        content,
        path,
        |part| match part["type"].as_str() {
            Some("refusal") if !user => Some(part["refusal"].as_str().unwrap_or_default()),
            _ => text_part(part, &["input_text", "output_text", "text"]),
        },
        "only text content is supported (the model has no vision input)",
    )
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
