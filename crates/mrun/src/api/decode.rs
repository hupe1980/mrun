//! Helpers shared by the three request decoders: typed field access with JSON paths in errors,
//! and the request limits checked before queuing (002 data model, validation).

use crate::api::model::{
    ApiError, ChatRequest, NAMESPACE_SEP, ResponseFormat, ToolChoice, ToolDef,
};
use crate::chat::stop::StopMatcher;
use serde_json::{Value, json};

/// Text standing in for an image inside a tool result (FR-020).
pub const IMAGE_PLACEHOLDER: &str = "[image omitted: model has no vision input]";

/// Server settings the decoders apply.
#[derive(Clone, Copy, Debug, Default)]
pub struct DecodeOptions {
    /// `--reject-tool-images`: refuse images in tool results instead of replacing them.
    pub reject_tool_images: bool,
}

/// An image inside a tool result: replaced by the placeholder, or refused.
pub fn tool_image(options: DecodeOptions, path: &str) -> Result<&'static str, ApiError> {
    if options.reject_tool_images {
        Err(ApiError::invalid(
            "images in tool results are not supported (--reject-tool-images)",
            path,
        ))
    } else {
        Ok(IMAGE_PLACEHOLDER)
    }
}

/// Paths of the fields of `body` outside `known`, each prefixed with `prefix`. Names that are
/// not plain identifiers come out as `(unnamed)`, so no content reaches the log.
pub fn unknown_fields(body: &Value, known: &[&str], prefix: &str) -> Vec<String> {
    let Value::Object(map) = body else {
        return Vec::new();
    };
    map.keys()
        .filter(|k| !known.contains(&k.as_str()))
        .map(|name| {
            let plain = name.len() <= 64
                && name
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-');
            if plain {
                format!("{prefix}{name}")
            } else {
                format!("{prefix}(unnamed)")
            }
        })
        .collect()
}

/// Logs `ignored-field {path}` once per process and path for each field of `body` outside
/// `known` (FR-041, D055: every field is accepted; an unknown one is ignored and named in the
/// log).
pub fn ignore_unknown(body: &Value, known: &[&str], prefix: &str) {
    static LOGGED: std::sync::Mutex<Vec<String>> = std::sync::Mutex::new(Vec::new());
    for name in unknown_fields(body, known, prefix) {
        let mut logged = LOGGED.lock().unwrap_or_else(|e| e.into_inner());
        if !logged.contains(&name) {
            crate::api::log::ignored_field(&name);
            logged.push(name);
        }
    }
}

/// Most messages per request.
pub const MAX_MESSAGES: usize = 4_096;
/// Most tools per request.
pub const MAX_TOOLS: usize = 256;
/// Largest tool or response schema, serialized.
pub const MAX_SCHEMA_BYTES: usize = 64 * 1024;
/// Deepest tool or response schema.
pub const MAX_SCHEMA_DEPTH: usize = 16;

pub fn obj<'a>(
    value: &'a Value,
    path: &str,
) -> Result<&'a serde_json::Map<String, Value>, ApiError> {
    value
        .as_object()
        .ok_or_else(|| ApiError::invalid(format!("{path} must be an object"), path))
}

pub fn arr<'a>(value: &'a Value, path: &str) -> Result<&'a Vec<Value>, ApiError> {
    value
        .as_array()
        .ok_or_else(|| ApiError::invalid(format!("{path} must be an array"), path))
}

pub fn str_field<'a>(value: &'a Value, key: &str, path: &str) -> Result<&'a str, ApiError> {
    value[key].as_str().ok_or_else(|| {
        ApiError::invalid(
            format!("{path}.{key} must be a string"),
            format!("{path}.{key}"),
        )
    })
}

pub fn opt_str<'a>(value: &'a Value, key: &str, path: &str) -> Result<Option<&'a str>, ApiError> {
    match &value[key] {
        Value::Null => Ok(None),
        Value::String(s) => Ok(Some(s)),
        _ => Err(ApiError::invalid(
            format!("{path}.{key} must be a string"),
            format!("{path}.{key}"),
        )),
    }
}

pub fn opt_u32(value: &Value, key: &str) -> Result<Option<u32>, ApiError> {
    match &value[key] {
        Value::Null => Ok(None),
        v => v
            .as_u64()
            .map(|n| Some(n.min(u64::from(u32::MAX)) as u32))
            .ok_or_else(|| ApiError::invalid(format!("{key} must be a non-negative integer"), key)),
    }
}

pub fn opt_f32(value: &Value, key: &str) -> Result<Option<f32>, ApiError> {
    match &value[key] {
        Value::Null => Ok(None),
        v => v
            .as_f64()
            .map(|n| Some(n as f32))
            .ok_or_else(|| ApiError::invalid(format!("{key} must be a number"), key)),
    }
}

pub fn opt_bool(value: &Value, key: &str) -> Result<Option<bool>, ApiError> {
    match &value[key] {
        Value::Null => Ok(None),
        Value::Bool(b) => Ok(Some(*b)),
        _ => Err(ApiError::invalid(format!("{key} must be a boolean"), key)),
    }
}

/// `stop` / `stop_sequences`: a string or an array of strings.
pub fn stops(value: &Value, key: &str) -> Result<Vec<String>, ApiError> {
    let stops = match &value[key] {
        Value::Null => Vec::new(),
        Value::String(s) => vec![s.clone()],
        Value::Array(items) => items
            .iter()
            .map(|s| {
                s.as_str()
                    .map(str::to_owned)
                    .ok_or_else(|| ApiError::invalid("stop sequences must be strings", key))
            })
            .collect::<Result<_, _>>()?,
        _ => return Err(ApiError::invalid("stop must be a string or an array", key)),
    };
    StopMatcher::new(stops.clone()).map_err(|e| ApiError::invalid(e.to_string(), key))?;
    Ok(stops)
}

/// Depth of nested objects and arrays.
fn depth(value: &Value) -> usize {
    match value {
        Value::Object(map) => 1 + map.values().map(depth).max().unwrap_or(0),
        Value::Array(items) => 1 + items.iter().map(depth).max().unwrap_or(0),
        _ => 0,
    }
}

pub fn check_schema(schema: &Value, path: &str) -> Result<(), ApiError> {
    if serde_json::to_string(schema).map_or(0, |s| s.len()) > MAX_SCHEMA_BYTES {
        return Err(ApiError::invalid(
            format!("schema is larger than {MAX_SCHEMA_BYTES} bytes"),
            path,
        ));
    }
    if depth(schema) > MAX_SCHEMA_DEPTH {
        return Err(ApiError::invalid(
            format!("schema is nested deeper than {MAX_SCHEMA_DEPTH} levels"),
            path,
        ));
    }
    Ok(())
}

pub fn check_tool(tool: &ToolDef, path: &str) -> Result<(), ApiError> {
    if tool.name.is_empty() {
        return Err(ApiError::invalid("tool name must not be empty", path));
    }
    check_schema(&tool.schema, path)
}

/// Request-wide limits, on the raw arrays before anything is decoded or copied.
pub fn check_counts(body: &Value, messages: &str) -> Result<(), ApiError> {
    let len = |key: &str| body[key].as_array().map_or(0, Vec::len);
    if len(messages) > MAX_MESSAGES {
        return Err(ApiError::invalid(
            format!("more than {MAX_MESSAGES} messages"),
            messages,
        ));
    }
    if len("tools") > MAX_TOOLS {
        return Err(ApiError::invalid(
            format!("more than {MAX_TOOLS} tools"),
            "tools",
        ));
    }
    Ok(())
}

/// Tool-call arguments as text: a string verbatim, an object as JSON, nothing as `{}`.
pub fn raw_args(value: &Value) -> String {
    match value {
        Value::String(s) => s.clone(),
        Value::Null => "{}".to_string(),
        other => other.to_string(),
    }
}

/// A string, or the text of the parts `text_of` accepts, joined; any other part is refused
/// with its path and `unsupported` as the message.
pub fn join_text_parts<'a>(
    content: &'a Value,
    path: &str,
    text_of: impl Fn(&'a Value) -> Option<&'a str>,
    unsupported: &str,
) -> Result<String, ApiError> {
    match content {
        Value::Null => Ok(String::new()),
        Value::String(s) => Ok(s.clone()),
        Value::Array(parts) => {
            let mut text = String::new();
            for (j, part) in parts.iter().enumerate() {
                match text_of(part) {
                    Some(t) => text.push_str(t),
                    None => {
                        return Err(ApiError::invalid(
                            unsupported,
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

/// The text of a part whose `type` is one of `kinds`.
pub fn text_part<'a>(part: &'a Value, kinds: &[&str]) -> Option<&'a str> {
    let kind = part["type"].as_str()?;
    kinds
        .contains(&kind)
        .then(|| part["text"].as_str().unwrap_or_default())
}

/// A function tool from its definition object (`name`, `description`, the schema under
/// `schema_key`, `strict`, `defer_loading`), flattened into `namespace` when it came from one.
pub fn function_tool(
    tool: &Value,
    path: &str,
    schema_key: &str,
    namespace: Option<&str>,
) -> Result<ToolDef, ApiError> {
    let name = str_field(tool, "name", path)?;
    let def = ToolDef {
        name: match namespace {
            Some(namespace) => format!("{namespace}{NAMESPACE_SEP}{name}"),
            None => name.to_string(),
        },
        description: opt_str(tool, "description", path)?
            .unwrap_or_default()
            .to_string(),
        schema: match &tool[schema_key] {
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

/// OpenAI `tool_choice`: `auto`, `none`, `required`, or a function named by `name` or
/// `function.name`; a named tool must be declared.
pub fn openai_tool_choice(request: &ChatRequest, choice: &Value) -> Result<ToolChoice, ApiError> {
    let choice = match choice {
        Value::Null => ToolChoice::Auto,
        Value::String(s) => match s.as_str() {
            "auto" => ToolChoice::Auto,
            "none" => ToolChoice::None,
            "required" => ToolChoice::Required,
            _ => return Err(ApiError::invalid("unknown tool_choice", "tool_choice")),
        },
        Value::Object(_) if choice["type"] == "function" => {
            let name = choice["function"]["name"]
                .as_str()
                .or(choice["name"].as_str())
                .ok_or_else(|| {
                    ApiError::invalid("tool_choice must name a function", "tool_choice")
                })?;
            ToolChoice::Named(name.to_string())
        }
        _ => {
            return Err(ApiError::invalid(
                "tool_choice must be auto, none, required or a function",
                "tool_choice",
            ));
        }
    };
    declared(request, &choice, "tool_choice")?;
    Ok(choice)
}

/// A named tool choice must name a declared tool.
pub fn declared(request: &ChatRequest, choice: &ToolChoice, path: &str) -> Result<(), ApiError> {
    match choice {
        ToolChoice::Named(name) if request.tool(name).is_none() => Err(ApiError::invalid(
            format!("tool_choice names an undeclared tool '{name}'"),
            path,
        )),
        _ => Ok(()),
    }
}

/// OpenAI response format: `type` from `format`, `name`/`schema`/`strict` from `spec` (Chat
/// nests them under `json_schema`, Responses does not).
pub fn response_format(
    format: &Value,
    spec: &Value,
    path: &str,
) -> Result<ResponseFormat, ApiError> {
    Ok(match format {
        Value::Null => ResponseFormat::Text,
        format => match format["type"].as_str() {
            Some("text") => ResponseFormat::Text,
            Some("json_object") => ResponseFormat::JsonObject,
            Some("json_schema") => {
                let schema = spec["schema"].clone();
                check_schema(&schema, &format!("{path}.schema"))?;
                ResponseFormat::JsonSchema {
                    name: spec["name"].as_str().unwrap_or("response").to_string(),
                    schema,
                    strict: spec["strict"].as_bool().unwrap_or(false),
                }
            }
            _ => {
                return Err(ApiError::invalid(
                    format!("unsupported {path} type"),
                    format!("{path}.type"),
                ));
            }
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn xapi_unknown_field_ignored_field_logged() {
        use crate::api::{chat_completions, messages, responses};
        let body = json!({"model": "m", "frobnicate": 1, "weird key!": 2});
        for known in [chat_completions::KNOWN, messages::KNOWN, responses::KNOWN] {
            assert_eq!(
                unknown_fields(&body, known, ""),
                vec!["frobnicate".to_string(), "(unnamed)".to_string()]
            );
        }
        assert_eq!(
            crate::api::log::line(crate::api::log::Kind::IgnoredField, "frobnicate"),
            "ignored-field frobnicate"
        );
    }

    #[test]
    fn chat_unknown_field_ignored() {
        let body = json!({"messages": [{"role": "user", "content": "hi"}],
            "frequency_penalty": 0.5, "logit_bias": {"1": 2}, "frobnicate": true});
        let request =
            crate::api::chat_completions::decode(&body, DecodeOptions::default()).unwrap();
        assert_eq!(request.messages.len(), 1);
        assert_eq!(
            unknown_fields(&body, crate::api::chat_completions::KNOWN, ""),
            ["frequency_penalty", "logit_bias", "frobnicate"]
        );
    }

    #[test]
    fn msg_output_config_effort_task_budget_ignored_logged() {
        use crate::api::messages;
        let body = json!({"max_tokens": 10, "messages": [{"role": "user", "content": "hi"}],
            "output_config": {"effort": "high", "task_budget": {"type": "tokens", "total": 1000}}});
        // Accepted: the request decodes and nothing of the two fields reaches it.
        let request = messages::decode(&body, DecodeOptions::default()).unwrap();
        assert_eq!(
            request.response_format,
            crate::api::model::ResponseFormat::Text
        );
        assert_eq!(
            unknown_fields(
                &body["output_config"],
                messages::KNOWN_OUTPUT_CONFIG,
                "output_config."
            ),
            ["output_config.effort", "output_config.task_budget"]
        );
        assert!(unknown_fields(&body, messages::KNOWN, "").is_empty());
    }

    #[test]
    fn xapi_schema_depth_limit() {
        let mut schema = json!({"type": "string"});
        for _ in 0..MAX_SCHEMA_DEPTH {
            schema = json!({"type": "object", "properties": {"x": schema}});
        }
        let error = check_schema(&schema, "tools[0]").unwrap_err();
        assert_eq!(error.param.as_deref(), Some("tools[0]"));
        assert!(check_schema(&json!({"type": "object", "properties": {}}), "t").is_ok());
    }

    #[test]
    fn counts_are_checked_on_the_raw_arrays() {
        let many = json!({"messages": vec![json!({}); MAX_MESSAGES + 1]});
        let error = check_counts(&many, "messages").unwrap_err();
        assert_eq!(error.param.as_deref(), Some("messages"));
        let tools = json!({"input": [], "tools": vec![json!({}); MAX_TOOLS + 1]});
        assert_eq!(
            check_counts(&tools, "input").unwrap_err().param.as_deref(),
            Some("tools")
        );
        assert!(check_counts(&json!({"messages": "x"}), "messages").is_ok());
    }

    #[test]
    fn shared_field_helpers() {
        assert_eq!(raw_args(&json!({"a": 1})), r#"{"a":1}"#);
        assert_eq!(raw_args(&json!(null)), "{}");
        assert_eq!(raw_args(&json!("x")), "x");
        let parts = json!([{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]);
        let joined = join_text_parts(&parts, "m", |p| text_part(p, &["text"]), "no").unwrap();
        assert_eq!(joined, "ab");
        let bad = json!([{"type": "image"}]);
        let error = join_text_parts(&bad, "m", |p| text_part(p, &["text"]), "no").unwrap_err();
        assert_eq!(error.param.as_deref(), Some("m.content[0]"));
        let mut request = ChatRequest::new(crate::api::model::ApiKind::Chat);
        request.tools =
            vec![function_tool(&json!({"name": "f"}), "tools[0]", "parameters", None).unwrap()];
        for choice in [
            json!({"type": "function", "function": {"name": "f"}}),
            json!({"type": "function", "name": "f"}),
        ] {
            assert_eq!(
                openai_tool_choice(&request, &choice).unwrap(),
                ToolChoice::Named("f".into())
            );
        }
        assert!(openai_tool_choice(&request, &json!({"type": "function", "name": "g"})).is_err());
        let format = json!({"type": "json_schema", "json_schema": {"name": "x", "schema": {"type": "object"}, "strict": true}});
        assert!(matches!(
            response_format(&format, &format["json_schema"], "response_format").unwrap(),
            ResponseFormat::JsonSchema { strict: true, .. }
        ));
    }

    #[test]
    fn stops_accept_string_or_array_within_limits() {
        assert_eq!(stops(&json!({"stop": "END"}), "stop").unwrap(), vec!["END"]);
        assert!(stops(&json!({"stop": ["a", "b", "c", "d", "e"]}), "stop").is_err());
        assert!(stops(&json!({}), "stop").unwrap().is_empty());
    }
}
