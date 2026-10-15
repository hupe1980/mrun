//! Helpers shared by the three request decoders: typed field access with JSON paths in errors,
//! and the request limits checked before queuing (002 data model, validation).

use crate::api::model::{ApiError, ChatRequest, ToolDef};
use crate::chat::stop::StopMatcher;
use serde_json::Value;

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

/// Request-wide limits, after decoding.
pub fn check_limits(request: &ChatRequest) -> Result<(), ApiError> {
    if request.messages.len() > MAX_MESSAGES {
        return Err(ApiError::invalid(
            format!("more than {MAX_MESSAGES} messages"),
            "messages",
        ));
    }
    if request.tools.len() > MAX_TOOLS {
        return Err(ApiError::invalid(
            format!("more than {MAX_TOOLS} tools"),
            "tools",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

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
    fn stops_accept_string_or_array_within_limits() {
        assert_eq!(stops(&json!({"stop": "END"}), "stop").unwrap(), vec!["END"]);
        assert!(stops(&json!({"stop": ["a", "b", "c", "d", "e"]}), "stop").is_err());
        assert!(stops(&json!({}), "stop").unwrap().is_empty());
    }
}
