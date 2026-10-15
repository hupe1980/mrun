//! One timed request against a system under test, in any of the three API styles (005 FR-013):
//! the request is streamed and every timing is taken client-side from the bytes as they arrive.
//!
//! Time to first token ends at the first content event: the first `content_block_delta`
//! (Messages), the first chunk with non-empty `content`, `reasoning_content` or `tool_calls`
//! (Chat Completions), the first `response.output_text.delta` or
//! `response.function_call_arguments.delta` (Responses). Pings, SSE comments, keepalives and
//! role-only chunks never count. Time to first useful token ends at the first visible text or
//! completed tool call. Token counts come from the usage fields, completed by mrun's `: mrun`
//! comment (or `x-mrun-cache-*` headers), which also carry the miss cause and message index.

use crate::api::model::ApiKind;
use serde::Deserialize;
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::time::Instant;

/// How a system is told to ignore end-of-sequence (FR-007).
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum IgnoreEos {
    /// Not supported: replies may end early (counted).
    #[default]
    None,
    /// mrun's `x-mrun-ignore-eos: on` request header (002 FR-031).
    MrunHeader,
    /// `"ignore_eos": true` in the body (llama.cpp).
    Body,
}

/// Why generation ended.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum Finish {
    #[default]
    Unknown,
    /// The output limit.
    Length,
    /// End of sequence, a stop sequence or a tool call.
    Stop,
}

/// What one request measured.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Outcome {
    pub first_byte_ms: Option<f64>,
    pub ttft_ms: Option<f64>,
    pub tfut_ms: Option<f64>,
    /// Arrival of each content event, from request send.
    pub token_times_ms: Vec<f64>,
    pub prompt_tokens: Option<u32>,
    pub cached_tokens: Option<u32>,
    pub output_tokens: Option<u32>,
    /// mrun's miss cause and the API message it points at.
    pub cause: Option<String>,
    pub message_index: Option<u32>,
    pub finish: Finish,
    /// `(name, raw arguments)` of each tool call.
    pub tool_calls: Vec<(String, String)>,
}

impl Outcome {
    /// Send → last content event (TTFT + fixed-length decode).
    pub fn total_ms(&self) -> Option<f64> {
        self.token_times_ms.last().copied()
    }

    /// Decode rate between the first and last content event.
    pub fn decode_tok_s(&self) -> Option<f64> {
        let (first, last) = (self.token_times_ms.first()?, self.token_times_ms.last()?);
        let tokens = self
            .output_tokens
            .unwrap_or(self.token_times_ms.len() as u32);
        (last > first && tokens > 1).then(|| f64::from(tokens - 1) * 1000.0 / (last - first))
    }

    /// Ended before `max_tokens` although the limit was the goal (FR-003a, FR-007).
    pub fn early_eos(&self, max_tokens: u32) -> bool {
        self.finish != Finish::Length && self.output_tokens.is_none_or(|n| n < max_tokens)
    }
}

/// The 2% rule of FR-015. Only a shortfall fails: counts above the expected one are template
/// overhead, which the expected count (visible text) leaves out, never truncation.
pub fn truncated(reported: Option<u32>, expected: u32) -> bool {
    match reported {
        Some(reported) if expected > 0 => u64::from(reported) * 100 < u64::from(expected) * 98,
        _ => false,
    }
}

pub fn path(api: ApiKind) -> &'static str {
    match api {
        ApiKind::Chat => "/v1/chat/completions",
        ApiKind::Messages => "/v1/messages",
        ApiKind::Responses => "/v1/responses",
    }
}

/// The request as sent: streamed, with the system's model name, the output limit and usage
/// reporting; everything else unchanged.
pub fn prepare(
    api: ApiKind,
    body: &Value,
    model: Option<&str>,
    max_tokens: u32,
    ignore_eos: IgnoreEos,
) -> Value {
    let mut body = body.clone();
    if let Some(model) = model.filter(|m| !m.is_empty()) {
        body["model"] = model.into();
    }
    body["stream"] = true.into();
    match api {
        ApiKind::Chat => {
            body["max_tokens"] = max_tokens.into();
            if body.get("max_completion_tokens").is_some() {
                body["max_completion_tokens"] = max_tokens.into();
            }
            body["stream_options"] = json!({"include_usage": true});
        }
        ApiKind::Messages => body["max_tokens"] = max_tokens.into(),
        ApiKind::Responses => body["max_output_tokens"] = max_tokens.into(),
    }
    if ignore_eos == IgnoreEos::Body {
        body["ignore_eos"] = true.into();
    }
    body
}

/// An HTTP client that talks to the given systems only: no proxy (FR-011).
pub fn client() -> reqwest::Client {
    reqwest::Client::builder()
        .no_proxy()
        .connect_timeout(std::time::Duration::from_secs(5))
        .build()
        .unwrap_or_default()
}

/// Sends `body` (already [`prepare`]d) and reads the stream to its end, or until `stop_after`
/// content events, when the connection is dropped (cancellation).
pub async fn send(
    client: &reqwest::Client,
    base_url: &str,
    api: ApiKind,
    body: &Value,
    ignore_eos: IgnoreEos,
    stop_after: Option<usize>,
) -> Result<Outcome, String> {
    let url = format!("{}{}", base_url.trim_end_matches('/'), path(api));
    let mut request = client
        .post(&url)
        .header("content-type", "application/json")
        .body(body.to_string());
    if api == ApiKind::Messages {
        request = request.header("anthropic-version", "2023-06-01");
    }
    if ignore_eos == IgnoreEos::MrunHeader {
        request = request.header("x-mrun-ignore-eos", "on");
    }
    let start = Instant::now();
    let mut response = request
        .send()
        .await
        .map_err(|e| format!("unreachable: {e}"))?;
    let status = response.status();
    if !status.is_success() {
        let text = response.text().await.unwrap_or_default();
        return Err(format!("HTTP {}: {}", status.as_u16(), excerpt(&text)));
    }
    let mut reader = Reader::new(api);
    reader.headers(response.headers());
    if stop_after == Some(0) {
        return Ok(reader.finish(start.elapsed().as_secs_f64() * 1000.0));
    }
    let mut buffer = String::new();
    while let Some(chunk) = response.chunk().await.map_err(|e| e.to_string())? {
        let at = start.elapsed().as_secs_f64() * 1000.0;
        if chunk.is_empty() {
            continue;
        }
        reader.out.first_byte_ms.get_or_insert(at);
        buffer.push_str(&String::from_utf8_lossy(&chunk).replace("\r\n", "\n"));
        while let Some(end) = buffer.find("\n\n") {
            let block: String = buffer.drain(..end + 2).collect();
            reader.block(&block, at)?;
        }
        if stop_after.is_some_and(|n| reader.out.token_times_ms.len() >= n) {
            return Ok(reader.finish(at));
        }
    }
    let at = start.elapsed().as_secs_f64() * 1000.0;
    Ok(reader.finish(at))
}

fn excerpt(text: &str) -> String {
    text.chars().take(200).collect()
}

/// Folds SSE blocks into an [`Outcome`].
#[derive(Debug)]
struct Reader {
    api: ApiKind,
    out: Outcome,
    /// Chat: the index of the call being streamed, by index → (name, arguments).
    open_call: Option<u64>,
    calls: BTreeMap<u64, (String, String)>,
    /// Messages: the open `tool_use` block.
    block_call: Option<(String, String)>,
}

impl Reader {
    fn new(api: ApiKind) -> Self {
        Self {
            api,
            out: Outcome::default(),
            open_call: None,
            calls: BTreeMap::new(),
            block_call: None,
        }
    }

    /// `x-mrun-cache-*` headers of non-streamed replies.
    fn headers(&mut self, headers: &reqwest::header::HeaderMap) {
        let get = |name: &str| headers.get(name).and_then(|v| v.to_str().ok());
        let num = |name: &str| get(name).and_then(|v| v.parse::<u32>().ok());
        if let (Some(cached), Some(processed)) =
            (num("x-mrun-cache-cached"), num("x-mrun-cache-processed"))
        {
            self.out.cached_tokens = Some(cached);
            self.out.prompt_tokens = Some(cached + processed);
        }
        self.out.cause = get("x-mrun-cache-cause").map(Into::into);
        self.out.message_index = get("x-mrun-cache-message")
            .and_then(|m| m.split(':').next())
            .and_then(|i| i.parse().ok());
    }

    fn block(&mut self, block: &str, at: f64) -> Result<(), String> {
        let mut name = None;
        let mut data = String::new();
        for line in block.lines() {
            if let Some(comment) = line.strip_prefix(':') {
                self.comment(comment.trim_start());
            } else if let Some(n) = line.strip_prefix("event:") {
                name = Some(n.trim().to_string());
            } else if let Some(d) = line.strip_prefix("data:") {
                if !data.is_empty() {
                    data.push('\n');
                }
                data.push_str(d.strip_prefix(' ').unwrap_or(d));
            }
        }
        if data.is_empty() || data == "[DONE]" {
            return Ok(());
        }
        let Ok(value) = serde_json::from_str::<Value>(&data) else {
            return Ok(());
        };
        let name = name.or_else(|| value["type"].as_str().map(Into::into));
        match self.api {
            ApiKind::Chat => self.chat(&value, at),
            ApiKind::Messages => self.messages(name.as_deref().unwrap_or(""), &value, at),
            ApiKind::Responses => self.responses(name.as_deref().unwrap_or(""), &value, at),
        }
    }

    /// `: mrun {json}`: reuse and the miss report; the usage fields take precedence.
    fn comment(&mut self, text: &str) {
        let Some(meta) = text
            .strip_prefix("mrun ")
            .and_then(|j| serde_json::from_str::<Value>(j).ok())
        else {
            return;
        };
        let num = |key: &str| meta[key].as_u64().map(|n| n as u32);
        if let (Some(cached), Some(processed)) = (num("cached"), num("processed")) {
            self.out.cached_tokens.get_or_insert(cached);
            self.out.prompt_tokens.get_or_insert(cached + processed);
        }
        if let Some(cause) = meta["cause"].as_str() {
            self.out.cause = Some(cause.into());
        }
        if let Some(index) = meta["message"]["index"].as_u64() {
            self.out.message_index = Some(index as u32);
        }
    }

    fn content(&mut self, at: f64) {
        self.out.ttft_ms.get_or_insert(at);
        self.out.token_times_ms.push(at);
    }

    fn useful(&mut self, at: f64) {
        self.out.tfut_ms.get_or_insert(at);
    }

    /// Usage reported by the system wins over the `: mrun` estimate.
    fn usage(&mut self, prompt: Option<u64>, cached: Option<u64>, output: Option<u64>) {
        if let Some(p) = prompt {
            self.out.prompt_tokens = Some(p as u32);
        }
        if let Some(c) = cached {
            self.out.cached_tokens = Some(c as u32);
        }
        if let Some(o) = output {
            self.out.output_tokens = Some(o as u32);
        }
    }

    fn close_chat_call(&mut self, at: f64) {
        if self.open_call.take().is_some() {
            self.useful(at);
        }
    }

    fn chat(&mut self, v: &Value, at: f64) -> Result<(), String> {
        if let Some(error) = v.get("error") {
            return Err(error_text(error));
        }
        let usage = &v["usage"];
        if usage.is_object() {
            self.usage(
                usage["prompt_tokens"].as_u64(),
                usage["prompt_tokens_details"]["cached_tokens"].as_u64(),
                usage["completion_tokens"].as_u64(),
            );
        }
        let Some(choice) = v["choices"].get(0) else {
            return Ok(());
        };
        let delta = &choice["delta"];
        let non_empty = |key: &str| delta[key].as_str().is_some_and(|s| !s.is_empty());
        let calls = delta["tool_calls"].as_array().filter(|c| !c.is_empty());
        if non_empty("content") || non_empty("reasoning_content") || non_empty("reasoning") {
            self.content(at);
        }
        if non_empty("content") {
            self.close_chat_call(at);
            self.useful(at);
        }
        if let Some(calls) = calls {
            self.content(at);
            for call in calls {
                let index = call["index"].as_u64().unwrap_or(0);
                if self.open_call.is_some_and(|open| open != index) {
                    self.close_chat_call(at);
                }
                self.open_call = Some(index);
                let entry = self.calls.entry(index).or_default();
                if let Some(name) = call["function"]["name"].as_str() {
                    entry.0.push_str(name);
                }
                if let Some(args) = call["function"]["arguments"].as_str() {
                    entry.1.push_str(args);
                }
            }
        }
        if let Some(reason) = choice["finish_reason"].as_str() {
            self.close_chat_call(at);
            self.out.finish = if reason == "length" {
                Finish::Length
            } else {
                Finish::Stop
            };
        }
        Ok(())
    }

    fn messages(&mut self, name: &str, v: &Value, at: f64) -> Result<(), String> {
        match name {
            "error" => return Err(error_text(&v["error"])),
            "message_start" => {
                let u = &v["message"]["usage"];
                let cached = u["cache_read_input_tokens"].as_u64().unwrap_or(0);
                let prompt = u["input_tokens"].as_u64().map(|input| {
                    input + cached + u["cache_creation_input_tokens"].as_u64().unwrap_or(0)
                });
                self.usage(prompt, Some(cached), None);
            }
            "content_block_start" if v["content_block"]["type"] == "tool_use" => {
                let name = v["content_block"]["name"].as_str().unwrap_or_default();
                self.block_call = Some((name.into(), String::new()));
            }
            "content_block_delta" => {
                self.content(at);
                let delta = &v["delta"];
                match delta["type"].as_str() {
                    Some("text_delta") => self.useful(at),
                    Some("input_json_delta") => {
                        if let Some((_, args)) = &mut self.block_call {
                            args.push_str(delta["partial_json"].as_str().unwrap_or_default());
                        }
                    }
                    _ => {}
                }
            }
            "content_block_stop" => {
                if let Some(call) = self.block_call.take() {
                    self.out.tool_calls.push(call);
                    self.useful(at);
                }
            }
            "message_delta" => {
                self.usage(None, None, v["usage"]["output_tokens"].as_u64());
                if let Some(reason) = v["delta"]["stop_reason"].as_str() {
                    self.out.finish = if reason == "max_tokens" {
                        Finish::Length
                    } else {
                        Finish::Stop
                    };
                }
            }
            _ => {}
        }
        Ok(())
    }

    fn responses(&mut self, name: &str, v: &Value, at: f64) -> Result<(), String> {
        match name {
            "response.output_text.delta" => {
                self.content(at);
                self.useful(at);
            }
            "response.function_call_arguments.delta" => self.content(at),
            "response.output_item.done" if v["item"]["type"] == "function_call" => {
                let item = &v["item"];
                self.out.tool_calls.push((
                    item["name"].as_str().unwrap_or_default().into(),
                    item["arguments"].as_str().unwrap_or_default().into(),
                ));
                self.useful(at);
            }
            "response.completed" | "response.incomplete" => {
                let r = &v["response"];
                let u = &r["usage"];
                self.usage(
                    u["input_tokens"].as_u64(),
                    u["input_tokens_details"]["cached_tokens"].as_u64(),
                    u["output_tokens"].as_u64(),
                );
                self.out.finish = if r["incomplete_details"]["reason"] == "max_output_tokens" {
                    Finish::Length
                } else {
                    Finish::Stop
                };
            }
            "response.failed" | "error" => {
                return Err(error_text(
                    v["response"].get("error").unwrap_or(&v["error"]),
                ));
            }
            _ => {}
        }
        Ok(())
    }

    fn finish(mut self, at: f64) -> Outcome {
        self.close_chat_call(at);
        let calls = std::mem::take(&mut self.calls);
        self.out.tool_calls.extend(calls.into_values());
        self.out
    }
}

fn error_text(error: &Value) -> String {
    error["message"]
        .as_str()
        .map_or_else(|| error.to_string(), Into::into)
}

/// Tool calls that are malformed against the request's tools (FR-003a): arguments that are not
/// valid JSON, an undeclared tool name, or arguments that fail the tool's schema.
pub fn malformed_calls(api: ApiKind, request: &Value, calls: &[(String, String)]) -> u32 {
    let tools: Vec<(&str, &Value)> = request["tools"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|t| match api {
            ApiKind::Chat => Some((
                t["function"]["name"].as_str()?,
                &t["function"]["parameters"],
            )),
            ApiKind::Messages => Some((t["name"].as_str()?, &t["input_schema"])),
            ApiKind::Responses => Some((t["name"].as_str()?, &t["parameters"])),
        })
        .collect();
    calls
        .iter()
        .filter(|(name, args)| {
            let args = if args.trim().is_empty() { "{}" } else { args };
            let Ok(value) = serde_json::from_str::<Value>(args) else {
                return true;
            };
            match tools.iter().find(|(n, _)| n == name) {
                Some((_, schema)) => !schema_ok(schema, &value),
                None => true,
            }
        })
        .count() as u32
}

/// The JSON Schema keywords tool definitions use: `type`, `enum`, `required`, `properties`,
/// `additionalProperties: false` and `items`. Unknown keywords pass.
fn schema_ok(schema: &Value, value: &Value) -> bool {
    let types: Vec<&str> = match &schema["type"] {
        Value::String(t) => vec![t.as_str()],
        Value::Array(ts) => ts.iter().filter_map(Value::as_str).collect(),
        _ => Vec::new(),
    };
    let type_ok = types.is_empty()
        || types.iter().any(|t| match *t {
            "object" => value.is_object(),
            "array" => value.is_array(),
            "string" => value.is_string(),
            "number" => value.is_number(),
            "integer" => value.is_i64() || value.is_u64(),
            "boolean" => value.is_boolean(),
            "null" => value.is_null(),
            _ => true,
        });
    if !type_ok {
        return false;
    }
    if let Some(options) = schema["enum"].as_array()
        && !options.contains(value)
    {
        return false;
    }
    if let Some(object) = value.as_object() {
        let props = schema["properties"].as_object();
        let required = schema["required"].as_array().into_iter().flatten();
        if required
            .filter_map(Value::as_str)
            .any(|k| !object.contains_key(k))
        {
            return false;
        }
        for (key, v) in object {
            match props.and_then(|p| p.get(key)) {
                Some(sub) if !schema_ok(sub, v) => return false,
                None if schema["additionalProperties"] == false => return false,
                _ => {}
            }
        }
    }
    if let (Some(items), Some(array)) = (schema.get("items"), value.as_array()) {
        return array.iter().all(|v| schema_ok(items, v));
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    fn read(api: ApiKind, frames: &[&str]) -> Outcome {
        let mut reader = Reader::new(api);
        for (i, frame) in frames.iter().enumerate() {
            reader.block(frame, i as f64 * 10.0).unwrap();
        }
        reader.finish(frames.len() as f64 * 10.0)
    }

    #[test]
    fn chat_ttft_ignores_role_only_chunks_and_comments() {
        let out = read(
            ApiKind::Chat,
            &[
                ": keepalive\n\n",
                ": mrun {\"cached\":3,\"processed\":7,\"max_tokens\":5,\"cause\":\"changed\",\"message\":{\"index\":2,\"role\":\"user\"}}\n\n",
                "data: {\"choices\":[{\"delta\":{\"role\":\"assistant\",\"content\":\"\"}}]}\n\n",
                "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"hm\"}}]}\n\n",
                "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n",
                "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"length\"}]}\n\n",
                "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":2,\"prompt_tokens_details\":{\"cached_tokens\":3}}}\n\n",
                "data: [DONE]\n\n",
            ],
        );
        assert_eq!(out.ttft_ms, Some(30.0));
        assert_eq!(out.tfut_ms, Some(40.0));
        assert_eq!(
            (out.prompt_tokens, out.cached_tokens, out.output_tokens),
            (Some(10), Some(3), Some(2))
        );
        assert_eq!(out.cause.as_deref(), Some("changed"));
        assert_eq!(out.message_index, Some(2));
        assert_eq!(out.finish, Finish::Length);
        assert!(!out.early_eos(2));
    }

    #[test]
    fn messages_and_responses_tool_calls() {
        let out = read(
            ApiKind::Messages,
            &[
                "event: ping\ndata: {\"type\":\"ping\"}\n\n",
                "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":4,\"cache_read_input_tokens\":6}}}\n\n",
                "event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"tool_use\",\"name\":\"read\"}}\n\n",
                "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"p\\\":1}\"}}\n\n",
                "event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
                "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"tool_use\"},\"usage\":{\"output_tokens\":3}}\n\n",
            ],
        );
        assert_eq!((out.ttft_ms, out.tfut_ms), (Some(30.0), Some(40.0)));
        assert_eq!((out.prompt_tokens, out.cached_tokens), (Some(10), Some(6)));
        assert_eq!(out.tool_calls, vec![("read".into(), "{\"p\":1}".into())]);
        assert!(out.early_eos(8));

        let out = read(
            ApiKind::Responses,
            &[
                "event: response.created\ndata: {\"type\":\"response.created\"}\n\n",
                "event: response.reasoning_text.delta\ndata: {\"delta\":\"hm\"}\n\n",
                "event: response.function_call_arguments.delta\ndata: {\"delta\":\"{\"}\n\n",
                "event: response.output_item.done\ndata: {\"item\":{\"type\":\"function_call\",\"name\":\"x\",\"arguments\":\"{\"}}\n\n",
                "event: response.incomplete\ndata: {\"response\":{\"incomplete_details\":{\"reason\":\"max_output_tokens\"},\"usage\":{\"input_tokens\":9,\"input_tokens_details\":{\"cached_tokens\":1},\"output_tokens\":4}}}\n\n",
            ],
        );
        assert_eq!((out.ttft_ms, out.tfut_ms), (Some(20.0), Some(30.0)));
        assert_eq!(out.finish, Finish::Length);
        assert_eq!(out.prompt_tokens, Some(9));
    }

    #[test]
    fn malformed_tool_calls_are_counted() {
        let request = json!({"tools": [{"type": "function", "function": {"name": "read",
            "parameters": {"type": "object", "properties": {"path": {"type": "string"}},
                "required": ["path"], "additionalProperties": false}}}]});
        let calls = |c: &[(&str, &str)]| -> Vec<(String, String)> {
            c.iter()
                .map(|(n, a)| (n.to_string(), a.to_string()))
                .collect()
        };
        let count = |c: &[(&str, &str)]| malformed_calls(ApiKind::Chat, &request, &calls(c));
        assert_eq!(count(&[("read", r#"{"path": "a"}"#)]), 0);
        assert_eq!(count(&[("read", r#"{"path": "a""#)]), 1, "invalid JSON");
        assert_eq!(count(&[("write", r#"{"path": "a"}"#)]), 1, "undeclared");
        assert_eq!(count(&[("read", r#"{"path": 3}"#)]), 1, "wrong type");
        assert_eq!(count(&[("read", r#"{}"#)]), 1, "missing required");
        assert_eq!(count(&[("read", r#"{"path": "a", "x": 1}"#)]), 1, "extra");
    }

    #[test]
    fn truncation_is_a_shortfall_over_two_percent() {
        assert!(truncated(Some(97), 100));
        assert!(!truncated(Some(98), 100));
        assert!(!truncated(Some(150), 100));
        assert!(!truncated(None, 100));
        assert!(!truncated(Some(1), 0));
    }

    #[test]
    fn prepare_sets_stream_limit_model_and_ignore_eos() {
        let body = json!({"model": "a", "messages": [], "max_completion_tokens": 9});
        let chat = prepare(ApiKind::Chat, &body, Some("b"), 64, IgnoreEos::Body);
        assert_eq!(chat["model"], "b");
        assert_eq!(chat["max_tokens"], 64);
        assert_eq!(chat["max_completion_tokens"], 64);
        assert_eq!(chat["stream_options"]["include_usage"], true);
        assert_eq!(chat["ignore_eos"], true);
        let resp = prepare(ApiKind::Responses, &body, None, 8, IgnoreEos::MrunHeader);
        assert_eq!(resp["model"], "a");
        assert_eq!(resp["max_output_tokens"], 8);
        assert!(resp.get("ignore_eos").is_none());
    }
}
