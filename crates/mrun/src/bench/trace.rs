//! Agent traces (005 data model): one JSON line per recorded request in arrival order, and the
//! maintainer step that builds them from `mrun serve --record` output (`mrun bench trace`):
//! order, group into conversations, derive reply lengths, redact with one mapping per trace and
//! refuse to write anything that looks like a secret.

use crate::api::model::ApiKind;
use crate::chat::tokenizer::Tokenizer;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use std::collections::HashMap;
use std::path::Path;

/// One replayed request.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TraceRecord {
    /// `chat`, `messages` or `responses`.
    pub api: String,
    pub conversation: String,
    /// The request exactly as the client sent it (redacted).
    pub body: Value,
    /// Expected prompt tokens for the truncation check; 0 when unknown.
    #[serde(default)]
    pub prompt_tokens: u32,
    /// Reply length, sent as the output limit.
    pub reply_tokens: u32,
    /// `x-claude-code-prev-tool-durations` when recorded; kept, never replayed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_gaps: Option<String>,
}

pub fn parse_api(name: &str) -> Option<ApiKind> {
    match name {
        "chat" => Some(ApiKind::Chat),
        "messages" => Some(ApiKind::Messages),
        "responses" => Some(ApiKind::Responses),
        _ => None,
    }
}

pub fn load(path: &Path) -> Result<Vec<TraceRecord>, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    text.lines()
        .enumerate()
        .filter(|(_, line)| !line.trim().is_empty())
        .map(|(i, line)| {
            let record: TraceRecord = serde_json::from_str(line)
                .map_err(|e| format!("{}:{}: {e}", path.display(), i + 1))?;
            parse_api(&record.api).ok_or_else(|| {
                format!("{}:{}: unknown api '{}'", path.display(), i + 1, record.api)
            })?;
            Ok(record)
        })
        .collect()
}

/// A request's messages as API content (005 FR-017): role, visible text and tool calls as JSON,
/// reasoning excluded. The leading system block is message 0 when present, as in mrun's miss
/// reports.
pub fn api_messages(api: ApiKind, body: &Value) -> Vec<Value> {
    let mut out = Vec::new();
    match api {
        ApiKind::Chat => {
            for m in body["messages"].as_array().into_iter().flatten() {
                let calls: Vec<Value> = m["tool_calls"]
                    .as_array()
                    .into_iter()
                    .flatten()
                    .map(|c| call(&c["function"]["name"], &c["function"]["arguments"]))
                    .collect();
                out.push(message(&m["role"], text(&m["content"]), calls));
            }
        }
        ApiKind::Messages => {
            let system = text(&body["system"]);
            if !system.is_empty() {
                out.push(message(&"system".into(), system, Vec::new()));
            }
            for m in body["messages"].as_array().into_iter().flatten() {
                let calls: Vec<Value> = blocks(&m["content"])
                    .filter(|b| b["type"] == "tool_use")
                    .map(|b| call(&b["name"], &b["input"]))
                    .collect();
                out.push(message(&m["role"], text(&m["content"]), calls));
            }
        }
        ApiKind::Responses => {
            let instructions = text(&body["instructions"]);
            if !instructions.is_empty() {
                out.push(message(&"system".into(), instructions, Vec::new()));
            }
            match &body["input"] {
                Value::String(s) => out.push(message(&"user".into(), s.clone(), Vec::new())),
                Value::Array(items) => {
                    for item in items {
                        out.push(match item["type"].as_str() {
                            Some("function_call") => message(
                                &"assistant".into(),
                                String::new(),
                                vec![call(&item["name"], &item["arguments"])],
                            ),
                            Some("function_call_output") => {
                                message(&"tool".into(), text(&item["output"]), Vec::new())
                            }
                            Some("reasoning") => {
                                message(&"reasoning".into(), String::new(), vec![])
                            }
                            _ => message(&item["role"], text(&item["content"]), Vec::new()),
                        });
                    }
                }
                _ => {}
            }
        }
    }
    out
}

fn message(role: &Value, text: String, calls: Vec<Value>) -> Value {
    json!({"role": role, "text": text, "tool_calls": calls})
}

/// A tool call with its arguments parsed when they are a JSON string.
fn call(name: &Value, arguments: &Value) -> Value {
    let arguments = match arguments {
        Value::String(s) => serde_json::from_str(s).unwrap_or_else(|_| s.clone().into()),
        other => other.clone(),
    };
    json!({"name": name, "arguments": arguments})
}

fn blocks(content: &Value) -> impl Iterator<Item = &Value> {
    content.as_array().into_iter().flatten()
}

/// Visible text of a string or a list of content blocks (thinking and reasoning skipped).
fn text(content: &Value) -> String {
    match content {
        Value::String(s) => s.clone(),
        Value::Array(parts) => parts
            .iter()
            .filter_map(|p| match p["type"].as_str() {
                Some("text" | "input_text" | "output_text") => p["text"].as_str().map(Into::into),
                Some("tool_result") => Some(text(&p["content"])),
                _ => None,
            })
            .collect::<Vec<String>>()
            .join("\n"),
        _ => String::new(),
    }
}

/// Builds a trace from `mrun serve --record` lines. `tokenizer` counts expected prompt and
/// reply tokens over the requests' visible text; without it prompt counts stay 0 (no truncation
/// check) and reply lengths are estimated at 4 bytes per token.
pub fn build(record: &str, tokenizer: Option<&Tokenizer>) -> Result<Vec<TraceRecord>, String> {
    let mut lines: Vec<Value> = record
        .lines()
        .filter(|l| !l.trim().is_empty())
        .enumerate()
        .map(|(i, l)| serde_json::from_str(l).map_err(|e| format!("line {}: {e}", i + 1)))
        .collect::<Result<_, _>>()?;
    // RFC 3339 UTC with fixed width sorts as text; the timestamps are used for nothing else.
    lines.sort_by(|a, b| a["arrived"].as_str().cmp(&b["arrived"].as_str()));

    let count = |s: &str| match tokenizer {
        Some(t) => t.encode(s).map_or(0, |ids| ids.len() as u32),
        None => s.len().div_ceil(4) as u32,
    };
    let mut records = Vec::new();
    // Per conversation: its id and the messages of its latest request.
    let mut open: Vec<(String, Vec<Value>)> = Vec::new();
    let mut next_id = 0;
    for line in &lines {
        let api = line["api"]
            .as_str()
            .and_then(parse_api)
            .ok_or("a record has no known api")?;
        let body = line["body"].clone();
        let messages = api_messages(api, &body);
        let hint = body["prompt_cache_key"]
            .as_str()
            .or_else(|| {
                crate::api::record::SESSION_HEADERS[..3]
                    .iter()
                    .find_map(|h| line["session"][h].as_str())
            })
            .map(str::to_string);
        let conversation = match hint {
            Some(hint) => hint,
            None => {
                // Longest shared message prefix (system + first user message at least).
                let best = open
                    .iter()
                    .map(|(id, prev)| (id, shared(prev, &messages)))
                    .filter(|(_, n)| *n >= 2)
                    .max_by_key(|(_, n)| *n)
                    .map(|(id, _)| id.clone());
                best.unwrap_or_else(|| {
                    next_id += 1;
                    format!("c{next_id}")
                })
            }
        };
        match open.iter_mut().find(|(id, _)| *id == conversation) {
            Some(entry) => entry.1 = messages.clone(),
            None => open.push((conversation.clone(), messages.clone())),
        }
        let visible: String = messages
            .iter()
            .map(|m| format!("{}\n{}\n{}", m["role"], m["text"], m["tool_calls"]))
            .chain(
                body["tools"]
                    .as_array()
                    .map(|t| Value::from(t.clone()).to_string()),
            )
            .collect();
        records.push(TraceRecord {
            api: api.as_str().into(),
            conversation,
            prompt_tokens: if tokenizer.is_some() {
                count(&visible)
            } else {
                0
            },
            reply_tokens: 0,
            tool_gaps: line["session"]["x-claude-code-prev-tool-durations"]
                .as_str()
                .map(Into::into),
            body,
        });
    }
    fill_reply_tokens(&mut records, count);
    redact(&mut records);
    let found = secrets(&records);
    if !found.is_empty() {
        return Err(format!(
            "possible secrets, nothing written: {}",
            found.join(", ")
        ));
    }
    Ok(records)
}

fn shared(a: &[Value], b: &[Value]) -> usize {
    a.iter().zip(b).take_while(|(x, y)| x == y).count()
}

/// Reply length of each request from the next request of its conversation (the assistant
/// messages appended after its history); the conversation's median for its last request.
fn fill_reply_tokens(records: &mut [TraceRecord], count: impl Fn(&str) -> u32) {
    const FALLBACK: u32 = 256;
    let mut by_conversation: HashMap<String, Vec<usize>> = HashMap::new();
    for (i, r) in records.iter().enumerate() {
        by_conversation
            .entry(r.conversation.clone())
            .or_default()
            .push(i);
    }
    for indices in by_conversation.values() {
        let mut lengths = Vec::new();
        for pair in indices.windows(2) {
            let api = parse_api(&records[pair[0]].api).unwrap_or(ApiKind::Chat);
            let now = api_messages(api, &records[pair[0]].body);
            let next = api_messages(api, &records[pair[1]].body);
            let reply: String = next
                .iter()
                .skip(now.len())
                .take_while(|m| m["role"] == "assistant" || m["role"] == "reasoning")
                .map(|m| {
                    let calls = m["tool_calls"].as_array().filter(|c| !c.is_empty());
                    let calls =
                        calls.map_or_else(String::new, |c| Value::from(c.clone()).to_string());
                    format!("{}{calls}", m["text"].as_str().unwrap_or(""))
                })
                .collect();
            let n = count(&reply).max(1);
            records[pair[0]].reply_tokens = n;
            lengths.push(n);
        }
        lengths.sort_unstable();
        let last = *indices.last().unwrap_or(&0);
        records[last].reply_tokens = lengths.get(lengths.len() / 2).copied().unwrap_or(FALLBACK);
    }
}

/// Replaces home-directory users, email addresses and private host names with stable
/// placeholders, one mapping for the whole trace so shared prefixes stay byte-identical.
fn redact(records: &mut [TraceRecord]) {
    let mut mapping: HashMap<String, String> = HashMap::new();
    for record in records {
        redact_value(&mut record.body, &mut mapping);
    }
}

fn redact_value(value: &mut Value, mapping: &mut HashMap<String, String>) {
    match value {
        Value::String(s) => *s = redact_text(s, mapping),
        Value::Array(items) => items.iter_mut().for_each(|v| redact_value(v, mapping)),
        Value::Object(map) => {
            let keys: Vec<String> = map.keys().cloned().collect();
            let mut out = Map::new();
            for key in keys {
                let mut v = map.remove(&key).unwrap_or(Value::Null);
                redact_value(&mut v, mapping);
                out.insert(key, v);
            }
            *map = out;
        }
        _ => {}
    }
}

const PRIVATE_SUFFIXES: [&str; 5] = [".local", ".lan", ".internal", ".home", ".corp"];

fn word_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-' | '+' | '@')
}

pub fn redact_text(text: &str, mapping: &mut HashMap<String, String>) -> String {
    let mut placeholder = |kind: &str, original: &str| {
        let n = mapping.len() + 1;
        mapping
            .entry(format!("{kind}:{original}"))
            .or_insert_with(|| match kind {
                "email" => format!("user{n}@example.com"),
                "host" => format!("host{n}.example"),
                _ => format!("user{n}"),
            })
            .clone()
    };
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while !rest.is_empty() {
        let token_len = rest.find(|c: char| !word_char(c)).unwrap_or(rest.len());
        if token_len == 0 {
            let c = rest.chars().next().unwrap_or(' ');
            out.push(c);
            rest = &rest[c.len_utf8()..];
            // `/Users/<name>` and `/home/<name>`.
            for prefix in ["Users/", "home/"] {
                if c == '/' && rest.starts_with(prefix) {
                    let after = &rest[prefix.len()..];
                    let name_len = after.find(|c: char| !word_char(c)).unwrap_or(after.len());
                    if name_len > 0 {
                        out.push_str(prefix);
                        out.push_str(&placeholder("user", &after[..name_len]));
                        rest = &after[name_len..];
                    }
                }
            }
            continue;
        }
        // A sentence's full stop is not part of a host name.
        let token = rest[..token_len].trim_end_matches('.');
        let token_len = token.len().max(1);
        let token = &rest[..token_len];
        if let Some(at) = token.find('@')
            && at > 0
            && token[at + 1..].contains('.')
        {
            out.push_str(&placeholder("email", token));
        } else if PRIVATE_SUFFIXES
            .iter()
            .any(|s| token.len() > s.len() && token.to_ascii_lowercase().ends_with(s))
        {
            out.push_str(&placeholder("host", token));
        } else {
            out.push_str(token);
        }
        rest = &rest[token_len..];
    }
    out
}

/// Token prefixes of common credentials, with the minimum length of what follows.
const SECRET_PREFIXES: [(&str, usize); 9] = [
    ("sk-", 20),
    ("ghp_", 30),
    ("github_pat_", 30),
    ("gho_", 30),
    ("xoxb-", 20),
    ("xoxp-", 20),
    ("hf_", 30),
    ("AKIA", 16),
    ("AIza", 30),
];

/// Names of the patterns found in the trace (empty when clean).
pub fn secrets(records: &[TraceRecord]) -> Vec<String> {
    let mut found = Vec::new();
    for (i, record) in records.iter().enumerate() {
        let text = record.body.to_string();
        if text.contains("-----BEGIN ") && text.contains("PRIVATE KEY") {
            found.push(format!("request {}: private key", i + 1));
        }
        for token in text.split(|c: char| !(c.is_ascii_alphanumeric() || c == '_' || c == '-')) {
            for (prefix, min) in SECRET_PREFIXES {
                if token.starts_with(prefix) && token.len() >= prefix.len() + min {
                    found.push(format!("request {}: {prefix}…", i + 1));
                }
            }
        }
    }
    found.dedup();
    found
}

#[cfg(test)]
mod tests {
    use super::*;

    fn line(arrived: &str, session: Value, body: Value) -> String {
        json!({"api": "chat", "arrived": arrived, "session": session, "body": body}).to_string()
    }

    fn chat(messages: Value) -> Value {
        json!({"model": "m", "messages": messages})
    }

    #[test]
    fn builds_ordered_grouped_records_with_reply_lengths() {
        let sys = json!({"role": "system", "content": "you are an agent"});
        let u1 = json!({"role": "user", "content": "fix the bug"});
        let a1 = json!({"role": "assistant", "content": "", "tool_calls": [{"id": "c1",
            "type": "function", "function": {"name": "read", "arguments": "{\"path\":\"a.rs\"}"}}]});
        let t1 = json!({"role": "tool", "tool_call_id": "c1", "content": "fn main() {}"});
        let other = json!({"role": "user", "content": "unrelated"});
        let record = [
            // Out of order on purpose: arrival time decides.
            line(
                "2026-10-15T08:00:02.000Z",
                json!({}),
                chat(json!([sys, u1, a1, t1])),
            ),
            line(
                "2026-10-15T08:00:01.000Z",
                json!({}),
                chat(json!([sys, u1])),
            ),
            line(
                "2026-10-15T08:00:03.000Z",
                json!({"x-claude-code-prev-tool-durations": "1200"}),
                chat(json!([sys, other])),
            ),
        ]
        .join("\n");
        let trace = build(&record, None).unwrap();
        assert_eq!(trace.len(), 3);
        assert_eq!(trace[0].body["messages"].as_array().unwrap().len(), 2);
        assert_eq!(trace[0].conversation, trace[1].conversation);
        assert_ne!(trace[2].conversation, trace[0].conversation);
        // The reply is the tool call appended by the next request (bytes / 4 without a tokenizer).
        assert!(trace[0].reply_tokens > 1);
        // Last request of a conversation: the conversation's median.
        assert_eq!(trace[1].reply_tokens, trace[0].reply_tokens);
        assert_eq!(trace[2].reply_tokens, 256);
        assert_eq!(trace[2].tool_gaps.as_deref(), Some("1200"));
        assert_eq!(trace[0].prompt_tokens, 0);
    }

    #[test]
    fn session_hints_group_before_prefixes() {
        let body = |key: &str| {
            json!({"model": "m", "prompt_cache_key": key,
            "messages": [{"role": "user", "content": "same"}]})
        };
        let record = [
            line("2026-10-15T08:00:01.000Z", json!({}), body("a")),
            line("2026-10-15T08:00:02.000Z", json!({}), body("b")),
        ]
        .join("\n");
        let trace = build(&record, None).unwrap();
        assert_eq!(trace[0].conversation, "a");
        assert_eq!(trace[1].conversation, "b");
    }

    #[test]
    fn redaction_is_one_deterministic_mapping() {
        let mut mapping = HashMap::new();
        let a = redact_text(
            "open /Users/alice/src/x.rs and mail alice@corp.com on box.local",
            &mut mapping,
        );
        let b = redact_text("cd /Users/alice; ssh box.local", &mut mapping);
        assert_eq!(
            a,
            "open /Users/user1/src/x.rs and mail user2@example.com on host3.example"
        );
        assert_eq!(b, "cd /Users/user1; ssh host3.example");
        assert_eq!(redact_text("/home/bob/x", &mut mapping), "/home/user4/x");
    }

    #[test]
    fn secrets_block_the_trace() {
        let record = line(
            "2026-10-15T08:00:01.000Z",
            json!({}),
            chat(
                json!([{"role": "user", "content": "token ghp_0123456789abcdefghijklmnopqrstuvwxyz"}]),
            ),
        );
        let error = build(&record, None).unwrap_err();
        assert!(error.contains("ghp_"), "{error}");
    }

    #[test]
    fn api_content_excludes_reasoning_and_counts_system_first() {
        let body = json!({"system": "sys", "messages": [
            {"role": "assistant", "content": [{"type": "thinking", "thinking": "hm"},
                {"type": "text", "text": "hi"},
                {"type": "tool_use", "id": "t", "name": "read", "input": {"p": 1}}]}]});
        let m = api_messages(ApiKind::Messages, &body);
        assert_eq!(m.len(), 2);
        assert_eq!(m[1]["text"], "hi");
        assert_eq!(m[1]["tool_calls"][0]["arguments"]["p"], 1);
        let r = api_messages(
            ApiKind::Responses,
            &json!({"input": [{"type": "function_call", "name": "x", "arguments": "{\"a\":2}"}]}),
        );
        assert_eq!(r[0]["tool_calls"][0]["arguments"]["a"], 2);
    }
}
