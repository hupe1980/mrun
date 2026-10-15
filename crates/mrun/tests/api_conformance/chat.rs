//! `chat_*` cases.

use crate::common::{self, Setup};
use serde_json::{Value, json};

fn user(text: &str) -> Value {
    json!({"model": "any", "messages": [{"role": "user", "content": text}]})
}

#[tokio::test]
async fn chat_basic_nonstream() {
    let server = common::start(&["Hello there."]).await;
    let reply = server.post("/v1/chat/completions", &user("hi")).await;
    assert_eq!(reply.status, 200, "{}", reply.text);
    let body = reply.json();
    assert_eq!(body["object"], "chat.completion");
    assert_eq!(body["model"], "scripted-model", "FR-018: the real model id");
    assert_eq!(body["choices"][0]["message"]["content"], "Hello there.");
    assert_eq!(body["choices"][0]["finish_reason"], "stop");
    assert_eq!(body["usage"]["completion_tokens"], 12, "EOS is not output");
    assert!(body["usage"]["prompt_tokens_details"]["cached_tokens"].is_number());
    assert!(reply.header("x-mrun-max-tokens").is_some());
}

#[tokio::test]
async fn chat_basic_stream_done_terminator() {
    let server = common::start(&["Hi!"]).await;
    let mut body = user("hi");
    body["stream"] = json!(true);
    let reply = server.post("/v1/chat/completions", &body).await;
    assert_eq!(reply.header("content-type"), Some("text/event-stream"));
    let frames = reply.frames();
    assert_eq!(frames.last().unwrap().1, "[DONE]");
    assert!(frames[0].1.starts_with("mrun "), "metadata comment first");
    let events = reply.events();
    // chat_stream_role_first_chunk
    assert_eq!(events[0].1["choices"][0]["delta"]["role"], "assistant");
    let text: String = events
        .iter()
        .filter_map(|(_, e)| e["choices"][0]["delta"]["content"].as_str())
        .collect();
    assert_eq!(text, "Hi!");
    // chat_no_delta_after_finish_reason
    let finish = events
        .iter()
        .position(|(_, e)| !e["choices"][0]["finish_reason"].is_null())
        .unwrap();
    assert_eq!(finish, events.len() - 1);
}

#[tokio::test]
async fn chat_tool_call_roundtrip_stream() {
    let call =
        "<tool_call>\n{\"name\": \"read\", \"arguments\": {\"path\": \"a.rs\"}}\n</tool_call>";
    let server = common::start(&[call, "Done."]).await;
    let tools = json!([{"type": "function", "function": {"name": "read",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}}}]);
    let body = json!({"messages": [{"role": "user", "content": "open a.rs"}],
        "tools": tools, "stream": true, "stream_options": {"include_usage": true}});
    let reply = server.post("/v1/chat/completions", &body).await;
    let events = reply.events();
    let calls: Vec<&Value> = events
        .iter()
        .filter_map(|(_, e)| e["choices"][0]["delta"]["tool_calls"].get(0))
        .collect();
    let id = calls[0]["id"].as_str().unwrap().to_string();
    assert!(id.starts_with("call_"));
    assert_eq!(calls[0]["function"]["name"], "read");
    let args: String = calls
        .iter()
        .filter_map(|c| c["function"]["arguments"].as_str())
        .collect();
    assert_eq!(
        serde_json::from_str::<Value>(&args).unwrap(),
        json!({"path": "a.rs"})
    );
    let finish = events
        .iter()
        .find_map(|(_, e)| e["choices"][0]["finish_reason"].as_str())
        .unwrap();
    assert_eq!(finish, "tool_calls");
    // chat_include_usage_chunk_empty_choices
    let usage = &events.last().unwrap().1;
    assert_eq!(usage["choices"], json!([]));
    assert!(usage["usage"]["prompt_tokens"].as_u64().unwrap() > 0);

    // chat_tool_result_roundtrip
    let next = json!({"messages": [
        {"role": "user", "content": "open a.rs"},
        {"role": "assistant", "content": null, "tool_calls": [{"id": id, "type": "function",
            "function": {"name": "read", "arguments": args}}]},
        {"role": "tool", "tool_call_id": id, "content": "fn main() {}"}],
        "tools": tools});
    let reply = server.post("/v1/chat/completions", &next).await;
    assert_eq!(reply.json()["choices"][0]["message"]["content"], "Done.");
}

#[tokio::test]
async fn chat_reasoning_content_separate_from_content() {
    let server = common::start(&["Let me think.\n</think>\n\nAnswer."]).await;
    let mut body = user("why?");
    body["reasoning_effort"] = json!("high");
    let reply = server.post("/v1/chat/completions", &body).await;
    let message = &reply.json()["choices"][0]["message"];
    assert_eq!(message["reasoning_content"], "Let me think.");
    assert_eq!(message["content"], "Answer.");
}

#[tokio::test]
async fn chat_tool_result_unknown_id_400() {
    let server = common::start(&[]).await;
    let body = json!({"messages": [{"role": "user", "content": "x"},
        {"role": "tool", "tool_call_id": "nope", "content": "y"}]});
    let reply = server.post("/v1/chat/completions", &body).await;
    assert_eq!(reply.status, 400);
    assert_eq!(reply.json()["error"]["type"], "invalid_request_error");
    assert_eq!(reply.json()["error"]["param"], "messages[1]");
}

#[tokio::test]
async fn chat_context_length_exceeded_400_code() {
    let server = common::start_with(
        &["x"],
        Setup {
            capacity: 32,
            ..Setup::default()
        },
    )
    .await;
    let reply = server
        .post("/v1/chat/completions", &user(&"long ".repeat(20)))
        .await;
    assert_eq!(reply.status, 400);
    assert_eq!(reply.json()["error"]["code"], "context_length_exceeded");
}

#[tokio::test]
async fn chat_max_tokens_length_and_clamp() {
    let server = common::start(&["one two three four five"]).await;
    let mut body = user("count");
    body["max_tokens"] = json!(3);
    body["max_completion_tokens"] = json!(5);
    let reply = server.post("/v1/chat/completions", &body).await;
    let body = reply.json();
    assert_eq!(body["choices"][0]["finish_reason"], "length");
    // chat_max_completion_tokens_precedence
    assert_eq!(body["usage"]["completion_tokens"], 5);
    assert_eq!(reply.header("x-mrun-max-tokens"), Some("5"));
}

#[tokio::test]
async fn chat_stop_sequence_holdback() {
    let server = common::start(&["alpha END beta"]).await;
    let mut body = user("go");
    body["stop"] = json!(["END"]);
    body["stream"] = json!(true);
    let reply = server.post("/v1/chat/completions", &body).await;
    let text: String = reply
        .events()
        .iter()
        .filter_map(|(_, e)| e["choices"][0]["delta"]["content"].as_str())
        .collect();
    assert_eq!(text, "alpha ");
}

#[tokio::test]
async fn chat_unknown_field_ignored_and_bad_json_400() {
    let server = common::start(&["ok"]).await;
    let mut body = user("hi");
    body["prompt_cache_key"] = json!("k");
    body["metadata"] = json!({"a": 1});
    assert_eq!(server.post("/v1/chat/completions", &body).await.status, 200);
    let reply = server
        .client
        .post(server.url("/v1/chat/completions"))
        .body("{not json")
        .send()
        .await
        .unwrap();
    assert_eq!(reply.status().as_u16(), 400);
}

#[tokio::test]
async fn xapi_body_over_64mib_413() {
    let server = common::start(&[]).await;
    let reply = server
        .client
        .post(server.url("/v1/chat/completions"))
        .header("content-type", "application/json")
        .body(vec![b' '; 64 * 1024 * 1024 + 1])
        .send()
        .await
        .unwrap();
    assert_eq!(reply.status().as_u16(), 413);
}
