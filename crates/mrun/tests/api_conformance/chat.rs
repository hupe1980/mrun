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

fn read_tool(strict: bool) -> Value {
    json!([{"type": "function", "function": {"name": "read", "strict": strict,
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}},
            "required": ["path"], "additionalProperties": false}}}])
}

#[tokio::test]
async fn chat_tool_choice_required_forces_call() {
    // Only the argument value and the choices after it are sampled; the opener, the envelope,
    // the key and the closer are forced and never come from the model (xapi_forced_tokens_not_sampled).
    let body = json!({"messages": [{"role": "user", "content": "go"}],
        "tools": read_tool(false), "tool_choice": "required"});
    let lark = common::plan(&body).answer.unwrap();
    let full =
        "<tool_call>\n{\"name\": \"read\", \"arguments\": {\"path\":\"a.rs\"}}\n</tool_call>";
    let script = common::sampled(&lark, "", full);
    // The tool is not strict, so only the envelope is forced; its arguments are free JSON.
    assert!(
        script.len() * 3 < full.len(),
        "most of the call is forced: {} sampled",
        script.len()
    );
    let server = common::start_tokens(vec![script], Setup::default()).await;
    let reply = server.post("/v1/chat/completions", &body).await;
    let message = &reply.json()["choices"][0]["message"];
    assert_eq!(
        message["tool_calls"][0]["function"]["name"], "read",
        "{}",
        reply.text
    );
    let args: Value = serde_json::from_str(
        message["tool_calls"][0]["function"]["arguments"]
            .as_str()
            .unwrap(),
    )
    .unwrap();
    assert_eq!(args, json!({"path": "a.rs"}));
    assert_eq!(reply.json()["choices"][0]["finish_reason"], "tool_calls");
}

#[tokio::test]
async fn chat_strict_tool_args_validate_schema() {
    let body = json!({"messages": [{"role": "user", "content": "go"}], "tools": read_tool(true)});
    let lark = common::plan(&body).on_opener.unwrap();
    let good = common::sampled(
        &lark,
        "<tool_call>",
        "\n{\"name\": \"read\", \"arguments\": {\"path\":\"a.rs\"}}\n</tool_call>",
    );
    // A model trying to write a number where the schema wants a string.
    let tokenizer = mrun::chat::tokenizer::test_support::byte_tokenizer();
    let mut bad = common::sampled(
        &lark,
        "<tool_call>",
        "\n{\"name\": \"read\", \"arguments\": {\"path\":\"",
    );
    bad.pop();
    bad.extend(tokenizer.encode("1}").unwrap());
    let server = common::start_tokens(vec![good, bad], Setup::default()).await;
    let reply = server.post("/v1/chat/completions", &body).await;
    assert_eq!(
        reply.json()["choices"][0]["finish_reason"],
        "tool_calls",
        "{}",
        reply.text
    );
    // The masked token cannot be produced: the scripted engine refuses it, which surfaces as
    // an engine error (a real model samples an allowed token instead).
    let reply = server.post("/v1/chat/completions", &body).await;
    assert_ne!(reply.status, 200, "{}", reply.text);
}

#[tokio::test]
async fn chat_tool_choice_none_no_calls() {
    let server = common::start(&["plain answer", "<tool_call>\n{}"]).await;
    let body = json!({"messages": [{"role": "user", "content": "go"}],
        "tools": read_tool(false), "tool_choice": "none"});
    let reply = server.post("/v1/chat/completions", &body).await;
    assert_eq!(
        reply.json()["choices"][0]["message"]["content"],
        "plain answer"
    );
    // The opener is banned for the whole reply (xapi_tool_choice_none_bias_pipelined).
    assert_ne!(server.post("/v1/chat/completions", &body).await.status, 200);
}

#[tokio::test]
async fn chat_response_format_json_object() {
    let mut body = user("json please");
    body["response_format"] = json!({"type": "json_object"});
    let lark = common::plan(&body).answer.unwrap();
    let script = common::sampled(&lark, "", "{\"a\":1}");
    let tokenizer = mrun::chat::tokenizer::test_support::byte_tokenizer();
    let server = common::start_tokens(
        vec![script, tokenizer.encode("hello").unwrap()],
        Setup::default(),
    )
    .await;
    let reply = server.post("/v1/chat/completions", &body).await;
    let content = reply.json()["choices"][0]["message"]["content"]
        .as_str()
        .unwrap()
        .to_string();
    assert_eq!(
        serde_json::from_str::<Value>(&content).unwrap(),
        json!({"a": 1})
    );
    assert_ne!(server.post("/v1/chat/completions", &body).await.status, 200);
}

#[tokio::test]
async fn xapi_max_tokens_clamped_reported() {
    let server = common::start_with(
        &["x"],
        Setup {
            capacity: 200,
            ..Setup::default()
        },
    )
    .await;
    let mut body = user("hi");
    body["max_tokens"] = json!(100_000);
    let reply = server.post("/v1/chat/completions", &body).await;
    assert_eq!(reply.status, 200, "{}", reply.text);
    let applied: u32 = reply.header("x-mrun-max-tokens").unwrap().parse().unwrap();
    let prompt = reply.json()["usage"]["prompt_tokens"].as_u64().unwrap() as u32;
    assert_eq!(applied, 200 - prompt);
    assert_eq!(reply.header("x-mrun-warning"), Some("max-tokens-clamped"));
    // Streaming: the first `: mrun` comment carries the limit, the last one the warning.
    body["stream"] = json!(true);
    let server = common::start_with(
        &["x"],
        Setup {
            capacity: 200,
            ..Setup::default()
        },
    )
    .await;
    let reply = server.post("/v1/chat/completions", &body).await;
    let comments: Vec<String> = reply
        .frames()
        .into_iter()
        .filter(|(name, _)| name == ":")
        .map(|(_, text)| text)
        .collect();
    assert!(
        comments[0].contains(&format!("\"max_tokens\":{applied}")),
        "{comments:?}"
    );
    assert!(
        comments.last().unwrap().contains("max-tokens-clamped"),
        "{comments:?}"
    );
}

#[tokio::test]
async fn chat_nonstrict_invalid_args_returned_raw_with_warning() {
    let call = "<tool_call>\n{\"name\": \"read\", \"arguments\": {\"path\": oops}}\n</tool_call>";
    let server = common::start(&[call]).await;
    let body = json!({"messages": [{"role": "user", "content": "go"}], "tools": read_tool(false)});
    let reply = server.post("/v1/chat/completions", &body).await;
    let message = &reply.json()["choices"][0]["message"];
    assert_eq!(
        message["tool_calls"][0]["function"]["arguments"],
        "{\"path\": oops}"
    );
    assert_eq!(reply.header("x-mrun-warning"), Some("tool-args-invalid"));
}

#[tokio::test]
async fn xapi_cached_tokens_resume_p_minus_1() {
    // The same prompt again: everything but the last prompt token is reused.
    let server = common::start(&["a", "b"]).await;
    let first = server
        .post("/v1/chat/completions", &user("same prompt"))
        .await
        .json();
    let second = server
        .post("/v1/chat/completions", &user("same prompt"))
        .await
        .json();
    let prompt = second["usage"]["prompt_tokens"].as_u64().unwrap();
    assert_eq!(first["usage"]["prompt_tokens"], prompt);
    assert_eq!(
        second["usage"]["prompt_tokens_details"]["cached_tokens"],
        prompt - 1
    );
}

#[tokio::test]
async fn xapi_five_screenshots_in_history_fit_the_body_limit() {
    // Five ~6 MB base64 screenshots in tool results: accepted and replaced by placeholders.
    let server = common::start(&["seen"]).await;
    let image = "A".repeat(6 * 1024 * 1024);
    let mut messages = vec![json!({"role": "user", "content": "look"})];
    for i in 0..5 {
        let id = format!("c{i}");
        messages.push(
            json!({"role": "assistant", "content": null, "tool_calls": [{"id": id,
            "type": "function", "function": {"name": "read", "arguments": "{}"}}]}),
        );
        messages.push(json!({"role": "tool", "tool_call_id": id, "content": [
            {"type": "image_url", "image_url": {"url": format!("data:image/png;base64,{image}")}}]}));
    }
    let body = json!({"messages": messages, "tools": read_tool(false)});
    let reply = server.post("/v1/chat/completions", &body).await;
    assert_eq!(
        reply.status,
        200,
        "{}",
        &reply.text[..reply.text.len().min(300)]
    );
    assert_eq!(reply.header("x-mrun-warning"), Some("image-omitted"));
}
