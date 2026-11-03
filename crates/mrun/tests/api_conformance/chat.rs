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
    assert_eq!(
        body["model"], "any",
        "D033: the requested model id is echoed"
    );
    assert_eq!(body["choices"][0]["message"]["content"], "Hello there.");
    assert_eq!(body["choices"][0]["finish_reason"], "stop");
    assert_eq!(body["usage"]["completion_tokens"], 12, "EOS is not output");
    assert!(body["usage"]["prompt_tokens_details"]["cached_tokens"].is_number());
    assert!(reply.header("x-mrun-max-tokens").is_some());
    // Without a model id the served model's is reported.
    let reply = server
        .post(
            "/v1/chat/completions",
            &json!({"messages": [{"role": "user", "content": "hi"}]}),
        )
        .await;
    assert_eq!(reply.json()["model"], "scripted-model", "{}", reply.text);
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
    let server = common::start(&["<think>\nLet me think.\n</think>\n\nAnswer."]).await;
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
async fn chat_unmet_memory_is_context_length_exceeded_not_overloaded() {
    // FR-043: the prompt fits the token capacity but not the state memory, with nothing cached
    // to shed: one lane can never fit it later, so the client must compact.
    let server = common::start_with(
        &["x"],
        Setup {
            state_tokens: Some(64),
            ..Setup::default()
        },
    )
    .await;
    let reply = server
        .post("/v1/chat/completions", &user(&"long ".repeat(60)))
        .await;
    assert_eq!(reply.status, 400, "{}", reply.text);
    let error = &reply.json()["error"];
    assert_eq!(error["type"], "invalid_request_error");
    assert_eq!(error["code"], "context_length_exceeded");
    let message = error["message"].as_str().unwrap();
    assert!(message.starts_with("prompt is too long: "), "{message}");
    assert!(message.contains(" maximum"), "{message}");
    assert!(reply.headers.get("retry-after").is_none());
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
    // In the API's own error shape, not axum's plain text.
    let body: Value = reply.json().await.unwrap();
    assert_eq!(body["error"]["type"], "invalid_request_error");
    assert_eq!(body["error"]["code"], "request_too_large");
    let reply = server
        .client
        .post(server.url("/v1/messages"))
        .header("content-type", "application/json")
        .body(vec![b' '; 64 * 1024 * 1024 + 1])
        .send()
        .await
        .unwrap();
    assert_eq!(reply.status().as_u16(), 413);
    let body: Value = reply.json().await.unwrap();
    assert_eq!(body["error"]["type"], "request_too_large");
}

#[tokio::test]
async fn xapi_invalid_sampling_400_before_headers() {
    let server = common::start(&["unused"]).await;
    for (field, value) in [("temperature", json!(-1.0)), ("top_p", json!(0.0))] {
        let mut body = user("hi");
        body[field] = value;
        body["stream"] = json!(true);
        let reply = server.post("/v1/chat/completions", &body).await;
        assert_eq!(reply.status, 400, "{}", reply.text);
        assert_eq!(reply.json()["error"]["param"], field);
    }
    // Nothing reached the engine: the script's reply is still there.
    let reply = server.post("/v1/chat/completions", &user("hi")).await;
    assert_eq!(reply.json()["choices"][0]["message"]["content"], "unused");
}

fn read_tool(strict: bool) -> Value {
    json!([{"type": "function", "function": {"name": "read", "strict": strict,
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}},
            "required": ["path"], "additionalProperties": false}}}])
}

#[tokio::test]
async fn chat_tool_choice_required_constrains_call() {
    // Nothing is forced: the model samples every token of the call, the grammar only masks
    // (xapi_no_forced_tokens).
    let body = json!({"messages": [{"role": "user", "content": "go"}],
        "tools": read_tool(false), "tool_choice": "required"});
    assert!(common::plan(&body).answer.is_some());
    let server = common::start(&[
        "<tool_call>\n{\"name\": \"read\", \"arguments\": {\"path\":\"a.rs\"}}\n</tool_call>",
        "Sure, reading it.",
    ])
    .await;
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
    // Prose is masked: the scripted engine refuses the token (a real model samples an
    // allowed one), which surfaces as an engine error, not an overload.
    let reply = server.post("/v1/chat/completions", &body).await;
    assert_eq!(reply.status, 500, "{}", reply.text);
    assert_eq!(reply.json()["error"]["type"], "server_error");
}

#[tokio::test]
async fn chat_strict_tool_under_auto_not_enforced() {
    // D032: `strict` under `auto` is logged, not enforced; arguments come back as written.
    let body = json!({"messages": [{"role": "user", "content": "go"}], "tools": read_tool(true)});
    let plan = common::plan(&body);
    assert!(plan.answer.is_none());
    assert_eq!(plan.unenforced, ["tools[].strict"]);
    let server = common::start(&[
        "<tool_call>\n{\"name\": \"read\", \"arguments\": {\"path\": 1}}\n</tool_call>",
    ])
    .await;
    let reply = server.post("/v1/chat/completions", &body).await;
    let call = &reply.json()["choices"][0]["message"]["tool_calls"][0];
    assert_eq!(
        call["function"]["arguments"], "{\"path\": 1}",
        "{}",
        reply.text
    );
}

#[tokio::test]
async fn chat_parallel_tool_calls_false_ends_after_the_first_call() {
    let call =
        "<tool_call>\n{\"name\": \"read\", \"arguments\": {\"path\":\"a.rs\"}}\n</tool_call>";
    let two = format!("{call}\n{call}");
    let server = common::start(&[&two, &two, call]).await;
    let mut body = json!({"messages": [{"role": "user", "content": "go"}],
        "tools": read_tool(false)});
    let reply = server.post("/v1/chat/completions", &body).await;
    let calls = &reply.json()["choices"][0]["message"]["tool_calls"];
    assert_eq!(calls.as_array().unwrap().len(), 2, "{}", reply.text);
    // Off: after the first call's closer only EOS may follow; the scripted engine refuses the
    // second call's tokens, a model ends its turn.
    body["parallel_tool_calls"] = json!(false);
    assert_eq!(server.post("/v1/chat/completions", &body).await.status, 500);
    let reply = server.post("/v1/chat/completions", &body).await;
    let calls = &reply.json()["choices"][0]["message"]["tool_calls"];
    assert_eq!(calls.as_array().unwrap().len(), 1, "{}", reply.text);
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
    assert!(common::plan(&body).answer.is_some());
    let server = common::start(&["{\"a\":1}", "hello"]).await;
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

#[tokio::test]
async fn chat_no_limit_defaults_min_remaining_32768() {
    // FR-015 (D052): one default for all APIs, min(remaining capacity, 32,768).
    for (capacity, small) in [(4096_u32, true), (65_536, false)] {
        let server = common::start_with(
            &["ok"],
            Setup {
                capacity,
                ..Setup::default()
            },
        )
        .await;
        let reply = server.post("/v1/chat/completions", &user("hi")).await;
        assert_eq!(reply.status, 200, "{}", reply.text);
        let prompt = reply.json()["usage"]["prompt_tokens"].as_u64().unwrap();
        let expected = if small {
            u64::from(capacity) - prompt
        } else {
            32_768
        };
        assert_eq!(
            reply.header("x-mrun-max-tokens"),
            Some(expected.to_string().as_str())
        );
        assert!(reply.header("x-mrun-warning").is_none());
    }
}

#[tokio::test]
async fn chat_nonstream_refused_by_estimated_wait_503_server_busy() {
    // FR-010 (D052): the admission refusal is the queue-deadline error, 503 `server_busy` with
    // `retry-after`, in the OpenAI shape; a streaming request is admitted (no deadline).
    use mrun::api::run::Timing;
    use std::time::{Duration, Instant};
    let setup = Setup {
        timing: Timing {
            keepalive: Duration::from_secs(10),
            queue_wait: Duration::from_millis(300),
        },
        step_delay: Some(Duration::from_millis(100)),
        ..Setup::default()
    };
    let server = common::start_with(&["busy busy", "long long", "streamed"], setup).await;
    assert_eq!(
        server.post("/v1/chat/completions", &user("a")).await.status,
        200
    );
    let client = server.client.clone();
    let url = server.url("/v1/chat/completions");
    let running = tokio::spawn(async move { client.post(url).json(&user("b")).send().await });
    tokio::time::sleep(Duration::from_millis(100)).await;
    let started = Instant::now();
    let reply = server.post("/v1/chat/completions", &user("c")).await;
    assert_eq!(reply.status, 503, "{}", reply.text);
    assert!(
        started.elapsed() < Duration::from_millis(200),
        "refused at once"
    );
    let error = &reply.json()["error"];
    assert_eq!(error["type"], "server_error");
    assert_eq!(error["code"], "server_busy");
    let seconds: u64 = reply.header("retry-after").unwrap().parse().unwrap();
    assert!((1..=60).contains(&seconds));
    let mut body = user("d");
    body["stream"] = json!(true);
    let reply = server.post("/v1/chat/completions", &body).await;
    assert_eq!(reply.status, 200);
    assert_eq!(reply.frames().last().unwrap().1, "[DONE]", "{}", reply.text);
    running.await.unwrap().unwrap();
}
