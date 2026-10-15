//! `msg_*` cases and cross-cutting queue, keepalive and disconnect cases.

use crate::common::{self, Setup};
use mrun::api::run::Timing;
use serde_json::{Value, json};
use std::time::{Duration, Instant};

fn user(text: &str) -> Value {
    json!({"model": "claude-sonnet-4-5", "max_tokens": 1000,
        "messages": [{"role": "user", "content": text}]})
}

fn slow(keepalive_ms: u64, queue_wait_ms: u64, step_ms: u64) -> Setup {
    Setup {
        timing: Timing {
            keepalive: Duration::from_millis(keepalive_ms),
            queue_wait: Duration::from_millis(queue_wait_ms),
        },
        step_delay: Some(Duration::from_millis(step_ms)),
        ..Setup::default()
    }
}

#[tokio::test]
async fn msg_basic_nonstream() {
    let server = common::start(&["Hello."]).await;
    let reply = server.post("/v1/messages?beta=true", &user("hi")).await;
    assert_eq!(reply.status, 200, "{}", reply.text);
    let body = reply.json();
    assert_eq!(body["type"], "message");
    assert_eq!(body["model"], "scripted-model");
    assert_eq!(body["content"], json!([{"type": "text", "text": "Hello."}]));
    assert_eq!(body["stop_reason"], "end_turn");
    assert_eq!(body["usage"]["cache_creation_input_tokens"], 0);
}

#[tokio::test]
async fn msg_stream_tool_use_input_json_delta() {
    let call = "<tool_call>\n{\"name\": \"Read\", \"arguments\": {\"file_path\": \"/r/a.rs\"}}\n</tool_call>";
    let server = common::start(&[call]).await;
    let mut body = user("read it");
    body["stream"] = json!(true);
    body["tools"] = json!([{"name": "Read", "input_schema": {"type": "object",
        "properties": {"file_path": {"type": "string"}}}}]);
    let reply = server
        .post_with(
            "/v1/messages",
            &body,
            &[
                ("anthropic-beta", "interleaved-thinking-2025-05-14"),
                ("x-api-key", ""),
            ],
        )
        .await;
    let events = reply.events();
    let names: Vec<&str> = events.iter().map(|(n, _)| n.as_str()).collect();
    assert_eq!(names.first(), Some(&"message_start"));
    assert_eq!(names.last(), Some(&"message_stop"));
    let start = events
        .iter()
        .find(|(n, e)| n == "content_block_start" && e["content_block"]["type"] == "tool_use")
        .unwrap();
    assert!(
        start.1["content_block"]["id"]
            .as_str()
            .unwrap()
            .starts_with("toolu_")
    );
    let json: String = events
        .iter()
        .filter_map(|(_, e)| e["delta"]["partial_json"].as_str())
        .collect();
    assert_eq!(
        serde_json::from_str::<Value>(&json).unwrap(),
        json!({"file_path": "/r/a.rs"})
    );
    let delta = events.iter().find(|(n, _)| n == "message_delta").unwrap();
    assert_eq!(delta.1["delta"]["stop_reason"], "tool_use");
}

#[tokio::test]
async fn msg_thinking_enabled_blocks_with_signature() {
    let server = common::start(&["Plan first.\n</think>\n\nDone."]).await;
    let mut body = user("do it");
    body["thinking"] = json!({"type": "enabled", "budget_tokens": 2048});
    let reply = server.post("/v1/messages", &body).await;
    let content = &reply.json()["content"];
    assert_eq!(content[0]["type"], "thinking");
    assert_eq!(content[0]["thinking"], "Plan first.");
    assert!(
        content[0]["signature"]
            .as_str()
            .unwrap()
            .starts_with("mrun1.")
    );
    assert_eq!(content[1], json!({"type": "text", "text": "Done."}));
}

#[tokio::test]
async fn msg_stop_sequence_reports_value() {
    let server = common::start(&["one STOP two"]).await;
    let mut body = user("go");
    body["stop_sequences"] = json!(["STOP"]);
    let body = server.post("/v1/messages", &body).await.json();
    assert_eq!(body["stop_reason"], "stop_sequence");
    assert_eq!(body["stop_sequence"], "STOP");
}

#[tokio::test]
async fn msg_max_tokens_stop_reason() {
    let server = common::start(&["a long answer"]).await;
    let mut body = user("go");
    body["max_tokens"] = json!(4);
    assert_eq!(
        server.post("/v1/messages", &body).await.json()["stop_reason"],
        "max_tokens"
    );
}

#[tokio::test]
async fn msg_context_overflow_prompt_too_long() {
    let server = common::start_with(
        &[],
        Setup {
            capacity: 40,
            ..Setup::default()
        },
    )
    .await;
    let reply = server
        .post("/v1/messages", &user(&"word ".repeat(30)))
        .await;
    assert_eq!(reply.status, 400);
    let error = &reply.json()["error"];
    assert_eq!(error["type"], "invalid_request_error");
    assert!(
        error["message"]
            .as_str()
            .unwrap()
            .starts_with("prompt is too long: ")
    );
}

#[tokio::test]
async fn msg_count_tokens_matches_prompt() {
    let server = common::start(&["Hello."]).await;
    let counted = server
        .post(
            "/v1/messages/count_tokens",
            &json!({"model": "x",
            "messages": [{"role": "user", "content": "hi"}]}),
        )
        .await
        .json()["input_tokens"]
        .as_u64()
        .unwrap();
    let reply = server.post("/v1/messages", &user("hi")).await.json();
    let usage = &reply["usage"];
    let input = usage["input_tokens"].as_u64().unwrap()
        + usage["cache_read_input_tokens"].as_u64().unwrap();
    assert_eq!(counted, input);
}

#[tokio::test]
async fn msg_assistant_prefill_continues() {
    let server = common::start(&["1, 2]}"]).await;
    let body = json!({"max_tokens": 100, "messages": [
        {"role": "user", "content": "JSON list"},
        {"role": "assistant", "content": "{\"a\": ["}]});
    let body = server.post("/v1/messages", &body).await.json();
    assert_eq!(body["content"][0]["text"], "1, 2]}");
}

#[tokio::test]
async fn msg_ping_during_silence() {
    let server = common::start_with(&["slow"], slow(30, 60_000, 60)).await;
    let mut body = user("hi");
    body["stream"] = json!(true);
    let reply = server.post("/v1/messages", &body).await;
    let events = reply.events();
    assert!(events.iter().any(|(n, _)| n == "ping"), "{}", reply.text);
    assert_eq!(events.last().unwrap().0, "message_stop");
}

#[tokio::test]
async fn msg_overloaded_529_retry_after_int() {
    let mut setup = slow(10_000, 60_000, 100);
    setup.queue = 0;
    let server = common::start_with(&["busy busy", "x"], setup).await;
    let client = server.client.clone();
    let url = server.url("/v1/messages");
    let first = tokio::spawn(async move { client.post(url).json(&user("a")).send().await });
    tokio::time::sleep(Duration::from_millis(100)).await;
    let reply = server.post("/v1/messages", &user("b")).await;
    assert_eq!(reply.status, 529, "{}", reply.text);
    assert_eq!(reply.json()["error"]["type"], "overloaded_error");
    let seconds: u64 = reply.header("retry-after").unwrap().parse().unwrap();
    assert!((1..=60).contains(&seconds));
    first.await.unwrap().unwrap();
}

#[tokio::test]
async fn msg_stream_deadline_after_headers_overloaded_error() {
    let server = common::start_with(&["busy busy busy busy", "never"], slow(20, 150, 100)).await;
    let client = server.client.clone();
    let url = server.url("/v1/messages");
    let first = tokio::spawn(async move { client.post(url).json(&user("a")).send().await });
    tokio::time::sleep(Duration::from_millis(50)).await;
    let mut body = user("b");
    body["stream"] = json!(true);
    let reply = server.post("/v1/messages", &body).await;
    let events = reply.events();
    assert!(
        events.iter().any(|(n, _)| n == "ping"),
        "keepalive while queued"
    );
    let last = events.last().unwrap();
    assert_eq!(last.0, "error");
    assert_eq!(last.1["error"]["type"], "overloaded_error");
    first.await.unwrap().unwrap();
}

#[tokio::test]
async fn xapi_disconnect_cancels_within_one_token() {
    // A long reply at 20 ms per token; the client leaves after the first bytes.
    let server = common::start_with(&[&"z".repeat(500), "next"], slow(10_000, 60_000, 20)).await;
    let mut body = user("long");
    body["stream"] = json!(true);
    let mut response = server
        .client
        .post(server.url("/v1/messages"))
        .json(&body)
        .send()
        .await
        .unwrap();
    response.chunk().await.unwrap();
    drop(response);
    // The worker is free again well before the 10 s the full reply would take. (The scripted
    // engine resumes its script where the cancelled request stopped, so the text is not checked.)
    let started = Instant::now();
    let mut next = user("again");
    next["max_tokens"] = json!(3);
    let reply = server.post("/v1/messages", &next).await;
    assert_eq!(reply.status, 200);
    assert!(
        started.elapsed() < Duration::from_secs(2),
        "{:?}",
        started.elapsed()
    );
}

#[tokio::test]
async fn xapi_disconnect_while_queued_releases_slot() {
    let mut setup = slow(10_000, 60_000, 50);
    setup.queue = 1;
    let server = common::start_with(&["busy busy", "after"], setup).await;
    let client = server.client.clone();
    let url = server.url("/v1/messages");
    let first = tokio::spawn(async move { client.post(url).json(&user("a")).send().await });
    tokio::time::sleep(Duration::from_millis(50)).await;
    // A queued non-streaming request whose client gives up.
    let queued = server
        .client
        .post(server.url("/v1/messages"))
        .json(&user("b"))
        .timeout(Duration::from_millis(100))
        .send()
        .await;
    assert!(queued.is_err());
    first.await.unwrap().unwrap();
    // The abandoned request never ran: the next one gets the next scripted reply.
    let reply = server.post("/v1/messages", &user("c")).await;
    assert_eq!(reply.json()["content"][0]["text"], "after");
}

#[tokio::test]
async fn status_and_models_report_the_machine() {
    let server = common::start_with(&["busy busy busy"], slow(10_000, 60_000, 50)).await;
    let models = server
        .client
        .get(server.url("/v1/models"))
        .send()
        .await
        .unwrap()
        .json::<Value>()
        .await
        .unwrap();
    assert_eq!(models["data"][0]["id"], "scripted-model");
    assert_eq!(models["data"][0]["context_window"], 8192);
    assert_eq!(models["data"][0]["max_model_len"], 8192);
    assert_eq!(models["data"][0]["max_output_tokens"], 8191);

    let client = server.client.clone();
    let url = server.url("/v1/messages");
    let running = tokio::spawn(async move { client.post(url).json(&user("a")).send().await });
    tokio::time::sleep(Duration::from_millis(150)).await;
    let status = server
        .client
        .get(server.url("/mrun/status"))
        .send()
        .await
        .unwrap()
        .json::<Value>()
        .await
        .unwrap();
    assert_eq!(status["schema_version"], 1);
    assert_eq!(status["model"]["syntax"], "hermes_json");
    assert_eq!(status["budget"]["kv_capacity_tokens"], 8192);
    assert_eq!(status["active"]["api"], "messages");
    assert_eq!(status["active"]["phase"], "generating");
    assert_eq!(status["queued"], 0);
    running.await.unwrap().unwrap();
    let after: Value = server
        .client
        .get(server.url("/mrun/status"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert!(after["active"].is_null());
}

#[tokio::test]
async fn xapi_ignore_eos_runs_to_the_limit() {
    let server = common::start(&["ab", "never used"]).await;
    let mut body = user("go");
    body["max_tokens"] = json!(6);
    let reply = server
        .post_with("/v1/messages", &body, &[("x-mrun-ignore-eos", "on")])
        .await
        .json();
    assert_eq!(reply["stop_reason"], "max_tokens");
    assert_eq!(reply["usage"]["output_tokens"], 6);
}

#[tokio::test]
async fn msg_diagnostics_and_context_management_ignored() {
    let server = common::start(&["ok"]).await;
    let mut body = user("hi");
    body["diagnostics"] = json!({"cache_misses": true});
    body["context_management"] = json!({"edits": [{"type": "clear_tool_uses_20250919"}]});
    body["metadata"] = json!({"user_id": "u"});
    body["system"] = json!([{"type": "text", "text": "S", "cache_control": {"type": "ephemeral"}}]);
    let reply = server.post("/v1/messages", &body).await;
    assert_eq!(reply.status, 200, "{}", reply.text);
}

#[tokio::test]
async fn msg_thinking_budget_closes_the_block() {
    // Budget 5: after five reasoning tokens the block is closed by forced tokens and the rest of
    // the reply is the answer.
    let server = common::start(&["abcdefghij"]).await;
    let mut body = user("think briefly");
    body["thinking"] = json!({"type": "enabled", "budget_tokens": 5});
    let reply = server.post("/v1/messages", &body).await.json();
    assert_eq!(reply["content"][0]["thinking"], "abcde", "{reply}");
    assert_eq!(reply["content"][1]["text"], "fghij");
}

#[tokio::test]
async fn msg_tool_reference_inline_no_prefix_change() {
    // A deferred tool is not in the tools region; a tool search result renders its definition
    // inline, so the prompt only grows at the end.
    let server = common::start(&["one", "two"]).await;
    let tools = json!([
        {"name": "Read", "input_schema": {"type": "object"}},
        {"name": "mcp__github__list_issues", "defer_loading": true,
         "description": "List issues", "input_schema": {"type": "object"}}]);
    let first = json!({"max_tokens": 50, "tools": tools, "messages": [
        {"role": "user", "content": "find issues"}]});
    let count = |body: &Value| {
        let body = body.clone();
        let server = &server;
        async move {
            let reply = server.post("/v1/messages/count_tokens", &body).await;
            assert_eq!(reply.status, 200, "{}", reply.text);
            reply.json()["input_tokens"].as_u64().unwrap()
        }
    };
    let mut without_deferred = first.clone();
    without_deferred["tools"] = json!([tools[0]]);
    assert_eq!(count(&first).await, count(&without_deferred).await);

    let second = json!({"max_tokens": 50, "tools": tools, "messages": [
        {"role": "user", "content": "find issues"},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "ToolSearch",
            "input": {"query": "issues"}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1",
            "content": [{"type": "tool_reference", "tool_name": "mcp__github__list_issues"}]}]}]});
    let reply = server.post("/v1/messages", &second).await;
    assert_eq!(reply.status, 200, "{}", reply.text);

    let mut unknown = second.clone();
    unknown["messages"][2]["content"][0]["content"][0]["tool_name"] = json!("nope");
    let reply = server.post("/v1/messages", &unknown).await;
    assert_eq!(reply.status, 400);
    assert!(
        reply.json()["error"]["message"]
            .as_str()
            .unwrap()
            .contains("nope")
    );

    let mut all_deferred = first.clone();
    all_deferred["tools"] = json!([tools[1]]);
    assert_eq!(server.post("/v1/messages", &all_deferred).await.status, 400);
}

#[tokio::test]
async fn msg_midconversation_system_in_place() {
    let server = common::start(&["ok"]).await;
    let body = json!({"max_tokens": 50, "system": "S", "messages": [
        {"role": "user", "content": "a"},
        {"role": "assistant", "content": "b"},
        {"role": "system", "content": "Reminder: be brief."},
        {"role": "user", "content": "c"}]});
    let with_reminder =
        server.post("/v1/messages/count_tokens", &body).await.json()["input_tokens"]
            .as_u64()
            .unwrap();
    let mut plain = body.clone();
    plain["messages"].as_array_mut().unwrap().remove(2);
    let without = server
        .post("/v1/messages/count_tokens", &plain)
        .await
        .json()["input_tokens"]
        .as_u64()
        .unwrap();
    assert!(with_reminder > without);
    assert_eq!(server.post("/v1/messages", &body).await.status, 200);
}

#[tokio::test]
async fn cache_report_status_and_clear() {
    let server = common::start(&["Sure.", "Again.", "Once more.", "Fourth."]).await;
    let first = json!({"max_tokens": 50, "system": "S", "messages": [
        {"role": "user", "content": "first question"}]});
    let reply = server.post("/v1/messages", &first).await;
    assert_eq!(reply.header("x-mrun-cache-cause"), Some("new"));
    let input = reply.json()["usage"]["input_tokens"].as_u64().unwrap();

    // The next turn: this Qwen3-style template renders an empty think block in the generation
    // header but not in history, so the previous reply re-renders differently. The report says
    // so (changed, at the assistant message), and everything before the reply is reused.
    let second = json!({"max_tokens": 50, "system": "S", "messages": [
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": "Sure."},
        {"role": "user", "content": "and then?"}]});
    let reply = server.post("/v1/messages", &second).await;
    assert_eq!(reply.header("x-mrun-cache-cause"), Some("changed"));
    assert_eq!(reply.header("x-mrun-cache-message"), Some("1:assistant"));
    let usage = &reply.json()["usage"];
    let cached: u64 = reply
        .header("x-mrun-cache-cached")
        .unwrap()
        .parse()
        .unwrap();
    assert_eq!(cached, usage["cache_read_input_tokens"].as_u64().unwrap());
    assert!(
        cached + 10 >= input,
        "only the reply header differs: {cached} vs {input}"
    );
    // The same request again is a full hit.
    let reply = server.post("/v1/messages", &second).await;
    assert_eq!(
        reply.header("x-mrun-cache-cause"),
        None,
        "{:?}",
        reply.headers
    );

    let status: Value = server
        .client
        .get(server.url("/mrun/status"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert!(status["cache"]["live_tokens"].as_u64().unwrap() > 0);
    assert!(status["cache"]["cached_tokens_total"].as_u64().unwrap() >= cached);
    assert!(status["last"]["cause"].is_null());

    // Clear while idle: the live state goes now; the next request starts from nothing.
    let cleared: Value = server
        .client
        .post(server.url("/mrun/cache/clear"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(cleared["live_reset"], "now");
    let reply = server.post("/v1/messages", &second).await;
    assert_eq!(reply.header("x-mrun-cache-cause"), Some("new"));
    assert_eq!(reply.header("x-mrun-cache-cached"), Some("0"));
}

#[tokio::test]
async fn no_cache_reports_disabled() {
    let setup = Setup {
        no_cache: true,
        ..Setup::default()
    };
    let server = common::start_with(&["a", "b"], setup).await;
    server.post("/v1/messages", &user("same")).await;
    let reply = server.post("/v1/messages", &user("same")).await;
    assert_eq!(reply.header("x-mrun-cache-cause"), Some("disabled"));
    assert_eq!(reply.json()["usage"]["cache_read_input_tokens"], 0);
}

#[tokio::test]
async fn cache_divergence_names_the_message_in_the_stream_comment() {
    let server = common::start(&["one", "two"]).await;
    let body = |edited: &str| {
        json!({"max_tokens": 50, "stream": true, "system": "S", "messages": [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": edited},
            {"role": "user", "content": "follow-up"}]})
    };
    server.post("/v1/messages", &body("original answer")).await;
    let reply = server.post("/v1/messages", &body("edited answer")).await;
    let comment = &reply.frames()[0].1;
    assert!(comment.contains("\"cause\":\"changed\""), "{comment}");
    assert!(comment.contains("\"role\":\"assistant\""), "{comment}");
    assert!(
        comment.contains("\"index\":1"),
        "API message index: {comment}"
    );
}
