//! `resp_*` cases.

use crate::common::{self, Reply, Server, Setup};
use serde_json::{Value, json};

const PATH: &str = "/v1/responses";

fn user(text: &str) -> Value {
    json!({"model": "any", "input": [{"type": "message", "role": "user",
        "content": [{"type": "input_text", "text": text}]}]})
}

fn shell_tool() -> Value {
    json!([{"type": "function", "name": "shell", "description": "Run a command",
        "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}}},
        "strict": false}])
}

const SHELL_CALL: &str =
    "<tool_call>\n{\"name\": \"shell\", \"arguments\": {\"cmd\": \"ls\"}}\n</tool_call>";

fn input_tokens(reply: &Reply) -> u64 {
    assert_eq!(reply.status, 200, "{}", reply.text);
    reply.json()["usage"]["input_tokens"].as_u64().unwrap()
}

/// Named events of a streamed reply.
fn events(reply: &Reply) -> Vec<(String, Value)> {
    assert_eq!(reply.status, 200, "{}", reply.text);
    assert_eq!(reply.header("content-type"), Some("text/event-stream"));
    reply.events()
}

async fn post(server: &Server, body: &Value) -> Reply {
    server.post(PATH, body).await
}

#[tokio::test]
async fn resp_basic_nonstream() {
    let server = common::start(&["Hello there."]).await;
    let reply = post(&server, &user("hi")).await;
    assert_eq!(reply.status, 200, "{}", reply.text);
    let body = reply.json();
    assert!(body["id"].as_str().unwrap().starts_with("resp_"));
    assert_eq!(body["object"], "response");
    assert_eq!(body["status"], "completed");
    assert!(body["incomplete_details"].is_null());
    assert_eq!(body["model"], "scripted-model");
    let output = body["output"].as_array().unwrap();
    assert_eq!(output.len(), 1, "no reasoning item without effort");
    assert_eq!(output[0]["type"], "message");
    assert!(output[0]["id"].as_str().unwrap().starts_with("msg_"));
    assert_eq!(output[0]["role"], "assistant");
    assert_eq!(output[0]["status"], "completed");
    assert_eq!(
        output[0]["content"],
        json!([{"type": "output_text", "text": "Hello there.", "annotations": []}])
    );
    assert_eq!(body["usage"]["output_tokens"], 12);
    assert!(reply.header("x-mrun-max-tokens").is_some());

    // A plain string input is one user message.
    let server = common::start(&["ok"]).await;
    let reply = post(&server, &json!({"input": "hi"})).await;
    assert_eq!(reply.json()["output"][0]["content"][0]["text"], "ok");
}

#[tokio::test]
async fn resp_stream_event_order_and_sequence_numbers() {
    let server = common::start(&["Hi!"]).await;
    let mut body = user("hi");
    body["stream"] = json!(true);
    let reply = post(&server, &body).await;
    let events = events(&reply);
    let names: Vec<&str> = events.iter().map(|(n, _)| n.as_str()).collect();
    let n = names.len();
    assert_eq!(
        names[..4],
        [
            "response.created",
            "response.in_progress",
            "response.output_item.added",
            "response.content_part.added",
        ]
    );
    assert!(n > 8, "{names:?}");
    assert!(
        names[4..n - 4]
            .iter()
            .all(|n| *n == "response.output_text.delta"),
        "{names:?}"
    );
    assert_eq!(
        names[n - 4..],
        [
            "response.output_text.done",
            "response.content_part.done",
            "response.output_item.done",
            "response.completed",
        ]
    );
    for (i, (name, data)) in events.iter().enumerate() {
        assert_eq!(data["type"], name.as_str(), "every event names its type");
        assert_eq!(data["sequence_number"], i as u64);
    }
    let text: String = events
        .iter()
        .filter(|(n, _)| n == "response.output_text.delta")
        .map(|(_, e)| {
            assert_eq!(e["output_index"], 0);
            assert_eq!(e["content_index"], 0);
            assert!(e["item_id"].as_str().unwrap().starts_with("msg_"));
            e["delta"].as_str().unwrap()
        })
        .collect();
    assert_eq!(text, "Hi!");
    let completed = &events.last().unwrap().1["response"];
    assert_eq!(completed["status"], "completed");
    assert!(completed["id"].as_str().unwrap().starts_with("resp_"));
    assert_eq!(completed["id"], events[0].1["response"]["id"]);
    assert_eq!(completed["output"][0]["content"][0]["text"], "Hi!");
    assert!(completed["usage"]["input_tokens"].as_u64().unwrap() > 0);
}

#[tokio::test]
async fn resp_created_sent_immediately() {
    let server = common::start(&["Hello."]).await;
    let mut body = user("hi");
    body["stream"] = json!(true);
    let mut response = server
        .client
        .post(server.url(PATH))
        .json(&body)
        .send()
        .await
        .unwrap();
    assert_eq!(response.status().as_u16(), 200);
    // The first write carries only the lifecycle events, before anything is generated.
    let first = response.chunk().await.unwrap().unwrap();
    let first = String::from_utf8(first.to_vec()).unwrap();
    assert!(
        first.starts_with("event: response.created\n"),
        "first bytes: {first}"
    );
    assert!(first.contains("event: response.in_progress\n"));
    assert!(!first.contains("output_item"), "{first}");
}

#[tokio::test]
async fn resp_developer_message_as_system() {
    let server = common::start(&["a", "b", "c"]).await;
    let developer = json!({"input": [
        {"type": "message", "role": "developer", "content": [
            {"type": "input_text", "text": "You are terse."}]},
        {"type": "message", "role": "user", "content": "hi"}]});
    let instructions = json!({"instructions": "You are terse.",
        "input": [{"type": "message", "role": "user", "content": "hi"}]});
    let none = json!({"input": [{"type": "message", "role": "user", "content": "hi"}]});
    let with_developer = input_tokens(&post(&server, &developer).await);
    let with_instructions = input_tokens(&post(&server, &instructions).await);
    let without = input_tokens(&post(&server, &none).await);
    assert_eq!(
        with_developer, with_instructions,
        "rendered as the system turn"
    );
    assert!(with_developer > without);
}

#[tokio::test]
async fn resp_function_call_item_done_parsable_by_codex() {
    let server = common::start(&[SHELL_CALL, "Done.", "Done.", "Done."]).await;
    let body = json!({"input": [{"type": "message", "role": "user", "content": "list"}],
        "tools": shell_tool(), "tool_choice": "auto", "parallel_tool_calls": false,
        "store": false, "include": ["reasoning.encrypted_content"],
        "reasoning": {"summary": "auto"}, "stream": true});
    let events = events(&post(&server, &body).await);
    let added = events
        .iter()
        .find(|(n, _)| n == "response.output_item.added")
        .unwrap();
    assert_eq!(added.1["item"]["type"], "function_call");
    assert_eq!(added.1["item"]["arguments"], "");
    let args: String = events
        .iter()
        .filter(|(n, _)| n == "response.function_call_arguments.delta")
        .filter_map(|(_, e)| e["delta"].as_str())
        .collect();
    let done = events
        .iter()
        .find(|(n, _)| n == "response.output_item.done")
        .unwrap();
    // Codex reads function calls only from this event (research §2).
    let item = &done.1["item"];
    assert_eq!(item["type"], "function_call");
    assert!(item["id"].as_str().unwrap().starts_with("fc_"));
    let call_id = item["call_id"].as_str().unwrap().to_string();
    assert!(call_id.starts_with("call_"));
    assert_eq!(item["name"], "shell");
    assert_eq!(item["status"], "completed");
    let arguments = item["arguments"].as_str().unwrap().to_string();
    assert_eq!(arguments, args, "deltas add up to the item's arguments");
    assert_eq!(
        serde_json::from_str::<Value>(&arguments).unwrap(),
        json!({"cmd": "ls"})
    );
    assert_eq!(done.1["output_index"], 0);
    assert_eq!(events.last().unwrap().0, "response.completed");

    // resp_function_call_output_string: the call and its result come back as items.
    let history = |output: Value| {
        json!({"input": [
            {"type": "message", "role": "user", "content": "list"},
            item.clone(),
            {"type": "function_call_output", "call_id": call_id, "output": output}],
            "tools": shell_tool()})
    };
    let reply = post(&server, &history(json!("a.rs\nb.rs"))).await;
    let string_tokens = input_tokens(&reply);
    assert_eq!(reply.json()["output"][0]["content"][0]["text"], "Done.");

    // resp_function_call_output_array_text: text parts render like the string.
    let reply = post(
        &server,
        &history(json!([{"type": "input_text", "text": "a.rs\n"},
            {"type": "input_text", "text": "b.rs"}])),
    )
    .await;
    assert_eq!(input_tokens(&reply), string_tokens);
    assert!(reply.header("x-mrun-warnings").is_none());

    // resp_function_call_output_image_placeholder (FR-020).
    let reply = post(
        &server,
        &history(json!([{"type": "input_text", "text": "screenshot:"},
            {"type": "input_image", "image_url": "data:image/png;base64,iVBORw0KGgo="}])),
    )
    .await;
    assert_eq!(reply.status, 200, "{}", reply.text);
    assert_eq!(reply.header("x-mrun-warnings"), Some("image-omitted"));
}

#[tokio::test]
async fn resp_function_call_output_unknown_call_400() {
    let server = common::start(&[]).await;
    let body = json!({"input": [{"role": "user", "content": "x"},
        {"type": "function_call_output", "call_id": "call_nope", "output": "y"}]});
    let reply = post(&server, &body).await;
    assert_eq!(reply.status, 400);
    assert_eq!(reply.json()["error"]["type"], "invalid_request_error");
}

#[tokio::test]
async fn resp_user_image_rejected_with_path() {
    let server = common::start(&[]).await;
    let body = json!({"input": [{"role": "user", "content": [
        {"type": "input_text", "text": "see"},
        {"type": "input_image", "image_url": "data:image/png;base64,AA"}]}]});
    let reply = post(&server, &body).await;
    assert_eq!(reply.status, 400);
    assert_eq!(reply.json()["error"]["param"], "input[0].content[1]");
}

#[tokio::test]
async fn resp_reasoning_item_has_summary_array() {
    let server = common::start(&["Let me look.\n</think>\n\nIt is fine."]).await;
    let mut body = user("why?");
    body["reasoning"] = json!({"effort": "high", "summary": "auto"});
    body["stream"] = json!(true);
    let events = events(&post(&server, &body).await);
    let reasoning: String = events
        .iter()
        .filter(|(n, _)| n == "response.reasoning_text.delta")
        .filter_map(|(_, e)| e["delta"].as_str())
        .collect();
    assert_eq!(reasoning, "Let me look.");
    let items: Vec<&Value> = events
        .iter()
        .filter(|(n, _)| n == "response.output_item.done")
        .map(|(_, e)| &e["item"])
        .collect();
    assert_eq!(items.len(), 2);
    assert_eq!(items[0]["type"], "reasoning");
    assert!(items[0]["id"].as_str().unwrap().starts_with("rs_"));
    assert_eq!(items[0]["summary"], json!([]), "Codex needs summary");
    assert!(items[0]["encrypted_content"].as_str().is_some());
    assert_eq!(items[1]["type"], "message");
    assert_eq!(items[1]["content"][0]["text"], "It is fine.");
    let usage = &events.last().unwrap().1["response"]["usage"];
    assert!(
        usage["output_tokens_details"]["reasoning_tokens"]
            .as_u64()
            .unwrap()
            > 0
    );
}

#[tokio::test]
async fn resp_encrypted_content_roundtrip_restores_think() {
    let server = common::start(&["Plan it.\n</think>\n\nAnswer.", "x", "y", "z"]).await;
    let mut first = user("q");
    first["reasoning"] = json!({"effort": "medium"});
    let reply = post(&server, &first).await;
    let output = reply.json()["output"].as_array().unwrap().clone();
    assert_eq!(output[0]["type"], "reasoning");
    assert_eq!(output[1]["content"][0]["text"], "Answer.");

    // Codex replays the reasoning item without `content`, keeping `encrypted_content`.
    let mut replayed = output[0].clone();
    replayed.as_object_mut().unwrap().remove("content");
    let next = |reasoning: Option<Value>| {
        let mut input = vec![json!({"role": "user", "content": "q"})];
        input.extend(reasoning);
        input.push(output[1].clone());
        input.push(json!({"role": "user", "content": "more"}));
        json!({"input": input})
    };
    let with = input_tokens(&post(&server, &next(Some(replayed))).await);
    let without = input_tokens(&post(&server, &next(None)).await);
    let foreign = json!({"type": "reasoning", "id": "rs_x", "summary": [],
        "encrypted_content": "gAAAAABforeign"});
    let dropped = input_tokens(&post(&server, &next(Some(foreign))).await);
    assert!(
        with >= without + "Plan it.".len() as u64,
        "the reasoning is rendered again ({with} vs {without})"
    );
    assert_eq!(dropped, without, "another provider's blob is dropped");
}

#[tokio::test]
async fn resp_reasoning_present_without_effort_no_think() {
    let server = common::start(&["Answer.", "Answer."]).await;
    // Codex's fallback for unknown models: `reasoning` without an effort.
    let mut body = user("q");
    body["reasoning"] = json!({"summary": "auto"});
    let output = post(&server, &body).await.json()["output"].clone();
    assert_eq!(output.as_array().unwrap().len(), 1);
    assert_eq!(output[0]["type"], "message");
    assert_eq!(output[0]["content"][0]["text"], "Answer.");
    body["reasoning"] = json!({"effort": "none"});
    let output = post(&server, &body).await.json()["output"].clone();
    assert_eq!(output[0]["type"], "message");
}

#[tokio::test]
async fn resp_previous_response_id_rejected() {
    let server = common::start(&[]).await;
    let mut body = user("x");
    body["previous_response_id"] = json!("resp_123");
    let reply = post(&server, &body).await;
    assert_eq!(reply.status, 400);
    let error = &reply.json()["error"];
    assert_eq!(error["type"], "invalid_request_error");
    assert_eq!(error["param"], "previous_response_id");
}

#[tokio::test]
async fn resp_item_reference_rejected() {
    let server = common::start(&[]).await;
    let body = json!({"input": [{"role": "user", "content": "x"},
        {"type": "item_reference", "id": "msg_1"}]});
    let reply = post(&server, &body).await;
    assert_eq!(reply.status, 400);
    assert_eq!(reply.json()["error"]["param"], "input[1]");
}

#[tokio::test]
async fn resp_store_true_accepted_not_persisted() {
    let server = common::start(&["ok"]).await;
    let mut body = user("x");
    body["store"] = json!(true);
    let reply = post(&server, &body).await;
    assert_eq!(reply.status, 200, "{}", reply.text);
    let stored = reply.json();
    assert_eq!(stored["store"], false, "nothing is stored");
    // The id cannot be continued from.
    let mut next = user("y");
    next["previous_response_id"] = stored["id"].clone();
    assert_eq!(post(&server, &next).await.status, 400);
}

#[tokio::test]
async fn resp_usage_cached_tokens() {
    let server = common::start(&["one", "two"]).await;
    let body = user("the same prompt twice");
    let first = post(&server, &body).await.json();
    let second = post(&server, &body).await.json();
    for body in [&first, &second] {
        let usage = &body["usage"];
        let input = usage["input_tokens"].as_u64().unwrap();
        let cached = usage["input_tokens_details"]["cached_tokens"]
            .as_u64()
            .unwrap();
        assert!(
            cached < input,
            "input includes cached tokens, P − 1 at most"
        );
        assert_eq!(
            usage["total_tokens"].as_u64().unwrap(),
            input + usage["output_tokens"].as_u64().unwrap()
        );
        assert!(usage["output_tokens_details"]["reasoning_tokens"].is_number());
    }
    assert_eq!(first["usage"]["input_tokens_details"]["cached_tokens"], 0);
    assert!(
        second["usage"]["input_tokens_details"]["cached_tokens"]
            .as_u64()
            .unwrap()
            > 0,
        "the repeated prompt is reused: {}",
        second["usage"]
    );
}

#[tokio::test]
async fn resp_incomplete_max_output_tokens() {
    let server = common::start(&["one two three four five", "one two three four five"]).await;
    let mut body = user("count");
    body["max_output_tokens"] = json!(4);
    let reply = post(&server, &body).await;
    let json = reply.json();
    assert_eq!(json["status"], "incomplete");
    assert_eq!(json["incomplete_details"]["reason"], "max_output_tokens");
    assert_eq!(json["output"][0]["status"], "incomplete");
    assert_eq!(json["usage"]["output_tokens"], 4);
    assert_eq!(reply.header("x-mrun-max-tokens"), Some("4"));

    body["stream"] = json!(true);
    let events = events(&post(&server, &body).await);
    let (name, last) = events.last().unwrap();
    assert_eq!(name, "response.incomplete");
    assert_eq!(last["response"]["status"], "incomplete");
    assert_eq!(
        last["response"]["incomplete_details"]["reason"],
        "max_output_tokens"
    );
    assert_eq!(last["response"]["usage"]["output_tokens"], 4);
}

#[tokio::test]
async fn resp_failed_context_length_exceeded_code() {
    let server = common::start_with(
        &["x"],
        Setup {
            capacity: 32,
            ..Setup::default()
        },
    )
    .await;
    let mut body = user(&"long ".repeat(20));
    let reply = post(&server, &body).await;
    assert_eq!(reply.status, 400);
    assert_eq!(reply.json()["error"]["code"], "context_length_exceeded");

    // resp_stream_context_overflow_in_stream: Codex compacts only on the in-stream form.
    body["stream"] = json!(true);
    let events = events(&post(&server, &body).await);
    let names: Vec<&str> = events.iter().map(|(n, _)| n.as_str()).collect();
    assert_eq!(
        names,
        [
            "response.created",
            "response.in_progress",
            "response.failed"
        ]
    );
    let failed = &events[2].1;
    assert_eq!(failed["sequence_number"], 2);
    assert_eq!(failed["response"]["status"], "failed");
    assert_eq!(
        failed["response"]["error"]["code"],
        "context_length_exceeded"
    );
    assert!(
        failed["response"]["error"]["message"]
            .as_str()
            .unwrap()
            .contains("too long")
    );
}

#[tokio::test]
async fn resp_web_search_tool_dropped_logged() {
    let server = common::start(&["a", "b"]).await;
    let mut body = user("search");
    body["tools"] = shell_tool();
    let plain = input_tokens(&post(&server, &body).await);
    body["tools"]
        .as_array_mut()
        .unwrap()
        .push(json!({"type": "web_search", "search_context_size": "medium"}));
    // Dropped (and logged `tool-dropped`): the prompt is the same as without it.
    assert_eq!(input_tokens(&post(&server, &body).await), plain);
}

#[tokio::test]
async fn resp_custom_tool_rejected() {
    let server = common::start(&[]).await;
    let mut body = user("patch");
    body["tools"] = json!([{"type": "custom", "name": "apply_patch",
        "description": "freeform", "format": {"type": "grammar"}}]);
    let reply = post(&server, &body).await;
    assert_eq!(reply.status, 400);
    let error = &reply.json()["error"];
    assert_eq!(error["type"], "invalid_request_error");
    assert_eq!(error["param"], "tools[0].type");
    assert!(error["message"].as_str().unwrap().contains("custom"));
}

#[tokio::test]
async fn resp_no_limit_defaults_to_remaining_capacity() {
    let capacity = 4096;
    let server = common::start_with(
        &["ok", "ok"],
        Setup {
            capacity,
            ..Setup::default()
        },
    )
    .await;
    let reply = post(&server, &user("hi")).await;
    let prompt = input_tokens(&reply);
    let expected = (u64::from(capacity) - prompt).to_string();
    assert_eq!(reply.header("x-mrun-max-tokens"), Some(expected.as_str()));
    assert!(
        reply.header("x-mrun-warnings").is_none(),
        "the default is not a clamp"
    );
    assert_eq!(reply.json()["status"], "completed");

    // Streaming: the first `: mrun` comment carries the same limit.
    let mut body = user("hi");
    body["stream"] = json!(true);
    let reply = post(&server, &body).await;
    let meta = reply
        .frames()
        .into_iter()
        .find_map(|(n, c)| (n == ":").then_some(c))
        .unwrap();
    let meta: Value = serde_json::from_str(meta.strip_prefix("mrun ").unwrap()).unwrap();
    assert_eq!(meta["max_tokens"].to_string(), expected);
}
