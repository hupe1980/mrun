//! Conformance on the real Qwen3.5 and Qwen3.6 templates (002 T075): the same tool loop through
//! all three APIs, rendered by the families' own templates with Qwen XML calls.

use crate::common::{self, QWEN3_5_TEMPLATE, QWEN3_6_TEMPLATE, Setup};
use mrun_engine::SyntaxId;
use serde_json::{Value, json};

const CALL: &str = "<tool_call>\n<function=read>\n<parameter=path>\na.rs\n</parameter>\n<parameter=opts>\n{\"lines\": [1, 2]}\n</parameter>\n</function>\n</tool_call>";

const TEMPLATES: [(&str, &str); 2] = [("qwen3.5", QWEN3_5_TEMPLATE), ("qwen3.6", QWEN3_6_TEMPLATE)];

fn setup(template: &'static str) -> Setup {
    Setup {
        template,
        syntax: SyntaxId::QwenXml,
        ..Setup::default()
    }
}

fn expected_args() -> Value {
    // `opts` has no declared type: JSON because it is an object (mlx-lm #1910).
    json!({"path": "a.rs", "opts": {"lines": [1, 2]}})
}

fn schema() -> Value {
    json!({"type": "object", "properties": {"path": {"type": "string"}, "opts": {}}})
}

#[tokio::test]
async fn real_templates_chat_tool_loop() {
    for (name, template) in TEMPLATES {
        let server = common::start_with(&[CALL, "Done."], setup(template)).await;
        let tools =
            json!([{"type": "function", "function": {"name": "read", "parameters": schema()}}]);
        // Thinking off on purpose (the default is on, D063): this checks the tool-call round trip.
        let body = json!({"messages": [{"role": "user", "content": "open a.rs"}], "tools": tools,
            "reasoning_effort": "none"});
        let reply = server.post("/v1/chat/completions", &body).await;
        assert_eq!(reply.status, 200, "{name}: {}", reply.text);
        let choice = &reply.json()["choices"][0];
        assert_eq!(choice["finish_reason"], "tool_calls", "{name}");
        let call = &choice["message"]["tool_calls"][0];
        assert_eq!(call["function"]["name"], "read", "{name}");
        let args = call["function"]["arguments"].as_str().unwrap();
        assert_eq!(
            serde_json::from_str::<Value>(args).unwrap(),
            expected_args(),
            "{name}"
        );
        // The call and its result render back through the template.
        let next = json!({"messages": [
            {"role": "user", "content": "open a.rs"},
            {"role": "assistant", "content": null, "tool_calls": [call]},
            {"role": "tool", "tool_call_id": call["id"], "content": "fn main() {}"}],
            "tools": tools, "reasoning_effort": "none"});
        let reply = server.post("/v1/chat/completions", &next).await;
        assert_eq!(reply.status, 200, "{name}: {}", reply.text);
        assert_eq!(
            reply.json()["choices"][0]["message"]["content"],
            "Done.",
            "{name}"
        );
    }
}

#[tokio::test]
async fn real_templates_messages_thinking_and_tool_use() {
    for (name, template) in TEMPLATES {
        // Thinking on: the generation prompt opens `<think>`, the reply closes it.
        let reply_text = format!("Plan.\n</think>\n\n{CALL}");
        let server = common::start_with(&[&reply_text], setup(template)).await;
        let body = json!({"max_tokens": 256, "thinking": {"type": "enabled", "budget_tokens": 1024},
            "tools": [{"name": "read", "input_schema": schema()}],
            "messages": [{"role": "user", "content": "open a.rs"}]});
        let reply = server.post("/v1/messages", &body).await;
        assert_eq!(reply.status, 200, "{name}: {}", reply.text);
        let body = reply.json();
        let content = body["content"].as_array().unwrap();
        assert_eq!(content[0]["type"], "thinking", "{name}: {body}");
        assert_eq!(content[0]["thinking"], "Plan.", "{name}");
        let tool = content.iter().find(|b| b["type"] == "tool_use").unwrap();
        assert_eq!(tool["name"], "read", "{name}");
        assert_eq!(tool["input"], expected_args(), "{name}");
        assert_eq!(body["stop_reason"], "tool_use", "{name}");
    }
}

#[tokio::test]
async fn real_templates_responses_function_call() {
    for (name, template) in TEMPLATES {
        let server = common::start_with(&[CALL], setup(template)).await;
        let body = json!({"input": "open a.rs",
            "tools": [{"type": "function", "name": "read", "parameters": schema()}]});
        let reply = server.post("/v1/responses", &body).await;
        assert_eq!(reply.status, 200, "{name}: {}", reply.text);
        let body = reply.json();
        let item = body["output"]
            .as_array()
            .unwrap()
            .iter()
            .find(|i| i["type"] == "function_call")
            .unwrap_or_else(|| panic!("{name}: {body}"));
        assert_eq!(item["name"], "read", "{name}");
        let args = item["arguments"].as_str().unwrap();
        assert_eq!(
            serde_json::from_str::<Value>(args).unwrap(),
            expected_args(),
            "{name}"
        );
    }
}
