//! OpenAI Responses, stateless subset (`POST /v1/responses`): decoder and encoders (002 research
//! §1c, §2 Codex).
//!
//! Nothing is stored: `previous_response_id`, `conversation` and `item_reference` are refused,
//! and reasoning survives a turn only through `encrypted_content` (FR-016).

mod decode;
mod encode;
mod naming;

pub use decode::{KNOWN, decode};
pub use encode::{StreamEncoder, usage};
pub use naming::{Naming, TOOL_SEARCH};

use crate::api::chat_completions::error_body;
use crate::api::decode::DecodeOptions;
use crate::api::handler::Api;
use crate::api::model::{ApiError, ApiKind, ChatRequest, ErrorKind};
use serde_json::Value;

/// `POST /v1/responses`
#[derive(Debug)]
pub struct Responses;

impl Api for Responses {
    const KIND: ApiKind = ApiKind::Responses;
    type Encoder = StreamEncoder;

    fn decode(body: &Value, options: DecodeOptions) -> Result<ChatRequest, ApiError> {
        decode(body, options)
    }

    fn encoder(model: &str, request: &ChatRequest) -> StreamEncoder {
        StreamEncoder::new(model, Naming::of(request))
    }

    fn error_body(error: &ApiError) -> Value {
        error_body(error)
    }

    /// Codex compacts only on the in-stream form of this error (contract: status codes).
    fn in_stream(error: &ApiError) -> bool {
        error.kind == ErrorKind::ContextLengthExceeded
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::decode::DecodeOptions;
    use crate::api::handler::StreamEncode;
    use crate::api::model::{
        ApiError, ApiKind, AssistantPart, ErrorKind, Finish, GenEvent, Message, ResponseFormat,
        ToolChoice, ToolResult, Usage,
    };
    use crate::chat::reasoning;
    use serde_json::json;

    fn decode_ok(body: Value) -> ChatRequest {
        decode(&body, DecodeOptions::default()).unwrap()
    }

    fn decode_err(body: Value) -> ApiError {
        decode(&body, DecodeOptions::default()).unwrap_err()
    }

    #[test]
    fn repetition_penalty_is_read_and_checked() {
        let body = |penalty: Value| json!({"input": "hi", "repetition_penalty": penalty});
        let request = decode_ok(body(json!(1.2)));
        assert_eq!(request.sampling.repetition_penalty, Some(1.2));
        assert!(KNOWN.contains(&"repetition_penalty"));
        for bad in [json!(0), json!(-0.5)] {
            let error = decode_ok(body(bad))
                .sampling
                .resolve_valid(mrun_engine::Sampling::GREEDY)
                .unwrap_err();
            assert_eq!(error.param.as_deref(), Some("repetition_penalty"));
            assert_eq!(error.status(ApiKind::Responses).as_u16(), 400);
        }
        let error = decode_err(body(json!("high")));
        assert_eq!(error.param.as_deref(), Some("repetition_penalty"));
    }

    /// `(event name, data)` of each frame.
    fn events(frames: &[String]) -> Vec<(String, Value)> {
        frames
            .iter()
            .map(|f| {
                let mut lines = f.lines();
                let name = lines.next().unwrap().strip_prefix("event: ").unwrap();
                let data = lines.next().unwrap().strip_prefix("data: ").unwrap();
                (name.to_string(), serde_json::from_str(data).unwrap())
            })
            .collect()
    }

    fn usage3() -> Usage {
        Usage {
            input: 10,
            cached: 4,
            output: 3,
            reasoning: 1,
            speculation: None,
        }
    }

    #[test]
    fn decode_maps_items_into_turns() {
        let encrypted = reasoning::encode_content("plan");
        let request = decode_ok(json!({
            "model": "gpt-x",
            "instructions": "Base.",
            "input": [
                {"type": "message", "role": "developer", "content": [
                    {"type": "input_text", "text": "Dev."}]},
                {"role": "user", "content": "hi"},
                {"type": "reasoning", "id": "rs_1", "summary": [], "encrypted_content": encrypted},
                {"type": "message", "role": "assistant", "content": [
                    {"type": "output_text", "text": "Looking."}]},
                {"type": "function_call", "id": "fc_1", "call_id": "call_1", "name": "shell",
                 "arguments": "{\"cmd\":\"ls\"}"},
                {"type": "function_call_output", "call_id": "call_1", "output": "a.rs"},
                {"type": "reasoning", "summary": [], "encrypted_content": "gAAAA-foreign"},
                {"type": "message", "role": "assistant", "content": "Done."},
                {"type": "message", "role": "system", "content": "mid"},
            ],
            "store": true, "include": ["reasoning.encrypted_content"],
            "prompt_cache_key": "k", "max_output_tokens": 50, "temperature": 0.2,
            "reasoning": {"effort": "high", "summary": "auto"},
            "text": {"format": {"type": "json_schema", "name": "x",
                "schema": {"type": "object"}, "strict": true}},
        }));
        assert_eq!(request.api, ApiKind::Responses);
        assert_eq!(request.system, vec!["Base.", "Dev."]);
        assert_eq!(
            request.messages,
            vec![
                Message::user_text("hi"),
                Message::Assistant(vec![
                    AssistantPart::Reasoning("plan".into()),
                    AssistantPart::Text("Looking.".into()),
                    AssistantPart::ToolCall {
                        id: "call_1".into(),
                        name: "shell".into(),
                        raw_args: "{\"cmd\":\"ls\"}".into(),
                    },
                ]),
                Message::Tool(ToolResult {
                    call_id: "call_1".into(),
                    content: "a.rs".into(),
                    is_error: false,
                    images_omitted: 0,
                }),
                Message::Assistant(vec![AssistantPart::Text("Done.".into())]),
                Message::System("mid".into()),
            ]
        );
        assert_eq!(request.max_output_tokens, Some(50));
        assert!(request.reasoning.is_on());
        assert!(matches!(
            request.response_format,
            ResponseFormat::JsonSchema { strict: true, .. }
        ));
    }

    #[test]
    fn string_input_and_no_limit() {
        let request = decode_ok(json!({"input": "hello"}));
        assert_eq!(request.messages, vec![Message::user_text("hello")]);
        assert_eq!(request.max_output_tokens, None, "FR-015: capacity − prompt");
        assert!(!request.stream);
    }

    #[test]
    fn resp_reasoning_follows_the_model_default_unless_turned_off() {
        // D063: no effort means the model's default (thinking on); only none/minimal turn it off.
        for (reasoning, on) in [
            (json!({"summary": "auto"}), true),
            (json!(null), true),
            (json!({"effort": "high"}), true),
            (json!({"effort": "minimal"}), false),
            (json!({"effort": "none"}), false),
        ] {
            let request = decode_ok(json!({"input": "x", "reasoning": reasoning}));
            assert_eq!(request.reasoning.is_on(), on, "{reasoning}");
        }
        let request = decode_ok(json!({"input": "x"}));
        assert!(request.reasoning.is_on(), "no reasoning object");
    }

    #[test]
    fn resp_previous_response_id_and_item_reference_rejected() {
        let error = decode_err(json!({"input": "x", "previous_response_id": "resp_1"}));
        assert_eq!(error.param.as_deref(), Some("previous_response_id"));
        let error = decode_err(json!({"input": [{"role": "user", "content": "x"},
            {"type": "item_reference", "id": "msg_1"}]}));
        assert_eq!(error.param.as_deref(), Some("input[1]"));
    }

    #[test]
    fn user_media_rejected_with_path() {
        for kind in ["input_image", "input_file"] {
            let error = decode_err(json!({"input": [{"role": "user", "content": [
                {"type": "input_text", "text": "see"}, {"type": kind}]}]}));
            assert_eq!(
                error.param.as_deref(),
                Some("input[0].content[1]"),
                "{kind}"
            );
        }
    }

    #[test]
    fn resp_function_call_output_array_and_images() {
        let body = json!({"input": [
            {"role": "user", "content": "look"},
            {"type": "function_call", "call_id": "c", "name": "view", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "c", "output": [
                {"type": "input_text", "text": "shot: "},
                {"type": "input_image", "image_url": "data:image/png;base64,AA"}]}]});
        let request = decode_ok(body.clone());
        let Message::Tool(result) = &request.messages[2] else {
            panic!("{:?}", request.messages)
        };
        assert_eq!(
            result.content,
            format!("shot: {}", crate::api::decode::IMAGE_PLACEHOLDER)
        );
        assert_eq!(result.images_omitted, 1);
        // resp_reject_tool_images_flag
        let error = decode(
            &body,
            DecodeOptions {
                reject_tool_images: true,
            },
        )
        .unwrap_err();
        assert_eq!(error.param.as_deref(), Some("input[2].output[1]"));
    }

    #[test]
    fn tools_function_web_search_custom() {
        let request = decode_ok(json!({"input": "x", "tools": [
            {"type": "function", "name": "shell", "description": "run",
             "parameters": {"type": "object", "properties": {}}, "strict": false},
            {"type": "web_search"}, {"type": "web_search_preview"}],
            "tool_choice": {"type": "function", "name": "shell"}, "parallel_tool_calls": false}));
        assert_eq!(request.tools.len(), 1);
        assert_eq!(request.tools[0].name, "shell");
        assert_eq!(request.tool_choice, ToolChoice::Named("shell".into()));
        assert!(!request.parallel_tool_calls);
        for kind in ["custom", "file_search"] {
            let error = decode_err(json!({"input": "x", "tools": [{"type": kind, "name": "t"}]}));
            assert_eq!(error.param.as_deref(), Some("tools[0].type"), "{kind}");
            assert!(error.message.contains(kind));
        }
    }

    #[test]
    fn resp_namespace_roundtrip() {
        let body = json!({"input": [
            {"role": "user", "content": "list issues"},
            {"type": "function_call", "call_id": "c1", "namespace": "mcp__github",
             "name": "list_issues", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "c1", "output": "[]"}],
            "tools": [{"type": "namespace", "name": "mcp__github", "description": "GitHub",
                "tools": [{"type": "function", "name": "list_issues",
                    "parameters": {"type": "object", "properties": {}}}]}]});
        let request = decode_ok(body);
        assert_eq!(request.tools[0].name, "mcp__github__list_issues");
        let Message::Assistant(parts) = &request.messages[1] else {
            panic!()
        };
        assert!(
            matches!(&parts[0], AssistantPart::ToolCall { name, .. } if name == "mcp__github__list_issues")
        );
        // Calls go back split into namespace and name, streamed and final.
        let naming = Naming::of(&request);
        let item = naming.item(
            "fc_1",
            "call_2",
            "mcp__github__list_issues",
            "{}",
            "completed",
        );
        assert_eq!(
            (item["namespace"].as_str(), item["name"].as_str()),
            (Some("mcp__github"), Some("list_issues"))
        );
        let mut encoder = StreamEncoder::new("m", naming);
        let frames = encoder.encode(&GenEvent::ToolStart {
            idx: 0,
            id: "call_2".into(),
            name: "mcp__github__list_issues".into(),
        });
        let added = &events(&frames)[0].1;
        assert_eq!(added["item"]["namespace"], "mcp__github");
        assert_eq!(added["item"]["name"], "list_issues");
    }

    #[test]
    fn tool_search_call_and_output_expand_deferred_tools() {
        let body = json!({"input": [
            {"role": "user", "content": "find a tool"},
            {"type": "tool_search_call", "call_id": "s1", "execution": "client",
             "arguments": {"query": "issues"}},
            {"type": "tool_search_output", "call_id": "s1", "execution": "client",
             "tools": [{"type": "namespace", "name": "mcp__github", "tools": [
                {"type": "function", "name": "list_issues", "defer_loading": true,
                 "parameters": {"type": "object", "properties": {}}}]}]}],
            "tools": [{"type": "tool_search", "execution": "client"},
                {"type": "function", "name": "shell", "parameters": {"type": "object"}}]});
        let request = decode_ok(body);
        let found = request.tool("mcp__github__list_issues").unwrap();
        assert!(found.deferred, "callable but never in the tools region");
        let Message::Tool(result) = &request.messages[2] else {
            panic!()
        };
        assert!(
            result
                .content
                .contains("\"name\":\"mcp__github__list_issues\""),
            "{}",
            result.content
        );
        let naming = Naming::of(&request);
        let item = naming.item("ts_1", "s2", TOOL_SEARCH, "{\"query\":\"pr\"}", "completed");
        assert_eq!(item["type"], "tool_search_call");
        assert_eq!(item["arguments"], json!({"query": "pr"}));
        assert_eq!(item["execution"], "client");
    }

    #[test]
    fn non_streaming_items_and_usage() {
        let mut encoder = StreamEncoder::new("m", Naming::default());
        for event in [
            GenEvent::ReasoningDelta("hm".into()),
            GenEvent::TextDelta("Calling.".into()),
            GenEvent::ToolStart {
                idx: 0,
                id: "call_9".into(),
                name: "shell".into(),
            },
            GenEvent::ToolEnd {
                idx: 0,
                raw_args: "{\"cmd\":\"ls\"}".into(),
                valid: true,
            },
            GenEvent::Finished {
                reason: Finish::ToolCalls,
                stop_seq: None,
                usage: usage3(),
            },
        ] {
            encoder.encode(&event);
        }
        let body = encoder.snapshot();
        assert_eq!(body["object"], "response");
        assert_eq!(body["status"], "completed");
        assert!(body["incomplete_details"].is_null());
        let output = body["output"].as_array().unwrap();
        assert_eq!(output[0]["type"], "reasoning");
        assert_eq!(output[0]["summary"], json!([]));
        assert_eq!(
            reasoning::decode_content(output[0]["encrypted_content"].as_str().unwrap()).unwrap(),
            "hm"
        );
        assert_eq!(output[1]["content"][0]["text"], "Calling.");
        assert_eq!(output[1]["content"][0]["annotations"], json!([]));
        assert_eq!(output[2]["call_id"], "call_9");
        assert!(output[2]["id"].as_str().unwrap().starts_with("fc_"));
        assert_eq!(output[2]["status"], "completed");
        assert_eq!(body["usage"]["input_tokens_details"]["cached_tokens"], 4);
        assert_eq!(
            body["usage"]["output_tokens_details"]["reasoning_tokens"],
            1
        );
        assert_eq!(body["usage"]["total_tokens"], 13);
    }

    #[test]
    fn truncated_call_is_incomplete() {
        let mut encoder = StreamEncoder::new("m", Naming::default());
        for event in [
            GenEvent::ToolStart {
                idx: 0,
                id: "call_1".into(),
                name: "shell".into(),
            },
            GenEvent::ToolEnd {
                idx: 0,
                raw_args: "{\"cmd\":".into(),
                valid: false,
            },
            GenEvent::Finished {
                reason: Finish::Length,
                stop_seq: None,
                usage: usage3(),
            },
        ] {
            encoder.encode(&event);
        }
        let body = encoder.snapshot();
        assert_eq!(body["status"], "incomplete");
        assert_eq!(body["incomplete_details"]["reason"], "max_output_tokens");
        assert_eq!(body["output"][0]["status"], "incomplete");
        assert_eq!(body["output"][0]["arguments"], "{\"cmd\":");

        let mut encoder = StreamEncoder::new("m", Naming::default());
        let mut frames = Vec::new();
        for event in [
            GenEvent::ToolStart {
                idx: 0,
                id: "call_1".into(),
                name: "shell".into(),
            },
            GenEvent::ToolEnd {
                idx: 0,
                raw_args: "{\"cmd\":".into(),
                valid: false,
            },
            GenEvent::Finished {
                reason: Finish::Length,
                stop_seq: None,
                usage: usage3(),
            },
        ] {
            frames.extend(encoder.encode(&event));
        }
        let events = events(&frames);
        let done = events
            .iter()
            .find(|(n, _)| n == "response.output_item.done")
            .unwrap();
        assert_eq!(done.1["item"]["status"], "incomplete");
        let (name, last) = events.last().unwrap();
        assert_eq!(name, "response.incomplete");
        assert_eq!(last["response"]["output"][0]["status"], "incomplete");
    }

    #[test]
    fn stream_event_order_and_sequence_numbers() {
        let mut encoder = StreamEncoder::new("m", Naming::default());
        let mut frames = encoder.open();
        frames.push(encoder.keepalive());
        frames.extend(encoder.encode(&GenEvent::TextDelta("x".into())));
        // resp_in_progress_keepalive_has_no_output: a keepalive after output carries none.
        let keepalive = encoder.keepalive();
        let data: Value = serde_json::from_str(
            keepalive
                .lines()
                .nth(1)
                .unwrap()
                .trim_start_matches("data: "),
        )
        .unwrap();
        assert_eq!(data["response"]["output"], json!([]));
        let mut encoder = StreamEncoder::new("m", Naming::default());
        let mut frames = encoder.open();
        frames.push(encoder.keepalive());
        for event in [
            GenEvent::Started {
                input: 10,
                cached: 4,
                max_tokens: 100,
                miss: None,
            },
            GenEvent::ReasoningDelta("a".into()),
            GenEvent::ReasoningDelta("b".into()),
            GenEvent::TextDelta("Hi".into()),
            GenEvent::ToolStart {
                idx: 0,
                id: "call_1".into(),
                name: "shell".into(),
            },
            GenEvent::ToolArgsDelta {
                idx: 0,
                json: "{}".into(),
            },
            GenEvent::ToolEnd {
                idx: 0,
                raw_args: "{}".into(),
                valid: true,
            },
            GenEvent::ToolStart {
                idx: 1,
                id: "call_2".into(),
                name: "shell".into(),
            },
            GenEvent::ToolEnd {
                idx: 1,
                raw_args: "not json".into(),
                valid: false,
            },
            GenEvent::Finished {
                reason: Finish::ToolCalls,
                stop_seq: None,
                usage: usage3(),
            },
            GenEvent::TextDelta("late".into()),
        ] {
            frames.extend(encoder.encode(&event));
        }
        let events = events(&frames);
        let names: Vec<&str> = events.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(
            names,
            [
                "response.created",
                "response.in_progress",
                "response.in_progress",
                "response.output_item.added",
                "response.reasoning_text.delta",
                "response.reasoning_text.delta",
                "response.reasoning_text.done",
                "response.output_item.done",
                "response.output_item.added",
                "response.content_part.added",
                "response.output_text.delta",
                "response.output_text.done",
                "response.content_part.done",
                "response.output_item.done",
                "response.output_item.added",
                "response.function_call_arguments.delta",
                "response.function_call_arguments.done",
                "response.output_item.done",
                "response.output_item.added",
                "response.function_call_arguments.delta",
                "response.function_call_arguments.done",
                "response.output_item.done",
                "response.completed",
            ]
        );
        for (i, (name, data)) in events.iter().enumerate() {
            assert_eq!(data["type"], name.as_str());
            assert_eq!(data["sequence_number"], i as u64);
        }
        // resp_function_call_item_done_parsable_by_codex: complete items with call_id.
        let calls: Vec<&Value> = events
            .iter()
            .filter(|(n, d)| {
                n == "response.output_item.done" && d["item"]["type"] == "function_call"
            })
            .map(|(_, d)| &d["item"])
            .collect();
        assert_eq!(calls[0]["call_id"], "call_1");
        assert_eq!(calls[0]["arguments"], "{}");
        assert_eq!(calls[1]["arguments"], "not json", "invalid args raw");
        let completed = &events.last().unwrap().1["response"];
        assert_eq!(completed["status"], "completed");
        assert_eq!(completed["output"].as_array().unwrap().len(), 4);
        assert_eq!(completed["usage"]["input_tokens"], 10);
        assert_eq!(completed["output"][0]["summary"], json!([]));
    }

    #[test]
    fn resp_stream_failed_codes() {
        for (kind, code) in [
            (ErrorKind::ContextLengthExceeded, "context_length_exceeded"),
            (ErrorKind::QueueTimeout, "server_busy"),
            (ErrorKind::NotReady, "server_busy"),
        ] {
            let mut encoder = StreamEncoder::new("m", Naming::default());
            let frames = encoder.encode(&GenEvent::Failed(ApiError::new(kind, "x")));
            let (name, data) = &events(&frames)[0];
            assert_eq!(name, "response.failed");
            assert_eq!(data["response"]["status"], "failed");
            assert_eq!(data["response"]["error"]["code"], code);
            assert!(encoder.encode(&GenEvent::TextDelta("x".into())).is_empty());
        }
    }
}
