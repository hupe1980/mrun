//! Responses bodies and the streaming encoder.

use super::naming::Naming;
use crate::api::handler::StreamEncode;
use crate::api::model::{ApiError, ErrorKind, Finish, GenEvent, Usage};
use crate::api::sse::{self, new_id};
use crate::chat::reasoning;
use serde_json::{Value, json};

pub fn usage(usage: Usage) -> Value {
    json!({
        "input_tokens": usage.input,
        "input_tokens_details": {"cached_tokens": usage.cached},
        "output_tokens": usage.output,
        "output_tokens_details": {"reasoning_tokens": usage.reasoning},
        "total_tokens": usage.input + usage.output,
    })
}

/// `error.code` of a failed response.
fn error_code(error: &ApiError) -> &'static str {
    match error.kind {
        ErrorKind::ContextLengthExceeded => "context_length_exceeded",
        ErrorKind::QueueTimeout | ErrorKind::NotReady | ErrorKind::Overloaded => "server_busy",
        ErrorKind::QueueFull => "rate_limit_exceeded",
        ErrorKind::InvalidRequest | ErrorKind::PayloadTooLarge => "invalid_request",
        ErrorKind::Forbidden => "forbidden",
        ErrorKind::Unauthorized => "invalid_api_key",
        ErrorKind::Internal => "server_error",
    }
}

fn truncated(finish: Finish) -> bool {
    matches!(
        finish,
        Finish::Length | Finish::ContextFull | Finish::Repetition
    )
}

fn status(complete: bool) -> &'static str {
    if complete { "completed" } else { "incomplete" }
}

fn reasoning_item(id: &str, text: &str) -> Value {
    json!({"type": "reasoning", "id": id, "summary": [],
        "content": [{"type": "reasoning_text", "text": text}],
        "encrypted_content": reasoning::encode_content(text)})
}

fn output_text(text: &str) -> Value {
    json!({"type": "output_text", "text": text, "annotations": []})
}

fn message_item(id: &str, text: &str, complete: bool) -> Value {
    json!({"type": "message", "id": id, "role": "assistant", "status": status(complete),
        "content": [output_text(text)]})
}

/// The response object of bodies and lifecycle events.
fn response_object(
    id: &str,
    created_at: u64,
    model: &str,
    status: &str,
    output: Vec<Value>,
    usage: Option<Usage>,
) -> Value {
    json!({
        "id": id,
        "object": "response",
        "created_at": created_at,
        "status": status,
        "error": null,
        "incomplete_details": if status == "incomplete" {
            json!({"reason": "max_output_tokens"})
        } else {
            Value::Null
        },
        "model": model,
        "output": output,
        "store": false,
        "usage": usage.map(self::usage),
    })
}

/// The output item being streamed.
#[derive(Debug)]
enum Open {
    Reasoning {
        id: String,
        text: String,
    },
    Message {
        id: String,
        text: String,
    },
    Call {
        id: String,
        call_id: String,
        name: String,
        args: String,
        /// Argument deltas were sent.
        streamed: bool,
        /// `ToolEnd` arrived; `output_item.done` waits for the next event, which tells whether
        /// the call was cut by the output limit.
        ended: bool,
    },
}

/// Streaming encoder: `response.created` with the headers, then item events with increasing
/// `sequence_number`s, and one terminal `response.completed`, `response.incomplete` or
/// `response.failed`. Its final response object is the non-streaming body.
#[derive(Debug)]
pub struct StreamEncoder {
    id: String,
    created_at: u64,
    model: String,
    seq: u64,
    output: Vec<Value>,
    open: Option<Open>,
    done: bool,
    naming: Naming,
    /// The terminal status and usage.
    finished: Option<(&'static str, Usage)>,
}

impl StreamEncoder {
    pub fn new(model: &str, naming: Naming) -> Self {
        Self {
            naming,
            id: new_id("resp_"),
            created_at: sse::unix_now(),
            model: model.to_string(),
            seq: 0,
            output: Vec::new(),
            open: None,
            done: false,
            finished: None,
        }
    }

    fn event(&mut self, name: &str, mut body: Value) -> String {
        body["type"] = name.into();
        body["sequence_number"] = self.seq.into();
        self.seq += 1;
        sse::event(name, &body)
    }

    fn object(&self, status: &str, usage: Option<Usage>) -> Value {
        response_object(
            &self.id,
            self.created_at,
            &self.model,
            status,
            self.output.clone(),
            usage,
        )
    }

    fn index(&self) -> usize {
        self.output.len()
    }

    /// Closes the open item; `complete: false` when the output limit cut it.
    fn close(&mut self, complete: bool, frames: &mut Vec<String>) {
        let Some(open) = self.open.take() else { return };
        let index = self.index();
        let item = match open {
            Open::Reasoning { id, text } => {
                frames.push(self.event(
                    "response.reasoning_text.done",
                    json!({"item_id": id, "output_index": index, "content_index": 0,
                        "text": text}),
                ));
                reasoning_item(&id, &text)
            }
            Open::Message { id, text } => {
                frames.push(self.event(
                    "response.output_text.done",
                    json!({"item_id": id, "output_index": index, "content_index": 0,
                        "text": text, "logprobs": []}),
                ));
                frames.push(self.event(
                    "response.content_part.done",
                    json!({"item_id": id, "output_index": index, "content_index": 0,
                        "part": output_text(&text)}),
                ));
                message_item(&id, &text, complete)
            }
            Open::Call {
                id,
                call_id,
                name,
                args,
                ended,
                ..
            } => {
                if !ended {
                    frames.push(self.event(
                        "response.function_call_arguments.done",
                        json!({"item_id": id, "output_index": index, "arguments": args}),
                    ));
                }
                self.naming
                    .item(&id, &call_id, &name, &args, status(complete && ended))
            }
        };
        frames.push(self.event(
            "response.output_item.done",
            json!({"output_index": index, "item": item}),
        ));
        self.output.push(item);
    }
}

impl StreamEncode for StreamEncoder {
    /// Frames for one event; nothing after the terminal event.
    fn encode(&mut self, event: &GenEvent) -> Vec<String> {
        if self.done {
            return Vec::new();
        }
        let mut frames = Vec::new();
        // An ended call is closed by whatever comes next, unless that is the end of the output.
        if matches!(self.open, Some(Open::Call { ended: true, .. }))
            && !matches!(event, GenEvent::Finished { .. } | GenEvent::Started { .. })
        {
            self.close(true, &mut frames);
        }
        match event {
            GenEvent::Started { .. } => {}
            GenEvent::ReasoningDelta(delta) => {
                if delta.is_empty() {
                    return frames;
                }
                if !matches!(self.open, Some(Open::Reasoning { .. })) {
                    self.close(true, &mut frames);
                    let id = new_id("rs_");
                    let index = self.index();
                    frames.push(self.event(
                        "response.output_item.added",
                        json!({"output_index": index, "item": {"type": "reasoning", "id": id,
                            "summary": [], "content": []}}),
                    ));
                    self.open = Some(Open::Reasoning {
                        id,
                        text: String::new(),
                    });
                }
                let index = self.index();
                if let Some(Open::Reasoning { id, text }) = &mut self.open {
                    text.push_str(delta);
                    let id = id.clone();
                    frames.push(self.event(
                        "response.reasoning_text.delta",
                        json!({"item_id": id, "output_index": index, "content_index": 0,
                            "delta": delta}),
                    ));
                }
            }
            GenEvent::TextDelta(delta) => {
                if delta.is_empty() {
                    return frames;
                }
                if !matches!(self.open, Some(Open::Message { .. })) {
                    self.close(true, &mut frames);
                    let id = new_id("msg_");
                    let index = self.index();
                    frames.push(self.event(
                        "response.output_item.added",
                        json!({"output_index": index, "item": {"type": "message", "id": id,
                            "role": "assistant", "status": "in_progress", "content": []}}),
                    ));
                    frames.push(self.event(
                        "response.content_part.added",
                        json!({"item_id": id, "output_index": index, "content_index": 0,
                            "part": output_text("")}),
                    ));
                    self.open = Some(Open::Message {
                        id,
                        text: String::new(),
                    });
                }
                let index = self.index();
                if let Some(Open::Message { id, text }) = &mut self.open {
                    text.push_str(delta);
                    let id = id.clone();
                    frames.push(self.event(
                        "response.output_text.delta",
                        json!({"item_id": id, "output_index": index, "content_index": 0,
                            "delta": delta, "logprobs": []}),
                    ));
                }
            }
            GenEvent::ToolStart {
                id: call_id, name, ..
            } => {
                self.close(true, &mut frames);
                let id = new_id("fc_");
                let index = self.index();
                let item = self.naming.item(&id, call_id, name, "", "in_progress");
                frames.push(self.event(
                    "response.output_item.added",
                    json!({"output_index": index, "item": item}),
                ));
                self.open = Some(Open::Call {
                    id,
                    call_id: call_id.clone(),
                    name: name.clone(),
                    args: String::new(),
                    streamed: false,
                    ended: false,
                });
            }
            GenEvent::ToolArgsDelta { json: delta, .. } => {
                let index = self.index();
                if let Some(Open::Call {
                    id, args, streamed, ..
                }) = &mut self.open
                {
                    args.push_str(delta);
                    *streamed = true;
                    let id = id.clone();
                    frames.push(self.event(
                        "response.function_call_arguments.delta",
                        json!({"item_id": id, "output_index": index, "delta": delta}),
                    ));
                }
            }
            GenEvent::ToolEnd { raw_args, .. } => {
                let index = self.index();
                if let Some(Open::Call {
                    id,
                    args,
                    streamed,
                    ended,
                    ..
                }) = &mut self.open
                {
                    // The model's raw text is the arguments, valid or not; buffered arguments
                    // go out in one delta.
                    let send_delta = !*streamed && !raw_args.is_empty();
                    *args = raw_args.clone();
                    *ended = true;
                    let id = id.clone();
                    if send_delta {
                        frames.push(self.event(
                            "response.function_call_arguments.delta",
                            json!({"item_id": id, "output_index": index, "delta": raw_args}),
                        ));
                    }
                    frames.push(self.event(
                        "response.function_call_arguments.done",
                        json!({"item_id": id, "output_index": index, "arguments": raw_args}),
                    ));
                }
            }
            GenEvent::Finished { reason, usage, .. } => {
                self.done = true;
                let complete = !truncated(*reason);
                self.close(complete, &mut frames);
                let (name, status) = if complete {
                    ("response.completed", "completed")
                } else {
                    ("response.incomplete", "incomplete")
                };
                self.finished = Some((status, *usage));
                let response = self.object(status, Some(*usage));
                frames.push(self.event(name, json!({"response": response})));
            }
            GenEvent::Failed(error) => {
                self.done = true;
                let mut response = self.object("failed", None);
                response["error"] = json!({"code": error_code(error), "message": error.message});
                frames.push(self.event("response.failed", json!({"response": response})));
            }
        }
        frames
    }

    /// Codex's SSE client drops comments, so the keepalive is a real event; its snapshot
    /// carries no output (the items so far are not re-sent).
    fn keepalive(&mut self) -> String {
        let response = response_object(
            &self.id,
            self.created_at,
            &self.model,
            "in_progress",
            Vec::new(),
            None,
        );
        self.event("response.in_progress", json!({"response": response}))
    }

    fn snapshot(&self) -> Value {
        let (status, usage) = self.finished.unwrap_or(("completed", Usage::default()));
        self.object(status, Some(usage))
    }

    fn open(&mut self) -> Vec<String> {
        let response = self.object("in_progress", None);
        vec![
            self.event("response.created", json!({"response": response.clone()})),
            self.event("response.in_progress", json!({"response": response})),
        ]
    }
}
