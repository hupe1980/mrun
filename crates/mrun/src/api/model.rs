//! The one internal request and event model behind the three APIs (002 data model).
//!
//! Decoders turn each wire format into a [`ChatRequest`]; the generation worker turns engine
//! events into [`GenEvent`]s; encoders fold those into each API's native body or SSE stream.

use crate::cache::miss::MissReport;
use axum::http::StatusCode;
use mrun_engine::Sampling;
use serde_json::Value;

/// Which API a request arrived on.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ApiKind {
    Chat,
    Messages,
    Responses,
}

impl ApiKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Chat => "chat",
            Self::Messages => "messages",
            Self::Responses => "responses",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ChatRequest {
    pub api: ApiKind,
    /// Requested model id; informational while one model is served (FR-018).
    pub model: String,
    /// Leading system, developer and instructions text, in order.
    pub system: Vec<String>,
    pub messages: Vec<Message>,
    pub tools: Vec<ToolDef>,
    pub tool_choice: ToolChoice,
    pub parallel_tool_calls: bool,
    pub response_format: ResponseFormat,
    /// Sampling the request sets; the rest comes from the model's generation config.
    pub sampling: SamplingParams,
    /// `None`: the API's default (Chat: server default; Responses: remaining capacity, FR-019).
    pub max_output_tokens: Option<u32>,
    pub stop: Vec<String>,
    pub reasoning: ReasoningCtl,
    /// Anthropic: the trailing assistant message is continued, not answered.
    pub prefill: bool,
    pub stream: bool,
    /// Chat `stream_options.include_usage`.
    pub include_usage: bool,
    /// `x-mrun-ignore-eos: on` (FR-031): generate to the output limit.
    pub ignore_eos: bool,
}

impl ChatRequest {
    /// An empty request for `api` with neutral defaults, filled in by the decoders.
    pub fn new(api: ApiKind) -> Self {
        Self {
            api,
            model: String::new(),
            system: Vec::new(),
            messages: Vec::new(),
            tools: Vec::new(),
            tool_choice: ToolChoice::Auto,
            parallel_tool_calls: true,
            response_format: ResponseFormat::Text,
            sampling: SamplingParams::default(),
            max_output_tokens: None,
            stop: Vec::new(),
            reasoning: ReasoningCtl::Off,
            prefill: false,
            stream: false,
            include_usage: false,
            ignore_eos: false,
        }
    }

    pub fn tool(&self, name: &str) -> Option<&ToolDef> {
        self.tools.iter().find(|t| t.name == name)
    }
}

/// Sampling fields as sent; unset ones fall back to the model's defaults.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SamplingParams {
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<u32>,
    pub min_p: Option<f32>,
    pub seed: Option<u64>,
}

impl SamplingParams {
    pub fn resolve(&self, defaults: Sampling) -> Sampling {
        Sampling {
            temperature: self.temperature.unwrap_or(defaults.temperature),
            top_p: self.top_p.unwrap_or(defaults.top_p),
            top_k: self.top_k.unwrap_or(defaults.top_k),
            min_p: self.min_p.unwrap_or(defaults.min_p),
            seed: self.seed.or(defaults.seed),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Message {
    User(Vec<UserPart>),
    Assistant(Vec<AssistantPart>),
    Tool(ToolResult),
    /// A system or developer message after the first non-system message (FR-029).
    System(String),
}

impl Message {
    pub fn user_text(text: impl Into<String>) -> Self {
        Self::User(vec![UserPart::Text(text.into())])
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum UserPart {
    Text(String),
}

#[derive(Clone, Debug, PartialEq)]
pub enum AssistantPart {
    Text(String),
    /// Reasoning restored from a signature or `encrypted_content` (FR-016).
    Reasoning(String),
    /// Arguments are kept verbatim as the model or client wrote them.
    ToolCall {
        id: String,
        name: String,
        raw_args: String,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub struct ToolResult {
    pub call_id: String,
    pub content: String,
    pub is_error: bool,
    /// Images in the result that were replaced by a placeholder (FR-020).
    pub images_omitted: u32,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct ToolDef {
    /// The name the model sees; for a namespaced tool `<namespace>__<tool>` (FR-027).
    pub name: String,
    pub description: String,
    /// JSON Schema of the arguments (key order preserved).
    pub schema: Value,
    pub strict: bool,
    /// `defer_loading` (FR-028): not rendered in the tools region; its definition appears
    /// inline where a tool search returns it.
    pub deferred: bool,
    /// Responses namespace the tool came from, restored on calls.
    pub namespace: Option<String>,
}

/// Separator of flattened namespace tool names.
pub const NAMESPACE_SEP: &str = "__";

impl ToolDef {
    /// The definition as tools regions render it (OpenAI function shape).
    pub fn definition(&self) -> Value {
        serde_json::json!({"type": "function", "function": {
            "name": self.name, "description": self.description, "parameters": self.schema}})
    }

    /// `(namespace, tool)` for a namespaced tool's call.
    pub fn split_name(&self) -> Option<(&str, &str)> {
        let namespace = self.namespace.as_deref()?;
        let tool = self
            .name
            .strip_prefix(namespace)?
            .strip_prefix(NAMESPACE_SEP)?;
        Some((namespace, tool))
    }

    /// JSON Schema type of a top-level parameter, if declared.
    pub fn param_type(&self, param: &str) -> Option<&str> {
        let ty = self.schema.get("properties")?.get(param)?.get("type")?;
        match ty {
            Value::String(s) => Some(s),
            // `["string", "null"]`: the first non-null type.
            Value::Array(types) => types
                .iter()
                .filter_map(Value::as_str)
                .find(|t| *t != "null"),
            _ => None,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ToolChoice {
    Auto,
    None,
    Required,
    Named(String),
}

#[derive(Clone, Debug, PartialEq)]
pub enum ResponseFormat {
    Text,
    JsonObject,
    JsonSchema {
        name: String,
        schema: Value,
        strict: bool,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Effort {
    Low,
    Medium,
    High,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReasoningCtl {
    Off,
    On {
        /// Anthropic `budget_tokens`: the thinking block is closed when reached.
        budget: Option<u32>,
        effort: Option<Effort>,
    },
}

impl ReasoningCtl {
    pub fn is_on(self) -> bool {
        matches!(self, Self::On { .. })
    }
}

/// How a generation ended, as the encoders report it.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Finish {
    Stop,
    StopSequence,
    Length,
    /// The output limit was clamped by the context capacity and reached.
    ContextFull,
    ToolCalls,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Usage {
    /// All prompt tokens, including cached ones.
    pub input: u32,
    pub cached: u32,
    pub output: u32,
    pub reasoning: u32,
}

/// In-band warnings (the only three in 0.1); every other condition is logged.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Warning {
    ToolArgsInvalid,
    MaxTokensClamped,
    ImageOmitted,
}

impl Warning {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::ToolArgsInvalid => "tool-args-invalid",
            Self::MaxTokensClamped => "max-tokens-clamped",
            Self::ImageOmitted => "image-omitted",
        }
    }
}

/// Events from the generation worker to the encoders; exactly one terminal event
/// (`Finished` or `Failed`) ends each stream.
#[derive(Clone, Debug, PartialEq)]
pub enum GenEvent {
    Started {
        input: u32,
        cached: u32,
        max_tokens: u32,
        /// The cache's miss report (003), when the server keeps a cache.
        miss: Option<MissReport>,
    },
    ReasoningDelta(String),
    TextDelta(String),
    ToolStart {
        idx: u32,
        id: String,
        name: String,
    },
    ToolArgsDelta {
        idx: u32,
        json: String,
    },
    /// `valid`: the arguments parse as a JSON object (and match the schema when strict).
    ToolEnd {
        idx: u32,
        raw_args: String,
        valid: bool,
    },
    Finished {
        reason: Finish,
        stop_seq: Option<String>,
        usage: Usage,
    },
    Failed(ApiError),
}

impl GenEvent {
    pub fn is_terminal(&self) -> bool {
        matches!(self, Self::Finished { .. } | Self::Failed(_))
    }
}

/// What went wrong, independent of the API that reports it.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ErrorKind {
    InvalidRequest,
    ContextLengthExceeded,
    QueueFull,
    QueueTimeout,
    Overloaded,
    NotReady,
    Forbidden,
    PayloadTooLarge,
    Internal,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ApiError {
    pub kind: ErrorKind,
    pub message: String,
    /// JSON path of the offending field, e.g. `messages[2].content[0]`.
    pub param: Option<String>,
    /// Seconds for `retry-after`, clamped to 1–60.
    pub retry_after: Option<u64>,
}

impl ApiError {
    pub fn new(kind: ErrorKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
            param: None,
            retry_after: None,
        }
    }

    pub fn invalid(message: impl Into<String>, param: impl Into<String>) -> Self {
        Self {
            param: Some(param.into()),
            ..Self::new(ErrorKind::InvalidRequest, message)
        }
    }

    pub fn with_retry_after(mut self, seconds: u64) -> Self {
        self.retry_after = Some(seconds.clamp(1, 60));
        self
    }

    /// HTTP status before headers are sent (contract: status codes).
    pub fn status(&self, api: ApiKind) -> StatusCode {
        match (self.kind, api) {
            (ErrorKind::InvalidRequest | ErrorKind::ContextLengthExceeded, _) => {
                StatusCode::BAD_REQUEST
            }
            (ErrorKind::QueueFull, ApiKind::Messages) => {
                StatusCode::from_u16(529).expect("529 is a valid status")
            }
            (ErrorKind::QueueFull, _) => StatusCode::TOO_MANY_REQUESTS,
            (ErrorKind::QueueTimeout, ApiKind::Messages) => StatusCode::GATEWAY_TIMEOUT,
            (ErrorKind::QueueTimeout | ErrorKind::Overloaded | ErrorKind::NotReady, _) => {
                StatusCode::SERVICE_UNAVAILABLE
            }
            (ErrorKind::Forbidden, _) => StatusCode::FORBIDDEN,
            (ErrorKind::PayloadTooLarge, _) => StatusCode::PAYLOAD_TOO_LARGE,
            (ErrorKind::Internal, _) => StatusCode::INTERNAL_SERVER_ERROR,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn param_types_come_from_the_schema() {
        let tool = ToolDef {
            name: "edit".into(),
            description: String::new(),
            schema: json!({"type": "object", "properties": {
                "path": {"type": "string"},
                "line": {"type": ["integer", "null"]},
                "opts": {}
            }}),
            strict: false,
            ..Default::default()
        };
        assert_eq!(tool.param_type("path"), Some("string"));
        assert_eq!(tool.param_type("line"), Some("integer"));
        assert_eq!(tool.param_type("opts"), None);
        assert_eq!(tool.param_type("missing"), None);
    }

    #[test]
    fn statuses_follow_the_contract() {
        let full = ApiError::new(ErrorKind::QueueFull, "queue full").with_retry_after(300);
        assert_eq!(full.retry_after, Some(60));
        assert_eq!(full.status(ApiKind::Messages).as_u16(), 529);
        assert_eq!(full.status(ApiKind::Chat).as_u16(), 429);
        let late = ApiError::new(ErrorKind::QueueTimeout, "deadline");
        assert_eq!(late.status(ApiKind::Messages).as_u16(), 504);
        assert_eq!(late.status(ApiKind::Responses).as_u16(), 503);
    }
}
