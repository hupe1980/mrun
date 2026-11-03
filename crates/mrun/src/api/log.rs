//! Server-log lines for what a request asks and mrun does not do (002 FR-022, D055): exactly two
//! kinds, each carrying the field's JSON path and never content.
//!
//! - `ignored-field {path}`: accepted, changes nothing (unknown fields, `output_config.effort`,
//!   `output_config.task_budget`);
//! - `unenforced {path}`: asked for, not done (`strict` under `auto`, uncompilable schemas,
//!   `budget_tokens`, thinking on a family without it, undeclared tools, raw history arguments,
//!   rewritten content).

use crate::api::model::ApiKind;

/// One of the two log kinds.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Kind {
    IgnoredField,
    Unenforced,
}

impl Kind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::IgnoredField => "ignored-field",
            Self::Unenforced => "unenforced",
        }
    }
}

/// The log line without the `mrun: ` prefix.
pub fn line(kind: Kind, path: &str) -> String {
    format!("{} {path}", kind.as_str())
}

/// Writes one line to the server log.
pub fn write(kind: Kind, path: &str) {
    eprintln!("mrun: {}", line(kind, path));
}

pub fn ignored_field(path: &str) {
    write(Kind::IgnoredField, path);
}

pub fn unenforced(path: &str) {
    write(Kind::Unenforced, path);
}

/// The request field that turns thinking on.
pub fn thinking_field(api: ApiKind) -> &'static str {
    match api {
        ApiKind::Chat => "reasoning_effort",
        ApiKind::Messages => "thinking",
        ApiKind::Responses => "reasoning.effort",
    }
}

/// The request array that holds the conversation.
pub fn messages_field(api: ApiKind) -> &'static str {
    match api {
        ApiKind::Chat | ApiKind::Messages => "messages",
        ApiKind::Responses => "input",
    }
}

/// Where a history tool call's arguments are in API message `index`.
pub fn history_args_field(api: ApiKind, index: u32) -> String {
    match api {
        ApiKind::Chat => format!("messages[{index}].tool_calls"),
        ApiKind::Messages => format!("messages[{index}].content"),
        ApiKind::Responses => format!("input[{index}].arguments"),
    }
}

/// The response-format field.
pub fn format_field(api: ApiKind) -> &'static str {
    match api {
        ApiKind::Responses => "text.format",
        ApiKind::Chat | ApiKind::Messages => "response_format",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn xapi_two_log_kinds_carry_the_path() {
        assert_eq!(
            line(Kind::IgnoredField, "output_config.effort"),
            "ignored-field output_config.effort"
        );
        assert_eq!(line(Kind::Unenforced, "tools"), "unenforced tools");
        assert_eq!(thinking_field(ApiKind::Responses), "reasoning.effort");
        assert_eq!(
            history_args_field(ApiKind::Chat, 3),
            "messages[3].tool_calls"
        );
    }
}
