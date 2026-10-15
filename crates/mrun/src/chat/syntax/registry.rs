//! Syntax registry (FR-032, data-model "Syntax"): per [`SyntaxId`] the parser, the template
//! check, message-start markers, the `tool_choice: none` opener, argument kind, history rule and
//! thinking control. No syntax-specific logic lives outside its parser and its entry here.

use super::harmony::{self, HarmonyParser};
use super::hermes::{self, HermesParser};
use super::qwen_xml::QwenXmlParser;
use super::{ParserConfig, SyntaxId, SyntaxParser};

/// How a syntax carries tool-call arguments (FR-004).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ArgKind {
    /// The model's own JSON text: streamed as it arrives (Chat, Responses).
    Raw,
    /// JSON synthesized from another format: buffered per call.
    Synthesized,
}

/// Where reasoning appears in the output.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReasoningMarkers {
    /// `open…close` tags; the initial state comes from the prompt tail.
    Tags {
        open: &'static str,
        close: &'static str,
    },
    /// A named channel (Harmony `analysis`).
    Channel(&'static str),
}

/// The `tool_choice: none` bias: these markers get −inf logits.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NoneBias {
    pub markers: &'static [&'static str],
    /// Only while the parser is inside a message header (Harmony ` to`).
    pub header_only: bool,
}

/// How history arguments are passed to the template (T004b).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HistoryArgs {
    /// Strings rendered verbatim (byte-stable re-render).
    Verbatim,
    /// Parsed to an object (the template re-serializes it); a non-object is rendered raw.
    Object,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HistoryRule {
    pub args: HistoryArgs,
    /// The template renders one call per assistant message: split multi-call messages.
    pub one_call_per_message: bool,
}

/// How thinking is switched (FR-035).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ThinkingControl {
    /// `enable_thinking` template variable (Qwen); Qwen3-Coder ignores it.
    EnableThinking,
    /// `reasoning_effort` (`low|medium|high`); always reasons.
    ReasoningEffort,
}

/// One syntax's registry entry.
#[derive(Debug)]
pub struct ChatSyntax {
    pub id: SyntaxId,
    /// Name in logs and errors.
    pub name: &'static str,
    /// Text the rendered probe conversation must contain (tool named `mrun_probe`).
    pub signature: &'static str,
    /// Markers that must be single added tokens in the vocabulary.
    pub required_tokens: &'static [&'static str],
    /// Tokens that start a message in the rendered prompt (T004a).
    pub message_starts: &'static [&'static str],
    pub none_bias: NoneBias,
    pub args: ArgKind,
    pub parallel_calls: bool,
    pub reasoning: ReasoningMarkers,
    pub history: HistoryRule,
    pub thinking: ThinkingControl,
}

const THINK: ReasoningMarkers = ReasoningMarkers::Tags {
    open: hermes::THINK_OPEN,
    close: hermes::THINK_CLOSE,
};
const CHATML_STARTS: &[&str] = &["<|im_start|>", "<tool_response>"];
const QWEN_TOKENS: &[&str] = &[
    "<|im_start|>",
    "<|im_end|>",
    hermes::CALL_OPEN,
    hermes::CALL_CLOSE,
    "<tool_response>",
];

pub static HERMES: ChatSyntax = ChatSyntax {
    id: SyntaxId::Hermes,
    name: "hermes_json",
    signature: "<tool_call>\n{\"name\": \"mrun_probe\"",
    required_tokens: QWEN_TOKENS,
    message_starts: CHATML_STARTS,
    none_bias: NoneBias {
        markers: &[hermes::CALL_OPEN],
        header_only: false,
    },
    args: ArgKind::Raw,
    parallel_calls: true,
    reasoning: THINK,
    history: HistoryRule {
        args: HistoryArgs::Verbatim,
        one_call_per_message: false,
    },
    thinking: ThinkingControl::EnableThinking,
};

pub static QWEN_XML: ChatSyntax = ChatSyntax {
    id: SyntaxId::QwenXml,
    name: "qwen_xml",
    signature: "<function=mrun_probe>",
    required_tokens: QWEN_TOKENS,
    message_starts: CHATML_STARTS,
    none_bias: NoneBias {
        markers: &[hermes::CALL_OPEN],
        header_only: false,
    },
    args: ArgKind::Synthesized,
    parallel_calls: true,
    reasoning: THINK,
    history: HistoryRule {
        args: HistoryArgs::Object,
        one_call_per_message: false,
    },
    thinking: ThinkingControl::EnableThinking,
};

pub static HARMONY: ChatSyntax = ChatSyntax {
    id: SyntaxId::Harmony,
    name: "harmony",
    signature: "to=functions.mrun_probe",
    required_tokens: &[
        harmony::START,
        harmony::CHANNEL,
        harmony::CONSTRAIN,
        harmony::MESSAGE,
        harmony::END,
        harmony::CALL,
        harmony::RETURN,
    ],
    message_starts: &[harmony::START],
    none_bias: NoneBias {
        markers: &[" to"],
        header_only: true,
    },
    args: ArgKind::Raw,
    parallel_calls: false,
    reasoning: ReasoningMarkers::Channel("analysis"),
    history: HistoryRule {
        args: HistoryArgs::Object,
        one_call_per_message: true,
    },
    thinking: ThinkingControl::ReasoningEffort,
};

impl ChatSyntax {
    pub fn get(id: SyntaxId) -> &'static ChatSyntax {
        match id {
            SyntaxId::Hermes => &HERMES,
            SyntaxId::QwenXml => &QWEN_XML,
            SyntaxId::Harmony => &HARMONY,
        }
    }

    /// A parser for one request's output.
    pub fn new_parser(&self, cfg: &ParserConfig) -> Box<dyn SyntaxParser> {
        match self.id {
            SyntaxId::Hermes => Box::new(HermesParser::new(cfg)),
            SyntaxId::QwenXml => Box::new(QwenXmlParser::new(cfg)),
            SyntaxId::Harmony => Box::new(HarmonyParser::new(cfg)),
        }
    }

    pub fn history(&self) -> HistoryRule {
        self.history
    }
}

/// The template does not render the family's syntax (load refused, FR-032).
#[derive(Debug, thiserror::Error, Eq, PartialEq)]
#[error(
    "chat template does not render the {syntax} tool-call syntax; missing: {}",
    missing.join(", ")
)]
pub struct TemplateCheckError {
    pub syntax: &'static str,
    /// The signature (if not rendered) and markers that are not added tokens, quoted.
    pub missing: Vec<String>,
}

/// Checks the rendered probe conversation against `syntax`: its signature must appear and its
/// markers must be added tokens. The caller adds the family name to the refusal.
pub fn check_template(
    syntax: &ChatSyntax,
    rendered_probe: &str,
    is_added_token: impl Fn(&str) -> bool,
) -> Result<(), TemplateCheckError> {
    let mut missing = Vec::new();
    if !rendered_probe.contains(syntax.signature) {
        missing.push(format!("{:?}", syntax.signature));
    }
    missing.extend(
        syntax
            .required_tokens
            .iter()
            .filter(|token| !is_added_token(token))
            .map(|token| format!("{token:?} (not an added token)")),
    );
    if missing.is_empty() {
        Ok(())
    } else {
        Err(TemplateCheckError {
            syntax: syntax.name,
            missing,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::super::{ParseEnd, ParseOut};
    use super::*;

    #[test]
    fn syn_registry_template_check_refused() {
        let all = |_: &str| true;
        let qwen3 = "<|im_start|>assistant\n<tool_call>\n{\"name\": \"mrun_probe\", \"arguments\": {}}\n</tool_call><|im_end|>";
        assert_eq!(check_template(&HERMES, qwen3, all), Ok(()));
        // Hermes-rendering template loaded for an XML family: refused, naming the syntax.
        let err = check_template(&QWEN_XML, qwen3, all).unwrap_err();
        assert_eq!(err.syntax, "qwen_xml");
        assert_eq!(err.missing, vec!["\"<function=mrun_probe>\"".to_string()]);
        assert!(err.to_string().contains("qwen_xml"));
        // Signature present but a marker is not an added token.
        let err = check_template(&HERMES, qwen3, |t| t != "<tool_call>").unwrap_err();
        assert_eq!(
            err.missing,
            vec!["\"<tool_call>\" (not an added token)".to_string()]
        );
        let harmony = "<|start|>assistant to=functions.mrun_probe<|channel|>commentary json<|message|>{}<|call|>";
        assert_eq!(check_template(&HARMONY, harmony, all), Ok(()));
        assert!(check_template(&HARMONY, qwen3, all).is_err());
    }

    #[test]
    fn entries_build_their_parsers() {
        let cfg = ParserConfig::default();
        for (id, text) in [
            (
                SyntaxId::Hermes,
                "<tool_call>\n{\"name\": \"x\", \"arguments\": {}}\n</tool_call>",
            ),
            (
                SyntaxId::QwenXml,
                "<tool_call>\n<function=x>\n</function>\n</tool_call>",
            ),
            (
                SyntaxId::Harmony,
                "<|channel|>commentary to=functions.x<|message|>{}<|call|>",
            ),
        ] {
            let syntax = ChatSyntax::get(id);
            assert_eq!(syntax.id, id);
            let mut parser = syntax.new_parser(&cfg);
            let mut out = ParseOut::default();
            parser.feed(0, text, &mut out);
            parser.finish(ParseEnd::Stop, &mut out);
            assert_eq!(parser.calls(), 1, "{}", syntax.name);
        }
        assert_eq!(HERMES.args, ArgKind::Raw);
        assert_eq!(QWEN_XML.args, ArgKind::Synthesized);
        assert!(!HARMONY.parallel_calls);
        assert_eq!(HARMONY.history().args, HistoryArgs::Object);
    }
}
