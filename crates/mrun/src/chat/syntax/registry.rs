//! Syntax registry (FR-032, data-model "Syntax"): per [`SyntaxId`] the parser, the template
//! check, argument kind and history rule. No syntax-specific logic lives outside its parser and
//! its entry here.

use super::hermes::{self, HermesParser};
use super::qwen_xml::QwenXmlParser;
use super::{ParserConfig, SyntaxId, SyntaxParser};
use crate::chat::template::{ChatTemplate, RenderOptions};
use crate::chat::tokenizer::Tokenizer;
use serde_json::json;
use std::collections::HashSet;

/// How a syntax carries tool-call arguments (FR-004).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ArgKind {
    /// The model's own JSON text: streamed as it arrives (Chat, Responses).
    Raw,
    /// JSON synthesized from another format: buffered per call.
    Synthesized,
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
    pub args: ArgKind,
    pub history: HistoryRule,
}

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
    args: ArgKind::Raw,
    history: HistoryRule {
        args: HistoryArgs::Verbatim,
    },
};

pub static QWEN_XML: ChatSyntax = ChatSyntax {
    id: SyntaxId::QwenXml,
    name: "qwen_xml",
    signature: "<function=mrun_probe>",
    required_tokens: QWEN_TOKENS,
    args: ArgKind::Synthesized,
    history: HistoryRule {
        args: HistoryArgs::Object,
    },
};

impl ChatSyntax {
    pub fn get(id: SyntaxId) -> &'static ChatSyntax {
        match id {
            SyntaxId::Hermes => &HERMES,
            SyntaxId::QwenXml => &QWEN_XML,
        }
    }

    /// A parser for one request's output.
    pub fn new_parser(&self, cfg: &ParserConfig) -> Box<dyn SyntaxParser> {
        match self.id {
            SyntaxId::Hermes => Box::new(HermesParser::new(cfg)),
            SyntaxId::QwenXml => Box::new(QwenXmlParser::new(cfg)),
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

/// The fixed probe conversation (010 R8): a user turn, then one call to `mrun_probe` with empty
/// arguments, with `mrun_probe` declared as the only tool.
fn probe() -> (serde_json::Value, RenderOptions) {
    let messages = json!([
        {"role": "user", "content": "probe"},
        {"role": "assistant", "content": "", "tool_calls": [{
            "id": "call_probe",
            "type": "function",
            "function": {"name": "mrun_probe", "arguments": {}}
        }]}
    ]);
    let options = RenderOptions {
        tools: Some(json!([{
            "type": "function",
            "function": {
                "name": "mrun_probe",
                "description": "probe",
                "parameters": {"type": "object", "properties": {}}
            }
        }])),
        ..RenderOptions::default()
    };
    (messages, options)
}

/// The load-time template check (FR-032): renders the probe conversation with the model's own
/// template and checks it against `syntax` and the tokenizer's added tokens. A template that
/// fails to render the probe is refused the same way (its signature is missing).
pub fn check_loaded(
    syntax: &ChatSyntax,
    template: &ChatTemplate,
    tokenizer: &Tokenizer,
) -> Result<(), TemplateCheckError> {
    let (messages, options) = probe();
    let rendered = template.render(&messages, &options).unwrap_or_default();
    let added: HashSet<String> = tokenizer
        .added_tokens()
        .into_iter()
        .map(|(_, text)| text)
        .collect();
    check_template(syntax, &rendered, |token| added.contains(token))
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
    }

    #[test]
    fn load_check_renders_the_probe_with_the_models_template() {
        use crate::chat::tokenizer::test_support::byte_tokenizer;
        let tokenizer = byte_tokenizer();
        let qwen3 = ChatTemplate::new(include_str!("../testdata/qwen3.jinja").into()).unwrap();
        assert_eq!(check_loaded(&HERMES, &qwen3, &tokenizer), Ok(()));
        // The Hermes-rendering Qwen3 template for an XML family: refused, naming the syntax.
        let err = check_loaded(&QWEN_XML, &qwen3, &tokenizer).unwrap_err();
        assert_eq!(err.syntax, "qwen_xml");
        assert_eq!(err.missing, vec!["\"<function=mrun_probe>\"".to_string()]);
        // The `qwen3_5` family's template (mlx-community/Qwen3.5-9B-4bit) renders Qwen XML.
        let qwen3_5 = ChatTemplate::new(include_str!("../testdata/qwen3_5.jinja").into()).unwrap();
        assert_eq!(check_loaded(&QWEN_XML, &qwen3_5, &tokenizer), Ok(()));
        assert!(check_loaded(&HERMES, &qwen3_5, &tokenizer).is_err());
        // A template that drops tool calls, and one that raises on the probe: refused.
        for source in [
            "{% for m in messages %}{{ m.content }}{% endfor %}",
            "{{ raise_exception('no tools') }}",
        ] {
            let template = ChatTemplate::new(source.into()).unwrap();
            let err = check_loaded(&HERMES, &template, &tokenizer).unwrap_err();
            assert_eq!(err.syntax, "hermes_json");
            assert_eq!(err.missing.len(), 1, "{source}");
        }
    }

    /// The real templates of the cached Qwen models (skipped when they are not downloaded).
    #[test]
    fn load_check_accepts_the_real_qwen_templates() {
        let Ok(home) = std::env::var("HOME") else {
            return;
        };
        for (repo, syntax) in [
            ("models--mlx-community--Qwen3-8B-4bit", &HERMES),
            ("models--mlx-community--Qwen3.6-35B-A3B-4bit", &QWEN_XML),
        ] {
            let snapshots = std::path::Path::new(&home)
                .join(".cache/huggingface/hub")
                .join(repo)
                .join("snapshots");
            let Some(dir) = std::fs::read_dir(snapshots)
                .ok()
                .and_then(|mut d| d.next())
                .and_then(Result::ok)
            else {
                continue;
            };
            let dir = dir.path();
            let Ok(tokenizer) = Tokenizer::from_file(&dir.join("tokenizer.json")) else {
                continue;
            };
            let template = ChatTemplate::from_model_dir(&dir).unwrap();
            assert_eq!(
                check_loaded(syntax, &template, &tokenizer),
                Ok(()),
                "{repo}"
            );
            let other = if syntax.id == SyntaxId::Hermes {
                &QWEN_XML
            } else {
                &HERMES
            };
            assert!(
                check_loaded(other, &template, &tokenizer).is_err(),
                "{repo}"
            );
        }
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
        ] {
            let syntax = ChatSyntax::get(id);
            assert_eq!(syntax.id, id);
            let mut parser = syntax.new_parser(&cfg);
            let mut out = ParseOut::default();
            parser.feed(super::super::NO_TOKEN, text, 0, &mut out);
            parser.finish(ParseEnd::Stop, &mut out);
            let ends = out
                .events
                .iter()
                .filter(|e| matches!(e, crate::api::model::GenEvent::ToolEnd { valid: true, .. }))
                .count();
            assert_eq!(ends, 1, "{}", syntax.name);
        }
        assert_eq!(HERMES.args, ArgKind::Raw);
        assert_eq!(QWEN_XML.args, ArgKind::Synthesized);
        assert_eq!(QWEN_XML.history().args, HistoryArgs::Object);
    }
}
