//! Qwen XML syntax (Qwen3-Coder, Qwen3.6): `<think>` reasoning as Hermes and calls written as
//! `<tool_call>\n<function=NAME>\n<parameter=K>\nVALUE\n</parameter>\n…</function>\n</tool_call>`
//! (research §3).
//!
//! JSON arguments are synthesized incrementally: string-typed parameters (by schema; unknown or
//! missing types too) stream JSON-escaped, other types are buffered and must parse as JSON as
//! written (no coercion, never evaluated). A value ends only at `\n</parameter>\n` followed by
//! `<parameter=` or `</function>`, so argument text containing the syntax's own tags round-trips.
//! A `<function=NAME>` without `<tool_call>` is recognized for declared names only.

use super::hermes::{CALL_CLOSE, CALL_OPEN, THINK_CLOSE, THINK_OPEN, take_reasoning, trim_lead};
use super::holdback::HoldBack;
use super::{ParseEnd, ParseOut, ParserConfig, SyntaxParser, check_declared};
use crate::api::model::ToolDef;

const FUNC_OPEN: &str = "<function=";
const FUNC_CLOSE: &str = "</function>";
const PARAM_OPEN: &str = "<parameter=";
const PARAM_CLOSE: &str = "\n</parameter>\n";
const VALUE_ENDS: &[&str] = &["\n</parameter>\n<parameter=", "\n</parameter>\n</function>"];
const TEXT_MARKERS: &[&str] = &[CALL_OPEN, THINK_OPEN, THINK_CLOSE, FUNC_OPEN];
const REASONING_MARKERS: &[&str] = &[THINK_CLOSE, CALL_OPEN];
/// Longest function or parameter name looked for before a tag is taken as text.
const MAX_NAME: usize = 256;

#[derive(Debug)]
enum Mode {
    Text,
    Reasoning,
    Call(Box<XmlCall>),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Stage {
    /// After `<tool_call>`, before `<function=NAME>`.
    AwaitFunction,
    /// Between parameters.
    Between,
    /// Inside a parameter value; `nl` while its leading newline is still expected.
    Value { nl: bool },
    /// After `</function>`, expecting `</tool_call>` (optional for a recovered call).
    AfterFunction,
}

/// Streaming parser for the Qwen XML syntax.
#[derive(Debug)]
pub struct QwenXmlParser {
    cfg: ParserConfig,
    mode: Mode,
    hb: HoldBack,
    skip_nl: bool,
    next_idx: u32,
    completed: u32,
}

#[derive(Debug)]
struct XmlCall {
    stage: Stage,
    /// The call began with `<tool_call>` (else it was recovered from `<function=`).
    opener: bool,
    name: String,
    idx: Option<u32>,
    /// The call's text from `<function=` on: `raw_args` when the call is invalid.
    raw: String,
    /// Synthesized JSON arguments.
    json: String,
    emitted: usize,
    params: u32,
    /// The open parameter is not string-typed: its value is buffered here.
    typed: Option<String>,
    invalid: bool,
}

impl XmlCall {
    fn new(opener: bool) -> Self {
        Self {
            stage: Stage::AwaitFunction,
            opener,
            name: String::new(),
            idx: None,
            raw: String::new(),
            json: String::new(),
            emitted: 0,
            params: 0,
            typed: None,
            invalid: false,
        }
    }

    fn start_param(&mut self, key: &str, tool: Option<&ToolDef>) {
        self.json
            .push_str(if self.params == 0 { "{" } else { ", " });
        self.json.push_str(&json_string(key));
        self.json.push_str(": ");
        self.params += 1;
        let ty = tool
            .and_then(|tool| tool.param_type(key))
            .unwrap_or("string");
        if matches!(
            ty,
            "integer" | "number" | "boolean" | "object" | "array" | "null"
        ) {
            self.typed = Some(String::new());
        } else {
            self.typed = None;
            self.json.push('"');
        }
    }

    fn value_chunk(&mut self, chunk: &str) {
        self.raw.push_str(chunk);
        match &mut self.typed {
            Some(buf) => buf.push_str(chunk),
            None => {
                let escaped = json_string(chunk);
                self.json.push_str(&escaped[1..escaped.len() - 1]);
            }
        }
    }

    fn end_value(&mut self) {
        match self.typed.take() {
            None => self.json.push('"'),
            Some(value) => {
                let value = value.trim();
                if serde_json::from_str::<serde_json::Value>(value).is_ok() {
                    self.json.push_str(value);
                } else {
                    self.invalid = true;
                }
            }
        }
    }
}

fn json_string(text: &str) -> String {
    serde_json::Value::from(text).to_string()
}

/// The name in `<tag=NAME>` at the start of `s`: `Some(Some((name, tag length)))` when complete,
/// `Some(None)` while it may still complete, `None` when it is not such a tag.
fn tag_name<'a>(s: &'a str, tag: &str) -> Option<Option<(&'a str, usize)>> {
    let rest = &s[tag.len()..];
    match rest.find('>') {
        Some(q) if q <= MAX_NAME && !rest[..q].contains('\n') => {
            Some(Some((&rest[..q], tag.len() + q + 1)))
        }
        Some(_) => None,
        None if rest.len() <= MAX_NAME && !rest.contains('\n') => Some(None),
        None => None,
    }
}

impl QwenXmlParser {
    pub fn new(cfg: &ParserConfig) -> Self {
        Self {
            cfg: cfg.clone(),
            mode: if cfg.reasoning_open {
                Mode::Reasoning
            } else {
                Mode::Text
            },
            hb: HoldBack::default(),
            skip_nl: cfg.reasoning_open,
            next_idx: 0,
            completed: 0,
        }
    }

    fn declared(&self, name: &str) -> bool {
        self.cfg.tools.iter().any(|tool| tool.name == name)
    }

    fn run(&mut self, out: &mut ParseOut) {
        loop {
            let progressed = match &self.mode {
                Mode::Text => self.step_text(out),
                Mode::Reasoning => self.step_reasoning(out),
                Mode::Call(_) => self.step_call(out),
            };
            self.emit_args(out);
            if !progressed {
                return;
            }
        }
    }

    fn step_text(&mut self, out: &mut ParseOut) -> bool {
        trim_lead(&mut self.hb, &mut self.skip_nl);
        let Some((at, i)) = self.hb.find(TEXT_MARKERS) else {
            out.text(self.hb.take_safe(TEXT_MARKERS));
            return false;
        };
        let marker = TEXT_MARKERS[i];
        if marker == FUNC_OPEN {
            match tag_name(&self.hb.as_str()[at..], FUNC_OPEN) {
                Some(Some((name, _))) if self.declared(name.trim()) => {
                    // Missing `<tool_call>` before a declared function: recovered.
                    out.text(self.hb.take(at));
                    self.mode = Mode::Call(Box::new(XmlCall::new(false)));
                }
                Some(None) => {
                    out.text(self.hb.take(at));
                    return false;
                }
                _ => out.text(self.hb.take(at + FUNC_OPEN.len())),
            }
            return true;
        }
        out.text(self.hb.take(at));
        self.hb.skip(marker.len());
        self.skip_nl = true;
        self.mode = match marker {
            CALL_OPEN => Mode::Call(Box::new(XmlCall::new(true))),
            THINK_OPEN => Mode::Reasoning,
            _ => Mode::Text,
        };
        true
    }

    fn step_reasoning(&mut self, out: &mut ParseOut) -> bool {
        trim_lead(&mut self.hb, &mut self.skip_nl);
        let Some((at, i)) = self.hb.find(REASONING_MARKERS) else {
            out.reasoning(take_reasoning(&mut self.hb, REASONING_MARKERS));
            return false;
        };
        out.reasoning(self.hb.take(at).trim_end_matches('\n'));
        self.hb.skip(REASONING_MARKERS[i].len());
        self.skip_nl = true;
        self.mode = if REASONING_MARKERS[i] == CALL_OPEN {
            Mode::Call(Box::new(XmlCall::new(true)))
        } else {
            Mode::Text
        };
        true
    }

    fn step_call(&mut self, out: &mut ParseOut) -> bool {
        let Mode::Call(call) = &mut self.mode else {
            unreachable!()
        };
        let s = self.hb.as_str();
        let body = s.trim_start();
        let ws = s.len() - body.len();
        match call.stage {
            Stage::AwaitFunction => {
                if body.is_empty() || FUNC_OPEN.starts_with(body) && body.len() < FUNC_OPEN.len() {
                    return false;
                }
                let tag = body
                    .starts_with(FUNC_OPEN)
                    .then(|| tag_name(body, FUNC_OPEN))
                    .flatten();
                match tag {
                    Some(None) => false,
                    Some(Some((name, len))) => {
                        call.name = name.trim().to_string();
                        call.raw.push_str(&body[..len]);
                        call.stage = Stage::Between;
                        self.hb.skip(ws + len);
                        if !self.cfg.buffer_args {
                            check_declared(&self.cfg.tools, &call.name, out);
                            out.tool_start(self.next_idx, &call.name);
                            call.idx = Some(self.next_idx);
                            self.next_idx += 1;
                        }
                        true
                    }
                    None => {
                        // Not the XML call shape: the opener is passed through as text.
                        out.text(CALL_OPEN);
                        self.mode = Mode::Text;
                        self.skip_nl = false;
                        true
                    }
                }
            }
            Stage::Between => {
                if body.is_empty() {
                    return false;
                }
                if body.starts_with(PARAM_OPEN) {
                    match tag_name(body, PARAM_OPEN) {
                        Some(None) => return false,
                        Some(Some((key, len))) => {
                            let key = key.to_string();
                            call.raw.push_str(&s[..ws + len]);
                            self.hb.skip(ws + len);
                            let tool = self.cfg.tools.iter().find(|tool| tool.name == call.name);
                            call.start_param(&key, tool);
                            call.stage = Stage::Value { nl: true };
                            return true;
                        }
                        None => {}
                    }
                } else if body.starts_with(FUNC_CLOSE) {
                    call.raw.push_str(&s[..ws + FUNC_CLOSE.len()]);
                    self.hb.skip(ws + FUNC_CLOSE.len());
                    call.json
                        .push_str(if call.params == 0 { "{}" } else { "}" });
                    call.stage = Stage::AfterFunction;
                    return true;
                } else if body.starts_with(CALL_CLOSE) {
                    // `</tool_call>` before `</function>`: an incomplete call.
                    self.hb.skip(ws + CALL_CLOSE.len());
                    self.close(false, out);
                    return true;
                } else if [PARAM_OPEN, FUNC_CLOSE, CALL_CLOSE]
                    .iter()
                    .any(|tag| tag.starts_with(body))
                {
                    return false;
                }
                // Stray text between parameters.
                let n = ws + body.chars().next().map_or(0, char::len_utf8);
                call.raw.push_str(&s[..n]);
                call.invalid = true;
                self.hb.skip(n);
                true
            }
            Stage::Value { nl: true } => {
                if s.is_empty() {
                    return false;
                }
                if s.starts_with('\n') {
                    call.raw.push('\n');
                    self.hb.skip(1);
                }
                call.stage = Stage::Value { nl: false };
                true
            }
            Stage::Value { nl: false } => match self.hb.find(VALUE_ENDS) {
                Some((at, _)) => {
                    let chunk = self.hb.take(at);
                    call.value_chunk(&chunk);
                    call.end_value();
                    call.raw.push_str(PARAM_CLOSE);
                    self.hb.skip(PARAM_CLOSE.len());
                    call.stage = Stage::Between;
                    true
                }
                None => {
                    let chunk = self.hb.take_safe(VALUE_ENDS);
                    call.value_chunk(&chunk);
                    false
                }
            },
            Stage::AfterFunction => {
                if body.is_empty() || CALL_CLOSE.starts_with(body) && body.len() < CALL_CLOSE.len()
                {
                    return false;
                }
                if body.starts_with(CALL_CLOSE) {
                    self.hb.skip(ws + CALL_CLOSE.len());
                } else {
                    self.hb.skip(ws);
                }
                self.close(true, out);
                true
            }
        }
    }

    /// Streams synthesized arguments of a valid, unbuffered call.
    fn emit_args(&mut self, out: &mut ParseOut) {
        if let Mode::Call(call) = &mut self.mode
            && let Some(idx) = call.idx
            && !call.invalid
            && call.emitted < call.json.len()
        {
            out.tool_args(idx, &call.json[call.emitted..]);
            call.emitted = call.json.len();
        }
    }

    /// Ends the open call: complete ones carry the synthesized JSON, others the raw call text.
    fn close(&mut self, complete: bool, out: &mut ParseOut) {
        let Mode::Call(call) = std::mem::replace(&mut self.mode, Mode::Text) else {
            return;
        };
        self.skip_nl = true;
        let valid = complete && !call.invalid;
        let raw_args = if valid { call.json } else { call.raw };
        let idx = match call.idx {
            Some(idx) => {
                if valid {
                    out.tool_args(idx, &raw_args[call.emitted..]);
                }
                idx
            }
            None => {
                let idx = self.next_idx;
                self.next_idx += 1;
                check_declared(&self.cfg.tools, &call.name, out);
                out.tool_start(idx, &call.name);
                out.tool_args(idx, &raw_args);
                idx
            }
        };
        out.tool_end(idx, raw_args, valid);
        if complete {
            self.completed += 1;
        }
    }
}

impl SyntaxParser for QwenXmlParser {
    fn feed(&mut self, _id: u32, text: &str, out: &mut ParseOut) {
        self.hb.push(text);
        self.run(out);
    }

    fn finish(&mut self, _end: ParseEnd, out: &mut ParseOut) {
        match &mut self.mode {
            Mode::Text => {
                trim_lead(&mut self.hb, &mut self.skip_nl);
                out.text(self.hb.take_all());
            }
            Mode::Reasoning => out.reasoning(self.hb.take_all().trim_end_matches('\n')),
            Mode::Call(call) => match call.stage {
                Stage::AwaitFunction => {
                    if call.opener {
                        out.text(CALL_OPEN);
                    }
                    out.text(self.hb.take_all());
                    self.mode = Mode::Text;
                }
                Stage::AfterFunction => {
                    self.hb.take_all();
                    self.close(true, out);
                }
                // An unclosed call is incomplete at `Length` and `Stop` alike.
                Stage::Between | Stage::Value { .. } => {
                    call.raw.push_str(&self.hb.take_all());
                    self.close(false, out);
                }
            },
        }
    }

    fn calls(&self) -> u32 {
        self.completed
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_util::*;
    use super::super::{GenEvent, ParseLog};
    use super::*;
    use serde_json::json;

    fn cfg(open: bool, buffer: bool) -> ParserConfig {
        ParserConfig {
            tools: vec![
                tool(
                    "edit",
                    json!({"properties": {
                        "path": {"type": "string"},
                        "text": {"type": ["string", "null"]},
                        "line": {"type": "integer"},
                        "force": {"type": "boolean"},
                        "opts": {"type": "object"},
                    }}),
                ),
                tool("ls", json!({})),
            ],
            reasoning_open: open,
            buffer_args: buffer,
        }
    }

    fn check(c: ParserConfig, text: &str, end: ParseEnd) -> (Vec<GenEvent>, Vec<ParseLog>) {
        assert_split_invariant(&|| Box::new(QwenXmlParser::new(&c)), text, end)
    }

    fn call(name: &str, params: &[(&str, &str)]) -> String {
        let mut s = format!("<tool_call>\n<function={name}>\n");
        for (k, v) in params {
            s.push_str(&format!("<parameter={k}>\n{v}\n</parameter>\n"));
        }
        s.push_str("</function>\n</tool_call>");
        s
    }

    fn one_call(events: &[GenEvent]) -> (String, bool) {
        match events.last() {
            Some(GenEvent::ToolEnd {
                raw_args, valid, ..
            }) => (raw_args.clone(), *valid),
            other => panic!("no ToolEnd: {other:?}"),
        }
    }

    #[test]
    fn typed_parameters_synthesize_json() {
        let out = format!(
            "Editing.\n\n{}",
            call(
                "edit",
                &[
                    ("path", "src/a \"b\".rs"),
                    ("line", "42"),
                    ("force", "true"),
                    ("opts", "{\"a\": [1]}")
                ]
            )
        );
        for buffer in [false, true] {
            let (events, logs) = check(cfg(false, buffer), &out, ParseEnd::Stop);
            let a = r#"{"path": "src/a \"b\".rs", "line": 42, "force": true, "opts": {"a": [1]}}"#;
            assert_eq!(
                events,
                vec![
                    text("Editing.\n\n"),
                    start(0, "edit"),
                    args(0, a),
                    end(0, a, true)
                ]
            );
            assert!(logs.is_empty());
        }
    }

    #[test]
    fn closing_tags_inside_values_round_trip() {
        let v = "x\n</parameter>\ny </tool_call> </function>\n</parameter>\n</parameter>";
        let out = call("edit", &[("text", v), ("path", "p")]);
        let (events, _) = check(cfg(false, false), &out, ParseEnd::Stop);
        let (raw, valid) = one_call(&events);
        assert!(valid);
        let parsed: serde_json::Value = serde_json::from_str(&raw).unwrap();
        assert_eq!(parsed, json!({"text": v, "path": "p"}));
    }

    #[test]
    fn null_stays_a_string_for_string_params() {
        let out = call("edit", &[("text", "null"), ("path", "null")]);
        let (events, _) = check(cfg(false, false), &out, ParseEnd::Stop);
        assert_eq!(
            one_call(&events),
            (r#"{"text": "null", "path": "null"}"#.into(), true)
        );
    }

    #[test]
    fn non_exact_booleans_are_invalid() {
        for bad in ["True", "yes", "1 == 1", "'x'"] {
            let out = call("edit", &[("force", bad)]);
            let (events, _) = check(cfg(false, true), &out, ParseEnd::Stop);
            let raw =
                format!("<function=edit>\n<parameter=force>\n{bad}\n</parameter>\n</function>");
            assert_eq!(
                events,
                vec![start(0, "edit"), args(0, &raw), end(0, &raw, false)],
                "{bad}"
            );
        }
    }

    #[test]
    fn missing_tool_call_recovered_for_declared_names_only() {
        let out = "ok <function=ls>\n</function>\nthen <function=nope>\n</function>";
        let (events, logs) = check(cfg(false, false), out, ParseEnd::Stop);
        assert_eq!(
            events,
            vec![
                text("ok "),
                start(0, "ls"),
                args(0, "{}"),
                end(0, "{}", true),
                text("then <function=nope>\n</function>"),
            ]
        );
        assert!(logs.is_empty());
    }

    #[test]
    fn parallel_calls_and_reasoning() {
        let out = format!(
            "<think>\nwhich?\n</think>\n\n{}\n{}",
            call("ls", &[]),
            call("edit", &[("path", "é✓")])
        );
        let (events, _) = check(cfg(false, false), &out, ParseEnd::Stop);
        assert_eq!(
            events,
            vec![
                reasoning("which?"),
                start(0, "ls"),
                args(0, "{}"),
                end(0, "{}", true),
                start(1, "edit"),
                args(1, r#"{"path": "é✓"}"#),
                end(1, r#"{"path": "é✓"}"#, true),
            ]
        );
        let mut p = QwenXmlParser::new(&cfg(true, false));
        let mut o = ParseOut::default();
        p.feed(0, &format!("hm\n</think>{out}"), &mut o);
        p.finish(ParseEnd::Stop, &mut o);
        assert_eq!(p.calls(), 2);
    }

    #[test]
    fn tool_call_ends_reasoning() {
        let out = format!("<think>\nhm\n{}", call("ls", &[]));
        let (events, _) = check(cfg(false, false), &out, ParseEnd::Stop);
        assert_eq!(
            events,
            vec![
                reasoning("hm"),
                start(0, "ls"),
                args(0, "{}"),
                end(0, "{}", true)
            ]
        );
    }

    #[test]
    fn string_values_stream_before_the_call_ends() {
        let mut p = QwenXmlParser::new(&cfg(false, false));
        let mut o = ParseOut::default();
        p.feed(
            0,
            "<tool_call>\n<function=edit>\n<parameter=text>\nhello wor",
            &mut o,
        );
        assert_eq!(
            normalize(o.events),
            vec![start(0, "edit"), args(0, r#"{"text": "hello wor"#)]
        );
    }

    #[test]
    fn syn_qwen_xml_truncated_call_length_incomplete() {
        let out = "<tool_call>\n<function=edit>\n<parameter=path>\nsrc/ma";
        let (events, _) = check(cfg(false, true), out, ParseEnd::Length);
        let raw = "<function=edit>\n<parameter=path>\nsrc/ma";
        assert_eq!(
            events,
            vec![start(0, "edit"), args(0, raw), end(0, raw, false)]
        );
        let (events, _) = check(cfg(false, false), out, ParseEnd::Length);
        assert_eq!(events.last(), Some(&end(0, raw, false)));
    }

    #[test]
    fn syn_qwen_xml_undeclared_tool_logged() {
        let out = call("rm", &[("path", "/")]);
        let (events, logs) = check(cfg(false, false), &out, ParseEnd::Stop);
        let a = r#"{"path": "/"}"#;
        assert_eq!(events, vec![start(0, "rm"), args(0, a), end(0, a, true)]);
        assert_eq!(logs, vec![ParseLog::ToolUnknown("rm".into())]);
    }

    #[test]
    fn non_xml_call_body_passes_through() {
        let out = "<tool_call>\n{\"name\": \"ls\"}\n</tool_call>";
        let (events, _) = check(cfg(false, false), out, ParseEnd::Stop);
        assert_eq!(events, vec![text(out)]);
    }
}
