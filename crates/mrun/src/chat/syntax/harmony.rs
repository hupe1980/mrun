//! Harmony syntax (gpt-oss, research §3b, FR-033). Output starts after the generation prompt
//! `<|start|>assistant` and is a sequence of messages
//! `[ to=RECIPIENT]<|channel|>CHANNEL[ to=RECIPIENT][ <|constrain|>TYPE]<|message|>BODY` ended by
//! `<|end|>` (followed by `<|start|>assistant…`), `<|call|>` or `<|return|>`.
//!
//! `analysis` → reasoning; `final` and recipient-less `commentary` → text; recipient
//! `functions.NAME` → a tool call whose arguments are the raw body; other recipients (built-in
//! `browser.*`, `python`) are dropped and logged. Generation stops at `<|call|>`, so a response
//! has at most one call. Markers are special tokens; they are matched on their text.

use super::holdback::HoldBack;
use super::{
    ParseEnd, ParseLog, ParseOut, ParserConfig, SyntaxParser, check_declared, is_json_object,
};

pub(crate) const START: &str = "<|start|>";
pub(crate) const CHANNEL: &str = "<|channel|>";
pub(crate) const CONSTRAIN: &str = "<|constrain|>";
pub(crate) const MESSAGE: &str = "<|message|>";
pub(crate) const END: &str = "<|end|>";
pub(crate) const CALL: &str = "<|call|>";
pub(crate) const RETURN: &str = "<|return|>";

const BODY_ENDS: &[&str] = &[END, CALL, RETURN];

#[derive(Debug)]
enum Body {
    Reasoning,
    Text,
    /// A function call; arguments so far and what was streamed.
    Call {
        idx: Option<u32>,
        name: String,
        raw: String,
        emitted: usize,
    },
    Drop,
}

#[derive(Debug)]
enum State {
    /// Inside a message header, before `<|message|>`.
    Header,
    Body(Body),
    /// After `<|end|>`, before the next `<|start|>`.
    Between,
    /// After `<|call|>` or `<|return|>`: generation is over.
    Done,
}

/// Streaming parser for the Harmony syntax.
#[derive(Debug)]
pub struct HarmonyParser {
    cfg: ParserConfig,
    state: State,
    hb: HoldBack,
    next_idx: u32,
    completed: u32,
}

/// Channel and recipient of a header, the recipient's `<|constrain|>` suffix stripped.
fn parse_header(header: &str) -> (Option<&str>, Option<&str>) {
    fn word(s: &str) -> &str {
        let end = s
            .find(|c: char| c.is_whitespace() || c == '<')
            .unwrap_or(s.len());
        &s[..end]
    }
    let channel = header
        .find(CHANNEL)
        .map(|at| word(&header[at + CHANNEL.len()..]))
        .filter(|c| !c.is_empty());
    let recipient = header
        .match_indices("to=")
        .find(|&(at, _)| {
            at == 0
                || header[..at].ends_with(|c: char| c.is_whitespace())
                || header[..at].ends_with('>')
        })
        .map(|(at, _)| word(&header[at + 3..]))
        .filter(|r| !r.is_empty());
    (channel, recipient)
}

impl HarmonyParser {
    pub fn new(cfg: &ParserConfig) -> Self {
        Self {
            cfg: cfg.clone(),
            state: State::Header,
            hb: HoldBack::default(),
            next_idx: 0,
            completed: 0,
        }
    }

    fn run(&mut self, out: &mut ParseOut) {
        loop {
            match &mut self.state {
                State::Header => {
                    let Some((at, _)) = self.hb.find(&[MESSAGE]) else {
                        return;
                    };
                    let header = self.hb.take(at);
                    self.hb.skip(MESSAGE.len());
                    let body = self.open_body(&header, out);
                    self.state = State::Body(body);
                }
                State::Body(body) => {
                    let Some((at, i)) = self.hb.find(BODY_ENDS) else {
                        let safe = self.hb.take_safe(BODY_ENDS);
                        body_text(body, &safe, self.cfg.buffer_args, out);
                        return;
                    };
                    let rest = self.hb.take(at);
                    self.hb.skip(BODY_ENDS[i].len());
                    body_text(body, &rest, self.cfg.buffer_args, out);
                    let State::Body(body) = std::mem::replace(&mut self.state, State::Done) else {
                        unreachable!()
                    };
                    if let Body::Call { .. } = body {
                        self.end_call(body, true, out);
                    }
                    if BODY_ENDS[i] == END {
                        self.state = State::Between;
                    }
                }
                State::Between => match self.hb.find(&[START]) {
                    Some((at, _)) => {
                        self.hb.skip(at + START.len());
                        self.state = State::Header;
                    }
                    None => {
                        self.hb.take_safe(&[START]);
                        return;
                    }
                },
                State::Done => {
                    self.hb.take_all();
                    return;
                }
            }
        }
    }

    fn open_body(&mut self, header: &str, out: &mut ParseOut) -> Body {
        match parse_header(header) {
            (_, Some(recipient)) => match recipient.strip_prefix("functions.") {
                Some(name) => {
                    let mut idx = None;
                    if !self.cfg.buffer_args {
                        check_declared(&self.cfg.tools, name, out);
                        out.tool_start(self.next_idx, name);
                        idx = Some(self.next_idx);
                        self.next_idx += 1;
                    }
                    Body::Call {
                        idx,
                        name: name.to_string(),
                        raw: String::new(),
                        emitted: 0,
                    }
                }
                None => {
                    out.logs
                        .push(ParseLog::BuiltinRecipient(recipient.to_string()));
                    Body::Drop
                }
            },
            (Some("analysis"), None) => Body::Reasoning,
            _ => Body::Text,
        }
    }

    fn end_call(&mut self, body: Body, complete: bool, out: &mut ParseOut) {
        let Body::Call {
            idx,
            name,
            raw,
            emitted,
        } = body
        else {
            return;
        };
        let idx = match idx {
            Some(idx) => {
                out.tool_args(idx, &raw[emitted..]);
                idx
            }
            None => {
                let idx = self.next_idx;
                self.next_idx += 1;
                check_declared(&self.cfg.tools, &name, out);
                out.tool_start(idx, &name);
                out.tool_args(idx, &raw);
                idx
            }
        };
        let valid = complete && is_json_object(&raw);
        out.tool_end(idx, raw, valid);
        if complete {
            self.completed += 1;
        }
    }
}

fn body_text(body: &mut Body, text: &str, buffer: bool, out: &mut ParseOut) {
    match body {
        Body::Reasoning => out.reasoning(text),
        Body::Text => out.text(text),
        Body::Call {
            idx, raw, emitted, ..
        } => {
            raw.push_str(text);
            if let (Some(idx), false) = (*idx, buffer) {
                out.tool_args(idx, &raw[*emitted..]);
                *emitted = raw.len();
            }
        }
        Body::Drop => {}
    }
}

impl SyntaxParser for HarmonyParser {
    fn feed(&mut self, _id: u32, text: &str, out: &mut ParseOut) {
        self.hb.push(text);
        self.run(out);
    }

    fn finish(&mut self, end: ParseEnd, out: &mut ParseOut) {
        let rest = self.hb.take_all();
        if let State::Body(mut body) = std::mem::replace(&mut self.state, State::Done) {
            body_text(&mut body, &rest, self.cfg.buffer_args, out);
            // `<|call|>` ends generation as an EOS token whose text may not be fed: a call open
            // at `Stop` is complete, at `Length` it is not.
            self.end_call(body, end == ParseEnd::Stop, out);
        }
    }

    fn calls(&self) -> u32 {
        self.completed
    }
}

#[cfg(test)]
mod tests {
    use super::super::GenEvent;
    use super::super::test_util::*;
    use super::*;
    use serde_json::json;

    fn cfg(buffer: bool) -> ParserConfig {
        ParserConfig {
            tools: vec![tool("get_weather", json!({}))],
            reasoning_open: false,
            buffer_args: buffer,
        }
    }

    fn check(buffer: bool, text: &str, end: ParseEnd) -> (Vec<GenEvent>, Vec<ParseLog>) {
        let c = cfg(buffer);
        assert_split_invariant(&|| Box::new(HarmonyParser::new(&c)), text, end)
    }

    const ARGS: &str = r#"{"city":"Berlin"}"#;

    #[test]
    fn syn_harmony_analysis_is_reasoning() {
        let out = "<|channel|>analysis<|message|>Think é.<|end|><|start|>assistant<|channel|>final<|message|>Hi.<|return|>";
        let (events, logs) = check(false, out, ParseEnd::Stop);
        assert_eq!(events, vec![reasoning("Think é."), text("Hi.")]);
        assert!(logs.is_empty());
    }

    #[test]
    fn syn_harmony_commentary_preamble_is_text() {
        let out = format!(
            "<|channel|>commentary<|message|>Checking.<|end|><|start|>assistant to=functions.get_weather<|channel|>commentary json<|message|>{ARGS}<|call|>"
        );
        let (events, _) = check(false, &out, ParseEnd::Stop);
        assert_eq!(
            events,
            vec![
                text("Checking."),
                start(0, "get_weather"),
                args(0, ARGS),
                end(0, ARGS, true)
            ]
        );
    }

    #[test]
    fn syn_harmony_recipient_in_role_header() {
        let out =
            format!(" to=functions.get_weather<|channel|>commentary json<|message|>{ARGS}<|call|>");
        for buffer in [false, true] {
            let (events, _) = check(buffer, &out, ParseEnd::Stop);
            assert_eq!(
                events,
                vec![start(0, "get_weather"), args(0, ARGS), end(0, ARGS, true)]
            );
        }
    }

    #[test]
    fn syn_harmony_recipient_after_channel() {
        let out = format!(
            "<|channel|>analysis<|message|>Need weather.<|end|><|start|>assistant<|channel|>commentary to=functions.get_weather <|constrain|>json<|message|>{ARGS}<|call|>"
        );
        let (events, _) = check(false, &out, ParseEnd::Stop);
        assert_eq!(
            events,
            vec![
                reasoning("Need weather."),
                start(0, "get_weather"),
                args(0, ARGS),
                end(0, ARGS, true)
            ]
        );
    }

    #[test]
    fn syn_harmony_constrain_suffix_stripped() {
        let out = format!(
            "<|channel|>commentary to=functions.get_weather<|constrain|>json<|message|>{ARGS}<|call|>"
        );
        let (events, logs) = check(false, &out, ParseEnd::Stop);
        assert_eq!(events[0], start(0, "get_weather"));
        assert!(logs.is_empty());
        assert_eq!(
            parse_header("<|channel|>commentary to=functions.x<|constrain|>json"),
            (Some("commentary"), Some("functions.x"))
        );
    }

    #[test]
    fn syn_harmony_stop_at_call_one_call_per_response() {
        // Anything after `<|call|>` (the engine stops there) is ignored.
        let out = format!(
            "<|channel|>commentary to=functions.get_weather <|constrain|>json<|message|>{ARGS}<|call|><|start|>assistant to=functions.get_weather<|channel|>commentary<|message|>{{}}<|call|>"
        );
        let (events, _) = check(false, &out, ParseEnd::Stop);
        assert_eq!(
            events,
            vec![start(0, "get_weather"), args(0, ARGS), end(0, ARGS, true)]
        );
        let mut p = HarmonyParser::new(&cfg(false));
        let mut o = ParseOut::default();
        p.feed(0, &out, &mut o);
        p.finish(ParseEnd::Stop, &mut o);
        assert_eq!(p.calls(), 1);
        // The stop token's text not fed: the call still completes at `Stop`.
        let out = format!("<|channel|>commentary to=functions.get_weather<|message|>{ARGS}");
        let (events, _) = check(false, &out, ParseEnd::Stop);
        assert_eq!(events.last(), Some(&end(0, ARGS, true)));
    }

    #[test]
    fn syn_harmony_builtin_recipient_ignored_logged() {
        let out = "<|channel|>analysis to=browser.search<|message|>{\"query\":\"x\"}<|call|>";
        let (events, logs) = check(false, out, ParseEnd::Stop);
        assert!(events.is_empty());
        assert_eq!(
            logs,
            vec![ParseLog::BuiltinRecipient("browser.search".into())]
        );
        let out = " to=python<|channel|>analysis<|message|>print(1)<|call|>";
        let (_, logs) = check(false, out, ParseEnd::Stop);
        assert_eq!(logs, vec![ParseLog::BuiltinRecipient("python".into())]);
    }

    #[test]
    fn syn_harmony_truncated_call_length_incomplete() {
        let out = "<|channel|>commentary to=functions.get_weather <|constrain|>json<|message|>{\"city\":\"Ber";
        let (events, _) = check(false, out, ParseEnd::Length);
        let raw = r#"{"city":"Ber"#;
        assert_eq!(
            events,
            vec![start(0, "get_weather"), args(0, raw), end(0, raw, false)]
        );
    }

    #[test]
    fn syn_harmony_undeclared_tool_logged() {
        let out = "<|channel|>commentary to=functions.rm<|message|>{}<|call|>";
        let (events, logs) = check(false, out, ParseEnd::Stop);
        assert_eq!(
            events,
            vec![start(0, "rm"), args(0, "{}"), end(0, "{}", true)]
        );
        assert_eq!(logs, vec![ParseLog::ToolUnknown("rm".into())]);
    }

    #[test]
    fn malformed_arguments_are_invalid() {
        let out = "<|channel|>commentary to=functions.get_weather<|message|>{city: x}<|call|>";
        let (events, _) = check(false, out, ParseEnd::Stop);
        assert_eq!(events.last(), Some(&end(0, "{city: x}", false)));
    }
}
