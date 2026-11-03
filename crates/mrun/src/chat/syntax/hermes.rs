//! Hermes JSON syntax (Qwen3): `<think>…</think>` reasoning and
//! `<tool_call>\n{"name": "…", "arguments": {…}}\n</tool_call>` calls (research §3).
//!
//! The call body is scanned with a small JSON scanner that tracks depth and string state, so the
//! name is reported as soon as it closes, the raw `arguments` text streams as it arrives, and a
//! `</tool_call>` inside a JSON string stays argument text.

use super::holdback::HoldBack;
use super::{CallSink, ParseEnd, ParseOut, ParserConfig, SyntaxParser, is_json_object};

pub(crate) const THINK_OPEN: &str = "<think>";
pub(crate) const THINK_CLOSE: &str = "</think>";
pub(crate) const CALL_OPEN: &str = "<tool_call>";
pub(crate) const CALL_CLOSE: &str = "</tool_call>";

const TEXT_MARKERS: &[&str] = &[CALL_OPEN, THINK_OPEN, THINK_CLOSE];
/// Text markers once output has begun: `<think>` opens reasoning only as the first output
/// (FR-032); later it is text.
const LATER_MARKERS: &[&str] = &[CALL_OPEN, THINK_CLOSE];
const REASONING_MARKERS: &[&str] = &[THINK_CLOSE, CALL_OPEN];

#[derive(Debug)]
enum Mode {
    Text,
    Reasoning,
    Call(Box<JsonCall>),
}

/// Streaming parser for the Hermes JSON syntax.
#[derive(Debug)]
pub struct HermesParser {
    cfg: ParserConfig,
    mode: Mode,
    hb: HoldBack,
    /// Drop leading newlines of the next text or reasoning (after `<think>`, `</think>`, a call).
    skip_nl: bool,
    /// Calls started so far (the next call's index).
    next_idx: u32,
    /// Output has begun: something other than leading whitespace was read.
    begun: bool,
}

impl HermesParser {
    pub fn new(cfg: &ParserConfig) -> Self {
        Self {
            cfg: cfg.clone(),
            mode: if cfg.reasoning_open {
                Mode::Reasoning
            } else {
                Mode::Text
            },
            hb: HoldBack::new(cfg.specials.clone()),
            skip_nl: cfg.reasoning_open,
            next_idx: 0,
            begun: cfg.reasoning_open,
        }
    }

    fn run(&mut self, out: &mut ParseOut) {
        loop {
            match &mut self.mode {
                Mode::Text => {
                    trim_lead(&mut self.hb, &mut self.skip_nl);
                    let markers = if self.begun {
                        LATER_MARKERS
                    } else {
                        TEXT_MARKERS
                    };
                    match self.hb.find(markers) {
                        Some((at, _))
                            if !self.begun && !self.hb.as_str()[..at].trim().is_empty() =>
                        {
                            // Text came first: `<think>` is no longer a marker.
                            self.begun = true;
                        }
                        Some((at, i)) => {
                            self.begun = true;
                            out.text(self.hb.take(at));
                            let base = self.hb.offset();
                            self.hb.skip(markers[i].len());
                            self.skip_nl = true;
                            self.mode = match markers[i] {
                                CALL_OPEN => Mode::Call(Box::new(JsonCall::new(base))),
                                THINK_OPEN => Mode::Reasoning,
                                // A bare `</think>` (Thinking-2507, duplicate close) is absorbed.
                                _ => Mode::Text,
                            };
                        }
                        None => {
                            let safe = self.hb.take_safe(markers);
                            self.begun |= !safe.trim().is_empty();
                            out.text(safe);
                            return;
                        }
                    }
                }
                Mode::Reasoning => {
                    trim_lead(&mut self.hb, &mut self.skip_nl);
                    match self.hb.find(REASONING_MARKERS) {
                        Some((at, i)) => {
                            out.reasoning(self.hb.take(at).trim_end_matches('\n'));
                            let base = self.hb.offset();
                            self.hb.skip(REASONING_MARKERS[i].len());
                            self.skip_nl = true;
                            // `<tool_call>` implicitly ends reasoning.
                            self.mode = if REASONING_MARKERS[i] == CALL_OPEN {
                                Mode::Call(Box::new(JsonCall::new(base)))
                            } else {
                                Mode::Text
                            };
                        }
                        None => {
                            out.reasoning(take_reasoning(&mut self.hb, REASONING_MARKERS));
                            return;
                        }
                    }
                }
                Mode::Call(call) => {
                    if call.body.is_empty() {
                        call.base = self.hb.offset();
                    }
                    call.body.push_str(&self.hb.take_all());
                    let closed = call.scan(&self.hb);
                    self.emit_progress(out);
                    let Some((at, end)) = closed else { return };
                    let Mode::Call(call) = std::mem::replace(&mut self.mode, Mode::Text) else {
                        unreachable!()
                    };
                    self.hb.unshift(&call.body[end..]);
                    self.close_call(*call, at, true, out);
                    self.skip_nl = true;
                }
            }
        }
    }

    /// Reports the name once known and streams argument text scanned so far.
    fn emit_progress(&mut self, out: &mut ParseOut) {
        let Mode::Call(call) = &mut self.mode else {
            return;
        };
        if self.cfg.buffer_args {
            return;
        }
        if let Some(name) = &call.name {
            call.sink
                .open(&mut self.next_idx, &self.cfg.tools, name, out);
        }
        if let Some(start) = call.args_start {
            let upto = call.args_end.unwrap_or(call.pos);
            if upto > start {
                call.sink.stream(&call.body[start..upto], out);
            }
        }
    }

    /// Ends a call whose body ends at `at`; a call whose name never parsed is passed through as
    /// text.
    fn close_call(&mut self, call: JsonCall, at: usize, complete: bool, out: &mut ParseOut) {
        let Some(name) = call.name.clone() else {
            out.text(format!("{CALL_OPEN}{}", &call.body[..at]));
            if complete {
                out.text(CALL_CLOSE);
            }
            return;
        };
        let raw = match (call.args_start, call.args_end) {
            (Some(start), Some(end)) => call.body[start..end].to_string(),
            (Some(start), None) => call.body[start..at].to_string(),
            // A complete object without `arguments`: no arguments.
            (None, _) if call.outer == Outer::Done => "{}".to_string(),
            (None, _) => String::new(),
        };
        let valid = complete && is_json_object(&raw);
        call.sink
            .close(&mut self.next_idx, &self.cfg.tools, &name, raw, valid, out);
    }
}

impl SyntaxParser for HermesParser {
    fn feed(&mut self, id: u32, text: &str, held: u32, out: &mut ParseOut) {
        self.hb.push_token(id, text, held);
        self.run(out);
    }

    fn finish(&mut self, _end: ParseEnd, out: &mut ParseOut) {
        match std::mem::replace(&mut self.mode, Mode::Text) {
            Mode::Text => {
                trim_lead(&mut self.hb, &mut self.skip_nl);
                out.text(self.hb.take_all());
            }
            Mode::Reasoning => out.reasoning(self.hb.take_all().trim_end_matches('\n')),
            // An unclosed call is incomplete at `Length` and `Stop` alike.
            Mode::Call(mut call) => {
                call.body.push_str(&self.hb.take_all());
                let at = call.body.len();
                call.pos = at;
                self.close_call(*call, at, false, out);
            }
        }
    }
}

/// Drops leading newlines while `skip` is set; clears it at the first other character.
pub(crate) fn trim_lead(hb: &mut HoldBack, skip: &mut bool) {
    if *skip {
        let n = hb.as_str().len() - hb.as_str().trim_start_matches('\n').len();
        hb.skip(n);
        if !hb.is_empty() {
            *skip = false;
        }
    }
}

/// Safe reasoning text, also holding back trailing newlines (dropped before `</think>`).
pub(crate) fn take_reasoning(hb: &mut HoldBack, markers: &[&str]) -> String {
    let mut safe = hb.take_safe(markers);
    let trimmed = safe.trim_end_matches('\n').len();
    let held = safe.split_off(trimmed);
    let rest = hb.take_all();
    hb.push(&held);
    hb.push(&rest);
    safe
}

/// Where the scanner is in the call's outer `{"name": …, "arguments": …}` object.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
enum Outer {
    #[default]
    Start,
    Key,
    InKey,
    Colon,
    ValueStart,
    Value,
    AfterValue,
    Done,
    /// Not the expected shape: only `</tool_call>` is looked for.
    Broken,
}

/// One Hermes call body (the text after `<tool_call>`), scanned incrementally.
#[derive(Debug, Default)]
struct JsonCall {
    body: String,
    /// Scan position in `body`.
    pos: usize,
    outer: Outer,
    in_str: bool,
    esc: bool,
    depth: u32,
    tok_start: usize,
    key: String,
    name: Option<String>,
    args_start: Option<usize>,
    args_end: Option<usize>,
    /// Stream offset of `body[0]`.
    base: usize,
    sink: CallSink,
}

impl JsonCall {
    fn new(base: usize) -> Self {
        Self {
            base,
            ..Self::default()
        }
    }

    /// Scans new body text; returns `(close offset, offset after the closer)` once
    /// `</tool_call>` appears outside a JSON string.
    fn scan(&mut self, hb: &HoldBack) -> Option<(usize, usize)> {
        while self.pos < self.body.len() {
            let b = self.body.as_bytes()[self.pos];
            if b == b'<' {
                let rest = &self.body[self.pos..];
                let at = self.base + self.pos;
                // The emitted token closes the call even inside a broken string.
                if rest.starts_with(CALL_CLOSE)
                    && (hb.is_token_at(at) || !self.in_str && hb.counts_at(at, CALL_CLOSE))
                {
                    if self.outer == Outer::Value {
                        self.end_value(self.pos);
                    }
                    return Some((self.pos, self.pos + CALL_CLOSE.len()));
                }
                if !self.in_str && CALL_CLOSE.starts_with(rest) {
                    return None;
                }
            }
            if self.step(b) {
                self.pos += 1;
            }
        }
        None
    }

    /// Advances the outer-object state by one byte; false when the byte must be re-read in the
    /// new state.
    fn step(&mut self, b: u8) -> bool {
        let ws = b.is_ascii_whitespace();
        match self.outer {
            Outer::Start if ws => {}
            Outer::Start if b == b'{' => self.outer = Outer::Key,
            Outer::Key | Outer::AfterValue | Outer::Colon | Outer::ValueStart | Outer::Done
                if ws => {}
            Outer::Key if b == b'"' => {
                self.outer = Outer::InKey;
                self.in_str = true;
                self.tok_start = self.pos;
            }
            Outer::Key if b == b'}' => self.outer = Outer::Done,
            Outer::InKey => {
                if self.string_byte(b) {
                    self.key = serde_json::from_str(&self.body[self.tok_start..=self.pos])
                        .unwrap_or_default();
                    self.outer = Outer::Colon;
                }
            }
            Outer::Colon if b == b':' => self.outer = Outer::ValueStart,
            Outer::ValueStart => {
                self.outer = Outer::Value;
                self.depth = 0;
                self.tok_start = self.pos;
                if self.key == "arguments" && self.args_start.is_none() {
                    self.args_start = Some(self.pos);
                }
                return false;
            }
            Outer::Value => {
                if self.in_str {
                    if self.string_byte(b) && self.depth == 0 {
                        self.end_value(self.pos + 1);
                    }
                } else {
                    match b {
                        b'"' => self.in_str = true,
                        b'{' | b'[' => self.depth += 1,
                        b'}' | b']' if self.depth > 0 => {
                            self.depth -= 1;
                            if self.depth == 0 {
                                self.end_value(self.pos + 1);
                            }
                        }
                        b'}' | b']' | b',' if self.depth == 0 => {
                            self.end_value(self.pos);
                            return false;
                        }
                        _ if ws && self.depth == 0 => {
                            self.end_value(self.pos);
                            return false;
                        }
                        _ => {}
                    }
                }
            }
            Outer::AfterValue if b == b',' => self.outer = Outer::Key,
            Outer::AfterValue if b == b'}' => self.outer = Outer::Done,
            // Trailing text after the object is ignored.
            Outer::Done | Outer::Broken => {}
            _ => {
                self.outer = Outer::Broken;
                self.in_str = false;
            }
        }
        true
    }

    /// One byte inside a string; true at its closing quote.
    fn string_byte(&mut self, b: u8) -> bool {
        if self.esc {
            self.esc = false;
        } else if b == b'\\' {
            self.esc = true;
        } else if b == b'"' {
            self.in_str = false;
            return true;
        }
        false
    }

    fn end_value(&mut self, end: usize) {
        self.outer = Outer::AfterValue;
        match self.key.as_str() {
            "name" if self.name.is_none() => {
                self.name = serde_json::from_str::<String>(&self.body[self.tok_start..end]).ok();
            }
            "arguments" if self.args_start == Some(self.tok_start) => self.args_end = Some(end),
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::ParseLog;
    use super::super::test_util::*;
    use super::*;
    use serde_json::json;

    fn cfg(open: bool, buffer: bool) -> ParserConfig {
        ParserConfig {
            tools: vec![
                tool(
                    "read_file",
                    json!({"properties": {"path": {"type": "string"}}}),
                ),
                tool("ls", json!({})),
            ],
            reasoning_open: open,
            buffer_args: buffer,
            specials: None,
        }
    }

    fn check(
        c: ParserConfig,
        text: &str,
        end: ParseEnd,
    ) -> (Vec<super::super::GenEvent>, Vec<ParseLog>) {
        assert_split_invariant(&|| Box::new(HermesParser::new(&c)), text, end)
    }

    #[test]
    fn reasoning_text_and_call() {
        let out = "<think>\nplan it\n</think>\n\nReading.\n<tool_call>\n{\"name\": \"read_file\", \"arguments\": {\"path\": \"src/main.rs\"}}\n</tool_call>";
        let (events, logs) = check(cfg(false, false), out, ParseEnd::Stop);
        let a = r#"{"path": "src/main.rs"}"#;
        assert_eq!(
            events,
            vec![
                reasoning("plan it"),
                text("Reading.\n"),
                start(0, "read_file"),
                args(0, a),
                end(0, a, true),
            ]
        );
        assert!(logs.is_empty());
    }

    #[test]
    fn reasoning_open_from_prompt_and_bare_close() {
        let (events, _) = check(
            cfg(true, false),
            "thinking\n</think>\n\nanswer",
            ParseEnd::Stop,
        );
        assert_eq!(events, vec![reasoning("thinking"), text("answer")]);
        // Bare `</think>` while not in reasoning is absorbed.
        let (events, _) = check(cfg(false, false), "a</think>\n\nb", ParseEnd::Stop);
        assert_eq!(events, vec![text("ab")]);
    }

    #[test]
    fn tool_call_implicitly_ends_reasoning() {
        let out = "<think>\nhm<tool_call>\n{\"name\": \"ls\", \"arguments\": {}}\n</tool_call>";
        let (events, _) = check(cfg(false, false), out, ParseEnd::Stop);
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
    fn parallel_calls_and_escaped_strings() {
        let a0 = r#"{"path": "a \"q\" \\ </tool_call> {"}"#;
        let a1 = r#"{"path": "é✓\n"}"#;
        let out = format!(
            "<tool_call>\n{{\"name\": \"read_file\", \"arguments\": {a0}}}\n</tool_call>\n<tool_call>\n{{\"name\": \"read_file\", \"arguments\": {a1}}}\n</tool_call>"
        );
        let (events, _) = check(cfg(false, false), &out, ParseEnd::Stop);
        assert_eq!(
            events,
            vec![
                start(0, "read_file"),
                args(0, a0),
                end(0, a0, true),
                start(1, "read_file"),
                args(1, a1),
                end(1, a1, true),
            ]
        );
    }

    #[test]
    fn syn_hermes_multi_key_arguments() {
        let a = r#"{"path": "a.rs", "limit": [1, {"x": 2}], "n": 3}"#;
        let out =
            format!("<tool_call>\n{{\"name\": \"read_file\", \"arguments\": {a}}}\n</tool_call>");
        for buffer in [false, true] {
            let (events, _) = check(cfg(false, buffer), &out, ParseEnd::Stop);
            assert_eq!(
                events,
                vec![start(0, "read_file"), args(0, a), end(0, a, true)]
            );
        }
    }

    #[test]
    fn arguments_before_name_and_buffered() {
        let a = r#"{"path": "x"}"#;
        let out =
            format!("<tool_call>\n{{\"arguments\": {a}, \"name\": \"read_file\"}}\n</tool_call>");
        let (events, _) = check(cfg(false, false), &out, ParseEnd::Stop);
        assert_eq!(
            events,
            vec![start(0, "read_file"), args(0, a), end(0, a, true)]
        );
        let (events, _) = check(cfg(false, true), &out, ParseEnd::Stop);
        assert_eq!(
            events,
            vec![start(0, "read_file"), args(0, a), end(0, a, true)]
        );
    }

    #[test]
    fn buffered_arguments_arrive_at_close() {
        let mut p = HermesParser::new(&cfg(false, true));
        let mut o = ParseOut::default();
        p.feed(
            super::super::NO_TOKEN,
            "<tool_call>\n{\"name\": \"ls\", \"arguments\": {\"a\"",
            0,
            &mut o,
        );
        assert!(o.events.is_empty());
        p.feed(super::super::NO_TOKEN, ": 1}}\n</tool_call>", 0, &mut o);
        assert_eq!(o.events.len(), 3);
    }

    #[test]
    fn malformed_json_is_invalid_with_raw_text() {
        let out = "<tool_call>\n{\"name\": \"ls\", \"arguments\": {\"a\": tru}}\n</tool_call>after";
        let (events, _) = check(cfg(false, false), out, ParseEnd::Stop);
        let raw = r#"{"a": tru}"#;
        assert_eq!(
            events,
            vec![
                start(0, "ls"),
                args(0, raw),
                end(0, raw, false),
                text("after")
            ]
        );
        // Unparseable envelope (no name): passed through as text.
        let out = "<tool_call>\nnot json\n</tool_call>";
        let (events, _) = check(cfg(false, false), out, ParseEnd::Stop);
        assert_eq!(events, vec![text(out)]);
    }

    #[test]
    fn syn_hermes_truncated_call_length_incomplete() {
        let out = "<tool_call>\n{\"name\": \"read_file\", \"arguments\": {\"path\": \"sr";
        let (events, _) = check(cfg(false, false), out, ParseEnd::Length);
        let raw = r#"{"path": "sr"#;
        assert_eq!(
            events,
            vec![start(0, "read_file"), args(0, raw), end(0, raw, false)]
        );
    }

    #[test]
    fn syn_hermes_undeclared_tool_logged() {
        let out = "<tool_call>\n{\"name\": \"rm\", \"arguments\": {}}\n</tool_call>";
        let (events, logs) = check(cfg(false, false), out, ParseEnd::Stop);
        assert_eq!(
            events,
            vec![start(0, "rm"), args(0, "{}"), end(0, "{}", true)]
        );
        assert_eq!(logs, vec![ParseLog::ToolUnknown("rm".into())]);
    }

    #[test]
    fn plain_text_with_angle_brackets_passes() {
        let out = "if a <b and <tool> then </thinker> é";
        let (events, _) = check(cfg(false, false), out, ParseEnd::Stop);
        assert_eq!(events, vec![text(out)]);
    }

    fn gated(open: bool) -> ParserConfig {
        ParserConfig {
            specials: Some(specials()),
            ..cfg(open, false)
        }
    }

    #[test]
    fn syn_hermes_spelled_markers_in_prose_stay_text() {
        let text_ = "Use <tool_call> tags and <think> blocks; </think> too.";
        let (events, _) = run_tokens(
            &mut HermesParser::new(&gated(false)),
            &format!("[[<think>]]plan[[</think>]]\n\n{text_}"),
            ParseEnd::Stop,
        );
        assert_eq!(events, vec![reasoning("plan"), text(text_)]);
    }

    #[test]
    fn syn_hermes_think_only_at_start() {
        // `<think>` as the first output opens reasoning.
        let (events, _) = check(
            cfg(false, false),
            "<think>\nhm\n</think>\n\nok",
            ParseEnd::Stop,
        );
        assert_eq!(events, vec![reasoning("hm"), text("ok")]);
        // After text it is text, by spelling and by token id alike.
        let (events, _) = check(cfg(false, false), "ok <think>x</think> y", ParseEnd::Stop);
        assert_eq!(events, vec![text("ok <think>x y")]);
        let (events, _) = run_tokens(
            &mut HermesParser::new(&gated(false)),
            "ok [[<think>]]x",
            ParseEnd::Stop,
        );
        assert_eq!(events, vec![text("ok <think>x")]);
        // After a call too.
        let out = "<tool_call>\n{\"name\": \"ls\", \"arguments\": {}}\n</tool_call><think>z";
        let (events, _) = check(cfg(false, false), out, ParseEnd::Stop);
        assert_eq!(events.last(), Some(&text("<think>z")));
    }

    #[test]
    fn syn_hermes_marker_text_not_token_is_content() {
        let (events, _) = run_tokens(
            &mut HermesParser::new(&gated(false)),
            "<think>a</think> <tool_call>\n{\"name\": \"ls\"}\n</tool_call>",
            ParseEnd::Stop,
        );
        assert_eq!(
            events,
            vec![text(
                "<think>a</think> <tool_call>\n{\"name\": \"ls\"}\n</tool_call>"
            )]
        );
    }

    #[test]
    fn syn_hermes_call_by_token_id() {
        let a = r#"{"path": "</tool_call>"}"#;
        let (events, _) = run_tokens(
            &mut HermesParser::new(&gated(false)),
            &format!(
                "ok\n[[<tool_call>]]\n{{\"name\": \"read_file\", \"arguments\": {a}}}\n[[</tool_call>]]"
            ),
            ParseEnd::Stop,
        );
        assert_eq!(
            events,
            vec![
                text("ok\n"),
                start(0, "read_file"),
                args(0, a),
                end(0, a, true)
            ]
        );
        // The emitted closer ends a call even inside a broken string.
        let (events, _) = run_tokens(
            &mut HermesParser::new(&gated(false)),
            "[[<tool_call>]]\n{\"name\": \"ls\", \"arguments\": {\"a\": \"x[[</tool_call>]]after",
            ParseEnd::Stop,
        );
        let raw = r#"{"a": "x"#;
        assert_eq!(
            events,
            vec![
                start(0, "ls"),
                args(0, raw),
                end(0, raw, false),
                text("after")
            ]
        );
    }
}
