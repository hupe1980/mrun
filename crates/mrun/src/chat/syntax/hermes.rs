//! Hermes JSON syntax (Qwen3): `<think>…</think>` reasoning and
//! `<tool_call>\n{"name": "…", "arguments": {…}}\n</tool_call>` calls (research §3).
//!
//! The call body is scanned with a small JSON scanner that tracks depth and string state, so the
//! name is reported as soon as it closes, the raw `arguments` text streams as it arrives, and a
//! `</tool_call>` inside a JSON string stays argument text.

use super::holdback::HoldBack;
use super::{ParseEnd, ParseOut, ParserConfig, SyntaxParser, check_declared, is_json_object};

pub(crate) const THINK_OPEN: &str = "<think>";
pub(crate) const THINK_CLOSE: &str = "</think>";
pub(crate) const CALL_OPEN: &str = "<tool_call>";
pub(crate) const CALL_CLOSE: &str = "</tool_call>";

const TEXT_MARKERS: &[&str] = &[CALL_OPEN, THINK_OPEN, THINK_CLOSE];
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
    completed: u32,
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
            hb: HoldBack::default(),
            skip_nl: cfg.reasoning_open,
            next_idx: 0,
            completed: 0,
        }
    }

    fn run(&mut self, out: &mut ParseOut) {
        loop {
            match &mut self.mode {
                Mode::Text => {
                    trim_lead(&mut self.hb, &mut self.skip_nl);
                    match self.hb.find(TEXT_MARKERS) {
                        Some((at, i)) => {
                            out.text(self.hb.take(at));
                            self.hb.skip(TEXT_MARKERS[i].len());
                            self.skip_nl = true;
                            self.mode = match TEXT_MARKERS[i] {
                                CALL_OPEN => Mode::Call(Box::default()),
                                THINK_OPEN => Mode::Reasoning,
                                // A bare `</think>` (Thinking-2507, duplicate close) is absorbed.
                                _ => Mode::Text,
                            };
                        }
                        None => {
                            out.text(self.hb.take_safe(TEXT_MARKERS));
                            return;
                        }
                    }
                }
                Mode::Reasoning => {
                    trim_lead(&mut self.hb, &mut self.skip_nl);
                    match self.hb.find(REASONING_MARKERS) {
                        Some((at, i)) => {
                            out.reasoning(self.hb.take(at).trim_end_matches('\n'));
                            self.hb.skip(REASONING_MARKERS[i].len());
                            self.skip_nl = true;
                            // `<tool_call>` implicitly ends reasoning.
                            self.mode = if REASONING_MARKERS[i] == CALL_OPEN {
                                Mode::Call(Box::default())
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
                    call.body.push_str(&self.hb.take_all());
                    let closed = call.scan();
                    self.emit_progress(out);
                    let Some((at, end)) = closed else { return };
                    let Mode::Call(call) = std::mem::replace(&mut self.mode, Mode::Text) else {
                        unreachable!()
                    };
                    self.hb.push(&call.body[end..]);
                    if self.close_call(*call, at, true, out) {
                        self.completed += 1;
                    }
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
        if call.idx.is_none()
            && let Some(name) = &call.name
        {
            check_declared(&self.cfg.tools, name, out);
            out.tool_start(self.next_idx, name);
            call.idx = Some(self.next_idx);
            self.next_idx += 1;
        }
        if let (Some(idx), Some(start)) = (call.idx, call.args_start) {
            let upto = call.args_end.unwrap_or(call.pos);
            if upto > call.emitted.max(start) {
                let from = call.emitted.max(start);
                out.tool_args(idx, &call.body[from..upto]);
                call.emitted = upto;
            }
        }
    }

    /// Ends a call whose body ends at `at`. Returns whether a `ToolEnd` was emitted; a call whose
    /// name never parsed is passed through as text.
    fn close_call(
        &mut self,
        call: JsonCall,
        at: usize,
        complete: bool,
        out: &mut ParseOut,
    ) -> bool {
        let Some(name) = call.name.clone() else {
            out.text(format!("{CALL_OPEN}{}", &call.body[..at]));
            if complete {
                out.text(CALL_CLOSE);
            }
            return false;
        };
        let (raw, start) = match (call.args_start, call.args_end) {
            (Some(start), Some(end)) => (call.body[start..end].to_string(), start),
            (Some(start), None) => (call.body[start..at].to_string(), start),
            // A complete object without `arguments`: no arguments.
            (None, _) if call.outer == Outer::Done => ("{}".to_string(), 0),
            (None, _) => (String::new(), 0),
        };
        let valid = complete && is_json_object(&raw);
        let idx = match call.idx {
            Some(idx) => {
                let sent = call.emitted.saturating_sub(start).min(raw.len());
                out.tool_args(idx, &raw[sent..]);
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
        out.tool_end(idx, raw, valid);
        true
    }
}

impl SyntaxParser for HermesParser {
    fn feed(&mut self, _id: u32, text: &str, out: &mut ParseOut) {
        self.hb.push(text);
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

    fn calls(&self) -> u32 {
        self.completed
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
    /// Bytes of `body` already sent as argument deltas.
    emitted: usize,
    idx: Option<u32>,
}

impl JsonCall {
    /// Scans new body text; returns `(close offset, offset after the closer)` once
    /// `</tool_call>` appears outside a JSON string.
    fn scan(&mut self) -> Option<(usize, usize)> {
        while self.pos < self.body.len() {
            let b = self.body.as_bytes()[self.pos];
            if !self.in_str && b == b'<' {
                let rest = &self.body[self.pos..];
                if rest.starts_with(CALL_CLOSE) {
                    if self.outer == Outer::Value {
                        self.end_value(self.pos);
                    }
                    return Some((self.pos, self.pos + CALL_CLOSE.len()));
                }
                if CALL_CLOSE.starts_with(rest) {
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
                        b'}' | b']' | b',' => {
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
        let mut p = HermesParser::new(&cfg(false, false));
        let mut o = ParseOut::default();
        p.feed(0, &out, &mut o);
        assert_eq!(p.calls(), 2);
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
            0,
            "<tool_call>\n{\"name\": \"ls\", \"arguments\": {\"a\"",
            &mut o,
        );
        assert!(o.events.is_empty());
        p.feed(0, ": 1}}\n</tool_call>", &mut o);
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
        let mut p = HermesParser::new(&cfg(false, false));
        let mut o = ParseOut::default();
        p.feed(0, out, &mut o);
        p.finish(ParseEnd::Length, &mut o);
        assert_eq!(p.calls(), 0);
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
}
