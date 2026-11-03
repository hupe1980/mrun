//! Qwen XML syntax (Qwen3-Coder, Qwen3.6): `<think>` reasoning as Hermes and calls written as
//! `<tool_call>\n<function=NAME>\n<parameter=K>\nVALUE\n</parameter>\n…</function>\n</tool_call>`
//! (research §3).
//!
//! JSON arguments are synthesized incrementally: string-typed parameters stream JSON-escaped; a
//! parameter without a declared type (mlx-lm #1910) is JSON when its value is an object or array
//! (buffered from the leading `{` or `[`) and streamed text otherwise; other types are buffered,
//! key included, until the value is complete and then coerced by the schema as the Qwen reference
//! parser does (booleans case-insensitive, numbers, unions, `null`; a value that fits no type
//! stays a string), so the streamed deltas always concatenate to the final arguments. A value ends at
//! `\n</parameter>\n` followed by `<parameter=` or `</function>`, so argument text containing the
//! syntax's own tags round-trips; when the call closes without that (`</tool_call>`, or the end
//! of output), the rest is read tolerantly: a value ends at `</parameter>`, `<parameter=` or
//! `</function>`. A `<function=NAME>` without `<tool_call>` is recognized for declared names only.

use super::hermes::{CALL_CLOSE, CALL_OPEN, THINK_CLOSE, THINK_OPEN, take_reasoning, trim_lead};
use super::holdback::{HoldBack, partial_suffix};
use super::{CallSink, ParseEnd, ParseOut, ParserConfig, SyntaxParser};
use crate::api::model::ToolDef;
use serde_json::Value;

const FUNC_OPEN: &str = "<function=";
const FUNC_CLOSE: &str = "</function>";
const PARAM_OPEN: &str = "<parameter=";
const PARAM_END: &str = "</parameter>";
const PARAM_CLOSE: &str = "\n</parameter>\n";
const VALUE_ENDS: &[&str] = &["\n</parameter>\n<parameter=", "\n</parameter>\n</function>"];
/// Tags that end a value in the tolerant reading.
const LOOSE_ENDS: &[&str] = &[PARAM_END, PARAM_OPEN, FUNC_CLOSE];
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
}

/// How a parameter value travels into the JSON arguments.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Kind {
    /// A string by schema: streams JSON-escaped.
    Stream,
    /// Declared non-string types: buffered, then coerced.
    Typed,
    /// No declared type, nothing but whitespace seen yet.
    Undecided,
    /// No declared type, the value starts with `{` or `[`: buffered, JSON when it parses as an
    /// object or array, else text (mlx-lm #1910).
    Json,
}

/// The parameter whose value is being read.
#[derive(Debug)]
struct Param {
    key: String,
    /// Declared non-string types (empty unless `kind` is `Typed`).
    types: Vec<String>,
    kind: Kind,
    /// Buffered value text (`Typed`, `Undecided`, `Json`).
    value: String,
}

#[derive(Debug)]
struct XmlCall {
    stage: Stage,
    /// The call began with `<tool_call>` (else it was recovered from `<function=`).
    opener: bool,
    name: String,
    /// The call's text from `<function=` on: `raw_args` of an incomplete buffered call.
    raw: String,
    /// Synthesized JSON arguments: what the deltas carried so far.
    json: String,
    params: u32,
    param: Option<Param>,
    sink: CallSink,
}

/// JSON Schema types of a top-level parameter, lowercase; `type` arrays and `anyOf`/`oneOf`
/// alternatives included.
fn param_types(tool: Option<&ToolDef>, key: &str) -> Vec<String> {
    let Some(schema) = tool.and_then(|t| t.schema.get("properties")?.get(key)) else {
        return Vec::new();
    };
    let mut types = Vec::new();
    let mut add = |schema: &Value| match schema.get("type") {
        Some(Value::String(t)) => types.push(t.to_ascii_lowercase()),
        Some(Value::Array(list)) => types.extend(
            list.iter()
                .filter_map(Value::as_str)
                .map(str::to_ascii_lowercase),
        ),
        _ => {}
    };
    add(schema);
    for key in ["anyOf", "oneOf"] {
        for alternative in schema
            .get(key)
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            add(alternative);
        }
    }
    types
}

/// How a value of these declared types is read: no type is `Undecided` (text unless it is a
/// JSON object or array, mlx-lm #1910), a `string` alternative streams.
fn kind_of(types: &[String]) -> Kind {
    if types.is_empty() {
        Kind::Undecided
    } else if types.iter().any(|t| t == "string") {
        Kind::Stream
    } else {
        Kind::Typed
    }
}

/// A typed value as the Qwen reference parser reads it; text that fits no declared type stays a
/// string.
pub(crate) fn coerce(text: &str, types: &[String]) -> Value {
    let t = text.trim();
    for ty in types {
        let value = match ty.as_str() {
            "null" if t.eq_ignore_ascii_case("null") || t == "None" => Some(Value::Null),
            "boolean" if t.eq_ignore_ascii_case("true") => Some(Value::Bool(true)),
            "boolean" if t.eq_ignore_ascii_case("false") => Some(Value::Bool(false)),
            "integer" => t.parse::<i64>().ok().map(Value::from).or_else(|| {
                t.parse::<f64>()
                    .ok()
                    .filter(|f| f.fract() == 0.0 && f.abs() < 9.0e15)
                    .map(|f| Value::from(f as i64))
            }),
            "number" => t.parse::<i64>().ok().map(Value::from).or_else(|| {
                t.parse::<f64>()
                    .ok()
                    .and_then(serde_json::Number::from_f64)
                    .map(Value::Number)
            }),
            "object" => serde_json::from_str::<Value>(t)
                .ok()
                .filter(Value::is_object),
            "array" => serde_json::from_str::<Value>(t)
                .ok()
                .filter(Value::is_array),
            _ => None,
        };
        if let Some(value) = value {
            return value;
        }
    }
    if types.iter().any(|t| t == "null") && t.eq_ignore_ascii_case("null") {
        return Value::Null;
    }
    Value::String(text.to_string())
}

impl XmlCall {
    fn new(opener: bool) -> Self {
        Self {
            stage: Stage::AwaitFunction,
            opener,
            name: String::new(),
            raw: String::new(),
            json: String::new(),
            params: 0,
            param: None,
            sink: CallSink::default(),
        }
    }

    fn separator(&self) -> &'static str {
        if self.params == 0 { "{" } else { ", " }
    }

    fn start_param(&mut self, key: &str, tool: Option<&ToolDef>) {
        let types = param_types(tool, key);
        let kind = kind_of(&types);
        if kind == Kind::Stream {
            self.open_string(key);
        }
        self.param = Some(Param {
            key: key.to_string(),
            types: if kind == Kind::Typed {
                types
            } else {
                Vec::new()
            },
            kind,
            value: String::new(),
        });
    }

    /// Writes the key and the opening quote of a streamed string value.
    fn open_string(&mut self, key: &str) {
        self.json.push_str(self.separator());
        self.json.push_str(&json_string(key));
        self.json.push_str(": \"");
        self.params += 1;
    }

    /// Writes a complete value.
    fn put_value(&mut self, key: &str, value: &str) {
        self.json.push_str(self.separator());
        self.json.push_str(&json_string(key));
        self.json.push_str(": ");
        self.json.push_str(value);
        self.params += 1;
    }

    /// Value text: to the raw call text and the open parameter.
    fn value_chunk(&mut self, chunk: &str) {
        self.raw.push_str(chunk);
        self.append_value(chunk);
    }

    fn append_value(&mut self, chunk: &str) {
        let Some(param) = &mut self.param else {
            return;
        };
        match param.kind {
            Kind::Stream => push_escaped(&mut self.json, chunk),
            Kind::Typed | Kind::Json => param.value.push_str(chunk),
            Kind::Undecided => {
                param.value.push_str(chunk);
                match param.value.trim_start().chars().next() {
                    None => {}
                    Some('{' | '[') => param.kind = Kind::Json,
                    Some(_) => {
                        // Text: stream it from here on.
                        param.kind = Kind::Stream;
                        let (key, value) = (param.key.clone(), std::mem::take(&mut param.value));
                        self.open_string(&key);
                        push_escaped(&mut self.json, &value);
                    }
                }
            }
        }
    }

    fn end_value(&mut self) {
        let Some(param) = self.param.take() else {
            return;
        };
        let value = match param.kind {
            Kind::Stream => {
                self.json.push('"');
                return;
            }
            Kind::Undecided => json_string(&param.value),
            Kind::Json => match serde_json::from_str::<Value>(param.value.trim()) {
                // JSON as the model wrote it.
                Ok(Value::Object(_) | Value::Array(_)) => param.value.trim().to_string(),
                _ => json_string(&param.value),
            },
            Kind::Typed => match coerce(&param.value, &param.types) {
                Value::Object(_) | Value::Array(_) => param.value.trim().to_string(),
                other => other.to_string(),
            },
        };
        self.put_value(&param.key, &value);
    }

    fn end_function(&mut self) {
        self.end_value();
        self.json
            .push_str(if self.params == 0 { "{}" } else { "}" });
    }

    /// Reads the rest of a call that closed without the strict terminators: the open value
    /// ends at the first loose end, then parameters follow until `</function>`. Returns whether
    /// `</function>` was seen.
    fn read_loose(&mut self, text: &str, tool: Option<&ToolDef>) -> bool {
        self.raw.push_str(text);
        let mut rest = text;
        if self.param.is_some() {
            rest = self.loose_value(rest);
        }
        loop {
            rest = rest.trim_start();
            if let Some(after) = rest.strip_prefix(PARAM_END) {
                rest = after;
            } else if rest.starts_with(PARAM_OPEN) {
                let Some(Some((key, len))) = tag_name(rest, PARAM_OPEN) else {
                    break;
                };
                let key = key.to_string();
                self.start_param(&key, tool);
                let value = &rest[len..];
                rest = self.loose_value(value.strip_prefix('\n').unwrap_or(value));
            } else if rest.starts_with(FUNC_CLOSE) {
                return true;
            } else if let Some(ch) = rest.chars().next() {
                rest = &rest[ch.len_utf8()..];
            } else {
                return false;
            }
        }
        false
    }

    /// The open value up to the first loose end (one trailing newline dropped); returns the rest.
    fn loose_value<'a>(&mut self, text: &'a str) -> &'a str {
        let end = LOOSE_ENDS
            .iter()
            .filter_map(|tag| text.find(tag))
            .min()
            .unwrap_or(text.len());
        let value = &text[..end];
        self.append_value(value.strip_suffix('\n').unwrap_or(value));
        self.end_value();
        &text[end..]
    }
}

fn json_string(text: &str) -> String {
    serde_json::Value::from(text).to_string()
}

/// Appends `text` JSON-escaped, without quotes.
fn push_escaped(json: &mut String, text: &str) {
    let escaped = json_string(text);
    json.push_str(&escaped[1..escaped.len() - 1]);
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
            hb: HoldBack::new(cfg.specials.clone()),
            skip_nl: cfg.reasoning_open,
            next_idx: 0,
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
            if let Mode::Call(call) = &mut self.mode {
                call.sink.stream(&call.json, out);
            }
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
        let tool = self.cfg.tools.iter().find(|tool| tool.name == call.name);
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
                            call.sink
                                .open(&mut self.next_idx, &self.cfg.tools, &call.name, out);
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
                            call.start_param(&key, tool);
                            call.stage = Stage::Value { nl: true };
                            return true;
                        }
                        None => {}
                    }
                } else if body.starts_with(FUNC_CLOSE) {
                    call.raw.push_str(&s[..ws + FUNC_CLOSE.len()]);
                    self.hb.skip(ws + FUNC_CLOSE.len());
                    call.end_function();
                    call.stage = Stage::AfterFunction;
                    return true;
                } else if self.hb.marker_at(ws, CALL_CLOSE) {
                    // `</tool_call>` before `</function>`: read as closing the function.
                    self.hb.skip(ws + CALL_CLOSE.len());
                    call.end_function();
                    self.close(true, out);
                    return true;
                } else if [PARAM_OPEN, FUNC_CLOSE, CALL_CLOSE]
                    .iter()
                    .any(|tag| tag.starts_with(body))
                {
                    return false;
                }
                // Stray text between parameters is skipped.
                let n = ws + body.chars().next().map_or(0, char::len_utf8);
                call.raw.push_str(&s[..n]);
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
            Stage::Value { nl: false } => {
                let strict = self.hb.find(VALUE_ENDS).map(|(at, _)| at);
                let closer = self.hb.find(&[CALL_CLOSE]).map(|(at, _)| at);
                match (strict, closer) {
                    (Some(at), _) if closer.is_none_or(|c| at < c) => {
                        let chunk = self.hb.take(at);
                        call.value_chunk(&chunk);
                        call.end_value();
                        call.raw.push_str(PARAM_CLOSE);
                        self.hb.skip(PARAM_CLOSE.len());
                        call.stage = Stage::Between;
                        true
                    }
                    (_, Some(at)) => {
                        // The call closed inside a value: read the rest tolerantly.
                        let rest = self.hb.take(at);
                        self.hb.skip(CALL_CLOSE.len());
                        call.read_loose(&rest, tool);
                        call.end_function();
                        self.close(true, out);
                        true
                    }
                    _ => {
                        // Stream up to a possible end; text from a loose end on waits until
                        // the value's end is known.
                        let text = self.hb.as_str();
                        let ends = [VALUE_ENDS[0], VALUE_ENDS[1], CALL_CLOSE];
                        let mut safe = text.len()
                            - partial_suffix(text, &ends).max(partial_suffix(text, LOOSE_ENDS));
                        if let Some(loose) = LOOSE_ENDS.iter().filter_map(|t| text.find(t)).min() {
                            safe = safe.min(loose);
                        }
                        // A newline right before held text may belong to the tag.
                        if safe < text.len() && text[..safe].ends_with('\n') {
                            safe -= 1;
                        }
                        let chunk = self.hb.take(safe);
                        call.value_chunk(&chunk);
                        false
                    }
                }
            }
            Stage::AfterFunction => {
                if body.is_empty() || CALL_CLOSE.starts_with(body) && body.len() < CALL_CLOSE.len()
                {
                    return false;
                }
                if self.hb.marker_at(ws, CALL_CLOSE) {
                    self.hb.skip(ws + CALL_CLOSE.len());
                } else {
                    self.hb.skip(ws);
                }
                self.close(true, out);
                true
            }
        }
    }

    /// Ends the open call: complete ones carry the synthesized JSON; an incomplete one what its
    /// deltas carried, or (buffered) the raw call text.
    fn close(&mut self, complete: bool, out: &mut ParseOut) {
        let Mode::Call(call) = std::mem::replace(&mut self.mode, Mode::Text) else {
            return;
        };
        self.skip_nl = true;
        let raw_args = if complete || call.sink.is_open() {
            call.json
        } else {
            call.raw
        };
        call.sink.close(
            &mut self.next_idx,
            &self.cfg.tools,
            &call.name,
            raw_args,
            complete,
            out,
        );
    }
}

impl SyntaxParser for QwenXmlParser {
    fn feed(&mut self, id: u32, text: &str, held: u32, out: &mut ParseOut) {
        self.hb.push_token(id, text, held);
        self.run(out);
    }

    fn finish(&mut self, end: ParseEnd, out: &mut ParseOut) {
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
                // A call cut by the output limit is incomplete; at a stop, one whose function
                // closed is read tolerantly.
                Stage::Between | Stage::Value { .. } => {
                    let rest = self.hb.take_all();
                    let tool = self.cfg.tools.iter().find(|tool| tool.name == call.name);
                    if end == ParseEnd::Stop && loose_closes(&rest, call.stage) {
                        call.read_loose(&rest, tool);
                        call.end_function();
                        call.sink.stream(&call.json, out);
                        self.close(true, out);
                    } else {
                        call.raw.push_str(&rest);
                        self.close(false, out);
                    }
                }
            },
        }
    }
}

/// Whether the tolerant reading of `rest` reaches `</function>`.
fn loose_closes(rest: &str, stage: Stage) -> bool {
    let mut probe = XmlCall::new(true);
    if matches!(stage, Stage::Value { .. }) {
        probe.param = Some(Param {
            key: String::new(),
            types: Vec::new(),
            kind: Kind::Stream,
            value: String::new(),
        });
    }
    probe.read_loose(rest, None)
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
            specials: None,
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
        let body = call("edit", &[("text", v), ("path", "p")]);
        let body = body
            .strip_prefix("<tool_call>")
            .unwrap()
            .strip_suffix("</tool_call>")
            .unwrap();
        // The model's own tokens open and close the call; the spelled tags are text.
        let c = ParserConfig {
            specials: Some(specials()),
            ..cfg(false, false)
        };
        let (events, _) = run_tokens(
            &mut QwenXmlParser::new(&c),
            &format!("[[<tool_call>]]{body}[[</tool_call>]]"),
            ParseEnd::Stop,
        );
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
    fn syn_qwen_xml_values_coerced_by_schema() {
        for (bad, value) in [
            ("True", json!(true)),
            ("FALSE", json!(false)),
            ("yes", json!("yes")),
            ("1 == 1", json!("1 == 1")),
            ("'x'", json!("'x'")),
        ] {
            let out = call("edit", &[("force", bad)]);
            let (events, _) = check(cfg(false, true), &out, ParseEnd::Stop);
            let (raw, valid) = one_call(&events);
            assert!(valid, "{bad}");
            let parsed: serde_json::Value = serde_json::from_str(&raw).unwrap();
            assert_eq!(parsed, json!({"force": value}), "{bad}");
        }
        let types = |t: &[&str]| t.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        assert_eq!(coerce(" 42 ", &types(&["integer"])), json!(42));
        assert_eq!(coerce("4.0", &types(&["integer"])), json!(4));
        assert_eq!(coerce("2.5", &types(&["number"])), json!(2.5));
        assert_eq!(coerce("7", &types(&["number"])), json!(7));
        assert_eq!(coerce("null", &types(&["integer", "null"])), json!(null));
        assert_eq!(coerce("None", &types(&["object", "null"])), json!(null));
        assert_eq!(coerce("3", &types(&["boolean", "integer"])), json!(3));
        assert_eq!(coerce("[1, 2]", &types(&["array"])), json!([1, 2]));
        assert_eq!(coerce("[1, 2", &types(&["array"])), json!("[1, 2"));
    }

    #[test]
    fn syn_qwen_xml_untyped_param_json_if_object_or_array_else_text() {
        let c = ParserConfig {
            tools: vec![tool(
                "t",
                json!({"properties": {"u": {"description": "any"}}}),
            )],
            ..cfg(false, false)
        };
        for (value, expected) in [
            ("{\"a\": [1, true]}", json!({"a": [1, true]})),
            ("[1, \"x\"]", json!([1, "x"])),
            ("42", json!("42")),
            ("true", json!("true")),
            ("\"quoted\"", json!("\"quoted\"")),
            ("{not json", json!("{not json")),
            ("plain text", json!("plain text")),
        ] {
            // A declared tool without a type, and an undeclared parameter.
            for key in ["u", "other"] {
                for buffer in [false, true] {
                    let c = ParserConfig {
                        buffer_args: buffer,
                        ..c.clone()
                    };
                    let (events, _) = check(c, &call("t", &[(key, value)]), ParseEnd::Stop);
                    let (raw, valid) = one_call(&events);
                    assert!(valid, "{value}");
                    let parsed: Value = serde_json::from_str(&raw).unwrap();
                    assert_eq!(parsed, json!({ key: expected.clone() }), "{value}");
                }
            }
        }
    }

    #[test]
    fn syn_qwen_xml_untyped_json_values_are_buffered() {
        let c = ParserConfig {
            tools: vec![tool("t", json!({}))],
            ..cfg(false, false)
        };
        let mut p = QwenXmlParser::new(&c);
        let mut o = ParseOut::default();
        p.feed(
            super::super::NO_TOKEN,
            "<tool_call>\n<function=t>\n<parameter=a>\ntext\n</parameter>\n<parameter=b>\n{\"k\": ",
            0,
            &mut o,
        );
        // `a` streamed as text, `b` (an object so far) held back.
        assert_eq!(
            normalize(o.events),
            vec![start(0, "t"), args(0, r#"{"a": "text""#)]
        );
    }

    #[test]
    fn syn_qwen_xml_union_types_from_any_of() {
        let tool = tool(
            "t",
            json!({"properties": {"n": {"anyOf": [{"type": "integer"}, {"type": "null"}]}}}),
        );
        assert_eq!(param_types(Some(&tool), "n"), vec!["integer", "null"]);
        assert!(param_types(Some(&tool), "m").is_empty());
    }

    #[test]
    fn syn_qwen_xml_tolerant_termination() {
        let c = ParserConfig {
            specials: Some(specials()),
            ..cfg(false, false)
        };
        for (body, expected) in [
            // `</parameter>` without its newlines.
            (
                "<function=edit>\n<parameter=path>\na.rs</parameter><parameter=line>\n3</parameter></function>",
                json!({"path": "a.rs", "line": 3}),
            ),
            // No `</parameter>` before the next parameter, no `</function>`.
            (
                "<function=edit>\n<parameter=path>\na.rs\n<parameter=force>\ntrue\n",
                json!({"path": "a.rs", "force": true}),
            ),
            // A value that ends at `</function>`.
            (
                "<function=edit>\n<parameter=path>\na.rs\n</function>\n",
                json!({"path": "a.rs"}),
            ),
        ] {
            let (events, _) = run_tokens(
                &mut QwenXmlParser::new(&c),
                &format!("[[<tool_call>]]\n{body}[[</tool_call>]]"),
                ParseEnd::Stop,
            );
            let (raw, valid) = one_call(&events);
            assert!(valid, "{body}");
            let parsed: serde_json::Value = serde_json::from_str(&raw).unwrap();
            assert_eq!(parsed, expected, "{body}");
        }
        // Without `</tool_call>`, a stop reads a closed function tolerantly too.
        let out = "<tool_call>\n<function=edit>\n<parameter=path>\na.rs</parameter>\n</function>";
        let (events, _) = check(cfg(false, false), out, ParseEnd::Stop);
        assert_eq!(one_call(&events), (r#"{"path": "a.rs"}"#.into(), true));
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
            super::super::NO_TOKEN,
            "<tool_call>\n<function=edit>\n<parameter=text>\nhello wor",
            0,
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
        // Streamed: the arguments are what the deltas carried.
        let (events, _) = check(cfg(false, false), out, ParseEnd::Length);
        assert_eq!(events.last(), Some(&end(0, r#"{"path": "src/ma"#, false)));
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
    fn syn_qwen_xml_marker_text_not_token_is_content() {
        let c = ParserConfig {
            specials: Some(specials()),
            ..cfg(false, false)
        };
        let prose = "Use <think> and <tool_call>\n<function=ls>\n</function>\n</tool_call> here.";
        let (events, _) = run_tokens(&mut QwenXmlParser::new(&c), prose, ParseEnd::Stop);
        // `<function=ls>` is not a token: a declared name still recovers a call (research §3),
        // so only check that the spelled `<think>` and `<tool_call>` did not switch modes.
        assert!(
            matches!(events.first(), Some(GenEvent::TextDelta(t)) if t.starts_with("Use <think> and <tool_call>"))
        );
        let (events, _) = run_tokens(
            &mut QwenXmlParser::new(&c),
            "[[<think>]]a[[</think>]]b <think>c</think>",
            ParseEnd::Stop,
        );
        assert_eq!(events, vec![reasoning("a"), text("b <think>c</think>")]);
    }

    #[test]
    fn non_xml_call_body_passes_through() {
        let out = "<tool_call>\n{\"name\": \"ls\"}\n</tool_call>";
        let (events, _) = check(cfg(false, false), out, ParseEnd::Stop);
        assert_eq!(events, vec![text(out)]);
    }
}
