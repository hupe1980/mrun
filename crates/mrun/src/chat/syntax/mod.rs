//! Chat syntaxes: streaming parsers that split model output into text, reasoning and tool calls
//! (002 FR-004, FR-032). Each family names one [`SyntaxId`]; [`registry`] maps it to its parser
//! and template facts.

pub mod hermes;
pub mod holdback;
pub mod qwen_xml;
pub mod registry;

use crate::api::model::{GenEvent, ToolDef};
pub use holdback::Specials;
pub use mrun_engine::SyntaxId;
use std::sync::Arc;

/// Id fed with text that no token produced (stop-sequence hold-back released at the end).
pub const NO_TOKEN: u32 = u32::MAX;

/// Per-request parser settings.
#[derive(Clone, Debug, Default)]
pub struct ParserConfig {
    /// Declared tools (names, schemas for typed XML parameters).
    pub tools: Vec<ToolDef>,
    /// The prompt ends inside an open reasoning block (`<think>` in the generation prompt).
    pub reasoning_open: bool,
    /// Buffer each call's arguments until it ends (Anthropic, FR-004 fallback); otherwise raw
    /// JSON arguments stream as they arrive.
    pub buffer_args: bool,
    /// The vocabulary's added tokens: markers that are added tokens count only where the model
    /// emitted them as tokens (D032). `None`: markers are recognized by text.
    pub specials: Option<Arc<Specials>>,
}

/// Why output ended, as far as the parser cares.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ParseEnd {
    /// EOS, a stop token or a stop sequence.
    Stop,
    /// Output limit reached: an open call is incomplete, never reported as complete.
    Length,
}

/// A server-log entry (never in-band, except `tool-args-invalid` which the encoder raises from
/// `ToolEnd { valid: false }`).
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ParseLog {
    /// A call named a tool the request did not declare; passed through.
    ToolUnknown(String),
}

/// Parser output: API events in order, plus log entries.
#[derive(Debug, Default)]
pub struct ParseOut {
    /// Only `TextDelta`, `ReasoningDelta`, `ToolStart`, `ToolArgsDelta` and `ToolEnd`.
    /// `ToolStart.id` is left empty: encoders assign ids in each API's style.
    pub events: Vec<GenEvent>,
    pub logs: Vec<ParseLog>,
}

impl ParseOut {
    pub(crate) fn text(&mut self, text: impl Into<String>) {
        let text = text.into();
        if !text.is_empty() {
            self.events.push(GenEvent::TextDelta(text));
        }
    }

    pub(crate) fn reasoning(&mut self, text: impl Into<String>) {
        let text = text.into();
        if !text.is_empty() {
            self.events.push(GenEvent::ReasoningDelta(text));
        }
    }

    pub(crate) fn tool_start(&mut self, idx: u32, name: &str) {
        self.events.push(GenEvent::ToolStart {
            idx,
            id: String::new(),
            name: name.to_string(),
        });
    }

    pub(crate) fn tool_args(&mut self, idx: u32, json: &str) {
        if !json.is_empty() {
            self.events.push(GenEvent::ToolArgsDelta {
                idx,
                json: json.to_string(),
            });
        }
    }

    pub(crate) fn tool_end(&mut self, idx: u32, raw_args: String, valid: bool) {
        self.events.push(GenEvent::ToolEnd {
            idx,
            raw_args,
            valid,
        });
    }
}

/// Event lifecycle of one tool call: `ToolStart` when it opens (at its end when arguments are
/// buffered), argument deltas, `ToolEnd`. Deltas always concatenate to the final `raw_args`.
#[derive(Debug, Default)]
pub(crate) struct CallSink {
    idx: Option<u32>,
    /// Bytes of the arguments sent as deltas.
    sent: usize,
}

impl CallSink {
    pub fn is_open(&self) -> bool {
        self.idx.is_some()
    }

    /// Emits `ToolStart` (once) with the next call index; logs `unenforced tools` for a name the
    /// request did not declare (the call is still passed through, FR-004).
    pub fn open(&mut self, next: &mut u32, tools: &[ToolDef], name: &str, out: &mut ParseOut) {
        if self.idx.is_some() {
            return;
        }
        if !tools.iter().any(|tool| tool.name == name) {
            out.logs.push(ParseLog::ToolUnknown(name.to_string()));
        }
        out.tool_start(*next, name);
        self.idx = Some(*next);
        *next += 1;
    }

    /// Sends what `args` (the arguments so far) adds to the deltas, once the call is open.
    pub fn stream(&mut self, args: &str, out: &mut ParseOut) {
        if let Some(idx) = self.idx
            && args.len() > self.sent
            && args.is_char_boundary(self.sent)
        {
            out.tool_args(idx, &args[self.sent..]);
            self.sent = args.len();
        }
    }

    /// Opens the call if needed, sends the rest of `raw_args` and ends it.
    pub fn close(
        mut self,
        next: &mut u32,
        tools: &[ToolDef],
        name: &str,
        raw_args: String,
        valid: bool,
        out: &mut ParseOut,
    ) {
        self.open(next, tools, name, out);
        debug_assert!(
            raw_args.len() >= self.sent,
            "deltas must prefix the arguments"
        );
        self.stream(&raw_args, out);
        let idx = self.idx.expect("opened above");
        out.tool_end(idx, raw_args, valid);
    }
}

/// Whether `raw` parses as a JSON object.
pub(crate) fn is_json_object(raw: &str) -> bool {
    serde_json::from_str::<serde_json::Map<String, serde_json::Value>>(raw).is_ok()
}

/// A streaming parser for one request's output. Input is the lossless token stream of 001
/// (`Event::Token { id, text, held }`): every token id with the text it made visible and the
/// bytes the stop matcher still holds; released text comes with [`NO_TOKEN`].
pub trait SyntaxParser: Send {
    fn feed(&mut self, id: u32, text: &str, held: u32, out: &mut ParseOut);
    /// Releases held-back text and closes open blocks. An open tool call at `Length` ends with
    /// `ToolEnd { valid: false }` carrying what arrived.
    fn finish(&mut self, end: ParseEnd, out: &mut ParseOut);
}

/// Shared fixtures for the parser tests: feeding split text and normalizing event streams.
#[cfg(test)]
pub(crate) mod test_util {
    use super::*;

    /// Feeds `chunks` in order and finishes with `end`.
    pub fn run_chunks(
        parser: &mut dyn SyntaxParser,
        chunks: &[&str],
        end: ParseEnd,
    ) -> (Vec<GenEvent>, Vec<ParseLog>) {
        let mut out = ParseOut::default();
        for chunk in chunks {
            parser.feed(NO_TOKEN, chunk, 0, &mut out);
        }
        parser.finish(end, &mut out);
        let events = normalize(out.events);
        assert_deltas_match(&events);
        (events, out.logs)
    }

    /// Streamed argument deltas of each call concatenate to its final arguments.
    pub fn assert_deltas_match(events: &[GenEvent]) {
        for event in events {
            if let GenEvent::ToolEnd { idx, raw_args, .. } = event {
                let sent: String = events
                    .iter()
                    .filter_map(|e| match e {
                        GenEvent::ToolArgsDelta { idx: i, json } if i == idx => Some(json.as_str()),
                        _ => None,
                    })
                    .collect();
                assert_eq!(&sent, raw_args, "deltas of call {idx}");
            }
        }
    }

    /// Merges adjacent deltas of the same kind (and call), so streams split differently compare
    /// equal.
    pub fn normalize(events: Vec<GenEvent>) -> Vec<GenEvent> {
        let mut merged: Vec<GenEvent> = Vec::new();
        for event in events {
            match (merged.last_mut(), event) {
                (Some(GenEvent::TextDelta(a)), GenEvent::TextDelta(b))
                | (Some(GenEvent::ReasoningDelta(a)), GenEvent::ReasoningDelta(b)) => {
                    a.push_str(&b)
                }
                (
                    Some(GenEvent::ToolArgsDelta { idx: i, json: a }),
                    GenEvent::ToolArgsDelta { idx: j, json: b },
                ) if *i == j => a.push_str(&b),
                (_, event) => merged.push(event),
            }
        }
        merged
    }

    /// Splits `text` at the given byte offsets (snapped down to character boundaries).
    pub fn split_at(text: &str, cuts: &[usize]) -> Vec<String> {
        let mut pieces = Vec::new();
        let mut start = 0;
        for &cut in cuts {
            let mut cut = cut.min(text.len());
            while !text.is_char_boundary(cut) {
                cut -= 1;
            }
            if cut > start {
                pieces.push(text[start..cut].to_string());
                start = cut;
            }
        }
        pieces.push(text[start..].to_string());
        pieces
    }

    /// Asserts that every two-way split, a char-by-char feed and seeded random splits of
    /// `text` produce the same events and logs as one-shot parsing.
    pub fn assert_split_invariant(
        make: &dyn Fn() -> Box<dyn SyntaxParser>,
        text: &str,
        end: ParseEnd,
    ) -> (Vec<GenEvent>, Vec<ParseLog>) {
        let whole = run_chunks(make().as_mut(), &[text], end);
        let check = |pieces: Vec<String>| {
            let refs: Vec<&str> = pieces.iter().map(String::as_str).collect();
            let got = run_chunks(make().as_mut(), &refs, end);
            assert_eq!(got, whole, "split {refs:?}");
        };
        for cut in 1..text.len() {
            if text.is_char_boundary(cut) {
                check(split_at(text, &[cut]));
            }
        }
        check(text.chars().map(String::from).collect());
        let mut seed = 0x2545_f491_4f6c_dd1d_u64;
        for _ in 0..64 {
            let mut cuts = Vec::new();
            let mut at = 0;
            while at < text.len() {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                at += 1 + (seed % 7) as usize;
                cuts.push(at);
            }
            check(split_at(text, &cuts));
        }
        whole
    }

    /// The byte tokenizer's added tokens, as parsers see them.
    pub fn specials() -> Arc<Specials> {
        let tokenizer = crate::chat::tokenizer::test_support::byte_tokenizer();
        Arc::new(Specials::new(tokenizer.added_tokens()))
    }

    /// Feeds `text` as the model's tokens: `[[name]]` is the added token `name`, everything else
    /// ordinary text (one token per character), so spelled markers stay text.
    pub fn run_tokens(
        parser: &mut dyn SyntaxParser,
        text: &str,
        end: ParseEnd,
    ) -> (Vec<GenEvent>, Vec<ParseLog>) {
        let tokenizer = crate::chat::tokenizer::test_support::byte_tokenizer();
        let mut out = ParseOut::default();
        let mut rest = text;
        while !rest.is_empty() {
            if let Some(after) = rest.strip_prefix("[[")
                && let Some(close) = after.find("]]")
            {
                let name = &after[..close];
                let id = tokenizer.token_id(name).expect("an added token");
                parser.feed(id, name, 0, &mut out);
                rest = &after[close + 2..];
                continue;
            }
            let ch = rest.chars().next().unwrap();
            parser.feed(0, &rest[..ch.len_utf8()], 0, &mut out);
            rest = &rest[ch.len_utf8()..];
        }
        parser.finish(end, &mut out);
        let events = normalize(out.events);
        assert_deltas_match(&events);
        (events, out.logs)
    }

    pub fn tool(name: &str, schema: serde_json::Value) -> ToolDef {
        ToolDef {
            name: name.to_string(),
            description: String::new(),
            schema,
            strict: false,
            ..Default::default()
        }
    }

    pub fn text(s: &str) -> GenEvent {
        GenEvent::TextDelta(s.to_string())
    }

    pub fn reasoning(s: &str) -> GenEvent {
        GenEvent::ReasoningDelta(s.to_string())
    }

    pub fn start(idx: u32, name: &str) -> GenEvent {
        GenEvent::ToolStart {
            idx,
            id: String::new(),
            name: name.to_string(),
        }
    }

    pub fn args(idx: u32, json: &str) -> GenEvent {
        GenEvent::ToolArgsDelta {
            idx,
            json: json.to_string(),
        }
    }

    pub fn end(idx: u32, raw_args: &str, valid: bool) -> GenEvent {
        GenEvent::ToolEnd {
            idx,
            raw_args: raw_args.to_string(),
            valid,
        }
    }
}
