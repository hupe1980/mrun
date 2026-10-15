//! Chat syntaxes: streaming parsers that split model output into text, reasoning and tool calls
//! (002 FR-004, FR-032). Each family names one [`SyntaxId`]; [`registry`] maps it to its parser
//! and template facts.

pub mod harmony;
pub mod hermes;
pub mod holdback;
pub mod qwen_xml;
pub mod registry;

use crate::api::model::{GenEvent, ToolDef};
pub use mrun_engine::SyntaxId;

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
    /// A Harmony message to a built-in recipient (`browser.*`, `python`); dropped.
    BuiltinRecipient(String),
}

/// Parser output: API events in order, plus log entries.
#[derive(Debug, Default)]
pub struct ParseOut {
    /// Only `TextDelta`, `ReasoningDelta`, `ToolStart`, `ToolArgsDelta` and `ToolEnd`.
    /// `ToolStart.id` is left empty: encoders assign ids in each API's style.
    pub events: Vec<GenEvent>,
    pub logs: Vec<ParseLog>,
}

/// A streaming parser for one request's output. Input is the lossless token stream of 001
/// (`Event::Token { id, text }`): every token id with the text it made visible.
pub trait SyntaxParser: Send {
    fn feed(&mut self, id: u32, text: &str, out: &mut ParseOut);
    /// Releases held-back text and closes open blocks. An open tool call at `Length` ends with
    /// `ToolEnd { valid: false }` carrying what arrived.
    fn finish(&mut self, end: ParseEnd, out: &mut ParseOut);
    /// Number of tool calls completed so far.
    fn calls(&self) -> u32;
}
