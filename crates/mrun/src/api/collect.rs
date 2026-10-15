//! Folds a `GenEvent` stream into one result, for non-streaming responses of every API.

use crate::api::model::{ApiError, Finish, GenEvent, Usage};

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Call {
    pub id: String,
    pub name: String,
    pub raw_args: String,
    pub valid: bool,
}

/// Output in the order it was produced.
#[derive(Clone, Debug, PartialEq)]
pub enum Block {
    Reasoning(String),
    Text(String),
    Call(Call),
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Collected {
    pub blocks: Vec<Block>,
    /// Prompt tokens, of which `cached` were reused.
    pub input: u32,
    pub cached: u32,
    pub max_tokens: u32,
    pub finish: Option<(Finish, Option<String>, Usage)>,
    pub error: Option<ApiError>,
}

impl Collected {
    pub fn push(&mut self, event: GenEvent) {
        match event {
            GenEvent::Started {
                input,
                cached,
                max_tokens,
            } => {
                self.input = input;
                self.cached = cached;
                self.max_tokens = max_tokens;
            }
            GenEvent::ReasoningDelta(text) => match self.blocks.last_mut() {
                Some(Block::Reasoning(r)) => r.push_str(&text),
                _ => self.blocks.push(Block::Reasoning(text)),
            },
            GenEvent::TextDelta(text) => match self.blocks.last_mut() {
                Some(Block::Text(t)) => t.push_str(&text),
                _ => self.blocks.push(Block::Text(text)),
            },
            GenEvent::ToolStart { id, name, .. } => self.blocks.push(Block::Call(Call {
                id,
                name,
                ..Call::default()
            })),
            GenEvent::ToolArgsDelta { .. } => {}
            GenEvent::ToolEnd {
                raw_args, valid, ..
            } => {
                if let Some(Block::Call(call)) = self
                    .blocks
                    .iter_mut()
                    .rev()
                    .find(|b| matches!(b, Block::Call(_)))
                {
                    call.raw_args = raw_args;
                    call.valid = valid;
                }
            }
            GenEvent::Finished {
                reason,
                stop_seq,
                usage,
            } => self.finish = Some((reason, stop_seq, usage)),
            GenEvent::Failed(error) => self.error = Some(error),
        }
    }

    pub fn text(&self) -> String {
        self.blocks
            .iter()
            .filter_map(|b| match b {
                Block::Text(t) => Some(t.as_str()),
                _ => None,
            })
            .collect()
    }

    pub fn reasoning(&self) -> String {
        self.blocks
            .iter()
            .filter_map(|b| match b {
                Block::Reasoning(t) => Some(t.as_str()),
                _ => None,
            })
            .collect()
    }

    pub fn calls(&self) -> impl Iterator<Item = &Call> {
        self.blocks.iter().filter_map(|b| match b {
            Block::Call(c) => Some(c),
            _ => None,
        })
    }
}
