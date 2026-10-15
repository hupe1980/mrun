//! The rendered prompt a generation request carries: its tokens plus the positions that reuse
//! (prefix cache, hybrid checkpoints) needs.

/// Role of a rendered message.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

/// Where a message starts in the token stream.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MessageMark {
    pub start: u32,
    pub role: Role,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct RenderedPrompt {
    pub tokens: Vec<u32>,
    /// Message starts in order; the first covers the system prompt and tools, if any.
    pub messages: Vec<MessageMark>,
    /// End of the system prompt and tool definitions (0 without them).
    pub system_tools_end: u32,
    /// Start of the generation header, where the next turn's re-rendered reply diverges;
    /// `tokens.len()` when the prompt has none.
    pub generation_start: u32,
}

impl RenderedPrompt {
    /// A prompt without message structure: no marks and no generation header.
    pub fn from_tokens(tokens: Vec<u32>) -> Self {
        let generation_start = tokens.len() as u32;
        Self {
            tokens,
            messages: Vec::new(),
            system_tools_end: 0,
            generation_start,
        }
    }

    pub fn len(&self) -> u32 {
        self.tokens.len() as u32
    }

    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }
}

/// Start of the generation header: the common token prefix of one conversation rendered without
/// and with the generation prompt.
pub fn generation_start(without_header: &[u32], with_header: &[u32]) -> u32 {
    without_header
        .iter()
        .zip(with_header)
        .take_while(|(a, b)| a == b)
        .count() as u32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chat::template::{ChatTemplate, RenderOptions};
    use crate::chat::tokenizer::test_support::byte_tokenizer;

    #[test]
    fn plain_prompts_have_no_header() {
        let prompt = RenderedPrompt::from_tokens(vec![5, 6, 7]);
        assert_eq!((prompt.len(), prompt.generation_start), (3, 3));
        assert!(prompt.messages.is_empty() && !prompt.is_empty());
    }

    #[test]
    fn generation_start_is_where_the_header_begins() {
        let template = ChatTemplate::new(
            "{%- for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n\
             {%- endfor %}{%- if add_generation_prompt %}<|im_start|>assistant\n{%- endif %}"
                .into(),
        )
        .unwrap();
        let tokenizer = byte_tokenizer();
        let messages = serde_json::json!([{"role": "user", "content": "hi"}]);
        let render = |add_generation_prompt| {
            let options = RenderOptions {
                add_generation_prompt,
                ..RenderOptions::default()
            };
            tokenizer
                .encode(&template.render(&messages, &options).unwrap())
                .unwrap()
        };
        let (without, with) = (render(false), render(true));
        assert_eq!(generation_start(&without, &with), without.len() as u32);
        assert!(with.len() > without.len());
    }

    #[test]
    fn generation_start_stops_at_a_retokenized_boundary() {
        // The header merged with the last message token: only the common prefix counts.
        assert_eq!(generation_start(&[1, 2, 3], &[1, 2, 9, 4]), 2);
        assert_eq!(generation_start(&[], &[1]), 0);
    }
}
