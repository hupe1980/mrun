//! Hugging Face `tokenizer.json` tokenization and UTF-8-safe streaming detokenization.

use std::path::Path;

#[derive(Debug, thiserror::Error)]
#[error("tokenizer: {0}")]
pub struct TokenizerError(String);

/// Thin wrapper over `tokenizers::Tokenizer` (built without network features).
#[derive(Debug, Clone)]
pub struct Tokenizer {
    inner: tokenizers::Tokenizer,
}

impl Tokenizer {
    pub fn from_file(path: &Path) -> Result<Self, TokenizerError> {
        tokenizers::Tokenizer::from_file(path)
            .map(Self::from_inner)
            .map_err(|error| TokenizerError(format!("{}: {error}", path.display())))
    }

    pub fn from_inner(inner: tokenizers::Tokenizer) -> Self {
        Self { inner }
    }

    /// Encodes already-rendered text; special tokens in the text are recognized, none are added.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>, TokenizerError> {
        self.inner
            .encode(text, false)
            .map(|encoding| encoding.get_ids().to_vec())
            .map_err(|error| TokenizerError(error.to_string()))
    }

    pub fn token_id(&self, token: &str) -> Option<u32> {
        self.inner.token_to_id(token)
    }

    /// A streaming decoder that never emits incomplete UTF-8.
    pub fn stream(&self) -> StreamDecoder<'_> {
        StreamDecoder {
            tokenizer: self,
            ids: Vec::new(),
            prefix: String::new(),
            prefix_index: 0,
        }
    }
}

/// Incremental detokenizer: text appears only once it forms complete characters.
#[derive(Debug)]
pub struct StreamDecoder<'a> {
    tokenizer: &'a Tokenizer,
    ids: Vec<u32>,
    prefix: String,
    prefix_index: usize,
}

impl StreamDecoder<'_> {
    pub fn step(&mut self, id: u32) -> Result<Option<String>, TokenizerError> {
        tokenizers::tokenizer::step_decode_stream(
            &self.tokenizer.inner,
            vec![id],
            false,
            &mut self.ids,
            &mut self.prefix,
            &mut self.prefix_index,
        )
        .map_err(|error| TokenizerError(error.to_string()))
    }
}

/// Fixtures for unit and integration tests.
#[doc(hidden)]
pub mod test_support {
    use super::Tokenizer;
    use tokenizers::models::bpe::{BPE, Vocab};
    use tokenizers::pre_tokenizers::byte_level::ByteLevel;
    use tokenizers::{AddedToken, decoders::byte_level::ByteLevel as ByteLevelDecoder};

    /// Byte-level tokenizer with one token per byte plus `<|im_start|>`, `<|im_end|>`.
    pub fn byte_tokenizer() -> Tokenizer {
        let mut alphabet: Vec<char> = ByteLevel::alphabet().into_iter().collect();
        alphabet.sort_unstable();
        let vocab: Vocab = alphabet
            .into_iter()
            .enumerate()
            .map(|(id, ch)| (ch.to_string(), id as u32))
            .collect();
        let model = BPE::builder()
            .vocab_and_merges(vocab, Vec::new())
            .build()
            .unwrap();
        let mut inner = tokenizers::Tokenizer::new(model);
        inner.with_pre_tokenizer(Some(ByteLevel::new(false, true, false)));
        inner.with_decoder(Some(ByteLevelDecoder::default()));
        inner
            .add_special_tokens([
                AddedToken::from("<|im_start|>", true),
                AddedToken::from("<|im_end|>", true),
            ])
            .expect("add special tokens");
        Tokenizer::from_inner(inner)
    }
}

#[cfg(test)]
mod tests {
    use super::test_support::byte_tokenizer;

    #[test]
    fn encode_recognizes_special_tokens_without_adding_any() {
        let tokenizer = byte_tokenizer();
        let ids = tokenizer.encode("<|im_start|>hi").unwrap();
        assert_eq!(ids.len(), 3);
        assert_eq!(Some(ids[0]), tokenizer.token_id("<|im_start|>"));
    }

    #[test]
    fn streaming_holds_back_split_multibyte_characters() {
        let tokenizer = byte_tokenizer();
        let ids = tokenizer.encode("a€b").unwrap();
        assert_eq!(ids.len(), 5, "€ is three byte tokens");
        let mut stream = tokenizer.stream();
        let pieces: Vec<_> = ids.iter().map(|&id| stream.step(id).unwrap()).collect();
        assert_eq!(
            pieces,
            [
                Some("a".to_string()),
                None,
                None,
                Some("€".to_string()),
                Some("b".to_string())
            ]
        );
    }
}
