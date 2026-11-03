//! Hugging Face `tokenizer.json` tokenization and UTF-8-safe streaming detokenization.
//!
//! Prompts are tokenized in segments (D032): template scaffolding with special-token parsing,
//! message content without it, so text like `<|im_end|>` in a user message stays text.
//!
//! Rendered prompts are encoded turn by turn through a segment memo (FR-046, SC-009): the text
//! is split before every `<|im_start|>` (an added token without strip flags, where the encoder
//! splits anyway, so the pieces encode independently) and each piece's ids come from an LRU of
//! at most [`MEMO_BYTES`], so a follow-up request re-encodes only its new turns.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::Path;
use std::sync::{Arc, Mutex, OnceLock, PoisonError};

/// Upper bound of the segment memo: piece text, ids and offsets.
pub const MEMO_BYTES: usize = 64 * 1024 * 1024;

/// The added token prompts are split before (ChatML turn start).
const TURN_START: &str = "<|im_start|>";

#[derive(Debug, thiserror::Error)]
#[error("tokenizer: {0}")]
pub struct TokenizerError(String);

/// Byte range of a token in the encoded text.
pub type Span = (usize, usize);

/// Thin wrapper over `tokenizers::Tokenizer` (built without network features).
#[derive(Debug, Clone)]
pub struct Tokenizer {
    inner: tokenizers::Tokenizer,
    /// Built on first use: segmented encoding needs it only when content holds added tokens.
    plain: Arc<OnceLock<Result<Plain, String>>>,
    /// Whether rendered text may be split before [`TURN_START`] (decided on first use).
    splits: Arc<OnceLock<bool>>,
    memo: Arc<Mutex<Memo>>,
}

/// One memoized piece: ids and offsets relative to the piece.
#[derive(Debug)]
struct Piece {
    ids: Vec<u32>,
    offsets: Vec<Span>,
    used: u64,
}

impl Piece {
    fn bytes(&self, text: &str) -> usize {
        text.len() + self.ids.len() * 4 + self.offsets.len() * std::mem::size_of::<Span>() + 64
    }
}

/// LRU from piece text to its encoding, bounded by `limit` bytes.
#[derive(Debug)]
struct Memo {
    pieces: HashMap<String, Piece>,
    /// Use tick → piece text, oldest first.
    order: BTreeMap<u64, String>,
    tick: u64,
    bytes: usize,
    limit: usize,
    hits: u64,
}

impl Memo {
    fn new(limit: usize) -> Self {
        Self {
            pieces: HashMap::new(),
            order: BTreeMap::new(),
            tick: 0,
            bytes: 0,
            limit,
            hits: 0,
        }
    }

    fn get(&mut self, text: &str) -> Option<(Vec<u32>, Vec<Span>)> {
        let piece = self.pieces.get_mut(text)?;
        self.tick += 1;
        let key = self.order.remove(&piece.used).expect("ordered");
        piece.used = self.tick;
        self.order.insert(self.tick, key);
        self.hits += 1;
        Some((piece.ids.clone(), piece.offsets.clone()))
    }

    fn put(&mut self, text: &str, ids: &[u32], offsets: &[Span]) {
        self.tick += 1;
        let piece = Piece {
            ids: ids.to_vec(),
            offsets: offsets.to_vec(),
            used: self.tick,
        };
        let size = piece.bytes(text);
        if size > self.limit || self.pieces.contains_key(text) {
            return;
        }
        while self.bytes + size > self.limit {
            let Some((_, oldest)) = self.order.pop_first() else {
                break;
            };
            if let Some(old) = self.pieces.remove(&oldest) {
                self.bytes -= old.bytes(&oldest);
            }
        }
        self.bytes += size;
        self.order.insert(self.tick, text.to_string());
        self.pieces.insert(text.to_string(), piece);
    }
}

/// The same tokenizer without its added vocabulary, and the ids that vocabulary holds.
#[derive(Debug)]
struct Plain {
    tokenizer: tokenizers::Tokenizer,
    added: HashSet<u32>,
}

impl Tokenizer {
    pub fn from_file(path: &Path) -> Result<Self, TokenizerError> {
        tokenizers::Tokenizer::from_file(path)
            .map(Self::from_inner)
            .map_err(|error| TokenizerError(format!("{}: {error}", path.display())))
    }

    pub fn from_inner(inner: tokenizers::Tokenizer) -> Self {
        Self {
            inner,
            plain: Arc::new(OnceLock::new()),
            splits: Arc::new(OnceLock::new()),
            memo: Arc::new(Mutex::new(Memo::new(MEMO_BYTES))),
        }
    }

    /// Like [`Self::encode_with_offsets`], through the segment memo: pieces of `text` that begin
    /// at a [`TURN_START`] are looked up, and only new ones are encoded (FR-046). The result
    /// equals the fresh encoding.
    pub fn encode_memoized(&self, text: &str) -> Result<(Vec<u32>, Vec<Span>), TokenizerError> {
        if !self.splits() {
            return self.encode_with_offsets(text);
        }
        let mut cuts: Vec<usize> = text.match_indices(TURN_START).map(|(at, _)| at).collect();
        if cuts.first() != Some(&0) {
            cuts.insert(0, 0);
        }
        cuts.push(text.len());
        let (mut ids, mut offsets) = (Vec::new(), Vec::new());
        for pair in cuts.windows(2) {
            let (from, to) = (pair[0], pair[1]);
            let piece = &text[from..to];
            if piece.is_empty() {
                continue;
            }
            let cached = self
                .memo
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .get(piece);
            let (piece_ids, piece_offsets) = match cached {
                Some(found) => found,
                None => {
                    let fresh = self.encode_with_offsets(piece)?;
                    self.memo
                        .lock()
                        .unwrap_or_else(PoisonError::into_inner)
                        .put(piece, &fresh.0, &fresh.1);
                    fresh
                }
            };
            ids.extend(piece_ids);
            offsets.extend(piece_offsets.into_iter().map(|(s, e)| (s + from, e + from)));
        }
        Ok((ids, offsets))
    }

    /// Pieces served from the memo so far (tests, diagnostics).
    pub fn memo_hits(&self) -> u64 {
        self.memo
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .hits
    }

    /// Bytes the memo holds.
    pub fn memo_bytes(&self) -> usize {
        self.memo
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .bytes
    }

    /// [`TURN_START`] is an added token the encoder splits at exactly, without stripping or
    /// normalizing around it.
    fn splits(&self) -> bool {
        *self.splits.get_or_init(|| {
            let Some(id) = self.inner.token_to_id(TURN_START) else {
                return false;
            };
            self.inner
                .get_added_tokens_decoder()
                .get(&id)
                .is_some_and(|t| !t.lstrip && !t.rstrip && !t.single_word && !t.normalized)
        })
    }

    /// Like [`Self::encode`], with each token's byte range in `text`.
    pub fn encode_with_offsets(&self, text: &str) -> Result<(Vec<u32>, Vec<Span>), TokenizerError> {
        self.inner
            .encode(text, false)
            .map(|encoding| (encoding.get_ids().to_vec(), encoding.get_offsets().to_vec()))
            .map_err(|error| TokenizerError(error.to_string()))
    }

    /// Encodes already-rendered text; special tokens in the text are recognized, none are added.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>, TokenizerError> {
        self.inner
            .encode(text, false)
            .map(|encoding| encoding.get_ids().to_vec())
            .map_err(|error| TokenizerError(error.to_string()))
    }

    /// Encodes `text` with added-token parsing everywhere except inside `content` (sorted,
    /// disjoint byte ranges): an added token found inside content is re-encoded as ordinary
    /// text, together with the text around it up to the neighbouring scaffolding tokens, exactly
    /// as the whole text would encode if that token were not in the vocabulary. Everything else
    /// is the plain encoding of `text`.
    pub fn encode_segmented(
        &self,
        text: &str,
        content: &[Span],
    ) -> Result<(Vec<u32>, Vec<Span>), TokenizerError> {
        let (ids, offsets) = self.encode_memoized(text)?;
        if content.is_empty() {
            return Ok((ids, offsets));
        }
        let plain = self.plain()?;
        let inside = |(start, end): Span| {
            let k = content.partition_point(|&(_, c_end)| c_end <= start);
            content
                .get(k)
                .is_some_and(|&(c_start, c_end)| c_start <= start && end <= c_end)
        };
        let added = |k: usize| plain.added.contains(&ids[k]);
        let stray: Vec<bool> = (0..ids.len())
            .map(|k| added(k) && inside(offsets[k]))
            .collect();
        if !stray.contains(&true) {
            return Ok((ids, offsets));
        }
        let (mut out_ids, mut out_offsets) = (Vec::with_capacity(ids.len()), Vec::new());
        let mut k = 0;
        while k < ids.len() {
            if added(k) && !stray[k] {
                out_ids.push(ids[k]);
                out_offsets.push(offsets[k]);
                k += 1;
                continue;
            }
            // A run between scaffolding added tokens: the encoder splits there, so the run is
            // independent of the rest.
            let end = (k..ids.len())
                .find(|&j| added(j) && !stray[j])
                .unwrap_or(ids.len());
            if stray[k..end].contains(&true) {
                let from = if k == 0 { 0 } else { offsets[k - 1].1 };
                let to = offsets.get(end).map_or(text.len(), |o| o.0);
                let encoding = plain
                    .tokenizer
                    .encode(&text[from..to], false)
                    .map_err(|error| TokenizerError(error.to_string()))?;
                out_ids.extend_from_slice(encoding.get_ids());
                out_offsets.extend(
                    encoding
                        .get_offsets()
                        .iter()
                        .map(|&(s, e)| (s + from, e + from)),
                );
            } else {
                out_ids.extend_from_slice(&ids[k..end]);
                out_offsets.extend_from_slice(&offsets[k..end]);
            }
            k = end;
        }
        Ok((out_ids, out_offsets))
    }

    fn plain(&self) -> Result<&Plain, TokenizerError> {
        self.plain
            .get_or_init(|| {
                let json = self.inner.to_string(false).map_err(|e| e.to_string())?;
                let mut value: serde_json::Value =
                    serde_json::from_str(&json).map_err(|e| e.to_string())?;
                value["added_tokens"] = serde_json::Value::Array(Vec::new());
                let tokenizer = value
                    .to_string()
                    .parse::<tokenizers::Tokenizer>()
                    .map_err(|e| e.to_string())?;
                let added = self.inner.get_added_tokens_decoder().into_keys().collect();
                Ok(Plain { tokenizer, added })
            })
            .as_ref()
            .map_err(|error| TokenizerError(error.clone()))
    }

    /// Added tokens (`<|im_end|>`, `<tool_call>`, ...) as `(id, text)`.
    pub fn added_tokens(&self) -> Vec<(u32, String)> {
        self.inner
            .get_added_tokens_decoder()
            .into_iter()
            .map(|(id, token)| (id, token.content))
            .collect()
    }

    /// The Hugging Face tokenizer (vocabulary access for grammar engines).
    pub fn inner(&self) -> &tokenizers::Tokenizer {
        &self.inner
    }

    pub fn token_id(&self, token: &str) -> Option<u32> {
        self.inner.token_to_id(token)
    }

    /// Vocabulary size including added tokens.
    pub fn vocab_size(&self) -> u32 {
        self.inner.get_vocab_size(true) as u32
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
#[cfg(any(test, feature = "test-support"))]
pub mod test_support {
    use super::Tokenizer;
    use tokenizers::models::bpe::{BPE, Vocab};
    use tokenizers::pre_tokenizers::byte_level::ByteLevel;
    use tokenizers::{AddedToken, decoders::byte_level::ByteLevel as ByteLevelDecoder};

    /// Added tokens of the byte tokenizer: ChatML, Qwen tool and thinking tags.
    pub const SPECIAL: [&str; 8] = [
        "<|im_start|>",
        "<|im_end|>",
        "<tool_call>",
        "</tool_call>",
        "<tool_response>",
        "</tool_response>",
        "<think>",
        "</think>",
    ];

    /// Byte-level tokenizer with one token per byte plus the [`SPECIAL`] added tokens.
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
            .add_special_tokens(SPECIAL.map(|token| AddedToken::from(token, true)))
            .expect("add special tokens");
        Tokenizer::from_inner(inner)
    }

    /// Text the BPE fixture is trained on: ChatML scaffolding, tool calls and agent prose, so
    /// prompts and replies merge into multi-byte tokens as with a real vocabulary.
    const BPE_CORPUS: &[&str] = &[
        "<|im_start|>system\nYou are a coding agent.<|im_end|>\n",
        "<|im_start|>user\nopen a.rs<|im_end|>\n<|im_start|>assistant\n",
        "<think>\nLook at a.rs.\n</think>\n\nReading the file now.\n",
        "<tool_call>\n{\"name\": \"read\", \"arguments\": {\"path\": \"a.rs\", \"limit\": 20}}\n</tool_call>",
        "<|im_start|>user\n<tool_response>\nfn main() {}\n</tool_response><|im_end|>\n",
        "# Tools\n\nYou may call one or more functions to assist with the user query.\n",
        "You are provided with function signatures within <tools></tools> XML tags:\n<tools>\n",
        "{\"type\": \"function\", \"function\": {\"name\": \"read\", \"description\": \"Read a file\",",
        " \"parameters\": {\"type\": \"object\", \"properties\": {\"path\": {\"type\": \"string\"}}}}}",
        "For each function call, return a json object with function name and arguments within",
        " <tool_call></tool_call> XML tags. Both files are fine. Now b.rs. Done. Step thanks",
    ];

    /// A small byte-level BPE (merges trained on [`BPE_CORPUS`], the same pre-tokenizer and
    /// decoder family as Qwen's) with the [`SPECIAL`] added tokens. Unlike the byte tokenizer,
    /// re-encoding text a model sampled token by token gives different ids (D051 tests).
    pub fn bpe_tokenizer() -> Tokenizer {
        use tokenizers::models::TrainerWrapper;
        use tokenizers::models::bpe::BpeTrainerBuilder;
        let model = BPE::builder().build().unwrap();
        let mut inner = tokenizers::Tokenizer::new(model);
        inner.with_pre_tokenizer(Some(ByteLevel::new(false, true, true)));
        inner.with_decoder(Some(ByteLevelDecoder::default()));
        let mut trainer: TrainerWrapper = BpeTrainerBuilder::new()
            .vocab_size(700)
            .min_frequency(1)
            .show_progress(false)
            .initial_alphabet(ByteLevel::alphabet().into_iter().collect())
            .build()
            .into();
        let corpus = BPE_CORPUS.iter().flat_map(|text| [*text; 3]);
        inner
            .train(&mut trainer, corpus)
            .expect("train the BPE fixture");
        inner
            .add_special_tokens(SPECIAL.map(|token| AddedToken::from(token, true)))
            .expect("add special tokens");
        Tokenizer::from_inner(inner)
    }

    /// Ids a model could sample for `text` that differ from its encoding: every ordinary token
    /// split into its single-byte tokens (added tokens stay whole). They decode to `text`.
    pub fn sampled_ids(tokenizer: &Tokenizer, text: &str) -> Vec<u32> {
        let inner = tokenizer.inner();
        let added = tokenizer.added_tokens();
        tokenizer
            .encode(text)
            .unwrap()
            .into_iter()
            .flat_map(|id| {
                let piece = inner.id_to_token(id).unwrap();
                if added.iter().any(|(a, _)| *a == id) {
                    vec![id]
                } else {
                    piece
                        .chars()
                        .map(|c| inner.token_to_id(&c.to_string()).unwrap())
                        .collect()
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::test_support::byte_tokenizer;

    #[test]
    fn bpe_fixture_merges_and_sampled_ids_decode_alike() {
        use super::test_support::{bpe_tokenizer, sampled_ids};
        let tokenizer = bpe_tokenizer();
        let text = "<|im_start|>assistant\n<think>\nLook at a.rs.\n</think>\n\nReading.";
        let ids = tokenizer.encode(text).unwrap();
        let sampled = sampled_ids(&tokenizer, text);
        assert!(ids.len() < text.len() / 2, "merges apply: {}", ids.len());
        assert_ne!(ids, sampled, "re-encoding drifts from sampled ids");
        assert_eq!(tokenizer.inner().decode(&sampled, false).unwrap(), text);
        assert_eq!(bpe_tokenizer().encode(text).unwrap(), ids, "deterministic");
    }

    #[test]
    fn encode_recognizes_special_tokens_without_adding_any() {
        let tokenizer = byte_tokenizer();
        let ids = tokenizer.encode("<|im_start|>hi").unwrap();
        assert_eq!(ids.len(), 3);
        assert_eq!(Some(ids[0]), tokenizer.token_id("<|im_start|>"));
    }

    #[test]
    fn segmented_encoding_keeps_added_tokens_in_content_as_text() {
        let tokenizer = byte_tokenizer();
        let text = "<|im_start|>user\nsay <|im_end|> please<|im_end|>\n";
        let at = text.find("say").unwrap();
        let content = (at, at + "say <|im_end|> please".len());
        let (ids, offsets) = tokenizer.encode_segmented(text, &[content]).unwrap();
        let end = tokenizer.token_id("<|im_end|>").unwrap();
        let start = tokenizer.token_id("<|im_start|>").unwrap();
        assert_eq!(ids.iter().filter(|&&id| id == end).count(), 1);
        assert_eq!(ids[0], start);
        assert_eq!(
            ids[ids.len() - 2],
            end,
            "the scaffolding token stays special"
        );
        assert_eq!(offsets.len(), ids.len());
        let decoded = tokenizer.inner().decode(&ids, false).unwrap();
        assert_eq!(decoded, text);
        // Content without added tokens encodes exactly as the whole text.
        let plain = "<|im_start|>user\nhello<|im_end|>";
        assert_eq!(
            tokenizer.encode_segmented(plain, &[(17, 22)]).unwrap().0,
            tokenizer.encode(plain).unwrap()
        );
    }

    #[test]
    fn xapi_segment_memo_equals_fresh_encode() {
        use super::test_support::bpe_tokenizer;
        for tokenizer in [byte_tokenizer(), bpe_tokenizer()] {
            assert!(tokenizer.splits());
            let turn = |i: usize| {
                format!(
                    "<|im_start|>user\nopen {i}.rs <|im_start|> in text<|im_end|>\n<|im_start|>assistant\n<think>\nok\n</think>\n\nReading.<|im_end|>\n"
                )
            };
            let mut text =
                "lead <|im_start|>system\nYou are a coding agent.<|im_end|>\n".to_string();
            for i in 0..20 {
                text.push_str(&turn(i));
                let fresh = tokenizer.encode_with_offsets(&text).unwrap();
                assert_eq!(tokenizer.encode_memoized(&text).unwrap(), fresh);
            }
            assert!(tokenizer.memo_hits() > 100, "{}", tokenizer.memo_hits());
            assert!(tokenizer.memo_bytes() > 0);
        }
    }

    #[test]
    fn segment_memo_evicts_the_least_recently_used_within_its_bound() {
        let mut memo = super::Memo::new(3_000);
        let piece = |i: usize| format!("{i:0>100}");
        for i in 0..20 {
            memo.put(&piece(i), &[1; 20], &[(0, 1); 20]);
            memo.get(&piece(0));
            assert!(memo.bytes <= 3_000);
        }
        assert!(memo.get(&piece(0)).is_some(), "recently used stays");
        assert!(memo.get(&piece(1)).is_none(), "oldest went first");
        assert!(memo.get(&piece(19)).is_some());
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
