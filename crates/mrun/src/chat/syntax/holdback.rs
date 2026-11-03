//! Hold-back of streamed text that may be the start of a marker split across tokens, and the
//! special-token gate (D032): a marker that is an added token of the vocabulary counts only
//! where the model emitted that token, so the same text in prose stays text. Shared by every
//! syntax parser; all split points fall on UTF-8 character boundaries.

use std::collections::{BTreeSet, HashMap, HashSet};
use std::sync::Arc;

/// The added tokens of a vocabulary (`<tool_call>`, `<|im_end|>`, ...).
#[derive(Clone, Debug, Default)]
pub struct Specials {
    by_id: HashMap<u32, String>,
    texts: HashSet<String>,
}

impl Specials {
    pub fn new(tokens: impl IntoIterator<Item = (u32, String)>) -> Self {
        let by_id: HashMap<u32, String> = tokens.into_iter().collect();
        let texts = by_id.values().cloned().collect();
        Self { by_id, texts }
    }

    pub fn text(&self, id: u32) -> Option<&str> {
        self.by_id.get(&id).map(String::as_str)
    }

    pub fn is_special(&self, text: &str) -> bool {
        self.texts.contains(text)
    }
}

/// Pending text a parser has received but not yet classified.
#[derive(Debug, Default)]
pub struct HoldBack {
    buf: String,
    /// Stream offset of `buf[0]`: bytes taken so far.
    base: usize,
    /// `None`: markers are recognized by their text alone.
    specials: Option<Arc<Specials>>,
    /// Stream offsets where an emitted special token's text starts.
    emitted: BTreeSet<usize>,
}

impl HoldBack {
    pub fn new(specials: Option<Arc<Specials>>) -> Self {
        Self {
            specials,
            ..Self::default()
        }
    }

    /// Text without a token (tests, released stop-sequence hold-back).
    pub fn push(&mut self, text: &str) {
        self.buf.push_str(text);
    }

    /// One output token and the text it made visible; `held` counts bytes the stop matcher still
    /// holds back, which arrive later (the token's own text may be among them).
    pub fn push_token(&mut self, id: u32, text: &str, held: u32) {
        self.buf.push_str(text);
        if let Some(special) = self.specials.as_ref().and_then(|s| s.text(id)) {
            let end = self.base + self.buf.len() + held as usize;
            self.emitted.insert(end.saturating_sub(special.len()));
        }
    }

    pub fn as_str(&self) -> &str {
        &self.buf
    }

    pub fn is_empty(&self) -> bool {
        self.buf.is_empty()
    }

    /// Stream offset of the start of the pending text.
    pub fn offset(&self) -> usize {
        self.base
    }

    /// Whether `marker` at stream offset `at` is a marker: always for text markers and without
    /// a gate; for special-token markers only where the model emitted the token.
    pub fn counts_at(&self, at: usize, marker: &str) -> bool {
        match &self.specials {
            Some(specials) if specials.is_special(marker) => self.emitted.contains(&at),
            _ => true,
        }
    }

    /// Whether the model emitted a special token whose text starts at stream offset `at`.
    pub fn is_token_at(&self, at: usize) -> bool {
        self.specials.is_some() && self.emitted.contains(&at)
    }

    /// [`Self::counts_at`] for byte `at` of the pending text.
    pub fn counts(&self, at: usize, marker: &str) -> bool {
        self.counts_at(self.base + at, marker)
    }

    /// Whether the pending text continues with `marker` at byte `at`, as a marker.
    pub fn marker_at(&self, at: usize, marker: &str) -> bool {
        self.buf[at..].starts_with(marker) && self.counts(at, marker)
    }

    /// Earliest marker occurrence as `(byte offset, marker index)`; at equal offsets the longest
    /// marker wins.
    pub fn find(&self, markers: &[&str]) -> Option<(usize, usize)> {
        markers
            .iter()
            .enumerate()
            .filter_map(|(i, marker)| {
                self.buf
                    .match_indices(marker)
                    .map(|(at, _)| at)
                    .find(|&at| self.counts(at, marker))
                    .map(|at| (at, i))
            })
            .min_by_key(|&(at, i)| (at, usize::MAX - markers[i].len()))
    }

    /// Removes and returns the first `at` bytes.
    pub fn take(&mut self, at: usize) -> String {
        let rest = self.buf.split_off(at);
        self.base += at;
        std::mem::replace(&mut self.buf, rest)
    }

    /// Drops the first `n` bytes.
    pub fn skip(&mut self, n: usize) {
        self.buf.drain(..n);
        self.base += n;
    }

    /// Puts taken text back in front.
    pub fn unshift(&mut self, text: &str) {
        self.base -= text.len();
        self.buf.insert_str(0, text);
    }

    /// Removes and returns everything except a suffix that could still begin one of `markers`.
    pub fn take_safe(&mut self, markers: &[&str]) -> String {
        let len = self.buf.len();
        let keep = markers
            .iter()
            .filter_map(|marker| {
                (1..marker.len().min(len + 1)).rev().find(|&n| {
                    marker.is_char_boundary(n)
                        && self.buf.ends_with(&marker[..n])
                        && self.counts(len - n, marker)
                })
            })
            .max()
            .unwrap_or(0);
        self.take(len - keep)
    }

    pub fn take_all(&mut self) -> String {
        self.take(self.buf.len())
    }
}

/// Length of the longest suffix of `text` that is a proper prefix of one of `markers`.
pub fn partial_suffix(text: &str, markers: &[&str]) -> usize {
    markers
        .iter()
        .filter_map(|marker| {
            (1..marker.len().min(text.len() + 1))
                .rev()
                .find(|&n| marker.is_char_boundary(n) && text.ends_with(&marker[..n]))
        })
        .max()
        .unwrap_or(0)
}

/// Earliest occurrence of any marker in `text` as `(byte offset, marker index)`.
pub fn find_first(text: &str, markers: &[&str]) -> Option<(usize, usize)> {
    markers
        .iter()
        .enumerate()
        .filter_map(|(i, marker)| text.find(marker).map(|at| (at, i)))
        .min_by_key(|&(at, i)| (at, usize::MAX - markers[i].len()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn holds_partial_marker_and_releases_on_mismatch() {
        let mut hb = HoldBack::default();
        hb.push("hello <tool");
        assert_eq!(hb.take_safe(&["<tool_call>"]), "hello ");
        hb.push("box>");
        assert_eq!(hb.take_safe(&["<tool_call>"]), "<toolbox>");
        assert!(hb.is_empty());
    }

    #[test]
    fn earliest_then_longest_marker_wins() {
        assert_eq!(
            find_first("a</think>b<think>", &["<think>", "</think>"]),
            Some((1, 1))
        );
        assert_eq!(find_first("x<|end|>", &["<|e", "<|end|>"]), Some((1, 1)));
        assert_eq!(find_first("none", &["<x>"]), None);
    }

    #[test]
    fn utf8_text_splits_on_char_boundaries() {
        let mut hb = HoldBack::default();
        hb.push("héllo ✓<");
        assert_eq!(hb.take_safe(&["<✓>"]), "héllo ✓");
        hb.push("✓");
        assert_eq!(hb.take_safe(&["<✓>"]), "");
        assert_eq!(hb.as_str(), "<✓");
        hb.push("x");
        assert_eq!(hb.take_safe(&["<✓>"]), "<✓x");
        // A multi-byte marker prefix is only matched at character boundaries.
        assert_eq!(partial_suffix("ab\u{2713}", &["\u{2713}\u{2713}"]), 3);
    }

    #[test]
    fn take_and_skip() {
        let mut hb = HoldBack::default();
        hb.push("abc<m>def");
        let (at, _) = hb.find(&["<m>"]).unwrap();
        assert_eq!(hb.take(at), "abc");
        hb.skip(3);
        assert_eq!(hb.take_all(), "def");
        assert_eq!(hb.offset(), 9);
    }

    #[test]
    fn special_markers_count_only_where_emitted_as_tokens() {
        let specials = Arc::new(Specials::new([(7, "<tool_call>".to_string())]));
        let mut hb = HoldBack::new(Some(specials));
        // Prose that spells the marker: text.
        hb.push_token(1, "use <tool_call> here ", 0);
        assert_eq!(hb.find(&["<tool_call>"]), None);
        assert_eq!(hb.take_safe(&["<tool_call>"]), "use <tool_call> here ");
        // The token itself, its text held back by a stop matcher and released later.
        hb.push_token(7, "", 11);
        hb.push_token(2, "<tool_call>x", 0);
        assert_eq!(hb.find(&["<tool_call>"]), Some((0, 0)));
        // Text markers are not gated.
        assert_eq!(hb.find(&["x"]), Some((11, 0)));
    }
}
