//! Hold-back of streamed text that may be the start of a marker split across tokens. Shared by
//! every syntax parser; all split points fall on UTF-8 character boundaries.

/// Pending text a parser has received but not yet classified.
#[derive(Debug, Default)]
pub struct HoldBack {
    buf: String,
}

impl HoldBack {
    pub fn push(&mut self, text: &str) {
        self.buf.push_str(text);
    }

    pub fn as_str(&self) -> &str {
        &self.buf
    }

    pub fn is_empty(&self) -> bool {
        self.buf.is_empty()
    }

    /// Earliest occurrence of any marker as `(byte offset, marker index)`; at equal offsets the
    /// longest marker wins.
    pub fn find(&self, markers: &[&str]) -> Option<(usize, usize)> {
        find_first(&self.buf, markers)
    }

    /// Removes and returns the first `at` bytes.
    pub fn take(&mut self, at: usize) -> String {
        let rest = self.buf.split_off(at);
        std::mem::replace(&mut self.buf, rest)
    }

    /// Drops the first `n` bytes.
    pub fn skip(&mut self, n: usize) {
        self.buf.drain(..n);
    }

    /// Removes and returns everything except a suffix that could still begin one of `markers`.
    pub fn take_safe(&mut self, markers: &[&str]) -> String {
        let keep = partial_suffix(&self.buf, markers);
        self.take(self.buf.len() - keep)
    }

    pub fn take_all(&mut self) -> String {
        std::mem::take(&mut self.buf)
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
    }
}
