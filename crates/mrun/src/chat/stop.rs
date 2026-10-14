//! Stop-sequence matching on streamed text with hold-back of possible partial matches.

/// Maximum number of stop sequences and their length in bytes (001 FR-016).
pub const MAX_STOPS: usize = 4;
pub const MAX_STOP_BYTES: usize = 256;

#[derive(Debug, thiserror::Error, Eq, PartialEq)]
pub enum StopError {
    #[error("at most {MAX_STOPS} stop sequences are allowed, got {0}")]
    TooMany(usize),
    #[error("stop sequences must be 1..={MAX_STOP_BYTES} bytes, got {0}")]
    BadLength(usize),
}

/// What the matcher lets through after each piece of text.
#[derive(Debug, Default, Eq, PartialEq)]
pub struct Fed {
    /// Text that is safe to show.
    pub emit: String,
    /// The stop sequence that ended generation, if any.
    pub stopped: Option<String>,
}

#[derive(Debug)]
pub struct StopMatcher {
    stops: Vec<String>,
    held: String,
}

impl StopMatcher {
    pub fn new(stops: Vec<String>) -> Result<Self, StopError> {
        if stops.len() > MAX_STOPS {
            return Err(StopError::TooMany(stops.len()));
        }
        if let Some(bad) = stops
            .iter()
            .find(|stop| stop.is_empty() || stop.len() > MAX_STOP_BYTES)
        {
            return Err(StopError::BadLength(bad.len()));
        }
        Ok(Self {
            stops,
            held: String::new(),
        })
    }

    /// Feeds new text; returns what may be shown now and whether a stop sequence matched.
    pub fn feed(&mut self, text: &str) -> Fed {
        self.held.push_str(text);
        let earliest = self
            .stops
            .iter()
            .filter_map(|stop| self.held.find(stop.as_str()).map(|at| (at, stop)))
            .min_by_key(|(at, _)| *at);
        if let Some((at, stop)) = earliest {
            let stop = stop.clone();
            let emit = self.held[..at].to_string();
            self.held.clear();
            return Fed {
                emit,
                stopped: Some(stop),
            };
        }
        // Hold back the longest suffix that could still start a stop sequence.
        let keep = self
            .stops
            .iter()
            .flat_map(|stop| {
                (1..stop.len())
                    .filter(|&n| stop.is_char_boundary(n) && self.held.ends_with(&stop[..n]))
            })
            .max()
            .unwrap_or(0);
        let split = self.held.len() - keep;
        let emit = self.held[..split].to_string();
        self.held.drain(..split);
        Fed {
            emit,
            stopped: None,
        }
    }

    /// Releases held text at the end of generation (no stop matched).
    pub fn finish(&mut self) -> String {
        std::mem::take(&mut self.held)
    }

    /// Bytes currently held back.
    pub fn held_len(&self) -> usize {
        self.held.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stop_split_across_pieces_is_detected_and_not_emitted() {
        let mut matcher = StopMatcher::new(vec!["</end>".into()]).unwrap();
        assert_eq!(matcher.feed("hello </e").emit, "hello ");
        let fed = matcher.feed("nd> tail");
        assert_eq!(fed.emit, "");
        assert_eq!(fed.stopped.as_deref(), Some("</end>"));
    }

    #[test]
    fn false_partial_match_is_released() {
        let mut matcher = StopMatcher::new(vec!["STOP".into()]).unwrap();
        assert_eq!(matcher.feed("ST").emit, "");
        assert_eq!(matcher.feed("ART").emit, "START");
        assert_eq!(matcher.finish(), "");
    }

    #[test]
    fn earliest_of_several_stops_wins() {
        let mut matcher = StopMatcher::new(vec!["b".into(), "a".into()]).unwrap();
        let fed = matcher.feed("xab");
        assert_eq!(fed.emit, "x");
        assert_eq!(fed.stopped.as_deref(), Some("a"));
    }

    #[test]
    fn held_text_is_flushed_at_finish() {
        let mut matcher = StopMatcher::new(vec!["##".into()]).unwrap();
        assert_eq!(matcher.feed("a#").emit, "a");
        assert_eq!(matcher.finish(), "#");
    }

    #[test]
    fn limits_are_enforced() {
        assert_eq!(
            StopMatcher::new(vec!["a".into(); 5]).unwrap_err(),
            StopError::TooMany(5)
        );
        assert_eq!(
            StopMatcher::new(vec![String::new()]).unwrap_err(),
            StopError::BadLength(0)
        );
        assert_eq!(
            StopMatcher::new(vec!["x".repeat(257)]).unwrap_err(),
            StopError::BadLength(257)
        );
    }
}
