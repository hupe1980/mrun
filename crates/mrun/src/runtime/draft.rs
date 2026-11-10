//! Speculative decoding's runtime side (006): the prompt-lookup [`Drafter`] and the [`Pace`]
//! that decides whether drafting pays on this request.
//!
//! Agents resend their history, so the prompt holds the files and replies a reply copies from.
//! The drafter finds the latest, longest earlier occurrence of the text's end in the prompt and
//! the reply so far and proposes what followed. Whatever it proposes, the engine's verify keeps
//! only what plain decoding would produce: the drafter and the pace trade speed, never output.

use std::collections::HashMap;
use std::time::Duration;

/// Positions kept per 2-gram (the latest).
const CANDIDATES: usize = 8;
/// Longest backward match measured.
const MATCH_LIMIT: usize = 32;

/// The latest positions of one 2-gram, oldest first (a ring of [`CANDIDATES`]).
#[derive(Clone, Copy, Default)]
struct Slots {
    ends: [u32; CANDIDATES],
    len: u8,
    next: u8,
}

impl Slots {
    fn push(&mut self, end: u32) {
        self.ends[self.next as usize] = end;
        self.next = (self.next + 1) % CANDIDATES as u8;
        self.len = (self.len + 1).min(CANDIDATES as u8);
    }

    /// Newest first.
    fn latest(&self) -> impl Iterator<Item = u32> + '_ {
        (1..=self.len as usize)
            .map(move |back| self.ends[(self.next as usize + CANDIDATES - back) % CANDIDATES])
    }
}

/// Prompt lookup over the prompt and the reply so far.
pub struct Drafter {
    tokens: Vec<u32>,
    /// 2-gram → the positions of its second token.
    index: HashMap<u64, Slots>,
}

impl std::fmt::Debug for Drafter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Drafter")
            .field("tokens", &self.tokens.len())
            .field("keys", &self.index.len())
            .finish()
    }
}

fn key(a: u32, b: u32) -> u64 {
    (u64::from(a) << 32) | u64::from(b)
}

impl Drafter {
    pub fn new(prompt: &[u32]) -> Self {
        let mut drafter = Self {
            tokens: Vec::with_capacity(prompt.len() + 1024),
            index: HashMap::with_capacity(prompt.len()),
        };
        for &token in prompt {
            drafter.push(token);
        }
        drafter
    }

    /// Appends a committed token.
    pub fn push(&mut self, token: u32) {
        if let Some(&before) = self.tokens.last() {
            let end = self.tokens.len() as u32;
            self.index.entry(key(before, token)).or_default().push(end);
        }
        self.tokens.push(token);
    }

    /// Up to `max` tokens proposed after the text's end: the continuation of its longest earlier
    /// match (at least 2 tokens; the newest wins a tie), `match − 1` tokens at most. Empty when
    /// nothing matches or `max` is 0.
    pub fn draft(&self, max: usize) -> Vec<u32> {
        let n = self.tokens.len();
        if max == 0 || n < 2 {
            return Vec::new();
        }
        let Some(slots) = self.index.get(&key(self.tokens[n - 2], self.tokens[n - 1])) else {
            return Vec::new();
        };
        let mut best: Option<(usize, usize)> = None;
        for end in slots.latest().map(|e| e as usize) {
            // The end itself, and matches with nothing after them, propose nothing.
            if end + 1 >= n {
                continue;
            }
            let matched = (0..MATCH_LIMIT.min(end + 1))
                .take_while(|&back| self.tokens[end - back] == self.tokens[n - 1 - back])
                .count();
            if best.is_none_or(|(_, m)| matched > m) {
                best = Some((end, matched));
            }
        }
        let Some((end, matched)) = best else {
            return Vec::new();
        };
        let len = (matched - 1).clamp(1, max).min(n - 1 - end);
        self.tokens[end + 1..end + 1 + len].to_vec()
    }
}

/// Plain steps skipped after a losing speculative round: the first back-off, and the most.
const BACKOFF_FIRST: u32 = 16;
const BACKOFF_MAX: u32 = 256;
/// Weight of a new sample in the moving rates.
const WEIGHT: f64 = 0.25;

/// Whether speculating pays on this request: moving rates (tokens per second) of plain steps
/// and of speculative rounds. Drafting goes on while the speculative rate keeps up with the plain
/// one; after a losing round it pauses for a back-off of plain steps that doubles each time.
#[derive(Debug)]
pub struct Pace {
    plain: Option<f64>,
    spec: Option<f64>,
    skip: u32,
    backoff: u32,
}

impl Default for Pace {
    fn default() -> Self {
        Self {
            plain: None,
            spec: None,
            skip: 0,
            backoff: BACKOFF_FIRST,
        }
    }
}

fn mix(average: Option<f64>, sample: f64) -> f64 {
    average.map_or(sample, |a| a + WEIGHT * (sample - a))
}

impl Pace {
    /// Whether to draft now.
    pub fn drafting(&self) -> bool {
        self.skip == 0
    }

    /// A plain step of `took`.
    pub fn plain(&mut self, took: Duration) {
        self.skip = self.skip.saturating_sub(1);
        if let Some(rate) = rate(1, took) {
            self.plain = Some(mix(self.plain, rate));
        }
    }

    /// A speculative round that committed `tokens` in `took`.
    pub fn round(&mut self, tokens: usize, took: Duration) {
        let Some(rate) = rate(tokens, took) else {
            return;
        };
        self.spec = Some(mix(self.spec, rate));
        if let (Some(plain), Some(spec)) = (self.plain, self.spec) {
            if spec < plain {
                self.skip = self.backoff;
                self.backoff = (self.backoff * 2).min(BACKOFF_MAX);
                // Probe afresh after the pause.
                self.spec = None;
            } else {
                self.backoff = BACKOFF_FIRST;
            }
        }
    }
}

fn rate(tokens: usize, took: Duration) -> Option<f64> {
    let secs = took.as_secs_f64();
    (secs > 0.0).then(|| tokens as f64 / secs)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn drafter(tokens: &[u32]) -> Drafter {
        Drafter::new(tokens)
    }

    #[test]
    fn a_copy_is_found_and_continued() {
        // "1 2 3 4 5 6 … 1 2 3": the end matches 3 tokens, so 2 tokens follow (match − 1).
        let d = drafter(&[1, 2, 3, 4, 5, 6, 9, 1, 2, 3]);
        assert_eq!(d.draft(7), [4, 5]);
        // The cap.
        assert_eq!(d.draft(1), [4]);
        assert!(d.draft(0).is_empty());
    }

    #[test]
    fn the_longest_match_wins_and_the_newest_breaks_ties() {
        // "7 1 2" then "8 1 2 …" earlier; the end "8 1 2" matches the second one longer.
        let d = drafter(&[7, 1, 2, 3, 8, 1, 2, 4, 9, 8, 1, 2]);
        assert_eq!(d.draft(7)[0], 4);
        // Equal matches: the newest.
        let d = drafter(&[1, 2, 3, 1, 2, 4, 1, 2]);
        assert_eq!(d.draft(1), [4]);
    }

    #[test]
    fn nothing_below_two_tokens_and_appends_extend_the_index() {
        let mut d = drafter(&[1, 2, 3, 4]);
        assert!(d.draft(7).is_empty(), "the end is new");
        d.push(1);
        assert!(d.draft(7).is_empty(), "one matching token is not enough");
        d.push(2);
        assert_eq!(d.draft(7), [3], "2 matched: 1 token");
        d.push(3);
        assert_eq!(d.draft(7), [4, 1], "3 matched: 2 tokens, as far as known");
    }

    #[test]
    fn long_matches_draft_up_to_the_cap() {
        let text: Vec<u32> = (0..40).collect();
        let mut d = drafter(&text);
        for t in 0..20 {
            d.push(t);
        }
        assert_eq!(d.draft(7), (20..27).collect::<Vec<_>>());
    }

    const MS: Duration = Duration::from_millis(1);

    #[test]
    fn the_pace_keeps_drafting_while_it_pays() {
        let mut pace = Pace::default();
        assert!(pace.drafting(), "the first round probes");
        pace.plain(20 * MS); // 50 tok/s
        pace.round(3, 30 * MS); // 100 tok/s
        assert!(pace.drafting());
    }

    #[test]
    fn a_losing_round_pauses_drafting_with_a_growing_backoff() {
        let mut pace = Pace::default();
        pace.plain(20 * MS); // 50 tok/s
        pace.round(1, 40 * MS); // 25 tok/s: loses
        assert!(!pace.drafting());
        for _ in 0..BACKOFF_FIRST {
            pace.plain(20 * MS);
        }
        assert!(pace.drafting(), "a probe after 16 plain steps");
        pace.round(1, 40 * MS);
        for _ in 0..2 * BACKOFF_FIRST - 1 {
            pace.plain(20 * MS);
        }
        assert!(!pace.drafting(), "the second pause is 32 steps");
        pace.plain(20 * MS);
        assert!(pace.drafting());
        // A winning round resets the back-off; the moving rate then needs several losing
        // rounds to fall below plain decoding's, and the next pause is 16 steps again.
        pace.round(4, 40 * MS);
        let mut losing = 0;
        while pace.drafting() {
            pace.round(1, 40 * MS);
            losing += 1;
        }
        assert!(losing > 1, "one bad round does not stop it");
        for _ in 0..BACKOFF_FIRST {
            pace.plain(20 * MS);
        }
        assert!(pace.drafting());
    }
}
