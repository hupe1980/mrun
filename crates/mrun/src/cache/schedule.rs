//! Forward schedules (003 FR-015, R10): which forward passes produced a path's state. Decode
//! computes keys and values with width-1 kernels and prefill with chunk-wide ones, which differ
//! in the last bits, so bit equality with a recomputation (D049, SC-005) needs the same passes.
//! The store keeps one schedule per path (the trunk and each branch, from position 0), cut and
//! moved with the state; the exactness oracle replays it, and a branch is reachable only through
//! a holder whose schedule of the shared prefix is the branch's own (the prefix it was computed
//! on), so reuse never mixes state computed on two different prefixes.

use std::sync::Arc;

/// One run of forward passes.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Span {
    /// One `prefill_chunk` over positions `[start, end)`. `tail` holds the chunk's last tokens
    /// that the path no longer holds (positions `end − tail.len()..end`, cut off later), so a
    /// replay feeds the same chunk and trims back.
    Chunk {
        start: u32,
        end: u32,
        tail: Arc<[u32]>,
    },
    /// One single-token `forward` per position in `[start, end)` (decode, teacher-forced in a
    /// replay).
    Steps { start: u32, end: u32 },
}

impl Span {
    pub fn start(&self) -> u32 {
        match self {
            Self::Chunk { start, .. } | Self::Steps { start, .. } => *start,
        }
    }

    /// End of the positions the path holds from this span.
    pub fn held_end(&self) -> u32 {
        match self {
            Self::Chunk { end, tail, .. } => end - tail.len() as u32,
            Self::Steps { end, .. } => *end,
        }
    }
}

/// A path's schedule, from position 0 to its length.
pub type Schedule = Vec<Span>;

/// Appends prefill chunks ending at `ends` (from the schedule's end).
pub fn push_chunks(schedule: &mut Schedule, ends: impl IntoIterator<Item = u32>) {
    let mut at = len(schedule);
    for end in ends {
        if end > at {
            schedule.push(Span::Chunk {
                start: at,
                end,
                tail: Arc::from(Vec::new()),
            });
            at = end;
        }
    }
}

/// Appends decode steps up to `end`.
pub fn push_steps(schedule: &mut Schedule, end: u32) {
    let at = len(schedule);
    if end <= at {
        return;
    }
    if let Some(Span::Steps { end: last, .. }) = schedule.last_mut()
        && *last == at
    {
        *last = end;
        return;
    }
    schedule.push(Span::Steps { start: at, end });
}

/// Positions the schedule covers.
pub fn len(schedule: &[Span]) -> u32 {
    schedule.last().map_or(0, Span::held_end)
}

/// Cuts `schedule` (of the path `tokens`) to `at` positions; a chunk that straddles the cut keeps
/// the tokens beyond it as its tail.
pub fn truncate(schedule: &mut Schedule, tokens: &[u32], at: u32) {
    while schedule.last().is_some_and(|s| s.start() >= at) {
        schedule.pop();
    }
    match schedule.last_mut() {
        Some(Span::Steps { end, .. }) if *end > at => *end = at,
        Some(Span::Chunk { end, tail, .. }) => {
            let held = *end - tail.len() as u32;
            if held > at {
                let mut beyond = tokens[at as usize..held as usize].to_vec();
                beyond.extend_from_slice(tail);
                *tail = Arc::from(beyond);
            }
        }
        _ => {}
    }
}

/// `schedule` (of `tokens`) cut to `at`.
pub fn prefix(schedule: &[Span], tokens: &[u32], at: u32) -> Schedule {
    let keep = schedule.partition_point(|s| s.start() < at);
    let mut out = schedule[..keep].to_vec();
    truncate(&mut out, tokens, at);
    out
}

/// Spans are contiguous from 0 (invariant 8): each starts where the previous one's held
/// positions end (a cut chunk is replayed and trimmed back), and the last ends at `length`.
pub fn is_contiguous(schedule: &[Span], length: u32) -> bool {
    let mut at = 0;
    for span in schedule {
        if span.start() != at || span.held_end() <= span.start() {
            return false;
        }
        at = span.held_end();
    }
    at == length
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cuts_keep_straddling_chunks_with_their_tail() {
        let tokens: Vec<u32> = (100..140).collect();
        let mut s = Schedule::new();
        push_chunks(&mut s, [10, 25]);
        push_steps(&mut s, 30);
        push_steps(&mut s, 32);
        assert_eq!(s.len(), 3, "step runs merge");
        assert!(is_contiguous(&s, 32));
        let cut = prefix(&s, &tokens, 20);
        assert_eq!(
            cut[1],
            Span::Chunk {
                start: 10,
                end: 25,
                tail: Arc::from(tokens[20..25].to_vec()),
            }
        );
        assert!(is_contiguous(&cut, 20));
        assert_eq!(
            prefix(&s, &tokens, 31).last(),
            Some(&Span::Steps { start: 25, end: 31 })
        );
        assert_eq!(prefix(&s, &tokens, 10).len(), 1);
        // Cutting the cut again extends the tail from the path's tokens.
        let mut twice = cut.clone();
        truncate(&mut twice, &tokens, 15);
        let Some(Span::Chunk { tail, .. }) = twice.last() else {
            panic!()
        };
        assert_eq!(&tail[..], &tokens[15..25]);
        assert_eq!(prefix(&s, &tokens, 15), twice);
        // A chunk after a straddling one starts at the held end.
        let mut grown = cut;
        push_chunks(&mut grown, [28]);
        assert_eq!(grown[2].start(), 20);
        assert!(is_contiguous(&grown, 28));
        assert!(!is_contiguous(&grown, 27));
    }
}
