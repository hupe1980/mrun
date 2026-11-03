//! Prefill chunk plan (D028: the runtime owns it; 003 FR-005). From position 0, the next end
//! after `s` is the first checkpoint mark in `(s, s + chunk_len(s)]`, else the last message
//! boundary in `[s + 256, s + chunk_len(s)]`, else `s + chunk_len(s)`. A pure function of the
//! prompt (tokens and marks): never of the resume position, cache state or traffic, so a request
//! served from the cache and a recomputation from empty state split its prompt alike (D049). A
//! request resuming at `r` runs the planned ends above `r` ([`above`]). `chunk_len` is the
//! engine's ([`ModelInfo::chunk_len`]: the memory-bound schedule, lowered where the warm-up
//! calibration found a layer slower than 250 ms; fixed for the process, D059), so the runtime and
//! the engine agree and the plan stays a pure function of the prompt for a given engine.

use crate::chat::prompt::RenderedPrompt;
use mrun_engine::ModelInfo;

/// Shortest chunk that may end at a message boundary.
const MIN_BOUNDARY_CHUNK: u32 = 256;

/// Ends of the chunks that prefill `[0, P − 1)` (the last prompt token is fed by the first
/// decode step), ascending; the last one is `P − 1`. `marks` are the checkpoint marks (empty on
/// attention-only models).
pub fn plan(info: &ModelInfo, prompt: &RenderedPrompt, marks: &[u32]) -> Vec<u32> {
    let end = prompt.len().saturating_sub(1);
    let mut ends = Vec::new();
    let mut at = 0;
    while at < end {
        let limit = at.saturating_add(info.chunk_len(at)).min(end);
        let next = marks
            .iter()
            .copied()
            .filter(|&c| c > at && c <= limit)
            .min()
            .or_else(|| {
                prompt
                    .messages
                    .iter()
                    .map(|m| m.start)
                    .filter(|&b| b >= at + MIN_BOUNDARY_CHUNK && b <= limit)
                    .max()
            })
            .unwrap_or(limit);
        ends.push(next);
        at = next;
    }
    ends
}

/// The planned ends above `resume`: what a request resuming there runs (its first chunk starts
/// at `resume`).
pub fn above(ends: &[u32], resume: u32) -> Vec<u32> {
    ends.iter().copied().filter(|&e| e > resume).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chat::prompt::{MessageMark, Role};
    use mrun_engine::{TimeLen, chunk_len};

    fn info() -> ModelInfo {
        ModelInfo::default()
    }

    fn prompt(len: u32, starts: &[u32]) -> RenderedPrompt {
        RenderedPrompt {
            tokens: vec![1; len as usize],
            messages: starts
                .iter()
                .map(|&start| MessageMark {
                    start,
                    role: Role::User,
                    message: Some(0),
                })
                .collect(),
            system_tools_end: 0,
            generation_start: len,
        }
    }

    #[test]
    fn chunks_end_at_boundaries_checkpoints_or_length() {
        let p = prompt(5000, &[0, 100, 1500, 1900, 4000]);
        let ends = plan(&info(), &p, &[3000]);
        // 100 is too early (< 256), 1900 is the last boundary within the first 2048; then the
        // checkpoint at 3000, then 4000 (≥ 256 from 3000), then the end.
        assert_eq!(ends, [1900, 3000, 4000, 4999]);
        assert_eq!(plan(&info(), &p, &[3000]), ends);
        assert_eq!(above(&ends, 1900), ends[1..]);
        assert_eq!(
            above(&ends, 2500),
            [3000, 4000, 4999],
            "the first chunk starts at 2500"
        );
        assert!(plan(&info(), &prompt(1, &[]), &[]).is_empty());
    }

    #[test]
    fn long_prompts_shrink_chunks_with_depth() {
        let p = prompt(40_000, &[]);
        let ends = plan(&info(), &p, &[]);
        let mut at = 0;
        for &end in &ends {
            assert_eq!(end - at, chunk_len(at).min(39_999 - at));
            at = end;
        }
        assert_eq!(at, 39_999);
    }

    #[test]
    fn a_calibrated_engine_shortens_the_plan() {
        let slow = ModelInfo {
            time_len: TimeLen::from_points(vec![(0, 1024), (16_384, 512)]),
            ..ModelInfo::default()
        };
        let p = prompt(5000, &[]);
        let ends = plan(&slow, &p, &[]);
        assert_eq!(ends[0], 1024);
        let mut at = 0;
        for &end in &ends {
            assert_eq!(end - at, slow.chunk_len(at).min(4999 - at));
            assert!(end - at < chunk_len(at) || at + chunk_len(at) > 4999);
            at = end;
        }
        assert_eq!(plan(&slow, &p, &[]), plan(&slow, &p, &[]), "pure");
    }

    #[test]
    fn the_plan_ignores_the_resume_position() {
        // Property (FR-005): for any prompt and any resume, the ends a resumed request runs are
        // the full plan's ends above it.
        let mut seed = 0x2545_f491_u64;
        let mut rand = |n: u32| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed % u64::from(n.max(1))) as u32
        };
        for _ in 0..200 {
            let len = 2 + rand(60_000);
            let mut starts: Vec<u32> = (0..rand(40)).map(|_| rand(len)).collect();
            starts.sort_unstable();
            starts.dedup();
            let marks: Vec<u32> = (0..rand(6)).map(|_| rand(len)).collect();
            let p = prompt(len, &starts);
            let ends = plan(&info(), &p, &marks);
            assert_eq!(ends.last(), Some(&(len - 1)));
            assert!(ends.windows(2).all(|w| w[0] < w[1]));
            let resume = rand(len);
            let resumed = above(&ends, resume);
            assert!(resumed.iter().all(|&e| e > resume));
            assert_eq!(resumed.len(), ends.iter().filter(|&&e| e > resume).count());
        }
    }
}
