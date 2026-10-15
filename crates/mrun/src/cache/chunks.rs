//! Prefill chunk plan (003 R4): chunk ends at checkpoints, else at the last message boundary at
//! least 256 tokens in, else at the depth-adaptive chunk length. A pure function of the prompt,
//! the checkpoints and the resume position, so cache-on and cache-off split alike.

use crate::chat::prompt::RenderedPrompt;
use crate::runtime::generate::ChunkPlan;
use mrun_engine::chunk_len;

/// Shortest chunk that may end at a message boundary.
const MIN_BOUNDARY_CHUNK: u32 = 256;

pub fn plan(prompt: &RenderedPrompt, resume: u32, checkpoints: Vec<u32>) -> ChunkPlan {
    let end = prompt.len().saturating_sub(1);
    let boundaries: Vec<u32> = prompt.messages.iter().map(|m| m.start).collect();
    let mut chosen = Vec::new();
    let mut at = resume;
    while at < end {
        let limit = at.saturating_add(chunk_len(at)).min(end);
        let next = checkpoints
            .iter()
            .copied()
            .filter(|&c| c > at && c <= limit)
            .min()
            .or_else(|| {
                boundaries
                    .iter()
                    .copied()
                    .filter(|&b| b >= at + MIN_BOUNDARY_CHUNK && b <= limit)
                    .max()
            })
            .unwrap_or(limit);
        if next < end && !checkpoints.contains(&next) {
            chosen.push(next);
        }
        at = next;
    }
    ChunkPlan {
        boundaries: chosen,
        checkpoints,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chat::prompt::{MessageMark, Role};

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
        let plan = plan(&p, 0, vec![3000]);
        // 100 is too early (< 256), 1900 is the last boundary within the first 2048.
        assert_eq!(plan.boundaries[0], 1900);
        // From 1900: the checkpoint at 3000 comes first, then 4000 (≥ 256 from 3000).
        assert!(!plan.boundaries.contains(&3000));
        assert!(plan.boundaries.contains(&4000));
        assert_eq!(plan.checkpoints, vec![3000]);
        // Deterministic and independent of earlier state.
        assert_eq!(super::plan(&p, 0, vec![3000]), plan);
        let resumed = super::plan(&p, 1900, vec![3000]);
        assert_eq!(&resumed.boundaries[..], &plan.boundaries[1..]);
    }
}
