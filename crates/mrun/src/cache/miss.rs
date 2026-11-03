//! Why a request was not served entirely from cached state (003, D053): cached and processed
//! tokens, the divergence position with its message index and role, the restore gap, the tokens
//! that could have been reused (`unreused`) and a cause.

use crate::chat::prompt::{RenderedPrompt, Role};

/// A hybrid resume further than this behind the match is reported as [`Cause::Gap`].
pub const GAP_CAUSE: u32 = 256;

/// Miss causes in precedence order (highest first), D053.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum Cause {
    /// `--no-cache`.
    Disabled,
    /// The request continues state that was dropped (memory, `cache clear`, a short tail).
    Evicted,
    /// The prompt differs from stored state inside a conversation message: the message index
    /// and role name what changed.
    Diverged,
    /// Nothing reusable beyond the system prompt and tools: a new conversation or a side request.
    New,
    /// A hybrid resumed from a checkpoint more than [`GAP_CAUSE`] tokens behind the match.
    Gap,
}

impl Cause {
    /// Every cause in precedence order: the one list `/metrics` and `/mrun/status` label from
    /// (012 FR-013).
    pub const ALL: [Self; 5] = [
        Self::Disabled,
        Self::Evicted,
        Self::Diverged,
        Self::New,
        Self::Gap,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Disabled => "disabled",
            Self::Evicted => "evicted",
            Self::Diverged => "diverged",
            Self::New => "new",
            Self::Gap => "gap",
        }
    }
}

/// The API message containing a position.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MessageRef {
    pub index: u32,
    pub role: Role,
}

pub fn role_str(role: Role) -> &'static str {
    match role {
        Role::System => "system",
        Role::User => "user",
        Role::Assistant => "assistant",
        Role::Tool => "tool",
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MissReport {
    pub prompt: u32,
    /// `min(resume, P − 1)`.
    pub cached: u32,
    /// `P − cached`.
    pub processed: u32,
    /// First position where the prompt differs from the closest stored path; `None` when it
    /// extends that path.
    pub divergence: Option<u32>,
    pub message: Option<MessageRef>,
    /// Tokens between a restored checkpoint and the match, processed again.
    pub gap: u32,
    /// Tokens of the longest prefix shared with a stored or recently dropped path that were not
    /// reused: `min(shared, P − 1) − cached`.
    pub unreused: u32,
    /// The dropped-path ring has overwritten entries since the last clear: a `new` may be an old
    /// eviction.
    pub ring_wrapped: bool,
    /// `None` on a hit.
    pub cause: Option<Cause>,
}

/// What lookup found, before classification.
#[derive(Clone, Copy, Debug, Default)]
pub struct Found {
    /// Longest exact match.
    pub matched: u32,
    /// Prefill starts here (the match, or a checkpoint at or below it).
    pub resume: u32,
    /// The closest stored path has a token at `matched` that differs from the prompt.
    pub diverges: bool,
    /// The request continues where stored state was dropped.
    pub evicted: bool,
    pub disabled: bool,
    /// Longest prefix shared with a stored path or a dropped-path ring entry.
    pub shared: u32,
    pub ring_wrapped: bool,
}

/// The API message containing `pos`: the last mark starting at or before it.
pub fn message_at(prompt: &RenderedPrompt, pos: u32) -> Option<MessageRef> {
    let mark = prompt.messages.iter().rev().find(|m| m.start <= pos)?;
    let role = if pos < prompt.system_tools_end {
        Role::System
    } else {
        mark.role
    };
    Some(MessageRef {
        index: mark.message.unwrap_or(0),
        role,
    })
}

/// A whole conversation message (not system) lies before `pos`.
fn conversation_matched(prompt: &RenderedPrompt, pos: u32) -> bool {
    let marks = &prompt.messages;
    marks.iter().enumerate().any(|(i, mark)| {
        let end = marks
            .get(i + 1)
            .map_or(prompt.generation_start, |next| next.start);
        mark.role != Role::System && mark.start >= prompt.system_tools_end && end <= pos
    })
}

pub fn classify(prompt: &RenderedPrompt, found: Found) -> MissReport {
    let len = prompt.len();
    let cached = found.resume.min(len.saturating_sub(1));
    let matched = found.matched.min(len.saturating_sub(1));
    let divergence = found.diverges.then_some(found.matched);
    // A divergence after a whole conversation message matched is an edit of that conversation;
    // anything earlier is a new conversation or a side request (subagent, title).
    let in_conversation = |at: u32| conversation_matched(prompt, at);
    let cause = if found.disabled {
        Some(Cause::Disabled)
    } else if found.evicted {
        Some(Cause::Evicted)
    } else if let Some(at) = divergence {
        Some(if in_conversation(at) {
            Cause::Diverged
        } else {
            Cause::New
        })
    } else if matched.saturating_sub(cached) > GAP_CAUSE {
        Some(Cause::Gap)
    } else if found.matched <= prompt.system_tools_end && len > prompt.system_tools_end.max(1) {
        Some(Cause::New)
    } else {
        None
    };
    MissReport {
        prompt: len,
        cached,
        processed: len - cached,
        divergence,
        message: divergence.and_then(|at| message_at(prompt, at)),
        gap: matched.saturating_sub(cached),
        unreused: found
            .shared
            .max(found.matched)
            .min(len.saturating_sub(1))
            .saturating_sub(cached),
        ring_wrapped: found.ring_wrapped,
        cause,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chat::prompt::MessageMark;

    /// System+tools [0, 10), user [10, 20), assistant [20, 30), user [30, 40), header at 40.
    fn prompt() -> RenderedPrompt {
        let mark = |start, role, message| MessageMark {
            start,
            role,
            message: Some(message),
        };
        RenderedPrompt {
            tokens: (0..45).collect(),
            messages: vec![
                mark(0, Role::System, 0),
                mark(10, Role::User, 1),
                mark(20, Role::Assistant, 2),
                mark(30, Role::User, 3),
            ],
            system_tools_end: 10,
            generation_start: 40,
        }
    }

    fn found(matched: u32, diverges: bool) -> Found {
        Found {
            matched,
            resume: matched,
            diverges,
            ..Found::default()
        }
    }

    #[test]
    fn causes_by_divergence_position() {
        let p = prompt();
        let hit = classify(&p, found(44, false));
        assert_eq!((hit.cause, hit.cached, hit.processed), (None, 44, 1));
        // A different system prompt or tool list: a new conversation or a side request.
        let r = classify(&p, found(5, true));
        assert_eq!(r.cause, Some(Cause::New));
        assert_eq!(
            r.message,
            Some(MessageRef {
                index: 0,
                role: Role::System
            })
        );
        // The assistant reply re-rendered differently.
        let r = classify(&p, found(25, true));
        assert_eq!(r.cause, Some(Cause::Diverged));
        assert_eq!(
            r.message,
            Some(MessageRef {
                index: 2,
                role: Role::Assistant
            })
        );
        assert_eq!(r.divergence, Some(25));
        // Nothing stored at all.
        assert_eq!(classify(&p, Found::default()).cause, Some(Cause::New));
    }

    #[test]
    fn precedence_and_gap() {
        let p = prompt();
        let mut f = found(25, true);
        f.evicted = true;
        assert_eq!(classify(&p, f).cause, Some(Cause::Evicted));
        f.disabled = true;
        assert_eq!(classify(&p, f).cause, Some(Cause::Disabled));
        let mut checkpointed = found(44, false);
        checkpointed.resume = 30;
        let r = classify(&p, checkpointed);
        assert_eq!((r.cause, r.gap, r.cached, r.unreused), (None, 14, 30, 14));
    }

    #[test]
    fn gap_and_unreused() {
        let mut p = prompt();
        p.tokens = (0..1000).collect();
        p.generation_start = 990;
        // A hybrid resumed 300 tokens behind a full match: `gap`.
        let mut f = found(999, false);
        f.resume = 699;
        let r = classify(&p, f);
        assert_eq!((r.cause, r.gap, r.unreused), (Some(Cause::Gap), 300, 300));
        // A divergence takes precedence over the gap.
        f.matched = 980;
        f.diverges = true;
        assert_eq!(classify(&p, f).cause, Some(Cause::Diverged));
        // A dropped path shared further than the match counts as unreused.
        let mut f = found(25, true);
        f.shared = 40;
        f.evicted = true;
        let r = classify(&p, f);
        assert_eq!((r.cause, r.unreused), (Some(Cause::Evicted), 15));
    }
}
