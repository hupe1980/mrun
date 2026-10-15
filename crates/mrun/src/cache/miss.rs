//! Why a request was not served entirely from cached state (003 FR-008, data model MissReport).

use crate::chat::prompt::{RenderedPrompt, Role};

/// Miss causes in precedence order (highest first).
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum Cause {
    /// `--no-cache`.
    Disabled,
    /// State existed under another cache key (model or template file changed).
    Model,
    /// The request continues into state removed by LRU, `cache clear`, the tiny-tail rule or
    /// memory pressure.
    Evicted,
    /// A system prompt, tool list or message was changed, removed or re-rendered.
    Changed,
    /// A new conversation (shares at most the system and tools region), or nothing stored.
    New,
}

impl Cause {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Disabled => "disabled",
            Self::Model => "model",
            Self::Evicted => "evicted",
            Self::Changed => "changed",
            Self::New => "new",
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
    /// Something is stored under this key.
    pub stored: bool,
    /// The request continues where a child was evicted.
    pub evicted: bool,
    /// State existed under a previous key.
    pub rekeyed: bool,
    pub disabled: bool,
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
    let cause = if found.disabled {
        Some(Cause::Disabled)
    } else if found.rekeyed && found.matched == 0 {
        Some(Cause::Model)
    } else if found.evicted {
        Some(Cause::Evicted)
    } else if !found.stored {
        Some(Cause::New)
    } else if let Some(at) = divergence {
        if at < prompt.system_tools_end || conversation_matched(prompt, at) {
            Some(Cause::Changed)
        } else {
            Some(Cause::New)
        }
    } else if found.matched == 0 {
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
            stored: true,
            ..Found::default()
        }
    }

    #[test]
    fn causes_by_divergence_position() {
        let p = prompt();
        let hit = classify(&p, found(44, false));
        assert_eq!((hit.cause, hit.cached, hit.processed), (None, 44, 1));
        // Tool list or system prompt edited.
        let r = classify(&p, found(5, true));
        assert_eq!(r.cause, Some(Cause::Changed));
        assert_eq!(
            r.message,
            Some(MessageRef {
                index: 0,
                role: Role::System
            })
        );
        // A different first user message: a new conversation.
        assert_eq!(classify(&p, found(12, true)).cause, Some(Cause::New));
        // The re-rendered assistant reply after a matched user message.
        let r = classify(&p, found(25, true));
        assert_eq!(r.cause, Some(Cause::Changed));
        assert_eq!(
            r.message,
            Some(MessageRef {
                index: 2,
                role: Role::Assistant
            })
        );
        assert_eq!(r.divergence, Some(25));
    }

    #[test]
    fn precedence_and_gap() {
        let p = prompt();
        let mut f = found(25, true);
        f.evicted = true;
        assert_eq!(classify(&p, f).cause, Some(Cause::Evicted));
        f.disabled = true;
        assert_eq!(classify(&p, f).cause, Some(Cause::Disabled));
        let nothing = Found::default();
        assert_eq!(classify(&p, nothing).cause, Some(Cause::New));
        let rekeyed = Found {
            rekeyed: true,
            ..Found::default()
        };
        assert_eq!(classify(&p, rekeyed).cause, Some(Cause::Model));
        let mut checkpointed = found(44, false);
        checkpointed.resume = 30;
        let r = classify(&p, checkpointed);
        assert_eq!((r.cause, r.gap, r.cached), (None, 14, 30));
    }
}
