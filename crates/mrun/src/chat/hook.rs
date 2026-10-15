//! The per-request decode hook (002 FR-036) running on the engine thread: token masks,
//! grammar-forced tokens and `ignore_eos`.
//!
//! Masks for step n+1 are computed in `on_token(n)`, on the CPU while the engine runs the
//! pipelined forward pass of token n.

use crate::chat::constrain::{GrammarEnv, mask_of};
use crate::runtime::generate::DecodeHook;
use llguidance::Matcher;
use mrun_engine::TokenMask;
use std::sync::Arc;

/// Grammar state of one request.
pub struct Constraint {
    pub env: Arc<GrammarEnv>,
    /// Compiled grammar of one call, cloned at each opener (strict tools under `auto`).
    pub on_opener: Option<(u32, Matcher)>,
    /// Compiled grammar of the whole answer, armed once reasoning is over.
    pub answer: Option<Matcher>,
    /// Token that closes the reasoning block while it is open (`</think>`).
    pub reasoning_end: Option<u32>,
    /// Fixed mask banning the call opener (`tool_choice: none`).
    pub ban: Option<TokenMask>,
}

/// What a request asks of the decode loop beyond sampling.
#[derive(Default)]
pub struct RequestHook {
    /// FR-031: EOS tokens are ordinary output; only the output limit ends generation.
    pub ignore_eos: bool,
    pub constraint: Option<Constraint>,
    active: Option<Matcher>,
}

impl std::fmt::Debug for Constraint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Constraint")
            .field("on_opener", &self.on_opener.as_ref().map(|(id, _)| id))
            .field("answer", &self.answer.is_some())
            .field("reasoning_end", &self.reasoning_end)
            .field("ban", &self.ban.is_some())
            .finish()
    }
}

impl std::fmt::Debug for RequestHook {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RequestHook")
            .field("ignore_eos", &self.ignore_eos)
            .field("constraint", &self.constraint)
            .field("active", &self.active.is_some())
            .finish()
    }
}

impl RequestHook {
    pub fn new(ignore_eos: bool, constraint: Option<Constraint>) -> Self {
        Self {
            ignore_eos,
            constraint,
            active: None,
        }
    }

    /// `None` when the request needs nothing from the hook, so decode stays unhooked.
    pub fn boxed(self) -> Option<Box<dyn DecodeHook>> {
        (self.ignore_eos || self.constraint.is_some())
            .then(|| Box::new(self) as Box<dyn DecodeHook>)
    }

    /// The mask the next step needs: the active grammar's, else the fixed ban.
    fn next_mask(&mut self) -> Option<TokenMask> {
        let constraint = self.constraint.as_ref()?;
        if let Some(matcher) = &mut self.active {
            if matcher.is_stopped() {
                self.active = None;
            } else {
                return mask_of(matcher, constraint.env.vocab());
            }
        }
        constraint.ban.clone()
    }

    fn arm_answer(&mut self) {
        if let Some(constraint) = &mut self.constraint
            && constraint.reasoning_end.is_none()
            && let Some(answer) = constraint.answer.take()
        {
            self.active = Some(answer);
        }
    }
}

impl DecodeHook for RequestHook {
    fn first_mask(&mut self) -> Option<TokenMask> {
        self.arm_answer();
        self.next_mask()
    }

    fn on_token(&mut self, id: u32) -> Option<TokenMask> {
        let constraint = self.constraint.as_mut()?;
        if let Some(matcher) = &mut self.active {
            if matcher.consume_token(id).is_err() {
                // A token outside the grammar (forced by the engine, e.g. EOS): stop constraining.
                eprintln!("mrun: grammar-abandoned");
                self.active = None;
            }
        } else if constraint.reasoning_end == Some(id) {
            constraint.reasoning_end = None;
            self.arm_answer();
        } else if let Some((opener, prototype)) = &constraint.on_opener
            && *opener == id
        {
            self.active = Some(prototype.deep_clone());
        }
        self.next_mask()
    }

    fn forced_tokens(&mut self) -> Vec<u32> {
        match &mut self.active {
            Some(matcher) if !matcher.is_stopped() => matcher.compute_ff_tokens(),
            _ => Vec::new(),
        }
    }

    fn ignore_eos(&self) -> bool {
        self.ignore_eos
    }
}
