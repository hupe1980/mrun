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

/// Anthropic `budget_tokens` (FR-013): once the reasoning block has used its budget, the
/// tokens that close it are forced.
#[derive(Clone, Debug, PartialEq)]
pub struct ThinkBudget {
    pub budget: u32,
    pub used: u32,
    /// Inside the reasoning block now.
    pub inside: bool,
    /// Token that opens a block the model writes itself (`<think>`).
    pub start: Option<u32>,
    /// Token that ends the block when the model closes it itself (`</think>`, Harmony `<|end|>`).
    pub end: u32,
    /// Tokens forced to close it (`\n</think>\n\n`; Harmony: end analysis, open `final`).
    pub closer: Vec<u32>,
    /// Harmony: reasoning starts after `<|channel|>analysis<|message|>`.
    pub harmony: Option<HarmonyIds>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct HarmonyIds {
    pub channel: u32,
    pub message: u32,
    pub analysis: Vec<u32>,
}

impl ThinkBudget {
    /// Tracks one output token; returns the closer once the budget is spent.
    fn on_token(&mut self, id: u32, header: &mut Vec<u32>, in_header: &mut bool) -> Vec<u32> {
        if let Some(h) = &self.harmony {
            if id == h.channel {
                *in_header = true;
                header.clear();
                return Vec::new();
            }
            if *in_header {
                if id == h.message {
                    *in_header = false;
                    self.inside = header.as_slice() == h.analysis.as_slice();
                } else {
                    header.push(id);
                }
                return Vec::new();
            }
        }
        if !self.inside {
            self.inside = self.start == Some(id);
            return Vec::new();
        }
        if id == self.end {
            self.inside = false;
            return Vec::new();
        }
        self.used += 1;
        if self.used >= self.budget {
            self.inside = false;
            return self.closer.clone();
        }
        Vec::new()
    }
}

/// What a request asks of the decode loop beyond sampling.
#[derive(Default)]
pub struct RequestHook {
    /// FR-031: EOS tokens are ordinary output; only the output limit ends generation.
    pub ignore_eos: bool,
    pub constraint: Option<Constraint>,
    pub budget: Option<ThinkBudget>,
    active: Option<Matcher>,
    /// Tokens to force next (a budget closer).
    queued: Vec<u32>,
    harmony_header: Vec<u32>,
    in_header: bool,
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
            .field("budget", &self.budget)
            .field("constraint", &self.constraint)
            .field("active", &self.active.is_some())
            .finish()
    }
}

impl RequestHook {
    pub fn new(
        ignore_eos: bool,
        constraint: Option<Constraint>,
        budget: Option<ThinkBudget>,
    ) -> Self {
        Self {
            ignore_eos,
            constraint,
            budget,
            ..Self::default()
        }
    }

    /// `None` when the request needs nothing from the hook, so decode stays unhooked.
    pub fn boxed(self) -> Option<Box<dyn DecodeHook>> {
        (self.ignore_eos || self.constraint.is_some() || self.budget.is_some())
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
        if let Some(budget) = &mut self.budget {
            let closer = budget.on_token(id, &mut self.harmony_header, &mut self.in_header);
            if !closer.is_empty() {
                self.queued = closer;
            }
        }
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
        if !self.queued.is_empty() {
            return std::mem::take(&mut self.queued);
        }
        match &mut self.active {
            Some(matcher) if !matcher.is_stopped() => matcher.compute_ff_tokens(),
            _ => Vec::new(),
        }
    }

    fn ignore_eos(&self) -> bool {
        self.ignore_eos
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn drive(hook: &mut RequestHook, ids: &[u32]) -> Vec<u32> {
        let mut out = Vec::new();
        for &id in ids {
            hook.on_token(id);
            let forced = hook.forced_tokens();
            for &f in &forced {
                hook.on_token(f);
            }
            out.extend(forced);
        }
        out
    }

    #[test]
    fn syn_harmony_budget_closes_analysis() {
        // ids: 1 <|channel|>, 2 <|message|>, 3 <|end|>, 7 "analysis", 8 "final", 9 text.
        let budget = ThinkBudget {
            budget: 3,
            used: 0,
            inside: false,
            start: None,
            end: 3,
            closer: vec![3, 100, 101, 1, 8, 2],
            harmony: Some(HarmonyIds {
                channel: 1,
                message: 2,
                analysis: vec![7],
            }),
        };
        let mut hook = RequestHook::new(false, None, Some(budget.clone()));
        // A commentary message is not reasoning.
        assert!(drive(&mut hook, &[1, 8, 2, 9, 9, 9, 9, 3]).is_empty());
        // Analysis: the third reasoning token triggers the closer once.
        let forced = drive(&mut hook, &[1, 7, 2, 9, 9, 9, 9]);
        assert_eq!(forced, vec![3, 100, 101, 1, 8, 2]);
        // A model that closes analysis in time is left alone.
        let mut hook = RequestHook::new(false, None, Some(budget));
        assert!(drive(&mut hook, &[1, 7, 2, 9, 3, 9, 9, 9]).is_empty());
    }

    #[test]
    fn think_budget_counts_from_an_open_block_or_a_written_think() {
        let budget = ThinkBudget {
            budget: 2,
            used: 0,
            inside: false,
            start: Some(5),
            end: 6,
            closer: vec![6],
            harmony: None,
        };
        let mut hook = RequestHook::new(false, None, Some(budget));
        assert!(drive(&mut hook, &[9, 9]).is_empty());
        assert_eq!(drive(&mut hook, &[5, 9, 9, 9]), vec![6]);
    }
}
