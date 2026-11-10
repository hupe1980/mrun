//! The per-request decode hook (002 FR-036, D032) running on the engine thread: token masks
//! and `ignore_eos`. It never forces tokens; `budget_tokens` is accepted and not enforced.
//!
//! Masks for step n+1 are computed in `on_token(n)`, on the CPU while the engine runs the
//! pipelined forward pass of token n.

use crate::chat::constrain::{GrammarEnv, mask_of};
use crate::runtime::generate::DecodeHook;
use llguidance::Matcher;
use mrun_engine::TokenMask;
use std::sync::Arc;

/// When the answer grammar takes over.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Arm {
    /// From the first token.
    Now,
    /// The prompt opened a reasoning block: after the token that closes it (`</think>`).
    AfterReasoning { end: u32 },
    /// Thinking is on and the model opens the block itself (Qwen3): the first token may be
    /// `open` instead of the grammar's, and then the grammar arms after `end`.
    OptionalThink { open: u32, end: u32 },
}

/// The answer grammar of one request (`required`/named calls, JSON formats).
pub struct Constraint {
    pub env: Arc<GrammarEnv>,
    pub answer: Matcher,
    pub arm: Arm,
}

impl std::fmt::Debug for Constraint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Constraint")
            .field("arm", &self.arm)
            .finish()
    }
}

/// `parallel_tool_calls: false` under `auto`: after the first call's closer only EOS is allowed.
#[derive(Clone, Debug)]
pub struct SingleCall {
    pub closer: u32,
    pub eos_only: TokenMask,
}

#[derive(Debug)]
enum State {
    /// Before the grammar arms (or without one).
    Waiting,
    /// The grammar constrains every step.
    Active,
    /// The grammar finished or was abandoned.
    Done,
}

/// What a request asks of the decode loop beyond sampling.
#[derive(Debug)]
pub struct RequestHook {
    /// FR-031: EOS tokens are ordinary output; only the output limit ends generation.
    pub ignore_eos: bool,
    pub constraint: Option<Constraint>,
    /// Fixed mask banning the call opener (`tool_choice: none`).
    pub ban: Option<TokenMask>,
    pub single_call: Option<SingleCall>,
    state: State,
    /// Tokens seen so far.
    seen: u32,
    /// The single call ended: only EOS from now on.
    ended: bool,
}

impl RequestHook {
    pub fn new(
        ignore_eos: bool,
        constraint: Option<Constraint>,
        ban: Option<TokenMask>,
        single_call: Option<SingleCall>,
    ) -> Self {
        Self {
            ignore_eos,
            constraint,
            ban,
            single_call,
            state: State::Waiting,
            seen: 0,
            ended: false,
        }
    }

    /// `None` when the request needs nothing from the hook, so decode stays unhooked.
    pub fn boxed(self) -> Option<Box<dyn DecodeHook>> {
        (self.ignore_eos
            || self.constraint.is_some()
            || self.ban.is_some()
            || self.single_call.is_some())
        .then(|| Box::new(self) as Box<dyn DecodeHook>)
    }

    /// The mask the next step needs: EOS only after the single call, the active grammar's,
    /// else the fixed ban.
    fn next_mask(&mut self) -> Option<TokenMask> {
        if self.ended {
            return self.single_call.as_ref().map(|s| s.eos_only.clone());
        }
        if let (State::Active, Some(constraint)) = (&self.state, &mut self.constraint) {
            if constraint.answer.is_stopped() {
                self.state = State::Done;
            } else {
                let vocab = constraint.env.vocab();
                let mut mask = mask_of(&mut constraint.answer, vocab);
                // Before the first token, Qwen3 may open its reasoning block instead.
                if let (0, Arm::OptionalThink { open, .. }, Some(mask)) =
                    (self.seen, constraint.arm, mask.as_mut())
                {
                    mask.allow(open);
                }
                return mask;
            }
        }
        self.ban.clone()
    }
}

impl DecodeHook for RequestHook {
    fn masks(&self) -> bool {
        self.constraint.is_some() || self.ban.is_some() || self.single_call.is_some()
    }

    fn first_mask(&mut self) -> Option<TokenMask> {
        if let Some(constraint) = &self.constraint
            && matches!(constraint.arm, Arm::Now | Arm::OptionalThink { .. })
        {
            self.state = State::Active;
        }
        self.next_mask()
    }

    fn on_token(&mut self, id: u32) -> Option<TokenMask> {
        let first = self.seen == 0;
        self.seen += 1;
        if let Some(single) = &self.single_call
            && single.closer == id
        {
            self.ended = true;
        }
        if let Some(constraint) = &mut self.constraint {
            match (&self.state, constraint.arm) {
                (State::Active, Arm::OptionalThink { open, end }) if first && id == open => {
                    // The model thinks first: the grammar waits for the block's end.
                    constraint.arm = Arm::AfterReasoning { end };
                    self.state = State::Waiting;
                }
                (State::Active, _) => {
                    if constraint.answer.consume_token(id).is_err() {
                        // A token outside the grammar (EOS under `ignore_eos`): stop constraining.
                        eprintln!("mrun: grammar-abandoned");
                        self.state = State::Done;
                    }
                }
                (State::Waiting, Arm::AfterReasoning { end }) if id == end => {
                    self.state = State::Active;
                }
                _ => {}
            }
        }
        self.next_mask()
    }

    fn ignore_eos(&self) -> bool {
        self.ignore_eos
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::model::{ApiKind, ChatRequest, ResponseFormat};
    use crate::chat::constrain::plan;
    use crate::chat::tokenizer::test_support::byte_tokenizer;
    use mrun_engine::SyntaxId;

    fn json_hook(arm: Arm) -> (RequestHook, crate::chat::tokenizer::Tokenizer) {
        let tokenizer = byte_tokenizer();
        let env = Arc::new(GrammarEnv::new(&tokenizer, &[]).unwrap());
        let mut request = ChatRequest::new(ApiKind::Chat);
        request.response_format = ResponseFormat::JsonObject;
        let lark = plan(&request, SyntaxId::Hermes, &env).answer.unwrap();
        let answer = env.matcher(&lark).unwrap();
        let constraint = Constraint { env, answer, arm };
        (
            RequestHook::new(false, Some(constraint), None, None),
            tokenizer,
        )
    }

    /// Feeds `text`; returns whether every token was allowed by the mask before it.
    fn allowed(
        hook: &mut RequestHook,
        tokenizer: &crate::chat::tokenizer::Tokenizer,
        text: &str,
    ) -> bool {
        let mut mask = hook.first_mask();
        for id in tokenizer.encode(text).unwrap() {
            if mask.as_ref().is_some_and(|m| !m.is_allowed(id)) {
                return false;
            }
            mask = hook.on_token(id);
        }
        true
    }

    #[test]
    fn grammar_masks_from_the_first_token() {
        let (mut hook, tokenizer) = json_hook(Arm::Now);
        assert!(allowed(&mut hook, &tokenizer, "{\"a\": 1}"));
        let (mut hook, tokenizer) = json_hook(Arm::Now);
        assert!(!allowed(&mut hook, &tokenizer, "Sure: {}"));
    }

    #[test]
    fn qwen3_may_think_before_the_grammar() {
        let think = |text: &str| {
            let (mut hook, tokenizer) = json_hook(Arm::OptionalThink {
                open: byte_tokenizer().token_id("<think>").unwrap(),
                end: byte_tokenizer().token_id("</think>").unwrap(),
            });
            allowed(&mut hook, &tokenizer, text)
        };
        assert!(think(
            "<think>\nplan: prose is fine here\n</think>\n\n{\"a\": 1}"
        ));
        assert!(think("{\"a\": 1}"), "or answer at once");
        assert!(
            !think("<think>\nx\n</think>\n\nSure."),
            "the answer is still constrained"
        );
    }

    #[test]
    fn grammar_arms_after_an_open_reasoning_block() {
        let tokenizer = byte_tokenizer();
        let end = tokenizer.token_id("</think>").unwrap();
        let (mut hook, tokenizer) = json_hook(Arm::AfterReasoning { end });
        assert!(allowed(&mut hook, &tokenizer, "free text</think>{}"));
        let (mut hook, tokenizer) = json_hook(Arm::AfterReasoning { end });
        assert!(!allowed(&mut hook, &tokenizer, "free text</think>no"));
    }

    #[test]
    fn single_call_allows_only_eos_after_the_closer() {
        let tokenizer = byte_tokenizer();
        let closer = tokenizer.token_id("</tool_call>").unwrap();
        let mut eos_only = TokenMask::none_allowed(tokenizer.vocab_size());
        eos_only.allow(3);
        let mut hook = RequestHook::new(false, None, None, Some(SingleCall { closer, eos_only }));
        assert!(hook.first_mask().is_none());
        assert!(hook.on_token(5).is_none());
        let mask = hook.on_token(closer).unwrap();
        assert!(mask.is_allowed(3) && !mask.is_allowed(5));
        assert!(hook.on_token(3).is_some());
    }
}
