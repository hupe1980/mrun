//! Truncation canary (D038).
//!
//! A server that silently truncates a long prompt (a context window smaller than asked for,
//! context shift, dropped messages) answers faster, so a truncated turn must never count. The
//! harness cannot compare the reported prompt size with an expected count: every system has its
//! own tokenizer and chat template. Instead each system measures itself:
//!
//! 1. **Calibration**: a short prompt `S` (a fresh nonce, so nothing is cached) is sent alone,
//!    then twice with the canary line `C` inserted at the start. Neither can be truncated, so
//!    `prompt_tokens(S + C) − prompt_tokens(S)` is the canary's size `c` in this system's own
//!    tokens. The repeat shows whether the system's count includes cached tokens (equal counts),
//!    as the OpenAI and Anthropic usage fields specify.
//! 2. **Check**: the same prompt `P` is seen twice, once plain (the timed repetitions) and once
//!    with the canary (the natural-stop pass, or the synthetic warm-up). Untruncated,
//!    `prompt_tokens(P + C) − prompt_tokens(P)` is `c` again. A truncating server cuts both to its
//!    window, so the difference collapses: below `c / 2` the turn is truncated.
//!
//! The canary is a run-unique line (`mrun-canary` and random hex groups) placed at the very start
//! of the prompt: the first system or user text, so every turn of a conversation carries it at
//! the same place and prefix caching works as without it. A check is "unverified" (never passed)
//! when the system reports no prompt counts, when the canary measures under
//! [`MIN_CANARY_TOKENS`], when it cannot be placed in the request, or on a turn served from cache
//! by a system whose counts leave cached tokens out.

use crate::api::model::ApiKind;
use crate::bench::driver;
use crate::bench::suites::Rng;
use crate::bench::systems::System;
use serde_json::{Value, json};

/// A canary that measures fewer tokens than this cannot tell truncation from noise.
pub const MIN_CANARY_TOKENS: u32 = 8;
const GROUPS: usize = 8;

/// The run's canary line.
pub fn line(rng: &mut Rng) -> String {
    let groups: Vec<String> = (0..GROUPS)
        .map(|_| format!("{:04x}", rng.below(0x1_0000)))
        .collect();
    format!("mrun-canary {}", groups.join(" "))
}

/// Prepends `text` to `slot`: a string, or an array of content parts.
fn prepend(slot: &mut Value, text: &str, part_type: &str) -> bool {
    match slot {
        Value::String(s) if !s.is_empty() => {
            s.insert_str(0, &format!("{text}\n"));
            true
        }
        Value::Array(parts) if !parts.is_empty() => {
            parts.insert(0, json!({"type": part_type, "text": format!("{text}\n")}));
            true
        }
        _ => false,
    }
}

/// `body` with the canary at the start of its first system or user text; `None` when the request
/// has no text slot there.
pub fn insert(api: ApiKind, body: &Value, canary: &str) -> Option<Value> {
    let mut body = body.clone();
    let placed = match api {
        ApiKind::Chat => body
            .get_mut("messages")
            .and_then(|m| m.get_mut(0))
            .and_then(|m| m.get_mut("content"))
            .is_some_and(|c| prepend(c, canary, "text")),
        ApiKind::Messages => {
            let system = body
                .get_mut("system")
                .is_some_and(|s| prepend(s, canary, "text"));
            system
                || body
                    .get_mut("messages")
                    .and_then(|m| m.get_mut(0))
                    .and_then(|m| m.get_mut("content"))
                    .is_some_and(|c| prepend(c, canary, "text"))
        }
        ApiKind::Responses => {
            let instructions = body
                .get_mut("instructions")
                .is_some_and(|i| prepend(i, canary, "input_text"));
            instructions
                || match body.get_mut("input") {
                    Some(input @ Value::String(_)) => prepend(input, canary, "input_text"),
                    Some(Value::Array(items)) => items
                        .first_mut()
                        .and_then(|i| i.get_mut("content"))
                        .is_some_and(|c| prepend(c, canary, "input_text")),
                    _ => false,
                }
        }
    };
    placed.then_some(body)
}

/// What calibration found for one system and API.
#[derive(Clone, Debug, PartialEq)]
pub struct Canary {
    /// The canary's size in the system's tokens; `None` when it cannot be measured.
    pub tokens: Option<u32>,
    /// The system's prompt count includes cached tokens.
    pub counts_cached: bool,
    /// Why checks are unverified, when they are.
    pub unverified: Option<String>,
}

/// One turn's verdict.
#[derive(Clone, Debug, PartialEq)]
pub enum Check {
    Complete,
    /// The canary added `added` of its `expected` tokens.
    Truncated {
        added: i64,
        expected: u32,
    },
    Unverified(String),
}

/// A prompt as one system reported it: total prompt tokens and the cached part.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Count {
    pub prompt: Option<u32>,
    pub cached: Option<u32>,
}

impl Canary {
    pub fn unverified(reason: &str) -> Self {
        Self {
            tokens: None,
            counts_cached: false,
            unverified: Some(reason.into()),
        }
    }

    /// From the calibration counts: `S`, `S + C`, `S + C` again.
    pub fn from_counts(plain: Count, with: Count, again: Count) -> Self {
        let (Some(p), Some(w)) = (plain.prompt, with.prompt) else {
            return Self::unverified("the system reports no prompt token counts");
        };
        let tokens = w.saturating_sub(p);
        if tokens < MIN_CANARY_TOKENS {
            return Self::unverified(&format!(
                "the canary measured {tokens} tokens (under {MIN_CANARY_TOKENS})"
            ));
        }
        // A repeat served from cache that reports fewer prompt tokens leaves cached tokens out.
        let counts_cached = !(again.cached.unwrap_or(0) > 0 && again.prompt < with.prompt);
        Self {
            tokens: Some(tokens),
            counts_cached,
            unverified: None,
        }
    }

    /// Compares the same prompt with the canary (`with`) and without it (`plain`).
    pub fn check(&self, with: Count, plain: Count) -> Check {
        let Some(expected) = self.tokens else {
            return Check::Unverified(self.unverified.clone().unwrap_or_default());
        };
        let (Some(w), Some(p)) = (with.prompt, plain.prompt) else {
            return Check::Unverified("no prompt token count on this turn".into());
        };
        let cached = with.cached.unwrap_or(0) > 0 || plain.cached.unwrap_or(0) > 0;
        if cached && !self.counts_cached {
            return Check::Unverified(
                "cached turn on a system whose counts leave cache out".into(),
            );
        }
        let added = i64::from(w) - i64::from(p);
        if added * 2 < i64::from(expected) {
            Check::Truncated { added, expected }
        } else {
            Check::Complete
        }
    }
}

impl Check {
    pub fn reason(&self, turn: usize) -> Option<String> {
        match self {
            Check::Truncated { added, expected } => Some(format!(
                "turn {turn} truncated (the canary added {added} of its {expected} tokens)"
            )),
            _ => None,
        }
    }
}

/// The count of an outcome, for checks.
pub fn of(out: &driver::Outcome) -> Count {
    Count {
        prompt: out.prompt_tokens,
        cached: out.cached_tokens,
    }
}

/// The short calibration prompt: a nonce (nothing cached) and an instruction.
fn short_body(api: ApiKind, rng: &mut Rng) -> Value {
    let nonce = line(rng).replace("mrun-canary", "nonce");
    let text = format!("{nonce} Reply with ok.");
    match api {
        ApiKind::Responses => json!({"input": text, "temperature": 0}),
        _ => json!({"messages": [{"role": "user", "content": text}], "temperature": 0}),
    }
}

/// Measures `canary` on `system` over `api` (three one-token requests).
pub async fn calibrate(
    client: &reqwest::Client,
    rng: &mut Rng,
    system: &System,
    api: ApiKind,
    canary: &str,
) -> Canary {
    let plain = short_body(api, rng);
    let Some(with) = insert(api, &plain, canary) else {
        return Canary::unverified("the canary cannot be placed");
    };
    let mut counts = Vec::new();
    for body in [&plain, &with, &with] {
        let request = driver::prepare(api, body, Some(&system.model), 1, system.ignore_eos());
        match driver::send(
            client,
            &system.url,
            api,
            &request,
            system.ignore_eos(),
            None,
        )
        .await
        {
            Ok(out) => counts.push(of(&out)),
            Err(error) => return Canary::unverified(&format!("calibration failed: {error}")),
        }
    }
    Canary::from_counts(counts[0], counts[1], counts[2])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn c(prompt: u32, cached: u32) -> Count {
        Count {
            prompt: Some(prompt),
            cached: Some(cached),
        }
    }

    #[test]
    fn the_canary_goes_first_in_every_api() {
        let canary = "mrun-canary 0001";
        let chat = json!({"messages": [{"role": "system", "content": "be brief"},
            {"role": "user", "content": "hi"}]});
        let out = insert(ApiKind::Chat, &chat, canary).unwrap();
        assert_eq!(out["messages"][0]["content"], "mrun-canary 0001\nbe brief");
        assert_eq!(out["messages"][1]["content"], "hi");

        let parts = json!({"system": [{"type": "text", "text": "s"}], "messages": []});
        let out = insert(ApiKind::Messages, &parts, canary).unwrap();
        assert_eq!(out["system"][0]["text"], "mrun-canary 0001\n");
        assert_eq!(out["system"][1]["text"], "s");
        let no_system = json!({"messages": [{"role": "user", "content": "q"}]});
        let out = insert(ApiKind::Messages, &no_system, canary).unwrap();
        assert_eq!(out["messages"][0]["content"], "mrun-canary 0001\nq");

        let responses = json!({"input": [{"role": "user",
            "content": [{"type": "input_text", "text": "q"}]}]});
        let out = insert(ApiKind::Responses, &responses, canary).unwrap();
        assert_eq!(out["input"][0]["content"][0]["type"], "input_text");
        let out = insert(
            ApiKind::Responses,
            &json!({"instructions": "i", "input": "q"}),
            canary,
        )
        .unwrap();
        assert_eq!(out["instructions"], "mrun-canary 0001\ni");
        assert!(insert(ApiKind::Chat, &json!({"messages": []}), canary).is_none());
    }

    #[test]
    fn calibration_measures_the_canary_in_the_systems_tokens() {
        let canary = Canary::from_counts(c(20, 0), c(44, 0), c(44, 40));
        assert_eq!(canary.tokens, Some(24));
        assert!(canary.counts_cached);
        let excludes = Canary::from_counts(c(20, 0), c(44, 0), c(4, 40));
        assert!(!excludes.counts_cached);
        assert!(
            Canary::from_counts(c(20, 0), c(22, 0), c(22, 0))
                .unverified
                .is_some()
        );
        assert!(
            Canary::from_counts(Count::default(), c(1, 0), c(1, 0))
                .tokens
                .is_none()
        );
    }

    #[test]
    fn a_truncating_server_loses_the_canary() {
        let canary = Canary::from_counts(c(20, 0), c(44, 0), c(44, 40));
        assert_eq!(canary.check(c(30_024, 0), c(30_000, 0)), Check::Complete);
        // Truncated to an 8K window either way: the difference collapses.
        let truncated = canary.check(c(8_192, 0), c(8_192, 0));
        assert_eq!(
            truncated,
            Check::Truncated {
                added: 0,
                expected: 24
            }
        );
        assert!(truncated.reason(3).unwrap().starts_with("turn 3 truncated"));
        assert!(matches!(
            canary.check(Count::default(), c(1, 0)),
            Check::Unverified(_)
        ));
        let excludes = Canary::from_counts(c(20, 0), c(44, 0), c(4, 40));
        assert!(matches!(
            excludes.check(c(100, 90), c(76, 70)),
            Check::Unverified(_)
        ));
        assert_eq!(excludes.check(c(124, 0), c(100, 0)), Check::Complete);
    }

    #[test]
    fn canaries_differ_between_runs() {
        let (a, b) = (line(&mut Rng::new(1)), line(&mut Rng::new(2)));
        assert_ne!(a, b);
        assert!(a.starts_with("mrun-canary ") && a.split(' ').count() == 9);
    }
}
