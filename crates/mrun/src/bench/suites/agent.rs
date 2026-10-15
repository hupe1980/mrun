//! Agent trace replay (005 FR-003, FR-003a, FR-017): the recorded requests are sent in order,
//! unchanged except for the output limit (`reply_tokens`, ignore-EOS where supported) and
//! streaming. Each system caches its own replies, so reuse is measured up to the start of the
//! previous reply, equally for all systems. Session time = Σ (TTFT + fixed-length decode).
//!
//! The cell's samples are the per-turn TTFTs of every repetition; `turns` keeps each turn.

use crate::api::model::ApiKind;
use crate::bench::driver;
use crate::bench::machine;
use crate::bench::result::{Cell, CellStatus, Failure, Rerenders, Turn, TurnKind};
use crate::bench::stats;
use crate::bench::suites::{Ctx, not_run};
use crate::bench::systems::System;
use crate::bench::trace::{TraceRecord, api_messages, parse_api};
use serde_json::Value;
use std::collections::HashMap;

/// `new` for a conversation's first request, `repeated` when the previous request was from the
/// same conversation, `branching` otherwise (data model, SC-005).
pub fn kinds(records: &[TraceRecord]) -> Vec<TurnKind> {
    let mut seen = std::collections::HashSet::new();
    let mut previous: Option<&str> = None;
    records
        .iter()
        .map(|r| {
            let kind = if seen.insert(r.conversation.as_str()) {
                TurnKind::New
            } else if previous == Some(r.conversation.as_str()) {
                TurnKind::Repeated
            } else {
                TurnKind::Branching
            };
            previous = Some(&r.conversation);
            kind
        })
        .collect()
}

/// A `changed` miss at a message whose API content equals the previous request's message at
/// that index is a template re-render (FR-017).
pub fn is_rerender(
    api: ApiKind,
    previous: Option<&Value>,
    body: &Value,
    cause: Option<&str>,
    index: Option<u32>,
) -> bool {
    let (Some(previous), Some("changed"), Some(index)) = (previous, cause, index) else {
        return false;
    };
    let (before, now) = (api_messages(api, previous), api_messages(api, body));
    let i = index as usize;
    i < before.len() && before.get(i) == now.get(i)
}

pub async fn run(
    ctx: &mut Ctx,
    system: &System,
    name: &str,
    records: &[TraceRecord],
    reps: u32,
) -> Cell {
    let cell_name = format!("agent/{name}");
    let api = records
        .first()
        .and_then(|r| parse_api(&r.api))
        .unwrap_or(ApiKind::Chat);
    if !system.apis.contains(&api) {
        let mut cell = not_run(
            &system.name,
            &cell_name,
            api.as_str(),
            "ms",
            "the system lacks this API",
        );
        cell.status = CellStatus::NotApplicable;
        return cell;
    }
    let kinds = kinds(records);
    let mut cell = Cell {
        system: system.name.clone(),
        cell: cell_name,
        api: api.as_str().into(),
        unit: "ms".into(),
        thermal_start: machine::thermal(),
        ..Cell::default()
    };
    let mut turns = Vec::new();
    let mut sessions = Vec::new();
    let mut rerenders = Rerenders {
        turns: 0,
        tokens: 0,
    };
    for rep in 1..=reps {
        ctx.reset(system, rep).await;
        let mut session_ms = 0.0;
        let mut last_body: HashMap<&str, &Value> = HashMap::new();
        for (i, record) in records.iter().enumerate() {
            let request = driver::prepare(
                api,
                &record.body,
                Some(&system.model),
                record.reply_tokens,
                system.ignore_eos(),
            );
            let outcome = driver::send(
                &ctx.client,
                &system.url,
                api,
                &request,
                system.ignore_eos(),
                None,
            )
            .await;
            let previous = last_body.insert(&record.conversation, &record.body);
            let out = match outcome {
                Ok(out) => out,
                Err(error) => {
                    cell.failures.push(Failure {
                        rep,
                        error: format!("turn {}: {error}", i + 1),
                    });
                    continue;
                }
            };
            if driver::truncated(out.prompt_tokens, record.prompt_tokens) {
                cell.failures.push(Failure {
                    rep,
                    error: "truncated input".into(),
                });
            }
            cell.early_eos += u32::from(out.early_eos(record.reply_tokens));
            cell.malformed_tool_calls +=
                driver::malformed_calls(api, &record.body, &out.tool_calls);
            let (input, cached) = (
                out.prompt_tokens.unwrap_or(0),
                out.cached_tokens.unwrap_or(0),
            );
            if rep == 1
                && system.is_mrun()
                && is_rerender(
                    api,
                    previous,
                    &record.body,
                    out.cause.as_deref(),
                    out.message_index,
                )
            {
                rerenders.turns += 1;
                rerenders.tokens += u64::from(input - cached.min(input));
            }
            session_ms += out.total_ms().unwrap_or(0.0);
            let Some(ttft) = out.ttft_ms else {
                cell.failures.push(Failure {
                    rep,
                    error: format!("turn {}: no content", i + 1),
                });
                continue;
            };
            cell.samples.push(ttft);
            turns.push(Turn {
                n: i as u32 + 1,
                conversation: record.conversation.clone(),
                kind: kinds[i],
                ttft_ms: ttft,
                tfut_ms: out.tfut_ms.unwrap_or(ttft),
                input,
                cached,
                output: out.output_tokens.unwrap_or(out.token_times_ms.len() as u32),
                cause: out.cause,
                message_index: out.message_index,
                rep: Some(rep),
            });
        }
        sessions.push(session_ms / 1000.0);
    }
    cell.thermal_end = machine::thermal();
    cell.session_s = stats::summarize(&sessions).map(|s| s.median);
    cell.turns = Some(turns);
    if system.is_mrun() {
        cell.template_rerenders = Some(rerenders);
    }
    if !cell.failures.is_empty() || cell.samples.is_empty() {
        cell.status = CellStatus::Failed;
    }
    cell.summarize();
    cell
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn record(conversation: &str) -> TraceRecord {
        TraceRecord {
            api: "chat".into(),
            conversation: conversation.into(),
            body: json!({}),
            prompt_tokens: 0,
            reply_tokens: 8,
            tool_gaps: None,
        }
    }

    #[test]
    fn turn_kinds() {
        let r: Vec<_> = ["a", "a", "b", "a", "b"].into_iter().map(record).collect();
        use TurnKind::*;
        assert_eq!(kinds(&r), vec![New, Repeated, New, Branching, Branching]);
    }

    #[test]
    fn rerender_needs_equal_content_at_the_index() {
        let before = json!({"messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]});
        let same = json!({"messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "u"},
            {"role": "assistant", "content": "a"}]});
        let edited = json!({"messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "v"}]});
        assert!(is_rerender(
            ApiKind::Chat,
            Some(&before),
            &same,
            Some("changed"),
            Some(1)
        ));
        assert!(!is_rerender(
            ApiKind::Chat,
            Some(&before),
            &edited,
            Some("changed"),
            Some(1)
        ));
        assert!(!is_rerender(
            ApiKind::Chat,
            Some(&before),
            &same,
            Some("new"),
            Some(1)
        ));
        assert!(!is_rerender(
            ApiKind::Chat,
            None,
            &same,
            Some("changed"),
            Some(1)
        ));
    }
}
