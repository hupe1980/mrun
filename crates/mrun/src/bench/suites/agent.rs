//! Agent trace replay (005 FR-003, FR-003a, FR-017, FR-020, FR-022, D038).
//!
//! **Timing pass**: the recorded requests are sent in order, unchanged except for the output
//! limit (`reply_tokens`, ignore-EOS where supported) and streaming, once per repetition after a
//! cache reset. The history in every request is the *recorded* conversation, including the
//! recorded replies, not the system's own: every system sees byte-identical prompts, and reuse
//! is measured up to the start of the previous reply, equally for all. A repetition is one
//! session; session time = Σ (TTFT + fixed-length decode).
//!
//! **Concurrent class** (FR-020): auxiliary requests recorded with an offset after a main
//! request are sent at that offset beside it, in parallel; the cell reports main-turn and
//! auxiliary TTFT separately. Traces without offsets replay sequentially.
//!
//! **Natural-stop pass**: one more replay after a reset with the client's own output limit and
//! no ignore-EOS, the canary at the start of every prompt. Quality signals (malformed tool calls,
//! empty replies at end-of-sequence) come from this pass only: forced-length decoding makes a
//! model run past where it would stop and invalidates them. For mrun it also measures the
//! splice hit rate per client (FR-022) from `/mrun/status` `cache.spliced_tokens_total`.
//!
//! **Validity** (FR-005): a session with any failed turn (an error, no content), a turn that
//! ended early (more than 10 % short of the forced length, FR-004) or a truncated turn (the
//! canary check against the natural-stop pass) is excluded as a whole and listed in
//! `incomplete` with the reason; skipping a turn would make the session look faster. The cell's
//! samples, turns and session times come from complete sessions only.

use crate::api::model::ApiKind;
use crate::bench::canary::{self, Check, Count};
use crate::bench::driver::{self, Finish, IgnoreEos, Outcome};
use crate::bench::machine;
use crate::bench::result::{
    Cell, CellStatus, Failure, Quality, SessionTime, SpliceRate, Turn, TurnClass, TurnKind,
};
use crate::bench::stats;
use crate::bench::suites::{Ctx, not_run};
use crate::bench::systems::System;
use crate::bench::trace::{TraceRecord, holds_earlier_reply, parse_api};
use std::collections::BTreeMap;
use std::time::Duration;

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

/// The replay groups: a main request with the auxiliary requests recorded at an offset after
/// it (sent beside it), or a single request. Indices into the records, in order.
pub fn groups(records: &[TraceRecord]) -> Vec<Vec<usize>> {
    let mut out: Vec<Vec<usize>> = Vec::new();
    for (i, record) in records.iter().enumerate() {
        let joins = record.class == TurnClass::Auxiliary
            && record.offset_ms.is_some()
            && out
                .last()
                .and_then(|group| group.first())
                .is_some_and(|&first| records[first].class == TurnClass::Main);
        match out.last_mut() {
            Some(group) if joins => group.push(i),
            _ => out.push(vec![i]),
        }
    }
    out
}

/// One timing repetition.
#[derive(Debug, Default)]
struct Session {
    turns: Vec<Turn>,
    counts: Vec<Count>,
    /// The first invalid turn ends the session.
    error: Option<String>,
    ms: f64,
    short: u32,
}

/// The natural-stop pass: quality counts, each turn's count with the canary and, for mrun,
/// the splice hit rate per client.
async fn natural_pass(
    ctx: &mut Ctx,
    system: &System,
    api: ApiKind,
    records: &[TraceRecord],
) -> (Quality, Vec<Option<Count>>, Option<Vec<SpliceRate>>) {
    ctx.reset(system, 0).await;
    let mut quality = Quality {
        complete: true,
        ..Quality::default()
    };
    let mut splices: BTreeMap<String, (u32, u32)> = BTreeMap::new();
    let mut counts = Vec::with_capacity(records.len());
    for (i, record) in records.iter().enumerate() {
        let canaried = canary::insert(api, &record.body, &ctx.canary);
        let body = canaried.as_ref().unwrap_or(&record.body);
        let request = driver::prepare_natural(api, body, Some(&system.model));
        let eligible = system.is_mrun() && holds_earlier_reply(api, &record.body);
        let before = if eligible {
            spliced_total(ctx, system).await
        } else {
            None
        };
        let outcome = driver::send(
            &ctx.client,
            &system.url,
            api,
            &request,
            IgnoreEos::None,
            None,
        )
        .await;
        match outcome {
            Ok(out) => {
                if eligible {
                    let after = spliced_total(ctx, system).await;
                    let spliced = match (before, after) {
                        (Some(before), Some(after)) => after > before,
                        _ => out.spliced.is_some_and(|n| n > 0),
                    };
                    let entry = splices.entry(record.client.clone()).or_default();
                    entry.0 += 1;
                    entry.1 += u32::from(spliced);
                }
                quality.turns += 1;
                quality.malformed_tool_calls +=
                    driver::malformed_calls(api, &record.body, &out.tool_calls);
                quality.early_eos += u32::from(out.empty_reply());
                quality.length_capped += u32::from(out.finish == Finish::Length);
                counts.push(canaried.is_some().then(|| canary::of(&out)));
            }
            Err(error) => {
                quality.complete = false;
                quality.failures.push(Failure {
                    rep: 0,
                    error: format!("turn {}: {error}", i + 1),
                });
                counts.push(None);
            }
        }
    }
    let splice_rate = system.is_mrun().then(|| {
        splices
            .into_iter()
            .map(|(client, (eligible, spliced))| SpliceRate {
                client,
                eligible,
                spliced,
                rate: (eligible > 0).then(|| f64::from(spliced) / f64::from(eligible)),
            })
            .collect()
    });
    (quality, counts, splice_rate)
}

/// `/mrun/status` `cache.spliced_tokens_total` (003 FR-016), when the server reports it.
async fn spliced_total(ctx: &Ctx, system: &System) -> Option<u64> {
    let response = ctx
        .client
        .get(format!("{}/mrun/status", system.url))
        .timeout(Duration::from_secs(5))
        .send()
        .await
        .ok()?;
    let status: serde_json::Value = response.json().await.ok()?;
    status["cache"]["spliced_tokens_total"].as_u64()
}

/// Sends one timed request, `delay` after the group started.
async fn timed(
    ctx: &Ctx,
    system: &System,
    api: ApiKind,
    record: &TraceRecord,
    delay: Duration,
) -> Result<Outcome, String> {
    if !delay.is_zero() {
        tokio::time::sleep(delay).await;
    }
    let request = driver::prepare(
        api,
        &record.body,
        Some(&system.model),
        record.reply_tokens,
        system.ignore_eos(),
    );
    driver::send(
        &ctx.client,
        &system.url,
        api,
        &request,
        system.ignore_eos(),
        None,
    )
    .await
}

/// One timing repetition over every group; returns the session and whether a group ran
/// concurrently.
async fn session(
    ctx: &Ctx,
    system: &System,
    api: ApiKind,
    records: &[TraceRecord],
    kinds: &[TurnKind],
    rep: u32,
) -> (Session, bool) {
    let mut session = Session::default();
    let mut concurrent = false;
    for group in groups(records) {
        concurrent |= group.len() > 1;
        let sends = group.iter().map(|&i| {
            let record = &records[i];
            let delay = if i == group[0] {
                Duration::ZERO
            } else {
                Duration::from_secs_f64(record.offset_ms.unwrap_or(0.0).max(0.0) / 1000.0)
            };
            timed(ctx, system, api, record, delay)
        });
        let outcomes = futures_util::future::join_all(sends).await;
        let mut group_ms: f64 = 0.0;
        for (&i, outcome) in group.iter().zip(outcomes) {
            let record = &records[i];
            let out = match outcome {
                Ok(out) => out,
                Err(error) => {
                    session.error = Some(format!("turn {}: {error}", i + 1));
                    return (session, concurrent);
                }
            };
            let Some(ttft) = out.ttft_ms else {
                session.error = Some(format!("turn {}: no content", i + 1));
                return (session, concurrent);
            };
            if out.ended_early(record.reply_tokens) {
                session.error = Some(format!(
                    "turn {} ended early ({} of {} tokens)",
                    i + 1,
                    out.output_tokens(),
                    record.reply_tokens
                ));
                return (session, concurrent);
            }
            session.short += u32::from(out.stopped_short(record.reply_tokens));
            let offset = (i != group[0]).then(|| record.offset_ms.unwrap_or(0.0));
            // A concurrent group takes as long as its longest request (from the group start).
            group_ms = group_ms.max(offset.unwrap_or(0.0) + out.total_ms().unwrap_or(ttft));
            session.counts.push(canary::of(&out));
            session.turns.push(Turn {
                n: i as u32 + 1,
                conversation: record.conversation.clone(),
                kind: kinds[i],
                ttft_ms: ttft,
                tfut_ms: out.tfut_ms.unwrap_or(ttft),
                input: out.prompt_tokens.unwrap_or(0),
                cached: out.cached_tokens.unwrap_or(0),
                output: out.output_tokens(),
                cause: out.cause,
                message_index: out.message_index,
                message_role: out.message_role,
                gap: out.gap,
                unreused: out.unreused,
                rep: Some(rep),
                class: record.class,
                offset_ms: offset,
            });
        }
        session.ms += group_ms;
    }
    (session, concurrent)
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
    let unsupported = if !system.runs("agent") {
        Some("the profile runs synthetic cells only")
    } else if !system.apis.contains(&api) {
        Some("the system lacks this API")
    } else {
        None
    };
    if let Some(reason) = unsupported {
        let mut cell = not_run(&system.name, &cell_name, api.as_str(), "ms", reason);
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
    let mut sessions = Vec::new();
    let mut concurrent = false;
    for rep in 1..=reps {
        ctx.reset(system, rep).await;
        let (one, parallel) = session(ctx, system, api, records, &kinds, rep).await;
        concurrent |= parallel;
        sessions.push(one);
    }
    let measured = ctx.canary(system, api).await;
    let (quality, canaried, splice_rate) = natural_pass(ctx, system, api, records).await;
    cell.thermal_end = machine::thermal();

    let mut verified = 0;
    let mut times = Vec::new();
    let mut turns = Vec::new();
    for (rep, session) in (1..).zip(sessions) {
        if let Some(error) = session.error {
            cell.failures.push(Failure {
                rep,
                error: error.clone(),
            });
            cell.incomplete.push(Failure { rep, error });
            continue;
        }
        let mut truncated = None;
        let mut session_verified = 0;
        for (i, plain) in session.counts.iter().enumerate() {
            let check = match canaried.get(i).copied().flatten() {
                Some(with) => measured.check(with, *plain),
                None => Check::Unverified("no canary count".into()),
            };
            session_verified += u32::from(check == Check::Complete);
            if let Some(reason) = check.reason(i + 1) {
                truncated = Some(reason);
                break;
            }
        }
        if let Some(reason) = truncated {
            cell.failures.push(Failure {
                rep,
                error: reason.clone(),
            });
            cell.incomplete.push(Failure { rep, error: reason });
            continue;
        }
        verified += session_verified;
        cell.short_replies += session.short;
        times.push(session.ms / 1000.0);
        cell.session_times.push(SessionTime {
            rep,
            s: session.ms / 1000.0,
        });
        cell.samples.extend(session.turns.iter().map(|t| t.ttft_ms));
        turns.extend(session.turns);
    }
    let checked = turns.len() as u32;
    cell.truncation_check = Some(match &measured.unverified {
        Some(reason) => format!("unverified: {reason}"),
        None if verified < checked => format!(
            "canary ({} of {checked} turns unverified)",
            checked - verified
        ),
        None => "canary".into(),
    });
    if concurrent {
        let median_of = |class: TurnClass| {
            let ttfts: Vec<f64> = turns
                .iter()
                .filter(|t| t.class == class)
                .map(|t| t.ttft_ms)
                .collect();
            stats::median(&ttfts)
        };
        cell.main_ttft = median_of(TurnClass::Main);
        cell.aux_ttft = median_of(TurnClass::Auxiliary);
    }
    cell.class = Some(
        if concurrent {
            "concurrent"
        } else {
            "sequential"
        }
        .into(),
    );
    cell.complete_sessions = Some(times.len() as u32);
    cell.session_s = stats::median(&times);
    cell.turns = Some(turns);
    cell.quality = Some(quality);
    cell.splice_hit_rate = splice_rate;
    if cell.samples.is_empty() {
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
            reply_tokens: 8,
            ..TraceRecord::default()
        }
    }

    #[test]
    fn turn_kinds() {
        let r: Vec<_> = ["a", "a", "b", "a", "b"].into_iter().map(record).collect();
        use TurnKind::*;
        assert_eq!(kinds(&r), vec![New, Repeated, New, Branching, Branching]);
    }

    #[test]
    fn auxiliary_requests_with_an_offset_join_the_main_turn_before_them() {
        let aux = |offset: Option<f64>| TraceRecord {
            class: TurnClass::Auxiliary,
            offset_ms: offset,
            ..record("t")
        };
        let r = vec![
            record("a"),
            aux(Some(100.0)),
            aux(Some(300.0)),
            record("a"),
            aux(None),
            aux(Some(5.0)),
        ];
        // An auxiliary request without an offset replays alone, as does one after it.
        assert_eq!(groups(&r), vec![vec![0, 1, 2], vec![3], vec![4], vec![5]]);
        assert_eq!(groups(&[record("a"), record("a")]), vec![vec![0], vec![1]]);
    }
}
