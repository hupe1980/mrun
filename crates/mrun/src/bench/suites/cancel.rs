//! Cancellation latency, mrun only (005 FR-001, FR-004): at each of the random interruption
//! points a streamed request is dropped (during prefill, or after a random number of content
//! events), and the sample is the time from the disconnect to the first byte of the next
//! request, a one-token reply to a short prompt.

use crate::api::model::ApiKind;
use crate::bench::driver;
use crate::bench::machine;
use crate::bench::result::{Cell, CellStatus, Failure};
use crate::bench::suites::{Ctx, not_run};
use crate::bench::systems::System;
use serde_json::{Value, json};

/// Interrupted requests ask for this many tokens; points fall in the first quarter.
const LONG_REPLY: u32 = 256;

fn body(api: ApiKind, text: &str) -> Value {
    match api {
        ApiKind::Responses => json!({"input": text, "temperature": 0}),
        _ => json!({"messages": [{"role": "user", "content": text}], "temperature": 0}),
    }
}

pub async fn run(ctx: &mut Ctx, system: &System, points: u32) -> Cell {
    let Some(api) = system.preferred_api() else {
        return not_run(&system.name, "cancel/disconnect", "chat", "ms", "no API");
    };
    let mut cell = Cell {
        system: system.name.clone(),
        cell: "cancel/disconnect".into(),
        api: api.as_str().into(),
        unit: "ms".into(),
        thermal_start: machine::thermal(),
        ..Cell::default()
    };
    ctx.reset(system, 1).await;
    let long_prompt = "Write a long story about a lighthouse keeper. ".repeat(20);
    let eos = system.ignore_eos();
    for point in 1..=points {
        let after = ctx.rng.below(u64::from(LONG_REPLY / 4) + 1) as usize;
        let long = driver::prepare(
            api,
            &body(api, &long_prompt),
            Some(&system.model),
            LONG_REPLY,
            eos,
        );
        // Returning drops the response: the connection closes.
        if let Err(error) =
            driver::send(&ctx.client, &system.url, api, &long, eos, Some(after)).await
        {
            cell.failures.push(Failure { rep: point, error });
            continue;
        }
        let probe = driver::prepare(api, &body(api, "ok"), Some(&system.model), 1, eos);
        match driver::send(&ctx.client, &system.url, api, &probe, eos, None).await {
            Ok(out) => match out.first_byte_ms {
                Some(ms) => cell.samples.push(ms),
                None => cell.failures.push(Failure {
                    rep: point,
                    error: "empty reply".into(),
                }),
            },
            Err(error) => cell.failures.push(Failure { rep: point, error }),
        }
    }
    cell.thermal_end = machine::thermal();
    if !cell.failures.is_empty() || cell.samples.is_empty() {
        cell.status = CellStatus::Failed;
    }
    cell.summarize();
    cell
}
