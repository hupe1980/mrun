//! `mrun bench` against a local fake SSE server (005 T002, D038): scripted delays in all three
//! API styles with pings, SSE comments, keepalives, role-only chunks and `: mrun` metadata, a
//! context window that truncates, failing requests and empty replies. Nothing here leaves
//! 127.0.0.1.

use axum::Router;
use axum::body::{Body, Bytes};
use axum::extract::State;
use axum::http::HeaderMap;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use mrun::api::model::ApiKind;
use mrun::bench::driver::{self, IgnoreEos};
use mrun::bench::result::test_support::{schema, validate};
use mrun::bench::result::{CellStatus, RunResult};
use mrun::bench::suites::{Ctx, agent};
use mrun::bench::systems::System;
use mrun::bench::trace::TraceRecord;
use serde_json::{Value, json};
use std::process::Command;
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// What the fake server does with each request.
#[derive(Clone, Debug)]
struct Script {
    /// Delay before the first content event; pings, comments and role-only chunks come first.
    ttft: Duration,
    /// Stop with an end-of-sequence after half the requested tokens.
    early_eos: bool,
    /// Reply with this tool call (name, raw arguments) instead of text.
    tool_call: Option<(&'static str, &'static str)>,
    cached: u32,
    /// `cause` and message index in the `: mrun` comment.
    cause: Option<(&'static str, u32)>,
    /// Truncate prompts to this many tokens (a server with a smaller window than asked for).
    window: Option<u32>,
    /// Reply with nothing at all, ended by end-of-sequence.
    empty: bool,
    /// Answer the n-th completion request (1-based) with HTTP 500.
    fail_on: Option<u32>,
    context: u32,
    /// Every completion request whose last message is a tool result splices this many tokens
    /// (`/mrun/status` `cache.spliced_tokens_total` grows).
    splice: u64,
}

impl Default for Script {
    fn default() -> Self {
        Self {
            ttft: Duration::from_millis(5),
            early_eos: false,
            tool_call: None,
            cached: 0,
            cause: None,
            window: None,
            empty: false,
            fail_on: None,
            context: 9_000,
            splice: 0,
        }
    }
}

#[derive(Default)]
struct Fake {
    script: Mutex<Script>,
    /// Path, headers and body of each completion request.
    seen: Mutex<Vec<(String, HeaderMap, Value)>>,
    /// Arrival of each completion request.
    arrivals: Mutex<Vec<std::time::Instant>>,
    resets: Mutex<u32>,
    spliced_total: Mutex<u64>,
}

type Shared = Arc<Fake>;

/// Without ignore-EOS (header or body flag) the fake model stops on its own after this many.
const NATURAL_STOP: u32 = 8;

/// Words in the request's text plus a 5-token template overhead.
fn prompt_tokens(body: &Value) -> u32 {
    let mut words = 0;
    let mut count = |v: &Value| {
        if let Some(s) = v.as_str() {
            words += s.split_whitespace().count() as u32;
        }
    };
    count(&body["input"]);
    for m in body["messages"].as_array().into_iter().flatten() {
        count(&m["content"]);
    }
    words + 5
}

fn sse(frames: Vec<(Duration, String)>) -> Response {
    let stream = futures_util::stream::unfold(frames.into_iter(), |mut frames| async move {
        let (delay, text) = frames.next()?;
        if !delay.is_zero() {
            tokio::time::sleep(delay).await;
        }
        Some((Ok::<_, std::convert::Infallible>(Bytes::from(text)), frames))
    });
    let mut response = Body::from_stream(stream).into_response();
    response
        .headers_mut()
        .insert("content-type", "text/event-stream".parse().unwrap());
    response
}

fn data(v: Value) -> String {
    format!("data: {v}\n\n")
}

fn event(name: &str, v: Value) -> String {
    format!("event: {name}\ndata: {v}\n\n")
}

/// The reply's shape: prompt, cached, output tokens and whether the limit ended it.
fn plan(
    script: &Script,
    headers: &HeaderMap,
    body: &Value,
    limit_key: &str,
) -> (u32, u32, u32, bool) {
    let max = body[limit_key].as_u64().unwrap_or(16) as u32;
    let natural = !headers.contains_key("x-mrun-ignore-eos") && body.get("ignore_eos").is_none();
    let full = prompt_tokens(body);
    let prompt = script.window.map_or(full, |w| full.min(w));
    let output = if script.empty {
        0
    } else if script.early_eos {
        (max / 2).max(1)
    } else if natural {
        max.min(NATURAL_STOP)
    } else {
        max
    };
    (prompt, script.cached.min(prompt), output, output == max)
}

fn failure() -> Response {
    (axum::http::StatusCode::INTERNAL_SERVER_ERROR, "boom").into_response()
}

/// Records a completion request; `None` when the script fails it.
fn arrive(fake: &Fake, api: &str, headers: &HeaderMap, body: &Value) -> Option<Script> {
    let mut seen = fake.seen.lock().unwrap();
    seen.push((api.into(), headers.clone(), body.clone()));
    fake.arrivals
        .lock()
        .unwrap()
        .push(std::time::Instant::now());
    let script = fake.script.lock().unwrap().clone();
    let tool_result_last = body["messages"]
        .as_array()
        .and_then(|m| m.last())
        .is_some_and(|m| m["role"] == "tool");
    if tool_result_last {
        *fake.spliced_total.lock().unwrap() += script.splice;
    }
    if script.fail_on == Some(seen.len() as u32) {
        return None;
    }
    Some(script)
}

fn mrun_comment(script: &Script, prompt: u32, cached: u32) -> String {
    let mut meta = json!({"cached": cached, "processed": prompt - cached, "max_tokens": 1});
    if let Some((cause, index)) = script.cause {
        meta["cause"] = cause.into();
        meta["message"] = json!({"index": index, "role": "user"});
    }
    format!(": mrun {meta}\n\n")
}

async fn chat(State(fake): State<Shared>, headers: HeaderMap, body: Bytes) -> Response {
    let body: Value = serde_json::from_slice(&body).unwrap();
    let script = match arrive(&fake, "chat", &headers, &body) {
        Some(script) => script,
        None => return failure(),
    };
    let (prompt, cached, mut output, limited) = plan(&script, &headers, &body, "max_tokens");
    if script.tool_call.is_some() {
        output = 3;
    }
    let zero = Duration::ZERO;
    let mut frames = vec![
        (zero, ": keepalive\n\n".to_string()),
        (zero, mrun_comment(&script, prompt, cached)),
        (
            zero,
            data(json!({"choices": [{"delta": {"role": "assistant", "content": ""}}]})),
        ),
    ];
    let mut first = script.ttft;
    let finish = match script.tool_call {
        Some((name, args)) => {
            frames.push((
                first,
                data(json!({"choices": [{"delta": {"tool_calls": [{"index": 0,
                "id": "c1", "type": "function", "function": {"name": name, "arguments": ""}}]}}]})),
            ));
            frames.push((
                zero,
                data(json!({"choices": [{"delta": {"tool_calls": [{"index": 0,
                "function": {"arguments": args}}]}}]})),
            ));
            "tool_calls"
        }
        None => {
            for _ in 0..output {
                frames.push((
                    first,
                    data(json!({"choices": [{"delta": {"content": "tok "}}]})),
                ));
                first = Duration::from_micros(200);
            }
            if limited { "length" } else { "stop" }
        }
    };
    frames.push((
        zero,
        data(json!({"choices": [{"delta": {}, "finish_reason": finish}]})),
    ));
    frames.push((
        zero,
        data(json!({"choices": [], "usage": {"prompt_tokens": prompt,
        "completion_tokens": output, "prompt_tokens_details": {"cached_tokens": cached}}})),
    ));
    frames.push((zero, "data: [DONE]\n\n".into()));
    sse(frames)
}

async fn messages(State(fake): State<Shared>, headers: HeaderMap, body: Bytes) -> Response {
    let body: Value = serde_json::from_slice(&body).unwrap();
    let script = match arrive(&fake, "messages", &headers, &body) {
        Some(script) => script,
        None => return failure(),
    };
    let (prompt, cached, output, limited) = plan(&script, &headers, &body, "max_tokens");
    let zero = Duration::ZERO;
    let mut frames = vec![
        (zero, event("ping", json!({"type": "ping"}))),
        (
            zero,
            event(
                "message_start",
                json!({"type": "message_start", "message": {"usage":
            {"input_tokens": prompt - cached, "cache_read_input_tokens": cached}}}),
            ),
        ),
        (
            zero,
            event(
                "content_block_start",
                json!({"type": "content_block_start",
            "index": 0, "content_block": {"type": "text", "text": ""}}),
            ),
        ),
        (zero, event("ping", json!({"type": "ping"}))),
    ];
    let mut first = script.ttft;
    for _ in 0..output {
        frames.push((
            first,
            event(
                "content_block_delta",
                json!({"type": "content_block_delta",
            "index": 0, "delta": {"type": "text_delta", "text": "tok "}}),
            ),
        ));
        first = Duration::from_micros(200);
    }
    frames.push((
        zero,
        event(
            "content_block_stop",
            json!({"type": "content_block_stop", "index": 0}),
        ),
    ));
    let reason = if limited { "max_tokens" } else { "end_turn" };
    frames.push((
        zero,
        event(
            "message_delta",
            json!({"type": "message_delta",
        "delta": {"stop_reason": reason}, "usage": {"output_tokens": output}}),
        ),
    ));
    frames.push((zero, event("message_stop", json!({"type": "message_stop"}))));
    sse(frames)
}

async fn responses(State(fake): State<Shared>, headers: HeaderMap, body: Bytes) -> Response {
    let body: Value = serde_json::from_slice(&body).unwrap();
    let script = match arrive(&fake, "responses", &headers, &body) {
        Some(script) => script,
        None => return failure(),
    };
    let (prompt, cached, output, limited) = plan(&script, &headers, &body, "max_output_tokens");
    let zero = Duration::ZERO;
    let mut frames = vec![
        (
            zero,
            event("response.created", json!({"type": "response.created"})),
        ),
        (zero, ": keepalive\n\n".into()),
        (
            zero,
            event(
                "response.reasoning_text.delta",
                json!({"delta": "thinking"}),
            ),
        ),
    ];
    let mut first = script.ttft;
    for _ in 0..output {
        frames.push((
            first,
            event("response.output_text.delta", json!({"delta": "tok "})),
        ));
        first = Duration::from_micros(200);
    }
    let usage = json!({"input_tokens": prompt, "input_tokens_details": {"cached_tokens": cached},
        "output_tokens": output});
    frames.push(if limited {
        (
            zero,
            event(
                "response.incomplete",
                json!({"response": {"status": "incomplete",
            "incomplete_details": {"reason": "max_output_tokens"}, "usage": usage}}),
            ),
        )
    } else {
        (
            zero,
            event(
                "response.completed",
                json!({"response": {"status": "completed", "usage": usage}}),
            ),
        )
    });
    sse(frames)
}

async fn spawn(script: Script) -> (String, Shared) {
    let fake: Shared = Arc::new(Fake {
        script: Mutex::new(script),
        ..Fake::default()
    });
    let app = Router::new()
        .route("/health", get(|| async { "ok" }))
        .route(
            "/v1/models",
            get(|State(f): State<Shared>| async move {
                let context = f.script.lock().unwrap().context;
                axum::Json(json!({"data": [{"id": "fake", "context_window": context}]}))
            }),
        )
        .route(
            "/mrun/status",
            get(|State(f): State<Shared>| async move {
                let total = *f.spliced_total.lock().unwrap();
                axum::Json(
                    json!({"version": "9.9.9", "lanes": {"configured": 1, "busy": 0},
                    "cache": {"spliced_tokens_total": total}}),
                )
            }),
        )
        .route(
            "/mrun/cache/clear",
            post(|State(f): State<Shared>| async move {
                *f.resets.lock().unwrap() += 1;
                axum::Json(json!({}))
            }),
        )
        .route("/slots/0", post(|| async { axum::Json(json!({})) }))
        .route("/v1/chat/completions", post(chat))
        .route("/v1/messages", post(messages))
        .route("/v1/responses", post(responses))
        .with_state(fake.clone());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    (url, fake)
}

fn request(api: ApiKind) -> Value {
    match api {
        ApiKind::Responses => json!({"model": "recorded", "input": "one two three"}),
        _ => {
            json!({"model": "recorded", "messages": [{"role": "user", "content": "one two three"}]})
        }
    }
}

#[tokio::test]
async fn ttft_skips_pings_comments_keepalives_and_role_chunks_in_every_api() {
    let (url, _fake) = spawn(Script {
        ttft: Duration::from_millis(150),
        cached: 2,
        cause: Some(("diverged", 1)),
        ..Script::default()
    })
    .await;
    let client = driver::client();
    for api in [ApiKind::Chat, ApiKind::Messages, ApiKind::Responses] {
        let body = driver::prepare(api, &request(api), Some("m"), 4, IgnoreEos::None);
        let out = driver::send(&client, &url, api, &body, IgnoreEos::None, None)
            .await
            .unwrap();
        let ttft = out.ttft_ms.unwrap();
        assert!(ttft >= 150.0, "{api:?}: {ttft}");
        assert!(
            out.first_byte_ms.unwrap() < 140.0,
            "{api:?}: early frames arrive at once"
        );
        assert_eq!(out.tfut_ms, Some(ttft), "{api:?}");
        assert_eq!(out.token_times_ms.len(), 4, "{api:?}");
        assert_eq!(out.prompt_tokens, Some(8), "{api:?}");
        assert_eq!(out.cached_tokens, Some(2), "{api:?}");
        assert_eq!(out.output_tokens, Some(4), "{api:?}");
        assert!(!out.stopped_short(4), "{api:?}");
        assert!(out.decode_tok_s().is_some(), "{api:?}");
    }
    // The miss report rides on the `: mrun` comment (Chat here).
    let body = driver::prepare(
        ApiKind::Chat,
        &request(ApiKind::Chat),
        None,
        2,
        IgnoreEos::None,
    );
    let out = driver::send(&client, &url, ApiKind::Chat, &body, IgnoreEos::None, None)
        .await
        .unwrap();
    assert_eq!(out.cause.as_deref(), Some("diverged"));
    assert_eq!(out.message_index, Some(1));
}

#[tokio::test]
async fn ignore_eos_and_the_model_name_reach_the_system() {
    let (url, fake) = spawn(Script::default()).await;
    let client = driver::client();
    for (eos, api) in [
        (IgnoreEos::MrunHeader, ApiKind::Chat),
        (IgnoreEos::Body, ApiKind::Chat),
        (IgnoreEos::None, ApiKind::Messages),
    ] {
        let body = driver::prepare(api, &request(api), Some("sys-model"), 3, eos);
        driver::send(&client, &url, api, &body, eos, None)
            .await
            .unwrap();
    }
    let seen = fake.seen.lock().unwrap();
    assert_eq!(seen[0].1["x-mrun-ignore-eos"], "on");
    assert!(seen[0].2.get("ignore_eos").is_none());
    assert_eq!(seen[1].2["ignore_eos"], true);
    assert!(seen[1].1.get("x-mrun-ignore-eos").is_none());
    assert_eq!(seen[2].1["anthropic-version"], "2023-06-01");
    for (_, _, body) in seen.iter() {
        assert_eq!(body["model"], "sys-model");
        assert_eq!(body["stream"], true);
        assert_eq!(body["max_tokens"], 3);
    }
}

fn trace_record(api: &str, conversation: &str, body: Value, prompt_tokens: u32) -> TraceRecord {
    TraceRecord {
        api: api.into(),
        conversation: conversation.into(),
        body,
        prompt_tokens,
        reply_tokens: 6,
        ..TraceRecord::default()
    }
}

fn tool_trace() -> Vec<TraceRecord> {
    let body = |text: &str| {
        json!({"model": "recorded", "messages": [{"role": "user", "content": text}],
            "tools": [{"type": "function", "function": {"name": "read",
                "parameters": {"type": "object", "properties": {"path": {"type": "string"}},
                    "required": ["path"]}}}]})
    };
    // A tool call reply is 3 tokens long: the forced length.
    let mut records = vec![
        trace_record("chat", "c1", body("read the file"), 0),
        trace_record("chat", "c1", body("read the file again"), 0),
    ];
    for record in &mut records {
        record.reply_tokens = 3;
    }
    records
}

#[tokio::test]
async fn quality_signals_come_from_the_natural_stop_pass() {
    // Arguments that are not JSON.
    let (url, fake) = spawn(Script {
        tool_call: Some(("read", "{\"path\": ")),
        ..Script::default()
    })
    .await;
    let system = System::parse(&format!("mrun={url}")).unwrap();
    let mut ctx = Ctx::new(false);
    let cell = agent::run(&mut ctx, &system, "tools", &tool_trace(), 1).await;
    let quality = cell.quality.clone().unwrap();
    assert_eq!((quality.turns, quality.malformed_tool_calls), (2, 2));
    assert_eq!(quality.early_eos, 0, "a tool call is a reply");
    assert!(quality.complete);
    assert_eq!(cell.short_replies, 0, "the tool call is the forced length");
    assert_eq!(cell.samples.len(), 2);
    assert_eq!(cell.status, CellStatus::Ok);
    // The natural-stop pass sent no ignore-EOS and the client's own (absent: default) limit,
    // with the canary at the start of the prompt.
    {
        let seen = fake.seen.lock().unwrap();
        let natural: Vec<_> = seen
            .iter()
            .filter(|(_, _, b)| b["max_tokens"] == driver::NATURAL_MAX_TOKENS)
            .collect();
        assert_eq!(natural.len(), 2);
        for (_, headers, body) in natural {
            assert!(!headers.contains_key("x-mrun-ignore-eos"));
            let text = body["messages"][0]["content"].as_str().unwrap();
            assert!(text.starts_with(&ctx.canary), "{text}");
        }
    }

    // An undeclared tool.
    *fake.script.lock().unwrap() = Script {
        tool_call: Some(("write", "{\"path\": \"a\"}")),
        ..Script::default()
    };
    let cell = agent::run(&mut ctx, &system, "tools", &tool_trace(), 1).await;
    assert_eq!(cell.quality.unwrap().malformed_tool_calls, 2);

    // Text cut short by end-of-sequence in the timed pass (FR-004: more than 10 % short):
    // ended early, the session is excluded; not a quality signal.
    *fake.script.lock().unwrap() = Script {
        early_eos: true,
        ..Script::default()
    };
    let mut long = tool_trace();
    for record in &mut long {
        record.reply_tokens = 6;
    }
    let early = agent::run(&mut ctx, &system, "tools", &long, 1).await;
    let quality = early.quality.clone().unwrap();
    assert_eq!((quality.malformed_tool_calls, quality.early_eos), (0, 0));
    assert_eq!(early.status, CellStatus::Failed);
    assert_eq!(
        early.incomplete[0].error, "turn 1 ended early (3 of 6 tokens)",
        "{:?}",
        early.incomplete
    );
    // At most 10 % short: counted, the session stays (20 of 21 tokens).
    let mut near = tool_trace();
    for record in &mut near {
        record.reply_tokens = 21;
    }
    *fake.script.lock().unwrap() = Script {
        early_eos: false,
        ..Script::default()
    };
    let mut cell = agent::run(&mut ctx, &system, "tools", &near, 1).await;
    assert_eq!(cell.status, CellStatus::Ok, "{:?}", cell.incomplete);
    assert_eq!(cell.short_replies, 0);
    // The fake answers the forced length exactly; mark both replies short by hand for the
    // report row below.
    cell.short_replies = 2;

    // The report shows the timing caveat and the truncation check in the cell's row.
    let mut result = RunResult {
        cells: vec![cell],
        ..RunResult::default()
    };
    result.cells[0].cell = "agent/tools".into();
    let report = mrun::bench::report::render(&result);
    let row = report
        .lines()
        .find(|l| l.starts_with("| agent/tools |"))
        .unwrap();
    assert!(row.ends_with("| 2 | canary |"), "{row}");
    assert!(
        report
            .contains("Quality (natural-stop pass, 2 turns): malformed tool calls 0, early EOS 0")
    );

    // Empty replies: every timed session fails (no content), the natural pass counts early EOS.
    *fake.script.lock().unwrap() = Script {
        empty: true,
        ..Script::default()
    };
    let cell = agent::run(&mut ctx, &system, "tools", &tool_trace(), 2).await;
    assert_eq!(cell.quality.unwrap().early_eos, 2);
    assert_eq!(cell.status, CellStatus::Failed);
    assert_eq!(cell.incomplete.len(), 2);
    assert!(cell.samples.is_empty());
}

#[tokio::test]
async fn a_session_with_a_failed_turn_is_excluded_as_a_whole() {
    // Request 2 is turn 2 of session 1.
    let (url, _fake) = spawn(Script {
        fail_on: Some(2),
        ..Script::default()
    })
    .await;
    let system = System::parse(&format!("mrun={url}")).unwrap();
    let cell = agent::run(&mut Ctx::new(false), &system, "t", &tool_trace(), 3).await;
    assert_eq!(cell.status, CellStatus::Ok);
    assert_eq!(cell.complete_sessions, Some(2));
    assert_eq!(cell.incomplete.len(), 1);
    assert_eq!(cell.incomplete[0].rep, 1);
    assert!(
        cell.incomplete[0].error.starts_with("turn 2: HTTP 500"),
        "{:?}",
        cell.incomplete
    );
    // Turn 1 of the failed session is not a sample either.
    assert_eq!(cell.samples.len(), 4);
    let turns = cell.turns.as_ref().unwrap();
    assert!(turns.iter().all(|t| t.rep != Some(1)));
}

#[tokio::test]
async fn agent_replay_records_turns_resets_and_truncation() {
    let (url, fake) = spawn(Script {
        cached: 4,
        cause: Some(("diverged", 0)),
        ..Script::default()
    })
    .await;
    let system = System::parse(&format!("mrun={url}")).unwrap();
    let msg = |t: &str| json!({"messages": [{"role": "user", "content": t}]});
    let long = vec!["word"; 60].join(" ");
    let records = vec![
        trace_record("chat", "a", msg("x"), 0),
        trace_record("chat", "a", msg("x"), 0),
        trace_record("chat", "b", msg(&long), 0),
        trace_record("chat", "a", msg("x"), 0),
    ];
    let mut ctx = Ctx::new(false);
    let cell = agent::run(&mut ctx, &system, "t", &records, 2).await;
    assert_eq!(
        *fake.resets.lock().unwrap(),
        3,
        "one reset per repetition and one before the natural-stop pass"
    );
    assert_eq!(ctx.resets.len(), 3);
    assert_eq!(ctx.resets[2].rep, 0, "rep 0 is the natural-stop pass");
    assert!(
        ctx.resets
            .iter()
            .all(|r| r.ok && r.step == "POST /mrun/cache/clear")
    );
    assert_eq!(cell.status, CellStatus::Ok);
    assert_eq!(cell.truncation_check.as_deref(), Some("canary"));
    let turns = cell.turns.as_ref().unwrap();
    assert_eq!(turns.len(), 8);
    let kinds: Vec<String> = turns[..4].iter().map(|t| format!("{:?}", t.kind)).collect();
    assert_eq!(kinds, ["New", "Repeated", "New", "Branching"]);
    assert_eq!((turns[0].input, turns[0].cached), (6, 4));
    assert!(cell.session_s.unwrap() > 0.0);

    // A 50-token window: turn 3 (65 tokens) is cut to 50 with and without the canary, so the
    // canary vanishes; every session contains it and is excluded.
    fake.script.lock().unwrap().window = Some(50);
    let cell = agent::run(&mut ctx, &system, "t", &records, 2).await;
    assert_eq!(cell.status, CellStatus::Failed);
    assert_eq!(cell.complete_sessions, Some(0));
    assert_eq!(cell.incomplete.len(), 2);
    assert!(
        cell.incomplete
            .iter()
            .all(|f| f.error == "turn 3 truncated (the canary added 0 of its 9 tokens)"),
        "{:?}",
        cell.incomplete
    );
    assert!(cell.samples.is_empty());
}

#[tokio::test]
async fn a_system_without_the_trace_api_is_not_applicable() {
    let (url, _fake) = spawn(Script::default()).await;
    let system = System::parse(&format!("x={url},apis=chat")).unwrap();
    let records = vec![trace_record(
        "responses",
        "a",
        request(ApiKind::Responses),
        0,
    )];
    let cell = agent::run(&mut Ctx::new(false), &system, "codex", &records, 1).await;
    assert_eq!(cell.status, CellStatus::NotApplicable);
}

/// Runs the binary off the async runtime, with every proxy variable pointing at a closed port:
/// bench must reach the systems directly and nothing else (FR-011).
async fn mrun(args: Vec<String>) -> std::process::Output {
    tokio::task::spawn_blocking(move || {
        Command::new(env!("CARGO_BIN_EXE_mrun"))
            .args(&args)
            .env("HTTP_PROXY", "http://127.0.0.1:9")
            .env("HTTPS_PROXY", "http://127.0.0.1:9")
            .env("ALL_PROXY", "http://127.0.0.1:9")
            .env("http_proxy", "http://127.0.0.1:9")
            .env_remove("MRUN_PORT")
            .stdin(std::process::Stdio::null())
            .output()
            .unwrap()
    })
    .await
    .unwrap()
}

fn s(parts: &[&str]) -> Vec<String> {
    parts.iter().map(|p| p.to_string()).collect()
}

#[tokio::test]
async fn bench_writes_a_result_that_validates_against_the_schema() {
    let (url, fake) = spawn(Script::default()).await;
    let (llama, _) = spawn(Script {
        ttft: Duration::from_millis(40),
        ..Script::default()
    })
    .await;
    let dir = tempfile::tempdir().unwrap();
    let traces = dir.path().join("traces");
    std::fs::create_dir(&traces).unwrap();
    let trace = traces.join("refactor.jsonl");
    let lines: Vec<String> = tool_trace()
        .iter()
        .map(|r| serde_json::to_string(r).unwrap())
        .collect();
    std::fs::write(&trace, lines.join("\n")).unwrap();
    let out = dir.path().join("out");
    let output = mrun(s(&[
        "bench",
        "--compare",
        "--system",
        &format!("mrun={url}"),
        "--system",
        &format!("llama.cpp={llama}"),
        "--system",
        "ollama=http://127.0.0.1:1",
        "--trace",
        traces.to_str().unwrap(),
        "--reps",
        "1",
        "--out",
        out.to_str().unwrap(),
    ]))
    .await;
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let text = std::fs::read_to_string(out.join("result.json")).unwrap();
    let value: Value = serde_json::from_str(&text).unwrap();
    validate(&schema(), &value, "$").unwrap();
    let result: RunResult = serde_json::from_value(value).unwrap();
    let cell = |system: &str, name: &str| {
        result
            .cells
            .iter()
            .find(|c| c.system == system && c.cell == name)
            .unwrap_or_else(|| panic!("{system} {name}"))
    };
    // Context 9000 (mrun's): 1K, 8K and the maximum fit, 32K does not.
    assert_eq!(cell("mrun", "synthetic/prefill-1k").status, CellStatus::Ok);
    assert_eq!(cell("mrun", "synthetic/prefill-1k").samples.len(), 5);
    assert_eq!(cell("mrun", "synthetic/decode-max").status, CellStatus::Ok);
    assert_eq!(
        cell("mrun", "synthetic/prefill-32k").status,
        CellStatus::NotRun
    );
    assert_eq!(cell("mrun", "agent/refactor").samples.len(), 6);
    assert_eq!(cell("mrun", "cancel/disconnect").samples.len(), 50);
    assert_eq!(cell("llama.cpp", "agent/refactor").status, CellStatus::Ok);
    assert!(
        result
            .cells
            .iter()
            .all(|c| c.system != "llama.cpp" || c.cell != "cancel/disconnect")
    );
    assert_eq!(cell("ollama", "agent/refactor").status, CellStatus::NotRun);
    assert_eq!(
        cell("ollama", "synthetic/prefill-1k").status,
        CellStatus::NotRun
    );
    let llama_sys = result
        .systems
        .iter()
        .find(|s| s.name == "llama.cpp")
        .unwrap();
    assert_eq!(llama_sys.context_length, 9_000);
    assert_eq!(llama_sys.flags[0], "-c 9000");
    assert_eq!(result.systems[0].version, "9.9.9");
    assert!(
        result
            .comparability
            .iter()
            .any(|c| c.system == "llama.cpp" && c.kind == "different-quantization")
    );
    assert!(
        result
            .resets
            .iter()
            .any(|r| r.system == "llama.cpp" && r.ok)
    );
    // Agent: mrun (5 ms) against llama.cpp (40 ms), paired over both turns.
    let agent = result
        .comparisons
        .iter()
        .find(|c| c.cell == "agent/refactor" && c.system == "llama.cpp")
        .unwrap();
    assert_eq!((agent.method.as_str(), agent.n), ("paired-turns", 2));
    assert!(
        agent.ci_low > 1.0 && agent.speedup >= agent.ci_low,
        "{agent:?}"
    );
    assert_eq!(agent.verdict, "mrun-faster");
    // The bundled llama.cpp profile is pinned to its verified release (T022): a headline cell.
    assert!(agent.headline);
    assert!(
        result
            .comparisons
            .iter()
            .any(|c| c.cell == "synthetic/prefill-1k" && c.method == "median-ratio")
    );
    assert_eq!(result.method.resamples, 2_000);
    let agent_cell = cell("mrun", "agent/refactor");
    assert_eq!(agent_cell.complete_sessions, Some(3));
    assert_eq!(agent_cell.quality.as_ref().unwrap().turns, 2);
    assert_eq!(agent_cell.truncation_check.as_deref(), Some("canary"));
    // Timed requests carry mrun's ignore-EOS header; the natural-stop pass (the client's own
    // limit) never does.
    let seen = fake.seen.lock().unwrap();
    let (natural, timed): (Vec<_>, Vec<_>) = seen
        .iter()
        .partition(|(_, _, b)| b["max_tokens"] == driver::NATURAL_MAX_TOKENS);
    assert_eq!(natural.len(), 2);
    assert!(
        natural
            .iter()
            .all(|(_, h, _)| !h.contains_key("x-mrun-ignore-eos"))
    );
    assert!(timed.iter().all(|(_, h, _)| h["x-mrun-ignore-eos"] == "on"));
    let report = std::fs::read_to_string(out.join("report.md")).unwrap();
    assert!(report.contains("## Losing cells"));
    assert!(report.contains("| agent/refactor | all turns | llama.cpp | 2 |"));
    assert!(report.contains("| agent/refactor | chat | ollama | no | none | not run |"));
    // v3: mrun's lanes and numerics from /mrun/status (no numerics reported: unknown), the
    // session comparison and the sequential class.
    assert_eq!(result.schema_version, 3);
    assert_eq!(result.systems[0].lanes, 1);
    assert_eq!(result.systems[0].numerics.nax, "unknown");
    assert!(
        result
            .comparisons
            .iter()
            .any(|c| c.cell == "agent/refactor" && c.scope.as_deref() == Some("session"))
    );
    assert_eq!(agent_cell.class.as_deref(), Some("sequential"));
    assert_eq!(agent_cell.session_times.len(), 3);
}

/// T032 (FR-020, SC-006): auxiliary requests replay at their recorded offsets beside the main
/// turn, and both TTFTs are reported.
#[tokio::test]
async fn the_concurrent_class_holds_offsets_and_reports_both_ttfts() {
    let (url, fake) = spawn(Script {
        ttft: Duration::from_millis(400),
        ..Script::default()
    })
    .await;
    let system = System::parse(&format!("mrun={url}")).unwrap();
    let msg = |t: &str| json!({"messages": [{"role": "user", "content": t}]});
    let mut aux = trace_record("chat", "title", msg("title please"), 0);
    aux.class = mrun::bench::result::TurnClass::Auxiliary;
    aux.offset_ms = Some(150.0);
    let records = vec![trace_record("chat", "a", msg("fix the bug"), 0), aux];
    let cell = agent::run(&mut Ctx::new(false), &system, "c", &records, 2).await;
    assert_eq!(cell.status, CellStatus::Ok, "{:?}", cell.incomplete);
    assert_eq!(cell.class.as_deref(), Some("concurrent"));
    assert!(cell.main_ttft.unwrap() >= 400.0);
    assert!(cell.aux_ttft.unwrap() >= 400.0);
    // In each timed repetition the auxiliary request arrives 150 ms after the main one
    // (within 50 ms), while the main one is still in its 400 ms prefill.
    let arrivals = fake.arrivals.lock().unwrap().clone();
    let seen = fake.seen.lock().unwrap();
    // Timed requests carry the forced length (6); calibration and the natural pass do not.
    let timed: Vec<usize> = (0..seen.len())
        .filter(|&i| seen[i].2["max_tokens"] == 6)
        .collect();
    assert_eq!(timed.len(), 4, "two repetitions of two requests");
    for pair in timed.chunks(2) {
        let gap = arrivals[pair[1]].duration_since(arrivals[pair[0]]);
        assert!(
            gap >= Duration::from_millis(150) && gap < Duration::from_millis(200),
            "{gap:?}"
        );
    }
    let turns = cell.turns.unwrap();
    assert_eq!(turns[1].offset_ms, Some(150.0));
}

/// T034 (FR-022): the natural-stop pass measures mrun's splice hit rate per client from
/// `/mrun/status` `cache.spliced_tokens_total`, over requests whose turn holds a reply.
#[tokio::test]
async fn the_splice_hit_rate_is_counted_per_client() {
    let (url, _fake) = spawn(Script {
        splice: 7,
        ..Script::default()
    })
    .await;
    let system = System::parse(&format!("mrun={url}")).unwrap();
    let user = json!({"role": "user", "content": "fix it"});
    let call = json!({"role": "assistant", "content": "", "tool_calls": [{"id": "c",
        "type": "function", "function": {"name": "read", "arguments": "{}"}}]});
    let result = json!({"role": "tool", "tool_call_id": "c", "content": "ok"});
    let mut first = trace_record("chat", "a", json!({"messages": [user]}), 0);
    first.client = "claude-code".into();
    let mut loop_turn = trace_record("chat", "a", json!({"messages": [user, call, result]}), 0);
    loop_turn.client = "claude-code".into();
    let mut other = trace_record("chat", "b", json!({"messages": [user, call, result]}), 0);
    other.client = "opencode".into();
    // The first request starts a turn (not eligible); the other two continue a tool loop and
    // the fake splices both.
    let cell = agent::run(
        &mut Ctx::new(false),
        &system,
        "s",
        &[first, loop_turn, other],
        1,
    )
    .await;
    let rates = cell.splice_hit_rate.unwrap();
    assert_eq!(rates.len(), 2, "{rates:?}");
    let claude = rates.iter().find(|r| r.client == "claude-code").unwrap();
    assert_eq!(
        (claude.eligible, claude.spliced, claude.rate),
        (1, 1, Some(1.0))
    );
    let opencode = rates.iter().find(|r| r.client == "opencode").unwrap();
    assert_eq!(opencode.eligible, 1);
    // A competitor gets no splice column.
    let (plain, _) = spawn(Script::default()).await;
    let llama = System::parse(&format!("llama.cpp={plain}")).unwrap();
    let records = vec![trace_record("chat", "a", json!({"messages": [user]}), 0)];
    let cell = agent::run(&mut Ctx::new(false), &llama, "s", &records, 1).await;
    assert!(cell.splice_hit_rate.is_none());
}

#[tokio::test]
async fn unreachable_systems_exit_5_and_several_need_compare() {
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("out");
    let output = mrun(s(&[
        "bench",
        "--suite",
        "synthetic",
        "--system",
        "mrun=http://127.0.0.1:1",
        "--out",
        out.to_str().unwrap(),
    ]))
    .await;
    assert_eq!(output.status.code(), Some(5));
    let result: RunResult =
        serde_json::from_str(&std::fs::read_to_string(out.join("result.json")).unwrap()).unwrap();
    assert!(result.cells.iter().all(|c| c.status == CellStatus::NotRun));

    let output = mrun(s(&[
        "bench",
        "--system",
        "a=http://127.0.0.1:1",
        "--system",
        "b=http://127.0.0.1:2",
    ]))
    .await;
    assert_eq!(output.status.code(), Some(2));

    // No bundled traces: the agent suite needs --trace; a missing path is named.
    let output = mrun(s(&[
        "bench",
        "--suite",
        "agent",
        "--system",
        "mrun=http://127.0.0.1:1",
    ]))
    .await;
    assert_eq!(output.status.code(), Some(2));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("needs --trace <file|dir>"), "{stderr}");
    let output = mrun(s(&[
        "bench",
        "--suite",
        "agent",
        "--trace",
        "/nonexistent/x.jsonl",
    ]))
    .await;
    assert_eq!(output.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&output.stderr).contains("neither a file nor a directory"));

    // --suite all without a trace runs the rest and says the agent suite was skipped.
    let output = mrun(s(&[
        "bench",
        "--system",
        "mrun=http://127.0.0.1:1",
        "--out",
        out.to_str().unwrap(),
    ]))
    .await;
    assert_eq!(output.status.code(), Some(5));
    let result: RunResult =
        serde_json::from_str(&std::fs::read_to_string(out.join("result.json")).unwrap()).unwrap();
    assert!(
        result
            .warnings
            .iter()
            .any(|w| w.starts_with("agent suite skipped"))
    );
    assert!(result.cells.iter().all(|c| !c.cell.starts_with("agent/")));
}

#[tokio::test]
async fn bench_trace_builds_a_trace_from_a_record() {
    let dir = tempfile::tempdir().unwrap();
    let record = dir.path().join("mrun-1.jsonl");
    let line = |at: &str, messages: Value| {
        json!({"api": "messages", "arrived": at, "session": {},
            "body": {"system": "agent", "messages": messages}})
        .to_string()
    };
    let u = json!({"role": "user", "content": "look at /Users/alice/x.rs"});
    let a = json!({"role": "assistant", "content": "done"});
    let u2 = json!({"role": "user", "content": "thanks"});
    std::fs::write(
        &record,
        [
            line("2026-10-15T08:00:00.000Z", json!([u])),
            line("2026-10-15T08:00:05.000Z", json!([u, a, u2])),
        ]
        .join("\n"),
    )
    .unwrap();
    let out = dir.path().join("trace.jsonl");
    let output = mrun(s(&[
        "bench",
        "trace",
        record.to_str().unwrap(),
        "-o",
        out.to_str().unwrap(),
    ]))
    .await;
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let trace = mrun::bench::trace::load(&out).unwrap();
    assert_eq!(trace.len(), 2);
    assert_eq!(trace[0].conversation, trace[1].conversation);
    assert_eq!(trace[0].reply_tokens, 1);
    assert!(trace[0].body.to_string().contains("/Users/user1/x.rs"));
}
