//! `mrun bench` against a local fake SSE server (005 T002): scripted delays in all three API
//! styles with pings, SSE comments, keepalives, role-only chunks and `: mrun` metadata. Nothing
//! here leaves 127.0.0.1.

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
    /// Report this many prompt tokens (a truncating server).
    prompt_tokens: Option<u32>,
    context: u32,
}

impl Default for Script {
    fn default() -> Self {
        Self {
            ttft: Duration::from_millis(5),
            early_eos: false,
            tool_call: None,
            cached: 0,
            cause: None,
            prompt_tokens: None,
            context: 9_000,
        }
    }
}

#[derive(Default)]
struct Fake {
    script: Mutex<Script>,
    /// Path, headers and body of each completion request.
    seen: Mutex<Vec<(String, HeaderMap, Value)>>,
    resets: Mutex<u32>,
}

type Shared = Arc<Fake>;

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
fn plan(script: &Script, body: &Value, limit_key: &str) -> (u32, u32, u32, bool) {
    let max = body[limit_key].as_u64().unwrap_or(16) as u32;
    let prompt = script.prompt_tokens.unwrap_or_else(|| prompt_tokens(body));
    let output = if script.early_eos {
        (max / 2).max(1)
    } else {
        max
    };
    (prompt, script.cached.min(prompt), output, output == max)
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
    fake.seen
        .lock()
        .unwrap()
        .push(("chat".into(), headers, body.clone()));
    let script = fake.script.lock().unwrap().clone();
    let (prompt, cached, mut output, limited) = plan(&script, &body, "max_tokens");
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
    fake.seen
        .lock()
        .unwrap()
        .push(("messages".into(), headers, body.clone()));
    let script = fake.script.lock().unwrap().clone();
    let (prompt, cached, output, limited) = plan(&script, &body, "max_tokens");
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
    fake.seen
        .lock()
        .unwrap()
        .push(("responses".into(), headers, body.clone()));
    let script = fake.script.lock().unwrap().clone();
    let (prompt, cached, output, limited) = plan(&script, &body, "max_output_tokens");
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
            get(|| async { axum::Json(json!({"version": "9.9.9"})) }),
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
        cause: Some(("changed", 1)),
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
        assert!(!out.early_eos(4), "{api:?}");
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
    assert_eq!(out.cause.as_deref(), Some("changed"));
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
        tool_gaps: None,
    }
}

fn tool_trace() -> Vec<TraceRecord> {
    let body = |text: &str| {
        json!({"model": "recorded", "messages": [{"role": "user", "content": text}],
            "tools": [{"type": "function", "function": {"name": "read",
                "parameters": {"type": "object", "properties": {"path": {"type": "string"}},
                    "required": ["path"]}}}]})
    };
    vec![
        trace_record("chat", "c1", body("read the file"), 0),
        trace_record("chat", "c1", body("read the file again"), 0),
    ]
}

#[tokio::test]
async fn malformed_tool_calls_and_early_eos_are_counted_next_to_speed() {
    // Arguments that are not JSON.
    let (url, _fake) = spawn(Script {
        tool_call: Some(("read", "{\"path\": ")),
        ..Script::default()
    })
    .await;
    let system = System::parse(&format!("mrun={url}")).unwrap();
    let mut ctx = Ctx::new(false);
    let cell = agent::run(&mut ctx, &system, "tools", &tool_trace(), 1).await;
    assert_eq!(cell.malformed_tool_calls, 2);
    assert_eq!(
        cell.early_eos, 2,
        "a tool call ends the reply before its length"
    );
    assert_eq!(cell.samples.len(), 2);
    assert_eq!(cell.status, CellStatus::Ok);

    // An undeclared tool, and plain text cut short by an end-of-sequence.
    let (url, fake) = spawn(Script {
        tool_call: Some(("write", "{\"path\": \"a\"}")),
        ..Script::default()
    })
    .await;
    let system = System::parse(&format!("mrun={url}")).unwrap();
    let cell = agent::run(&mut ctx, &system, "tools", &tool_trace(), 1).await;
    assert_eq!(cell.malformed_tool_calls, 2);
    *fake.script.lock().unwrap() = Script {
        early_eos: true,
        ..Script::default()
    };
    let cell = agent::run(&mut ctx, &system, "tools", &tool_trace(), 1).await;
    assert_eq!((cell.malformed_tool_calls, cell.early_eos), (0, 2));

    // The report shows both counts in the cell's row, next to its median.
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
    assert!(row.ends_with("| 2 | 0 |"), "{row}");
}

#[tokio::test]
async fn agent_replay_records_turns_resets_and_truncation() {
    let (url, fake) = spawn(Script {
        cached: 4,
        cause: Some(("changed", 0)),
        prompt_tokens: Some(50),
        ..Script::default()
    })
    .await;
    let system = System::parse(&format!("mrun={url}")).unwrap();
    let msg = |t: &str| json!({"messages": [{"role": "user", "content": t}]});
    let records = vec![
        trace_record("chat", "a", msg("x"), 50),
        trace_record("chat", "a", msg("x"), 50),
        trace_record("chat", "b", msg("y"), 100),
        trace_record("chat", "a", msg("x"), 50),
    ];
    let mut ctx = Ctx::new(false);
    let cell = agent::run(&mut ctx, &system, "t", &records, 2).await;
    assert_eq!(*fake.resets.lock().unwrap(), 2, "one reset per repetition");
    assert_eq!(ctx.resets.len(), 2);
    assert!(
        ctx.resets
            .iter()
            .all(|r| r.ok && r.step == "POST /mrun/cache/clear")
    );
    let turns = cell.turns.as_ref().unwrap();
    assert_eq!(turns.len(), 8);
    let kinds: Vec<String> = turns[..4].iter().map(|t| format!("{:?}", t.kind)).collect();
    assert_eq!(kinds, ["New", "Repeated", "New", "Branching"]);
    assert_eq!((turns[0].input, turns[0].cached), (50, 4));
    assert!(cell.session_s.unwrap() > 0.0);
    // Turn 3 expected 100 prompt tokens and got 50: truncated input, once per repetition.
    let truncated = cell
        .failures
        .iter()
        .filter(|f| f.error == "truncated input")
        .count();
    assert_eq!(truncated, 2);
    assert_eq!(cell.status, CellStatus::Failed);
    // Turns 2 and 4 repeat their conversation's message 0 unchanged after a `changed` miss.
    let rerenders = cell.template_rerenders.as_ref().unwrap();
    assert_eq!((rerenders.turns, rerenders.tokens), (2, 92));
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
    let (llama, _) = spawn(Script::default()).await;
    let dir = tempfile::tempdir().unwrap();
    let trace = dir.path().join("refactor.jsonl");
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
        trace.to_str().unwrap(),
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
    // The prefill nonce changes each repetition; ignore-EOS went out as mrun's header.
    let seen = fake.seen.lock().unwrap();
    assert!(seen.iter().all(|(_, h, _)| h["x-mrun-ignore-eos"] == "on"));
    let report = std::fs::read_to_string(out.join("report.md")).unwrap();
    assert!(report.contains("## Losing cells"));
    assert!(report.contains("| agent/refactor | chat | ollama | not run |"));
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
