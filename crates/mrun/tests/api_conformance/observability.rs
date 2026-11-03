//! 012 (D056): live per-request status, `/metrics`, `/v1/models` while loading, and the
//! SIGTERM drain (D052).

use crate::common::{self, Setup};
use mrun::api::run::Timing;
use mrun::cli::serve::{self, EngineFactory};
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::time::{Duration, Instant};

fn slow(step_ms: u64) -> Setup {
    Setup {
        timing: Timing {
            keepalive: Duration::from_secs(10),
            queue_wait: Duration::from_secs(60),
        },
        step_delay: Some(Duration::from_millis(step_ms)),
        ..Setup::default()
    }
}

fn messages(text: &str) -> Value {
    json!({"max_tokens": 100, "messages": [{"role": "user", "content": text}]})
}

async fn status(server: &common::Server) -> Value {
    let response = server.client.get(server.url("/mrun/status")).send().await;
    response.unwrap().json().await.unwrap()
}

async fn metrics(server: &common::Server) -> (String, BTreeMap<String, f64>) {
    let response = server
        .client
        .get(server.url("/metrics"))
        .send()
        .await
        .unwrap();
    let kind = response.headers()["content-type"]
        .to_str()
        .unwrap()
        .to_string();
    let text = response.text().await.unwrap();
    assert_eq!(kind, "text/plain; version=0.0.4; charset=utf-8");
    let samples = text
        .lines()
        .filter(|l| !l.starts_with('#'))
        .map(|l| {
            let (name, value) = l.rsplit_once(' ').unwrap();
            (name.to_string(), value.parse().unwrap())
        })
        .collect();
    (text, samples)
}

/// Every line is a comment or a sample of the contract's grammar; every `# TYPE` precedes its
/// samples.
fn check_format(text: &str) {
    assert!(text.ends_with('\n'));
    let mut typed = Vec::new();
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("# TYPE ") {
            typed.push(rest.split(' ').next().unwrap().to_string());
        } else if !line.starts_with("# HELP ") {
            let name = line.split(['{', ' ']).next().unwrap();
            assert!(
                name.bytes().all(|b| b.is_ascii_lowercase() || b == b'_'),
                "{line}"
            );
            let family = ["_bucket", "_sum", "_count"]
                .iter()
                .find_map(|s| name.strip_suffix(s))
                .filter(|base| typed.iter().any(|t| t == base))
                .unwrap_or(name);
            assert!(typed.iter().any(|t| t == family), "untyped sample: {line}");
        }
    }
}

#[tokio::test]
async fn status_lists_queued_and_running_requests() {
    let server = common::start_with(&["busy busy busy", "second"], slow(50)).await;
    let spawn = |text: &'static str| {
        let client = server.client.clone();
        let url = server.url("/v1/messages");
        tokio::spawn(async move { client.post(url).json(&messages(text)).send().await })
    };
    let first = spawn("a");
    tokio::time::sleep(Duration::from_millis(150)).await;
    let second = spawn("b");
    tokio::time::sleep(Duration::from_millis(100)).await;
    let started = Instant::now();
    let now = status(&server).await;
    assert!(started.elapsed() < Duration::from_millis(100), "FR-012");
    let requests = now["requests"].as_array().unwrap();
    assert_eq!(requests.len(), 2, "{now}");
    assert_eq!(requests[0]["phase"], "generating");
    assert_eq!(requests[0]["api"], "messages");
    assert_eq!(requests[0]["lane"], 0);
    assert!(requests[0]["generated"].as_u64().unwrap() >= 1);
    assert_eq!(requests[1]["phase"], "queued");
    assert!(requests[1]["cached"].is_null());
    assert!(requests[1]["speculation"].is_null());
    assert!(requests[0]["id"].as_u64() < requests[1]["id"].as_u64());
    assert_eq!(now["active"]["id"], requests[0]["id"]);
    assert_eq!(now["lanes"], json!({"configured": 1, "busy": 1}));
    first.await.unwrap().unwrap();
    second.await.unwrap().unwrap();
    assert_eq!(status(&server).await["requests"], json!([]));
}

#[tokio::test]
async fn metrics_deltas_equal_reported_usage() {
    let server = common::start(&["one", "two two", "three", "four"]).await;
    let (before_text, before) = metrics(&server).await;
    check_format(&before_text);
    let mut prompt = 0.0;
    let mut cached = 0.0;
    let mut generated = 0.0;
    let mut misses = 0.0;
    let mut miss = |reply: &common::Reply| {
        if reply.header("x-mrun-cache-cause").is_some() {
            misses += 1.0;
        }
        reply.json()
    };
    let reply = miss(&server.post("/v1/messages", &messages("hello")).await);
    let usage = &reply["usage"];
    prompt += usage["input_tokens"].as_f64().unwrap()
        + usage["cache_read_input_tokens"].as_f64().unwrap();
    cached += usage["cache_read_input_tokens"].as_f64().unwrap();
    generated += usage["output_tokens"].as_f64().unwrap();
    let chat = json!({"messages": [{"role": "user", "content": "hello"}]});
    let reply = miss(&server.post("/v1/chat/completions", &chat).await);
    prompt += reply["usage"]["prompt_tokens"].as_f64().unwrap();
    cached += reply["usage"]["prompt_tokens_details"]["cached_tokens"]
        .as_f64()
        .unwrap();
    generated += reply["usage"]["completion_tokens"].as_f64().unwrap();
    let reply = miss(
        &server
            .post("/v1/responses", &json!({"input": "hello"}))
            .await,
    );
    prompt += reply["usage"]["input_tokens"].as_f64().unwrap();
    cached += reply["usage"]["input_tokens_details"]["cached_tokens"]
        .as_f64()
        .unwrap();
    generated += reply["usage"]["output_tokens"].as_f64().unwrap();
    // A rejected request (invalid body).
    assert_eq!(
        server
            .post("/v1/messages", &json!({"messages": []}))
            .await
            .status,
        400
    );
    let (text, after) = metrics(&server).await;
    check_format(&text);
    let delta = |name: &str| after[name] - before.get(name).copied().unwrap_or(0.0);
    let sum = |prefix: &str| {
        after
            .iter()
            .filter(|(k, _)| k.starts_with(prefix))
            .map(|(k, v)| v - before.get(k).copied().unwrap_or(0.0))
            .sum::<f64>()
    };
    assert_eq!(sum("mrun_prompt_tokens_total"), prompt);
    assert_eq!(sum("mrun_cached_tokens_total"), cached);
    assert_eq!(sum("mrun_generated_tokens_total"), generated);
    assert_eq!(sum("mrun_requests_total"), 4.0);
    assert_eq!(
        delta("mrun_requests_total{api=\"messages\",outcome=\"rejected\"}"),
        1.0
    );
    assert_eq!(
        delta("mrun_requests_total{api=\"responses\",outcome=\"completed\"}"),
        1.0
    );
    assert_eq!(sum("mrun_cache_misses_total"), misses, "{text}");
    assert_eq!(after["mrun_model_loaded"], 1.0);
    assert_eq!(delta("mrun_time_to_first_token_seconds_count"), 3.0);
    assert!(text.contains("mrun_cache_misses_total{cause=\"gap\"}"));
    assert!(!text.contains("hello"), "no content");
}

#[tokio::test]
async fn models_and_metrics_answer_while_loading() {
    let (release, wait) = std::sync::mpsc::channel::<()>();
    let factory: EngineFactory = Box::new(move |_| {
        let _ = wait.recv();
        Err(mrun::errors::CliError::new(
            mrun::errors::Code::Internal,
            "never loads",
        ))
    });
    let mut args = common::args(16);
    args.model = Some("qwen3.6".into());
    let server = serve::start(args, factory).await.unwrap();
    let client = reqwest::Client::new();
    let base = format!("http://{}", server.addr);
    let started = Instant::now();
    let response = client
        .get(format!("{base}/v1/models"))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status().as_u16(), 200);
    let models: Value = response.json().await.unwrap();
    assert!(started.elapsed() < Duration::from_secs(3));
    assert_eq!(models["data"][0]["id"], "qwen3.6");
    assert_eq!(models["data"][0]["display_name"], "qwen3.6");
    assert_eq!(models["data"][0]["type"], "model");
    assert_eq!(models["has_more"], false);
    let text = client
        .get(format!("{base}/metrics"))
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();
    assert!(text.contains("mrun_model_loaded 0\n"), "{text}");
    assert!(text.contains("mrun_requests_total{"));
    assert!(!text.contains("mrun_memory_bytes"), "{text}");
    release.send(()).unwrap();
    server.shutdown().await.unwrap();
}

#[tokio::test]
async fn shutdown_cancels_queued_and_lets_running_finish() {
    let server = common::start_with(&["busy busy", "never"], slow(50)).await;
    let spawn = |text: &'static str| {
        let client = server.client.clone();
        let url = server.url("/v1/messages");
        tokio::spawn(async move {
            let response = client.post(url).json(&messages(text)).send().await.unwrap();
            (response.status().as_u16(), response.text().await.unwrap())
        })
    };
    let running = spawn("a");
    tokio::time::sleep(Duration::from_millis(100)).await;
    let queued = spawn("b");
    tokio::time::sleep(Duration::from_millis(100)).await;
    let started = Instant::now();
    let common::Server { inner, .. } = server;
    let shutdown = tokio::spawn(inner.shutdown());
    let (status, body) = queued.await.unwrap();
    assert_eq!(status, 529, "{body}");
    assert!(body.contains("shutting down"), "{body}");
    assert!(started.elapsed() < Duration::from_millis(300));
    let (status, body) = running.await.unwrap();
    assert_eq!(status, 200, "{body}");
    shutdown.await.unwrap().unwrap();
    assert!(started.elapsed() < Duration::from_secs(5));
}

/// What the responses of a session reported (002 FR-022 metadata and usage).
#[derive(Debug, Default)]
struct Reported {
    requests: f64,
    prompt: f64,
    cached: f64,
    generated: f64,
    spliced: f64,
    unreused: f64,
    causes: BTreeMap<String, f64>,
}

impl Reported {
    /// Adds a non-streaming response of `api`.
    fn add(&mut self, api: &str, reply: &common::Reply) -> Value {
        assert_eq!(reply.status, 200, "{}", reply.text);
        let header = |name: &str| reply.header(name).map_or(0.0, |v| v.parse().unwrap());
        self.requests += 1.0;
        self.spliced += header("x-mrun-spliced");
        self.unreused += header("x-mrun-cache-unreused");
        if let Some(cause) = reply.header("x-mrun-cache-cause") {
            *self.causes.entry(cause.to_string()).or_default() += 1.0;
        }
        let body = reply.json();
        let usage = &body["usage"];
        let n = |v: &Value| v.as_f64().unwrap();
        let (prompt, cached, output) = match api {
            "messages" => (
                n(&usage["input_tokens"]) + n(&usage["cache_read_input_tokens"]),
                n(&usage["cache_read_input_tokens"]),
                n(&usage["output_tokens"]),
            ),
            "chat" => (
                n(&usage["prompt_tokens"]),
                n(&usage["prompt_tokens_details"]["cached_tokens"]),
                n(&usage["completion_tokens"]),
            ),
            _ => (
                n(&usage["input_tokens"]),
                n(&usage["input_tokens_details"]["cached_tokens"]),
                n(&usage["output_tokens"]),
            ),
        };
        self.prompt += prompt;
        self.cached += cached;
        self.generated += output;
        body
    }
}

/// The deterministic 30-turn Claude-Code-shaped Messages session of 012 research §8: a
/// billing-header system prompt, 21 tools, a growing `tool_use`/`tool_result` loop, one
/// auxiliary title request and one edited earlier message. Returns the scripted replies in
/// request order, and drives the session against `server` once it runs.
const SESSION_TURNS: usize = 30;

fn session_replies() -> Vec<String> {
    let mut replies: Vec<String> = (0..SESSION_TURNS)
        .map(|i| {
            format!(
                "<tool_call>\n{{\"name\": \"Read\", \"arguments\": {{\"file_path\": \"src/f{i}.rs\"}}}}\n</tool_call>"
            )
        })
        .collect();
    // The title request comes after turn 10.
    replies.insert(11, "Session title".into());
    replies
}

async fn run_session(server: &common::Server, reported: &mut Reported) {
    let system = json!([
        {"type": "text", "text": "x-anthropic-billing-header: cc_version=2.1.3; cch=4f2a;"},
        {"type": "text", "text": "You are Claude Code, a coding agent.", "cache_control": {"type": "ephemeral"}}
    ]);
    let mut tools: Vec<Value> = (0..20)
        .map(|i| {
            json!({"name": format!("Tool{i}"), "description": format!("Tool number {i}"),
            "input_schema": {"type": "object", "properties": {"x": {"type": "string"}}}})
        })
        .collect();
    tools.push(json!({"name": "Read", "description": "Read a file",
        "input_schema": {"type": "object", "properties": {"file_path": {"type": "string"}}}}));
    let mut messages = vec![json!({"role": "user", "content": "Refactor the parser."})];
    for turn in 0..SESSION_TURNS {
        if turn == 11 {
            let title = json!({"max_tokens": 50, "system": "Write a 5-word title.",
                "messages": [{"role": "user", "content": "Refactor the parser."}]});
            reported.add("messages", &server.post("/v1/messages", &title).await);
        }
        if turn == 20 {
            // The client edits an earlier message (a compaction-like rewrite of the first tool
            // result): `diverged`. The replies after it keep splicing from their own records.
            messages[2]["content"][0]["content"] = json!("fn f0() { /* elided */ }");
        }
        let body = json!({"model": "claude-sonnet-4-5", "max_tokens": 1000, "system": system,
            "tools": tools, "messages": messages});
        let reply = reported.add("messages", &server.post("/v1/messages", &body).await);
        let content = reply["content"].clone();
        let tool = content
            .as_array()
            .unwrap()
            .iter()
            .find(|b| b["type"] == "tool_use")
            .unwrap_or_else(|| panic!("turn {turn}: {reply}"))
            .clone();
        messages.push(json!({"role": "assistant", "content": content}));
        messages.push(json!({"role": "user", "content": [{"type": "tool_result",
            "tool_use_id": tool["id"], "content": format!("fn f{turn}() {{}}")}]}));
    }
}

#[tokio::test]
async fn metrics_deltas_equal_reported_usage_session() {
    // SC-003: 10 requests across the three APIs, the 30-turn session, one rejected and one
    // cancelled request; every counter delta equals what the responses reported.
    let mut replies: Vec<String> = (0..10).map(|i| format!("answer {i}")).collect();
    replies.extend(session_replies());
    replies.push("z".repeat(3_000));
    let replies: Vec<&str> = replies.iter().map(String::as_str).collect();
    let server = common::start_with(&replies, slow(1)).await;
    let (before_text, before) = metrics(&server).await;
    check_format(&before_text);
    let mut reported = Reported::default();
    for i in 0..10 {
        let text = format!("question {i}");
        match i % 3 {
            0 => reported.add(
                "messages",
                &server.post("/v1/messages", &messages(&text)).await,
            ),
            1 => {
                let body = json!({"messages": [{"role": "user", "content": text}]});
                reported.add("chat", &server.post("/v1/chat/completions", &body).await)
            }
            _ => reported.add(
                "responses",
                &server.post("/v1/responses", &json!({"input": text})).await,
            ),
        };
    }
    run_session(&server, &mut reported).await;
    // One rejected request.
    let rejected = server.post("/v1/messages", &json!({"messages": []})).await;
    assert_eq!(rejected.status, 400);
    // One cancelled request: the client leaves while it generates.
    let mut body = messages("long");
    body["stream"] = json!(true);
    body["max_tokens"] = json!(3_000);
    let mut response = server
        .client
        .post(server.url("/v1/messages"))
        .json(&body)
        .send()
        .await
        .unwrap();
    response.chunk().await.unwrap();
    drop(response);
    let deadline = Instant::now() + Duration::from_secs(5);
    while !status(&server).await["requests"]
        .as_array()
        .unwrap()
        .is_empty()
    {
        assert!(
            Instant::now() < deadline,
            "the cancelled request is still listed"
        );
        tokio::time::sleep(Duration::from_millis(5)).await;
    }

    let (text, after) = metrics(&server).await;
    check_format(&text);
    let delta = |name: &str| after[name] - before.get(name).copied().unwrap_or(0.0);
    let sum = |prefix: &str| {
        after
            .iter()
            .filter(|(k, _)| k.starts_with(prefix))
            .map(|(k, v)| v - before.get(k).copied().unwrap_or(0.0))
            .sum::<f64>()
    };
    assert_eq!(reported.requests, 10.0 + 31.0);
    assert_eq!(sum("mrun_prompt_tokens_total"), reported.prompt);
    assert_eq!(sum("mrun_cached_tokens_total"), reported.cached);
    assert_eq!(sum("mrun_generated_tokens_total"), reported.generated);
    assert_eq!(delta("mrun_spliced_tokens_total"), reported.spliced);
    assert_eq!(delta("mrun_cache_unreused_tokens_total"), reported.unreused);
    for cause in ["new", "diverged", "evicted", "gap", "disabled"] {
        let name = format!("mrun_cache_misses_total{{cause=\"{cause}\"}}");
        let expected = reported.causes.get(cause).copied().unwrap_or(0.0);
        assert_eq!(delta(&name), expected, "{cause}: {:?}", reported.causes);
    }
    assert!(
        reported.causes.contains_key("diverged"),
        "the edited message: {:?}",
        reported.causes
    );
    assert_eq!(sum("mrun_requests_total"), reported.requests + 2.0);
    let outcome = |o: &str| {
        sum(&format!(
            "mrun_requests_total{{api=\"messages\",outcome=\"{o}\"}}"
        ))
    };
    assert_eq!(outcome("rejected"), 1.0);
    assert_eq!(outcome("cancelled"), 1.0);
    assert_eq!(
        sum("mrun_requests_total{api=\"chat\",outcome=\"completed\"}"),
        3.0
    );
}

#[tokio::test]
async fn no_content_in_metrics_or_status() {
    // FR-011, SC-004: neither endpoint carries prompt text, tool names, paths or client ids.
    let canary = mrun::bench::canary::line(&mut mrun::bench::suites::Rng::new(7));
    let tool = "zq_private_tool_8472";
    let path = "/Users/alice/secret-project/src/vault.rs";
    let client_model = "client-model-id-5521";
    let call = format!(
        "<tool_call>\n{{\"name\": \"{tool}\", \"arguments\": {{\"path\": \"{path}\"}}}}\n</tool_call>"
    );
    let replies = [call.as_str(), "fine", "fine", &"z".repeat(400)];
    let server = common::start_with(&replies, slow(5)).await;
    let tools = json!([{"name": tool, "description": format!("reads {path}"),
        "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}}}]);
    let body = json!({"model": client_model, "max_tokens": 100,
        "system": format!("{canary}\nWork in {path}."), "tools": tools,
        "messages": [{"role": "user", "content": format!("{canary} open {path}")}]});
    assert_eq!(server.post("/v1/messages", &body).await.status, 200);
    let chat = json!({"model": client_model, "messages": [
        {"role": "system", "content": canary}, {"role": "user", "content": path}]});
    assert_eq!(server.post("/v1/chat/completions", &chat).await.status, 200);
    let resp = json!({"model": client_model, "instructions": canary, "input": path});
    assert_eq!(server.post("/v1/responses", &resp).await.status, 200);
    // And while a request with the same content runs.
    let client = server.client.clone();
    let url = server.url("/v1/messages");
    let running = tokio::spawn(async move { client.post(url).json(&body).send().await });
    let deadline = Instant::now() + Duration::from_secs(5);
    while status(&server).await["requests"]
        .as_array()
        .unwrap()
        .is_empty()
    {
        assert!(Instant::now() < deadline);
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    let (text, _) = metrics(&server).await;
    let now = status(&server).await.to_string();
    for (name, scraped) in [("/metrics", &text), ("/mrun/status", &now)] {
        for secret in [
            canary.as_str(),
            "mrun-canary",
            tool,
            path,
            "secret-project",
            client_model,
        ] {
            assert!(!scraped.contains(secret), "{name} contains {secret:?}");
        }
    }
    running.await.unwrap().unwrap();
}

#[tokio::test]
async fn status_and_metrics_answer_during_prefill() {
    // FR-012: during a 30K-token prefill (~1.2 s: 20 ms per layer, four layers per 2,048-token
    // chunk), 50 polls of each endpoint answer with p95 < 100 ms, and `progress` rises.
    let setup = Setup {
        capacity: 40_000,
        layer_delay: Some(Duration::from_millis(20)),
        ..Setup::default()
    };
    let server = common::start_with(&["done"], setup).await;
    let client = server.client.clone();
    let url = server.url("/v1/messages");
    let long = messages(&"y".repeat(30_000));
    let running = tokio::spawn(async move { client.post(url).json(&long).send().await });
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        let now = status(&server).await;
        if now["requests"][0]["phase"] == "prefill" {
            break;
        }
        assert!(Instant::now() < deadline, "{now}");
        tokio::time::sleep(Duration::from_millis(2)).await;
    }
    let (mut status_times, mut metrics_times, mut progress) = (Vec::new(), Vec::new(), Vec::new());
    for _ in 0..50 {
        let started = Instant::now();
        let now = status(&server).await;
        status_times.push(started.elapsed());
        if now["requests"][0]["phase"] == "prefill" {
            progress.push(now["requests"][0]["progress"].as_u64().unwrap());
        }
        let started = Instant::now();
        metrics(&server).await;
        metrics_times.push(started.elapsed());
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    for (name, mut times) in [("status", status_times), ("metrics", metrics_times)] {
        times.sort();
        let p95 = times[50 * 95 / 100 - 1];
        assert!(p95 < Duration::from_millis(100), "{name}: p95 {p95:?}");
    }
    assert!(
        progress.windows(2).any(|w| w[1] > w[0]),
        "progress rises during prefill: {progress:?}"
    );
    running.await.unwrap().unwrap();
}

/// A non-streaming Messages request on `messages` (alternating user/assistant texts).
async fn converse(server: &common::Server, turns: &[&str], reported: &mut Reported) -> Value {
    let messages: Vec<Value> = turns
        .iter()
        .enumerate()
        .map(|(i, text)| {
            let role = if i % 2 == 0 { "user" } else { "assistant" };
            json!({"role": role, "content": text})
        })
        .collect();
    let body = json!({"max_tokens": 1000, "messages": messages});
    reported.add("messages", &server.post("/v1/messages", &body).await)
}

fn text_of(reply: &Value) -> String {
    reply["content"][0]["text"].as_str().unwrap().to_string()
}

#[tokio::test]
async fn metrics_count_every_cause_and_unreused() {
    // SC-006: a session producing each cause, a request with `unreused > 0`, and the ring flag.
    let mut replies = vec!["first", "other", "again", "edited", "c1", "c2"];
    replies.extend(std::iter::repeat_n("ok", 1_030));
    let setup = Setup {
        hybrid: true,
        ..Setup::default()
    };
    let server = common::start_with(&replies, setup).await;
    let (first_text, before) = metrics(&server).await;
    for cause in ["new", "diverged", "evicted", "gap", "disabled"] {
        let line = format!("mrun_cache_misses_total{{cause=\"{cause}\"}} 0\n");
        assert!(
            first_text.contains(&line),
            "{cause} at 0 from the first scrape"
        );
    }
    assert!(first_text.contains("mrun_cache_dropped_ring_wrapped 0\n"));
    let mut reported = Reported::default();
    // `new`, then a second conversation that drops the first one's short tail.
    let a = text_of(&converse(&server, &["conversation A about parsers"], &mut reported).await);
    converse(&server, &["conversation B about lexers"], &mut reported).await;
    // Continuing A: its state was dropped (`evicted`), with tokens it shared left unreused.
    let a2 = text_of(
        &converse(
            &server,
            &["conversation A about parsers", &a, "go on"],
            &mut reported,
        )
        .await,
    );
    // Editing A's second user message: `diverged`.
    converse(
        &server,
        &[
            "conversation A about parsers",
            &a,
            "go on, please",
            &a2,
            "more",
        ],
        &mut reported,
    )
    .await;
    // A prompt inside stored state with no checkpoint near its end resumes far behind the
    // match: `gap` (hybrid). An assistant prefill of 600 tokens, then of its first 300.
    let x = "x".repeat(600);
    converse(&server, &["conversation C", &x], &mut reported).await;
    converse(&server, &["conversation C", &x[..300]], &mut reported).await;
    // `disabled` on a `--no-cache` server.
    let off = common::start_with(
        &["x"],
        Setup {
            no_cache: true,
            ..Setup::default()
        },
    )
    .await;
    let mut off_reported = Reported::default();
    converse(&off, &["anything"], &mut off_reported).await;
    let (off_text, _) = metrics(&off).await;
    assert!(
        off_text.contains("mrun_cache_misses_total{cause=\"disabled\"} 1\n"),
        "{off_text}"
    );

    let (text, after) = metrics(&server).await;
    let delta = |name: &str| after[name] - before.get(name).copied().unwrap_or(0.0);
    for cause in ["new", "diverged", "evicted", "gap", "disabled"] {
        let name = format!("mrun_cache_misses_total{{cause=\"{cause}\"}}");
        let expected = reported.causes.get(cause).copied().unwrap_or(0.0);
        assert_eq!(delta(&name), expected, "{cause}: {:?}", reported.causes);
    }
    for cause in ["new", "diverged", "evicted", "gap"] {
        assert!(
            reported.causes.contains_key(cause),
            "{cause}: {:?}",
            reported.causes
        );
    }
    assert!(reported.unreused > 0.0, "{text}");
    assert_eq!(delta("mrun_cache_unreused_tokens_total"), reported.unreused);
    assert_eq!(after["mrun_cache_dropped_ring_wrapped"], 0.0);

    // More than 1,024 dropped paths wrap the ring.
    for i in 0..1_030 {
        let body = messages(&format!("side request {i}"));
        assert_eq!(server.post("/v1/messages", &body).await.status, 200);
    }
    let (_, wrapped) = metrics(&server).await;
    assert_eq!(wrapped["mrun_cache_dropped_ring_wrapped"], 1.0);
    assert_eq!(status(&server).await["cache"]["dropped_ring_wrapped"], true);
}
