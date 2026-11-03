//! Remote GPU mode (013) on loopback: the agent with its stub upstream, the local relay in
//! front of it, and a mock RunPod v2 endpoint for self-stop. No network.

use axum::Router;
use axum::extract::{Request, State};
use mrun::agent::lifecycle::Stopper;
use mrun::agent::stub::StubStats;
use mrun::agent::{AgentConfig, Backend, RunningAgent};
use mrun::api::relay::{PodMeta, RelayConfig, RunningRelay};
use serde_json::{Value, json};
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

const TOKEN: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
const KEEPALIVE: Duration = Duration::from_millis(200);

fn config(stopper: Stopper) -> AgentConfig {
    AgentConfig {
        bind: "127.0.0.1:0".parse().unwrap(),
        token: TOKEN.into(),
        model: "stub".into(),
        max_model_len: 65_536,
        max_num_seqs: 16,
        tool_parser: "hermes".into(),
        reasoning_parser: None,
        vllm_args: vec![],
        idle_stop: Duration::from_secs(3600),
        max_runtime: Duration::from_secs(3600),
        failed_stop: Duration::ZERO,
        grace: Duration::from_secs(5),
        cost_per_hr: 0.06,
        gpu: Some("CPU".into()),
        keepalive: KEEPALIVE,
        backend: Backend::Stub,
        stopper,
        tick: Duration::from_millis(50),
        load_timeout: Duration::from_secs(60),
        log_path: None,
        log_max_bytes: mrun::agent::logfile::MAX_BYTES,
    }
}

async fn agent(config: AgentConfig) -> (RunningAgent, String, Arc<StubStats>) {
    let agent = mrun::agent::start(config).await.unwrap();
    let base = format!("http://{}", agent.addr);
    let start = Instant::now();
    loop {
        if agent.shared.phase() == mrun::agent::Phase::Ready {
            break;
        }
        assert!(start.elapsed() < Duration::from_secs(5), "agent not ready");
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    let stats = agent.shared.stub.lock().unwrap().clone().unwrap();
    (agent, base, stats)
}

async fn relay(agent_url: &str) -> (RunningRelay, String) {
    let (relay, base) = relay_saving(agent_url, None).await;
    let http = reqwest::Client::new();
    let start = Instant::now();
    loop {
        let health = http.get(format!("{base}/health")).send().await.unwrap();
        if health.status() == 200 {
            break;
        }
        assert!(start.elapsed() < Duration::from_secs(5), "relay not ready");
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    (relay, base)
}

/// A relay that saves the pod's log to `failure_log` when the agent fails (not waited on).
async fn relay_saving(
    agent_url: &str,
    failure_log: Option<std::path::PathBuf>,
) -> (RunningRelay, String) {
    let relay = mrun::api::relay::start(RelayConfig {
        port: 0,
        agent_url: agent_url.into(),
        token: TOKEN.into(),
        pod: PodMeta {
            id: Some("pod1".into()),
            gpu: Some("CPU".into()),
            cost_per_hr: Some(0.06),
            since_unix: None,
        },
        keepalive: KEEPALIVE,
        poll: Duration::from_millis(50),
        max_output: None,
        failure_log,
    })
    .await
    .unwrap();
    let base = format!("http://{}", relay.addr);
    (relay, base)
}

fn chat(text: &str, stream: bool) -> Value {
    json!({"model": "my-alias", "stream": stream,
        "messages": [{"role": "user", "content": text}]})
}

fn messages(text: &str, stream: bool) -> Value {
    json!({"model": "claude-sonnet-x", "max_tokens": 100, "stream": stream,
        "system": [{"type": "text", "text": "x-anthropic-billing-header: cc_version=2.1; cch=9f3c1;\nYou are Claude Code."}],
        "messages": [{"role": "user", "content": text}],
        "context_management": {"edits": []}})
}

fn responses(text: &str, stream: bool) -> Value {
    json!({"model": "gpt-x", "stream": stream, "input": [{"role": "user", "content": text}]})
}

/// Reads a whole SSE body as text frames with their arrival times.
async fn frames(mut response: reqwest::Response) -> Vec<(Duration, String)> {
    let start = Instant::now();
    let mut out = Vec::new();
    while let Some(chunk) = response.chunk().await.unwrap() {
        out.push((
            start.elapsed(),
            String::from_utf8_lossy(&chunk).into_owned(),
        ));
    }
    out
}

#[tokio::test]
async fn requests_without_the_pod_token_get_401_and_reach_nothing() {
    let (agent, base, stats) = agent(config(Stopper::Exit)).await;
    let http = reqwest::Client::new();
    let previous_pod = "f".repeat(64);
    for auth in [
        None,
        Some("Bearer wrong"),
        Some("Basic abc"),
        Some(previous_pod.as_str()),
    ] {
        for (path, body) in [
            ("/v1/chat/completions", chat("hi", false)),
            ("/v1/messages", messages("hi", false)),
            ("/v1/messages/count_tokens", messages("hi", false)),
            ("/v1/responses", responses("hi", false)),
        ] {
            let mut request = http.post(format!("{base}{path}")).json(&body);
            if let Some(auth) = auth {
                let value = if auth.len() == 64 {
                    format!("Bearer {auth}")
                } else {
                    auth.to_string()
                };
                request = request.header("authorization", value);
            }
            let response = request.send().await.unwrap();
            assert_eq!(response.status(), 401, "{path} {auth:?}");
            let body: Value = response.json().await.unwrap();
            if path.starts_with("/v1/messages") {
                assert_eq!(body["error"]["type"], "authentication_error");
            } else {
                assert_eq!(body["error"]["code"], "invalid_api_key");
            }
        }
        for path in ["/v1/models", "/mrun/status"] {
            let mut request = http.get(format!("{base}{path}"));
            if let Some(auth) = auth {
                request = request.header("authorization", auth);
            }
            assert_eq!(request.send().await.unwrap().status(), 401, "{path}");
        }
    }
    assert_eq!(
        stats.requests.load(Ordering::SeqCst),
        0,
        "nothing was forwarded"
    );
    // The idle timer was not reset by refused requests: no activity was recorded.
    assert_eq!(agent.shared.requests_total.load(Ordering::SeqCst), 0);
    let health: Value = http
        .get(format!("{base}/health"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(health, json!({"status": "ok", "phase": "ready"}));
    let ok = http
        .get(format!("{base}/mrun/status"))
        .bearer_auth(TOKEN)
        .send()
        .await
        .unwrap();
    assert_eq!(ok.status(), 200);
    let status: Value = ok.json().await.unwrap();
    assert_eq!(status["role"], "agent");
    assert_eq!(status["backend"], "stub stub");
    assert!(!status.to_string().contains(TOKEN));
    agent.shutdown().await;
}

#[tokio::test]
async fn the_relay_serves_all_three_apis_with_model_echo_and_the_local_surfaces() {
    let (agent, agent_url, stats) = agent(config(Stopper::Exit)).await;
    let (relay, base) = relay(&agent_url).await;
    let http = reqwest::Client::new();

    // Local surfaces: /v1/models with the remote context, status with backend and pod.
    let models: Value = http
        .get(format!("{base}/v1/models"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(models["data"][0]["id"], "stub");
    assert_eq!(models["data"][0]["context_window"], 65_536);
    assert_eq!(models["data"][0]["max_output_tokens"], 32_768);
    let status: Value = http
        .get(format!("{base}/mrun/status"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(status["backend"], "stub stub");
    assert_eq!(status["pod"]["id"], "pod1");
    assert!(status["budget"].is_null() && status["cache"].is_null() && status["last"].is_null());
    assert!(!status.to_string().contains(TOKEN));
    let port = relay.addr.port();
    let info = tokio::task::spawn_blocking(move || mrun::cli::server::ServerInfo::fetch(port))
        .await
        .unwrap()
        .unwrap();
    assert_eq!((info.model_id.as_str(), info.context), ("stub", 65_536));

    // Chat, not streaming: the requested id is echoed; vLLM saw its served name.
    let response = http
        .post(format!("{base}/v1/chat/completions"))
        .json(&chat("hi", false))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
    assert_eq!(response.headers()["mrun-backend"], "stub stub");
    let body: Value = response.json().await.unwrap();
    assert_eq!(body["model"], "my-alias");
    assert_eq!(
        body["choices"][0]["message"]["content"],
        "hello from the stub"
    );
    assert_eq!(
        stats.last_body.lock().unwrap().as_ref().unwrap()["model"],
        "stub"
    );

    // Messages: nonce zeroed, unknown fields dropped, model echoed in message_start.
    let response = http
        .post(format!("{base}/v1/messages?beta=true"))
        .header("anthropic-beta", "x")
        .json(&messages("hi", true))
        .send()
        .await
        .unwrap();
    let text: String = frames(response).await.into_iter().map(|(_, t)| t).collect();
    assert!(text.contains("event: message_stop"), "{text}");
    assert!(text.contains("\"model\":\"claude-sonnet-x\""), "{text}");
    let seen = stats.last_body.lock().unwrap().clone().unwrap();
    assert!(
        seen["system"][0]["text"]
            .as_str()
            .unwrap()
            .contains("cch=00000;")
    );
    assert!(seen.get("context_management").is_none());
    assert_eq!(seen["model"], "stub");

    // count_tokens and Responses (streaming and not).
    let count: Value = http
        .post(format!("{base}/v1/messages/count_tokens"))
        .json(&messages("hi", false))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert!(count["input_tokens"].as_u64().unwrap() > 0);
    let whole: Value = http
        .post(format!("{base}/v1/responses"))
        .json(&responses("hi", false))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(whole["model"], "gpt-x");
    let response = http
        .post(format!("{base}/v1/responses"))
        .json(&responses("hi", true))
        .send()
        .await
        .unwrap();
    let text: String = frames(response).await.into_iter().map(|(_, t)| t).collect();
    assert!(text.contains("event: response.completed"), "{text}");
    assert!(text.contains("\"model\":\"gpt-x\""), "{text}");
    let response = http
        .post(format!("{base}/v1/chat/completions"))
        .json(&chat("hi", true))
        .send()
        .await
        .unwrap();
    let text: String = frames(response).await.into_iter().map(|(_, t)| t).collect();
    assert!(text.ends_with("data: [DONE]\n\n"), "{text}");
    assert!(text.contains("\"model\":\"my-alias\""), "{text}");
    assert!(!text.contains("\"model\":\"stub\""), "{text}");

    // The relay keeps the loopback guard.
    let foreign = http
        .get(format!("{base}/v1/models"))
        .header("host", "evil.example")
        .send()
        .await
        .unwrap();
    assert_eq!(foreign.status(), 403);
    relay.shutdown().await;
    agent.shutdown().await;
}

#[tokio::test]
async fn slow_first_tokens_get_headers_at_once_and_keepalives_in_each_apis_format() {
    let (agent, agent_url, _) = agent(config(Stopper::Exit)).await;
    let (relay, base) = relay(&agent_url).await;
    let http = reqwest::Client::new();
    for (path, body, keepalive) in [
        (
            "/v1/messages",
            messages("slow:1.2", true),
            "event: ping\ndata: {\"type\":\"ping\"}\n\n",
        ),
        (
            "/v1/chat/completions",
            chat("slow:1.2", true),
            "\"choices\":[]",
        ),
        (
            "/v1/responses",
            responses("slow:1.2", true),
            ": keepalive\n\n",
        ),
    ] {
        for target in [&agent_url, &base] {
            // The agent forwards as is: it needs the served name (the relay maps any name).
            let mut body = body.clone();
            if target == &agent_url {
                body["model"] = "stub".into();
            }
            let start = Instant::now();
            let response = http
                .post(format!("{target}{path}"))
                .bearer_auth(TOKEN)
                .json(&body)
                .send()
                .await
                .unwrap();
            assert!(
                start.elapsed() < Duration::from_millis(500),
                "headers took {:?} ({path} via {target})",
                start.elapsed()
            );
            assert_eq!(response.headers()["content-type"], "text/event-stream");
            let frames = frames(response).await;
            let pings: Vec<_> = frames
                .iter()
                .filter(|(_, t)| t.contains(keepalive))
                .collect();
            assert!(pings.len() >= 3, "{path} via {target}: {frames:?}");
            let mut last = Duration::ZERO;
            for (at, _) in &pings {
                assert!(*at - last <= KEEPALIVE * 2, "{path}: gap {:?}", *at - last);
                last = *at;
            }
            let text: String = frames.iter().map(|(_, t)| t.as_str()).collect();
            assert!(text.contains("hello") || text.contains("stub"), "{text}");
        }
    }
    relay.shutdown().await;
    agent.shutdown().await;
}

#[tokio::test]
async fn a_client_disconnect_aborts_the_upstream_through_relay_and_agent() {
    let (agent, agent_url, stats) = agent(config(Stopper::Exit)).await;
    let (relay, base) = relay(&agent_url).await;
    let http = reqwest::Client::new();
    let mut response = http
        .post(format!("{base}/v1/chat/completions"))
        .json(&chat("long:100", true))
        .send()
        .await
        .unwrap();
    let first = response.chunk().await.unwrap().unwrap();
    assert!(!first.is_empty());
    let dropped = Instant::now();
    drop(response);
    while stats.aborted.load(Ordering::SeqCst) == 0 {
        assert!(
            dropped.elapsed() < Duration::from_secs(1),
            "the upstream stream was not aborted within 1 s"
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    // The agent's activity count drops back too.
    let start = Instant::now();
    while agent.shared.activity.active.load(Ordering::SeqCst) > 0 {
        assert!(start.elapsed() < Duration::from_secs(1));
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    // An upstream error after the headers becomes one API error event.
    let response = http
        .post(format!("{base}/v1/chat/completions"))
        .json(&chat("error", true))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
    let text: String = frames(response).await.into_iter().map(|(_, t)| t).collect();
    assert!(
        text.contains("\"error\"") && text.contains("stub error"),
        "{text}"
    );
    let response = http
        .post(format!("{base}/v1/messages"))
        .json(&messages("error", true))
        .send()
        .await
        .unwrap();
    let text: String = frames(response).await.into_iter().map(|(_, t)| t).collect();
    assert!(text.starts_with("event: error\n"), "{text}");
    relay.shutdown().await;
    agent.shutdown().await;
}

type Calls = Arc<Mutex<Vec<(String, String, String, Value)>>>;

/// A mock RunPod v2 endpoint recording (path, user agent, authorization, body).
async fn mock_v2() -> (String, Calls) {
    let calls: Calls = Arc::default();
    async fn record(State(calls): State<Calls>, request: Request) -> axum::Json<Value> {
        let (parts, body) = request.into_parts();
        let bytes = axum::body::to_bytes(body, 1 << 16).await.unwrap();
        let header = |name: &str| {
            parts
                .headers
                .get(name)
                .and_then(|v| v.to_str().ok())
                .unwrap_or_default()
                .to_string()
        };
        calls.lock().unwrap().push((
            parts.uri.path().to_string(),
            header("user-agent"),
            header("authorization"),
            serde_json::from_slice(&bytes).unwrap_or(Value::Null),
        ));
        axum::Json(json!({"status": "ok"}))
    }
    let app = Router::new().fallback(record).with_state(calls.clone());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await });
    (format!("http://{addr}"), calls)
}

fn runpod_stopper(base: &str) -> Stopper {
    let client = mrun_runpod::Client::with_endpoints(
        mrun_runpod::ApiKey::new("pod-scoped-key"),
        mrun_runpod::Endpoints {
            rest: format!("{base}/v1"),
            graphql: format!("{base}/graphql"),
            v2: format!("{base}/v2"),
        },
    )
    .unwrap();
    Stopper::RunPod {
        client,
        pod_id: "pod1".into(),
    }
}

#[tokio::test]
async fn an_idle_agent_stops_its_pod_but_not_during_a_stream() {
    let (v2, calls) = mock_v2().await;
    let mut cfg = config(runpod_stopper(&v2));
    cfg.idle_stop = Duration::from_millis(400);
    let (agent, base, _) = agent(cfg).await;
    let http = reqwest::Client::new();
    // A 2 s stream (20 tokens × 100 ms) keeps the agent busy past the idle limit.
    let response = http
        .post(format!("{base}/v1/chat/completions"))
        .bearer_auth(TOKEN)
        .json(&json!({"model": "stub", "stream": true,
            "messages": [{"role": "user", "content": "long:20"}]}))
        .send()
        .await
        .unwrap();
    let ended = {
        let frames = frames(response).await;
        assert!(frames.iter().any(|(_, t)| t.contains("[DONE]")));
        Instant::now()
    };
    assert!(calls.lock().unwrap().is_empty(), "stopped during a stream");
    while calls.lock().unwrap().is_empty() {
        assert!(ended.elapsed() < Duration::from_secs(3), "no idle stop");
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    assert!(
        ended.elapsed() >= Duration::from_millis(350),
        "{:?}",
        ended.elapsed()
    );
    let (path, user_agent, auth, body) = calls.lock().unwrap()[0].clone();
    assert_eq!(path, "/v2/pods/pod1/action");
    assert!(user_agent.starts_with("mrun/"), "{user_agent}");
    assert_eq!(auth, "Bearer pod-scoped-key");
    assert_eq!(body, json!({"action": "stop"}));
    let status: Value = http
        .get(format!("{base}/mrun/status"))
        .bearer_auth(TOKEN)
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(
        (status["phase"].as_str(), status["reason"].as_str()),
        (Some("stopping"), Some("idle"))
    );
    // New requests are refused while stopping.
    let refused = http
        .post(format!("{base}/v1/chat/completions"))
        .bearer_auth(TOKEN)
        .json(&chat("hi", false))
        .send()
        .await
        .unwrap();
    assert_eq!(refused.status(), 503);
    agent.shutdown().await;
}

#[tokio::test]
async fn the_maximum_runtime_stops_the_pod_after_the_running_response() {
    let (v2, calls) = mock_v2().await;
    let mut cfg = config(runpod_stopper(&v2));
    cfg.max_runtime = Duration::from_millis(600);
    let (agent, base, _) = agent(cfg).await;
    let http = reqwest::Client::new();
    let response = http
        .post(format!("{base}/v1/chat/completions"))
        .bearer_auth(TOKEN)
        .json(&json!({"model": "stub", "stream": true,
            "messages": [{"role": "user", "content": "long:15"}]}))
        .send()
        .await
        .unwrap();
    let frames = frames(response).await;
    let ended = Instant::now();
    let text: String = frames.iter().map(|(_, t)| t.as_str()).collect();
    assert!(
        text.contains("[DONE]"),
        "the running response finished: {text}"
    );
    while calls.lock().unwrap().is_empty() {
        assert!(ended.elapsed() < Duration::from_secs(2), "no runtime stop");
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    let status = agent.shared.status_json();
    assert_eq!(status["reason"], "max_runtime");
    agent.shutdown().await;
}

fn header(response: &reqwest::Response, name: &str) -> u64 {
    response.headers()[name].to_str().unwrap().parse().unwrap()
}

#[tokio::test]
async fn the_log_route_needs_the_token_reads_by_offset_and_follows() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("logs/vllm.log");
    let (agent, base, _) = agent(AgentConfig {
        log_path: Some(path.clone()),
        ..config(Stopper::Exit)
    })
    .await;
    let http = reqwest::Client::new();
    // 401 without (or with a wrong) token, nothing read.
    let none = http.get(format!("{base}/mrun/logs")).send().await.unwrap();
    assert_eq!(none.status(), 401);
    let wrong = http
        .get(format!("{base}/mrun/logs?offset=0"))
        .bearer_auth("f".repeat(64))
        .send()
        .await
        .unwrap();
    assert_eq!(wrong.status(), 401);
    assert!(!wrong.text().await.unwrap().contains("[agent]"));

    // The agent's own lines are there, timestamped; the headers carry the offsets.
    let all = http
        .get(format!("{base}/mrun/logs"))
        .bearer_auth(TOKEN)
        .send()
        .await
        .unwrap();
    assert_eq!(all.status(), 200);
    let (start, next, end) = (
        header(&all, "x-mrun-log-start"),
        header(&all, "x-mrun-log-offset"),
        header(&all, "x-mrun-log-end"),
    );
    let text = all.text().await.unwrap();
    assert_eq!((start, next), (0, text.len() as u64));
    assert!(next <= end);
    assert!(text.contains("[agent] mrun agent "), "{text}");
    assert!(text.contains("[agent] phase ready"), "{text}");
    assert!(std::fs::read_to_string(&path).unwrap().starts_with(&text));

    // From an offset: only what follows it; past the end: nothing, at the end.
    agent.shared.log.append("vllm line one");
    let rest = http
        .get(format!("{base}/mrun/logs?offset={next}"))
        .bearer_auth(TOKEN)
        .send()
        .await
        .unwrap();
    assert_eq!(header(&rest, "x-mrun-log-start"), next);
    let rest_text = rest.text().await.unwrap();
    assert_eq!(rest_text.lines().count(), 1);
    assert!(rest_text.ends_with("Z vllm line one\n"), "{rest_text}");
    let past = http
        .get(format!("{base}/mrun/logs?offset=99999999"))
        .bearer_auth(TOKEN)
        .send()
        .await
        .unwrap();
    let past_end = header(&past, "x-mrun-log-end");
    assert_eq!(header(&past, "x-mrun-log-offset"), past_end);
    assert!(past.text().await.unwrap().is_empty());
    let bad = http
        .get(format!("{base}/mrun/logs?offset=x"))
        .bearer_auth(TOKEN)
        .send()
        .await
        .unwrap();
    assert_eq!(bad.status(), 400);

    // follow=1 streams new lines as they are written, keepalives (empty lines) in between.
    let mut follow = http
        .get(format!("{base}/mrun/logs?offset={past_end}&follow=1"))
        .bearer_auth(TOKEN)
        .send()
        .await
        .unwrap();
    assert_eq!(follow.status(), 200);
    assert_eq!(header(&follow, "x-mrun-log-start"), past_end);
    let writer = agent.shared.log.clone();
    tokio::spawn(async move {
        tokio::time::sleep(KEEPALIVE * 3).await;
        writer.append("late line two");
        writer.append("late line three");
    });
    let mut received = String::new();
    let start = Instant::now();
    while !received.contains("late line three\n") {
        assert!(start.elapsed() < Duration::from_secs(5), "{received:?}");
        let chunk = tokio::time::timeout(Duration::from_secs(5), follow.chunk())
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        received.push_str(&String::from_utf8_lossy(&chunk));
    }
    assert!(
        received.starts_with('\n'),
        "a keepalive came first: {received:?}"
    );
    let lines: Vec<&str> = received.lines().filter(|l| !l.is_empty()).collect();
    assert_eq!(lines.len(), 2, "{received:?}");
    assert!(lines[0].ends_with("Z late line two"));
    // Reading the log is no request: the idle timer and counters are untouched.
    assert_eq!(agent.shared.requests_total.load(Ordering::Relaxed), 0);
    drop(follow);
    agent.shutdown().await;
}

#[tokio::test]
async fn a_failed_agent_reaches_clients_with_its_reason_and_the_relay_saves_the_pod_log() {
    let dir = tempfile::tempdir().unwrap();
    // vLLM cannot start: the agent fails at once (before serving).
    let agent = mrun::agent::start(AgentConfig {
        backend: mrun::agent::Backend::Vllm {
            program: dir.path().join("no-vllm").display().to_string(),
            port: 9,
        },
        log_path: Some(dir.path().join("pod/vllm.log")),
        ..config(Stopper::Exit)
    })
    .await
    .unwrap();
    let saved = dir.path().join("saved/cloud-pod1.log");
    let (_relay, base) = relay_saving(&format!("http://{}", agent.addr), Some(saved.clone())).await;
    let start = Instant::now();
    while !saved.exists() {
        assert!(start.elapsed() < Duration::from_secs(5), "no log saved");
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    let log = std::fs::read_to_string(&saved).unwrap();
    assert!(log.contains("[agent] cannot start vllm"), "{log}");
    let response = reqwest::Client::new()
        .post(format!("{base}/v1/chat/completions"))
        .json(&chat("hi", false))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 500);
    let text = response.text().await.unwrap();
    assert!(
        text.contains("the remote model failed to start: cannot start vllm"),
        "{text}"
    );
}
