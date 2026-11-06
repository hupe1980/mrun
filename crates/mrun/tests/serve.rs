//! `mrun serve` skeleton (002 T000/T001): lifecycle, readiness, the loopback guard and probes.

use mrun::chat::template::ChatTemplate;
use mrun::chat::tokenizer::test_support::byte_tokenizer;
use mrun::cli::serve::{self, EngineFactory, Loaded, RunningServer, ServeArgs};
use mrun::errors::Code;
use mrun::runtime::handle::{DEFAULT_STALL_TIMEOUT, EngineHandle};
use mrun::runtime::scripted::ScriptedEngine;
use mrun_engine::{Sampling, SyntaxId};
use std::process::Command;
use std::sync::Arc;

fn args() -> ServeArgs {
    use clap::Parser;
    #[derive(Parser)]
    struct Wrap {
        #[command(flatten)]
        serve: ServeArgs,
    }
    let mut serve = Wrap::parse_from(["mrun"]).serve;
    serve.port = 0;
    serve
}

fn scripted() -> EngineFactory {
    scripted_with("{{ messages[0].content }}")
}

fn scripted_with(template: &'static str) -> EngineFactory {
    Box::new(move |_| {
        let tokenizer = Arc::new(byte_tokenizer());
        let handle = EngineHandle::spawn(
            || Ok(ScriptedEngine::new(Vec::new(), 4096)),
            tokenizer.clone(),
            DEFAULT_STALL_TIMEOUT,
        )?;
        Ok(Loaded {
            model_id: "scripted".into(),
            handle,
            tokenizer,
            template: Arc::new(ChatTemplate::new(template.into()).unwrap()),
            syntax: SyntaxId::Hermes,
            supports_thinking: true,
            defaults: Sampling::GREEDY,
            grammar: None,
        })
    })
}

async fn get(server: &RunningServer, path: &str, headers: &[(&str, &str)]) -> (u16, String) {
    let mut request = reqwest::Client::new().get(format!("http://{}{path}", server.addr));
    for (name, value) in headers {
        request = request.header(*name, *value);
    }
    let response = request.send().await.unwrap();
    (response.status().as_u16(), response.text().await.unwrap())
}

#[tokio::test]
async fn health_is_503_while_loading_then_ok() {
    let (release, wait) = std::sync::mpsc::channel::<()>();
    let inner = scripted();
    let factory: EngineFactory = Box::new(move |args| {
        wait.recv().unwrap();
        inner(args)
    });
    let mut server = serve::start(args(), factory).await.unwrap();
    let (status, body) = get(&server, "/health", &[]).await;
    assert_eq!(status, 503);
    assert!(body.contains(r#""status":"loading""#), "{body}");

    release.send(()).unwrap();
    server.loaded().await.unwrap();
    let (status, body) = get(&server, "/health", &[]).await;
    assert_eq!(status, 200);
    assert!(body.contains(r#""model":"scripted""#), "{body}");
    server.shutdown().await.unwrap();
}

#[tokio::test]
async fn failed_load_is_reported() {
    // A model that does not exist (builds without `engine` have no engine at all).
    let mut args = args();
    args.model = Some("/nonexistent/mrun-test-model".into());
    let mut server = serve::start(args, serve::default_factory()).await.unwrap();
    let error = server.loaded().await.unwrap_err();
    assert!(
        error.message.contains("no inference engine") || error.code == Code::ModelNotFound,
        "{error}"
    );
    assert_ne!(error.exit() as u8, 0);
    let (status, body) = get(&server, "/health", &[]).await;
    assert_eq!(status, 503);
    assert!(body.contains(r#""status":"error""#), "{body}");
    // A failed load is permanent: a server error, not overload that clients retry.
    let response = reqwest::Client::new()
        .post(format!("http://{}/v1/messages", server.addr))
        .json(&serde_json::json!({"model": "m", "max_tokens": 1,
            "messages": [{"role": "user", "content": "hi"}]}))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status().as_u16(), 500);
    let body: serde_json::Value = response.json().await.unwrap();
    assert_eq!(body["error"]["type"], "api_error", "{body}");
    server.shutdown().await.unwrap();
}

/// 002 SC-006: recording is a command-line decision only; no environment variable enables it.
#[tokio::test]
async fn xapi_record_cli_only_audit() {
    use clap::CommandFactory;
    #[derive(clap::Parser)]
    struct Wrap {
        #[command(flatten)]
        serve: ServeArgs,
    }
    let command = Wrap::command();
    let record = command
        .get_arguments()
        .find(|a| a.get_id() == "record")
        .unwrap();
    assert!(
        record.get_env().is_none(),
        "--record must not read the environment"
    );

    let mut server = serve::start(args(), scripted()).await.unwrap();
    server.loaded().await.unwrap();
    let (_, body) = get(&server, "/mrun/status", &[]).await;
    let status: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(status["recording"], false, "{status}");
    server.shutdown().await.unwrap();
}

#[tokio::test]
async fn xapi_hello_probe_and_beta_query_accepted() {
    let mut server = serve::start(args(), scripted()).await.unwrap();
    server.loaded().await.unwrap();
    let response = reqwest::Client::new()
        .head(format!("http://{}/api/hello", server.addr))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status().as_u16(), 200);
    let (status, _) = get(
        &server,
        "/health?beta=true",
        &[("anthropic-beta", "interleaved-thinking-2025-05-14")],
    )
    .await;
    assert_eq!(status, 200);
    server.shutdown().await.unwrap();
}

#[tokio::test]
async fn xapi_host_and_origin_guard() {
    let mut server = serve::start(args(), scripted()).await.unwrap();
    server.loaded().await.unwrap();
    let port = server.addr.port();
    let own = format!("http://localhost:{port}");
    let (status, body) = get(&server, "/health", &[("host", "evil.example")]).await;
    assert_eq!(status, 403, "{body}");
    let (status, _) = get(&server, "/health", &[("origin", "http://evil.example")]).await;
    assert_eq!(status, 403);
    let response = reqwest::Client::new()
        .get(format!("http://{}/health", server.addr))
        .header("origin", &own)
        .send()
        .await
        .unwrap();
    assert_eq!(response.status().as_u16(), 200);
    assert!(
        response
            .headers()
            .keys()
            .all(|name| !name.as_str().starts_with("access-control-")),
        "no CORS headers"
    );
    server.shutdown().await.unwrap();
}

#[test]
fn xapi_host_flag_usage_error() {
    let output = Command::new(env!("CARGO_BIN_EXE_mrun"))
        .args(["serve", "--host", "0.0.0.0"])
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(2));
}

#[test]
fn serve_flags_read_mrun_environment() {
    let output = Command::new(env!("CARGO_BIN_EXE_mrun"))
        .args(["serve"])
        .env("MRUN_PORT", "not-a-port")
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&output.stderr).contains("not-a-port"));
}

/// 025: a tool loop and an edited prompt land on one session's main row; the edit's excerpt
/// shows the changed text; `/mrun/sessions` carries no text.
#[tokio::test]
async fn sessions_group_a_tool_loop_and_explain_an_edit() {
    let template = "{%- for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n\
                    {%- endfor %}{%- if add_generation_prompt %}<|im_start|>assistant\n{%- endif %}";
    let mut server = serve::start(args(), scripted_with(template)).await.unwrap();
    server.loaded().await.unwrap();
    let client = reqwest::Client::new();
    let post = |messages: serde_json::Value| {
        let request = client
            .post(format!("http://{}/v1/chat/completions", server.addr))
            .header("user-agent", "claude-cli/2.1.3 (external, cli)")
            .json(&serde_json::json!({
                "model": "scripted", "max_tokens": 1, "messages": messages,
                "tools": [{"type": "function", "function": {"name": "bash", "parameters": {}}}],
            }));
        async move { assert_eq!(request.send().await.unwrap().status(), 200) }
    };
    let m = |role: &str, content: &str| serde_json::json!({"role": role, "content": content});
    let sys = m("system", "You are a coding agent.");
    let ask = m("user", "run the build");
    post(serde_json::json!([sys, ask])).await;
    let ran = |secs: &str| m("user", &format!("build ok in {secs} and the tests pass"));
    post(serde_json::json!([
        sys,
        ask,
        m("assistant", "running"),
        ran("12.4s")
    ]))
    .await;
    post(serde_json::json!([
        sys,
        ask,
        m("assistant", "running"),
        ran("12.4s"),
        m("assistant", "done"),
        m("user", "thanks")
    ]))
    .await;
    post(serde_json::json!([
        sys,
        ask,
        m("assistant", "running"),
        ran("11.9s"),
        m("assistant", "done"),
        m("user", "thanks")
    ]))
    .await;

    let (status, body) = get(&server, "/mrun/sessions", &[]).await;
    assert_eq!(status, 200);
    let sessions: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(
        sessions["sessions"].as_array().unwrap().len(),
        1,
        "{sessions}"
    );
    let session = &sessions["sessions"][0];
    assert_eq!(
        (session["label"].as_str(), session["requests"].as_u64()),
        (Some("Claude Code"), Some(4))
    );
    assert_eq!(
        session["rows"],
        serde_json::json!([{"id": "main", "kind": "main", "requests": 4}]),
        "{body}"
    );
    let requests = sessions["requests"].as_array().unwrap();
    assert!(
        requests
            .iter()
            .all(|r| r["row"] == "main" && r["session"] == "s1")
    );
    let edit = requests.last().unwrap();
    assert_eq!(
        (edit["cause"].as_str(), edit["excerpt"].as_bool()),
        (Some("diverged"), Some(true)),
        "{edit}"
    );
    assert!(
        !body.contains("build ok"),
        "no prompt text in /mrun/sessions"
    );
    let (_, status_body) = get(&server, "/mrun/status", &[]).await;
    assert!(
        !status_body.contains("build ok"),
        "no prompt text in /mrun/status"
    );
    let since = edit["id"].as_u64().unwrap() - 1;
    let (_, newer) = get(&server, &format!("/mrun/sessions?since={since}"), &[]).await;
    let newer: serde_json::Value = serde_json::from_str(&newer).unwrap();
    assert_eq!(newer["requests"].as_array().unwrap().len(), 1);

    let id = edit["id"].as_u64().unwrap();
    let (status, body) = get(&server, &format!("/mrun/requests/{id}/excerpt"), &[]).await;
    assert_eq!(status, 200, "{body}");
    let excerpt: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert!(
        excerpt["context"]
            .as_str()
            .unwrap()
            .ends_with("build ok in 1"),
        "{excerpt}"
    );
    assert!(
        excerpt["before"].as_str().unwrap().starts_with("2.4s"),
        "{excerpt}"
    );
    assert!(
        excerpt["after"].as_str().unwrap().starts_with("1.9s"),
        "{excerpt}"
    );
    let (status, _) = get(&server, "/mrun/requests/999999/excerpt", &[]).await;
    assert_eq!(status, 404);
    let first = requests[0]["id"].as_u64().unwrap();
    let (status, _) = get(&server, &format!("/mrun/requests/{first}/excerpt"), &[]).await;
    assert_eq!(status, 404);
    server.shutdown().await.unwrap();
}
