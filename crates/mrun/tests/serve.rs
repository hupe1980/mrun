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
    Box::new(|_| {
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
            template: Arc::new(ChatTemplate::new("{{ messages[0].content }}".into()).unwrap()),
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
    // A model that does not exist (builds without `mlx` have no engine at all).
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
