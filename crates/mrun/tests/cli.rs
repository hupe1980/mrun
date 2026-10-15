//! The client-side commands of 004 against a server started in-process: `connect`, `launch
//! --dry-run`, `ps`, `doctor`, `stop`, `serve --detach` and `completions`.

use mrun::chat::template::ChatTemplate;
use mrun::chat::tokenizer::test_support::byte_tokenizer;
use mrun::cli::serve::{self, EngineFactory, Loaded, RunningServer, ServeArgs};
use mrun::runtime::handle::{DEFAULT_STALL_TIMEOUT, EngineHandle};
use mrun::runtime::scripted::ScriptedEngine;
use mrun_engine::{Sampling, SyntaxId};
use std::process::{Command, Output};
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

async fn server(capacity: u32) -> RunningServer {
    let factory: EngineFactory = Box::new(move |args| {
        let tokenizer = Arc::new(byte_tokenizer());
        let handle = EngineHandle::spawn_with(
            move || Ok(ScriptedEngine::new(Vec::new(), capacity)),
            tokenizer.clone(),
            DEFAULT_STALL_TIMEOUT,
            serve::cache_policy(args),
        )?;
        Ok(Loaded {
            model_id: "mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit".into(),
            handle,
            tokenizer,
            template: Arc::new(ChatTemplate::new("{{ messages[0].content }}".into()).unwrap()),
            syntax: SyntaxId::QwenXml,
            supports_thinking: false,
            defaults: Sampling::GREEDY,
            grammar: None,
        })
    });
    let mut server = serve::start(args(), factory).await.unwrap();
    server.loaded().await.unwrap();
    server
}

/// Runs the binary off the async runtime with an isolated `MRUN_HOME` and no client binaries.
async fn mrun(home: &std::path::Path, args: &[String]) -> Output {
    let home = home.to_path_buf();
    let args = args.to_vec();
    tokio::task::spawn_blocking(move || {
        Command::new(env!("CARGO_BIN_EXE_mrun"))
            .args(&args)
            .env("MRUN_HOME", &home)
            .env("PATH", "/nonexistent")
            .env_remove("MRUN_PORT")
            .output()
            .unwrap()
    })
    .await
    .unwrap()
}

fn strings(parts: &[&str]) -> Vec<String> {
    parts.iter().map(|s| s.to_string()).collect()
}

fn stdout(output: &Output) -> String {
    String::from_utf8_lossy(&output.stdout).into_owned()
}

#[tokio::test]
async fn connect_and_launch_use_the_running_servers_values() {
    let server = server(105_000).await;
    let port = server.addr.port().to_string();
    let home = tempfile::tempdir().unwrap();

    let out = mrun(
        home.path(),
        &strings(&["connect", "claude-code", "--port", &port]),
    )
    .await;
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let text = stdout(&out);
    assert!(text.contains("~/.claude/settings.json"));
    assert!(text.contains(&format!(
        "\"ANTHROPIC_BASE_URL\": \"http://127.0.0.1:{port}\""
    )));
    assert!(text.contains("\"CLAUDE_CODE_MAX_CONTEXT_TOKENS\": \"105000\""));
    assert!(
        !home.path().join("codex").exists(),
        "connect writes nothing"
    );

    let out = mrun(
        home.path(),
        &strings(&["launch", "codex", "--dry-run", "--port", &port]),
    )
    .await;
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let text = stdout(&out);
    assert!(text.contains("MRUN_API_KEY=mrun"));
    assert!(text.contains("codex -c"), "{text}");
    assert!(text.contains("model_providers.mrun.wire_api"), "{text}");

    // A missing client binary is reported with its install command.
    let out = mrun(
        home.path(),
        &strings(&["launch", "opencode", "--port", &port]),
    )
    .await;
    assert_eq!(out.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&out.stderr).contains("E020"));

    // A model other than the served one: usage error with a hint.
    let out = mrun(
        home.path(),
        &strings(&[
            "launch",
            "claude-code",
            "--model",
            "gpt-oss",
            "--dry-run",
            "--port",
            &port,
        ]),
    )
    .await;
    assert_eq!(out.status.code(), Some(2));
    server.shutdown().await.unwrap();
}

#[tokio::test]
async fn small_context_warns() {
    let server = server(40_960).await;
    let port = server.addr.port().to_string();
    let home = tempfile::tempdir().unwrap();
    let out = mrun(
        home.path(),
        &strings(&["launch", "claude-code", "--dry-run", "--port", &port]),
    )
    .await;
    assert!(String::from_utf8_lossy(&out.stderr).contains("warning: only 40960 tokens"));
    server.shutdown().await.unwrap();
}

#[tokio::test]
async fn ps_shows_the_model_and_cache() {
    let server = server(105_000).await;
    let port = server.addr.port().to_string();
    let home = tempfile::tempdir().unwrap();
    let out = mrun(home.path(), &strings(&["ps", "--port", &port])).await;
    assert!(out.status.success());
    let text = stdout(&out);
    assert!(
        text.contains(
            "model     mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit (qwen3, qwen_xml)"
        ),
        "{text}"
    );
    assert!(text.contains("active    idle"));
    assert!(text.contains("cache     0 live tokens"));
    let out = mrun(home.path(), &strings(&["ps", "--json", "--port", &port])).await;
    let json: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(json["schema_version"], 1);
    server.shutdown().await.unwrap();
}

#[tokio::test]
async fn commands_without_a_server_exit_5() {
    let home = tempfile::tempdir().unwrap();
    // A port nothing listens on.
    let port = std::net::TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port()
        .to_string();
    for args in [
        strings(&["ps", "--port", &port]),
        strings(&["connect", "codex", "--port", &port]),
        strings(&["stop", "--port", &port]),
        strings(&["launch", "claude-code", "--dry-run", "--port", &port]),
    ] {
        let out = mrun(home.path(), &args).await;
        assert_eq!(
            out.status.code(),
            Some(5),
            "{args:?}: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        assert!(String::from_utf8_lossy(&out.stderr).contains("error[E010]"));
    }
}

#[tokio::test]
async fn detach_reports_a_failed_load_with_its_log() {
    // This build has no inference engine: the detached server fails to load and detach says so.
    let home = tempfile::tempdir().unwrap();
    let port = std::net::TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port()
        .to_string();
    let out = mrun(
        home.path(),
        &strings(&["serve", "--detach", "--port", &port]),
    )
    .await;
    assert_eq!(out.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&out.stderr).contains("no inference engine"));
    assert!(home.path().join("server.log").exists());
    assert!(!home.path().join("server.pid").exists());
}

#[tokio::test]
async fn doctor_and_completions() {
    let home = tempfile::tempdir().unwrap();
    let out = mrun(home.path(), &strings(&["doctor", "--port", "1"])).await;
    let text = stdout(&out);
    assert!(
        text.contains("platform") && text.contains("claude-code"),
        "{text}"
    );
    if !cfg!(all(target_os = "macos", target_arch = "aarch64")) {
        assert_eq!(
            out.status.code(),
            Some(8),
            "not an Apple Silicon Mac: {text}"
        );
    }
    let out = mrun(home.path(), &strings(&["completions", "bash"])).await;
    assert!(out.status.success());
    assert!(stdout(&out).contains("mrun"));
}

#[tokio::test]
async fn cache_clear_reports_and_quiet_silences_info() {
    let server = server(105_000).await;
    let port = server.addr.port().to_string();
    let home = tempfile::tempdir().unwrap();
    let out = mrun(home.path(), &strings(&["cache", "clear", "--port", &port])).await;
    assert!(out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("cleared"));
    let out = mrun(
        home.path(),
        &strings(&["-q", "cache", "clear", "--port", &port]),
    )
    .await;
    assert!(out.status.success());
    assert!(
        out.stderr.is_empty(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    server.shutdown().await.unwrap();
}
