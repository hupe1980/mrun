//! The client-side commands of 004 against a server started in-process: `connect` (also
//! `--write`/`--restore`), `launch --dry-run`, `ps` (also `--watch`), `doctor`, `stop`, `logs`,
//! `serve --detach`, bare `mrun` and `completions`.

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
    let (open, server) = loading_server(capacity).await;
    open.send(()).unwrap();
    let mut server = server;
    server.loaded().await.unwrap();
    server
}

/// A server whose model loads once the returned sender sends (or is dropped); returned while it
/// loads.
async fn loading_server(capacity: u32) -> (std::sync::mpsc::Sender<()>, RunningServer) {
    let (open, gate) = std::sync::mpsc::channel::<()>();
    let factory: EngineFactory = Box::new(move |args| {
        let _ = gate.recv();
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
    (open, serve::start(args(), factory).await.unwrap())
}

/// Runs the binary off the async runtime with an isolated `HOME` and `MRUN_HOME`, and no client
/// binaries on `PATH`.
async fn mrun(home: &std::path::Path, args: &[String]) -> Output {
    let home = home.to_path_buf();
    let args = args.to_vec();
    tokio::task::spawn_blocking(move || {
        Command::new(env!("CARGO_BIN_EXE_mrun"))
            .args(&args)
            .env("HOME", &home)
            .env("MRUN_HOME", &home)
            .env("PATH", "/nonexistent")
            .env_remove("MRUN_PORT")
            .env_remove("HF_HOME")
            .env_remove("HF_HUB_CACHE")
            .env_remove("XDG_CACHE_HOME")
            .env_remove("CLAUDE_CONFIG_DIR")
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
    assert!(text.contains("claude --settings"), "{text}");
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
    assert!(!text.contains("MRUN_API_KEY"), "{text}");
    assert!(text.contains("codex -c"), "{text}");
    assert!(text.contains("model_providers.mrun.wire_api"), "{text}");

    // A missing client binary is reported with its install command.
    let out = mrun(
        home.path(),
        &strings(&["launch", "opencode", "--port", &port]),
    )
    .await;
    assert_eq!(out.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&out.stderr).contains("E020"));
    // No client named and none installed.
    let out = mrun(home.path(), &strings(&["launch", "--port", &port])).await;
    assert_eq!(out.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&out.stderr).contains("no supported coding agent"));

    // settings.json env that beats launch is reported.
    std::fs::create_dir_all(home.path().join(".claude")).unwrap();
    std::fs::write(
        home.path().join(".claude/settings.json"),
        r#"{"env": {"ANTHROPIC_BASE_URL": "https://proxy.example"}}"#,
    )
    .unwrap();
    let out = mrun(
        home.path(),
        &strings(&["launch", "claude-code", "--dry-run", "--port", &port]),
    )
    .await;
    assert!(out.status.success());
    assert!(
        String::from_utf8_lossy(&out.stderr).contains("sets ANTHROPIC_BASE_URL in \"env\""),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(stdout(&out).contains("CLAUDE_CODE_SUBAGENT_MODEL="));

    // A model other than the served one: usage error with a hint.
    let out = mrun(
        home.path(),
        &strings(&[
            "launch",
            "claude-code",
            "--model",
            "qwen3.5",
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
async fn each_client_is_refused_below_its_minimum_context() {
    let server = server(40_960).await;
    let port = server.addr.port().to_string();
    let home = tempfile::tempdir().unwrap();
    // Codex's minimum is 32K: 40,960 tokens launch.
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
    // Below Claude Code's and OpenCode's 64K minimum launch refuses (FR-027, D048).
    for client in ["claude-code", "opencode"] {
        let out = mrun(
            home.path(),
            &strings(&["launch", client, "--dry-run", "--port", &port]),
        )
        .await;
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert_eq!(out.status.code(), Some(4), "{client}: {stderr}");
        assert!(
            stderr.contains("error[E040]") && stderr.contains("needs at least 65536"),
            "{stderr}"
        );
    }
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
async fn commands_without_a_server() {
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
    // Nothing to stop is not an error.
    let out = mrun(home.path(), &strings(&["stop", "--port", &port])).await;
    assert_eq!(out.status.code(), Some(0));
    assert!(String::from_utf8_lossy(&out.stderr).contains("nothing to stop"));
    // connect prints this machine's defaults.
    let out = mrun(
        home.path(),
        &strings(&["connect", "codex", "--port", &port]),
    )
    .await;
    assert_eq!(
        out.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(String::from_utf8_lossy(&out.stderr).contains("no mrun server is running"));
    assert!(stdout(&out).contains("[model_providers.mrun]"));
    for client in ["crush", "zed"] {
        let out = mrun(home.path(), &strings(&["connect", client, "--port", &port])).await;
        assert_eq!(out.status.code(), Some(0), "{client}");
        let out = mrun(
            home.path(),
            &strings(&["connect", client, "--write", "--yes", "--port", &port]),
        )
        .await;
        assert_eq!(out.status.code(), Some(2), "{client} is print only");
    }
}

#[tokio::test]
async fn connect_write_and_restore() {
    let server = server(105_000).await;
    let port = server.addr.port().to_string();
    let home = tempfile::tempdir().unwrap();
    let out = mrun(
        home.path(),
        &strings(&[
            "connect",
            "claude-code",
            "--write",
            "--yes",
            "--port",
            &port,
        ]),
    )
    .await;
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "{stderr}");
    assert!(stderr.contains("verified: the server on port"), "{stderr}");
    let settings = home.path().join("claude/settings.json");
    let text = std::fs::read_to_string(&settings).unwrap();
    assert!(text.contains("\"idleCompaction\": false"), "{text}");
    let out = mrun(
        home.path(),
        &strings(&["connect", "claude-code", "--restore", "--yes"]),
    )
    .await;
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(!settings.exists());
    server.shutdown().await.unwrap();
}

#[tokio::test]
async fn ps_watch_streams_json_lines_without_a_terminal() {
    let server = server(105_000).await;
    let port = server.addr.port().to_string();
    let home = tempfile::tempdir().unwrap();
    let mut child = Command::new(env!("CARGO_BIN_EXE_mrun"))
        .args(["ps", "--watch", "--port", &port])
        .env("HOME", home.path())
        .env("MRUN_HOME", home.path())
        .stdout(std::process::Stdio::piped())
        .spawn()
        .unwrap();
    let stdout = child.stdout.take().unwrap();
    // Two polls arrive within a few seconds (one every 500 ms once the binary is up).
    let lines = tokio::task::spawn_blocking(move || {
        use std::io::BufRead;
        std::io::BufReader::new(stdout)
            .lines()
            .take(2)
            .map(Result::unwrap)
            .collect::<Vec<String>>()
    });
    let lines = tokio::time::timeout(std::time::Duration::from_secs(20), lines)
        .await
        .expect("two status lines")
        .unwrap();
    child.kill().unwrap();
    let _ = child.wait();
    assert_eq!(lines.len(), 2, "{lines:?}");
    let first: serde_json::Value = serde_json::from_str(&lines[0]).unwrap();
    assert_eq!(first["schema_version"], 1);
    server.shutdown().await.unwrap();
}

#[tokio::test]
async fn detach_reports_a_failed_load_with_its_log() {
    // The model does not exist (and builds without `mlx` have no engine): the detached server
    // fails to load and detach says so.
    let home = tempfile::tempdir().unwrap();
    let missing = home.path().join("no-such-model").display().to_string();
    let port = std::net::TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port()
        .to_string();
    let out = mrun(
        home.path(),
        &strings(&["serve", "--detach", "--port", &port, "--model", &missing]),
    )
    .await;
    let stderr = String::from_utf8_lossy(&out.stderr);
    // Without `mlx` there is no engine (E001, exit 1); with it the model is missing (E031, 3).
    let expected = if stderr.contains("no inference engine") {
        1
    } else {
        assert!(stderr.contains("error[E031]"), "{stderr}");
        3
    };
    assert_eq!(out.status.code(), Some(expected), "{stderr}");
    assert!(home.path().join("server.log").exists());
    assert!(!home.path().join(format!("server-{port}.pid")).exists());

    let out = mrun(home.path(), &strings(&["logs", "-n", "3"])).await;
    assert!(out.status.success());
    assert!(stdout(&out).lines().count() <= 3);
}

#[tokio::test]
async fn stop_never_signals_a_pid_that_is_not_the_server() {
    let home = tempfile::tempdir().unwrap();
    let port = std::net::TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port()
        .to_string();
    let pid_file = home.path().join(format!("server-{port}.pid"));
    std::fs::write(&pid_file, "1").unwrap();
    // Nothing answers on the port: pid 1 is never signalled, the stale file goes, and there is
    // nothing to stop (exit 0).
    let out = mrun(home.path(), &strings(&["stop", "--port", &port])).await;
    assert_eq!(out.status.code(), Some(0));
    assert!(String::from_utf8_lossy(&out.stderr).contains("stale pid file"));
    assert!(!pid_file.exists(), "the stale pid file is removed");
}

#[tokio::test]
async fn stop_reaches_a_foreground_server_but_checks_its_process() {
    // A server started without --detach has no pid file; `stop` takes the pid from
    // /mrun/status. Here that pid is this test binary, not an mrun executable, so it is refused
    // (the PID-reuse guard) and nothing is signalled.
    let server = server(32_768).await;
    let port = server.addr.port().to_string();
    let home = tempfile::tempdir().unwrap();
    let out = mrun(home.path(), &strings(&["stop", "--port", &port])).await;
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_ne!(out.status.code(), Some(0), "{stderr}");
    assert!(stderr.contains("is not an mrun server"), "{stderr}");
    assert!(!stderr.contains("--detach"), "{stderr}");
}

/// T017: `launch` waits for a loading server instead of starting another, and offers a restart
/// (never without consent) when another model is served.
#[tokio::test]
async fn launch_waits_for_a_loading_server_and_asks_before_a_restart() {
    let (open, server) = loading_server(105_000).await;
    let port = server.addr.port().to_string();
    let home = tempfile::tempdir().unwrap();
    // The model finishes loading while launch waits for it.
    std::thread::spawn(move || {
        std::thread::sleep(std::time::Duration::from_millis(1500));
        let _ = open.send(());
    });
    let out = mrun(
        home.path(),
        &strings(&["launch", "codex", "--dry-run", "--port", &port]),
    )
    .await;
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "{stderr}");
    assert!(
        stdout(&out).contains("model_provider=\"mrun\""),
        "{}",
        stdout(&out)
    );

    // Another model is served: --dry-run never restarts, it says how to.
    let out = mrun(
        home.path(),
        &strings(&[
            "launch",
            "codex",
            "--dry-run",
            "--model",
            "org/other",
            "--port",
            &port,
        ]),
    )
    .await;
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(2), "{stderr}");
    assert!(
        stderr
            .contains("serves 'mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit', not 'org/other'"),
        "{stderr}"
    );
    assert!(stderr.contains("'mrun stop'"), "{stderr}");

    // Without a terminal and without --yes the restart needs confirmation (E002); the server
    // keeps running.
    let out = mrun(
        home.path(),
        &strings(&["launch", "codex", "--model", "org/other", "--port", &port]),
    )
    .await;
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(2), "{stderr}");
    assert!(stderr.contains("error[E002]"), "{stderr}");
    let ps = mrun(home.path(), &strings(&["ps", "--json", "--port", &port])).await;
    assert!(ps.status.success());
    server.shutdown().await.unwrap();
}

/// T017: with no server, `launch` asks once before starting one; without a terminal it refuses
/// and names `--yes`, starting nothing.
#[tokio::test]
async fn launch_asks_before_starting_a_server() {
    let home = tempfile::tempdir().unwrap();
    let port = std::net::TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port()
        .to_string();
    let model = home.path().join("model");
    std::fs::create_dir_all(&model).unwrap();
    std::fs::write(model.join("config.json"), "{}").unwrap();
    let out = mrun(
        home.path(),
        &strings(&[
            "launch",
            "codex",
            "--model",
            model.to_str().unwrap(),
            "--port",
            &port,
        ]),
    )
    .await;
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_ne!(out.status.code(), Some(0), "{stderr}");
    assert!(stderr.contains("--yes"), "{stderr}");
    assert!(!home.path().join(format!("server-{port}.pid")).exists());
}

#[tokio::test]
async fn bare_mrun_prints_status_and_next_steps() {
    let home = tempfile::tempdir().unwrap();
    let port = std::net::TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port()
        .to_string();
    let out = tokio::task::spawn_blocking({
        let home = home.path().to_path_buf();
        move || {
            Command::new(env!("CARGO_BIN_EXE_mrun"))
                .env("HOME", &home)
                .env("MRUN_HOME", &home)
                .env("MRUN_PORT", &port)
                .env_remove("HF_HOME")
                .env_remove("HF_HUB_CACHE")
                .env_remove("XDG_CACHE_HOME")
                .output()
                .unwrap()
        }
    })
    .await
    .unwrap();
    assert!(out.status.success());
    let text = stdout(&out);
    assert!(text.contains("server  not running"), "{text}");
    assert!(text.contains("models  none pulled"), "{text}");
    assert!(text.contains("next:\n  mrun launch claude-code"), "{text}");
}

/// T020: bare `mrun` shows the loading time while loading and the context while serving.
#[tokio::test]
async fn bare_mrun_shows_loading_time_and_context() {
    let (open, mut server) = loading_server(105_000).await;
    let port = server.addr.port().to_string();
    let home = tempfile::tempdir().unwrap();
    let bare = |home: std::path::PathBuf, port: String| {
        tokio::task::spawn_blocking(move || {
            Command::new(env!("CARGO_BIN_EXE_mrun"))
                .env("HOME", &home)
                .env("MRUN_HOME", &home)
                .env("MRUN_PORT", &port)
                .env_remove("HF_HOME")
                .env_remove("HF_HUB_CACHE")
                .env_remove("XDG_CACHE_HOME")
                .output()
                .unwrap()
        })
    };
    let out = bare(home.path().to_path_buf(), port.clone()).await.unwrap();
    let text = stdout(&out);
    assert!(
        text.contains("server  loading") && text.contains(" s)"),
        "{text}"
    );
    open.send(()).unwrap();
    server.loaded().await.unwrap();
    let out = bare(home.path().to_path_buf(), port.clone()).await.unwrap();
    let text = stdout(&out);
    assert!(text.contains("(105000 tokens of context)"), "{text}");
    server.shutdown().await.unwrap();
}

/// T032: `--help` shows examples for bare `mrun`, `launch` without a client and `run` without a
/// model.
#[tokio::test]
async fn help_shows_the_short_forms() {
    let home = tempfile::tempdir().unwrap();
    for (args, example) in [
        (vec!["--help"], "  mrun                     the server"),
        (
            vec!["launch", "--help"],
            "  mrun launch              the only installed client",
        ),
        (
            vec!["run", "--help"],
            "  mrun run                 this Mac's default model",
        ),
        (vec!["stop", "--help"], "  mrun stop\n"),
    ] {
        let out = mrun(home.path(), &strings(&args)).await;
        assert!(stdout(&out).contains(example), "{args:?}: {}", stdout(&out));
    }
}

#[tokio::test]
async fn doctor_and_completions() {
    let home = tempfile::tempdir().unwrap();
    let out = mrun(home.path(), &strings(&["doctor", "--port", "1"])).await;
    let text = stdout(&out);
    assert!(
        text.contains("platform") && text.contains("info  clients"),
        "{text}"
    );
    assert!(text.contains("no model is pulled yet"), "{text}");
    if !cfg!(all(target_os = "macos", target_arch = "aarch64")) {
        assert_eq!(
            out.status.code(),
            Some(8),
            "not an Apple Silicon Mac: {text}"
        );
    }
    // The hardware report (D060): Markdown, exit 0, the home directory shown as `~`.
    let out = mrun(
        home.path(),
        &strings(&["doctor", "--report", "--port", "1"]),
    )
    .await;
    let text = stdout(&out);
    assert_eq!(out.status.code(), Some(0), "{text}");
    for needle in [
        "### mrun hardware report",
        "| mrun | `",
        "| Physical memory |",
        "#### Models\n\nnone pulled",
        "#### Calibration (`~/calibration.toml`)",
        "#### Doctor",
        "info clients",
    ] {
        assert!(text.contains(needle), "missing {needle:?} in {text}");
    }
    let home_text = home.path().to_string_lossy();
    assert!(!text.contains(home_text.as_ref()), "{text}");
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

/// FR-032: `mrun run` refuses a model whose chat template does not render its family's tool-call
/// syntax (here the Hermes-rendering Qwen3 template on a `qwen3_moe` config, which needs Qwen XML),
/// naming family, syntax and the missing marker, before any weights are read (exit 3).
#[tokio::test]
async fn run_refuses_a_template_without_the_family_syntax() {
    let home = tempfile::tempdir().unwrap();
    let dir = tempfile::tempdir().unwrap();
    std::fs::copy(
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../mrun-engine/tests/fixtures/tiny_qwen3_moe/config.json"
        ),
        dir.path().join("config.json"),
    )
    .unwrap();
    std::fs::write(
        dir.path().join("chat_template.jinja"),
        include_str!("../src/chat/testdata/qwen3.jinja"),
    )
    .unwrap();
    byte_tokenizer()
        .inner()
        .save(dir.path().join("tokenizer.json"), false)
        .unwrap();
    let model = dir.path().display().to_string();
    let output = mrun(home.path(), &strings(&["run", &model, "--prompt", "hi"])).await;
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert_eq!(output.status.code(), Some(3), "{stderr}");
    assert!(stderr.contains("E034"), "{stderr}");
    assert!(stderr.contains("qwen3_moe"), "{stderr}");
    assert!(stderr.contains("qwen_xml"), "{stderr}");
    assert!(stderr.contains("<function=mrun_probe>"), "{stderr}");
}
