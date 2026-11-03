//! FR-024: every command except `pull` works with the network denied. The hub endpoint and
//! every proxy point at a closed port, a model is pulled into the cache beforehand, and a
//! server runs in-process; each command must succeed (and `pull` must fail fast with E050).

use mrun::chat::template::ChatTemplate;
use mrun::chat::tokenizer::test_support::byte_tokenizer;
use mrun::cli::serve::{self, EngineFactory, Loaded, ServeArgs};
use mrun::hub::{Paths, test_support};
use mrun::runtime::handle::{DEFAULT_STALL_TIMEOUT, EngineHandle};
use mrun::runtime::scripted::ScriptedEngine;
use mrun_engine::{Sampling, SyntaxId};
use std::path::Path;
use std::process::{Command, Output};
use std::sync::Arc;

/// A port nothing listens on: every network attempt fails at once.
fn closed_port() -> u16 {
    std::net::TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port()
}

async fn mrun(home: &Path, dead: u16, args: &[&str]) -> Output {
    let home = home.to_path_buf();
    let args: Vec<String> = args.iter().map(|s| s.to_string()).collect();
    let nowhere = format!("http://127.0.0.1:{dead}");
    tokio::task::spawn_blocking(move || {
        Command::new(env!("CARGO_BIN_EXE_mrun"))
            .args(&args)
            .env("HOME", &home)
            .env("MRUN_HOME", home.join(".mrun"))
            .env("PATH", "/usr/bin:/bin:/usr/sbin:/sbin")
            .env("HF_ENDPOINT", &nowhere)
            .env("HTTP_PROXY", &nowhere)
            .env("HTTPS_PROXY", &nowhere)
            .env("ALL_PROXY", &nowhere)
            .env("NO_PROXY", "")
            .env_remove("MRUN_PORT")
            .env_remove("HF_HOME")
            .env_remove("HF_HUB_CACHE")
            .env_remove("XDG_CACHE_HOME")
            .env_remove("CLAUDE_CONFIG_DIR")
            .env_remove("CODEX_HOME")
            .output()
            .unwrap()
    })
    .await
    .unwrap()
}

async fn server() -> serve::RunningServer {
    use clap::Parser;
    #[derive(Parser)]
    struct Wrap {
        #[command(flatten)]
        serve: ServeArgs,
    }
    let mut args = Wrap::parse_from(["mrun"]).serve;
    args.port = 0;
    let factory: EngineFactory = Box::new(move |args| {
        let tokenizer = Arc::new(byte_tokenizer());
        let handle = EngineHandle::spawn_with(
            move || Ok(ScriptedEngine::new(Vec::new(), 105_000)),
            tokenizer.clone(),
            DEFAULT_STALL_TIMEOUT,
            serve::cache_policy(args),
        )?;
        Ok(Loaded {
            model_id: "org/tiny".into(),
            handle,
            tokenizer,
            template: Arc::new(ChatTemplate::new("{{ messages[0].content }}".into()).unwrap()),
            syntax: SyntaxId::QwenXml,
            supports_thinking: false,
            defaults: Sampling::GREEDY,
            grammar: None,
        })
    });
    let mut server = serve::start(args, factory).await.unwrap();
    server.loaded().await.unwrap();
    server
}

#[tokio::test]
async fn every_command_but_pull_works_without_a_network() {
    let home = tempfile::tempdir().unwrap();
    let root = home.path().to_str().unwrap().to_string();
    let paths = Paths::from_vars(&|key: &str| match key {
        "HOME" => Some(root.clone()),
        "MRUN_HOME" => Some(format!("{root}/.mrun")),
        _ => None,
    });
    test_support::cached_model(
        &paths,
        None,
        "org/tiny",
        &"a".repeat(40),
        &[("config.json", b"{}"), ("model.safetensors", b"weights")],
    );
    std::fs::create_dir_all(&paths.mrun_home).unwrap();
    std::fs::write(paths.mrun_home.join("server.log"), "mrun listening\n").unwrap();
    let dead = closed_port();
    let server = server().await;
    let port = server.addr.port().to_string();
    let idle = closed_port().to_string();

    let commands: Vec<Vec<&str>> = vec![
        vec![],
        vec!["--version"],
        vec!["--help"],
        vec!["ls"],
        vec!["ls", "--json"],
        vec!["show", "org/tiny"],
        vec!["verify", "org/tiny"],
        vec!["ps", "--port", &port],
        vec!["ps", "--json", "--port", &port],
        vec!["connect", "codex", "--port", &port],
        vec!["connect", "claude-code", "--port", &idle],
        vec!["launch", "codex", "--dry-run", "--port", &port],
        vec!["cache", "clear", "--port", &port],
        vec!["logs"],
        vec!["stop", "--port", &idle],
        vec!["completions", "zsh"],
        vec!["rm", "org/tiny", "--yes"],
    ];
    for args in &commands {
        let out = mrun(home.path(), dead, args).await;
        assert!(
            out.status.success(),
            "mrun {args:?} failed offline: {}",
            String::from_utf8_lossy(&out.stderr)
        );
    }
    // doctor: no check fails for lack of a network. Checks that fail for local reasons (the
    // platform on a non-Mac, a full disk on the test machine) are allowed.
    let out = mrun(home.path(), dead, &["doctor", "--port", &port]).await;
    let text = String::from_utf8_lossy(&out.stdout);
    const LOCAL: [&str; 2] = ["platform", "disk"];
    let failed: Vec<&str> = text
        .lines()
        .filter_map(|l| l.strip_prefix("FAIL"))
        .filter_map(|l| l.split_whitespace().next())
        .collect();
    assert!(
        failed.iter().all(|name| LOCAL.contains(name)),
        "doctor offline: {text}"
    );

    // pull needs the network and says so quickly (E050, exit 6).
    let started = std::time::Instant::now();
    let out = mrun(home.path(), dead, &["pull", "org/other", "--yes"]).await;
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(6), "{stderr}");
    assert!(stderr.contains("error[E050]"), "{stderr}");
    assert!(started.elapsed() < std::time::Duration::from_secs(30));
    server.shutdown().await.unwrap();
}
