//! A pool carries on (026 US4, T024): two Macs as two servers on loopback homes serve a tiny
//! real model across the ring (on the CPU). When the member leaves, the host serves the model
//! again on its own, with the same tokens; when the member is lost and comes back, Rejoin needs
//! no code and Re-plan serves on both Macs again, with the same tokens.

#![cfg(feature = "engine")]

use mrun::api::run::Timing;
use mrun::chat::tokenizer::test_support::byte_tokenizer;
use mrun::cli::serve::{self, RunningServer, ServeArgs};
use mrun::keychain::Vault;
use mrun::pool::PoolConfig;
use serde_json::{Value, json};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

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

fn free_port() -> u16 {
    std::net::TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port()
}

/// A tiny `qwen3` with the byte tokenizer and the Qwen3 template: a model `open` accepts.
fn tiny_model(dir: &Path) {
    mrun_engine::testing::write_tiny_qwen3(dir, 512).unwrap();
    std::fs::write(
        dir.join("chat_template.jinja"),
        include_str!("../src/chat/testdata/qwen3.jinja"),
    )
    .unwrap();
    byte_tokenizer()
        .inner()
        .save(dir.join("tokenizer.json"), false)
        .unwrap();
}

/// A Mac with no model of its own (its server reports a failed load) and a pool port.
struct Mac {
    server: RunningServer,
    home: PathBuf,
    port: u16,
}

impl Mac {
    async fn new(name: &str, home: &Path, port: u16) -> Self {
        let config = PoolConfig {
            home: home.to_path_buf(),
            vault: Vault::dir(home.join("keychain")),
            port,
            bind: "127.0.0.1".parse().unwrap(),
            announce: false,
            name: name.into(),
            chip: "Apple M4 Pro".into(),
            memory_bytes: 48 << 30,
        };
        let args = args();
        let timing = Timing::from_args(&args);
        let none: serve::EngineFactory = Box::new(|_| {
            Err(mrun::errors::CliError::new(
                mrun::errors::Code::Internal,
                "no model of its own",
            ))
        });
        let server = serve::start_with_pool(args, none, timing, config)
            .await
            .unwrap();
        Self {
            server,
            home: home.to_path_buf(),
            port,
        }
    }

    async fn get(&self, path: &str) -> Value {
        reqwest::get(format!("http://{}{path}", self.server.addr))
            .await
            .unwrap()
            .json()
            .await
            .unwrap()
    }

    async fn post(&self, path: &str, body: Value) -> (u16, Value) {
        let response = reqwest::Client::new()
            .post(format!("http://{}{path}", self.server.addr))
            .json(&body)
            .timeout(Duration::from_secs(120))
            .send()
            .await
            .unwrap();
        let status = response.status().as_u16();
        (status, response.json().await.unwrap_or_default())
    }

    fn pool_address(&self) -> String {
        format!("127.0.0.1:{}", self.port)
    }

    /// Greedy reply to the same prompt, once the server's model serves.
    async fn reply(&self, model: &str) -> String {
        let start = Instant::now();
        loop {
            let (status, body) = self
                .post(
                    "/v1/chat/completions",
                    json!({"model": model, "max_tokens": 6, "temperature": 0,
                        "messages": [{"role": "user", "content": "hello pool"}]}),
                )
                .await;
            if status == 200 {
                return body["choices"][0]["message"]["content"]
                    .as_str()
                    .unwrap_or_default()
                    .to_string();
            }
            assert!(
                start.elapsed() < Duration::from_secs(60),
                "no reply: {status} {body}"
            );
            tokio::time::sleep(Duration::from_millis(200)).await;
        }
    }

    /// Waits until the host's pool serves on `macs` Macs.
    async fn serving_on(&self, macs: usize) {
        let start = Instant::now();
        loop {
            let pool = self.get("/mrun/pool").await;
            let stages = pool["pool"]["plan"]["stages"]
                .as_array()
                .map_or(0, Vec::len);
            if pool["pool"]["replanning"] == false && stages == macs {
                return;
            }
            assert!(
                start.elapsed() < Duration::from_secs(60),
                "not serving on {macs} Macs: {pool}"
            );
            tokio::time::sleep(Duration::from_millis(200)).await;
        }
    }
}

/// A Mac on a runtime of its own: dropping the runtime takes its whole mrun away at once (lid
/// closed, cable pulled).
fn on_its_own(name: &'static str, home: PathBuf, port: u16) -> (Mac, std::sync::mpsc::Sender<()>) {
    let (tx, rx) = std::sync::mpsc::channel::<Mac>();
    let (stop, stopped) = std::sync::mpsc::channel::<()>();
    std::thread::spawn(move || {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .unwrap();
        let mac = runtime.block_on(Mac::new(name, &home, port));
        tx.send(mac).unwrap();
        let _ = stopped.recv();
        runtime.shutdown_background();
    });
    (rx.recv().unwrap(), stop)
}

async fn join(host: &Mac, who: &Mac) -> String {
    let (status, asked) = who
        .post("/mrun/pool/join", json!({"address": host.pool_address()}))
        .await;
    assert_eq!(status, 200, "{asked}");
    let id = host.get("/mrun/pool").await["pool"]["waiting"][0]["mac_id"]
        .as_str()
        .unwrap()
        .to_string();
    let member = who.post("/mrun/pool/join/confirm", json!({"yes": true}));
    let admit = async {
        tokio::time::sleep(Duration::from_millis(100)).await;
        host.post("/mrun/pool/admit", json!({"mac_id": id, "yes": true}))
            .await
    };
    let ((status, joined), _) = tokio::join!(member, admit);
    assert_eq!(status, 200, "{joined}");
    id
}

#[tokio::test(flavor = "multi_thread")]
async fn the_pool_serves_again_when_a_mac_leaves_or_comes_back() {
    // The servers read MRUN_HOME and the CPU switch from the environment: run in a child that
    // has them.
    if std::env::var_os("MRUN_POOL_CPU").is_none() {
        let home = tempfile::tempdir().unwrap();
        let status = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "the_pool_serves_again_when_a_mac_leaves_or_comes_back",
                "--exact",
                "--nocapture",
            ])
            .env("MRUN_POOL_CPU", "1")
            .env("MRUN_HOME", home.path())
            .status()
            .unwrap();
        assert!(status.success());
        return;
    }
    let model_dir = tempfile::tempdir().unwrap();
    tiny_model(model_dir.path());
    let model = model_dir.path().display().to_string();
    let (ben_home, anna_home) = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());
    let ben = Mac::new("Ben's MacBook Pro", ben_home.path(), free_port()).await;
    let anna_port = free_port();
    let (anna, vanish) = on_its_own(
        "Anna's MacBook Pro",
        anna_home.path().to_path_buf(),
        anna_port,
    );

    ben.post("/mrun/pool/start", json!({})).await;
    join(&ben, &anna).await;
    let (status, served) = ben.post("/mrun/pool/serve", json!({"model": model})).await;
    assert_eq!(status, 200, "{served}");
    ben.serving_on(2).await;
    let both = ben.reply(&model).await;
    assert!(!both.is_empty());

    // Anna leaves: Ben serves the model alone, with the same tokens.
    let (status, _) = anna.post("/mrun/pool/leave", json!({})).await;
    assert_eq!(status, 200);
    ben.serving_on(1).await;
    assert_eq!(ben.reply(&model).await, both, "the same tokens on one Mac");

    // Anna joins again, and her mrun goes away without a word: lost, and Ben serves alone.
    let anna_id = join(&ben, &anna).await;
    ben.post("/mrun/pool/replan", json!({})).await;
    ben.serving_on(2).await;
    vanish.send(()).unwrap();
    let anna_home_path = anna.home.clone();
    std::mem::forget(anna); // its runtime is gone; nothing to shut down
    let start = Instant::now();
    loop {
        let pool = ben.get("/mrun/pool").await;
        let lost = pool["pool"]["members"][0]["state"] == "lost";
        let stages = pool["pool"]["plan"]["stages"]
            .as_array()
            .map_or(0, Vec::len);
        if lost && stages == 1 && pool["pool"]["replanning"] == false {
            break;
        }
        assert!(start.elapsed() < Duration::from_secs(30), "{pool}");
        tokio::time::sleep(Duration::from_millis(200)).await;
    }
    assert_eq!(ben.reply(&model).await, both);

    // Anna's mrun is back: Rejoin needs no code, Re-plan serves on both again.
    tokio::time::sleep(Duration::from_millis(300)).await; // its pool port is free again
    let anna = Mac::new("Anna's MacBook Pro", &anna_home_path, anna_port).await;
    assert_eq!(
        anna.get("/mrun/pool").await["rejoin"]["host"],
        "Ben's MacBook Pro"
    );
    let (status, back) = anna.post("/mrun/pool/rejoin", json!({})).await;
    assert_eq!(status, 200, "{back}");
    assert_eq!(back["pool"], "Ben's pool");
    let pool = ben.get("/mrun/pool").await;
    assert_eq!(pool["pool"]["members"][0]["mac_id"], anna_id.as_str());
    assert_eq!(pool["pool"]["members"][0]["state"], "joined");
    let (status, replanned) = ben.post("/mrun/pool/replan", json!({})).await;
    assert_eq!(status, 200, "{replanned}");
    ben.serving_on(2).await;
    assert_eq!(
        ben.reply(&model).await,
        both,
        "the same tokens on two Macs again"
    );

    // A Mac the pool does not know cannot rejoin.
    let stranger_home = tempfile::tempdir().unwrap();
    let stranger = Mac::new("Mallory's Mac", stranger_home.path(), free_port()).await;
    let (status, refused) = stranger.post("/mrun/pool/rejoin", json!({})).await;
    assert_eq!(status, 409, "{refused}");
    for mac in [ben, anna, stranger] {
        mac.server.shutdown().await.unwrap();
    }
}
