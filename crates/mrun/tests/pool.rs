//! Mac pool (026 US1): three Macs as three servers on loopback homes. A member joins with the
//! code both show, a stranger is refused before any body, a declined join stores nothing, and
//! Leave and Remove drop the membership on both sides. Nothing listens on the pool port before a
//! pool starts (SC-007).

use mrun::api::run::Timing;
use mrun::chat::template::ChatTemplate;
use mrun::chat::tokenizer::test_support::byte_tokenizer;
use mrun::cli::serve::{self, EngineFactory, Loaded, RunningServer, ServeArgs};
use mrun::keychain::Vault;
use mrun::pool::PoolConfig;
use mrun::runtime::handle::{DEFAULT_STALL_TIMEOUT, EngineHandle};
use mrun::runtime::scripted::ScriptedEngine;
use mrun_engine::{Sampling, SyntaxId};
use serde_json::{Value, json};
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
            supports_thinking: false,
            defaults: Sampling::GREEDY,
            grammar: None,
        })
    })
}

fn free_port() -> u16 {
    std::net::TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port()
}

struct Mac {
    server: RunningServer,
    _home: tempfile::TempDir,
    port: u16,
}

impl Mac {
    async fn new(name: &str) -> Self {
        let home = tempfile::tempdir().unwrap();
        let port = free_port();
        let config = PoolConfig {
            home: home.path().to_path_buf(),
            vault: Vault::dir(home.path().join("keychain")),
            port,
            bind: "127.0.0.1".parse().unwrap(),
            announce: false,
            name: name.into(),
            chip: "Apple M4 Pro".into(),
            memory_bytes: 48 << 30,
        };
        let args = args();
        let timing = Timing::from_args(&args);
        let server = serve::start_with_pool(args, scripted(), timing, config)
            .await
            .unwrap();
        Self {
            server,
            _home: home,
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
            .send()
            .await
            .unwrap();
        (response.status().as_u16(), response.json().await.unwrap())
    }

    fn pool_address(&self) -> String {
        format!("127.0.0.1:{}", self.port)
    }

    fn listening(&self) -> bool {
        std::net::TcpStream::connect(self.pool_address()).is_ok()
    }
}

/// `who` asks to join `host`'s pool and both say yes; returns `who`'s id.
async fn join(host: &Mac, who: &Mac) -> String {
    let (status, asked) = who
        .post("/mrun/pool/join", json!({"address": host.pool_address()}))
        .await;
    assert_eq!(status, 200, "{asked}");
    let code = asked["code"].as_str().unwrap().to_string();
    let waiting = &host.get("/mrun/pool").await["pool"]["waiting"][0];
    assert_eq!(
        waiting["code"],
        code.as_str(),
        "both Macs show the same code"
    );
    let id = waiting["mac_id"].as_str().unwrap().to_string();
    let member = who.post("/mrun/pool/join/confirm", json!({"yes": true}));
    let admit = async {
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        host.post("/mrun/pool/admit", json!({"mac_id": id, "yes": true}))
            .await
    };
    let ((status, joined), (admitted, _)) = tokio::join!(member, admit);
    assert_eq!((status, admitted), (200, 200), "{joined}");
    assert_eq!(joined["joined"], true);
    id
}

#[tokio::test]
async fn macs_join_with_the_code_and_strangers_are_refused() {
    let (ben, anna, mallory) = (
        Mac::new("Ben's MacBook Pro").await,
        Mac::new("Anna's MacBook Pro").await,
        Mac::new("Mallory's Mac").await,
    );
    // SC-007: no pool, nothing on the pool port.
    assert!(!ben.listening());
    let (status, started) = ben.post("/mrun/pool/start", json!({})).await;
    assert_eq!(status, 200, "{started}");
    assert_eq!(started["pool"]["name"], "Ben's pool");
    assert!(ben.listening());

    let anna_id = join(&ben, &anna).await;
    let host_view = ben.get("/mrun/pool").await;
    assert_eq!(
        host_view["pool"]["members"][0]["name"],
        "Anna's MacBook Pro"
    );
    assert_eq!(host_view["memberships"][0]["role"], "host");
    let member_view = anna.get("/mrun/pool").await;
    assert_eq!(member_view["pool"]["role"], "member");
    assert_eq!(member_view["pool"]["host"]["name"], "Ben's MacBook Pro");
    assert!(
        !serde_json::to_string(&host_view)
            .unwrap()
            .contains("secret"),
        "no secret in the status"
    );

    // A member's agents reach the host's model through the pool port; the host names the
    // member from its certificate, and keeps its sessions apart.
    let anna_vault = Vault::dir(anna._home.path().join("keychain"));
    let anna_identity = mrun::pool::identity::Identity::load_or_create(&anna_vault).unwrap();
    let host_id = member_view["pool"]["host"]["mac_id"].as_str().unwrap();
    let pin = mrun::pool::identity::Members::at(anna._home.path(), anna_vault.clone())
        .pin(host_id)
        .unwrap();
    let relay = mrun::pool::tls::client(&anna_identity, Some(pin)).unwrap();
    let answer = relay
        .post(format!(
            "https://{}/v1/chat/completions",
            ben.pool_address()
        ))
        .header("user-agent", "claude-cli/2.1.3")
        .json(&json!({"model": "scripted", "max_tokens": 1,
            "messages": [{"role": "user", "content": "hi"}]}))
        .send()
        .await
        .unwrap();
    assert_eq!(answer.status(), 200);
    let sessions = ben.get("/mrun/sessions").await;
    assert_eq!(
        sessions["requests"][0]["member"],
        anna_id.as_str(),
        "{sessions}"
    );
    assert_eq!(sessions["sessions"][0]["member"], anna_id.as_str());
    assert_eq!(sessions["sessions"][0]["label"], "Claude Code");
    let own: Value = relay
        .get(format!("https://{}/pool/v1/sessions", ben.pool_address()))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(
        own["requests"].as_array().unwrap().len(),
        1,
        "a member sees its own requests"
    );

    // A stranger's own certificate reaches no member route: refused before any body.
    let stranger = mrun::pool::identity::Identity::load_or_create(&Vault::dir(
        mallory._home.path().join("keychain"),
    ))
    .unwrap();
    let client = mrun::pool::tls::client(&stranger, None).unwrap();
    // A 403, or the connection closed while the client still writes the body.
    match client
        .post(format!("https://{}/pool/v1/bye", ben.pool_address()))
        .body(vec![b'x'; 1 << 20])
        .send()
        .await
    {
        Ok(refused) => assert_eq!(refused.status(), 403),
        Err(e) => assert!(e.is_request() || e.is_body(), "{e:?}"),
    }

    // A declined join stores nothing.
    let (_, asked) = mallory
        .post("/mrun/pool/join", json!({"address": ben.pool_address()}))
        .await;
    assert!(asked["code"].is_string());
    let waiting: Vec<Value> = ben.get("/mrun/pool").await["pool"]["waiting"]
        .as_array()
        .unwrap()
        .clone();
    let mallory_id = waiting[0]["mac_id"].as_str().unwrap().to_string();
    let confirm = mallory.post("/mrun/pool/join/confirm", json!({"yes": true}));
    let decline = async {
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        ben.post(
            "/mrun/pool/admit",
            json!({"mac_id": mallory_id, "yes": false}),
        )
        .await
    };
    let ((status, body), _) = tokio::join!(confirm, decline);
    assert_eq!(status, 403, "{body}");
    assert_eq!(body["error"]["code"], "E061");
    assert_eq!(
        ben.get("/mrun/pool").await["memberships"]
            .as_array()
            .unwrap()
            .len(),
        1
    );
    assert_eq!(mallory.get("/mrun/pool").await["memberships"], json!([]));

    // Leave: both Macs forget.
    let (status, _) = anna.post("/mrun/pool/leave", json!({})).await;
    assert_eq!(status, 200);
    assert_eq!(ben.get("/mrun/pool").await["pool"]["members"], json!([]));
    assert_eq!(ben.get("/mrun/pool").await["memberships"], json!([]));
    assert_eq!(anna.get("/mrun/pool").await["memberships"], json!([]));

    // Remove: both Macs forget too.
    let again = join(&ben, &anna).await;
    assert_eq!(again, anna_id, "the same Mac, the same id");
    let (status, _) = ben
        .post("/mrun/pool/remove", json!({"mac_id": anna_id}))
        .await;
    assert_eq!(status, 200);
    assert_eq!(anna.get("/mrun/pool").await["pool"], Value::Null);
    assert_eq!(anna.get("/mrun/pool").await["memberships"], json!([]));

    // Ending the pool closes the pool port.
    let (status, _) = ben.post("/mrun/pool/leave", json!({})).await;
    assert_eq!(status, 200);
    tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    assert!(!ben.listening());
    for mac in [ben, anna, mallory] {
        mac.server.shutdown().await.unwrap();
    }
}

/// A Mac whose whole mrun goes away at once (its runtime dropped): lid closed, cable pulled.
fn mac_on_its_own(name: &'static str) -> (Mac, std::sync::mpsc::Sender<()>) {
    let (tx, rx) = std::sync::mpsc::channel::<Mac>();
    let (stop, stopped) = std::sync::mpsc::channel::<()>();
    std::thread::spawn(move || {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .unwrap();
        let mac = runtime.block_on(Mac::new(name));
        tx.send(mac).unwrap();
        let _ = stopped.recv();
        runtime.shutdown_background();
    });
    (rx.recv().unwrap(), stop)
}

#[tokio::test(flavor = "multi_thread")]
async fn a_mac_that_vanishes_is_shown_lost_within_five_seconds() {
    let ben = Mac::new("Ben's MacBook Pro").await;
    let (anna, vanish) = mac_on_its_own("Anna's MacBook Pro");
    ben.post("/mrun/pool/start", json!({})).await;
    join(&ben, &anna).await;
    tokio::time::sleep(std::time::Duration::from_millis(1500)).await;
    assert_eq!(
        ben.get("/mrun/status").await["pool"]["members"][0]["state"],
        "joined"
    );
    let gone = std::time::Instant::now();
    vanish.send(()).unwrap();
    std::mem::forget(anna); // its runtime is gone; nothing to shut down
    loop {
        let status = ben.get("/mrun/status").await;
        if status["pool"]["members"][0]["state"] == "lost" {
            assert_eq!(
                status["pool"]["degraded"],
                "Anna's MacBook Pro stopped answering"
            );
            break;
        }
        assert!(
            gone.elapsed() < std::time::Duration::from_secs(5),
            "not lost within 5 s: {status}"
        );
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    }
    ben.server.shutdown().await.unwrap();
}

/// Constitution 9.0.0 IV: a host may start its pool unencrypted; the joining Mac learns it before
/// it confirms, and both Macs' status says it.
#[tokio::test]
async fn an_unencrypted_pool_says_so_before_anyone_joins() {
    let (ben, anna) = (
        Mac::new("Ben's MacBook Pro").await,
        Mac::new("Anna's MacBook Pro").await,
    );
    let (status, started) = ben
        .post("/mrun/pool/start", json!({"unencrypted": true}))
        .await;
    assert_eq!(status, 200, "{started}");
    assert_eq!(started["pool"]["unencrypted"], true);
    let (_, asked) = anna
        .post("/mrun/pool/join", json!({"address": ben.pool_address()}))
        .await;
    assert_eq!(
        asked["unencrypted"], true,
        "told before confirming: {asked}"
    );
    assert_eq!(anna.get("/mrun/pool").await["joining"]["unencrypted"], true);
    let id = ben.get("/mrun/pool").await["pool"]["waiting"][0]["mac_id"]
        .as_str()
        .unwrap()
        .to_string();
    let member = anna.post("/mrun/pool/join/confirm", json!({"yes": true}));
    let admit = async {
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        ben.post("/mrun/pool/admit", json!({"mac_id": id, "yes": true}))
            .await
    };
    let ((status, _), _) = tokio::join!(member, admit);
    assert_eq!(status, 200);
    assert_eq!(anna.get("/mrun/status").await["pool"]["unencrypted"], true);
    for mac in [ben, anna] {
        mac.server.shutdown().await.unwrap();
    }
}
