//! Speculative decoding through the server (006 FR-007, FR-008): a reply that copies its prompt
//! is the same with `--speculation auto` and `off`; with `auto` the request's record,
//! `/mrun/status` and `/metrics` count what was drafted and kept, with `off` nothing is drafted.
//! The engine is `ScriptedEngine` with a verify (its verify is serial decoding; the GPU verify
//! pass is checked against it in `mrun-engine`'s `tests/verify.rs`).

use mrun::chat::template::ChatTemplate;
use mrun::chat::tokenizer::test_support::byte_tokenizer;
use mrun::cli::serve::{self, EngineFactory, Loaded, RunningServer, ServeArgs, Speculation};
use mrun::runtime::handle::{DEFAULT_STALL_TIMEOUT, EngineHandle};
use mrun::runtime::scripted::ScriptedEngine;
use mrun_engine::{Sampling, SyntaxId};
use serde_json::{Value, json};
use std::sync::Arc;

const PROMPT: &str = "the quick brown fox jumps over the lazy dog. the quick";
const REPLY: &str = " brown fox jumps over the lazy dog.";

fn args(speculation: Speculation) -> ServeArgs {
    use clap::Parser;
    #[derive(Parser)]
    struct Wrap {
        #[command(flatten)]
        serve: ServeArgs,
    }
    let mut serve = Wrap::parse_from(["mrun"]).serve;
    serve.port = 0;
    serve.speculation = speculation;
    serve
}

fn copying() -> EngineFactory {
    Box::new(|args: &ServeArgs| {
        let tokenizer = Arc::new(byte_tokenizer());
        let script = tokenizer.encode(REPLY).unwrap();
        let handle = EngineHandle::spawn(
            move || {
                let mut engine = ScriptedEngine::new(script, 4096);
                engine.max_draft = 7;
                Ok(engine)
            },
            tokenizer.clone(),
            DEFAULT_STALL_TIMEOUT,
        )?;
        handle.set_speculation(args.speculation.on());
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

async fn reply(server: &RunningServer) -> String {
    let body: Value = reqwest::Client::new()
        .post(format!("http://{}/v1/messages", server.addr))
        .json(&json!({"model": "scripted", "max_tokens": 100,
            "messages": [{"role": "user", "content": PROMPT}]}))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    body["content"][0]["text"].as_str().unwrap().to_string()
}

async fn get(server: &RunningServer, path: &str) -> String {
    reqwest::get(format!("http://{}{path}", server.addr))
        .await
        .unwrap()
        .text()
        .await
        .unwrap()
}

#[tokio::test]
async fn a_copying_reply_is_the_same_and_counted() {
    let mut replies = Vec::new();
    for speculation in [Speculation::Off, Speculation::Auto] {
        let mut server = serve::start(args(speculation), copying()).await.unwrap();
        server.loaded().await.unwrap();
        replies.push(reply(&server).await);
        let status: Value = serde_json::from_str(&get(&server, "/mrun/status").await).unwrap();
        let totals = &status["speculation"];
        let finished = &status["finished"][0]["speculation"];
        let metrics = get(&server, "/metrics").await;
        match speculation {
            Speculation::Off => {
                assert_eq!(totals["mode"], "off");
                assert_eq!(totals["drafted"], 0);
                assert!(finished.is_null(), "{status}");
                assert!(metrics.contains("mrun_speculation_drafted_tokens_total 0"));
            }
            Speculation::Auto => {
                assert_eq!(totals["mode"], "auto");
                let drafted = totals["drafted"].as_u64().unwrap();
                let accepted = totals["accepted"].as_u64().unwrap();
                assert!(drafted > 0 && accepted > 0, "{totals}");
                assert_eq!(finished["drafted"], drafted, "{status}");
                assert!(finished["passes"].as_u64().unwrap() > 0);
                assert!(
                    metrics.contains(&format!(
                        "mrun_speculation_accepted_tokens_total {accepted}"
                    )),
                    "{metrics}"
                );
            }
        }
        server.shutdown().await.unwrap();
    }
    assert_eq!(replies[0], REPLY);
    assert_eq!(replies[1], replies[0]);
}
