//! Harness: a real server on an ephemeral port whose engine replays scripted replies, one per
//! request, through Qwen3's own (Hermes) chat template, or the real Qwen3.5 and Qwen3.6
//! templates (Qwen XML) where a case sets [`Setup::template`].

#![allow(dead_code)]

use clap::Parser;
use mrun::api::run::Timing;
use mrun::chat::constrain::GrammarEnv;
use mrun::chat::template::ChatTemplate;
use mrun::chat::tokenizer::test_support::byte_tokenizer;
use mrun::cli::serve::{self, EngineFactory, Loaded, RunningServer, ServeArgs};
use mrun::runtime::handle::{DEFAULT_STALL_TIMEOUT, EngineHandle};
use mrun::runtime::scripted::{KV_BYTES, SCRIPT_EOS, ScriptedEngine};
use mrun_engine::{Sampling, SyntaxId};
use serde_json::Value;
use std::sync::Arc;
use std::time::Duration;

/// Qwen3's chat template as shipped (Hermes calls, merged tool results, thinking switch).
pub const TEMPLATE: &str = include_str!("../../src/chat/testdata/qwen3.jinja");
/// The `qwen3_5` family's template (mlx-community/Qwen3.5-9B-4bit): Qwen XML calls.
pub const QWEN3_5_TEMPLATE: &str = include_str!("../../src/chat/testdata/qwen3_5.jinja");
/// The `qwen3_5_moe` family's template (mlx-community/Qwen3.6-35B-A3B-4bit): Qwen XML calls,
/// `preserve_thinking`.
pub const QWEN3_6_TEMPLATE: &str = include_str!("../../src/chat/testdata/qwen3_6.jinja");

pub struct Setup {
    pub capacity: u32,
    pub queue: usize,
    pub timing: Timing,
    /// Delay per generated token.
    pub step_delay: Option<Duration>,
    /// Delay per layer of a prefill chunk (four layers per chunk).
    pub layer_delay: Option<Duration>,
    /// `--no-cache`.
    pub no_cache: bool,
    /// A hybrid model (linear-attention layers): resumes from checkpoints.
    pub hybrid: bool,
    /// Sequence-state memory in tokens' worth of bytes, below `capacity` to run out of memory.
    pub state_tokens: Option<u32>,
    /// The model's chat template and its tool-call syntax.
    pub template: &'static str,
    pub syntax: SyntaxId,
}

impl Default for Setup {
    fn default() -> Self {
        Self {
            capacity: 8192,
            queue: 16,
            timing: Timing {
                keepalive: Duration::from_secs(10),
                queue_wait: Duration::from_secs(600),
            },
            step_delay: None,
            layer_delay: None,
            no_cache: false,
            hybrid: false,
            state_tokens: None,
            template: TEMPLATE,
            syntax: SyntaxId::Hermes,
        }
    }
}

pub struct Server {
    pub inner: RunningServer,
    pub client: reqwest::Client,
}

pub fn args(queue: usize) -> ServeArgs {
    #[derive(Parser)]
    struct Wrap {
        #[command(flatten)]
        serve: ServeArgs,
    }
    let mut args = Wrap::parse_from(["mrun"]).serve;
    args.port = 0;
    args.queue = queue;
    args
}

/// Each reply is the text the model writes for one request, in order.
pub async fn start(replies: &[&str]) -> Server {
    start_with(replies, Setup::default()).await
}

pub async fn start_with(replies: &[&str], setup: Setup) -> Server {
    let tokenizer = byte_tokenizer();
    let replies = replies
        .iter()
        .map(|r| tokenizer.encode(r).unwrap())
        .collect();
    start_tokens(replies, setup).await
}

/// Replies as token ids: what the engine samples; grammar-forced tokens are not in them.
pub async fn start_tokens(replies: Vec<Vec<u32>>, setup: Setup) -> Server {
    let tokenizer = Arc::new(byte_tokenizer());
    let mut script = Vec::new();
    for reply in replies {
        script.extend(reply);
        script.push(SCRIPT_EOS);
    }
    let capacity = setup.capacity;
    let step_delay = setup.step_delay;
    let layer_delay = setup.layer_delay;
    let hybrid = setup.hybrid;
    let state_tokens = setup.state_tokens;
    let (template, syntax) = (setup.template, setup.syntax);
    let factory: EngineFactory = Box::new(move |args| {
        let handle = EngineHandle::spawn_with(
            move || {
                let mut engine = if hybrid {
                    ScriptedEngine::new_hybrid(script, capacity, 1_000)
                } else {
                    ScriptedEngine::new(script, capacity)
                };
                engine.step_delay = step_delay;
                engine.layer_delay = layer_delay;
                if let Some(tokens) = state_tokens {
                    engine = engine.with_state_capacity_bytes(u64::from(tokens) * KV_BYTES);
                }
                Ok(engine)
            },
            tokenizer.clone(),
            DEFAULT_STALL_TIMEOUT,
            serve::cache_policy(args),
        )?;
        let grammar = Some(Arc::new(GrammarEnv::new(&tokenizer, &[]).unwrap()));
        Ok(Loaded {
            model_id: "scripted-model".into(),
            handle,
            tokenizer,
            template: Arc::new(ChatTemplate::new(template.into()).unwrap()),
            syntax,
            supports_thinking: true,
            defaults: Sampling::GREEDY,
            grammar,
        })
    });
    let mut serve_args = args(setup.queue);
    serve_args.no_cache = setup.no_cache;
    let mut inner = serve::start_with(serve_args, factory, setup.timing)
        .await
        .unwrap();
    inner.loaded().await.unwrap();
    Server {
        inner,
        client: reqwest::Client::new(),
    }
}

pub struct Reply {
    pub status: u16,
    pub headers: reqwest::header::HeaderMap,
    pub text: String,
}

impl Reply {
    pub fn json(&self) -> Value {
        serde_json::from_str(&self.text).unwrap_or_else(|e| panic!("{e}: {}", self.text))
    }

    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers.get(name).and_then(|v| v.to_str().ok())
    }

    /// SSE frames as `(event name, data)`; comments come as `(":", text)`.
    pub fn frames(&self) -> Vec<(String, String)> {
        let mut frames = Vec::new();
        for block in self.text.split("\n\n").filter(|b| !b.is_empty()) {
            let mut name = String::new();
            let mut data = String::new();
            for line in block.lines() {
                if let Some(c) = line.strip_prefix(": ") {
                    frames.push((":".to_string(), c.to_string()));
                } else if let Some(n) = line.strip_prefix("event: ") {
                    name = n.to_string();
                } else if let Some(d) = line.strip_prefix("data: ") {
                    data = d.to_string();
                }
            }
            if !data.is_empty() {
                frames.push((name, data));
            }
        }
        frames
    }

    /// Data frames parsed as JSON (skips `[DONE]` and comments).
    pub fn events(&self) -> Vec<(String, Value)> {
        self.frames()
            .into_iter()
            .filter(|(name, data)| name != ":" && data != "[DONE]")
            .map(|(name, data)| (name, serde_json::from_str(&data).unwrap()))
            .collect()
    }
}

impl Server {
    pub fn url(&self, path: &str) -> String {
        format!("http://{}{path}", self.inner.addr)
    }

    pub async fn post(&self, path: &str, body: &Value) -> Reply {
        self.post_with(path, body, &[]).await
    }

    pub async fn post_with(&self, path: &str, body: &Value, headers: &[(&str, &str)]) -> Reply {
        let mut request = self.client.post(self.url(path)).json(body);
        for (k, v) in headers {
            request = request.header(*k, *v);
        }
        let response = request.send().await.unwrap();
        let status = response.status().as_u16();
        let headers = response.headers().clone();
        let text = response.text().await.unwrap();
        Reply {
            status,
            headers,
            text,
        }
    }
}

/// The grammar mrun plans for a Chat Completions `body` on this harness's model.
pub fn plan(body: &Value) -> mrun::chat::constrain::Plan {
    let tokenizer = byte_tokenizer();
    let env = GrammarEnv::new(&tokenizer, &[]).unwrap();
    let request =
        mrun::api::chat_completions::decode(body, mrun::api::decode::DecodeOptions::default())
            .unwrap();
    mrun::chat::constrain::plan(&request, SyntaxId::Hermes, &env)
}
