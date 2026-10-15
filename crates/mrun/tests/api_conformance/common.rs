//! Harness: a real server on an ephemeral port whose engine replays scripted replies, one per
//! request, through a Qwen3-style (Hermes) ChatML template.

#![allow(dead_code)]

use clap::Parser;
use mrun::api::run::Timing;
use mrun::chat::template::ChatTemplate;
use mrun::chat::tokenizer::test_support::byte_tokenizer;
use mrun::cli::serve::{self, EngineFactory, Loaded, RunningServer, ServeArgs};
use mrun::runtime::handle::{DEFAULT_STALL_TIMEOUT, EngineHandle};
use mrun::runtime::scripted::{SCRIPT_EOS, ScriptedEngine};
use mrun_engine::{Sampling, SyntaxId};
use serde_json::Value;
use std::sync::Arc;
use std::time::Duration;

/// Qwen3-style template: tools in the system turn, Hermes calls, merged tool results, thinking
/// switch in the generation prompt.
pub const TEMPLATE: &str = r#"{%- if tools %}<|im_start|>system
{% if messages[0].role == 'system' %}{{ messages[0].content }}

{% endif %}<tools>{% for t in tools %}
{{ t | tojson }}{% endfor %}
</tools><|im_end|>
{% elif messages[0].role == 'system' %}<|im_start|>system
{{ messages[0].content }}<|im_end|>
{% endif %}
{%- for m in messages %}
{%- if m.role == 'system' and loop.first %}
{%- elif m.role == 'user' or m.role == 'system' %}<|im_start|>{{ m.role }}
{{ m.content }}<|im_end|>
{% elif m.role == 'assistant' %}<|im_start|>assistant
{% if m.reasoning_content %}<think>
{{ m.reasoning_content }}
</think>

{% endif %}{{ m.content }}{% for c in m.tool_calls or [] %}
<tool_call>
{"name": "{{ c.function.name }}", "arguments": {% if c.function.arguments is string %}{{ c.function.arguments }}{% else %}{{ c.function.arguments | tojson }}{% endif %}}
</tool_call>{% endfor %}<|im_end|>
{% elif m.role == 'tool' %}{% if loop.first or loop.previtem.role != 'tool' %}<|im_start|>user{% endif %}{{ '\n' }}<tool_response>
{{ m.content }}
</tool_response>{% if loop.last or loop.nextitem.role != 'tool' %}<|im_end|>
{% endif %}{% endif %}
{%- endfor %}
{%- if add_generation_prompt %}<|im_start|>assistant
{% if enable_thinking %}<think>
{% else %}<think>

</think>

{% endif %}{% endif %}"#;

pub struct Setup {
    pub capacity: u32,
    pub queue: usize,
    pub timing: Timing,
    /// Delay per generated token.
    pub step_delay: Option<Duration>,
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
    let tokenizer = Arc::new(byte_tokenizer());
    let mut script = Vec::new();
    for reply in replies {
        script.extend(tokenizer.encode(reply).unwrap());
        script.push(SCRIPT_EOS);
    }
    let capacity = setup.capacity;
    let step_delay = setup.step_delay;
    let factory: EngineFactory = Box::new(move |_| {
        let handle = EngineHandle::spawn(
            move || {
                let mut engine = ScriptedEngine::new(script, capacity);
                engine.step_delay = step_delay;
                Ok(engine)
            },
            tokenizer.clone(),
            DEFAULT_STALL_TIMEOUT,
        )?;
        Ok(Loaded {
            model_id: "scripted-model".into(),
            handle,
            tokenizer,
            template: Arc::new(ChatTemplate::new(TEMPLATE.into()).unwrap()),
            syntax: SyntaxId::Hermes,
            supports_thinking: true,
            defaults: Sampling::GREEDY,
        })
    });
    let mut inner = serve::start_with(args(setup.queue), factory, setup.timing)
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
