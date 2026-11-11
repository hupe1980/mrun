//! Tools in mrun's own chats (028): the MCP servers of `~/.mrun/mcp.json`, started on first use
//! and kept for the process (the window or `mrun run`), each local one in a procage cage. The
//! server (`mrun serve`) never runs tools; mrun's own tools are `mrun mcp serve`.

pub mod cage;
pub mod client;
pub mod config;
pub mod serve;

use client::{Client, Outcome, Tool};
use serde_json::{Value, json};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::Mutex;

/// The model sees `<server>__<tool>`.
pub const SEPARATOR: &str = "__";

/// A started server: its tools (listed without waiting for a running call) and its client.
#[derive(Debug)]
struct Started {
    tools: Vec<Tool>,
    client: Mutex<Client>,
}

/// The started servers of one process. Locks are held only to look up or insert, never across a
/// start or a call; one server's calls run one after another on its own client lock.
#[derive(Debug)]
pub struct Registry {
    home: PathBuf,
    running: Mutex<HashMap<String, Arc<Started>>>,
}

impl Registry {
    pub fn new(home: PathBuf) -> Self {
        Registry {
            home,
            running: Mutex::new(HashMap::new()),
        }
    }

    pub fn home(&self) -> &std::path::Path {
        &self.home
    }

    /// Server `name`, started if it is not yet.
    async fn client(&self, name: &str) -> Result<Arc<Started>, String> {
        let running = self.running.lock().await.get(name).cloned();
        if let Some(c) = running {
            return Ok(c);
        }
        let (servers, _) = config::load(&self.home);
        let server = servers
            .iter()
            .find(|s| s.name == name)
            .ok_or_else(|| format!("no server {name} in mcp.json"))?;
        let client = Client::start(server).await?;
        let started = Arc::new(Started {
            tools: client.tools.clone(),
            client: Mutex::new(client),
        });
        // Two first uses at once: the first one started stays, the other is dropped (stopped).
        let mut running = self.running.lock().await;
        Ok(running.entry(name.into()).or_insert(started).clone())
    }

    /// Stops every server (the next use starts them again with the file as it is then).
    pub async fn stop(&self, name: Option<&str>) {
        let mut running = self.running.lock().await;
        match name {
            Some(n) => drop(running.remove(n)),
            None => running.clear(),
        }
    }

    /// Every server of `mcp.json` without secrets; with `start`, each one's tools or the error
    /// that stopped it (servers start in parallel).
    pub async fn list(&self, start: bool) -> Value {
        let (servers, errors) = config::load(&self.home);
        let views = futures_util::future::join_all(servers.iter().map(|s| async move {
            let mut view = s.public();
            let running = self.running.lock().await.get(&s.name).cloned();
            let client = match (running, start) {
                (Some(c), _) => Some(Ok(c)),
                (None, true) => Some(self.client(&s.name).await),
                (None, false) => None,
            };
            match client {
                Some(Ok(c)) => {
                    view["tools"] = json!(
                        c.tools
                            .iter()
                            .map(|t| json!({
                                "name": t.name,
                                "qualified": qualified(&s.name, &t.name),
                                "description": t.description,
                                "input_schema": t.input_schema,
                            }))
                            .collect::<Vec<_>>()
                    );
                }
                Some(Err(e)) => view["error"] = e.into(),
                None => {}
            }
            view
        }))
        .await;
        json!({
            "path": config::path(&self.home),
            "servers": views,
            "errors": errors,
        })
    }

    /// Calls `tool` of `server`. A server that stopped is started once more for the call.
    pub async fn call(&self, server: &str, tool: &str, arguments: Value) -> Outcome {
        let client = match self.client(server).await {
            Ok(c) => c,
            Err(e) => return Outcome::error(e),
        };
        let outcome = client.client.lock().await.call(tool, arguments).await;
        if outcome.is_error && outcome.text.starts_with("the server stopped") {
            self.stop(Some(server)).await;
        }
        outcome
    }
}

/// `<server>__<tool>`, at most 64 characters (the longest name model APIs accept).
pub fn qualified(server: &str, tool: &str) -> Option<String> {
    let name = format!("{server}{SEPARATOR}{tool}");
    (name.len() <= 64).then_some(name)
}

/// The server and tool of a qualified name.
pub fn split(name: &str) -> Option<(&str, &str)> {
    name.split_once(SEPARATOR)
}
