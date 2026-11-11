//! A small MCP client (028): `initialize`, `tools/list` (with cursors) and `tools/call`, over
//! stdio (newline-delimited JSON-RPC to a caged child) or streamable HTTP (JSON or SSE answers,
//! `Mcp-Session-Id`). Every request waits at most 60 s. Env values reach a child through its
//! environment, never argv; they are masked in the stderr lines kept for errors.

use super::cage::{self, Denial, Proc};
use super::config::{Server, Transport};
use serde::Serialize;
use serde_json::{Value, json};
use std::collections::{BTreeMap, VecDeque};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader, Lines};
use tokio::process::{ChildStdin, ChildStdout};

pub const TIMEOUT: Duration = Duration::from_secs(60);
const PROTOCOL: &str = "2025-06-18";

/// A tool as the server lists it.
#[derive(Clone, Debug, Serialize)]
pub struct Tool {
    pub name: String,
    pub description: String,
    pub input_schema: Value,
}

/// What a call returns to the model.
#[derive(Clone, Debug, Default, Serialize)]
pub struct Outcome {
    pub text: String,
    pub is_error: bool,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub denials: Vec<Denial>,
}

impl Outcome {
    pub fn error(text: impl Into<String>) -> Self {
        Outcome {
            text: text.into(),
            is_error: true,
            denials: Vec::new(),
        }
    }
}

#[derive(Debug)]
enum Conn {
    Stdio {
        proc: Box<Proc>,
        stdin: ChildStdin,
        lines: Lines<BufReader<ChildStdout>>,
        stderr: Arc<Mutex<VecDeque<String>>>,
    },
    Http {
        url: String,
        headers: BTreeMap<String, String>,
        session: Option<String>,
        http: reqwest::Client,
    },
}

/// A started server.
#[derive(Debug)]
pub struct Client {
    conn: Conn,
    next: u64,
    pub tools: Vec<Tool>,
}

/// The program a stdio entry runs: `mrun` is this executable (its own tools), other bare names
/// are looked up on `PATH`.
pub fn program(command: &str) -> PathBuf {
    if command == "mrun"
        && let Ok(exe) = std::env::current_exe()
    {
        return exe;
    }
    let path = PathBuf::from(command);
    if path.components().count() > 1 {
        return super::config::expand(command);
    }
    crate::cli::which(command).unwrap_or(path)
}

impl Client {
    /// Starts `server` and lists its tools.
    pub async fn start(server: &Server) -> Result<Client, String> {
        let conn = match &server.transport {
            Transport::Stdio {
                command,
                args,
                env,
                cwd,
            } => {
                let program = program(command);
                let mut cmd = tokio::process::Command::new(&program);
                cmd.args(args)
                    .envs(env)
                    .stdin(std::process::Stdio::piped())
                    .stdout(std::process::Stdio::piped())
                    .stderr(std::process::Stdio::piped())
                    .kill_on_drop(true);
                if let Some(cwd) = cwd {
                    cmd.current_dir(cwd);
                }
                let mut names: Vec<String> = env.keys().cloned().collect();
                // mrun's own tools need no config file (and a cage would deny reading it).
                if program.file_name().is_some_and(|n| n == "mrun")
                    && args.first().is_some_and(|a| a == "mcp")
                {
                    cmd.env("MRUN_CONFIG", "");
                    names.push("MRUN_CONFIG".into());
                }
                let mut proc = cage::spawn(&mut cmd, server.sandbox.as_ref(), &program, &names)
                    .map_err(|e| format!("{command}: {e}"))?;
                let child = proc.child();
                let stdin = child.stdin.take().ok_or("no stdin")?;
                let stdout = child.stdout.take().ok_or("no stdout")?;
                let stderr = Arc::new(Mutex::new(VecDeque::new()));
                if let Some(err) = child.stderr.take() {
                    let keep = stderr.clone();
                    let secrets: Vec<String> =
                        env.values().filter(|v| v.len() >= 4).cloned().collect();
                    tokio::spawn(async move {
                        let mut lines = BufReader::new(err).lines();
                        while let Ok(Some(mut line)) = lines.next_line().await {
                            for s in &secrets {
                                line = line.replace(s.as_str(), "•••");
                            }
                            let mut keep = keep.lock().unwrap_or_else(|e| e.into_inner());
                            if keep.len() == 20 {
                                keep.pop_front();
                            }
                            keep.push_back(line);
                        }
                    });
                }
                Conn::Stdio {
                    proc: Box::new(proc),
                    stdin,
                    lines: BufReader::new(stdout).lines(),
                    stderr,
                }
            }
            Transport::Http { url, headers } => Conn::Http {
                url: url.clone(),
                headers: headers.clone(),
                session: None,
                http: reqwest::Client::builder()
                    .timeout(TIMEOUT)
                    .build()
                    .map_err(|e| e.to_string())?,
            },
        };
        let mut client = Client {
            conn,
            next: 0,
            tools: Vec::new(),
        };
        client
            .request(
                "initialize",
                json!({
                    "protocolVersion": PROTOCOL,
                    "capabilities": {},
                    "clientInfo": {"name": "mrun", "version": env!("CARGO_PKG_VERSION")},
                }),
            )
            .await?;
        client.notify("notifications/initialized").await?;
        client.tools = client.list().await?;
        Ok(client)
    }

    async fn list(&mut self) -> Result<Vec<Tool>, String> {
        let mut tools = Vec::new();
        let mut cursor: Option<String> = None;
        for _ in 0..100 {
            let params = cursor.as_ref().map_or(json!({}), |c| json!({"cursor": c}));
            let page = self.request("tools/list", params).await?;
            for t in page["tools"].as_array().into_iter().flatten() {
                let Some(name) = t["name"].as_str() else {
                    continue;
                };
                tools.push(Tool {
                    name: name.into(),
                    description: t["description"].as_str().unwrap_or("").into(),
                    input_schema: match &t["inputSchema"] {
                        Value::Object(_) => t["inputSchema"].clone(),
                        _ => json!({"type": "object"}),
                    },
                });
            }
            cursor = page["nextCursor"].as_str().map(str::to_string);
            if cursor.is_none() {
                break;
            }
        }
        Ok(tools)
    }

    /// Calls `tool`; failures, timeouts and sandbox denials come back as error outcomes.
    pub async fn call(&mut self, tool: &str, arguments: Value) -> Outcome {
        let args = if arguments.is_object() {
            arguments
        } else {
            json!({})
        };
        let result = self
            .request("tools/call", json!({"name": tool, "arguments": args}))
            .await;
        let mut outcome = match result {
            Ok(r) => Outcome {
                text: text_of(&r["content"]),
                is_error: r["isError"].as_bool().unwrap_or(false),
                denials: Vec::new(),
            },
            Err(e) => Outcome::error(e),
        };
        if let Conn::Stdio { proc, .. } = &mut self.conn {
            outcome.denials = proc.denials();
        }
        outcome
    }

    async fn notify(&mut self, method: &str) -> Result<(), String> {
        self.send(json!({"jsonrpc": "2.0", "method": method}), None)
            .await
            .map(|_| ())
    }

    async fn request(&mut self, method: &str, params: Value) -> Result<Value, String> {
        self.next += 1;
        let id = self.next;
        let message = json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params});
        let reply = tokio::time::timeout(TIMEOUT, self.send(message, Some(id)))
            .await
            .map_err(|_| format!("{method}: no answer in {} s", TIMEOUT.as_secs()))??
            .ok_or_else(|| format!("{method}: no answer"))?;
        if let Some(err) = reply.get("error") {
            return Err(err["message"].as_str().unwrap_or("error").to_string());
        }
        Ok(reply["result"].clone())
    }

    /// Sends one message; with `id`, returns the answer to it.
    async fn send(&mut self, message: Value, id: Option<u64>) -> Result<Option<Value>, String> {
        match &mut self.conn {
            Conn::Stdio {
                stdin,
                lines,
                stderr,
                ..
            } => {
                let died = |stderr: &Arc<Mutex<VecDeque<String>>>| {
                    let lines = stderr.lock().unwrap_or_else(|e| e.into_inner());
                    let tail: Vec<&str> = lines
                        .iter()
                        .rev()
                        .take(3)
                        .rev()
                        .map(String::as_str)
                        .collect();
                    if tail.is_empty() {
                        "the server stopped".to_string()
                    } else {
                        format!("the server stopped: {}", tail.join(" / "))
                    }
                };
                let mut line = message.to_string();
                line.push('\n');
                if stdin.write_all(line.as_bytes()).await.is_err() || stdin.flush().await.is_err() {
                    return Err(died(stderr));
                }
                let Some(id) = id else { return Ok(None) };
                loop {
                    let Ok(Some(line)) = lines.next_line().await else {
                        return Err(died(stderr));
                    };
                    let Ok(reply) = serde_json::from_str::<Value>(&line) else {
                        continue;
                    };
                    if reply["id"].as_u64() == Some(id) && reply.get("method").is_none() {
                        return Ok(Some(reply));
                    }
                    if reply.get("method").is_some() && reply.get("id").is_some() {
                        // A request from the server (roots, sampling): not offered.
                        let no = json!({"jsonrpc": "2.0", "id": reply["id"], "error": {"code": -32601, "message": "not supported"}});
                        let _ = stdin.write_all(format!("{no}\n").as_bytes()).await;
                    }
                }
            }
            Conn::Http {
                url,
                headers,
                session,
                http,
            } => {
                let mut req = http
                    .post(url.as_str())
                    .header("Accept", "application/json, text/event-stream")
                    .header("MCP-Protocol-Version", PROTOCOL)
                    .json(&message);
                for (k, v) in headers.iter() {
                    req = req.header(k, v);
                }
                if let Some(s) = session.as_ref() {
                    req = req.header("Mcp-Session-Id", s);
                }
                let host = super::config::host_of(url);
                let resp = req
                    .send()
                    .await
                    .map_err(|e| format!("{host}: {}", e.without_url()))?;
                if let Some(s) = resp
                    .headers()
                    .get("mcp-session-id")
                    .and_then(|v| v.to_str().ok())
                {
                    *session = Some(s.to_string());
                }
                let status = resp.status();
                if !status.is_success() {
                    return Err(format!("{host}: HTTP {}", status.as_u16()));
                }
                let Some(id) = id else { return Ok(None) };
                let sse = resp
                    .headers()
                    .get("content-type")
                    .and_then(|v| v.to_str().ok())
                    .is_some_and(|t| t.starts_with("text/event-stream"));
                let body = resp
                    .text()
                    .await
                    .map_err(|e| format!("{host}: {}", e.without_url()))?;
                let messages: Vec<Value> = if sse {
                    events(&body)
                } else {
                    serde_json::from_str::<Value>(&body).into_iter().collect()
                };
                messages
                    .into_iter()
                    .flat_map(|m| match m {
                        Value::Array(batch) => batch,
                        one => vec![one],
                    })
                    .find(|m| m["id"].as_u64() == Some(id))
                    .map(Some)
                    .ok_or_else(|| format!("{host}: no answer"))
            }
        }
    }
}

/// The JSON messages of an SSE body (`data:` lines, one event per blank line).
fn events(body: &str) -> Vec<Value> {
    let mut out = Vec::new();
    let mut data = String::new();
    for line in body.lines().chain(std::iter::once("")) {
        if let Some(rest) = line.strip_prefix("data:") {
            data.push_str(rest.strip_prefix(' ').unwrap_or(rest));
        } else if line.is_empty() && !data.is_empty() {
            if let Ok(v) = serde_json::from_str(&data) {
                out.push(v);
            }
            data.clear();
        }
    }
    out
}

/// Text parts joined; other parts named, not sent.
fn text_of(content: &Value) -> String {
    let parts: Vec<String> = content
        .as_array()
        .into_iter()
        .flatten()
        .map(|p| match p["type"].as_str() {
            Some("text") => p["text"].as_str().unwrap_or("").to_string(),
            Some("resource") if p["resource"]["text"].is_string() => {
                p["resource"]["text"].as_str().unwrap_or("").to_string()
            }
            Some(kind) => format!("[{kind} not shown]"),
            None => String::new(),
        })
        .collect();
    parts.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sse_bodies_and_content_parts() {
        let body = "event: message\ndata: {\"jsonrpc\":\"2.0\",\"id\":1,\n\ndata: {\"jsonrpc\":\"2.0\",\"id\":2,\"result\":{}}\n\n";
        let got = events(body);
        assert_eq!(got.len(), 1, "a broken event is skipped");
        assert_eq!(got[0]["id"], 2);
        let content = json!([{"type": "text", "text": "a"}, {"type": "image", "data": "…"}, {"type": "text", "text": "b"}]);
        assert_eq!(text_of(&content), "a\n[image not shown]\nb");
    }

    #[test]
    fn mrun_is_this_executable() {
        assert_eq!(program("mrun"), std::env::current_exe().unwrap());
        assert_eq!(program("/bin/sh"), PathBuf::from("/bin/sh"));
    }
}
