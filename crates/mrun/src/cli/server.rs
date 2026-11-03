//! A tiny client for the running server's loopback endpoints (`/v1/models`, `/health`,
//! `/mrun/status`), used by `launch`, `connect`, `ps`, `doctor` and `stop`. Plain HTTP/1.1 over
//! a `TcpStream`: the server is local and its JSON bodies are small, so no HTTP stack is needed.

use crate::errors::{CliError, Code};
use serde_json::Value;
use std::io::{IsTerminal, Read, Write};
use std::net::{SocketAddr, TcpStream};
use std::process::Child;
use std::time::{Duration, Instant};

const TIMEOUT: Duration = Duration::from_secs(5);
/// How long a model may take to load before `serve --detach` and `launch` give up waiting.
pub const LOAD_TIMEOUT: Duration = Duration::from_secs(600);

/// The port commands talk to: `MRUN_PORT`, else the default.
pub fn default_port() -> u16 {
    std::env::var("MRUN_PORT")
        .ok()
        .and_then(|p| p.parse().ok())
        .unwrap_or(crate::cli::serve::DEFAULT_PORT)
}

/// What clients need from the running server.
#[derive(Clone, Debug, PartialEq)]
pub struct ServerInfo {
    pub port: u16,
    pub model_id: String,
    /// Context capacity on this machine, in tokens.
    pub context: u32,
    /// The server's default output limit.
    pub max_output: u32,
    /// The server records requests (`--record`).
    pub recording: bool,
}

impl ServerInfo {
    pub fn base_url(&self) -> String {
        format!("http://127.0.0.1:{}", self.port)
    }

    /// Reads `/v1/models` of the server on `port`.
    pub fn fetch(port: u16) -> Result<Self, CliError> {
        let (status, body) = request(port, "GET", "/v1/models")?;
        let model = &body["data"][0];
        if status != 200 || model.is_null() {
            return Err(unreachable(port).hint(
                "the server is still loading or failed to load; check 'mrun ps' or its log",
            ));
        }
        let recording = request(port, "GET", "/mrun/status")
            .is_ok_and(|(_, status)| status["recording"] == true);
        Ok(Self {
            port,
            model_id: model["id"].as_str().unwrap_or_default().to_string(),
            context: model["context_window"].as_u64().unwrap_or(0) as u32,
            max_output: model["max_output_tokens"].as_u64().unwrap_or(0) as u32,
            recording,
        })
    }
}

/// How long the server on `port` has been running (its process age, from the pid it reports
/// in `/mrun/status`): while it loads, the time spent loading.
pub fn loading_seconds(port: u16) -> Option<u64> {
    let (_, status) = request(port, "GET", "/mrun/status").ok()?;
    let pid = status["pid"].as_u64()?;
    let output = std::process::Command::new("ps")
        .args(["-p", &pid.to_string(), "-o", "etime="])
        .output()
        .ok()?;
    parse_etime(String::from_utf8_lossy(&output.stdout).trim())
}

/// `ps -o etime` (`[[dd-]hh:]mm:ss`) in seconds.
fn parse_etime(text: &str) -> Option<u64> {
    let (days, clock) = match text.split_once('-') {
        Some((days, clock)) => (days.parse::<u64>().ok()?, clock),
        None => (0, text),
    };
    let mut seconds = 0;
    for part in clock.split(':') {
        seconds = seconds * 60 + part.parse::<u64>().ok()?;
    }
    Some(days * 86_400 + seconds)
}

/// What `/health` says about the server on a port.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Health {
    /// Nothing (or not mrun) answers.
    Down,
    /// Loading the named model.
    Loading(Option<String>),
    Ready(String),
    /// Loading failed with this error text.
    Failed(String),
}

pub fn health(port: u16) -> Health {
    let Ok((status, body)) = request(port, "GET", "/health") else {
        return Health::Down;
    };
    let text = |key: &str| body[key].as_str().map(str::to_string);
    match (status, body["status"].as_str()) {
        (200, _) => Health::Ready(text("model").unwrap_or_default()),
        (_, Some("loading")) => Health::Loading(text("model")),
        (_, Some("error")) => Health::Failed(text("error").unwrap_or_default()),
        _ => Health::Down,
    }
}

/// How waiting for a loading server ended.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum LoadEnd {
    Ready,
    /// The server process (`child`) exited with this code.
    Exited(u8),
    /// Loading failed (seen through `/health` of a server that is not our child).
    Failed(String),
    TimedOut,
}

/// Waits until the server on `port` has loaded its model, showing `label… <n> s` on a terminal.
/// With `child`, its exit ends the wait (a failed load exits the server).
pub fn wait_until_loaded(port: u16, label: &str, mut child: Option<&mut Child>) -> LoadEnd {
    let start = Instant::now();
    let live = std::io::stderr().is_terminal() && !crate::cli::is_quiet();
    if !live {
        crate::cli::info(&format!("{label}…"));
    }
    let end = loop {
        if let Some(child) = child.as_deref_mut()
            && let Ok(Some(status)) = child.try_wait()
        {
            break LoadEnd::Exited(status.code().unwrap_or(1).clamp(1, 255) as u8);
        }
        match health(port) {
            Health::Ready(_) => break LoadEnd::Ready,
            Health::Failed(error) if child.is_none() => break LoadEnd::Failed(error),
            _ => {}
        }
        if start.elapsed() > LOAD_TIMEOUT {
            break LoadEnd::TimedOut;
        }
        if live {
            eprint!("\r{label}… {} s", start.elapsed().as_secs());
            let _ = std::io::stderr().flush();
        }
        std::thread::sleep(Duration::from_millis(250));
    };
    let seconds = start.elapsed().as_secs();
    if live {
        eprintln!("\r{label}… {seconds} s");
    } else {
        crate::cli::info(&format!("{label}… {seconds} s"));
    }
    end
}

fn unreachable(port: u16) -> CliError {
    CliError::new(
        Code::ServerUnreachable,
        format!("no mrun server is running on 127.0.0.1:{port}"),
    )
    .hint("start one with 'mrun serve --detach' (or 'mrun launch <client>', which offers to)")
}

/// One request with an empty body; returns the status and the JSON body (`Null` if none).
pub fn request(port: u16, method: &str, path: &str) -> Result<(u16, Value), CliError> {
    let addr = SocketAddr::from(([127, 0, 0, 1], port));
    let mut stream = TcpStream::connect_timeout(&addr, TIMEOUT).map_err(|_| unreachable(port))?;
    stream.set_read_timeout(Some(TIMEOUT)).ok();
    let head = format!(
        "{method} {path} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
    );
    stream
        .write_all(head.as_bytes())
        .map_err(|_| unreachable(port))?;
    let mut raw = Vec::new();
    stream
        .read_to_end(&mut raw)
        .map_err(|_| unreachable(port))?;
    parse_response(&raw).ok_or_else(|| {
        CliError::new(
            Code::ServerUnreachable,
            format!("unexpected answer from 127.0.0.1:{port}{path}"),
        )
        .hint("is another program using this port? Try 'mrun doctor'")
    })
}

/// Status and JSON body of a `Connection: close` HTTP/1.1 response.
fn parse_response(raw: &[u8]) -> Option<(u16, Value)> {
    let split = raw.windows(4).position(|w| w == b"\r\n\r\n")?;
    let head = std::str::from_utf8(&raw[..split]).ok()?;
    let status = head.split_whitespace().nth(1)?.parse().ok()?;
    let body = &raw[split + 4..];
    let body = if head
        .to_ascii_lowercase()
        .contains("transfer-encoding: chunked")
    {
        dechunk(body)?
    } else {
        body.to_vec()
    };
    let value = if body.is_empty() {
        Value::Null
    } else {
        serde_json::from_slice(&body).ok()?
    };
    Some((status, value))
}

fn dechunk(mut body: &[u8]) -> Option<Vec<u8>> {
    let mut out = Vec::new();
    loop {
        let line = body.windows(2).position(|w| w == b"\r\n")?;
        let size =
            usize::from_str_radix(std::str::from_utf8(&body[..line]).ok()?.trim(), 16).ok()?;
        body = &body[line + 2..];
        if size == 0 {
            return Some(out);
        }
        out.extend_from_slice(body.get(..size)?);
        body = body.get(size + 2..)?;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_plain_and_chunked_bodies() {
        let plain = b"HTTP/1.1 200 OK\r\ncontent-length: 11\r\n\r\n{\"a\": true}";
        assert_eq!(
            parse_response(plain),
            Some((200, serde_json::json!({"a": true})))
        );
        let chunked =
            b"HTTP/1.1 503 X\r\nTransfer-Encoding: chunked\r\n\r\n4\r\n{\"b\"\r\n4\r\n: 1}\r\n0\r\n\r\n";
        assert_eq!(
            parse_response(chunked),
            Some((503, serde_json::json!({"b": 1})))
        );
        assert_eq!(parse_response(b"garbage"), None);
    }

    #[test]
    fn process_ages_parse() {
        assert_eq!(parse_etime("00:42"), Some(42));
        assert_eq!(parse_etime("01:02:03"), Some(3723));
        assert_eq!(parse_etime("2-00:00:01"), Some(172_801));
        assert_eq!(parse_etime(""), None);
    }

    #[test]
    fn a_closed_port_is_down() {
        let port = std::net::TcpListener::bind("127.0.0.1:0")
            .unwrap()
            .local_addr()
            .unwrap()
            .port();
        assert_eq!(health(port), Health::Down);
    }
}
