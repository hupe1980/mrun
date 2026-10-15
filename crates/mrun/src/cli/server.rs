//! A tiny client for the running server's loopback endpoints (`/v1/models`, `/health`,
//! `/mrun/status`), used by `launch`, `connect`, `ps`, `doctor` and `stop`. Plain HTTP/1.1 over
//! a `TcpStream`: the server is local and its JSON bodies are small, so no HTTP stack is needed.

use crate::errors::{CliError, Code};
use serde_json::Value;
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpStream};
use std::time::Duration;

const TIMEOUT: Duration = Duration::from_secs(5);

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
        Ok(Self {
            port,
            model_id: model["id"].as_str().unwrap_or_default().to_string(),
            context: model["context_window"].as_u64().unwrap_or(0) as u32,
            max_output: model["max_output_tokens"].as_u64().unwrap_or(0) as u32,
        })
    }
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
}
