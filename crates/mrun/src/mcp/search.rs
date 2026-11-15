//! mrun's `search` tool (029): web search through a service the user picks. Exa answers without a
//! key over its hosted MCP endpoint; Brave Search takes a key; SearXNG is the user's own instance.
//! Keys travel in request headers only, and errors never carry URLs.

use serde_json::{Value, json};
use std::time::Duration;

/// The most results a search returns.
pub const MAX_RESULTS: usize = 8;
/// The longest passage per result, in characters.
pub const MAX_PASSAGE: usize = 1_200;
/// The most bytes read from a service.
const MAX_REPLY: usize = 2 << 20;
const TIMEOUT: Duration = Duration::from_secs(30);
const EXA: &str = "https://mcp.exa.ai/mcp";
const BRAVE: &str = "https://api.search.brave.com/res/v1/web/search";

/// Where queries go.
#[derive(Clone, PartialEq)]
pub enum Provider {
    Exa { key: Option<String>, base: String },
    Brave { key: String, base: String },
    SearXng { url: String },
}

/// Never the key.
impl std::fmt::Debug for Provider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} ({})", self.name(), self.host())
    }
}

/// One result.
#[derive(Clone, Debug, PartialEq)]
pub struct Hit {
    pub title: String,
    pub url: String,
    pub date: Option<String>,
    pub passage: String,
}

impl Provider {
    /// The provider of `--provider` and `--url`, with its key from `env` (`EXA_API_KEY`,
    /// `BRAVE_API_KEY`).
    pub fn new(
        provider: &str,
        url: Option<&str>,
        env: impl Fn(&str) -> Option<String>,
    ) -> Result<Self, String> {
        let key = |name: &str| env(name).filter(|k| !k.trim().is_empty());
        match provider {
            "exa" => Ok(Provider::Exa {
                key: key("EXA_API_KEY"),
                base: EXA.into(),
            }),
            "brave" => Ok(Provider::Brave {
                key: key("BRAVE_API_KEY")
                    .ok_or("Brave Search needs a key: set BRAVE_API_KEY (search.brave.com/api)")?,
                base: BRAVE.into(),
            }),
            "searxng" => {
                let url = url.ok_or("SearXNG needs --url, the address of your instance")?;
                let parsed = reqwest::Url::parse(url).map_err(|e| format!("{url}: {e}"))?;
                if !matches!(parsed.scheme(), "http" | "https") || parsed.host_str().is_none() {
                    return Err(format!("{url}: not an http or https address"));
                }
                Ok(Provider::SearXng {
                    url: url.trim_end_matches('/').into(),
                })
            }
            other => Err(format!("{other}: the providers are exa, brave and searxng")),
        }
    }

    /// The service as people say it.
    pub fn name(&self) -> &'static str {
        match self {
            Provider::Exa { .. } => "Exa",
            Provider::Brave { .. } => "Brave Search",
            Provider::SearXng { .. } => "SearXNG",
        }
    }

    /// The host that receives the queries.
    pub fn host(&self) -> String {
        let url = match self {
            Provider::Exa { base, .. } | Provider::Brave { base, .. } => base,
            Provider::SearXng { url } => url,
        };
        reqwest::Url::parse(url)
            .ok()
            .and_then(|u| {
                u.host_str()
                    .map(|h| u.port().map_or(h.to_string(), |p| format!("{h}:{p}")))
            })
            .unwrap_or_default()
    }

    /// The results for `query`, at most `count`.
    pub async fn search(&self, query: &str, count: usize) -> Result<Vec<Hit>, String> {
        let count = count.clamp(1, MAX_RESULTS);
        let client = reqwest::Client::builder()
            .timeout(TIMEOUT)
            .user_agent(concat!("mrun/", env!("CARGO_PKG_VERSION")))
            .build()
            .map_err(|e| e.without_url().to_string())?;
        let request = match self {
            Provider::Exa { key, base } => {
                let mut r = client
                    .post(base)
                    .header("accept", "application/json, text/event-stream")
                    .json(&json!({
                        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
                        "params": {"name": "web_search_exa", "arguments": {"query": query, "numResults": count}},
                    }));
                if let Some(k) = key {
                    r = r.header("x-api-key", k);
                }
                r
            }
            Provider::Brave { key, base } => client
                .get(with_query(
                    base,
                    &[("q", query), ("count", &count.to_string())],
                )?)
                .header("accept", "application/json")
                .header("x-subscription-token", key),
            Provider::SearXng { url } => client.get(with_query(
                &format!("{url}/search"),
                &[("q", query), ("format", "json")],
            )?),
        };
        let resp = request
            .send()
            .await
            .map_err(|e| format!("{}: {}", self.name(), e.without_url()))?;
        let status = resp.status().as_u16();
        let body = read(resp)
            .await
            .map_err(|e| format!("{}: {e}", self.name()))?;
        if status != 200 {
            return Err(self.failed(status));
        }
        let hits = match self {
            Provider::Exa { .. } => exa(&body).map_err(|e| format!("Exa: {e}"))?,
            Provider::Brave { .. } => brave(&body).map_err(|e| format!("Brave Search: {e}"))?,
            Provider::SearXng { .. } => searxng(&body).map_err(|e| format!("SearXNG: {e}"))?,
        };
        Ok(hits.into_iter().take(count).collect())
    }

    /// A failed status as what to do about it.
    fn failed(&self, status: u16) -> String {
        let what = match status {
            429 => "rate limited",
            401 | 403 if matches!(self, Provider::SearXng { .. }) => {
                return format!(
                    "SearXNG answered HTTP {status}: add `json` to `search.formats` in its settings.yml"
                );
            }
            401 | 403 => "the key was refused",
            _ => "the search failed",
        };
        let fix = match self {
            Provider::Exa { key: None, .. } => {
                "; set EXA_API_KEY, or use another service: mrun mcp add search --provider brave"
            }
            Provider::Exa { .. } | Provider::Brave { .. } => "; check the key in ~/.mrun/mcp.json",
            Provider::SearXng { .. } => "; check that the instance is running",
        };
        format!("{} answered HTTP {status}: {what}{fix}", self.name())
    }
}

fn with_query(base: &str, pairs: &[(&str, &str)]) -> Result<reqwest::Url, String> {
    let mut url =
        reqwest::Url::parse(base).map_err(|_| "not a valid service address".to_string())?;
    url.query_pairs_mut().extend_pairs(pairs);
    Ok(url)
}

async fn read(resp: reqwest::Response) -> Result<String, String> {
    use futures_util::StreamExt;
    let mut body = Vec::new();
    let mut stream = resp.bytes_stream();
    while let Some(chunk) = stream.next().await {
        body.extend_from_slice(&chunk.map_err(|e| e.without_url().to_string())?);
        if body.len() > MAX_REPLY {
            return Err("the reply is larger than 2 MiB".into());
        }
    }
    Ok(String::from_utf8_lossy(&body).into_owned())
}

/// Exa's MCP reply (JSON, or an event stream whose `data:` line is the JSON): one text whose
/// results are separated by `---` lines, each with `Title:`, `URL:`, `Published:` and the passage.
fn exa(body: &str) -> Result<Vec<Hit>, String> {
    let json = body
        .lines()
        .find_map(|l| l.strip_prefix("data:"))
        .unwrap_or(body);
    let v: Value = serde_json::from_str(json.trim()).map_err(|_| "unreadable reply".to_string())?;
    if let Some(e) = v.get("error") {
        return Err(e["message"].as_str().unwrap_or("error").to_string());
    }
    let text = v["result"]["content"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|c| c["text"].as_str())
        .collect::<Vec<_>>()
        .join("\n\n---\n\n");
    if v["result"]["isError"].as_bool() == Some(true) {
        return Err(text);
    }
    let field = |block: &str, name: &str| {
        block
            .lines()
            .find_map(|l| l.strip_prefix(name))
            .map(|v| v.trim().to_string())
            .filter(|v| !v.is_empty() && v != "N/A")
    };
    Ok(text
        .split("\n\n---\n\n")
        .filter_map(|block| {
            let url = field(block, "URL:")?;
            let passage = ["Highlights:", "Text:", "Summary:"]
                .iter()
                .find_map(|h| block.find(h).map(|i| block[i + h.len()..].trim()))
                .unwrap_or("");
            Some(Hit {
                title: field(block, "Title:").unwrap_or_else(|| url.clone()),
                date: field(block, "Published:"),
                url,
                passage: passage.to_string(),
            })
        })
        .collect())
}

fn brave(body: &str) -> Result<Vec<Hit>, String> {
    let v: Value = serde_json::from_str(body).map_err(|_| "unreadable reply".to_string())?;
    Ok(v["web"]["results"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|r| {
            Some(Hit {
                title: plain(r["title"].as_str().unwrap_or("")),
                url: r["url"].as_str()?.to_string(),
                date: r["page_age"]
                    .as_str()
                    .or(r["age"].as_str())
                    .map(str::to_string),
                passage: plain(r["description"].as_str().unwrap_or("")),
            })
        })
        .collect())
}

fn searxng(body: &str) -> Result<Vec<Hit>, String> {
    let v: Value = serde_json::from_str(body)
        .map_err(|_| "not JSON: add `json` to `search.formats` in its settings.yml".to_string())?;
    Ok(v["results"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|r| {
            Some(Hit {
                title: r["title"].as_str().unwrap_or("").to_string(),
                url: r["url"].as_str()?.to_string(),
                date: r["publishedDate"].as_str().map(str::to_string),
                passage: r["content"].as_str().unwrap_or("").to_string(),
            })
        })
        .collect())
}

/// Brave marks matches with `<strong>`; the model gets text.
fn plain(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut tag = false;
    for c in s.chars() {
        match c {
            '<' => tag = true,
            '>' => tag = false,
            c if !tag => out.push(c),
            _ => {}
        }
    }
    out.replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&#39;", "'")
}

/// At most `max` characters, cut at a word boundary.
fn passage(text: &str, max: usize) -> String {
    let text = text.trim();
    if text.chars().count() <= max {
        return text.to_string();
    }
    let cut: String = text.chars().take(max).collect();
    let end = cut.rfind(char::is_whitespace).unwrap_or(cut.len());
    format!("{}…", cut[..end].trim_end())
}

/// The results as the model reads them.
pub fn render(query: &str, hits: &[Hit]) -> String {
    if hits.is_empty() {
        return format!("No results for {query}.");
    }
    hits.iter()
        .map(|h| {
            let title = if h.title.trim().is_empty() {
                &h.url
            } else {
                &h.title
            };
            let mut s = format!("## [{}]({})", title.trim(), h.url);
            if let Some(d) = &h.date {
                s.push_str(&format!("\nPublished: {d}"));
            }
            let p = passage(&h.passage, MAX_PASSAGE);
            if !p.is_empty() {
                s.push_str("\n\n");
                s.push_str(&p);
            }
            s
        })
        .collect::<Vec<_>>()
        .join("\n\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn serve(app: axum::Router) -> String {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await });
        format!("http://{addr}")
    }

    const EXA_TEXT: &str = "Title: select in tokio - Rust\nURL: https://docs.rs/tokio/latest/tokio/macro.select.html\nPublished: N/A\nAuthor: N/A\nHighlights:\nWaits on multiple concurrent branches.\n\n---\n\nTitle: Tokio 1.48\nURL: https://tokio.rs/blog/1-48\nPublished: 2026-10-01T00:00:00.000Z\nAuthor: N/A\nText:\nRelease notes.";

    #[tokio::test]
    async fn exa_reads_an_event_stream_and_sends_the_key_as_a_header() {
        use axum::http::HeaderMap;
        let app = axum::Router::new().route(
            "/mcp",
            axum::routing::post(|h: HeaderMap, body: axum::Json<Value>| async move {
                assert_eq!(body["params"]["name"], "web_search_exa");
                assert_eq!(body["params"]["arguments"]["numResults"], 2);
                assert_eq!(h["x-api-key"], "sentinel-key");
                let reply = json!({"jsonrpc": "2.0", "id": 1, "result": {"content": [{"type": "text", "text": EXA_TEXT}]}});
                ([("content-type", "text/event-stream")], format!("event: message\ndata: {reply}\n\n"))
            }),
        );
        let base = format!("{}/mcp", serve(app).await);
        let exa = Provider::Exa {
            key: Some("sentinel-key".into()),
            base,
        };
        let hits = exa.search("tokio select", 2).await.unwrap();
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].title, "select in tokio - Rust");
        assert_eq!(hits[0].date, None);
        assert_eq!(hits[1].date.as_deref(), Some("2026-10-01T00:00:00.000Z"));
        assert_eq!(hits[1].passage, "Release notes.");
        assert!(!format!("{exa:?}").contains("sentinel"));
        let text = render("tokio select", &hits);
        assert!(text.starts_with("## [select in tokio - Rust](https://docs.rs/"));
        assert!(text.contains("Published: 2026-10-01"));
    }

    #[tokio::test]
    async fn failures_name_the_service_the_status_and_the_fix() {
        let app = axum::Router::new()
            .route(
                "/mcp",
                axum::routing::post(|| async {
                    (axum::http::StatusCode::TOO_MANY_REQUESTS, "slow down")
                }),
            )
            .route(
                "/search",
                axum::routing::get(|| async { (axum::http::StatusCode::FORBIDDEN, "no") }),
            );
        let base = serve(app).await;
        let exa = Provider::Exa {
            key: None,
            base: format!("{base}/mcp"),
        };
        let e = exa.search("q", 8).await.unwrap_err();
        assert!(e.starts_with("Exa answered HTTP 429: rate limited"), "{e}");
        assert!(e.contains("--provider brave"), "{e}");
        let sx = Provider::SearXng { url: base.clone() };
        let e = sx.search("q", 8).await.unwrap_err();
        assert!(e.contains("search.formats"), "{e}");
        assert!(!e.contains(&base), "errors carry no URL: {e}");
    }

    #[tokio::test]
    async fn brave_and_searxng_read_their_json() {
        let app = axum::Router::new()
            .route(
                "/brave",
                axum::routing::get(|h: axum::http::HeaderMap| async move {
                    assert_eq!(h["x-subscription-token"], "k");
                    axum::Json(json!({"web": {"results": [
                        {"title": "The <strong>Rust</strong> Book", "url": "https://doc.rust-lang.org/book/", "description": "Learn &amp; build", "page_age": "2026-01-02"}
                    ]}}))
                }),
            )
            .route(
                "/search",
                axum::routing::get(|q: axum::extract::Query<std::collections::HashMap<String, String>>| async move {
                    assert_eq!(q["format"], "json");
                    axum::Json(json!({"results": [{"title": "A", "url": "https://a.example/", "content": "aa"}, {"url": "https://b.example/"}]}))
                }),
            );
        let base = serve(app).await;
        let brave = Provider::Brave {
            key: "k".into(),
            base: format!("{base}/brave"),
        };
        let hits = brave.search("rust", 8).await.unwrap();
        assert_eq!(hits[0].title, "The Rust Book");
        assert_eq!(hits[0].passage, "Learn & build");
        let sx = Provider::SearXng { url: base };
        let hits = sx.search("x", 1).await.unwrap();
        assert_eq!(hits.len(), 1, "count is honoured");
        assert_eq!(render("x", &[]), "No results for x.");
    }

    #[test]
    fn providers_come_from_flags_and_keys_from_the_environment() {
        let none = |_: &str| None;
        assert!(matches!(
            Provider::new("exa", None, none).unwrap(),
            Provider::Exa { key: None, .. }
        ));
        assert_eq!(
            Provider::new("exa", None, none).unwrap().host(),
            "mcp.exa.ai"
        );
        let e = Provider::new("brave", None, none).unwrap_err();
        assert!(e.contains("BRAVE_API_KEY"), "{e}");
        let brave = Provider::new("brave", None, |k| {
            (k == "BRAVE_API_KEY").then(|| "x".into())
        })
        .unwrap();
        assert_eq!(brave.host(), "api.search.brave.com");
        assert!(Provider::new("searxng", None, none).is_err());
        assert_eq!(
            Provider::new("searxng", Some("http://127.0.0.1:8888/"), none)
                .unwrap()
                .host(),
            "127.0.0.1:8888"
        );
        assert!(Provider::new("google", None, none).is_err());
    }

    #[test]
    fn passages_are_cut_at_a_word() {
        let long = "word ".repeat(400);
        let p = passage(&long, MAX_PASSAGE);
        assert!(p.chars().count() <= MAX_PASSAGE + 1);
        assert!(p.ends_with("word…"));
    }
}
