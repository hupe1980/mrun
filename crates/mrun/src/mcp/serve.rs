//! mrun's own tools (028, FR-010): a stdio MCP server in the binary. `workspace` reads, writes,
//! edits, finds and runs in one folder (every path inside it; mcp.json cages it there too);
//! `fetch` reads web pages. Any MCP client can start them: `mrun mcp serve workspace --root DIR`.

use serde_json::{Value, json};
use std::path::{Component, Path, PathBuf};
use std::time::Duration;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

/// A `bash` command's limit.
pub const BASH_TIMEOUT: Duration = Duration::from_secs(120);
/// The most characters a result carries.
pub const MAX_CHARS: usize = 30_000;
/// The most bytes `fetch` reads.
pub const MAX_FETCH: usize = 1 << 20;
const SKIP: [&str; 4] = [".git", "node_modules", "target", ".venv"];

/// Which tools a server offers.
#[derive(Clone, Debug)]
pub enum Kit {
    Workspace(PathBuf),
    Fetch,
}

impl Kit {
    pub fn name(&self) -> &'static str {
        match self {
            Kit::Workspace(_) => "workspace",
            Kit::Fetch => "fetch",
        }
    }

    pub fn tools(&self) -> Value {
        let tool = |name: &str, description: &str, properties: Value, required: &[&str]| {
            json!({
                "name": name,
                "description": description,
                "inputSchema": {"type": "object", "properties": properties, "required": required},
            })
        };
        let s = |d: &str| json!({"type": "string", "description": d});
        let n = |d: &str| json!({"type": "integer", "description": d});
        match self {
            Kit::Workspace(_) => json!([
                tool(
                    "read",
                    "Read a text file of the workspace; lines are numbered.",
                    json!({"path": s("Path, relative to the workspace"), "offset": n("First line (1-based)"), "limit": n("Number of lines (default 2000)")}),
                    &["path"]
                ),
                tool(
                    "write",
                    "Create or overwrite a file of the workspace.",
                    json!({"path": s("Path, relative to the workspace"), "content": s("The whole new content")}),
                    &["path", "content"]
                ),
                tool(
                    "edit",
                    "Replace an exact string in a file. It must occur once, unless all is true.",
                    json!({"path": s("Path, relative to the workspace"), "old": s("The exact text to replace"), "new": s("The replacement"), "all": {"type": "boolean", "description": "Replace every occurrence"}}),
                    &["path", "old", "new"]
                ),
                tool(
                    "glob",
                    "List files matching a glob (`**/*.rs`, `src/*`), newest first.",
                    json!({"pattern": s("Glob, relative to the workspace")}),
                    &["pattern"]
                ),
                tool(
                    "grep",
                    "Search file contents with a regular expression; prints path:line:text.",
                    json!({"pattern": s("Regular expression"), "path": s("Folder or file to search (default the workspace)"), "glob": s("Only files matching this glob")}),
                    &["pattern"]
                ),
                tool(
                    "bash",
                    "Run a shell command in the workspace (120 s at most); prints its output and exit code.",
                    json!({"command": s("The command line")}),
                    &["command"]
                ),
            ]),
            Kit::Fetch => json!([tool(
                "fetch",
                "Fetch a web page (http or https) as text; HTML is reduced to its readable text.",
                json!({"url": s("The URL")}),
                &["url"]
            ),]),
        }
    }

    /// Runs one tool: `Ok` text or `Err` text for an error result.
    pub async fn call(&self, tool: &str, args: &Value) -> Result<String, String> {
        let arg = |k: &str| args[k].as_str().ok_or_else(|| format!("{k} is missing"));
        match (self, tool) {
            (Kit::Workspace(root), "read") => {
                let path = inside(root, arg("path")?)?;
                let text = std::fs::read_to_string(&path)
                    .map_err(|e| format!("{}: {e}", shown(root, &path)))?;
                let offset = args["offset"].as_u64().unwrap_or(1).max(1) as usize;
                let limit = args["limit"].as_u64().unwrap_or(2000) as usize;
                let lines: Vec<String> = text
                    .lines()
                    .enumerate()
                    .skip(offset - 1)
                    .take(limit)
                    .map(|(i, l)| format!("{:>6}\t{l}", i + 1))
                    .collect();
                Ok(cut(lines.join("\n")))
            }
            (Kit::Workspace(root), "write") => {
                let path = inside(root, arg("path")?)?;
                let content = arg("content")?;
                if let Some(dir) = path.parent() {
                    std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
                }
                std::fs::write(&path, content)
                    .map_err(|e| format!("{}: {e}", shown(root, &path)))?;
                Ok(format!(
                    "wrote {} ({} bytes)",
                    shown(root, &path),
                    content.len()
                ))
            }
            (Kit::Workspace(root), "edit") => {
                let path = inside(root, arg("path")?)?;
                let (old, new) = (arg("old")?, arg("new")?);
                let text = std::fs::read_to_string(&path)
                    .map_err(|e| format!("{}: {e}", shown(root, &path)))?;
                let count = if old.is_empty() {
                    0
                } else {
                    text.matches(old).count()
                };
                let all = args["all"].as_bool().unwrap_or(false);
                match count {
                    0 => Err(format!(
                        "{}: the text to replace is not there; nothing changed",
                        shown(root, &path)
                    )),
                    n if n > 1 && !all => Err(format!(
                        "{}: the text occurs {n} times; give more context or all: true; nothing changed",
                        shown(root, &path)
                    )),
                    n => {
                        std::fs::write(&path, text.replace(old, new)).map_err(|e| e.to_string())?;
                        Ok(format!("edited {} ({n} replaced)", shown(root, &path)))
                    }
                }
            }
            (Kit::Workspace(root), "glob") => {
                let pattern = arg("pattern")?;
                let mut found: Vec<(std::time::SystemTime, String)> = files(root, root)
                    .into_iter()
                    .filter_map(|p| {
                        let rel = shown(root, &p);
                        glob(pattern, &rel).then(|| {
                            let t = std::fs::metadata(&p)
                                .and_then(|m| m.modified())
                                .unwrap_or(std::time::UNIX_EPOCH);
                            (t, rel)
                        })
                    })
                    .collect();
                found.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
                if found.is_empty() {
                    return Ok("no files match".into());
                }
                Ok(cut(found
                    .into_iter()
                    .map(|(_, p)| p)
                    .collect::<Vec<_>>()
                    .join("\n")))
            }
            (Kit::Workspace(root), "grep") => {
                let re = regex::Regex::new(arg("pattern")?).map_err(|e| e.to_string())?;
                let start = match args["path"].as_str() {
                    Some(p) => inside(root, p)?,
                    None => root.clone(),
                };
                let only = args["glob"].as_str();
                let mut out = String::new();
                for path in if start.is_file() {
                    vec![start]
                } else {
                    files(root, &start)
                } {
                    let rel = shown(root, &path);
                    if only.is_some_and(|g| !glob(g, &rel) && !glob(g, file_name(&rel))) {
                        continue;
                    }
                    let Ok(text) = std::fs::read_to_string(&path) else {
                        continue;
                    };
                    for (i, line) in text.lines().enumerate() {
                        if re.is_match(line) {
                            out.push_str(&format!("{rel}:{}:{line}\n", i + 1));
                            if out.len() > MAX_CHARS {
                                return Ok(cut(out));
                            }
                        }
                    }
                }
                Ok(if out.is_empty() {
                    "no matches".into()
                } else {
                    out
                })
            }
            (Kit::Workspace(root), "bash") => bash(root, arg("command")?, BASH_TIMEOUT).await,
            (Kit::Fetch, "fetch") => fetch(arg("url")?).await,
            _ => Err(format!("no tool {tool}")),
        }
    }
}

/// `path` inside `root` (relative to it, or absolute within it), symlinks resolved: `..`, an
/// absolute path elsewhere or a link out of the root is refused.
pub fn inside(root: &Path, path: &str) -> Result<PathBuf, String> {
    let root = root
        .canonicalize()
        .map_err(|e| format!("{}: {e}", root.display()))?;
    let given = Path::new(path);
    let joined = if given.is_absolute() {
        given.to_path_buf()
    } else {
        root.join(given)
    };
    // Resolve the deepest part that exists; the rest must not climb.
    let mut existing = joined.as_path();
    let mut rest = Vec::new();
    while !existing.exists() {
        rest.push(existing.file_name().ok_or("not a path")?.to_owned());
        existing = existing.parent().ok_or("not a path")?;
    }
    let mut resolved = existing.canonicalize().map_err(|e| e.to_string())?;
    for part in rest.iter().rev() {
        if Path::new(part)
            .components()
            .any(|c| !matches!(c, Component::Normal(_)))
        {
            return Err(format!("{path} is outside the workspace"));
        }
        resolved.push(part);
    }
    if resolved.starts_with(&root) {
        Ok(resolved)
    } else {
        Err(format!("{path} is outside the workspace"))
    }
}

fn shown(root: &Path, path: &Path) -> String {
    let root = root.canonicalize().unwrap_or_else(|_| root.to_path_buf());
    path.strip_prefix(&root)
        .unwrap_or(path)
        .display()
        .to_string()
}

fn file_name(rel: &str) -> &str {
    rel.rsplit('/').next().unwrap_or(rel)
}

/// The files under `dir` (not following links; skipping `.git`, `node_modules`, `target`).
fn files(root: &Path, dir: &Path) -> Vec<PathBuf> {
    let root = root.canonicalize().unwrap_or_else(|_| root.to_path_buf());
    let mut out = Vec::new();
    let mut stack = vec![dir.canonicalize().unwrap_or_else(|_| dir.to_path_buf())];
    while let Some(d) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&d) else {
            continue;
        };
        for e in entries.flatten() {
            let Ok(kind) = e.file_type() else { continue };
            let path = e.path();
            if kind.is_dir() {
                if !SKIP.iter().any(|s| e.file_name() == *s) {
                    stack.push(path);
                }
            } else if kind.is_file() && path.starts_with(&root) {
                out.push(path);
                if out.len() >= 100_000 {
                    return out;
                }
            }
        }
    }
    out
}

/// A glob over `/`-separated paths: `*` and `?` within a part, `**` over parts.
pub fn glob(pattern: &str, path: &str) -> bool {
    fn part(p: &[u8], s: &[u8]) -> bool {
        match (p.first(), s.first()) {
            (None, None) => true,
            (Some(b'*'), _) => part(&p[1..], s) || (!s.is_empty() && part(p, &s[1..])),
            (Some(b'?'), Some(_)) => part(&p[1..], &s[1..]),
            (Some(a), Some(b)) if a == b => part(&p[1..], &s[1..]),
            _ => false,
        }
    }
    fn parts(p: &[&str], s: &[&str]) -> bool {
        match (p.first(), s.first()) {
            (None, None) => true,
            (Some(&"**"), _) => parts(&p[1..], s) || (!s.is_empty() && parts(p, &s[1..])),
            (Some(a), Some(b)) => part(a.as_bytes(), b.as_bytes()) && parts(&p[1..], &s[1..]),
            _ => false,
        }
    }
    let p: Vec<&str> = pattern.trim_start_matches("./").split('/').collect();
    let s: Vec<&str> = path.split('/').collect();
    parts(&p, &s)
}

/// At most `MAX_CHARS` characters, saying so when cut.
fn cut(mut text: String) -> String {
    if text.chars().count() > MAX_CHARS {
        let end = text
            .char_indices()
            .nth(MAX_CHARS)
            .map_or(text.len(), |(i, _)| i);
        text.truncate(end);
        text.push_str(&format!("\n[cut at {MAX_CHARS} characters]"));
    }
    text
}

async fn bash(root: &Path, command: &str, limit: Duration) -> Result<String, String> {
    let child = tokio::process::Command::new("/bin/sh")
        .arg("-c")
        .arg(command)
        .current_dir(root)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .kill_on_drop(true)
        .process_group(0)
        .spawn()
        .map_err(|e| e.to_string())?;
    let pid = child.id();
    match tokio::time::timeout(limit, child.wait_with_output()).await {
        Ok(Ok(out)) => {
            let mut text = String::from_utf8_lossy(&out.stdout).into_owned();
            text.push_str(&String::from_utf8_lossy(&out.stderr));
            let code = out
                .status
                .code()
                .map_or("a signal".to_string(), |c| c.to_string());
            let text = format!("{}\n[exit {code}]", cut(text).trim_end());
            if out.status.success() {
                Ok(text)
            } else {
                Err(text)
            }
        }
        Ok(Err(e)) => Err(e.to_string()),
        Err(_) => {
            if let Some(pid) = pid {
                // The whole group: the command and what it started.
                let _ = std::process::Command::new("/bin/kill")
                    .args(["-9", &format!("-{pid}")])
                    .stderr(std::process::Stdio::null())
                    .status();
            }
            Err(format!("stopped after {} s", limit.as_secs()))
        }
    }
}

async fn fetch(url: &str) -> Result<String, String> {
    let parsed = reqwest::Url::parse(url).map_err(|e| format!("{url}: {e}"))?;
    if !matches!(parsed.scheme(), "http" | "https") {
        return Err("only http and https URLs".into());
    }
    let resp = reqwest::Client::builder()
        .timeout(Duration::from_secs(30))
        .user_agent(concat!("mrun/", env!("CARGO_PKG_VERSION")))
        .build()
        .map_err(|e| e.to_string())?
        .get(parsed)
        .send()
        .await
        .map_err(|e| e.without_url().to_string())?;
    let status = resp.status();
    let kind = resp
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_ascii_lowercase();
    let textual = kind.is_empty()
        || kind.starts_with("text/")
        || ["json", "xml", "javascript", "yaml", "toml"]
            .iter()
            .any(|t| kind.contains(t));
    if !textual {
        return Err(format!("{kind} is not text"));
    }
    let mut body = Vec::new();
    let mut stream = resp.bytes_stream();
    use futures_util::StreamExt;
    while let Some(chunk) = stream.next().await {
        body.extend_from_slice(&chunk.map_err(|e| e.to_string())?);
        if body.len() >= MAX_FETCH {
            body.truncate(MAX_FETCH);
            break;
        }
    }
    let raw = String::from_utf8_lossy(&body);
    let text = if kind.contains("html") || (kind.is_empty() && raw.contains("<html")) {
        readable(&raw)
    } else {
        raw.into_owned()
    };
    let text = cut(text);
    if status.is_success() {
        Ok(text)
    } else {
        Err(format!("HTTP {}\n{text}", status.as_u16()))
    }
}

/// The readable text of an HTML page: no scripts, styles or tags; paragraphs kept.
pub fn readable(html: &str) -> String {
    let lower = html.to_ascii_lowercase();
    let mut out = String::new();
    let mut i = 0;
    let bytes = html.as_bytes();
    while i < bytes.len() {
        if bytes[i] == b'<' {
            let rest = &lower[i..];
            let skip = ["script", "style", "noscript", "svg", "head"]
                .into_iter()
                .find(|t| {
                    rest[1..].starts_with(t)
                        && rest[1 + t.len()..].starts_with(['>', ' ', '\n', '\t'])
                });
            let end = match skip {
                Some(t) => rest
                    .find(&format!("</{t}"))
                    .and_then(|e| rest[e..].find('>').map(|g| e + g)),
                None => rest.find('>'),
            };
            let Some(end) = end else { break };
            let tag = &rest[1..end.min(rest.len())];
            if [
                "p", "br", "div", "li", "h1", "h2", "h3", "h4", "tr", "/p", "/div", "/li", "pre",
                "/pre",
            ]
            .iter()
            .any(|t| tag == *t || tag.starts_with(&format!("{t} ")))
            {
                out.push('\n');
            }
            i += end + 1;
        } else {
            let next = html[i..].find('<').map_or(html.len(), |n| i + n);
            out.push_str(&html[i..next]);
            i = next;
        }
    }
    let decoded = out
        .replace("&nbsp;", " ")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&#39;", "'")
        .replace("&amp;", "&");
    let mut lines: Vec<String> = Vec::new();
    for line in decoded.lines() {
        let line = line.split_whitespace().collect::<Vec<_>>().join(" ");
        if !line.is_empty() {
            lines.push(line);
        }
    }
    lines.join("\n").trim().to_string()
}

/// Serves `kit` on stdin and stdout until stdin closes.
pub async fn run(kit: Kit) -> std::io::Result<()> {
    let mut lines = BufReader::new(tokio::io::stdin()).lines();
    let mut out = tokio::io::stdout();
    while let Some(line) = lines.next_line().await? {
        let Ok(msg) = serde_json::from_str::<Value>(&line) else {
            continue;
        };
        let Some(reply) = answer(&kit, &msg).await else {
            continue;
        };
        out.write_all(format!("{reply}\n").as_bytes()).await?;
        out.flush().await?;
    }
    Ok(())
}

/// The answer to one JSON-RPC message (none for notifications).
pub async fn answer(kit: &Kit, msg: &Value) -> Option<Value> {
    let id = msg.get("id")?.clone();
    let result = match msg["method"].as_str().unwrap_or("") {
        "initialize" => json!({
            "protocolVersion": msg["params"]["protocolVersion"].as_str().unwrap_or("2025-06-18"),
            "capabilities": {"tools": {}},
            "serverInfo": {"name": format!("mrun-{}", kit.name()), "version": env!("CARGO_PKG_VERSION")},
        }),
        "ping" => json!({}),
        "tools/list" => json!({"tools": kit.tools()}),
        "tools/call" => {
            let name = msg["params"]["name"].as_str().unwrap_or("");
            let (text, is_error) = match kit.call(name, &msg["params"]["arguments"]).await {
                Ok(t) => (t, false),
                Err(t) => (t, true),
            };
            json!({"content": [{"type": "text", "text": text}], "isError": is_error})
        }
        other => {
            return Some(
                json!({"jsonrpc": "2.0", "id": id, "error": {"code": -32601, "message": format!("no method {other}")}}),
            );
        }
    };
    Some(json!({"jsonrpc": "2.0", "id": id, "result": result}))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kit() -> (tempfile::TempDir, Kit) {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join("src")).unwrap();
        std::fs::write(dir.path().join("src/a.rs"), "fn a() {}\nfn b() {}\n").unwrap();
        let kit = Kit::Workspace(dir.path().to_path_buf());
        (dir, kit)
    }

    #[tokio::test]
    async fn read_write_edit_inside_the_root() {
        let (dir, kit) = kit();
        let got = kit
            .call("read", &json!({"path": "src/a.rs"}))
            .await
            .unwrap();
        assert_eq!(got, "     1\tfn a() {}\n     2\tfn b() {}");
        kit.call("write", &json!({"path": "new/b.txt", "content": "x x"}))
            .await
            .unwrap();
        assert_eq!(
            std::fs::read_to_string(dir.path().join("new/b.txt")).unwrap(),
            "x x"
        );
        let twice = kit
            .call(
                "edit",
                &json!({"path": "new/b.txt", "old": "x", "new": "y"}),
            )
            .await
            .unwrap_err();
        assert!(twice.contains("2 times"), "{twice}");
        let missing = kit
            .call(
                "edit",
                &json!({"path": "new/b.txt", "old": "z", "new": "y"}),
            )
            .await
            .unwrap_err();
        assert!(missing.contains("not there"));
        assert_eq!(
            std::fs::read_to_string(dir.path().join("new/b.txt")).unwrap(),
            "x x",
            "nothing changed"
        );
        kit.call(
            "edit",
            &json!({"path": "new/b.txt", "old": "x", "new": "y", "all": true}),
        )
        .await
        .unwrap();
        assert_eq!(
            std::fs::read_to_string(dir.path().join("new/b.txt")).unwrap(),
            "y y"
        );
    }

    #[tokio::test]
    async fn paths_outside_the_root_are_refused() {
        let (dir, kit) = kit();
        let other = tempfile::tempdir().unwrap();
        std::fs::write(other.path().join("secret"), "s").unwrap();
        std::os::unix::fs::symlink(other.path(), dir.path().join("link")).unwrap();
        for path in [
            "../x",
            "src/../../x",
            "link/secret",
            other.path().join("secret").to_str().unwrap(),
        ] {
            let err = kit.call("read", &json!({"path": path})).await.unwrap_err();
            assert!(err.contains("outside the workspace"), "{path}: {err}");
            let err = kit
                .call("write", &json!({"path": path, "content": "x"}))
                .await
                .unwrap_err();
            assert!(err.contains("outside the workspace"), "{path}: {err}");
        }
        assert_eq!(
            std::fs::read_to_string(other.path().join("secret")).unwrap(),
            "s"
        );
        let abs = dir.path().join("src/a.rs");
        assert!(
            kit.call("read", &json!({"path": abs.to_str().unwrap()}))
                .await
                .is_ok(),
            "absolute inside is fine"
        );
    }

    #[tokio::test]
    async fn glob_grep_and_bash() {
        let (dir, kit) = kit();
        std::fs::create_dir_all(dir.path().join("target")).unwrap();
        std::fs::write(dir.path().join("target/x.rs"), "fn a() {}").unwrap();
        assert_eq!(
            kit.call("glob", &json!({"pattern": "**/*.rs"}))
                .await
                .unwrap(),
            "src/a.rs"
        );
        assert_eq!(
            kit.call("grep", &json!({"pattern": "fn b"})).await.unwrap(),
            "src/a.rs:2:fn b() {}\n"
        );
        assert_eq!(
            kit.call("grep", &json!({"pattern": "fn", "glob": "*.md"}))
                .await
                .unwrap(),
            "no matches"
        );
        let out = kit
            .call("bash", &json!({"command": "ls src && pwd"}))
            .await
            .unwrap();
        assert!(
            out.starts_with("a.rs\n") && out.ends_with("[exit 0]"),
            "{out}"
        );
        let failed = kit
            .call("bash", &json!({"command": "echo no >&2; exit 3"}))
            .await
            .unwrap_err();
        assert_eq!(failed, "no\n[exit 3]");
        let slow = bash(dir.path(), "sleep 5", Duration::from_millis(200))
            .await
            .unwrap_err();
        assert!(slow.starts_with("stopped after"));
        let long = kit
            .call("bash", &json!({"command": "yes | head -c 100000"}))
            .await
            .unwrap();
        assert!(long.contains("[cut at 30000 characters]"));
    }

    #[test]
    fn globs() {
        assert!(glob("**/*.rs", "a.rs"));
        assert!(glob("**/*.rs", "src/x/a.rs"));
        assert!(glob("src/*", "src/a.rs"));
        assert!(!glob("src/*", "src/x/a.rs"));
        assert!(glob("?.md", "a.md"));
        assert!(!glob("*.rs", "src/a.rs"));
    }

    #[test]
    fn html_becomes_readable_text() {
        let html = "<html><head><title>t</title><style>p{}</style></head><body><h1>Hi</h1><script>x()</script><p>a &amp; b</p><p>c</p></body></html>";
        assert_eq!(readable(html), "Hi\na & b\nc");
    }

    #[tokio::test]
    async fn fetch_reads_a_local_page() {
        let app = axum::Router::new()
            .route(
                "/",
                axum::routing::get(|| async { axum::response::Html("<p>hello</p>") }),
            )
            .route(
                "/bin",
                axum::routing::get(|| async { ([("content-type", "image/png")], vec![0u8; 4]) }),
            );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await });
        assert_eq!(
            Kit::Fetch
                .call("fetch", &json!({"url": format!("http://{addr}/")}))
                .await
                .unwrap(),
            "hello"
        );
        let bin = Kit::Fetch
            .call("fetch", &json!({"url": format!("http://{addr}/bin")}))
            .await
            .unwrap_err();
        assert!(bin.contains("not text"));
        assert!(
            Kit::Fetch
                .call("fetch", &json!({"url": "file:///etc/passwd"}))
                .await
                .is_err()
        );
    }

    #[tokio::test]
    async fn speaks_mcp() {
        let (_dir, kit) = kit();
        let init = answer(&kit, &json!({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-06-18"}})).await.unwrap();
        assert_eq!(init["result"]["serverInfo"]["name"], "mrun-workspace");
        assert!(
            answer(
                &kit,
                &json!({"jsonrpc": "2.0", "method": "notifications/initialized"})
            )
            .await
            .is_none()
        );
        let list = answer(
            &kit,
            &json!({"jsonrpc": "2.0", "id": 2, "method": "tools/list"}),
        )
        .await
        .unwrap();
        assert_eq!(list["result"]["tools"].as_array().unwrap().len(), 6);
        let call = answer(&kit, &json!({"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "read", "arguments": {"path": "../x"}}})).await.unwrap();
        assert_eq!(call["result"]["isError"], true);
    }
}
