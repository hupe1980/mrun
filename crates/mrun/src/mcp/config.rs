//! `~/.mrun/mcp.json` (028): the MCP servers mrun's chats may use, in the `mcpServers` format
//! every MCP client reads, plus a `sandbox` block per local server. mrun writes the file only to
//! import other clients' servers, add its own tools or record an allowed path or host, keeping it
//! at mode 0600: env values and
//! headers are secrets and never leave it (no page, log, argv or CLI output).

use serde::Serialize;
use serde_json::{Map, Value, json};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// Hosts the default cage reaches: package registries, so `npx -y …` and `uvx …` can fetch the
/// server they start.
pub const REGISTRIES: [&str; 4] = [
    "registry.npmjs.org",
    "pypi.org",
    "files.pythonhosted.org",
    "github.com",
];

/// One server of `mcp.json`.
#[derive(Clone, Debug, PartialEq)]
pub struct Server {
    pub name: String,
    pub transport: Transport,
    /// `None`: runs uncaged (`"sandbox": false`, or a remote server).
    pub sandbox: Option<Sandbox>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Transport {
    /// A local process speaking JSON-RPC on stdin and stdout.
    Stdio {
        command: String,
        args: Vec<String>,
        env: BTreeMap<String, String>,
        cwd: Option<PathBuf>,
    },
    /// Streamable HTTP.
    Http {
        url: String,
        headers: BTreeMap<String, String>,
    },
}

/// The cage of a local server (procage): its folder, whether it may write there, the hosts it
/// may reach through the cage's proxy, and paths allowed one by one after a denial.
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct Sandbox {
    pub workspace: Option<PathBuf>,
    pub write: bool,
    pub hosts: Vec<String>,
    pub allow: Vec<Allowed>,
}

/// A path the user allowed after the sandbox stopped a server (procage's suggestion).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, serde::Deserialize)]
pub struct Allowed {
    /// `read`, `write` or `exec`.
    pub access: String,
    pub path: PathBuf,
}

/// The file's location under mrun's home.
pub fn path(home: &Path) -> PathBuf {
    home.join("mcp.json")
}

/// The servers of `mcp.json`, in name order; a missing file is no servers. Entries that are
/// neither a command nor a URL are skipped with their name in the error list.
pub fn load(home: &Path) -> (Vec<Server>, Vec<String>) {
    let Ok(text) = std::fs::read_to_string(path(home)) else {
        return (Vec::new(), Vec::new());
    };
    let Ok(file) = serde_json::from_str::<Value>(&text) else {
        return (Vec::new(), vec!["mcp.json is not valid JSON".into()]);
    };
    let mut servers = Vec::new();
    let mut errors = Vec::new();
    for (name, entry) in servers_of(&file) {
        match parse(&name, &entry) {
            Some(server) => servers.push(server),
            None => errors.push(format!("{name}: neither a command nor a url")),
        }
    }
    (servers, errors)
}

fn servers_of(file: &Value) -> Vec<(String, Value)> {
    file["mcpServers"]
        .as_object()
        .map(|m| m.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
        .unwrap_or_default()
}

fn strings(value: &Value) -> BTreeMap<String, String> {
    value
        .as_object()
        .map(|m| {
            m.iter()
                .filter_map(|(k, v)| v.as_str().map(|v| (k.clone(), v.to_string())))
                .collect()
        })
        .unwrap_or_default()
}

/// `~` at the start of a path is the home directory.
pub fn expand(path: &str) -> PathBuf {
    match (path.strip_prefix("~/"), std::env::home_dir()) {
        (Some(rest), Some(home)) => home.join(rest),
        _ if path == "~" => std::env::home_dir().unwrap_or_else(|| PathBuf::from(path)),
        _ => PathBuf::from(path),
    }
}

fn parse(name: &str, entry: &Value) -> Option<Server> {
    if let Some(url) = entry["url"].as_str() {
        return Some(Server {
            name: name.into(),
            transport: Transport::Http {
                url: url.into(),
                headers: strings(&entry["headers"]),
            },
            sandbox: None,
        });
    }
    let command = entry["command"].as_str()?.to_string();
    let cwd = entry["cwd"].as_str().map(expand);
    let sandbox = match &entry["sandbox"] {
        Value::Bool(false) => None,
        block => Some(Sandbox {
            workspace: block["workspace"]
                .as_str()
                .map(expand)
                .or_else(|| cwd.clone()),
            write: block["write"].as_bool().unwrap_or(false),
            hosts: match block["hosts"].as_array() {
                Some(hosts) => hosts
                    .iter()
                    .filter_map(|h| h.as_str().map(str::to_string))
                    .collect(),
                None => REGISTRIES.iter().map(|h| h.to_string()).collect(),
            },
            allow: block["allow"]
                .as_array()
                .map(|a| {
                    a.iter()
                        .filter_map(|r| serde_json::from_value(r.clone()).ok())
                        .collect()
                })
                .unwrap_or_default(),
        }),
    };
    Some(Server {
        name: name.into(),
        transport: Transport::Stdio {
            command,
            args: entry["args"]
                .as_array()
                .map(|a| {
                    a.iter()
                        .filter_map(|v| v.as_str().map(str::to_string))
                        .collect()
                })
                .unwrap_or_default(),
            env: strings(&entry["env"]),
            cwd,
        },
        sandbox,
    })
}

impl Server {
    /// What the page and `mrun mcp ls` show: no env values, no headers, a URL's host only.
    pub fn public(&self) -> Value {
        let mut view = match &self.transport {
            Transport::Stdio {
                command, args, env, ..
            } => json!({
                "kind": "stdio",
                "command": std::iter::once(command.as_str())
                    .chain(args.iter().map(String::as_str))
                    .collect::<Vec<_>>()
                    .join(" "),
                "env": env.keys().collect::<Vec<_>>(),
            }),
            Transport::Http { url, headers } => json!({
                "kind": "http",
                "host": host_of(url),
                "headers": headers.keys().collect::<Vec<_>>(),
            }),
        };
        view["name"] = self.name.clone().into();
        view["sandbox"] = self.sandbox.as_ref().map_or(Value::Bool(false), |s| {
            json!({"workspace": s.workspace, "write": s.write, "hosts": s.hosts, "allow": s.allow})
        });
        view
    }

    /// The host of a remote server (shown on its call cards and the privacy line).
    pub fn host(&self) -> Option<String> {
        match &self.transport {
            Transport::Http { url, .. } => Some(host_of(url)),
            Transport::Stdio { .. } => None,
        }
    }
}

/// The host of `url`, without scheme, port or path.
pub fn host_of(url: &str) -> String {
    reqwest::Url::parse(url)
        .ok()
        .and_then(|u| u.host_str().map(str::to_string))
        .unwrap_or_else(|| url.to_string())
}

/// Reads `mcp.json` as JSON (an empty object when it does not exist).
fn read_file(home: &Path) -> Result<Map<String, Value>, String> {
    match std::fs::read_to_string(path(home)) {
        Ok(text) => match serde_json::from_str::<Value>(&text) {
            Ok(Value::Object(map)) => Ok(map),
            _ => Err("mcp.json is not a JSON object; fix it first".into()),
        },
        Err(_) => Ok(Map::new()),
    }
}

/// Writes `mcp.json` at mode 0600 (it holds secrets).
fn write_file(home: &Path, file: &Map<String, Value>) -> Result<(), String> {
    let text = serde_json::to_string_pretty(file).map_err(|e| e.to_string())?;
    std::fs::create_dir_all(home).map_err(|e| e.to_string())?;
    let target = path(home);
    let temp = target.with_extension("json.tmp");
    std::fs::write(&temp, text + "\n").map_err(|e| e.to_string())?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&temp, std::fs::Permissions::from_mode(0o600))
            .map_err(|e| e.to_string())?;
    }
    std::fs::rename(&temp, &target).map_err(|e| e.to_string())
}

/// Creates an empty `mcp.json` when there is none (for Edit mcp.json).
pub fn ensure(home: &Path) -> Result<(), String> {
    if path(home).is_file() {
        return Ok(());
    }
    let mut file = Map::new();
    file.insert("mcpServers".into(), Value::Object(Map::new()));
    write_file(home, &file)
}

/// Adds `entries` whose names `mcp.json` does not have yet; returns the names added.
pub fn add(home: &Path, entries: Vec<(String, Value)>) -> Result<Vec<String>, String> {
    let mut file = read_file(home)?;
    let servers = file
        .entry("mcpServers")
        .or_insert_with(|| Value::Object(Map::new()));
    let Some(servers) = servers.as_object_mut() else {
        return Err("mcpServers in mcp.json is not an object".into());
    };
    let mut added = Vec::new();
    for (name, entry) in entries {
        if !servers.contains_key(&name) {
            servers.insert(name.clone(), entry);
            added.push(name);
        }
    }
    if !added.is_empty() {
        write_file(home, &file)?;
    }
    Ok(added)
}

/// Adds `allowed` to server `name`'s sandbox (after a denial the user chose to allow), or a host.
pub fn allow(
    home: &Path,
    name: &str,
    allowed: Option<Allowed>,
    host: Option<&str>,
) -> Result<(), String> {
    let mut file = read_file(home)?;
    let entry = file
        .get_mut("mcpServers")
        .and_then(|s| s.get_mut(name))
        .and_then(Value::as_object_mut)
        .ok_or_else(|| format!("no server {name} in mcp.json"))?;
    let block = entry
        .entry("sandbox")
        .or_insert_with(|| Value::Object(Map::new()));
    let Some(block) = block.as_object_mut() else {
        return Err(format!("{name} runs without a sandbox"));
    };
    if let Some(allowed) = allowed {
        let list = block
            .entry("allow")
            .or_insert_with(|| Value::Array(Vec::new()));
        if let Some(list) = list.as_array_mut() {
            let rule = serde_json::to_value(&allowed).map_err(|e| e.to_string())?;
            if !list.contains(&rule) {
                list.push(rule);
            }
        }
    }
    if let Some(host) = host {
        let list = block
            .entry("hosts")
            .or_insert_with(|| Value::Array(REGISTRIES.iter().map(|h| Value::from(*h)).collect()));
        if let Some(list) = list.as_array_mut()
            && !list.iter().any(|h| h == host)
        {
            list.push(host.into());
        }
    }
    write_file(home, &file)
}

/// Where another client keeps its MCP servers.
pub fn client_config(client: &str) -> Option<PathBuf> {
    let home = std::env::home_dir()?;
    match client {
        "claude-code" => Some(home.join(".claude.json")),
        "claude-desktop" => {
            Some(home.join("Library/Application Support/Claude/claude_desktop_config.json"))
        }
        _ => None,
    }
}

/// The servers of another client's config file (its `mcpServers`).
pub fn servers_in(file: &Path) -> Result<Vec<(String, Value)>, String> {
    let text =
        std::fs::read_to_string(file).map_err(|_| format!("{} not found", file.display()))?;
    let value: Value =
        serde_json::from_str(&text).map_err(|e| format!("{}: {e}", file.display()))?;
    Ok(servers_of(&value))
}

/// mrun's own tools (028): the `workspace` tools in `root` (read, write, edit, list, glob, grep,
/// bash; caged, writing allowed there) and `fetch` (web pages; it needs the network, so uncaged,
/// and every call names its host).
pub fn builtin(kind: &str, root: Option<&Path>) -> Option<(String, Value)> {
    match (kind, root) {
        ("workspace", Some(root)) => Some((
            "workspace".into(),
            json!({
                "command": "mrun",
                "args": ["mcp", "serve", "workspace", "--root", root],
                "sandbox": {"workspace": root, "write": true, "hosts": []},
            }),
        )),
        ("fetch", _) => Some((
            "fetch".into(),
            json!({"command": "mrun", "args": ["mcp", "serve", "fetch"], "sandbox": false}),
        )),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write(home: &Path, text: &str) {
        std::fs::write(path(home), text).unwrap();
    }

    #[test]
    fn servers_parse_with_a_default_cage_and_secrets_stay_out_of_the_view() {
        let home = tempfile::tempdir().unwrap();
        write(
            home.path(),
            r#"{"mcpServers": {
                "fs": {"command": "npx", "args": ["-y", "server-fs", "~/notes"], "env": {"TOKEN": "s3cret"}},
                "gh": {"url": "https://api.example.com/mcp", "headers": {"Authorization": "Bearer s3cret"}},
                "open": {"command": "x", "sandbox": false},
                "bad": {"nothing": 1}
            }}"#,
        );
        let (servers, errors) = load(home.path());
        assert_eq!(errors, ["bad: neither a command nor a url"]);
        let fs = servers.iter().find(|s| s.name == "fs").unwrap();
        let cage = fs.sandbox.as_ref().unwrap();
        assert!(!cage.write);
        assert!(cage.hosts.contains(&"registry.npmjs.org".to_string()));
        assert!(
            servers
                .iter()
                .find(|s| s.name == "open")
                .unwrap()
                .sandbox
                .is_none()
        );
        let gh = servers.iter().find(|s| s.name == "gh").unwrap();
        assert_eq!(gh.host().as_deref(), Some("api.example.com"));
        for server in &servers {
            let view = server.public().to_string();
            assert!(!view.contains("s3cret"), "{view}");
        }
        assert_eq!(fs.public()["env"], json!(["TOKEN"]));
    }

    #[test]
    fn import_adds_missing_names_only_at_mode_0600_and_allow_widens_one_server() {
        let home = tempfile::tempdir().unwrap();
        write(
            home.path(),
            r#"{"mcpServers": {"fs": {"command": "mine"}}}"#,
        );
        let other = home.path().join("claude.json");
        std::fs::write(
            &other,
            r#"{"mcpServers": {"fs": {"command": "theirs"}, "git": {"command": "uvx", "args": ["mcp-server-git"]}}}"#,
        )
        .unwrap();
        let added = add(home.path(), servers_in(&other).unwrap()).unwrap();
        assert_eq!(added, ["git"]);
        let (servers, _) = load(home.path());
        assert_eq!(servers.len(), 2);
        assert!(
            matches!(&servers[0].transport, Transport::Stdio { command, .. } if command == "mine")
        );
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = std::fs::metadata(path(home.path()))
                .unwrap()
                .permissions()
                .mode();
            assert_eq!(mode & 0o777, 0o600);
        }
        let rule = Allowed {
            access: "write".into(),
            path: "/tmp/x".into(),
        };
        allow(
            home.path(),
            "git",
            Some(rule.clone()),
            Some("api.github.com"),
        )
        .unwrap();
        let (servers, _) = load(home.path());
        let cage = servers[1].sandbox.as_ref().unwrap();
        assert_eq!(cage.allow, [rule]);
        assert!(cage.hosts.contains(&"api.github.com".to_string()));
        assert!(
            cage.hosts.contains(&"pypi.org".to_string()),
            "the registries stay"
        );
    }

    #[test]
    fn mruns_own_tools_are_ordinary_entries() {
        let (name, entry) = builtin("workspace", Some(Path::new("/tmp/w"))).unwrap();
        assert_eq!(name, "workspace");
        assert_eq!(entry["sandbox"]["write"], true);
        let (_, fetch) = builtin("fetch", None).unwrap();
        assert_eq!(fetch["sandbox"], false);
        assert!(builtin("workspace", None).is_none());
    }
}
