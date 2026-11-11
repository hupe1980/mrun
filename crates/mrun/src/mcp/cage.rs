//! The sandbox local MCP servers run in (028, FR-009): procage's Seatbelt cage on macOS, from
//! the server's `sandbox` block. Without a block: its folder read-only, network only through the
//! cage's proxy to package registries; secrets and persistence locations stay denied by
//! procage's base profile. Denials come back with procage's suggested change for one-click allow.

use super::config::{Allowed, Sandbox};
use serde::Serialize;
use std::path::{Path, PathBuf};

/// A denial the sandbox reported, with the change that would allow it.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Denial {
    pub text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fix: Option<Fix>,
    /// The fix as a button: "Allow writing /x".
    #[serde(skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
}

/// What "Allow …" adds to the server's block.
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(untagged)]
pub enum Fix {
    Path(Allowed),
    Host { host: String },
}

impl Fix {
    /// The CLI command this fix is (constitution VII).
    pub fn command(&self, server: &str) -> String {
        match self {
            Fix::Host { host } => format!("mrun mcp allow {server} --host {host}"),
            Fix::Path(a) => format!(
                "mrun mcp allow {server} --{} '{}'",
                a.access,
                a.path.display()
            ),
        }
    }

    /// The button's label: "Allow writing /x", "Allow api.github.com".
    pub fn label(&self) -> String {
        match self {
            Fix::Path(a) => {
                let verb = match a.access.as_str() {
                    "write" => "writing",
                    "exec" => "running",
                    _ => "reading",
                };
                format!("Allow {verb} {}", a.path.display())
            }
            Fix::Host { host } => format!("Allow {host}"),
        }
    }
}

/// A started server process: caged, or uncaged when its block says `"sandbox": false`.
#[derive(Debug)]
pub enum Proc {
    #[cfg(target_os = "macos")]
    Caged(
        Box<procage::native::AsyncChild>,
        Box<procage::policy::Policy>,
    ),
    Open(tokio::process::Child),
}

impl Proc {
    pub fn child(&mut self) -> &mut tokio::process::Child {
        match self {
            #[cfg(target_os = "macos")]
            Proc::Caged(child, _) => child,
            Proc::Open(child) => child,
        }
    }

    /// The denials so far, each once, with a fix where one exists.
    pub fn denials(&mut self) -> Vec<Denial> {
        match self {
            #[cfg(target_os = "macos")]
            Proc::Caged(child, policy) => {
                let mut out: Vec<Denial> = Vec::new();
                for v in child.violations().iter() {
                    let fix = fix(policy, v);
                    let denial = Denial {
                        text: v.to_string(),
                        label: fix.as_ref().map(Fix::label),
                        fix,
                    };
                    if !out.contains(&denial) {
                        out.push(denial);
                    }
                }
                out
            }
            Proc::Open(_) => Vec::new(),
        }
    }
}

#[cfg(target_os = "macos")]
fn fix(policy: &procage::policy::Policy, v: &procage::Violation) -> Option<Fix> {
    use procage::policy::{Access, FsTarget};
    if v.denied_by == procage::DeniedBy::Allowlist {
        // "host:port (reason)"
        let target = v.target.as_deref()?;
        let host = target.split(' ').next()?.rsplit_once(':')?.0;
        return Some(Fix::Host { host: host.into() });
    }
    match procage::suggest(policy, v)? {
        procage::Suggestion::Allow(rule) => {
            let path = match &rule.target {
                FsTarget::Literal(p) | FsTarget::Subpath(p) => PathBuf::from(p.to_str()?),
                FsTarget::Glob(_) => return None,
            };
            let access = match rule.access {
                Access::Write => "write",
                Access::Exec => "exec",
                _ => "read",
            };
            Some(Fix::Path(Allowed {
                access: access.into(),
                path,
            }))
        }
        // Weakening changes (protections, hard denies, signals) are never offered by one click.
        _ => None,
    }
}

/// Package-manager caches a launcher needs to write (`npx -y …` installs into `~/.npm`).
#[cfg(target_os = "macos")]
fn caches(program: &Path) -> Vec<PathBuf> {
    let Some(home) = std::env::home_dir() else {
        return Vec::new();
    };
    let name = program.file_name().and_then(|n| n.to_str()).unwrap_or("");
    match name {
        "npx" | "npm" | "pnpm" | "bunx" => vec![home.join(".npm")],
        "uvx" | "uv" => vec![home.join(".cache/uv"), home.join(".local/share/uv")],
        _ => Vec::new(),
    }
}

/// The policy of a block: the folder read-only (or writable), the program itself runnable, the
/// env names passed, the network through the proxy when hosts are listed.
#[cfg(target_os = "macos")]
pub fn policy(
    sandbox: &Sandbox,
    program: &Path,
    env_names: &[String],
) -> Result<procage::policy::Policy, String> {
    use procage::policy::presets::{agent_readonly, agent_workspace};
    use procage::policy::{AbsPath, Access, EnvPolicy, FsRule, FsTarget, Network, Port};
    let abs = |p: &Path| {
        AbsPath::new(p.to_string_lossy().as_bytes()).map_err(|e| format!("{}: {e}", p.display()))
    };
    let root = workspace(sandbox)?;
    let mut policy = if sandbox.write {
        agent_workspace::v3(&abs(&root)?)
    } else {
        agent_readonly::v3(&abs(&root)?)
    };
    if program.is_absolute() {
        policy
            .fs
            .push(FsRule::new(Access::Exec, FsTarget::Literal(abs(program)?)));
    }
    for dir in caches(program) {
        if std::fs::create_dir_all(&dir).is_ok() {
            policy
                .fs
                .push(FsRule::new(Access::Write, FsTarget::Subpath(abs(&dir)?)));
        }
    }
    for a in &sandbox.allow {
        let access = match a.access.as_str() {
            "write" => Access::Write,
            "exec" => Access::Exec,
            _ => Access::Read,
        };
        policy
            .fs
            .push(FsRule::new(access, FsTarget::Literal(abs(&a.path)?)));
    }
    if let EnvPolicy::Allowlist(list) = &mut policy.env {
        list.extend(env_names.iter().cloned());
    }
    policy.net = if sandbox.hosts.is_empty() {
        Network::None
    } else {
        // The cage picks the port; this one is not used (`with_proxy`).
        let port = Port::new(3128).map_err(|e| e.to_string())?;
        Network::Proxy {
            http: Some(port),
            socks: None,
        }
    };
    Ok(policy)
}

/// The folder a caged server works in: its block's `workspace`, else the home folder.
pub fn workspace(sandbox: &Sandbox) -> Result<PathBuf, String> {
    sandbox
        .workspace
        .clone()
        .or_else(std::env::home_dir)
        .ok_or_else(|| "no workspace folder for the sandbox".into())
}

/// Starts `command` in the cage of `sandbox`, or uncaged without one.
pub fn spawn(
    command: &mut tokio::process::Command,
    sandbox: Option<&Sandbox>,
    program: &Path,
    env_names: &[String],
) -> Result<Proc, String> {
    let Some(sandbox) = sandbox else {
        return command.spawn().map(Proc::Open).map_err(|e| e.to_string());
    };
    if !available() {
        return Err(format!("this system cannot sandbox it; {UNCAGED_HINT}"));
    }
    #[cfg(target_os = "macos")]
    {
        use procage::native::{Cage, CageOptions, Grants};
        let policy = policy(sandbox, program, env_names)?;
        let mut options = CageOptions::default().with_violations();
        if !sandbox.hosts.is_empty() {
            let allow =
                procage::proxy::Allowlist::new(&sandbox.hosts).map_err(|e| e.to_string())?;
            options = options.with_proxy(procage::proxy::Proxy::new(allow));
        }
        let cage = Cage::with_options(&policy, &options)
            .map_err(|e| format!("sandbox: {e} ({UNCAGED_HINT})"))?;
        command.current_dir(workspace(sandbox)?);
        let child = cage
            .spawn_async(command, &Grants::new())
            .map_err(|e| format!("sandbox: {e}"))?;
        Ok(Proc::Caged(Box::new(child), Box::new(policy)))
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = (sandbox, program, env_names);
        Err("no sandbox on this system; set \"sandbox\": false to run this server uncaged".into())
    }
}

/// What a refused cage tells the user to do.
const UNCAGED_HINT: &str =
    "set \"sandbox\": false for this server in mcp.json to run it unconfined";

/// Whether this process can cage children, found once: macOS, and not sandboxed itself. Other
/// systems have no cage; their local servers run only with `"sandbox": false`.
pub fn available() -> bool {
    static AVAILABLE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *AVAILABLE.get_or_init(|| {
        #[cfg(target_os = "macos")]
        {
            matches!(procage::native::is_sandboxed(), Ok(false))
        }
        #[cfg(not(target_os = "macos"))]
        {
            false
        }
    })
}

#[cfg(all(test, target_os = "macos"))]
mod tests {
    use super::*;
    use tokio::io::AsyncReadExt;

    /// Runs a script, failing (not hanging) when the sandbox does not answer in 30 s.
    async fn run(sandbox: Option<&Sandbox>, script: &str) -> (String, Proc) {
        tokio::time::timeout(
            std::time::Duration::from_secs(30),
            run_inner(sandbox, script),
        )
        .await
        .expect("the caged script did not finish in 30 s")
    }

    async fn run_inner(sandbox: Option<&Sandbox>, script: &str) -> (String, Proc) {
        let mut cmd = tokio::process::Command::new("/bin/sh");
        cmd.arg("-c")
            .arg(script)
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::null());
        let mut proc = match spawn(&mut cmd, sandbox, Path::new("/bin/sh"), &[]) {
            Ok(p) => p,
            Err(e) if e.contains("capability gaps") => {
                eprintln!("skipped: {e}");
                return (
                    String::from("skipped"),
                    Proc::Open(tokio::process::Command::new("/bin/true").spawn().unwrap()),
                );
            }
            Err(e) => panic!("{e}"),
        };
        let mut out = String::new();
        let child = proc.child();
        child
            .stdout
            .take()
            .unwrap()
            .read_to_string(&mut out)
            .await
            .unwrap();
        child.wait().await.unwrap();
        (out, proc)
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn a_caged_server_writes_only_in_its_workspace() {
        if !available() {
            eprintln!("skipped: this system cannot cage children");
            return;
        }
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().canonicalize().unwrap();
        let outside = tempfile::tempdir_in(std::env::home_dir().unwrap()).unwrap();
        let outside = outside.path().canonicalize().unwrap();
        let sandbox = Sandbox {
            workspace: Some(root.clone()),
            write: true,
            ..Sandbox::default()
        };
        let script = format!(
            "echo in > '{}/a' && echo ok; echo out > '{}/b' 2>/dev/null && echo escaped",
            root.display(),
            outside.display()
        );
        let (out, _) = run(Some(&sandbox), &script).await;
        if out == "skipped" {
            return;
        }
        assert_eq!(out.trim(), "ok");
        assert!(root.join("a").exists());
        assert!(!outside.join("b").exists());

        let readonly = Sandbox {
            workspace: Some(root.clone()),
            ..Sandbox::default()
        };
        let (out, _) = run(
            Some(&readonly),
            &format!(
                "cat '{}/a'; echo x > '{}/c' && echo wrote",
                root.display(),
                root.display()
            ),
        )
        .await;
        assert_eq!(out.trim(), "in");

        let (out, _) = run(
            None,
            &format!("echo out > '{}/b' && echo open", outside.display()),
        )
        .await;
        assert_eq!(out.trim(), "open", "\"sandbox\": false runs uncaged");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn the_proxy_refuses_hosts_off_the_list_and_offers_them() {
        if !available() {
            eprintln!("skipped: this system cannot cage children");
            return;
        }
        let dir = tempfile::tempdir().unwrap();
        let sandbox = Sandbox {
            workspace: Some(dir.path().canonicalize().unwrap()),
            hosts: vec!["registry.npmjs.org".into()],
            ..Sandbox::default()
        };
        let (out, mut proc) = run(
            Some(&sandbox),
            "/usr/bin/curl -s -o /dev/null -w '%{http_code}' https://example.com/ || echo refused",
        )
        .await;
        if out == "skipped" {
            return;
        }
        assert!(out.contains("refused") || out.starts_with("000"), "{out}");
        let denials = proc.denials();
        assert!(
            denials.iter().any(|d| d.fix
                == Some(Fix::Host {
                    host: "example.com".into()
                })),
            "{denials:?}"
        );
    }

    #[test]
    fn labels_name_the_change() {
        let fix = Fix::Path(Allowed {
            access: "write".into(),
            path: "/x/y".into(),
        });
        assert_eq!(fix.label(), "Allow writing /x/y");
        assert_eq!(Fix::Host { host: "a.b".into() }.label(), "Allow a.b");
    }
}
