//! Every action of the window is the CLI's own command, run as a child process (018 research
//! R2): the UI cannot do what the CLI cannot, and "Copy as Command" copies exactly what ran.
//! One function builds the argv for both ([`steps`]), and jobs follow the children.

use serde::Serialize;
use std::collections::HashMap;
use std::process::Stdio;
use std::sync::{Arc, Mutex};
use std::time::Instant;
use tokio::io::{AsyncBufReadExt, AsyncReadExt, BufReader};
use tokio::process::Command;
use tokio::sync::Notify;

/// What the user can do from the window (data model: Action).
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum Kind {
    Pull,
    Use,
    Remove,
    Start,
    Stop,
    Connect,
    Disconnect,
    CloudUp,
    CloudDown,
    CloudLogin,
    CloudLogout,
}

impl Kind {
    pub fn parse(s: &str) -> Option<Self> {
        Some(match s {
            "pull" => Self::Pull,
            "use" => Self::Use,
            "remove" => Self::Remove,
            "start" => Self::Start,
            "stop" => Self::Stop,
            "connect" => Self::Connect,
            "disconnect" => Self::Disconnect,
            "cloud-up" => Self::CloudUp,
            "cloud-down" => Self::CloudDown,
            "cloud-login" => Self::CloudLogin,
            "cloud-logout" => Self::CloudLogout,
            _ => return None,
        })
    }

    /// Whether the action stops or replaces the running server.
    pub fn touches_server(self) -> bool {
        matches!(
            self,
            Self::Use | Self::Stop | Self::Start | Self::CloudUp | Self::CloudDown
        )
    }

    /// The host listing a finished job of this kind makes stale (019 research R9).
    pub fn listing(self) -> &'static str {
        match self {
            Self::Pull | Self::Remove | Self::Use | Self::Start | Self::Stop => "models",
            Self::Connect | Self::Disconnect => "agents",
            Self::CloudUp | Self::CloudDown | Self::CloudLogin | Self::CloudLogout => "cloud",
        }
    }
}

/// The clients `mrun connect` knows.
pub const CLIENTS: [&str; 3] = ["claude-code", "codex", "opencode"];

/// The print-only targets of `mrun connect`, shown as Other apps (020 FR-008).
pub const APPS: [&str; 2] = ["zed", "crush"];

/// A model name or repo id as `pull`, `serve --model` and `rm` take it: never an option, never
/// a path outside the hub's names.
fn valid_model(target: &str) -> bool {
    !target.is_empty()
        && !target.starts_with('-')
        && target.len() <= 200
        && target
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"._/@:-".contains(&b))
}

/// A `cloud-up` target, `<repo>|<gpu id>|<secure|community>` (024 data model): the model, the
/// GPU as `cloud gpus --json` names it, and whether to rent in Community Cloud.
fn cloud_target(target: &str) -> Result<(String, String, bool), String> {
    let refuse = || format!("not a cloud start: {target:?}");
    let mut parts = target.split('|');
    let (Some(repo), Some(gpu), Some(cloud), None) =
        (parts.next(), parts.next(), parts.next(), parts.next())
    else {
        return Err(refuse());
    };
    let gpu_ok = (1..=80).contains(&gpu.len())
        && !gpu.starts_with(['-', ' '])
        && gpu
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b" ._-".contains(&b));
    let community = match cloud {
        "secure" => false,
        "community" => true,
        _ => return Err(refuse()),
    };
    if valid_model(repo) && gpu_ok {
        Ok((repo.to_string(), gpu.to_string(), community))
    } else {
        Err(refuse())
    }
}

/// The commands an action runs, in order (`use` stops the server, then starts it with the
/// model). `last_model` is what Start serves when no target is given.
pub fn steps(
    kind: Kind,
    target: &str,
    server_port: u16,
    last_model: Option<&str>,
) -> Result<Vec<Vec<String>>, String> {
    let port = server_port.to_string();
    let s = |args: &[&str]| args.iter().map(|a| (*a).to_string()).collect::<Vec<_>>();
    let model = |t: &str| {
        if valid_model(t) {
            Ok(t.to_string())
        } else {
            Err(format!("not a model name: {t:?}"))
        }
    };
    let client = |t: &str| {
        if CLIENTS.contains(&t) {
            Ok(t.to_string())
        } else {
            Err(format!("not a client mrun connects: {t:?}"))
        }
    };
    Ok(match kind {
        Kind::Pull => vec![s(&["pull", &model(target)?, "--yes", "--json"])],
        Kind::Use => vec![
            s(&["stop", "--port", &port]),
            s(&[
                "serve",
                "--detach",
                "--model",
                &model(target)?,
                "--port",
                &port,
            ]),
        ],
        Kind::Start => {
            let chosen = if target.is_empty() {
                last_model.map(str::to_string)
            } else {
                Some(model(target)?)
            };
            match chosen {
                Some(m) if valid_model(&m) => {
                    vec![s(&["serve", "--detach", "--model", &m, "--port", &port])]
                }
                _ => vec![s(&["serve", "--detach", "--port", &port])],
            }
        }
        Kind::Stop => vec![s(&["stop", "--port", &port])],
        Kind::Remove => vec![s(&["rm", &model(target)?, "--yes"])],
        Kind::Connect => vec![s(&[
            "connect",
            &client(target)?,
            "--port",
            &port,
            "--write",
            "--yes",
            "--json",
        ])],
        Kind::Disconnect => vec![s(&[
            "connect",
            &client(target)?,
            "--restore",
            "--yes",
            "--json",
        ])],
        // `cloud up` refuses while a local server holds the port: the start replaces it.
        Kind::CloudUp => {
            let (repo, gpu, community) = cloud_target(target)?;
            let mut up = s(&["cloud", "up", "--model", &repo, "--gpu", &gpu]);
            if community {
                up.push("--community".into());
            }
            up.extend(s(&["--port", &port, "--yes", "--json"]));
            vec![s(&["stop", "--port", &port]), up]
        }
        Kind::CloudDown => vec![s(&["cloud", "down", "--port", &port, "--yes"])],
        // The key goes to the child's stdin, never into argv (024 FR-005).
        Kind::CloudLogin => vec![s(&["cloud", "login"])],
        Kind::CloudLogout => vec![s(&["cloud", "logout"])],
    })
}

/// What a cancelled job leaves to undo: a cancelled `cloud-up` may have rented a pod.
pub fn undo(kind: Kind, server_port: u16) -> Option<Vec<String>> {
    (kind == Kind::CloudUp).then(|| {
        steps(Kind::CloudDown, "", server_port, None)
            .unwrap_or_default()
            .concat()
    })
}

/// The Models place's read: every model, marked serving by the window's own server (019 FR-008).
pub fn models_argv(server_port: u16) -> Vec<String> {
    ["ls", "--json", "--all", "--port", &server_port.to_string()]
        .map(str::to_string)
        .to_vec()
}

/// What Add a model shows (020 FR-005): `pull <repo> --dry-run`; the host adds `--json`.
pub fn plan_argv(target: &str) -> Result<Vec<String>, String> {
    if valid_model(target) {
        Ok(["pull", target, "--dry-run"].map(str::to_string).to_vec())
    } else {
        Err(format!("not a model name: {target:?}"))
    }
}

/// What Add shows in the cloud section (024): `cloud models <repo> --json`.
pub fn cloud_model_argv(target: &str) -> Result<Vec<String>, String> {
    if valid_model(target) {
        Ok(["cloud", "models", target, "--json"]
            .map(str::to_string)
            .to_vec())
    } else {
        Err(format!("not a model name: {target:?}"))
    }
}

/// The commands as one shell line, `mrun` first (Copy as Command).
pub fn command_line(steps: &[Vec<String>]) -> String {
    steps
        .iter()
        .map(|argv| {
            std::iter::once("mrun".to_string())
                .chain(argv.iter().map(|a| shell_quote(a)))
                .collect::<Vec<_>>()
                .join(" ")
        })
        .collect::<Vec<_>>()
        .join(" && ")
}

fn shell_quote(a: &str) -> String {
    if a.bytes()
        .all(|b| b.is_ascii_alphanumeric() || b"._/@:=-".contains(&b))
    {
        a.to_string()
    } else {
        format!("'{}'", a.replace('\'', r"'\''"))
    }
}

/// The `mrun` the window runs: this executable.
pub fn cli_program() -> std::path::PathBuf {
    std::env::current_exe().unwrap_or_else(|_| "mrun".into())
}

/// Runs `program argv` to completion and returns its stdout (reads such as `ls --json`).
pub async fn read(program: &std::path::Path, argv: &[&str]) -> Result<String, String> {
    let (ok, stdout, error) = output(program, argv).await?;
    if ok { Ok(stdout) } else { Err(error) }
}

/// Runs `program argv` to completion: whether it succeeded, its stdout, the error line of its
/// stderr (`pull --dry-run --json`, whose refusal is a JSON line on stdout).
pub async fn output(
    program: &std::path::Path,
    argv: &[&str],
) -> Result<(bool, String, String), String> {
    let out = Command::new(program)
        .args(argv)
        .stdin(Stdio::null())
        .output()
        .await
        .map_err(|e| format!("cannot run mrun: {e}"))?;
    Ok((
        out.status.success(),
        String::from_utf8_lossy(&out.stdout).into_owned(),
        last_line(&String::from_utf8_lossy(&out.stderr)),
    ))
}

fn last_line(stderr: &str) -> String {
    let text = stderr.trim();
    let start = text.len().saturating_sub(4096);
    let start = (start..=text.len())
        .find(|&i| text.is_char_boundary(i))
        .unwrap_or(0);
    let tail = &text[start..];
    tail.lines()
        .find(|l| l.trim_start().starts_with("error"))
        .unwrap_or_else(|| tail.lines().last().unwrap_or(""))
        .trim()
        .to_string()
}

/// Download progress of a pull, or the step of a cloud start (data model: Job.progress).
#[derive(Clone, Debug, Default, Serialize, PartialEq)]
pub struct Progress {
    pub file: String,
    pub bytes: u64,
    pub total: u64,
    pub rate: u64,
    pub eta_s: Option<u64>,
    /// `cloud up`'s step (`pod`, `agent`, `loading`, `ready`) and its line.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub step: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum State {
    Running,
    Done,
    Failed,
    Cancelled,
}

/// One action in flight or finished (data model: Job).
#[derive(Clone, Debug, Serialize)]
pub struct Job {
    pub id: String,
    pub kind: Kind,
    pub target: String,
    pub state: State,
    pub progress: Option<Progress>,
    pub error: Option<String>,
    /// The last JSON line the command printed (connect's files, pull's `done`).
    pub output: Option<serde_json::Value>,
    #[serde(skip)]
    started: Instant,
}

/// The window's jobs: one per kind and target at a time, the last [`KEEP`] kept.
#[derive(Debug, Default)]
pub struct Jobs {
    jobs: Mutex<Vec<Job>>,
    cancels: Mutex<HashMap<String, Arc<Notify>>>,
    next: Mutex<u64>,
}

const KEEP: usize = 20;

impl Jobs {
    pub fn list(&self) -> Vec<Job> {
        self.lock().clone()
    }

    pub fn get(&self, id: &str) -> Option<Job> {
        self.lock().iter().find(|j| j.id == id).cloned()
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Vec<Job>> {
        self.jobs
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    fn update(&self, id: &str, f: impl FnOnce(&mut Job)) {
        if let Some(job) = self.lock().iter_mut().find(|j| j.id == id) {
            f(job);
        }
    }

    /// Starts `steps` as a job; `None` when a job of this kind and target is running. `stdin`
    /// goes to the first step's stdin (a secret: never stored in the job).
    pub fn start(
        self: &Arc<Self>,
        program: std::path::PathBuf,
        kind: Kind,
        target: &str,
        steps: Vec<Vec<String>>,
        stdin: Option<String>,
        on_done: impl FnOnce(State) + Send + 'static,
    ) -> Option<Job> {
        let job = {
            let mut jobs = self.lock();
            if jobs
                .iter()
                .any(|j| j.kind == kind && j.target == target && j.state == State::Running)
            {
                return None;
            }
            let mut next = self
                .next
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            *next += 1;
            let job = Job {
                id: format!("j{}", *next),
                kind,
                target: target.to_string(),
                state: State::Running,
                progress: None,
                error: None,
                output: None,
                started: Instant::now(),
            };
            jobs.push(job.clone());
            let excess = jobs.len().saturating_sub(KEEP);
            jobs.drain(..excess);
            job
        };
        let cancel = Arc::new(Notify::new());
        self.cancels
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .insert(job.id.clone(), cancel.clone());
        let this = self.clone();
        let id = job.id.clone();
        tokio::spawn(async move {
            let state = this.run(&program, &id, steps, stdin, kind, &cancel).await;
            this.cancels
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .remove(&id);
            // Before the state shows: a page that sees the job finished reads what it changed.
            on_done(state);
            this.update(&id, |j| j.state = state);
        });
        Some(job)
    }

    /// Cancels a running job (kills its child; a pull resumes next time).
    pub fn cancel(&self, id: &str) -> Option<Job> {
        if let Some(notify) = self
            .cancels
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .get(id)
        {
            notify.notify_one();
        }
        self.get(id)
    }

    /// Runs the steps and returns how they ended; the caller sets the job's state.
    async fn run(
        &self,
        program: &std::path::Path,
        id: &str,
        steps: Vec<Vec<String>>,
        mut stdin: Option<String>,
        kind: Kind,
        cancel: &Notify,
    ) -> State {
        let last = steps.len().saturating_sub(1);
        for (i, argv) in steps.into_iter().enumerate() {
            let input = stdin.take();
            let child = Command::new(program)
                .args(&argv)
                .stdin(if input.is_some() {
                    Stdio::piped()
                } else {
                    Stdio::null()
                })
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .kill_on_drop(true)
                .spawn();
            let mut child = match child {
                Ok(child) => child,
                Err(e) => {
                    self.update(id, |j| j.error = Some(format!("cannot run mrun: {e}")));
                    return State::Failed;
                }
            };
            if let (Some(input), Some(mut pipe)) = (input, child.stdin.take()) {
                use tokio::io::AsyncWriteExt;
                // Dropping the pipe closes it: the child reads to the end.
                let _ = pipe.write_all(format!("{input}\n").as_bytes()).await;
            }
            let stdout = child.stdout.take();
            let mut stderr = child.stderr.take();
            let lines = async {
                if let Some(out) = stdout {
                    let mut lines = BufReader::new(out).lines();
                    while let Ok(Some(line)) = lines.next_line().await {
                        self.observe(id, &line);
                    }
                }
            };
            let err_text = async {
                let mut text = String::new();
                if let Some(err) = stderr.as_mut() {
                    let _ = err.read_to_string(&mut text).await;
                }
                text
            };
            let finished = async {
                let ((), text, status) = tokio::join!(lines, err_text, child.wait());
                (text, status)
            };
            let outcome: Option<(String, std::io::Result<std::process::ExitStatus>)> = tokio::select! {
                done = finished => Some(done),
                () = cancel.notified() => None,
            };
            let Some((text, status)) = outcome else {
                let _ = child.kill().await;
                return State::Cancelled;
            };
            let ok = status.is_ok_and(|s| s.success());
            // `use` and `cloud-up` stop first; nothing to stop is fine.
            let tolerated = matches!(kind, Kind::Use | Kind::CloudUp) && i < last;
            if !ok && !tolerated {
                self.update(id, |j| {
                    j.error = Some(
                        match j.output.as_ref().and_then(|o| o["message"].as_str()) {
                            Some(message) => message.to_string(),
                            None => last_line(&text),
                        },
                    );
                });
                return State::Failed;
            }
        }
        State::Done
    }

    /// A JSON line from the child: pull progress, a cloud start's step, or the command's result.
    fn observe(&self, id: &str, line: &str) {
        let Ok(value) = serde_json::from_str::<serde_json::Value>(line) else {
            return;
        };
        self.update(id, |job| {
            if value["event"] == "progress" {
                let bytes = value["bytes"].as_u64().unwrap_or(0);
                let total = value["total"].as_u64().unwrap_or(0);
                let rate = value["rate"].as_u64().unwrap_or(0);
                job.progress = Some(Progress {
                    file: value["file"].as_str().unwrap_or_default().to_string(),
                    bytes,
                    total,
                    rate,
                    eta_s: (rate > 0).then(|| total.saturating_sub(bytes) / rate),
                    ..Progress::default()
                });
            } else if value["event"] == "step" {
                job.progress = Some(Progress {
                    bytes: value["bytes"].as_u64().unwrap_or(0),
                    total: value["total"].as_u64().unwrap_or(0),
                    step: value["step"].as_str().map(str::to_string),
                    text: value["text"].as_str().map(str::to_string),
                    ..Progress::default()
                });
            } else {
                if value["event"] == "error" {
                    job.error = value["message"].as_str().map(str::to_string);
                }
                job.output = Some(value);
            }
            let _ = job.started;
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn commands_are_the_clis() {
        let line = |k, t| command_line(&steps(k, t, 8484, None).unwrap());
        assert_eq!(
            line(Kind::Pull, "qwen3.6"),
            "mrun pull qwen3.6 --yes --json"
        );
        assert_eq!(
            line(Kind::Use, "mlx-community/Qwen3-8B-4bit"),
            "mrun stop --port 8484 && mrun serve --detach --model mlx-community/Qwen3-8B-4bit --port 8484"
        );
        assert_eq!(
            line(Kind::Connect, "claude-code"),
            "mrun connect claude-code --port 8484 --write --yes --json"
        );
        assert_eq!(
            line(Kind::Disconnect, "codex"),
            "mrun connect codex --restore --yes --json"
        );
        assert_eq!(line(Kind::Start, ""), "mrun serve --detach --port 8484");
        assert_eq!(
            command_line(&steps(Kind::Start, "", 9000, Some("qwen3.5")).unwrap()),
            "mrun serve --detach --model qwen3.5 --port 9000"
        );
        assert_eq!(
            command_line(&[models_argv(9000)]),
            "mrun ls --json --all --port 9000"
        );
        assert_eq!(
            command_line(&[plan_argv("org/name@main").unwrap()]),
            "mrun pull org/name@main --dry-run"
        );
        assert!(plan_argv("--help").is_err());
        assert!(cloud_model_argv("--help").is_err());
        assert_eq!(
            cloud_model_argv("org/m").unwrap().join(" "),
            "cloud models org/m --json"
        );
        assert_eq!(
            line(Kind::CloudUp, "Qwen/Qwen3-4B|NVIDIA RTX A5000|community"),
            "mrun stop --port 8484 && mrun cloud up --model Qwen/Qwen3-4B --gpu 'NVIDIA RTX A5000' --community --port 8484 --yes --json"
        );
        assert_eq!(
            line(Kind::CloudUp, "Qwen/Qwen3-4B|NVIDIA L40S|secure"),
            "mrun stop --port 8484 && mrun cloud up --model Qwen/Qwen3-4B --gpu 'NVIDIA L40S' --port 8484 --yes --json"
        );
        assert_eq!(
            line(Kind::CloudDown, ""),
            "mrun cloud down --port 8484 --yes"
        );
        assert_eq!(line(Kind::CloudLogin, "ignored"), "mrun cloud login");
        assert_eq!(line(Kind::CloudLogout, ""), "mrun cloud logout");
        assert_eq!(
            undo(Kind::CloudUp, 9000).unwrap().join(" "),
            "cloud down --port 9000 --yes"
        );
        assert_eq!(undo(Kind::Pull, 9000), None);
    }

    #[test]
    fn targets_never_become_options_or_unknown_clients() {
        assert!(steps(Kind::Pull, "--help", 8484, None).is_err());
        assert!(steps(Kind::Remove, "a b", 8484, None).is_err());
        assert!(steps(Kind::Use, "x;rm -rf /", 8484, None).is_err());
        assert!(steps(Kind::Connect, "bash", 8484, None).is_err());
        assert!(steps(Kind::Start, "", 8484, Some("--evil")).unwrap()[0].len() == 4);
        for target in [
            "",
            "Qwen/Qwen3-4B",
            "Qwen/Qwen3-4B|L40S",
            "Qwen/Qwen3-4B|L40S|moon",
            "Qwen/Qwen3-4B|L40S|secure|x",
            "--help|L40S|secure",
            "Qwen/Qwen3-4B|--cpu|secure",
            "Qwen/Qwen3-4B|L40S;rm|secure",
        ] {
            assert!(
                steps(Kind::CloudUp, target, 8484, None).is_err(),
                "{target}"
            );
        }
        assert_eq!(Kind::parse("cloud-up"), Some(Kind::CloudUp));
        assert_eq!(
            serde_json::to_value(Kind::CloudLogout).unwrap(),
            "cloud-logout"
        );
    }

    #[test]
    fn progress_lines_become_job_progress() {
        let jobs = Jobs::default();
        jobs.lock().push(Job {
            id: "j1".into(),
            kind: Kind::Pull,
            target: "m".into(),
            state: State::Running,
            progress: None,
            error: None,
            output: None,
            started: Instant::now(),
        });
        jobs.observe(
            "j1",
            r#"{"event":"progress","file":"a","bytes":50,"total":150,"rate":10}"#,
        );
        let p = jobs.get("j1").unwrap().progress.unwrap();
        assert_eq!((p.bytes, p.total, p.eta_s), (50, 150, Some(10)));
        jobs.observe(
            "j1",
            r#"{"event":"error","code":"E030","message":"no network"}"#,
        );
        assert_eq!(jobs.get("j1").unwrap().error.as_deref(), Some("no network"));
    }

    #[test]
    fn step_lines_become_job_progress() {
        let jobs = Jobs::default();
        jobs.lock().push(Job {
            id: "j1".into(),
            kind: Kind::CloudUp,
            target: "m|g|secure".into(),
            state: State::Running,
            progress: None,
            error: None,
            output: None,
            started: Instant::now(),
        });
        jobs.observe(
            "j1",
            r#"{"event":"step","step":"loading","text":"loading: 12 GB","bytes":12,"total":35}"#,
        );
        let p = jobs.get("j1").unwrap().progress.unwrap();
        assert_eq!(
            (p.step.as_deref(), p.text.as_deref(), p.bytes, p.total),
            (Some("loading"), Some("loading: 12 GB"), 12, 35)
        );
        jobs.observe("j1", r#"{"event":"done","model":"m","cost_per_hr":0.86}"#);
        assert_eq!(jobs.get("j1").unwrap().output.unwrap()["cost_per_hr"], 0.86);
    }

    #[tokio::test]
    async fn stdin_reaches_the_child() {
        let jobs = Arc::new(Jobs::default());
        let (tx, rx) = tokio::sync::oneshot::channel();
        let job = jobs
            .start(
                "/bin/cat".into(),
                Kind::CloudLogin,
                "",
                vec![vec![]],
                Some(r#"{"secret":"rpa_X"}"#.into()),
                move |state| {
                    let _ = tx.send(state);
                },
            )
            .unwrap();
        assert_eq!(rx.await.unwrap(), State::Done);
        // `cat` echoed the line: the payload reached the child's stdin.
        let done = jobs.get(&job.id).unwrap();
        assert_eq!(done.output.unwrap()["secret"], "rpa_X");
        assert_eq!(done.target, "");
    }
}
