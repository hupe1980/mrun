//! `mrun connect <client>`: prints the persistent configuration for a client and where it goes,
//! with the running server's values (without a server: this machine's defaults). With `--write`
//! it applies it (D055): plan → diff → backup (`MRUN_HOME/backup/<client>/<stamp>/`) → atomic
//! write → verify (`/health` and `/v1/models`); `--restore` puts the last backup back. `--json`
//! prints the client's state (018 agent entry) with the files `--write` would change, or with
//! `--write`/`--restore` the files it changed.

use crate::cli::clients::{self, Apply, Block, Client, Homes, Target};
use crate::cli::server::{self, Health, ServerInfo};
use crate::errors::{CliError, Code};
use crate::hub::{self, Ctx, aliases};
use serde::{Deserialize, Serialize};
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

#[derive(Debug, clap::Args)]
pub struct ConnectArgs {
    pub client: Target,
    /// Port of the running server.
    #[arg(long, env = "MRUN_PORT", default_value_t = server::default_port())]
    pub port: u16,
    /// Write the configuration (after showing the plan and diff, with a backup) instead of
    /// printing it.
    #[arg(long, conflicts_with = "restore")]
    pub write: bool,
    /// Put back the files the last 'connect --write' changed for this client.
    #[arg(long)]
    pub restore: bool,
    /// Answer yes to the confirmation of --write and --restore.
    #[arg(long, short = 'y')]
    pub yes: bool,
    /// Print the client's state and the files --write would change (with --write or
    /// --restore: the files changed) as JSON.
    #[arg(long)]
    pub json: bool,
}

pub fn run(args: ConnectArgs) -> u8 {
    if args.json {
        return super::exit(run_json(&args));
    }
    let homes = Homes::from_env();
    if args.restore {
        return super::exit(restore(&homes.mrun, args.client, args.yes).map(drop));
    }
    let server = server_or_defaults(args.port);
    let blocks = clients::persistent(args.client, &server, &homes);
    if !args.write {
        for block in &blocks {
            println!("# {} ({})", block.path.display(), block.note);
            println!("{}", block.text);
            println!();
        }
        return 0;
    }
    super::exit(write(&homes.mrun, args.client, &blocks, args.yes).map(|_| {
        verify(args.port);
    }))
}

/// `connect --json`: the state, after `--write` or `--restore` when asked.
fn run_json(args: &ConnectArgs) -> Result<(), CliError> {
    let homes = Homes::from_env();
    let server = server_or_defaults(args.port);
    let blocks = clients::persistent(args.client, &server, &homes);
    let done = if args.restore {
        let restored = restore(&homes.mrun, args.client, args.yes)?;
        Some(
            restored
                .into_iter()
                .map(|path| FileChange {
                    path,
                    change: "restore",
                })
                .collect(),
        )
    } else if args.write {
        let written = write(&homes.mrun, args.client, &blocks, args.yes)?;
        verify(args.port);
        Some(written)
    } else {
        None
    };
    let installed = binary(args.client).and_then(crate::cli::which);
    let version = installed.as_deref().and_then(version_of);
    let mut state = state(args.client, &server, &blocks, installed.is_some(), version);
    if let Some(files) = done {
        state.files = files;
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&state).expect("serializes")
    );
    Ok(())
}

/// One client's state for `connect --json` (018 agent entry).
#[derive(Debug, Eq, PartialEq, Serialize)]
pub struct AgentState {
    pub client: &'static str,
    pub installed: bool,
    pub version: Option<String>,
    /// Every file `--write` manages already holds this server's configuration.
    pub connected: bool,
    /// What `--write` would change, or what `--write`/`--restore` changed.
    pub files: Vec<FileChange>,
    /// Why the client cannot use this server or `--write` cannot apply (print only, a file
    /// mrun does not rewrite, a model with less context than the client's minimum, D055).
    pub refusal: Option<String>,
    /// The blocks `mrun connect <client>` prints, print-only clients included (020 FR-008).
    pub blocks: Vec<BlockText>,
}

/// A printed block: the file it goes in, how to use it, the text.
#[derive(Debug, Eq, PartialEq, Serialize)]
pub struct BlockText {
    pub path: PathBuf,
    pub note: String,
    pub text: String,
}

#[derive(Debug, Eq, PartialEq, Serialize)]
pub struct FileChange {
    pub path: PathBuf,
    /// `create`, `modify` or `restore`.
    pub change: &'static str,
}

/// The state of `target` against `server`'s configuration blocks; `installed` and `version`
/// come from `PATH` in [`run`].
fn state(
    target: Target,
    server: &ServerInfo,
    blocks: &[Block],
    installed: bool,
    version: Option<String>,
) -> AgentState {
    let planned = if target.writable() {
        plan(blocks)
    } else {
        Err(CliError::new(
            Code::InvalidArguments,
            format!(
                "{} is print only: paste the block from 'mrun connect {}'",
                target.name(),
                target.name()
            ),
        ))
    };
    let (files, connected, refusal) = match planned {
        Ok(changes) => {
            let files: Vec<FileChange> = changes
                .iter()
                .filter(|change| change.old.as_deref() != Some(change.new.as_str()))
                .map(|change| FileChange {
                    path: change.path.clone(),
                    change: change.label(),
                })
                .collect();
            let connected = files.is_empty();
            (files, connected, None)
        }
        Err(error) => (Vec::new(), false, Some(error.message)),
    };
    let short = client(target).filter(|client| server.context < client.min_context());
    let refusal = refusal.or_else(|| {
        short.map(|client| {
            format!(
                "{} leaves {} tokens of context; {} needs at least {}",
                server.model_id,
                server.context,
                client.name(),
                client.min_context()
            )
        })
    });
    AgentState {
        client: target.name(),
        installed,
        version,
        connected,
        files,
        refusal,
        blocks: blocks
            .iter()
            .map(|block| BlockText {
                path: block.path.clone(),
                note: block.note.clone(),
                text: block.text.clone(),
            })
            .collect(),
    }
}

/// The launchable client behind a target (Crush and Zed are print only).
fn client(target: Target) -> Option<Client> {
    match target {
        Target::ClaudeCode => Some(Client::ClaudeCode),
        Target::Codex => Some(Client::Codex),
        Target::OpenCode => Some(Client::OpenCode),
        Target::Crush | Target::Zed => None,
    }
}

/// The program on `PATH` that tells whether a target is installed.
fn binary(target: Target) -> Option<&'static str> {
    match target {
        Target::Crush => Some("crush"),
        Target::Zed => Some("zed"),
        other => client(other).map(Client::binary),
    }
}

/// `<binary> --version` as `a.b.c`, else its first line; `None` when it does not answer.
fn version_of(binary: &Path) -> Option<String> {
    let output = std::process::Command::new(binary)
        .arg("--version")
        .stdin(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .output()
        .ok()?;
    let text = String::from_utf8_lossy(&output.stdout);
    match clients::parse_version(&text) {
        Some((a, b, c)) => Some(format!("{a}.{b}.{c}")),
        None => text
            .lines()
            .next()
            .map(str::trim)
            .filter(|line| !line.is_empty())
            .map(str::to_string),
    }
}

/// The running server's values, else this machine's defaults (with a note on stderr).
fn server_or_defaults(port: u16) -> ServerInfo {
    match ServerInfo::fetch(port) {
        Ok(server) => server,
        Err(_) => {
            let defaults = defaults(port);
            eprintln!(
                "note: no mrun server is running on port {}; showing the values for {} on this Mac ({} tokens of context). Start it with 'mrun serve --detach --model {}'.",
                port, defaults.model_id, defaults.context, defaults.model_id
            );
            defaults
        }
    }
}

/// What `connect` shows without a server: this machine's default alias, the context it gets
/// here and the server's default output limit.
fn defaults(port: u16) -> ServerInfo {
    let ctx = Ctx::from_env();
    let alias = aliases::default_alias(ctx.device);
    let name = hub::default_model(&ctx).unwrap_or_else(|_| alias.name.to_string());
    let context = ctx
        .device
        .map(|device| alias.fit(device).max_context)
        .filter(|&context| context > 0)
        .unwrap_or(aliases::DEFAULT_MIN_CONTEXT);
    ServerInfo {
        port,
        model_id: name,
        context,
        max_output: crate::api::run::DEFAULT_MAX_OUTPUT,
        recording: false,
    }
}

/// One file of a write: what is there now and what will be.
#[derive(Debug)]
struct Change {
    path: PathBuf,
    old: Option<String>,
    new: String,
}

impl Change {
    /// `create` or `modify` (018 agent entry).
    fn label(&self) -> &'static str {
        if self.old.is_some() {
            "modify"
        } else {
            "create"
        }
    }
}

/// The files `--write` would change, with their new content.
fn plan(blocks: &[Block]) -> Result<Vec<Change>, CliError> {
    let mut changes = Vec::new();
    for block in blocks {
        let old = match fs::read_to_string(&block.path) {
            Ok(text) => Some(text),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
            Err(error) => return Err(io_error(&block.path, &error)),
        };
        let new = match &block.apply {
            Apply::PrintOnly => {
                return Err(CliError::new(
                    Code::InvalidArguments,
                    format!("{} is print only", block.path.display()),
                )
                .hint("run 'mrun connect' without --write and paste the block yourself"));
            }
            Apply::Replace => block.text.clone(),
            Apply::MergeJson(key) => merge_json(old.as_deref(), &block.text, key, &block.path)?,
        };
        let new = if new.ends_with('\n') {
            new
        } else {
            format!("{new}\n")
        };
        changes.push(Change {
            path: block.path.clone(),
            old,
            new,
        });
    }
    Ok(changes)
}

/// `existing` with the value at `key` replaced by the one in `block`; a new file is the block.
fn merge_json(
    existing: Option<&str>,
    block: &str,
    key: &[&str],
    path: &Path,
) -> Result<String, CliError> {
    let block: serde_json::Value = serde_json::from_str(block).expect("blocks are JSON");
    let Some(existing) = existing.filter(|text| !text.trim().is_empty()) else {
        return Ok(serde_json::to_string_pretty(&block).expect("serializes"));
    };
    let mut root: serde_json::Value = serde_json::from_str(existing).map_err(|error| {
        CliError::new(
            Code::InvalidArguments,
            format!(
                "{} is not plain JSON ({error}); mrun does not rewrite it",
                path.display()
            ),
        )
        .hint("merge the block from 'mrun connect' (without --write) by hand")
    })?;
    let mut value = &block;
    for part in key {
        value = &value[*part];
    }
    let mut slot = &mut root;
    for part in key {
        if !slot.is_object() {
            return Err(CliError::new(
                Code::InvalidArguments,
                format!("{}: \"{part}\" is not an object", path.display()),
            ));
        }
        slot = slot
            .as_object_mut()
            .expect("checked")
            .entry(part.to_string())
            .or_insert_with(|| serde_json::json!({}));
    }
    *slot = value.clone();
    Ok(serde_json::to_string_pretty(&root).expect("serializes"))
}

/// A line diff (longest common subsequence), `-`/`+`/` ` prefixed.
fn diff(old: &str, new: &str) -> String {
    let (a, b): (Vec<&str>, Vec<&str>) = (old.lines().collect(), new.lines().collect());
    let mut lcs = vec![vec![0usize; b.len() + 1]; a.len() + 1];
    for i in (0..a.len()).rev() {
        for j in (0..b.len()).rev() {
            lcs[i][j] = if a[i] == b[j] {
                lcs[i + 1][j + 1] + 1
            } else {
                lcs[i + 1][j].max(lcs[i][j + 1])
            };
        }
    }
    let (mut i, mut j, mut out) = (0, 0, String::new());
    while i < a.len() || j < b.len() {
        if i < a.len() && j < b.len() && a[i] == b[j] {
            out.push_str(&format!("  {}\n", a[i]));
            (i, j) = (i + 1, j + 1);
        } else if i < a.len() && (j == b.len() || lcs[i + 1][j] >= lcs[i][j + 1]) {
            out.push_str(&format!("- {}\n", a[i]));
            i += 1;
        } else {
            out.push_str(&format!("+ {}\n", b[j]));
            j += 1;
        }
    }
    out
}

/// The record of one backup: every file `--write` touched and whether it existed.
#[derive(Debug, Default, Deserialize, Serialize)]
struct Record {
    files: Vec<Saved>,
}

#[derive(Debug, Deserialize, Serialize)]
struct Saved {
    path: PathBuf,
    /// File name of the saved copy inside the backup directory; `None` when the file did not
    /// exist (restore removes it).
    copy: Option<String>,
}

fn backup_root(mrun_home: &Path, target: Target) -> PathBuf {
    mrun_home.join("backup").join(target.name())
}

/// Plan, diff, confirmation, backup, atomic writes. Returns the files written.
fn write(
    mrun_home: &Path,
    target: Target,
    blocks: &[Block],
    yes: bool,
) -> Result<Vec<FileChange>, CliError> {
    if !target.writable() {
        return Err(CliError::new(
            Code::InvalidArguments,
            format!("'connect {} --write' is not supported", target.name()),
        )
        .hint(format!(
            "run 'mrun connect {}' and paste the block yourself",
            target.name()
        )));
    }
    let changes: Vec<Change> = plan(blocks)?
        .into_iter()
        .filter(|change| change.old.as_deref() != Some(change.new.as_str()))
        .collect();
    if changes.is_empty() {
        eprintln!("{} is already configured; nothing to write", target.name());
        return Ok(Vec::new());
    }
    for change in &changes {
        let what = if change.old.is_some() {
            "change"
        } else {
            "create"
        };
        eprintln!("{what} {}:", change.path.display());
        eprint!("{}", diff(change.old.as_deref().unwrap_or(""), &change.new));
    }
    let question = format!(
        "write {} file(s) for {} (backup in {})?",
        changes.len(),
        target.name(),
        backup_root(mrun_home, target).display()
    );
    if !super::confirm(&question, yes)? {
        eprintln!("nothing written");
        return Ok(Vec::new());
    }
    let dir = backup_root(mrun_home, target).join(stamp());
    fs::create_dir_all(&dir).map_err(|e| io_error(&dir, &e))?;
    let mut record = Record::default();
    for (index, change) in changes.iter().enumerate() {
        let copy = change.old.as_ref().map(|old| {
            let name = format!("{index}.orig");
            (name, old)
        });
        if let Some((name, old)) = &copy {
            fs::write(dir.join(name), old).map_err(|e| io_error(&dir, &e))?;
        }
        record.files.push(Saved {
            path: change.path.clone(),
            copy: copy.map(|(name, _)| name),
        });
    }
    let index = dir.join("record.json");
    fs::write(
        &index,
        serde_json::to_string_pretty(&record).expect("serializes"),
    )
    .map_err(|e| io_error(&index, &e))?;
    let mut written = Vec::new();
    for change in &changes {
        atomic_write(&change.path, &change.new)?;
        eprintln!("wrote {}", change.path.display());
        written.push(FileChange {
            path: change.path.clone(),
            change: change.label(),
        });
    }
    eprintln!(
        "backup in {}; 'mrun connect {} --restore' undoes this",
        dir.display(),
        target.name()
    );
    Ok(written)
}

/// Writes through a temporary file in the same directory and a rename.
fn atomic_write(path: &Path, text: &str) -> Result<(), CliError> {
    let dir = path.parent().unwrap_or(Path::new("."));
    fs::create_dir_all(dir).map_err(|e| io_error(dir, &e))?;
    let mut temp = tempfile::NamedTempFile::new_in(dir).map_err(|e| io_error(dir, &e))?;
    temp.write_all(text.as_bytes())
        .map_err(|e| io_error(path, &e))?;
    if let Ok(meta) = fs::metadata(path) {
        let _ = fs::set_permissions(temp.path(), meta.permissions());
    }
    temp.persist(path).map_err(|e| io_error(path, &e.error))?;
    Ok(())
}

/// The newest backup of `target` is put back (files that did not exist are removed) and then
/// deleted. Returns the restored paths.
fn restore(mrun_home: &Path, target: Target, yes: bool) -> Result<Vec<PathBuf>, CliError> {
    let root = backup_root(mrun_home, target);
    let latest = fs::read_dir(&root)
        .into_iter()
        .flatten()
        .flatten()
        .filter(|entry| entry.path().join("record.json").is_file())
        .map(|entry| entry.path())
        .max()
        .ok_or_else(|| {
            CliError::new(
                Code::InvalidArguments,
                format!("no backup of {} to restore", target.name()),
            )
            .hint(format!(
                "backups are made by 'mrun connect {} --write' in {}",
                target.name(),
                root.display()
            ))
        })?;
    let index = latest.join("record.json");
    let record: Record = fs::read_to_string(&index)
        .map_err(|e| io_error(&index, &e))
        .and_then(|text| {
            serde_json::from_str(&text)
                .map_err(|e| CliError::new(Code::Internal, format!("{}: {e}", index.display())))
        })?;
    for saved in &record.files {
        let action = if saved.copy.is_some() {
            "restore"
        } else {
            "remove"
        };
        eprintln!("{action} {}", saved.path.display());
    }
    if !super::confirm(
        &format!("undo the last 'connect {} --write'?", target.name()),
        yes,
    )? {
        eprintln!("nothing restored");
        return Ok(Vec::new());
    }
    let mut restored = Vec::new();
    for saved in &record.files {
        match &saved.copy {
            Some(copy) => {
                // Only names this module wrote are read back.
                if copy.contains(['/', '\\']) || copy.starts_with('.') {
                    return Err(CliError::new(
                        Code::Internal,
                        format!("{}: invalid copy name '{copy}'", index.display()),
                    ));
                }
                let text = fs::read_to_string(latest.join(copy))
                    .map_err(|e| io_error(&latest.join(copy), &e))?;
                atomic_write(&saved.path, &text)?;
            }
            None => match fs::remove_file(&saved.path) {
                Err(error) if error.kind() != std::io::ErrorKind::NotFound => {
                    return Err(io_error(&saved.path, &error));
                }
                _ => {}
            },
        }
        restored.push(saved.path.clone());
    }
    fs::remove_dir_all(&latest).map_err(|e| io_error(&latest, &e))?;
    eprintln!("restored {} file(s) for {}", restored.len(), target.name());
    Ok(restored)
}

/// After a write: the server the configuration points at answers `/health` and `/v1/models`.
fn verify(port: u16) {
    match server::health(port) {
        Health::Ready(_) => match ServerInfo::fetch(port) {
            Ok(info) => eprintln!(
                "verified: the server on port {port} serves {} ({} tokens of context)",
                info.model_id, info.context
            ),
            Err(error) => eprintln!("warning: /v1/models on port {port} failed: {error}"),
        },
        Health::Loading(_) => {
            eprintln!(
                "note: the server on port {port} is still loading; the configuration is written"
            )
        }
        Health::Failed(error) => eprintln!("warning: the server on port {port} failed: {error}"),
        Health::Down => eprintln!(
            "note: no server answers on port {port}; start one with 'mrun serve --detach' before using the client"
        ),
    }
}

/// Sortable backup directory name: seconds and nanoseconds since the epoch.
fn stamp() -> String {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    format!("{:012}-{:09}", now.as_secs(), now.subsec_nanos())
}

fn io_error(path: &Path, error: &dyn std::fmt::Display) -> CliError {
    CliError::new(
        Code::Internal,
        format!("cannot access {}: {error}", path.display()),
    )
    .hint("check the permissions of that directory")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn server() -> ServerInfo {
        ServerInfo {
            port: 8484,
            model_id: "org/m".into(),
            context: 70_000,
            max_output: 16_384,
            recording: false,
        }
    }

    fn homes(root: &Path) -> Homes {
        Homes {
            mrun: root.join("mrun"),
            codex: root.join("codex"),
            config: root.join("config"),
        }
    }

    #[test]
    fn write_backs_up_merges_and_restore_undoes_it() {
        let root = tempfile::tempdir().unwrap();
        let homes = homes(root.path());
        let opencode = homes.config.join("opencode/opencode.json");
        fs::create_dir_all(opencode.parent().unwrap()).unwrap();
        let mine = "{\"theme\": \"dark\", \"provider\": {\"other\": {\"name\": \"x\"}}}\n";
        fs::write(&opencode, mine).unwrap();

        let blocks = clients::persistent(Target::OpenCode, &server(), &homes);
        write(&homes.mrun, Target::OpenCode, &blocks, true).unwrap();
        let merged: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(&opencode).unwrap()).unwrap();
        assert_eq!(merged["theme"], "dark");
        assert_eq!(merged["provider"]["other"]["name"], "x");
        assert_eq!(
            merged["provider"]["mrun"]["models"]["org/m"]["limit"]["context"],
            70_000
        );
        // Unchanged: nothing written, no second backup.
        write(&homes.mrun, Target::OpenCode, &blocks, true).unwrap();
        let backups = fs::read_dir(backup_root(&homes.mrun, Target::OpenCode))
            .unwrap()
            .count();
        assert_eq!(backups, 1);

        let restored = restore(&homes.mrun, Target::OpenCode, true).unwrap();
        assert_eq!(restored, std::slice::from_ref(&opencode));
        assert_eq!(fs::read_to_string(&opencode).unwrap(), mine);
        assert!(restore(&homes.mrun, Target::OpenCode, true).is_err());
    }

    #[test]
    fn restore_removes_files_that_did_not_exist() {
        let root = tempfile::tempdir().unwrap();
        let homes = homes(root.path());
        let blocks = clients::persistent(Target::Codex, &server(), &homes);
        write(&homes.mrun, Target::Codex, &blocks, true).unwrap();
        let profile = homes.codex.join("mrun.config.toml");
        assert!(
            fs::read_to_string(&profile)
                .unwrap()
                .contains("model = \"org/m\"")
        );
        restore(&homes.mrun, Target::Codex, true).unwrap();
        assert!(!profile.exists());
        assert!(!homes.mrun.join("codex/models.json").exists());
    }

    #[test]
    fn print_only_and_non_json_targets_are_refused() {
        let root = tempfile::tempdir().unwrap();
        let homes = homes(root.path());
        for target in [Target::Crush, Target::Zed] {
            let blocks = clients::persistent(target, &server(), &homes);
            let error = write(&homes.mrun, target, &blocks, true).unwrap_err();
            assert_eq!(error.code, Code::InvalidArguments);
        }
        let opencode = homes.config.join("opencode/opencode.json");
        fs::create_dir_all(opencode.parent().unwrap()).unwrap();
        fs::write(&opencode, "{ // a comment\n}").unwrap();
        let blocks = clients::persistent(Target::OpenCode, &server(), &homes);
        let error = write(&homes.mrun, Target::OpenCode, &blocks, true).unwrap_err();
        assert!(error.message.contains("not plain JSON"), "{error}");
        assert_eq!(fs::read_to_string(&opencode).unwrap(), "{ // a comment\n}");
    }

    /// T033: `connect --json`'s state in a temporary home: the planned files, connected after
    /// `--write`, back to planned after `--restore`, and the refusals.
    #[test]
    fn json_state_shows_planned_written_and_restored_files() {
        let root = tempfile::tempdir().unwrap();
        let homes = homes(root.path());
        let blocks = clients::persistent(Target::Codex, &server(), &homes);
        let profile = homes.codex.join("mrun.config.toml");
        let before = state(
            Target::Codex,
            &server(),
            &blocks,
            true,
            Some("0.50.0".into()),
        );
        assert_eq!(
            serde_json::to_value(&before).unwrap(),
            serde_json::json!({
                "client": "codex",
                "installed": true,
                "version": "0.50.0",
                "connected": false,
                "files": [{"path": profile, "change": "create"}],
                "refusal": null,
                "blocks": blocks
                    .iter()
                    .map(|b| serde_json::json!({"path": b.path, "note": b.note, "text": b.text}))
                    .collect::<Vec<_>>(),
            })
        );
        assert!(!profile.exists(), "the state writes nothing");

        let written = write(&homes.mrun, Target::Codex, &blocks, true).unwrap();
        assert_eq!(
            written,
            [FileChange {
                path: profile.clone(),
                change: "create"
            }]
        );
        let after = state(Target::Codex, &server(), &blocks, false, None);
        assert!(after.connected && after.files.is_empty(), "{after:?}");
        // Another model: the profile would be modified.
        let other = ServerInfo {
            model_id: "org/other".into(),
            ..server()
        };
        let moved = state(
            Target::Codex,
            &other,
            &clients::persistent(Target::Codex, &other, &homes),
            false,
            None,
        );
        assert!(!moved.connected);
        assert_eq!(moved.files[0].change, "modify");

        restore(&homes.mrun, Target::Codex, true).unwrap();
        assert_eq!(
            state(Target::Codex, &server(), &blocks, false, None),
            before_without_version(before)
        );

        let zed = state(
            Target::Zed,
            &server(),
            &clients::persistent(Target::Zed, &server(), &homes),
            false,
            None,
        );
        assert!(zed.files.is_empty() && !zed.connected);
        assert!(zed.refusal.unwrap().contains("print only"));

        let small = ServerInfo {
            context: 40_000,
            ..server()
        };
        let claude = state(
            Target::ClaudeCode,
            &small,
            &clients::persistent(Target::ClaudeCode, &small, &homes),
            true,
            None,
        );
        assert_eq!(
            claude.refusal.as_deref(),
            Some("org/m leaves 40000 tokens of context; claude-code needs at least 65536")
        );
        assert_eq!(claude.files[0].change, "create", "{claude:?}");
    }

    fn before_without_version(state: AgentState) -> AgentState {
        AgentState {
            installed: false,
            version: None,
            ..state
        }
    }

    #[test]
    fn diffs_show_changed_lines() {
        assert_eq!(diff("a\nb\nc\n", "a\nx\nc\n"), "  a\n- b\n+ x\n  c\n");
        assert_eq!(diff("", "new\n"), "+ new\n");
    }
}
