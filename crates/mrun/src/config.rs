//! Configuration sources (004 FR-021, 017 FR-003), later sources winning: built-in defaults, the
//! config file (shared top-level keys, then the command's table), `MRUN_<FLAG>` environment
//! variables, flags. Flags and their variables are declared on the clap arguments; the file's
//! values become those arguments' default values (`apply`), so clap resolves the order itself.
//! This module also catches misspelled variables, keys and sections, which would otherwise be
//! ignored silently.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use clap::{Arg, ArgAction, Command};

/// Every `MRUN_*` variable mrun reads (`--record` has none by design).
pub const KNOWN: [&str; 34] = [
    "MRUN_HOME",
    "MRUN_CONFIG",
    "MRUN_PORT",
    "MRUN_MODEL",
    "MRUN_MAX_OUTPUT",
    "MRUN_QUEUE",
    "MRUN_QUEUE_WAIT_S",
    "MRUN_BUDGET_GIB",
    "MRUN_NO_CACHE",
    "MRUN_SPECULATION",
    "MRUN_REJECT_TOOL_IMAGES",
    "MRUN_DETACH",
    "MRUN_CALIBRATION_DEBUG",
    // The window (018).
    "MRUN_UI_PORT",
    // Remote GPU mode (013): the relay and the in-pod agent and bootstrap.
    "MRUN_POD_TOKEN",
    "MRUN_RELAY_TOKEN",
    "MRUN_AGENT_PORT",
    "MRUN_AGENT_LOG",
    "MRUN_IDLE_STOP",
    "MRUN_MAX_RUNTIME",
    "MRUN_FAILED_STOP",
    "MRUN_MAX_MODEL_LEN",
    "MRUN_MAX_NUM_SEQS",
    "MRUN_TOOL_PARSER",
    "MRUN_REASONING_PARSER",
    "MRUN_VLLM_ARGS",
    "MRUN_COST_PER_HR",
    "MRUN_GPU",
    "MRUN_AGENT_STUB",
    "MRUN_AGENT_SHA256",
    "MRUN_AGENT_URL",
    "MRUN_BOOTSTRAP",
    // The Keychain stand-in for tests (024).
    "MRUN_KEYCHAIN_STUB",
    // Mac pool (026).
    "MRUN_POOL_PORT",
];

/// `MRUN_*` variables of the build (`crates/mrun/build.rs`): not read at runtime, but often still
/// exported in the shell that built mrun, so they draw no warning (008 local release dry run).
pub const BUILD_TIME: [&str; 1] = ["MRUN_BUILD_COMMIT"];

/// Warnings for unknown `MRUN_*` names, with the closest known one.
pub fn unknown_variables(names: impl Iterator<Item = String>) -> Vec<String> {
    names
        .filter(|name| {
            name.starts_with("MRUN_")
                && !KNOWN.contains(&name.as_str())
                && !BUILD_TIME.contains(&name.as_str())
        })
        .map(|name| {
            let closest = KNOWN
                .iter()
                .min_by_key(|known| distance(&name, known))
                .filter(|known| distance(&name, known) <= 3);
            match closest {
                Some(known) => {
                    format!("warning: unknown variable {name} is ignored; did you mean {known}?")
                }
                None => format!("warning: unknown variable {name} is ignored"),
            }
        })
        .collect()
}

/// Levenshtein distance.
fn distance(a: &str, b: &str) -> usize {
    let b: Vec<char> = b.chars().collect();
    let mut row: Vec<usize> = (0..=b.len()).collect();
    for (i, ca) in a.chars().enumerate() {
        let mut prev = row[0];
        row[0] = i + 1;
        for (j, cb) in b.iter().enumerate() {
            let next = (row[j + 1] + 1)
                .min(row[j] + 1)
                .min(prev + usize::from(ca != *cb));
            prev = row[j + 1];
            row[j + 1] = next;
        }
    }
    row[b.len()]
}

// ---- The config file (017) ----

/// The commands with settings: the table name and the subcommand path it configures.
pub const TABLES: [(&str, &[&str]); 5] = [
    ("serve", &["serve"]),
    ("launch", &["launch"]),
    ("run", &["run"]),
    ("cloud", &["cloud", "up"]),
    ("bench", &["bench"]),
];

/// Flags that are not settings (017 FR-004): confirmations, one-shot actions, output formats, and
/// remote mode (`serve --relay`: a file must never make `serve` relay to a pod).
pub const NOT_SETTINGS: [&str; 15] = [
    "yes", "dry-run", "json", "write", "restore", "out", "follow", "record", "trace", "info",
    "quiet", "verbose", "relay", "help", "version",
];

/// Settings shared by several commands under one variable: top-level keys (017 FR-002).
pub const SHARED: [(&str, &str); 3] = [
    ("port", "MRUN_PORT"),
    ("model", "MRUN_MODEL"),
    ("budget-gib", "MRUN_BUDGET_GIB"),
];

/// Settings that take effect only with another (clap's `requires`, which clap does not expose).
pub const NEEDS: [(&str, &str); 3] = [
    ("cloud.max-price", "retry"),
    ("cloud.any-cloud", "retry"),
    ("cloud.agent-url", "agent-sha256"),
];

/// Names that look like secrets or addresses: never settings (017 FR-010, FR-011).
const SECRET_LIKE: [&str; 6] = ["api-key", "runpod-api-key", "token", "key", "host", "bind"];

/// The kind of value a setting takes (from its flag's value parser).
#[derive(Clone, Debug, PartialEq)]
pub enum Kind {
    Boolean,
    Integer,
    Number,
    Choice(Vec<String>),
    String,
}

/// One settable key (data model "Setting").
#[derive(Clone, Debug)]
pub struct Setting {
    /// `None` for a shared top-level key.
    pub table: Option<&'static str>,
    /// The flag's long name: the key.
    pub name: String,
    /// The clap argument id.
    pub arg: String,
    /// The subcommand paths it applies to.
    pub targets: Vec<Vec<&'static str>>,
    /// Built-in default values (empty: none).
    pub default: Vec<String>,
    pub env: Option<String>,
    pub help: String,
    pub kind: Kind,
    pub repeatable: bool,
}

impl Setting {
    /// `table.name`, or `name` for a shared key.
    pub fn key(&self) -> String {
        match self.table {
            Some(table) => format!("{table}.{}", self.name),
            None => self.name.clone(),
        }
    }

    /// The setting this one needs to take effect, if any.
    pub fn needs(&self) -> Option<&'static str> {
        let key = self.key();
        NEEDS.iter().find(|(k, _)| *k == key).map(|(_, n)| *n)
    }
}

fn find<'a>(cmd: &'a Command, path: &[&str]) -> Option<&'a Command> {
    path.iter()
        .try_fold(cmd, |cmd, name| cmd.find_subcommand(name))
}

fn is_setting(arg: &Arg) -> bool {
    arg.get_long()
        .is_some_and(|long| !NOT_SETTINGS.contains(&long))
        && !arg.is_hide_set()
        && !arg.is_positional()
}

fn env_of(arg: &Arg) -> Option<String> {
    arg.get_env().map(|e| e.to_string_lossy().into_owned())
}

/// Parses `values` with `arg`'s own value parser (through a one-argument command: clap's parser
/// entry point is internal), returning clap's reason on the first rejection.
fn parse_with(arg: &Arg, values: &[String]) -> Result<(), String> {
    let probe = Arg::new("v")
        .long("v")
        .action(ArgAction::Append)
        .value_parser(arg.get_value_parser().clone());
    let argv =
        std::iter::once("probe".to_string()).chain(values.iter().map(|v| format!("--v={v}")));
    match Command::new("probe")
        .no_binary_name(false)
        .arg(probe)
        .try_get_matches_from(argv)
    {
        Ok(_) => Ok(()),
        Err(e) => {
            let text = e.to_string();
            let first = text.lines().next().unwrap_or_default();
            let message = first.strip_prefix("error: ").unwrap_or(first);
            // clap names the probe flag; keep the reason after it.
            let reason = message.split_once("': ").map_or(message, |(_, r)| r);
            Err(reason.trim().to_string())
        }
    }
}

/// The kind of `arg`'s values, by trying its own parser.
fn kind(arg: &Arg) -> Kind {
    if matches!(arg.get_action(), ArgAction::SetTrue | ArgAction::SetFalse) {
        return Kind::Boolean;
    }
    let choices: Vec<String> = arg
        .get_possible_values()
        .iter()
        .filter(|v| !v.is_hide_set())
        .map(|v| v.get_name().to_string())
        .collect();
    if !choices.is_empty() {
        return Kind::Choice(choices);
    }
    let parses = |text: &str| parse_with(arg, &[text.to_string()]).is_ok();
    // A duration parser takes "1" (seconds) but also "1m": only a parser that takes plain numbers
    // and nothing else is numeric.
    if parses("1m") || parses("x") {
        return Kind::String;
    }
    match (parses("1"), parses("1.5")) {
        (true, false) => Kind::Integer,
        (true, true) => Kind::Number,
        _ => Kind::String,
    }
}

fn setting(table: Option<&'static str>, arg: &Arg) -> Setting {
    Setting {
        table,
        name: arg.get_long().unwrap_or_default().to_string(),
        arg: arg.get_id().to_string(),
        targets: Vec::new(),
        default: arg
            .get_default_values()
            .iter()
            .map(|v| v.to_string_lossy().into_owned())
            .collect(),
        env: env_of(arg),
        help: arg.get_help().map(ToString::to_string).unwrap_or_default(),
        kind: kind(arg),
        repeatable: matches!(arg.get_action(), ArgAction::Append),
    }
}

/// Every user-facing subcommand path of `cmd` (depth first): hidden ones (the in-pod `agent`)
/// never take settings from the user's file.
fn paths(cmd: &Command, prefix: Vec<&'static str>, out: &mut Vec<Vec<&'static str>>) {
    for sub in cmd.get_subcommands().filter(|s| !s.is_hide_set()) {
        let name: &'static str = Box::leak(sub.get_name().to_string().into_boxed_str());
        let mut path = prefix.clone();
        path.push(name);
        out.push(path.clone());
        paths(sub, path, out);
    }
}

/// The settable keys of `cli` (017 FR-004): the flags of each table's command that are
/// preferences, and the shared top-level keys on every command with their variable.
pub fn settings(cli: &Command) -> Vec<Setting> {
    let mut out = Vec::new();
    let mut all = Vec::new();
    paths(cli, Vec::new(), &mut all);
    for (name, env) in SHARED {
        let mut shared: Option<Setting> = None;
        for path in &all {
            let Some(cmd) = find(cli, path) else { continue };
            for arg in cmd.get_arguments() {
                if is_setting(arg)
                    && arg.get_long() == Some(name)
                    && env_of(arg).as_deref() == Some(env)
                {
                    let entry = shared.get_or_insert_with(|| setting(None, arg));
                    entry.targets.push(path.clone());
                }
            }
        }
        out.extend(shared);
    }
    for (table, path) in TABLES {
        let Some(cmd) = find(cli, path) else { continue };
        for arg in cmd.get_arguments().filter(|a| is_setting(a)) {
            let mut s = setting(Some(table), arg);
            s.targets.push(path.to_vec());
            out.push(s);
        }
    }
    out
}

/// Where the config file is (017 FR-001).
#[derive(Clone, Debug, PartialEq)]
pub enum Location {
    /// `MRUN_CONFIG` is set and empty: no file.
    Disabled,
    /// A path; `explicit` when named by `MRUN_CONFIG` (then it must exist).
    File { path: PathBuf, explicit: bool },
}

impl Location {
    pub fn path(&self) -> Option<&Path> {
        match self {
            Self::Disabled => None,
            Self::File { path, .. } => Some(path),
        }
    }
}

/// `MRUN_CONFIG` (empty: none), else `config.toml` in `MRUN_HOME` (default `~/.mrun`).
pub fn location() -> Location {
    location_from(|name| std::env::var_os(name))
}

fn location_from(var: impl Fn(&str) -> Option<std::ffi::OsString>) -> Location {
    match var("MRUN_CONFIG") {
        Some(path) if path.is_empty() => Location::Disabled,
        Some(path) => Location::File {
            path: PathBuf::from(path),
            explicit: true,
        },
        None => {
            let home = var("MRUN_HOME").map(PathBuf::from).unwrap_or_else(|| {
                var("HOME")
                    .map(PathBuf::from)
                    .or_else(std::env::home_dir)
                    .unwrap_or_else(|| PathBuf::from("."))
                    .join(".mrun")
            });
            Location::File {
                path: home.join("config.toml"),
                explicit: false,
            }
        }
    }
}

/// One valid value of the file.
#[derive(Clone, Debug)]
pub struct FileValue {
    /// Index into the settings.
    pub setting: usize,
    /// The flag's string form, one per value.
    pub values: Vec<String>,
    pub line: usize,
}

/// The config file read and checked (data model "ConfigFile").
#[derive(Clone, Debug, Default)]
pub struct ConfigFile {
    pub path: Option<PathBuf>,
    pub exists: bool,
    pub values: Vec<FileValue>,
    /// `<path>:<line>: …` warnings (unknown sections and keys).
    pub warnings: Vec<String>,
}

/// A config file that cannot be used: unreadable, not TOML, or an invalid value (017 FR-006).
#[derive(Clone, Debug, PartialEq, thiserror::Error)]
#[error("{0}")]
pub struct ConfigError(pub String);

/// 1-based line and column of byte `offset` in `text`.
fn line_col(text: &str, offset: usize) -> (usize, usize) {
    let before = &text[..offset.min(text.len())];
    let line = before.matches('\n').count() + 1;
    let col = before.len() - before.rfind('\n').map_or(0, |i| i + 1) + 1;
    (line, col)
}

fn closest<'a>(name: &str, known: impl Iterator<Item = &'a str>) -> Option<&'a str> {
    known
        .map(|k| (distance(name, k), k))
        .filter(|(d, _)| *d <= 3)
        .min_by_key(|(d, _)| *d)
        .map(|(_, k)| k)
}

/// Reads and checks the file at `location` against `settings` of `cli`.
pub fn read(
    cli: &Command,
    location: &Location,
    settings: &[Setting],
) -> Result<ConfigFile, ConfigError> {
    let Location::File { path, explicit } = location else {
        return Ok(ConfigFile::default());
    };
    let text = match std::fs::read(path) {
        Ok(bytes) => String::from_utf8(bytes)
            .map_err(|_| ConfigError(format!("config file {} is not UTF-8", path.display())))?,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound && !explicit => {
            return Ok(ConfigFile {
                path: Some(path.clone()),
                ..ConfigFile::default()
            });
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            return Err(ConfigError(format!(
                "config file {} (MRUN_CONFIG) does not exist",
                path.display()
            )));
        }
        Err(e) => {
            return Err(ConfigError(format!(
                "cannot read config file {}: {e}",
                path.display()
            )));
        }
    };
    parse(cli, path, &text, settings)
}

/// Parses and checks `text` (the content of `path`) against `settings` of `cli`.
pub fn parse(
    cli: &Command,
    path: &Path,
    text: &str,
    settings: &[Setting],
) -> Result<ConfigFile, ConfigError> {
    use toml::de::{DeTable, DeValue};
    let shown = path.display();
    let doc = DeTable::parse(text).map_err(|e| {
        let (line, col) = e.span().map_or((1, 1), |s| line_col(text, s.start));
        ConfigError(format!("{shown}:{line}:{col}: {}", e.message().trim()))
    })?;
    let mut file = ConfigFile {
        path: Some(path.to_path_buf()),
        exists: true,
        ..ConfigFile::default()
    };
    let line_of = |span: std::ops::Range<usize>| line_col(text, span.start).0;
    let tables: Vec<&str> = TABLES.iter().map(|(t, _)| *t).collect();
    let mut entries: Vec<(Option<&str>, &str, &toml::Spanned<DeValue>, usize)> = Vec::new();
    for (key, value) in doc.get_ref() {
        let name = key.get_ref().as_ref();
        let line = line_of(key.span());
        match value.get_ref() {
            DeValue::Table(table) => {
                if !tables.contains(&name) {
                    let hint = closest(name, tables.iter().copied())
                        .map(|t| format!("; did you mean [{t}]?"))
                        .unwrap_or_default();
                    file.warnings
                        .push(format!("{shown}:{line}: unknown section [{name}]{hint}"));
                    continue;
                }
                for (k, v) in table {
                    entries.push((Some(name), k.get_ref().as_ref(), v, line_of(k.span())));
                }
            }
            _ => entries.push((None, name, value, line)),
        }
    }
    // Shared keys first, so a command's table overrides them when installed in order.
    entries.sort_by_key(|(table, ..)| table.is_some());
    for (table, name, value, line) in entries {
        let key = table.map_or_else(|| name.to_string(), |t| format!("{t}.{name}"));
        let found = settings
            .iter()
            .position(|s| s.table == table && s.name == name);
        let Some(index) = found else {
            file.warnings
                .push(unknown_key(&shown.to_string(), line, table, name, settings));
            continue;
        };
        let setting = &settings[index];
        let values = string_values(setting, value.get_ref()).map_err(|expected| {
            ConfigError(format!(
                "{shown}:{line}: {key} = {}: expected {expected}",
                display(value.get_ref())
            ))
        })?;
        if setting.kind != Kind::Boolean {
            check_values(cli, setting, &values).map_err(|message| {
                ConfigError(format!(
                    "{shown}:{line}: {key} = {}: {message}",
                    display(value.get_ref())
                ))
            })?;
        }
        file.values.push(FileValue {
            setting: index,
            values,
            line,
        });
    }
    // In file order (warnings carry `<path>:<line>:`).
    file.warnings.sort_by_key(|w| {
        w.strip_prefix(&format!("{shown}:"))
            .and_then(|rest| rest.split(':').next())
            .and_then(|line| line.parse::<usize>().ok())
            .unwrap_or(0)
    });
    Ok(file)
}

fn unknown_key(
    shown: &str,
    line: usize,
    table: Option<&str>,
    name: &str,
    settings: &[Setting],
) -> String {
    let key = table.map_or_else(|| name.to_string(), |t| format!("{t}.{name}"));
    let mut text = format!("{shown}:{line}: unknown key {key}");
    if SECRET_LIKE.contains(&name) {
        text.push_str(
            "; secrets come from the environment (RUNPOD_API_KEY or .env), never from the config file, and mrun binds to 127.0.0.1 only",
        );
        return text;
    }
    if let Some(shared) = settings
        .iter()
        .find(|s| s.table.is_none() && s.name == name)
    {
        let _ = shared;
        text.push_str("; this command has no such flag: set it at the top level of the file");
        return text;
    }
    let here = settings
        .iter()
        .filter(|s| s.table == table)
        .map(|s| s.name.as_str());
    if let Some(near) = closest(name, here) {
        text.push_str(&format!("; did you mean {near}?"));
        return text;
    }
    if let Some(other) = settings
        .iter()
        .find(|s| s.name == name && s.table.is_some())
    {
        text.push_str(&format!(
            "; it belongs to [{}]",
            other.table.unwrap_or_default()
        ));
    }
    text
}

/// A TOML value as it appeared, for messages.
fn display(value: &toml::de::DeValue) -> String {
    use toml::de::DeValue;
    match value {
        DeValue::String(s) => format!("{:?}", s.as_ref()),
        DeValue::Integer(i) => i.as_str().to_string(),
        DeValue::Float(f) => f.as_str().to_string(),
        DeValue::Boolean(b) => b.to_string(),
        DeValue::Datetime(d) => d.to_string(),
        DeValue::Array(_) => "[…]".into(),
        DeValue::Table(_) => "{…}".into(),
    }
}

/// The flag's string form of a TOML value, or what was expected instead.
fn string_values(setting: &Setting, value: &toml::de::DeValue) -> Result<Vec<String>, String> {
    use toml::de::DeValue;
    // The TOML type must match the setting's kind (as the schema says): `idle-stop = 30` is not
    // the string "30m" a reader of the file would expect, even though the flag takes 30 (seconds).
    let one = |value: &DeValue| -> Result<String, String> {
        match (value, &setting.kind) {
            (DeValue::Boolean(b), Kind::Boolean) => Ok(b.to_string()),
            (DeValue::Integer(i), Kind::Integer | Kind::Number) => Ok(i.as_str().replace('_', "")),
            (DeValue::Float(f), Kind::Number) => Ok(f.as_str().replace('_', "")),
            (DeValue::String(s), Kind::String | Kind::Choice(_)) => Ok(s.to_string()),
            _ => Err(expected(setting)),
        }
    };
    match value {
        DeValue::Array(items) if setting.repeatable => {
            items.iter().map(|item| one(item.get_ref())).collect()
        }
        DeValue::Array(_) => Err(format!("one value ({}), not a list", expected(setting))),
        other => Ok(vec![one(other)?]),
    }
}

fn expected(setting: &Setting) -> String {
    match &setting.kind {
        Kind::Boolean => "true or false".into(),
        Kind::Integer => "an integer".into(),
        Kind::Number => "a number".into(),
        Kind::Choice(choices) => format!("one of {}", choices.join(", ")),
        Kind::String => match setting.default.first() {
            Some(example) => format!("a string such as {example:?}"),
            None => "a string".into(),
        },
    }
}

/// Runs the flag's own value parser over `values`.
fn check_values(cli: &Command, setting: &Setting, values: &[String]) -> Result<(), String> {
    let path = setting.targets.first().cloned().unwrap_or_default();
    let cmd = find(cli, &path).ok_or("no such command")?;
    let arg = cmd
        .get_arguments()
        .find(|a| a.get_id() == setting.arg.as_str())
        .ok_or("no such flag")?;
    parse_with(arg, values)
}

/// A boolean preference flag as `--flag[=<BOOL>]`, defaulting to `default` (017 R4).
fn negatable(arg: Arg, default: String) -> Arg {
    arg.action(ArgAction::Set)
        .num_args(0..=1)
        .require_equals(true)
        .default_missing_value("true")
        .value_parser(clap::builder::BoolishValueParser::new())
        .value_name("BOOL")
        .hide_possible_values(true)
        .hide_default_value(true)
        .default_value(default)
}

fn mut_at(cmd: Command, path: &[&'static str], f: &dyn Fn(Command) -> Command) -> Command {
    match path.split_first() {
        None => f(cmd),
        Some((first, rest)) => cmd.mut_subcommand(*first, |sub| mut_at(sub, rest, f)),
    }
}

/// `cmd` with every boolean setting negatable and the file's values as defaults (017 R1, R4).
pub fn apply(mut cmd: Command, settings: &[Setting], file: &ConfigFile) -> Command {
    for setting in settings.iter().filter(|s| s.kind == Kind::Boolean) {
        for path in &setting.targets {
            let id = setting.arg.clone();
            let default = setting
                .default
                .first()
                .cloned()
                .unwrap_or_else(|| "false".into());
            cmd = mut_at(cmd, path, &move |c: Command| {
                let id = id.clone();
                let default = default.clone();
                c.mut_arg(id, move |a| negatable(a, default))
            });
        }
    }
    for value in &file.values {
        let setting = &settings[value.setting];
        for path in &setting.targets {
            let id = setting.arg.clone();
            let values = value.values.clone();
            cmd = mut_at(cmd, path, &move |c: Command| {
                let values = values.clone();
                // A file value satisfies a required flag (`[cloud] model` makes `--model` optional).
                c.mut_arg(id.clone(), move |a| {
                    a.default_values(values).required(false)
                })
            });
        }
    }
    // A file value counts as given for `requires` (clap ignores defaults there): `retry` in the
    // file lets `--max-price` alone on the command line work.
    for (key, needed) in NEEDS {
        let Some(dependent) = settings.iter().find(|s| s.key() == key) else {
            continue;
        };
        let satisfied = file.values.iter().any(|v| {
            let s = &settings[v.setting];
            s.table == dependent.table && s.name == needed
        });
        if !satisfied {
            continue;
        }
        for path in &dependent.targets {
            let id = dependent.arg.clone();
            cmd = mut_at(cmd, path, &move |c: Command| {
                c.mut_arg(id.clone(), |a| {
                    a.requires(clap::builder::Resettable::<clap::Id>::Reset)
                })
            });
        }
    }
    cmd
}

/// The file's values by `(command path, arg id)`: line and flag form, for consent marks (FR-010)
/// and the detached server (R5).
static FROM_FILE: OnceLock<FileValues> = OnceLock::new();

/// `(command path, arg id)` → `(line, values)`.
type FileValues = BTreeMap<(String, String), (usize, Vec<String>)>;

/// Records `file` for [`from_file`] and [`file_args`] (once per process).
pub fn remember(settings: &[Setting], file: &ConfigFile) {
    let mut map = BTreeMap::new();
    for value in &file.values {
        let setting = &settings[value.setting];
        for path in &setting.targets {
            map.insert(
                (path.join(" "), setting.arg.clone()),
                (value.line, value.values.clone()),
            );
        }
    }
    let _ = FROM_FILE.set(map);
}

/// The file's line for `arg` of `command` (`"cloud up"`), when the file set it.
pub fn from_file(command: &str, arg: &str) -> Option<usize> {
    FROM_FILE
        .get()?
        .get(&(command.to_string(), arg.to_string()))
        .map(|(line, _)| *line)
}

/// The file's values for `command` as flags (`--port=9000`), for a child process that must use
/// this process's resolution without reading the file again (017 R5).
pub fn file_args(command: &str, settings: &[Setting]) -> Vec<String> {
    let Some(map) = FROM_FILE.get() else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for ((path, arg), (_, values)) in map {
        if path != command {
            continue;
        }
        let Some(setting) = settings
            .iter()
            .find(|s| &s.arg == arg && s.targets.iter().any(|t| t.join(" ") == *path))
        else {
            continue;
        };
        // A set variable beats the file here, and would lose to a flag in the child: leave it to
        // the inherited environment.
        if setting
            .env
            .as_deref()
            .is_some_and(|var| std::env::var_os(var).is_some_and(|v| !v.is_empty()))
        {
            continue;
        }
        let name = setting.name.clone();
        out.extend(values.iter().map(|v| format!("--{name}={v}")));
    }
    out
}

/// Where an effective value came from (data model "EffectiveValue").
#[derive(Clone, Debug, PartialEq)]
pub enum Source {
    Default,
    File(usize),
    Env(String),
}

impl std::fmt::Display for Source {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Default => f.write_str("default"),
            Self::File(line) => write!(f, "file:{line}"),
            Self::Env(var) => write!(f, "env:{var}"),
        }
    }
}

/// A setting's value in force when a command starts with no flags.
#[derive(Clone, Debug)]
pub struct Effective {
    pub setting: usize,
    pub values: Vec<String>,
    pub source: Source,
    /// Lower sources that had a value: `(source, values)`.
    pub overridden: Vec<(Source, Vec<String>)>,
}

/// The effective value of every setting (FR-007): a command table's own value only when the file
/// sets it there (shared keys are listed at the top).
pub fn effective(
    settings: &[Setting],
    file: &ConfigFile,
    env: impl Fn(&str) -> Option<String>,
) -> Vec<Effective> {
    let mut out = Vec::new();
    for (index, setting) in settings.iter().enumerate() {
        let shared_arg = setting.table.is_some()
            && SHARED
                .iter()
                .any(|(_, var)| setting.env.as_deref() == Some(var));
        let in_file = file.values.iter().find(|v| v.setting == index);
        if shared_arg && in_file.is_none() {
            continue;
        }
        let mut layers: Vec<(Source, Vec<String>)> = Vec::new();
        if !setting.default.is_empty() {
            layers.push((Source::Default, setting.default.clone()));
        }
        if let Some(v) = in_file {
            layers.push((Source::File(v.line), v.values.clone()));
        }
        if let Some(var) = &setting.env
            && let Some(value) = env(var).filter(|v| !v.is_empty())
        {
            layers.push((Source::Env(var.clone()), vec![value]));
        }
        let (source, values) = layers.pop().unwrap_or((Source::Default, Vec::new()));
        out.push(Effective {
            setting: index,
            values,
            source,
            overridden: layers
                .into_iter()
                .filter(|(s, _)| *s != Source::Default)
                .collect(),
        });
    }
    out
}

/// The settings, the file's location and the file as read (or why it cannot be used).
#[derive(Clone, Debug)]
pub struct Resolved {
    pub settings: Vec<Setting>,
    pub location: Location,
    pub file: Result<ConfigFile, ConfigError>,
}

/// Reads the config file for `cli` (the unapplied command tree) from the environment's location.
pub fn resolve(cli: &Command) -> Resolved {
    let settings = settings(cli);
    let location = location();
    let file = read(cli, &location, &settings);
    Resolved {
        settings,
        location,
        file,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn misspelled_variables_get_a_suggestion() {
        let names = [
            "MRUN_PROT",
            "MRUN_PORT",
            "MRUN_ENTIRELY_ELSE",
            "PATH",
            "MRUN_BUILD_COMMIT",
        ]
        .map(String::from)
        .into_iter();
        let warnings = unknown_variables(names);
        assert_eq!(
            warnings,
            vec![
                "warning: unknown variable MRUN_PROT is ignored; did you mean MRUN_PORT?",
                "warning: unknown variable MRUN_ENTIRELY_ELSE is ignored",
            ]
        );
    }

    // ---- The config file (017) ----

    use crate::cli::app::cli_command;

    fn all() -> (Command, Vec<Setting>) {
        let cli = cli_command();
        let settings = settings(&cli);
        (cli, settings)
    }

    fn parse_text(text: &str) -> Result<ConfigFile, ConfigError> {
        let (cli, settings) = all();
        parse(&cli, Path::new("c.toml"), text, &settings)
    }

    /// A value `setting`'s own parser accepts, other than its default when it has one.
    fn sample(setting: &Setting) -> String {
        let cli = cli_command();
        let path = &setting.targets[0];
        let cmd = find(&cli, path).unwrap();
        let arg = cmd
            .get_arguments()
            .find(|a| a.get_id() == setting.arg.as_str())
            .unwrap();
        let candidates: Vec<String> = match &setting.kind {
            Kind::Boolean => vec!["true".into()],
            Kind::Choice(choices) => choices.iter().rev().cloned().collect(),
            _ => ["7", "2.5", "45m", "qwen3", "/tmp/x", "a=b", "x"]
                .map(String::from)
                .to_vec(),
        };
        candidates
            .into_iter()
            .find(|c| {
                setting.default.first() != Some(c)
                    && parse_with(arg, std::slice::from_ref(c)).is_ok()
            })
            .unwrap_or_else(|| panic!("no sample for {}", setting.key()))
    }

    fn toml_literal(setting: &Setting, value: &str) -> String {
        match setting.kind {
            Kind::Boolean | Kind::Integer | Kind::Number => value.to_string(),
            _ => format!("{value:?}"),
        }
    }

    /// The argv that reaches `path`, with every other required argument given.
    fn argv(cli: &Command, path: &[&str], extra: &[String], skip: &str) -> Vec<String> {
        let mut out = vec!["mrun".to_string()];
        out.extend(path.iter().map(|p| p.to_string()));
        let cmd = find(cli, path).unwrap();
        for arg in cmd
            .get_arguments()
            .filter(|a| a.is_required_set() && a.get_id() != skip)
        {
            let given = extra.iter().any(|e| {
                arg.get_long()
                    .is_some_and(|l| e.starts_with(&format!("--{l}")))
            });
            match arg.get_long() {
                Some(long) if !given => out.push(format!("--{long}=required")),
                None => {
                    let first = arg
                        .get_possible_values()
                        .first()
                        .map_or("required".to_string(), |v| v.get_name().to_string());
                    out.push(first);
                }
                _ => {}
            }
        }
        out.extend(extra.iter().cloned());
        out
    }

    fn raw(matches: &clap::ArgMatches, path: &[&str], id: &str) -> Vec<String> {
        let m = path
            .iter()
            .try_fold(matches, |m, p| m.subcommand_matches(p))
            .unwrap();
        m.get_raw(id)
            .map(|v| v.map(|x| x.to_string_lossy().into_owned()).collect())
            .unwrap_or_default()
    }

    #[test]
    fn the_file_is_found_by_mrun_config_then_mrun_home() {
        let env = |pairs: &'static [(&'static str, &'static str)]| {
            move |name: &str| {
                pairs
                    .iter()
                    .find(|(k, _)| *k == name)
                    .map(|(_, v)| std::ffi::OsString::from(v))
            }
        };
        assert_eq!(
            location_from(env(&[("MRUN_CONFIG", "")])),
            Location::Disabled
        );
        assert_eq!(
            location_from(env(&[("MRUN_CONFIG", "/x.toml"), ("MRUN_HOME", "/h")])),
            Location::File {
                path: PathBuf::from("/x.toml"),
                explicit: true
            }
        );
        assert_eq!(
            location_from(env(&[("MRUN_HOME", "/h")])),
            Location::File {
                path: PathBuf::from("/h/config.toml"),
                explicit: false
            }
        );
        assert_eq!(
            location_from(env(&[("HOME", "/u")])),
            Location::File {
                path: PathBuf::from("/u/.mrun/config.toml"),
                explicit: false
            }
        );
    }

    #[test]
    fn no_setting_selects_remote_mode_holds_a_secret_or_answers_a_prompt() {
        let (_, settings) = all();
        for s in &settings {
            assert!(
                !NOT_SETTINGS.contains(&s.name.as_str()) && !SECRET_LIKE.contains(&s.name.as_str()),
                "{} must not be a setting",
                s.key()
            );
            assert!(!s.name.starts_with("relay"), "{}", s.key());
        }
        for table in ["serve", "launch", "run", "cloud", "bench"] {
            assert!(settings.iter().any(|s| s.table == Some(table)), "{table}");
        }
        let port = settings.iter().find(|s| s.key() == "port").unwrap();
        for command in [
            vec!["serve"],
            vec!["ps"],
            vec!["stop"],
            vec!["doctor"],
            vec!["launch"],
        ] {
            assert!(
                port.targets.contains(&command),
                "port applies to {command:?}"
            );
        }
        let file =
            parse_text("[serve]\nrelay = \"https://x\"\n[cloud]\nyes = true\napi-key = \"k\"\n")
                .unwrap();
        assert!(file.values.is_empty());
        assert_eq!(file.warnings.len(), 3, "{:?}", file.warnings);
        assert!(file.warnings[2].contains("secrets come from the environment"));
    }

    #[test]
    fn every_setting_resolves_default_then_file_then_flag() {
        let (cli, settings) = all();
        for (index, setting) in settings.iter().enumerate() {
            let value = sample(setting);
            let text = match setting.table {
                Some(table) => format!(
                    "[{table}]\n{} = {}\n",
                    setting.name,
                    toml_literal(setting, &value)
                ),
                None => format!("{} = {}\n", setting.name, toml_literal(setting, &value)),
            };
            let file = parse(&cli, Path::new("c.toml"), &text, &settings)
                .unwrap_or_else(|e| panic!("{}: {e}", setting.key()));
            assert_eq!(
                file.values.len(),
                1,
                "{}: {:?}",
                setting.key(),
                file.warnings
            );
            assert_eq!(file.values[0].setting, index);
            for path in &setting.targets {
                // File over default.
                let cmd = apply(cli.clone(), &settings, &file);
                let m = cmd
                    .try_get_matches_from(argv(&cli, path, &[], &setting.arg))
                    .unwrap_or_else(|e| panic!("{} {path:?}: {e}", setting.key()));
                assert_eq!(
                    raw(&m, path, &setting.arg),
                    std::slice::from_ref(&value),
                    "{} {path:?}",
                    setting.key()
                );
                // Flag over file (booleans in the other direction).
                let flag = match setting.kind {
                    Kind::Boolean => "false".to_string(),
                    _ => setting
                        .default
                        .first()
                        .cloned()
                        .unwrap_or_else(|| value.clone()),
                };
                let cmd = apply(cli.clone(), &settings, &file);
                let mut given = vec![format!("--{}={flag}", setting.name)];
                if let Some(needed) = setting.needs() {
                    let other = settings
                        .iter()
                        .find(|s| s.table == setting.table && s.name == needed)
                        .unwrap();
                    given.push(format!("--{needed}={}", sample(other)));
                }
                let m = cmd
                    .try_get_matches_from(argv(&cli, path, &given, &setting.arg))
                    .unwrap_or_else(|e| panic!("{} {path:?}: {e}", setting.key()));
                assert_eq!(
                    raw(&m, path, &setting.arg),
                    [flag],
                    "{} {path:?}",
                    setting.key()
                );
            }
            // A variable over the file (as `config show` resolves it).
            if let Some(var) = &setting.env {
                let eff = effective(&settings, &file, |name| {
                    (name == var).then(|| "from-env".into())
                });
                let e = eff.iter().find(|e| e.setting == index).unwrap();
                assert_eq!(e.source, Source::Env(var.clone()), "{}", setting.key());
                assert_eq!(e.overridden[0].0, Source::File(file.values[0].line));
            }
        }
    }

    #[test]
    fn a_command_table_overrides_the_shared_key() {
        let (cli, settings) = all();
        let file = parse(
            &cli,
            Path::new("c.toml"),
            "model = \"a\"\n[launch]\nmodel = \"b\"\n",
            &settings,
        )
        .unwrap();
        let cmd = apply(cli.clone(), &settings, &file);
        let m = cmd
            .clone()
            .try_get_matches_from(argv(&cli, &["launch"], &[], ""))
            .unwrap();
        assert_eq!(raw(&m, &["launch"], "model"), ["b"]);
        let m = cmd
            .try_get_matches_from(argv(&cli, &["serve"], &[], ""))
            .unwrap();
        assert_eq!(raw(&m, &["serve"], "model"), ["a"]);
    }

    #[test]
    fn a_needed_setting_from_the_file_satisfies_a_flag() {
        let (cli, settings) = all();
        let file = parse(
            &cli,
            Path::new("c.toml"),
            "[cloud]\nretry = \"1h\"\nmodel = \"m\"\n",
            &settings,
        )
        .unwrap();
        let cmd = apply(cli.clone(), &settings, &file);
        let m = cmd
            .try_get_matches_from(["mrun", "cloud", "up", "--max-price=3"])
            .unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(raw(&m, &["cloud", "up"], "retry"), ["1h"]);
        assert_eq!(raw(&m, &["cloud", "up"], "max_price"), ["3"]);
    }

    #[test]
    fn values_of_the_wrong_kind_fail_with_file_and_line() {
        let cases = [
            (
                "[cloud]\n\nidle-stop = 30\n",
                "c.toml:3: cloud.idle-stop = 30: expected a string such as \"30m\"",
            ),
            (
                "[cloud]\nidle-stop = \"soon\"\n",
                "c.toml:2: cloud.idle-stop = \"soon\": 'soon' is not a duration",
            ),
            (
                "[cloud]\nany-cloud = \"yes\"\n",
                "c.toml:2: cloud.any-cloud = \"yes\": expected true or false",
            ),
            (
                "[serve]\nqueue = \"many\"\n",
                "c.toml:2: serve.queue = \"many\": expected an integer",
            ),
            ("[serve]\nqueue = 0\n", "c.toml:2: serve.queue = 0:"),
            (
                "[cloud]\nmax-price = \"cheap\"\n",
                "c.toml:2: cloud.max-price = \"cheap\": expected a number",
            ),
            ("[cloud]\nidle-stop = [\"30m\"]\n", "not a list"),
            (
                "[bench]\nsuite = \"everything\"\n",
                "c.toml:2: bench.suite = \"everything\":",
            ),
            ("[cloud\n", "c.toml:1:7: unclosed table"),
        ];
        for (text, want) in cases {
            let error = parse_text(text).unwrap_err().0;
            assert!(error.contains(want), "{text:?}: {error}");
        }
        assert!(
            parse_text("[cloud]\nmax-price = 2\n").is_ok(),
            "an integer is a number"
        );
    }

    #[test]
    fn misspellings_get_the_closest_name() {
        let (_, settings) = all();
        let file = parse_text("[clould]\nx = 1\n").unwrap();
        assert!(
            file.warnings[0].ends_with("did you mean [cloud]?"),
            "{:?}",
            file.warnings
        );
        for setting in settings.iter().filter(|s| s.name.len() > 4) {
            let typo: String = setting.name.chars().skip(1).collect();
            let text = match setting.table {
                Some(t) => format!("[{t}]\n{typo} = 1\n"),
                None => format!("{typo} = 1\n"),
            };
            let file = parse_text(&text).unwrap();
            let warning = file.warnings.first().unwrap_or_else(|| panic!("{typo}"));
            let suggested = warning
                .rsplit("did you mean ")
                .next()
                .unwrap()
                .trim_end_matches('?');
            assert!(
                distance(&typo, suggested) <= distance(&typo, &setting.name),
                "{typo}: {warning}"
            );
        }
        let file = parse_text("[serve]\nidle-stop = \"1m\"\n").unwrap();
        assert!(
            file.warnings[0].ends_with("it belongs to [cloud]"),
            "{:?}",
            file.warnings
        );
    }

    #[test]
    fn known_covers_every_variable_of_the_command_line() {
        let cli = cli_command();
        let mut all_paths = Vec::new();
        paths(&cli, Vec::new(), &mut all_paths);
        for path in all_paths {
            let cmd = find(&cli, &path).unwrap();
            for arg in cmd.get_arguments() {
                if let Some(var) = env_of(arg) {
                    assert!(
                        KNOWN.contains(&var.as_str()),
                        "{var} ({path:?}) is not in KNOWN"
                    );
                }
            }
        }
    }

    #[test]
    fn a_full_file_resolves_quickly() {
        let (cli, settings) = all();
        let text: String = settings
            .iter()
            .map(|s| {
                let line = format!("{} = {}\n", s.name, toml_literal(s, &sample(s)));
                match s.table {
                    Some(t) => format!("[{t}]\n{line}"),
                    None => line,
                }
            })
            .collect::<Vec<_>>()
            .join("");
        // Tables may repeat in this generated text: merge them into one per table.
        let mut merged = String::new();
        for (t, _) in TABLES {
            merged.push_str(&format!("[{t}]\n"));
            for line in text
                .split("\n[")
                .filter(|b| b.starts_with(&format!("{t}]")))
            {
                merged.push_str(line.split_once('\n').map_or("", |(_, l)| l));
                merged.push('\n');
            }
        }
        let shared: String = settings
            .iter()
            .filter(|s| s.table.is_none())
            .map(|s| format!("{} = {}\n", s.name, toml_literal(s, &sample(s))))
            .collect();
        let full = format!("{shared}{merged}");
        let start = std::time::Instant::now();
        let file = parse(&cli, Path::new("c.toml"), &full, &settings).unwrap();
        let _ = apply(cli.clone(), &settings, &file);
        let took = start.elapsed();
        assert_eq!(file.values.len(), settings.len(), "{:?}", file.warnings);
        let bound = if cfg!(debug_assertions) { 200 } else { 5 };
        assert!(took.as_millis() < bound, "{took:?}");
    }
}
