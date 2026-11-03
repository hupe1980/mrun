//! The config file through the binary (017): location, `config path/show/schema`, cloud policy
//! from the file, precedence with variables, mistakes, `doctor`, and that mrun never writes the
//! file. Every run gets its own `MRUN_HOME` and no inherited `MRUN_*` variables.

use std::process::{Command, Output};

use serde_json::Value;

struct Home {
    dir: tempfile::TempDir,
}

impl Home {
    fn new() -> Self {
        Self {
            dir: tempfile::tempdir().unwrap(),
        }
    }

    fn with_file(text: &str) -> Self {
        let home = Self::new();
        std::fs::write(home.file(), text).unwrap();
        home
    }

    fn file(&self) -> std::path::PathBuf {
        self.dir.path().join("config.toml")
    }

    fn mrun(&self, args: &[&str], env: &[(&str, &str)]) -> Output {
        let mut cmd = Command::new(env!("CARGO_BIN_EXE_mrun"));
        for (name, _) in std::env::vars().filter(|(n, _)| n.starts_with("MRUN_")) {
            cmd.env_remove(name);
        }
        cmd.env("MRUN_HOME", self.dir.path())
            .env("HOME", self.dir.path())
            .args(args)
            .envs(env.iter().copied())
            .output()
            .unwrap()
    }

    fn show(&self, env: &[(&str, &str)]) -> Value {
        let out = self.mrun(&["config", "show", "--json"], env);
        assert!(
            out.status.success(),
            "{}",
            String::from_utf8_lossy(&out.stderr)
        );
        serde_json::from_slice(&out.stdout).unwrap()
    }
}

fn text(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes).into_owned()
}

fn setting<'a>(shown: &'a Value, key: &str) -> &'a Value {
    shown["settings"]
        .as_array()
        .unwrap()
        .iter()
        .find(|s| s["key"] == key)
        .unwrap_or_else(|| panic!("no {key} in {shown}"))
}

#[test]
fn without_a_file_every_setting_is_the_default() {
    let home = Home::new();
    let path = home.mrun(&["config", "path"], &[]);
    assert!(text(&path.stdout).contains("(not found: built-in defaults apply)"));
    let shown = home.show(&[]);
    assert_eq!(shown["exists"], false);
    for s in shown["settings"].as_array().unwrap() {
        assert_eq!(s["source"], "default", "{s}");
    }
    let toml = home.mrun(&["config", "show"], &[]);
    assert!(text(&toml.stdout).starts_with("# config file: "));
    assert!(
        text(&toml.stdout)
            .lines()
            .next()
            .unwrap()
            .ends_with("(not found: built-in defaults apply)")
    );
    let none = home.mrun(&["config", "path"], &[("MRUN_CONFIG", "")]);
    assert_eq!(
        text(&none.stdout).trim(),
        "no config file (MRUN_CONFIG is empty)"
    );
}

#[test]
fn cloud_policy_comes_from_the_file() {
    // SC-001: the long `cloud up` command of 2026-10-24 as a five-line table.
    let home = Home::with_file(
        "[cloud]\nidle-stop = \"30m\"\nmax-runtime = \"8h\"\nretry = \"1h\"\nany-cloud = true\nmax-price = 2.6\n",
    );
    let shown = home.show(&[]);
    for (key, value, line) in [
        ("cloud.idle-stop", Value::from("30m"), 2),
        ("cloud.max-runtime", Value::from("8h"), 3),
        ("cloud.retry", Value::from("1h"), 4),
        ("cloud.any-cloud", Value::from(true), 5),
        ("cloud.max-price", Value::from(2.6), 6),
    ] {
        let s = setting(&shown, key);
        assert_eq!(s["value"], value, "{key}");
        assert_eq!(s["source"], format!("file:{line}"), "{key}");
    }
    assert_eq!(setting(&shown, "cloud.max-price")["needs"], "retry");
    // A confirmation is not a setting: it warns and the prompt stays.
    let yes = Home::with_file("[cloud]\nyes = true\n");
    let out = yes.mrun(&["config", "show", "--json"], &[]);
    let shown: Value = serde_json::from_slice(&out.stdout).unwrap();
    assert!(
        shown["warnings"][0]
            .as_str()
            .unwrap()
            .contains("unknown key cloud.yes")
    );
}

#[test]
fn variables_beat_the_file_and_tables_override_shared_keys() {
    let home = Home::with_file("port = 9000\nmodel = \"a\"\n\n[launch]\nmodel = \"b\"\n");
    let shown = home.show(&[("MRUN_PORT", "9100")]);
    let port = setting(&shown, "port");
    assert_eq!(port["value"], 9100);
    assert_eq!(port["source"], "env:MRUN_PORT");
    assert_eq!(port["overridden"][0]["source"], "file:1");
    assert_eq!(port["overridden"][0]["value"], 9000);
    assert_eq!(setting(&shown, "model")["value"], "a");
    assert_eq!(setting(&shown, "launch.model")["value"], "b");
    // The TOML view says the same.
    let out = home.mrun(&["config", "show"], &[("MRUN_PORT", "9100")]);
    assert!(
        text(&out.stdout).contains("port = 9100"),
        "{}",
        text(&out.stdout)
    );
    assert!(text(&out.stdout).contains("env MRUN_PORT (file:1 had 9000)"));
}

#[test]
fn shown_settings_read_back_to_the_same_values_and_fit_the_schema() {
    let home = Home::with_file(
        "model = \"qwen3.6\"\n[cloud]\nidle-stop = \"30m\"\nany-cloud = true\n[run]\ntemperature = 0.2\n",
    );
    let shown = home.mrun(&["config", "show"], &[]);
    let copy = home.dir.path().join("copy.toml");
    std::fs::write(&copy, &shown.stdout).unwrap();
    let again = home.show(&[("MRUN_CONFIG", copy.to_str().unwrap())]);
    assert!(again["warnings"].as_array().unwrap().is_empty(), "{again}");
    let first = home.show(&[]);
    let values = |v: &Value| -> Vec<(String, Value)> {
        v["settings"]
            .as_array()
            .unwrap()
            .iter()
            .map(|s| (s["key"].as_str().unwrap().to_string(), s["value"].clone()))
            .filter(|(_, v)| !v.is_null())
            .collect()
    };
    assert_eq!(values(&first), values(&again));
    // The schema accepts that file and rejects unknown keys (SC-007).
    let schema: Value =
        serde_json::from_slice(&home.mrun(&["config", "schema"], &[]).stdout).unwrap();
    let doc: toml::Table = toml::from_str(&text(&shown.stdout)).unwrap();
    assert_eq!(valid(&schema, &toml_to_json(&doc)), Vec::<String>::new());
    let wrong: toml::Table = toml::from_str("prot = 1\n[cloud]\nidle_stop = \"1m\"\n").unwrap();
    assert_eq!(valid(&schema, &toml_to_json(&wrong)).len(), 2);
    let typed: toml::Table =
        toml::from_str("[cloud]\nidle-stop = 30\nany-cloud = \"yes\"\n").unwrap();
    assert_eq!(valid(&schema, &toml_to_json(&typed)).len(), 2);
}

fn toml_to_json(table: &toml::Table) -> Value {
    serde_json::to_value(table).unwrap()
}

/// The subset of JSON Schema the config schema uses: object properties, `additionalProperties:
/// false`, `type`, `enum`, `anyOf`.
fn valid(schema: &Value, value: &Value) -> Vec<String> {
    let mut errors = Vec::new();
    check(schema, value, "", &mut errors);
    errors
}

fn check(schema: &Value, value: &Value, at: &str, errors: &mut Vec<String>) {
    if let Some(options) = schema["anyOf"].as_array() {
        if !options.iter().any(|o| valid(o, value).is_empty()) {
            errors.push(format!("{at}: matches no option"));
        }
        return;
    }
    let ok = match schema["type"].as_str() {
        Some("object") => value.is_object(),
        Some("string") => value.is_string(),
        Some("boolean") => value.is_boolean(),
        Some("integer") => value.is_i64() || value.is_u64(),
        Some("number") => value.is_number(),
        Some("array") => value.is_array(),
        _ => true,
    };
    if !ok {
        errors.push(format!("{at}: not a {}", schema["type"]));
        return;
    }
    if let Some(choices) = schema["enum"].as_array()
        && !choices.contains(value)
    {
        errors.push(format!("{at}: not one of {choices:?}"));
    }
    if let (Some(props), Some(object)) = (schema["properties"].as_object(), value.as_object()) {
        for (key, v) in object {
            match props.get(key) {
                Some(sub) => check(sub, v, &format!("{at}.{key}"), errors),
                None if schema["additionalProperties"] == false => {
                    errors.push(format!("{at}.{key}: unknown"));
                }
                None => {}
            }
        }
    }
    if let (Some(items), Some(list)) = (schema.get("items"), value.as_array()) {
        for (i, v) in list.iter().enumerate() {
            check(items, v, &format!("{at}[{i}]"), errors);
        }
    }
}

#[test]
fn mistakes_warn_or_stop_before_anything_runs() {
    // A misspelled section warns; the command still runs (and fails only for its own reason).
    let home = Home::with_file("[clould]\nidle-stop = \"30m\"\n");
    let out = home.mrun(&["ps"], &[("MRUN_PORT", "1")]);
    assert!(text(&out.stderr).contains("unknown section [clould]; did you mean [cloud]?"));
    assert!(
        text(&out.stderr).contains("no mrun server is running"),
        "{}",
        text(&out.stderr)
    );
    // An invalid value stops every command before it runs.
    let home = Home::with_file("[cloud]\nidle-stop = 30\n");
    let out = home.mrun(&["ps"], &[]);
    assert_eq!(out.status.code(), Some(2));
    let err = text(&out.stderr);
    assert!(
        err.contains("config.toml:2: cloud.idle-stop = 30: expected a string such as \"30m\""),
        "{err}"
    );
    assert!(!err.contains("no mrun server"), "nothing ran: {err}");
    // A syntax error too, but `config path` still answers.
    let home = Home::with_file("[cloud\n");
    let out = home.mrun(&["ps"], &[]);
    assert_eq!(out.status.code(), Some(2));
    assert!(text(&out.stderr).contains("config.toml:1:7:"));
    let out = home.mrun(&["config", "path"], &[]);
    assert!(out.status.success());
    assert!(text(&out.stdout).contains("(exists)"));
    // `MRUN_CONFIG` naming a missing file is an error; empty disables the file.
    let out = home.mrun(&["ps"], &[("MRUN_CONFIG", "/no/such/config.toml")]);
    assert!(text(&out.stderr).contains("(MRUN_CONFIG) does not exist"));
    let out = home.mrun(&["config", "show", "--json"], &[("MRUN_CONFIG", "")]);
    assert!(out.status.success());
}

#[test]
fn doctor_reports_the_config_file() {
    for (file, want) in [
        (None, "no config file at"),
        (Some("[serve]\nqueue = 4\n"), "config.toml: 1 setting"),
        (Some("[clould]\nx = 1\n"), "unknown section [clould]"),
        // A broken file stops every other command, but doctor runs and reports it.
        (Some("[cloud\n"), "config.toml:1:7:"),
    ] {
        let home = file.map_or_else(Home::new, Home::with_file);
        let out = home.mrun(&["doctor"], &[("MRUN_PORT", "1")]);
        let all = text(&out.stdout) + &text(&out.stderr);
        let line = all
            .lines()
            .find(|l| l.contains("config"))
            .unwrap_or_else(|| panic!("no config finding in {all}"));
        assert!(all.contains(want), "{line}\n{all}");
    }
}

#[test]
fn booleans_can_be_turned_off_on_the_command_line() {
    let home = Home::new();
    let help = text(&home.mrun(&["cloud", "up", "--help"], &[]).stdout);
    assert!(help.contains("--any-cloud[=<BOOL>]"), "{help}");
    let serve = text(&home.mrun(&["serve", "--help"], &[]).stdout);
    assert!(serve.contains("--no-cache[=<BOOL>]"), "{serve}");
}

#[test]
fn mrun_never_writes_the_file() {
    let body = "port = 9000\n[serve]\nqueue = 4\n[cloud]\napi-key = \"secret\"\n";
    let home = Home::with_file(body);
    let before = std::fs::metadata(home.file()).unwrap().modified().unwrap();
    for args in [
        &["config", "show"][..],
        &["config", "show", "--json"],
        &["config", "schema"],
        &["config", "path"],
        &["doctor"],
        &["serve", "--detach", "--model", "/no/such/model"],
    ] {
        let _ = home.mrun(args, &[("MRUN_PORT", "1")]);
    }
    assert_eq!(std::fs::read_to_string(home.file()).unwrap(), body);
    assert_eq!(
        std::fs::metadata(home.file()).unwrap().modified().unwrap(),
        before
    );
    // The secret-like key was refused, not used.
    let shown = home.show(&[]);
    assert!(
        shown["warnings"][0]
            .as_str()
            .unwrap()
            .contains("secrets come from the environment")
    );
    assert!(!shown.to_string().contains("\"secret\""));
}

#[test]
fn run_and_launch_defaults_come_from_the_file() {
    let home = Home::with_file("[run]\ntemperature = 0.2\n[launch]\nmodel = \"x\"\n");
    let shown = home.show(&[]);
    assert_eq!(setting(&shown, "run.temperature")["value"], 0.2);
    assert_eq!(setting(&shown, "launch.model")["value"], "x");
}
