//! `mrun config` (017): where the config file is (`path`), every setting in force with its source
//! (`show`), and a JSON Schema of the file for editors (`schema`). mrun never writes the file.

use serde_json::{Value, json};

use crate::config::{Effective, Kind, Location, Resolved, Setting, Source, effective};
use crate::errors::{CliError, Code};

#[derive(Debug, clap::Subcommand)]
pub enum ConfigCommand {
    /// The config file mrun reads, and whether it exists.
    Path,
    /// Every setting with the value a command would use and where it comes from (valid TOML).
    Show {
        /// Print JSON instead of TOML.
        #[arg(long)]
        json: bool,
    },
    /// A JSON Schema (draft 2020-12) of the config file, for editors.
    Schema,
}

pub fn run(command: ConfigCommand, resolved: &Resolved) -> u8 {
    match command {
        ConfigCommand::Path => {
            println!("{}", path_line(resolved));
            if let Err(error) = &resolved.file {
                eprintln!("{error}");
            }
            0
        }
        ConfigCommand::Show { json } => match &resolved.file {
            Err(error) => CliError::new(Code::InvalidArguments, error.to_string())
                .hint("fix the line above; 'mrun config path' shows the file")
                .report(),
            Ok(_) => {
                let env = |name: &str| std::env::var(name).ok();
                if json {
                    println!("{}", show_json(resolved, &env));
                } else {
                    print!("{}", show_toml(resolved, &env));
                }
                0
            }
        },
        ConfigCommand::Schema => {
            println!(
                "{}",
                serde_json::to_string_pretty(&schema(&resolved.settings)).unwrap_or_default()
            );
            0
        }
    }
}

/// `path (exists)`, `path (not found: built-in defaults apply)`, or no file.
pub fn path_line(resolved: &Resolved) -> String {
    match &resolved.location {
        Location::Disabled => "no config file (MRUN_CONFIG is empty)".into(),
        Location::File { path, .. } => {
            let exists = resolved.file.as_ref().map_or(true, |f| f.exists);
            if exists {
                format!("{} (exists)", path.display())
            } else {
                format!("{} (not found: built-in defaults apply)", path.display())
            }
        }
    }
}

/// A value in TOML syntax.
fn toml_value(setting: &Setting, values: &[String]) -> String {
    let one = |v: &String| match &setting.kind {
        Kind::Boolean | Kind::Integer | Kind::Number if !v.is_empty() => v.clone(),
        _ => toml::Value::String(v.clone()).to_string(),
    };
    if setting.repeatable {
        format!(
            "[{}]",
            values.iter().map(one).collect::<Vec<_>>().join(", ")
        )
    } else {
        values.first().map(one).unwrap_or_default()
    }
}

/// A value as JSON, typed like the setting.
fn json_value(setting: &Setting, values: &[String]) -> Value {
    let one = |v: &String| -> Value {
        match &setting.kind {
            Kind::Boolean => v.parse::<bool>().map_or_else(|_| json!(v), Value::Bool),
            Kind::Integer => v.parse::<i64>().map_or_else(|_| json!(v), |n| json!(n)),
            Kind::Number => v.parse::<f64>().map_or_else(|_| json!(v), |n| json!(n)),
            _ => json!(v),
        }
    };
    if setting.repeatable {
        Value::Array(values.iter().map(one).collect())
    } else {
        values.first().map_or(Value::Null, one)
    }
}

fn comment(setting: &Setting, e: &Effective) -> String {
    let mut text = match &e.source {
        Source::Env(var) => format!("env {var}"),
        other => other.to_string(),
    };
    let overridden: Vec<String> = e
        .overridden
        .iter()
        .map(|(source, values)| format!("{source} had {}", toml_value(setting, values)))
        .collect();
    if !overridden.is_empty() {
        text.push_str(&format!(" ({})", overridden.join(", ")));
    }
    if let Some(needs) = setting.needs() {
        text.push_str(&format!(" (needs {needs})"));
    }
    text
}

/// `config show`: valid TOML, each value with its source as a comment (017 FR-007).
pub fn show_toml(resolved: &Resolved, env: &dyn Fn(&str) -> Option<String>) -> String {
    let Ok(file) = &resolved.file else {
        return String::new();
    };
    let mut out = format!("# config file: {}\n", path_line(resolved));
    for warning in &file.warnings {
        out.push_str(&format!("# warning: {warning}\n"));
    }
    let all = effective(&resolved.settings, file, env);
    let line = |s: &Setting, e: &Effective| -> String {
        if e.values.is_empty() {
            let help = s.help.lines().next().unwrap_or_default();
            format!(
                "# {} ={:width$}# {help}\n",
                s.name,
                "",
                width = 24usize.saturating_sub(s.name.len() + 4)
            )
        } else {
            let assign = format!("{} = {}", s.name, toml_value(s, &e.values));
            format!("{assign:<27} # {}\n", comment(s, e))
        }
    };
    for e in all
        .iter()
        .filter(|e| resolved.settings[e.setting].table.is_none())
    {
        out.push_str(&line(&resolved.settings[e.setting], e));
    }
    for (table, _) in crate::config::TABLES {
        out.push_str(&format!("\n[{table}]\n"));
        for e in all
            .iter()
            .filter(|e| resolved.settings[e.setting].table == Some(table))
        {
            out.push_str(&line(&resolved.settings[e.setting], e));
        }
    }
    out
}

/// `config show --json`.
pub fn show_json(resolved: &Resolved, env: &dyn Fn(&str) -> Option<String>) -> String {
    let Ok(file) = &resolved.file else {
        return String::new();
    };
    let settings: Vec<Value> = effective(&resolved.settings, file, env)
        .iter()
        .map(|e| {
            let s = &resolved.settings[e.setting];
            json!({
                "key": s.key(),
                "value": json_value(s, &e.values),
                "source": e.source.to_string(),
                "overridden": e.overridden.iter().map(|(source, values)| json!({
                    "source": source.to_string(),
                    "value": json_value(s, values),
                })).collect::<Vec<_>>(),
                "needs": s.needs(),
            })
        })
        .collect();
    let value = json!({
        "path": resolved.location.path().map(|p| p.display().to_string()),
        "exists": file.exists,
        "settings": settings,
        "warnings": file.warnings,
    });
    serde_json::to_string_pretty(&value).unwrap_or_default()
}

fn property(setting: &Setting) -> Value {
    let item = match &setting.kind {
        Kind::Boolean => json!({ "type": "boolean" }),
        Kind::Integer => json!({ "type": "integer" }),
        Kind::Number => json!({ "type": "number" }),
        Kind::Choice(choices) => json!({ "type": "string", "enum": choices }),
        Kind::String => json!({ "type": "string" }),
    };
    let mut prop = if setting.repeatable {
        json!({ "anyOf": [item, { "type": "array", "items": item }] })
    } else {
        item
    };
    let object = prop.as_object_mut().expect("an object");
    if !setting.help.is_empty() {
        object.insert("description".into(), json!(setting.help));
    }
    if !setting.default.is_empty() {
        object.insert("default".into(), json_value(setting, &setting.default));
    }
    prop
}

/// The JSON Schema (draft 2020-12) of the config file (017 FR-014).
pub fn schema(settings: &[Setting]) -> Value {
    let mut top = serde_json::Map::new();
    for s in settings.iter().filter(|s| s.table.is_none()) {
        top.insert(s.name.clone(), property(s));
    }
    for (table, path) in crate::config::TABLES {
        let mut props = serde_json::Map::new();
        for s in settings.iter().filter(|s| s.table == Some(table)) {
            props.insert(s.name.clone(), property(s));
        }
        top.insert(
            table.to_string(),
            json!({
                "type": "object",
                "description": format!("Settings of `mrun {}`.", path.join(" ")),
                "additionalProperties": false,
                "properties": props,
            }),
        );
    }
    json!({
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "mrun config file",
        "description": "~/.mrun/config.toml (or $MRUN_CONFIG). Keys are the flags' long names; later sources win: built-in defaults, this file, MRUN_* variables, flags.",
        "type": "object",
        "additionalProperties": false,
        "properties": top,
    })
}
