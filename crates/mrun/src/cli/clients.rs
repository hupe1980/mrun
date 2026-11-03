//! What each coding agent needs to use the local server (004 FR-005, FR-006, research R5, D037):
//! the process-scoped setup `mrun launch` applies, and the persistent form `mrun connect` prints.
//! Pure functions of the running server's values, so the output is snapshot-tested.

use crate::cli::server::ServerInfo;
use serde_json::json;
use std::path::{Path, PathBuf};

#[derive(Clone, Copy, Debug, Eq, PartialEq, clap::ValueEnum)]
pub enum Client {
    #[value(name = "claude-code")]
    ClaudeCode,
    Codex,
    #[value(name = "opencode")]
    OpenCode,
}

impl Client {
    pub const ALL: [Client; 3] = [Self::ClaudeCode, Self::Codex, Self::OpenCode];

    /// The name on the command line.
    pub fn name(self) -> &'static str {
        match self {
            Self::ClaudeCode => "claude-code",
            Self::Codex => "codex",
            Self::OpenCode => "opencode",
        }
    }

    pub fn binary(self) -> &'static str {
        match self {
            Self::ClaudeCode => "claude",
            Self::Codex => "codex",
            Self::OpenCode => "opencode",
        }
    }

    pub fn install_hint(self) -> &'static str {
        match self {
            Self::ClaudeCode => "npm install -g @anthropic-ai/claude-code",
            Self::Codex => "npm install -g @openai/codex (or brew install codex)",
            Self::OpenCode => "npm install -g opencode-ai (or brew install opencode)",
        }
    }

    /// Clients whose program is on `PATH`.
    pub fn installed() -> Vec<Client> {
        Self::ALL
            .into_iter()
            .filter(|client| crate::cli::which(client.binary()).is_some())
            .collect()
    }
}

/// Minimum Claude Code version mrun supports (`ANTHROPIC_DEFAULT_MODEL` needs 2.1.236, gateway
/// hint headers 2.1.273).
pub const CLAUDE_CODE_MIN: (u32, u32, u32) = (2, 1, 290);
/// Claude Code's lower bound for `CLAUDE_CODE_AUTO_COMPACT_WINDOW`.
const COMPACT_MIN: u32 = 100_000;
/// `API_TIMEOUT_MS`: 50 minutes, for long local prefills.
const API_TIMEOUT_MS: &str = "3000000";
/// Placeholder credential; mrun checks none on loopback.
const TOKEN: &str = "mrun";

impl Client {
    /// The context below which `launch` refuses a model for this client (FR-027, D048, D055;
    /// research R5 "Minimum context per client"). What each client spends before the first
    /// user turn, plus room to work before it compacts:
    /// - Claude Code: system prompt, tool schemas and compaction need 64K.
    /// - OpenCode: system prompt and tool descriptions (about 12K tokens) and it subtracts the
    ///   output limit mrun announces (32,768 by default) from the window before compacting, so
    ///   less than 64K leaves almost no room.
    /// - Codex: base instructions for an unknown model (about 5K tokens), tools (about 2K) and
    ///   `AGENTS.md` (at most 32 KiB, about 8K) take about 15K; 32K leaves half the window.
    ///
    /// Engineering minimums until 002's client gate measures the real floor.
    pub fn min_context(self) -> u32 {
        match self {
            Self::ClaudeCode | Self::OpenCode => 65_536,
            Self::Codex => 32_768,
        }
    }
}

/// Every variable `launch` may set for Claude Code; `~/.claude/settings.json` overriding one of
/// them defeats `launch`. Each one was checked against Claude Code's environment variable
/// reference (code.claude.com/docs/en/env-vars, 2026-10-16). Not set, on purpose:
/// `CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY` (discovery keeps only ids containing `claude` or
/// `anthropic`, so mrun's models would never appear), `CLAUDE_CODE_WORKFLOW_SUBAGENT_MODEL` (not
/// documented; `CLAUDE_CODE_SUBAGENT_MODEL` covers workflow agents) and
/// `CLAUDE_CODE_IDLE_COMPACT_MIN_TOKENS` (idle compaction is switched off with the
/// `idleCompaction` setting instead, D047).
pub const CLAUDE_KEYS: [&str; 27] = [
    "ANTHROPIC_BASE_URL",
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_MODEL",
    "ANTHROPIC_DEFAULT_MODEL",
    "ANTHROPIC_DEFAULT_OPUS_MODEL",
    "ANTHROPIC_DEFAULT_SONNET_MODEL",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL",
    "ANTHROPIC_DEFAULT_FABLE_MODEL",
    "CLAUDE_CODE_SUBAGENT_MODEL",
    "CLAUDE_CODE_SUBAGENT_MODEL_FORCE",
    "CLAUDE_CODE_USE_BEDROCK",
    "CLAUDE_CODE_USE_VERTEX",
    "CLAUDE_CODE_USE_FOUNDRY",
    "CLAUDE_CODE_USE_ANTHROPIC_AWS",
    "CLAUDE_CODE_USE_MANTLE",
    "CLAUDE_CODE_ATTRIBUTION_HEADER",
    "CLAUDE_CODE_MAX_CONTEXT_TOKENS",
    "CLAUDE_CODE_MAX_OUTPUT_TOKENS",
    "CLAUDE_CODE_AUTO_COMPACT_WINDOW",
    "CLAUDE_CODE_DISABLE_NONSTREAMING_FALLBACK",
    "CLAUDE_CODE_GATEWAY_HINT_HEADERS",
    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC",
    "DISABLE_ERROR_REPORTING",
    "CLAUDE_CODE_DISABLE_FEEDBACK_SURVEY",
    "API_TIMEOUT_MS",
    "ENABLE_TOOL_SEARCH",
];

/// Claude Code settings `launch` passes with `--settings` and `connect` writes: idle compaction
/// off (mrun's cache does not expire, so compacting an idle conversation only loses context).
fn claude_settings() -> serde_json::Value {
    json!({"idleCompaction": false})
}

/// The process-scoped configuration of one client.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct Setup {
    pub env: Vec<(String, String)>,
    pub args: Vec<String>,
    /// Files mrun writes under its own state directory (never the client's files).
    pub files: Vec<(PathBuf, String)>,
}

fn claude_env(server: &ServerInfo) -> Vec<(String, String)> {
    let id = server.model_id.as_str();
    // Claude Code appends `/v1/messages` itself; a base URL ending in `/v1` doubles it.
    let base = server
        .base_url()
        .trim_end_matches('/')
        .trim_end_matches("/v1")
        .to_string();
    let mut env: Vec<(&str, String)> = vec![
        ("ANTHROPIC_BASE_URL", base),
        ("ANTHROPIC_AUTH_TOKEN", TOKEN.into()),
        ("ANTHROPIC_API_KEY", String::new()),
        ("ANTHROPIC_MODEL", id.into()),
        ("ANTHROPIC_DEFAULT_MODEL", id.into()),
        ("ANTHROPIC_DEFAULT_OPUS_MODEL", id.into()),
        ("ANTHROPIC_DEFAULT_SONNET_MODEL", id.into()),
        ("ANTHROPIC_DEFAULT_HAIKU_MODEL", id.into()),
        ("ANTHROPIC_DEFAULT_FABLE_MODEL", id.into()),
        ("CLAUDE_CODE_SUBAGENT_MODEL", id.into()),
        // Agent definitions that name a Claude model (`model: sonnet`) run on the local one too
        // (Claude Code 2.1.257).
        ("CLAUDE_CODE_SUBAGENT_MODEL_FORCE", "1".into()),
        // A provider switch exported by a shell profile would route around ANTHROPIC_BASE_URL.
        ("CLAUDE_CODE_USE_BEDROCK", "0".into()),
        ("CLAUDE_CODE_USE_VERTEX", "0".into()),
        ("CLAUDE_CODE_USE_FOUNDRY", "0".into()),
        ("CLAUDE_CODE_USE_ANTHROPIC_AWS", "0".into()),
        ("CLAUDE_CODE_USE_MANTLE", "0".into()),
        // The attribution block changes per request and would defeat prefix reuse.
        ("CLAUDE_CODE_ATTRIBUTION_HEADER", "0".into()),
        ("CLAUDE_CODE_MAX_CONTEXT_TOKENS", server.context.to_string()),
        (
            "CLAUDE_CODE_MAX_OUTPUT_TOKENS",
            server.max_output.to_string(),
        ),
    ];
    if server.context >= COMPACT_MIN {
        env.push((
            "CLAUDE_CODE_AUTO_COMPACT_WINDOW",
            server.context.to_string(),
        ));
    }
    if server.recording {
        // Session and turn ids in headers, for traces recorded with `serve --record`.
        env.push(("CLAUDE_CODE_GATEWAY_HINT_HEADERS", "1".into()));
    }
    env.extend([
        // The non-streaming fallback re-sends the whole prompt after a stream fails.
        ("CLAUDE_CODE_DISABLE_NONSTREAMING_FALLBACK", "1".into()),
        ("CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC", "1".into()),
        // Presence-only: any value (even 0) opts out.
        ("DISABLE_ERROR_REPORTING", "1".into()),
        ("CLAUDE_CODE_DISABLE_FEEDBACK_SURVEY", "1".into()),
        ("API_TIMEOUT_MS", API_TIMEOUT_MS.into()),
        ("ENABLE_TOOL_SEARCH", "auto:10".into()),
    ]);
    env.into_iter().map(|(k, v)| (k.to_string(), v)).collect()
}

/// Claude Code's arguments: the settings above and WebSearch removed from the tool list.
fn claude_args() -> Vec<String> {
    vec![
        "--settings".into(),
        claude_settings().to_string(),
        "--disallowedTools".into(),
        "WebSearch".into(),
    ]
}

/// Where `connect` tells Claude Code users to save the settings for `claude --settings`.
fn claude_settings_path(home: &Path) -> PathBuf {
    home.join("claude").join("settings.json")
}

fn opencode_config(server: &ServerInfo) -> serde_json::Value {
    let id = &server.model_id;
    json!({
        "$schema": "https://opencode.ai/config.json",
        "provider": {"mrun": {
            "npm": "@ai-sdk/openai-compatible",
            "name": "mrun",
            "options": {"baseURL": format!("{}/v1", server.base_url()), "apiKey": TOKEN},
            "models": {id.as_str(): {
                "name": id,
                "limit": {
                    "context": server.context,
                    "input": server.context.saturating_sub(server.max_output),
                    "output": server.max_output,
                },
            }},
        }},
        "model": format!("mrun/{id}"),
    })
}

/// `-c` keys a user may not override: they select the provider mrun configures.
/// `model_context_window` replaces the 272K Codex assumes for an unknown model; a model catalog
/// file is not written (its schema changes between Codex releases: 0.163 requires a dozen fields).
pub const CODEX_RESERVED: [&str; 3] = ["model_provider", "model_providers", "model_context_window"];

/// The setup `mrun launch` applies.
pub fn setup(client: Client, server: &ServerInfo) -> Setup {
    match client {
        Client::ClaudeCode => Setup {
            env: claude_env(server),
            args: claude_args(),
            ..Setup::default()
        },
        Client::Codex => {
            let c = |kv: String| ["-c".to_string(), kv];
            let args = [
                c(format!("model=\"{}\"", server.model_id)),
                c("model_provider=\"mrun\"".into()),
                c("model_providers.mrun.name=\"mrun\"".into()),
                c(format!(
                    "model_providers.mrun.base_url=\"{}/v1\"",
                    server.base_url()
                )),
                c("model_providers.mrun.wire_api=\"responses\"".into()),
                c(format!("model_context_window={}", server.context)),
                c("web_search=\"disabled\"".into()),
            ]
            .concat();
            Setup {
                args,
                ..Setup::default()
            }
        }
        Client::OpenCode => {
            let mut config = opencode_config(server);
            // OpenCode 2.x's Code Mode tool (`execute`: JavaScript that calls OpenCode's own and
            // MCP tools) is one the served models were not trained on: Qwen3-Coder-Next routes
            // shell commands through it (`tools.shell(...)`, which Code Mode does not have) and
            // fails. The direct tools (shell, read, edit, ...) stay.
            config["tools"] = json!({"execute": false});
            Setup {
                env: vec![("OPENCODE_CONFIG_CONTENT".into(), config.to_string())],
                ..Setup::default()
            }
        }
    }
}

/// What `mrun connect` configures: the three launchable clients plus two print-only ones.
#[derive(Clone, Copy, Debug, Eq, PartialEq, clap::ValueEnum)]
pub enum Target {
    #[value(name = "claude-code")]
    ClaudeCode,
    Codex,
    #[value(name = "opencode")]
    OpenCode,
    Crush,
    Zed,
}

impl Target {
    pub fn name(self) -> &'static str {
        match self {
            Self::ClaudeCode => "claude-code",
            Self::Codex => "codex",
            Self::OpenCode => "opencode",
            Self::Crush => "crush",
            Self::Zed => "zed",
        }
    }

    /// Whether `connect --write` may write this client's configuration (Crush and Zed are
    /// print only: their files are the user's own, with comments and settings mrun cannot merge
    /// safely).
    pub fn writable(self) -> bool {
        !matches!(self, Self::Crush | Self::Zed)
    }
}

/// How a block of `connect` output lands in its file.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Apply {
    /// The whole file is mrun's (or a profile named for mrun): written as is.
    Replace,
    /// A JSON file of the user's: the value at this key path is set, the rest is kept.
    MergeJson(Vec<&'static str>),
    /// Shown only; `--write` never touches it.
    PrintOnly,
}

/// One block of `mrun connect` output: the target file, how to use it, the text.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Block {
    pub path: PathBuf,
    /// What to do with it (shown next to the path).
    pub note: String,
    pub text: String,
    pub apply: Apply,
}

/// The user's directories `connect` targets; tests pass their own.
#[derive(Clone, Debug)]
pub struct Homes {
    /// mrun's state directory.
    pub mrun: PathBuf,
    /// `CODEX_HOME`, else `~/.codex`.
    pub codex: PathBuf,
    /// `XDG_CONFIG_HOME`, else `~/.config`.
    pub config: PathBuf,
}

impl Homes {
    pub fn from_env() -> Self {
        let var = |key: &str| {
            std::env::var_os(key)
                .filter(|value| !value.is_empty())
                .map(PathBuf::from)
        };
        let home = var("HOME")
            .or_else(std::env::home_dir)
            .unwrap_or_else(|| PathBuf::from("."));
        Self {
            mrun: crate::cli::mrun_home(),
            codex: var("CODEX_HOME").unwrap_or_else(|| home.join(".codex")),
            config: var("XDG_CONFIG_HOME").unwrap_or_else(|| home.join(".config")),
        }
    }
}

/// Codex's separate profile file (D055): `codex --profile mrun` reads it, `config.toml` stays
/// untouched.
fn codex_profile(server: &ServerInfo) -> String {
    format!(
        "model = \"{id}\"\nmodel_provider = \"mrun\"\nmodel_context_window = {context}\nweb_search = \"disabled\"\n\n[model_providers.mrun]\nname = \"mrun\"\nbase_url = \"{base}/v1\"\nwire_api = \"responses\"\n",
        id = server.model_id,
        context = server.context,
        base = server.base_url(),
    )
}

/// Crush's `crushrc` commands for an OpenAI-compatible provider.
fn crush_rc(server: &ServerInfo) -> String {
    let id = &server.model_id;
    format!(
        "provider add mrun --name mrun --type openai-compat \\\n  --base-url \"{base}/v1\" \\\n  --api-key {TOKEN}\n\nmodel add mrun/{id} \\\n  --name \"{id}\" \\\n  --context-window {context} \\\n  --default-max-tokens {output}\n",
        base = server.base_url(),
        context = server.context,
        output = server.max_output,
    )
}

/// Zed's `language_models` entry; `max_tokens` is the capacity.
fn zed_settings(server: &ServerInfo) -> serde_json::Value {
    json!({"language_models": {"openai_compatible": {"mrun": {
        "api_url": format!("{}/v1", server.base_url()),
        "available_models": [{
            "name": server.model_id,
            "display_name": server.model_id,
            "max_tokens": server.context,
            "max_output_tokens": server.max_output,
            "capabilities": {
                "tools": true,
                "images": false,
                "parallel_tool_calls": false,
                "prompt_cache_key": false,
            },
        }],
    }}}})
}

/// The persistent form `mrun connect` prints (and `--write` applies).
pub fn persistent(target: Target, server: &ServerInfo, homes: &Homes) -> Vec<Block> {
    match target {
        Target::ClaudeCode => {
            let env: serde_json::Map<String, serde_json::Value> = claude_env(server)
                .into_iter()
                .map(|(k, v)| (k, v.into()))
                .collect();
            let mut settings = claude_settings();
            settings["env"] = env.into();
            settings["permissions"] = json!({"deny": ["WebSearch"]});
            let path = claude_settings_path(&homes.mrun);
            vec![Block {
                note: format!(
                    "then run: claude --settings {}; your ~/.claude/settings.json stays as it is",
                    path.display()
                ),
                path,
                text: serde_json::to_string_pretty(&settings).expect("serializes"),
                apply: Apply::Replace,
            }]
        }
        Target::Codex => vec![Block {
            path: homes.codex.join("mrun.config.toml"),
            note: "then run: codex --profile mrun; config.toml stays as it is".into(),
            text: codex_profile(server),
            apply: Apply::Replace,
        }],
        Target::OpenCode => {
            let config = opencode_config(server);
            vec![Block {
                path: homes.config.join("opencode").join("opencode.json"),
                note: "merge the \"provider\" entry; pick mrun with /models (a top-level \"model\" would make it opencode's default for every project)".into(),
                text: serde_json::to_string_pretty(&json!({
                    "$schema": config["$schema"],
                    "provider": config["provider"],
                }))
                .expect("serializes"),
                apply: Apply::MergeJson(vec!["provider", "mrun"]),
            }]
        }
        Target::Crush => vec![Block {
            path: homes.config.join("crush").join("crushrc"),
            note: "append these lines (print only)".into(),
            text: crush_rc(server),
            apply: Apply::PrintOnly,
        }],
        Target::Zed => vec![Block {
            path: homes.config.join("zed").join("settings.json"),
            note: "merge into Zed's settings (print only); enter any API key (e.g. mrun) when Zed asks".into(),
            text: serde_json::to_string_pretty(&zed_settings(server)).expect("serializes"),
            apply: Apply::PrintOnly,
        }],
    }
}

/// Keys of the `env` block in Claude Code's user settings (`$CLAUDE_CONFIG_DIR/settings.json`,
/// else `~/.claude/settings.json`) that override what `launch` sets.
/// Claude Code settings files whose `env` overrides what `launch` sets, with the keys: the user
/// file (`CLAUDE_CONFIG_DIR` or `~/.claude/settings.json`) and the project files of the current
/// directory (`./.claude/settings.json`, `./.claude/settings.local.json`).
pub fn claude_settings_overrides() -> Vec<(String, Vec<String>)> {
    let user = std::env::var_os("CLAUDE_CONFIG_DIR")
        .filter(|dir| !dir.is_empty())
        .map(|dir| {
            let path = PathBuf::from(dir).join("settings.json");
            (path.display().to_string(), path)
        })
        .or_else(|| {
            std::env::home_dir().map(|home| {
                (
                    "~/.claude/settings.json".to_string(),
                    home.join(".claude/settings.json"),
                )
            })
        });
    let project = ["settings.json", "settings.local.json"]
        .map(|name| (format!("./.claude/{name}"), Path::new(".claude").join(name)));
    user.into_iter()
        .chain(project)
        .filter_map(|(label, path)| {
            let keys = overriding_keys(&std::fs::read_to_string(path).ok()?);
            (!keys.is_empty()).then_some((label, keys))
        })
        .collect()
}

/// `~/.claude/settings.json sets A, B; ./.claude/settings.json sets C`.
pub fn describe_overrides(overrides: &[(String, Vec<String>)]) -> String {
    overrides
        .iter()
        .map(|(file, keys)| format!("{file} sets {}", keys.join(", ")))
        .collect::<Vec<_>>()
        .join("; ")
}

fn overriding_keys(settings: &str) -> Vec<String> {
    let Ok(settings) = serde_json::from_str::<serde_json::Value>(settings) else {
        return Vec::new();
    };
    let Some(env) = settings["env"].as_object() else {
        return Vec::new();
    };
    env.keys()
        .filter(|key| CLAUDE_KEYS.contains(&key.as_str()))
        .cloned()
        .collect()
}

/// Parses `claude --version` output such as `2.1.300 (Claude Code)`.
pub fn parse_version(text: &str) -> Option<(u32, u32, u32)> {
    let word = text.split_whitespace().find(|w| w.contains('.'))?;
    let mut parts = word.split('.').map(|p| p.parse::<u32>().ok());
    Some((parts.next()??, parts.next()??, parts.next()??))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn server(context: u32) -> ServerInfo {
        ServerInfo {
            port: 8484,
            model_id: "mlx-community/Qwen3.6-35B-A3B-4bit".into(),
            context,
            max_output: 16_384,
            recording: false,
        }
    }

    fn env_of(setup: &Setup) -> std::collections::HashMap<&str, &str> {
        setup
            .env
            .iter()
            .map(|(k, v)| (k.as_str(), v.as_str()))
            .collect()
    }

    #[test]
    fn claude_code_gets_every_slot_and_the_real_window() {
        let s = setup(Client::ClaudeCode, &server(105_000));
        let env = env_of(&s);
        assert_eq!(env["ANTHROPIC_BASE_URL"], "http://127.0.0.1:8484");
        assert!(!env["ANTHROPIC_BASE_URL"].ends_with("/v1"));
        assert_eq!(env["ANTHROPIC_API_KEY"], "");
        for slot in [
            "ANTHROPIC_MODEL",
            "ANTHROPIC_DEFAULT_MODEL",
            "ANTHROPIC_DEFAULT_OPUS_MODEL",
            "ANTHROPIC_DEFAULT_SONNET_MODEL",
            "ANTHROPIC_DEFAULT_HAIKU_MODEL",
            "ANTHROPIC_DEFAULT_FABLE_MODEL",
            "CLAUDE_CODE_SUBAGENT_MODEL",
        ] {
            assert_eq!(env[slot], "mlx-community/Qwen3.6-35B-A3B-4bit", "{slot}");
        }
        for provider in ["BEDROCK", "VERTEX", "FOUNDRY", "ANTHROPIC_AWS", "MANTLE"] {
            assert_eq!(env[format!("CLAUDE_CODE_USE_{provider}").as_str()], "0");
        }
        assert_eq!(env["CLAUDE_CODE_SUBAGENT_MODEL_FORCE"], "1");
        assert_eq!(env["CLAUDE_CODE_ATTRIBUTION_HEADER"], "0");
        assert_eq!(env["CLAUDE_CODE_MAX_CONTEXT_TOKENS"], "105000");
        assert_eq!(env["CLAUDE_CODE_AUTO_COMPACT_WINDOW"], "105000");
        assert_eq!(env["CLAUDE_CODE_DISABLE_NONSTREAMING_FALLBACK"], "1");
        assert_eq!(env["DISABLE_ERROR_REPORTING"], "1");
        assert_eq!(env["CLAUDE_CODE_DISABLE_FEEDBACK_SURVEY"], "1");
        assert_eq!(env["ENABLE_TOOL_SEARCH"], "auto:10");
        for absent in [
            "CLAUDE_CODE_IDLE_COMPACT_MIN_TOKENS",
            "CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY",
            "CLAUDE_CODE_WORKFLOW_SUBAGENT_MODEL",
            "CLAUDE_CODE_GATEWAY_HINT_HEADERS",
        ] {
            assert!(!env.contains_key(absent), "{absent}");
        }
        assert_eq!(
            s.args,
            [
                "--settings",
                r#"{"idleCompaction":false}"#,
                "--disallowedTools",
                "WebSearch"
            ]
        );
        assert!(s.files.is_empty());
        for key in env.keys() {
            assert!(CLAUDE_KEYS.contains(key), "{key} is in CLAUDE_KEYS");
        }

        // Below Claude Code's lower bound the auto-compact window is not sent.
        let small_setup = setup(Client::ClaudeCode, &server(40_960));
        let small_env = env_of(&small_setup);
        assert!(!small_env.contains_key("CLAUDE_CODE_AUTO_COMPACT_WINDOW"));
        assert_eq!(small_env.len(), env.len() - 1);

        let recording = ServerInfo {
            recording: true,
            ..server(105_000)
        };
        let s = setup(Client::ClaudeCode, &recording);
        assert_eq!(env_of(&s)["CLAUDE_CODE_GATEWAY_HINT_HEADERS"], "1");
    }

    #[test]
    fn codex_selects_the_mrun_provider_with_its_context_window() {
        let s = setup(Client::Codex, &server(105_000));
        let args = s.args.join(" ");
        assert!(args.contains("-c model_provider=\"mrun\""));
        assert!(args.contains("model_providers.mrun.wire_api=\"responses\""));
        assert!(args.contains("web_search=\"disabled\""));
        assert!(args.contains("-c model_context_window=105000"));
        assert!(!args.contains("model_catalog_json"));
        assert!(!args.contains("env_key"));
        assert!(s.env.is_empty());
        assert!(
            s.files.is_empty(),
            "no catalog file: its schema changes per Codex release"
        );
    }

    #[test]
    fn opencode_gets_limits_through_the_config_content() {
        let s = setup(Client::OpenCode, &server(262_144));
        let config: serde_json::Value =
            serde_json::from_str(env_of(&s)["OPENCODE_CONFIG_CONTENT"]).unwrap();
        let model = &config["provider"]["mrun"]["models"]["mlx-community/Qwen3.6-35B-A3B-4bit"];
        assert_eq!(model["limit"]["context"], 262_144);
        assert_eq!(model["limit"]["input"], 262_144 - 16_384);
        assert_eq!(
            config["provider"]["mrun"]["options"]["baseURL"],
            "http://127.0.0.1:8484/v1"
        );
        assert_eq!(
            config["tools"]["execute"], false,
            "Code Mode off: the models call the direct tools"
        );
    }

    fn homes() -> Homes {
        Homes {
            mrun: "/home/u/.mrun".into(),
            codex: "/home/u/.codex".into(),
            config: "/home/u/.config".into(),
        }
    }

    #[test]
    fn connect_prints_targets_for_every_client() {
        use clap::ValueEnum;
        for target in Target::value_variants() {
            let blocks = persistent(*target, &server(105_000), &homes());
            assert!(!blocks.is_empty());
            for block in blocks {
                assert!(!block.note.is_empty() && !block.text.is_empty());
                assert_eq!(block.apply == Apply::PrintOnly, !target.writable());
            }
        }
        let codex = persistent(Target::Codex, &server(105_000), &homes());
        assert_eq!(codex[0].path, Path::new("/home/u/.codex/mrun.config.toml"));
        assert!(codex[0].text.contains("[model_providers.mrun]"));
        assert!(!codex[0].text.contains("env_key"));
        let claude = persistent(Target::ClaudeCode, &server(105_000), &homes());
        assert!(
            claude[0]
                .note
                .contains("claude --settings /home/u/.mrun/claude/settings.json")
        );
        let settings: serde_json::Value = serde_json::from_str(&claude[0].text).unwrap();
        assert_eq!(settings["idleCompaction"], false);
        assert_eq!(settings["permissions"]["deny"][0], "WebSearch");
        let opencode = persistent(Target::OpenCode, &server(105_000), &homes());
        assert!(opencode[0].note.contains("top-level \"model\""));
        assert!(!opencode[0].text.contains("\"model\": \"mrun/"));

        let crush = &persistent(Target::Crush, &server(105_000), &homes())[0];
        assert!(
            crush.text.contains("--type openai-compat"),
            "{}",
            crush.text
        );
        assert!(crush.text.contains("--context-window 105000"));
        assert!(
            crush
                .text
                .contains("--base-url \"http://127.0.0.1:8484/v1\"")
        );
        let zed = &persistent(Target::Zed, &server(105_000), &homes())[0];
        let zed: serde_json::Value = serde_json::from_str(&zed.text).unwrap();
        let mrun = &zed["language_models"]["openai_compatible"]["mrun"];
        assert_eq!(mrun["available_models"][0]["max_tokens"], 105_000);
        assert_eq!(mrun["api_url"], "http://127.0.0.1:8484/v1");
    }

    #[test]
    fn settings_env_that_overrides_launch_is_found() {
        let settings = r#"{"env": {"ANTHROPIC_BASE_URL": "https://proxy", "FOO": "1",
            "ANTHROPIC_MODEL": "claude-x"}, "theme": "dark"}"#;
        assert_eq!(
            overriding_keys(settings),
            ["ANTHROPIC_BASE_URL", "ANTHROPIC_MODEL"]
        );
        assert!(overriding_keys(r#"{"theme": "dark"}"#).is_empty());
        assert!(overriding_keys("not json").is_empty());
    }

    #[test]
    fn versions_parse() {
        assert_eq!(parse_version("2.1.300 (Claude Code)"), Some((2, 1, 300)));
        assert_eq!(parse_version("claude 2.1.9"), Some((2, 1, 9)));
        assert!(parse_version("2.1.289 (Claude Code)").unwrap() < CLAUDE_CODE_MIN);
        assert_eq!(parse_version("unknown"), None);
    }
}
