//! What each coding agent needs to use the local server (004 FR-005, FR-006, research R5): the
//! process-scoped setup `mrun launch` applies, and the persistent form `mrun connect` prints.
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
}

/// Minimum Claude Code version mrun supports.
pub const CLAUDE_CODE_MIN: (u32, u32, u32) = (2, 1, 273);
/// Context below which agents' system prompts and tool schemas leave too little room.
pub const SMALL_CONTEXT: u32 = 48 * 1024;
/// Claude Code's lower bound for `CLAUDE_CODE_AUTO_COMPACT_WINDOW`.
const AUTO_COMPACT_MIN: u32 = 100_000;
/// `API_TIMEOUT_MS`: 50 minutes, for long local prefills.
const API_TIMEOUT_MS: &str = "3000000";
/// Placeholder credential; mrun checks none on loopback.
const TOKEN: &str = "mrun";
/// Environment variable Codex reads the provider key from.
const CODEX_KEY_ENV: &str = "MRUN_API_KEY";

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
    let mut env: Vec<(&str, String)> = vec![
        ("ANTHROPIC_BASE_URL", server.base_url()),
        ("ANTHROPIC_AUTH_TOKEN", TOKEN.into()),
        ("ANTHROPIC_API_KEY", String::new()),
        ("ANTHROPIC_MODEL", id.into()),
        ("ANTHROPIC_DEFAULT_OPUS_MODEL", id.into()),
        ("ANTHROPIC_DEFAULT_SONNET_MODEL", id.into()),
        ("ANTHROPIC_DEFAULT_HAIKU_MODEL", id.into()),
        ("CLAUDE_CODE_SUBAGENT_MODEL", id.into()),
        ("CLAUDE_CODE_MAX_CONTEXT_TOKENS", server.context.to_string()),
        (
            "CLAUDE_CODE_MAX_OUTPUT_TOKENS",
            server.max_output.to_string(),
        ),
    ];
    if server.context >= AUTO_COMPACT_MIN {
        env.push((
            "CLAUDE_CODE_AUTO_COMPACT_WINDOW",
            server.context.to_string(),
        ));
    }
    env.extend([
        // Above capacity: an idle conversation is never compacted (the cache does not expire).
        (
            "CLAUDE_CODE_IDLE_COMPACT_MIN_TOKENS",
            (server.context + 1).to_string(),
        ),
        ("CLAUDE_CODE_GATEWAY_HINT_HEADERS", "1".into()),
        ("CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC", "1".into()),
        ("CLAUDE_CODE_TOTAL_TOKENS_REMINDER", "off".into()),
        ("API_TIMEOUT_MS", API_TIMEOUT_MS.into()),
        ("ENABLE_TOOL_SEARCH", "auto:10".into()),
    ]);
    env.into_iter().map(|(k, v)| (k.to_string(), v)).collect()
}

/// Codex's model catalog: the real window and output limit (it assumes 272K for unknown
/// models) and tool search for deferred MCP tools.
fn codex_catalog(server: &ServerInfo) -> String {
    let catalog = json!({"models": [{
        "slug": server.model_id,
        "display_name": server.model_id,
        "context_window": server.context,
        "max_output_tokens": server.max_output,
        "supports_search_tool": true,
    }]});
    serde_json::to_string_pretty(&catalog).expect("serializes")
}

fn codex_catalog_path(home: &Path) -> PathBuf {
    home.join("codex").join("models.json")
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
pub const CODEX_RESERVED: [&str; 3] = ["model_provider", "model_providers", "model_catalog_json"];

/// The setup `mrun launch` applies. `home` is mrun's state directory.
pub fn setup(client: Client, server: &ServerInfo, home: &Path) -> Setup {
    match client {
        Client::ClaudeCode => Setup {
            env: claude_env(server),
            ..Setup::default()
        },
        Client::Codex => {
            let catalog = codex_catalog_path(home);
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
                c(format!("model_providers.mrun.env_key=\"{CODEX_KEY_ENV}\"")),
                c(format!("model_catalog_json=\"{}\"", catalog.display())),
                c("web_search=\"disabled\"".into()),
            ]
            .concat();
            Setup {
                env: vec![(CODEX_KEY_ENV.into(), TOKEN.into())],
                args,
                files: vec![(catalog, codex_catalog(server))],
            }
        }
        Client::OpenCode => Setup {
            env: vec![(
                "OPENCODE_CONFIG_CONTENT".into(),
                opencode_config(server).to_string(),
            )],
            ..Setup::default()
        },
    }
}

/// The persistent form `mrun connect` prints: `(target path, text)` blocks to paste.
pub fn persistent(client: Client, server: &ServerInfo, home: &Path) -> Vec<(String, String)> {
    match client {
        Client::ClaudeCode => {
            let env: serde_json::Map<String, serde_json::Value> = claude_env(server)
                .into_iter()
                .map(|(k, v)| (k, v.into()))
                .collect();
            let text = serde_json::to_string_pretty(&json!({"env": env})).expect("serializes");
            vec![(
                "~/.claude/settings.json (merge the \"env\" block)".into(),
                text,
            )]
        }
        Client::Codex => {
            let catalog = codex_catalog_path(home);
            let profile = format!(
                "model = \"{id}\"\nmodel_provider = \"mrun\"\nmodel_catalog_json = \"{catalog}\"\nweb_search = \"disabled\"\n\n[model_providers.mrun]\nname = \"mrun\"\nbase_url = \"{base}/v1\"\nwire_api = \"responses\"\nenv_key = \"{CODEX_KEY_ENV}\"\n",
                id = server.model_id,
                catalog = catalog.display(),
                base = server.base_url(),
            );
            vec![
                (
                    "~/.codex/mrun.config.toml (then run: codex --profile mrun, with MRUN_API_KEY set to any value)".into(),
                    profile,
                ),
                (catalog.display().to_string(), codex_catalog(server)),
            ]
        }
        Client::OpenCode => vec![(
            "~/.config/opencode/opencode.json (merge the \"provider\" entry)".into(),
            serde_json::to_string_pretty(&opencode_config(server)).expect("serializes"),
        )],
    }
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
            model_id: "mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit".into(),
            context,
            max_output: 16_384,
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
        let s = setup(Client::ClaudeCode, &server(105_000), Path::new("/m"));
        let env = env_of(&s);
        assert_eq!(env["ANTHROPIC_BASE_URL"], "http://127.0.0.1:8484");
        assert_eq!(env["ANTHROPIC_API_KEY"], "");
        for slot in [
            "ANTHROPIC_MODEL",
            "ANTHROPIC_DEFAULT_OPUS_MODEL",
            "ANTHROPIC_DEFAULT_SONNET_MODEL",
            "ANTHROPIC_DEFAULT_HAIKU_MODEL",
            "CLAUDE_CODE_SUBAGENT_MODEL",
        ] {
            assert!(!env[slot].starts_with("claude-"), "{slot}");
        }
        assert_eq!(env["CLAUDE_CODE_MAX_CONTEXT_TOKENS"], "105000");
        assert_eq!(env["CLAUDE_CODE_AUTO_COMPACT_WINDOW"], "105000");
        assert_eq!(env["CLAUDE_CODE_IDLE_COMPACT_MIN_TOKENS"], "105001");
        assert_eq!(env["ENABLE_TOOL_SEARCH"], "auto:10");
        assert!(s.args.is_empty() && s.files.is_empty());
        // Below Claude Code's minimum the auto-compact window is not set.
        let small = setup(Client::ClaudeCode, &server(40_960), Path::new("/m"));
        assert!(!env_of(&small).contains_key("CLAUDE_CODE_AUTO_COMPACT_WINDOW"));
    }

    #[test]
    fn codex_selects_the_mrun_provider_with_a_catalog() {
        let s = setup(Client::Codex, &server(105_000), Path::new("/home/u/.mrun"));
        let args = s.args.join(" ");
        assert!(args.contains("-c model_provider=\"mrun\""));
        assert!(args.contains("model_providers.mrun.wire_api=\"responses\""));
        assert!(args.contains("web_search=\"disabled\""));
        assert!(args.contains("model_catalog_json=\"/home/u/.mrun/codex/models.json\""));
        let (path, catalog) = &s.files[0];
        assert_eq!(path, Path::new("/home/u/.mrun/codex/models.json"));
        let catalog: serde_json::Value = serde_json::from_str(catalog).unwrap();
        assert_eq!(catalog["models"][0]["context_window"], 105_000);
        assert_eq!(catalog["models"][0]["supports_search_tool"], true);
        assert_eq!(env_of(&s)[CODEX_KEY_ENV], "mrun");
    }

    #[test]
    fn opencode_gets_limits_through_the_config_content() {
        let s = setup(Client::OpenCode, &server(262_144), Path::new("/m"));
        let config: serde_json::Value =
            serde_json::from_str(env_of(&s)["OPENCODE_CONFIG_CONTENT"]).unwrap();
        let model = &config["provider"]["mrun"]["models"]["mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit"];
        assert_eq!(model["limit"]["context"], 262_144);
        assert_eq!(model["limit"]["input"], 262_144 - 16_384);
        assert_eq!(
            config["provider"]["mrun"]["options"]["baseURL"],
            "http://127.0.0.1:8484/v1"
        );
    }

    #[test]
    fn connect_prints_targets_for_every_client() {
        for client in [Client::ClaudeCode, Client::Codex, Client::OpenCode] {
            let blocks = persistent(client, &server(105_000), Path::new("/home/u/.mrun"));
            assert!(!blocks.is_empty());
            for (target, text) in blocks {
                assert!(!target.is_empty() && !text.is_empty());
            }
        }
        let codex = persistent(Client::Codex, &server(105_000), Path::new("/m"));
        assert!(codex[0].1.contains("[model_providers.mrun]"));
    }

    #[test]
    fn versions_parse() {
        assert_eq!(parse_version("2.1.300 (Claude Code)"), Some((2, 1, 300)));
        assert_eq!(parse_version("claude 2.1.9"), Some((2, 1, 9)));
        assert!(parse_version("2.1.272 (Claude Code)").unwrap() < CLAUDE_CODE_MIN);
        assert_eq!(parse_version("unknown"), None);
    }
}
