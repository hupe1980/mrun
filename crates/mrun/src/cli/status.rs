//! `mrun` without a command: the server, the pulled models and the next three commands (D037).

use crate::cli::server::{self, Health};
use crate::hub::{self, Ctx, aliases};
use std::fmt::Write;

pub fn run() -> u8 {
    let ctx = Ctx::from_env();
    let port = server::default_port();
    let pulled: Vec<String> = hub::manifests(&ctx.paths)
        .iter()
        .map(|m| match ctx.device {
            Some(device) => {
                let fit = m.fit(device);
                let estimated = if device.estimated { ", estimated" } else { "" };
                if fit.fits {
                    format!("{} (fits: {} tokens{estimated})", m.name(), fit.max_context)
                } else {
                    format!("{} (does not fit{estimated})", m.name())
                }
            }
            None => m.name().to_string(),
        })
        .collect();
    let default = aliases::default_alias(ctx.device).name;
    let health = server::health(port);
    let detail = match &health {
        Health::Ready(_) => server::ServerInfo::fetch(port)
            .ok()
            .map(|info| Detail::Context(info.context)),
        Health::Loading(_) => server::loading_seconds(port).map(Detail::Elapsed),
        _ => None,
    };
    print!("{}", render(&health, detail, port, &pulled, default));
    0
}

/// What the server line adds: the context while serving, the time spent while loading.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Detail {
    Context(u32),
    Elapsed(u64),
}

pub fn render(
    health: &Health,
    detail: Option<Detail>,
    port: u16,
    pulled: &[String],
    default: &str,
) -> String {
    let mut out = String::new();
    let server = match health {
        Health::Ready(model) => match detail {
            Some(Detail::Context(tokens)) if tokens > 0 => {
                format!("serving {model} on 127.0.0.1:{port} ({tokens} tokens of context)")
            }
            _ => format!("serving {model} on 127.0.0.1:{port}"),
        },
        Health::Loading(model) => {
            let model = model.as_deref().unwrap_or("a model");
            match detail {
                Some(Detail::Elapsed(seconds)) => {
                    format!("loading {model} on 127.0.0.1:{port} (for {seconds} s)")
                }
                _ => format!("loading {model} on 127.0.0.1:{port}"),
            }
        }
        Health::Failed(error) => format!(
            "failed to load on 127.0.0.1:{port}: {}",
            error.lines().next().unwrap_or_default()
        ),
        Health::Down => format!("not running (port {port})"),
    };
    let _ = writeln!(out, "server  {server}");
    let models = if pulled.is_empty() {
        "none pulled".to_string()
    } else {
        pulled.join(", ")
    };
    let _ = writeln!(out, "models  {models}");
    let pull = format!("mrun pull {default}");
    let next: [(&str, &str); 3] = match health {
        Health::Ready(_) => [
            (
                "mrun launch claude-code",
                "start Claude Code on it (or codex, opencode)",
            ),
            ("mrun ps", "requests, cache and memory"),
            ("mrun stop", "stop the server"),
        ],
        Health::Loading(_) => [
            ("mrun logs -f", "follow the load"),
            (
                "mrun launch claude-code",
                "waits for it, then starts Claude Code",
            ),
            ("mrun ps", "the server's state"),
        ],
        Health::Failed(_) => [
            ("mrun logs", "what went wrong"),
            ("mrun doctor", "check this Mac and the models"),
            ("mrun ls", "the models here and whether they fit"),
        ],
        Health::Down if pulled.is_empty() => [
            (
                "mrun launch claude-code",
                "pull, serve and start Claude Code in one step",
            ),
            (pull.as_str(), "only download this Mac's default model"),
            ("mrun doctor", "check this Mac"),
        ],
        Health::Down => [
            (
                "mrun launch claude-code",
                "serve and start Claude Code (or codex, opencode)",
            ),
            ("mrun serve --detach", "start the server in the background"),
            ("mrun ls", "the models here and whether they fit"),
        ],
    };
    out.push_str("\nnext:\n");
    let width = next
        .iter()
        .map(|(command, _)| command.len())
        .max()
        .unwrap_or(0);
    for (command, what) in next {
        let _ = writeln!(out, "  {command:<width$}  {what}");
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_fresh_mac_is_told_how_to_start() {
        let text = render(&Health::Down, None, 8484, &[], "qwen3");
        assert!(text.starts_with("server  not running (port 8484)\nmodels  none pulled\n"));
        assert!(text.contains("  mrun pull qwen3 "), "{text}");
        assert_eq!(
            text.lines()
                .filter(|line| line.starts_with("  mrun"))
                .count(),
            3
        );
    }

    #[test]
    fn a_running_server_is_shown_with_its_model() {
        let text = render(
            &Health::Ready("qwen3".into()),
            Some(Detail::Context(105_000)),
            8484,
            &["qwen3 (fits: 105000 tokens)".into()],
            "qwen3",
        );
        assert!(
            text.contains("server  serving qwen3 on 127.0.0.1:8484 (105000 tokens of context)")
        );
        assert!(text.contains("models  qwen3 (fits: 105000 tokens)"));
        assert!(text.contains("models  qwen3"));
        assert!(text.contains("mrun stop"));
    }

    #[test]
    fn a_loading_server_shows_how_long_it_has_been_loading() {
        let text = render(
            &Health::Loading(Some("qwen3.6".into())),
            Some(Detail::Elapsed(42)),
            8484,
            &[],
            "qwen3.6",
        );
        assert!(text.contains("server  loading qwen3.6 on 127.0.0.1:8484 (for 42 s)"));
        assert!(text.contains("mrun logs -f"));
    }
}
