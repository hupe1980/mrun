//! D062: one server at a time, and only into memory that is there. Before a system is started
//! (the switch pause) no known inference server may still run and the available memory must
//! hold its model plus the margin; once it runs, it must be the only one. Pure checks over what
//! `ps` and `vm_stat` report; only server kinds and process names are printed, never arguments.

use crate::bench::systems::System;
use mrun_engine::budget::{self, GIB};
use mrun_engine::sysmem::{self, Process};

/// The kind of inference server a command line runs, if it is one bench knows: mrun, Ollama,
/// llama.cpp, mlx-lm, oMLX, Rapid-MLX or Splash.
pub fn server_kind(command: &str) -> Option<&'static str> {
    let tokens: Vec<&str> = command.split_whitespace().collect();
    let base = |token: &str| token.rsplit('/').next().unwrap_or(token).to_string();
    let followed = |name: &str, next: &[&str]| {
        tokens
            .windows(2)
            .any(|pair| base(pair[0]) == name && next.contains(&pair[1]))
    };
    if followed("mrun", &["serve"]) {
        Some("mrun")
    } else if followed("ollama", &["serve", "runner"]) {
        Some("ollama")
    } else if tokens.iter().any(|t| base(t) == "llama-server") {
        Some("llama-server")
    } else if tokens
        .iter()
        .any(|t| t.contains("mlx_lm.server") || t.ends_with("mlx_lm/server.py"))
        || followed("mlx_lm", &["server"])
    {
        Some("mlx_lm")
    } else if followed("omlx", &["serve"]) {
        Some("omlx")
    } else if followed("rapid-mlx", &["serve"]) {
        Some("rapid-mlx")
    } else if followed("splash", &["serve"]) {
        Some("splash")
    } else {
        None
    }
}

/// The known server kinds in `ps -axo pid=,command=` text, sorted and deduplicated (Ollama's
/// server and runner count once), leaving out `own` (this process).
pub fn servers_in(ps: &str, own: u32) -> Vec<&'static str> {
    let mut kinds: Vec<&'static str> = ps
        .lines()
        .filter_map(|line| {
            let line = line.trim_start();
            let (pid, command) = line.split_once(char::is_whitespace)?;
            (pid.parse::<u32>().ok()? != own).then_some(())?;
            server_kind(command)
        })
        .collect();
    kinds.sort_unstable();
    kinds.dedup();
    kinds
}

/// The known servers running now; empty outside macOS or when `ps` fails.
pub fn running_servers() -> Vec<&'static str> {
    if !cfg!(target_os = "macos") {
        return Vec::new();
    }
    std::process::Command::new("ps")
        .args(["-axo", "pid=,command="])
        .output()
        .map(|out| servers_in(&String::from_utf8_lossy(&out.stdout), std::process::id()))
        .unwrap_or_default()
}

/// Before a start no known server may run; while a system runs, at most one (itself).
pub fn check_servers(running: &[&str], started: bool) -> Result<(), String> {
    let allowed = usize::from(started);
    if running.len() <= allowed {
        return Ok(());
    }
    Err(if started {
        format!(
            "{} servers run at once ({}); bench measures one at a time",
            running.len(),
            running.join(", ")
        )
    } else {
        format!(
            "{} still runs; stop it and wait until it has exited",
            running.join(", ")
        )
    })
}

/// Memory the system's model needs: the profile's `memory_gib`, else mrun's model scaled by the
/// profile's bits per weight (every bundled profile serves the same model at its own
/// quantization).
pub fn model_bytes(system: &System, mrun_weights: u64) -> u64 {
    let model = &system.profile.model;
    if let Some(gib) = model.memory_gib {
        return (gib * GIB as f64) as u64;
    }
    let bits = model.bits_per_weight.unwrap_or(4.5);
    (mrun_weights as f64 * bits / 4.5) as u64
}

/// The available memory must hold the model plus the margin; the refusal names the largest
/// other processes.
pub fn check_memory(
    name: &str,
    model: u64,
    available: Option<u64>,
    physical: u64,
    others: &[Process],
) -> Result<(), String> {
    let Some(available) = available else {
        return Ok(());
    };
    let margin = budget::margin(physical);
    if available >= model + margin {
        return Ok(());
    }
    let gib = |bytes: u64| bytes as f64 / GIB as f64;
    let largest = if others.is_empty() {
        String::new()
    } else {
        format!(
            "; largest memory users: {}",
            sysmem::describe(&others[..others.len().min(3)])
        )
    };
    Err(format!(
        "{name} needs about {:.1} GiB for its model plus a {:.1} GiB margin, but only {:.1} GiB are available{largest}; quit some and rerun",
        gib(model),
        gib(margin),
        gib(available)
    ))
}

/// Whether the system runs on this machine (bench only guards local servers).
pub fn is_local(system: &System) -> bool {
    let rest = system
        .url
        .split_once("://")
        .map_or(system.url.as_str(), |(_, rest)| rest);
    ["127.0.0.1", "localhost", "[::1]"]
        .iter()
        .any(|host| rest.starts_with(host))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_servers_are_recognized_by_their_command() {
        let cases = [
            ("/opt/homebrew/bin/mrun serve --model qwen3.6", Some("mrun")),
            ("mrun bench --compare", None),
            (
                "/Applications/Ollama.app/Contents/Resources/ollama serve",
                Some("ollama"),
            ),
            (
                "/Applications/Ollama.app/Contents/Resources/ollama runner --port 1",
                Some("ollama"),
            ),
            ("/Applications/Ollama.app/Contents/MacOS/Ollama", None),
            ("llama-server -m q.gguf --port 9931", Some("llama-server")),
            (
                "/usr/bin/python3 /venv/bin/mlx_lm.server --model m --port 8080",
                Some("mlx_lm"),
            ),
            ("python -m mlx_lm.server --model m", Some("mlx_lm")),
            ("mlx_lm server --model m", Some("mlx_lm")),
            ("/venv/bin/omlx serve --model-dir d", Some("omlx")),
            ("rapid-mlx serve m", Some("rapid-mlx")),
            ("splash serve --model m", Some("splash")),
            ("/usr/bin/rust-analyzer", None),
        ];
        for (command, kind) in cases {
            assert_eq!(server_kind(command), kind, "{command}");
        }
        let ps = "  1 /sbin/launchd
 10 /Applications/Ollama.app/Contents/Resources/ollama serve
 11 /Applications/Ollama.app/Contents/Resources/ollama runner --model x
 12 /opt/homebrew/bin/mrun serve
 13 /opt/homebrew/bin/mrun serve
";
        assert_eq!(servers_in(ps, 1), ["mrun", "ollama"]);
        assert_eq!(servers_in(ps, 12), ["mrun", "ollama"]);
        assert!(servers_in("", 1).is_empty());
    }

    #[test]
    fn one_server_at_a_time() {
        assert!(check_servers(&[], false).is_ok());
        assert!(check_servers(&["ollama"], true).is_ok());
        let still = check_servers(&["ollama"], false).unwrap_err();
        assert_eq!(
            still,
            "ollama still runs; stop it and wait until it has exited"
        );
        let two = check_servers(&["mlx_lm", "mrun"], true).unwrap_err();
        assert!(
            two.starts_with("2 servers run at once (mlx_lm, mrun)"),
            "{two}"
        );
    }

    #[test]
    fn the_model_must_fit_the_available_memory_with_the_margin() {
        let others = [
            Process {
                pid: 2,
                name: "com.apple.Virtualization.VirtualMachine".into(),
                resident: 34_681_651_200,
            },
            Process {
                pid: 3,
                name: "Python".into(),
                resident: 20_508_468_838,
            },
        ];
        let physical = 48 * GIB;
        assert!(check_memory("mlx-lm", 19 * GIB, None, physical, &others).is_ok());
        assert!(check_memory("mlx-lm", 19 * GIB, Some(25 * GIB), physical, &others).is_ok());
        let error =
            check_memory("mlx-lm", 19 * GIB, Some(14 * GIB), physical, &others).unwrap_err();
        assert_eq!(
            error,
            "mlx-lm needs about 19.0 GiB for its model plus a 6.0 GiB margin, but only 14.0 GiB \
             are available; largest memory users: com.apple.Virtualization.VirtualMachine \
             32.3 GiB, Python 19.1 GiB; quit some and rerun"
        );
    }

    #[test]
    fn model_size_follows_the_profile() {
        let mut llama = System::parse("llama.cpp=http://127.0.0.1:9931").unwrap();
        assert!(is_local(&llama));
        let scaled = model_bytes(&llama, 18 * GIB);
        assert!(scaled > 18 * GIB, "4.96 bits per weight > mrun's 4.5");
        llama.profile.model.memory_gib = Some(21.5);
        assert_eq!(model_bytes(&llama, 18 * GIB), 21 * GIB + GIB / 2);
        let remote = System::parse("far=http://10.0.0.2:8080").unwrap();
        assert!(!is_local(&remote));
    }
}
