//! `mrun show <model>`: manifest, family, architecture, quantization, license, identity and the
//! fit breakdown on this machine.

use crate::hub::{self, Ctx, Manifest, aliases, human_bytes};
use std::fmt::Write;

#[derive(Debug, Clone, clap::Args)]
pub struct ShowArgs {
    /// Alias or repo id (org/name[@revision]) of a pulled model.
    pub model: String,
    /// Print JSON: the manifest, status, memory and fit.
    #[arg(long)]
    pub json: bool,
}

/// `show --json`: the manifest as recorded plus what `show` derives on this machine.
pub fn render_json(ctx: &Ctx, manifest: &Manifest) -> serde_json::Value {
    let status = hub::check(&ctx.paths, &mut manifest.clone());
    let memory = ctx.device.map(|device| {
        serde_json::json!({
            "physical": device.physical,
            "recommended_working_set": device.recommended_working_set,
            "available": device.available,
            "estimated": device.estimated,
            "source": hub::memory_source(device),
        })
    });
    let fit = ctx.device.map(|device| {
        let fit = manifest.fit(device);
        serde_json::json!({
            "budget": fit.budget,
            "weights": fit.weights,
            "scratch": fit.scratch,
            "fits": fit.fits,
            "max_context": fit.max_context,
        })
    });
    serde_json::json!({
        "schema_version": 1,
        "name": manifest.name(),
        "path": manifest.dir(&ctx.paths),
        "status": status.label(),
        "size": manifest.size(),
        "manifest": manifest,
        "memory": memory,
        "fit": fit,
        "preview": aliases::preview_note(&manifest.repo),
    })
}

pub fn render(ctx: &Ctx, manifest: &Manifest) -> String {
    let dir = manifest.dir(&ctx.paths);
    let mut out = String::new();
    let mut line = |key: &str, value: String| {
        let _ = writeln!(out, "{:<14}{value}", format!("{key}:"));
    };
    line("model", manifest.name().to_string());
    line("repo", manifest.repo.clone());
    line("revision", manifest.revision.clone());
    line("family", manifest.family.clone());
    line(
        "architecture",
        match &manifest.text_architecture {
            Some(text) => format!("{} (text: {text})", manifest.architecture),
            None => manifest.architecture.clone(),
        },
    );
    line(
        "quantization",
        manifest.quantization.as_ref().map_or("none".into(), |q| {
            let mut parts = Vec::new();
            if let Some(bits) = q.bits {
                parts.push(format!("{bits}-bit"));
            }
            if let Some(group) = q.group_size {
                parts.push(format!("group {group}"));
            }
            parts.push(q.mode.clone().unwrap_or_else(|| "affine".into()));
            parts.join(", ")
        }),
    );
    line("syntax", manifest.syntax.clone());
    line("license", manifest.license.clone());
    line("identity", manifest.identity.clone());
    line(
        "size",
        format!(
            "{} in {} files",
            human_bytes(manifest.size()),
            manifest.files.len()
        ),
    );
    line("path", dir.display().to_string());
    let status = hub::check(&ctx.paths, &mut manifest.clone());
    line("status", status.label().into());
    line(
        "state",
        format!(
            "{} bytes per token, {} position-bound bytes, max position {}",
            manifest.state_bytes_per_token, manifest.checkpoint_bytes, manifest.max_position
        ),
    );
    match ctx.device {
        Some(device) => {
            line("memory", hub::describe_memory(device));
            let fit = manifest.fit(device);
            line(
                "fit",
                format!(
                    "budget {}{}: weights {} + scratch ~{} -> {}",
                    human_bytes(fit.budget),
                    if device.estimated { " (estimated)" } else { "" },
                    human_bytes(fit.weights),
                    human_bytes(fit.scratch),
                    if fit.fits {
                        format!("fits, up to {} tokens of context", fit.max_context)
                    } else {
                        "does not fit".into()
                    }
                ),
            );
        }
        None => line("fit", "unknown (cannot read this machine's memory)".into()),
    }
    line(
        "on 48 GB",
        format!("{} tokens of context", manifest.capacity_48gb),
    );
    if let Some(note) = aliases::preview_note(&manifest.repo) {
        line("preview", note);
    }
    out
}

pub fn run(args: ShowArgs) -> u8 {
    let ctx = Ctx::from_env();
    super::exit(hub::find_manifest(&ctx.paths, &args.model).map(|manifest| {
        if args.json {
            println!("{:#}", render_json(&ctx, &manifest));
        } else {
            print!("{}", render(&ctx, &manifest));
        }
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hub::test_support;

    #[test]
    fn shows_family_identity_and_fit() {
        let root = tempfile::tempdir().unwrap();
        let ctx = test_support::ctx(root.path(), "http://unused");
        let manifest = test_support::cached_model(
            &ctx.paths,
            Some("small"),
            "org/small",
            &"a".repeat(40),
            &[("config.json", b"{}"), ("model.safetensors", b"w")],
        );
        let text = render(&ctx, &manifest);
        for expected in [
            "model:        small\n",
            "family:       qwen3\n",
            "quantization: none\n",
            "status:       ready\n",
            "fits, up to 40960 tokens of context",
            "memory:       51.5 GB physical, 40.2 GB working set (Metal recommended working set)\n",
            &format!("identity:     {}\n", "0".repeat(64)),
        ] {
            assert!(text.contains(expected), "{expected:?} in\n{text}");
        }
        let json = render_json(&ctx, &manifest);
        assert_eq!(json["name"], "small");
        assert_eq!(json["status"], "ready");
        assert_eq!(json["manifest"]["repo"], "org/small");
        assert_eq!(json["manifest"]["state_bytes_per_token"], 147_456);
        assert_eq!(json["fit"]["max_context"], 40_960);
        assert_eq!(json["memory"]["estimated"], false);
        assert!(json["preview"].is_null());
        assert!(!text.contains("preview"), "{text}");
    }

    #[test]
    fn a_preview_alias_says_so() {
        let root = tempfile::tempdir().unwrap();
        let ctx = test_support::ctx(root.path(), "http://unused");
        let manifest = test_support::cached_model(
            &ctx.paths,
            Some("qwen3.5"),
            "mlx-community/Qwen3.5-9B-4bit",
            &"a".repeat(40),
            &[("config.json", b"{}"), ("model.safetensors", b"w")],
        );
        let note = "qwen3.5 is a preview: below the client gate (Claude Code 27/30, Codex 26/30, \
                    OpenCode 28/30); larger Macs get qwen3.6";
        let text = render(&ctx, &manifest);
        assert!(text.ends_with(&format!("preview:      {note}\n")), "{text}");
        assert_eq!(render_json(&ctx, &manifest)["preview"], note);
    }
}
