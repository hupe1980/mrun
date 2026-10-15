//! `mrun show <model>`: manifest, family, architecture, quantization, license, identity and the
//! fit breakdown on this machine.

use crate::hub::{self, Ctx, Manifest, human_bytes, verify};
use std::fmt::Write;

#[derive(Debug, Clone, clap::Args)]
pub struct ShowArgs {
    /// Alias or repo id (org/name[@revision]) of a pulled model.
    pub model: String,
}

pub fn render(ctx: &Ctx, manifest: &Manifest) -> String {
    let dir = manifest.snapshot_dir(&ctx.paths);
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
    line("status", verify::fast_check(manifest, &dir).label().into());
    line(
        "state",
        format!(
            "{} bytes per token, {} position-bound bytes, max position {}",
            manifest.state_bytes_per_token, manifest.checkpoint_bytes, manifest.max_position
        ),
    );
    match ctx.device {
        Some(device) => {
            let fit = hub::fit(device, manifest.weights(), &manifest.model_info());
            line(
                "fit",
                format!(
                    "budget {}: weights {} + scratch ~{} -> {}",
                    human_bytes(fit.budget),
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
    out
}

pub fn run(args: ShowArgs) -> u8 {
    let ctx = Ctx::from_env();
    super::exit(
        hub::find_manifest(&ctx.paths, &args.model)
            .map(|manifest| print!("{}", render(&ctx, &manifest))),
    )
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
            &format!("identity:     {}\n", "0".repeat(64)),
        ] {
            assert!(text.contains(expected), "{expected:?} in\n{text}");
        }
    }
}
