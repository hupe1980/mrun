//! `mrun ls [--json]`: pulled models with family, size, fit and maximum context on this machine,
//! and their status from fast checks (size, modification time).

use crate::hub::{self, Ctx, aliases, verify};
use serde::Serialize;

#[derive(Debug, Clone, clap::Args)]
pub struct LsArgs {
    /// Print JSON instead of a table.
    #[arg(long)]
    pub json: bool,
}

/// One pulled model.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct Row {
    pub alias: Option<String>,
    pub repo: String,
    pub revision: String,
    pub family: String,
    /// Bytes on disk.
    pub size: u64,
    /// `None` when this machine's memory is unknown.
    pub fits: Option<bool>,
    pub max_context: Option<u32>,
    /// `ready` or `corrupt`.
    pub status: &'static str,
}

pub fn rows(ctx: &Ctx) -> Vec<Row> {
    hub::manifests(&ctx.paths)
        .into_iter()
        .map(|manifest| {
            let fit = ctx
                .device
                .map(|device| hub::fit(device, manifest.weights(), &manifest.model_info()));
            let status = verify::fast_check(&manifest, &manifest.snapshot_dir(&ctx.paths));
            Row {
                size: manifest.size(),
                fits: fit.map(|fit| fit.fits),
                max_context: fit.map(|fit| fit.max_context),
                status: status.label(),
                alias: manifest.alias,
                repo: manifest.repo,
                revision: manifest.revision,
                family: manifest.family,
            }
        })
        .collect()
}

/// The table; a hint when nothing is pulled.
pub fn render(rows: &[Row]) -> String {
    if rows.is_empty() {
        return format!("no models yet; run 'mrun pull {}'\n", aliases::DEFAULT);
    }
    let header = [
        "ALIAS", "REPO", "FAMILY", "SIZE", "FITS", "MAX CTX", "STATUS",
    ];
    let cells: Vec<[String; 7]> = rows
        .iter()
        .map(|row| {
            [
                row.alias.clone().unwrap_or_else(|| "-".into()),
                row.repo.clone(),
                row.family.clone(),
                hub::human_bytes(row.size),
                match row.fits {
                    Some(true) => "yes".into(),
                    Some(false) => "no".into(),
                    None => "?".into(),
                },
                row.max_context
                    .filter(|tokens| *tokens > 0)
                    .map_or_else(|| "-".into(), |tokens| tokens.to_string()),
                row.status.to_string(),
            ]
        })
        .collect();
    let mut widths = header.map(str::len);
    for line in &cells {
        for (width, cell) in widths.iter_mut().zip(line) {
            *width = (*width).max(cell.len());
        }
    }
    let mut out = String::new();
    let mut push = |line: &[String]| {
        let padded: Vec<String> = line
            .iter()
            .zip(widths)
            .map(|(cell, width)| format!("{cell:<width$}"))
            .collect();
        out.push_str(padded.join("  ").trim_end());
        out.push('\n');
    };
    push(&header.map(str::to_string));
    for line in &cells {
        push(line);
    }
    out
}

pub fn run(args: LsArgs) -> u8 {
    let rows = rows(&Ctx::from_env());
    if args.json {
        println!(
            "{}",
            serde_json::to_string_pretty(&rows).unwrap_or_default()
        );
    } else {
        print!("{}", render(&rows));
    }
    0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hub::test_support;
    use std::fs;

    #[test]
    fn lists_fit_and_marks_corrupt_models() {
        let root = tempfile::tempdir().unwrap();
        let ctx = test_support::ctx(root.path(), "http://unused");
        assert_eq!(
            render(&rows(&ctx)),
            "no models yet; run 'mrun pull qwen3.6'\n"
        );
        let files: &[(&str, &[u8])] = &[("config.json", b"{}"), ("model.safetensors", b"w")];
        test_support::cached_model(
            &ctx.paths,
            Some("small"),
            "org/small",
            &"a".repeat(40),
            files,
        );
        let broken =
            test_support::cached_model(&ctx.paths, None, "org/broken", &"b".repeat(40), files);
        fs::write(
            broken.snapshot_dir(&ctx.paths).join("model.safetensors"),
            b"ww",
        )
        .unwrap();

        let rows = rows(&ctx);
        assert_eq!(
            render(&rows),
            "ALIAS  REPO        FAMILY  SIZE  FITS  MAX CTX  STATUS\n\
             -      org/broken  qwen3   3 B   yes   40960    corrupt\n\
             small  org/small   qwen3   3 B   yes   40960    ready\n"
        );
        let json = serde_json::to_value(&rows).unwrap();
        assert_eq!(json[1]["alias"], "small");
        assert_eq!(json[1]["status"], "ready");
        assert_eq!(json[1]["max_context"], 40960);

        let unknown = Ctx {
            device: None,
            ..ctx
        };
        assert_eq!(super::rows(&unknown)[0].fits, None);
    }
}
