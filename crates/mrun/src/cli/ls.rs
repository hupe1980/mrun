//! `mrun ls [--all] [--json] [--port <p>]`: pulled models with family, size, fit and maximum context on this
//! machine, and their status from fast checks (size, modification time); curated aliases not
//! pulled yet follow with their fit, so the choice is visible before a download. Never touches
//! the network: an alias's size is its pinned one. `--json` rows are the 018 model entries.

use crate::hub::download::{self, FitClass};
use crate::hub::{self, Ctx, Fit, aliases};
use mrun_engine::Binding;
use mrun_engine::budget::{self, DeviceMemory, GIB};
use serde::Serialize;
use std::io::{Read, Write};
use std::time::Duration;

#[derive(Debug, Clone, clap::Args)]
pub struct LsArgs {
    /// Print JSON instead of a table.
    #[arg(long)]
    pub json: bool,
    /// Include the curated aliases that are not pulled (`ls` lists them by default too; the
    /// flag states it for scripts and the UI).
    #[arg(long)]
    pub all: bool,
    /// Fit as with this lower memory budget (GiB), as 'serve --budget-gib' would load.
    #[arg(long, env = "MRUN_BUDGET_GIB", value_parser = crate::cli::model::parse_budget_gib)]
    pub budget_gib: Option<u64>,
    /// Show the memory each model needs for this many tokens of context (`64K`), as
    /// 'serve --context' would load it.
    #[arg(long, value_parser = crate::cli::model::parse_tokens, conflicts_with = "budget_gib")]
    pub context: Option<u32>,
    /// Port of the server whose model is marked as serving.
    #[arg(long, env = "MRUN_PORT", default_value_t = crate::cli::server::default_port())]
    pub port: u16,
}

/// One model: pulled, or a curated alias not pulled yet.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct Row {
    /// What users type: the alias, else the repo id.
    pub name: String,
    pub alias: Option<String>,
    pub repo: String,
    pub revision: String,
    pub family: String,
    /// Bytes on disk.
    pub size: u64,
    /// `None` when this machine's memory is unknown.
    pub fits: Option<bool>,
    pub max_context: Option<u32>,
    /// Fit and context rest on an estimated working set (no Metal device answered).
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub estimated: bool,
    /// `ready`, `corrupt` or `not pulled`.
    pub status: &'static str,
    pub pulled: bool,
    /// `fits`, `tight` (less than `DEFAULT_ALIAS_MIN_TOKENS` of context) or `wont_fit`; `None`
    /// when memory is unknown.
    pub fit: Option<FitClass>,
    /// The fit as one sentence ([`download::verdict`], as `pull` says it).
    pub verdict: String,
    /// The per-machine default ([`hub::default_model`], D048).
    pub recommended: bool,
    /// The model is a preview alias, below the client gate (D061).
    pub preview: bool,
    /// The running server serves this model (set by `run`; `false` when none answers).
    pub serving: bool,
    /// The budget behind fit and context is bound by the memory available now: other
    /// processes hold the rest (D062).
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub limited_by_available: bool,
    /// A part (026): only layers `[first, end)` (0-based), for pool stages.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub part: Option<[usize; 2]>,
    /// What a context costs (027): `fixed + tokens × per_token` bytes; `None` when memory is
    /// unknown.
    pub memory: Option<hub::Cost>,
    /// The budget those bytes count against.
    pub budget: Option<u64>,
}

pub fn rows(ctx: &Ctx) -> Vec<Row> {
    let estimated = ctx.device.is_some_and(|device| device.estimated);
    let limited_by_available = ctx
        .device
        .and_then(|device| budget::plan(device, budget::Overrides::default()).ok())
        .is_some_and(|plan| plan.binding == Binding::Available);
    let recommended = hub::default_model(ctx).ok();
    let row = |alias: Option<String>, repo: String, fit: Option<Fit>| {
        let name = alias.clone().unwrap_or_else(|| repo.clone());
        Row {
            recommended: recommended.as_deref() == Some(name.as_str()),
            name,
            alias,
            repo,
            revision: String::new(),
            family: String::new(),
            size: 0,
            fits: fit.map(|fit| fit.fits),
            max_context: fit.map(|fit| fit.max_context),
            estimated,
            status: "not pulled",
            pulled: false,
            fit: fit.map(|fit| FitClass::of(&fit)),
            verdict: download::verdict(fit.map(|fit| (fit, estimated))),
            preview: false,
            serving: false,
            limited_by_available,
            part: None,
            memory: fit.map(|fit| fit.memory),
            budget: fit.map(|fit| fit.budget),
        }
    };
    let pulled = hub::manifests(&ctx.paths);
    let available = aliases::ALIASES
        .iter()
        .filter(|alias| !pulled.iter().any(|m| m.alias_name() == Some(alias.name)))
        .map(|alias| {
            let fit = ctx.device.map(|device| alias.fit(device));
            Row {
                revision: alias.revision.to_string(),
                family: alias.family.to_string(),
                size: alias.size,
                preview: alias.preview.is_some(),
                ..row(Some(alias.name.to_string()), alias.repo.to_string(), fit)
            }
        })
        .collect::<Vec<_>>();
    pulled
        .into_iter()
        .map(|mut manifest| {
            let fit = ctx.device.map(|device| manifest.fit(device));
            let status = hub::check(&ctx.paths, &mut manifest);
            let preview = aliases::named(&manifest.repo).is_some_and(|a| a.preview.is_some());
            let alias = manifest.alias_name().map(str::to_string);
            Row {
                preview,
                part: manifest.part,
                size: manifest.size(),
                status: status.label(),
                pulled: true,
                revision: manifest.revision,
                family: manifest.family,
                ..row(alias, manifest.repo, fit)
            }
        })
        .chain(available)
        .collect()
}

/// Marks the row of `served` (the server's model id: an alias or a repo id).
pub fn mark_serving(rows: &mut [Row], served: &str) {
    for row in rows {
        row.serving = [
            Some(row.name.as_str()),
            row.alias.as_deref(),
            Some(&row.repo),
        ]
        .into_iter()
        .flatten()
        .any(|name| name.eq_ignore_ascii_case(served));
    }
}

/// Longest wait for the server's answer: `ls` stays fast without one.
const SERVER_TIMEOUT: Duration = Duration::from_millis(300);

/// The model the server on `port` serves (`/v1/models`), or `None` within
/// [`SERVER_TIMEOUT`]. HTTP/1.0, so the answer is not chunked.
fn served_model(port: u16) -> Option<String> {
    let addr = std::net::SocketAddr::from(([127, 0, 0, 1], port));
    let mut stream = std::net::TcpStream::connect_timeout(&addr, SERVER_TIMEOUT).ok()?;
    stream.set_read_timeout(Some(SERVER_TIMEOUT)).ok()?;
    stream.set_write_timeout(Some(SERVER_TIMEOUT)).ok()?;
    stream
        .write_all(b"GET /v1/models HTTP/1.0\r\nHost: 127.0.0.1\r\n\r\n")
        .ok()?;
    let mut raw = Vec::new();
    stream.take(1 << 20).read_to_end(&mut raw).ok()?;
    let split = raw.windows(4).position(|w| w == b"\r\n\r\n")?;
    let head = std::str::from_utf8(&raw[..split]).ok()?;
    if head.split_whitespace().nth(1) != Some("200") {
        return None;
    }
    let body: serde_json::Value = serde_json::from_slice(&raw[split + 4..]).ok()?;
    body["data"][0]["id"].as_str().map(str::to_string)
}

/// `65536` as `64K`; other counts as they are.
fn tokens_word(tokens: u32) -> String {
    if tokens.is_multiple_of(1024) {
        format!("{}K", tokens / 1024)
    } else {
        tokens.to_string()
    }
}

/// The memory a row needs for `context` tokens, or that it does not fit (027).
fn at_context(row: &Row, context: u32) -> String {
    match (row.memory, row.max_context) {
        (Some(memory), Some(max)) if context <= max => hub::human_bytes(memory.at(context)),
        (Some(_), Some(_)) => "won't fit".into(),
        _ => "?".into(),
    }
}

/// The table, with a hint when nothing is pulled; with `context`, the memory each model needs for
/// it instead of the maximum context.
pub fn render(rows: &[Row], default: &str, context: Option<u32>) -> String {
    let at = context.map(|tokens| format!("AT {}", tokens_word(tokens)));
    let header = [
        "ALIAS",
        "REPO",
        "FAMILY",
        "SIZE",
        "FITS",
        at.as_deref().unwrap_or("MAX CTX"),
        "STATUS",
    ];
    let cells: Vec<[String; 7]> = rows
        .iter()
        .map(|row| {
            [
                row.alias.clone().unwrap_or_else(|| "-".into()),
                row.repo.clone(),
                row.family.clone(),
                hub::human_bytes(row.size),
                row.fit.map_or("?", FitClass::word).to_string(),
                match context {
                    Some(tokens) => at_context(row, tokens),
                    None => row
                        .max_context
                        .filter(|tokens| *tokens > 0)
                        .map_or_else(|| "-".into(), |tokens| tokens.to_string()),
                },
                match (row.part, row.preview) {
                    (Some([first, end]), _) => {
                        format!("{}, layers {}–{end} · for pools", row.status, first + 1)
                    }
                    (None, true) => format!("{}, preview", row.status),
                    (None, false) => row.status.to_string(),
                },
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
    let mut notes: Vec<String> = rows
        .iter()
        .filter(|row| row.preview)
        .filter_map(|row| aliases::preview_note(&row.repo))
        .collect();
    notes.dedup();
    if !notes.is_empty() {
        out.push('\n');
        for note in notes {
            out.push_str(&note);
            out.push('\n');
        }
    }
    if rows.iter().any(|row| row.estimated) {
        out.push_str("\nfit is estimated: no Metal device reported its working set\n");
    }
    if rows.iter().all(|row| row.status == "not pulled") {
        out.push_str(&format!("\nno models yet; run 'mrun pull {default}'\n"));
    }
    out
}

/// D062: fit counts what other processes hold now; the line says how much is available and
/// whether that binds. `None` when available memory is not read (not macOS).
pub fn memory_note(device: Option<DeviceMemory>) -> Option<String> {
    let device = device?;
    let available = device.available? as f64 / GIB as f64;
    let plan = budget::plan(device, budget::Overrides::default()).ok();
    Some(match plan {
        Some(plan) if plan.binding != Binding::Available => format!(
            "fit uses the memory available now ({available:.1} GiB, other apps included); the working set binds"
        ),
        _ => format!(
            "fit uses the memory available now ({available:.1} GiB): other apps hold the rest and lower it; 'mrun doctor' lists them"
        ),
    })
}

pub fn run(args: LsArgs) -> u8 {
    let ctx = Ctx::from_env().with_budget(args.budget_gib);
    let mut rows = rows(&ctx);
    if let Some(served) = served_model(args.port) {
        mark_serving(&mut rows, &served);
    }
    if args.json {
        println!(
            "{}",
            serde_json::to_string_pretty(&rows).unwrap_or_default()
        );
    } else {
        print!(
            "{}",
            render(&rows, aliases::default_alias(ctx.device).name, args.context)
        );
        if let Some(note) = memory_note(ctx.device) {
            println!("\n{note}");
        }
    }
    0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hub::test_support;
    use std::fs;

    #[derive(clap::Parser)]
    struct Wrap {
        #[command(flatten)]
        ls: LsArgs,
    }

    #[test]
    fn a_context_shows_what_each_model_needs() {
        use clap::Parser;
        assert!(Wrap::try_parse_from(["mrun", "--context", "64K", "--budget-gib", "10"]).is_err());
        let root = tempfile::tempdir().unwrap();
        let mut ctx = test_support::ctx(root.path(), "http://unused");
        ctx.device = Some(DeviceMemory {
            physical: 48 * GIB,
            recommended_working_set: 37 * GIB,
            available: None,
            estimated: false,
        });
        let rows = rows(&ctx);
        let row = rows.iter().find(|r| r.name == "qwen3.6").unwrap();
        let (memory, max, budget) = (
            row.memory.unwrap(),
            row.max_context.unwrap(),
            row.budget.unwrap(),
        );
        // The cost model is the fit's: the maximum context fits, one more step would not.
        assert!(memory.at(max) <= budget);
        let text = render(&rows, "qwen3.6", Some(65_536));
        assert!(text.contains("AT 64K"), "{text}");
        assert!(
            text.contains(&hub::human_bytes(memory.at(65_536))),
            "{text}"
        );
        assert!(render(&rows, "qwen3.6", Some(max + 1)).contains("won't fit"));
        let json = serde_json::to_value(row).unwrap();
        assert_eq!(json["memory"]["per_token"], memory.per_token);
        assert_eq!(json["budget"], budget);
    }

    #[test]
    fn the_port_names_the_server_to_ask() {
        use clap::Parser;
        let args = Wrap::try_parse_from(["mrun", "--json", "--all", "--port", "9123"]).unwrap();
        assert_eq!(args.ls.port, 9123);
        assert!(args.ls.json && args.ls.all);
        let default = Wrap::try_parse_from(["mrun"]).unwrap();
        assert_eq!(default.ls.port, crate::cli::server::default_port());
        assert!(Wrap::try_parse_from(["mrun", "--port", "http"]).is_err());
    }

    /// Columns of a rendered line, whitespace collapsed.
    fn columns(line: &str) -> String {
        line.split_whitespace().collect::<Vec<_>>().join(" ")
    }

    #[test]
    fn fit_says_it_counts_other_apps() {
        let device = |available| DeviceMemory {
            physical: 48 * GIB,
            recommended_working_set: 37 * GIB,
            available,
            estimated: false,
        };
        assert_eq!(memory_note(None), None);
        assert_eq!(memory_note(Some(device(None))), None);
        assert!(
            memory_note(Some(device(Some(44 * GIB))))
                .unwrap()
                .ends_with("the working set binds")
        );
        let short = memory_note(Some(device(Some(20 * GIB)))).unwrap();
        assert!(
            short.contains("20.0 GiB): other apps hold the rest"),
            "{short}"
        );
        let root = tempfile::tempdir().unwrap();
        let mut ctx = test_support::ctx(root.path(), "http://unused");
        ctx.device = Some(device(Some(20 * GIB)));
        let rows = rows(&ctx);
        assert!(rows.iter().all(|row| row.limited_by_available));
        assert_eq!(
            serde_json::to_value(&rows).unwrap()[0]["limited_by_available"],
            true
        );
    }

    #[test]
    fn lists_fit_marks_corrupt_models_and_offers_aliases() {
        let root = tempfile::tempdir().unwrap();
        let ctx = test_support::ctx(root.path(), "http://unused");
        let empty = render(&rows(&ctx), "qwen3.6", None);
        assert!(
            empty.ends_with("\nno models yet; run 'mrun pull qwen3.6'\n"),
            "{empty}"
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
        fs::write(broken.dir(&ctx.paths).join("model.safetensors"), b"ww").unwrap();

        let rows = rows(&ctx);
        let text = render(&rows, "qwen3.6", None);
        let lines: Vec<String> = text.lines().map(columns).collect();
        assert_eq!(
            lines,
            [
                "ALIAS REPO FAMILY SIZE FITS MAX CTX STATUS",
                "- org/broken qwen3 3 B tight 40960 corrupt",
                "small org/small qwen3 3 B tight 40960 ready",
                "qwen3.5 mlx-community/Qwen3.5-9B-4bit qwen3_5 6.0 GB yes 262144 not pulled, preview",
                "qwen3.6 mlx-community/Qwen3.6-35B-A3B-4bit qwen3_5_moe 20.4 GB yes 262144 not pulled",
                "",
                "qwen3.5 is a preview: below the client gate (Claude Code 27/30, Codex 26/30, OpenCode 28/30); larger Macs get qwen3.6",
            ]
        );
        let json = serde_json::to_value(&rows).unwrap();
        assert_eq!(json[1]["alias"], "small");
        assert_eq!(json[1]["status"], "ready");
        assert_eq!(json[1]["max_context"], 40960);
        assert_eq!(json[1]["preview"], false, "not an alias");
        assert_eq!(json[2]["preview"], true);
        assert_eq!(json[3]["preview"], false);
        // 018 model entries.
        assert_eq!(json[1]["name"], "small");
        assert_eq!(json[0]["name"], "org/broken");
        assert_eq!(json[1]["pulled"], true);
        assert_eq!(json[1]["fit"], "tight");
        assert_eq!(
            json[1]["verdict"],
            "Fits with 40,960 tokens, less than the 65,536 coding agents need."
        );
        assert_eq!(json[3]["name"], "qwen3.6");
        assert_eq!(json[3]["pulled"], false);
        assert_eq!(json[3]["size"], aliases::get("qwen3.6").unwrap().size);
        assert_eq!(json[3]["fit"], "fits");
        assert_eq!(json[3]["verdict"], "Fits with 262,144 tokens.");
        // `default_model`: a pulled model that fits before an alias that needs a pull.
        let default = hub::default_model(&ctx).unwrap();
        let recommended: Vec<_> = rows.iter().filter(|row| row.recommended).collect();
        assert_eq!(recommended.len(), 1);
        assert_eq!(recommended[0].name, default);
        assert!(recommended[0].pulled);
        assert_eq!(json[3]["serving"], false);

        assert!(json[1].get("estimated").is_none(), "measured: no marker");
        assert!(!text.contains("estimated"), "{text}");

        let mut guessed = ctx.clone();
        guessed.device = guessed
            .device
            .map(|device| mrun_engine::budget::DeviceMemory {
                estimated: true,
                ..device
            });
        let rows = super::rows(&guessed);
        assert!(rows.iter().all(|row| row.estimated));
        assert_eq!(serde_json::to_value(&rows).unwrap()[0]["estimated"], true);
        assert!(render(&rows, "qwen3.6", None).contains("\nfit is estimated: no Metal device"));

        let unknown = Ctx {
            device: None,
            ..ctx
        };
        let rows = super::rows(&unknown);
        assert_eq!(rows[0].fits, None);
        assert_eq!(rows[0].fit, None);
        assert_eq!(rows[0].verdict, "Fit unknown.");
    }

    /// SC-005: on simulated 16/24/32/48 GB Macs the rows' fit, verdict and recommendation are
    /// the CLI's own (`pull`'s fit phrase, `default_model`), one model is recommended and it
    /// leaves the agents' context.
    #[test]
    fn fit_and_recommendation_follow_the_budget() {
        let root = tempfile::tempdir().unwrap();
        let base = test_support::ctx(root.path(), "http://unused");
        let mut seen = Vec::new();
        for gb in [16, 24, 32, 48] {
            let device = DeviceMemory {
                physical: gb * GIB,
                recommended_working_set: gb * GIB / 100 * 78,
                available: None,
                estimated: false,
            };
            let ctx = Ctx {
                device: Some(device),
                ..base.clone()
            };
            let rows = rows(&ctx);
            for row in &rows {
                let alias = aliases::get(&row.name).unwrap();
                let fit = alias.fit(device);
                assert_eq!(row.fit, Some(FitClass::of(&fit)), "{gb} GB {}", row.name);
                assert_eq!(row.verdict, download::verdict(Some((fit, false))));
                assert!(!row.pulled && row.size == alias.size);
            }
            let recommended: Vec<_> = rows.iter().filter(|row| row.recommended).collect();
            assert_eq!(recommended.len(), 1, "{gb} GB");
            assert_eq!(
                recommended[0].name,
                hub::default_model(&ctx).unwrap(),
                "{gb} GB"
            );
            assert_eq!(recommended[0].fit, Some(FitClass::Fits), "{gb} GB");
            seen.push((
                gb,
                recommended[0].name.clone(),
                rows.iter()
                    .map(|row| row.fit.unwrap().word())
                    .collect::<Vec<_>>(),
            ));
        }
        assert_eq!(
            seen,
            [
                (16, "qwen3.5".to_string(), vec!["yes", "no"]),
                (24, "qwen3.5".to_string(), vec!["yes", "no"]),
                (32, "qwen3.5".to_string(), vec!["yes", "tight"]),
                (48, "qwen3.6".to_string(), vec!["yes", "yes"]),
            ]
        );
        // `--budget-gib` lowers a 48 GB Mac to the 16 GB picture.
        let lowered = Ctx {
            device: Some(hub::TARGET_48GB),
            ..base
        }
        .with_budget(Some(12 * GIB));
        let rows = rows(&lowered);
        assert!(
            rows.iter()
                .any(|row| row.recommended && row.name == "qwen3.5")
        );
    }

    #[test]
    fn the_served_model_is_marked() {
        let root = tempfile::tempdir().unwrap();
        let ctx = test_support::ctx(root.path(), "http://unused");
        let mut rows = rows(&ctx);
        mark_serving(&mut rows, "mlx-community/Qwen3.6-35B-A3B-4bit");
        let serving: Vec<_> = rows
            .iter()
            .filter(|row| row.serving)
            .map(|row| &row.name)
            .collect();
        assert_eq!(serving, ["qwen3.6"]);
        // Nothing listens on a port just freed: no server, quickly.
        let port = std::net::TcpListener::bind("127.0.0.1:0")
            .unwrap()
            .local_addr()
            .unwrap()
            .port();
        let start = std::time::Instant::now();
        assert_eq!(served_model(port), None);
        assert!(start.elapsed() < Duration::from_secs(1));
    }

    #[test]
    fn the_served_model_is_read_from_v1_models() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let server = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut request = [0u8; 512];
            let _ = stream.read(&mut request).unwrap();
            let body = r#"{"data":[{"id":"qwen3.5"}]}"#;
            let reply = format!(
                "HTTP/1.0 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\n\r\n{body}",
                body.len()
            );
            stream.write_all(reply.as_bytes()).unwrap();
        });
        assert_eq!(served_model(port).as_deref(), Some("qwen3.5"));
        server.join().unwrap();
    }
}
