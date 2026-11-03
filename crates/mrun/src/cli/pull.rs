//! `mrun pull <alias|org/name[@rev]> [--yes] [--json] [--dry-run]`: the only command that uses
//! the network (FR-013). `--json` prints the 018 pull progress lines on stdout instead of bars:
//! `plan`, `progress`, `done`, or `confirm` and `error`. `--dry-run` stops after the plan (020).

use crate::errors::{CliError, Code};
use crate::hub::download::{self, Plan};
use crate::hub::progress::{JsonLines, Report};
use crate::hub::{Ctx, Manifest};
use serde_json::json;

#[derive(Debug, Clone, clap::Args)]
pub struct PullArgs {
    /// Curated alias (qwen3.5, qwen3.6) or Hugging Face repo id (org/name[@revision]).
    pub model: String,
    /// Download even when the model does not fit this machine's memory budget.
    #[arg(long, short)]
    pub yes: bool,
    /// Print JSON lines (plan, progress, done, confirm, error) on stdout instead of progress
    /// bars; never asks: a confirmation needs --yes.
    #[arg(long)]
    pub json: bool,
    /// Print the plan (size, family, fit) and exit: no confirmation, no download.
    #[arg(long)]
    pub dry_run: bool,
}

pub fn run(args: PullArgs) -> u8 {
    let ctx = Ctx::from_env();
    let runtime = match tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
    {
        Ok(runtime) => runtime,
        Err(error) => return CliError::new(Code::Internal, error.to_string()).report(),
    };
    if args.dry_run {
        let result = if args.json {
            runtime.block_on(plan_json(&ctx, &args.model, &JsonLines::stdout()))
        } else {
            runtime.block_on(download::dry_run(&ctx, &args.model, &Report::Human))
        };
        return super::exit(result.map(drop));
    }
    if args.json {
        let result = runtime.block_on(pull_json(&ctx, &args.model, args.yes, &JsonLines::stdout()));
        return super::exit(result.map(drop));
    }
    let mut confirm = |question: &str| super::confirm(question, args.yes);
    super::exit(
        runtime
            .block_on(download::pull(&ctx, &args.model, &mut confirm))
            .map(drop),
    )
}

/// `pull --json`: the pull reporting JSON lines on `out`. A confirmation is a `confirm` line
/// and E002 unless `yes`; every failure ends with an `error` line (the error is also returned,
/// for stderr and the exit code).
pub async fn pull_json(
    ctx: &Ctx,
    model: &str,
    yes: bool,
    out: &JsonLines,
) -> Result<Manifest, CliError> {
    let mut asked = None;
    let mut confirm = |question: &str| {
        if yes {
            return Ok(true);
        }
        out.emit(&json!({"event": "confirm", "question": question}));
        asked = Some(question.to_string());
        Err(CliError::new(Code::NeedsConfirmation, "needs --yes"))
    };
    let result = download::pull_with(ctx, model, &mut confirm, &Report::Json(out.clone())).await;
    let result = result.map_err(|error| match asked {
        Some(question) => {
            CliError::new(Code::NeedsConfirmation, format!("{question} (needs --yes)"))
                .hint("pass --yes to download it anyway")
        }
        None => error,
    });
    if let Err(error) = &result {
        emit_error(out, error);
    }
    result
}

/// `pull --dry-run --json` (020 FR-005): one `plan` event, or an `error` event.
pub async fn plan_json(ctx: &Ctx, model: &str, out: &JsonLines) -> Result<Plan, CliError> {
    let result = download::dry_run(ctx, model, &Report::Json(out.clone())).await;
    if let Err(error) = &result {
        emit_error(out, error);
    }
    result
}

fn emit_error(out: &JsonLines, error: &CliError) {
    out.emit(&json!({
        "event": "error",
        "code": error.code.id(),
        "message": error.message,
        "hint": error.hint.as_deref().unwrap_or(error.code.next_step()),
    }));
}
