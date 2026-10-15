//! `mrun pull <alias|org/name[@rev]> [--yes]`: the only command that uses the network (FR-013).

use crate::errors::{CliError, Code};
use crate::hub::{Ctx, download};

#[derive(Debug, Clone, clap::Args)]
pub struct PullArgs {
    /// Curated alias (qwen3.6, qwen3-coder, gpt-oss) or Hugging Face repo id (org/name[@revision]).
    pub model: String,
    /// Download even when the model does not fit this machine's memory budget.
    #[arg(long, short)]
    pub yes: bool,
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
    let mut confirm = |question: &str| super::confirm(question, args.yes);
    super::exit(
        runtime
            .block_on(download::pull(&ctx, &args.model, &mut confirm))
            .map(drop),
    )
}
