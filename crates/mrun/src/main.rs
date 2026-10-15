use clap::{Parser, Subcommand};
use mrun::chat::limits::MAX_PROMPT_BYTES;
use mrun::cli::serve::{self, ServeArgs};
use std::path::PathBuf;
use std::process::ExitCode;

/// Local inference server for coding agents on Apple Silicon.
#[derive(Debug, Parser)]
#[command(name = "mrun", version, about, propagate_version = true)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Chat with a local model in the terminal.
    #[command(
        after_help = "Examples:\n  mrun run mlx-community/Qwen3-8B-4bit\n  mrun run ./models/qwen3 --prompt \"Explain this error\" --max-tokens 512"
    )]
    Run(RunArgs),
    /// Serve the Anthropic Messages, OpenAI Chat Completions and Responses APIs on 127.0.0.1.
    #[command(
        after_help = "Examples:\n  mrun serve --model qwen3.6\n  MRUN_PORT=9000 mrun serve --model qwen3-coder"
    )]
    Serve(ServeArgs),
}

#[derive(Debug, clap::Args)]
struct RunArgs {
    /// Local model directory or Hugging Face cache repo id (org/name).
    model: String,
    /// One-shot prompt; without it an interactive session starts.
    #[arg(long)]
    prompt: Option<String>,
    /// System prompt.
    #[arg(long)]
    system: Option<String>,
    #[arg(long, default_value_t = 0.0)]
    temperature: f32,
    #[arg(long, default_value_t = 1.0)]
    top_p: f32,
    #[arg(long, default_value_t = 0)]
    top_k: u32,
    #[arg(long, default_value_t = 0.0)]
    min_p: f32,
    #[arg(long)]
    seed: Option<u64>,
    #[arg(long, default_value_t = 2048)]
    max_tokens: u32,
    /// Stop sequence (up to 4, each at most 256 bytes).
    #[arg(long)]
    stop: Vec<String>,
    /// Let models that support it think before answering.
    #[arg(long)]
    think: bool,
    /// Fraction of physical memory mrun may plan with.
    #[arg(long, default_value_t = 0.75)]
    memory_fraction: f64,
    /// Memory kept free for other applications and mrun's host heap, in GiB
    /// [default: physical memory / 6, clamped to 2..=8].
    #[arg(long)]
    reserve_gib: Option<u64>,
    /// Allow a budget above the automatic one.
    #[arg(long)]
    allow_over: bool,
    /// Print the memory budget breakdown after loading and exit.
    #[arg(long)]
    info: bool,
    /// Run the greedy conformance set for this model and exit.
    #[arg(long)]
    check: bool,
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    match cli.command {
        Command::Run(args) => run(args),
        Command::Serve(args) => ExitCode::from(serve::run(args, serve::default_factory())),
    }
}

fn run(args: RunArgs) -> ExitCode {
    if let Some(prompt) = &args.prompt
        && prompt.len() > MAX_PROMPT_BYTES
    {
        eprintln!(
            "error: the prompt is {} bytes; the limit is {MAX_PROMPT_BYTES} bytes",
            prompt.len()
        );
        return ExitCode::from(2);
    }
    if let Err(error) = mrun::chat::stop::StopMatcher::new(args.stop.clone()) {
        eprintln!("error: {error}");
        return ExitCode::from(2);
    }
    let path = PathBuf::from(&args.model);
    if path.exists() && !path.join("config.json").exists() {
        eprintln!(
            "error: {} is not a model directory (no config.json)\n  hint: point at a directory with config.json, tokenizer.json and *.safetensors",
            path.display()
        );
        return ExitCode::from(3);
    }
    eprintln!(
        "error: this build of mrun has no inference engine yet\n  hint: the MLX engine is under development; follow the project README for progress"
    );
    ExitCode::from(1)
}
