use clap::{Parser, Subcommand};
use mrun::chat::limits::MAX_PROMPT_BYTES;
use mrun::cli::cache::CacheCommand;
use mrun::cli::connect::{self, ConnectArgs};
use mrun::cli::doctor::{self, DoctorArgs};
use mrun::cli::launch::{self, LaunchArgs};
use mrun::cli::ps::{self, PsArgs};
use mrun::cli::serve::{self, ServeArgs};
use mrun::cli::stop::{self, StopArgs};
use std::path::PathBuf;
use std::process::ExitCode;

/// Local inference server for coding agents on Apple Silicon.
#[derive(Debug, Parser)]
#[command(name = "mrun", version, about, propagate_version = true)]
struct Cli {
    /// Print only warnings and errors.
    #[arg(short, long, global = true)]
    quiet: bool,
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
    /// Start a coding agent configured for the local server (this process only).
    #[command(
        after_help = "Examples:\n  mrun launch claude-code\n  mrun launch codex --dry-run\n  mrun launch opencode -- --continue"
    )]
    Launch(LaunchArgs),
    /// Print the persistent configuration for a client; writes nothing.
    #[command(after_help = "Examples:\n  mrun connect claude-code\n  mrun connect codex")]
    Connect(ConnectArgs),
    /// Show the running server: model, budget, requests, cache and the last miss.
    #[command(after_help = "Examples:\n  mrun ps\n  mrun ps --json")]
    Ps(PsArgs),
    /// Check this Mac, the server and the clients; prints a fix for each problem.
    #[command(after_help = "Examples:\n  mrun doctor")]
    Doctor(DoctorArgs),
    /// Stop a server started with `mrun serve --detach`.
    #[command(after_help = "Examples:\n  mrun stop")]
    Stop(StopArgs),
    /// Manage the running server's prefix cache.
    #[command(subcommand, after_help = "Examples:\n  mrun cache clear")]
    Cache(CacheCommand),
    /// Print a shell completion script.
    #[command(
        after_help = "Examples:\n  mrun completions zsh > ~/.zfunc/_mrun\n  mrun completions bash > /etc/bash_completion.d/mrun"
    )]
    Completions { shell: clap_complete::Shell },
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
    for warning in mrun::config::unknown_variables(std::env::vars().map(|(name, _)| name)) {
        eprintln!("{warning}");
    }
    let cli = Cli::parse();
    mrun::cli::set_quiet(cli.quiet);
    match cli.command {
        Command::Run(args) => run(args),
        Command::Serve(args) => ExitCode::from(serve::run(args, serve::default_factory())),
        Command::Launch(args) => ExitCode::from(launch::run(args)),
        Command::Connect(args) => ExitCode::from(connect::run(args)),
        Command::Ps(args) => ExitCode::from(ps::run(args)),
        Command::Doctor(args) => ExitCode::from(doctor::run(args)),
        Command::Stop(args) => ExitCode::from(stop::run(args)),
        Command::Cache(command) => ExitCode::from(mrun::cli::cache::run(command)),
        Command::Completions { shell } => {
            use clap::CommandFactory;
            clap_complete::generate(shell, &mut Cli::command(), "mrun", &mut std::io::stdout());
            ExitCode::SUCCESS
        }
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
