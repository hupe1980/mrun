//! The command line: `mrun`'s clap tree, in the library so tests and the config file resolver
//! (017) see the same arguments as the binary.

use clap::{Parser, Subcommand};

use crate::agent::AgentArgs;
use crate::bench::BenchArgs;
use crate::cli::cache::CacheCommand;
use crate::cli::cloud::CloudCommand;
use crate::cli::config::ConfigCommand;
use crate::cli::connect::ConnectArgs;
use crate::cli::doctor::DoctorArgs;
use crate::cli::launch::LaunchArgs;
use crate::cli::logs::LogsArgs;
use crate::cli::ls::LsArgs;
use crate::cli::ps::PsArgs;
use crate::cli::pull::PullArgs;
use crate::cli::rm::RmArgs;
use crate::cli::run::RunArgs;
use crate::cli::serve::ServeArgs;
use crate::cli::show::ShowArgs;
use crate::cli::stop::StopArgs;
use crate::cli::verify::VerifyArgs;
use crate::ui::UiArgs;

/// Local inference server for coding agents on Apple Silicon.
#[derive(Debug, Parser)]
#[command(
    name = "mrun",
    version = crate::cli::version::LONG,
    about,
    propagate_version = true,
    after_help = "Examples:\n  mrun                     the server, pulled models and what to do next\n  mrun launch              start the only installed coding agent (pulls and serves as needed)\n  mrun doctor              check this Mac"
)]
pub struct Cli {
    /// Print only warnings and errors.
    #[arg(short, long, global = true)]
    pub quiet: bool,
    /// Without a command: the server, the pulled models and what to do next.
    #[command(subcommand)]
    pub command: Option<Command>,
}

#[derive(Debug, Subcommand)]
pub enum Command {
    /// Chat with a local model in the terminal.
    #[command(
        after_help = "Examples:\n  mrun run                 this Mac's default model (or the running server's)\n  mrun run qwen3.5\n  mrun run ./models/qwen3.5 --prompt \"Explain this error\" --max-tokens 512"
    )]
    Run(RunArgs),
    /// Serve the Anthropic Messages, OpenAI Chat Completions and Responses APIs on 127.0.0.1.
    #[command(
        after_help = "Examples:\n  mrun serve --detach\n  MRUN_PORT=9000 mrun serve --model qwen3.5"
    )]
    Serve(ServeArgs),
    /// Start a coding agent configured for the local server (this process only); pulls the
    /// model and starts the server when needed.
    #[command(
        after_help = "Examples:\n  mrun launch              the only installed client\n  mrun launch claude-code\n  mrun launch codex --dry-run\n  mrun launch claude-code --model qwen3.5 --yes -- --continue"
    )]
    Launch(LaunchArgs),
    /// Print the persistent configuration for a client; --write applies it with a backup.
    #[command(
        after_help = "Examples:\n  mrun connect claude-code\n  mrun connect codex --write\n  mrun connect codex --restore\n  mrun connect zed"
    )]
    Connect(ConnectArgs),
    /// Show the running server: model, budget, requests, cache and the last miss.
    #[command(after_help = "Examples:\n  mrun ps\n  mrun ps --json\n  mrun ps --watch")]
    Ps(PsArgs),
    /// Check this Mac, the server and the clients; prints a fix for each problem.
    #[command(
        after_help = "Examples:\n  mrun doctor\n  mrun doctor --report      a Markdown block to paste into a hardware report"
    )]
    Doctor(DoctorArgs),
    /// Stop the mrun server on a port (started with --detach or in another terminal); open
    /// requests finish first.
    #[command(after_help = "Examples:\n  mrun stop\n  mrun stop --port 9000")]
    Stop(StopArgs),
    /// Show the log of servers started in the background.
    #[command(after_help = "Examples:\n  mrun logs\n  mrun logs -f -n 200")]
    Logs(LogsArgs),
    /// Manage the running server's prefix cache.
    #[command(subcommand, after_help = "Examples:\n  mrun cache clear")]
    Cache(CacheCommand),
    /// Print a shell completion script.
    #[command(
        after_help = "Examples:\n  mrun completions zsh > ~/.zfunc/_mrun\n  mrun completions bash > /etc/bash_completion.d/mrun"
    )]
    Completions { shell: clap_complete::Shell },
    /// Download a model into mrun's model store (or import it from the Hugging Face cache), verify it and record it.
    #[command(
        after_help = "Examples:\n  mrun pull qwen3.5\n  mrun pull mlx-community/Qwen3-8B-4bit@main --yes"
    )]
    Pull(PullArgs),
    /// List pulled models and the curated aliases with family, size, fit and maximum context on
    /// this machine.
    #[command(after_help = "Examples:\n  mrun ls\n  mrun ls --json")]
    Ls(LsArgs),
    /// Show a pulled model: family, architecture, quantization, license, identity and fit.
    #[command(after_help = "Examples:\n  mrun show qwen3.6")]
    Show(ShowArgs),
    /// Remove a pulled model from mrun's model store.
    #[command(after_help = "Examples:\n  mrun rm qwen3.5\n  mrun rm org/name@abc1234 --yes")]
    Rm(RmArgs),
    /// Re-hash every file of a pulled model against its manifest.
    #[command(after_help = "Examples:\n  mrun verify qwen3.6")]
    Verify(VerifyArgs),
    /// Measure mrun (and, with --compare, other local servers): agent replay, synthetic cells
    /// and cancellation; writes result.json and report.md.
    #[command(
        after_help = "Examples:\n  mrun bench --suite agent\n  mrun bench --compare --system mrun=http://127.0.0.1:8484 --system llama.cpp=http://127.0.0.1:9931,model=qwen3.5"
    )]
    Bench(BenchArgs),
    /// Run a model on a rented RunPod GPU behind the usual local APIs (opt-in: prompts and code
    /// are sent to RunPod).
    #[command(
        subcommand,
        after_help = "Examples:\n  mrun cloud gpus\n  mrun cloud models\n  mrun cloud up --model Qwen/Qwen3.6-35B-A3B-FP8 --gpu L40S\n  mrun cloud status\n  mrun cloud down"
    )]
    Cloud(CloudCommand),
    /// Pool this Mac with others on the local network to run a model none of them fits alone
    /// (opt-in: prompts are processed on the pool's Macs).
    #[command(
        subcommand,
        after_help = "Examples:\n  mrun pool start\n  mrun pool ls\n  mrun pool join \"Ben's pool\"\n  mrun pool admit\n  mrun pool leave"
    )]
    Pool(crate::cli::pool::PoolCommand),
    /// Tools for mrun's own chats (the window, `mrun run`): MCP servers in ~/.mrun/mcp.json,
    /// sandboxed, each call asked first; and mrun's own tools as MCP servers.
    #[command(
        subcommand,
        after_help = "Examples:\n  mrun mcp add workspace          read, edit and run in this folder\n  mrun mcp add fetch\n  mrun mcp import claude-code\n  mrun mcp ls\n  mrun mcp allow github --host api.github.com\n  mrun run qwen3.6 --mcp workspace"
    )]
    Mcp(crate::cli::mcp::McpCommand),
    /// The config file: where it is, every setting in force with its source, and its schema.
    #[command(
        subcommand,
        after_help = "Settings are read in this order, later sources winning: built-in defaults, the config file (top-level keys, then the command's table), MRUN_* variables, flags.\n\nExamples:\n  mrun config path\n  mrun config show\n  mrun config schema > ~/.mrun/config.schema.json"
    )]
    Config(ConfigCommand),
    /// Open mrun's window: what the server is doing, a chat with the model, models that fit
    /// this Mac, and one-click agent setup (loopback only; spec 018).
    #[command(
        after_help = "Examples:\n  mrun ui\n  mrun ui --browser\n  mrun ui --background      menu bar only (open at login)"
    )]
    Ui(UiArgs),
    /// The process inside a RunPod pod: supervises vLLM and serves it to the local mrun
    /// (started by `mrun cloud up`; Linux).
    #[command(hide = true)]
    Agent(AgentArgs),
}

/// The command tree before the config file is applied.
pub fn cli_command() -> clap::Command {
    <Cli as clap::CommandFactory>::command()
}
