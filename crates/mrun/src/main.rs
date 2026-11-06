use clap::FromArgMatches;
use mrun::agent;
use mrun::cli::app::{Cli, Command};
use mrun::cli::cloud;
use mrun::cli::connect;
use mrun::cli::doctor;
use mrun::cli::launch;
use mrun::cli::logs;
use mrun::cli::ls;
use mrun::cli::ps;
use mrun::cli::pull;
use mrun::cli::rm;
use mrun::cli::run;
use mrun::cli::serve;
use mrun::cli::show;
use mrun::cli::stop;
use mrun::cli::verify;
use std::process::ExitCode;

fn main() -> ExitCode {
    for warning in mrun::config::unknown_variables(std::env::vars().map(|(name, _)| name)) {
        eprintln!("{warning}");
    }
    // `mrun --version` prints the numerics class with its GPU half, which needs the Metal device,
    // so it is not part of the constant clap prints for subcommands (008 T030).
    let args: Vec<String> = std::env::args().skip(1).collect();
    if let [flag] = args.as_slice()
        && (flag == "--version" || flag == "-V")
    {
        println!("mrun {}", mrun::cli::version::long());
        return ExitCode::SUCCESS;
    }
    // The config file (017): its values become the flags' defaults, so clap resolves the order
    // default < file < MRUN_* < flag. A file that cannot be used stops every command but
    // `mrun config`, which reports it.
    let base = mrun::cli::app::cli_command();
    let resolved = mrun::config::resolve(&base);
    // `mrun config` reports a broken file itself, and `doctor` lists it as a failed check (017
    // FR-009); every other command stops.
    let config_command = args
        .iter()
        .find(|a| !a.starts_with('-'))
        .is_some_and(|a| a == "config" || a == "doctor");
    let file = match &resolved.file {
        Ok(file) => file.clone(),
        Err(error) if !config_command => {
            return ExitCode::from(
                mrun::errors::CliError::new(
                    mrun::errors::Code::InvalidArguments,
                    error.to_string(),
                )
                .hint("fix the config file, or run with MRUN_CONFIG= to ignore it")
                .report(),
            );
        }
        Err(_) => mrun::config::ConfigFile::default(),
    };
    if !config_command {
        for warning in &file.warnings {
            eprintln!("warning: {warning}");
        }
    }
    mrun::config::remember(&resolved.settings, &file);
    let mut applied = mrun::config::apply(base, &resolved.settings, &file);
    let cli = Cli::from_arg_matches(&applied.clone().get_matches()).unwrap_or_else(|e| e.exit());
    mrun::cli::set_quiet(cli.quiet);
    let Some(command) = cli.command else {
        return ExitCode::from(mrun::cli::status::run());
    };
    match command {
        Command::Run(args) => ExitCode::from(run::run(args)),
        Command::Serve(args) => ExitCode::from(serve::run(args, serve::default_factory())),
        Command::Launch(args) => ExitCode::from(launch::run(args)),
        Command::Connect(args) => ExitCode::from(connect::run(args)),
        Command::Ps(args) => ExitCode::from(ps::run(args)),
        Command::Doctor(args) => ExitCode::from(doctor::run(args)),
        Command::Stop(args) => ExitCode::from(stop::run(args)),
        Command::Logs(args) => ExitCode::from(logs::run(args)),
        Command::Cache(command) => ExitCode::from(mrun::cli::cache::run(command)),
        Command::Completions { shell } => {
            clap_complete::generate(shell, &mut applied, "mrun", &mut std::io::stdout());
            ExitCode::SUCCESS
        }
        Command::Pull(args) => ExitCode::from(pull::run(args)),
        Command::Ls(args) => ExitCode::from(ls::run(args)),
        Command::Show(args) => ExitCode::from(show::run(args)),
        Command::Rm(args) => ExitCode::from(rm::run(args)),
        Command::Verify(args) => ExitCode::from(verify::run(args)),
        Command::Bench(args) => ExitCode::from(mrun::bench::run(args)),
        Command::Cloud(command) => ExitCode::from(cloud::run(command)),
        Command::Pool(command) => ExitCode::from(mrun::cli::pool::run(command)),
        Command::Config(command) => ExitCode::from(mrun::cli::config::run(command, &resolved)),
        Command::Agent(args) => ExitCode::from(agent::run(args)),
        Command::Ui(args) => ExitCode::from(mrun::ui::run(args)),
    }
}
