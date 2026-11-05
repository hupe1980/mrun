//! `mrun run`: chat with a local model in the terminal (001 US1). Each turn is rendered by the
//! same renderer as the APIs and generated through the engine handle with `Session` reuse, so a
//! follow-up only processes the new tokens.

use crate::api::model::{ApiKind, AssistantPart, ChatRequest, Message, ReasoningCtl};
use crate::chat::limits::MAX_PROMPT_BYTES;
use crate::chat::render::{RenderContext, render};
use crate::chat::splice::{self, Reply, ReplyParts};
use crate::chat::stop::StopMatcher;
use crate::cli::model::{self, OpenModel};
use crate::errors::{CliError, Exit};
use crate::hub::human_bytes;
use crate::runtime::generate::{Event, FinishReason, Finished, GenerateRequest};
use crate::runtime::handle::{EngineHandle, Ticket};
use mrun_engine::{MemoryBudget, ModelInfo, Sampling};
use std::io::{BufRead, IsTerminal, Write};
use std::path::Path;
use std::sync::{Arc, Mutex, PoisonError};
use std::time::{Duration, Instant};

/// Flags of `mrun run`.
#[derive(Debug, clap::Args)]
pub struct RunArgs {
    /// Model: alias, Hugging Face repo id (org/name) or local directory [default: the running
    /// server's model, else this machine's default].
    pub model: Option<String>,
    /// One-shot prompt; without it an interactive session starts.
    #[arg(long)]
    pub prompt: Option<String>,
    /// System prompt.
    #[arg(long)]
    pub system: Option<String>,
    /// Sampling temperature [default: the model's generation_config.json, else 0].
    #[arg(long)]
    pub temperature: Option<f32>,
    /// [default: the model's generation_config.json, else 1]
    #[arg(long)]
    pub top_p: Option<f32>,
    /// [default: the model's generation_config.json, else 0 (off)]
    #[arg(long)]
    pub top_k: Option<u32>,
    /// [default: the model's generation_config.json, else 0 (off)]
    #[arg(long)]
    pub min_p: Option<f32>,
    #[arg(long)]
    pub seed: Option<u64>,
    #[arg(long, default_value_t = 2048)]
    pub max_tokens: u32,
    /// Stop sequence (up to 4, each at most 256 bytes).
    #[arg(long)]
    pub stop: Vec<String>,
    /// Let models that support it think before answering.
    #[arg(long)]
    pub think: bool,
    /// Lower the memory budget to this many GiB [default: the GPU's recommended working set,
    /// at most physical memory − 2 GiB]; it cannot raise it.
    #[arg(long, env = "MRUN_BUDGET_GIB", value_parser = crate::cli::model::parse_budget_gib)]
    pub budget_gib: Option<u64>,
    /// Print the memory budget breakdown after loading and exit.
    #[arg(long)]
    pub info: bool,
}

impl RunArgs {
    /// The flags over the model's sampling defaults.
    fn sampling(&self, defaults: Sampling) -> Sampling {
        Sampling {
            temperature: self.temperature.unwrap_or(defaults.temperature),
            top_p: self.top_p.unwrap_or(defaults.top_p),
            top_k: self.top_k.unwrap_or(defaults.top_k),
            min_p: self.min_p.unwrap_or(defaults.min_p),
            seed: self.seed.or(defaults.seed),
            repetition_penalty: defaults.repetition_penalty,
        }
    }
}

/// The model `run` opens when none is named: the running server's, else this machine's default.
fn default_model(budget: Option<u64>) -> Result<String, crate::errors::CliError> {
    crate::cli::server::ServerInfo::fetch(crate::cli::server::default_port()).map_or_else(
        |_| crate::hub::default_model(&crate::hub::Ctx::from_env().with_budget(budget)),
        |server| Ok(server.model_id),
    )
}

/// Usage errors found from the flags alone, before any model is opened (exit 2): an empty or
/// oversized prompt, invalid stop sequences, invalid sampling values, `--max-tokens 0`.
fn check_args(args: &RunArgs) -> Result<(), String> {
    if let Some(prompt) = &args.prompt {
        if prompt.trim().is_empty() {
            return Err("the prompt is empty".into());
        }
        if prompt.len() > MAX_PROMPT_BYTES {
            return Err(format!(
                "the prompt is {} bytes; the limit is {MAX_PROMPT_BYTES} bytes",
                prompt.len()
            ));
        }
    }
    StopMatcher::new(args.stop.clone()).map_err(|error| error.to_string())?;
    args.sampling(Sampling::GREEDY).validate()?;
    if args.max_tokens == 0 {
        return Err("--max-tokens must be at least 1".into());
    }
    Ok(())
}

/// Runs `mrun run`; returns the exit code.
pub fn run(args: RunArgs) -> u8 {
    if let Err(error) = check_args(&args) {
        eprintln!("error: {error}");
        return Exit::Usage as u8;
    }
    let name = match args
        .model
        .clone()
        .map_or_else(|| default_model(args.budget_gib), Ok)
    {
        Ok(name) => name,
        Err(error) => return error.report(),
    };
    let path = Path::new(&name);
    if path.exists() && !path.join("config.json").exists() {
        eprintln!(
            "error: {} is not a model directory (no config.json)\n  hint: point at a directory with config.json, tokenizer.json and *.safetensors",
            path.display()
        );
        return Exit::Model as u8;
    }
    let model = match model::open(&name) {
        Ok(model) => model,
        Err(error) => return error.report(),
    };
    let sampling = args.sampling(model.defaults);
    if let Err(error) = sampling.validate() {
        eprintln!("error: {error}");
        return Exit::Usage as u8;
    }
    if let Err(error) = model::preflight(&model) {
        return error.report();
    }
    let handle = match spawn(&model, &args) {
        Ok(handle) => handle,
        Err(error) => return CliError::from(error).report(),
    };
    if args.info {
        print_info(&model, &handle);
        return 0;
    }
    let mut chat = Chat::new(&model, &handle, &args, sampling);
    chat.interrupt = Interrupt::install();
    match &args.prompt {
        Some(prompt) => chat.turn(prompt),
        None => chat.repl(),
    }
}

#[cfg(feature = "engine")]
fn spawn(model: &OpenModel, args: &RunArgs) -> Result<EngineHandle, String> {
    let overrides = model::overrides(args.budget_gib);
    crate::cli::info(&format!("loading {} …", model.name));
    EngineHandle::spawn(
        model::engine_factory(model, overrides),
        model.tokenizer.clone(),
        crate::runtime::handle::DEFAULT_STALL_TIMEOUT,
    )
}

#[cfg(not(feature = "engine"))]
fn spawn(_model: &OpenModel, _args: &RunArgs) -> Result<EngineHandle, String> {
    Err(CliError::new(
        crate::errors::Code::Internal,
        "this build of mrun has no inference engine",
    )
    .hint("build with the default features (the `engine` feature)")
    .into())
}

fn print_info(model: &OpenModel, handle: &EngineHandle) {
    let syntax = crate::chat::syntax::registry::ChatSyntax::get(model.syntax()).name;
    print!(
        "{}",
        info_text(
            &model.name,
            syntax,
            &model.dir.display().to_string(),
            handle.info(),
            handle.budget(),
            raised_wired_limit(),
        )
    );
}

/// `iogpu.wired_limit_mb` when it has been raised (non-zero), read once (FR-016).
fn raised_wired_limit() -> Option<u64> {
    static LIMIT: std::sync::OnceLock<Option<u64>> = std::sync::OnceLock::new();
    *LIMIT.get_or_init(|| {
        let out = std::process::Command::new("sysctl")
            .args(["-n", "iogpu.wired_limit_mb"])
            .output()
            .ok()?;
        let mb: u64 = String::from_utf8_lossy(&out.stdout).trim().parse().ok()?;
        (mb > 0).then_some(mb << 20)
    })
}

/// The `--info` breakdown (US3-4): memory and budget split, the raised wired limit, the scratch
/// depth, the calibrated chunk lengths and the numerics class.
fn info_text(
    name: &str,
    syntax: &str,
    dir: &str,
    info: &ModelInfo,
    budget: &MemoryBudget,
    wired_limit: Option<u64>,
) -> String {
    use std::fmt::Write as _;
    let mut out = String::new();
    let _ = writeln!(out, "model      {name} ({}, {syntax})", info.family);
    let _ = writeln!(out, "directory  {dir}");
    let _ = writeln!(out, "identity   {}", info.identity);
    let _ = writeln!(out, "numerics   {}", info.numerics);
    let available = budget
        .available
        .map(|bytes| format!(", {} available", human_bytes(bytes)))
        .unwrap_or_default();
    let _ = writeln!(
        out,
        "memory     {} physical, {} working set{available}",
        human_bytes(budget.physical),
        human_bytes(budget.recommended_working_set)
    );
    if let Some(limit) = wired_limit {
        let _ = writeln!(
            out,
            "wired      limit raised to {} (the budget stays at most physical − 2 GiB)",
            human_bytes(limit)
        );
    }
    let _ = writeln!(
        out,
        "budget     {} (bound by {})",
        human_bytes(budget.budget),
        budget.binding
    );
    let _ = writeln!(out, "  weights  {} (measured)", human_bytes(budget.weights));
    let _ = writeln!(
        out,
        "  scratch  {} (at depth {})",
        human_bytes(budget.scratch),
        budget.scratch_depth
    );
    let _ = writeln!(
        out,
        "  state    {} = {} tokens of context ({} per token; max position {})",
        human_bytes(budget.state_capacity_bytes),
        budget.kv_capacity_tokens,
        human_bytes(info.kv_bytes_per_token),
        info.max_position
    );
    if info.recurrent_bytes > 0 {
        let _ = writeln!(
            out,
            "  position-bound {} per checkpoint",
            human_bytes(info.recurrent_bytes)
        );
    }
    let chunks: Vec<String> = [0u32, 16_384, 65_536]
        .iter()
        .filter(|&&d| d == 0 || d < budget.kv_capacity_tokens)
        .map(|&d| format!("{} at {}K", info.chunk_len(d), d / 1024))
        .collect();
    let calibrated = if info.time_len.is_unbounded() {
        "memory bound"
    } else {
        "calibrated"
    };
    let _ = writeln!(out, "chunk_len  {} ({calibrated})", chunks.join(", "));
    out
}

/// Ctrl-C during `run` (US2): the first press cancels the running request (the prompt's
/// completed chunks stay cached); a second within a second, or one while no request runs,
/// exits (130).
#[derive(Clone, Debug, Default)]
struct Interrupt {
    state: Arc<Mutex<InterruptState>>,
}

#[derive(Debug, Default)]
struct InterruptState {
    active: Option<Ticket>,
    last: Option<Instant>,
}

#[derive(Debug, Eq, PartialEq)]
enum OnInterrupt {
    Cancel,
    Exit,
}

/// A second press within this long exits.
const DOUBLE_PRESS: Duration = Duration::from_secs(1);

fn on_interrupt(running: bool, last: Option<Instant>, now: Instant) -> OnInterrupt {
    let double = last.is_some_and(|last| now.saturating_duration_since(last) < DOUBLE_PRESS);
    if running && !double {
        OnInterrupt::Cancel
    } else {
        OnInterrupt::Exit
    }
}

impl Interrupt {
    /// Handles SIGINT on a background thread from now on.
    fn install() -> Self {
        let interrupt = Self::default();
        let handler = interrupt.clone();
        let spawned = std::thread::Builder::new()
            .name("ctrl-c".into())
            .spawn(move || {
                let Ok(runtime) = tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                else {
                    return;
                };
                runtime.block_on(async move {
                    while tokio::signal::ctrl_c().await.is_ok() {
                        if handler.pressed(Instant::now()) == OnInterrupt::Exit {
                            eprintln!();
                            std::process::exit(130);
                        }
                    }
                });
            });
        if spawned.is_err() {
            eprintln!("warning: Ctrl-C will not cancel a running request");
        }
        interrupt
    }

    fn pressed(&self, now: Instant) -> OnInterrupt {
        let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        let action = on_interrupt(state.active.is_some(), state.last, now);
        state.last = Some(now);
        if action == OnInterrupt::Cancel
            && let Some(ticket) = &state.active
        {
            ticket.cancel();
            eprintln!("\n(cancelling; press Ctrl-C again to quit)");
        }
        action
    }

    fn running(&self, ticket: Option<Ticket>) {
        self.state
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .active = ticket;
    }
}

struct Chat<'a> {
    model: &'a OpenModel,
    handle: &'a EngineHandle,
    request: ChatRequest,
    sampling: Sampling,
    max_tokens: u32,
    stop: Vec<String>,
    interrupt: Interrupt,
}

impl<'a> Chat<'a> {
    fn new(
        model: &'a OpenModel,
        handle: &'a EngineHandle,
        args: &RunArgs,
        sampling: Sampling,
    ) -> Self {
        let mut request = ChatRequest::new(ApiKind::Chat);
        request.system = args.system.iter().cloned().collect();
        if args.think {
            request.reasoning = ReasoningCtl::On { budget: None };
        }
        Self {
            model,
            handle,
            request,
            sampling,
            max_tokens: args.max_tokens,
            stop: args.stop.clone(),
            interrupt: Interrupt::default(),
        }
    }

    fn repl(&mut self) -> u8 {
        let interactive = std::io::stdin().is_terminal();
        if interactive {
            eprintln!("Type a message; /clear starts over, /info shows the budget, /exit quits.");
        }
        let stdin = std::io::stdin();
        let mut lines = stdin.lock().lines();
        loop {
            if interactive {
                print!(">>> ");
                let _ = std::io::stdout().flush();
            }
            let Some(Ok(line)) = lines.next() else {
                return 0;
            };
            let line = line.trim();
            match line {
                "" => continue,
                "/exit" | "/quit" | "/bye" => return 0,
                "/clear" => {
                    self.request.messages.clear();
                    if let Err(error) = self.handle.reset() {
                        eprintln!("error: {error}");
                        return 1;
                    }
                    eprintln!("(conversation cleared)");
                }
                "/info" => print_info(self.model, self.handle),
                text if text.starts_with('/') => {
                    eprintln!("unknown command {text}; try /clear, /info or /exit");
                }
                text => {
                    let code = self.turn(text);
                    if code != 0 && !interactive {
                        return code;
                    }
                }
            }
        }
    }

    /// One user turn: render, generate, stream the reply and print the stats line.
    fn turn(&mut self, text: &str) -> u8 {
        self.request.messages.push(Message::user_text(text));
        let ctx = RenderContext {
            template: &self.model.template,
            tokenizer: &self.model.tokenizer,
            syntax: self.model.syntax(),
            supports_thinking: self.model.supports_thinking(),
        };
        let mut rendered = match render(&self.request, &ctx) {
            Ok(rendered) => rendered,
            Err(error) => {
                self.request.messages.pop();
                eprintln!("error: {}", error.message);
                return 1;
            }
        };
        // D051: the current turn's replies come back as the ids the model generated.
        {
            let mut ledger = self
                .handle
                .ledger()
                .lock()
                .unwrap_or_else(PoisonError::into_inner);
            let spliced = splice::splice(
                &ledger,
                &mut rendered.prompt,
                &rendered.replies,
                &self.request.messages,
                &self.request.tools,
            );
            ledger.note_spliced(spliced.tokens);
        }
        let prompt = rendered.prompt.tokens.clone();
        let generation_start = rendered.prompt.generation_start;
        let mut request = GenerateRequest::new(rendered.prompt, self.max_tokens);
        request.sampling = self.sampling;
        request.stop = self.stop.clone();
        let generation = match self.handle.generate(request) {
            Ok(generation) => generation,
            Err(error) => {
                self.request.messages.pop();
                eprintln!("error: {error}");
                return 1;
            }
        };
        self.interrupt.running(Some(generation.ticket.clone()));
        let mut reply = String::new();
        let mut generated = Vec::new();
        let mut out = std::io::stdout();
        for event in generation.events.iter() {
            let text = match event {
                Event::Token { id, text, .. } => {
                    generated.push(id);
                    text
                }
                Event::Flush { text } => text,
                _ => continue,
            };
            reply.push_str(&text);
            let _ = out.write_all(text.as_bytes());
            let _ = out.flush();
        }
        let finished = generation.finished.recv();
        self.interrupt.running(None);
        let Ok(finished) = finished else {
            eprintln!("\nerror: the engine thread has stopped");
            return 1;
        };
        if !reply.ends_with('\n') {
            println!();
        }
        eprintln!("{}", stats_line(&finished));
        match finished.reason {
            FinishReason::Rejected(_) | FinishReason::Error(_) => {
                self.request.messages.pop();
                1
            }
            reason => {
                let parts = vec![AssistantPart::Text(reply)];
                if let FinishReason::Stop {
                    stop: None,
                    eos_token: Some(eos),
                } = reason
                {
                    self.handle
                        .ledger()
                        .lock()
                        .unwrap_or_else(PoisonError::into_inner)
                        .record(Reply {
                            prompt,
                            generation_start,
                            generated,
                            eos,
                            canonical: ReplyParts::of_message(&parts)
                                .canonical(&self.request.tools),
                        });
                }
                self.request.messages.push(Message::Assistant(parts));
                0
            }
        }
    }
}

fn stats_line(finished: &Finished) -> String {
    let s = &finished.stats;
    let reason = match &finished.reason {
        FinishReason::Stop {
            stop: Some(stop), ..
        } => format!("stop {stop:?}"),
        FinishReason::Stop { .. } => "stop".into(),
        FinishReason::Length => "length".into(),
        FinishReason::Repetition => "stopped: repeating output".into(),
        FinishReason::Cancelled => "cancelled".into(),
        FinishReason::Rejected(why) => format!("rejected: {why}"),
        FinishReason::Error(why) => format!("error: {why}"),
    };
    let ttft = s
        .ttft
        .map_or_else(|| "-".into(), |t| format!("{:.2} s", t.as_secs_f64()));
    format!(
        "[{} prompt ({} cached) · {} generated (limit {}) · TTFT {ttft} · prefill {:.0} tok/s · decode {:.1} tok/s · {reason}]",
        s.prompt_tokens,
        s.cached_tokens,
        s.generated_tokens,
        s.max_tokens,
        s.prefill_tps,
        s.decode_tps
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;
    use mrun_engine::{LayerKind, Numerics, TimeLen};

    #[derive(Debug, clap::Parser)]
    struct Cli {
        #[command(flatten)]
        run: RunArgs,
    }

    fn args(argv: &[&str]) -> RunArgs {
        Cli::try_parse_from(std::iter::once("run").chain(argv.iter().copied()))
            .unwrap()
            .run
    }

    #[test]
    fn usage_errors_are_found_before_loading() {
        assert!(check_args(&args(&["m", "--prompt", "hi"])).is_ok());
        for (argv, says) in [
            (&["m", "--prompt", "  "][..], "empty"),
            (&["m", "--max-tokens", "0"], "--max-tokens"),
            (&["m", "--temperature=-1"], "temperature"),
            (&["m", "--top-p", "0"], "top_p"),
            (&["m", "--min-p", "2"], "min_p"),
            (
                &[
                    "m", "--stop", "a", "--stop", "b", "--stop", "c", "--stop", "d", "--stop", "e",
                ],
                "stop",
            ),
        ] {
            let error = check_args(&args(argv)).unwrap_err();
            assert!(error.contains(says), "{argv:?}: {error}");
        }
        let huge = "x".repeat(MAX_PROMPT_BYTES + 1);
        let error = check_args(&args(&["m", "--prompt", &huge])).unwrap_err();
        assert!(error.contains("bytes"), "{error}");
    }

    #[test]
    fn ctrl_c_cancels_once_and_exits_on_a_second_press() {
        let now = Instant::now();
        assert_eq!(on_interrupt(true, None, now), OnInterrupt::Cancel);
        assert_eq!(
            on_interrupt(true, Some(now), now + Duration::from_millis(400)),
            OnInterrupt::Exit
        );
        assert_eq!(
            on_interrupt(true, Some(now), now + Duration::from_millis(1500)),
            OnInterrupt::Cancel
        );
        // At the prompt, Ctrl-C quits.
        assert_eq!(on_interrupt(false, None, now), OnInterrupt::Exit);
        // Without a running request a press only records the time.
        let interrupt = Interrupt::default();
        assert_eq!(interrupt.pressed(now), OnInterrupt::Exit);
    }

    #[test]
    fn info_shows_the_breakdown() {
        let gib = 1u64 << 30;
        let info = ModelInfo {
            identity: "abc123".into(),
            family: "qwen3_5".into(),
            layer_kinds: vec![LayerKind::Linear, LayerKind::FullAttention],
            max_position: 262_144,
            kv_bytes_per_token: 32 << 10,
            recurrent_bytes: 51 << 20,
            numerics: Numerics {
                nax: false,
                gpu_arch: "applegpu_g16s".into(),
            },
            time_len: TimeLen::from_points(vec![(0, 2048), (16_384, 1536), (65_536, 512)]),
            ..ModelInfo::default()
        };
        let budget = MemoryBudget {
            physical: 48 * gib,
            recommended_working_set: 37 * gib,
            available: Some(40 * gib),
            budget: 37 * gib,
            binding: mrun_engine::Binding::WorkingSet,
            weights: 5 * gib,
            scratch: 3 * gib,
            scratch_depth: 262_144,
            state_capacity_bytes: 29 * gib,
            kv_capacity_tokens: 262_144,
        };
        let text = info_text("qwen3.5", "qwen-xml", "/m", &info, &budget, Some(46 * gib));
        let want = format!(
            "model      qwen3.5 (qwen3_5, qwen-xml)\n\
             directory  /m\n\
             identity   abc123\n\
             numerics   applegpu_g16s, nuvec\n\
             memory     {} physical, {} working set, {} available\n\
             wired      limit raised to {} (the budget stays at most physical − 2 GiB)\n\
             budget     {} (bound by working set)\n  weights  {} (measured)\n  scratch  {} (at depth 262144)\n  \
             state    {} = 262144 tokens of context ({} per token; max position 262144)\n  \
             position-bound {} per checkpoint\n\
             chunk_len  2048 at 0K, 1536 at 16K, 512 at 64K (calibrated)\n",
            human_bytes(48 * gib),
            human_bytes(37 * gib),
            human_bytes(40 * gib),
            human_bytes(46 * gib),
            human_bytes(37 * gib),
            human_bytes(5 * gib),
            human_bytes(3 * gib),
            human_bytes(29 * gib),
            human_bytes(32 << 10),
            human_bytes(51 << 20),
        );
        assert_eq!(text, want);
        // No raised limit, no calibration.
        let plain = ModelInfo {
            time_len: TimeLen::unbounded(),
            ..info
        };
        let text = info_text("q", "s", "/m", &plain, &budget, None);
        assert!(!text.contains("wired"));
        assert!(text.ends_with("chunk_len  2048 at 0K, 2048 at 16K, 512 at 64K (memory bound)\n"));
    }
}
