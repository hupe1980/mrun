//! Opening a model for `mrun run` and `mrun serve`: resolve the name, read the family, tokenizer,
//! chat template, identity and sampling defaults, and (with `engine`) build the engine factory.

use crate::chat::syntax::registry::{self, ChatSyntax};
use crate::chat::template::ChatTemplate;
use crate::chat::tokenizer::Tokenizer;
use crate::errors::{CliError, Code};
use crate::hub::{self, Ctx};
use mrun_engine::budget::{GIB, Overrides};
use mrun_engine::identity::{self, Identity};
use mrun_engine::models::{Family, FamilyConfig, LoadError};
use mrun_engine::{ModelInfo, Sampling, SyntaxId};
use serde_json::Value;
use std::path::PathBuf;
use std::sync::Arc;

/// A resolved model with everything the chat layer needs; the engine loads separately.
#[derive(Debug)]
pub struct OpenModel {
    /// The name as given (alias, repo id or path).
    pub name: String,
    pub dir: PathBuf,
    pub family: &'static Family,
    pub tokenizer: Arc<Tokenizer>,
    pub template: Arc<ChatTemplate>,
    pub identity: Identity,
    /// Sampling defaults from `generation_config.json` (greedy without one).
    pub defaults: Sampling,
    /// Weight bytes the engine will load (the manifest's figure, else the `*.safetensors`
    /// sizes) and the fields the budget reads, for the pre-load check (D062).
    pub weights: u64,
    pub info: ModelInfo,
    /// The alias pinned to this model's repo and commit (its drafter, 030).
    pub alias: Option<&'static hub::aliases::Alias>,
}

impl OpenModel {
    pub fn syntax(&self) -> SyntaxId {
        self.family.syntax
    }

    /// Families with a thinking switch in their template: every 0.1 family (Qwen3-Coder's
    /// template ignores it).
    pub fn supports_thinking(&self) -> bool {
        true
    }
}

/// Budget overrides from `--budget-gib` (bytes; D048: it lowers the budget only) or `--context`
/// (tokens; 027: the budget those need).
pub fn overrides(budget: Option<u64>, context: Option<u32>) -> Overrides {
    Overrides { budget, context }
}

/// `--context`: tokens, with `K` for 1,024 (`65536`, `64K`, `64k`).
pub fn parse_tokens(text: &str) -> Result<u32, String> {
    let text = text.trim();
    let (digits, unit) = match text.strip_suffix(['K', 'k']) {
        Some(digits) => (digits, 1024),
        None => (text, 1),
    };
    digits
        .parse::<u32>()
        .ok()
        .and_then(|n| n.checked_mul(unit))
        .filter(|&n| n > 0)
        .ok_or_else(|| format!("'{text}' is not a number of tokens (65536 or 64K)"))
}

/// `--budget-gib`: a positive number of GiB, fractions allowed (`10.5`), in bytes.
pub fn parse_budget_gib(text: &str) -> Result<u64, String> {
    match text.trim().parse::<f64>() {
        Ok(gib) if gib.is_finite() && gib > 0.0 && gib <= 4096.0 => Ok((gib * GIB as f64) as u64),
        _ => Err(format!("'{text}' is not a positive number of GiB")),
    }
}

fn model_error(name: &str, error: impl std::fmt::Display) -> CliError {
    CliError::new(Code::Internal, format!("cannot open {name}: {error}"))
}

/// Resolves `name` and reads the model's family, tokenizer, template and identity. A part (026:
/// a pool stage's layers only) does not open as a model.
pub fn open(name: &str) -> Result<OpenModel, CliError> {
    open_with(name, false)
}

/// [`open`] for a pool stage, which may be a part.
pub fn open_stage(name: &str) -> Result<OpenModel, CliError> {
    open_with(name, true)
}

fn open_with(name: &str, stage: bool) -> Result<OpenModel, CliError> {
    let ctx = Ctx::from_env();
    let local = hub::resolve(&ctx.paths, name)?;
    let dir = local.dir;
    let manifest = local.manifest;
    if let Some([first, end]) = manifest.as_ref().and_then(|m| m.part)
        && !stage
    {
        return Err(CliError::new(
            Code::ModelNotFound,
            format!(
                "{name} is here only as layers {}–{end}, for pools",
                first + 1
            ),
        )
        .hint(format!("'mrun pull {name}' completes it")));
    }
    let config: Value = std::fs::read_to_string(dir.join("config.json"))
        .map_err(|e| model_error(name, e))
        .and_then(|text| serde_json::from_str(&text).map_err(|e| model_error(name, e)))?;
    let family = FamilyConfig::from_config(&config).map_err(|error| match &error {
        LoadError::UnsupportedArchitecture { .. } => {
            CliError::new(Code::UnsupportedArchitecture, error.report(name))
        }
        LoadError::InvalidConfig { .. } => model_error(name, error),
    })?;
    let tokenizer =
        Tokenizer::from_file(&dir.join("tokenizer.json")).map_err(|e| model_error(name, e))?;
    let template = ChatTemplate::from_model_dir(&dir).map_err(|e| model_error(name, e))?;
    // FR-032: the template must render the family's tool-call syntax with added-token markers.
    let syntax = ChatSyntax::get(family.family.syntax);
    registry::check_loaded(syntax, &template, &tokenizer).map_err(|error| {
        CliError::new(
            Code::UnsupportedArchitecture,
            format!(
                "cannot load {name}: family {} needs the {} syntax: {error}",
                family.family.name, syntax.name
            ),
        )
        .hint("use the model's own chat template (chat_template.jinja or tokenizer_config.json)")
    })?;
    // A pulled model's files passed the fast check: its manifest's Hub hashes stand for them.
    let known = manifest.as_ref().map_or_else(Default::default, |manifest| {
        manifest
            .files
            .iter()
            .map(|file| (file.path.clone(), file.sha256.clone()))
            .collect()
    });
    let identity = identity::identify_with(&dir, &ctx.paths.identity_cache(), &known)
        .map_err(|e| model_error(name, e))?;
    if let Some(previous) = &identity.changed_from {
        eprintln!(
            "note: the files of {name} changed since it was last loaded (identity {} → {}); cached state for it is not reused",
            &previous[..previous.len().min(12)],
            &identity.value[..12]
        );
    }
    let checkpoint = family.model_info(String::new(), Vec::new());
    let (weights, info) = match &manifest {
        // The manifest's sizes, with the checkpoint's vocabulary and layers (the drafter's check).
        Some(manifest) => (
            manifest.loaded_weights(),
            ModelInfo {
                vocab: checkpoint.vocab,
                layer_kinds: checkpoint.layer_kinds,
                ..manifest.model_info()
            },
        ),
        None => (safetensors_bytes(&dir), checkpoint),
    };
    let alias = manifest
        .as_ref()
        .and_then(|m| hub::aliases::pinned(&m.repo, &m.revision));
    Ok(OpenModel {
        name: name.to_string(),
        alias,
        weights,
        info,
        defaults: generation_defaults(&dir),
        family: family.family,
        tokenizer: Arc::new(tokenizer),
        template: Arc::new(template),
        identity,
        dir,
    })
}

/// Bytes of the `*.safetensors` files in `dir` (links followed).
fn safetensors_bytes(dir: &std::path::Path) -> u64 {
    std::fs::read_dir(dir)
        .map(|entries| {
            entries
                .filter_map(Result::ok)
                .map(|entry| entry.path())
                .filter(|path| path.extension().is_some_and(|x| x == "safetensors"))
                .filter_map(|path| std::fs::metadata(path).ok())
                .map(|meta| meta.len())
                .sum()
        })
        .unwrap_or(0)
}

/// D062: `serve`, `run` and `launch` refuse (E041) before loading weights when other processes
/// hold the memory the model needs for the default floor of context.
pub fn preflight(model: &OpenModel, context: Option<u32>) -> Result<(), CliError> {
    let Some(device) = hub::device_memory() else {
        return Ok(());
    };
    if let Some(context) = context {
        check_context(device, &model.name, model.weights, &model.info, context)?;
    }
    hub::check_available(device, &model.name, model.weights, &model.info)
}

/// Whether to load the vision tower (009) for `--vision` `choice`: `on` and `off` as said;
/// `auto` loads it unless it alone pushes the context below 64K tokens (or below `context`).
pub fn vision(
    model: &OpenModel,
    choice: &str,
    budget_gib: Option<u64>,
    context: Option<u32>,
) -> bool {
    match choice {
        "on" => true,
        "off" => false,
        _ => {
            let Some(device) = hub::device_memory() else {
                return true;
            };
            let tower = tower_bytes(model);
            if tower == 0 {
                return true;
            }
            let budget = budget_gib.map(|g| g << 30);
            let need = context.unwrap_or(hub::aliases::DEFAULT_MIN_CONTEXT);
            let room =
                |weights: u64| hub::fit_with(device, budget, weights, &model.info).max_context;
            room(model.weights) >= need || room(model.weights.saturating_sub(tower)) < need
        }
    }
}

/// The DFlash drafter to load for `--speculation` `choice` (030): `auto` takes the alias's
/// pulled drafter when it fits the target and the model still leaves 64K tokens (or `context`)
/// with it, and says why when it does not; `dflash` requires it (E043); `lookup` and `off` load
/// none.
pub fn drafter(
    model: &OpenModel,
    choice: crate::cli::serve::Speculation,
    vision: bool,
    budget_gib: Option<u64>,
    context: Option<u32>,
) -> Result<Option<std::path::PathBuf>, CliError> {
    if !choice.drafts_with_model() {
        return Ok(None);
    }
    let required = choice == crate::cli::serve::Speculation::Dflash;
    let refuse = |message: String| {
        if required {
            Err(CliError::new(Code::NoDrafter, message))
        } else {
            Ok(None)
        }
    };
    let Some((alias, pin)) = model
        .alias
        .and_then(|alias| alias.drafter.as_ref().map(|pin| (alias, pin)))
    else {
        return refuse(format!("{} has no DFlash drafter", model.name));
    };
    let ctx = Ctx::from_env();
    let Some(dir) = hub::drafter::pulled(&ctx.paths, alias) else {
        let message = format!(
            "{}'s drafter is not pulled: 'mrun pull {}' fetches it ({}): faster replies, the same output",
            alias.name,
            alias.name,
            hub::human_bytes(pin.size())
        );
        if !required {
            eprintln!("note: {message}");
        }
        return refuse(message);
    };
    if let Err(reason) = fits_target(&dir, &model.info) {
        let message = format!(
            "the drafter in {} does not fit {}: {reason}",
            dir.display(),
            model.name
        );
        if !required {
            eprintln!("note: {message}; decoding without it");
        }
        return refuse(message);
    }
    if !required && let Some(device) = hub::device_memory() {
        let tower = if vision { 0 } else { tower_bytes(model) };
        let base = model.weights.saturating_sub(tower);
        let budget = budget_gib.map(|g| g << 30);
        let need = context.unwrap_or(hub::aliases::DEFAULT_MIN_CONTEXT);
        let room = |weights: u64| hub::fit_with(device, budget, weights, &model.info).max_context;
        if room(base) >= need && room(base + pin.memory(1)) < need {
            eprintln!(
                "note: the DFlash drafter is left out: with it {} would hold fewer than {need} tokens of context",
                model.name
            );
            return Ok(None);
        }
    }
    Ok(Some(dir))
}

/// Whether the drafter in `dir` reads this target: a DFlash config of its vocabulary and layers
/// (the engine checks the width when it loads it).
fn fits_target(dir: &std::path::Path, info: &ModelInfo) -> Result<(), String> {
    #[cfg(feature = "engine")]
    {
        let config: Value = std::fs::read_to_string(dir.join("config.json"))
            .map_err(|e| e.to_string())
            .and_then(|t| serde_json::from_str(&t).map_err(|e| e.to_string()))?;
        let c = mrun_engine::models::dflash::DFlashConfig::from_config(&config)
            .map_err(|e| e.to_string())?;
        if c.vocab != info.vocab {
            return Err(format!("vocabulary {} against {}", c.vocab, info.vocab));
        }
        let layers = info.layer_kinds.len();
        if let Some(i) = c.target_layer_ids.iter().find(|&&i| i >= layers) {
            return Err(format!("it reads layer {i} of {layers}"));
        }
    }
    #[cfg(not(feature = "engine"))]
    let _ = (dir, info);
    Ok(())
}

/// Bytes of `model`'s vision tower (009), 0 without one.
fn tower_bytes(model: &OpenModel) -> u64 {
    #[cfg(feature = "engine")]
    return mrun_engine::load::weights_bytes_of(
        &model.dir,
        mrun_engine::models::vision::is_vision_tensor,
    );
    #[cfg(not(feature = "engine"))]
    {
        let _ = model;
        0
    }
}

/// Refuses (E042) a context outside 4,096 to the model's maximum, or one whose estimated need
/// exceeds the budget, before any weight is read. The engine checks again with its measured
/// scratch.
fn check_context(
    device: mrun_engine::budget::DeviceMemory,
    name: &str,
    weights: u64,
    info: &mrun_engine::ModelInfo,
    context: u32,
) -> Result<(), CliError> {
    use mrun_engine::budget::{self, INITIAL_STATE_TOKENS};
    let max = info.max_position;
    if context < INITIAL_STATE_TOKENS as u32 || context > max {
        return Err(CliError::new(
            Code::ContextTooLarge,
            format!(
                "a context of {context} tokens is outside {INITIAL_STATE_TOKENS} to {name}'s {max}"
            ),
        ));
    }
    let needed = hub::budget_needed(context, weights, info);
    let Ok(plan) = budget::plan(device, Overrides::default()) else {
        return Ok(());
    };
    if needed <= plan.budget {
        return Ok(());
    }
    let fits = hub::fit(device, weights, info).max_context;
    Err(CliError::new(
        Code::ContextTooLarge,
        format!(
            "{context} tokens of context for {name} need {}, more than the budget of {}; up to {fits} tokens fit",
            budget::gib(needed),
            budget::gib(plan.budget)
        ),
    ))
}

/// `temperature`, `top_p`, `top_k` and `min_p` from `generation_config.json`.
fn generation_defaults(dir: &std::path::Path) -> Sampling {
    let Some(config) = std::fs::read_to_string(dir.join("generation_config.json"))
        .ok()
        .and_then(|text| serde_json::from_str::<Value>(&text).ok())
    else {
        return Sampling::GREEDY;
    };
    if config["do_sample"] == Value::Bool(false) {
        return Sampling::GREEDY;
    }
    let f = |key: &str, default: f32| config[key].as_f64().map_or(default, |v| v as f32);
    let sampling = Sampling {
        temperature: f("temperature", 0.0),
        top_p: f("top_p", 1.0),
        top_k: config["top_k"].as_u64().map_or(0, |k| k as u32),
        min_p: f("min_p", 0.0),
        seed: None,
        // Off, as in mlx-lm (it does not read `generation_config.json`'s value).
        repetition_penalty: 1.0,
    };
    if sampling.validate().is_ok() {
        sampling
    } else {
        Sampling::GREEDY
    }
}

/// The engine factory for `model` (runs on the engine thread).
#[cfg(feature = "engine")]
pub fn engine_factory(
    model: &OpenModel,
    overrides: Overrides,
    lanes: Option<usize>,
    vision: bool,
    drafter: Option<std::path::PathBuf>,
) -> impl FnOnce() -> Result<mrun_engine::NuvecEngine, String> + Send + 'static {
    let dir = model.dir.clone();
    let name = model.name.clone();
    let options = mrun_engine::LoadOptions {
        identity: model.identity.value.clone(),
        overrides,
        fallback_device: hub::device_memory(),
        cpu: false,
        calibration: mrun_engine::calibration::default_path(),
        part: None,
        lanes,
        no_vision: !vision,
        drafter,
    };
    move || {
        use mrun_engine::NuvecLoadError;
        mrun_engine::NuvecEngine::load(&dir, options).map_err(|error| {
            let code = match &error {
                NuvecLoadError::Model(LoadError::UnsupportedArchitecture { .. }) => {
                    Code::UnsupportedArchitecture
                }
                NuvecLoadError::Budget(
                    mrun_engine::budget::BudgetError::ContextTooLarge { .. }
                    | mrun_engine::budget::BudgetError::ContextOutOfRange { .. },
                ) => Code::ContextTooLarge,
                NuvecLoadError::Lanes { .. } => Code::ContextTooLarge,
                NuvecLoadError::Budget(_) => Code::DoesNotFit,
                _ => Code::Internal,
            };
            let message = match error {
                NuvecLoadError::Model(error) => error.report(&name),
                other => format!("cannot load {name}: {other}"),
            };
            CliError::new(code, message).to_string()
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mrun_engine::budget::DeviceMemory;

    #[test]
    fn tokens_take_k_for_1024() {
        assert_eq!(parse_tokens("65536"), Ok(65_536));
        assert_eq!(parse_tokens("64K"), Ok(65_536));
        assert_eq!(parse_tokens(" 64k "), Ok(65_536));
        for bad in ["", "0", "-5", "64M", "1.5K", "99999999K"] {
            assert!(parse_tokens(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn a_context_that_does_not_fit_is_refused_with_its_numbers() {
        let device = DeviceMemory {
            physical: 48 * GIB,
            recommended_working_set: 37 * GIB,
            available: None,
            estimated: false,
        };
        // 18 GiB of weights, 20 KiB per token (qwen3.6's shape), 262,144 positions.
        let info = hub::model_info("qwen3_5_moe", 20 * 1024, 64 << 20, 262_144);
        let weights = 18 * GIB;
        let fits = hub::fit(device, weights, &info).max_context;
        assert!(fits > 65_536);
        assert_eq!(check_context(device, "m", weights, &info, 65_536), Ok(()));
        let short = DeviceMemory {
            physical: 24 * GIB,
            recommended_working_set: 20 * GIB,
            ..device
        };
        let error = check_context(short, "m", weights, &info, 200_000).unwrap_err();
        assert_eq!(error.code, Code::ContextTooLarge);
        assert!(
            error
                .message
                .starts_with("200000 tokens of context for m need ")
                && error.message.contains("tokens fit"),
            "{}",
            error.message
        );
        let error = check_context(device, "m", weights, &info, 1_000).unwrap_err();
        assert_eq!(
            error.message,
            "a context of 1000 tokens is outside 4096 to m's 262144"
        );
    }
}
