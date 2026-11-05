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

/// Budget overrides from `--budget-gib` (bytes; D048: it lowers the budget only).
pub fn overrides(budget: Option<u64>) -> Overrides {
    Overrides { budget }
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

/// Resolves `name` and reads the model's family, tokenizer, template and identity.
pub fn open(name: &str) -> Result<OpenModel, CliError> {
    let ctx = Ctx::from_env();
    let local = hub::resolve(&ctx.paths, name)?;
    let dir = local.dir;
    let manifest = local.manifest;
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
    let (weights, info) = match &manifest {
        Some(manifest) => (manifest.loaded_weights(), manifest.model_info()),
        None => (
            safetensors_bytes(&dir),
            family.model_info(String::new(), Vec::new()),
        ),
    };
    Ok(OpenModel {
        name: name.to_string(),
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
pub fn preflight(model: &OpenModel) -> Result<(), CliError> {
    match hub::device_memory() {
        Some(device) => hub::check_available(device, &model.name, model.weights, &model.info),
        None => Ok(()),
    }
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
) -> impl FnOnce() -> Result<mrun_engine::NuvecEngine, String> + Send + 'static {
    let dir = model.dir.clone();
    let name = model.name.clone();
    let options = mrun_engine::LoadOptions {
        identity: model.identity.value.clone(),
        overrides,
        fallback_device: hub::device_memory(),
        cpu: false,
        calibration: mrun_engine::calibration::default_path(),
    };
    move || {
        use mrun_engine::NuvecLoadError;
        mrun_engine::NuvecEngine::load(&dir, options).map_err(|error| {
            let code = match &error {
                NuvecLoadError::Model(LoadError::UnsupportedArchitecture { .. }) => {
                    Code::UnsupportedArchitecture
                }
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
