//! Model families: the registry from `config.json` to a family, and (with `mlx`) the graphs.

pub mod registry;

pub use registry::{Family, FamilyConfig, StateKind, SyntaxId};

/// Why a model cannot be loaded; all are refused before any weights are read.
#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
pub enum LoadError {
    #[error("{}", unsupported(model_type, text_model_type.as_deref(), None))]
    UnsupportedArchitecture {
        model_type: String,
        text_model_type: Option<String>,
    },
    #[error("invalid config.json: {key} {reason}")]
    InvalidConfig { key: String, reason: String },
}

impl LoadError {
    /// The message naming the model (`org/name` or a path), as the CLI prints it.
    pub fn report(&self, model: &str) -> String {
        match self {
            Self::UnsupportedArchitecture {
                model_type,
                text_model_type,
            } => unsupported(model_type, text_model_type.as_deref(), Some(model)),
            other => other.to_string(),
        }
    }
}

fn unsupported(model_type: &str, text_model_type: Option<&str>, model: Option<&str>) -> String {
    let arch = match text_model_type {
        Some(text) => format!("{model_type}+{text}"),
        None => model_type.to_string(),
    };
    let location = model
        .map(|model| format!(" in {model}"))
        .unwrap_or_default();
    format!(
        "unsupported architecture '{arch}'{location}\n  supported families: {}",
        registry::supported()
    )
}
