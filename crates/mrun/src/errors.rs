//! User-facing errors: a stable code, a message and the next step, with the CLI exit codes of the
//! 004 contract.
//!
//! ```text
//! error[E031]: model 'qwen3.7' not found
//!   hint: did you mean 'qwen3.6'? Run 'mrun ls' to see local models.
//! ```

use std::fmt;

/// Process exit codes (004 contract).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum Exit {
    Success = 0,
    Unexpected = 1,
    Usage = 2,
    /// Model not found, unsupported or corrupt.
    Model = 3,
    /// Does not fit the memory budget.
    Fit = 4,
    ServerUnreachable = 5,
    Network = 6,
    DoctorFailures = 7,
}

/// Error codes of the catalog. Numbers are stable once released.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Code {
    /// E001 unexpected internal error.
    Internal,
    /// E002 a destructive or risky step needs confirmation and stdin is not a terminal.
    NeedsConfirmation,
    /// E003 the arguments contradict each other or the running server.
    InvalidArguments,
    /// E010 the server is not running, not reachable or did not stop.
    ServerUnreachable,
    /// E020 the client program is not installed.
    ClientMissing,
    /// E031 the model is not in the local cache.
    ModelNotFound,
    /// E032 a model file does not match its recorded hash.
    ModelCorrupt,
    /// E033 several cached models match the name.
    ModelAmbiguous,
    /// E034 the architecture is not supported (single message for every command).
    UnsupportedArchitecture,
    /// E040 the model does not fit this Mac's memory budget.
    DoesNotFit,
    /// E041 the model would fit, but other processes hold the memory it needs now (D062).
    MemoryShort,
    /// E050 network or download failure.
    Network,
    /// E051 a gated model needs a Hugging Face token.
    Gated,
    /// E052 not enough disk space for the download.
    DiskFull,
}

impl Code {
    pub fn id(self) -> &'static str {
        match self {
            Self::Internal => "E001",
            Self::NeedsConfirmation => "E002",
            Self::InvalidArguments => "E003",
            Self::ServerUnreachable => "E010",
            Self::ClientMissing => "E020",
            Self::ModelNotFound => "E031",
            Self::ModelCorrupt => "E032",
            Self::ModelAmbiguous => "E033",
            Self::UnsupportedArchitecture => "E034",
            Self::DoesNotFit => "E040",
            Self::MemoryShort => "E041",
            Self::Network => "E050",
            Self::Gated => "E051",
            Self::DiskFull => "E052",
        }
    }

    pub fn exit(self) -> Exit {
        match self {
            Self::Internal => Exit::Unexpected,
            Self::NeedsConfirmation | Self::InvalidArguments | Self::ClientMissing => Exit::Usage,
            Self::ServerUnreachable => Exit::ServerUnreachable,
            Self::ModelNotFound
            | Self::ModelCorrupt
            | Self::ModelAmbiguous
            | Self::UnsupportedArchitecture => Exit::Model,
            Self::DoesNotFit | Self::MemoryShort => Exit::Fit,
            Self::Network | Self::Gated | Self::DiskFull => Exit::Network,
        }
    }

    /// The generic next step, printed when an error carries no specific hint (FR-022: every
    /// error names a next step).
    pub fn next_step(self) -> &'static str {
        match self {
            Self::Internal => "this is a bug; rerun with RUST_LOG=debug and report the output",
            Self::NeedsConfirmation => "rerun in a terminal, or pass --yes",
            Self::InvalidArguments => "see 'mrun <command> --help'",
            Self::ServerUnreachable => {
                "'mrun ps' shows the server; 'mrun serve --detach' starts one"
            }
            Self::ClientMissing => "install the client, or run 'mrun connect' for its settings",
            Self::ModelNotFound => {
                "'mrun ls' lists local models; 'mrun pull <model>' downloads one"
            }
            Self::ModelCorrupt => "'mrun pull <model>' re-downloads the damaged files",
            Self::ModelAmbiguous => "name the model as org/name",
            Self::UnsupportedArchitecture => "'mrun ls' lists the models this build supports",
            Self::DoesNotFit => {
                "pick a smaller model (`mrun ls` shows what fits), or quit other apps and run without --budget-gib, which can only lower the budget"
            }
            Self::MemoryShort => {
                "quit memory-hungry apps or VMs ('mrun doctor' lists the largest), or pick a smaller model"
            }
            Self::Network => "check the connection and rerun; downloads resume",
            Self::Gated => "accept the license on huggingface.co and set HF_TOKEN",
            Self::DiskFull => "free disk space, or point HF_HUB_CACHE at a larger volume",
        }
    }

    /// Every code, for the catalog snapshot.
    pub const ALL: [Code; 14] = [
        Self::Internal,
        Self::NeedsConfirmation,
        Self::InvalidArguments,
        Self::ServerUnreachable,
        Self::ClientMissing,
        Self::ModelNotFound,
        Self::ModelCorrupt,
        Self::ModelAmbiguous,
        Self::UnsupportedArchitecture,
        Self::DoesNotFit,
        Self::MemoryShort,
        Self::Network,
        Self::Gated,
        Self::DiskFull,
    ];
}

/// An error with its next step.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CliError {
    pub code: Code,
    pub message: String,
    pub hint: Option<String>,
}

impl CliError {
    pub fn new(code: Code, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
            hint: None,
        }
    }

    pub fn hint(mut self, hint: impl Into<String>) -> Self {
        self.hint = Some(hint.into());
        self
    }

    pub fn exit(&self) -> Exit {
        self.code.exit()
    }

    /// Prints to stderr and returns the exit code.
    pub fn report(&self) -> u8 {
        eprintln!("{self}");
        self.exit() as u8
    }
}

impl fmt::Display for CliError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "error[{}]: {}", self.code.id(), self.message)?;
        let hint = self.hint.as_deref().unwrap_or(self.code.next_step());
        write!(f, "\n  hint: {hint}")
    }
}

impl std::error::Error for CliError {}

/// Parses the [`fmt::Display`] form back (`error[E031]: message\n  hint: …`), so an error that
/// crossed a `String` boundary (the engine thread) keeps its code. Text without a known code is
/// E001.
impl From<String> for CliError {
    fn from(text: String) -> Self {
        let parsed = text.strip_prefix("error[").and_then(|rest| {
            let (id, rest) = rest.split_once("]: ")?;
            let code = Code::ALL.into_iter().find(|code| code.id() == id)?;
            Some(match rest.split_once("\n  hint: ") {
                // The generic next step is what Display prints for no hint.
                Some((message, hint)) if hint == code.next_step() => Self::new(code, message),
                Some((message, hint)) => Self::new(code, message).hint(hint),
                None => Self::new(code, rest),
            })
        });
        parsed.unwrap_or_else(|| Self::new(Code::Internal, text))
    }
}

impl From<CliError> for String {
    fn from(error: CliError) -> Self {
        error.to_string()
    }
}

/// Serialized as its text, as `/health` reports a failed load.
impl serde::Serialize for CliError {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_str(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn catalog_codes_are_unique_and_map_to_contract_exits() {
        let mut ids: Vec<&str> = Code::ALL.iter().map(|c| c.id()).collect();
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(ids.len(), Code::ALL.len());
        assert_eq!(Code::UnsupportedArchitecture.exit() as u8, 3);
        assert_eq!(Code::DoesNotFit.exit() as u8, 4);
        assert_eq!(Code::ServerUnreachable.exit() as u8, 5);
        assert_eq!(Code::Gated.exit() as u8, 6);
        assert_eq!(Code::ClientMissing.exit() as u8, 2);
    }

    /// The catalog as documented; a change here is a change of the public contract.
    #[test]
    fn catalog_snapshot() {
        let catalog: Vec<String> = Code::ALL
            .iter()
            .map(|code| format!("{} {}", code.id(), code.exit() as u8))
            .collect();
        assert_eq!(
            catalog.join(", "),
            "E001 1, E002 2, E003 2, E010 5, E020 2, E031 3, E032 3, E033 3, E034 3, E040 4, \
             E041 4, E050 6, E051 6, E052 6"
        );
    }

    #[test]
    fn errors_survive_a_string_round_trip() {
        let error = CliError::new(Code::DoesNotFit, "too big").hint("pick a smaller model");
        assert_eq!(CliError::from(String::from(error.clone())), error);
        assert_eq!(
            CliError::from("boom".to_string()),
            CliError::new(Code::Internal, "boom")
        );
        assert_eq!(
            serde_json::to_value(&error).unwrap(),
            "error[E040]: too big\n  hint: pick a smaller model"
        );
    }

    /// FR-022, SC-005: every code has a next step and is emitted somewhere in `src` (outside
    /// this catalog), so the catalog holds no dead entries.
    #[test]
    fn every_code_has_a_next_step_and_is_emitted() {
        fn sources(dir: &std::path::Path, out: &mut String) {
            for entry in std::fs::read_dir(dir).unwrap() {
                let path = entry.unwrap().path();
                if path.is_dir() {
                    sources(&path, out);
                } else if path.extension().is_some_and(|e| e == "rs")
                    && !path.ends_with("errors.rs")
                {
                    out.push_str(&std::fs::read_to_string(&path).unwrap());
                }
            }
        }
        let mut text = String::new();
        sources(
            &std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src"),
            &mut text,
        );
        for code in Code::ALL {
            assert!(!code.next_step().is_empty(), "{code:?}");
            let name = format!("Code::{code:?}");
            assert!(
                text.match_indices(&name).any(|(at, _)| {
                    !text[at + name.len()..].starts_with(|c: char| c.is_alphanumeric())
                }),
                "{} ({code:?}) is never emitted",
                code.id()
            );
        }
        let bare = CliError::new(Code::ModelCorrupt, "bad blob");
        assert!(bare.to_string().ends_with(Code::ModelCorrupt.next_step()));
        assert_eq!(CliError::from(bare.to_string()), bare);
    }

    #[test]
    fn display_has_code_message_and_hint() {
        let error = CliError::new(Code::ModelNotFound, "model 'qwen3.7' not found")
            .hint("did you mean 'qwen3.6'? Run 'mrun ls' to see local models.");
        assert_eq!(
            error.to_string(),
            "error[E031]: model 'qwen3.7' not found\n  hint: did you mean 'qwen3.6'? Run 'mrun ls' to see local models."
        );
    }
}
