//! User-facing errors: a stable code, a message and the next step, with the CLI exit codes of the
//! 004 contract.
//!
//! ```text
//! error[E031]: model 'qwen3-codr' not found
//!   hint: did you mean 'qwen3-coder'? Run 'mrun ls' to see local models.
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
    /// Reserved: bench regression mode (Later).
    Regression = 7,
    DoctorFailures = 8,
}

/// Error codes of the catalog. Numbers are stable once released.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Code {
    /// E001 unexpected internal error.
    Internal,
    /// E010 the server is not running or not reachable.
    ServerUnreachable,
    /// E011 the port is taken by another program.
    PortInUse,
    /// E020 the client program is not installed.
    ClientMissing,
    /// E021 the client is older than mrun supports.
    ClientTooOld,
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
            Self::ServerUnreachable => "E010",
            Self::PortInUse => "E011",
            Self::ClientMissing => "E020",
            Self::ClientTooOld => "E021",
            Self::ModelNotFound => "E031",
            Self::ModelCorrupt => "E032",
            Self::ModelAmbiguous => "E033",
            Self::UnsupportedArchitecture => "E034",
            Self::DoesNotFit => "E040",
            Self::Network => "E050",
            Self::Gated => "E051",
            Self::DiskFull => "E052",
        }
    }

    pub fn exit(self) -> Exit {
        match self {
            Self::Internal => Exit::Unexpected,
            Self::ServerUnreachable => Exit::ServerUnreachable,
            Self::PortInUse | Self::ClientMissing | Self::ClientTooOld => Exit::Unexpected,
            Self::ModelNotFound
            | Self::ModelCorrupt
            | Self::ModelAmbiguous
            | Self::UnsupportedArchitecture => Exit::Model,
            Self::DoesNotFit => Exit::Fit,
            Self::Network | Self::Gated | Self::DiskFull => Exit::Network,
        }
    }

    /// Every code, for the catalog snapshot.
    pub const ALL: [Code; 13] = [
        Self::Internal,
        Self::ServerUnreachable,
        Self::PortInUse,
        Self::ClientMissing,
        Self::ClientTooOld,
        Self::ModelNotFound,
        Self::ModelCorrupt,
        Self::ModelAmbiguous,
        Self::UnsupportedArchitecture,
        Self::DoesNotFit,
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
        if let Some(hint) = &self.hint {
            write!(f, "\n  hint: {hint}")?;
        }
        Ok(())
    }
}

impl std::error::Error for CliError {}

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
    }

    #[test]
    fn display_has_code_message_and_hint() {
        let error = CliError::new(Code::ModelNotFound, "model 'qwen3-codr' not found")
            .hint("did you mean 'qwen3-coder'? Run 'mrun ls' to see local models.");
        assert_eq!(
            error.to_string(),
            "error[E031]: model 'qwen3-codr' not found\n  hint: did you mean 'qwen3-coder'? Run 'mrun ls' to see local models."
        );
    }
}
