use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Subject {
    pub runtime_sha256: Option<String>,
    pub backend: String,
    pub model: String,
    pub profile: String,
    pub render: String,
    pub build: String,
    pub fixtures_revision: String,
    pub fixtures_sha256: String,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Eligibility {
    pub cache_off: bool,
    pub reuse: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReportError {
    Unavailable,
    Bounds,
    Syntax,
    Version,
    Stale,
    Incomplete,
    Hash,
    Draft,
}

pub fn validate(
    _bytes: &[u8],
    _current: &Subject,
    _read_evidence: impl FnMut(&str) -> Result<Vec<u8>, ReportError>,
) -> Result<Eligibility, ReportError> {
    Err(ReportError::Unavailable)
}
