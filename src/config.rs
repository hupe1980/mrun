use std::fmt;
use std::time::Duration;

use serde::Serialize;
use sha2::{Digest, Sha256};

const KIB: u64 = 1024;
const MIB: u64 = 1024 * KIB;
const GIB: u64 = 1024 * MIB;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(transparent)]
pub struct Bytes(u64);

impl Bytes {
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    pub const fn get(self) -> u64 {
        self.0
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(transparent)]
pub struct Tokens(u32);

impl Tokens {
    pub const fn new(value: u32) -> Self {
        Self(value)
    }

    pub const fn get(self) -> u32 {
        self.0
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct ResourceProfile {
    pub version: u32,
    pub total_context: Tokens,
    pub max_output: Tokens,
    pub active_requests: u8,
    pub queued_requests: u8,
    pub queue_deadline: Duration,
    pub stalled_write_deadline: Duration,
    pub waiting_bodies: Bytes,
    pub body: Bytes,
    pub headers: Bytes,
    pub connections: u16,
    pub header_deadline: Duration,
    pub body_deadline: Duration,
    pub messages: u16,
    pub tools: u16,
    pub schema_depth: u16,
    pub json_depth: u16,
    pub per_schema: Bytes,
    pub all_schemas: Bytes,
    pub schema_nodes: u32,
    pub render_temporaries: Bytes,
    pub grammar_storage: Bytes,
    pub generated_payload: Bytes,
    pub streaming_queue: Bytes,
    pub bridge_slots: u16,
    pub stop_strings: u8,
    pub stop_string: Bytes,
    pub physical_kv_cells: Tokens,
    pub sequence_ids: u8,
    pub retained_entries: u8,
    pub retained_cells: Tokens,
    pub retained_metadata: Bytes,
    pub prefill_chunk: Tokens,
    pub decode_chunk: Tokens,
    pub diagnostics: Bytes,
    pub host_reserve: Bytes,
    pub automatic_physical_percent: u8,
    pub minimum_scratch: Bytes,
    pub scratch_backing_percent: u8,
}

pub const QUALIFIED_PROFILE: ResourceProfile = ResourceProfile {
    version: 1,
    total_context: Tokens::new(16_384),
    max_output: Tokens::new(2_048),
    active_requests: 1,
    queued_requests: 8,
    queue_deadline: Duration::from_secs(30),
    stalled_write_deadline: Duration::from_secs(10),
    waiting_bodies: Bytes::new(16 * MIB),
    body: Bytes::new(2 * MIB),
    headers: Bytes::new(32 * KIB),
    connections: 16,
    header_deadline: Duration::from_secs(5),
    body_deadline: Duration::from_secs(30),
    messages: 512,
    tools: 64,
    schema_depth: 8,
    json_depth: 32,
    per_schema: Bytes::new(32 * KIB),
    all_schemas: Bytes::new(256 * KIB),
    schema_nodes: 4_096,
    render_temporaries: Bytes::new(64 * MIB),
    grammar_storage: Bytes::new(16 * MIB),
    generated_payload: Bytes::new(256 * KIB),
    streaming_queue: Bytes::new(64 * KIB),
    bridge_slots: 16,
    stop_strings: 4,
    stop_string: Bytes::new(256),
    physical_kv_cells: Tokens::new(32_768),
    sequence_ids: 5,
    retained_entries: 4,
    retained_cells: Tokens::new(16_384),
    retained_metadata: Bytes::new(4 * MIB),
    prefill_chunk: Tokens::new(128),
    decode_chunk: Tokens::new(1),
    diagnostics: Bytes::new(MIB),
    host_reserve: Bytes::new(8 * GIB),
    automatic_physical_percent: 75,
    minimum_scratch: Bytes::new(2 * GIB),
    scratch_backing_percent: 25,
};

impl ResourceProfile {
    pub fn identity(&self) -> Result<String, serde_json::Error> {
        let encoded = serde_json::to_vec(self)?;
        Ok(format!("{:x}", Sha256::digest(encoded)))
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Overrides {
    pub context_tokens: Option<u32>,
    pub output_tokens: Option<u32>,
    pub cache_entries: Option<u8>,
    pub cache_cells: Option<u32>,
    pub idle_seconds: Option<u64>,
    pub budget_bytes: Option<u64>,
    pub allow_budget_increase: bool,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct RuntimeConfig {
    context: Tokens,
    output: Tokens,
    cache_entries: u8,
    cache_cells: Tokens,
    idle_timeout: Option<Duration>,
    budget: Option<Bytes>,
    allow_budget_increase: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ConfigError {
    Context { minimum: u32, maximum: u32 },
    Output { maximum: u32 },
    CacheEntries { maximum: u8 },
    CacheCells { maximum: u32 },
    ZeroBudget,
}

impl fmt::Display for ConfigError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Context { minimum, maximum } => {
                write!(formatter, "context tokens must be {minimum}..={maximum}")
            }
            Self::Output { maximum } => {
                write!(formatter, "output tokens must be 1..={maximum}")
            }
            Self::CacheEntries { maximum } => {
                write!(formatter, "cache entries must be 0..={maximum}")
            }
            Self::CacheCells { maximum } => {
                write!(formatter, "cache cells must be 0..={maximum}")
            }
            Self::ZeroBudget => formatter.write_str("budget bytes must be positive"),
        }
    }
}

impl std::error::Error for ConfigError {}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            context: QUALIFIED_PROFILE.total_context,
            output: QUALIFIED_PROFILE.max_output,
            cache_entries: QUALIFIED_PROFILE.retained_entries,
            cache_cells: QUALIFIED_PROFILE.retained_cells,
            idle_timeout: None,
            budget: None,
            allow_budget_increase: false,
        }
    }
}

impl RuntimeConfig {
    pub fn from_overrides(overrides: Overrides) -> Result<Self, ConfigError> {
        let context = overrides
            .context_tokens
            .unwrap_or(QUALIFIED_PROFILE.total_context.get());
        if !(256..=QUALIFIED_PROFILE.total_context.get()).contains(&context) {
            return Err(ConfigError::Context {
                minimum: 256,
                maximum: QUALIFIED_PROFILE.total_context.get(),
            });
        }
        let output = overrides
            .output_tokens
            .unwrap_or(QUALIFIED_PROFILE.max_output.get());
        let maximum_output = context.min(QUALIFIED_PROFILE.max_output.get());
        if !(1..=maximum_output).contains(&output) {
            return Err(ConfigError::Output {
                maximum: maximum_output,
            });
        }
        let cache_entries = overrides
            .cache_entries
            .unwrap_or(QUALIFIED_PROFILE.retained_entries);
        if cache_entries > QUALIFIED_PROFILE.retained_entries {
            return Err(ConfigError::CacheEntries {
                maximum: QUALIFIED_PROFILE.retained_entries,
            });
        }
        let cache_cells = overrides
            .cache_cells
            .unwrap_or(QUALIFIED_PROFILE.retained_cells.get());
        if cache_cells > QUALIFIED_PROFILE.retained_cells.get() {
            return Err(ConfigError::CacheCells {
                maximum: QUALIFIED_PROFILE.retained_cells.get(),
            });
        }
        if overrides.budget_bytes == Some(0) {
            return Err(ConfigError::ZeroBudget);
        }
        Ok(Self {
            context: Tokens::new(context),
            output: Tokens::new(output),
            cache_entries,
            cache_cells: Tokens::new(cache_cells),
            idle_timeout: overrides
                .idle_seconds
                .filter(|seconds| *seconds > 0)
                .map(Duration::from_secs),
            budget: overrides.budget_bytes.map(Bytes::new),
            allow_budget_increase: overrides.allow_budget_increase,
        })
    }

    pub const fn context(&self) -> Tokens {
        self.context
    }

    pub const fn output(&self) -> Tokens {
        self.output
    }

    pub const fn cache_entries(&self) -> u8 {
        self.cache_entries
    }

    pub const fn cache_cells(&self) -> Tokens {
        self.cache_cells
    }

    pub const fn idle_timeout(&self) -> Option<Duration> {
        self.idle_timeout
    }

    pub const fn budget(&self) -> Option<Bytes> {
        self.budget
    }

    pub const fn allows_budget_increase(&self) -> bool {
        self.allow_budget_increase
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_preserve_request_and_physical_geometry() {
        let config = RuntimeConfig::from_overrides(Overrides::default()).unwrap();
        assert_eq!(config.context().get(), 16_384);
        assert_eq!(config.output().get(), 2_048);
        assert_eq!(QUALIFIED_PROFILE.physical_kv_cells.get(), 32_768);
        assert_eq!(QUALIFIED_PROFILE.sequence_ids, 5);
        assert_eq!(config.cache_entries(), 4);
        assert_eq!(config.cache_cells().get(), 16_384);
        assert_eq!(config.idle_timeout(), None);
        assert_eq!(config.budget(), None);
        assert!(!config.allows_budget_increase());
    }

    #[test]
    fn lower_limits_and_disabled_retention_are_preserved() {
        let config = RuntimeConfig::from_overrides(Overrides {
            context_tokens: Some(256),
            output_tokens: Some(256),
            cache_entries: Some(0),
            cache_cells: Some(0),
            idle_seconds: Some(12),
            budget_bytes: Some(1024),
            ..Overrides::default()
        })
        .unwrap();
        assert_eq!(config.context().get(), 256);
        assert_eq!(config.output().get(), 256);
        assert_eq!(config.cache_entries(), 0);
        assert_eq!(config.cache_cells().get(), 0);
        assert_eq!(config.idle_timeout(), Some(Duration::from_secs(12)));
        assert_eq!(config.budget(), Some(Bytes::new(1024)));
    }

    #[test]
    fn rejects_limits_outside_qualified_envelope() {
        for value in [0, 255, 16_385, u32::MAX] {
            assert!(
                RuntimeConfig::from_overrides(Overrides {
                    context_tokens: Some(value),
                    ..Overrides::default()
                })
                .is_err()
            );
        }
        for value in [0, 2_049, u32::MAX] {
            assert!(
                RuntimeConfig::from_overrides(Overrides {
                    output_tokens: Some(value),
                    ..Overrides::default()
                })
                .is_err()
            );
        }
        assert!(
            RuntimeConfig::from_overrides(Overrides {
                context_tokens: Some(256),
                output_tokens: Some(257),
                ..Overrides::default()
            })
            .is_err()
        );
        assert!(
            RuntimeConfig::from_overrides(Overrides {
                cache_entries: Some(5),
                ..Overrides::default()
            })
            .is_err()
        );
        assert!(
            RuntimeConfig::from_overrides(Overrides {
                cache_cells: Some(16_385),
                ..Overrides::default()
            })
            .is_err()
        );
        assert!(
            RuntimeConfig::from_overrides(Overrides {
                budget_bytes: Some(0),
                ..Overrides::default()
            })
            .is_err()
        );
    }

    #[test]
    fn profile_identity_binds_frozen_limits() {
        let digest = QUALIFIED_PROFILE.identity().unwrap();
        assert_eq!(digest.len(), 64);
        assert_eq!(digest, QUALIFIED_PROFILE.identity().unwrap());
        let mut changed = QUALIFIED_PROFILE;
        changed.body = Bytes::new(changed.body.get() + 1);
        assert_ne!(digest, changed.identity().unwrap());
    }
}
