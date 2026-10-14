use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub const MODEL_ID: &str = "qwen3-8b-q4-k-m";
pub const MODEL_SHA256: &str = "d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ArtifactProfile {
    pub repository: &'static str,
    pub revision: &'static str,
    pub filename: &'static str,
    pub bytes: u64,
    pub sha256: &'static str,
    pub license: &'static str,
}

pub const SELECTED_ARTIFACT: ArtifactProfile = ArtifactProfile {
    repository: "Qwen/Qwen3-8B-GGUF",
    revision: "7c41481f57cb95916b40956ab2f0b139b296d974",
    filename: "Qwen3-8B-Q4_K_M.gguf",
    bytes: 5_027_783_488,
    sha256: MODEL_SHA256,
    license: "Apache-2.0",
};

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct StateIdentity {
    pub weight_sha256: String,
    pub artifact_revision: String,
    pub tokenizer_sha256: String,
    pub template_sha256: String,
    pub render_controls_sha256: String,
    pub architecture: String,
    pub state_abi: String,
    pub backend_commit: String,
    pub build_sha256: String,
    pub kv_precision: String,
    pub kv_layout: String,
    pub physical_cells: u32,
    pub sequence_ids: u32,
    pub position_origin: u32,
    pub rope_from_artifact: bool,
    pub rope_metadata_sha256: String,
    pub adapters: Vec<String>,
    pub pool_epoch: u64,
}

impl StateIdentity {
    pub fn digest(&self) -> Result<String, serde_json::Error> {
        Ok(format!("{:x}", Sha256::digest(serde_json::to_vec(self)?)))
    }

    pub fn is_selected_profile(&self) -> bool {
        false
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Capabilities {
    full_attention: bool,
    text_only: bool,
    non_thinking: bool,
    single_tool_call: bool,
    schema_subset: &'static str,
    context_tokens: u32,
    output_tokens: u32,
}

pub struct QualifiedModel {
    identity: StateIdentity,
    capabilities: Capabilities,
}

impl QualifiedModel {
    pub fn model_id(&self) -> &'static str {
        MODEL_ID
    }
    pub fn identity(&self) -> &StateIdentity {
        &self.identity
    }
    pub fn capabilities(&self) -> &Capabilities {
        &self.capabilities
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn identity() -> StateIdentity {
        StateIdentity {
            weight_sha256: MODEL_SHA256.into(),
            artifact_revision: SELECTED_ARTIFACT.revision.into(),
            tokenizer_sha256: "a".repeat(64),
            template_sha256: "b".repeat(64),
            render_controls_sha256: "c".repeat(64),
            architecture: "qwen3".into(),
            state_abi: "unqualified-state".into(),
            backend_commit: "unqualified-backend".into(),
            build_sha256: "d".repeat(64),
            kv_precision: "unqualified".into(),
            kv_layout: "unqualified".into(),
            physical_cells: 0,
            sequence_ids: 0,
            position_origin: 0,
            rope_from_artifact: true,
            rope_metadata_sha256: "e".repeat(64),
            adapters: vec![],
            pool_epoch: 1,
        }
    }

    #[test]
    fn artifact_is_pinned_without_disclosing_qualification() {
        assert_eq!(SELECTED_ARTIFACT.bytes, 5_027_783_488);
        assert_eq!(SELECTED_ARTIFACT.sha256, MODEL_SHA256);
        assert_eq!(MODEL_ID, "qwen3-8b-q4-k-m");
        assert!(!identity().is_selected_profile());
    }

    #[test]
    fn every_state_identity_field_is_bound_and_full_identity_is_comparable() {
        let original = identity();
        let hash = original.digest().unwrap();
        let encoded = serde_json::to_value(&original).unwrap();
        for (name, value) in encoded.as_object().unwrap() {
            let mut changed = encoded.clone();
            changed[name] = match value {
                serde_json::Value::String(_) => serde_json::json!("changed"),
                serde_json::Value::Bool(value) => serde_json::json!(!value),
                serde_json::Value::Number(value) => serde_json::json!(value.as_u64().unwrap() + 1),
                serde_json::Value::Array(_) => serde_json::json!(["unsupported-adapter"]),
                _ => unreachable!(),
            };
            let decoded: StateIdentity = serde_json::from_value(changed).unwrap();
            assert_ne!(decoded, original, "{name}");
            assert_ne!(decoded.digest().unwrap(), hash, "{name}");
        }
    }

    #[test]
    fn unsupported_geometry_adapters_and_position_profiles_are_rejected() {
        let mut changed = identity();
        changed.adapters.push("unsupported".into());
        assert!(!changed.is_selected_profile());
        changed = identity();
        changed.position_origin = 1;
        assert!(!changed.is_selected_profile());
        changed = identity();
        changed.physical_cells = 16_384;
        assert!(!changed.is_selected_profile());
        changed = identity();
        changed.rope_from_artifact = false;
        assert!(!changed.is_selected_profile());
    }
}
