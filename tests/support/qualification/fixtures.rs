use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Component, Path};

const MAX_MANIFEST_BYTES: usize = 1024 * 1024;
const MAX_SOURCE_BYTES: usize = 1024 * 1024;
const MAX_RENDERED_BYTES: usize = 64 * 1024 * 1024;

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum Stage {
    Draft,
    Source,
    Rendered,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Purpose {
    PrimitiveConstruction,
    DependentEvidence,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Manifest {
    pub schema_version: u32,
    pub revision: String,
    pub stage: Stage,
    pub catalogs: Vec<Catalog>,
    pub assets: Vec<Asset>,
    pub rendered: Option<Rendered>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Catalog {
    pub kind: String,
    pub path: String,
    pub sha256: String,
    pub count: usize,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Asset {
    pub path: String,
    pub sha256: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Identity {
    pub model: String,
    pub tokenizer: String,
    pub template: String,
    pub renderer: String,
    pub profile: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Rendered {
    pub identity: Identity,
    pub path: String,
    pub sha256: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct TokenRecord {
    id: String,
    source_sha256: String,
    tokens: Vec<u32>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct TokenFile {
    schema_version: u32,
    source_revision: String,
    identity: Identity,
    records: Vec<TokenRecord>,
}

#[derive(Debug, Eq, PartialEq)]
pub enum FixtureError {
    Bounds,
    Syntax,
    Version,
    Path,
    Hash,
    Counts,
    License,
    Draft,
    Rendered,
    Missing,
}

pub fn expand_replay_turn(
    turn: &serde_json::Value,
    case: &serde_json::Value,
) -> Result<serde_json::Value, FixtureError> {
    let mut expanded = turn.clone();
    expand_replay_value(&mut expanded, case)?;
    Ok(expanded)
}

fn expand_replay_value(
    value: &mut serde_json::Value,
    case: &serde_json::Value,
) -> Result<(), FixtureError> {
    match value {
        serde_json::Value::Object(fields) => {
            for field in fields.values_mut() {
                expand_replay_value(field, case)?;
            }
        }
        serde_json::Value::Array(values) => {
            for item in values {
                expand_replay_value(item, case)?;
            }
        }
        serde_json::Value::String(text) => {
            for key in ["path", "source", "fixed", "issue", "check"] {
                let replacement = case[key].as_str().ok_or(FixtureError::Syntax)?;
                *text = text.replace(&format!("${key}"), replacement);
            }
        }
        _ => {}
    }
    Ok(())
}

pub fn load(
    bytes: &[u8],
    purpose: Purpose,
    mut read: impl FnMut(&str) -> Result<Vec<u8>, FixtureError>,
) -> Result<Manifest, FixtureError> {
    if bytes.len() > MAX_MANIFEST_BYTES {
        return Err(FixtureError::Bounds);
    }
    let manifest: Manifest = serde_json::from_slice(bytes).map_err(|_| FixtureError::Syntax)?;
    if manifest.schema_version != 1 {
        return Err(FixtureError::Version);
    }
    if manifest.revision.is_empty() || manifest.revision.len() > 128 {
        return Err(FixtureError::Bounds);
    }
    if purpose == Purpose::DependentEvidence && manifest.stage != Stage::Rendered {
        return Err(FixtureError::Draft);
    }
    let expected = [
        ("interoperability", "IC", 20),
        ("state", "", 100),
        ("coding", "C", 20),
        ("replay", "R", 10),
    ];
    if manifest.catalogs.len() != expected.len() || manifest.assets.len() > 1024 {
        return Err(FixtureError::Counts);
    }
    let mut all_ids = BTreeSet::new();
    let mut paths = BTreeSet::new();
    let mut source_hashes = BTreeMap::new();
    for (catalog, (kind, prefix, count)) in manifest.catalogs.iter().zip(expected) {
        if catalog.kind != kind || catalog.count != count || !paths.insert(catalog.path.clone()) {
            return Err(FixtureError::Counts);
        }
        let data = checked_read(&catalog.path, &catalog.sha256, MAX_SOURCE_BYTES, &mut read)?;
        let value: serde_json::Value =
            serde_json::from_slice(&data).map_err(|_| FixtureError::Syntax)?;
        if value["schema_version"] != 1 {
            return Err(FixtureError::Version);
        }
        if value["license"] != "MIT OR Apache-2.0" {
            return Err(FixtureError::License);
        }
        reject_token_arrays(&value)?;
        let cases = value["cases"].as_array().ok_or(FixtureError::Counts)?;
        let ids: Vec<_> = cases
            .iter()
            .map(|case| case["id"].as_str().ok_or(FixtureError::Counts))
            .collect::<Result<_, _>>()?;
        let expected_ids: Vec<_> = if kind == "state" {
            ["P", "B", "I", "S", "F"]
                .into_iter()
                .flat_map(|category| (1..=20).map(move |index| format!("{category}-{index:03}")))
                .collect()
        } else {
            (1..=count)
                .map(|index| format!("{prefix}-{index:03}"))
                .collect()
        };
        if ids != expected_ids {
            return Err(FixtureError::Counts);
        }
        let mut controls = BTreeSet::new();
        for case in cases {
            let mut control = case.as_object().ok_or(FixtureError::Counts)?.clone();
            control.remove("id");
            if !controls.insert(serde_json::to_string(&control).map_err(|_| FixtureError::Syntax)?)
            {
                return Err(FixtureError::Counts);
            }
        }
        for id in expected_ids {
            source_hashes.insert(id.clone(), catalog.sha256.clone());
            all_ids.insert(id);
        }
    }
    for asset in &manifest.assets {
        if !paths.insert(asset.path.clone()) {
            return Err(FixtureError::Path);
        }
        checked_read(&asset.path, &asset.sha256, MAX_SOURCE_BYTES, &mut read)?;
    }
    match (manifest.stage, &manifest.rendered) {
        (Stage::Draft | Stage::Source, None) => {}
        (Stage::Rendered, Some(rendered)) => {
            if !paths.insert(rendered.path.clone()) {
                return Err(FixtureError::Path);
            }
            for hash in [
                &rendered.identity.model,
                &rendered.identity.tokenizer,
                &rendered.identity.template,
                &rendered.identity.renderer,
                &rendered.identity.profile,
            ] {
                valid_hash(hash)?;
            }
            let data = checked_read(
                &rendered.path,
                &rendered.sha256,
                MAX_RENDERED_BYTES,
                &mut read,
            )?;
            let file: TokenFile =
                serde_json::from_slice(&data).map_err(|_| FixtureError::Syntax)?;
            if file.schema_version != 1
                || file.source_revision != manifest.revision
                || file.identity != rendered.identity
                || file.records.len() != all_ids.len()
            {
                return Err(FixtureError::Rendered);
            }
            for record in file.records {
                if !all_ids.remove(&record.id)
                    || record.tokens.is_empty()
                    || record.tokens.len() > 16_384
                    || source_hashes.get(&record.id) != Some(&record.source_sha256)
                {
                    return Err(FixtureError::Rendered);
                }
            }
            if !all_ids.is_empty() {
                return Err(FixtureError::Rendered);
            }
        }
        _ => return Err(FixtureError::Rendered),
    }
    Ok(manifest)
}

fn checked_read(
    path: &str,
    hash: &str,
    maximum: usize,
    read: &mut impl FnMut(&str) -> Result<Vec<u8>, FixtureError>,
) -> Result<Vec<u8>, FixtureError> {
    if path.is_empty()
        || path.len() > 4096
        || Path::new(path)
            .components()
            .any(|part| !matches!(part, Component::Normal(_)))
    {
        return Err(FixtureError::Path);
    }
    valid_hash(hash)?;
    let bytes = read(path)?;
    if bytes.len() > maximum {
        return Err(FixtureError::Bounds);
    }
    if format!("{:x}", Sha256::digest(&bytes)) != hash {
        return Err(FixtureError::Hash);
    }
    Ok(bytes)
}

fn valid_hash(hash: &str) -> Result<(), FixtureError> {
    if hash.len() != 64
        || !hash
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(FixtureError::Hash);
    }
    Ok(())
}

fn reject_token_arrays(value: &serde_json::Value) -> Result<(), FixtureError> {
    match value {
        serde_json::Value::Object(fields) => {
            for (key, value) in fields {
                if matches!(key.as_str(), "rendered_tokens" | "token_ids") && !value.is_null() {
                    return Err(FixtureError::Draft);
                }
                reject_token_arrays(value)?;
            }
        }
        serde_json::Value::Array(values) => {
            for value in values {
                reject_token_arrays(value)?;
            }
        }
        _ => {}
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn input(path: &str) -> Result<Vec<u8>, FixtureError> {
        let bytes: &[u8] = match path {
            "interoperability/cases.json" => {
                include_bytes!("../../fixtures/interoperability/cases.json")
            }
            "state/cases.json" => include_bytes!("../../fixtures/state/cases.json"),
            "coding/cases.json" => include_bytes!("../../fixtures/coding/cases.json"),
            "replay/cases.json" => include_bytes!("../../fixtures/replay/cases.json"),
            "interoperability/workspace/Cargo.toml" => {
                include_bytes!("../../fixtures/interoperability/workspace/Cargo.toml")
            }
            "interoperability/workspace/src/lib.rs" => {
                include_bytes!("../../fixtures/interoperability/workspace/src/lib.rs")
            }
            "interoperability/workspace/tests/supplied.rs" => {
                include_bytes!("../../fixtures/interoperability/workspace/tests/supplied.rs")
            }
            _ => return Err(FixtureError::Missing),
        };
        Ok(bytes.to_vec())
    }

    #[test]
    fn primitive_probes_can_use_complete_verified_drafts() {
        let manifest = load(
            include_bytes!("../../fixtures/manifest.json"),
            Purpose::PrimitiveConstruction,
            input,
        )
        .unwrap();
        assert_eq!(manifest.stage, Stage::Draft);
    }

    #[test]
    fn dependent_evidence_rejects_draft_inputs() {
        assert_eq!(
            load(
                include_bytes!("../../fixtures/manifest.json"),
                Purpose::DependentEvidence,
                input,
            )
            .unwrap_err(),
            FixtureError::Draft
        );
    }

    #[test]
    fn missing_and_changed_sources_are_rejected() {
        let bytes = include_bytes!("../../fixtures/manifest.json");
        assert_eq!(
            load(bytes, Purpose::PrimitiveConstruction, |_| Err(
                FixtureError::Missing
            ))
            .unwrap_err(),
            FixtureError::Missing
        );
        assert_eq!(
            load(
                bytes,
                Purpose::PrimitiveConstruction,
                |_| Ok(b"{}".to_vec())
            )
            .unwrap_err(),
            FixtureError::Hash
        );
    }

    fn synthetic_rendered() -> (serde_json::Value, serde_json::Value) {
        let mut manifest: serde_json::Value =
            serde_json::from_slice(include_bytes!("../../fixtures/manifest.json")).unwrap();
        manifest["stage"] = serde_json::json!("rendered");
        let identity = serde_json::json!({
            "model": "a".repeat(64), "tokenizer": "b".repeat(64),
            "template": "c".repeat(64), "renderer": "d".repeat(64),
            "profile": "e".repeat(64)
        });
        let mut records = Vec::new();
        for catalog in manifest["catalogs"].as_array().unwrap() {
            let data: serde_json::Value =
                serde_json::from_slice(&input(catalog["path"].as_str().unwrap()).unwrap()).unwrap();
            for case in data["cases"].as_array().unwrap() {
                records.push(serde_json::json!({
                    "id": case["id"], "source_sha256": catalog["sha256"], "tokens": [1]
                }));
            }
        }
        let tokens = serde_json::json!({
            "schema_version": 1, "source_revision": manifest["revision"],
            "identity": identity, "records": records
        });
        manifest["rendered"] = serde_json::json!({
            "identity": identity, "path": "synthetic-test-only.json", "sha256": ""
        });
        (manifest, tokens)
    }

    fn load_synthetic(
        mut manifest: serde_json::Value,
        tokens: serde_json::Value,
    ) -> Result<Manifest, FixtureError> {
        let bytes = serde_json::to_vec(&tokens).unwrap();
        manifest["rendered"]["sha256"] = serde_json::json!(format!("{:x}", Sha256::digest(&bytes)));
        load(
            &serde_json::to_vec(&manifest).unwrap(),
            Purpose::DependentEvidence,
            |path| {
                if path == "synthetic-test-only.json" {
                    Ok(bytes.clone())
                } else {
                    input(path)
                }
            },
        )
    }

    #[test]
    fn rendered_format_checks_complete_case_and_identity_bindings() {
        let (manifest, tokens) = synthetic_rendered();
        assert_eq!(
            load_synthetic(manifest.clone(), tokens.clone())
                .unwrap()
                .stage,
            Stage::Rendered
        );
        let mut changed = tokens.clone();
        changed["identity"]["model"] = serde_json::json!("f".repeat(64));
        assert_eq!(
            load_synthetic(manifest.clone(), changed).unwrap_err(),
            FixtureError::Rendered
        );
        let mut changed = tokens.clone();
        changed["records"].as_array_mut().unwrap().pop();
        assert_eq!(
            load_synthetic(manifest.clone(), changed).unwrap_err(),
            FixtureError::Rendered
        );
        let mut changed = tokens;
        changed["records"][0]["source_sha256"] = manifest["catalogs"][1]["sha256"].clone();
        assert_eq!(
            load_synthetic(manifest, changed).unwrap_err(),
            FixtureError::Rendered
        );
    }

    #[test]
    fn schema_and_source_only_eligibility_fail_closed() {
        let mut manifest: serde_json::Value =
            serde_json::from_slice(include_bytes!("../../fixtures/manifest.json")).unwrap();
        manifest["stage"] = serde_json::json!("source");
        assert!(
            load(
                &serde_json::to_vec(&manifest).unwrap(),
                Purpose::PrimitiveConstruction,
                input
            )
            .is_ok()
        );
        assert_eq!(
            load(
                &serde_json::to_vec(&manifest).unwrap(),
                Purpose::DependentEvidence,
                input
            )
            .unwrap_err(),
            FixtureError::Draft
        );
        manifest["schema_version"] = serde_json::json!(2);
        assert_eq!(
            load(
                &serde_json::to_vec(&manifest).unwrap(),
                Purpose::PrimitiveConstruction,
                input
            )
            .unwrap_err(),
            FixtureError::Version
        );
        let duplicated = String::from_utf8(include_bytes!("../../fixtures/manifest.json").to_vec())
            .unwrap()
            .replacen(
                "\"schema_version\": 1",
                "\"schema_version\": 1, \"schema_version\": 1",
                1,
            );
        assert_eq!(
            load(duplicated.as_bytes(), Purpose::PrimitiveConstruction, input).unwrap_err(),
            FixtureError::Syntax
        );
    }
}
