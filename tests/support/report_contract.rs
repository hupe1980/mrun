use crate::backend::qualification::{self, Eligibility, ReportError, Subject};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;

fn subject() -> Subject {
    Subject {
        runtime_sha256: Some("a".repeat(64)),
        backend: "b".repeat(64),
        model: "c".repeat(64),
        profile: "d".repeat(64),
        render: "e".repeat(64),
        build: "f".repeat(64),
        fixtures_revision: "synthetic-report-tests-only-1".into(),
        fixtures_sha256: "0".repeat(64),
    }
}

fn lawful(reuse: bool) -> (Value, BTreeMap<String, Vec<u8>>) {
    let harness = json!({"sha256": "1".repeat(64), "build": "2".repeat(64), "run_id": "synthetic-run-1", "target_runtime_sha256": "a".repeat(64)});
    let mut report = json!({
        "schema_version": 1,
        "stage": if reuse { "candidate_reuse" } else { "candidate_capability" },
        "subject": subject(), "harness": harness, "fixture_stage": "rendered", "records": []
    });
    let mut artifacts = BTreeMap::new();
    let mut gates = vec![(
        "capability",
        vec![
            "load",
            "geometry",
            "render",
            "grammar",
            "accounting",
            "fault",
            "cancel",
            "unload",
            "admission",
            "pressure",
            "transport",
        ]
        .into_iter()
        .map(String::from)
        .collect::<Vec<_>>(),
    )];
    if reuse {
        gates.push((
            "state",
            ["P", "B", "I", "S", "F"]
                .into_iter()
                .flat_map(|prefix| (1..=20).map(move |number| format!("{prefix}-{number:03}")))
                .collect(),
        ));
        gates.push((
            "numerics",
            (1..=20).map(|number| format!("P-{number:03}")).collect(),
        ));
        gates.push((
            "protocol",
            (1..=20).map(|number| format!("IC-{number:03}")).collect(),
        ));
        gates.push((
            "quality",
            (1..=20)
                .flat_map(|number| [7, 19, 41].map(move |seed| format!("C-{number:03}@{seed}")))
                .collect(),
        ));
    }
    for (gate, cases) in gates {
        let name = format!("{gate}.json");
        let artifact = serde_json::to_vec(&json!({
            "schema_version": 1, "subject": subject(), "harness": harness,
            "gate": gate, "disposition": "pass", "cases": cases,
            "details": {"complete": true, "exact_positive_cases": 20,
                "numerical_positions_pass": true, "paired_outcomes": 60,
                "lost_successes": 0, "systematic_new_failure": false, "all_completed_tools_valid": true}
        })).unwrap();
        report["records"].as_array_mut().unwrap().push(json!({
            "gate": gate, "disposition": "pass", "path": name,
            "sha256": format!("{:x}", Sha256::digest(&artifact))
        }));
        artifacts.insert(name, artifact);
    }
    (report, artifacts)
}

fn check(
    report: &Value,
    artifacts: &BTreeMap<String, Vec<u8>>,
) -> Result<Eligibility, ReportError> {
    qualification::validate(&serde_json::to_vec(report).unwrap(), &subject(), |path| {
        artifacts.get(path).cloned().ok_or(ReportError::Incomplete)
    })
}

#[test]
fn complete_capability_and_reuse_records_have_distinct_eligibility() {
    let (report, artifacts) = lawful(false);
    assert_eq!(
        check(&report, &artifacts),
        Ok(Eligibility {
            cache_off: true,
            reuse: false
        })
    );
    let (report, artifacts) = lawful(true);
    assert_eq!(
        check(&report, &artifacts),
        Ok(Eligibility {
            cache_off: true,
            reuse: true
        })
    );
}

#[test]
fn adapter_only_report_never_authorizes_serving() {
    let (mut report, artifacts) = lawful(false);
    report["stage"] = json!("adapter_spike");
    report["subject"]["runtime_sha256"] = Value::Null;
    assert!(check(&report, &artifacts).is_err());
}

#[test]
fn protocol_or_quality_omission_blocks_reuse() {
    for missing in ["protocol", "quality"] {
        let (mut report, artifacts) = lawful(true);
        report["records"]
            .as_array_mut()
            .unwrap()
            .retain(|record| record["gate"] != missing);
        assert_eq!(check(&report, &artifacts), Err(ReportError::Incomplete));
    }
}

#[test]
fn replaced_binary_or_changed_fixture_subject_is_stale() {
    for field in [
        "runtime_sha256",
        "backend",
        "model",
        "profile",
        "render",
        "build",
        "fixtures_sha256",
    ] {
        let (mut report, artifacts) = lawful(false);
        report["subject"][field] = json!("9".repeat(64));
        assert_eq!(check(&report, &artifacts), Err(ReportError::Stale));
    }
}

#[test]
fn draft_manifests_cannot_authorize_reports() {
    let (mut report, artifacts) = lawful(false);
    report["fixture_stage"] = json!("draft");
    assert_eq!(check(&report, &artifacts), Err(ReportError::Draft));
}

#[test]
fn hashed_artifacts_must_exist_and_match() {
    let (report, mut artifacts) = lawful(false);
    assert_eq!(
        check(&report, &BTreeMap::new()),
        Err(ReportError::Incomplete)
    );
    artifacts.get_mut("capability.json").unwrap().push(b' ');
    assert_eq!(check(&report, &artifacts), Err(ReportError::Hash));
}

#[test]
fn summary_pass_cannot_hide_failed_or_incomplete_artifact() {
    for mutation in [
        "failed",
        "missing_case",
        "wrong_subject",
        "wrong_target",
        "lost_quality",
        "duplicate_case",
    ] {
        let (mut report, mut artifacts) = lawful(true);
        let name = if mutation == "lost_quality" {
            "quality.json"
        } else {
            "capability.json"
        };
        let mut evidence: Value = serde_json::from_slice(&artifacts[name]).unwrap();
        match mutation {
            "failed" => evidence["disposition"] = json!("fail"),
            "missing_case" => {
                evidence["cases"].as_array_mut().unwrap().pop();
            }
            "wrong_subject" => evidence["subject"]["runtime_sha256"] = json!("9".repeat(64)),
            "wrong_target" => evidence["harness"]["target_runtime_sha256"] = json!("8".repeat(64)),
            "lost_quality" => evidence["details"]["lost_successes"] = json!(2),
            _ => {
                let duplicate = evidence["cases"][0].clone();
                evidence["cases"][1] = duplicate;
            }
        }
        let bytes = serde_json::to_vec(&evidence).unwrap();
        let record = report["records"]
            .as_array_mut()
            .unwrap()
            .iter_mut()
            .find(|record| record["path"] == name)
            .unwrap();
        record["sha256"] = json!(format!("{:x}", Sha256::digest(&bytes)));
        artifacts.insert(name.into(), bytes);
        assert!(check(&report, &artifacts).is_err(), "{mutation}");
    }
}

#[test]
fn unknown_fields_versions_duplicates_and_trailing_bytes_are_rejected() {
    let (mut report, artifacts) = lawful(false);
    report["schema_version"] = json!(2);
    assert_eq!(check(&report, &artifacts), Err(ReportError::Version));
    report["schema_version"] = json!(1);
    report["operator_pass"] = json!(true);
    assert_eq!(check(&report, &artifacts), Err(ReportError::Syntax));
    report.as_object_mut().unwrap().remove("operator_pass");
    let text = serde_json::to_string(&report).unwrap();
    for bytes in [
        format!("{text}{{}}"),
        text.replacen(
            "\"schema_version\":1",
            "\"schema_version\":1,\"schema_version\":1",
            1,
        ),
    ] {
        assert_eq!(
            qualification::validate(bytes.as_bytes(), &subject(), |_| Err(
                ReportError::Incomplete
            )),
            Err(ReportError::Syntax)
        );
    }
}

#[test]
fn report_bytes_depth_nodes_strings_and_reference_counts_are_bounded() {
    for bytes in [
        vec![b' '; 1_048_577],
        format!("{}0{}", "[".repeat(17), "]".repeat(17)).into_bytes(),
        format!("[{}]", vec!["0"; 65_536].join(",")).into_bytes(),
        serde_json::to_vec(&"a".repeat(4097)).unwrap(),
    ] {
        assert_eq!(
            qualification::validate(&bytes, &subject(), |_| Err(ReportError::Incomplete)),
            Err(ReportError::Bounds)
        );
    }
    let (mut report, artifacts) = lawful(false);
    let record = report["records"][0].clone();
    report["records"] = json!(vec![record; 1025]);
    assert_eq!(check(&report, &artifacts), Err(ReportError::Bounds));
}
