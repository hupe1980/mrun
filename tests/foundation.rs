mod support;

#[path = "../src/backend/mod.rs"]
#[expect(
    dead_code,
    reason = "Qualification construction follows native capability evidence"
)]
mod backend;
#[path = "../src/budget/mod.rs"]
mod budget;

use std::path::Path;
use std::time::Duration;

use support::{BackendEvent, EventError, EventLog, FixtureRoot, ManualClock, Pressure};

#[test]
fn controlled_clocks_share_time_without_sleeping() {
    let clock = ManualClock::default();
    let observer = clock.clone();
    assert_eq!(clock.now(), Duration::ZERO);
    assert_eq!(
        clock.advance(Duration::from_millis(100)),
        Some(Duration::from_millis(100))
    );
    assert_eq!(observer.now(), Duration::from_millis(100));
    assert_eq!(clock.advance(Duration::MAX), None);
    assert_eq!(clock.now(), Duration::from_millis(100));
}

#[test]
fn fixture_roots_are_unique_owned_and_removed_once() {
    let first = FixtureRoot::new("parallel").unwrap();
    let second = FixtureRoot::new("parallel").unwrap();
    assert_ne!(first.path(), second.path());
    let filename = first.child(Path::new("data.json")).unwrap();
    std::fs::write(&filename, b"{}").unwrap();
    assert!(!second.child(Path::new("data.json")).unwrap().exists());
    assert!(first.child(Path::new("../outside")).is_err());
    assert!(first.child(Path::new("/outside")).is_err());
    assert!(FixtureRoot::new("../../outside").is_err());
    let owned = first.path().to_owned();
    drop(first);
    assert!(!owned.exists());
    assert!(second.path().exists());
}

#[test]
fn backend_events_are_bounded_and_keep_ordered_timestamps() {
    let clock = ManualClock::default();
    assert!(matches!(
        EventLog::new(clock.clone(), 0),
        Err(EventError::InvalidCapacity)
    ));
    assert!(matches!(
        EventLog::new(clock.clone(), 1025),
        Err(EventError::InvalidCapacity)
    ));
    let mut log = EventLog::new(clock.clone(), 1).unwrap();
    let events = [
        BackendEvent::Reserved(1024),
        BackendEvent::Evaluated(128),
        BackendEvent::Submitted,
        BackendEvent::CancelObserved,
        BackendEvent::Synchronized,
        BackendEvent::Released(1024),
        BackendEvent::Fault,
        BackendEvent::Pressure(Pressure::Nominal),
        BackendEvent::Pressure(Pressure::Elevated),
        BackendEvent::Pressure(Pressure::Critical),
        BackendEvent::Pressure(Pressure::Missing),
    ];
    for event in events {
        clock.advance(Duration::from_millis(1)).unwrap();
        log.push(event).unwrap();
        assert_eq!(log.push(BackendEvent::Fault), Err(EventError::Full));
        assert_eq!(log.pop(), Some((clock.now(), event)));
        assert_eq!(log.pop(), None);
    }
}

#[test]
fn coding_fixtures_compile_fail_behaviorally_and_have_passing_references() {
    let catalog: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/coding/cases.json")).unwrap();
    let cases = catalog["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 20);
    assert_eq!(catalog["stage"], "draft");
    assert_eq!(catalog["seeds"], serde_json::json!([7, 19, 41]));
    let root = FixtureRoot::new("coding-authoring").unwrap();
    let cargo = std::path::Path::new(env!("CARGO"));
    for (index, case) in cases.iter().enumerate() {
        assert_eq!(case["id"], format!("C-{:03}", index + 1));
        let workspace = root.child(Path::new(case["id"].as_str().unwrap())).unwrap();
        std::fs::create_dir_all(workspace.join("src")).unwrap();
        std::fs::write(
            workspace.join("Cargo.toml"),
            "[package]\nname=\"mrun-authored-task\"\nversion=\"0.0.0\"\nedition=\"2024\"\nrust-version=\"1.94\"\n[dependencies]\nserde_json=\"=1.0.154\"\n",
        )
        .unwrap();
        for (source_key, files_key, should_pass) in [
            ("source", "files", false),
            ("reference", "reference_files", true),
        ] {
            if let Some(files) = case[files_key].as_object() {
                for (name, contents) in files {
                    let destination = FixtureRoot::child(&root, Path::new(name)).unwrap();
                    assert!(destination.starts_with(root.path()));
                    assert!(name.starts_with("src/"));
                    std::fs::write(workspace.join(name), contents.as_str().unwrap()).unwrap();
                }
            }
            let source = format!(
                "{}\n#[cfg(test)] mod supplied {{ use super::*; {} }}\n",
                case[source_key].as_str().unwrap(),
                case["tests"].as_str().unwrap()
            );
            std::fs::write(workspace.join("src/lib.rs"), source).unwrap();
            let result = std::process::Command::new(cargo)
                .args(["test", "--offline", "--lib"])
                .current_dir(&workspace)
                .env("RUSTUP_TOOLCHAIN", "1.94.0")
                .env("RUSTC", cargo.with_file_name("rustc"))
                .env("CARGO_TARGET_DIR", root.path().join("target"))
                .output()
                .unwrap();
            let stdout = String::from_utf8_lossy(&result.stdout);
            let stderr = String::from_utf8_lossy(&result.stderr);
            assert_eq!(
                result.status.success(),
                should_pass,
                "{} {source_key}: {stdout}\n{stderr}",
                case["id"]
            );
            if !should_pass {
                assert!(stdout.contains("test result: FAILED"), "{stderr}");
                assert!(stdout.contains("panicked") || stdout.contains("assertion"));
            }
        }
    }
}

#[test]
fn replay_sources_expand_to_ten_distinct_ten_turn_histories() {
    let catalog: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/replay/cases.json")).unwrap();
    assert_eq!(catalog["stage"], "draft");
    assert!(catalog["rendered_tokens"].is_null());
    let cases = catalog["cases"].as_array().unwrap();
    let turns = catalog["turns"].as_array().unwrap();
    assert_eq!(cases.len(), 10);
    assert_eq!(turns.len(), 10);
    let names: Vec<_> = catalog["tools"]
        .as_array()
        .unwrap()
        .iter()
        .map(|tool| tool["function"]["name"].as_str().unwrap())
        .collect();
    let mut histories = std::collections::HashSet::new();
    for (index, case) in cases.iter().enumerate() {
        assert_eq!(case["id"], format!("R-{:03}", index + 1));
        assert_ne!(case["source"], case["fixed"]);
        let mut history = String::new();
        for (turn_index, turn) in turns.iter().enumerate() {
            assert_eq!(turn["number"], turn_index + 1);
            assert!(names.contains(&turn["call"]["name"].as_str().unwrap()));
            let decoded = support::qualification::fixtures::expand_replay_turn(turn, case).unwrap();
            let expanded = serde_json::to_string(&decoded).unwrap();
            for key in ["path", "source", "fixed", "issue", "check"] {
                assert!(!expanded.contains(&format!("${key}")));
            }
            assert!(decoded["call"]["arguments"].is_object());
            assert!(decoded["recorded_result"].is_object());
            history.push_str(&expanded);
        }
        assert!(histories.insert(history));
    }
}
