use serde::de::{self, DeserializeSeed, MapAccess, SeqAccess, Visitor};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};
use std::cell::Cell;
use std::collections::BTreeSet;
use std::fmt;

/// Parse limits from the data model; they apply before any field is interpreted.
const MAX_BYTES: usize = 1024 * 1024;
const MAX_DEPTH: usize = 16;
const MAX_NODES: usize = 65_536;
const MAX_STRING: usize = 4 * 1024;
const MAX_RECORDS: usize = 1_024;

const CAPABILITY_CASES: [&str; 11] = [
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
];
const QUALITY_SEEDS: [u32; 3] = [7, 19, 41];
const MIN_EXACT_POSITIVE_CASES: u32 = 20;
const PAIRED_OUTCOMES: u32 = 60;
const MAX_LOST_SUCCESSES: u32 = 1;

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

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
enum Stage {
    AdapterSpike,
    CandidateCapability,
    CandidateReuse,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
enum FixtureStage {
    Draft,
    Source,
    Rendered,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd)]
#[serde(rename_all = "snake_case")]
enum Gate {
    Capability,
    State,
    Numerics,
    Protocol,
    Quality,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
enum Disposition {
    Pass,
    Fail,
    Blocked,
    Inconclusive,
}

#[derive(Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
struct Harness {
    sha256: String,
    build: String,
    run_id: String,
    target_runtime_sha256: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Report {
    schema_version: u32,
    stage: Stage,
    subject: Subject,
    harness: Harness,
    fixture_stage: FixtureStage,
    records: Vec<Record>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Record {
    gate: Gate,
    disposition: Disposition,
    path: String,
    sha256: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Evidence {
    schema_version: u32,
    subject: Subject,
    harness: Harness,
    gate: Gate,
    disposition: Disposition,
    cases: Vec<String>,
    details: Details,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Details {
    complete: bool,
    exact_positive_cases: u32,
    numerical_positions_pass: bool,
    paired_outcomes: u32,
    lost_successes: u32,
    systematic_new_failure: bool,
    all_completed_tools_valid: bool,
}

/// Derives serving eligibility from a report and its separately hashed evidence artifacts.
///
/// Summary dispositions never authorize anything on their own: every required artifact is read,
/// hash-checked and re-validated against the current subject before eligibility is returned.
pub fn validate(
    bytes: &[u8],
    current: &Subject,
    mut read_evidence: impl FnMut(&str) -> Result<Vec<u8>, ReportError>,
) -> Result<Eligibility, ReportError> {
    let report: Report = parse(bytes)?;
    if report.schema_version != 1 {
        return Err(ReportError::Version);
    }
    if report.records.len() > MAX_RECORDS {
        return Err(ReportError::Bounds);
    }
    check_subject(&report.subject)?;
    check_harness(&report.harness)?;
    let reuse = match report.stage {
        Stage::AdapterSpike => return Err(ReportError::Incomplete),
        Stage::CandidateCapability => false,
        Stage::CandidateReuse => true,
    };
    let runtime = current
        .runtime_sha256
        .as_ref()
        .ok_or(ReportError::Unavailable)?;
    if report.subject != *current || report.harness.target_runtime_sha256.as_ref() != Some(runtime)
    {
        return Err(ReportError::Stale);
    }
    if report.harness.sha256 == *runtime {
        return Err(ReportError::Incomplete);
    }
    if report.fixture_stage != FixtureStage::Rendered {
        return Err(ReportError::Draft);
    }

    let mut passed = BTreeSet::new();
    for record in &report.records {
        check_hash(&record.sha256)?;
        check_path(&record.path)?;
        if !passed.insert(record.gate) {
            return Err(ReportError::Syntax);
        }
        let artifact = read_evidence(&record.path)?;
        if format!("{:x}", Sha256::digest(&artifact)) != record.sha256 {
            return Err(ReportError::Hash);
        }
        let evidence: Evidence = parse(&artifact)?;
        if evidence.schema_version != 1 {
            return Err(ReportError::Version);
        }
        if evidence.subject != *current || evidence.harness != report.harness {
            return Err(ReportError::Stale);
        }
        if evidence.gate != record.gate
            || record.disposition != Disposition::Pass
            || evidence.disposition != Disposition::Pass
            || !evidence.details.complete
            || !covers(record.gate, &evidence.cases)
            || !gate_details_pass(record.gate, &evidence.details)
        {
            return Err(ReportError::Incomplete);
        }
    }
    let required: &[Gate] = if reuse {
        &[
            Gate::Capability,
            Gate::State,
            Gate::Numerics,
            Gate::Protocol,
            Gate::Quality,
        ]
    } else {
        &[Gate::Capability]
    };
    if !required.iter().all(|gate| passed.contains(gate)) {
        return Err(ReportError::Incomplete);
    }
    Ok(Eligibility {
        cache_off: true,
        reuse,
    })
}

fn required_cases(gate: Gate) -> BTreeSet<String> {
    let numbered =
        |prefix: &'static str| (1..=20).map(move |number| format!("{prefix}-{number:03}"));
    match gate {
        Gate::Capability => CAPABILITY_CASES.iter().map(|case| (*case).into()).collect(),
        Gate::State => ["P", "B", "I", "S", "F"]
            .into_iter()
            .flat_map(numbered)
            .collect(),
        Gate::Numerics => numbered("P").collect(),
        Gate::Protocol => numbered("IC").collect(),
        Gate::Quality => numbered("C")
            .flat_map(|case| QUALITY_SEEDS.map(|seed| format!("{case}@{seed}")))
            .collect(),
    }
}

fn covers(gate: Gate, cases: &[String]) -> bool {
    let unique: BTreeSet<String> = cases.iter().cloned().collect();
    unique.len() == cases.len() && unique == required_cases(gate)
}

fn gate_details_pass(gate: Gate, details: &Details) -> bool {
    match gate {
        Gate::Capability | Gate::Protocol => true,
        Gate::State => details.exact_positive_cases >= MIN_EXACT_POSITIVE_CASES,
        Gate::Numerics => details.numerical_positions_pass,
        Gate::Quality => {
            details.paired_outcomes == PAIRED_OUTCOMES
                && details.lost_successes <= MAX_LOST_SUCCESSES
                && !details.systematic_new_failure
                && details.all_completed_tools_valid
        }
    }
}

fn check_subject(subject: &Subject) -> Result<(), ReportError> {
    for hash in [
        &subject.backend,
        &subject.model,
        &subject.profile,
        &subject.render,
        &subject.build,
        &subject.fixtures_sha256,
    ] {
        check_hash(hash)?;
    }
    if let Some(runtime) = &subject.runtime_sha256 {
        check_hash(runtime)?;
    }
    if subject.fixtures_revision.is_empty() {
        return Err(ReportError::Syntax);
    }
    Ok(())
}

fn check_harness(harness: &Harness) -> Result<(), ReportError> {
    check_hash(&harness.sha256)?;
    check_hash(&harness.build)?;
    if let Some(target) = &harness.target_runtime_sha256 {
        check_hash(target)?;
    }
    if harness.run_id.is_empty() {
        return Err(ReportError::Syntax);
    }
    Ok(())
}

fn check_hash(hash: &str) -> Result<(), ReportError> {
    if hash.len() == 64
        && hash
            .bytes()
            .all(|byte| matches!(byte, b'0'..=b'9' | b'a'..=b'f'))
    {
        Ok(())
    } else {
        Err(ReportError::Syntax)
    }
}

fn check_path(path: &str) -> Result<(), ReportError> {
    let lawful = !path.is_empty()
        && !path.starts_with('/')
        && !path.contains('\\')
        && path
            .split('/')
            .all(|part| !part.is_empty() && part != "." && part != "..");
    if lawful {
        Ok(())
    } else {
        Err(ReportError::Syntax)
    }
}

/// Parses bounded JSON, rejecting duplicate keys and trailing data before typed decoding.
fn parse<T: for<'de> Deserialize<'de>>(bytes: &[u8]) -> Result<T, ReportError> {
    if bytes.len() > MAX_BYTES {
        return Err(ReportError::Bounds);
    }
    let limits = Limits::default();
    let mut deserializer = serde_json::Deserializer::from_slice(bytes);
    let value = Bounded {
        limits: &limits,
        depth: 0,
    }
    .deserialize(&mut deserializer)
    .and_then(|value| deserializer.end().map(|()| value))
    .map_err(|_| limits.failure.get().unwrap_or(ReportError::Syntax))?;
    serde_json::from_value(value).map_err(|_| ReportError::Syntax)
}

#[derive(Default)]
struct Limits {
    nodes: Cell<usize>,
    failure: Cell<Option<ReportError>>,
}

impl Limits {
    fn fail<E: de::Error>(&self, error: ReportError) -> E {
        self.failure.set(Some(error));
        E::custom("report rejected")
    }

    fn node<E: de::Error>(&self) -> Result<(), E> {
        let nodes = self.nodes.get() + 1;
        self.nodes.set(nodes);
        if nodes > MAX_NODES {
            return Err(self.fail(ReportError::Bounds));
        }
        Ok(())
    }

    fn string<E: de::Error>(&self, value: &str) -> Result<String, E> {
        if value.len() > MAX_STRING {
            return Err(self.fail(ReportError::Bounds));
        }
        Ok(value.to_owned())
    }
}

#[derive(Clone, Copy)]
struct Bounded<'a> {
    limits: &'a Limits,
    depth: usize,
}

impl Bounded<'_> {
    fn nested<E: de::Error>(self) -> Result<Self, E> {
        let depth = self.depth + 1;
        if depth > MAX_DEPTH {
            return Err(self.limits.fail(ReportError::Bounds));
        }
        Ok(Self { depth, ..self })
    }
}

impl<'de> DeserializeSeed<'de> for Bounded<'_> {
    type Value = Value;

    fn deserialize<D: de::Deserializer<'de>>(self, deserializer: D) -> Result<Value, D::Error> {
        deserializer.deserialize_any(self)
    }
}

impl<'de> Visitor<'de> for Bounded<'_> {
    type Value = Value;

    fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        formatter.write_str("bounded JSON")
    }

    fn visit_bool<E: de::Error>(self, value: bool) -> Result<Value, E> {
        self.limits.node()?;
        Ok(Value::Bool(value))
    }

    fn visit_i64<E: de::Error>(self, value: i64) -> Result<Value, E> {
        self.limits.node()?;
        Ok(value.into())
    }

    fn visit_u64<E: de::Error>(self, value: u64) -> Result<Value, E> {
        self.limits.node()?;
        Ok(value.into())
    }

    fn visit_f64<E: de::Error>(self, value: f64) -> Result<Value, E> {
        self.limits.node()?;
        Ok(value.into())
    }

    fn visit_str<E: de::Error>(self, value: &str) -> Result<Value, E> {
        self.limits.node()?;
        self.limits.string(value).map(Value::String)
    }

    fn visit_unit<E: de::Error>(self) -> Result<Value, E> {
        self.limits.node()?;
        Ok(Value::Null)
    }

    fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Value, A::Error> {
        self.limits.node()?;
        let inner = self.nested()?;
        let mut values = Vec::new();
        while let Some(value) = seq.next_element_seed(inner)? {
            values.push(value);
        }
        Ok(Value::Array(values))
    }

    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Value, A::Error> {
        self.limits.node()?;
        let inner = self.nested()?;
        let mut values = Map::new();
        while let Some(key) = map.next_key::<String>()? {
            let key = self.limits.string(&key)?;
            let value = map.next_value_seed(inner)?;
            if values.insert(key, value).is_some() {
                return Err(self.limits.fail(ReportError::Syntax));
            }
        }
        Ok(Value::Object(values))
    }
}
