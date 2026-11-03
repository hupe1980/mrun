//! The run result (005 data model, `schema_version: 3`): what `mrun bench` writes as JSON and
//! renders as markdown. `crates/mrun/bench/result.schema.json` is the hand-written contract;
//! tests validate emitted results against it.
//!
//! Version 2 (D038): agent cells keep complete sessions only and list the excluded ones in
//! `incomplete`; quality signals moved into `quality` (the natural-stop pass) and the timing
//! pass counts `short_replies`; `comparisons` hold mrun's speed-up with bootstrap intervals;
//! systems carry `release` and `verified`.
//!
//! Version 3 (D041, D044, D045, D053, FR-020–FR-022): per system `lossy_kv`, `speculation`,
//! `lanes` and `numerics`; per turn `class`, `offset_ms`, `gap`, `unreused` and the message
//! role; concurrent cells (`class`, `main_ttft`, `aux_ttft`); `splice_hit_rate` per client;
//! session times and per-kind and session comparisons (`scope`); comparability
//! `different-numerics`. Hard cut, unreleased.

use crate::bench::stats::{self, Summary};
use serde::{Deserialize, Serialize};

pub const SCHEMA_VERSION: u32 = 3;

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct RunResult {
    pub schema_version: u32,
    pub started_at: String,
    pub finished_at: String,
    pub mrun_version: String,
    pub machine: Machine,
    pub systems: Vec<SystemUnderTest>,
    pub resets: Vec<Reset>,
    pub cells: Vec<Cell>,
    pub comparability: Vec<Comparability>,
    /// mrun against every other system, per cell (D038).
    pub comparisons: Vec<Comparison>,
    pub method: Method,
    pub warnings: Vec<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Machine {
    pub chip: String,
    pub memory_bytes: u64,
    pub macos: String,
    pub power_source: String,
    pub low_power: bool,
    pub per_system: Vec<SystemMemory>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct SystemMemory {
    pub system: String,
    pub free_memory_before: u64,
    pub swap: u64,
    /// 1-minute load average before the system ran.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub load_average: Option<f64>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct SystemUnderTest {
    pub name: String,
    pub version: String,
    /// The release the profile's flags were checked against (`unpinned` when none was).
    pub release: String,
    /// False for profiles that could not be checked: reported, never in the headline.
    pub verified: bool,
    pub endpoint: String,
    /// `chat`, `messages`, `responses`.
    pub apis: Vec<String>,
    pub context_length: u32,
    pub flags: Vec<String>,
    pub model: ModelUnderTest,
    /// The KV format is not exact (FR-015): its cells are labeled, never dropped.
    pub lossy_kv: bool,
    /// `none`, `mtp`, `dflash`, `pld`, `suffix`, `draft`, or as reported (FR-011).
    pub speculation: String,
    /// Parallel sequences configured; 0 = unknown (FR-011, FR-014).
    pub lanes: u32,
    pub numerics: Numerics,
}

/// The numerics class (D044, FR-021): NAX kernels on or off and the GPU family.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Numerics {
    /// The tensor backend (`mlx`, `nuvec`; spec 016 FR-008), when the server reports it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub backend: Option<String>,
    /// The backend's version (`0.32.4`, `0.4.1`; 016 FR-008), when the server reports it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub backend_version: Option<String>,
    /// How its GPU kernels are provided (`precompiled`, `runtime`; 016 FR-009).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kernels: Option<String>,
    /// `on`, `off` or `unknown`.
    pub nax: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gpu_family: Option<String>,
}

impl Default for Numerics {
    fn default() -> Self {
        Self {
            backend: None,
            backend_version: None,
            kernels: None,
            nax: "unknown".into(),
            gpu_family: None,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ModelUnderTest {
    pub name: String,
    /// `mlx` or `gguf`.
    pub format: String,
    pub bits_per_weight: Option<f64>,
    /// `bf16`, `f16`, `int8`, …
    pub kv_format: String,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Reset {
    pub system: String,
    pub rep: u32,
    pub step: String,
    pub ok: bool,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum CellStatus {
    #[default]
    Ok,
    Failed,
    NotRun,
    #[serde(rename = "n/a")]
    NotApplicable,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TurnKind {
    New,
    Repeated,
    Branching,
}

/// `main` (the agent loop) or `auxiliary` (side requests: titles, summaries; FR-020).
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TurnClass {
    #[default]
    Main,
    Auxiliary,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Turn {
    pub n: u32,
    pub conversation: String,
    pub kind: TurnKind,
    pub ttft_ms: f64,
    pub tfut_ms: f64,
    pub input: u32,
    pub cached: u32,
    pub output: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cause: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message_index: Option<u32>,
    /// Role of the message the miss points at.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub message_role: Option<String>,
    /// 003 restore gap and unreused tokens (D053).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gap: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub unreused: Option<u32>,
    /// Repetition (1-based).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rep: Option<u32>,
    #[serde(default)]
    pub class: TurnClass,
    /// Concurrent class: the recorded arrival after the start of the main turn it overlapped.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub offset_ms: Option<f64>,
}

/// One complete session's time (Σ TTFT + decode), for paired session comparisons.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SessionTime {
    pub rep: u32,
    pub s: f64,
}

/// mrun's splice hit rate for one trace client (FR-022): natural-stop-pass requests whose
/// current turn holds an earlier assistant reply, and those mrun spliced.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SpliceRate {
    pub client: String,
    pub eligible: u32,
    pub spliced: u32,
    /// `spliced / eligible`; `None` with no eligible request.
    pub rate: Option<f64>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Failure {
    pub rep: u32,
    pub error: String,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Cell {
    pub system: String,
    pub cell: String,
    pub api: String,
    pub status: CellStatus,
    pub samples: Vec<f64>,
    pub min: Option<f64>,
    pub median: Option<f64>,
    pub max: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub p95: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub iqr: Option<f64>,
    /// `ms`, `tok/s`, `s`.
    pub unit: String,
    /// `nominal`, `fair`, `serious`, `critical` (`NSProcessInfo.thermalState`) or `unknown`.
    pub thermal_start: String,
    pub thermal_end: String,
    /// Timing pass: replies that stopped before the forced length (the system cannot ignore
    /// end-of-sequence); their decode is shorter than everyone else's.
    pub short_replies: u32,
    /// Agent cells: the natural-stop pass.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub quality: Option<Quality>,
    /// `canary`, or `unverified: <reason>` (D038).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub truncation_check: Option<String>,
    pub failures: Vec<Failure>,
    /// Agent cells: sessions (repetitions) left out as a whole, with the reason.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub incomplete: Vec<Failure>,
    /// Agent cells: sessions in which every turn completed untruncated.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub complete_sessions: Option<u32>,
    /// Agent cells: the turns of complete sessions.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub turns: Option<Vec<Turn>>,
    /// Agent cells: median session time over complete sessions.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub session_s: Option<f64>,
    /// Agent cells: each complete session's time.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub session_times: Vec<SessionTime>,
    /// Agent cells: `sequential` or `concurrent` (FR-020).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub class: Option<String>,
    /// Concurrent cells: median TTFT of main and of auxiliary turns.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub main_ttft: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub aux_ttft: Option<f64>,
    /// The system's KV is lossy (FR-015); copied from the system.
    #[serde(default)]
    pub lossy_kv: bool,
    /// The system's speculation method; copied from the system.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub speculation: Option<String>,
    /// mrun agent cells: splice hit rate per trace client (FR-022).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub splice_hit_rate: Option<Vec<SpliceRate>>,
}

/// Quality signals of one natural-stop replay: the client's own output limit, no ignore-EOS.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Quality {
    /// Turns that returned a reply.
    pub turns: u32,
    /// Tool calls with invalid JSON, an undeclared tool or arguments that fail its schema.
    pub malformed_tool_calls: u32,
    /// Replies that ended at end-of-sequence with neither visible text nor a tool call.
    pub early_eos: u32,
    /// Replies that hit the output limit.
    pub length_capped: u32,
    /// False when a turn failed: the counts cover the completed turns only.
    pub complete: bool,
    pub failures: Vec<Failure>,
}

/// mrun's speed-up over `system` on `cell` (> 1: mrun is faster), with its bootstrap interval.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Comparison {
    pub cell: String,
    pub system: String,
    /// Agent cells: `new`, `repeated`, `branching` (turns of that kind only) or `session`
    /// (session time); absent: every turn.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scope: Option<String>,
    /// `paired-turns` (agent: geometric mean of per-turn TTFT ratios over the turns every
    /// compared system completed), `paired-sessions` (session time over the repetitions valid
    /// for both) or `median-ratio` (synthetic: unpaired).
    pub method: String,
    /// Turns paired, or the smaller sample count.
    pub n: u32,
    pub speedup: f64,
    pub ci_low: f64,
    pub ci_high: f64,
    /// `mrun-faster`, `mrun-slower` (the interval excludes 1) or `within-noise`.
    pub verdict: String,
    /// False when the other system's profile is unverified.
    pub headline: bool,
}

impl Cell {
    /// Fills the summary fields from `samples`.
    pub fn summarize(&mut self) {
        let summary: Option<Summary> = stats::summarize(&self.samples);
        self.min = summary.map(|s| s.min);
        self.median = summary.map(|s| s.median);
        self.max = summary.map(|s| s.max);
        self.p95 = summary.and_then(|s| s.p95);
        self.iqr = summary.and_then(|s| s.iqr);
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Comparability {
    pub cell: String,
    /// The system compared with mrun on this cell.
    #[serde(default)]
    pub system: String,
    /// `same-weights`, `different-quantization`, `lossy-kv`, `different-numerics`, `n/a`.
    pub kind: String,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Method {
    /// Always `in-order`.
    pub replay: String,
    pub reps: Reps,
    /// Bootstrap resamples and seed of every interval.
    pub resamples: u32,
    pub seed: u64,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Reps {
    pub synthetic: u32,
    pub cancel: u32,
    pub agent: u32,
}

/// The hand-written schema and a validator for the keywords it uses, for tests here and in
/// `tests/bench_driver.rs`.
#[cfg(any(test, feature = "test-support"))]
pub mod test_support {
    use serde_json::Value;

    /// The keywords `crates/mrun/bench/result.schema.json` uses: type, const, enum, properties, required,
    /// items, minimum.
    pub fn validate(schema: &Value, value: &Value, path: &str) -> Result<(), String> {
        if let Some(c) = schema.get("const")
            && c != value
        {
            return Err(format!("{path}: expected {c}"));
        }
        if let Some(options) = schema.get("enum").and_then(Value::as_array)
            && !options.contains(value)
        {
            return Err(format!("{path}: {value} not in {options:?}"));
        }
        if let Some(ty) = schema.get("type") {
            let types: Vec<&str> = match ty {
                Value::String(t) => vec![t.as_str()],
                Value::Array(ts) => ts.iter().filter_map(Value::as_str).collect(),
                _ => vec![],
            };
            let ok = types.iter().any(|t| match *t {
                "object" => value.is_object(),
                "array" => value.is_array(),
                "string" => value.is_string(),
                "number" => value.is_number(),
                "integer" => value.is_u64() || value.is_i64(),
                "boolean" => value.is_boolean(),
                "null" => value.is_null(),
                _ => false,
            });
            if !ok {
                return Err(format!("{path}: {value} is not {types:?}"));
            }
        }
        if let Some(min) = schema.get("minimum").and_then(Value::as_f64)
            && value.as_f64().is_some_and(|v| v < min)
        {
            return Err(format!("{path}: below {min}"));
        }
        if let Some(required) = schema.get("required").and_then(Value::as_array) {
            for key in required.iter().filter_map(Value::as_str) {
                if value.get(key).is_none() {
                    return Err(format!("{path}: missing {key}"));
                }
            }
        }
        if let (Some(props), Some(object)) = (schema.get("properties"), value.as_object()) {
            for (key, sub) in object {
                if let Some(subschema) = props.get(key) {
                    validate(subschema, sub, &format!("{path}.{key}"))?;
                }
            }
        }
        if let (Some(items), Some(array)) = (schema.get("items"), value.as_array()) {
            for (i, item) in array.iter().enumerate() {
                validate(items, item, &format!("{path}[{i}]"))?;
            }
        }
        Ok(())
    }

    pub fn schema() -> Value {
        serde_json::from_str(include_str!("../../bench/result.schema.json")).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use test_support::{schema, validate};

    #[test]
    fn emitted_results_validate_against_the_schema() {
        let mut result = crate::bench::report::tests::sample();
        result.cells[0].turns = Some(vec![Turn {
            n: 1,
            conversation: "c1".into(),
            kind: TurnKind::Repeated,
            ttft_ms: 120.0,
            tfut_ms: 130.0,
            input: 9000,
            cached: 8000,
            output: 40,
            cause: Some("diverged".into()),
            message_index: Some(4),
            message_role: Some("assistant".into()),
            gap: None,
            unreused: Some(120),
            rep: Some(1),
            class: TurnClass::Auxiliary,
            offset_ms: Some(250.0),
        }]);
        result.cells[0].class = Some("concurrent".into());
        result.cells[0].main_ttft = Some(120.0);
        result.cells[0].aux_ttft = Some(80.0);
        result.cells[0].session_times = vec![SessionTime { rep: 1, s: 4.5 }];
        result.cells[0].splice_hit_rate = Some(vec![
            SpliceRate {
                client: "claude-code".into(),
                eligible: 4,
                spliced: 3,
                rate: Some(0.75),
            },
            SpliceRate {
                client: "codex".into(),
                eligible: 0,
                spliced: 0,
                rate: None,
            },
        ]);
        result.systems[0].numerics = Numerics {
            backend: Some("nuvec".into()),
            backend_version: Some("0.4.1".into()),
            kernels: Some("precompiled".into()),
            nax: "off".into(),
            gpu_family: Some("applegpu_g16s".into()),
        };
        result.comparability.push(Comparability {
            cell: "synthetic/prefill-8k".into(),
            system: "llama.cpp".into(),
            kind: "different-quantization".into(),
        });
        result.machine.per_system.push(SystemMemory {
            system: "mrun".into(),
            free_memory_before: 1 << 30,
            swap: 0,
            load_average: Some(1.5),
        });
        result.cells[0].failures.push(Failure {
            rep: 2,
            error: "turn 3 truncated (the canary added 0 of its 24 tokens)".into(),
        });
        result.cells[0].incomplete = result.cells[0].failures.clone();
        result.cells[0].complete_sessions = Some(2);
        result.cells[0].truncation_check = Some("canary".into());
        result.cells[0].quality = Some(Quality {
            turns: 20,
            malformed_tool_calls: 1,
            early_eos: 0,
            length_capped: 2,
            complete: true,
            failures: Vec::new(),
        });
        assert!(!result.comparisons.is_empty());
        let value = serde_json::to_value(&result).unwrap();
        assert_eq!(value["schema_version"], 3);
        assert_eq!(value["systems"][0]["lanes"], 0);
        assert_eq!(value["cells"][0]["turns"][0]["class"], "auxiliary");
        validate(&schema(), &value, "$").unwrap();
        // The validator catches a broken result.
        let mut broken = value.clone();
        broken["cells"][0]["status"] = "great".into();
        assert!(validate(&schema(), &broken, "$").is_err());
        broken = value.clone();
        broken.as_object_mut().unwrap().remove("method");
        assert!(validate(&schema(), &broken, "$").is_err());
        broken = value.clone();
        broken["comparisons"][0]["verdict"] = "fastest".into();
        assert!(validate(&schema(), &broken, "$").is_err());
        broken = value.clone();
        broken["cells"][0]["turns"][0]["cause"] = "re-render".into();
        assert!(
            validate(&schema(), &broken, "$").is_err(),
            "003 causes only"
        );
        broken = value.clone();
        broken["systems"][0]["numerics"]["nax"] = "maybe".into();
        assert!(validate(&schema(), &broken, "$").is_err());
        broken = value;
        broken["schema_version"] = 2.into();
        assert!(
            validate(&schema(), &broken, "$").is_err(),
            "v2 results are not v3"
        );
    }

    #[test]
    fn results_round_trip() {
        let result = crate::bench::report::tests::sample();
        let text = serde_json::to_string(&result).unwrap();
        assert_eq!(serde_json::from_str::<RunResult>(&text).unwrap(), result);
        assert!(text.contains("\"status\":\"not-run\""));
    }
}
