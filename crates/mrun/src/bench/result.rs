//! The run result (005 data model, `schema_version: 1`): what `mrun bench` writes as JSON and
//! renders as markdown. `bench/result.schema.json` is the hand-written contract; tests validate
//! emitted results against it.

use crate::bench::stats::{self, Summary};
use serde::{Deserialize, Serialize};

pub const SCHEMA_VERSION: u32 = 1;

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
    pub endpoint: String,
    /// `chat`, `messages`, `responses`.
    pub apis: Vec<String>,
    pub context_length: u32,
    pub flags: Vec<String>,
    pub model: ModelUnderTest,
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
    /// Repetition (1-based).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rep: Option<u32>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Failure {
    pub rep: u32,
    pub error: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Rerenders {
    pub turns: u32,
    pub tokens: u64,
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
    pub thermal_start: String,
    pub thermal_end: String,
    pub early_eos: u32,
    pub malformed_tool_calls: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub template_rerenders: Option<Rerenders>,
    pub failures: Vec<Failure>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub turns: Option<Vec<Turn>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub session_s: Option<f64>,
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

    /// Lower is better for latencies (`ms`, `s`), higher for rates (`tok/s`).
    pub fn lower_is_better(&self) -> bool {
        self.unit != "tok/s"
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Comparability {
    pub cell: String,
    /// The system compared with mrun on this cell.
    #[serde(default)]
    pub system: String,
    /// `same-weights`, `different-quantization`, `lossy-kv`, `n/a`.
    pub kind: String,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Method {
    /// Always `in-order`.
    pub replay: String,
    pub reps: Reps,
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

    /// The keywords `bench/result.schema.json` uses: type, const, enum, properties, required,
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
        serde_json::from_str(include_str!("../../../../bench/result.schema.json")).unwrap()
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
            cause: Some("changed".into()),
            message_index: Some(4),
            rep: Some(1),
        }]);
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
            error: "truncated input".into(),
        });
        let value = serde_json::to_value(&result).unwrap();
        validate(&schema(), &value, "$").unwrap();
        // The validator catches a broken result.
        let mut broken = value.clone();
        broken["cells"][0]["status"] = "great".into();
        assert!(validate(&schema(), &broken, "$").is_err());
        broken = value;
        broken.as_object_mut().unwrap().remove("method");
        assert!(validate(&schema(), &broken, "$").is_err());
    }

    #[test]
    fn results_round_trip() {
        let result = crate::bench::report::tests::sample();
        let text = serde_json::to_string(&result).unwrap();
        assert_eq!(serde_json::from_str::<RunResult>(&text).unwrap(), result);
        assert!(text.contains("\"status\":\"not-run\""));
    }
}
