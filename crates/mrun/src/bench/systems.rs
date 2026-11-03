//! Systems under test and their profiles (005 FR-007, FR-008, FR-014, FR-015): where each one
//! listens, its health check, how its caches are reset before every repetition (an HTTP call or
//! a manual restart pause), its recommended flags and model facts. Every request goes to the
//! system's own URL: profile steps are paths, never hosts (FR-011).

use crate::api::model::ApiKind;
use crate::bench::driver::IgnoreEos;
use crate::bench::result::{ModelUnderTest, Numerics, Reset, SystemUnderTest};
use crate::bench::trace::parse_api;
use serde::Deserialize;
use serde_json::Value;
use std::collections::BTreeMap;
use std::io::{BufRead, IsTerminal, Write};
use std::time::Duration;

/// Bundled profiles, `crates/mrun/bench/profiles/*.toml` (inside the crate, so they ship with
/// `cargo package`).
pub const BUNDLED: [(&str, &str); 7] = [
    ("mrun", include_str!("../../bench/profiles/mrun.toml")),
    (
        "llama.cpp",
        include_str!("../../bench/profiles/llama.cpp.toml"),
    ),
    ("ollama", include_str!("../../bench/profiles/ollama.toml")),
    ("omlx", include_str!("../../bench/profiles/omlx.toml")),
    (
        "rapid-mlx",
        include_str!("../../bench/profiles/rapid-mlx.toml"),
    ),
    ("mlx-lm", include_str!("../../bench/profiles/mlx-lm.toml")),
    ("splash", include_str!("../../bench/profiles/splash.toml")),
];

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Profile {
    pub name: String,
    /// The release the flags and endpoints were checked against (D038), `unpinned` otherwise.
    #[serde(default = "unpinned")]
    pub release: String,
    /// False when the profile could not be checked: its cells stay out of headline claims.
    #[serde(default = "yes")]
    pub verified: bool,
    pub url: String,
    pub apis: Vec<String>,
    /// The suites the system takes part in (`synthetic`, `agent`, `cancel`); empty means all.
    #[serde(default)]
    pub suites: Vec<String>,
    pub health: Option<String>,
    pub version_path: Option<String>,
    /// JSON pointer into the version response.
    pub version_pointer: Option<String>,
    pub context_path: Option<String>,
    pub context_pointer: Option<String>,
    /// Explicit context length; 0 means mrun's capacity N.
    #[serde(default)]
    pub context_length: u32,
    #[serde(default)]
    pub ignore_eos: IgnoreEos,
    #[serde(default)]
    pub flags: Vec<String>,
    /// Speculation method of the best exact configuration (FR-011, FR-015): `none`, `mtp`,
    /// `dflash`, `pld`, `suffix`, `draft`, or as the system names it.
    #[serde(default = "no_speculation")]
    pub speculation: String,
    /// Parallel sequences the start line configures (FR-014); mrun reads its own from
    /// `/mrun/status`.
    #[serde(default = "one_lane")]
    pub lanes: u32,
    /// Overrides the KV-format rule when a format name does not tell (FR-015).
    #[serde(default)]
    pub lossy_kv: Option<bool>,
    /// How to start it, shown at the switch pause.
    pub start: Option<String>,
    #[serde(default)]
    pub reset: Vec<HttpStep>,
    /// Reset by hand: the text of the restart pause.
    pub manual_reset: Option<String>,
    pub model: ModelSpec,
    #[serde(default)]
    pub variants: BTreeMap<String, Variant>,
}

fn unpinned() -> String {
    "unpinned".into()
}

fn yes() -> bool {
    true
}

fn no_speculation() -> String {
    "none".into()
}

fn one_lane() -> u32 {
    1
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelSpec {
    #[serde(default)]
    pub name: String,
    pub format: String,
    pub bits_per_weight: Option<f64>,
    pub kv_format: String,
    /// Memory the model needs once loaded, in GiB, for the pre-start check (D062); without it
    /// bench scales mrun's model by `bits_per_weight`.
    #[serde(default)]
    pub memory_gib: Option<f64>,
}

/// A labeled second configuration of the same system (Splash `splash-bf16`).
#[derive(Clone, Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Variant {
    pub flags: Vec<String>,
    pub start: Option<String>,
    pub kv_format: Option<String>,
    #[serde(default)]
    pub lossy_kv: Option<bool>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HttpStep {
    pub method: String,
    pub path: String,
    /// JSON body; `{model}` is replaced by the system's model name.
    pub body: Option<String>,
}

impl Profile {
    pub fn parse(text: &str) -> Result<Self, String> {
        let profile: Self = toml::from_str(text).map_err(|e| e.to_string())?;
        let paths = profile
            .reset
            .iter()
            .map(|s| s.path.as_str())
            .chain(profile.health.as_deref())
            .chain(profile.version_path.as_deref())
            .chain(profile.context_path.as_deref());
        for path in paths {
            if !path.starts_with('/') {
                return Err(format!("'{path}' must be a path on the system's own URL"));
            }
        }
        if let Some(api) = profile.apis.iter().find(|a| parse_api(a).is_none()) {
            return Err(format!("unknown api '{api}'"));
        }
        if let Some(suite) = profile
            .suites
            .iter()
            .find(|s| !matches!(s.as_str(), "synthetic" | "agent" | "cancel"))
        {
            return Err(format!("unknown suite '{suite}'"));
        }
        Ok(profile)
    }

    pub fn bundled(name: &str) -> Option<Self> {
        BUNDLED
            .iter()
            .find(|(n, _)| *n == name)
            .and_then(|(_, text)| Self::parse(text).ok())
    }

    /// The profile of a system name: a bundled profile or one of its variants.
    fn for_system(name: &str) -> Option<Self> {
        Self::bundled(name).or_else(|| {
            BUNDLED
                .iter()
                .filter_map(|(n, _)| Self::bundled(n))
                .find(|p| p.variants.contains_key(name))
        })
    }

    /// A custom system without a profile: no health path, no reset step.
    fn custom(url: &str) -> Self {
        Self {
            name: "custom".into(),
            release: unpinned(),
            verified: true,
            url: url.into(),
            apis: vec!["chat".into()],
            model: ModelSpec {
                format: "unknown".into(),
                kv_format: "unknown".into(),
                ..ModelSpec::default()
            },
            ..Self::default()
        }
    }
}

/// One system as the run uses it.
#[derive(Clone, Debug)]
pub struct System {
    pub name: String,
    pub url: String,
    pub apis: Vec<ApiKind>,
    pub model: String,
    pub profile: Profile,
}

impl System {
    /// `NAME=URL[,apis=chat+messages+responses][,model=M][,profile=P]`; `P` is a bundled
    /// profile name or a TOML file.
    pub fn parse(spec: &str) -> Result<Self, String> {
        let (name, rest) = spec
            .split_once('=')
            .ok_or_else(|| format!("'{spec}': expected NAME=URL[,apis=…][,model=…][,profile=…]"))?;
        let mut parts = rest.split(',');
        let url = parts.next().unwrap_or_default().to_string();
        if !url.starts_with("http://") && !url.starts_with("https://") {
            return Err(format!("'{url}' is not an http(s) URL"));
        }
        let (mut apis, mut model, mut profile) = (None, None, None);
        for part in parts {
            match part.split_once('=') {
                Some(("apis", v)) => apis = Some(v.split('+').map(String::from).collect()),
                Some(("model", v)) => model = Some(v.to_string()),
                Some(("profile", v)) => profile = Some(v.to_string()),
                _ => return Err(format!("'{part}': expected apis=, model= or profile=")),
            }
        }
        let profile = match profile {
            Some(p) => Profile::for_system(&p)
                .or_else(|| {
                    std::fs::read_to_string(&p)
                        .ok()
                        .and_then(|t| Profile::parse(&t).ok())
                })
                .ok_or_else(|| format!("profile '{p}' is neither bundled nor a readable file"))?,
            None => Profile::for_system(name).unwrap_or_else(|| Profile::custom(&url)),
        };
        Self::new(name, &url, apis, model, profile)
    }

    fn new(
        name: &str,
        url: &str,
        apis: Option<Vec<String>>,
        model: Option<String>,
        mut profile: Profile,
    ) -> Result<Self, String> {
        if let Some(variant) = profile.variants.get(name).cloned() {
            profile.flags = variant.flags;
            profile.start = variant.start.or(profile.start);
            if let Some(kv) = variant.kv_format {
                profile.model.kv_format = kv;
                profile.lossy_kv = None;
            }
            if variant.lossy_kv.is_some() {
                profile.lossy_kv = variant.lossy_kv;
            }
        }
        let apis = apis
            .unwrap_or_else(|| profile.apis.clone())
            .iter()
            .map(|a| parse_api(a).ok_or_else(|| format!("unknown api '{a}'")))
            .collect::<Result<_, _>>()?;
        Ok(Self {
            name: name.into(),
            url: url.trim_end_matches('/').into(),
            apis,
            model: model.unwrap_or_else(|| profile.model.name.clone()),
            profile,
        })
    }

    /// Every bundled profile and variant at its default URL, mrun first (`--compare`).
    pub fn all_bundled() -> Vec<Self> {
        let mut out = Vec::new();
        for (name, _) in BUNDLED {
            let Some(profile) = Profile::bundled(name) else {
                continue;
            };
            let names: Vec<String> = std::iter::once(name.to_string())
                .chain(profile.variants.keys().cloned())
                .collect();
            for n in names {
                if let Ok(system) = Self::new(&n, &profile.url.clone(), None, None, profile.clone())
                {
                    out.push(system);
                }
            }
        }
        out
    }

    pub fn is_mrun(&self) -> bool {
        self.profile.name == "mrun"
    }

    /// The first of Chat, Messages, Responses this system serves.
    pub fn preferred_api(&self) -> Option<ApiKind> {
        [ApiKind::Chat, ApiKind::Messages, ApiKind::Responses]
            .into_iter()
            .find(|a| self.apis.contains(a))
    }

    /// Whether the profile lets this system run `suite` (mlx-lm: synthetic only).
    pub fn runs(&self, suite: &str) -> bool {
        self.profile.suites.is_empty() || self.profile.suites.iter().any(|s| s == suite)
    }

    pub fn ignore_eos(&self) -> IgnoreEos {
        self.profile.ignore_eos
    }

    /// The record of this system for the result; `context` is its context length.
    pub fn describe(&self, version: String, context: u32) -> SystemUnderTest {
        let fill = |s: &str| s.replace("{context}", &context.to_string());
        SystemUnderTest {
            name: self.name.clone(),
            version,
            release: self.profile.release.clone(),
            // A profile whose flags were not checked against a pinned release never headlines.
            verified: self.profile.verified && self.profile.release != unpinned(),
            endpoint: self.url.clone(),
            apis: self.apis.iter().map(|a| a.as_str().to_string()).collect(),
            context_length: context,
            flags: self.profile.flags.iter().map(|f| fill(f)).collect(),
            model: ModelUnderTest {
                name: self.model.clone(),
                format: self.profile.model.format.clone(),
                bits_per_weight: self.profile.model.bits_per_weight,
                kv_format: self.profile.model.kv_format.clone(),
            },
            lossy_kv: self
                .profile
                .lossy_kv
                .unwrap_or_else(|| lossy_kv(&self.profile.model.kv_format)),
            speculation: self.profile.speculation.clone(),
            lanes: self.profile.lanes,
            numerics: Numerics::default(),
        }
    }

    /// mrun's lanes and numerics class from its `/mrun/status` (FR-011, FR-021): lanes from
    /// `lanes.configured` (absent: 1), numerics from `numerics {nax, gpu_family}` (absent:
    /// `nax: unknown`).
    pub async fn status_facts(&self, client: &reqwest::Client) -> (u32, Numerics) {
        match self.get(client, "/mrun/status").await {
            Ok(status) => status_facts(&status),
            Err(_) => (1, Numerics::default()),
        }
    }

    /// The backend of an mrun relay (`/mrun/status` with a `pod`, 013), `None` for the local
    /// engine.
    pub async fn remote_backend(&self, client: &reqwest::Client) -> Option<String> {
        let status = self.get(client, "/mrun/status").await.ok()?;
        remote_backend(&status)
    }

    pub fn start_hint(&self, context: u32) -> String {
        self.profile
            .start
            .as_deref()
            .map(|s| s.replace("{context}", &context.to_string()))
            .unwrap_or_else(|| self.name.clone())
    }

    async fn get(&self, client: &reqwest::Client, path: &str) -> Result<Value, String> {
        let response = client
            .get(format!("{}{path}", self.url))
            .timeout(Duration::from_secs(5))
            .send()
            .await
            .map_err(|e| format!("unreachable: {e}"))?;
        if !response.status().is_success() {
            return Err(format!("{path}: HTTP {}", response.status().as_u16()));
        }
        Ok(response.json().await.unwrap_or(Value::Null))
    }

    /// The health endpoint (or `/v1/models` without one).
    pub async fn health(&self, client: &reqwest::Client) -> Result<(), String> {
        let path = self.profile.health.as_deref().unwrap_or("/v1/models");
        self.get(client, path).await.map(drop)
    }

    pub async fn version(&self, client: &reqwest::Client) -> String {
        let (Some(path), Some(pointer)) =
            (&self.profile.version_path, &self.profile.version_pointer)
        else {
            return "unknown".into();
        };
        match self.get(client, path).await {
            Ok(v) => v.pointer(pointer).map_or_else(
                || "unknown".into(),
                |v| v.as_str().map_or_else(|| v.to_string(), Into::into),
            ),
            Err(_) => "unknown".into(),
        }
    }

    /// The context length the system reports, when its profile says where.
    pub async fn reported_context(&self, client: &reqwest::Client) -> Option<u32> {
        let (path, pointer) = (
            self.profile.context_path.as_ref()?,
            self.profile.context_pointer.as_ref()?,
        );
        let value = self.get(client, path).await.ok()?;
        value.pointer(pointer)?.as_u64().map(|n| n as u32)
    }

    pub fn needs_manual_reset(&self) -> bool {
        self.profile.manual_reset.is_some()
    }

    /// Clears the system's caches before repetition `rep` (FR-014).
    pub async fn reset(&self, client: &reqwest::Client, rep: u32, interactive: bool) -> Reset {
        let record = |step: String, ok| Reset {
            system: self.name.clone(),
            rep,
            step,
            ok,
        };
        if let Some(text) = &self.profile.manual_reset {
            if !interactive {
                return record(format!("manual: {text} (skipped: no terminal)"), false);
            }
            pause(&format!("{text}, then press Enter"));
            let ok = self.health(client).await.is_ok();
            return record(format!("manual: {text}"), ok);
        }
        if self.profile.reset.is_empty() {
            return record("none".into(), false);
        }
        let mut steps = Vec::new();
        let mut ok = true;
        for step in &self.profile.reset {
            let method = reqwest::Method::from_bytes(step.method.as_bytes())
                .unwrap_or(reqwest::Method::POST);
            let mut request = client
                .request(method, format!("{}{}", self.url, step.path))
                .timeout(Duration::from_secs(120));
            if let Some(body) = &step.body {
                request = request
                    .header("content-type", "application/json")
                    .body(body.replace("{model}", &self.model));
            }
            ok &= request.send().await.is_ok_and(|r| r.status().is_success());
            steps.push(format!("{} {}", step.method, step.path));
        }
        record(steps.join("; "), ok)
    }
}

/// `backend` of a relay's status (one with a `pod`).
pub fn remote_backend(status: &Value) -> Option<String> {
    if status["pod"].is_null() {
        return None;
    }
    Some(status["backend"].as_str().unwrap_or("remote").to_string())
}

/// Lanes and numerics from a `/mrun/status` body.
pub fn status_facts(status: &Value) -> (u32, Numerics) {
    let lanes = status["lanes"]["configured"]
        .as_u64()
        .and_then(|n| u32::try_from(n).ok())
        .filter(|&n| n > 0)
        .unwrap_or(1);
    let nax = match &status["numerics"]["nax"] {
        Value::Bool(true) => "on".to_string(),
        Value::Bool(false) => "off".to_string(),
        Value::String(s) if s == "on" || s == "off" => s.clone(),
        _ => "unknown".to_string(),
    };
    let engine = &status["engine"]["backend"];
    let numerics = Numerics {
        backend: status["numerics"]["backend"].as_str().map(Into::into),
        backend_version: engine["version"].as_str().map(Into::into),
        kernels: engine["kernels"].as_str().map(Into::into),
        nax,
        gpu_family: status["numerics"]["gpu_family"].as_str().map(Into::into),
    };
    (lanes, numerics)
}

/// Prints `message` and waits for Enter (callers check for a terminal first).
pub fn pause(message: &str) {
    eprint!("{message} ");
    let _ = std::io::stderr().flush();
    let _ = std::io::stdin().lock().read_line(&mut String::new());
}

/// Whether the harness may pause for the user.
pub fn interactive() -> bool {
    std::io::stdin().is_terminal()
}

/// KV formats that are not exact (FR-007 "lossy KV").
pub fn lossy_kv(kv_format: &str) -> bool {
    !matches!(
        kv_format.to_ascii_lowercase().as_str(),
        "bf16" | "f16" | "fp16" | "f32" | "fp32" | "unknown" | ""
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bundled_profiles_parse_and_use_paths_only() {
        for (name, text) in BUNDLED {
            let profile = Profile::parse(text).unwrap_or_else(|e| panic!("{name}: {e}"));
            assert_eq!(profile.name, name);
            assert!(profile.url.starts_with("http://127.0.0.1:"), "{name}");
            assert!(
                !profile.reset.is_empty() || profile.manual_reset.is_some(),
                "{name} has no reset"
            );
        }
        assert!(Profile::parse("name = \"x\"\nurl = \"http://a\"\napis = []\nhealth = \"http://evil/\"\n[model]\nformat = \"mlx\"\nkv_format = \"bf16\"").is_err());
    }

    #[test]
    fn profiles_carry_speculation_lanes_and_lossy_kv() {
        for (name, _) in BUNDLED {
            let profile = Profile::bundled(name).unwrap();
            assert!(!profile.speculation.is_empty(), "{name}");
            assert!(profile.lanes >= 1, "{name}");
            let start = profile.start.clone().unwrap_or_default();
            assert!(
                !start.is_empty(),
                "{name}: the start line shows the configuration"
            );
        }
        let omlx = Profile::bundled("omlx").unwrap();
        assert_eq!(omlx.apis, ["chat", "messages", "responses"]);
        let splash = System::parse("splash=http://127.0.0.1:8000").unwrap();
        assert!(splash.describe(String::new(), 0).lossy_kv);
        let bf16 = System::parse("splash-bf16=http://127.0.0.1:8000").unwrap();
        assert!(!bf16.describe(String::new(), 0).lossy_kv);
        let llama = System::parse("llama.cpp=http://127.0.0.1:9931").unwrap();
        let described = llama.describe(String::new(), 0);
        assert_eq!(
            (described.lanes, described.numerics.nax.as_str()),
            (1, "unknown")
        );
        assert!(
            Profile::bundled("mtplx").is_none(),
            "no mtplx profile in 0.1"
        );
    }

    #[test]
    fn a_relay_is_recognized_as_remote() {
        let relay = serde_json::json!({"backend": "vllm 0.32.0", "pod": {"id": "p"}});
        assert_eq!(remote_backend(&relay).as_deref(), Some("vllm 0.32.0"));
        assert_eq!(remote_backend(&serde_json::json!({"backend": null})), None);
    }

    #[test]
    fn mrun_status_gives_lanes_and_numerics() {
        let (lanes, numerics) = status_facts(&serde_json::json!({
            "lanes": {"configured": 2, "busy": 0},
            "numerics": {"nax": true, "gpu_family": "applegpu_g16s"}
        }));
        assert_eq!(lanes, 2);
        assert_eq!(numerics.nax, "on");
        assert_eq!(numerics.gpu_family.as_deref(), Some("applegpu_g16s"));
        let (_, nuvec) = status_facts(&serde_json::json!({
            "numerics": {"backend": "nuvec", "nax": "off", "gpu_family": "applegpu_g16s"},
            "engine": {"backend": {"name": "nuvec", "version": "0.4.1", "numerics": "nuvec", "kernels": "precompiled"}}
        }));
        assert_eq!(nuvec.backend.as_deref(), Some("nuvec"));
        assert_eq!(nuvec.backend_version.as_deref(), Some("0.4.1"));
        assert_eq!(nuvec.kernels.as_deref(), Some("precompiled"));
        let (lanes, numerics) = status_facts(&serde_json::json!({"version": "0.1.0"}));
        assert_eq!((lanes, numerics), (1, Numerics::default()));
    }

    #[test]
    fn system_specs_and_variants() {
        let default = System::all_bundled();
        let llama = default.iter().find(|s| s.name == "llama.cpp").unwrap();
        assert_eq!(
            llama.url, "http://127.0.0.1:9931",
            "llama-server's default port"
        );
        let s = System::parse("llama.cpp=http://127.0.0.1:9000/,model=q").unwrap();
        assert_eq!(s.url, "http://127.0.0.1:9000");
        assert_eq!(s.profile.name, "llama.cpp");
        assert_eq!(s.ignore_eos(), IgnoreEos::Body);
        assert_eq!(s.describe("b1".into(), 32_768).flags[0], "-c 32768");
        let bf16 = System::parse("splash-bf16=http://127.0.0.1:8000").unwrap();
        assert_eq!(bf16.profile.model.kv_format, "bf16");
        assert!(lossy_kv(
            &Profile::bundled("splash").unwrap().model.kv_format
        ));
        let custom = System::parse("x=http://127.0.0.1:1,apis=chat+responses").unwrap();
        assert_eq!(custom.apis, vec![ApiKind::Chat, ApiKind::Responses]);
        assert!(System::parse("x=ftp://a").is_err());
        assert!(System::parse("x=http://a,colour=red").is_err());
        let all = System::all_bundled();
        assert_eq!(all[0].name, "mrun");
        assert!(all.iter().any(|s| s.name == "splash-bf16"));
        assert_eq!(all.len(), 8);
        // The 0.1 competitors are pinned and verified (T022); Rapid-MLX and Splash are Later and
        // stay unverified; mlx-lm runs synthetic cells only; Rapid-MLX runs agent cells over
        // Messages.
        for s in &all {
            let pinned =
                ["mrun", "llama.cpp", "ollama", "omlx", "mlx-lm"].contains(&s.name.as_str());
            assert_eq!(s.describe(String::new(), 0).verified, pinned, "{}", s.name);
        }
        let mlx_lm = all.iter().find(|s| s.name == "mlx-lm").unwrap();
        assert!(mlx_lm.runs("synthetic") && !mlx_lm.runs("agent"));
        let rapid = all.iter().find(|s| s.name == "rapid-mlx").unwrap();
        assert!(rapid.runs("agent") && rapid.apis.contains(&ApiKind::Messages));
        assert!(Profile::parse("name = \"x\"\nurl = \"http://a\"\napis = []\nsuites = [\"x\"]\n[model]\nformat = \"mlx\"\nkv_format = \"bf16\"").is_err());
    }
}
