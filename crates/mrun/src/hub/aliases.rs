//! Curated aliases (D027 as revised by D040 and D048): ordered by size, each pinned to a commit with
//! the blob id of every file `pull` fetches, and the numbers the per-machine default is computed
//! from.
//!
//! Pins were read from `GET https://huggingface.co/api/models/<repo>/revision/<commit>?blobs=true`;
//! the state sizes come from each pinned `config.json` through the family formulas, the vision
//! bytes from the safetensors headers.
//! `qwen3` (Qwen3-8B) is no longer an alias (D040); `mlx-community/Qwen3-8B-4bit` stays pullable
//! by repo id.

use super::{Fit, fit, model_info};
use mrun_engine::budget::{DEFAULT_ALIAS_MIN_TOKENS, DeviceMemory, GIB, SCRATCH_ESTIMATE};

/// A curated alias.
#[derive(Debug)]
pub struct Alias {
    pub name: &'static str,
    pub repo: &'static str,
    /// Pinned commit (40 lowercase hex).
    pub revision: &'static str,
    /// 001 family registry name; `pull` refuses a repo whose `config.json` resolves to another.
    pub family: &'static str,
    pub license: &'static str,
    /// Bytes `pull` downloads.
    pub size: u64,
    /// Bytes of the `*.safetensors` files.
    pub weights: u64,
    /// Tensor bytes of the vision tower inside `weights`; the engine skips them at load, so fit
    /// leaves them out ([`Alias::loaded_weights`]).
    pub vision: u64,
    /// 001 `kv_bytes_per_token` of the pinned `config.json`.
    pub kv_bytes_per_token: u64,
    /// 001 `recurrent_bytes` of the pinned `config.json` (one checkpoint).
    pub recurrent_bytes: u64,
    pub max_position: u32,
    /// Scratch the hub plans a model of the family with before it is loaded (001 measures its own
    /// after warm-up): `scratch + scratch_per_position × depth`. qwen3.5 is fitted to the engine's
    /// two measurements on the M4 Pro (3.26 GiB at a 84,333-token capacity on a simulated 16 GB
    /// budget, 6.06 GiB at 262,144 on 48 GB; 001 research R18); qwen3.6 has one measurement
    /// (5.4 GiB at 262,144), so its per-position part is 0 until a second one exists.
    pub scratch: u64,
    pub scratch_per_position: u64,
    /// Blob id of every file at `revision`: the LFS sha256, else the git blob id.
    pub blobs: &'static [(&'static str, &'static str)],
    /// Set when the alias is below the client gate (D061): `ls`, `show`, `launch` and `doctor`
    /// say so.
    pub preview: Option<Preview>,
}

/// Why an alias ships as a preview, with the client gate's measured figures (002 SC-001).
#[derive(Debug)]
pub struct Preview {
    /// Short reason, e.g. "below the client gate".
    pub reason: &'static str,
    /// `(client, passed, runs)` per client of the gate run.
    pub gate: &'static [(&'static str, u32, u32)],
}

impl Preview {
    /// "Claude Code 27/30, Codex 26/30, OpenCode 28/30".
    pub fn figures(&self) -> String {
        self.gate
            .iter()
            .map(|(client, passed, runs)| format!("{client} {passed}/{runs}"))
            .collect::<Vec<_>>()
            .join(", ")
    }
}

impl Alias {
    /// Weight bytes the engine loads: the files minus the skipped vision tower.
    pub fn loaded_weights(&self) -> u64 {
        self.weights - self.vision
    }

    /// Fit on `device` from the pinned numbers (nothing needs to be downloaded).
    pub fn fit(&self, device: DeviceMemory) -> Fit {
        fit(
            device,
            self.loaded_weights(),
            &model_info(
                self.family,
                self.kv_bytes_per_token,
                self.recurrent_bytes,
                self.max_position,
            ),
        )
    }

    /// The one line `show`, `launch` and `doctor` print for a preview alias, naming the largest
    /// alias that is not a preview (D061).
    pub fn preview_note(&self) -> Option<String> {
        let preview = self.preview.as_ref()?;
        let larger = ALIASES
            .iter()
            .rev()
            .find(|alias| alias.preview.is_none() && alias.weights > self.weights)
            .map(|alias| format!("; larger Macs get {}", alias.name))
            .unwrap_or_default();
        Some(format!(
            "{} is a preview: {} ({}){larger}",
            self.name,
            preview.reason,
            preview.figures()
        ))
    }

    /// The pinned blob id of `path`.
    pub fn blob(&self, path: &str) -> Option<&'static str> {
        self.blobs
            .iter()
            .find(|(file, _)| *file == path)
            .map(|(_, blob)| *blob)
    }
}

/// Context the default alias must leave on this machine (D048: Claude Code's minimum).
pub const DEFAULT_MIN_CONTEXT: u32 = DEFAULT_ALIAS_MIN_TOKENS;

/// Smallest first.
pub static ALIASES: [Alias; 2] = [
    Alias {
        name: "qwen3.5",
        repo: "mlx-community/Qwen3.5-9B-4bit",
        revision: "8b2b98c00a6b4d291155e4890773ca8f769aee53",
        family: "qwen3_5",
        license: "apache-2.0",
        size: 5_977_071_067,
        weights: 5_950_221_072,
        vision: 912_020_960,
        // 8 attention layers × K and V × 4 KV heads × 256 × bf16 = 32 KiB.
        kv_bytes_per_token: 32_768,
        // 24 Gated DeltaNet layers × (f32 state 32 × 128 × 128 + conv tail 3 × 8192 × bf16)
        // ≈ 49 MiB.
        recurrent_bytes: 51_511_296,
        max_position: 262_144,
        scratch: 2_074_483_202,
        scratch_per_position: 16_908,
        // Client gate on the M4 Pro (002 research, D061): every failure was a
        // 600 s timeout of a stuck session (the 9B model loops or rambles), not an mrun fault.
        preview: Some(Preview {
            reason: "below the client gate",
            gate: &[
                ("Claude Code", 27, 30),
                ("Codex", 26, 30),
                ("OpenCode", 28, 30),
            ],
        }),
        blobs: &[
            (
                "chat_template.jinja",
                "a585dec894e63da457d9440ec6aa7caa16d20860",
            ),
            ("config.json", "0435ae3f4b3c8b907f7359a2f8f7150d316f3435"),
            (
                "model-00001-of-00002.safetensors",
                "a68b87558c6ef43f74c2bd63ce7e9092ceddc3101f3def0030774bae5f42aadd",
            ),
            (
                "model-00002-of-00002.safetensors",
                "b0a770bf8469c7f3f18756a0e0283f1c1174344a83e059a4e483f6af4907352d",
            ),
            (
                "model.safetensors.index.json",
                "d366fa524ccd8f4ed93a67c7857597dc56172b92",
            ),
            (
                "preprocessor_config.json",
                "2ea84a437d448ff71b08df68fdd949d5cc4ebb64",
            ),
            (
                "processor_config.json",
                "7ad6acdf4203f22b7b990e36ccc3a1fe38563d5e",
            ),
            (
                "tokenizer.json",
                "87a7830d63fcf43bf241c3c5242e96e62dd3fdc29224ca26fed8ea333db72de4",
            ),
            (
                "tokenizer_config.json",
                "a068e2468cff426a9b105006e74e044030a6faf4",
            ),
            (
                "video_preprocessor_config.json",
                "3ba673a5ad7d4d13f54155ecd38b2a94a6dac8fe",
            ),
            ("vocab.json", "0aa0ce0658d60ac4a5d609f4eadb0e8e43514176"),
        ],
    },
    Alias {
        name: "qwen3.6",
        repo: "mlx-community/Qwen3.6-35B-A3B-4bit",
        revision: "38740b847e4cb78f352aba30aa41c76e08e6eb46",
        family: "qwen3_5_moe",
        license: "apache-2.0",
        size: 20_429_166_969,
        weights: 20_402_204_271,
        vision: 893_142_496,
        kv_bytes_per_token: 20_480,
        recurrent_bytes: 64_389_120,
        max_position: 262_144,
        scratch: 55 * GIB / 10,
        scratch_per_position: 0,
        preview: None,
        blobs: &[
            (
                "chat_template.jinja",
                "a8755d827c0a7b614c246c4060dfd58ab352a8ff",
            ),
            ("config.json", "e3a2334ebf2df216742ef3ed4b784417bfffe6fc"),
            (
                "configuration.json",
                "d24dba949ee1fe70cc810e4c4709a0bddf4e06ba",
            ),
            (
                "generation_config.json",
                "023756cfadf88e5bf69eefeee3e172f38c448d64",
            ),
            (
                "model-00001-of-00004.safetensors",
                "09f3e6ecb0b7af6e6a38bc8169a134c821b0924c2679b2bb8f4426ad38d032b8",
            ),
            (
                "model-00002-of-00004.safetensors",
                "31dcdb1c49eebdb1505bd14e3cb33f9cf900bd2546b638f2464694ae763a033f",
            ),
            (
                "model-00003-of-00004.safetensors",
                "3e66de06a1f03dade16a612a368cfce4a4c9caa4efd7d28185454384082cec03",
            ),
            (
                "model-00004-of-00004.safetensors",
                "a5d0cf03519c26f8b506df6b0ba60526e5c08c8cea22d0c21ce92950e58a5422",
            ),
            (
                "model.safetensors.index.json",
                "f714d295484e01790bad8b40c2ce49323d7d2598",
            ),
            (
                "preprocessor_config.json",
                "2ea84a437d448ff71b08df68fdd949d5cc4ebb64",
            ),
            (
                "processor_config.json",
                "a3be3e79470d0a5befe0ba5247dfad8717d84529",
            ),
            (
                "tokenizer.json",
                "87a7830d63fcf43bf241c3c5242e96e62dd3fdc29224ca26fed8ea333db72de4",
            ),
            (
                "tokenizer_config.json",
                "a068e2468cff426a9b105006e74e044030a6faf4",
            ),
            (
                "video_preprocessor_config.json",
                "3ba673a5ad7d4d13f54155ecd38b2a94a6dac8fe",
            ),
            ("vocab.json", "0aa0ce0658d60ac4a5d609f4eadb0e8e43514176"),
        ],
    },
];

pub fn get(name: &str) -> Option<&'static Alias> {
    ALIASES.iter().find(|alias| alias.name == name)
}

/// The alias a served model id or a typed name refers to: an alias name or its repo id (any case,
/// an `@revision` suffix ignored).
pub fn named(name: &str) -> Option<&'static Alias> {
    let name = name.split('@').next().unwrap_or(name);
    ALIASES.iter().find(|alias| {
        alias.name.eq_ignore_ascii_case(name) || alias.repo.eq_ignore_ascii_case(name)
    })
}

/// [`Alias::preview_note`] of the alias `name` refers to ([`named`]).
pub fn preview_note(name: &str) -> Option<String> {
    named(name).and_then(Alias::preview_note)
}

/// The alias pinned to `repo` at `revision`, if any.
pub fn pinned(repo: &str, revision: &str) -> Option<&'static Alias> {
    ALIASES
        .iter()
        .find(|alias| alias.repo == repo && alias.revision == revision)
}

/// Scratch to plan a model of `family` with before it is loaded: the family's alias's measured
/// scratch, else 001's generic estimate.
pub fn scratch_estimate(family: &str) -> u64 {
    ALIASES
        .iter()
        .find(|alias| alias.family == family)
        .map_or(SCRATCH_ESTIMATE, |alias| alias.scratch)
}

/// The per-position part of [`scratch_estimate`] (0 for families without two measurements).
pub fn scratch_per_position(family: &str) -> u64 {
    ALIASES
        .iter()
        .find(|alias| alias.family == family)
        .map_or(0, |alias| alias.scratch_per_position)
}

/// The largest alias that leaves [`DEFAULT_MIN_CONTEXT`] tokens on `device` with the default
/// budget; `None` when even the smallest does not (E040 on such a Mac). No RAM-size boundary:
/// the fit decides (D048). Memory other processes hold does not count here (D062).
pub fn machine_alias(device: DeviceMemory) -> Option<&'static Alias> {
    let device = device.ignoring_available();
    ALIASES
        .iter()
        .rev()
        .find(|alias| alias.fit(device).max_context >= DEFAULT_MIN_CONTEXT)
}

/// The alias to suggest or show: [`machine_alias`], else the smallest.
pub fn default_alias(device: Option<DeviceMemory>) -> &'static Alias {
    device.and_then(machine_alias).unwrap_or(&ALIASES[0])
}

#[cfg(test)]
mod tests {
    use super::*;
    use mrun_engine::budget::GIB;
    use mrun_engine::models::registry;

    /// Families an alias names before 001 registers them; `pull` of such an alias stops at the
    /// architecture gate until then. Remove an entry once the family lands.
    const PENDING_FAMILIES: [&str; 1] = ["qwen3_5"];

    #[test]
    fn aliases_are_pinned_to_registered_families() {
        let supported = registry::supported();
        let registered: Vec<&str> = supported
            .split(", ")
            .map(|entry| entry.split(' ').next().unwrap_or_default())
            .collect();
        for alias in &ALIASES {
            assert!(
                registered.contains(&alias.family) || PENDING_FAMILIES.contains(&alias.family),
                "{} / {supported}",
                alias.name
            );
            assert!(super::super::is_commit(alias.revision), "{}", alias.name);
            assert!(alias.blob("config.json").is_some(), "{}", alias.name);
            for (path, blob) in alias.blobs {
                let lfs = path.ends_with(".safetensors") || *path == "tokenizer.json";
                let len = if lfs { 64 } else { 40 };
                assert!(
                    blob.len() == len && super::super::is_blob_id(blob),
                    "{path}"
                );
            }
            assert!(alias.weights < alias.size);
            assert!(alias.vision < alias.weights);
        }
        assert!(
            ALIASES
                .windows(2)
                .all(|w| w[0].loaded_weights() < w[1].loaded_weights())
        );
        assert!(get("qwen3").is_none(), "qwen3 is no longer an alias (D040)");
    }

    fn mac(gib: u64, recommended: f64) -> Option<DeviceMemory> {
        Some(DeviceMemory {
            physical: gib * GIB,
            recommended_working_set: (gib as f64 * recommended) as u64 * GIB,
            available: None,
            estimated: false,
        })
    }

    #[test]
    fn the_default_follows_the_machine() {
        assert_eq!(default_alias(None).name, "qwen3.5");
        for gib in [16, 24, 32] {
            assert_eq!(default_alias(mac(gib, 0.67)).name, "qwen3.5", "{gib} GB");
        }
        assert_eq!(default_alias(mac(36, 0.75)).name, "qwen3.6");
        assert_eq!(default_alias(mac(48, 0.78)).name, "qwen3.6");
        assert_eq!(default_alias(mac(64, 0.78)).name, "qwen3.6");
        assert_eq!(
            default_alias(mac(4, 0.67)).name,
            "qwen3.5",
            "the smallest as a suggestion"
        );
        assert!(
            machine_alias(mac(8, 0.67).unwrap()).is_none(),
            "no alias for 8 GB"
        );
        assert!(machine_alias(mac(4, 0.67).unwrap()).is_none());
        // 16 GB leaves Claude Code's minimum for qwen3.5 (D048).
        let fit = get("qwen3.5").unwrap().fit(mac(16, 0.67).unwrap());
        assert!(fit.max_context >= DEFAULT_MIN_CONTEXT, "{fit:?}");
    }

    /// Measured devices (SC-007, T052, research R6): physical memory and Metal's
    /// `recommendedMaxWorkingSetSize` as the engine reports them (`mrun run --info`).
    const M4_PRO_48GB: DeviceMemory = DeviceMemory {
        physical: 51_539_607_552,
        recommended_working_set: 40_200_896_512,
        available: None,
        estimated: false,
    };

    #[test]
    fn measured_devices_pick_the_expected_alias() {
        // 48 GB M4 Pro: qwen3.6 leaves at least the minimum context.
        let pick = machine_alias(M4_PRO_48GB).unwrap();
        assert_eq!(pick.name, "qwen3.6");
        assert!(pick.fit(M4_PRO_48GB).max_context >= DEFAULT_MIN_CONTEXT);
        // Synthetic: below the smallest alias's need nothing is picked.
        let need = ALIASES[0].loaded_weights() + ALIASES[0].scratch;
        let tiny = DeviceMemory {
            physical: need,
            recommended_working_set: need / 2,
            available: None,
            estimated: false,
        };
        assert!(machine_alias(tiny).is_none());
    }

    /// Simulated small Macs (D060, 004 SC-007, research R6): the qualification Mac is the
    /// 48 GB M4 Pro; smaller Macs run there with `--budget-gib` at these public working sets.
    /// `(physical GiB, recommended working set in bytes, expected pick)`.
    const SIMULATED: [(u64, u64, &str); 3] = [
        // 16 GB: llama.cpp ggml-metal log `recommendedMaxWorkingSetSize = 11453.25 MB`.
        (16, 11_453_250_000, "qwen3.5"),
        // 24 GB: ⅔ of physical, the ratio the 16 and 32 GB logs show (no log cited yet).
        (24, 17_179_869_184, "qwen3.5"),
        // 32 GB: llama.cpp ggml-metal log `recommendedMaxWorkingSetSize = 22906.50 MB`.
        (32, 22_906_500_000, "qwen3.5"),
    ];

    #[test]
    fn simulated_small_macs_pick_the_expected_alias() {
        for (gib, working_set, expected) in SIMULATED {
            let device = DeviceMemory {
                physical: gib << 30,
                recommended_working_set: working_set,
                available: None,
                estimated: false,
            };
            let pick = machine_alias(device).unwrap();
            let fit = pick.fit(device);
            assert_eq!(pick.name, expected, "{gib} GB");
            assert!(fit.max_context >= DEFAULT_MIN_CONTEXT, "{gib} GB: {fit:?}");
        }
    }

    /// D048: no RAM-size tier anywhere in `src` outside tests: no `N * GIB` / `N << 30` size
    /// literal of a Mac's memory and no comparison of physical memory with a literal. The one
    /// reference device (`TARGET_48GB`, the data model's `capacity_48gb`) is allowed.
    #[test]
    fn no_ram_size_tier_constants_exist() {
        fn sources(dir: &std::path::Path, out: &mut Vec<(std::path::PathBuf, String)>) {
            for entry in std::fs::read_dir(dir).unwrap() {
                let path = entry.unwrap().path();
                if path.is_dir() {
                    sources(&path, out);
                } else if path.extension().is_some_and(|e| e == "rs") {
                    let text = std::fs::read_to_string(&path).unwrap();
                    // Production code only: unit tests follow `#[cfg(test)]`.
                    let code = text.split("#[cfg(test)]").next().unwrap_or("").to_string();
                    out.push((path, code));
                }
            }
        }
        const SIZES: [&str; 10] = ["8", "16", "18", "24", "32", "36", "48", "64", "96", "128"];
        let tier = |line: &str| {
            let line = line.split("//").next().unwrap_or("");
            let size_literal = SIZES.iter().any(|n| {
                [
                    format!("{n} * GIB"),
                    format!("{n} * budget::GIB"),
                    format!("{n} << 30"),
                    format!("{n}_000_000_000"),
                ]
                .iter()
                .any(|p| {
                    line.match_indices(p.as_str()).any(|(at, _)| {
                        !line[..at].ends_with(|c: char| c.is_ascii_alphanumeric() || c == '_')
                    })
                })
            });
            let physical_compare = line.contains("physical")
                && [" < ", " > ", " <= ", " >= "].iter().any(|op| {
                    line.split(op).nth(1).is_some_and(|rhs| {
                        rhs.trim_start().starts_with(|c: char| c.is_ascii_digit())
                    })
                });
            (size_literal || physical_compare) && !line.contains("TARGET_48GB")
        };
        let mut files = Vec::new();
        sources(
            &std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src"),
            &mut files,
        );
        let mut found = Vec::new();
        for (path, code) in &files {
            let mut lines = code.lines().enumerate().peekable();
            while let Some((n, line)) = lines.next() {
                // The reference device's own initializer.
                if line.contains("pub const TARGET_48GB") {
                    for (_, line) in lines.by_ref() {
                        if line.trim() == "};" {
                            break;
                        }
                    }
                    continue;
                }
                if tier(line) {
                    found.push(format!("{}:{}: {}", path.display(), n + 1, line.trim()));
                }
            }
        }
        assert!(
            found.is_empty(),
            "RAM-size tiers (D048):\n{}",
            found.join("\n")
        );
        assert!(
            tier("if physical >= 36 * GIB {"),
            "the check catches a tier"
        );
    }

    #[test]
    fn the_small_tier_is_a_preview() {
        let note = "qwen3.5 is a preview: below the client gate (Claude Code 27/30, Codex 26/30, \
                    OpenCode 28/30); larger Macs get qwen3.6";
        assert_eq!(
            get("qwen3.5").unwrap().preview_note().as_deref(),
            Some(note)
        );
        assert_eq!(preview_note("qwen3.5").as_deref(), Some(note));
        assert_eq!(
            preview_note("mlx-community/qwen3.5-9b-4bit@main").as_deref(),
            Some(note)
        );
        assert!(get("qwen3.6").unwrap().preview.is_none());
        assert!(preview_note("qwen3.6").is_none());
        assert!(preview_note("org/other").is_none());
        // The release gate covers the largest alias (D061).
        assert!(ALIASES.last().unwrap().preview.is_none());
    }

    #[test]
    fn pins_are_found_by_repo_and_revision() {
        let small = get("qwen3.5").unwrap();
        assert_eq!(pinned(small.repo, small.revision).unwrap().name, "qwen3.5");
        assert!(pinned(small.repo, &"a".repeat(40)).is_none());
        assert_eq!(small.loaded_weights(), 5_038_200_112);
    }
}
