//! Curated aliases (FR-002, research R4): one per 0.1 family, each pinned to a repository and
//! revision.
//!
//! The pins are refreshed by a maintainer (`cargo xtask refresh-aliases`, not built yet). The hub
//! was unreachable when this table was written, so every entry is `verified: false`: it tracks
//! `main` and has no pinned file hashes. `pull` then pins the commit it resolved in the manifest.

/// A curated alias.
#[derive(Debug)]
pub struct Alias {
    pub name: &'static str,
    pub repo: &'static str,
    /// Commit (40 hex) when verified; a branch until then.
    pub revision: &'static str,
    /// Whether the revision and hashes were pinned against the hub.
    pub verified: bool,
    /// 001 family registry name; `pull` refuses a repo whose `config.json` resolves to another.
    pub family: &'static str,
    /// Expected sha256 per file at `revision`; checked by `pull` when not empty.
    pub sha256: &'static [(&'static str, &'static str)],
}

/// Used when nothing is named (`launch` with nothing pulled). Set from 005 T012a; `qwen3.6`
/// until then.
pub const DEFAULT: &str = "qwen3.6";

pub static ALIASES: [Alias; 3] = [
    // TODO(004 T005): pin revision, hashes and fit numbers once the hub is reachable.
    Alias {
        name: "qwen3.6",
        repo: "mlx-community/Qwen3.6-35B-A3B-4bit",
        revision: "main",
        verified: false,
        family: "qwen3_5_moe",
        sha256: &[],
    },
    // TODO(004 T005): pin revision and hashes.
    Alias {
        name: "qwen3-coder",
        repo: "mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit",
        revision: "main",
        verified: false,
        family: "qwen3_moe",
        sha256: &[],
    },
    // TODO(004 T005a): research R4 marks the repo's contents unverified (MXFP4 experts,
    // `quantization.mode`); pin revision and hashes after checking `config.json`.
    Alias {
        name: "gpt-oss",
        repo: "mlx-community/gpt-oss-20b-MXFP4-Q8",
        revision: "main",
        verified: false,
        family: "gpt_oss",
        sha256: &[],
    },
];

pub fn get(name: &str) -> Option<&'static Alias> {
    ALIASES.iter().find(|alias| alias.name == name)
}

#[cfg(test)]
mod tests {
    use super::*;
    use mrun_engine::models::registry;

    #[test]
    fn every_alias_names_a_registered_family_that_lists_it() {
        assert!(get(DEFAULT).is_some());
        let supported = registry::supported();
        for alias in &ALIASES {
            assert!(
                supported.contains(&format!("{} ({})", alias.family, alias.name)),
                "{} / {supported}",
                alias.name
            );
            assert!(alias.repo.contains('/'));
            let pinned =
                alias.revision.len() == 40 && alias.revision.bytes().all(|b| b.is_ascii_hexdigit());
            assert_eq!(
                alias.verified, pinned,
                "{}: verified means pinned",
                alias.name
            );
        }
    }
}
