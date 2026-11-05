//! `chunk_len` calibration (FR-007, D050): at warm-up the engine times one layer of the
//! memory-bound chunk at a few depths up to the capacity and derives [`TimeLen`], the longest
//! multiple of 128 tokens whose layer takes at most [`LAYER_TIME_BOUND_MS`]. The result is fixed
//! for the process and persisted in `~/.mrun/calibration.toml` per (device, family,
//! quantization, numerics class, model identity), so later starts reuse it and chunk plans stay
//! identical across restarts (007 keys on them).
//!
//! The search and the file need no tensor backend; the engine supplies the timer.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::engine::{CHUNK_ALIGN, LAYER_TIME_BOUND_MS, Numerics, TimeLen, chunk_len};

/// Depths the calibration times, below the capacity; the last reachable chunk start
/// (`capacity − 128`) is added. 4K and 8K are timed so that, where they pass, prompts up to 8K
/// keep the memory-bound 2,048 split (the mlx-lm parity set, SC-001).
pub const DEPTHS: [u32; 8] = [0, 4096, 8192, 16_384, 32_768, 65_536, 131_072, 262_144];

/// The depths to time for a token capacity of `capacity`.
pub fn depths(capacity: u32) -> Vec<u32> {
    let last = capacity.saturating_sub(CHUNK_ALIGN);
    let mut depths: Vec<u32> = DEPTHS.iter().copied().filter(|&d| d < last).collect();
    if last > 0 || depths.is_empty() {
        depths.push(last);
    }
    depths.dedup();
    depths
}

/// Finds [`TimeLen`] with `time_ms(depth, len)`, the time one layer of a `len`-token chunk at
/// `depth` takes. At each depth: the memory-bound chunk when it is fast enough, else the
/// longest multiple of 128 below it that is (binary search, assuming time grows with the
/// length), at least 128.
pub fn search<E>(
    depths: &[u32],
    mut time_ms: impl FnMut(u32, u32) -> Result<f64, E>,
) -> Result<TimeLen, E> {
    let bound = f64::from(LAYER_TIME_BOUND_MS);
    let mut points = Vec::with_capacity(depths.len());
    for &depth in depths {
        let memory = chunk_len(depth);
        if time_ms(depth, memory)? <= bound {
            points.push((depth, memory));
            continue;
        }
        // Largest k in [1, hi] with time(k × 128) ≤ bound; k = 1 when even 128 is too slow.
        let (mut lo, mut hi) = (1, (memory - 1) / CHUNK_ALIGN);
        while lo < hi {
            let mid = lo + (hi - lo).div_ceil(2);
            if time_ms(depth, mid * CHUNK_ALIGN)? <= bound {
                lo = mid;
            } else {
                hi = mid - 1;
            }
        }
        points.push((depth, lo.max(1) * CHUNK_ALIGN));
    }
    Ok(TimeLen::from_points(points))
}

/// What a calibration is valid for.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Key {
    /// GPU name and memory size, e.g. `Apple M4 Pro, 48 GiB`.
    pub device: String,
    pub family: String,
    /// e.g. `affine-4-64`, `dense`.
    pub quantization: String,
    pub numerics: Numerics,
    /// Content identity: two models of one family and quantization differ in size.
    pub identity: String,
}

impl Key {
    fn text(&self) -> String {
        format!(
            "{} | {} | {} | {} | {}",
            self.device,
            self.family,
            self.quantization,
            self.numerics.key(),
            self.identity
        )
    }
}

/// The quantization part of a key from `config.json`: `<mode>-<bits>-<group size>` of the
/// default (`quantization`, else `quantization_config`), `dense` without.
pub fn quantization_key(config: &serde_json::Value) -> String {
    let q = config
        .get("quantization")
        .or_else(|| config.get("quantization_config"));
    match q {
        Some(q) if q.is_object() => format!(
            "{}-{}-{}",
            q.get("mode").and_then(|m| m.as_str()).unwrap_or("affine"),
            q.get("bits").and_then(|b| b.as_u64()).unwrap_or(0),
            q.get("group_size").and_then(|g| g.as_u64()).unwrap_or(0)
        ),
        _ => "dense".into(),
    }
}

/// `$MRUN_HOME/calibration.toml`, else `~/.mrun/calibration.toml`.
pub fn default_path() -> Option<PathBuf> {
    let home = std::env::var_os("MRUN_HOME")
        .filter(|h| !h.is_empty())
        .map(PathBuf::from)
        .or_else(|| {
            std::env::var_os("HOME")
                .filter(|h| !h.is_empty())
                .map(|h| PathBuf::from(h).join(".mrun"))
        })?;
    Some(home.join("calibration.toml"))
}

#[derive(Debug, Default, Deserialize, Serialize)]
struct File {
    #[serde(default)]
    entries: BTreeMap<String, Entry>,
}

#[derive(Debug, Deserialize, Serialize)]
struct Entry {
    /// `[depth, len]` pairs.
    points: Vec<[u32; 2]>,
}

fn read(path: &Path) -> File {
    std::fs::read_to_string(path)
        .ok()
        .and_then(|text| toml::from_str(&text).ok())
        .unwrap_or_default()
}

/// The stored calibration for `key`, if any (an unreadable file counts as empty).
pub fn load(path: &Path, key: &Key) -> Option<TimeLen> {
    let entry = read(path).entries.remove(&key.text())?;
    let points = entry.points.iter().map(|[d, l]| (*d, *l)).collect();
    Some(TimeLen::from_points(points))
}

/// Every stored calibration as `(key text, [(depth, len)])`, in key order (`mrun doctor
/// --report`); an unreadable file counts as empty.
pub fn entries(path: &Path) -> Vec<(String, Vec<(u32, u32)>)> {
    read(path)
        .entries
        .into_iter()
        .map(|(key, entry)| (key, entry.points.iter().map(|[d, l]| (*d, *l)).collect()))
        .collect()
}

/// Stores `time_len` under `key`, keeping other entries (write to a temporary file, then rename).
pub fn store(path: &Path, key: &Key, time_len: &TimeLen) -> std::io::Result<()> {
    let dir = path.parent().unwrap_or(Path::new("."));
    std::fs::create_dir_all(dir)?;
    let mut file = read(path);
    file.entries.insert(
        key.text(),
        Entry {
            points: time_len.points().iter().map(|&(d, l)| [d, l]).collect(),
        },
    );
    let text = toml::to_string(&file).map_err(std::io::Error::other)?;
    let mut temp = tempfile::NamedTempFile::new_in(dir)?;
    std::io::Write::write_all(&mut temp, text.as_bytes())?;
    temp.persist(path).map_err(|e| e.error)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(identity: &str) -> Key {
        Key {
            device: "Apple M4 Pro, 48 GiB".into(),
            family: "qwen3".into(),
            quantization: "affine-4-64".into(),
            numerics: Numerics {
                nax: false,
                gpu_arch: "applegpu_g16s".into(),
            },
            identity: identity.into(),
        }
    }

    #[test]
    fn depths_stop_below_the_capacity() {
        assert_eq!(depths(40_960), [0, 4096, 8192, 16_384, 32_768, 40_832]);
        assert_eq!(
            depths(262_144),
            [0, 4096, 8192, 16_384, 32_768, 65_536, 131_072, 262_016]
        );
        assert_eq!(depths(100), [0]);
        assert_eq!(depths(0), [0]);
    }

    #[test]
    fn a_fast_gpu_keeps_the_memory_bound() {
        let fast = search::<()>(&depths(262_144), |_, _| Ok(5.0)).unwrap();
        let info = crate::ModelInfo {
            time_len: fast,
            ..Default::default()
        };
        for depth in [
            0, 1, 16_384, 20_000, 40_000, 65_536, 100_000, 200_000, 262_000,
        ] {
            assert_eq!(info.chunk_len(depth), chunk_len(depth), "depth {depth}");
        }
    }

    #[test]
    fn a_slow_timer_shortens_chunks() {
        // 0.1 ms per token per layer plus 0.001 ms per token of depth per 128 tokens of chunk.
        let mut calls = 0;
        let slow = search::<()>(&depths(65_536), |depth, len| {
            calls += 1;
            Ok(f64::from(len) * 0.1 + f64::from(depth) * f64::from(len) / 128.0 * 0.0001)
        })
        .unwrap();
        // depth 0: 0.1 ms/token → the 2,048 memory bound is within 250 ms.
        assert_eq!(slow.at(0), 2048);
        let info = crate::ModelInfo {
            time_len: slow,
            ..Default::default()
        };
        for depth in (0..65_536).step_by(1024).chain(depths(65_536)) {
            let len = info.chunk_len(depth);
            assert!(len >= CHUNK_ALIGN);
            let t = f64::from(len) * 0.1 + f64::from(depth) * f64::from(len) / 128.0 * 0.0001;
            assert!(
                t <= f64::from(LAYER_TIME_BOUND_MS),
                "{depth}: {len} takes {t}"
            );
        }
        assert!(calls <= 6 * 5, "binary search, not a scan: {calls}");
        // Hopelessly slow: 128 at every depth.
        let hopeless = search::<()>(&[0, 16_384], |_, _| Ok(1e6)).unwrap();
        assert_eq!(hopeless.points(), [(0, 128), (16_384, 128)]);
    }

    #[test]
    fn calibration_is_stored_per_key_and_reused() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("sub").join("calibration.toml");
        assert_eq!(load(&path, &key("a")), None);
        let a = TimeLen::from_points(vec![(0, 1536), (16_384, 1024)]);
        let b = TimeLen::from_points(vec![(0, 2048)]);
        store(&path, &key("a"), &a).unwrap();
        store(&path, &key("b"), &b).unwrap();
        assert_eq!(load(&path, &key("a")), Some(a));
        assert_eq!(load(&path, &key("b")), Some(b));
        let listed = entries(&path);
        assert_eq!(listed.len(), 2);
        assert_eq!(listed[0].0, key("a").text());
        assert_eq!(listed[0].1, [(0, 1536), (16_384, 1024)]);
        let mut other = key("a");
        other.numerics.nax = true;
        assert_eq!(load(&path, &other), None, "another numerics class");
        // A corrupt file is treated as empty and rewritten.
        std::fs::write(&path, "not [toml").unwrap();
        assert_eq!(load(&path, &key("a")), None);
        store(&path, &key("a"), &TimeLen::unbounded()).unwrap();
        assert_eq!(load(&path, &key("a")), Some(TimeLen::unbounded()));
    }

    #[test]
    fn quantization_keys_read_the_default() {
        let q =
            serde_json::json!({"quantization": {"group_size": 64, "bits": 4, "mode": "affine"}});
        assert_eq!(quantization_key(&q), "affine-4-64");
        assert_eq!(quantization_key(&serde_json::json!({})), "dense");
    }
}
