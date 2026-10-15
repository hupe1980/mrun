//! Memory budget: what mrun may use and how it splits into weights, scratch and sequence state.
//!
//! Pure arithmetic over [`DeviceMemory`]. The MLX engine fills it from `device_info`
//! (`memory_size`, `max_recommended_working_set_size`); Linux builds read `/proc/meminfo`.

use crate::engine::{MAX_PLAN_CHECKPOINTS, MemoryBudget, ModelInfo};

pub const GIB: u64 = 1 << 30;
/// Default share of physical memory mrun plans with.
pub const DEFAULT_FRACTION: f64 = 0.75;
/// Attention state the pre-load check requires room for, in tokens.
pub const INITIAL_STATE_TOKENS: u64 = 4096;
/// Scratch assumed by the pre-load check before the probe has measured it.
pub const SCRATCH_ESTIMATE: u64 = 2 * GIB;

/// Physical memory and the GPU's recommended working set, in bytes.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DeviceMemory {
    pub physical: u64,
    pub recommended_working_set: u64,
}

impl DeviceMemory {
    /// From `/proc/meminfo` text: `MemTotal`; the working set is all of it (no GPU limit).
    pub fn from_meminfo(text: &str) -> Option<Self> {
        let kib: u64 = text
            .lines()
            .find_map(|line| line.strip_prefix("MemTotal:"))?
            .trim()
            .strip_suffix("kB")?
            .trim()
            .parse()
            .ok()?;
        let physical = kib.checked_mul(1024)?;
        Some(Self {
            physical,
            recommended_working_set: physical,
        })
    }

    /// This machine's memory on Linux.
    #[cfg(target_os = "linux")]
    pub fn read_linux() -> std::io::Result<Self> {
        let text = std::fs::read_to_string("/proc/meminfo")?;
        Self::from_meminfo(&text).ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "no MemTotal line in /proc/meminfo",
            )
        })
    }
}

/// User overrides of the automatic budget.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Overrides {
    /// Share of physical memory (default [`DEFAULT_FRACTION`]).
    pub fraction: Option<f64>,
    /// Bytes kept free for other applications (default [`default_reserve`]).
    pub reserve: Option<u64>,
    /// Permits a budget above the automatic one.
    pub allow_over: bool,
}

#[derive(Clone, Debug, PartialEq, thiserror::Error)]
pub enum BudgetError {
    #[error("the memory fraction must be in (0, 1], got {0}")]
    InvalidFraction(f64),
    #[error(
        "a budget of {requested} bytes exceeds the automatic {automatic} bytes\n  hint: pass --allow-over to use it anyway"
    )]
    AboveAutomatic { requested: u64, automatic: u64 },
    #[error(
        "no memory left for mrun: the reserve of {reserve} bytes covers the {usable} usable bytes"
    )]
    NoRoom { usable: u64, reserve: u64 },
    #[error(
        "the model does not fit the budget of {budget} bytes: weights {weights} + scratch {scratch} + position-bound state {recurrent} + {INITIAL_STATE_TOKENS} tokens of attention state {initial_state}"
    )]
    DoesNotFit {
        budget: u64,
        weights: u64,
        scratch: u64,
        recurrent: u64,
        initial_state: u64,
    },
}

/// Memory kept free by default: `clamp(physical ÷ 6, 2 GiB, 8 GiB)`.
pub fn default_reserve(physical: u64) -> u64 {
    (physical / 6).clamp(2 * GIB, 8 * GIB)
}

/// The top-level split: physical, working set, reserve and the budget
/// `min(fraction × physical, working set) − reserve`.
pub fn plan(device: DeviceMemory, overrides: Overrides) -> Result<MemoryBudget, BudgetError> {
    let fraction = overrides.fraction.unwrap_or(DEFAULT_FRACTION);
    if !(fraction > 0.0 && fraction <= 1.0) {
        return Err(BudgetError::InvalidFraction(fraction));
    }
    let usable = |fraction: f64| {
        ((device.physical as f64 * fraction) as u64).min(device.recommended_working_set)
    };
    let automatic_reserve = default_reserve(device.physical);
    let automatic = usable(DEFAULT_FRACTION).saturating_sub(automatic_reserve);
    let reserve = overrides.reserve.unwrap_or(automatic_reserve);
    let usable = usable(fraction);
    if reserve >= usable {
        return Err(BudgetError::NoRoom { usable, reserve });
    }
    let budget = usable - reserve;
    if budget > automatic && !overrides.allow_over {
        return Err(BudgetError::AboveAutomatic {
            requested: budget,
            automatic,
        });
    }
    Ok(MemoryBudget {
        physical: device.physical,
        recommended_working_set: device.recommended_working_set,
        reserve,
        budget,
        ..MemoryBudget::default()
    })
}

/// Position-bound bytes the state capacity holds back: the live state plus the checkpoints one
/// prefill call may capture.
pub fn position_bound_reserve(recurrent_bytes: u64) -> u64 {
    recurrent_bytes * (1 + MAX_PLAN_CHECKPOINTS as u64)
}

/// Pre-load check (weights from file sizes, scratch estimated): the budget must hold the weights,
/// scratch, the position-bound state and [`INITIAL_STATE_TOKENS`] of attention state.
pub fn check_fits(
    budget: &MemoryBudget,
    weights: u64,
    scratch: u64,
    info: &ModelInfo,
) -> Result<(), BudgetError> {
    let initial_state = INITIAL_STATE_TOKENS * info.kv_bytes_per_token;
    let need = weights + scratch + info.recurrent_bytes + initial_state;
    if need > budget.budget {
        return Err(BudgetError::DoesNotFit {
            budget: budget.budget,
            weights,
            scratch,
            recurrent: info.recurrent_bytes,
            initial_state,
        });
    }
    Ok(())
}

/// Depth the scratch probe runs at: the deepest position the state could reach with the scratch
/// estimate, capped at the model's maximum position.
pub fn probe_depth(budget: &MemoryBudget, weights: u64, info: &ModelInfo) -> u32 {
    let room = budget
        .budget
        .saturating_sub(weights + SCRATCH_ESTIMATE + position_bound_reserve(info.recurrent_bytes));
    tokens(room, info).min(info.max_position)
}

/// Completes `budget` with measured weights and probed scratch: state capacity in bytes and
/// tokens, never above the maximum position or the probed depth.
pub fn with_state_capacity(
    budget: MemoryBudget,
    weights: u64,
    scratch: u64,
    probed_depth: u32,
    info: &ModelInfo,
) -> Result<MemoryBudget, BudgetError> {
    check_fits(&budget, weights, scratch, info)?;
    let state_capacity_bytes = budget.budget - weights - scratch;
    let attention =
        state_capacity_bytes.saturating_sub(position_bound_reserve(info.recurrent_bytes));
    Ok(MemoryBudget {
        weights,
        scratch,
        probed_depth,
        state_capacity_bytes,
        kv_capacity_tokens: tokens(attention, info)
            .min(info.max_position)
            .min(probed_depth),
        ..budget
    })
}

fn tokens(bytes: u64, info: &ModelInfo) -> u32 {
    (bytes / info.kv_bytes_per_token.max(1)).min(u64::from(u32::MAX)) as u32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::LayerKind;

    const KIB: u64 = 1024;
    const MIB: u64 = 1024 * KIB;

    fn mac(gib: u64) -> DeviceMemory {
        DeviceMemory {
            physical: gib * GIB,
            recommended_working_set: gib * GIB,
        }
    }

    fn model(kv_bytes_per_token: u64, recurrent_bytes: u64, max_position: u32) -> ModelInfo {
        ModelInfo {
            identity: "test".into(),
            family: "test".into(),
            layer_kinds: vec![LayerKind::FullAttention],
            kv_heads: 1,
            head_dim: 1,
            vocab: 1,
            max_position,
            kv_bytes_per_token,
            recurrent_bytes,
            eos_tokens: Vec::new(),
        }
    }

    #[test]
    fn reserve_is_a_sixth_clamped_to_two_and_eight_gib() {
        assert_eq!(default_reserve(8 * GIB), 2 * GIB);
        assert_eq!(default_reserve(16 * GIB), 16 * GIB / 6);
        assert_eq!(default_reserve(48 * GIB), 8 * GIB);
        assert_eq!(default_reserve(128 * GIB), 8 * GIB);
    }

    #[test]
    fn automatic_budget_on_16_48_and_128_gb() {
        for (gib, budget) in [
            (16, 12 * GIB - 16 * GIB / 6),
            (48, 28 * GIB),
            (128, 88 * GIB),
        ] {
            let plan = plan(mac(gib), Overrides::default()).unwrap();
            assert_eq!(plan.budget, budget, "{gib} GB");
            assert_eq!(plan.physical, gib * GIB);
        }
        // The GPU working set caps the fraction.
        let capped = DeviceMemory {
            physical: 48 * GIB,
            recommended_working_set: 30 * GIB,
        };
        assert_eq!(plan(capped, Overrides::default()).unwrap().budget, 22 * GIB);
    }

    #[test]
    fn overrides_lower_freely_and_raise_only_when_allowed() {
        let lower = Overrides {
            fraction: Some(0.5),
            reserve: Some(10 * GIB),
            allow_over: false,
        };
        assert_eq!(plan(mac(48), lower).unwrap().budget, 14 * GIB);
        let higher = Overrides {
            reserve: Some(4 * GIB),
            ..Overrides::default()
        };
        assert_eq!(
            plan(mac(48), higher),
            Err(BudgetError::AboveAutomatic {
                requested: 32 * GIB,
                automatic: 28 * GIB
            })
        );
        let allowed = Overrides {
            allow_over: true,
            ..higher
        };
        assert_eq!(plan(mac(48), allowed).unwrap().budget, 32 * GIB);
        for fraction in [0.0, 1.5, f64::NAN] {
            let bad = Overrides {
                fraction: Some(fraction),
                ..Overrides::default()
            };
            assert!(matches!(
                plan(mac(48), bad),
                Err(BudgetError::InvalidFraction(_))
            ));
        }
        let no_room = Overrides {
            reserve: Some(40 * GIB),
            ..Overrides::default()
        };
        assert!(matches!(
            plan(mac(48), no_room),
            Err(BudgetError::NoRoom { .. })
        ));
    }

    #[test]
    fn meminfo_total_is_physical_memory() {
        let text = "MemTotal:       16318412 kB\nMemFree:         1000 kB\n";
        let device = DeviceMemory::from_meminfo(text).unwrap();
        assert_eq!(device.physical, 16_318_412 * 1024);
        assert_eq!(device.recommended_working_set, device.physical);
        assert_eq!(DeviceMemory::from_meminfo("MemFree: 1 kB"), None);
        #[cfg(target_os = "linux")]
        assert!(DeviceMemory::read_linux().unwrap().physical > 0);
    }

    #[test]
    fn state_capacity_is_bounded_by_bytes_max_position_and_probed_depth() {
        let budget = plan(mac(48), Overrides::default()).unwrap();
        let info = model(100 * KIB, 0, 1_000_000);
        let full = with_state_capacity(budget, 16 * GIB, 2 * GIB, 1_000_000, &info).unwrap();
        assert_eq!(full.state_capacity_bytes, 10 * GIB);
        assert_eq!(full.kv_capacity_tokens, (10 * GIB / (100 * KIB)) as u32);
        let probed = with_state_capacity(budget, 16 * GIB, 2 * GIB, 50_000, &info).unwrap();
        assert_eq!(probed.kv_capacity_tokens, 50_000);
        let short = model(100 * KIB, 0, 4096);
        let capped = with_state_capacity(budget, 16 * GIB, 2 * GIB, 1_000_000, &short).unwrap();
        assert_eq!(capped.kv_capacity_tokens, 4096);
        // Position-bound state for the live sequence and three checkpoints comes off first.
        let hybrid = model(100 * KIB, 256 * MIB, 1_000_000);
        let held = with_state_capacity(budget, 16 * GIB, 2 * GIB, 1_000_000, &hybrid).unwrap();
        assert_eq!(
            held.kv_capacity_tokens,
            ((10 * GIB - GIB) / (100 * KIB)) as u32
        );
    }

    #[test]
    fn a_model_that_does_not_fit_is_refused_with_every_number() {
        let budget = plan(mac(16), Overrides::default()).unwrap();
        let info = model(144 * KIB, 0, 40_960);
        let error = check_fits(&budget, 9 * GIB, SCRATCH_ESTIMATE, &info).unwrap_err();
        assert_eq!(
            error,
            BudgetError::DoesNotFit {
                budget: budget.budget,
                weights: 9 * GIB,
                scratch: 2 * GIB,
                recurrent: 0,
                initial_state: 4096 * 144 * KIB,
            }
        );
        assert!(check_fits(&budget, 4 * GIB, SCRATCH_ESTIMATE, &info).is_ok());
    }

    #[test]
    fn probe_depth_uses_the_scratch_estimate_and_the_maximum_position() {
        let budget = plan(mac(48), Overrides::default()).unwrap();
        let info = model(96 * KIB, 0, 262_144);
        let depth = probe_depth(&budget, 16 * GIB, &info);
        assert_eq!(u64::from(depth), 10 * GIB / (96 * KIB));
        assert_eq!(
            probe_depth(&budget, 4 * GIB, &model(144 * KIB, 0, 40_960)),
            40_960
        );
    }
}
