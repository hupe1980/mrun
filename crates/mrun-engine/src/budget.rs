//! Memory budget (D062): what mrun may use and how it splits into weights, scratch and sequence
//! state.
//!
//! `budget = min(recommended working set, physical − 2 GiB, available − margin)`, lowered by
//! `--budget-gib` (it cannot raise it). The working set is the device's figure (a raised system
//! wired limit raises it too); available is what macOS can hand out without swapping at startup
//! ([`crate::sysmem::available`]), so memory other processes and VMs hold counts; the margin is
//! [`margin`]. mrun's own accounting enforces the budget: every allocation of sequence state is
//! checked against `state_capacity_bytes`. Scratch terms that grow with the position are taken at
//! the capacity-limited depth, not at the family's maximum position: capacity and scratch are
//! solved together ([`with_state_capacity`]). Pure arithmetic over [`DeviceMemory`]; the MLX
//! engine and the hub fill it from MLX `device_info` (Metal's `recommendedMaxWorkingSetSize`)
//! and `vm_stat`; only without a Metal device does the hub estimate the working set
//! (`DeviceMemory::estimated`).

pub use crate::engine::Binding;
use crate::engine::{MemoryBudget, ModelInfo};

pub const GIB: u64 = 1 << 30;
/// Attention state the pre-load check requires room for, in tokens.
pub const INITIAL_STATE_TOKENS: u64 = 4096;
/// Scratch assumed before a model is loaded (`pull`, `ls`, the pre-load check); the engine
/// computes its own from the model after warm-up.
pub const SCRATCH_ESTIMATE: u64 = GIB;
/// Checkpoints of position-bound state the capacity always leaves room for (beside the live state).
pub const RESERVED_CHECKPOINTS: u64 = 2;
/// Memory the budget always leaves to the system, whatever the working set says.
pub const SYSTEM_GUARD: u64 = 2 * GIB;
/// Tokens of state the default alias must leave on a machine (D048).
pub const DEFAULT_ALIAS_MIN_TOKENS: u32 = 64 * 1024;

/// Physical memory, the GPU's recommended working set and the memory available now, in bytes.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DeviceMemory {
    pub physical: u64,
    pub recommended_working_set: u64,
    /// What macOS can hand out without swapping (free + inactive + purgeable + speculative
    /// pages, D062); `None` where it is not read (not macOS, or a fixed test device): the budget
    /// then follows D048.
    pub available: Option<u64>,
    /// The working set is an estimate, not the Metal device's own figure (no Metal device:
    /// Linux, or a build without MLX); outputs say so.
    pub estimated: bool,
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
            available: None,
            estimated: true,
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

impl DeviceMemory {
    /// This device without the available-memory term: what the machine offers when nothing
    /// else runs. The default alias is picked from it, so other apps change the fit and the
    /// refusal, not which model is the default.
    pub fn ignoring_available(self) -> Self {
        Self {
            available: None,
            ..self
        }
    }

    /// `available − margin`: what the available memory leaves to mrun; `None` when unknown.
    pub fn available_room(&self) -> Option<u64> {
        self.available
            .map(|available| available.saturating_sub(margin(self.physical)))
    }
}

/// The share of available memory left to the system and other apps:
/// `clamp(physical ÷ 8, 2 GiB, 6 GiB)` (D062).
pub fn margin(physical: u64) -> u64 {
    (physical / 8).clamp(SYSTEM_GUARD, 3 * SYSTEM_GUARD)
}

/// User overrides of the automatic budget.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Overrides {
    /// Lowers the budget to this many bytes (`--budget-gib`); it cannot raise it.
    pub budget: Option<u64>,
}

#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
pub enum BudgetError {
    #[error(
        "a budget of {} exceeds the automatic {} (bound by {binding})\n  hint: --budget-gib can only lower the budget",
        gib(*.requested),
        gib(*.automatic)
    )]
    AboveAutomatic {
        requested: u64,
        automatic: u64,
        binding: Binding,
    },
    #[error(
        "no memory left for mrun: {} of memory leave nothing above the system's share",
        gib(*.physical)
    )]
    NoRoom { physical: u64 },
    #[error(
        "no memory left for mrun: {} are available, less than the {} left to other apps\n  hint: quit other apps or VMs ('mrun doctor' lists the largest)",
        gib(*.available),
        gib(*.margin)
    )]
    NoneAvailable { available: u64, margin: u64 },
    #[error(
        "the model does not fit the memory budget of {}: weights {} + scratch {} + position-bound state {} + attention state for {INITIAL_STATE_TOKENS} tokens {} = {}",
        gib(*.budget),
        gib(*.weights),
        gib(*.scratch),
        gib(*.recurrent),
        gib(*.initial_state),
        gib(.weights + .scratch + .recurrent + .initial_state)
    )]
    DoesNotFit {
        budget: u64,
        weights: u64,
        scratch: u64,
        recurrent: u64,
        initial_state: u64,
    },
}

/// `bytes` as GiB with two decimals (`4.30 GiB`), as refusals print them.
pub fn gib(bytes: u64) -> String {
    format!("{:.2} GiB", bytes as f64 / GIB as f64)
}

/// The automatic budget and its binding term: `min(recommended working set, physical −
/// SYSTEM_GUARD, available − margin)`; ties go to the earlier term.
pub fn automatic(device: DeviceMemory) -> (u64, Binding) {
    let mut terms = vec![
        (device.recommended_working_set, Binding::WorkingSet),
        (
            device.physical.saturating_sub(SYSTEM_GUARD),
            Binding::Physical,
        ),
    ];
    if let Some(room) = device.available_room() {
        terms.push((room, Binding::Available));
    }
    terms
        .into_iter()
        .reduce(|best, term| if term.0 < best.0 { term } else { best })
        .unwrap_or_default()
}

/// The top-level budget: [`automatic`], lowered by `overrides.budget`.
pub fn plan(device: DeviceMemory, overrides: Overrides) -> Result<MemoryBudget, BudgetError> {
    let (automatic, binding) = automatic(device);
    if automatic == 0 {
        return Err(match (binding, device.available) {
            (Binding::Available, Some(available)) => BudgetError::NoneAvailable {
                available,
                margin: margin(device.physical),
            },
            _ => BudgetError::NoRoom {
                physical: device.physical,
            },
        });
    }
    let (budget, binding) = match overrides.budget {
        Some(requested) if requested > automatic => {
            return Err(BudgetError::AboveAutomatic {
                requested,
                automatic,
                binding,
            });
        }
        Some(requested) if requested < automatic => (requested, Binding::Flag),
        _ => (automatic, binding),
    };
    Ok(MemoryBudget {
        physical: device.physical,
        recommended_working_set: device.recommended_working_set,
        available: device.available,
        budget,
        binding,
        ..MemoryBudget::default()
    })
}

/// Whether the weights may be wired (D062): only when they fit the available memory less the
/// margin, measured before the load; without an available figure, as D029.
pub fn may_wire(device: DeviceMemory, weights: u64) -> bool {
    device.available_room().is_none_or(|room| weights <= room)
}

/// Position-bound bytes the state capacity holds back: the live state plus
/// [`RESERVED_CHECKPOINTS`] checkpoints.
pub fn position_bound_reserve(recurrent_bytes: u64) -> u64 {
    recurrent_bytes * (1 + RESERVED_CHECKPOINTS)
}

/// Pre-load check: the budget must hold the weights, scratch, the position-bound reserve (as
/// the capacity holds it back) and [`INITIAL_STATE_TOKENS`] of attention state.
pub fn check_fits(
    budget: &MemoryBudget,
    weights: u64,
    scratch: u64,
    info: &ModelInfo,
) -> Result<(), BudgetError> {
    let initial_state = INITIAL_STATE_TOKENS * info.kv_bytes_per_token;
    let recurrent = position_bound_reserve(info.recurrent_bytes);
    let need = weights + scratch + recurrent + initial_state;
    if need > budget.budget {
        return Err(BudgetError::DoesNotFit {
            budget: budget.budget,
            weights,
            scratch,
            recurrent,
            initial_state,
        });
    }
    Ok(())
}

/// Completes `budget` with the weights and the scratch, which may grow with the depth a sequence
/// reaches (`scratch(depth)`, non-decreasing): the largest token capacity `c` (at most the
/// model's maximum position) whose attention state fits beside `scratch(c)`. That is the fixed
/// point of `c = tokens(budget − weights − scratch(c))`, found by bisection since the fit is
/// monotone in `c`. The state capacity in bytes is what remains beside `scratch(c)`.
pub fn with_state_capacity(
    budget: MemoryBudget,
    weights: u64,
    scratch: &dyn Fn(u32) -> u64,
    info: &ModelInfo,
) -> Result<MemoryBudget, BudgetError> {
    let initial = (INITIAL_STATE_TOKENS as u32).min(info.max_position);
    check_fits(&budget, weights, scratch(initial), info)?;
    let reserve = position_bound_reserve(info.recurrent_bytes);
    let fits = |depth: u32| {
        let need = weights
            .saturating_add(scratch(depth))
            .saturating_add(reserve)
            .saturating_add(u64::from(depth) * info.kv_bytes_per_token);
        need <= budget.budget
    };
    // fits(initial) holds by the check above; find the largest depth that fits.
    let (mut lo, mut hi) = (initial, info.max_position);
    while lo < hi {
        let mid = lo + (hi - lo).div_ceil(2);
        if fits(mid) {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    let scratch = scratch(lo);
    let state_capacity_bytes = budget.budget - weights - scratch;
    let attention = state_capacity_bytes.saturating_sub(reserve);
    Ok(MemoryBudget {
        weights,
        scratch,
        scratch_depth: lo,
        state_capacity_bytes,
        kv_capacity_tokens: lo.min(tokens(attention, info)),
        ..budget
    })
}

/// Scratch as a fixed part plus a part per position of depth: the form the hub's fit check uses
/// for an alias it has not loaded (its measured scratch, plus one attention layer per position).
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct LinearScratch {
    pub fixed: u64,
    pub per_position: u64,
}

impl LinearScratch {
    pub fn at(&self, depth: u32) -> u64 {
        self.fixed + self.per_position * u64::from(depth)
    }
}

/// Token capacity of a model on `device` before loading it: the budget from `overrides`, the
/// weights and the scratch as [`with_state_capacity`] solves them. `Err` when it does not fit.
pub fn capacity_on(
    device: DeviceMemory,
    overrides: Overrides,
    weights: u64,
    scratch: LinearScratch,
    info: &ModelInfo,
) -> Result<MemoryBudget, BudgetError> {
    with_state_capacity(plan(device, overrides)?, weights, &|d| scratch.at(d), info)
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

    fn mac(gib: u64, working_set: f64) -> DeviceMemory {
        DeviceMemory {
            physical: gib * GIB,
            recommended_working_set: (gib as f64 * working_set * GIB as f64) as u64,
            available: None,
            estimated: false,
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
            ..ModelInfo::default()
        }
    }

    fn fixed(bytes: u64) -> impl Fn(u32) -> u64 {
        move |_| bytes
    }

    #[test]
    fn budget_is_the_working_set_below_the_system_guard() {
        // 16 GB: the working set (⅔) binds; 48 GB as MLX reports it: 37.4 GiB.
        let small = plan(mac(16, 0.67), Overrides::default()).unwrap();
        assert_eq!(small.budget, mac(16, 0.67).recommended_working_set);
        let m4_pro = DeviceMemory {
            physical: 48 * GIB,
            recommended_working_set: 40_200_896_512,
            available: None,
            estimated: false,
        };
        assert_eq!(
            plan(m4_pro, Overrides::default()).unwrap().budget,
            40_200_896_512
        );
        // A raised wired limit raises the working set, never above physical − 2 GiB.
        let raised = plan(mac(32, 1.0), Overrides::default()).unwrap();
        assert_eq!(raised.budget, 30 * GIB);
        assert_eq!(raised.physical, 32 * GIB);
        assert_eq!(
            plan(mac(2, 1.0), Overrides::default()),
            Err(BudgetError::NoRoom { physical: 2 * GIB })
        );
    }

    /// D048 across the device set: the budget is the working set Metal reports (≈ ⅔ of
    /// physical below 36 GB, ≈ ¾ from 36 GB, measured 40.2 GB on the 48 GB M4 Pro), and with a
    /// raised `iogpu.wired_limit_mb` (Metal then reports a larger working set) never more than
    /// `physical − 2 GiB`. No device size is special-cased.
    #[test]
    fn budget_follows_the_working_set_on_every_device() {
        let devices: [(u64, f64); 6] = [
            (8, 2.0 / 3.0),
            (16, 2.0 / 3.0),
            (24, 2.0 / 3.0),
            (36, 0.75),
            (48, 40_200_896_512.0 / (48.0 * GIB as f64)),
            (128, 0.75),
        ];
        for (gib, fraction) in devices {
            let device = mac(gib, fraction);
            let plan = plan(device, Overrides::default()).unwrap();
            assert_eq!(plan.budget, device.recommended_working_set, "{gib} GB");
            assert!(plan.budget <= device.physical - SYSTEM_GUARD, "{gib} GB");
            // A raised wired limit: Metal's working set reaches (almost) all of memory.
            for raised in [device.physical - GIB, device.physical] {
                let raised = DeviceMemory {
                    recommended_working_set: raised,
                    ..device
                };
                let plan = super::plan(raised, Overrides::default()).unwrap();
                assert_eq!(
                    plan.budget,
                    device.physical - SYSTEM_GUARD,
                    "{gib} GB raised"
                );
            }
            // The override lowers it on every device and never raises it.
            let half = Overrides {
                budget: Some(device.recommended_working_set / 2),
            };
            assert_eq!(
                super::plan(device, half).unwrap().budget,
                device.recommended_working_set / 2
            );
        }
        assert_eq!(mac(48, 0.78).physical, 48 * GIB);
    }

    /// D048: no RAM-size boundary in the engine's production code (`mrun`'s
    /// `no_ram_size_tier_constants_exist` covers the CLI): no `N * GIB` / `N << 30` literal of a
    /// Mac's memory size and no comparison of `physical` with a literal. `testing.rs` (test
    /// support) and unit tests are exempt.
    #[test]
    fn no_ram_size_constant_in_the_engine() {
        const SIZES: [&str; 10] = ["8", "16", "18", "24", "32", "36", "48", "64", "96", "128"];
        let tier = |line: &str| {
            let line = line.split("//").next().unwrap_or("");
            let literal = SIZES.iter().any(|n| {
                [format!("{n} * GIB"), format!("{n} << 30")]
                    .iter()
                    .any(|p| {
                        line.match_indices(p.as_str()).any(|(at, _)| {
                            !line[..at].ends_with(|c: char| c.is_ascii_alphanumeric() || c == '_')
                        })
                    })
            });
            let compare = line.contains("physical")
                && [" < ", " > ", " <= ", " >= "].iter().any(|op| {
                    line.split(op).nth(1).is_some_and(|rhs| {
                        rhs.trim_start().starts_with(|c: char| c.is_ascii_digit())
                    })
                });
            literal || compare
        };
        assert!(tier("if physical >= 36 * GIB {"));
        let mut stack = vec![std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src")];
        let mut found = Vec::new();
        while let Some(dir) = stack.pop() {
            for entry in std::fs::read_dir(dir).unwrap() {
                let path = entry.unwrap().path();
                if path.is_dir() {
                    stack.push(path);
                } else if path.extension().is_some_and(|e| e == "rs")
                    && !path.ends_with("testing.rs")
                {
                    let text = std::fs::read_to_string(&path).unwrap();
                    let code = text.split("#[cfg(test)]").next().unwrap_or("");
                    for (n, line) in code.lines().enumerate() {
                        if tier(line) {
                            found.push(format!("{}:{}: {}", path.display(), n + 1, line.trim()));
                        }
                    }
                }
            }
        }
        assert!(
            found.is_empty(),
            "RAM-size constants:\n{}",
            found.join("\n")
        );
    }

    /// FR-017: scratch at the capacity-limited depth for the three alias families' shapes (the
    /// analytic terms: one attention layer per position, and for head dim 256 the unfused scores
    /// `2 × 16 heads × L × (depth + L) × 2 B` over the chunk schedule): the solution is a fixed
    /// point (it fits, one more position does not), monotone in the budget, and equals v2
    /// (scratch at the maximum position) once the capacity reaches it.
    #[test]
    fn scratch_depth_is_the_fixed_point_for_each_family() {
        fn scores(depth: u32) -> u64 {
            let (mut at, mut most) = (0u32, 0u64);
            while at < depth {
                let len = crate::engine::chunk_len(at);
                most = most.max(2 * 16 * u64::from(len) * u64::from(at + len) * 2);
                at += len;
            }
            most
        }
        type Scratch = Box<dyn Fn(u32) -> u64>;
        let families: [(&str, ModelInfo, u64, Scratch); 3] = [
            (
                "qwen3",
                model(144 * KIB, 0, 40_960),
                4600 * MIB,
                Box::new(|d| GIB + 4 * KIB * u64::from(d)),
            ),
            (
                "qwen3.5",
                model(32 * KIB, 24 * (32 * 128 * 128 * 4 + 3 * 8192 * 2), 262_144),
                6 * GIB,
                Box::new(|d| GIB + 4 * KIB * u64::from(d) + scores(d)),
            ),
            (
                "qwen3.6",
                model(20 * KIB, 30 * (32 * 128 * 128 * 4 + 3 * 8192 * 2), 262_144),
                18 * GIB,
                Box::new(|d| 2 * GIB + 2 * KIB * u64::from(d) + scores(d)),
            ),
        ];
        for (name, info, weights, scratch) in &families {
            let reserve = position_bound_reserve(info.recurrent_bytes);
            let need =
                |d: u32| weights + scratch(d) + reserve + u64::from(d) * info.kv_bytes_per_token;
            let mut previous = 0;
            for gib in [12, 16, 24, 32, 40, 64, 96, 120] {
                let device = mac(128, 1.0);
                let lower = Overrides {
                    budget: Some(gib * GIB),
                };
                let Ok(b) = capacity_on_fn(device, lower, *weights, scratch.as_ref(), info) else {
                    continue;
                };
                let d = b.scratch_depth;
                assert_eq!(b.scratch, scratch(d), "{name} {gib}");
                assert!(need(d) <= b.budget, "{name} {gib}: {d} does not fit");
                assert!(
                    d == info.max_position || need(d + 1) > b.budget,
                    "{name} {gib}: {d} is not the largest depth that fits"
                );
                assert!(d >= previous, "{name}: not monotone at {gib} GiB");
                assert_eq!(b.kv_capacity_tokens, d.min(b.kv_capacity_tokens), "{name}");
                previous = d;
            }
            // Enough memory: the capacity reaches the maximum position and scratch is v2's.
            let b = capacity_on_fn(
                mac(128, 1.0),
                Overrides::default(),
                *weights,
                scratch.as_ref(),
                info,
            )
            .unwrap();
            assert_eq!(b.scratch_depth, info.max_position, "{name}");
            assert_eq!(b.scratch, scratch(info.max_position), "{name}");
        }
    }

    fn capacity_on_fn(
        device: DeviceMemory,
        overrides: Overrides,
        weights: u64,
        scratch: &dyn Fn(u32) -> u64,
        info: &ModelInfo,
    ) -> Result<MemoryBudget, BudgetError> {
        with_state_capacity(plan(device, overrides)?, weights, scratch, info)
    }

    /// D062: each term of `min(working set, physical − 2 GiB, available − margin)` binds in
    /// its case, the flag only lowers, and the plan reports which term it is.
    #[test]
    fn each_term_of_the_formula_binds_in_its_case() {
        let m4_pro = |available: Option<u64>| DeviceMemory {
            physical: 48 * GIB,
            recommended_working_set: 40_200_896_512,
            available,
            estimated: false,
        };
        assert_eq!(margin(48 * GIB), 6 * GIB);
        assert_eq!(margin(16 * GIB), 2 * GIB);
        assert_eq!(margin(8 * GIB), 2 * GIB);
        assert_eq!(margin(32 * GIB), 4 * GIB);
        // Nothing else running: the working set binds.
        let idle = plan(m4_pro(Some(44 * GIB)), Overrides::default()).unwrap();
        assert_eq!(
            (idle.budget, idle.binding, idle.available),
            (40_200_896_512, Binding::WorkingSet, Some(44 * GIB))
        );
        // A 32 GiB VM runs: 20.2 GiB available, 6 GiB margin.
        let vm = plan(m4_pro(Some(20 * GIB + GIB / 5)), Overrides::default()).unwrap();
        assert_eq!(vm.budget, 14 * GIB + GIB / 5);
        assert_eq!(vm.binding, Binding::Available);
        // Unknown available memory (not macOS): D048's budget.
        let unknown = plan(m4_pro(None), Overrides::default()).unwrap();
        assert_eq!(unknown.binding, Binding::WorkingSet);
        assert_eq!(unknown.available, None);
        // A raised wired limit without an available figure: physical − 2 GiB binds. (With
        // one, available − margin ≤ physical − 2 GiB always, since the margin is ≥ 2 GiB.)
        let raised = DeviceMemory {
            recommended_working_set: 48 * GIB,
            ..m4_pro(None)
        };
        let raised = plan(raised, Overrides::default()).unwrap();
        assert_eq!(
            (raised.budget, raised.binding),
            (46 * GIB, Binding::Physical)
        );
        // The flag lowers below every term, and cannot raise past the available term.
        let lowered = plan(
            m4_pro(Some(30 * GIB)),
            Overrides {
                budget: Some(10 * GIB),
            },
        )
        .unwrap();
        assert_eq!((lowered.budget, lowered.binding), (10 * GIB, Binding::Flag));
        assert_eq!(
            plan(
                m4_pro(Some(30 * GIB)),
                Overrides {
                    budget: Some(30 * GIB)
                }
            ),
            Err(BudgetError::AboveAutomatic {
                requested: 30 * GIB,
                automatic: 24 * GIB,
                binding: Binding::Available,
            })
        );
        // The flag at exactly the automatic budget keeps the automatic term.
        let equal = plan(
            m4_pro(Some(30 * GIB)),
            Overrides {
                budget: Some(24 * GIB),
            },
        )
        .unwrap();
        assert_eq!(equal.binding, Binding::Available);
        // Less available than the margin: nothing for mrun, and the error says why.
        assert_eq!(
            plan(m4_pro(Some(5 * GIB)), Overrides::default()),
            Err(BudgetError::NoneAvailable {
                available: 5 * GIB,
                margin: 6 * GIB
            })
        );
        assert_eq!(Binding::Available.to_string(), "available memory");
        assert_eq!(Binding::Flag.label(), "--budget-gib");
        // The default alias ignores the available term.
        assert_eq!(m4_pro(Some(GIB)).ignoring_available(), m4_pro(None));
    }

    #[test]
    fn weights_are_wired_only_when_they_fit_the_available_memory() {
        let device = |available| DeviceMemory {
            physical: 48 * GIB,
            recommended_working_set: 40 * GIB,
            available,
            estimated: false,
        };
        assert!(may_wire(device(Some(30 * GIB)), 19 * GIB));
        assert!(may_wire(device(Some(25 * GIB)), 19 * GIB));
        assert!(!may_wire(device(Some(24 * GIB)), 19 * GIB));
        assert!(may_wire(device(None), 19 * GIB));
    }

    #[test]
    fn the_override_only_lowers_the_budget() {
        let lower = Overrides {
            budget: Some(20 * GIB),
        };
        assert_eq!(plan(mac(48, 0.78), lower).unwrap().budget, 20 * GIB);
        let higher = Overrides {
            budget: Some(46 * GIB),
        };
        assert!(matches!(
            plan(mac(48, 0.78), higher),
            Err(BudgetError::AboveAutomatic {
                requested,
                ..
            }) if requested == 46 * GIB
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
    fn state_capacity_is_bounded_by_bytes_and_max_position() {
        let budget = plan(
            mac(48, 1.0),
            Overrides {
                budget: Some(40 * GIB),
            },
        )
        .unwrap();
        let info = model(100 * KIB, 0, 1_000_000);
        let full = with_state_capacity(budget, 16 * GIB, &fixed(2 * GIB), &info).unwrap();
        assert_eq!(full.state_capacity_bytes, 22 * GIB);
        assert_eq!(full.kv_capacity_tokens, (22 * GIB / (100 * KIB)) as u32);
        let short = model(100 * KIB, 0, 4096);
        let capped = with_state_capacity(budget, 16 * GIB, &fixed(2 * GIB), &short).unwrap();
        assert_eq!(capped.kv_capacity_tokens, 4096);
        // Position-bound state for the live sequence and two checkpoints comes off first.
        let hybrid = model(100 * KIB, 256 * MIB, 1_000_000);
        let held = with_state_capacity(budget, 16 * GIB, &fixed(2 * GIB), &hybrid).unwrap();
        assert_eq!(
            held.kv_capacity_tokens,
            ((22 * GIB - 768 * MIB) / (100 * KIB)) as u32
        );
    }

    #[test]
    fn position_dependent_scratch_is_taken_at_the_capacity_limited_depth() {
        // 10 GiB beside the weights; scratch 1 GiB + 16 KiB per position, state 48 KiB per token.
        let budget = plan(
            mac(48, 1.0),
            Overrides {
                budget: Some(14 * GIB),
            },
        )
        .unwrap();
        let info = model(48 * KIB, 0, 262_144);
        let scratch = LinearScratch {
            fixed: GIB,
            per_position: 16 * KIB,
        };
        let fit = with_state_capacity(budget, 4 * GIB, &|d| scratch.at(d), &info).unwrap();
        // c · 48 KiB + 1 GiB + c · 16 KiB ≤ 10 GiB → c = 9 GiB / 64 KiB.
        assert_eq!(fit.kv_capacity_tokens, 147_456);
        assert_eq!(fit.scratch, scratch.at(147_456));
        assert!(fit.weights + fit.scratch + 147_456 * 48 * KIB <= fit.budget);
        // At the family maximum the scratch alone would leave far less.
        let at_max =
            with_state_capacity(budget, 4 * GIB, &fixed(scratch.at(262_144)), &info).unwrap();
        assert!(at_max.kv_capacity_tokens < 120_000);
        // The pure pre-load form gives the same answer.
        let device = mac(48, 1.0);
        let pre = capacity_on(
            device,
            Overrides {
                budget: Some(14 * GIB),
            },
            4 * GIB,
            scratch,
            &info,
        )
        .unwrap();
        assert_eq!(pre.kv_capacity_tokens, 147_456);
    }

    #[test]
    fn a_model_that_does_not_fit_is_refused_with_every_number() {
        let budget = plan(mac(16, 0.67), Overrides::default()).unwrap();
        let info = model(144 * KIB, 0, 40_960);
        let error = check_fits(&budget, 12 * GIB, SCRATCH_ESTIMATE, &info).unwrap_err();
        assert_eq!(
            error,
            BudgetError::DoesNotFit {
                budget: budget.budget,
                weights: 12 * GIB,
                scratch: SCRATCH_ESTIMATE,
                recurrent: 0,
                initial_state: 4096 * 144 * KIB,
            }
        );
        assert_eq!(
            error.to_string(),
            "the model does not fit the memory budget of 10.72 GiB: weights 12.00 GiB + scratch \
             1.00 GiB + position-bound state 0.00 GiB + attention state for 4096 tokens 0.56 GiB \
             = 13.56 GiB"
        );
        assert!(check_fits(&budget, 4 * GIB, SCRATCH_ESTIMATE, &info).is_ok());
        // The position-bound reserve counts as the capacity holds it back (3×).
        let hybrid = model(0, 3 * GIB, 4096);
        match check_fits(&budget, 4 * GIB, 0, &hybrid) {
            Err(BudgetError::DoesNotFit { recurrent, .. }) => assert_eq!(recurrent, 9 * GIB),
            other => panic!("{other:?}"),
        }
    }
}
