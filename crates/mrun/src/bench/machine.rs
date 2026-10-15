//! Machine state for a run (005 FR-006, FR-012), read from macOS command-line tools: `sysctl`,
//! `sw_vers`, `pmset` and `vm_stat`. No FFI: the thermal state is approximated by `pmset -g
//! therm` (`nominal` when no thermal or performance warning is recorded, else the warning), not
//! `NSProcessInfo.thermalState`. Elsewhere every value is "unknown" or 0 (driver tests on Linux).

use crate::bench::result::{Machine, SystemMemory};
use std::process::Command;

fn run(program: &str, args: &[&str]) -> Option<String> {
    if !cfg!(target_os = "macos") {
        return None;
    }
    let out = Command::new(program).args(args).output().ok()?;
    out.status
        .success()
        .then(|| String::from_utf8_lossy(&out.stdout).trim().to_string())
}

fn sysctl(name: &str) -> Option<String> {
    run("/usr/sbin/sysctl", &["-n", name])
}

pub fn machine() -> Machine {
    let pmset = run("/usr/bin/pmset", &["-g"]).unwrap_or_default();
    let batt = run("/usr/bin/pmset", &["-g", "batt"]).unwrap_or_default();
    Machine {
        chip: sysctl("machdep.cpu.brand_string").unwrap_or_else(|| "unknown".into()),
        memory_bytes: sysctl("hw.memsize")
            .and_then(|m| m.parse().ok())
            .unwrap_or(0),
        macos: run("/usr/bin/sw_vers", &["-productVersion"]).unwrap_or_else(|| "unknown".into()),
        power_source: power_source(&batt),
        low_power: low_power(&pmset),
        per_system: Vec::new(),
    }
}

/// Free memory, swap and load before a system runs.
pub fn snapshot(system: &str) -> SystemMemory {
    SystemMemory {
        system: system.into(),
        free_memory_before: run("/usr/bin/vm_stat", &[]).map_or(0, |v| free_memory(&v)),
        swap: sysctl("vm.swapusage").map_or(0, |s| swap_used(&s)),
        load_average: sysctl("vm.loadavg").and_then(|l| load_average(&l)),
    }
}

/// `nominal`, the recorded warning, or `unknown`.
pub fn thermal() -> String {
    run("/usr/bin/pmset", &["-g", "therm"]).map_or_else(|| "unknown".into(), |t| thermal_state(&t))
}

fn power_source(batt: &str) -> String {
    if batt.contains("'AC Power'") {
        "AC".into()
    } else if batt.contains("'Battery Power'") {
        "battery".into()
    } else {
        "unknown".into()
    }
}

fn low_power(pmset: &str) -> bool {
    pmset.lines().any(|line| {
        let mut words = line.split_whitespace();
        words.next() == Some("lowpowermode") && words.next() == Some("1")
    })
}

fn thermal_state(therm: &str) -> String {
    let warnings: Vec<&str> = therm
        .lines()
        .map(str::trim)
        .filter(|l| l.to_ascii_lowercase().contains("level") && !l.contains("No "))
        .collect();
    if warnings.is_empty() {
        "nominal".into()
    } else {
        warnings.join("; ")
    }
}

/// `(Pages free + Pages speculative) × page size` from `vm_stat`.
fn free_memory(vm_stat: &str) -> u64 {
    let page = vm_stat
        .split("page size of ")
        .nth(1)
        .and_then(|r| r.split_whitespace().next())
        .and_then(|n| n.parse::<u64>().ok())
        .unwrap_or(16_384);
    let pages = |name: &str| {
        vm_stat
            .lines()
            .find_map(|l| l.strip_prefix(name))
            .and_then(|r| r.trim().trim_end_matches('.').parse::<u64>().ok())
            .unwrap_or(0)
    };
    (pages("Pages free:") + pages("Pages speculative:")) * page
}

/// `used = 1024.50M` from `sysctl vm.swapusage`, in bytes.
fn swap_used(usage: &str) -> u64 {
    usage
        .split("used = ")
        .nth(1)
        .and_then(|r| r.split_whitespace().next())
        .and_then(|v| {
            let (n, unit) = v.split_at(v.len().saturating_sub(1));
            let scale = match unit {
                "K" => 1u64 << 10,
                "M" => 1 << 20,
                "G" => 1 << 30,
                _ => return None,
            };
            n.parse::<f64>().ok().map(|n| (n * scale as f64) as u64)
        })
        .unwrap_or(0)
}

/// The 1-minute load of `{ 1.23 1.45 1.67 }`.
fn load_average(loadavg: &str) -> Option<f64> {
    loadavg
        .trim_matches(|c| c == '{' || c == '}' || c == ' ')
        .split_whitespace()
        .next()?
        .parse()
        .ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_macos_tool_output() {
        assert_eq!(
            power_source("Now drawing from 'AC Power'\n -InternalBattery-0 100%"),
            "AC"
        );
        assert_eq!(power_source("Now drawing from 'Battery Power'"), "battery");
        assert!(low_power(" sleep 1\n lowpowermode         1\n"));
        assert!(!low_power(" lowpowermode 0"));
        let vm = "Mach Virtual Memory Statistics: (page size of 16384 bytes)\nPages free:      100.\nPages active:    5.\nPages speculative: 20.\n";
        assert_eq!(free_memory(vm), 120 * 16_384);
        assert_eq!(
            swap_used("total = 2048.00M  used = 1.50G  free = 0.50G  (encrypted)"),
            3 << 29
        );
        assert_eq!(load_average("{ 1.23 1.45 1.67 }"), Some(1.23));
        let quiet = "Note: No thermal warning level has been recorded\nNote: No performance warning level has been recorded\nNote: No CPU power status has been recorded";
        assert_eq!(thermal_state(quiet), "nominal");
        assert_eq!(
            thermal_state("Thermal Warning Level = 2"),
            "Thermal Warning Level = 2"
        );
    }

    #[cfg(not(target_os = "macos"))]
    #[test]
    fn other_platforms_get_the_stub() {
        assert_eq!(machine().chip, "unknown");
        assert_eq!(thermal(), "unknown");
        assert_eq!(snapshot("mrun").free_memory_before, 0);
    }
}
