//! System memory pressure (D031): a background thread reads macOS's
//! `kern.memorystatus_vm_pressure_level` with `sysctlbyname` once a second and raises the gauge when the level rises;
//! the generation loop takes the pending level between prefill chunks and every 64 decode tokens,
//! and the cache sheds stored state. Elsewhere nothing raises it.

use std::sync::Arc;
use std::sync::atomic::{AtomicU8, Ordering};
use std::time::Duration;

/// Memory-pressure level reported by the system.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum Pressure {
    Warning = 1,
    Critical = 2,
}

impl Pressure {
    fn from_u8(level: u8) -> Option<Self> {
        match level {
            1 => Some(Self::Warning),
            2 => Some(Self::Critical),
            _ => None,
        }
    }
}

/// A pending pressure level, set on a rising edge and taken once.
#[derive(Debug, Default)]
pub struct PressureGauge {
    pending: AtomicU8,
}

impl PressureGauge {
    /// Records `level` (the highest pending one wins).
    pub fn raise(&self, level: Pressure) {
        self.pending.fetch_max(level as u8, Ordering::AcqRel);
    }

    /// The pending level, once.
    pub fn take(&self) -> Option<Pressure> {
        Pressure::from_u8(self.pending.swap(0, Ordering::AcqRel))
    }
}

/// Starts the poller for `gauge` (macOS only). The thread ends when the gauge is dropped.
pub fn watch(gauge: &Arc<PressureGauge>) {
    if !cfg!(target_os = "macos") {
        return;
    }
    let weak = Arc::downgrade(gauge);
    let _ = std::thread::Builder::new()
        .name("mrun-pressure".into())
        .spawn(move || {
            let mut last = 0u8;
            while let Some(gauge) = weak.upgrade() {
                let level = system_level().map_or(0, |p| p as u8);
                if level > last
                    && let Some(pressure) = Pressure::from_u8(level)
                {
                    gauge.raise(pressure);
                }
                last = level;
                drop(gauge);
                std::thread::sleep(Duration::from_secs(1));
            }
        });
}

/// The kernel's level: 1 normal, 2 warning, 4 critical.
fn system_level() -> Option<Pressure> {
    level_of(raw_level()?)
}

fn level_of(raw: u32) -> Option<Pressure> {
    match raw {
        2 => Some(Pressure::Warning),
        4 => Some(Pressure::Critical),
        _ => None,
    }
}

/// `kern.memorystatus_vm_pressure_level` without forking `sysctl`.
#[cfg(target_os = "macos")]
#[allow(unsafe_code)]
fn raw_level() -> Option<u32> {
    use std::ffi::{c_char, c_int, c_void};
    unsafe extern "C" {
        // libSystem, always linked on macOS.
        fn sysctlbyname(
            name: *const c_char,
            oldp: *mut c_void,
            oldlenp: *mut usize,
            newp: *mut c_void,
            newlen: usize,
        ) -> c_int;
    }
    let mut value: u32 = 0;
    let mut len = std::mem::size_of::<u32>();
    // SAFETY: the name is NUL-terminated, `oldp` points to `len` writable bytes, and no new
    // value is set.
    let status = unsafe {
        sysctlbyname(
            c"kern.memorystatus_vm_pressure_level".as_ptr(),
            (&raw mut value).cast(),
            &mut len,
            std::ptr::null_mut(),
            0,
        )
    };
    (status == 0 && len == std::mem::size_of::<u32>()).then_some(value)
}

#[cfg(not(target_os = "macos"))]
fn raw_level() -> Option<u32> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn levels_parse_and_the_highest_pending_wins() {
        assert_eq!(level_of(1), None);
        assert_eq!(level_of(2), Some(Pressure::Warning));
        assert_eq!(level_of(4), Some(Pressure::Critical));
        let gauge = PressureGauge::default();
        gauge.raise(Pressure::Warning);
        gauge.raise(Pressure::Critical);
        assert_eq!(gauge.take(), Some(Pressure::Critical));
        assert_eq!(gauge.take(), None);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn the_system_reports_a_level() {
        // Normal pressure reads as `None`; the call itself must work.
        assert!(matches!(raw_level(), Some(1 | 2 | 4)), "{:?}", raw_level());
    }
}
