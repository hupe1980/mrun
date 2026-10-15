//! Keeps the Mac awake while requests run (008 FR-009): an idle-sleep assertion is taken when a
//! request is queued or running and released 60 seconds after the last one ends, so a laptop
//! does not sleep in the middle of a long prefill and still sleeps when mrun is idle.
//!
//! On macOS the assertion is a `caffeinate -i -w <pid>` child (the same IOPM assertion
//! `caffeinate` takes, without unsafe FFI; it also ends if mrun dies). Elsewhere it is a no-op.

use std::time::{Duration, Instant, SystemTime};

/// Time after the last request before the assertion is released.
pub const RELEASE_AFTER: Duration = Duration::from_secs(60);

/// What the state machine asks the platform to do.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Action {
    Acquire,
    Release,
}

/// Queue-driven assertion state.
#[derive(Debug, Default)]
pub struct PowerState {
    held_since: Option<SystemTime>,
    idle_since: Option<Instant>,
}

impl PowerState {
    pub fn held(&self) -> bool {
        self.held_since.is_some()
    }

    pub fn held_since(&self) -> Option<SystemTime> {
        self.held_since
    }

    /// One observation: `busy` when a request is queued or running.
    pub fn observe(&mut self, busy: bool, now: Instant, wall: SystemTime) -> Option<Action> {
        if busy {
            self.idle_since = None;
            if self.held_since.is_none() {
                self.held_since = Some(wall);
                return Some(Action::Acquire);
            }
            return None;
        }
        self.held_since?;
        let idle = *self.idle_since.get_or_insert(now);
        if now.duration_since(idle) >= RELEASE_AFTER {
            self.held_since = None;
            self.idle_since = None;
            return Some(Action::Release);
        }
        None
    }
}

/// The platform's idle-sleep assertion.
pub trait Assertion: Send {
    fn acquire(&mut self) -> bool;
    fn release(&mut self);
}

/// `caffeinate -i -w <pid>`: prevents idle sleep while it runs.
#[derive(Debug, Default)]
pub struct Caffeinate {
    child: Option<std::process::Child>,
}

impl Assertion for Caffeinate {
    fn acquire(&mut self) -> bool {
        if self.child.is_some() {
            return true;
        }
        self.child = std::process::Command::new("caffeinate")
            .args(["-i", "-w", &std::process::id().to_string()])
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn()
            .ok();
        self.child.is_some()
    }

    fn release(&mut self) {
        if let Some(mut child) = self.child.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

impl Drop for Caffeinate {
    fn drop(&mut self) {
        self.release();
    }
}

/// No power management (Linux, tests).
#[derive(Debug, Default)]
pub struct NoAssertion;

impl Assertion for NoAssertion {
    fn acquire(&mut self) -> bool {
        false
    }

    fn release(&mut self) {}
}

/// The platform default.
pub fn platform_assertion() -> Box<dyn Assertion> {
    if cfg!(target_os = "macos") {
        Box::new(Caffeinate::default())
    } else {
        Box::new(NoAssertion)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn acquires_on_work_and_releases_a_minute_after_the_last_request() {
        let mut state = PowerState::default();
        let t0 = Instant::now();
        let wall = SystemTime::UNIX_EPOCH;
        let at = |s: u64| t0 + Duration::from_secs(s);
        assert_eq!(
            state.observe(false, at(0), wall),
            None,
            "idle: nothing held"
        );
        assert_eq!(state.observe(true, at(1), wall), Some(Action::Acquire));
        assert_eq!(state.observe(true, at(2), wall), None);
        assert!(state.held());
        assert_eq!(state.observe(false, at(10), wall), None);
        // A new request within the minute keeps the assertion and restarts the countdown.
        assert_eq!(state.observe(true, at(50), wall), None);
        assert_eq!(state.observe(false, at(60), wall), None);
        assert_eq!(state.observe(false, at(119), wall), None);
        assert_eq!(state.observe(false, at(120), wall), Some(Action::Release));
        assert!(!state.held());
        assert_eq!(state.observe(false, at(500), wall), None);
    }
}
