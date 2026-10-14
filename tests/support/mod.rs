use std::collections::VecDeque;
use std::fs;
use std::io;
use std::path::{Component, Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

mod admission_foundation;
mod backend;
pub mod qualification;
mod report_contract;

static NEXT_FIXTURE: AtomicU64 = AtomicU64::new(0);

#[derive(Clone, Default)]
pub struct ManualClock(Arc<Mutex<Duration>>);

impl ManualClock {
    pub fn now(&self) -> Duration {
        *self.0.lock().expect("test clock lock poisoned")
    }

    pub fn advance(&self, delta: Duration) -> Option<Duration> {
        let mut current = self.0.lock().expect("test clock lock poisoned");
        let next = current.checked_add(delta)?;
        *current = next;
        Some(next)
    }
}

pub struct FixtureRoot(PathBuf);

impl FixtureRoot {
    pub fn new(label: &str) -> io::Result<Self> {
        if label.is_empty()
            || !label
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || byte == b'-' || byte == b'_')
        {
            return Err(io::Error::new(io::ErrorKind::InvalidInput, "invalid label"));
        }
        for _attempt in 0..1024 {
            let nonce = NEXT_FIXTURE.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "mrun-fixture-{}-{nonce}-{label}",
                std::process::id()
            ));
            match fs::create_dir(&path) {
                Ok(()) => return Ok(Self(path)),
                Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
                Err(error) => return Err(error),
            }
        }
        Err(io::Error::new(
            io::ErrorKind::AlreadyExists,
            "fixture root attempts exhausted",
        ))
    }

    pub fn path(&self) -> &Path {
        &self.0
    }

    pub fn child(&self, relative: &Path) -> io::Result<PathBuf> {
        if relative.as_os_str().is_empty()
            || relative
                .components()
                .any(|component| !matches!(component, Component::Normal(_)))
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "fixture path must be relative",
            ));
        }
        Ok(self.0.join(relative))
    }
}

impl Drop for FixtureRoot {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Pressure {
    Nominal,
    Elevated,
    Critical,
    Missing,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BackendEvent {
    Reserved(u64),
    Evaluated(u32),
    Submitted,
    CancelObserved,
    Synchronized,
    Released(u64),
    Fault,
    Pressure(Pressure),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EventError {
    Full,
    InvalidCapacity,
}

pub struct EventLog {
    clock: ManualClock,
    capacity: usize,
    events: VecDeque<(Duration, BackendEvent)>,
}

impl EventLog {
    pub fn new(clock: ManualClock, capacity: usize) -> Result<Self, EventError> {
        if !(1..=1024).contains(&capacity) {
            return Err(EventError::InvalidCapacity);
        }
        Ok(Self {
            clock,
            capacity,
            events: VecDeque::with_capacity(capacity),
        })
    }

    pub fn push(&mut self, event: BackendEvent) -> Result<(), EventError> {
        if self.events.len() == self.capacity {
            return Err(EventError::Full);
        }
        self.events.push_back((self.clock.now(), event));
        Ok(())
    }

    pub fn pop(&mut self) -> Option<(Duration, BackendEvent)> {
        self.events.pop_front()
    }
}
