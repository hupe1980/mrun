//! Counting global allocator shared by the probe test binaries.
#![allow(dead_code)]

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU64, Ordering::Relaxed};
use std::sync::{Mutex, MutexGuard};

pub struct Counting;

static ACTIVE: AtomicBool = AtomicBool::new(false);
static CALLS: AtomicU64 = AtomicU64::new(0);
static BYTES: AtomicU64 = AtomicU64::new(0);
static LIVE: AtomicI64 = AtomicI64::new(0);
static PEAK: AtomicI64 = AtomicI64::new(0);
static SERIAL: Mutex<()> = Mutex::new(());

thread_local! {
    // Const-initialized: reading it from the allocator never allocates.
    static THREAD_CALLS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
    static THREAD_BYTES: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
    static THREAD_LIVE: std::cell::Cell<i64> = const { std::cell::Cell::new(0) };
    static THREAD_PEAK: std::cell::Cell<i64> = const { std::cell::Cell::new(0) };
}

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if ACTIVE.load(Relaxed) {
            CALLS.fetch_add(1, Relaxed);
            THREAD_CALLS.with(|calls| calls.set(calls.get() + 1));
            THREAD_BYTES.with(|bytes| bytes.set(bytes.get() + layout.size() as u64));
            let local = THREAD_LIVE.with(|live| {
                live.set(live.get() + layout.size() as i64);
                live.get()
            });
            THREAD_PEAK.with(|peak| peak.set(peak.get().max(local)));
            BYTES.fetch_add(layout.size() as u64, Relaxed);
            let size = layout.size() as i64;
            let live = LIVE.fetch_add(size, Relaxed) + size;
            PEAK.fetch_max(live, Relaxed);
        }
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        if ACTIVE.load(Relaxed) {
            LIVE.fetch_sub(layout.size() as i64, Relaxed);
            THREAD_LIVE.with(|live| live.set(live.get() - layout.size() as i64));
        }
        unsafe { System.dealloc(pointer, layout) }
    }
}

/// Allocation-constructor-entry counts observed inside one measured scope.
#[derive(Clone, Copy, Debug)]
pub struct Observed {
    pub calls: u64,
    /// Allocation calls made by the measuring thread only (excludes harness threads).
    pub thread_calls: u64,
    /// Bytes requested by the measuring thread only.
    pub thread_bytes: u64,
    /// Peak bytes live on the measuring thread above the scope start.
    pub thread_peak: u64,
    pub bytes: u64,
    /// Peak bytes live above the scope start (frees of pre-scope memory saturate at zero).
    pub peak: u64,
}

/// Serializes measured scopes: the allocator counters are process-global.
pub fn serial() -> MutexGuard<'static, ()> {
    SERIAL
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Measures every allocation in the process while `f` runs, including helper threads.
pub fn measure<T>(f: impl FnOnce() -> T) -> (T, Observed) {
    for counter in [&CALLS, &BYTES] {
        counter.store(0, Relaxed);
    }
    LIVE.store(0, Relaxed);
    PEAK.store(0, Relaxed);
    let thread_start = THREAD_CALLS.with(|calls| calls.get());
    let bytes_start = THREAD_BYTES.with(|bytes| bytes.get());
    THREAD_LIVE.with(|live| live.set(0));
    THREAD_PEAK.with(|peak| peak.set(0));
    ACTIVE.store(true, Relaxed);
    let value = f();
    ACTIVE.store(false, Relaxed);
    let observed = Observed {
        calls: CALLS.load(Relaxed),
        thread_calls: THREAD_CALLS.with(|calls| calls.get()) - thread_start,
        thread_bytes: THREAD_BYTES.with(|bytes| bytes.get()) - bytes_start,
        thread_peak: THREAD_PEAK.with(|peak| peak.get()).max(0) as u64,
        bytes: BYTES.load(Relaxed),
        peak: PEAK.load(Relaxed).max(0) as u64,
    };
    (value, observed)
}
