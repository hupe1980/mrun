//! `GET /metrics` (012 FR-010, D056): Prometheus text format 0.0.4, written by hand. Counters
//! are atomics recorded by the relay and the handler; gauges are read at scrape time from the
//! readiness, the tracker, the queue gauge, the budget and the cache snapshot, never through
//! the engine worker (FR-012). Labels come from closed sets; nothing carries content (FR-011).

use crate::api::model::{ApiKind, Usage};
use crate::cache::miss::{Cause, MissReport};
use crate::cli::serve::{Readiness, ServerState};
use crate::runtime::generate::Stats;
use axum::extract::State;
use axum::http::{HeaderValue, header};
use axum::response::{IntoResponse, Response};
use std::fmt::Write;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, PoisonError};

const APIS: [ApiKind; 3] = [ApiKind::Chat, ApiKind::Messages, ApiKind::Responses];
const TTFT_BUCKETS: &[f64] = &[0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0];
const PREFILL_BUCKETS: &[f64] = &[50.0, 100.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0];
const DECODE_BUCKETS: &[f64] = &[5.0, 10.0, 20.0, 30.0, 40.0, 60.0, 80.0, 120.0, 200.0];

/// How a request ended, for `mrun_requests_total`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Outcome {
    /// A finished response was delivered.
    Completed,
    /// The client went away (or the server shut down under it).
    Cancelled,
    /// An error after admission, including the queue deadline.
    Failed,
    /// An error before admission.
    Rejected,
}

impl Outcome {
    const ALL: [Self; 4] = [
        Self::Completed,
        Self::Cancelled,
        Self::Failed,
        Self::Rejected,
    ];

    fn as_str(self) -> &'static str {
        match self {
            Self::Completed => "completed",
            Self::Cancelled => "cancelled",
            Self::Failed => "failed",
            Self::Rejected => "rejected",
        }
    }
}

fn api_index(api: ApiKind) -> usize {
    match api {
        ApiKind::Chat => 0,
        ApiKind::Messages => 1,
        ApiKind::Responses => 2,
    }
}

#[derive(Debug)]
struct Histogram {
    bounds: &'static [f64],
    counts: Vec<u64>,
    sum: f64,
    count: u64,
}

impl Histogram {
    fn new(bounds: &'static [f64]) -> Self {
        Self {
            bounds,
            counts: vec![0; bounds.len()],
            sum: 0.0,
            count: 0,
        }
    }

    fn observe(&mut self, value: f64) {
        if !value.is_finite() {
            return;
        }
        if let Some(i) = self.bounds.iter().position(|&b| value <= b) {
            self.counts[i] += 1;
        }
        self.sum += value;
        self.count += 1;
    }
}

/// Counters and histograms since the server started.
#[derive(Debug)]
pub struct Metrics {
    requests: [[AtomicU64; 4]; 3],
    prompt: [AtomicU64; 3],
    cached: [AtomicU64; 3],
    generated: [AtomicU64; 3],
    spliced: AtomicU64,
    unreused: AtomicU64,
    /// By position in [`Cause::ALL`] (003's list, FR-013), all present from the first scrape
    /// so `rate()` works.
    misses: [AtomicU64; Cause::ALL.len()],
    ttft: Mutex<Histogram>,
    prefill: Mutex<Histogram>,
    decode: Mutex<Histogram>,
}

impl Default for Metrics {
    fn default() -> Self {
        Self {
            requests: Default::default(),
            prompt: Default::default(),
            cached: Default::default(),
            generated: Default::default(),
            spliced: AtomicU64::new(0),
            unreused: AtomicU64::new(0),
            misses: Default::default(),
            ttft: Mutex::new(Histogram::new(TTFT_BUCKETS)),
            prefill: Mutex::new(Histogram::new(PREFILL_BUCKETS)),
            decode: Mutex::new(Histogram::new(DECODE_BUCKETS)),
        }
    }
}

impl Metrics {
    pub fn record_outcome(&self, api: ApiKind, outcome: Outcome) {
        self.requests[api_index(api)][outcome as usize].fetch_add(1, Ordering::Relaxed);
    }

    /// A completed response: its usage, miss report, spliced tokens and timings (the outcome
    /// is recorded separately).
    pub fn record_completed(
        &self,
        api: ApiKind,
        usage: Usage,
        miss: Option<&MissReport>,
        spliced: u32,
        stats: &Stats,
    ) {
        let i = api_index(api);
        self.prompt[i].fetch_add(u64::from(usage.input), Ordering::Relaxed);
        self.cached[i].fetch_add(u64::from(usage.cached), Ordering::Relaxed);
        self.generated[i].fetch_add(u64::from(usage.output), Ordering::Relaxed);
        self.spliced
            .fetch_add(u64::from(spliced), Ordering::Relaxed);
        if let Some(miss) = miss {
            self.unreused
                .fetch_add(u64::from(miss.unreused), Ordering::Relaxed);
            if let Some(i) = miss
                .cause
                .and_then(|c| Cause::ALL.iter().position(|&a| a == c))
            {
                self.misses[i].fetch_add(1, Ordering::Relaxed);
            }
        }
        let lock = |h: &Mutex<Histogram>, value: f64| {
            h.lock()
                .unwrap_or_else(PoisonError::into_inner)
                .observe(value);
        };
        if let Some(ttft) = stats.ttft.filter(|_| usage.output > 0) {
            lock(&self.ttft, ttft.as_secs_f64());
        }
        if stats.prompt_tokens > stats.cached_tokens && stats.prefill_tps > 0.0 {
            lock(&self.prefill, stats.prefill_tps);
        }
        if stats.generated_tokens >= 2 && stats.decode_tps > 0.0 {
            lock(&self.decode, stats.decode_tps);
        }
    }
}

/// Prometheus text writer.
struct Text(String);

impl Text {
    fn family(&mut self, name: &str, kind: &str, help: &str) {
        let _ = writeln!(self.0, "# HELP {name} {help}");
        let _ = writeln!(self.0, "# TYPE {name} {kind}");
    }

    fn sample(&mut self, name: &str, labels: &[(&str, &str)], value: impl std::fmt::Display) {
        self.0.push_str(name);
        if !labels.is_empty() {
            let list: Vec<String> = labels.iter().map(|(k, v)| format!("{k}=\"{v}\"")).collect();
            let _ = write!(self.0, "{{{}}}", list.join(","));
        }
        let _ = writeln!(self.0, " {value}");
    }

    fn single(&mut self, name: &str, kind: &str, help: &str, value: impl std::fmt::Display) {
        self.family(name, kind, help);
        self.sample(name, &[], value);
    }

    fn per_api(&mut self, name: &str, help: &str, values: &[AtomicU64; 3]) {
        self.family(name, "counter", help);
        for api in APIS {
            let value = values[api_index(api)].load(Ordering::Relaxed);
            self.sample(name, &[("api", api.as_str())], value);
        }
    }

    fn histogram(&mut self, name: &str, help: &str, histogram: &Mutex<Histogram>) {
        let h = histogram.lock().unwrap_or_else(PoisonError::into_inner);
        self.family(name, "histogram", help);
        let bucket = format!("{name}_bucket");
        let mut cumulative = 0;
        for (bound, count) in h.bounds.iter().zip(&h.counts) {
            cumulative += count;
            self.sample(&bucket, &[("le", &bound.to_string())], cumulative);
        }
        self.sample(&bucket, &[("le", "+Inf")], h.count);
        self.sample(&format!("{name}_sum"), &[], h.sum);
        self.sample(&format!("{name}_count"), &[], h.count);
    }
}

/// The scrape body. While the model loads (or after a failed load) only `mrun_info`,
/// `mrun_model_loaded` and `mrun_requests_total`.
impl Metrics {
    /// `mrun_cache_misses_total`, one sample per cause of [`Cause::ALL`].
    fn render_misses(&self, out: &mut Text) {
        out.family(
            "mrun_cache_misses_total",
            "counter",
            "Completed responses by miss cause.",
        );
        for (cause, count) in Cause::ALL.iter().zip(&self.misses) {
            out.sample(
                "mrun_cache_misses_total",
                &[("cause", cause.as_str())],
                count.load(Ordering::Relaxed),
            );
        }
    }
}

pub fn render(state: &ServerState) -> String {
    let metrics = &state.tracker.metrics;
    let mut out = Text(String::new());
    out.family("mrun_info", "gauge", "Server version.");
    out.sample("mrun_info", &[("version", env!("CARGO_PKG_VERSION"))], 1);
    let readiness = state.readiness.borrow().clone();
    let loaded = match readiness {
        Readiness::Ready(loaded) => Some(loaded),
        _ => None,
    };
    out.single(
        "mrun_model_loaded",
        "gauge",
        "1 when the model is loaded.",
        u8::from(loaded.is_some()),
    );
    out.family(
        "mrun_requests_total",
        "counter",
        "Requests by API and outcome.",
    );
    for api in APIS {
        for outcome in Outcome::ALL {
            let value = metrics.requests[api_index(api)][outcome as usize].load(Ordering::Relaxed);
            out.sample(
                "mrun_requests_total",
                &[("api", api.as_str()), ("outcome", outcome.as_str())],
                value,
            );
        }
    }
    let Some(loaded) = loaded else {
        return out.0;
    };
    out.per_api(
        "mrun_prompt_tokens_total",
        "Prompt tokens of completed responses, cached included.",
        &metrics.prompt,
    );
    out.per_api(
        "mrun_cached_tokens_total",
        "Cached prompt tokens of completed responses.",
        &metrics.cached,
    );
    out.per_api(
        "mrun_generated_tokens_total",
        "Output tokens of completed responses.",
        &metrics.generated,
    );
    out.single(
        "mrun_spliced_tokens_total",
        "counter",
        "Prompt tokens taken from earlier replies.",
        metrics.spliced.load(Ordering::Relaxed),
    );
    metrics.render_misses(&mut out);
    out.single(
        "mrun_cache_unreused_tokens_total",
        "counter",
        "Prompt tokens shared with a recently stored or dropped path but processed again.",
        metrics.unreused.load(Ordering::Relaxed),
    );
    let stats = loaded.handle.cache_stats().unwrap_or_default();
    out.single(
        "mrun_cache_dropped_ring_wrapped",
        "gauge",
        "1 once the dropped-path ring has wrapped (a new miss may be an old eviction).",
        u8::from(stats.ring_wrapped),
    );
    out.single(
        "mrun_cache_evicted_total",
        "counter",
        "Cache evictions.",
        stats.evicted_total,
    );
    out.single(
        "mrun_memory_pressure_events_total",
        "counter",
        "System memory-pressure events.",
        stats.pressure_events,
    );
    out.single("mrun_lanes_configured", "gauge", "Decode lanes.", 1);
    out.single(
        "mrun_lanes_busy",
        "gauge",
        "Requests prefilling or generating.",
        state.tracker.busy(),
    );
    out.single(
        "mrun_queue_length",
        "gauge",
        "Requests waiting.",
        loaded.handle.queue().waiting(),
    );
    let budget = loaded.handle.budget();
    let info = loaded.handle.info();
    let live = u64::from(stats.live_tokens);
    let state_bytes = if live > 0 {
        live * info.kv_bytes_per_token + info.recurrent_bytes
    } else {
        0
    };
    out.family("mrun_memory_bytes", "gauge", "Memory by part.");
    for (part, bytes) in [
        ("budget", budget.budget),
        ("weights", budget.weights),
        ("scratch", budget.scratch),
        ("state", state_bytes),
        ("cache", stats.buffer_bytes + stats.checkpoint_bytes),
    ] {
        out.sample("mrun_memory_bytes", &[("part", part)], bytes);
    }
    out.single(
        "mrun_kv_capacity_tokens",
        "gauge",
        "Tokens of sequence state that fit.",
        budget.kv_capacity_tokens,
    );
    out.single(
        "mrun_cache_branches",
        "gauge",
        "Stored branches.",
        stats.branches,
    );
    out.single(
        "mrun_cache_live_tokens",
        "gauge",
        "Tokens in the live state.",
        stats.live_tokens,
    );
    out.histogram(
        "mrun_time_to_first_token_seconds",
        "Time to first token of completed responses.",
        &metrics.ttft,
    );
    out.histogram(
        "mrun_prefill_tokens_per_second",
        "Prefill speed of completed responses that processed tokens.",
        &metrics.prefill,
    );
    out.histogram(
        "mrun_decode_tokens_per_second",
        "Decode speed of completed responses with two or more output tokens.",
        &metrics.decode,
    );
    out.0
}

/// `GET /metrics`
pub async fn metrics(State(state): State<ServerState>) -> Response {
    let mut response = render(&state).into_response();
    response.headers_mut().insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("text/plain; version=0.0.4; charset=utf-8"),
    );
    response
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache::miss::Cause;
    use std::time::Duration;

    #[test]
    fn histograms_are_cumulative_and_end_at_the_count() {
        let mut out = Text(String::new());
        let h = Mutex::new(Histogram::new(&[1.0, 2.0]));
        for v in [0.5, 1.5, 1.7, 9.0, f64::NAN] {
            h.lock().unwrap().observe(v);
        }
        out.histogram("x_seconds", "x.", &h);
        let body = out.0;
        assert!(body.contains("x_seconds_bucket{le=\"1\"} 1\n"), "{body}");
        assert!(body.contains("x_seconds_bucket{le=\"2\"} 3\n"));
        assert!(body.contains("x_seconds_bucket{le=\"+Inf\"} 4\n"));
        assert!(body.contains("x_seconds_count 4\n"));
        assert!(body.contains("x_seconds_sum 12.7\n"));
    }

    #[test]
    fn every_cause_renders_at_zero() {
        let metrics = Metrics::default();
        let mut out = Text(String::new());
        metrics.render_misses(&mut out);
        for cause in Cause::ALL {
            let line = format!(
                "mrun_cache_misses_total{{cause=\"{}\"}} 0\n",
                cause.as_str()
            );
            assert!(out.0.contains(&line), "{}", out.0);
        }
        assert_eq!(
            out.0.matches("mrun_cache_misses_total{").count(),
            Cause::ALL.len()
        );
    }

    #[test]
    fn completed_responses_add_up() {
        let metrics = Metrics::default();
        let usage = Usage {
            input: 100,
            cached: 60,
            output: 5,
            reasoning: 0,
        };
        let miss = MissReport {
            prompt: 100,
            cached: 60,
            processed: 40,
            divergence: Some(60),
            message: None,
            gap: 0,
            unreused: 12,
            ring_wrapped: false,
            cause: Some(Cause::Diverged),
        };
        let stats = Stats {
            prompt_tokens: 100,
            cached_tokens: 60,
            generated_tokens: 5,
            ttft: Some(Duration::from_millis(300)),
            prefill_tps: 400.0,
            decode_tps: 30.0,
            ..Stats::default()
        };
        metrics.record_completed(ApiKind::Messages, usage, Some(&miss), 7, &stats);
        metrics.record_outcome(ApiKind::Chat, Outcome::Rejected);
        assert_eq!(metrics.prompt[1].load(Ordering::Relaxed), 100);
        assert_eq!(metrics.spliced.load(Ordering::Relaxed), 7);
        assert_eq!(metrics.unreused.load(Ordering::Relaxed), 12);
        assert_eq!(metrics.misses[2].load(Ordering::Relaxed), 1, "diverged");
        assert_eq!(
            metrics.requests[0][Outcome::Rejected as usize].load(Ordering::Relaxed),
            1
        );
        assert_eq!(metrics.ttft.lock().unwrap().count, 1);
        assert_eq!(metrics.decode.lock().unwrap().count, 1);
    }
}
