//! `GET /v1/models`, `GET /mrun/status` (schema v1, 002 contract, 012 additions) and the
//! tracker: the live record of every queued and running request (012 FR-008), the metrics
//! registry, and the shutdown drain (D052).

use crate::api::chat_completions::Chat;
use crate::api::handler;
use crate::api::metrics::Metrics;
use crate::api::model::ApiKind;
use crate::api::run::DEFAULT_MAX_OUTPUT;
use crate::api::sessions::{self, Finished, History};
use crate::cache::miss::{Cause, MissReport, role_str};
use crate::chat::syntax::registry::ChatSyntax;
use crate::cli::serve::{Readiness, ServerState};
use crate::runtime::handle::Ticket;
use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::sync::{Mutex, PoisonError};
use std::time::{Duration, Instant};
use tokio::sync::watch;

/// Finished requests `/mrun/status` keeps (018).
pub const FINISHED: usize = 20;

/// Time running requests get to finish after SIGTERM before they are cancelled (D052).
pub const SHUTDOWN_GRACE: Duration = Duration::from_secs(5);

/// Where a request is.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Phase {
    Queued,
    Prefill,
    Generating,
}

impl Phase {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Queued => "queued",
            Self::Prefill => "prefill",
            Self::Generating => "generating",
        }
    }
}

/// The live record of one request (012 data model). No content: ids, counters and phases.
#[derive(Clone, Debug)]
pub struct Entry {
    pub id: u64,
    pub api: ApiKind,
    /// 0 until parallel lanes (011).
    pub lane: u32,
    pub phase: Phase,
    /// From the render while queued, then the engine's count.
    pub prompt_tokens: u32,
    /// `None` while queued.
    pub cached: Option<u32>,
    /// Prompt tokens in the state: cached plus processed so far.
    pub progress: u32,
    /// The miss cause (`None` on a hit or while queued).
    pub cause: Option<Cause>,
    /// The miss report's `unreused` tokens (003 FR-010, 012 FR-014); `None` while queued.
    pub unreused: Option<u32>,
    pub generated: u32,
    pub first_token: Option<Instant>,
    pub queued_at: Instant,
    /// Cancels the request (shutdown).
    pub ticket: Option<Ticket>,
    /// The pool member that sent it (026); `None` from this Mac.
    pub member: Option<String>,
}

impl Entry {
    pub fn queued(id: u64, api: ApiKind, prompt_tokens: u32) -> Self {
        Self {
            id,
            api,
            lane: 0,
            phase: Phase::Queued,
            prompt_tokens,
            cached: None,
            progress: 0,
            cause: None,
            unreused: None,
            generated: 0,
            first_token: None,
            queued_at: Instant::now(),
            ticket: None,
            member: None,
        }
    }

    /// Output tokens per second after the first one; `None` before the second.
    pub fn decode_tps(&self, now: Instant) -> Option<f64> {
        let first = self.first_token?;
        let secs = now.duration_since(first).as_secs_f64();
        (self.generated >= 2 && secs > 0.0).then(|| f64::from(self.generated - 1) / secs)
    }

    fn json(&self, now: Instant) -> Value {
        json!({
            "id": self.id,
            "api": self.api.as_str(),
            "lane": self.lane,
            "phase": self.phase.as_str(),
            "prompt_tokens": self.prompt_tokens,
            "cached": self.cached,
            "progress": self.progress,
            "cause": self.cause.map(Cause::as_str),
            "unreused": self.unreused,
            "generated": self.generated,
            "decode_tps": self.decode_tps(now),
            "speculation": Value::Null,
            "age_ms": now.duration_since(self.queued_at).as_millis() as u64,
        })
    }

    /// The schema v1 `active` shape.
    fn active_json(&self) -> Value {
        json!({
            "id": self.id,
            "api": self.api.as_str(),
            "prompt_tokens": self.prompt_tokens,
            "cached": self.cached.unwrap_or(0),
            "phase": self.phase.as_str(),
            "progress": self.progress,
        })
    }
}

/// This Mac's pool for `/mrun/status`; a host's carries the queue by sender.
fn pool_status(state: &ServerState) -> Value {
    let mut pool = state.pool.status()["pool"].clone();
    if pool["role"] == "host" {
        pool["queue"] = state.tracker.pool_queue();
    }
    pool
}

/// Shutdown stages, observed by every relay.
#[derive(Clone, Copy, Debug, Default, Eq, Ord, PartialEq, PartialOrd)]
pub enum Stage {
    #[default]
    Serving,
    /// New requests are refused and queued ones cancelled.
    Draining,
    /// Running requests are cancelled too.
    Stopping,
}

/// The server's live view: requests by id, metrics and the shutdown stage.
#[derive(Debug)]
pub struct Tracker {
    requests: Mutex<BTreeMap<u64, Entry>>,
    /// The last [`FINISHED`] requests that left the queue, newest last (018 FR-005: Now's
    /// reuse ribbons). Counters only, no content.
    finished: Mutex<std::collections::VecDeque<Value>>,
    /// Session replay's records (025).
    pub history: Mutex<History>,
    /// `ended_ms` counts from here.
    started: Instant,
    pub metrics: Metrics,
    stage: watch::Sender<Stage>,
}

impl Default for Tracker {
    fn default() -> Self {
        Self {
            requests: Mutex::default(),
            finished: Mutex::default(),
            history: Mutex::default(),
            started: Instant::now(),
            metrics: Metrics::default(),
            stage: watch::Sender::new(Stage::Serving),
        }
    }
}

impl Tracker {
    fn map(&self) -> std::sync::MutexGuard<'_, BTreeMap<u64, Entry>> {
        self.requests.lock().unwrap_or_else(PoisonError::into_inner)
    }

    pub fn insert(&self, entry: Entry) {
        self.map().insert(entry.id, entry);
    }

    /// The pool's queue (026 FR-007): who runs now, and how many requests wait per sender
    /// (`null`: this Mac), oldest first.
    pub fn pool_queue(&self) -> Value {
        let map = self.map();
        let running = map
            .values()
            .find(|e| e.phase != Phase::Queued)
            .map(|e| e.member.clone());
        let mut waiting: Vec<(Option<String>, u32)> = Vec::new();
        let mut queued: Vec<&Entry> = map.values().filter(|e| e.phase == Phase::Queued).collect();
        queued.sort_by_key(|e| e.queued_at);
        for e in queued {
            match waiting.iter_mut().find(|(m, _)| *m == e.member) {
                Some((_, n)) => *n += 1,
                None => waiting.push((e.member.clone(), 1)),
            }
        }
        json!({
            "running": running.map(|m| json!({"member": m})),
            "waiting": waiting.iter().map(|(m, n)| json!({"member": m, "waiting": n})).collect::<Vec<_>>(),
        })
    }

    pub fn update(&self, id: u64, f: impl FnOnce(&mut Entry)) {
        if let Some(entry) = self.map().get_mut(&id) {
            f(entry);
        }
    }

    pub fn remove(&self, id: u64) {
        self.map().remove(&id);
    }

    /// Removes a request that ended and keeps its record for `/mrun/status` `finished`: its
    /// reuse and miss report as `last` shows them, time to first token from arrival, decode
    /// speed and how it ended (`end`, `stopped`, `error`). Requests that never left the queue
    /// leave no record. With `replay`, session replay keeps it too, with its session and row.
    pub fn finish(
        &self,
        id: u64,
        miss: Option<&MissReport>,
        stop: &'static str,
        replay: Option<Finished>,
    ) {
        let Some(entry) = self.map().remove(&id) else {
            return;
        };
        let Some(cached) = entry.cached else {
            return;
        };
        let now = Instant::now();
        let mut record = miss.map_or_else(
            || {
                json!({"cached": cached, "processed": entry.prompt_tokens.saturating_sub(cached),
                    "cause": null, "divergence": null, "message": null, "unreused": 0})
            },
            miss_json,
        );
        record["id"] = entry.id.into();
        record["api"] = entry.api.as_str().into();
        record["prompt_tokens"] = entry.prompt_tokens.into();
        record["generated"] = entry.generated.into();
        record["ttft_ms"] = entry
            .first_token
            .map(|t| t.duration_since(entry.queued_at).as_millis() as u64)
            .into();
        record["decode_tps"] = entry.decode_tps(now).into();
        record["stop"] = stop.into();
        if let Some(replay) = replay {
            let ended_ms = self.uptime_ms();
            let mut history = self.history.lock().unwrap_or_else(PoisonError::into_inner);
            // A member's agents are sessions of their own.
            let key = match &replay.member {
                Some(member) => format!("{}@{member}", replay.client),
                None => replay.client.clone(),
            };
            let (session, row) = history.assign(&key, &replay.fingerprint, ended_ms);
            let mut kept = record.clone();
            kept["session"] = session.as_str().into();
            kept["row"] = row.into();
            kept["client"] = replay.client.into();
            kept["member"] = replay.member.into();
            kept["ended_ms"] = ended_ms.into();
            kept["attention"] = miss
                .is_some_and(|m| sessions::worth_fixing(m.cause, m.message, m.unreused))
                .into();
            kept["excerpt"] = replay.excerpt.is_some().into();
            history.push(id, session, kept, replay.excerpt);
        }
        let mut finished = self.finished.lock().unwrap_or_else(PoisonError::into_inner);
        finished.push_back(record);
        while finished.len() > FINISHED {
            finished.pop_front();
        }
    }

    /// Milliseconds since the tracker started: the clock of session replay's `ended_ms`.
    pub fn uptime_ms(&self) -> u64 {
        self.started.elapsed().as_millis() as u64
    }

    /// The kept records of [`Self::finish`], oldest first.
    pub fn finished(&self) -> Vec<Value> {
        self.finished
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .iter()
            .cloned()
            .collect()
    }

    /// Every entry, ordered by id.
    pub fn snapshot(&self) -> Vec<Entry> {
        self.map().values().cloned().collect()
    }

    /// The first request past the queue (schema v1 `active`).
    pub fn first_running(&self) -> Option<Entry> {
        self.map()
            .values()
            .find(|e| e.phase != Phase::Queued)
            .cloned()
    }

    /// Requests in `prefill` or `generating`.
    pub fn busy(&self) -> usize {
        self.map()
            .values()
            .filter(|e| e.phase != Phase::Queued)
            .count()
    }

    pub fn stage(&self) -> Stage {
        *self.stage.borrow()
    }

    pub fn watch_stage(&self) -> watch::Receiver<Stage> {
        self.stage.subscribe()
    }

    /// SIGTERM (D052): refuses new work and cancels queued requests at once, gives running
    /// ones `grace` to finish, then cancels them. Returns when no request is left or right
    /// after the cancel.
    pub async fn drain(&self, grace: Duration) {
        self.stage.send_replace(Stage::Draining);
        let cancel_queued = |map: &BTreeMap<u64, Entry>| {
            for entry in map.values().filter(|e| e.phase == Phase::Queued) {
                if let Some(ticket) = &entry.ticket {
                    ticket.cancel();
                }
            }
        };
        cancel_queued(&self.map());
        let deadline = tokio::time::Instant::now() + grace;
        while tokio::time::Instant::now() < deadline && !self.map().is_empty() {
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        self.stage.send_replace(Stage::Stopping);
        for entry in self.map().values() {
            if let Some(ticket) = &entry.ticket {
                ticket.cancel();
            }
        }
    }
}

/// The models `/v1/models` lists: the served one, or while loading the one being loaded.
fn listed_model(state: &ServerState) -> Option<(String, Option<u32>, Option<u32>)> {
    match &*state.readiness.borrow() {
        Readiness::Ready(loaded) => {
            let capacity = loaded.handle.budget().kv_capacity_tokens;
            let max_output = state.args.max_output.unwrap_or(DEFAULT_MAX_OUTPUT);
            Some((
                loaded.model_id.clone(),
                Some(capacity),
                Some(max_output.min(capacity.saturating_sub(1))),
            ))
        }
        Readiness::Loading => state.args.model.clone().map(|id| (id, None, None)),
        Readiness::Failed(_) => None,
    }
}

pub(crate) fn model_json(id: &str, capacity: Option<u32>, max_output: Option<u32>) -> Value {
    let mut model = json!({
        "id": id,
        "type": "model",
        "display_name": id,
        "created_at": "1970-01-01T00:00:00Z",
        "object": "model",
        "created": 0,
        "owned_by": "mrun",
    });
    if let (Some(capacity), Some(max_output)) = (capacity, max_output) {
        model["context_window"] = capacity.into();
        // The vLLM-style field OpenCode's local discovery reads.
        model["max_model_len"] = capacity.into();
        model["max_output_tokens"] = max_output.into();
    }
    model
}

/// `GET /v1/models`: answers at once in every state (D047), in a shape both Anthropic (`data`
/// of `{type, id, display_name, created_at}`, `has_more`) and OpenAI (`object: list`) clients
/// read; while loading the model being loaded, without capacity fields.
pub async fn models(State(state): State<ServerState>) -> Response {
    let data: Vec<Value> = listed_model(&state)
        .map(|(id, capacity, max_output)| model_json(&id, capacity, max_output))
        .into_iter()
        .collect();
    let first = data.first().map(|m| m["id"].clone()).unwrap_or(Value::Null);
    axum::Json(json!({
        "object": "list",
        "data": data,
        "has_more": false,
        "first_id": first,
        "last_id": first,
    }))
    .into_response()
}

/// `GET /v1/models/{id}`: the listed model under its id (any requested id names it, as
/// requests do), or 404.
pub async fn model(State(state): State<ServerState>, Path(id): Path<String>) -> Response {
    match listed_model(&state) {
        Some((_, capacity, max_output)) => {
            let mut model = model_json(&id, capacity, max_output);
            // The model's own sampling defaults and thinking mode, for the window's chat
            // inspector (018 FR-010a): mrun's extension, additive.
            if let Ok(loaded) = handler::ready(&state) {
                let d = loaded.defaults;
                model["defaults"] = json!({
                    "temperature": d.temperature,
                    "top_p": d.top_p,
                    "top_k": d.top_k,
                    "min_p": d.min_p,
                    "repetition_penalty": d.repetition_penalty,
                    "max_tokens": max_output,
                });
                model["thinking"] = loaded.supports_thinking.into();
            }
            axum::Json(model).into_response()
        }
        None => (
            StatusCode::NOT_FOUND,
            axum::Json(json!({"type": "error", "error": {"type": "not_found_error",
                "message": "no model is loaded"}})),
        )
            .into_response(),
    }
}

fn power_json(state: &ServerState) -> Value {
    let power = state.power.lock().unwrap_or_else(PoisonError::into_inner);
    match power.held_since() {
        Some(since) => json!({"assertion": true, "since": crate::api::record::rfc3339_of(since)}),
        None => json!({"assertion": false}),
    }
}

/// A miss report as `/mrun/status` and the `: mrun` comment show it.
pub fn miss_json(miss: &MissReport) -> Value {
    json!({
        "cached": miss.cached,
        "processed": miss.processed,
        "cause": miss.cause.map(|c| c.as_str()),
        "divergence": miss.divergence,
        "message": miss.message.map(|m| json!({"index": m.index, "role": role_str(m.role)})),
        "gap": miss.gap,
        "unreused": miss.unreused,
        "ring_wrapped": miss.ring_wrapped,
    })
}

/// `POST /mrun/cache/clear`: drops branches, checkpoints and the live state. The clear is a
/// worker command, so it runs between requests and answers once done.
pub async fn clear_cache(State(state): State<ServerState>) -> Response {
    let loaded = match handler::ready(&state) {
        Ok(loaded) => loaded,
        Err(error) => return handler::error::<Chat>(&error),
    };
    let report = match loaded.handle.clear_cache() {
        Ok(receiver) => receiver.recv_async().await.ok().flatten(),
        Err(error) => {
            let error = crate::api::model::ApiError::new(
                crate::api::model::ErrorKind::Internal,
                error.to_string(),
            );
            return handler::error::<Chat>(&error);
        }
    };
    let report = report.unwrap_or_default();
    axum::Json(json!({
        "branches_dropped": report.branches_dropped,
        "bytes_freed": report.bytes_freed,
    }))
    .into_response()
}

/// The engine's backend (016 FR-008): nuvec, the numerics class from the engine, the version and
/// kernel mode from this build (`null` without an engine).
fn backend_json(numerics: &mrun_engine::engine::Numerics) -> Value {
    let build = crate::cli::version::backend();
    json!({
        "name": mrun_engine::BACKEND,
        "version": build.map(|b| b.version),
        "numerics": numerics.class(),
        "kernels": build.map(|b| b.kernels.as_str()),
    })
}

/// `GET /mrun/status`: reads only the tracker, atomics and snapshots (012 FR-012).
pub async fn status(State(state): State<ServerState>) -> Response {
    let mut body = json!({
        "schema_version": 1,
        "version": env!("CARGO_PKG_VERSION"),
        "numerics": Value::Null,
        "engine": Value::Null,
        "pid": std::process::id(),
        "model": Value::Null,
        "budget": Value::Null,
        "active": Value::Null,
        "queued": 0,
        "cache": Value::Null,
        "recording": state.args.record.is_some(),
        "power": power_json(&state),
        "last": Value::Null,
        "requests": [],
        "finished": [],
        "lanes": {"configured": 1, "busy": 0},
        "pool": pool_status(&state),
    });
    if let Ok(loaded) = handler::ready(&state) {
        let info = loaded.handle.info();
        let budget = loaded.handle.budget();
        body["model"] = json!({
            "id": loaded.model_id,
            "identity": info.identity,
            "family": info.family,
            "syntax": ChatSyntax::get(loaded.syntax).name,
        });
        // The numerics class the engine runs with (FR-027, D044; 002 T067).
        body["numerics"] = json!({
            "backend": mrun_engine::BACKEND,
            "nax": if info.numerics.nax { "on" } else { "off" },
            "gpu_family": info.numerics.gpu_arch,
        });
        body["engine"] = json!({ "backend": backend_json(&info.numerics) });
        body["budget"] = json!({
            "budget": budget.budget,
            "weights": budget.weights,
            "scratch": budget.scratch,
            "kv_capacity_tokens": budget.kv_capacity_tokens,
            "kv_bytes_per_token": info.kv_bytes_per_token,
        });
        body["queued"] = loaded.handle.queue().waiting().into();
        if let Some(stats) = loaded.handle.cache_stats() {
            body["cache"] = json!({
                "live_tokens": stats.live_tokens,
                "branches": stats.branches,
                "branch_tokens": stats.branch_tokens,
                "buffer_bytes": stats.buffer_bytes,
                "checkpoints": stats.checkpoints,
                "checkpoint_bytes": stats.checkpoint_bytes,
                "prompt_tokens_total": stats.prompt_tokens_total,
                "cached_tokens_total": stats.cached_tokens_total,
                "spliced_tokens_total": loaded
                    .handle
                    .ledger()
                    .lock()
                    .unwrap_or_else(PoisonError::into_inner)
                    .spliced_total(),
                // FR-020: current-turn assistant messages seen and spliced, per client class.
                "splice": serde_json::to_value(
                    loaded
                        .handle
                        .ledger()
                        .lock()
                        .unwrap_or_else(PoisonError::into_inner)
                        .clients(),
                )
                .unwrap_or_default(),
                "evicted_total": stats.evicted_total,
                "pressure_events": stats.pressure_events,
                "dropped_ring_wrapped": stats.ring_wrapped,
                "enabled": !state.args.no_cache,
            });
            body["last"] = stats.last.as_ref().map_or(Value::Null, miss_json);
        }
        let now = Instant::now();
        let requests = state.tracker.snapshot();
        body["active"] = requests
            .iter()
            .find(|e| e.phase != Phase::Queued)
            .map_or(Value::Null, Entry::active_json);
        body["lanes"]["busy"] = requests
            .iter()
            .filter(|e| e.phase != Phase::Queued)
            .count()
            .into();
        body["requests"] = requests.iter().map(|e| e.json(now)).collect();
        body["finished"] = state.tracker.finished().into();
    }
    axum::Json(body).into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_engine_backend_names_backend_version_numerics_and_kernels() {
        use mrun_engine::engine::Numerics;
        let build = crate::cli::version::backend();
        for (nax, class) in [(false, "nuvec"), (true, "nuvec-nax")] {
            let json = backend_json(&Numerics {
                nax,
                gpu_arch: "applegpu_g16s".into(),
            });
            assert_eq!(json["name"], "nuvec");
            assert_eq!(json["numerics"], class);
            match build {
                Some(b) => {
                    assert_eq!(json["version"], b.version);
                    assert_eq!(json["kernels"], b.kernels.as_str());
                }
                None => assert!(json["version"].is_null() && json["kernels"].is_null()),
            }
        }
    }

    #[test]
    fn entries_move_through_their_phases() {
        let tracker = Tracker::default();
        tracker.insert(Entry::queued(2, ApiKind::Chat, 10));
        tracker.insert(Entry::queued(1, ApiKind::Messages, 300));
        assert!(tracker.first_running().is_none());
        tracker.update(1, |e| {
            e.phase = Phase::Prefill;
            e.cached = Some(100);
            e.progress = 100;
            e.cause = Some(Cause::Diverged);
            e.unreused = Some(2048);
        });
        assert_eq!(tracker.first_running().unwrap().id, 1);
        assert_eq!(tracker.busy(), 1);
        let start = Instant::now();
        tracker.update(1, |e| {
            e.phase = Phase::Generating;
            e.first_token = Some(start);
            e.generated = 5;
        });
        let entry = tracker.first_running().unwrap();
        let later = start + Duration::from_secs(2);
        assert_eq!(entry.decode_tps(later), Some(2.0));
        let json = entry.json(later);
        assert_eq!(json["phase"], "generating");
        assert_eq!(json["cause"], "diverged");
        assert_eq!(json["unreused"], 2048);
        assert!(json["speculation"].is_null());
        let ids: Vec<u64> = tracker.snapshot().iter().map(|e| e.id).collect();
        assert_eq!(ids, [1, 2]);
        assert!(tracker.snapshot()[1].json(later)["cached"].is_null());
        assert!(tracker.snapshot()[1].json(later)["unreused"].is_null());
        tracker.remove(1);
        tracker.remove(2);
        assert!(tracker.snapshot().is_empty());
    }

    #[tokio::test]
    async fn drain_returns_at_once_when_idle_and_moves_to_stopping() {
        let tracker = Tracker::default();
        let stage = tracker.watch_stage();
        let start = tokio::time::Instant::now();
        tracker.drain(Duration::from_secs(5)).await;
        assert!(start.elapsed() < Duration::from_secs(1));
        assert_eq!(*stage.borrow(), Stage::Stopping);
    }
}
