//! Cost guards inside the pod (013 FR-010): the agent stops its own pod after the idle period
//! (no request running or started) and after the maximum runtime (running responses get up to
//! [`GRACE`] to finish), through the v2 pod action with the pod-scoped key RunPod injects.

use super::{Phase, Shared};
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

/// Time running responses get to finish once the maximum runtime is reached.
pub const GRACE: Duration = Duration::from_secs(60);

/// How the agent stops its pod.
#[derive(Clone, Debug)]
pub enum Stopper {
    /// `POST <v2>/pods/{id}/action {"action":"stop"}` with the pod-scoped key.
    RunPod {
        client: mrun_runpod::Client,
        pod_id: String,
    },
    /// Outside a pod (no `RUNPOD_POD_ID`): the agent exits.
    Exit,
}

impl Stopper {
    /// From `RUNPOD_POD_ID` and the pod-scoped `RUNPOD_API_KEY`.
    pub fn from_env() -> Self {
        match (
            std::env::var("RUNPOD_POD_ID"),
            std::env::var(mrun_runpod::KEY_VARIABLE),
        ) {
            (Ok(pod_id), Ok(key)) if !pod_id.is_empty() && !key.is_empty() => {
                match mrun_runpod::Client::new(mrun_runpod::ApiKey::new(key)) {
                    Ok(client) => Self::RunPod { client, pod_id },
                    Err(_) => Self::Exit,
                }
            }
            _ => Self::Exit,
        }
    }

    pub fn pod_id(&self) -> Option<&str> {
        match self {
            Self::RunPod { pod_id, .. } => Some(pod_id),
            Self::Exit => None,
        }
    }
}

/// Why the agent stops its pod.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Reason {
    Idle,
    MaxRuntime,
    /// The model failed for good; the pod stops once its log could be fetched.
    Failed,
}

impl Reason {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Idle => "idle",
            Self::MaxRuntime => "max_runtime",
            Self::Failed => "failed",
        }
    }
}

/// The reason to stop now, if any.
pub fn due(shared: &Shared, now: Instant) -> Option<Reason> {
    if now.duration_since(shared.started) >= shared.config.max_runtime {
        return Some(Reason::MaxRuntime);
    }
    let failed_stop = shared.config.failed_stop;
    if !failed_stop.is_zero() && shared.failed_for(now).is_some_and(|f| f >= failed_stop) {
        return Some(Reason::Failed);
    }
    let idle = shared.idle_for(now)?;
    (idle >= shared.config.idle_stop).then_some(Reason::Idle)
}

/// Watches the timers every `tick`; stops the pod once one is due. Returns after a successful
/// stop (or, with [`Stopper::Exit`], to let the agent exit).
pub async fn watch(shared: Arc<Shared>, tick: Duration) {
    let reason = loop {
        tokio::time::sleep(tick).await;
        if let Some(reason) = due(&shared, Instant::now()) {
            break reason;
        }
    };
    shared.set_phase(Phase::Stopping(reason));
    shared.note(&format!(
        "stopping the pod ({}: idle limit {}s, runtime limit {}s)",
        reason.as_str(),
        shared.config.idle_stop.as_secs(),
        shared.config.max_runtime.as_secs()
    ));
    // New requests are refused from here; running ones may finish within the grace period.
    let deadline = Instant::now() + shared.config.grace;
    while shared.activity.active.load(Ordering::SeqCst) > 0 && Instant::now() < deadline {
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    match &shared.config.stopper {
        Stopper::Exit => shared.note("not in a RunPod pod; exiting"),
        Stopper::RunPod { client, pod_id } => {
            let mut attempt = 0u32;
            loop {
                attempt += 1;
                match client.self_stop(pod_id).await {
                    Ok(()) => {
                        shared.note(&format!("stop requested for pod {pod_id}"));
                        break;
                    }
                    Err(error) => {
                        shared.note(&format!(
                            "stopping pod {pod_id} failed (attempt {attempt}): {error}"
                        ));
                        tokio::time::sleep(Duration::from_secs(if attempt < 5 { 10 } else { 60 }))
                            .await;
                    }
                }
            }
        }
    }
}
