//! SC-005 (002 T053): a client that disconnects over a real socket stops its request within
//! 300 ms (p95 over 20 trials) whether it is queued, prefilling or generating, on every API,
//! streaming or not.

use crate::common::{self, Server, Setup};
use serde_json::{Value, json};
use std::time::{Duration, Instant};

const TRIALS: usize = 20;
const P95_BOUND: Duration = Duration::from_millis(300);

/// The request body of trial `i` for API `i % 3`, streaming on odd trials.
fn body(i: usize, text: &str) -> (&'static str, Value) {
    let stream = i % 2 == 1;
    let messages = json!([{"role": "user", "content": text}]);
    match i % 3 {
        0 => (
            "/v1/chat/completions",
            json!({"messages": messages, "stream": stream}),
        ),
        1 => (
            "/v1/messages",
            json!({"max_tokens": 30_000, "messages": messages, "stream": stream}),
        ),
        _ => ("/v1/responses", json!({"input": text, "stream": stream})),
    }
}

async fn requests(server: &Server) -> Vec<Value> {
    let status: Value = server
        .client
        .get(server.url("/mrun/status"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    status["requests"].as_array().cloned().unwrap_or_default()
}

/// Waits until `pred` holds for the status requests (5 s at most); returns them.
async fn wait_for(server: &Server, what: &str, pred: impl Fn(&[Value]) -> bool) -> Vec<Value> {
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        let requests = requests(server).await;
        if pred(&requests) {
            return requests;
        }
        assert!(Instant::now() < deadline, "timed out waiting for {what}");
        tokio::time::sleep(Duration::from_millis(2)).await;
    }
}

fn count(requests: &[Value], phase: &str) -> usize {
    requests.iter().filter(|r| r["phase"] == phase).count()
}

/// Sends trial `i` on its own task; the task reads the whole body until it is aborted.
fn send(server: &Server, i: usize, text: &str) -> tokio::task::JoinHandle<()> {
    let (path, body) = body(i, text);
    // A fresh client per trial: the aborted connection is not pooled.
    let client = reqwest::Client::new();
    let url = server.url(path);
    tokio::spawn(async move {
        if let Ok(response) = client.post(url).json(&body).send().await {
            let _ = response.bytes().await;
        }
    })
}

/// Runs the trials of one phase: start trial `i`, wait until it is in `phase`, disconnect, and
/// time until it is gone with only the `others` requests (in other phases) left. A running
/// trial (`others == 0`) counts as stopped only once the worker has taken the next request:
/// a probe sent right after the disconnect has left the queue. Returns the p95.
async fn trials(
    server: &Server,
    phase: &str,
    others: usize,
    text: impl Fn(usize) -> String,
) -> Duration {
    let mut times = Vec::with_capacity(TRIALS);
    for i in 0..TRIALS {
        let task = send(server, i, &text(i));
        let seen = wait_for(server, phase, |r| count(r, phase) == 1).await;
        let id = seen.iter().find(|r| r["phase"] == phase).unwrap()["id"].clone();
        task.abort();
        let started = Instant::now();
        if others == 0 {
            let probe = send(server, i + 1, "probe");
            wait_for(server, "the worker to take the probe", |r| {
                r.len() == 1 && r[0]["id"] != id && r[0]["phase"] != "queued"
            })
            .await;
            times.push(started.elapsed());
            probe.abort();
            wait_for(server, "the probe to stop", <[Value]>::is_empty).await;
        } else {
            wait_for(server, "the request to stop", |r| {
                r.len() == others && r.iter().all(|r| r["id"] != id)
            })
            .await;
            times.push(started.elapsed());
        }
    }
    times.sort();
    let p95 = times[TRIALS * 95 / 100 - 1];
    eprintln!(
        "disconnect {phase}: p95 {p95:?}, max {:?}",
        times[TRIALS - 1]
    );
    p95
}

#[tokio::test]
async fn xapi_disconnect_generating_p95_300ms() {
    let reply = "z".repeat(50_000);
    let setup = Setup {
        step_delay: Some(Duration::from_millis(10)),
        ..Setup::default()
    };
    let server = common::start_with(&[&reply], setup).await;
    let p95 = trials(&server, "generating", 0, |i| format!("go {i}")).await;
    assert!(p95 <= P95_BOUND, "{p95:?}");
}

#[tokio::test]
async fn xapi_disconnect_prefill_p95_300ms() {
    // 5 ms per layer, four layers per 2,048-token chunk: a 40K-token prompt prefills ~400 ms.
    let setup = Setup {
        capacity: 100_000,
        layer_delay: Some(Duration::from_millis(5)),
        ..Setup::default()
    };
    let server = common::start_with(&[], setup).await;
    // A distinct prompt per trial, so no trial resumes from the chunks of an earlier one.
    let p95 = trials(&server, "prefill", 0, |i| {
        format!("{i:03} {}", "x".repeat(40_000))
    })
    .await;
    assert!(p95 <= P95_BOUND, "{p95:?}");
}

#[tokio::test]
async fn xapi_disconnect_queued_p95_300ms() {
    let reply = "z".repeat(50_000);
    let setup = Setup {
        step_delay: Some(Duration::from_millis(10)),
        ..Setup::default()
    };
    let server = common::start_with(&[&reply], setup).await;
    // One request holds the worker while the trials wait behind it.
    let blocker = send(&server, 1, "hold");
    wait_for(&server, "the blocker", |r| count(r, "generating") == 1).await;
    let p95 = trials(&server, "queued", 1, |i| format!("wait {i}")).await;
    assert!(p95 <= P95_BOUND, "{p95:?}");
    blocker.abort();
}
