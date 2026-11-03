//! `mrun ps`: the running server's model, budget, requests and cache, from `/mrun/status`
//! (004 FR-008), with one line per queued and running request (012 FR-008). `--watch` redraws
//! every 500 ms on a terminal and prints one JSON line per second otherwise (012 FR-009,
//! contract `mrun ps --watch`); Ctrl-C ends it with exit 0.

use crate::cli::server;
use serde_json::Value;
use std::future::Future;
use std::io::{IsTerminal, Write};
use std::time::Duration;

/// Redraw interval of `--watch` on a terminal (at least one redraw per second).
const TTY_INTERVAL: Duration = Duration::from_millis(500);
/// Snapshot interval of `--watch` without a terminal or with `--json`: one line per second.
const LINE_INTERVAL: Duration = Duration::from_secs(1);

#[derive(Debug, clap::Args)]
pub struct PsArgs {
    /// Print the status JSON.
    #[arg(long)]
    pub json: bool,
    /// Keep polling: redraw on a terminal, one JSON line per second otherwise (or with --json).
    #[arg(long, short = 'w')]
    pub watch: bool,
    /// Port of the server.
    #[arg(long, env = "MRUN_PORT", default_value_t = server::default_port())]
    pub port: u16,
}

pub fn run(args: PsArgs) -> u8 {
    if args.watch {
        return watch_command(&args);
    }
    match server::request(args.port, "GET", "/mrun/status") {
        Ok((_, status)) => {
            if args.json {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&status).unwrap_or_default()
                );
            } else {
                print!("{}", render(&status));
            }
            0
        }
        Err(error) => error.report(),
    }
}

/// The poll interval of `--watch`.
fn interval(live: bool) -> Duration {
    if live { TTY_INTERVAL } else { LINE_INTERVAL }
}

/// `--watch` on stdout until Ctrl-C.
fn watch_command(args: &PsArgs) -> u8 {
    let runtime = match tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
    {
        Ok(runtime) => runtime,
        Err(error) => {
            return crate::errors::CliError::new(crate::errors::Code::Internal, error.to_string())
                .report();
        }
    };
    let live = std::io::stdout().is_terminal() && !args.json;
    let port = args.port;
    runtime.block_on(async move {
        let fetch = move || async move {
            tokio::task::spawn_blocking(move || server::request(port, "GET", "/mrun/status"))
                .await
                .ok()
                .and_then(Result::ok)
                .map(|(_, status)| status)
        };
        let ctrl_c = async {
            let _ = tokio::signal::ctrl_c().await;
        };
        watch(
            fetch,
            &mut std::io::stdout(),
            live,
            port,
            interval(live),
            ctrl_c,
        )
        .await
    })
}

/// The watch loop: every `every`, `fetch` the status (`None`: unreachable) and write a frame
/// to `out`, until `stop` resolves (exit 0, a newline on a terminal) or `out` is closed (a
/// closed pipe, `mrun ps --watch | head`, exit 0 too).
async fn watch<F, Fut>(
    mut fetch: F,
    out: &mut impl Write,
    live: bool,
    port: u16,
    every: Duration,
    stop: impl Future<Output = ()>,
) -> u8
where
    F: FnMut() -> Fut,
    Fut: Future<Output = Option<Value>>,
{
    tokio::pin!(stop);
    let mut ticker = tokio::time::interval(every);
    ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    loop {
        tokio::select! {
            () = &mut stop => break,
            _ = ticker.tick() => {}
        }
        let frame = match fetch().await {
            Some(status) => watch_frame(&status, live),
            None => watch_error(port, live),
        };
        if out
            .write_all(frame.as_bytes())
            .and_then(|()| out.flush())
            .is_err()
        {
            return 0;
        }
    }
    if live {
        let _ = writeln!(out);
    }
    0
}

/// One redraw (cursor home, clear to the end) on a terminal, else one compact JSON line.
fn watch_frame(status: &Value, live: bool) -> String {
    if live {
        format!("\x1b[H\x1b[J{}", render(status))
    } else {
        format!("{}\n", serde_json::to_string(status).unwrap_or_default())
    }
}

/// The unreachable-server line, on its own screen on a terminal.
fn watch_error(port: u16, live: bool) -> String {
    let line = format!("server unreachable on 127.0.0.1:{port}, retrying\n");
    if live {
        format!("\x1b[H\x1b[J{line}")
    } else {
        line
    }
}

/// One request line (contract `mrun ps --watch`).
fn request_line(request: &Value) -> String {
    let phase = request["phase"].as_str().unwrap_or("?");
    let mut detail = match phase {
        "queued" => format!("prompt {}", request["prompt_tokens"]),
        "prefill" => format!("{}/{}", request["progress"], request["prompt_tokens"]),
        _ => {
            let mut d = format!("{} tokens", request["generated"]);
            if let Some(tps) = request["decode_tps"].as_f64() {
                d.push_str(&format!(", {tps:.1} tok/s"));
            }
            let spec = &request["speculation"];
            if let (Some(drafted), Some(accepted)) =
                (spec["drafted"].as_u64(), spec["accepted"].as_u64())
            {
                let rate = (100 * accepted).checked_div(drafted).unwrap_or(0);
                d.push_str(&format!(", spec {rate}%"));
            }
            d
        }
    };
    if phase != "queued" {
        let mut cache = vec![format!(
            "cached {}",
            request["cached"].as_u64().unwrap_or(0)
        )];
        if let Some(cause) = request["cause"].as_str() {
            cache.push(cause.to_string());
        }
        if let Some(unreused) = request["unreused"].as_u64().filter(|&n| n > 0) {
            cache.push(format!("unreused {unreused}"));
        }
        detail.push_str(&format!(" ({})", cache.join(", ")));
    }
    let age = request["age_ms"].as_u64().unwrap_or(0) as f64 / 1000.0;
    format!(
        "#{} {:<9} lane {}  {phase:<10} {detail:<36}  age {age:.1}s\n",
        request["id"],
        request["api"].as_str().unwrap_or("?"),
        request["lane"],
    )
}

fn gib(bytes: &Value) -> String {
    format!(
        "{:.1} GiB",
        bytes.as_u64().unwrap_or(0) as f64 / (1u64 << 30) as f64
    )
}

/// The remote-mode view (013 FR-012, US2): backend, pod, cost, idle timer, privacy note.
fn render_remote(status: &Value) -> String {
    use crate::remote::human_duration;
    let pod = &status["pod"];
    let mut out = String::new();
    match status["model"]["id"].as_str() {
        Some(id) => out.push_str(&format!(
            "model     {id} (remote, {}), context {} tokens\n",
            status["backend"].as_str().unwrap_or("?"),
            status["context"]
        )),
        None => out.push_str(&format!(
            "model     loading on the pod ({})\n",
            pod["phase"].as_str().unwrap_or("unreachable")
        )),
    }
    let cost = pod["cost_per_hr"].as_f64().unwrap_or(0.0);
    out.push_str(&format!(
        "pod       {} on {}, {}, ${cost:.2}/h\n",
        pod["id"].as_str().unwrap_or("?"),
        pod["gpu"].as_str().unwrap_or("?"),
        pod["phase"].as_str().unwrap_or("unreachable"),
    ));
    if let Some(uptime) = pod["uptime_s"].as_u64() {
        let spend = pod["spend_usd"]
            .as_f64()
            .unwrap_or(cost * uptime as f64 / 3600.0);
        match pod["unreachable_s"].as_u64() {
            // Counted to when the pod last answered: after that it has most likely stopped
            // itself, and what it cost since is unknown here ('mrun cloud status' asks RunPod).
            Some(unseen) => out.push_str(&format!(
                "cost      up {} until it stopped answering {} ago, about ${spend:.3}; later cost unknown\n",
                human_duration(uptime),
                human_duration(unseen)
            )),
            None => out.push_str(&format!(
                "cost      up {}, about ${spend:.3} so far\n",
                human_duration(uptime)
            )),
        }
    }
    match (pod["idle_s"].as_u64(), pod["idle_stop_s"].as_u64()) {
        (Some(idle), Some(limit)) => out.push_str(&format!(
            "idle      {} of {}; stops itself after {} at most\n",
            human_duration(idle),
            human_duration(limit),
            human_duration(pod["max_runtime_s"].as_u64().unwrap_or(0))
        )),
        _ => out.push_str(&format!(
            "active    {} request(s)\n",
            status["lanes"]["busy"]
        )),
    }
    if pod["reachable"] == false {
        out.push_str(&format!(
            "warning   the pod is unreachable: {}\n",
            pod["error"].as_str().unwrap_or("?")
        ));
    }
    out.push_str(&format!(
        "note      {}\n",
        status["remote_note"]
            .as_str()
            .unwrap_or(crate::api::relay::PRIVACY_NOTE)
    ));
    out
}

/// Human-readable status.
pub fn render(status: &Value) -> String {
    if !status["pod"].is_null() {
        return render_remote(status);
    }
    let mut out = String::new();
    let model = &status["model"];
    if model.is_null() {
        out.push_str("model     loading\n");
        return out;
    }
    out.push_str(&format!(
        "model     {} ({}, {})\n",
        model["id"].as_str().unwrap_or("?"),
        model["family"].as_str().unwrap_or("?"),
        model["syntax"].as_str().unwrap_or("?"),
    ));
    let budget = &status["budget"];
    out.push_str(&format!(
        "budget    {} = weights {} + scratch {} + state; context {} tokens\n",
        gib(&budget["budget"]),
        gib(&budget["weights"]),
        gib(&budget["scratch"]),
        budget["kv_capacity_tokens"],
    ));
    match (&status["active"], status["requests"].as_array()) {
        // Servers with per-request status (012): one line each.
        (_, Some(requests)) if !requests.is_empty() => {
            for request in requests {
                out.push_str(&request_line(request));
            }
        }
        (Value::Null, _) => out.push_str("active    idle\n"),
        (active, _) => out.push_str(&format!(
            "active    {} request, {} prompt tokens ({} cached), {} {}\n",
            active["api"].as_str().unwrap_or("?"),
            active["prompt_tokens"],
            active["cached"],
            active["phase"].as_str().unwrap_or("?"),
            if active["phase"] == "prefill" {
                format!("{}/{}", active["progress"], active["prompt_tokens"])
            } else {
                String::new()
            },
        )),
    }
    out.push_str(&format!("queued    {}\n", status["queued"]));
    let cache = &status["cache"];
    if !cache.is_null() {
        let prompt = cache["prompt_tokens_total"].as_u64().unwrap_or(0);
        let cached = cache["cached_tokens_total"].as_u64().unwrap_or(0);
        let rate = if prompt > 0 {
            format!("{:.0}%", 100.0 * cached as f64 / prompt as f64)
        } else {
            "–".into()
        };
        let bytes = Value::from(
            cache["buffer_bytes"].as_u64().unwrap_or(0)
                + cache["checkpoint_bytes"].as_u64().unwrap_or(0),
        );
        out.push_str(&format!(
            "cache     {} live tokens, {} branches ({} tokens), {} checkpoints, {}; hit rate {rate}\n",
            cache["live_tokens"],
            cache["branches"],
            cache["branch_tokens"],
            cache["checkpoints"],
            gib(&bytes),
        ));
        if cache["dropped_ring_wrapped"] == true {
            out.push_str(
                "          dropped-path ring wrapped: a new miss may be an old eviction\n",
            );
        }
    }
    if status["recording"] == true {
        out.push_str("recording on (--record)\n");
    }
    let last = &status["last"];
    if !last.is_null() {
        let cause = last["cause"].as_str().unwrap_or("hit");
        let mut line = format!(
            "last      {cause}: {} cached, {} processed",
            last["cached"], last["processed"]
        );
        if let Some(at) = last["divergence"].as_u64() {
            line.push_str(&format!(", diverged at {at}"));
        }
        if !last["message"].is_null() {
            line.push_str(&format!(
                " (message {}, {})",
                last["message"]["index"],
                last["message"]["role"].as_str().unwrap_or("?")
            ));
        }
        if last["gap"].as_u64().unwrap_or(0) > 0 {
            line.push_str(&format!(", gap {}", last["gap"]));
        }
        if last["unreused"].as_u64().unwrap_or(0) > 0 {
            line.push_str(&format!(", unreused {}", last["unreused"]));
        }
        out.push_str(&line);
        out.push('\n');
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_pod_that_stopped_answering_shows_the_cost_until_then() {
        let status = json!({
            "model": null, "backend": "vllm 0.32.0",
            "pod": {"id": "abc", "gpu": "NVIDIA RTX PRO 6000", "cost_per_hr": 1.69,
                "phase": "ready", "uptime_s": 28_800, "spend_usd": 13.52,
                "unreachable_s": 5_400, "reachable": false},
        });
        let text = render_remote(&status);
        assert!(
            text.contains(
                "up 8h00m until it stopped answering 1h30m ago, about $13.520; later cost unknown"
            ),
            "{text}"
        );
        assert!(!text.contains("so far"), "{text}");
    }
    use serde_json::json;

    #[test]
    fn renders_a_relay_with_backend_pod_cost_and_idle() {
        let status = json!({
            "model": {"id": "Qwen/Qwen3-4B-Instruct-2507", "family": "remote"},
            "backend": "vllm 0.32.0", "context": 65536, "budget": null, "cache": null,
            "lanes": {"configured": 1, "busy": 0},
            "pod": {"id": "abc", "gpu": "NVIDIA RTX A5000", "cost_per_hr": 0.16,
                "phase": "ready", "uptime_s": 600, "spend_usd": 0.027, "idle_s": 125,
                "idle_stop_s": 1800, "max_runtime_s": 28800, "reachable": true},
            "remote_note": "remote mode: prompts and code are sent to RunPod",
        });
        let text = render(&status);
        assert!(text.contains("(remote, vllm 0.32.0)"), "{text}");
        assert!(
            text.contains("pod       abc on NVIDIA RTX A5000, ready, $0.16/h"),
            "{text}"
        );
        assert!(text.contains("about $0.027"), "{text}");
        assert!(text.contains("idle      2m of 30m"), "{text}");
        assert!(text.contains("sent to RunPod"), "{text}");
        assert!(!text.contains("budget"), "{text}");
    }

    #[test]
    fn renders_model_requests_cache_and_last_miss() {
        let status = json!({
            "model": {"id": "m", "family": "qwen3_moe", "syntax": "qwen_xml"},
            "budget": {"budget": 30u64 << 30, "weights": 17u64 << 30, "scratch": 1u64 << 30,
                "kv_capacity_tokens": 105000},
            "active": {"api": "messages", "prompt_tokens": 9000, "cached": 8000,
                "phase": "prefill", "progress": 8500},
            "queued": 1,
            "cache": {"live_tokens": 9000, "branches": 2, "branch_tokens": 40000,
                "checkpoints": 0, "prompt_tokens_total": 100, "cached_tokens_total": 90},
            "recording": false,
            "last": {"cached": 8000, "processed": 1000, "cause": "changed", "divergence": 8000,
                "message": {"index": 4, "role": "assistant"}, "gap": 0},
        });
        let text = render(&status);
        assert!(text.contains("model     m (qwen3_moe, qwen_xml)"));
        assert!(text.contains("prefill 8500/9000"));
        assert!(text.contains("hit rate 90%"));
        assert!(text.contains(
            "changed: 8000 cached, 1000 processed, diverged at 8000 (message 4, assistant)"
        ));
        assert_eq!(render(&json!({"model": null})), "model     loading\n");
    }

    #[test]
    fn request_lines_per_phase() {
        let queued = json!({"id": 13, "api": "chat", "lane": 0, "phase": "queued",
            "prompt_tokens": 812, "cached": null, "progress": 0, "cause": null, "unreused": null,
            "generated": 0, "decode_tps": null, "speculation": null, "age_ms": 300});
        let line = request_line(&queued);
        assert!(
            line.starts_with("#13 chat      lane 0  queued     prompt 812"),
            "{line}"
        );
        assert!(line.trim_end().ends_with("age 0.3s"), "{line}");
        let prefill = json!({"id": 12, "api": "messages", "lane": 0, "phase": "prefill",
            "prompt_tokens": 30000, "cached": 9000, "progress": 18432, "cause": "gap",
            "unreused": 2048, "generated": 0, "decode_tps": null, "speculation": null,
            "age_ms": 4100});
        let line = request_line(&prefill);
        assert!(
            line.contains("prefill    18432/30000 (cached 9000, gap, unreused 2048)"),
            "{line}"
        );
        let mut hit = prefill.clone();
        hit["cause"] = Value::Null;
        hit["unreused"] = json!(0);
        assert!(
            request_line(&hit).contains("(cached 9000)"),
            "unreused only when > 0"
        );
        let generating = json!({"id": 11, "api": "messages", "lane": 1, "phase": "generating",
            "prompt_tokens": 100, "cached": 90, "progress": 100, "cause": null, "unreused": 0,
            "generated": 214, "decode_tps": 41.25, "speculation": null, "age_ms": 9800});
        let line = request_line(&generating);
        assert!(
            line.contains("lane 1  generating 214 tokens, 41.2 tok/s (cached 90)"),
            "{line}"
        );
        assert!(!line.contains("spec"), "no speculation, no rate");
        let mut speculating = generating.clone();
        speculating["speculation"] = json!({"drafted": 100, "accepted": 78});
        assert!(request_line(&speculating).contains(", spec 78%"));
        // The view lists every request instead of the v1 `active` line.
        let status = json!({"model": {"id": "m"}, "budget": {}, "active": {"api": "chat"},
            "requests": [generating, queued], "queued": 1});
        let text = render(&status);
        assert!(!text.contains("active "), "{text}");
        assert_eq!(text.matches("lane ").count(), 2);
    }

    #[test]
    fn idle_view_shows_budget_cache_bytes_and_branches() {
        let status = json!({
            "model": {"id": "m", "family": "qwen3", "syntax": "hermes_json"},
            "budget": {"budget": 30u64 << 30, "weights": 17u64 << 30, "scratch": 1u64 << 30,
                "kv_capacity_tokens": 105000},
            "active": null, "requests": [], "queued": 0,
            "cache": {"live_tokens": 0, "branches": 3, "branch_tokens": 40000,
                "buffer_bytes": 3u64 << 30, "checkpoints": 2, "checkpoint_bytes": 1u64 << 29,
                "prompt_tokens_total": 0, "cached_tokens_total": 0, "dropped_ring_wrapped": true},
        });
        let text = render(&status);
        assert!(
            text.contains("budget    30.0 GiB = weights 17.0 GiB + scratch 1.0 GiB"),
            "{text}"
        );
        assert!(text.contains("active    idle"));
        assert!(
            text.contains(
                "cache     0 live tokens, 3 branches (40000 tokens), 2 checkpoints, 3.5 GiB"
            ),
            "{text}"
        );
        assert!(text.contains("dropped-path ring wrapped"));
    }

    #[test]
    fn watch_frames_redraw_or_stream_json_lines() {
        let status = json!({"model": null});
        assert_eq!(watch_frame(&status, false), "{\"model\":null}\n");
        assert!(watch_frame(&status, true).starts_with("\x1b[H\x1b[Jmodel     loading"));
        assert_eq!(
            watch_error(8484, false),
            "server unreachable on 127.0.0.1:8484, retrying\n"
        );
        assert_eq!(interval(true), Duration::from_millis(500));
        assert_eq!(interval(false), Duration::from_secs(1));
    }

    #[tokio::test]
    async fn watch_polls_until_stopped_and_survives_an_unreachable_server() {
        use std::sync::atomic::{AtomicU32, Ordering};
        let polls = std::sync::Arc::new(AtomicU32::new(0));
        let counter = polls.clone();
        // Up, down, up: the loop keeps polling through the outage.
        let fetch = move || {
            let n = counter.fetch_add(1, Ordering::SeqCst);
            async move { (n != 1).then(|| json!({"model": null, "n": n})) }
        };
        let stop = {
            let polls = polls.clone();
            async move {
                while polls.load(Ordering::SeqCst) < 4 {
                    tokio::time::sleep(Duration::from_millis(1)).await;
                }
            }
        };
        let mut out = Vec::new();
        let code = watch(fetch, &mut out, false, 9, Duration::from_millis(5), stop).await;
        assert_eq!(code, 0);
        let text = String::from_utf8(out).unwrap();
        let lines: Vec<&str> = text.lines().collect();
        assert!(lines.len() >= 3, "{text}");
        assert_eq!(lines[1], "server unreachable on 127.0.0.1:9, retrying");
        for line in [lines[0], lines[2]] {
            let parsed: Value = serde_json::from_str(line).unwrap();
            assert!(parsed["model"].is_null());
        }
        // On a terminal: redraws, and a newline at Ctrl-C.
        let mut out = Vec::new();
        let fetch = || async { Some(json!({"model": null})) };
        let stop = tokio::time::sleep(Duration::from_millis(30));
        watch(fetch, &mut out, true, 9, Duration::from_millis(5), stop).await;
        let text = String::from_utf8(out).unwrap();
        assert!(text.matches("\x1b[H\x1b[J").count() >= 2, "{text:?}");
        assert!(text.ends_with('\n'));
    }

    /// A closed pipe ends the watch with exit 0.
    #[tokio::test]
    async fn watch_ends_on_a_closed_pipe() {
        struct Closed;
        impl Write for Closed {
            fn write(&mut self, _: &[u8]) -> std::io::Result<usize> {
                Err(std::io::ErrorKind::BrokenPipe.into())
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }
        let fetch = || async { Some(json!({})) };
        let never = std::future::pending::<()>();
        let code = watch(
            fetch,
            &mut Closed,
            false,
            9,
            Duration::from_millis(1),
            never,
        )
        .await;
        assert_eq!(code, 0);
    }
}
