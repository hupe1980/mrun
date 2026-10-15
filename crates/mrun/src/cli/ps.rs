//! `mrun ps`: the running server's model, budget, requests and cache, from `/mrun/status`
//! (004 FR-008).

use crate::cli::server;
use serde_json::Value;

#[derive(Debug, clap::Args)]
pub struct PsArgs {
    /// Print the status JSON.
    #[arg(long)]
    pub json: bool,
    /// Port of the server.
    #[arg(long, env = "MRUN_PORT", default_value_t = server::default_port())]
    pub port: u16,
}

pub fn run(args: PsArgs) -> u8 {
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

fn gib(bytes: &Value) -> String {
    format!(
        "{:.1} GiB",
        bytes.as_u64().unwrap_or(0) as f64 / (1u64 << 30) as f64
    )
}

/// Human-readable status.
pub fn render(status: &Value) -> String {
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
    match &status["active"] {
        Value::Null => out.push_str("active    idle\n"),
        active => out.push_str(&format!(
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
        out.push_str(&format!(
            "cache     {} live tokens, {} branches ({} tokens), {} checkpoints; hit rate {rate}\n",
            cache["live_tokens"], cache["branches"], cache["branch_tokens"], cache["checkpoints"],
        ));
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
        out.push_str(&line);
        out.push('\n');
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

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
}
