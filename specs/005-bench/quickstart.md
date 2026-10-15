# Quickstart: Benchmark Harness

## Prerequisites

Target Mac idle, on AC power, Low Power Mode off. `mrun serve` running with the model under test.
Optional competitors installed and started per `bench/profiles/*.toml`.

## Scenarios

1. **Self benchmark**: `mrun bench --suite synthetic` → `mrun-bench/<ts>/report.md` with
   prefill/decode cells (median, p95, IQR). Run twice; medians agree within 5% (SC-001).
2. **Agent suite**: `mrun bench --suite agent` → per-turn TTFT, cached vs processed tokens.
3. **Comparison**: start llama.cpp server (default port 9931) and Ollama on :11434, then
   `mrun bench --compare --system mrun=http://127.0.0.1:8484 --system llama.cpp=http://127.0.0.1:9931 --system ollama=http://127.0.0.1:11434 --suite agent`
   → one report with every system and losing cells listed (SC-003).
4. **Regression**: `mrun bench --suite synthetic --baseline bench/baselines/m4pro.json`; inject a
   10% slowdown (debug flag) → exit 7 (SC-004).

Automated: `cargo test -p mrun --test bench_driver` runs the driver against a fake SSE server.
