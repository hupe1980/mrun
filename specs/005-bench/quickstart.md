# Quickstart: Benchmark Harness

## Prerequisites

Target Mac idle, on AC power, Low Power Mode off. `mrun serve` running with the model under test.
Optional competitors installed; their `bench/profiles/*.toml` set start/stop/health/reset commands
and the same context length as mrun.

## Scenarios

1. **Self benchmark**: `mrun bench --suite synthetic` → `mrun-bench/<ts>/report.md` with
   prefill/decode cells at 1K/8K/32K/max (min/median/max). Run twice; medians agree within 5%
   (SC-001).
2. **Agent suite**: `mrun bench --suite agent` → per-turn TTFT, cached vs processed tokens, peak
   memory, malformed tool calls and early EOS.
3. **Comparison**: stop all servers, then
   `mrun bench --compare --system mrun=http://127.0.0.1:8484,profile=mrun --system llama.cpp=http://127.0.0.1:8080,apis=chat,profile=llama.cpp --system ollama=http://127.0.0.1:11434,profile=ollama --suite agent`.
   The harness starts each system from its profile, resets caches before each repetition, stops
   it and checks free memory before the next; never two are loaded at once. For a profile without
   commands it prints "Stop X, start Y, press Enter" and waits. → one report with every system and
   losing cells listed (SC-003). A competitor that cannot be reached gets "not run" cells; exit 0.
4. **Regression**: `mrun bench --suite synthetic --baseline bench/baselines/m4pro.json`; inject a
   15% slowdown via the fake SSE server's delay (T014) → exit 7; five unchanged runs → exit 0 (SC-004).

Automated: `cargo test -p mrun --test bench_driver` runs the driver against a fake SSE server.
