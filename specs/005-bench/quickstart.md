# Quickstart: Benchmark Harness

## Prerequisites

Target Mac idle, on AC power, Low Power Mode off. `mrun serve` running with the model under test.
Optional competitors installed (llama.cpp, Ollama, oMLX, mlx-lm, Splash); their
`bench/profiles/*.toml` set reset steps, flags and the same context length as mrun.

## Scenarios

1. **Self benchmark**: `mrun bench --suite synthetic` → `mrun-bench/<ts>/report.md` with
   prefill/decode cells at 1K/8K/32K/max (min/median/max). Run twice; medians agree within 5%
   (SC-001).
2. **Agent suite**: `mrun bench --suite agent` → per-turn TTFT, cached vs processed tokens, miss
   causes and `template` re-renders, malformed tool calls and early EOS.
3. **Comparison**: stop all servers, then
   `mrun bench --compare --system mrun=http://127.0.0.1:8484,profile=mrun --system llama.cpp=http://127.0.0.1:8080,apis=chat,profile=llama.cpp --system ollama=http://127.0.0.1:11434,profile=ollama --system splash-bf16=http://127.0.0.1:8000,profile=splash --suite agent`.
   Before each system the harness prints "Stop X, start Y, press Enter" and waits; it resets
   caches before each repetition (or pauses for a restart where the profile says `manual`). →
   one report with every system and losing cells listed (SC-003, SC-005). A competitor that cannot
   be reached gets "not run" cells; exit 0.

Automated: `cargo test -p mrun --test bench_driver` runs the driver against a fake SSE server.
