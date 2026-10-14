# Quickstart: Prefix Cache

## Automated

```sh
cargo test -p mrun --test prefix_tree_props --test prefix_mutations     # any OS
MRUN_TEST_MODELS=qwen3-8b cargo test -p mrun-engine --test prefix_numerics -- --ignored   # macOS
```

## Manual

1. `mrun serve --model qwen3-coder`; run a Claude Code session for 10+ turns; `mrun ps` shows hit
   rate > 90% and cached tokens per turn.
2. Remove `CLAUDE_CODE_ATTRIBUTION_HEADER=0` from the client env, run two turns; `mrun ps` last miss
   says `volatile` with the divergence position; `mrun doctor` suggests the fix.
3. `mrun bench --suite agent` with and without `--no-cache` (SC-001: ≥ 5× median TTFT).
4. Memory pressure: run `memory_pressure -S -l warn` (macOS tool) while serving; cache bytes drop,
   running request continues (SC-005).
