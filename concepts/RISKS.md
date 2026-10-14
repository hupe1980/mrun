# Risks

| ID | Risk | Mitigation | Status |
| --- | --- | --- | --- |
| R01 | MLX Rust bindings immature, lag MLX releases, or lack ops we need | 001 opens with a binding spike and a go/no-go; thin own safe layer over the C API if needed | Open |
| R02 | Competitors (llama.cpp, Ollama MLX, oMLX) already match the cache story | Win on explainability, memory predictability, cancellation and API fidelity; publish a fair benchmark with their caches on | Open |
| R03 | Exact reuse breaks because clients inject volatile content (Claude Code attribution header) or templates re-render history differently | Miss diagnostics with first divergent token; `mrun connect` sets known client env (`CLAUDE_CODE_ATTRIBUTION_HEADER=0`); template round-trip tests | Open |
| R04 | Chunked vs one-shot prefill not bit-identical | Reuse stored KV, never recompute a cached prefix; fixed chunk grid recorded in cache key | Open |
| R05 | MoE model port slower than mlx-lm | Port from mlx-lm reference, compare tok/s in `mrun bench` against mlx-lm on the same weights | Open |
| R06 | Unified memory pressure from other apps after admission | Pressure observer sheds prefix store first; explicit failure if active work cannot continue | Open |
| R07 | Hybrid-attention models dominate new releases | Prefix store nodes carry state checkpoints from day one (D017) | Open |
| R08 | Scope creep toward a general-purpose server | Non-goals in [VISION](VISION.md); constitution Principle VI | Ongoing |
