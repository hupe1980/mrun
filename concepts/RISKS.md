# Risks

| ID | Risk | Mitigation | Status |
| --- | --- | --- | --- |
| R01 | MLX bindings lack ops or lag releases | Own `mlx-sys` over mlx-c v0.7.0 (every needed function checked); pin MLX ≥ 0.32.4 (deadlock and stream fixes); one-week binding spike with go/no-go | Open |
| R02 | Competitors match the cache story (SGLang-MLX radix cache, oMLX divergence reports, Dynamo next-turn prefill) | Win on moving branches instead of losing them, in-band miss causes, memory admission, cancellation that keeps progress, API fidelity; prove it with a fair, dated benchmark | Open |
| R03 | Templates re-render history differently (reasoning dropped, arguments re-serialized), so the previous reply never hits | HF-compatible Jinja with golden tests; checkpoint at the end of each prompt; `template` miss cause measured in bench | Open |
| R04 | Chunked and one-shot prefill differ in the last bits | Never recompute a cached prefix; compare cache-on with cache-off at the same chunk split | Open |
| R05 | Hybrid model port (Gated DeltaNet kernel) slower or harder than planned | Port mlx-lm's `qwen3_5` including its Metal kernel via `mlx_fast_metal_kernel`; Qwen3-Coder stays a complete alternative so the release does not depend on it | Open |
| R06 | Other apps grow after admission | Pressure observer drops branches, then an idle live path; budget reserve scales with RAM | Open |
| R07 | Clients change wire formats or settings between releases (oMLX broke on Claude Code 2.1.154) | Nightly compatibility matrix against the latest Claude Code, Codex and OpenCode; `mrun doctor --agent` | Open |
| R08 | Scope creep toward a general-purpose server | Non-goals in [VISION](VISION.md); constitution Principle VI | Ongoing |
| R09 | Cold prefills take minutes; clients time out | Keepalives every 10 s in each API's own format; `launch` raises client timeouts; completed chunks survive cancellation | Open |
| R10 | Long-session leaks (MLX graph references, Metal buffer limits) | Detached copies with pointer assertions; 2-hour soak with flat active memory and buffer count | Open |
| R11 | A resident 20 GB+ footprint on a shared laptop | Wire only weights and active state; cache shrinks under pressure; `mrun stop` | Accepted |
