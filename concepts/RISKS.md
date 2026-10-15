# Risks

| ID | Risk | Mitigation | Status |
| --- | --- | --- | --- |
| R01 | MLX bindings lack ops or lag releases | Own `mlx-sys` over mlx-c v0.7.0 (every needed function checked); pin MLX ≥ 0.32.4 (deadlock and stream fixes); one-week binding spike with go/no-go | Open |
| R02 | Competitors match the cache story (SGLang-MLX radix cache, oMLX divergence reports, Dynamo next-turn prefill) or the default model's speed: Inco Splash serves Qwen3.6-35B-A3B on M3+ with own Metal kernels, DFlash2 drafts and a persistent cache (vendor: 210 tok/s decode, 123 ms cached TTFT at 32K, M5 Pro) | Win on exact bf16 reuse with explained misses, moved branches, a fixed budget without idle unload, cancellation that keeps progress, client fidelity and open families (M1+); prove it with a fair, dated benchmark that includes Splash in bf16 KV (005 SC-005); decode gap: 006 n-gram drafts, DFlash2 as a measured option | Open |
| R03 | Templates re-render history differently (reasoning dropped, arguments re-serialized), so the previous reply never hits | HF-compatible Jinja with golden tests; checkpoint at the end of each prompt; re-renders reported as `changed` with message index and role, sized by bench's `template` diagnostic | Open |
| R04 | Chunked and one-shot prefill differ in the last bits | Never recompute a cached prefix; compare cache-on with cache-off at the same chunk split | Open |
| R05 | Hybrid model port (Gated DeltaNet kernel) slower or harder than planned | Port mlx-lm's `qwen3_5` including its Metal kernel via `mlx_fast_metal_kernel`; Qwen3-Coder stays a complete alternative so the release does not depend on it | Open |
| R06 | Other apps grow after admission | Pressure observer drops branches, then an idle live path; budget reserve scales with RAM | Open |
| R07 | Clients change wire formats or settings between releases (oMLX broke on Claude Code 2.1.154; Codex 0.163 ends a turn on `response.incomplete`; Claude Code 2.1.299 idle compaction) | Pinned-version smoke on every PR; a manual check against the latest Claude Code (npm `stable` and `latest`), Codex and OpenCode before each release; `mrun launch` sets the settings that matter | Open |
| R08 | Scope creep toward a general-purpose server | Non-goals in [VISION](VISION.md); constitution Principle VI | Ongoing |
| R09 | Cold prefills take minutes; clients time out (Claude Code before 2.1.299 aborts after 180 s of silence, Codex after 300 s) | Keepalives every 10 s in each API's own format; `launch` raises client timeouts; completed chunks survive cancellation | Open |
| R10 | Long-session leaks (MLX graph references, Metal buffer limits) | Detached copies with pointer assertions; 2-hour soak with flat active memory and buffer count | Open |
| R11 | A resident 20 GB+ footprint on a shared laptop | Wire only weights and active state; cache shrinks under pressure; `mrun stop` | Accepted |
| R12 | The lean cut (D025) leaves out something users need on day one (service, persistent client config, non-loopback access) | Every cut keeps its spec and ID; 0.2 picks the most requested; `connect` prints the config users can paste | Accepted |
