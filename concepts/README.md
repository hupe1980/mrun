# mrun Concepts

Design notes behind mrun. Specs own requirements; these pages explain direction and rationale.

| Page | What it covers |
| --- | --- |
| [VISION](VISION.md) | Problem, users, where mrun must win, non-goals |
| [ARCHITECTURE](ARCHITECTURE.md) | Components, data flow, KV and prefix store, memory model, threading |
| [DECISIONS](DECISIONS.md) | Architecture decision records |
| [ROADMAP](ROADMAP.md) | Milestones and gates |
| [RISKS](RISKS.md) | Top risks and mitigations |

## Research

| Study | Summary |
| --- | --- |
| [competitors-2026-10](research/competitors-2026-10.md) | Source study of 13 competitors; what mrun adopted and rejected |
| [arxiv-sota-2026-10](research/arxiv-sota-2026-10.md) | 2025–2026 papers: agent serving, eviction, speculation, hybrids, SSD tiers |
| [next-features-2026-10](research/next-features-2026-10.md) | Speculative decoding, KV quantization, hybrids, SSD tier, batching |
| [landscape-2026-10](research/landscape-2026-10.md) | Competitors, client API needs, models that fit 48 GB, DX patterns |
| [mlx-rust](research/mlx-rust.md) | MLX Rust bindings, mlx-lm internals, MLX model format |
| [literature-study](research/literature-study.md) | Papers on prefix reuse, agent serving, KV compression, speculation |
| [mistralrs-study](research/mistralrs-study.md) | What to adopt and avoid from mistral.rs |
| [fact-check-2026-10-15](research/fact-check-2026-10-15.md) | Corrections applied to the specs on 2026-10-15 |
| [archive](research/archive/) | Candle evaluation (why D010 and D012) |
