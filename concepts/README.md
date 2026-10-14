# mrun Concepts

Design notes behind mrun. Specs own requirements; these pages explain direction and rationale.

| Page | What it covers |
| --- | --- |
| [VISION](VISION.md) | Problem, users, what best-in-class means, non-goals |
| [ARCHITECTURE](ARCHITECTURE.md) | Components, data flow, KV and prefix store, memory model, threading |
| [DECISIONS](DECISIONS.md) | Architecture decision records |
| [ROADMAP](ROADMAP.md) | Milestones and gates |
| [RISKS](RISKS.md) | Top risks and mitigations |

## Research

| Study | Summary |
| --- | --- |
| [landscape-2026-10](research/landscape-2026-10.md) | Competitors, client API needs, models that fit 48 GB, DX patterns |
| [mlx-rust](research/mlx-rust.md) | MLX Rust bindings, mlx-lm internals, MLX model format |
| [literature-study](research/literature-study.md) | Papers on prefix reuse, agent serving, KV compression, speculation |
| [mistralrs-study](research/mistralrs-study.md) | What to adopt and avoid from mistral.rs |
| [candle-capabilities](research/candle-capabilities.md) | Why Candle was dropped (D010) |
| [candle-budget](research/candle-budget/results.md) | Allocation-bound experiment; why memory is budgeted by construction (D012) |

## Workflow

Features follow Spec Kit: `/speckit-specify` → `/speckit-clarify` → `/speckit-plan` →
`/speckit-tasks` → `/speckit-analyze` → `/speckit-implement`. Skills are installed for Claude Code
(`.claude/skills`) and Copilot (`.github/skills`). Select the active feature with
`SPECIFY_FEATURE_DIRECTORY=specs/NNN-name` or `.specify/feature.json`.
