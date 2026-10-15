# mrun Specs

**Updated**: 2026-10-15 · **Status**: Redesign (hard cut); no feature implemented yet.

mrun is a local inference server for coding agents on Apple Silicon. See
[concepts/VISION](../concepts/VISION.md) for the product and
[constitution](../.specify/memory/constitution.md) for engineering principles.

## Product requirements

| ID | Requirement |
| --- | --- |
| PR-1 | Coding agents (Claude Code, Codex, OpenCode and other OpenAI-compatible clients) complete real tool loops against a local model. |
| PR-2 | Repeated agent context is never recomputed: exact prefix reuse across turns, branches and tool loops, with explained misses. |
| PR-3 | Memory is budgeted at startup; requests that cannot fit are refused up front; the Mac never swaps because of mrun. |
| PR-4 | Interruptions take effect immediately; abandoned work never delays the next request. |
| PR-5 | Offline and private by default: loopback only, no telemetry, no prompt logging. |
| PR-6 | First-run to working agent in minutes: pull, serve, launch, doctor. |
| PR-7 | Performance claims are reproducible with `mrun bench` against competitors with their caches on. |

## Features

| # | Feature | Priority | Depends on | Covers | Status |
| --- | --- | --- | --- | --- | --- |
| 001 | [Engine core](001-engine-core/spec.md) | P1 | — | PR-3, PR-4, PR-5 | Planned |
| 002 | [Agent APIs](002-agent-apis/spec.md) | P1 | 001 | PR-1, PR-4, PR-5 | Planned |
| 003 | [Prefix cache](003-prefix-cache/spec.md) | P1 | 001, 002 (SC-001 trace from 005) | PR-2, PR-3 | Planned |
| 004 | [Models and CLI experience](004-models-cli/spec.md) | P2 | 001; 002 for `launch`/`connect`/`ps` | PR-6 | Planned |
| 005 | [Benchmark harness](005-bench/spec.md) | P2 | 002 (003 for SC-005) | PR-7 | Planned |

Order: 001 → 002 → 003; 004 and 005 start once 001 lands and finish after 002. Each feature has a
spec, clarifications, plan, research, data model, contracts, quickstart and tasks; all five were
analyzed with `/speckit-analyze` on 2026-10-14 and the findings fixed. Milestones and gates are in
[concepts/ROADMAP](../concepts/ROADMAP.md).

## Next candidates (not specified yet)

006 hybrid models with state checkpoints, 007 persistent SSD prefix tier, 008 copy-aware
speculative decoding, 009 8-bit attention state; later parallel sessions and Linux/CUDA. Each
becomes a spec only after its measure-first check; order and evidence are in
[concepts/ROADMAP](../concepts/ROADMAP.md).

## Rules

- Each spec owns its user stories, `FR-*` and `SC-*`; plans own technical choices; tasks own work.
- Success criteria are acceptance gates measured with `mrun bench` or the feature's tests on the
  target Mac (M4 Pro, 48 GB). A checked task is not a passed gate.
- "Best-in-class" claims need a dated `mrun bench --compare` report that includes losing cells.
- Changing a spec updates its plan and tasks in the same change.
