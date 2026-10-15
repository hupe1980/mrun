# mrun Specs

**Updated**: 2026-10-15 (design sprint 3) · **Status**: No feature implemented beyond the portable
foundation of 001.

mrun is a local inference server for coding agents on Apple Silicon. See
[concepts/VISION](../concepts/VISION.md) for the product, [concepts/ROADMAP](../concepts/ROADMAP.md)
for direction and gates, and the [constitution](../.specify/memory/constitution.md) (v3.2.0) for
engineering principles.

## Product requirements

| ID | Requirement |
| --- | --- |
| PR-1 | Coding agents (Claude Code, Codex, OpenCode and other OpenAI-compatible clients) complete real tool loops, including MCP tools, against a local model. |
| PR-2 | Repeated agent context is never recomputed: exact reuse across turns, subagents and edits, for attention and hybrid models, with explained misses. |
| PR-3 | Memory is budgeted at startup; prompts that cannot fit are refused up front, output limits clamped; memory pressure sheds cache first. |
| PR-4 | Interruptions take effect within a few hundred milliseconds at any depth and keep completed work. |
| PR-5 | Offline and private by default: loopback only, no telemetry, no prompt logging, no persisted state unless opted in. |
| PR-6 | Fresh install to working agent with one command (`mrun launch <client>`). |
| PR-7 | Performance claims are reproducible with `mrun bench` against competitors with their caches on. |

## Features

| # | Feature | Release | Depends on | Covers | Status |
| --- | --- | --- | --- | --- | --- |
| 001 | [Engine core](001-engine-core/spec.md): Qwen3.6 hybrid, Qwen3-Coder, Qwen3-8B; budget; checkpoints; cancellation | 0.1 (M0–M1) | — | PR-3, PR-4, PR-5 | Foundation implemented; MLX engine planned |
| 002 | [Agent APIs](002-agent-apis/spec.md): Messages, Chat Completions, Responses; MCP, namespaces, deferred tools | 0.1 (M2) | 001 | PR-1, PR-4, PR-5 | Planned |
| 003 | [Prefix cache](003-prefix-cache/spec.md): live path plus moved branches, hybrid checkpoints, miss reports | 0.1 (M3) | 001, 002 | PR-2, PR-3 | Planned |
| 004 | [Models and CLI](004-models-cli/spec.md): pull, launch, connect, doctor, ps | 0.1 (M4) | 001; 002 for launch/connect/ps | PR-6 | Planned |
| 005 | [Benchmark harness](005-bench/spec.md): in-order trace replay, fair comparisons | 0.1 (M5) | 002; 003 for cache cells | PR-7 | Planned |
| 008 | [Distribution](008-distribution/spec.md): Homebrew, signed binary, launchd service, `/metrics` | 0.1 (M6) | 002, 004 | PR-6 | Planned |
| 006 | [Speculative decoding](006-speculative-decoding/spec.md): copy-aware n-gram drafts (MTP heads gated on measurement) | After 0.1 | 001, 003, 005 | PR-1 | Planned (measure first) |
| 007 | [Persistent prefix tier](007-persistent-prefix-tier/spec.md): opt-in SSD branches and checkpoints | After 0.1 | 003, 005 | PR-2, PR-5 | Planned (measure first) |
| 009 | [Image input](009-image-input/spec.md): screenshots and image tool results for vision-capable models | After 0.1 | 001, 002, 003 | PR-1 | Planned (measure first) |

Order: 001 → 002 → 003; 004 starts once 001 lands, 005 once 002 lands, 008 once 002 and 004
land; all three run in parallel with 003. 010's 0.1 families land with 001. 006, 007
and 009 start after 0.1, each only after its measure-first check with `mrun bench` on recorded traces.
Each feature has a spec, plan, research, data model, contracts, quickstart and tasks, and is
analyzed with `/speckit-analyze` before implementation.

## Later candidates (not specified)

Parallel sessions with batched decode (only if traces show queueing), Linux/CUDA evaluation.
Dropped in sprint 3: 8-bit KV, idle warm-up, session and compaction eviction hints, Anthropic
`diagnostics`, the bench quality suite.

## Rules

- Each spec owns its user stories, `FR-*` and `SC-*`; plans own technical choices; tasks own work.
- Success criteria are acceptance gates measured with `mrun bench` or the feature's tests on the
  target Mac (M4 Pro, 48 GB). A checked task is not a passed gate.
- "Best-in-class" claims need a dated `mrun bench --compare` report that includes losing cells.
- Changing a spec updates its plan, tasks and the public docs in `site/` in the same change.
