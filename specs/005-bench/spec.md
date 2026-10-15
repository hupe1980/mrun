# Feature Specification: Benchmark Harness

**Feature Branch**: `005-bench`

**Created**: 2026-10-14

**Status**: Draft

**Input**: User description: "`mrun bench` measures what agents feel and compares fairly: agent-trace replay (recorded multi-turn sessions with tool results), synthetic long-context and decode cells, cancellation latency and memory; runs mrun and competitors (llama.cpp server, Ollama, oMLX, mlx-lm server) on the same machine with their caches enabled, records full settings and machine state, emits JSON plus a markdown report; quality checks so speed never hides wrong output."

## Clarifications

### Session 2026-10-14

- Q: Where do agent traces come from? → A: Recorded by maintainers running OpenCode and Codex on public open-source repositories with `mrun serve --record` (002 FR-021), then automatically redacted, secret-scanned and reviewed. Claude Code traces stay local (its system prompt is proprietary). Revised after analysis on 2026-10-14.
- Q: How does replay treat assistant turns? → A: Teacher-forced: each system is first given the recorded previous turn including the recorded reply (untimed), so caches hold identical history.
- Q: What is the primary comparison metric? → A: Median time to first token per turn and total session time on the agent suite; tok/s cells are secondary.
- Q: Which competitor versions? → A: The latest stable release of each at measurement time, recorded in the report; profiles document each one's recommended cache flags.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Measure my agent workload (Priority: P1)

A maintainer or user runs `mrun bench` and gets the numbers that matter for agent work: time to
first token per turn on a replayed agent session, prompt-processing and generation speed at short
and long context, cancellation latency, and peak memory.

**Why this priority**: Every performance decision and claim depends on trustworthy measurements.

**Independent Test**: Run `mrun bench --suite agent` against the running server and confirm a JSON
result and markdown report with all cells, settings and machine state.

**Acceptance Scenarios**:

1. **Given** a running mrun server, **When** `mrun bench --suite agent` runs, **Then** it replays
   the bundled agent traces and reports per-turn time to first token, processed/cached tokens and
   total session time.
2. **Given** `--suite synthetic`, **When** it runs, **Then** it reports prompt and generation speed
   at 1K, 8K, 16K, 30K, 64K and maximum context, with spread from repeated runs.
3. **Given** any run, **When** it finishes, **Then** the result records hardware, OS, power state,
   model hashes, mrun version and every setting, so it can be reproduced.

---

### User Story 2 - Fair comparison with competitors (Priority: P1)

The maintainer runs the same suites against llama.cpp server, Ollama, oMLX and the reference
MLX server on the same Mac, each with its prefix caching and recommended optimizations enabled,
and publishes a dated comparison including cells where mrun loses.

**Why this priority**: "Best-in-class" is only meaningful as a fair, reproducible comparison.

**Independent Test**: Run `mrun bench --compare` with competitor endpoints configured and check
that each competitor's version, flags and model files are recorded alongside results.

**Acceptance Scenarios**:

1. **Given** competitor endpoints, **When** the comparison runs, **Then** each system runs the same
   traces and cells sequentially (never concurrently) with warm-up and repeated measurements.
2. **Given** differing model files between systems, **When** reported, **Then** the report marks
   the cell as a quality-matched comparison, not same-weights.
3. **Given** a system that fails a cell, **When** reported, **Then** the failure is shown, not
   dropped.

---

### User Story 3 - Speed never hides wrong output (Priority: P2)

Each run checks output quality so a configuration that is faster but wrong is caught.

**Why this priority**: Fast wrong answers are worse than slow right ones for coding.

**Independent Test**: Run the quality suite and verify task pass rates are reported next to speed.

**Acceptance Scenarios**:

1. **Given** the quality suite, **When** run, **Then** it executes bundled coding tasks with
   objective tests and reports pass rates per system.
2. **Given** an mrun change, **When** the regression check runs, **Then** it fails if speed or
   pass rate drops beyond the configured tolerance (greedy-output changes are caught by feature
   001's `mrun run --check`).

### Edge Cases

- Machine under load or on battery/low-power: recorded, and the report warns.
- Thermal throttling during long runs: flagged when a cell slows by more than 5% (FR-012).
- A system without the trace's API (for example Responses traces on oMLX): cell marked "n/a".
- Competitor not installed: the cell is "not run" with the reason.
- A trace turn exceeds a system's context: recorded as a failure for that system.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: `mrun bench` MUST provide suites: `agent` (trace replay), `synthetic` (context and
  generation cells at 1K, 8K, 16K, 30K, 64K and the system's maximum), `cancel` (interruption
  latency), `memory` (peak and steady memory), `quality` (coding tasks with tests).
- **FR-002**: Bundled agent traces MUST come from open-source clients (OpenCode, Codex with NOTICE
  attribution) on public repositories, after automated redaction (paths, usernames, emails) and a
  secret scan. Traces from proprietary clients (Claude Code) MAY be recorded and replayed locally
  but MUST NOT be committed.
- **FR-003**: Agent replay MUST use teacher-forced history: before timing turn n, the harness sends
  turn n−1's request including the recorded assistant reply (output limited to one token, untimed),
  so every system's cache holds the same history the real client would have produced; the report
  states this method. Requests are sent in the trace's API style (Chat Completions, Messages or
  Responses); systems lacking that API are marked "n/a" for the trace.
- **FR-003a**: Traces MUST record the gap between each response's end and the next request (tool
  execution and user think time). Replay MUST reproduce gaps capped at 60 s by default
  (`--gaps recorded|none`), because time-based cache expiry and background work in other systems
  depend on them; the report states the mode. Each system's timed outputs MUST be checked for
  malformed tool calls and early end-of-sequence, reported per system next to its speed.
- **FR-004**: Repetitions: synthetic and cancel cells use one warm-up and at least 5 repetitions;
  agent traces use 3 repetitions; quality tasks use 3 seeds. Reports give median, p95 and IQR.
- **FR-005**: Results MUST be written as JSON (versioned, hand-written schema file) and a markdown
  report, including per-turn records for agent cells (time to first token, time to first useful
  token, input, cached and output tokens).
- **FR-006**: Results MUST record machine state (chip, memory, macOS, power source, low-power mode,
  thermal state, free memory and swap before each system) and, per system, version, flags, model
  name, file format, quantization (bits, group size, scheme) and file hashes when local.
- **FR-007**: `--compare` MUST drive other systems over HTTP with a per-system model name; decode
  cells use temperature 0, a fixed output length and the system's ignore-EOS option where
  available, labeling cells that ended early; each competitor's recommended cache and performance
  flags live in bundled profiles and are recorded per run.
- **FR-008**: Only the system under test may be running: others MUST be stopped or have no model
  loaded, verified by the recorded free memory.
- **FR-009**: The quality suite MUST run objective coding tasks (supplied tests) in a temporary copy
  with time limits and no network access, and report pass rate per system alongside speed.
- **FR-010**: A regression mode MUST compare against a stored baseline and fail (exit 7) on
  configured thresholds (speed, quality); the greedy-conformance check is feature 001's `--check`.
- **FR-011**: Bench MUST NOT contact any host other than the systems under test; publishing results
  is a manual step.
- **FR-012**: A thermal-throttling warning MUST be emitted when a cell's last repetition is more than
  5% slower than its first.

### Key Entities

- **Suite / Cell**: A named measurement with inputs and settings.
- **System Under Test**: Name, version, endpoint, flags, model files.
- **Run Result**: Machine state, system, cell, repetitions, statistics, failures.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Two consecutive runs of the synthetic suite on an idle target Mac agree within 5% on
  median speeds.
- **SC-002**: A full comparison of mrun and three competitors on the bundled agent suite (3 repetitions)
  completes unattended in under 4 hours on the target Mac.
- **SC-003**: Every published report contains all cells for all systems, with failures and "n/a"
  marked and a "losing cells" section (review checklist, 100%).
- **SC-004**: The regression mode catches a deliberately introduced 10% speed regression in 100% of
  trials.
- **SC-005**: Against llama.cpp server, Ollama and oMLX with their prompt caches enabled, on the
  agent suite with Qwen3-Coder-30B-A3B (MLX 4-bit for mrun and oMLX, GGUF Q4_K_M for llama.cpp
  and Ollama, labeled quality-matched), mrun's median time to first token is equal or better;
  every losing cell is published (moved here from feature 003).

## Assumptions

- Competitors are installed by the user; mrun does not install them.
- Agent traces are recorded with mrun's own server in a debug mode and reviewed for secrets before
  bundling.
- The harness talks to systems over HTTP only.
