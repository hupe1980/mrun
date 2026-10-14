# Feature Specification: Benchmark Harness

**Feature Branch**: `005-bench`

**Created**: 2026-10-14

**Status**: Draft

**Input**: User description: "`mrun bench` measures what agents feel and compares fairly: agent-trace replay (recorded multi-turn sessions with tool results), synthetic long-context and decode cells, cancellation latency and memory; runs mrun and competitors (llama.cpp server, Ollama, oMLX, mlx-lm server) on the same machine with their caches enabled, records full settings and machine state, emits JSON plus a markdown report; quality checks so speed never hides wrong output."

## Clarifications

### Session 2026-10-14

- Q: Where do agent traces come from? → A: Recorded by mrun maintainers running the supported clients on public open-source repositories, with mrun's request recorder (opt-in, debug only), then reviewed and redacted before bundling.
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
   at 1K, 8K, 16K and 30K context, with variance from repeated runs.
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
2. **Given** an mrun change, **When** the regression check runs, **Then** it fails if greedy outputs
   change on the conformance set or pass rate drops beyond the configured tolerance.

### Edge Cases

- Machine under load or on battery/low-power: recorded, and the report warns.
- Thermal throttling during long runs: detected via speed drift and flagged.
- Competitor not installed: the cell is "not run" with the reason.
- A trace turn exceeds a system's context: recorded as a failure for that system.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: `mrun bench` MUST provide suites: `agent` (trace replay), `synthetic` (context and
  generation cells), `cancel` (interruption latency), `memory` (peak and steady memory), `quality`
  (coding tasks with tests).
- **FR-002**: Agent traces MUST be bundled, redistributable recordings of real multi-turn coding
  sessions including tool results, replayed with their original timing gaps optional.
- **FR-003**: Every measurement MUST use warm-up runs, at least 5 repetitions and report median,
  p95 and spread.
- **FR-004**: Results MUST be written as JSON (stable schema, versioned) and a markdown report.
- **FR-005**: Results MUST record machine (chip, memory, macOS, power and thermal state), system
  under test (version, flags, model files with hashes) and suite version.
- **FR-006**: `--compare` MUST drive OpenAI-compatible or Anthropic-compatible endpoints of other
  systems; each competitor's recommended cache and performance flags MUST be documented in the
  bundled profiles and recorded per run.
- **FR-007**: Systems MUST be measured one at a time with the others idle.
- **FR-008**: The quality suite MUST run objective coding tasks (supplied tests) and report pass
  rate per system alongside speed.
- **FR-009**: A regression mode MUST compare against a stored baseline and fail on configured
  thresholds (speed, conformance, quality).
- **FR-010**: Bench MUST NOT upload results anywhere; publishing is a manual step.

### Key Entities

- **Suite / Cell**: A named measurement with inputs and settings.
- **System Under Test**: Name, version, endpoint, flags, model files.
- **Run Result**: Machine state, system, cell, repetitions, statistics, failures.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Two consecutive runs of the synthetic suite on an idle target Mac agree within 5% on
  median speeds.
- **SC-002**: A full comparison of four systems on the agent suite completes unattended in under
  2 hours.
- **SC-003**: Every published report contains all cells for all systems, with failures marked
  (review checklist, 100%).
- **SC-004**: The regression mode catches a deliberately introduced 10% speed regression and a
  greedy-output change in 100% of trials.

## Assumptions

- Competitors are installed by the user; mrun does not install them.
- Agent traces are recorded with mrun's own server in a debug mode and reviewed for secrets before
  bundling.
- The harness talks to systems over HTTP only.
