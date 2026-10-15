# Feature Specification: Benchmark Harness

**Feature Branch**: `005-bench`

**Created**: 2026-10-14

**Status**: Draft

**Input**: User description: "`mrun bench` measures what agents feel and compares fairly: agent-trace replay (recorded multi-turn sessions with tool results), synthetic long-context and decode cells, cancellation latency and memory; runs mrun and competitors (llama.cpp server, Ollama, oMLX, mlx-lm server) on the same machine with their caches enabled, records full settings and machine state, emits JSON plus a markdown report; quality checks so speed never hides wrong output."

## Clarifications

### Session 2026-10-14

- Q: Where do agent traces come from? → A: Recorded by maintainers running OpenCode and Codex on public open-source repositories with `mrun serve --record` (002 FR-021), then automatically redacted, secret-scanned and reviewed. Claude Code traces stay local (its system prompt is proprietary). Revised after analysis on 2026-10-14.
- Q: How does replay treat assistant turns? → A: Plain in-order replay: the recorded requests are sent in order exactly as recorded; they already contain the client's history. Each system caches its own replies, so reuse is measured up to the start of the previous reply, equally for all systems. Revised after audit on 2026-10-15 (replaces teacher-forced priming).
- Q: What is the primary comparison metric? → A: Median time to first token per turn and total session time on the agent suite; tok/s cells are secondary.
- Q: Which competitor versions? → A: The latest stable release of each at measurement time, recorded in the report; profiles document each one's recommended cache flags.

### Session 2026-10-15 (audit)

- Q: How are cross-system caches made comparable? → A: Before each repetition every system's caches are reset by its profile's `reset` step; synthetic prefill prompts carry a per-repetition nonce; all systems run with the same context length.
- Q: Are systems loaded together? → A: Never. One system at a time: start, run, stop, check free memory; or a manual "Stop X, start Y, press Enter" pause when a profile has no commands.
- Q: Is there a coding-task quality suite? → A: Not in this feature (see Later). Quality signals are malformed tool calls and early end-of-sequence per system, plus 001's `mrun run --check`.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Measure my agent workload (Priority: P1)

A maintainer or user runs `mrun bench` and gets the numbers that matter for agent work: time to
first token per turn on a replayed agent session, prompt-processing and generation speed at short
and long context, cancellation latency, and memory during the run.

**Why this priority**: Every performance decision and claim depends on trustworthy measurements.

**Independent Test**: Run `mrun bench --suite agent` against the running server and confirm a JSON
result and markdown report with all cells, settings and machine state.

**Acceptance Scenarios**:

1. **Given** a running mrun server, **When** `mrun bench --suite agent` runs, **Then** it replays
   the bundled agent traces and reports per-turn time to first token, processed/cached tokens,
   total session time and peak memory.
2. **Given** `--suite synthetic`, **When** it runs, **Then** it reports prompt and generation speed
   at 1K, 8K, 32K and the common maximum context, with min/median/max over repetitions.
3. **Given** any run, **When** it finishes, **Then** the result records hardware, OS, power state,
   model hashes, mrun version and every setting, so it can be reproduced.

---

### User Story 2 - Fair comparison with competitors (Priority: P1)

The maintainer runs the same suites against llama.cpp server, Ollama, oMLX and the reference
MLX server on the same Mac, each with its prefix caching and recommended optimizations enabled,
and publishes a dated comparison including cells where mrun loses.

**Why this priority**: "Best-in-class" is only meaningful as a fair, reproducible comparison.

**Independent Test**: Run `mrun bench --compare` with competitor profiles configured and check
that each competitor's version, flags, context length, model files and cache resets are recorded
alongside results.

**Acceptance Scenarios**:

1. **Given** competitor profiles, **When** the comparison runs, **Then** each system is started,
   measured and stopped in turn (never two loaded at once), with its caches reset before each
   repetition and the same context length as mrun.
2. **Given** different quantizations between systems, **When** reported, **Then** the cell is
   labeled "different quantization" with each system's bits per weight, not same-weights.
3. **Given** a system that fails a cell or is unreachable, **When** reported, **Then** the failure
   or "not run" is shown, not dropped.

---

### User Story 3 - Speed never hides wrong output (Priority: P2)

Each run reports per-system output-quality signals and a regression mode guards mrun changes, so a
configuration that is faster but wrong is caught.

**Why this priority**: Fast wrong answers are worse than slow right ones for coding.

**Independent Test**: Replay a trace against a fake server emitting a malformed tool call and an
early end-of-sequence; verify both counts appear next to that system's speed.

**Acceptance Scenarios**:

1. **Given** any agent run, **When** reported, **Then** each system's malformed-tool-call and
   early-EOS counts appear next to its speed.
2. **Given** an mrun change, **When** the regression check runs, **Then** it fails if speed drops
   beyond the configured threshold or the quality-signal counts rise (greedy-output changes are
   caught by feature 001's `mrun run --check`).

### Edge Cases

- Machine under load or on battery/low-power: recorded, and the report warns.
- Thermal throttling during long runs: flagged per FR-012.
- A system without the trace's API (for example Responses traces on oMLX): cell marked "n/a".
- Competitor not installed or unreachable: its cells are "not run" with the reason (FR-016).
- Reported prompt tokens differ from the expected count by more than 2%: the cell fails as
  "truncated input" (FR-015).
- Free memory has not recovered after stopping a system: warning before the next system starts.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: `mrun bench` MUST provide suites: `agent` (trace replay), `synthetic` (prefill and
  decode cells at 1K, 8K, 32K and the common maximum, i.e. the smallest context across the compared
  systems) and `cancel` (interruption latency; mrun only, never run for competitors).
- **FR-002**: Bundled agent traces MUST come from open-source clients (OpenCode, Codex with NOTICE
  attribution) on public repositories, after automated redaction (paths, usernames, emails, hosts)
  using one deterministic mapping per trace, so shared prefixes survive, and a secret scan. Traces
  from proprietary clients (Claude Code) MAY be recorded and replayed locally but MUST NOT be
  committed. Requests are grouped into conversations by session hint (`prompt_cache_key`, or
  session headers when recorded) or, failing that, longest shared prefix.
- **FR-003**: Agent replay MUST send the recorded requests in recorded order exactly as recorded
  (they contain the client's history), in the trace's API style (Chat Completions, Messages or
  Responses); systems lacking that API are "n/a" for the trace. Each timed turn uses
  `max_tokens` = the recorded reply length and the system's ignore-EOS option where supported;
  session time = sum over turns of TTFT + fixed-length decode time. Because each system caches its
  own replies, reuse is measured up to the start of the previous reply, equally for all systems;
  the report states this method.
- **FR-003a**: Each system's outputs MUST be checked for malformed tool calls (arguments that are
  not valid JSON, an undeclared tool name, or arguments failing the trace's tool schema) and early
  end-of-sequence; both counts are reported per system next to its speed.
- **FR-004**: Repetitions: synthetic cells use one warm-up and at least 5 repetitions, cancel uses
  50 interruption points, agent traces use 3 repetitions. With n < 20 samples reports give
  min/median/max; with n ≥ 20 they add p95 and IQR.
- **FR-005**: Results MUST be written as JSON (versioned, hand-written schema file) and a markdown
  report, including per-turn records for agent cells (time to first token, time to first useful
  token, input, cached and output tokens).
- **FR-006**: Results MUST record machine state (chip, memory, macOS, power source, low-power mode,
  thermal state, free memory and swap before each system) and, per system, version, flags, context
  length, cache resets performed, model name, file format, quantization (bits per weight, group
  size, scheme) and file hashes when local. During agent runs the harness MUST sample each system's
  RSS/footprint (PID from its profile) and system swap.
- **FR-007**: `--compare` MUST drive other systems over HTTP with a per-system model name; decode
  cells use temperature 0, a fixed output length and the system's ignore-EOS option where
  available (mrun: the loopback-only `x-mrun-ignore-eos: on` request header, 002), labeling cells
  that ended early; bundled profiles cover llama.cpp, Ollama, oMLX, mlx-lm and SGLang's MLX
  backend; each competitor's recommended cache and performance
  flags live in bundled profiles and are recorded per run. Cells whose systems use different
  quantizations are labeled "different quantization" with bits per weight.
- **FR-008**: Only one system may be loaded at a time. Profiles define `start`, `stop` and `health`
  commands; the harness starts a system, waits for health, runs it, stops it and checks that free
  memory has recovered before the next. Without commands it pauses with "Stop X, start Y, press
  Enter".
- **FR-009**: *Removed (quality suite moved to Later).*
- **FR-010**: A regression mode MUST compare against a stored baseline and fail (exit 7) on
  configured thresholds (median speed, malformed-tool-call and early-EOS counts); the
  greedy-conformance check is feature 001's `--check`.
- **FR-011**: Bench MUST NOT contact any host other than the systems under test; publishing results
  is a manual step.
- **FR-012**: A thermal-throttling warning MUST be emitted when a cell's fitted slope across
  repetitions exceeds 5% of its median over the run. Thermal state comes from the OS thermal state
  (powermetrics only when run as root).
- **FR-013**: Time to first token MUST be measured client-side from request send to the first
  content event: the first `content_block_delta` (Messages); the first chunk with non-empty
  `content`, `reasoning_content` or `tool_calls` (Chat Completions); the first
  `response.output_text.delta` or `response.function_call_arguments.delta` (Responses). Pings, SSE
  comments, keepalives and role-only chunks are ignored.
- **FR-014**: Before each repetition the harness MUST reset each system's caches through its
  profile's `reset` step (mrun: `mrun cache clear`; llama.cpp: `POST /slots/0?action=erase`; Ollama:
  unload with `keep_alive: 0` and reload; oMLX: restart and clear its SSD cache directory; mlx-lm:
  restart) and record each reset. Synthetic prefill cells prepend a per-repetition random nonce.
- **FR-015**: Profiles MUST give every system the same context length N = mrun's capacity (Ollama
  `OLLAMA_CONTEXT_LENGTH`/`num_ctx`, llama.cpp `-c N -np 1`). A cell fails as "truncated input"
  when a system's reported prompt tokens differ from the expected count by more than 2%.
- **FR-016**: With `--compare`, an unreachable competitor yields "not run" cells and the run
  continues; exit 5 only when every system (or the only system) is unreachable.

### Key Entities

- **Suite / Cell**: A named measurement with inputs and settings.
- **System Under Test**: Name, version, endpoint, profile (start/stop/health/reset, flags, context
  length), model files.
- **Trace**: Ordered recorded requests grouped into conversations.
- **Run Result**: Machine state, system, cell, repetitions, statistics, failures.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Two consecutive runs of the synthetic suite on an idle target Mac agree within 5% on
  median speeds.
- **SC-002**: A full comparison of mrun and three competitors on the bundled agent suite (3
  repetitions) completes unattended in under 4 hours on the target Mac when all profiles have
  start/stop commands.
- **SC-003**: Every published report contains all cells for all systems, with failures, "not run"
  and "n/a" marked and a "losing cells" section (review checklist, 100%).
- **SC-004**: With a 10% threshold, the regression mode detects an injected 15% slowdown in 100% of
  trials and raises 0 false alarms on 5 unchanged runs.
- **SC-005**: Against llama.cpp server, Ollama and oMLX with their prompt caches enabled, on the
  agent suite with Qwen3-Coder-30B-A3B (MLX 4-bit for mrun and oMLX, GGUF Q4_K_M for llama.cpp
  and Ollama, labeled "different quantization" with bits per weight), mrun's median time to first
  token is ≤ 1.05× the best competitor's in every agent cell; the report publishes losing cells
  (moved here from feature 003).

## Assumptions

- Competitors are installed by the user; mrun does not install them.
- Agent traces are recorded with `mrun serve --record` (002 FR-021: request bodies with arrival and
  response-end times); the timestamps are used only to order requests.
- The harness talks to systems over HTTP, plus the profile commands for start/stop/health/reset.

## Later

- A coding-task quality suite (objective tests in a sandboxed copy, several seeds) as a separate
  feature.
- Replaying recorded tool/think-time gaps.
