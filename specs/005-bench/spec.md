# Feature Specification: Benchmark Harness

**Feature Branch**: `005-bench`

**Created**: 2026-10-14

**Status**: Draft (revised 2026-10-15: lean 0.1 cut, D025; Splash added)

**Input**: User description: "`mrun bench` measures what agents feel and compares fairly: agent-trace replay (recorded multi-turn sessions with tool results), synthetic long-context and decode cells, cancellation latency and memory; runs mrun and competitors (llama.cpp server, Ollama, oMLX, mlx-lm server) on the same machine with their caches enabled, records full settings and machine state, emits JSON plus a markdown report; quality checks so speed never hides wrong output."

## Clarifications

### Session 2026-10-14

- Q: Where do agent traces come from? → A: Recorded by maintainers running OpenCode and Codex on public open-source repositories with `mrun serve --record` (002 FR-021), then automatically redacted, secret-scanned and reviewed. Claude Code traces stay local (its system prompt is proprietary). Revised after analysis on 2026-10-14.
- Q: How does replay treat assistant turns? → A: Plain in-order replay: the recorded requests are sent in order exactly as recorded; they already contain the client's history. Each system caches its own replies, so reuse is measured up to the start of the previous reply, equally for all systems. Revised after audit on 2026-10-15 (replaces teacher-forced priming).
- Q: What is the primary comparison metric? → A: Median time to first token per turn and total session time on the agent suite; tok/s cells are secondary.
- Q: Which competitor versions? → A: The latest stable release of each at measurement time, recorded in the report; profiles document each one's recommended cache flags.

### Session 2026-10-15 (audit)

- Q: How are cross-system caches made comparable? → A: Before each repetition every system's caches are reset by its profile's `reset` step; synthetic prefill prompts carry a per-repetition nonce; all systems run with the same context length.
- Q: Are systems loaded together? → A: Never. One system at a time, switched by hand: the harness pauses with "Stop X, start Y, press Enter" (automated start/stop is Later).
- Q: Is there a coding-task quality suite? → A: Not in this feature (see Later). Quality signals are malformed tool calls and early end-of-sequence per system, plus 001's `mrun run --check`.

### Session 2026-10-15 (lean 0.1 cut, D025; delta scan)

- Q: Which competitors? → A: llama.cpp, Ollama, oMLX, mlx-lm and Inco Splash (github.com/incoai/splash, Apache-2.0; Qwen3.6 and Qwen3.8 only, own Metal kernels, DFlash2 drafts, INT8 KV by default, unloads after 10 idle minutes). Splash runs with `--kv-format bf16` and idle release off for fairness (the exact-KV cell, compared against mrun), and once with its default INT8 KV as a separate, labeled cell; both settings are recorded. SGLang's MLX backend is Later.
- Q: What is deferred? → A: Automated start/stop and memory-recovery checks (manual pause instead), the thermal slope (thermal state is recorded only), regression mode and exit 7, RSS sampling, SC-002 (4-hour unattended run).
- Q: Where do reply lengths come from, now that `--record` stores no response data? → A: From the next request of the same conversation, whose history contains the reply; the last request of a conversation uses the median reply length of that conversation.
- Q: Who diagnoses template re-renders? → A: The bench: mrun reports `changed` with a message index; when that message's API content equals the previous request's, the turn is counted as `template` (FR-017).

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Measure my agent workload (Priority: P1)

A maintainer or user runs `mrun bench` and gets the numbers that matter for agent work: time to
first token per turn on a replayed agent session, prompt-processing and generation speed at short
and long context, cancellation latency, and machine state during the run.

**Why this priority**: Every performance decision and claim depends on trustworthy measurements.

**Independent Test**: Run `mrun bench --suite agent` against the running server and confirm a JSON
result and markdown report with all cells, settings and machine state.

**Acceptance Scenarios**:

1. **Given** a running mrun server, **When** `mrun bench --suite agent` runs, **Then** it replays
   the bundled agent traces and reports per-turn time to first token, processed/cached tokens and
   total session time.
2. **Given** `--suite synthetic`, **When** it runs, **Then** it reports prompt and generation speed
   at 1K, 8K, 32K and the common maximum context, with min/median/max over repetitions.
3. **Given** any run, **When** it finishes, **Then** the result records hardware, OS, power and
   thermal state, model hashes, mrun version and every setting, so it can be reproduced.

---

### User Story 2 - Fair comparison with competitors (Priority: P1)

The maintainer runs the same suites against llama.cpp server, Ollama, oMLX, the reference MLX
server and Splash on the same Mac, each with its prefix caching and recommended optimizations
enabled, and publishes a dated comparison including cells where mrun loses.

**Why this priority**: "Best-in-class" is only meaningful as a fair, reproducible comparison.

**Independent Test**: Run `mrun bench --compare` with competitor profiles configured and check
that each competitor's version, flags, context length, model files and cache resets are recorded
alongside results.

**Acceptance Scenarios**:

1. **Given** competitor profiles, **When** the comparison runs, **Then** the harness pauses for the
   user to switch to each system in turn (never two loaded at once), resets its caches before each
   repetition and uses the same context length as mrun.
2. **Given** different quantizations or KV formats between systems, **When** reported, **Then** the
   cell is labeled "different quantization" (with each system's bits per weight) or "lossy KV",
   not same-weights.
3. **Given** a system that fails a cell or is unreachable, **When** reported, **Then** the failure
   or "not run" is shown, not dropped.

---

### User Story 3 - Speed never hides wrong output (Priority: P2)

Each run reports per-system output-quality signals, so a configuration that is faster but wrong is
visible next to its speed.

**Why this priority**: Fast wrong answers are worse than slow right ones for coding.

**Independent Test**: Replay a trace against a fake server emitting a malformed tool call and an
early end-of-sequence; verify both counts appear next to that system's speed.

**Acceptance Scenarios**:

1. **Given** any agent run, **When** reported, **Then** each system's malformed-tool-call and
   early-EOS counts appear next to its speed. (Regression mode against a baseline: Later.)

### Edge Cases

- Machine under load or on battery/low-power: recorded, and the report warns.
- Thermal state other than nominal at a cell's start or end: recorded and warned (no slope fit).
- A system without the trace's API (for example Responses traces on oMLX): cell marked "n/a".
- Competitor not installed or unreachable: its cells are "not run" with the reason (FR-016).
- Reported prompt tokens differ from the expected count by more than 2%: the cell fails as
  "truncated input" (FR-015).
- Free memory before a system is more than 2 GiB below the value before the first system: warning
  (the previous system may still be loaded).

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
  session headers when recorded) or, failing that, longest shared prefix. Recorded tool-gap headers
  (`x-claude-code-prev-tool-durations`) are kept in the trace but not replayed.
- **FR-003**: Agent replay MUST send the recorded requests in recorded order exactly as recorded
  (they contain the client's history), in the trace's API style (Chat Completions, Messages or
  Responses); systems lacking that API are "n/a" for the trace. Each timed turn uses
  `max_tokens` = the reply length (taken from the next request's history; the conversation's
  median for its last request) and the system's ignore-EOS option where supported; session time =
  sum over turns of TTFT + fixed-length decode time. Because each system caches its own replies,
  reuse is measured up to the start of the previous reply, equally for all systems; the report
  states this method.
- **FR-003a**: Each system's outputs MUST be checked for malformed tool calls (arguments that are
  not valid JSON, an undeclared tool name, or arguments failing the trace's tool schema) and early
  end-of-sequence; both counts are reported per system next to its speed.
- **FR-004**: Repetitions: synthetic cells use one warm-up and at least 5 repetitions, cancel uses
  50 interruption points, agent traces use 3 repetitions. With n < 20 samples reports give
  min/median/max; with n ≥ 20 they add p95 and IQR.
- **FR-005**: Results MUST be written as JSON (versioned, hand-written schema file) and a markdown
  report, including per-turn records for agent cells (time to first token, time to first useful
  token, input, cached and output tokens, mrun's miss cause and message index where reported).
- **FR-006**: Results MUST record machine state (chip, memory, macOS, power source, low-power mode,
  thermal state at the start and end of each cell, free memory and swap before each system) and,
  per system, version, flags, context length, cache resets performed, model name, file format,
  quantization (bits per weight, group size, scheme), KV format and file hashes when local.
  *Removed (0.1 scope cut, 2026-10-15; see Later): RSS/footprint sampling during agent runs.*
- **FR-007**: `--compare` MUST drive other systems over HTTP with a per-system model name; decode
  cells use temperature 0, a fixed output length and the system's ignore-EOS option where
  available (mrun: the `x-mrun-ignore-eos: on` request header, 002 FR-031), labeling cells that
  ended early. Bundled profiles cover llama.cpp, Ollama, oMLX, mlx-lm and Splash; each
  competitor's recommended cache and performance flags live in the profile and are recorded per
  run. Splash runs `splash serve --model mlx-community/Qwen3.6-35B-A3B-4bit` with `--kv-format
  bf16` and idle release off (cell `splash-bf16`), and with its default INT8 KV (cell `splash`,
  labeled "lossy KV") [flags to verify on the pinned release]. Cells whose systems use different
  quantizations are labeled "different quantization" with bits per weight. *Removed (0.1 scope
  cut, 2026-10-15): the SGLang-MLX profile.*
- **FR-008**: Only one system may be loaded at a time. The harness pauses before each system with
  "Stop X, start Y, press Enter" and then checks the system's health endpoint. *Removed (0.1 scope
  cut, 2026-10-15; see Later): profile `start`/`stop` commands and the automated free-memory
  recovery check.*
- **FR-009**: *Removed (quality suite moved to Later).*
- **FR-010**: *Removed (0.1 scope cut, 2026-10-15; see Later)*: regression mode against a stored
  baseline (exit 7).
- **FR-011**: Bench MUST NOT contact any host other than the systems under test; publishing results
  is a manual step.
- **FR-012**: The OS thermal state (`NSProcessInfo.thermalState`) MUST be recorded at the start and
  end of each cell, and the report MUST warn when it is not nominal. *Removed (0.1 scope cut,
  2026-10-15): the fitted throttling slope.*
- **FR-013**: Time to first token MUST be measured client-side from request send to the first
  content event: the first `content_block_delta` (Messages); the first chunk with non-empty
  `content`, `reasoning_content` or `tool_calls` (Chat Completions); the first
  `response.output_text.delta` or `response.function_call_arguments.delta` (Responses). Pings, SSE
  comments, keepalives and role-only chunks are ignored.
- **FR-014**: Before each repetition the harness MUST reset each system's caches through its
  profile's `reset` step, either an HTTP call (mrun: `POST /mrun/cache/clear`; llama.cpp:
  `POST /slots/0?action=erase`; Ollama: unload with `keep_alive: 0` and reload) or a manual
  restart pause (oMLX: restart and clear its SSD cache directory; mlx-lm: restart; Splash: restart
  with its persistent and SSD caches off or cleared), and record each reset. Synthetic prefill
  cells prepend a per-repetition random nonce.
- **FR-015**: Profiles MUST give every system the same context length N = mrun's capacity (Ollama
  `OLLAMA_CONTEXT_LENGTH`/`num_ctx`, llama.cpp `-c N -np 1`). A cell fails as "truncated input"
  when a system's reported prompt tokens differ from the expected count by more than 2%.
- **FR-016**: With `--compare`, an unreachable competitor yields "not run" cells and the run
  continues; exit 5 only when every system (or the only system) is unreachable.
- **FR-017**: For mrun cells, a turn whose miss cause is `changed` at a message whose API content
  (role, visible text, tool calls as canonical JSON; reasoning excluded) equals the previous
  request's message at the same index MUST be counted as a `template` re-render; the report gives
  per-trace counts and the tokens they cost (the bench-side replacement for 003's old `template`
  cause).

### Key Entities

- **Suite / Cell**: A named measurement with inputs and settings.
- **System Under Test**: Name, version, endpoint, profile (reset, health, flags, context length,
  KV format), model files.
- **Trace**: Ordered recorded requests grouped into conversations.
- **Run Result**: Machine state, system, cell, repetitions, statistics, failures.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Two consecutive runs of the synthetic suite on an idle target Mac agree within 5% on
  median speeds.
- **SC-002**: *Removed (0.1 scope cut, 2026-10-15)*: unattended 4-hour comparison (needs automated
  start/stop).
- **SC-003**: Snapshot tests of the report renderer show every cell for every system, with
  failures, "not run" and "n/a" marked, and a "Losing cells" section that is present even when it
  says "none".
- **SC-004**: *Removed (0.1 scope cut, 2026-10-15)*: regression-mode detection (with FR-010).
- **SC-005**: Median repeated- and branching-turn TTFT ≤ the best competitor's on the agent suite
  (competitors: llama.cpp, Ollama, oMLX, mlx-lm and Splash's `splash-bf16` cell, caches on, the
  default model or the closest each supports, quantization labeled); all cells published, losing
  cells included.

## Assumptions

- Competitors are installed by the user; mrun does not install them. Splash needs an M3 or newer
  Mac and macOS 26.4+ (the M4 Pro target qualifies).
- Agent traces are recorded with `mrun serve --record` (002 FR-021: request bodies, arrival times
  and allow-listed session headers); the timestamps are used only to order requests.
- The harness talks to systems over HTTP, plus manual pauses for switching and restarts.

## Later

- A coding-task quality suite (objective tests in a sandboxed copy, several seeds) as a separate
  feature.
- Replaying recorded tool/think-time gaps (`x-claude-code-prev-tool-durations` is already kept).
- Moved out of 0.1 on 2026-10-15 (D025), IDs kept: automated profile `start`/`stop` with the
  free-memory recovery check (FR-008), regression mode with exit 7 (FR-010, SC-004), the thermal
  slope (FR-012), RSS/footprint sampling (FR-006), the SGLang-MLX profile (FR-007), the 4-hour
  unattended run (SC-002).
