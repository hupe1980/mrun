# Research: Benchmark Harness

## R1: What to measure

- **Decision**: Primary: per-turn time to first token (TTFT, FR-013), time to first useful token
  (first non-reasoning text or completed tool call), and session time (sum of TTFT + fixed-length
  decode per turn) on agent traces. Secondary: prompt tok/s and decode tok/s at 1K/8K/32K and the
  common maximum context; cancellation latency (mrun only: disconnect → next request's first
  byte); peak RSS/footprint and swap delta sampled during agent runs.
- **TTFT events**: Messages: first `content_block_delta`. Chat: first chunk with non-empty
  `content`, `reasoning_content` or `tool_calls`. Responses: first `response.output_text.delta` or
  `response.function_call_arguments.delta`. Pings, SSE comments, keepalives and role-only chunks
  are ignored, since servers differ in when they send them.
- **Rationale**: Constitution I; the literature study shows agent workloads are prefill-dominated
  with long shared prefixes (see [literature-study](../../concepts/research/literature-study.md)).
- **Alternatives**: Synthetic tok/s only (misses the product's value); server-reported timings only
  (not comparable across systems; we measure client-side and record server timings when present);
  a separate memory suite (redundant with sampling during agent runs).

## R2: Fairness rules

- **Decision**: Same machine, AC power, Low Power Mode off. One system loaded at a time: profile
  `start` → `health` → run → `stop` → free-memory check; without commands the harness pauses with
  "Stop X, start Y, press Enter". Before each repetition the profile `reset` clears caches (mrun
  `mrun cache clear`; llama.cpp `POST /slots/0?action=erase`; Ollama `keep_alive: 0` then reload;
  oMLX restart + clear its SSD cache directory; mlx-lm restart); resets are recorded. Synthetic
  prefill prompts start with a per-repetition random nonce. Equal context N = mrun's capacity
  (Ollama `OLLAMA_CONTEXT_LENGTH`/`num_ctx`, llama.cpp `-c N -np 1`); a > 2% gap between reported
  and expected prompt tokens fails the cell as "truncated input". Decode at temperature 0 with
  fixed output length and ignore-EOS where supported. Competitor caches enabled per their docs:
  llama.cpp `--cache-reuse 256 --cache-ram <MiB> --jinja`, Ollama defaults (MLX engine where it
  applies), oMLX defaults (in-memory + SSD cache), mlx-lm server prompt cache defaults. Profiles in
  `bench/profiles/` are the single place these flags live; re-verified at each published run.
  Cells with different quantizations (MLX vs GGUF) are labeled "different quantization" with bits
  per weight.
- **Rationale**: Without resets, run order decides who hits a warm cache; without equal context,
  Ollama's small default silently truncates; two loaded models distort memory and bandwidth.
  Landscape study; analysis U4, U5, U7; audit 2026-10-15.

## R3: Statistics

- **Decision**: n < 20: min/median/max. n ≥ 20 (cancel points, pooled per-turn TTFT): also
  nearest-rank p95 and IQR. Throttling: least-squares slope across repetitions, flagged when the
  fitted change over the run exceeds 5% of the median. Regression compares medians against the
  baseline (default threshold 10%). No bootstrap intervals.
- **Rationale**: Principle VI; p95/IQR from 3–5 samples are noise (analysis X1, audit).

## R4: Traces and replay

- **Decision**: Recorded with `mrun serve --record <dir>` (002 FR-021: request bodies with arrival
  and response-end times) while maintainers run OpenCode and Codex (Apache-2.0, NOTICE kept) on
  public repositories. `cargo xtask bench-trace` orders requests by arrival time (the only use of
  the timestamps), groups them into conversations (`prompt_cache_key`, session headers when
  recorded, else longest shared prefix), counts expected prompt and reply tokens with the model's
  tokenizer, redacts (home paths, usernames, emails, hostnames) with one deterministic mapping per
  trace, and secret-scans; the same check runs pre-commit on `bench/traces/`. Claude Code traces
  are never committed (proprietary system prompt) but can be recorded and replayed locally.
  Replay sends each request exactly as recorded, in order; timed turns use
  `max_tokens` = recorded reply length with ignore-EOS where supported.
- **Rationale**: The recorded requests already carry the client's history, so no priming step is
  needed. Each system caches its own (different) reply, so every system can reuse its cache only up
  to the start of the previous reply: the understatement is identical across systems and the
  comparison stays fair. A per-trace mapping keeps redacted prefixes byte-identical across turns;
  licensing and privacy (S6, S8).
- **Alternatives**: Teacher-forced priming with recorded replies (extra untimed requests, a reply
  extraction step, and an extra trace field; dropped in the audit).

## R5: Quality signals

- **Decision**: No coding-task suite in this feature. Per system: malformed tool calls (arguments
  not valid JSON, undeclared tool name, or failing the trace's tool schema) and early EOS counts;
  greedy conformance against stored reference outputs is 001's `mrun run --check`.
- **Rationale**: A sandboxed task harness (sandbox-exec, seeds, tests) is a feature of its own;
  the cheap signals catch broken tool calling and truncation. See spec "Later".

## R6: Machine state

- **Decision**: macOS only. `sysctl hw.model machdep.cpu.brand_string hw.memsize`, `sw_vers`,
  `pmset -g` (power source, low power mode), `vm_stat`/`sysctl vm.swapusage` before each system and
  sampled during agent runs, thermal state from `NSProcessInfo.thermalState` (powermetrics only
  when root); process RSS/footprint via `ps` for the PID from the profile. No Linux fallbacks.
