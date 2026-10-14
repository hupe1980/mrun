# Research: Benchmark Harness

## R1: What to measure

- **Decision**: Primary: per-turn time to first token (TTFT), time to first useful token (first
  non-reasoning text or completed tool call), and total session wall time on agent traces.
  Secondary: prompt tok/s and decode tok/s at 1K/8K/16K/30K, cancellation latency (disconnect →
  next request's first byte), peak RSS/footprint and swap delta.
- **Rationale**: Constitution I; the literature study shows agent workloads are prefill-dominated
  with long shared prefixes (see [literature-study](../../concepts/research/literature-study.md)).
- **Alternatives**: Synthetic tok/s only (misses the product's value); server-reported timings only
  (not comparable across systems; we measure client-side and record server timings when present).

## R2: Fairness rules

- **Decision**: Same machine, idle, AC power, Low Power Mode off; one system at a time; 1 warm-up
  pass then ≥5 repetitions; randomized cell order per repetition; record each system's version,
  flags and model file hashes; competitor caches enabled per their docs:
  llama.cpp `--cache-reuse 256 --cache-ram <MiB> --jinja`, Ollama defaults (MLX engine where it
  applies), oMLX defaults (in-memory + SSD cache), mlx-lm server prompt cache defaults.
  Profiles in `bench/profiles/` are the single place these flags live; they are re-verified at
  each published run.
- **Rationale**: Landscape study and the old protocol's sound parts (retain losing cells, mark
  quality-matched comparisons when weights differ).
- **Alternatives**: Concurrent runs (contaminated timings), same-weights-only (excludes GGUF-only
  systems; handled by labelling quality-matched cells).

## R3: Statistics

- **Decision**: Median and p95 (nearest-rank) per cell; spread as IQR; 95% bootstrap CI for
  medians (1,000 resamples); report the CI so small differences are not over-claimed.
- **Alternatives**: Means (skewed by outliers).

## R4: Traces

- **Decision**: JSONL, one record per turn: API style, full request body as the client sent it,
  recorded tool results already included in subsequent requests, think-time gap. Recorded with
  `mrun serve --record <dir>` (debug-only flag, off by default, writes request bodies) while
  maintainers run Claude Code, Codex and OpenCode on public repositories; redacted and reviewed
  before commit.
- **Rationale**: Replaying exact client bodies reproduces real prefix-reuse behavior including
  client quirks (volatile headers).

## R5: Quality suite

- **Decision**: ~20 small tasks on public code (Rust, Python, TypeScript): fix-a-bug and
  implement-a-function with supplied tests; the driver runs a minimal tool loop itself
  (read/write/run-tests tools executed by the harness, never by the server) so every system sees
  identical tool semantics. Score = tests pass. Three seeds per task.
- **Alternatives**: SWE-bench (too heavy for laptops); HumanEval (no tool loop).

## R6: Machine state

- **Decision**: `sysctl hw.model machdep.cpu.brand_string hw.memsize`, `sw_vers`, `pmset -g` (power
  source, low power mode), `vm_stat`/`sysctl vm.swapusage` before/after, thermal via
  `pmset -g therm`; sample process RSS with `ps` for the system under test when its PID is known.
