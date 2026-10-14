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

- **Decision**: Same machine, AC power, Low Power Mode off; only the system under test running
  (others stopped or with no model loaded; free memory and swap recorded before each system); one
  warm-up pass; repetitions per FR-004; per-system model names; decode cells at temperature 0 with
  fixed output length and ignore-EOS where supported; competitor caches enabled per their docs:
  llama.cpp `--cache-reuse 256 --cache-ram <MiB> --jinja`, Ollama defaults (MLX engine where it
  applies), oMLX defaults (in-memory + SSD cache), mlx-lm server prompt cache defaults. Profiles in
  `bench/profiles/` are the single place these flags live; re-verified at each published run.
  Cells with different weights (MLX vs GGUF) are labeled quality-matched.
- **Rationale**: Landscape study; analysis U4, U5, U7.

## R3: Statistics

- **Decision**: Median, nearest-rank p95 and IQR per cell. No bootstrap intervals: with ≥ 5
  repetitions the IQR shows spread, and claims require visible margins.
- **Rationale**: Principle VI (analysis X1).

## R4: Traces and replay

- **Decision**: JSONL, one record per turn: API style, full request body as the client sent it, the
  recorded assistant reply, think-time gap. Recorded with `mrun serve --record <dir>` (002 FR-021)
  while maintainers run OpenCode and Codex (Apache-2.0, NOTICE kept) on public repositories;
  automated redaction (home paths, usernames, emails, hostnames) and a secret scan run as a
  pre-commit check on `bench/traces/`; Claude Code traces are never committed (proprietary system
  prompt) but can be recorded and replayed locally. Replay is teacher-forced: before timing turn n,
  send turn n−1 with its recorded reply appended (max 1 output token, untimed), then time turn n.
- **Rationale**: Without teacher forcing, a server's own generated reply differs from the recorded
  one and cache reuse is understated for every cache-aware system (analysis U1); licensing and
  privacy (S6, S8).

## R5: Quality suite

- **Decision**: ~20 small tasks on public code (Rust, Python, TypeScript) with supplied tests; the
  harness runs a minimal tool loop itself (read/write/run-tests executed by the harness, never by
  the server) in a temporary copy, under `sandbox-exec` with a no-network profile, per-command
  timeout (60 s) and per-task wall limit (10 min). Score = tests pass; 3 seeds per task.
- **Rationale**: Model-written code must not touch the host (analysis S5).

## R6: Machine state

- **Decision**: `sysctl hw.model machdep.cpu.brand_string hw.memsize`, `sw_vers`, `pmset -g` (power
  source, low power mode), `vm_stat`/`sysctl vm.swapusage` before/after, thermal via
  `pmset -g therm`; sample process RSS with `ps` for the system under test when its PID is known.
