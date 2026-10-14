---
description: "Tasks for 001 Engine Core"
---

# Tasks: Engine Core

**Input**: [plan.md](plan.md), [spec.md](spec.md), [research.md](research.md), [data-model.md](data-model.md), [contracts/engine.md](contracts/engine.md)

**Tests**: Required by the constitution. Linux CI runs CPU tests with tiny random-weight models;
macOS-only tests are `#[ignore]` + `MRUN_TEST_MODELS`.

## Format: `[ID] [P?] [Story] Description`

## Phase 1: Setup

- [ ] T001 Replace the old scaffold with the workspace: root `Cargo.toml` (members `crates/mlx-sys`, `crates/mrun-engine`, `crates/mrun`; shared lints), move package metadata to `crates/mrun/Cargo.toml`, delete `src/`, `tests/` and the old fixtures
- [ ] T002 [P] CI `.github/workflows/ci.yml`: Linux job (OpenBLAS/LAPACKE, CMake): fmt, clippy `-D warnings`, `cargo test --workspace`, plus `cargo test -p mrun --no-default-features`; macOS job: build only (hosted runners have no reliable Metal GPU for MLX)
- [ ] T003 [P] `crates/mrun/src/main.rs` with clap: `run` only (other commands are added by their features)

## Phase 2: Foundational (blocks all stories)

- [ ] T004 `crates/mlx-sys`: vendor mlx-c v0.7.0 (`a341b49`) as a submodule and MLX v0.32.3 as a sha256-pinned tarball; `build.rs` builds via CMake with `FETCHCONTENT_SOURCE_DIR_MLX`, Metal + Foundation + Accelerate on macOS, CPU + OpenBLAS/LAPACKE on Linux; metallib next to the binary
- [ ] T005 `crates/mlx-sys/src/bindings.rs` generated once (`regen.sh`) and checked in; smoke test evaluating `add`
- [ ] T006 `crates/mrun-engine/src/mlx/` [feature mlx]: `Array` (RAII, `!Send`, `!Sync`), error handler → `Result`; ops: zeros, astype, reshape, transpose, matmul, quantized_matmul and gather_qmm with mode, take (embedding), argpartition, take_along_axis, sort, cumsum, argmax, logsumexp, softmax precise, where, slice, slice_update, concatenate, random key/categorical; `fast::{sdpa, rope, rms_norm}`; eval/async_eval; stream; safetensors load; memory (active/peak/cache, limits, clear_cache, wired limit); `device_info`; `deep_copy` (zeros + slice_update + eval) with a pointer-inequality debug assertion; CPU unit tests per op
- [ ] T007 **M0 gate on the M4 Pro**: `crates/mrun-engine/examples/spike.rs` runs 4-bit quantized matmul, gather_qmm, sdpa and memory APIs on Metal with timings; record go/no-go in research R1
- [ ] T008 [P] `crates/mrun-engine/src/engine.rs` + `identity.rs` (no MLX): `Engine` trait, ModelInfo, MemoryBudget, Segment, KvSegmentId, EngineError, CancelToken, Sampling, TokenMask; identity per research R10 with cache in `~/.mrun/identity.toml`; tests (HF blob-name shortcut, changed-file detection)
- [ ] T009 [P] `crates/mrun-engine/src/load.rs` [mlx]: `config.json` (qwen3, qwen3_moe), quantization with per-path overrides and mode inference, weights estimate from file sizes (pre-load check), safetensors load; named errors for unsupported architectures/quantizations; tests with generated tiny checkpoints
- [ ] T010 [P] `crates/mrun/src/chat/{tokenizer.rs, template.rs, utf8.rs, stop.rs}`: tokenizers 0.23 (`default-features = false`, no `http`), minijinja 3 + pycompat rendering (with `enable_thinking`), UTF-8 hold-back, stop matcher with hold-back; tests incl. split multibyte and split stop strings
- [ ] T011 `crates/mrun/src/runtime/{handle.rs, generate.rs, scripted.rs}`: `EngineHandle::spawn(factory)` (worker thread, crossbeam commands, bounded events cap 64, stall timeout 10 s → cancel); generic generation loop (reserve → prefill all but last → decode loop → stop/UTF-8 → exactly one `Finished` with `prefilled` → trim to committed tokens); `ScriptedEngine`; tests: one `Finished`, cancel at each phase, stall, trim after stop/cancel (FR-004)

## Phase 3: US1 — Chat in the terminal (P1) 🎯 MVP

- [ ] T012 [P] [US1] `crates/mrun-engine/src/kv.rs` [mlx]: per-layer K/V, cap = round_up(prompt + min(max_output, 4096), 256), 4,096-token growth one layer at a time, append, trim, snapshot (deep copy), restore (drop active first); tests: no copy on append (memory counters), growth peak ≤ cache + one layer, snapshot independence, restore + continue equals uninterrupted within 1e-3
- [ ] T013 [P] [US1] `crates/mrun-engine/src/models/qwen3.rs` [mlx]: port mlx-lm `qwen3.py` (`ba8a08f`) with layer-by-layer prefill (`async_eval`, at most two layers in flight, cancel check before each layer); tiny random config tests (prefill vs step-by-step logits within 1e-3; cancel stops within two layers)
- [ ] T014 [US1] `crates/mrun-engine/src/sample.rs` [mlx]: temperature/top-k/top-p/min-p/categorical with per-request seed key, argmax at T=0, optional mask; determinism and filter tests
- [ ] T015 [US1] `crates/mrun-engine/src/mlx_engine.rs`: `MlxEngine` implementing `Engine` (reserve, prefill in 2,048 chunks with per-layer cancel checks, pipelined decode unless masked, trim, snapshot/restore/drop)
- [ ] T016 [US1] `crates/mrun/src/cli/run.rs`: REPL (`/clear`, `/info`, `/exit`), `--prompt`, sampling flags, `--think`, stats line, model resolution (path or HF cache repo id), input caps (FR-016)
- [ ] T017 [US1] Errors with next steps in `crates/mrun/src/errors.rs`: missing/unsupported model (exit 3), empty prompt, max_tokens 0, input caps; model-changed-on-disk report at load; tests

## Phase 4: US2 — Interrupt instantly (P1)

- [ ] T018 [US2] Ctrl-C handling in `cli/run.rs` (first cancels, second within 1 s exits); state trimmed to committed tokens
- [ ] T019 [US2] Cancellation tests: `crates/mrun/tests/cancel.rs` (ScriptedEngine races incl. completion race) and `crates/mrun-engine/tests/cancel_cpu.rs` (tiny model: state after cancel equals state before request)
- [ ] T020 [US2] `crates/mrun-engine/examples/bench_engine.rs` (FR-017): prefill/decode tok/s at given contexts, cancel latency over 50 random interrupt points, JSON with machine + model identity; `scripts/bench_reference.sh` for pinned mlx-lm; run SC-004 on the M4 Pro for both models (per-layer eval, fallback every 4 layers) and record in research R5

## Phase 5: US3 — Never overcommit memory (P1)

- [ ] T021 [P] [US3] `crates/mrun-engine/src/budget.rs`: physical memory (`sysctl hw.memsize`, `/proc/meminfo`), recommended working set, budget formula and overrides; unit tests
- [ ] T022 [US3] Pre-load check from file sizes; scratch probe per research R6 (incl. per-layer growth transient) with cache in `~/.mrun/probe.toml`; `kv_capacity_tokens` capped by max position; MLX memory/cache/wired limits; refusal with all numbers (exit 4)
- [ ] T023 [US3] Admission: prompt + max_output vs capacity (minus others' pinned, plus evictable); boundary tests (= capacity passes, +1 rejected) on ScriptedEngine and tiny model
- [ ] T024 [US3] `mrun run --info` breakdown incl. current free memory (snapshot test)

## Phase 6: US4 — Trustworthy output (P2)

- [ ] T025 [P] [US4] `crates/mrun-engine/src/models/qwen3_moe.rs`: router (8-bit gate, softmax precise, argpartition top-k, norm_topk_prob), SwitchGLU via gather_qmm with sort-by-expert at ≥ 64 indices; tiny random MoE test
- [ ] T026 [US4] `scripts/gen_reference.py` (pinned mlx-lm, `--prefill-step-size 2048`, last prompt token processed alone): prompt ids + 256 greedy ids for ≥ 20 prompts up to 8K tokens, plus template token-id parity; run after T020 fixes the schedule
- [ ] T027 [US4] `crates/mrun-engine/tests/conformance.rs` (`#[ignore]`, macOS): greedy match for both models; `mrun run --check` reuses it and prints first mismatch with logit margin
- [ ] T028 [US4] Seeded sampling reproducibility and stop finish-reason tests in `crates/mrun/tests/sampling.rs`

## Phase 7: Polish

- [ ] T029 [P] SC-002/SC-003 with `bench_engine` vs `bench_reference.sh` at 1K/16K for both models; SC-007 cold start (probe cached) for 8B; record in research
- [ ] T030 [P] 30-minute soak (SC-006): `phys_footprint` sampling and `vm_stat` swap-outs; record
- [ ] T031 [P] Offline/no-write audit (FR-015): test that loading and generating make no network connections (no network features compiled; run with network namespace/`sandbox-exec` deny) and write nothing outside `~/.mrun`
- [ ] T032 Update `README.md` (build prerequisites, `mrun run`, `--info`) and run [quickstart](quickstart.md)

## Dependencies

T001 → T002/T003/T004/T008 → T005 → T006 → T007 (gate) → T009, T012, T013 → T014 → T015 → T016/T017 (MVP).
T008, T010, T011 need no MLX and can start right after T001. US2/US3 follow T015. T026 after T020.

## Parallel opportunities

T002/T003/T004/T008 · T009/T010/T011 · T012/T013 · T021 with T018 · T025 with T020 · T029/T030/T031.
