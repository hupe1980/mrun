---
description: "Tasks for 001 Engine Core"
---

# Tasks: Engine Core

**Input**: [plan.md](plan.md), [spec.md](spec.md), [research.md](research.md), [data-model.md](data-model.md), [contracts/engine.md](contracts/engine.md)

**Tests**: Required by the constitution (tests ship with each change). Linux CI runs CPU tests with
tiny random-weight models; macOS-only tests are `#[ignore]` + `MRUN_TEST_MODELS`.

## Format: `[ID] [P?] [Story] Description`

## Phase 1: Setup

- [ ] T001 Replace the old scaffold with the workspace: root `Cargo.toml` (members `crates/mlx-sys`, `crates/mrun-engine`, `crates/mrun`), move package metadata to `crates/mrun/Cargo.toml`, delete `src/`, `tests/` and the old fixtures
- [ ] T002 [P] CI in `.github/workflows/ci.yml`: fmt, clippy `-D warnings`, `cargo test --workspace` on ubuntu (installs OpenBLAS/LAPACKE, CMake); a `macos-14` job builds with Metal and runs non-ignored tests
- [ ] T003 [P] `crates/mrun/src/main.rs` with clap skeleton (`run`, `serve`, `pull`, `ls`, `rm`, `show`, `connect`, `doctor`, `ps`, `cache`, `config`, `bench`, `completions`), unimplemented commands exit 1 with "not yet implemented"

## Phase 2: Foundational (blocks all stories)

- [ ] T004 `crates/mlx-sys`: vendor mlx-c v0.7.0 (`a341b49`) as a git submodule and MLX v0.32.3 as a sha256-pinned tarball; `build.rs` builds via CMake with `FETCHCONTENT_SOURCE_DIR_MLX`, Metal + Foundation + Accelerate on macOS, CPU + OpenBLAS/LAPACKE on Linux; installs the metallib next to the binary
- [ ] T005 `crates/mlx-sys/src/bindings.rs`: generate once with bindgen (script `crates/mlx-sys/regen.sh`) and check in; smoke test creating an array and evaluating `add`
- [ ] T006 `crates/mrun-engine/src/mlx/`: safe `Array` (RAII free, `!Sync`), error handler that converts mlx-c errors to `Result`, ops needed by Qwen3/MoE (matmul, quantized_matmul with mode, gather_qmm with mode, argpartition, take_along_axis, softmax precise, slice, slice_update, concatenate, astype, where), `fast::{sdpa, rope, rms_norm}`, eval/async_eval, stream, safetensors load, memory (active/peak/cache, limits, clear_cache, wired limit), `device_info`; unit tests per op on CPU
- [ ] T007 **M0 gate on the M4 Pro**: `crates/mrun-engine/examples/spike.rs` runs a 4-bit quantized matmul, gather_qmm, sdpa and memory APIs on Metal and prints timings; record result and go/no-go in `specs/001-engine-core/research.md` (R1)
- [ ] T008 [P] `crates/mrun-engine/src/load.rs`: parse `config.json` (qwen3, qwen3_moe), quantization with per-path overrides and mode inference, load safetensors into named arrays; reject unknown architectures/quantizations with named errors; tests with generated tiny checkpoints
- [ ] T009 [P] `crates/mrun/src/chat/{tokenizer.rs, template.rs, utf8.rs, stop.rs}`: tokenizers 0.23 from `tokenizer.json`, minijinja 3 + pycompat chat template rendering (with `enable_thinking`), streaming UTF-8 hold-back, stop-sequence matcher with hold-back; tests incl. split multibyte and split stop strings
- [ ] T010 `crates/mrun/src/runtime/engine.rs`: `Engine` trait, `GenerateRequest`, `Event`, `EngineError`, `CancelToken`, `EngineHandle` (dedicated thread, crossbeam command channel, bounded event channel cap 64, stall timeout 10 s → cancel); `runtime/scripted.rs` `ScriptedEngine`; tests for exactly-one-`Finished`, cancel, stall

## Phase 3: US1 — Chat in the terminal (P1) 🎯 MVP

**Independent test**: `mrun run qwen3-8b` streams answers and continues the conversation.

- [ ] T011 [P] [US1] `crates/mrun-engine/src/kv.rs`: per-layer K/V `[1, kv_heads, cap, head_dim]`, 256-step growth up to reservation, `append`, `trim`, `offset`; tests: growth, trim, sole-ownership (no copies on append)
- [ ] T012 [P] [US1] `crates/mrun-engine/src/models/qwen3.rs`: port mlx-lm `qwen3.py` (`ba8a08f`): embeddings, per-head q/k RMSNorm, RoPE with cache offset, GQA via sdpa (causal), SwiGLU MLP, tied/quantized lm_head; test on a tiny random config (shapes, causal consistency: prefill vs step-by-step logits within 1e-3)
- [ ] T013 [US1] `crates/mrun-engine/src/sample.rs`: on-device temperature/top-k/top-p/min-p/categorical with per-request seed key, argmax at T=0, optional mask; tests for determinism with seed and each filter's effect
- [ ] T014 [US1] `crates/mrun-engine/src/lib.rs` `MlxEngine::generate`: chunked prefill (default 512, eval per chunk, cancel check), decode loop with async_eval pipelining and per-token cancel check, events and stats; session continuation (only new tokens processed)
- [ ] T015 [US1] `crates/mrun/src/cli/run.rs`: REPL (`/clear`, `/info`, `/exit`), `--prompt` one-shot, sampling flags, `--think`, stats line, model resolution (path or HF cache repo id until 004)
- [ ] T016 [US1] Errors for missing/unsupported models with next steps (exit 3) in `crates/mrun/src/errors.rs`

## Phase 4: US2 — Interrupt instantly (P1)

- [ ] T017 [US2] Ctrl-C handling in `cli/run.rs`: first cancels, second within 1 s exits; prompt-processing cancel drops partial state back to the last committed turn
- [ ] T018 [US2] Tests in `crates/mrun/tests/cancel.rs` with `ScriptedEngine` (cancel during prefill/decode/at completion race: exactly one outcome) and `crates/mrun-engine/tests/cancel_cpu.rs` on a tiny model (state after cancel equals state before request)
- [ ] T019 [US2] Measure cancellation latency on the M4 Pro at 16K with chunk 512 and 256 for both models; set the default chunk and record in research R5

## Phase 5: US3 — Never overcommit memory (P1)

- [ ] T020 [P] [US3] `crates/mrun-engine/src/budget.rs`: physical memory (`sysctl hw.memsize`, `/proc/meminfo` on Linux), `device_info` recommended working set, budget formula, overrides; unit tests for the formula and override rules
- [ ] T021 [US3] Scratch probe at load: one worst-case chunk at max position on a scratch KV, `peak − active`; compute `kv_capacity_tokens`; set MLX memory/cache/wired limits; refuse load with all numbers (exit 4)
- [ ] T022 [US3] Per-request reservation (`prompt + max_tokens`) with up-front `Capacity` error; boundary tests (exactly at capacity passes, +1 fails) on `ScriptedEngine` and tiny model
- [ ] T023 [US3] `mrun run --info` budget breakdown output (snapshot test)

## Phase 6: US4 — Trustworthy output (P2)

- [ ] T024 [P] [US4] `crates/mrun-engine/src/models/qwen3_moe.rs`: router (8-bit gate, softmax precise, argpartition top-k, norm_topk_prob), SwitchGLU via gather_qmm with sort-by-expert at ≥ 64 indices; tiny random MoE config test
- [ ] T025 [P] [US4] `scripts/gen_reference.py` (pinned mlx-lm, `--prefill-step-size` = mrun chunk): writes prompt token ids + 256 greedy token ids per prompt to `crates/mrun-engine/tests/reference/<model>.json` for ≥ 20 prompts up to 8K tokens; template token-id parity included
- [ ] T026 [US4] `crates/mrun-engine/tests/conformance.rs` (`#[ignore]`, macOS): greedy match vs references for Qwen3-8B-4bit and Qwen3-Coder-30B-A3B-4bit; `mrun run --check` uses the same harness and reports first mismatch
- [ ] T027 [US4] Seeded sampling reproducibility and stop-sequence finish-reason tests in `crates/mrun/tests/sampling.rs`

## Phase 7: Polish

- [ ] T028 [P] Speed check vs mlx-lm (SC-002/003) at 1K and 16K for both models; record numbers in `specs/001-engine-core/research.md`; open follow-ups for gaps > 10%
- [ ] T029 [P] 30-minute soak alternating 1K/30K prompts with memory sampling (SC-006); record
- [ ] T030 Update `README.md` (build prerequisites, `mrun run`, `--info`) and run [quickstart](quickstart.md)

## Dependencies

T001 → T002/T003/T004 → T005 → T006 → T007 (gate) → T008, T011, T012 → T013 → T014 → T015/T016 (MVP).
T010 can start after T001 (independent of MLX). US2 (T017–T019) and US3 (T020–T023) follow T014.
US4 T024/T025 can start after T008/T012. Polish last.

## Parallel opportunities

T002/T003/T004 · T008/T009/T010 · T011/T012 · T020 with T017 · T024/T025 · T028/T029.
