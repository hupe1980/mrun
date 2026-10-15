---
description: "Tasks for 001 Engine Core"
---

# Tasks: Engine Core

**Input**: [plan.md](plan.md), [spec.md](spec.md), [research.md](research.md), [data-model.md](data-model.md), [contracts/engine.md](contracts/engine.md)

**Tests**: Required by the constitution. Linux CI runs `ScriptedEngine` and tiny random-weight CPU
models; macOS-only tests are `#[ignore]` + `MRUN_TEST_MODELS`.

**Status (2026-10-15, after `d06852c`)**: `[X]` = done in code with tests. Partly done work is
split: the landed part is ticked, the remainder is a separate open task.

## Format: `[ID] [P?] [Story] Description`

## Phase 1: Setup

- [X] T001 Workspace: root `Cargo.toml` (members `crates/mrun-engine`, `crates/mrun`; shared lints), package metadata in `crates/mrun/Cargo.toml`, old scaffold removed
- [X] T002 [P] CI `.github/workflows/ci.yml` baseline: Linux fmt, clippy `-D warnings`, `cargo test --workspace`; macOS build only. (MLX dependencies and the no-MLX job move to T004a.)
- [X] T003 [P] `crates/mrun/src/main.rs` with clap: `run` arguments and input-cap checks; no engine wired yet

## Phase 2: Foundational (blocks all stories)

- [ ] T004 `crates/mlx-sys`: mlx-c v0.7.0 (`a341b49`) submodule, MLX v0.32.4 (`36969c0`) sha256-pinned tarball; `build.rs` via CMake with `FETCHCONTENT_SOURCE_DIR_MLX`; Metal + Foundation + Accelerate on macOS, CPU + OpenBLAS/LAPACKE on Linux; metallib next to the binary
- [ ] T004a CI: add CMake/OpenBLAS/LAPACKE to the Linux job, a `cargo test -p mrun --no-default-features` step, and the `mlx` feature build on macOS
- [ ] T005 `crates/mlx-sys/src/bindings.rs` generated once (`regen.sh`) and checked in; smoke test evaluating `add`
- [ ] T006 `crates/mrun-engine/src/mlx/` [mlx]: `Array` (RAII, `!Send`, `!Sync`), error handler → `Result`; ops: zeros, astype, reshape, transpose, matmul, quantized_matmul/gather_qmm with mode, take, argpartition, take_along_axis, put_along_axis, sort, cumsum, argmax, logsumexp, logaddexp, softmax precise, sigmoid, where, split, slice, slice_update, concatenate, contiguous, conv1d (groups), random key/categorical; `fast::{sdpa, rope, rms_norm, metal_kernel}`; `compile` (shapeless closures); eval/async_eval; stream; safetensors; memory (active/peak/cache, limits, clear_cache, wired); `device_info` (`memory_size`, `max_recommended_working_set_size`); `deep_copy` with pointer assertion; CPU unit tests per op
- [ ] T006a [mlx] Compiled-op parity (R14): on Metal compare `swiglu`, `precise_swiglu`, `compute_g` compiled vs unfused on the three models' shapes; bind `compile` for them unless bitwise equal; record in research R14
- [ ] T007 **M0 gate on the M4 Pro** `examples/spike.rs`: 4-bit qmm, gather_qmm, sdpa, the GDN `metal_kernel`, `compile`, memory APIs with timings; go/no-go in research R1
- [X] T008 [P] `engine.rs` + `identity.rs` (no MLX), first version: `Engine` trait with `reserve(total)`, `prefill`, `decode`, `trim`, `snapshot`/`restore`; ModelInfo, MemoryBudget, CancelToken, Sampling, TokenMask; identity over safetensors + 4 files with `~/.mrun/identity.toml`; tests
- [ ] T008a Reshape `engine.rs` to [contracts/engine.md](contracts/engine.md): `reserve(keep, total)`, `allocated()`, `PrefillPlan`/`Prefilled`, `chunk_len`, `forward`/`sample`/`top_logprobs`, fallible `trim`, `checkpoint`/`restore_checkpoint`/`drop_checkpoint`, `split_off`/`append`/`drop_segment`, `reset`; new errors (`NeedsMemory`, `NotTrimmable`, `NoRecurrentState`, `ReservationExceeded`, …); `Architecture::Qwen35Moe`, layer kinds, `recurrent_bytes`; MemoryBudget fields per data-model; remove `snapshot`/`restore`/`KvSegmentId`; unit tests for `chunk_len` (16K→2048, 32K→1024, 64K→512, 128K→256, 256K→128, 262K→128, depth 0); `CancelToken::yield_requested`, `Prefilled.yielded`, `split_off(at, shrink)`
- [X] T008b [P] `identity.rs`: `generation_config.json` and `special_tokens_map.json` are identifying files; `Sampling::validate` (NaN/negative temperature, top-p ∉ (0, 1], min-p ∉ [0, 1]) with tests
- [ ] T008c [P] `identity.rs`: `eos_union(dir)` (config + generation_config int/list + tokenizer EOS from `tokenizer_config.json`/`special_tokens_map.json`) feeding `ModelInfo.eos_tokens`; tests incl. a changed `generation_config.json`
- [ ] T009 [P] `load.rs` [mlx]: `qwen3`, `qwen3_moe`, `qwen3_5_moe` (`text_config` or root), quantization overrides (8-bit router/shared gate), mode inference, mlx-lm sanitize (drop vision/`mtp.*`, key remap, `gate_up_proj` split, unconverted conv1d axis + norm +1), `A_log` kept f32, weights estimate from file sizes, named rejections; tests with generated tiny checkpoints
- [X] T010 [P] `chat/{tokenizer.rs, template.rs, stop.rs}` first version: tokenizers 0.23 without `http`, minijinja + pycompat with `enable_thinking`, UTF-8-safe streaming decode, stop matcher with hold-back; tests
- [X] T010a [P] Template environment (FR-021): `trim_blocks`, `lstrip_blocks`, `tojson` = Python `json.dumps(ensure_ascii=False, indent)`; hand-written tests (`tojson_matches_python_json_dumps`, `block_tags_trim_like_hugging_face`)
- [ ] T010c Template goldens (SC-009): `scripts/gen_template_golden.py` (pinned transformers); fixtures for the Qwen3, Qwen3-Coder and Qwen3.6 templates (≥ 10 each: tools, tool calls/results, multi-turn, thinking on/off) checked on text and token IDs
- [ ] T010b [P] `chat/limits.rs`: `MAX_PROMPT_BYTES`, `MAX_STOPS`, `MAX_STOP_BYTES` in one place, used by `main.rs` and `stop.rs`
- [X] T011 `runtime/{handle.rs, generate.rs, scripted.rs}` first version: worker thread with bounded events (cap 64, stall timeout), generic loop on the old trait (reserve → prefill all but last → decode → stop/UTF-8 → `Finished` → trim to state tokens), output-limit clamping, `ScriptedEngine`; tests for one `Finished`, cancel, stall, clamping, reuse
- [ ] T011a `ScriptedEngine` on the new trait: byte costs per token/checkpoint, lazy allocation in 4,096 steps clamped to reservation and capacity, `NeedsMemory`, `chunk_len` + plan splits with per-chunk cancel and partial-chunk discard, yield after the current whole chunk (`Prefilled.yielded`), one pending pipelined step with rollback, hybrid mode (non-trimmable recurrent state, checkpoints required; `trim(0)`/`reserve(0, …)` reset it to zeros), checkpoints and segments recording token ranges, `split_off` with and without `shrink`
- [ ] T011b Generation loop on the new trait (`generate.rs`, `session.rs`), the single owner of the per-request loop: a reuse-policy parameter (`Session` here; 003's `PrefixStore` plugs in under `serve`, 002's worker only dequeues and calls `generate`); Session `committed` vs `tokens` (today it mirrors only the state, and a stop sequence's tokens stay in it); trim at every finish (EOS, stop sequence, length, cancel, error) incl. the pending pipelined token; hybrid fallback to the deepest session checkpoint (two newest kept, planned at `generation_start`, the start of the generation header); `NeedsMemory{bytes}` → policy evicts, retry, else error; on `yielded` → policy hook, then prefill the remaining tokens with the same plan; tests per finish path asserting `session.tokens == engine state`, and a yield test with an identical chunk split. (Already landed: TTFT at the first sampled token, `Sampling::validate` → `Rejected`.)
- [X] T011c `handle.rs`: `flume` channels (sync + async receive); a stalled consumer is cancelled and never waited on again; `catch_unwind` per request → `Finished{Error}` + state reset
- [ ] T011e `Finished` always delivered: today it is `try_send` on the event channel and is lost when that channel is full; move it to a one-slot channel sent after the event sender is dropped; tests: full channel still yields `Finished`, panicking engine, async consumer
- [X] T011d Dead code removal, part 1: `StopMatcher::held_len`, `EngineHandle::stall_timeout`, `Detokenizer` trait
- [ ] T011f Dead code removal, part 2: `TokenMask::vocab`, `ScriptedEngine::segment_tokens` (replaced by T011a's range records); clippy clean

## Phase 3: US1 — Chat in the terminal (P1) 🎯 MVP

- [ ] T012 [P] [US1] `state.rs` [mlx]: attention K/V with lazy allocation (input + 4,096, then +4,096 steps, round 256, clamp to reservation and capacity), one-layer growth, `allocated()` accounting, `NeedsMemory` pre-check, trim; tests: no copy on append (memory counters), growth peak ≤ state + one layer, never above reservation
- [ ] T013 [P] [US1] `models/qwen3.rs` [mlx]: port mlx-lm `qwen3.py`; per-layer prefill (`eval` each layer, cancel check, one in flight); tiny CPU test: chunked prefill vs step-by-step logits within 1e-3; cancel stops within one layer
- [ ] T014 [US1] `sample.rs` [mlx]: logprobs in logits dtype, greedy argmax, top-k (`argpartition` + `put_along_axis`), top-p, min-p, seeded categorical, mask; determinism and filter tests
- [ ] T015 [US1] `mlx_engine.rs`: `reserve`, `prefill` with plan/`chunk_len`/whole-chunk commit and yield after the current whole chunk, `forward`/`sample` with one-step pipelining and rollback, `top_logprobs`, `trim` (`trim(0)` resets recurrent state), `reset`
- [ ] T016 [US1] `cli/run.rs`: REPL (`/clear`, `/info`, `/exit`), `--prompt`, sampling flags, `--think`, stats line, model resolution, caps from `chat::limits`; drop `--chunk`; `--reserve-gib` default from the formula
- [ ] T017 [US1] `errors.rs`: missing/unsupported model (exit 3), empty prompt, limit 0, invalid sampling, caps; identity-changed report at load; tests

## Phase 4: US2 — Interrupt instantly (P1)

- [ ] T018 [US2] Ctrl-C in `cli/run.rs` (cancel; second within 1 s exits)
- [ ] T019 [US2] Cancel tests: `crates/mrun/tests/cancel.rs` (ScriptedEngine: each phase, completion race, partial chunk discarded, whole chunks kept, pending step trimmed) and `crates/mrun-engine/tests/cancel_cpu.rs` (tiny models incl. hybrid: state after cancel equals state after the last whole chunk)
- [ ] T020 [US2] `examples/bench_engine.rs` (FR-018): prefill/decode tok/s at given contexts; cancel latency, 50 random points per bucket (16K, 64K, max); JSON with machine + identity; `scripts/bench_reference.sh`; run SC-004 per model on the M4 Pro; record in research R5

## Phase 5: US3 — Never overcommit memory (P1)

- [ ] T021 [P] [US3] `budget.rs`: physical and working set from `device_info` (Linux: `/proc/meminfo`), reserve `clamp(physical ÷ 6, 2 GiB, 8 GiB)`, formula, overrides (`--allow-over`); unit tests incl. 16/48/128 GB
- [ ] T022 [US3] Pre-load check (incl. recurrent bytes); probe at two points (R6) incl. growth transient and recurrent rollback, cache key with MLX version and chunk policy; `kv_capacity_tokens ≤ min(max_position, P)`; MLX memory/cache/wired limits; refusal with all numbers (exit 4)
- [ ] T023 [US3] Admission boundary tests (SC-005) on ScriptedEngine and a tiny model: capacity − 1 → one token "length"; capacity → rejected with both numbers; oversized limit clamped and reported; cached state counted as available
- [ ] T024 [US3] `mrun run --info` breakdown (weights, scratch, recurrent, state bytes/tokens, probed depth, reserve, free memory); snapshot test
- [ ] T024a [US3] FR-022 resident model: no idle-unload timer anywhere in `runtime/`; test that the worker keeps the model loaded across a long idle period (paused clock) and only `reset()`s state

## Phase 6: US4 — Trustworthy output (P2)

- [ ] T025 [P] [US4] `models/qwen3_moe.rs`: router (8-bit gate, softmax precise, top-k, norm_topk_prob), SwitchGLU via gather_qmm with sort at ≥ 64 indices; tiny CPU test
- [ ] T026 [US4] `scripts/gen_reference.py` (mlx-lm `ba8a08f`, `--prefill-step-size 2048`): ≥ 20 prompts up to 8K, 256 greedy IDs, EOS set, per model
- [ ] T027 [US4] `crates/mrun-engine/tests/conformance.rs` (`#[ignore]`, macOS) + `mrun run --check` (first mismatch, margin from `top_logprobs`); EOS union equals mlx-lm's set
- [ ] T027a [US4] Parity run Qwen3-8B-4bit on the M4 Pro (SC-001); record
- [ ] T027b [US4] Parity run Qwen3-Coder-30B-A3B-Instruct-4bit (SC-001); record
- [ ] T027c [US4] Parity run Qwen3.6-35B-A3B 4-bit (SC-001, after T031); record
- [ ] T028 [US4] `crates/mrun/tests/sampling.rs`: seeded reproducibility, validation rejections, stop finish reason and trimmed tokens

## Phase 7: Hybrid model and state primitives (US1, US4, US5)

- [ ] T029 [P] `models/gated_delta.rs` + `kernels/gated_delta.metal.inc` [mlx]: kernel bodies verbatim from mlx-lm (packed `gated_delta_step_packed_btree` for Dk = 128, generic `gated_delta_step`), dispatch rules, grids and template args per R12; `gated_delta_ops` reference; `normalize_qk`, `compute_g`; tests: ops vs mlx-lm fixture (CPU), kernel vs ops and packed vs generic bitwise (Metal)
- [ ] T030 `state.rs` recurrent part: conv state `[1, k−1, conv_dim]` and delta state `[1, Hv, Dv, Dk]` f32 per linear layer; chunk-start rollback on cancel; pending-step rollback; `NotTrimmable`; `NoRecurrentState` after `split_off`
- [ ] T031 `models/qwen3_5_moe.rs` [mlx]: layer pattern from `full_attention_interval`, gated attention (q/gate split, q/k norm, partial RoPE), GDN layer (conv1d, SiLU, gated RMSNorm), sparse MoE with shared expert + sigmoid gate; config dimension checks; tiny random CPU config: chunked prefill vs step-by-step within 1e-3
- [ ] T032 [US5] Checkpoints in `MlxEngine`/`state.rs`: capture at plan positions during prefill, `checkpoint()`, `restore_checkpoint` (copy), `drop_checkpoint`, `NeedsMemory` pre-check, deep-copy pointer assertions; recurrent term in the budget
- [ ] T033 [US5] `split_off`/`append` in `MlxEngine` for attention and hybrid state (per layer copy + release, `shrink` vs kept allocation, consume vs copy); tests: peak ≤ capacity + one layer (memory counters). Attention part after T015; hybrid part after T032
- [ ] T034 [US5] `crates/mrun-engine/tests/exactness.rs` (SC-008): tiny CPU models always; real models `#[ignore]`: split_off/append and checkpoint-restore + re-prefill with the same plan → bit-identical next-token logits; same chunk split produced for cache-on and cache-off runs

## Phase 8: Polish

- [ ] T035 [P] SC-002/SC-003 per model with `bench_engine` vs `bench_reference.sh` (1K/16K); SC-007 cold start for 8B; record in research
- [ ] T036 [P] 30-minute soak and 2-hour leak soak (SC-006): `phys_footprint`, `vm_stat` swap-outs, MLX active memory and Metal buffer count flat; no idle unload
- [ ] T037 [P] Offline/no-write audit (FR-016): no network features compiled; run under `sandbox-exec` network deny; no writes outside `~/.mrun`
- [ ] T038 Update `README.md` and site pages (`getting-started.md`, `memory.md`) and run [quickstart](quickstart.md)

## Dependencies

- T001 → T002/T003/T004/T008.
- No MLX, can start now: T008a → T011a → T011b → T011e → T011f; T008c, T010b, T010c in parallel.
- T004 → T004a, T005 → T006 → T006a, T007 (gate) → T009, T012, T013, T029.
- T012 + T013 + T014 → T015 → T016/T017 (MVP on Qwen3-8B).
- US2/US3 after T015 (T024a after T011b). T026 after T009 + T015 (T020 fixes the schedule); T027 after T026; T027a after T027; T027b after T025; T027c after T031.
- Hybrid: T029 + T030 → T031 → T032 → T033 (hybrid part) → T034. T033 attention part after T015; hybrid part after T032.
- Polish after the stories it measures.

## Parallel opportunities

T004 with T008a · T008c/T010b/T010c with T008a · T009/T012/T013/T029 after T006 · T021 with T018 · T025 with T020 · T035/T036/T037.
