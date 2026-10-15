---
description: "Tasks for 001 Engine Core"
---

# Tasks: Engine Core

**Input**: [plan.md](plan.md), [spec.md](spec.md), [research.md](research.md), [data-model.md](data-model.md), [contracts/engine.md](contracts/engine.md)

**Tests**: Required by the constitution. Linux CI runs `ScriptedEngine` and tiny random-weight CPU
models; macOS-only tests are `#[ignore]` + `MRUN_TEST_MODELS`.

**Status (2026-10-15, after `98ac3d0`)**: `[X]` = done in code with tests. Partly done work is
split: the landed part is ticked, the remainder is a separate open task.

## Format: `[ID] [P?] [Story] Description`

## Phase 1: Setup

- [X] T001 Workspace: root `Cargo.toml` (members `crates/mrun-engine`, `crates/mrun`; shared lints), package metadata in `crates/mrun/Cargo.toml`, old scaffold removed
- [X] T002 [P] CI `.github/workflows/ci.yml` baseline: Linux fmt, clippy `-D warnings`, `cargo test --workspace`; macOS build only. (MLX dependencies and the no-MLX job move to T004a.)
- [X] T003 [P] `crates/mrun/src/main.rs` with clap: `run` arguments and input-cap checks; no engine wired yet

## Phase 2: Foundational (blocks all stories)

- [X] T004 `crates/mlx-sys`: mlx-c v0.7.0 (`a341b49`) submodule, MLX v0.32.4 (`36969c0`) sha256-pinned tarball; `build.rs` via CMake with `FETCHCONTENT_SOURCE_DIR_MLX`; Metal + Foundation + Accelerate on macOS, CPU + OpenBLAS/LAPACKE on Linux; metallib next to the binary. *Note*: mlx-c is a git submodule (`crates/mlx-sys/vendor/mlx-c`); the MLX pin is the sha256 of the uncompressed `git archive` tar of `36969c0` (`7b05da60…f592`), fetched from GitHub's archive or, as fallback, a shallow git fetch, both verified. mlx-c v0.7.0 does not compile against MLX v0.32.4 (`gather_qmm` gained `global_scale`): `patches/mlx-c-0001-gather-qmm-global-scale.patch` passes `std::nullopt`, applied to a copy in `OUT_DIR`. `mlx-sys` is excluded from the workspace (MLX build) and is tested with `cargo test --manifest-path crates/mlx-sys/Cargo.toml`; feature `mlx` of `mrun-engine` is off by default. `MLX_SYS_PREBUILT_DIR` skips the build. Linux verified (clean build ≈ 6 min on 4 cores); macOS path written but not yet built → T004b.
- [ ] T004b Verify the macOS half of T004 on Apple hardware/CI: Metal build of `mrun-engine --features mlx`, `mlx.metallib` copied next to the binaries (`target/<profile>/{,deps,examples}`) and found at runtime
- [X] T004a CI: new `mlx-sys-linux` job (CMake/OpenBLAS/LAPACKE, submodules, cached `crates/mlx-sys/target`, clippy + `cargo test --manifest-path crates/mlx-sys/Cargo.toml`); fast Linux job unchanged (default features have no MLX, so it is the no-MLX job); macOS job adds `cargo build -p mrun-engine --features mlx`
- [X] T005 `crates/mlx-sys/src/bindings.rs` generated once (`regen.sh`) and checked in; smoke test evaluating `add` (bindgen 0.73.2, `mlx_.*` allowlist, x86_64 LP64 host; aarch64-only f16/bf16 accessors omitted; `tests/smoke.rs` passes on Linux CPU)
- [ ] T006 `crates/mrun-engine/src/mlx/` [mlx]: `Array` (RAII, `!Send`, `!Sync`), error handler → `Result`; ops: zeros, astype, reshape, transpose, matmul, quantized_matmul/gather_qmm with mode, take, argpartition, take_along_axis, put_along_axis, sort, cumsum, argmax, logsumexp, logaddexp, softmax precise, sigmoid, where, split, slice, slice_update, concatenate, contiguous, conv1d (groups), random key/categorical; `fast::{sdpa, rope, rms_norm, metal_kernel}`; `compile` (shapeless closures); eval/async_eval; stream; safetensors; memory (active/peak/cache, limits, clear_cache, wired); `device_info` (`memory_size`, `max_recommended_working_set_size`); `deep_copy` with pointer assertion; CPU unit tests per op
- [ ] T006a [mlx] Compiled-op parity (R14): on Metal compare `swiglu`, `precise_swiglu`, `compute_g`, `swiglu_oai` compiled vs unfused on every 0.1 family's shapes; bind `compile` for them unless bitwise equal; record in research R14
- [ ] T007 **M0 gate on the M4 Pro** `examples/spike.rs`: 4-bit qmm, gather_qmm, sdpa, the GDN `metal_kernel`, `compile`, memory APIs with timings; go/no-go in research R1
- [X] T008 [P] `engine.rs` + `identity.rs` (no MLX), first version: `Engine` trait with `reserve(total)`, `prefill`, `decode`, `trim`, `snapshot`/`restore`; ModelInfo, MemoryBudget, CancelToken, Sampling, TokenMask; identity over safetensors + 4 files with `~/.mrun/identity.toml`; tests
- [X] T008a Reshape `engine.rs` to [contracts/engine.md](contracts/engine.md): `reserve(keep, total)`, `allocated()`, `PrefillPlan`/`Prefilled`, `chunk_len`, `forward`/`sample`/`top_logprobs`, fallible `trim`, `checkpoint`/`restore_checkpoint`/`drop_checkpoint`, `split_off`/`append`/`drop_segment`, `reset`; new errors (`NeedsMemory`, `NotTrimmable`, `NoRecurrentState`, `ReservationExceeded`, …); `Architecture::Qwen35Moe` (replaced by T040a), layer kinds, `recurrent_bytes`; MemoryBudget fields per data-model; remove `snapshot`/`restore`/`KvSegmentId`; unit tests for `chunk_len` (16K→2048, 32K→1024, 64K→512, 128K→256, 256K→128, 262K→128, depth 0); `CancelToken::yield_requested`, `Prefilled.yielded`, `split_off(at, shrink)`
- [X] T008b [P] `identity.rs`: `generation_config.json` and `special_tokens_map.json` are identifying files; `Sampling::validate` (NaN/negative temperature, top-p ∉ (0, 1], min-p ∉ [0, 1]) with tests
- [X] T008d [P] `TokenMask::from_words(&[u32], vocab)`: build a mask from a 32-bit-word bitset (llguidance layout), bits at or above `vocab` cleared; tests
- [X] T008e [P] `identity.rs` robustness: an unparsable `identity.toml` is rebuilt (hashes recomputed) instead of failing the load; the cache is written through a per-process temp file in the same directory (`tempfile::NamedTempFile::persist`), so concurrent processes never share a temp path; tests
- [X] T008c [P] `identity.rs`: `eos_union(dir)` (config + generation_config int/list + tokenizer EOS from `tokenizer_config.json`/`special_tokens_map.json`, plus the family's declared end tokens (010 `eos_extra`, e.g. Harmony `<|call|>`, `<|return|>`)) feeding `ModelInfo.eos_tokens`; tests incl. a changed `generation_config.json`
- [ ] T009 [P] `load.rs` [mlx]: `qwen3`, `qwen3_moe`, `qwen3_5_moe` (`text_config` or root), quantization overrides (8-bit router/shared gate), mode inference, mlx-lm sanitize (drop vision/`mtp.*`, key remap, `gate_up_proj` split, unconverted conv1d axis + norm +1), `A_log` kept f32, weights estimate from file sizes, named rejections; tests with generated tiny checkpoints
- [X] T010 [P] `chat/{tokenizer.rs, template.rs, stop.rs}` first version: tokenizers 0.23 without `http`, minijinja + pycompat with `enable_thinking`, UTF-8-safe streaming decode, stop matcher with hold-back; tests
- [X] T010a [P] Template environment (FR-021): `trim_blocks`, `lstrip_blocks`, `tojson` = Python `json.dumps(ensure_ascii=False, indent)`; hand-written tests (`tojson_matches_python_json_dumps`, `block_tags_trim_like_hugging_face`)
- [X] T010e [P] Template environment, rest of T010a (FR-021): `strftime_now(fmt)` returning the server start date; the probe stubs a fixed date; tests
- [ ] T010c Template goldens (SC-009): `scripts/gen_template_golden.py` (pinned transformers); fixtures for the Qwen3, Qwen3-Coder and Qwen3.6 templates (≥ 10 each: tools, tool calls/results, multi-turn, thinking on/off) checked on text and token IDs
- [ ] T010d [P] Template goldens (SC-009) for the gpt-oss template (Harmony; `reasoning_effort` low/medium/high, tool calls and results, analysis dropped from earlier turns), ≥ 10, text and token IDs (needs the repo pinned by 004 T005). (Devstral goldens: 0.2, 010 T010)
- [X] T010f [P] `chat/template.rs`: `RenderOptions.reasoning_effort: Option<String>` passed to the template as `reasoning_effort` (left undefined when `None`, so the template default applies) for gpt-oss; `chat/prompt.rs`: `RenderedPrompt::from_tokens` and the pure `generation_start(without_header, with_header)` (common token prefix of a turn rendered without and with the generation prompt) used by T016; tests
- [X] T010b [P] `chat/limits.rs`: `MAX_PROMPT_BYTES`, `MAX_STOPS`, `MAX_STOP_BYTES` in one place, used by `main.rs` and `stop.rs`
- [X] T011 `runtime/{handle.rs, generate.rs, scripted.rs}` first version: worker thread with bounded events (cap 64, stall timeout), generic loop on the old trait (reserve → prefill all but last → decode → stop/UTF-8 → `Finished` → trim to state tokens), output-limit clamping, `ScriptedEngine`; tests for one `Finished`, cancel, stall, clamping, reuse
- [X] T011a `ScriptedEngine` on the new trait: byte costs per token/checkpoint, lazy allocation in 4,096 steps clamped to reservation and capacity, `NeedsMemory`, `chunk_len` + plan splits with per-chunk cancel and partial-chunk discard, yield after the current whole chunk (`Prefilled.yielded`), one pending pipelined step with rollback, hybrid mode (non-trimmable recurrent state, checkpoints required; `trim(0)`/`reserve(0, …)` reset it to zeros), checkpoints and segments recording token ranges, `split_off` with and without `shrink`
- [X] T011b Generation loop on the new trait (`generate.rs`, `session.rs`), the single owner of the per-request loop: a reuse-policy parameter (`Session` here; 003's `PrefixStore` plugs in under `serve`, 002's worker only dequeues and calls `generate`); Session `committed` vs `tokens`; trim at every finish (EOS, stop sequence, length, cancel, error) incl. the pending pipelined token; hybrid fallback to the deepest session checkpoint (two newest kept, planned at `generation_start`, the start of the generation header); `NeedsMemory{bytes}` → policy evicts, retry, else error; on `yielded` → policy hook, then prefill the remaining tokens with the same plan; tests per finish path asserting `session.tokens == engine state`, and a yield test with an identical chunk split. (Already landed: TTFT at the first sampled token, `Sampling::validate` → `Rejected`.)
- [X] T011i Adapt `Session` (and the loop) to the reshaped reuse-policy seam of the [003 contract](../003-prefix-cache/contracts/cache.md): `initial_need(engine, &RenderedPrompt, total) -> u64` (growth to `P + min(out, 4K)` beyond the live state plus planned checkpoints) with a pre-prepare `evict(shortfall, Some(prompt))`; `evict(engine, bytes, protect) -> u64` (bytes freed, 0 = nothing left); `prepare(engine, &RenderedPrompt, total) -> Prepared{resume, chunks}` where the policy reserves (`plan()` removed; 003 adds `miss`); `on_pressure(engine, level, idle)` called with a pending `Pressure` level at prefill yields and between decode tokens (`PressureGauge`: edge-triggered, raising requests a yield of the running request); `commit(…, &RenderedPrompt)`; tests
- [X] T011j `GenerateRequest.prompt: RenderedPrompt` (`tokens`, `messages: Vec<MessageMark{start, role}>`, `system_tools_end`, `generation_start`) replacing `prompt: Vec<u32>` + `generation_start: Option<u32>`; `hook: Option<Box<dyn DecodeHook>>` (FR-011, 002 FR-036): `on_token(id) -> Option<TokenMask>` (mask for the next sampled step), `forced_tokens()` (appended without sampling, still fed and reported), `ignore_eos()`; tests for mask, forced tokens and ignore_eos on ScriptedEngine
- [X] T011k Lossless token events: `Event::Token { id, text, held }` for every output token id (`text` = visible now, possibly empty; `held` = bytes held back by stop matching), `Event::Flush { text }` for held text released at a finish without a stop match, `FinishReason::Stop { stop, eos_token: Option<u32> }`; tests
- [X] T011l `test-support` cargo feature of `mrun` gating `runtime::scripted` (`ScriptedEngine`, its pub test knobs) and `chat::tokenizer::test_support`; integration tests enable it through a self dev-dependency; clippy with and without the feature
- [X] T011m `EngineError::TooManyCheckpoints { planned, max }` replaces T011h's `InvalidRange` for more than `MAX_PLAN_CHECKPOINTS` planned checkpoints (contract, ScriptedEngine, test)
- [X] T011n `ScriptedEngine::trim` with a pending step per contract: `trim(active_len())` keeps (confirms) the pending token, `trim(active_len() − 1)` rolls it back; position-bound state allows only these (or 0); tests
- [X] T011o TTFT clock starts when `generate` is called (the worker's dequeue), not after reuse and reservation; prefill speed still counts from the start of prefill; test
- [X] T011c `handle.rs`: `flume` channels (sync + async receive); a stalled consumer is cancelled and never waited on again; `catch_unwind` per request → `Finished{Error}` + state reset. *Note*: 002 T005 refactors this file (no new `worker.rs`): worker generic over the reuse policy, lazy `Option<E>` from the server's factory, FIFO of 16 with a wait deadline, `Reset` handled inside `catch_unwind`
- [X] T011e `Finished` always delivered: today it is `try_send` on the event channel and is lost when that channel is full; move it to a one-slot channel sent after the event sender is dropped; tests: full channel still yields `Finished`, panicking engine, async consumer
- [X] T011g Engine-level `reset_all()` that also drops stored segments and checkpoints, called by the worker after a panic (checkpoints captured during the panicked request are otherwise unreachable); test with a panicking ScriptedEngine
- [X] T011h More than `MAX_PLAN_CHECKPOINTS` planned checkpoints in one `prefill` → `InvalidRange` (contract, ScriptedEngine, test); error variant replaced by T011m
- [X] T011d Dead code removal, part 1: `StopMatcher::held_len`, `EngineHandle::stall_timeout`, `Detokenizer` trait
- [X] T011f Dead code removal, part 2: `ScriptedEngine::segment_tokens` gone (replaced by T011a's range records); `TokenMask::vocab` kept (it bounds `allow`/`is_allowed`); clippy clean

## Phase 3: US1 — Chat in the terminal (P1) 🎯 MVP

- [ ] T012 [P] [US1] `state.rs` [mlx]: attention K/V with lazy allocation (input + 4,096, then +4,096 steps, round 256, clamp to reservation and capacity), one-layer growth, `allocated()` accounting, `NeedsMemory` pre-check, trim; tests: no copy on append (memory counters), growth peak ≤ state + one layer, never above reservation
- [ ] T013 [P] [US1] `models/qwen3.rs` [mlx]: port mlx-lm `qwen3.py`; per-layer prefill (`eval` each layer, cancel check, one in flight); tiny CPU test: chunked prefill vs step-by-step logits within 1e-3; cancel stops within one layer
- [ ] T014 [US1] `sample.rs` [mlx]: logprobs in logits dtype, greedy argmax, top-k (`argpartition` + `put_along_axis`), top-p, min-p, seeded categorical, mask; determinism and filter tests
- [ ] T015 [US1] `mlx_engine.rs`: `reserve`, `prefill` with plan/`chunk_len`/whole-chunk commit and yield after the current whole chunk, `forward`/`sample` with one-step pipelining and rollback, `top_logprobs`, `trim` (`trim(0)` resets position-bound state), `reset`, `reset_all`
- [ ] T016 [US1] `cli/run.rs`: REPL (`/clear`, `/info`, `/exit`), `--prompt`, sampling flags, `--think` (gpt-oss: `RenderOptions.reasoning_effort`, T010f), stats line, model resolution, caps from `chat::limits`; drop `--chunk`; `--reserve-gib` default from the formula; each turn's `RenderedPrompt` with `generation_start` computed by T010f's `generation_start` from the turn rendered without and with the generation prompt
- [ ] T017 [US1] `errors.rs`: missing/unsupported model (exit 3), empty prompt, limit 0, invalid sampling, caps; identity-changed report at load; tests

## Phase 4: US2 — Interrupt instantly (P1)

- [ ] T018 [US2] Ctrl-C in `cli/run.rs` (cancel; second within 1 s exits)
- [X] T019 [US2] Cancel tests `crates/mrun/tests/cancel.rs` (ScriptedEngine: each phase, completion race, partial chunk discarded, whole chunks kept, pending step trimmed)
- [ ] T019a [US2] [mlx] `crates/mrun-engine/tests/cancel_cpu.rs` (tiny models incl. hybrid: state after cancel equals state after the last whole chunk)
- [ ] T020 [US2] `examples/bench_engine.rs` (FR-018): prefill/decode tok/s at given contexts; cancel latency, 50 random points per bucket (16K, 64K, max); JSON with machine + identity; `scripts/bench_reference.sh`; run SC-004 per model on the M4 Pro; record in research R5

## Phase 5: US3 — Never overcommit memory (P1)

- [X] T021 [P] [US3] `budget.rs`: physical and working set from `device_info` (Linux: `/proc/meminfo`), reserve `clamp(physical ÷ 6, 2 GiB, 8 GiB)`, formula, overrides (`--allow-over`); unit tests incl. 16/48/128 GB
- [ ] T022 [US3] Pre-load check (incl. position-bound bytes (`recurrent_bytes`)); probe at two points (R6) incl. growth transient and position-bound rollback, cache key with MLX version and chunk policy; `kv_capacity_tokens ≤ min(max_position, P)`; MLX memory/cache/wired limits; refusal with all numbers (exit 4)
- [X] T023 [US3] Admission boundary tests (SC-005) on ScriptedEngine (`crates/mrun/tests/admission.rs`): capacity − 1 → one token "length"; capacity → rejected with both numbers; oversized limit clamped and reported; cached state counted as available
- [ ] T023a [US3] [mlx] The T023 boundary cases on a tiny CPU model
- [ ] T024 [US3] `mrun run --info` breakdown (weights, scratch, position-bound, state bytes/tokens, probed depth, reserve, free memory); snapshot test
- [X] T024a [US3] FR-022 resident model: no idle-unload timer anywhere in `runtime/`; test that the worker keeps the model loaded across a long idle period (paused clock) and only `reset()`s state

## Phase 6: US4 — Trustworthy output (P2)

- [ ] T025 [P] [US4] `models/qwen3_moe.rs`: router (8-bit gate, softmax precise, top-k, norm_topk_prob), SwitchGLU via gather_qmm with sort at ≥ 64 indices; tiny CPU test
- [ ] T026 [US4] `scripts/gen_reference.py` (mlx-lm `ba8a08f`, `--prefill-step-size 2048`): ≥ 20 prompts up to 8K, 256 greedy IDs, EOS set, per model
- [ ] T027 [US4] `crates/mrun-engine/tests/conformance.rs` (`#[ignore]`, macOS) + `mrun run --check` (first mismatch, margin from `top_logprobs`); EOS union equals mlx-lm's set
- [ ] T027a [US4] Parity run Qwen3-8B-4bit on the M4 Pro (SC-001); record
- [ ] T027b [US4] Parity run Qwen3-Coder-30B-A3B-Instruct-4bit (SC-001); record
- [ ] T027c [US4] Parity run Qwen3.6-35B-A3B 4-bit (SC-001, after T031); record
- [ ] T027d [US4] Parity run gpt-oss-20b MXFP4 (SC-001, after T042 + T043a); record whether stock mlx-lm (rotating cache) also matches bit for bit, closing the R16 caveat
- [ ] T027e *Removed (0.1 scope cut, 2026-10-15)*: Devstral Small 2 parity moved to 0.2 (010 T010)
- [X] T028 [US4] `crates/mrun/tests/sampling.rs` on ScriptedEngine: validation rejections, stop finish reason and trimmed tokens
- [ ] T028a [US4] [mlx] Seeded reproducibility on the real sampler (after T014)

## Phase 7: Hybrid model and state primitives (US1, US4, US5)

- [ ] T028b [P] [mlx] Evaluate MLX v0.32.4 `fast::gated_delta_update` (not exposed by mlx-c v0.7.0): add a C shim in `mlx-sys` (patch, like the `gather_qmm` one) and compare it bitwise against mlx-lm's `gated_delta` Metal kernel on Qwen3.6 shapes; if equal and not slower, T029 uses the native op instead of porting the kernel bodies
- [ ] T029 [P] `models/gated_delta.rs` + `kernels/gated_delta.metal.inc` [mlx]: kernel bodies verbatim from mlx-lm (packed `gated_delta_step_packed_btree` for Dk = 128, generic `gated_delta_step`), dispatch rules, grids and template args per R12; `gated_delta_ops` reference; `normalize_qk`, `compute_g`; tests: ops vs mlx-lm fixture (CPU), kernel vs ops and packed vs generic bitwise (Metal)
- [ ] T030 `state.rs` recurrent part: conv state `[1, k−1, conv_dim]` and delta state `[1, Hv, Dv, Dk]` f32 per linear layer; chunk-start rollback on cancel; pending-step rollback; `NotTrimmable`; `NoRecurrentState` after `split_off`
- [ ] T031 `models/qwen3_5_moe.rs` [mlx]: layer pattern from `full_attention_interval`, gated attention (q/gate split, q/k norm, partial RoPE), GDN layer (conv1d, SiLU, gated RMSNorm), sparse MoE with shared expert + sigmoid gate; config dimension checks; tiny random CPU config: chunked prefill vs step-by-step within 1e-3
- [ ] T032 [US5] Checkpoints in `MlxEngine`/`state.rs`: capture at plan positions during prefill, `checkpoint()`, `restore_checkpoint` (copy), `drop_checkpoint`, `NeedsMemory` pre-check, deep-copy pointer assertions; recurrent term in the budget
- [ ] T033 [US5] `split_off`/`append` in `MlxEngine` for attention and hybrid state (per layer copy + release, `shrink` vs kept allocation, consume vs copy); tests: peak ≤ capacity + one layer (memory counters). Attention part after T015; hybrid part after T032
- [ ] T034 [US5] `crates/mrun-engine/tests/exactness.rs` (SC-008): tiny CPU models always; real models `#[ignore]`: split_off/append and checkpoint-restore + re-prefill with the same plan → bit-identical next-token logits; same chunk split produced for cache-on and cache-off runs

## Phase 7b: Block library and the remaining 0.1 families (US1, US4, US5)

Reference sources: mlx-lm `ba8a08f` `mlx_lm/models/{gpt_oss.py, rope_utils.py, switch_layers.py, activations.py, cache.py, base.py}`. Registry, tiers and post-0.1 families are spec 010.

- [ ] T039 [P] `models/blocks/` [mlx] (layout per 010, R15): attention block (optional q/k/v/o bias, q/k RMSNorm, output gate, window `None | Some(w)` as a slice of the full KV plus windowed causal mask, optional sinks to SDPA), rope (default, partial, YaRN via `fast::rope` `freqs`; position-scaled queries come with `mistral` in 0.2), feed-forward (SwiGLU, `swiglu_oai`, routers softmax→top-k and top-k→softmax, SwitchGLU with optional per-expert bias, shared expert with sigmoid gate), RMSNorm; T013, T025 and T031 compose these blocks instead of private copies; tiny CPU tests per block option against mlx-lm fixtures
- [X] T040a `engine.rs`: `Architecture` enum replaced by `ModelInfo.family: String` (registry name) and `LayerKind::{FullAttention, SlidingAttention{window}, Linear}`; ScriptedEngine and tests updated
- [X] T040 `models/registry.rs` (no MLX): `model_type` (and wrapped `text_config.model_type`) → family entry, `ModelInfo.family`, layer kinds (`FullAttention`, `SlidingAttention{window}`, `Linear`) from `layer_types`/`full_attention_interval`/`sliding_window`, state kinds and byte costs, the family's chat syntax (002; one `SyntaxId` per family since T040d), declared end tokens (010 `eos_extra`) for the EOS union; unknown type refused with the 004 E034 text; fixture configs per registered family and refused ones
- [ ] T040b `load.rs` [mlx]: registry → family module, unknown type refused before reading weights (exit 3), `sinks`, rope scaling (FR-025); load-refusal fixtures (SC-010)
- [ ] T040c [mlx] Wire `identity::eos_union(dir, family.eos_extra)` into the load path (with T009/T040b): `ModelInfo.eos_tokens` comes from it; test on a generated tiny checkpoint that a family end token (Harmony `<|call|>`) ends generation
- [X] T040d `models/registry.rs`: one chat syntax per family in 0.1 (`syntax: SyntaxId`; `qwen3_moe` = `QwenXml` only, `qwen3_5_moe` = `QwenXml`, `gpt_oss` = `Harmony`, `qwen3` = `Hermes`); `SyntaxId` defined in `mrun-engine` and re-exported at the crate root (002's parsers key on it); E034 lists `qwen3_5_moe (qwen3.6), qwen3_moe (qwen3-coder), gpt_oss (gpt-oss), qwen3`; tests
- [ ] T041 [P] [mlx] MXFP4: `quantized_matmul`/`gather_qmm` with `mode = "mxfp4"` (null biases), per-path quantization `mode` from `config.json`; CPU test vs an mlx-lm fixture; Metal timing in the spike (T007)
- [ ] T042 `models/gpt_oss.rs` [mlx] per R16: sanitize (`gate_up_proj` interleaved split incl. bias, `_blocks`/`_scales`), alternating `layer_types`, attention with bias + sinks + YaRN, sliding layers on the full KV with the window slice, top-k→softmax router, SwitchGLU with biases and `swiglu_oai`; tiny CPU config: chunked prefill vs step-by-step within 1e-3 for prompts shorter and longer than the window
- [ ] T043 *Removed (0.1 scope cut, 2026-10-15)*: `models/mistral.rs` (Devstral Small 2) moved to 0.2 (010 T010; R17 kept for that port)
- [ ] T043a `scripts/gen_reference.py`: gpt_oss reference with a temporal-order windowed cache (full KV, window slice) substituted for `RotatingKVCache` in the reference process, plus a stock-mlx-lm token run (R16); reference for gpt-oss-20b
- [ ] T045 [US5] Sliding-window exactness in `tests/exactness.rs` (SC-008): tiny `gpt_oss` CPU model always, gpt-oss-20b `#[ignore]`: `trim`, `split_off`/`append` and continued prefill at positions before, at and beyond the window (incl. within the last 127 positions of a chunk) → bit-identical next-token logits; no checkpoint created (zero-byte)
- [X] T046 `ScriptedEngine` profiles and budget fixtures per 0.1 family (bytes per token, position-bound bytes from the data-model table)
- [ ] T046a `--info` shows family, syntax (template check) and the per-family numbers (with T024)

## Phase 8: Polish

- [ ] T035 [P] SC-002/SC-003 per 0.1 family with `bench_engine` vs `bench_reference.sh` (1K/16K); SC-007 cold start for 8B; record in research
- [ ] T036 [P] 30-minute soak and 2-hour leak soak (SC-006): `phys_footprint`, `vm_stat` swap-outs, MLX active memory and Metal buffer count flat; no idle unload
- [ ] T037 [P] Offline/no-write audit (FR-016): no network features compiled; run under `sandbox-exec` network deny; no writes outside `~/.mrun`
- [ ] T038 Update `README.md` and site pages (`getting-started.md`, `memory.md`) and run [quickstart](quickstart.md)

## Dependencies

- T001 → T002/T003/T004/T008.
- No MLX, can start now: T008a → T011a → T011b → T011e → T011f; T008c, T010b, T010c, T010e in parallel. T011b → T011i/T011j/T011k/T011o (before 002 T005 and 003 T008); T011l, T011m, T011n, T008d, T008e, T010f, T040d independent.
- T004 → T004a, T005 → T006 → T006a, T007 (gate) → T009, T012, T013, T029.
- T012 + T013 + T014 → T015 → T016/T017 (MVP on Qwen3-8B).
- US2/US3 after T015 (T024a after T011b). T026 after T009 + T015; T027 after T026; T027a after T027; T027b after T025; T027c after T031.
- Hybrid: T029 + T030 → T031 → T032 → T033 (hybrid part) → T034. T033 attention part after T015; hybrid part after T032.
- Families: T006 → T039 (before T013/T025/T031 compose blocks) → T040; T041 after T006; T042 after T039 + T040 + T041; T040c with T009 + T040b; T043a after T026; T027d after T042 + T043a; T045 after T042 + T033; T046 with T011a; T010d after the repos are pinned (004 T005).
- Polish after the stories it measures.

## Parallel opportunities

T004 with T008a · T008c/T010b/T010c with T008a · T009/T012/T013/T029 after T006 · T021 with T018 · T025 with T020 · T042 after T040 · T010d with T043a · T035/T036/T037.
