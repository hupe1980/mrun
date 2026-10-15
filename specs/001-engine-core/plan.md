# Implementation Plan: Engine Core

**Branch**: `001-engine-core` | **Date**: 2026-10-14 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `specs/001-engine-core/spec.md`

## Summary

Run Qwen3 dense and Qwen3-MoE MLX-quantized models from the `mrun` binary on Apple Silicon. We
build our own `mlx-sys` bindings over vendored mlx-c/MLX, a thin safe array layer, and 1:1 ports
of mlx-lm's `qwen3` and `qwen3_moe` graphs. One engine thread owns all MLX state: weights, the
active KV cache (allocated at admission, trim by offset, deep-copy snapshots), chunked prefill
(2,048 tokens like mlx-lm, cancellable after every layer), on-device sampling with async-eval
pipelining, and a startup memory budget that pre-checks file sizes, measures weights and a cached
worst-case scratch probe, then sizes KV capacity (capped by max position). The `Engine` trait is
primitive-based and MLX-free so the generation loop, 002 and 003 are testable everywhere. `mrun run` provides an interactive
streaming chat; `mrun run --check` runs the greedy conformance set against stored mlx-lm outputs.

## Technical Context

**Language/Version**: Rust 1.94.0 (edition 2024); C/C++17 for vendored MLX built via CMake
**Primary Dependencies**: MLX v0.32.4 + mlx-c v0.7.0 (`a341b49`), vendored and hash-checked;
bindgen output checked in (no libclang at build time); `tokenizers` 0.23 (HF tokenizer.json);
`minijinja` 3 + `minijinja-contrib` (pycompat) for chat templates (tokenizers built without its
`http` feature: no network, FR-015); `safetensors` via MLX loader;
tokio 1.53 (async side), crossbeam-channel (engine command/event channels); clap 4.6
**Storage**: Model files in the HF cache (feature 004 resolves paths); reference outputs for
conformance in `crates/mrun-engine/tests/reference/*.json`
**Testing**: cargo test. Linux CI: MLX CPU backend, tiny random-weight Qwen3/Qwen3-MoE configs
(generated in tests), KV/sampler/budget unit tests. macOS: logits oracle and greedy conformance vs
mlx-lm on real 4-bit weights (`MRUN_TEST_MODELS=...`), `mrun bench` speed cells (005)
**Target Platform**: macOS 15+ on Apple Silicon (Metal); Linux x86_64/aarch64 for CI (CPU)
**Project Type**: Rust workspace producing one CLI/server binary
**Performance Goals**: ≥ 90% of mlx-lm decode and prefill tok/s on the same weights (SC-002/003);
cancellation ≤ 300 ms p95 at 16K (SC-004); first token ≤ 15 s cold start for 8B (SC-007)
**Constraints**: Memory budget by construction (Principle III); no network (FR-015); input caps
(FR-016); exact match with mlx-lm greedy output under the same chunk size and schedule (SC-001)
**Scale/Scope**: 2 model families, 1 active sequence, contexts up to the budgeted capacity
(≈ 100–110K tokens for Qwen3-Coder-30B-A3B on 48 GB; Qwen3-8B capped at 40,960)

## Constitution Check

| Principle | Gate | Status |
| --- | --- | --- |
| I Agent latency | Gates measured with the in-repo engine benchmark (FR-017) on named models | Pass |
| II Exactness | Conformance vs mlx-lm (same schedule); deep-copy snapshots; trim to committed tokens | Pass |
| III Memory budget | Pre-load check, measured scratch, admission before execution, input caps, MLX limits | Pass |
| IV Local, private | No network, no prompt logging; content-hash identity computed in 001 (R10) | Pass |
| V Own runtime, reuse kernels | MLX kernels only; graphs ported from mlx-lm; no new kernels | Pass |
| VI Simple | One engine thread, one sequence, no batching, no paging | Pass |

Post-design re-check: Pass. Complexity note: vendoring MLX + our own sys crate is justified below.

## Project Structure

### Documentation (this feature)

```text
specs/001-engine-core/
├── plan.md  research.md  data-model.md  quickstart.md
├── contracts/engine.md   # internal Engine trait + `mrun run` CLI contract
└── tasks.md
```

### Source Code (repository root)

```text
Cargo.toml                     # workspace
rust-toolchain.toml
crates/
├── mlx-sys/
│   ├── build.rs               # cmake vendored mlx-c/MLX; Metal on macOS, CPU+OpenBLAS on Linux
│   ├── vendor/                # mlx-c (submodule @a341b49) + MLX source tarball sha256-pinned
│   └── src/bindings.rs        # checked-in bindgen output
├── mrun-engine/               # features: mlx (default) pulls in mlx-sys
│   ├── src/
│   │   ├── engine.rs          # Engine trait, ModelInfo, MemoryBudget, Segment, errors (no MLX)
│   │   ├── identity.rs        # model identity (R10), shared with 004
│   │   ├── mlx/               # [mlx] safe layer: Array (RAII, !Send, !Sync), ops, fast, memory, device
│   │   ├── load.rs            # config.json + safetensors, quantization (mode/bits/group, per-path overrides)
│   │   ├── models/{qwen3.rs, qwen3_moe.rs, common.rs}
│   │   ├── kv.rs              # per-layer K/V, cap rounded to 256, 4,096-token per-layer growth, trim, deep-copy snapshot/restore
│   │   ├── sample.rs          # temperature, top-k, top-p, min-p, seed (on device), optional mask
│   │   ├── budget.rs          # budget formula (no MLX) + [mlx] scratch probe with cache
│   │   └── mlx_engine.rs      # [mlx] MlxEngine implementing Engine
│   ├── examples/{spike.rs, bench_engine.rs}
│   └── tests/                 # tiny random models (CPU), reference conformance (macOS, opt-in)
└── mrun/
    └── src/
        ├── main.rs  cli/run.rs           # features: mlx (default) → mrun-engine/mlx
        ├── chat/{tokenizer.rs, template.rs, stop.rs, utf8.rs}
        └── runtime/{handle.rs (worker thread), generate.rs (loop), session.rs, scripted.rs}
```

**Structure Decision**: Three crates (see [ARCHITECTURE](../../concepts/ARCHITECTURE.md)): `mlx-sys`
isolates the C/C++ build, `mrun-engine` holds the MLX-free `Engine` contract plus the MLX
implementation behind a feature, `mrun` holds everything engine-agnostic. `cargo test -p mrun
--no-default-features` never builds MLX.

## Complexity Tracking

| Item | Why needed | Simpler alternative rejected because |
| --- | --- | --- |
| Own `mlx-sys` over vendored mlx-c | Linux CI build, quant `mode` parameter, `device_info`, pinned MLX with the gather_qmm fix | `mlx-rs` fails to link on Linux and hard-codes affine quantization ([mlx-rust §7.1](../../concepts/research/mlx-rust.md)) |
| Separate engine crate | Keeps MLX/C++ build out of fast iteration on API/CLI code; enforces one-thread ownership | Single crate makes every test build MLX |
