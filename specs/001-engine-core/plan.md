# Implementation Plan: Engine Core

**Branch**: `001-engine-core` | **Date**: 2026-10-15 (revised) | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `specs/001-engine-core/spec.md`

## Summary

Run the release-0.1 model families from the `mrun` binary on Apple Silicon: `qwen3_5_moe`
(Qwen3.6-35B-A3B, hybrid), `qwen3_moe` (Qwen3-Coder-30B-A3B), `gpt_oss` (gpt-oss-20b, MXFP4,
sliding window with sinks) and `qwen3` (Qwen3-8B, internal test reference, no alias); Devstral
Small 2 (`mistral`) moved to 0.2 (010 T010). Own
`mlx-sys` over vendored mlx-c/MLX, a thin safe layer, a shared block library (attention: full,
sliding window, sinks; Gated DeltaNet with mlx-lm's Metal kernel bound through
`mlx_fast_metal_kernel_*`; dense and MoE feed-forward; norms; RoPE incl. YaRN) and one small family
module per architecture, each a port of the mlx-lm graph with a parity test; the registry and block
design are shared with 010. Unknown architectures are refused with the supported families. One
worker thread owns all MLX state. The sequence state is positional (K/V of every attention layer;
sliding-window layers keep full KV) plus position-bound (linear-layer state, Qwen3.6 only), allocated lazily within
a per-request reservation, with growth reported before allocation so the caller evicts first. The
reshaped `Engine` trait exposes depth-adaptive chunked prefill with planned boundaries and
checkpoints (one layer in flight, whole chunks committed), a forward/sample split with one-token
pipelining, checkpoints, and move primitives (`split_off`, `append`) for 003. The runtime tracks
committed tokens, delivers `Finished` unconditionally, and survives panics. `mrun run` provides
an interactive chat; `--check` runs greedy conformance against stored mlx-lm outputs.

## Technical Context

**Language/Version**: Rust 1.94.0 (edition 2024); C/C++17 for vendored MLX via CMake
**Primary Dependencies**: MLX v0.32.4 (`36969c0`) + mlx-c v0.7.0 (`a341b49`), vendored and
hash-checked, bindings checked in; `tokenizers` 0.23 (no `http`); `minijinja` 3 +
`minijinja-contrib` (pycompat); `flume` 0.12 (sync + async channels); clap 4.6; serde/serde_json;
sha2; toml
**Storage**: HF cache for models; `~/.mrun/{identity,probe}.toml`; references in
`crates/mrun-engine/tests/reference/*.json`; template goldens in `crates/mrun/tests/templates/`
**Testing**: cargo test. Linux CI: `ScriptedEngine` (allocation, growth, pipelining, checkpoints,
moves), chat layer goldens, MLX CPU backend with tiny random-weight `qwen3`/`qwen3_moe`/`qwen3_5_moe`
configs and a tiny `gpt_oss` config
(GDN via the ops reference; MXFP4 on the CPU backend). macOS (`#[ignore]`, `MRUN_TEST_MODELS`): GDN kernel vs ops,
conformance on real weights, exactness of moves/checkpoints, bench cells
**Target Platform**: macOS 15+ on Apple Silicon; Linux x86_64/aarch64 for CI (CPU)
**Project Type**: Rust workspace producing one CLI/server binary
**Performance Goals**: ≥ 90% of mlx-lm decode (1K, 16K) and prefill (16K) per model; cancel
≤ 300 ms p95 at 16K and 64K (max depth reported only); first token ≤ 15 s cold for 8B
**Constraints**: Budget by construction (III); exact greedy parity (II); no network; input caps
**Scale/Scope**: 4 families (one model each), 1 live sequence; capacity on 48 GB ≈ 262K (Qwen3.6,
max position), ≈ 105–120K (Qwen3-Coder), 131,072 (gpt-oss-20b, max position; ≈ 14.7 GiB of KV
would allow > 300K (capped at 131K)), 40,960 (Qwen3-8B,
max position)

## Constitution Check

| Principle | Gate | Status |
| --- | --- | --- |
| I Agent latency | Speed and cancel gates per model and depth bucket via the in-repo bench (FR-018) | Pass |
| II Exactness | Greedy parity defined as mlx-lm's argmax over logprobs; same kernels and compiled ops; deterministic chunk splits; deep-copied checkpoints and segments; trim to committed tokens | Pass |
| III Memory budget | Pre-load check, probed scratch (two points), capacity ≤ probed depth, lazy allocation clamped to the reservation, `NeedsMemory` before every allocation, input caps, MLX limits | Pass |
| IV Local, private | No network; identity covers every output-determining file | Pass |
| V Own runtime, reuse kernels | MLX kernels (incl. `quantized_matmul`/`gather_qmm` mode `mxfp4`, SDPA with sinks); GDN kernel is mlx-lm's source ported unchanged (reference kernel, not a new one); open set of families from shared blocks, family logic only in its module and chat syntax | Pass (see Complexity) |
| VI Simple | One thread, one sequence, contiguous arrays, no paging; dead code removed; `--chunk` flag dropped | Pass |

Post-design re-check: Pass.

## Project Structure

### Documentation (this feature)

```text
specs/001-engine-core/
├── plan.md  research.md  data-model.md  quickstart.md
├── contracts/engine.md   # Engine trait, generation loop, `mrun run`
├── checklists/requirements.md
└── tasks.md
```

### Source Code (repository root)

```text
crates/
├── mlx-sys/                   # build.rs (cmake), vendor/ (mlx-c submodule, MLX tarball), src/bindings.rs
├── mrun-engine/               # feature `mlx` (off by default) pulls in mlx-sys
│   ├── src/
│   │   ├── engine.rs          # Engine trait, chunk_len, plan/step/segment/checkpoint types, errors (no MLX)
│   │   ├── identity.rs        # identity + EOS union (R10)
│   │   ├── budget.rs          # formula (no MLX) + [mlx] probe with cache
│   │   ├── mlx/               # [mlx] Array, ops, fast, metal_kernel, compile, memory, device
│   │   ├── load.rs            # config/text_config, quantization, sanitize, safetensors
│   │   ├── state.rs           # attention K/V + recurrent state, allocation, checkpoints, segments
│   │   ├── models/blocks/     # shared blocks (layout per 010): attention (full, sliding, sinks),
│   │   │                      # gated_delta, ffn (SwiGLU, SwigluOAI, MoE/SwitchGLU), norm, rope (default, YaRN)
│   │   ├── models/registry.rs # model_type → family module, state kinds, one SyntaxId (010)
│   │   ├── models/{qwen3.rs, qwen3_moe.rs, qwen3_5_moe.rs, gpt_oss.rs}  # one per family
│   │   ├── kernels/gated_delta.metal.inc   # mlx-lm kernel bodies, verbatim
│   │   ├── sample.rs          # logprobs, greedy, top-k/p, min-p, seed, mask
│   │   └── mlx_engine.rs      # MlxEngine
│   ├── examples/{spike.rs, bench_engine.rs}
│   └── tests/                 # tiny CPU models, conformance, exactness (macOS)
└── mrun/src/
    ├── main.rs  cli/run.rs  errors.rs
    ├── chat/{limits.rs, tokenizer.rs, template.rs, stop.rs, prompt.rs}
    └── runtime/{handle.rs, generate.rs, session.rs, scripted.rs}
scripts/{gen_reference.py, gen_template_golden.py, bench_reference.sh}
```

**Structure Decision**: Three crates (ARCHITECTURE): `mlx-sys` isolates the C/C++ build;
`mrun-engine` holds the MLX-free contract and the MLX implementation behind a feature; `mrun`
holds everything engine-agnostic. `cargo test -p mrun` (default features) never builds MLX; `--features mlx` (on `mrun`, forwarding
`mrun-engine/mlx`) builds it.

## Complexity Tracking

| Item | Why needed | Simpler alternative rejected because |
| --- | --- | --- |
| Own `mlx-sys` over vendored mlx-c | Linux CI, quant `mode`, `device_info`, metal kernels, pinned MLX | `mlx-rs` fails on Linux and lacks these ([mlx-rust §7.1](../../concepts/research/mlx-rust.md)) |
| Separate engine crate | Keeps MLX out of fast iteration; one-thread ownership | Single crate builds MLX for every test |
| Custom Metal kernel (GDN) | No MLX op implements the gated delta recurrence at speed; parity needs mlx-lm's exact kernel | Ops loop (`gated_delta_ops`) is per-token and not bitwise-equal to the kernel; kept only as CPU/test reference |
| Position-bound rollback (chunk start, pending step) | Whole-chunk commit and pipelining on non-trimmable state | Disabling pipelining for hybrids costs decode speed; re-prefill on every cancel loses progress |
| SC-004 max-depth bucket reported, not gated (until 006 or a chunk-floor change makes it measurable) | One layer at the 128-token chunk floor may take ~200 ms at 262K (R5), so 300 ms p95 is not reliably reachable | Gating it would block 0.1 on a bound the current chunk floor cannot meet; a smaller floor costs prefill speed |
| Full KV on sliding-window layers | Positional state stays exact at any position (trim, moves, 003 reuse) | A rotating buffer (mlx-lm `RotatingKVCache`) saves 24 KiB/token on gpt-oss but makes earlier positions unrecoverable, so every branch or edit would need checkpoints |
