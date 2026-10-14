# Quickstart: Engine Core

## Prerequisites

- Apple Silicon Mac, macOS 15+, full Xcode (or the Command Line Tools plus the Metal toolchain via
  `xcodebuild -downloadComponent MetalToolchain`), CMake ≥ 3.24.
- Rust per `rust-toolchain.toml`.
- Model weights in the HF cache: `mlx-community/Qwen3-8B-4bit` (until feature 004, download with
  `huggingface-cli download` or `hf download`).
- For conformance references: Python with `mlx-lm` at the pinned version (only when regenerating).

## Build and test

```sh
cargo build --release
cargo test --workspace                         # unit + tiny-model tests (CPU on Linux, Metal on macOS)
MRUN_TEST_MODELS=qwen3-8b cargo test -p mrun-engine --test conformance -- --ignored
```

## Scenarios

1. **Chat** (US1): `mrun run mlx-community/Qwen3-8B-4bit` → type a question → streamed answer and
   a stats line; follow-up reuses state (stats show cached tokens).
2. **Budget** (US3): `mrun run mlx-community/Qwen3-8B-4bit --info` → weights, scratch, KV capacity
   in tokens and bytes. With `--memory-fraction 0.2 --prompt "$(cat big.txt)"` → up-front
   rejection naming requested and available tokens.
3. **Cancel** (US2): paste a 16K-token prompt, press Ctrl-C during processing → prompt returns
   within 300 ms; ask again → normal answer.
4. **Conformance** (US4): `mrun run mlx-community/Qwen3-8B-4bit --check` → "20/20 prompts match";
   repeat with `mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit`.
5. **Speed and cancel** (SC-002/003/004):
   `cargo run --release -p mrun-engine --example bench_engine -- --model <repo> --ctx 1024,16384`
   and `scripts/bench_reference.sh <repo>` (pinned mlx-lm, `--prefill-step-size 2048`); compare
   the two JSON files.

## Regenerating references

`scripts/gen_reference.py` (pinned mlx-lm) writes `crates/mrun-engine/tests/reference/<model>.json`
with prompt token IDs and greedy output IDs using the same chunk size as mrun.
