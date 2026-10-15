# Quickstart: Engine Core

## Prerequisites

- Apple Silicon Mac, macOS 15+, Xcode (or Command Line Tools + `xcodebuild -downloadComponent
  MetalToolchain`), CMake ≥ 3.24; Rust per `rust-toolchain.toml`.
- Weights in the HF cache (until 004: `hf download <repo>`): `mlx-community/Qwen3-8B-4bit`,
  `mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit`, a 4-bit MLX conversion of Qwen3.6-35B-A3B,
  `mlx-community/gpt-oss-20b-MXFP4-Q8` and `mlx-community/Devstral-Small-2-24B-Instruct-2512-4bit`
  (pins per 004 T005a).
- Only to regenerate references or goldens: Python with mlx-lm `ba8a08f` and transformers at the
  pinned version.

## Build and test

```sh
cargo build --release
cargo test --workspace                          # unit, ScriptedEngine, tiny CPU models
cargo test -p mrun --no-default-features        # no MLX build
MRUN_TEST_MODELS=qwen3-8b,qwen3-coder,qwen3.6,gpt-oss,devstral cargo test -p mrun-engine --release -- --ignored
```

## Scenarios

1. **Chat** (US1): `mrun run mlx-community/Qwen3-8B-4bit` → ask, then follow up → the second stats
   line shows cached tokens ≈ the first turn. Repeat with the Qwen3.6 model.
2. **Budget** (US3), machine-independent: `mrun run mlx-community/Qwen3-8B-4bit --memory-fraction
   0.35 --reserve-gib 8 --info` → on a 48 GB Mac a budget of ≈ 8.8 GiB and a capacity C of roughly
   18K tokens (bounded by the probe depth). Then with the same flags:
   - a prompt of C − 1 tokens → one output token, finish "length";
   - a prompt of C tokens → rejected before processing, naming C and C + 1;
   - `--max-tokens 100000` with a short prompt → stats report the clamped limit.
   With `--memory-fraction 0.25 --reserve-gib 8` (budget ≈ 4 GiB < weights) → load refused, exit 4.
3. **Cancel** (US2): paste a 16K-token prompt, press Ctrl-C during processing → ready within
   300 ms; ask again → stats show the kept whole chunks as cached.
4. **Conformance** (US4): `mrun run <model> --check` → "20/20 prompts match" for each 0.1 family
   (Qwen3.6, Qwen3-Coder, gpt-oss-20b, Devstral Small 2, Qwen3-8B). A model with an unsupported
   `model_type` → exit 3 listing the supported families.
5. **State primitives** (US5): `cargo test -p mrun-engine --release --test exactness -- --ignored`
   → split-off/append and checkpoint-restore logits bit-identical on each model, and gpt-oss
   trims/moves at positions inside and beyond the sliding window.
6. **Speed and cancel** (SC-002/003/004):
   `cargo run --release -p mrun-engine --example bench_engine -- --model <repo> --ctx 1024,16384
   --cancel-depths 16384,65536,max` and `scripts/bench_reference.sh <repo>`; compare the JSON files.

## Regenerating references

- `scripts/gen_reference.py <repo>` (pinned mlx-lm, `--prefill-step-size 2048`) writes
  `crates/mrun-engine/tests/reference/<model>.json`: prompt IDs, 256 greedy IDs, EOS set.
- `scripts/gen_template_golden.py <repo>` (pinned transformers) writes
  `crates/mrun/tests/templates/<model>/*.json`: messages, tools, rendered text, token IDs.
