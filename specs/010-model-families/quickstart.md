# Quickstart: Model Families

## Automated

```sh
cargo test -p mrun-engine --test families          # tiny-model parity, every registered architecture (MLX CPU, any OS)
cargo test -p mrun --test syntax --test probe       # parser suites, probe, template goldens (any OS)
scripts/check_family_diff.sh origin/main            # locality rule on a family change
MRUN_TEST_MODELS=gpt-oss,devstral cargo test -p mrun-engine --test conformance -- --ignored   # macOS
MRUN_TEST_MODELS=gpt-oss cargo test -p mrun-engine --test prefix_equality -- --ignored              # macOS, window cases
```

## Manual (target Mac)

1. `mrun pull gpt-oss && mrun run gpt-oss --info`: family `gpt_oss`, syntax `harmony`,
   quantization `mxfp4 (experts) + affine`, positional state 48 KiB/token (12 windowed layers),
   capacity 131,072 (max position).
2. `mrun run gpt-oss --check` and `mrun run devstral --check`: no mismatch.
3. `mrun launch claude-code --model gpt-oss`; ask for a file edit: tool calls arrive, analysis
   text shows as thinking. Repeat with `codex` and `opencode`, then with `devstral`.
4. Continue a gpt-oss session past 5K tokens, edit a message near position 130: `mrun ps` shows
   cause `edit` and a divergence at that message; the output equals a `x-mrun-cache: off` rerun.
5. `mrun run some-org/unknown-arch-mlx`: exit 3, supported architectures and aliases listed.
6. `mrun run mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit` (Tier 2, `qwen3_moe`): `mrun show`
   prints `tier 2` and syntax `hermes` chosen by the probe.
7. `mrun bench --suite synthetic --model gpt-oss` and `--model devstral`: cells
   recorded (SC-005).
