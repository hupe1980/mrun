# Quickstart: Model Families

Post-0.1 (lean cut 2026-10-15). The 0.1 families are validated by the 001–005 quickstarts.

## Automated

```sh
cargo test -p mrun-engine --test families          # tiny-model parity, every registered architecture (MLX CPU, any OS)
cargo test -p mrun --test syntax                    # parser suites, template check, template goldens (any OS)
MRUN_TEST_MODELS=devstral cargo test -p mrun-engine --test conformance -- --ignored        # macOS, 0.2
MRUN_TEST_MODELS=devstral cargo test -p mrun --features mlx --test prefix_equality -- --ignored
```

## Manual (target Mac)

1. `mrun pull devstral && mrun run devstral --info`: family `mistral`, syntax `mistral`,
   positional state ≈160 KiB/token, capacity ≈78K on 48 GB.
2. `mrun run devstral --check`: no mismatch.
3. `mrun launch claude-code --model devstral`; ask for a file edit: `[TOOL_CALLS]` calls arrive as
   tool calls. Repeat with `codex` and `opencode` (002 SC-001).
4. Edit a message in a Devstral session: `mrun ps` shows cause `changed` and the message index;
   the output equals a `mrun serve --no-cache` rerun.
5. `mrun run some-org/unknown-arch-mlx`: exit 3, supported architectures and aliases listed.
6. With Tier 2: `mrun run <repo of a registered architecture>`: `mrun show` prints `tier 2`.
