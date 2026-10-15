# Quickstart: Speculative Decoding

Validation scenarios for feature 006 (after implementation), on the target Mac (M4 Pro, 48 GB).

## Prerequisites

`mrun` on `PATH`; `mrun pull qwen3.6` and `mrun pull qwen3-coder`; bundled agent traces.

## Scenarios

1. **Measure first** (FR-001):
   ```sh
   mrun bench --suite spec --offline            # acceptance histograms, copy_share_16 per trace
   mrun serve --model qwen3.6 --detach
   mrun bench --suite spec                      # off | ngram | auto per trace
   ```
   Expect: edit-heavy traces labelled; drafted/accepted per mode.
2. **No regression** (SC-001): `mrun bench --suite all` with `--speculation off`, then `auto`, per
   model (5 repetitions, medians); compare with `--baseline` at the 005 regression threshold (exit 0
   expected).
3. **Edit speed-up** (SC-002): in the `spec` cells, `auto` decode tok/s on edit-heavy traces ≥ 1.3×
   `off`.
4. **Greedy equality** (SC-003): `spec_conformance` (below) compares `auto` with `off`; any
   divergence is reported with its position and `off` logit margin. `mrun run qwen3.6 --check`
   always runs with speculation `off` (FR-017).
5. **Metadata**: `curl -si localhost:8484/v1/messages …` shows `x-mrun-spec-*` headers; with
   `-H 'x-mrun-speculation: off'` the reason is `off`; a strict-tool request shows `grammar`;
   `curl -s localhost:8484/mrun/status | jq .spec` shows the controller.

Automated equivalents: `cargo test -p mrun --test spec_mutations --test spec_controller --test
spec_index` (any OS); `cargo test -p mrun-engine --test spec_sampling` (CPU MLX);
`spec_conformance` on macOS.
