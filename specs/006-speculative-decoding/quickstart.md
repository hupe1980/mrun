# Quickstart: Speculative Decoding

Validation scenarios for feature 006 (after implementation), on the target Mac (M4 Pro, 48 GB).

## Prerequisites

`mrun` on `PATH`; `mrun pull qwen3.6` (with MTP sidecar) and `mrun pull qwen3-coder`; bundled
agent traces.

## Scenarios

1. **Measure first** (FR-001):
   ```sh
   mrun bench --suite spec --offline            # acceptance histograms, copy_share_16 per trace
   mrun serve --model qwen3.6 --detach
   mrun bench --suite spec                      # off | ngram | mtp | auto per trace
   ```
   Expect: edit-heavy traces labelled; drafted/accepted per mode.
2. **No regression** (SC-001): `mrun bench --suite all` with `--speculation off`, then `auto`, per
   model; compare medians with `--baseline` at a 3% threshold (exit 0 expected).
3. **Edit speed-up** (SC-002): in the `spec` cells, `auto` decode tok/s on edit-heavy traces ≥ 1.3×
   `off`.
4. **Greedy equality** (SC-003): `mrun run qwen3.6 --check --speculation auto`; any divergence is
   reported with its position and `off` logit margin.
5. **Metadata**: `curl -si localhost:8484/v1/messages …` shows `x-mrun-spec-*` headers; with
   `-H 'x-mrun-speculation: off'` the reason is `off`; a strict-tool request shows `grammar`.
6. **MTP sidecar**: `mrun show qwen3.6` lists the sidecar; delete it and restart: `mrun ps` shows
   `mtp: unavailable (no-sidecar)` and n-gram drafting continues; `mrun pull qwen3.6` restores it
   without re-downloading the base model.

Automated equivalents: `cargo test -p mrun --test spec_mutations --test spec_controller --test
spec_index` (any OS); `cargo test -p mrun-engine --test spec_sampling` (CPU MLX);
`spec_conformance` on macOS.
