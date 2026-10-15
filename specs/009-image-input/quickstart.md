# Quickstart: Image Input

Validation scenarios for feature 009 (after implementation).

## Gate (Phase 0)

```sh
mrun serve --record ~/traces/2026-11 …            # normal agent use for two weeks, several clients
mrun bench --trace-stats ~/traces/2026-11          # share of sessions with images per client (FR-001)
curl -s localhost:8484/mrun/status | jq .images    # live counters, incl. images omitted by FR-020
```

## Automated

```sh
cargo test -p mrun --test media_limits --test media_digest --test media_api --test prefix_images   # any OS
MRUN_TEST_MODELS=qwen3.6 cargo test -p mrun-engine --test vision_conformance --test vision_equality -- --ignored  # macOS
```

## Manual (target Mac, Qwen3.6 4-bit)

1. `mrun pull qwen3.6 && mrun show qwen3.6`: shows `vision: yes` and its memory cost.
   `mrun run qwen3.6 --info` lists vision weights, vision scratch and the embedding-cache cap.
2. `mrun launch claude-code`; paste a screenshot of a terminal with a red error line; ask what
   failed and what colour the line is (SC-001, 5 runs). Repeat with the Read tool on a PNG (OQ-2).
3. Continue that session for 10 turns: every turn's `: mrun` metadata shows `encoder_runs: 0`,
   no miss cause, cached ≥ the previous committed length (SC-002).
4. Start a subagent that receives the same screenshot: `embedding_hits: 1`, `encoder_runs: 0`.
5. `mrun launch codex`; ask it to view an image file: `view_image` is offered and answered
   (SC-001). `mrun launch codex --dry-run` with `--no-vision` on the server shows
   `input_modalities = ["text"]`.
6. Send a user message with an `https://` image URL: 400 naming the path; the same inside a tool
   result: placeholder and `image-omitted`.
7. `mrun serve --model qwen3-coder`: 002's image placeholder and rejection behaviour (SC-007).
8. `mrun bench --suite vision` reports encoder time per image size and TTFT overhead (SC-003).
